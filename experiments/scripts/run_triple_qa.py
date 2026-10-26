#!/usr/bin/env python3
"""Answer GraphRAG-Bench blind questions using pre-extracted HippoRAG triples + LLM.

Retrieval: token-overlap ranking of triples within the question's source report.
Generation: OpenAI-compatible chat API (from backend/config/.env).

Example:
  python run_triple_qa.py \\
    --triples /workspace/ESG_KG/output/triples/gpt-4o-mini/esg_ocr_triples.json \\
    --questions /workspace/Datasets/Questions/esg_questions_blind.json \\
    --gold /workspace/ESG_KG/output/qg/esg_qg_gold_for_eval.json \\
    --out /workspace/ESG_KG/output/qa/qwen25-3b_gpt4omini-triples/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.I)
_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "was",
    "were",
    "is",
    "are",
    "be",
    "by",
    "with",
    "as",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "did",
    "does",
    "do",
    "its",
    "it",
    "their",
    "they",
    "has",
    "have",
    "had",
    "been",
    "being",
    "about",
    "into",
    "over",
    "under",
    "than",
    "then",
    "also",
    "per",
    "vs",
}


def detect_newesg_root() -> Path:
    here = Path(__file__).resolve()
    euler = here.parents[2]
    sibling = euler.parent
    if (sibling / "Datasets" / "Questions").is_dir():
        return sibling
    if Path("/workspace/Datasets/Questions").is_dir():
        return Path("/workspace")
    return sibling


def load_dotenv() -> Dict[str, str]:
    env = {"LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""}
    for path in (
        Path(__file__).resolve().parents[1] / "config" / ".env",
        Path("/workspace/backend/config/.env"),
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in env and v:
                env[k] = v
        break
    for k in list(env):
        if not env[k]:
            env[k] = os.getenv(k, "") or ""
    return env


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def triple_text(t: Dict[str, Any]) -> str:
    head = ((t.get("head") or {}).get("name")) or ""
    rel = t.get("relation") or ""
    tail = ((t.get("tail") or {}).get("name")) or ""
    page = ((t.get("qualifiers") or {}).get("page")) or ""
    return f"({head}; {rel}; {tail})" + (f" [page {page}]" if page else "")


def load_triples_by_doc(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(data, dict) and "reports" in data:
        for r in data.get("reports") or []:
            doc = str(r.get("document_id") or "")
            if doc:
                out[doc] = list(r.get("triples") or [])
    elif isinstance(data, dict) and "triples" in data:
        doc = str(data.get("document_id") or "unknown")
        out[doc] = list(data.get("triples") or [])
    else:
        raise ValueError(f"Unrecognized triples format: {path}")
    return out


def load_questions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return list(data["questions"])
    raise ValueError(f"Unrecognized questions format: {path}")


def load_gold_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    return {str(x.get("id")): x for x in (items or []) if x.get("id")}


class TripleIndex:
    """Per-document inverted index for fast overlap retrieval."""

    def __init__(self, triples: List[Dict[str, Any]]):
        self.triples = triples
        self.texts = [triple_text(t) for t in triples]
        self.doc_tokens: List[List[str]] = [tokenize(x) for x in self.texts]
        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self.n = max(1, len(triples))
        self.idf = {
            t: math.log(1.0 + self.n / (1.0 + df)) for t, df in self.df.items()
        }

    def retrieve(self, question: str, top_k: int = 40) -> List[Tuple[float, int, str]]:
        q_toks = tokenize(question)
        if not q_toks or not self.triples:
            return []
        q_tf = Counter(q_toks)
        scored: List[Tuple[float, int]] = []
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            for term, qf in q_tf.items():
                if term in tf:
                    score += qf * tf[term] * self.idf.get(term, 0.0)
            # light boost for exact phrase hits of longer tokens
            text_l = self.texts[i].lower()
            for term in q_toks:
                if len(term) >= 5 and term in text_l:
                    score += 0.15
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for score, i in scored[:top_k]:
            out.append((score, i, self.texts[i]))
        return out


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qa_answer_from_triples.md"


def load_system_prompt() -> str:
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    # Fallback if the markdown prompt is missing.
    return (
        "Answer ONLY from the provided knowledge triples. "
        "If insufficient, reply exactly: UNKNOWN. "
        "Use ONE short English phrase/sentence (≤ 20 words) ending with a period. "
        'Canonical numeric form: "<value> <metric-or-unit>" — never a bare number.'
    )


SYSTEM_PROMPT = load_system_prompt()


def build_user_prompt(question: str, triple_lines: List[str], qtype: str) -> str:
    ctx = "\n".join(f"- {line}" for line in triple_lines) or "(none)"
    return (
        f"Question type: {qtype or 'n/a'}\n"
        f"Question: {question}\n\n"
        f"Knowledge triples:\n{ctx}\n\n"
        "Write the final answer only, following the Answer format rules:\n"
    )


def call_llm(
    client,
    *,
    model: str,
    question: str,
    triple_lines: List[str],
    qtype: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(question, triple_lines, qtype),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Qwen3 non-streaming requires enable_thinking=false on DashScope.
    if "qwen3" in (model or "").lower():
        kwargs["extra_body"] = {"enable_thinking": False}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as first_exc:
        # Some DashScope models prefer max_completion_tokens
        try:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            raise first_exc
    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    meta = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }
    return content, meta


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--triples",
        default=str(
            root / "ESG_KG" / "output" / "triples" / "hipporag" / "gpt-4o-mini" / "esg_ocr_triples.json"
        ),
    )
    p.add_argument(
        "--questions",
        default=str(root / "Datasets" / "Questions" / "esg_questions_blind.json"),
    )
    p.add_argument(
        "--gold",
        default=str(root / "ESG_KG" / "output" / "qg" / "esg_qg_gold_for_eval.json"),
    )
    p.add_argument(
        "--out",
        default=str(
            root
            / "ESG_KG"
            / "output"
            / "qa"
            / "qwen25-3b_gpt4omini-triples"
            / "results.json"
        ),
    )
    p.add_argument("--model", default=None)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source", default=None, help="Only answer questions for this document_id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=20)
    args = p.parse_args()

    cfg = load_dotenv()
    model = args.model or cfg.get("LLM_MODEL") or "Qwen2.5-3B-Instruct"
    api_key = cfg.get("LLM_API_KEY") or ""
    base_url = cfg.get("LLM_BASE_URL") or None
    if not api_key:
        print("LLM_API_KEY missing", file=sys.stderr)
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Triples: {args.triples}")
    print(f"Questions: {args.questions}")

    triples_path = Path(args.triples)
    # Prefer .../hipporag/gpt-5.1 or .../LightRAG/gpt-5.1 as source label.
    parent = triples_path.parent
    grand = parent.parent if parent else None
    if grand and grand.name.lower() in {"hipporag", "lightrag"}:
        triple_source = f"{grand.name}/{parent.name}"
    else:
        triple_source = parent.name if parent and parent.name else "triples"
    triples_by_doc = load_triples_by_doc(triples_path)
    indexes = {doc: TripleIndex(ts) for doc, ts in triples_by_doc.items()}
    questions = load_questions(Path(args.questions))
    gold_map = load_gold_map(Path(args.gold) if args.gold else None)

    if args.source:
        questions = [q for q in questions if q.get("source") == args.source]
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prev, list):
                results = prev
                done_ids = {str(x.get("id")) for x in results if x.get("id")}
                print(f"Resume: loaded {len(done_ids)} existing answers")
        except Exception as exc:
            print(f"WARN: could not resume from {out_path}: {exc}", file=sys.stderr)

    def save() -> None:
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    t0 = time.time()
    answered = 0
    errors = 0
    for i, q in enumerate(questions, 1):
        qid = str(q.get("id") or "")
        if qid in done_ids:
            continue
        source = str(q.get("source") or "")
        question = str(q.get("question") or "").strip()
        qtype = str(q.get("question_type") or "")
        gold = gold_map.get(qid) or {}
        gold_answer = str(gold.get("answer") or "")
        gold_evidence = str(gold.get("evidence") or "")

        index = indexes.get(source)
        retrieved = index.retrieve(question, top_k=args.top_k) if index else []
        triple_lines = [line for _s, _i, line in retrieved]
        context = "\n".join(triple_lines)

        generated = ""
        meta: Dict[str, Any] = {}
        err = None
        try:
            if not triple_lines:
                generated = "UNKNOWN"
            else:
                generated, meta = call_llm(
                    client,
                    model=model,
                    question=question,
                    triple_lines=triple_lines,
                    qtype=qtype,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
        except Exception as exc:
            err = str(exc)
            errors += 1
            print(f"[{i}/{len(questions)}] FAIL {qid}: {exc}", file=sys.stderr)

        row = {
            "id": qid,
            "question": question,
            "source": source,
            "context": context,
            "evidence": gold_evidence,
            "question_type": qtype,
            "generated_answer": generated,
            "gold_answer": gold_answer,
            "retrieval": {
                "top_k": args.top_k,
                "n_retrieved": len(triple_lines),
                "scores": [round(s, 4) for s, _i, _l in retrieved[:10]],
            },
            "model": model,
            "triple_source": triple_source,
            "error": err,
            "usage": meta,
        }
        results.append(row)
        answered += 1
        if answered % 10 == 0 or i == len(questions):
            elapsed = time.time() - t0
            print(
                f"[{i}/{len(questions)}] answered={answered} errors={errors} "
                f"elapsed={elapsed:.1f}s last={qid}"
            )
        if answered % max(1, args.save_every) == 0:
            save()
        if args.sleep:
            time.sleep(args.sleep)

    save()
    print(
        f"Done. wrote {len(results)} rows → {out_path} "
        f"(new={answered}, errors={errors}, seconds={time.time()-t0:.1f})"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
