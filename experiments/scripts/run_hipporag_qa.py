#!/usr/bin/env python3
"""Answer GraphRAG-Bench questions with official HippoRAG2 retrieve + rag_qa.

Uses prebuilt indexes from build_hipporag_index.py (OpenIE reused, no GPT re-extract).
Embedding at query time MUST match index build (default text-embedding-3-small).

Naming convention for outs (user-facing):
  ESG_KG/output/qa/hipporag2-{extractShort}-{qaShort}/results.json
  e.g. hipporag2-gpt5.1-qwen3-8b

Example:
  python run_hipporag_qa.py \\
    --extract-tag gpt-5.1 \\
    --model qwen3-8b \\
    --out /workspace/ESG_KG/output/qa/hipporag2-gpt5.1-qwen3-8b/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def detect_newesg_root() -> Path:
    here = Path(__file__).resolve()
    euler = here.parents[2]
    sibling = euler.parent
    if (sibling / "Datasets" / "Corpus").is_dir():
        return sibling
    if Path("/workspace/Datasets/Corpus").is_dir():
        return Path("/workspace")
    return sibling


def load_dotenv() -> Dict[str, str]:
    keys = [
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LIGHTRAG_EMBEDDING_MODEL",
        "LIGHTRAG_EMBEDDING_BASE_URL",
        "LIGHTRAG_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    ]
    env = {k: "" for k in keys}
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


def extract_short(tag: str) -> str:
    t = tag.strip().lower()
    mapping = {
        "gpt-4o-mini": "gpt4omini",
        "gpt-4o-mini-2024-07-18": "gpt4omini",
        "gpt-5.1": "gpt5.1",
        "gpt-5.1-2025-11-13": "gpt5.1",
        "gpt-5.4": "gpt5.4",
        "gpt-5.4-2026-03-05": "gpt5.4",
    }
    return mapping.get(t, t.replace("-", "").replace("_", ""))


def _patch_qwen3_non_thinking(rag, llm_model: str) -> None:
    """DashScope Qwen3 requires enable_thinking=false for non-streaming calls."""
    if "qwen3" not in (llm_model or "").lower():
        return
    client = getattr(getattr(rag, "llm_model", None), "openai_client", None)
    if client is None:
        return
    create = client.chat.completions.create

    def _wrapped(*args, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("enable_thinking", False)
        kwargs["extra_body"] = extra
        return create(*args, **kwargs)

    client.chat.completions.create = _wrapped  # type: ignore[method-assign]


def _ensure_index_alias(save_dir: Path, *, index_llm: str, qa_llm: str, emb: str) -> None:
    """HippoRAG uses save_dir/{llm}_{emb}/; copy indexed graph to QA llm folder.

    Symlinks often fail on Docker→Windows mounts, so we copy (once) instead.
    """
    import shutil

    qa_llm = qa_llm.replace("/", "_")
    src = save_dir / f"{index_llm}_{emb}"
    if not src.is_dir():
        alt = save_dir / f"seeded-openie_{emb}"
        if alt.is_dir():
            src = alt
        else:
            raise FileNotFoundError(f"Indexed graph dir missing: {src}")
    dst = save_dir / f"{qa_llm}_{emb}"
    marker = dst / ".copied_from_hippo_index"
    if marker.is_file():
        return
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    marker.write_text(str(src) + "\n", encoding="utf-8")

    # OpenIE file is also keyed by llm_name
    openie_src = save_dir / f"openie_results_ner_{index_llm}.json"
    if not openie_src.is_file():
        openie_src = save_dir / "openie_results_ner_seeded-openie.json"
    openie_dst = save_dir / f"openie_results_ner_{qa_llm}.json"
    if openie_src.is_file() and not openie_dst.is_file():
        shutil.copy2(openie_src, openie_dst)


def qa_short(model: str) -> str:
    m = (model or "").strip().lower()
    if m.startswith("qwen3-235b"):
        return "qwen3-235b"
    if m.startswith("qwen3-32b"):
        return "qwen3-32b"
    if m.startswith("qwen3-8b"):
        return "qwen3-8b"
    return m.replace("/", "_")


class HippoIndexPool:
    def __init__(
        self,
        index_root: Path,
        *,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str,
        embedding_model: str,
        embedding_base_url: str,
        embedding_api_key: str,
        qa_top_k: int,
        retrieval_top_k: int,
        linking_top_k: int,
    ):
        self.index_root = Path(index_root)
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self.qa_top_k = qa_top_k
        self.retrieval_top_k = retrieval_top_k
        self.linking_top_k = linking_top_k
        self._cache: Dict[str, Any] = {}

    def get(self, document_id: str):
        if document_id in self._cache:
            return self._cache[document_id]
        save_dir = self.index_root / document_id
        if not save_dir.is_dir():
            raise FileNotFoundError(f"HippoRAG index missing: {save_dir}")
        from hipporag import HippoRAG

        # HippoRAG stores graph under {llm_name}_{embedding}/. Index was built with
        # llm_name="hippo"; symlink so QA llm_name still loads that graph.
        _ensure_index_alias(save_dir, index_llm="hippo", qa_llm=self.llm_model, emb=self.embedding_model)

        # LLM client picks up OPENAI_API_KEY (= DashScope key) + llm_base_url.
        os.environ["OPENAI_API_KEY"] = self.llm_api_key
        os.environ["OPENAI_BASE_URL"] = self.llm_base_url
        rag = HippoRAG(
            save_dir=str(save_dir),
            llm_model_name=self.llm_model,
            llm_base_url=self.llm_base_url,
            embedding_model_name=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
        )
        rag.global_config.qa_top_k = self.qa_top_k
        rag.global_config.retrieval_top_k = self.retrieval_top_k
        rag.global_config.linking_top_k = self.linking_top_k
        # Patch embedding client to OpenAI key (different from DashScope LLM key).
        if self.embedding_api_key and hasattr(rag, "embedding_model"):
            from openai import OpenAI

            rag.embedding_model.client = OpenAI(
                api_key=self.embedding_api_key, base_url=self.embedding_base_url
            )
        _patch_qwen3_non_thinking(rag, self.llm_model)
        self._cache[document_id] = rag
        return rag

    def answer(self, document_id: str, question: str) -> Dict[str, Any]:
        rag = self.get(document_id)
        sols, messages, meta = rag.rag_qa(queries=[question])
        sol = sols[0] if sols else None
        answer = ""
        docs: List[str] = []
        if sol is not None:
            answer = str(getattr(sol, "answer", "") or "").strip()
            docs = list(getattr(sol, "docs", None) or [])
        usage = {}
        if isinstance(meta, list) and meta:
            m0 = meta[0] if isinstance(meta[0], dict) else {}
            usage = {
                "prompt_tokens": m0.get("prompt_tokens") or m0.get("input_tokens"),
                "completion_tokens": m0.get("completion_tokens") or m0.get("output_tokens"),
            }
        return {
            "answer": answer,
            "docs": docs[: self.qa_top_k],
            "raw_message": (messages[0] if messages else ""),
            "usage": usage,
        }


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract-tag", required=True, help="gpt-4o-mini | gpt-5.1 | gpt-5.4")
    p.add_argument("--index-root", default=None)
    p.add_argument(
        "--questions",
        default=str(root / "Datasets" / "Questions" / "esg_questions_blind.json"),
    )
    p.add_argument(
        "--gold",
        default=str(root / "ESG_KG" / "output" / "qg" / "esg_qg_gold_for_eval.json"),
    )
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=None, help="Answer LLM (default: LLM_MODEL)")
    # Paper (GraphRAG-Bench / HippoRAG2): qa_top_k=5, retrieval_top_k=5, linking_top_k=5
    p.add_argument("--qa-top-k", type=int, default=5)
    p.add_argument("--retrieval-top-k", type=int, default=5)
    p.add_argument("--linking-top-k", type=int, default=5)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    args = p.parse_args()

    cfg = load_dotenv()
    model = args.model or cfg.get("LLM_MODEL") or "qwen3-8b"
    api_key = cfg.get("LLM_API_KEY") or ""
    base_url = cfg.get("LLM_BASE_URL") or None
    if not api_key:
        print("LLM_API_KEY missing", file=sys.stderr)
        return 2

    emb_model = cfg.get("LIGHTRAG_EMBEDDING_MODEL") or "text-embedding-3-small"
    emb_base = cfg.get("LIGHTRAG_EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
    emb_key = cfg.get("LIGHTRAG_EMBEDDING_API_KEY") or cfg.get("OPENAI_API_KEY") or ""
    if not emb_key:
        print("Embedding API key missing", file=sys.stderr)
        return 2

    index_root = Path(args.index_root) if args.index_root else (
        root / "ESG_KG" / "output" / "indexes" / "hipporag" / args.extract_tag
    )
    if not index_root.is_dir():
        print(f"Index root missing: {index_root}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else (
        root
        / "ESG_KG"
        / "output"
        / "qa"
        / f"hipporag2-{extract_short(args.extract_tag)}-{qa_short(model)}"
        / "results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions") or questions.get("data") or []
    gold_by_id: Dict[str, Any] = {}
    gold_path = Path(args.gold)
    if gold_path.is_file():
        gold_raw = json.loads(gold_path.read_text(encoding="utf-8"))
        items = gold_raw if isinstance(gold_raw, list) else (gold_raw.get("questions") or [])
        for g in items:
            if isinstance(g, dict) and g.get("id"):
                gold_by_id[str(g["id"])] = g

    if args.source:
        questions = [q for q in questions if str(q.get("source") or "") == args.source]
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]

    results: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and out_path.is_file():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                loaded = []
            # Only resume official native rows; ignore legacy Top-40 answers.
            native = [
                r
                for r in loaded
                if isinstance(r, dict) and r.get("graph_backend") == "hipporag2-native"
            ]
            legacy_n = len(loaded) - len(native)
            if legacy_n:
                legacy_path = out_path.with_name("results.top40.json")
                if not legacy_path.is_file():
                    legacy_path.write_text(
                        json.dumps(loaded, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                print(f"Quarantined {legacy_n} non-native rows -> {legacy_path.name}")
            results = native
            done_ids = {
                str(r.get("id"))
                for r in results
                if r.get("id") and not r.get("error") and (r.get("generated_answer") or "")
            }
            print(f"Resume: {len(done_ids)} native answers kept ({legacy_n} legacy ignored)")
        except Exception:
            results = []

    # Point LLM env at answer model
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = model
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key

    pool = HippoIndexPool(
        index_root,
        llm_model=model,
        llm_base_url=base_url or "",
        llm_api_key=api_key,
        embedding_model=emb_model,
        embedding_base_url=emb_base,
        embedding_api_key=emb_key,
        qa_top_k=args.qa_top_k,
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
    )

    print(f"Answer model: {model}")
    print(f"Extract tag: {args.extract_tag}")
    print(f"Index root: {index_root}")
    print(f"Embedding: {emb_model}")
    print(f"qa_top_k={args.qa_top_k} retrieval_top_k={args.retrieval_top_k} linking_top_k={args.linking_top_k}")
    print(f"Out: {out_path}")
    print(f"Questions: {len(questions)}")

    t0 = time.time()
    new_n = 0
    err_n = 0
    for i, q in enumerate(questions, 1):
        qid = str(q.get("id") or "")
        if qid in done_ids:
            continue
        source = str(q.get("source") or "")
        question = str(q.get("question") or "")
        qtype = str(q.get("question_type") or q.get("type") or "")
        gold = gold_by_id.get(qid) or {}
        row: Dict[str, Any] = {
            "id": qid,
            "source": source,
            "question": question,
            "question_type": qtype,
            "gold_answer": gold.get("answer") or q.get("answer") or "",
            "evidence": gold.get("evidence") or q.get("evidence") or [],
            "generated_answer": "",
            "context": "",
            "model": model,
            "graph_backend": "hipporag2-native",
            "triple_source": f"hipporag/{args.extract_tag}",
            "retrieval": {
                "native": True,
                "qa_top_k": args.qa_top_k,
                "retrieval_top_k": args.retrieval_top_k,
                "linking_top_k": args.linking_top_k,
            },
            "usage": {},
            "error": None,
        }
        try:
            # LLM key for answer / DSPy filter
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = base_url or ""
            out = pool.answer(source, question)
            row["generated_answer"] = out.get("answer") or "UNKNOWN"
            row["context"] = "\n\n".join(out.get("docs") or [])
            row["usage"] = out.get("usage") or {}
        except Exception as exc:
            row["error"] = str(exc)
            row["generated_answer"] = "UNKNOWN"
            err_n += 1
            print(f"ERROR {qid}: {str(exc)[:300]}", file=sys.stderr)
        results.append(row)
        new_n += 1
        if args.sleep:
            time.sleep(args.sleep)
        if new_n % max(1, args.save_every) == 0 or i == len(questions):
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"[{len(results)}/{len(questions)}] answered={len(results)} "
                f"errors={err_n} elapsed={time.time()-t0:.1f}s last={qid}"
            )

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Done. wrote {len(results)} rows → {out_path} "
        f"(new={new_n}, errors={err_n}, seconds={time.time()-t0:.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
