#!/usr/bin/env python3
"""Extract triples with HippoRAG2 OpenIE from Datasets/Corpus/esg_ocr.json.

Writes ESG_KG/output/triples/hipporag/{out_name}/by_report/{doc}_triples.json
       ESG_KG/output/triples/hipporag/{out_name}/esg_ocr_triples.json (optional combined)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PAGE_SPLIT_RE = re.compile(r"(?m)^## Page (\d+)\s*$")


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
    """Prefer backend/config/.env over process env (Docker may pin stale LLM_MODEL)."""
    env = {
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "",
        "LLM_MODEL": "",
    }
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


def split_pages(context: str) -> List[Tuple[int, str]]:
    m = _PAGE_SPLIT_RE.search(context)
    if not m:
        body = context.strip()
        return [(1, body)] if body else []
    parts = _PAGE_SPLIT_RE.split(context[m.start() :])
    pages: List[Tuple[int, str]] = []
    i = 1
    while i + 1 < len(parts):
        pages.append((int(parts[i]), parts[i + 1].strip()))
        i += 2
    return pages


def page_has_statements(body: str) -> bool:
    m = re.search(r"(?ms)^### Statements\n(.*?)(?=^### |\Z)", body)
    block = (m.group(1) if m else "").strip()
    return bool(block) and block.lower() != "(none)"


def statements_block(body: str) -> str:
    m = re.search(r"(?ms)^### Statements\n(.*?)(?=^### |\Z)", body)
    return (m.group(1) if m else "").strip()


def normalize_hippo_triple(t: Any, *, page: int, chunk_id: str) -> Optional[Dict[str, Any]]:
    head = rel = tail = None
    if isinstance(t, (list, tuple)) and len(t) >= 3:
        head, rel, tail = t[0], t[1], t[2]
    elif isinstance(t, dict):
        head = t.get("head") or t.get("subject") or t.get("s")
        rel = t.get("relation") or t.get("predicate") or t.get("p")
        tail = t.get("tail") or t.get("object") or t.get("o")
        if isinstance(head, dict):
            head = head.get("name") or head.get("text")
        if isinstance(tail, dict):
            tail = tail.get("name") or tail.get("text")
    if not head or not rel or not tail:
        return None
    return {
        "head": {"type": "Other", "name": str(head).strip()},
        "relation": str(rel).strip(),
        "tail": {"type": "Other", "name": str(tail).strip()},
        "qualifiers": {"page": str(page), "year": None, "unit": None, "scope": None},
        "evidence_span": "",
        "chunk_id": chunk_id,
        "source": "hipporag2_openie",
    }


def build_docs(
    pages: List[Tuple[int, str]],
    *,
    statements_only: bool,
    pack_chars: int,
) -> List[Tuple[str, int, str]]:
    """Return list of (chunk_id, page_no_or_first, passage)."""
    units: List[Tuple[int, str]] = []
    for page_no, body in pages:
        if statements_only:
            if not page_has_statements(body):
                continue
            text = statements_block(body)
        else:
            text = body.strip()
        if not text:
            continue
        units.append((page_no, text))

    if pack_chars <= 0:
        return [(f"p{p}", p, t) for p, t in units]

    docs: List[Tuple[str, int, str]] = []
    buf_pages: List[int] = []
    buf_parts: List[str] = []
    buf_len = 0
    pack_i = 0

    def flush() -> None:
        nonlocal buf_pages, buf_parts, buf_len, pack_i
        if not buf_parts:
            return
        first = buf_pages[0]
        cid = f"pack{pack_i:03d}_p{first}"
        passage = "\n\n".join(buf_parts)
        docs.append((cid, first, passage))
        pack_i += 1
        buf_pages, buf_parts, buf_len = [], [], 0

    for page_no, text in units:
        chunk = f"## Page {page_no}\n{text}"
        if buf_len and buf_len + len(chunk) + 2 > pack_chars:
            flush()
        buf_pages.append(page_no)
        buf_parts.append(chunk)
        buf_len += len(chunk) + 2
    flush()
    return docs


def is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "insufficient_quota",
            "exceeded your current quota",
            "billing_hard_limit",
            "no credits remaining",
            "payment required",
            "quota exceeded",
            "you have insufficient",
        )
    )


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        default=str(root / "Datasets" / "Corpus" / "esg_ocr.json"),
    )
    p.add_argument(
        "--out-dir",
        default=str(root / "ESG_KG" / "output" / "triples" / "hipporag" / "gpt-4o-mini"),
    )
    p.add_argument("--pdf", default="coles", help="corpus_name filter substring")
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--statements-only", action="store_true", default=True)
    p.add_argument("--all-pages", action="store_true", help="Use full page text, not statements-only")
    p.add_argument(
        "--pack-chars",
        type=int,
        default=0,
        help="Pack pages into ~N-char docs (0 = one page per doc)",
    )
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if args.all_pages:
        args.statements_only = False

    cfg = load_dotenv()
    model = args.model or cfg.get("LLM_MODEL") or "gpt-4o-mini"
    api_key = cfg.get("LLM_API_KEY") or ""
    base_url = cfg.get("LLM_BASE_URL") or None
    if not api_key:
        print("LLM_API_KEY missing", file=sys.stderr)
        return 2

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url

    corpus_path = Path(args.corpus)
    items = json.loads(corpus_path.read_text(encoding="utf-8"))
    needle = (args.pdf or "").strip().lower()
    selected = []
    for item in items:
        name = str(item.get("corpus_name") or "")
        if args.all or (needle and needle in name.lower()):
            selected.append(item)
    if not selected:
        print(f"No corpus items matched pdf={args.pdf!r}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    by_report = out_dir / "by_report"
    by_report.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_hipporag_llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # HippoRAG2 OpenIE
    from hipporag.information_extraction import OpenIE
    from hipporag.llm.openai_gpt import CacheOpenAI
    from hipporag.utils.config_utils import BaseConfig

    global_config = BaseConfig()
    global_config.llm_name = model
    global_config.llm_base_url = base_url
    global_config.save_openie = False
    llm = CacheOpenAI(cache_dir=str(cache_dir), global_config=global_config)
    openie = OpenIE(llm_model=llm)

    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Reports: {[x.get('corpus_name') for x in selected]}")
    print(f"Mode: {'statements-only' if args.statements_only else 'full-page'} pack_chars={args.pack_chars}")

    combined: List[Dict[str, Any]] = []
    aborted = False

    for item in selected:
        doc_id = str(item["corpus_name"])
        out_path = by_report / f"{doc_id}_triples.json"
        if args.skip_existing and out_path.is_file():
            print(f"[{doc_id}] SKIP existing {out_path}")
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8"))
                combined.append(prev)
            except Exception:
                pass
            continue

        pages = split_pages(str(item.get("context") or ""))
        docs = build_docs(
            pages,
            statements_only=args.statements_only,
            pack_chars=args.pack_chars,
        )
        if args.max_docs is not None:
            docs = docs[: max(0, args.max_docs)]
        if not docs:
            print(f"[{doc_id}] no docs to process", file=sys.stderr)
            continue

        print(f"[{doc_id}] openie docs={len(docs)} (~{len(docs)*2} LLM calls)")

        # HippoRAG ChunkInfo-like dict
        chunks: Dict[str, Dict[str, Any]] = {}
        page_by_chunk: Dict[str, int] = {}
        for cid, page_no, passage in docs:
            chunks[cid] = {
                "num_tokens": max(1, len(passage) // 4),
                "content": passage,
                "chunk_order": [],
                "full_doc_ids": [doc_id],
            }
            page_by_chunk[cid] = page_no

        t0 = time.time()
        try:
            ner_dict, triple_dict = openie.batch_openie(chunks)
        except Exception as exc:
            print(f"[{doc_id}] FAIL batch_openie: {exc}", file=sys.stderr)
            if is_quota_error(exc):
                print("Stopping due to token/quota exhaustion.", file=sys.stderr)
                aborted = True
                break
            continue

        triples: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        pages_processed: List[Dict[str, Any]] = []

        for cid, page_no, _passage in docs:
            trip_out = triple_dict.get(cid)
            ner_out = ner_dict.get(cid)
            raw_triples = list(getattr(trip_out, "triples", None) or [])
            meta = dict(getattr(trip_out, "metadata", None) or {})
            if meta.get("error"):
                errors.append({"chunk_id": cid, "page": page_no, "error": meta["error"]})
                if is_quota_error(RuntimeError(str(meta["error"]))):
                    aborted = True
            kept = 0
            for t in raw_triples:
                nt = normalize_hippo_triple(t, page=page_no, chunk_id=cid)
                if nt:
                    triples.append(nt)
                    kept += 1
            pages_processed.append(
                {
                    "page": page_no,
                    "chunk_id": cid,
                    "entities": len(getattr(ner_out, "unique_entities", None) or []),
                    "triples": kept,
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens"),
                    "ok": "error" not in meta,
                }
            )
            if args.sleep:
                time.sleep(args.sleep)

        company = doc_id.split("-")[0]
        payload = {
            "document_id": doc_id,
            "company": company,
            "model": model,
            "extractor": "hipporag2_openie",
            "source_corpus": str(corpus_path).replace("\\", "/"),
            "statements_only": bool(args.statements_only),
            "pack_chars": args.pack_chars,
            "doc_count": len(docs),
            "triple_count": len(triples),
            "seconds": round(time.time() - t0, 2),
            "pages_processed": pages_processed,
            "errors": errors,
            "triples": triples,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        combined.append(payload)
        print(
            f"[{doc_id}] wrote {out_path} triples={len(triples)} "
            f"seconds={payload['seconds']}"
        )
        if aborted:
            print("Stopping due to token/quota exhaustion.", file=sys.stderr)
            break

    combined_path = out_dir / "esg_ocr_triples.json"
    reports_map: Dict[str, Dict[str, Any]] = {}
    if combined_path.is_file():
        try:
            prev = json.loads(combined_path.read_text(encoding="utf-8"))
            for r in prev.get("reports") or []:
                did = str(r.get("document_id") or "")
                if did:
                    reports_map[did] = r
        except Exception as exc:
            print(f"WARN: could not load existing combined: {exc}", file=sys.stderr)
    for r in combined:
        did = str(r.get("document_id") or "")
        if did:
            reports_map[did] = r
    # Also pick up any by_report files not in memory (e.g. partial run)
    for fp in sorted(by_report.glob("*_triples.json")):
        did = fp.name.replace("_triples.json", "")
        if did in reports_map:
            continue
        try:
            reports_map[did] = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
    reports_list = sorted(reports_map.values(), key=lambda x: str(x.get("document_id") or ""))
    combined_path.write_text(
        json.dumps(
            {
                "dataset": "esg_ocr",
                "extractor": "hipporag2_openie",
                "model": model,
                "report_count": len(reports_list),
                "triple_count": sum(int(x.get("triple_count") or 0) for x in reports_list),
                "reports": reports_list,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {combined_path} reports={len(reports_list)} "
        f"triples={sum(int(x.get('triple_count') or 0) for x in reports_list)}"
    )
    return 3 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
