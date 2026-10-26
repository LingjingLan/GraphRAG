#!/usr/bin/env python3
"""Extract triples with LightRAG from Datasets/Corpus/esg_ocr.json.

Document prep mirrors extract_hipporag_triples.py:
  - statements-only pages by default
  - pack_chars=0 → one page per doc
  - same page split / statements block parsing

Writes:
  ESG_KG/output/triples/LightRAG/{model}/by_report/{doc}_triples.json
  ESG_KG/output/triples/LightRAG/{model}/esg_ocr_triples.json
  ESG_KG/output/triples/LightRAG/{model}/indexes/{doc}/   (LightRAG working_dir)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import replace
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


def build_docs(
    pages: List[Tuple[int, str]],
    *,
    statements_only: bool,
    pack_chars: int,
) -> List[Tuple[str, int, str]]:
    """Same contract as extract_hipporag_triples.build_docs: (chunk_id, page, passage)."""
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


def docs_to_lightrag_passages(docs: List[Tuple[str, int, str]]) -> List[str]:
    """Embed page/chunk markers so exported edges keep page provenance."""
    out: List[str] = []
    for cid, page_no, passage in docs:
        out.append(
            f"__LIGHTRAG_SEGMENT_IDS__: {cid}\n"
            f"__LIGHTRAG_PAGES__: {page_no}\n\n"
            f"[{cid} - Page {page_no}]\n{passage}".strip()
        )
    return out


def out_name_from_model(model: str) -> str:
    """gpt-5.1-2025-11-13 -> gpt-5.1 (match hipporag folder naming)."""
    m = (model or "").strip()
    if m.startswith("gpt-5.1"):
        return "gpt-5.1"
    if m.startswith("gpt-5.4"):
        return "gpt-5.4"
    if m.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    return re.sub(r"[^\w.\-]+", "_", m) or "model"


def is_quota_error(exc: BaseException | str) -> bool:
    """Same stop condition as extract_hipporag_triples.py."""
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
            "rate_limit_exceeded",
            "too many requests",
            "429",
        )
    )


async def index_one(
    *,
    doc_id: str,
    passages: List[str],
    index_dir: Path,
    settings,
) -> List[Dict[str, Any]]:
    backend_src = Path(__file__).resolve().parents[1] / "src"
    if str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))

    from esg_encoding.retrieval.lightrag.export_triples import export_triples_from_working_dir
    from esg_encoding.retrieval.lightrag.factory import create_lightrag

    index_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale store for this report so reruns are clean.
    for p in index_dir.glob("*"):
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil

                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    rag = create_lightrag(index_dir, settings=settings, workspace=doc_id)
    await rag.initialize_storages()
    await rag.ainsert(passages)
    try:
        await rag.aexport_data(str(index_dir / "export.csv"), file_format="csv")
    except Exception as exc:
        print(f"[{doc_id}] export csv skipped: {exc}")
    await rag.finalize_storages()
    return export_triples_from_working_dir(index_dir, source="lightrag")


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(root / "Datasets" / "Corpus" / "esg_ocr.json"))
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: ESG_KG/output/triples/LightRAG/{model_short}",
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
        help="Pack pages into ~N-char docs (0 = one page per doc; same as HippoRAG default)",
    )
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    if args.all_pages:
        args.statements_only = False

    cfg = load_dotenv()
    model = args.model or cfg.get("LLM_MODEL") or "gpt-5.1-2025-11-13"
    api_key = cfg.get("LLM_API_KEY") or ""
    base_url = cfg.get("LLM_BASE_URL") or None
    if not api_key:
        print("LLM_API_KEY missing", file=sys.stderr)
        return 2

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = model
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["LLM_BASE_URL"] = base_url

    short = out_name_from_model(model)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "ESG_KG" / "output" / "triples" / "LightRAG" / short)
    by_report = out_dir / "by_report"
    indexes = out_dir / "indexes"
    by_report.mkdir(parents=True, exist_ok=True)
    indexes.mkdir(parents=True, exist_ok=True)

    backend_src = Path(__file__).resolve().parents[1] / "src"
    if str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))
    from esg_encoding.retrieval.lightrag.export_triples import write_report_triples
    from esg_encoding.retrieval.lightrag.settings import LightRAGSettings

    settings = replace(
        LightRAGSettings(),
        enabled=True,
        llm_model_name=model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        # Keep LightRAG entity/relation language aligned with HippoRAG ESG English corpus.
    )

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

    print(f"Extractor: lightrag")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Out: {out_dir}")
    print(f"Reports: {[x.get('corpus_name') for x in selected]}")
    print(
        f"Mode: {'statements-only' if args.statements_only else 'full-page'} "
        f"pack_chars={args.pack_chars} (HippoRAG-aligned)"
    )

    combined: List[Dict[str, Any]] = []
    aborted = False
    for item in selected:
        doc_id = str(item["corpus_name"])
        out_path = by_report / f"{doc_id}_triples.json"
        index_dir = indexes / doc_id
        if args.skip_existing and out_path.is_file():
            print(f"[{doc_id}] SKIP existing {out_path}")
            try:
                combined.append(json.loads(out_path.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue

        pages = split_pages(str(item.get("context") or ""))
        docs = build_docs(pages, statements_only=args.statements_only, pack_chars=args.pack_chars)
        if args.max_docs is not None:
            docs = docs[: max(0, args.max_docs)]
        if not docs:
            print(f"[{doc_id}] no docs to process", file=sys.stderr)
            continue

        passages = docs_to_lightrag_passages(docs)
        print(f"[{doc_id}] lightrag docs={len(passages)}")
        t0 = time.time()
        try:
            triples = asyncio.run(
                index_one(
                    doc_id=doc_id,
                    passages=passages,
                    index_dir=index_dir,
                    settings=settings,
                )
            )
        except Exception as exc:
            print(f"[{doc_id}] FAILED: {exc}", file=sys.stderr)
            write_report_triples(
                out_path,
                document_id=doc_id,
                company=str(item.get("company") or doc_id.split("-")[0]),
                model=model,
                triples=[],
                seconds=time.time() - t0,
                extra={
                    "statements_only": args.statements_only,
                    "pack_chars": args.pack_chars,
                    "doc_count": len(docs),
                    "index_dir": str(index_dir),
                    "errors": [str(exc)],
                },
            )
            if is_quota_error(exc):
                print("Stopping due to token/quota exhaustion.", file=sys.stderr)
                aborted = True
                break
            if args.sleep:
                time.sleep(args.sleep)
            continue

        seconds = time.time() - t0
        # Stamp page from chunk markers when GraphML lacks page text.
        page_by_cid = {cid: page for cid, page, _ in docs}
        for t in triples:
            cid = str(t.get("chunk_id") or "")
            if cid in page_by_cid:
                t.setdefault("qualifiers", {})["page"] = str(page_by_cid[cid])
            t["source"] = "lightrag"

        company = str(item.get("company") or doc_id.split("-")[0])
        payload = write_report_triples(
            out_path,
            document_id=doc_id,
            company=company,
            model=model,
            triples=triples,
            seconds=seconds,
            extra={
                "statements_only": args.statements_only,
                "pack_chars": args.pack_chars,
                "doc_count": len(docs),
                "pages_processed": sorted({p for _, p, _ in docs}),
                "index_dir": str(index_dir),
            },
        )
        combined.append(payload)
        print(f"[{doc_id}] triples={len(triples)} seconds={seconds:.1f} -> {out_path}")
        if args.sleep:
            time.sleep(args.sleep)

        # Refresh combined after each report (crash-safe).
        (out_dir / "esg_ocr_triples.json").write_text(
            json.dumps(
                {
                    "dataset": "esg_ocr",
                    "extractor": "lightrag",
                    "model": model,
                    "report_count": len(combined),
                    "triple_count": sum(int(r.get("triple_count") or 0) for r in combined),
                    "reports": combined,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        f"Done. reports={len(combined)} triples="
        f"{sum(int(r.get('triple_count') or 0) for r in combined)} "
        f"-> {out_dir / 'esg_ocr_triples.json'}"
    )
    return 3 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
