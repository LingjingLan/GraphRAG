#!/usr/bin/env python3
"""Answer GraphRAG-Bench blind questions via DualChannelRetriever on esg_ocr.json.

Mirrors the normal client SASB path (not GraphRAG chunk_token_size):
  OCR markdown → ContentExtractor markdown-block TextSegments
  → embed → DualChannelRetriever (BM25 + dense + fusion)
  → pack evidence with ~1200-char truncation (like disclosure_inference)
  → gpt-4o-mini answer.

Example:
  python run_dualchannel_qa.py --resume \\
    --model gpt-4o-mini \\
    --out /workspace/ESG_KG/output/qa/DualChannel-GPT4omini/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PAGE_SPLIT_RE = re.compile(r"(?m)^## Page (\d+)\s*$")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qa_answer_from_dualchannel.md"


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
    keys = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY", "LIGHTRAG_EMBEDDING_API_KEY"]
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


def statements_block(body: str) -> str:
    m = re.search(r"(?ms)^### Statements\n(.*?)(?=^### |\Z)", body)
    return (m.group(1) if m else "").strip()


def load_eval_system_prompt() -> str:
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Answer ONLY from the retrieved DualChannel evidence. "
        "If insufficient, reply exactly: UNKNOWN. "
        "Use ONE short English phrase/sentence (≤ 20 words) ending with a period."
    )


_TIKTOKEN_ENC = None


def count_tokens(text: str) -> Tuple[int, bool]:
    if not text:
        return 0, False
    global _TIKTOKEN_ENC
    try:
        if _TIKTOKEN_ENC is None:
            import tiktoken

            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        return len(_TIKTOKEN_ENC.encode(text)), False
    except Exception:
        return max(1, len(text) // 4), True


def _tiktoken_enc(model_name: str = "text-embedding-3-small"):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def normalize_ocr_page_markers(context: str) -> str:
    """Map corpus `## Page N` headings to client HTML page markers."""
    return re.sub(r"(?m)^## Page (\d+)\s*$", r"<!-- Page \1 -->", context or "")


def truncate_evidence(text: str, max_chars: int = 1200) -> str:
    """Same default as DisclosureInferenceEngine._truncate_segment_text."""
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def build_document_content(
    doc_id: str,
    context: str,
    *,
    statements_only: bool,
    config=None,
):
    """Build segments the same way as client upload: markdown blocks + tables."""
    from esg_encoding.content_extractor import ContentExtractor
    from esg_encoding.models import DocumentContent, ProcessingConfig

    md = normalize_ocr_page_markers(context)
    if statements_only:
        # Optional GraphRAG-style filter: keep only ### Statements bodies.
        pages = split_pages(context)
        kept: List[str] = []
        for page_no, body in pages:
            text = statements_block(body)
            if text and text.lower() != "(none)":
                kept.append(f"<!-- Page {page_no} -->\n{text}")
        md = "\n\n".join(kept)

    extractor = ContentExtractor(config or ProcessingConfig())
    segments = extractor._segments_from_markdown(md, doc_id)
    return DocumentContent(
        document_id=doc_id,
        file_path=f"ocr://{doc_id}",
        segments=segments,
        markdown_content=context or "",
    )


def openai_embed_texts(client, model: str, texts: List[str], *, batch_size: int = 64):
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs: List[List[float]] = []
    for i in range(0, len(texts), max(1, int(batch_size))):
        batch = [t if (t or "").strip() else " " for t in texts[i : i + batch_size]]
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda x: x.index)
        for row in ordered:
            v = np.asarray(row.embedding, dtype=np.float32)
            n = float(np.linalg.norm(v) + 1e-12)
            vecs.append((v / n).tolist())
    return np.asarray(vecs, dtype=np.float32)


class OpenAIContentEmbedder:
    """Drop-in embedder for DualChannel using OpenAI embeddings API."""

    def __init__(self, client, model: str, *, batch_size: int = 64, embed_max_chars: int = 6000):
        self.client = client
        self.model = model
        self.batch_size = int(batch_size)
        self.embed_max_chars = int(embed_max_chars)

    def embed_document(self, document_content):
        from esg_encoding.models import ReportContent, SegmentEmbedding

        # Truncate only the embedding input; segment.content stays full for BM25.
        texts = []
        for s in document_content.segments:
            t = str(s.content or "")
            if self.embed_max_chars and len(t) > self.embed_max_chars:
                t = t[: self.embed_max_chars]
            texts.append(t)
        mat = openai_embed_texts(self.client, self.model, texts, batch_size=self.batch_size)
        embeddings = [
            SegmentEmbedding(segment_id=seg.segment_id, embedding=mat[i].tolist())
            for i, seg in enumerate(document_content.segments)
        ]
        return ReportContent(
            document_id=document_content.document_id,
            document_content=document_content,
            embeddings=embeddings,
        )


def install_openai_query_encoder(client, model: str, *, batch_size: int = 64) -> None:
    """Make SemanticRetriever query encoding use the same OpenAI model as docs."""
    import esg_encoding.retrieval.semantic as semantic_mod
    import esg_encoding.shared_embedding_model as shared_mod

    def _encode(model_obj, texts, model_name_or_path=None, normalize_embeddings=True, **kwargs):
        return openai_embed_texts(client, model, list(texts), batch_size=batch_size)

    shared_mod.encode_query_texts = _encode  # type: ignore[attr-defined]
    semantic_mod.encode_query_texts = _encode  # type: ignore[attr-defined]

    def _init_noop(self):
        self.embedding_model = "openai-stub"

    semantic_mod.SemanticRetriever._init_embedding_model = _init_noop  # type: ignore[method-assign]


class DualChannelReportPool:
    def __init__(
        self,
        corpus_by_id: Dict[str, str],
        *,
        cache_root: Path,
        statements_only: bool,
        config,
        embedder,
        embed_backend: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        segment_mode: str = "md_blocks",
    ):
        self.corpus_by_id = corpus_by_id
        self.cache_root = Path(cache_root)
        self.statements_only = statements_only
        self.config = config
        self.embedder = embedder
        self.embed_backend = str(embed_backend)
        self.embedding_model = str(embedding_model)
        self.segment_mode = str(segment_mode)
        self._cache: Dict[str, Any] = {}
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_ok(self, meta_obj: Dict[str, Any]) -> bool:
        return (
            bool(meta_obj.get("statements_only")) == bool(self.statements_only)
            and str(meta_obj.get("segment_mode") or "") == self.segment_mode
            and str(meta_obj.get("embed_backend") or "") == self.embed_backend
            and str(meta_obj.get("embedding_model") or "") == self.embedding_model
        )

    def get(self, document_id: str):
        if document_id in self._cache:
            return self._cache[document_id]
        from esg_encoding.models import ReportContent, SegmentEmbedding

        cache_dir = self.cache_root / document_id
        pkl = cache_dir / "report_content.pkl"
        meta = cache_dir / "index_meta.json"
        if pkl.is_file() and meta.is_file():
            try:
                meta_obj = json.loads(meta.read_text(encoding="utf-8"))
                if self._cache_ok(meta_obj):
                    with pkl.open("rb") as f:
                        report = pickle.load(f)
                    self._cache[document_id] = report
                    return report
            except Exception:
                pass

        ctx = self.corpus_by_id.get(document_id)
        if ctx is None:
            raise FileNotFoundError(f"Corpus missing for {document_id}")
        doc = build_document_content(
            document_id,
            ctx,
            statements_only=self.statements_only,
            config=self.config,
        )
        if not doc.segments:
            raise ValueError(f"No segments for {document_id}")
        print(
            f"[{document_id}] embedding segments={len(doc.segments)} "
            f"mode={self.segment_mode} "
            f"backend={self.embed_backend} model={self.embedding_model} ...",
            flush=True,
        )
        report: ReportContent = self.embedder.embed_document(doc)
        if report.embeddings and not isinstance(report.embeddings[0], SegmentEmbedding):
            report.embeddings = [
                SegmentEmbedding(segment_id=e.segment_id, embedding=list(e.embedding))
                for e in report.embeddings
            ]
        cache_dir.mkdir(parents=True, exist_ok=True)
        with pkl.open("wb") as f:
            pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta.write_text(
            json.dumps(
                {
                    "method": "dualchannel",
                    "document_id": document_id,
                    "statements_only": bool(self.statements_only),
                    "segment_count": len(doc.segments),
                    "segment_mode": self.segment_mode,
                    "embed_backend": self.embed_backend,
                    "embedding_model": self.embedding_model,
                    "device": str(getattr(self.config, "device", "")),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._cache[document_id] = report
        return report


def _make_runtime_config(base_config, *, top_k: int, use_keyword: bool, use_semantic: bool, use_rerank: bool):
    """ProcessingConfig is a strict Pydantic model; DualChannel reads optional flags via getattr."""
    from types import SimpleNamespace

    data = base_config.model_dump() if hasattr(base_config, "model_dump") else dict(base_config.__dict__)
    data.update(
        {
            "top_k": int(top_k),
            "use_keyword_retrieval": bool(use_keyword),
            "use_semantic_retrieval": bool(use_semantic),
            "use_reranker": bool(use_rerank),
        }
    )
    return SimpleNamespace(**data)


def answer_one(
    *,
    pool: DualChannelReportPool,
    llm_client,
    model: str,
    document_id: str,
    question: str,
    top_k: int,
    context_k: int,
    evidence_max_chars: int,
    use_keyword: bool,
    use_semantic: bool,
    use_rerank: bool,
    system_prompt: Optional[str],
    response_type: str,
) -> Tuple[str, str, Dict[str, Any]]:
    from esg_encoding.content_extractor import enrich_document_with_pdf_links
    from esg_encoding.retrieval.dual_channel import DualChannelRetriever
    from esg_encoding.retrieval.evidence_retriever import _metric_from_spec

    report = pool.get(document_id)
    enrich_document_with_pdf_links(report.document_content)
    runtime_cfg = _make_runtime_config(
        pool.config,
        top_k=int(top_k),
        use_keyword=use_keyword,
        use_semantic=use_semantic,
        use_rerank=use_rerank,
    )
    metric = _metric_from_spec(None, question)
    retriever = DualChannelRetriever(runtime_cfg)
    result = retriever.retrieve_for_metric(report, metric)
    combined = list(getattr(result, "combined_results", None) or [])
    # Mirror SASB disclosure packing: DualChannel window + ~1200-char truncate per hit.
    passages = []
    for r in combined[: max(1, int(context_k))]:
        content = truncate_evidence(str(getattr(r, "content", "") or ""), int(evidence_max_chars))
        if content:
            page = getattr(r, "page_number", None)
            sid = getattr(r, "segment_id", "") or ""
            head = f"[Page {page} | {sid}]" if page is not None else f"[{sid}]"
            passages.append(f"{head}\n{content}")
    context = "\n\n".join(passages)
    if system_prompt:
        sys_content = (
            system_prompt.replace("{context_data}", context or "(no evidence)")
            .replace("{response_type}", response_type)
            .replace("{user_prompt}", "")
        )
    else:
        sys_content = f"Answer using only this context:\n\n{context}"

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    mlow = (model or "").lower()
    if mlow.startswith("gpt-5") or mlow.startswith("o1") or mlow.startswith("o3") or mlow.startswith("o4"):
        kwargs["max_completion_tokens"] = 512
    else:
        kwargs["max_tokens"] = 512
    resp = llm_client.chat.completions.create(**kwargs)
    answer = (resp.choices[0].message.content or "").strip()
    pt, pt_est = count_tokens((sys_content or "") + "\n" + question)
    ct, ct_est = count_tokens(answer)
    usage = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "prompt_tokens_estimated": pt_est,
        "completion_tokens_estimated": ct_est,
        "tokenizer": "cl100k_base" if not pt_est else "chars/4",
        "prompt_chars": len(sys_content or "") + len(question),
        "context_chars": len(context),
        "n_combined": len(combined),
        "n_context": len(passages),
        "evidence_max_chars": int(evidence_max_chars),
        "retrieval_types": [str(getattr(r, "retrieval_type", "")) for r in combined[: int(context_k)]],
    }
    return answer, context[:50000], usage


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(root / "Datasets" / "Corpus" / "esg_ocr.json"))
    p.add_argument("--questions", default=str(root / "Datasets" / "Questions" / "esg_questions_blind.json"))
    p.add_argument(
        "--out",
        default=str(root / "ESG_KG" / "output" / "qa" / "DualChannel-GPT4omini" / "results.json"),
    )
    p.add_argument(
        "--cache-root",
        default=str(root / "ESG_KG" / "output" / "indexes" / "dualchannel"),
    )
    p.add_argument("--model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--top-k", type=int, default=10, help="DualChannel base top_k (client default 10; adaptive window grows)")
    p.add_argument("--context-k", type=int, default=20, help="Max evidence hits packed into LLM (client SASB ~14-22)")
    p.add_argument(
        "--evidence-max-chars",
        type=int,
        default=1200,
        help="Per-evidence truncate, same as DisclosureInferenceEngine._truncate_segment_text",
    )
    p.add_argument("--statements-only", action="store_true", default=False)
    p.add_argument("--all-pages", action="store_true", default=True)
    p.add_argument(
        "--embed-backend",
        choices=["openai", "local"],
        default="openai",
        help="openai=text-embedding-3-small (fast); local=harrier/BGE (true client embedder)",
    )
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--device", default="cpu", help="Only used for --embed-backend local")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embed-max-chars", type=int, default=6000, help="Truncate embedding INPUT only (BM25 keeps full text)")
    p.add_argument("--no-keyword", action="store_true")
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--rerank", action="store_true", help="Enable FlagReranker (off by default; GPU-heavy)")
    p.add_argument("--eval-format", action="store_true", default=True)
    p.add_argument("--no-eval-format", action="store_true")
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--preembed-all", action="store_true", help="Embed all report sources before QA loop")
    args = p.parse_args()
    if args.statements_only:
        args.all_pages = False

    cfg = load_dotenv()
    model = args.model or cfg.get("LLM_MODEL") or "gpt-4o-mini"
    api_key = args.llm_api_key or cfg.get("LLM_API_KEY") or ""
    base_url = args.llm_base_url or cfg.get("LLM_BASE_URL") or None
    if not args.llm_base_url and str(model).lower().startswith("gpt-"):
        base_url = "https://api.openai.com/v1"
        if not args.llm_api_key:
            api_key = cfg.get("LIGHTRAG_EMBEDDING_API_KEY") or cfg.get("OPENAI_API_KEY") or api_key
    if not api_key:
        print("LLM_API_KEY missing", file=sys.stderr)
        return 2

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["LOCAL_EMBEDDINGS_DEVICE"] = str(args.device)
    os.environ.setdefault("BACKEND_LAZY_LOAD_MODELS", "1")

    backend_src = Path(__file__).resolve().parents[1] / "src"
    if str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))

    from openai import OpenAI

    from esg_encoding.models import ProcessingConfig

    config = ProcessingConfig(
        top_k=int(args.top_k),
        batch_size=int(args.batch_size),
        device=str(args.device),
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=model,
        embedding_model=str(args.embedding_model),
    )
    use_keyword = not bool(args.no_keyword)
    use_semantic = not bool(args.no_semantic)
    use_rerank = bool(args.rerank)

    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    if args.embed_backend == "openai":
        embedder = OpenAIContentEmbedder(
            llm_client,
            args.embedding_model,
            batch_size=int(args.batch_size),
            embed_max_chars=int(args.embed_max_chars),
        )
        install_openai_query_encoder(llm_client, args.embedding_model, batch_size=int(args.batch_size))
        cache_root = Path(args.cache_root) / "openai_mdblocks"
    else:
        from esg_encoding.content_embedder import ContentEmbedder

        embedder = ContentEmbedder(config)
        cache_root = Path(args.cache_root) / "local_mdblocks"

    corpus_items = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    corpus_by_id = {
        str(item.get("corpus_name") or ""): str(item.get("context") or "")
        for item in corpus_items
        if item.get("corpus_name")
    }

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.source:
        questions = [q for q in questions if str(q.get("source") or "") == args.source]
    if args.limit is not None:
        questions = questions[: max(0, int(args.limit))]

    use_eval = bool(args.eval_format) and not bool(args.no_eval_format)
    system_prompt = load_eval_system_prompt() if use_eval else None
    response_type = (
        "One short English phrase or sentence (≤20 words), ending with a period."
        if use_eval
        else "Multiple Paragraphs"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume and out_path.is_file():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            for r in results:
                if (
                    isinstance(r, dict)
                    and r.get("id")
                    and not r.get("error")
                    and (r.get("generated_answer") or "")
                ):
                    done_ids.add(str(r["id"]))
            print(f"Resume: {len(done_ids)} successful ids already present")
        except Exception as exc:
            print(f"Resume load failed: {exc}", file=sys.stderr)

    pool = DualChannelReportPool(
        corpus_by_id,
        cache_root=cache_root,
        statements_only=bool(args.statements_only),
        config=config,
        embedder=embedder,
        embed_backend=str(args.embed_backend),
        embedding_model=str(args.embedding_model),
        segment_mode="md_blocks",
    )

    print(
        f"DualChannel QA model={model} base={base_url} top_k={args.top_k} "
        f"context_k={args.context_k} evidence_max_chars={args.evidence_max_chars} "
        f"statements_only={args.statements_only} "
        f"keyword={use_keyword} semantic={use_semantic} rerank={use_rerank} "
        f"embed={args.embed_backend}:{args.embedding_model} segment_mode=md_blocks"
    )
    print(f"Questions={len(questions)} Out={out_path} Cache={cache_root}")

    if args.preembed_all:
        sources = sorted({str(q.get("source") or "") for q in questions if q.get("source")})
        print(f"Pre-embedding {len(sources)} reports...", flush=True)
        for sid in sources:
            pool.get(sid)
        print("Pre-embed done.", flush=True)

    total = len(questions)
    answered = 0
    for i, q in enumerate(questions, 1):
        qid = str(q.get("id") or "")
        if qid and qid in done_ids:
            continue
        doc_id = str(q.get("source") or q.get("document_id") or "")
        question = str(q.get("question") or q.get("query") or "").strip()
        if not doc_id or not question:
            continue
        t0 = time.time()
        try:
            answer, ctx, usage = answer_one(
                pool=pool,
                llm_client=llm_client,
                model=model,
                document_id=doc_id,
                question=question,
                top_k=int(args.top_k),
                context_k=int(args.context_k),
                evidence_max_chars=int(args.evidence_max_chars),
                use_keyword=use_keyword,
                use_semantic=use_semantic,
                use_rerank=use_rerank,
                system_prompt=system_prompt,
                response_type=response_type,
            )
            row = {
                "id": qid,
                "source": doc_id,
                "question": question,
                "question_type": q.get("question_type") or q.get("type"),
                "generated_answer": answer,
                "context": ctx,
                "usage": usage,
                "seconds": time.time() - t0,
                "method": "dualchannel",
                "model": model,
                "graph_backend": "dualchannel-keyword+semantic",
                "embed_backend": args.embed_backend,
                "embedding_model": args.embedding_model,
                "segment_mode": "md_blocks",
                "evidence_max_chars": int(args.evidence_max_chars),
            }
            answered += 1
            print(f"[{answered}/{total}] answered ok pt={usage.get('prompt_tokens')} last={qid}", flush=True)
        except Exception as exc:
            row = {
                "id": qid,
                "source": doc_id,
                "question": question,
                "error": str(exc),
                "generated_answer": "",
                "seconds": time.time() - t0,
                "method": "dualchannel",
                "model": model,
            }
            print(f"[{i}/{total}] FAILED {qid}: {exc}", file=sys.stderr, flush=True)

        results = [r for r in results if not (isinstance(r, dict) and str(r.get("id")) == qid)]
        results.append(row)
        if qid and not row.get("error") and row.get("generated_answer"):
            done_ids.add(qid)
        if answered % max(1, int(args.save_every)) == 0 or i == total:
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.sleep:
            time.sleep(args.sleep)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = sum(1 for r in results if r.get("generated_answer") and not r.get("error"))
    print(f"Done. ok={ok}/{len(results)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
