#!/usr/bin/env python3
"""Answer GraphRAG-Bench blind questions with LightRAG's native hybrid QA.

Uses each report's prebuilt LightRAG index (from extract_lightrag_triples.py)
and LightRAG.aquery / aquery_llm — NOT the Top-k triple token-overlap path.

Embedding at query time MUST match index build (gpt-5.1 extract used
text-embedding-3-small, dim=1536 via OpenAI). Set LIGHTRAG_EMBEDDING_* if
the answer LLM is on a different provider (e.g. DashScope Qwen).

Example:
  python run_lightrag_qa.py --all \\
    --index-root /workspace/ESG_KG/output/triples/LightRAG/gpt-5.1/indexes \\
    --out /workspace/ESG_KG/output/qa/lightrag-native-gpt5.1-qwen3-8b/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        "LIGHTRAG_EMBEDDING_DIM",
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


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "qa_answer_from_lightrag.md"


def load_eval_system_prompt() -> str:
    if _PROMPT_PATH.is_file():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Answer ONLY from the retrieved LightRAG context. "
        "If insufficient, reply exactly: UNKNOWN. "
        "Use ONE short English phrase/sentence (≤ 20 words) ending with a period."
    )


_TIKTOKEN_ENC = None


def count_tokens(text: str) -> Tuple[int, bool]:
    """Return (n_tokens, estimated). Prefer tiktoken cl100k; fall back to chars/4."""
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


def estimate_tokens(text: str) -> int:
    """Backward-compatible helper; prefer count_tokens() for new code."""
    n, _ = count_tokens(text)
    return n


# Keep stored context from exploding results.json; token counts use the full prompt.
_CONTEXT_STORE_MAX_CHARS = 50000


class LightRAGIndexPool:
    """Lazy per-document LightRAG instances under index-root/{document_id}."""

    def __init__(self, index_root: Path, settings):
        self.index_root = Path(index_root)
        self.settings = settings
        self._cache: Dict[str, Any] = {}

    async def get(self, document_id: str):
        if document_id in self._cache:
            return self._cache[document_id]
        working = self.index_root / document_id
        if not working.is_dir():
            raise FileNotFoundError(f"LightRAG index missing: {working}")
        backend_src = Path(__file__).resolve().parents[1] / "src"
        if str(backend_src) not in sys.path:
            sys.path.insert(0, str(backend_src))
        from esg_encoding.retrieval.lightrag.factory import create_lightrag
        from lightrag.kg.shared_storage import initialize_pipeline_status

        rag = create_lightrag(working, settings=self.settings, workspace=document_id)
        await rag.initialize_storages()
        try:
            await initialize_pipeline_status()
        except Exception:
            pass
        self._cache[document_id] = rag
        return rag

    def _query_param(
        self,
        *,
        mode: str,
        top_k: int,
        chunk_top_k: int,
        only_need_context: bool,
        only_need_prompt: bool,
        user_prompt: Optional[str],
        max_entity_tokens: int,
        max_relation_tokens: int,
        max_total_tokens: int,
    ):
        from lightrag import QueryParam

        kwargs: Dict[str, Any] = {
            "mode": mode,
            "only_need_context": only_need_context,
            "only_need_prompt": only_need_prompt,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "enable_rerank": False,
            "max_entity_tokens": max_entity_tokens,
            "max_relation_tokens": max_relation_tokens,
            "max_total_tokens": max_total_tokens,
        }
        if user_prompt:
            kwargs["user_prompt"] = user_prompt
        return QueryParam(**kwargs)

    async def native_answer(
        self,
        document_id: str,
        question: str,
        *,
        mode: str,
        top_k: int,
        chunk_top_k: int,
        system_prompt: Optional[str],
        user_prompt: Optional[str],
        max_entity_tokens: int,
        max_relation_tokens: int,
        max_total_tokens: int,
        capture_prompt: bool,
        extra_only_need_prompt: bool = False,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Run LightRAG native hybrid QA. Returns (answer, prompt_or_context, usage_meta).

        By default records the real final-answer prompt via QA_PROMPT_CAPTURE inside
        the single aquery_llm path (no extra retrieval round). Optional
        extra_only_need_prompt restores the slower double-round debug path.
        """
        from esg_encoding.retrieval.lightrag.factory import QA_PROMPT_CAPTURE

        rag = await self.get(document_id)
        prompt_text = ""
        capture_source = None
        api_pt = None
        api_ct = None

        if extra_only_need_prompt:
            p_param = self._query_param(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                only_need_context=False,
                only_need_prompt=True,
                user_prompt=user_prompt,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens,
                max_total_tokens=max_total_tokens,
            )
            prompt_raw = await rag.aquery(question, param=p_param, system_prompt=system_prompt)
            prompt_text = prompt_raw if isinstance(prompt_raw, str) else json.dumps(prompt_raw, ensure_ascii=False)
            capture_source = "only_need_prompt"

        param = self._query_param(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            only_need_context=False,
            only_need_prompt=False,
            user_prompt=user_prompt,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
        )

        bucket: Dict[str, Any] = {}
        token = QA_PROMPT_CAPTURE.set(bucket) if capture_prompt else None
        try:
            if hasattr(rag, "aquery_llm"):
                result = await rag.aquery_llm(question, param=param, system_prompt=system_prompt)
                llm_resp = (result or {}).get("llm_response") or {}
                answer = str(llm_resp.get("content") or "").strip()
                data = (result or {}).get("data") or {}
            else:
                answer_raw = await rag.aquery(question, param=param, system_prompt=system_prompt)
                answer = answer_raw if isinstance(answer_raw, str) else json.dumps(answer_raw, ensure_ascii=False)
                answer = answer.strip()
                data = {}
        finally:
            if token is not None:
                QA_PROMPT_CAPTURE.reset(token)

        if capture_prompt and bucket.get("prompt_text"):
            prompt_text = str(bucket["prompt_text"])
            capture_source = "answer_llm_intercept"
            if isinstance(bucket.get("api_prompt_tokens"), (int, float)):
                api_pt = int(bucket["api_prompt_tokens"])
            if isinstance(bucket.get("api_completion_tokens"), (int, float)):
                api_ct = int(bucket["api_completion_tokens"])
        elif capture_prompt and not prompt_text:
            # Answer-path cache hit: LLM wrapper never ran. One prompt-only round for metering.
            p_param = self._query_param(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                only_need_context=False,
                only_need_prompt=True,
                user_prompt=user_prompt,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens,
                max_total_tokens=max_total_tokens,
            )
            prompt_raw = await rag.aquery(question, param=p_param, system_prompt=system_prompt)
            prompt_text = (
                prompt_raw if isinstance(prompt_raw, str) else json.dumps(prompt_raw, ensure_ascii=False)
            )
            capture_source = "only_need_prompt_cache_fallback"
        elif not prompt_text and data:
            # Inspection fallback only — not a real LLM prompt.
            for key in ("prompt", "context", "raw_data"):
                if isinstance(data.get(key), str) and data[key].strip():
                    prompt_text = data[key]
                    capture_source = f"data.{key}"
                    break
            if not prompt_text:
                prompt_text = json.dumps(data, ensure_ascii=False)
                capture_source = "data.json"

        # Prefer provider usage for the final answer call when available; else tiktoken.
        pt_tik, pt_est = count_tokens(prompt_text) if prompt_text else (None, False)
        ct_tik, ct_est = count_tokens(answer) if answer else (None, False)
        if api_pt is not None:
            pt, pt_est, pt_src = api_pt, False, "api"
        elif pt_tik is not None and capture_source in (
            "answer_llm_intercept",
            "only_need_prompt",
            "only_need_prompt_cache_fallback",
        ):
            pt, pt_src = pt_tik, "tiktoken_cl100k"
        else:
            pt, pt_src = pt_tik, "tiktoken_or_estimate_untrusted"

        if api_ct is not None:
            ct, ct_est, ct_src = api_ct, False, "api"
        else:
            ct, ct_src = ct_tik, "tiktoken_cl100k" if not ct_est else "chars/4"

        prompt_is_real = capture_source in (
            "answer_llm_intercept",
            "only_need_prompt",
            "only_need_prompt_cache_fallback",
        )
        usage = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "prompt_tokens_estimated": bool(pt_est or (prompt_text and not prompt_is_real)),
            "completion_tokens_estimated": bool(ct_est and api_ct is None),
            "tokenizer": "api" if pt_src == "api" else ("cl100k_base" if not pt_est else "chars/4"),
            "prompt_chars": len(prompt_text) if prompt_text else 0,
            "prompt_captured": prompt_is_real,
            "prompt_capture_source": capture_source,
            "prompt_tokens_tiktoken": pt_tik,
            "completion_tokens_tiktoken": ct_tik,
            "prompt_tokens_source": pt_src,
            "completion_tokens_source": ct_src,
        }
        stored_context = prompt_text
        if stored_context and len(stored_context) > _CONTEXT_STORE_MAX_CHARS:
            stored_context = stored_context[:_CONTEXT_STORE_MAX_CHARS]
        return answer, stored_context, usage


def main() -> int:
    root = detect_newesg_root()
    default_index = root / "ESG_KG" / "output" / "triples" / "LightRAG" / "gpt-5.1" / "indexes"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-root", default=str(default_index))
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
            root / "ESG_KG" / "output" / "qa" / "lightrag-native-gpt5.1-qwen3-8b" / "results.json"
        ),
    )
    p.add_argument("--model", default=None, help="Answer/keyword LLM (default: LLM_MODEL)")
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="Override LLM_BASE_URL (e.g. https://api.openai.com/v1 for gpt-4o-mini)",
    )
    p.add_argument(
        "--llm-api-key",
        default=None,
        help="Override LLM_API_KEY (use OpenAI key when answering with GPT)",
    )
    p.add_argument(
        "--mode",
        default="hybrid",
        help="LightRAG query mode (paper: hybrid). Options: naive|local|global|hybrid|mix",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="LightRAG official default (DEFAULT_TOP_K=40). GraphRAG-Bench paper used 30.",
    )
    p.add_argument("--chunk-top-k", type=int, default=20, help="LightRAG official DEFAULT_CHUNK_TOP_K=20")
    p.add_argument(
        "--max-entity-tokens",
        type=int,
        default=6000,
        help="LightRAG official DEFAULT_MAX_ENTITY_TOKENS=6000",
    )
    p.add_argument(
        "--max-relation-tokens",
        type=int,
        default=8000,
        help="LightRAG official DEFAULT_MAX_RELATION_TOKENS=8000",
    )
    p.add_argument(
        "--max-total-tokens",
        type=int,
        default=30000,
        help="LightRAG official DEFAULT_MAX_TOTAL_TOKENS=30000",
    )
    p.add_argument(
        "--eval-format",
        action="store_true",
        default=True,
        help="Pass GraphRAG-Bench short-answer system prompt into LightRAG generation",
    )
    p.add_argument(
        "--no-eval-format",
        action="store_true",
        help="Use LightRAG default response style (Multiple Paragraphs)",
    )
    p.add_argument(
        "--capture-prompt",
        action="store_true",
        default=True,
        help="Record real final-answer prompt tokens via LLM intercept (no extra retrieval round)",
    )
    p.add_argument("--no-capture-prompt", action="store_true")
    p.add_argument(
        "--extra-only-need-prompt",
        action="store_true",
        help="Debug: also run only_need_prompt (extra keyword+retrieval round, slower)",
    )
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Disable LightRAG LLM response cache (safer when multiple QA processes share indexes)",
    )
    # Embedding overrides (must match index)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    args = p.parse_args()

    cfg = load_dotenv()
    model = args.model or cfg.get("LLM_MODEL") or "qwen3-8b"
    api_key = args.llm_api_key or cfg.get("LLM_API_KEY") or ""
    base_url = args.llm_base_url or cfg.get("LLM_BASE_URL") or None
    # Answer with GPT while .env LLM_* is DashScope: auto-route to OpenAI embedding creds
    if not args.llm_base_url and str(model).lower().startswith("gpt-"):
        base_url = cfg.get("LIGHTRAG_EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
        if not args.llm_api_key:
            api_key = (
                cfg.get("LIGHTRAG_EMBEDDING_API_KEY")
                or cfg.get("OPENAI_API_KEY")
                or api_key
            )
    if not api_key:
        print("LLM_API_KEY missing", file=sys.stderr)
        return 2

    emb_model = (
        args.embedding_model
        or cfg.get("LIGHTRAG_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    emb_dim = int(
        args.embedding_dim
        or cfg.get("LIGHTRAG_EMBEDDING_DIM")
        or 1536
    )
    emb_base = (
        args.embedding_base_url
        or cfg.get("LIGHTRAG_EMBEDDING_BASE_URL")
        or "https://api.openai.com/v1"
    )
    emb_key = (
        args.embedding_api_key
        or cfg.get("LIGHTRAG_EMBEDDING_API_KEY")
        or cfg.get("OPENAI_API_KEY")
        or ""
    )
    if not emb_key:
        print(
            "Embedding API key missing. Index was built with OpenAI text-embedding-3-small (1536). "
            "Set LIGHTRAG_EMBEDDING_API_KEY or OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    use_eval = bool(args.eval_format) and not bool(args.no_eval_format)
    capture_prompt = bool(args.capture_prompt) and not bool(args.no_capture_prompt)
    extra_only_need_prompt = bool(args.extra_only_need_prompt)
    system_prompt = load_eval_system_prompt() if use_eval else None

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = model
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["LLM_BASE_URL"] = base_url
    os.environ["LIGHTRAG_EMBEDDING_MODEL"] = emb_model
    os.environ["LIGHTRAG_EMBEDDING_DIM"] = str(emb_dim)
    os.environ["LIGHTRAG_EMBEDDING_BASE_URL"] = emb_base
    os.environ["LIGHTRAG_EMBEDDING_API_KEY"] = emb_key

    backend_src = Path(__file__).resolve().parents[1] / "src"
    if str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))
    from dataclasses import replace

    from esg_encoding.retrieval.lightrag.settings import LightRAGSettings

    settings = replace(
        LightRAGSettings(),
        enabled=True,
        query_mode=args.mode,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        llm_model_name=model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        embedding_model_name=emb_model,
        embedding_dim=emb_dim,
        embedding_base_url=emb_base,
        embedding_api_key=emb_key,
        enable_llm_cache=not bool(args.no_llm_cache),
    )

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions") or questions.get("data") or []
    gold_by_id: Dict[str, Any] = {}
    gold_path = Path(args.gold)
    if gold_path.is_file():
        gold_raw = json.loads(gold_path.read_text(encoding="utf-8"))
        if isinstance(gold_raw, list):
            for g in gold_raw:
                if isinstance(g, dict) and g.get("id"):
                    gold_by_id[str(g["id"])] = g
        elif isinstance(gold_raw, dict):
            for g in gold_raw.get("questions") or gold_raw.get("items") or []:
                if isinstance(g, dict) and g.get("id"):
                    gold_by_id[str(g["id"])] = g

    if args.source:
        questions = [q for q in questions if str(q.get("source") or "") == args.source]
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and out_path.is_file():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            done_ids = {str(r.get("id")) for r in results if r.get("id")}
            print(f"Resume: {len(done_ids)} already answered")
        except Exception:
            results = []

    pool = LightRAGIndexPool(Path(args.index_root), settings)
    index_root = Path(args.index_root)
    graph_backend = "lightrag-native"

    print(f"Answer model: {model}")
    print(f"LLM base: {base_url}")
    print(f"Embedding: {emb_model} dim={emb_dim} base={emb_base}")
    print(f"Index root: {index_root}")
    print(f"Query mode: {args.mode} top_k={args.top_k} (native LightRAG aquery)")
    print(f"Eval format prompt: {use_eval}")
    print(f"Capture prompt tokens: {capture_prompt} (intercept single answer call)")
    print(f"Extra only_need_prompt round: {extra_only_need_prompt}")
    print(f"LLM response cache: {not bool(args.no_llm_cache)}")
    print(f"Questions: {len(questions)}")

    t0 = time.time()
    new_n = 0
    err_n = 0

    async def _one(q: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(q.get("id") or "")
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
            "graph_backend": graph_backend,
            "triple_source": "LightRAG/gpt-5.1",
            "retrieval": {
                "mode": args.mode,
                "top_k": args.top_k,
                "chunk_top_k": args.chunk_top_k,
                "native": True,
                "max_entity_tokens": args.max_entity_tokens,
                "max_relation_tokens": args.max_relation_tokens,
                "max_total_tokens": args.max_total_tokens,
            },
            "usage": {},
            "error": None,
        }
        try:
            ans, prompt_text, usage = await pool.native_answer(
                source,
                question,
                mode=args.mode,
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                system_prompt=system_prompt,
                user_prompt=None,
                max_entity_tokens=args.max_entity_tokens,
                max_relation_tokens=args.max_relation_tokens,
                max_total_tokens=args.max_total_tokens,
                capture_prompt=capture_prompt,
                extra_only_need_prompt=extra_only_need_prompt,
            )
            row["generated_answer"] = ans or "UNKNOWN"
            row["context"] = prompt_text
            row["usage"] = usage
        except Exception as exc:
            row["error"] = str(exc)
            row["generated_answer"] = "UNKNOWN"
        return row

    async def _run_all() -> None:
        nonlocal new_n, err_n, results
        for i, q in enumerate(questions, 1):
            qid = str(q.get("id") or "")
            if qid in done_ids:
                continue
            row = await _one(q)
            results.append(row)
            new_n += 1
            if row.get("error"):
                err_n += 1
                print(f"ERROR {qid}: {row['error'][:300]}", file=sys.stderr)
            if args.sleep:
                await asyncio.sleep(args.sleep)
            if new_n % max(1, args.save_every) == 0 or i == len(questions):
                out_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                elapsed = time.time() - t0
                pt = (row.get("usage") or {}).get("prompt_tokens")
                est = (row.get("usage") or {}).get("prompt_tokens_estimated")
                cap = (row.get("usage") or {}).get("prompt_captured")
                src = (row.get("usage") or {}).get("prompt_tokens_source")
                print(
                    f"[{len(results)}/{len(questions)}] answered={len(results)} "
                    f"errors={err_n} elapsed={elapsed:.1f}s last={qid} "
                    f"prompt_tokens={pt} source={src} captured={cap} estimated={est}"
                )

    asyncio.run(_run_all())
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Done. wrote {len(results)} rows → {out_path} "
        f"(new={new_n}, errors={err_n}, seconds={time.time() - t0:.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
