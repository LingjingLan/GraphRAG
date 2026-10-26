#!/usr/bin/env python3
"""One-shot HyperGraphRAG prompt metering via only_need_prompt.

Keeps llm_response_cache storage loaded (required by openai_complete model lookup)
but forces cache misses so only_need_prompt builds a real sys prompt.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


def load_dotenv() -> None:
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
            if k and v and not os.getenv(k):
                os.environ[k] = v
        break


async def main() -> int:
    load_dotenv()
    backend = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend / "src"))
    vendor = backend / "vendor" / "HyperGraphRAG"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    api_key = (
        os.getenv("LIGHTRAG_EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or ""
    )
    base = os.getenv("LIGHTRAG_EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = "gpt-4o-mini"
    os.environ["OPENAI_BASE_URL"] = base
    os.environ["LLM_BASE_URL"] = base
    os.environ.setdefault("LIGHTRAG_EMBEDDING_MODEL", "text-embedding-3-small")
    os.environ.setdefault("LIGHTRAG_EMBEDDING_DIM", "1536")

    import tiktoken
    import hypergraphrag.operate as hg_operate
    import hypergraphrag.utils as hg_utils

    # Force miss: cache hits return the answer string and skip prompt build.
    # operate imports handle_cache by name — patch both namespaces.
    async def _no_cache(hashing_kv, args_hash, prompt, mode="default"):
        return None, None, None, None

    hg_utils.handle_cache = _no_cache
    hg_operate.handle_cache = _no_cache

    enc = tiktoken.get_encoding("cl100k_base")

    root = Path("/workspace") if Path("/workspace/Datasets").is_dir() else backend.parents[1]
    qs = json.loads((root / "Datasets" / "Questions" / "esg_questions_blind.json").read_text(encoding="utf-8"))
    prefer_id = "Apple-2024_L1_0229"
    q = next((r for r in qs if r.get("id") == prefer_id), None)
    if q is None:
        q = next(
            (r for r in qs if r.get("source") == "Apple-2024" and r.get("question_type") == "Fact Retrieval"),
            qs[0],
        )
    print("qid", q.get("id"), "source", q.get("source"), "type", q.get("question_type"))
    print("question", (q.get("question") or "")[:200])

    from esg_encoding.retrieval.hypergraphrag.factory import create_hypergraphrag
    from esg_encoding.retrieval.hypergraphrag.settings import HyperGraphRAGSettings
    from hypergraphrag import QueryParam

    settings = HyperGraphRAGSettings()
    index_root = root / "ESG_KG" / "output" / "triples" / "HyperGraphRAG" / "gpt-5.1" / "indexes" / str(q["source"])
    print("index", index_root, "exists", index_root.is_dir(), "cache_storage", settings.enable_llm_cache)
    rag = create_hypergraphrag(index_root, settings=settings)

    response_type = "One short English phrase or sentence (≤20 words), ending with a period."
    question = str(q.get("question") or "")

    # Retry keyword extraction a few times — empty keywords → fail_response (~20 toks).
    prompt = ""
    pt_sys = 0
    for attempt in range(1, 4):
        pparam = QueryParam(mode="hybrid", top_k=60, only_need_prompt=True, response_type=response_type)
        prompt = await rag.aquery(question, param=pparam)
        prompt = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        pt_sys = len(enc.encode(prompt))
        print(f"attempt {attempt}: sys_chars={len(prompt)} sys_tokens={pt_sys}")
        if pt_sys >= 500:
            break
        print("  too small (likely fail_response / empty keywords), retrying...")

    pt_user = len(enc.encode(question))
    pt_total = pt_sys + pt_user
    print("--- only_need_prompt (real sys prompt) ---")
    print("sys_prompt_chars", len(prompt), "sys_tokens", pt_sys)
    print("user_tokens", pt_user)
    print("prompt_tokens_total_est", pt_total)
    print("sys_prompt_head", prompt[:200].replace("\n", "\\n"))

    cparam = QueryParam(mode="hybrid", top_k=60, only_need_context=True, response_type=response_type)
    ctx = await rag.aquery(question, param=cparam)
    ctx = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False)
    print("--- only_need_context (old meter) ---")
    print("ctx_chars", len(ctx), "ctx_tokens", len(enc.encode(ctx or " ")))

    aparam = QueryParam(
        mode="hybrid",
        top_k=60,
        only_need_prompt=False,
        only_need_context=False,
        response_type=response_type,
    )
    ans = str(await rag.aquery(question, param=aparam)).strip()
    ct = len(enc.encode(ans))
    print("--- answer ---")
    print("answer", ans[:240])
    print("completion_tokens_est", ct)
    print("TOTAL_est", pt_total + ct)

    out = {
        "id": q.get("id"),
        "source": q.get("source"),
        "question_type": q.get("question_type"),
        "prompt_tokens_est": pt_total,
        "sys_prompt_tokens": pt_sys,
        "user_tokens": pt_user,
        "completion_tokens_est": ct,
        "total_tokens_est": pt_total + ct,
        "old_context_tokens": len(enc.encode(ctx or " ")),
        "capture": "only_need_prompt+user",
        "cache_bypassed": True,
        "note": "Single-question probe; not full-set mean. Keyword-extraction LLM tokens not included.",
    }
    out_path = root / "ESG_KG" / "output" / "qa" / "HyperGraphRAG-GPT5.1-GPT4omini" / "_token_probe_one.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
