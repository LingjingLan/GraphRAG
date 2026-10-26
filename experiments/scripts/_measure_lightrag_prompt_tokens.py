#!/usr/bin/env python3
"""Measure real LightRAG hybrid prompt size (no 50k log truncate)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path("/workspace")
sys.path.insert(0, "/workspace/backend/src")


def load_dotenv() -> dict:
    env = {}
    path = Path("/workspace/backend/config/.env")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


async def main() -> None:
    from lightrag import QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from esg_encoding.retrieval.lightrag.factory import create_lightrag
    from esg_encoding.retrieval.lightrag.settings import LightRAGSettings

    cfg = load_dotenv()
    for k in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LIGHTRAG_EMBEDDING_MODEL",
        "LIGHTRAG_EMBEDDING_DIM",
        "LIGHTRAG_EMBEDDING_BASE_URL",
        "LIGHTRAG_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    ):
        if cfg.get(k):
            os.environ[k] = cfg[k]
    os.environ["OPENAI_API_KEY"] = cfg.get("LLM_API_KEY") or cfg.get("OPENAI_API_KEY") or ""
    if cfg.get("LLM_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = cfg["LLM_BASE_URL"]
    if cfg.get("LIGHTRAG_EMBEDDING_API_KEY"):
        # embedding uses separate key; factory reads LIGHTRAG_EMBEDDING_*
        pass

    qs = json.loads((ROOT / "Datasets/Questions/esg_questions_blind.json").read_text(encoding="utf-8"))
    if isinstance(qs, dict):
        qs = qs.get("questions") or []
    idxs = [0, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    sample = [qs[i] for i in idxs if i < len(qs)]

    index_root = ROOT / "ESG_KG/output/triples/LightRAG/gpt-5.1/indexes"
    settings = replace(
        LightRAGSettings(),
        enabled=True,
        query_mode="hybrid",
        top_k=30,
        chunk_top_k=20,
        llm_model_name=cfg.get("LLM_MODEL") or "qwen3-8b",
        llm_base_url=cfg.get("LLM_BASE_URL"),
        llm_api_key=cfg.get("LLM_API_KEY"),
        embedding_model_name=cfg.get("LIGHTRAG_EMBEDDING_MODEL") or "text-embedding-3-small",
        embedding_dim=int(cfg.get("LIGHTRAG_EMBEDDING_DIM") or 1536),
        embedding_base_url=cfg.get("LIGHTRAG_EMBEDDING_BASE_URL") or "https://api.openai.com/v1",
        embedding_api_key=cfg.get("LIGHTRAG_EMBEDDING_API_KEY") or cfg.get("OPENAI_API_KEY") or "",
    )

    max_entity_tokens = 4000
    max_relation_tokens = 4000
    max_total_tokens = 30000

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def ntok(t: str) -> int:
            return len(enc.encode(t))
    except Exception:

        def ntok(t: str) -> int:
            return max(1, len(t) // 4)

    cache = {}
    rows = []
    for q in sample:
        doc = str(q.get("source") or "")
        question = str(q.get("question") or "")
        if doc not in cache:
            working = index_root / doc
            if not working.is_dir():
                print("MISSING", working)
                continue
            rag = create_lightrag(working, settings=settings, workspace=doc)
            await rag.initialize_storages()
            try:
                await initialize_pipeline_status()
            except Exception:
                pass
            cache[doc] = rag
        rag = cache[doc]
        param = QueryParam(
            mode="hybrid",
            only_need_prompt=True,
            only_need_context=False,
            top_k=30,
            chunk_top_k=20,
            enable_rerank=False,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
        )
        prompt = await rag.aquery(question, param=param)
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt, ensure_ascii=False)
        tok = ntok(prompt)
        rows.append(
            {
                "id": q.get("id"),
                "source": doc,
                "chars": len(prompt),
                "tokens_cl100k": tok,
                "chars_div4": max(1, len(prompt) // 4),
            }
        )
        print(f"{q.get('id')}: chars={len(prompt)} tokens={tok} chars/4={len(prompt)//4}", flush=True)

    toks = [r["tokens_cl100k"] for r in rows]
    print("---", flush=True)
    print(
        f"n={len(toks)} mean={sum(toks)/len(toks):.0f} min={min(toks)} max={max(toks)} "
        f"budget_max_total_tokens={max_total_tokens}",
        flush=True,
    )
    print(f"extrapolate_2015_mean_sum_M={(sum(toks)/len(toks)*2015)/1e6:.2f}", flush=True)
    out = ROOT / "ESG_KG/output/qa/_lightrag_prompt_token_sample.json"
    out.write_text(
        json.dumps(
            {
                "max_total_tokens": max_total_tokens,
                "max_entity_tokens": max_entity_tokens,
                "max_relation_tokens": max_relation_tokens,
                "sample": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote", out, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
