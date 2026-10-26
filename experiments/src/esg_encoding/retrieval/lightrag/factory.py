"""Build a configured LightRAG instance from LightRAGSettings / LLM env."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .settings import LightRAGSettings

# When set to a dict, the next non-keyword LLM call records the real QA prompt
# (and API usage if the provider returns it). Used by run_lightrag_qa.py to avoid
# a second only_need_prompt retrieval round.
QA_PROMPT_CAPTURE: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "esg_lightrag_qa_prompt_capture", default=None
)


def _resolve_llm(settings: LightRAGSettings, config: Any | None = None) -> tuple[str, str, str]:
    model = (
        settings.llm_model_name
        or getattr(config, "llm_model", None)
        or os.getenv("LLM_MODEL")
        or "qwen3-32b"
    )
    base_url = (
        settings.llm_base_url
        or getattr(config, "llm_base_url", None)
        or os.getenv("LLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = (
        settings.llm_api_key
        or getattr(config, "llm_api_key", None)
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    return str(model), str(base_url), str(api_key)


def _resolve_embedding(settings: LightRAGSettings) -> tuple[str, str, str, int]:
    base_url = (
        settings.embedding_base_url
        or settings.llm_base_url
        or os.getenv("LLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = (
        settings.embedding_api_key
        or settings.llm_api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    model = settings.embedding_model_name
    dim = int(settings.embedding_dim)
    # Auto-pick a sensible OpenAI embedding if still on DashScope defaults.
    base_l = str(base_url).lower()
    # OpenAI + OpenAI-compatible proxies (e.g. litellm / eulerai).
    openai_like = (
        "openai.com" in base_l
        or "api.openai" in base_l
        or "eulerai" in base_l
        or "litellm" in base_l
        or "openai" in base_l
    )
    if openai_like:
        if model in ("", "text-embedding-v3") or model.startswith("text-embedding-v"):
            model = (
                str(os.getenv("LIGHTRAG_EMBEDDING_MODEL", "") or "text-embedding-3-small").strip()
                or "text-embedding-3-small"
            )
        if dim == 1024 and model.startswith("text-embedding-3"):
            dim = int(os.getenv("LIGHTRAG_EMBEDDING_DIM", "") or 1536)
    return model, str(base_url), str(api_key), dim


def create_lightrag(
    working_dir: str | Path,
    *,
    settings: Optional[LightRAGSettings] = None,
    config: Any | None = None,
    workspace: str | None = None,
):
    """Create + return an uninitialized LightRAG instance (call initialize_storages next)."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    settings = settings or LightRAGSettings()
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    llm_model, llm_base, llm_key = _resolve_llm(settings, config)
    emb_model, emb_base, emb_key, emb_dim = _resolve_embedding(settings)
    if not llm_key:
        raise RuntimeError("LLM_API_KEY / LIGHTRAG_LLM_API_KEY missing for LightRAG")

    # Keep OpenAI SDK env in sync for libraries that read process env.
    os.environ.setdefault("OPENAI_API_KEY", llm_key)
    os.environ.setdefault("OPENAI_BASE_URL", llm_base)

    async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        if history_messages is None:
            history_messages = []
        if settings.disable_thinking and "qwen3" in llm_model.lower():
            # Qwen3 on DashScope: non-thinking mode (do not send to OpenAI GPT).
            extra = dict(kwargs.pop("extra_body", None) or {})
            extra.setdefault("enable_thinking", False)
            kwargs["extra_body"] = extra

        bucket = QA_PROMPT_CAPTURE.get()
        tracker = None
        if bucket is not None and not keyword_extraction:
            # Record the exact text sent on the final answer call.
            pieces: list[str] = []
            if system_prompt:
                pieces.append(str(system_prompt))
            for msg in history_messages or []:
                if isinstance(msg, dict) and msg.get("content"):
                    pieces.append(str(msg["content"]))
            if prompt:
                pieces.append(str(prompt))
            bucket["prompt_text"] = "\n\n".join(pieces)
            try:
                from lightrag.utils import TokenTracker

                tracker = TokenTracker()
                kwargs = {**kwargs, "token_tracker": tracker}
            except Exception:
                tracker = None

        result = await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=llm_key,
            base_url=llm_base,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

        if bucket is not None and not keyword_extraction and tracker is not None:
            try:
                usage = tracker.get_usage()
                if usage.get("call_count"):
                    bucket["api_prompt_tokens"] = usage.get("prompt_tokens")
                    bucket["api_completion_tokens"] = usage.get("completion_tokens")
            except Exception:
                pass
        return result

    @wrap_embedding_func_with_attrs(
        embedding_dim=emb_dim,
        max_token_size=8192,
        model_name=emb_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts,
            model=emb_model,
            api_key=emb_key,
            base_url=emb_base,
        )

    kwargs: dict[str, Any] = {
        "working_dir": str(working_dir),
        "llm_model_func": llm_model_func,
        "embedding_func": embedding_func,
        "llm_model_name": llm_model,
        "llm_model_max_async": settings.llm_model_max_async,
        "embedding_func_max_async": settings.embedding_func_max_async,
        "cosine_better_than_threshold": settings.cosine_better_than_threshold,
        "enable_llm_cache": bool(settings.enable_llm_cache),
        "addon_params": {
            "language": "English",
            "entity_types": [
                "organization",
                "person",
                "location",
                "event",
                "metric",
                "product",
                "policy",
                "other",
            ],
        },
    }
    if workspace:
        kwargs["workspace"] = workspace

    return LightRAG(**kwargs)
