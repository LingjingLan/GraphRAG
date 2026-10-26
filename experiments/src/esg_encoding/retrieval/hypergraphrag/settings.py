"""HyperGraphRAG settings (parallel to LightRAGSettings / HippoRAGSettings).

Upstream alignment with LightRAG + HippoRAG2:
  - statements-only pages, pack_chars=0
  - chunk_token_size=1200 (library default; GraphRAG-Bench-style)
  - extract LLM: GPT-5.1 (via LLM_* / overrides)
  - embedding: text-embedding-3-small (dim 1536) via LIGHTRAG_EMBEDDING_*
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "") or default))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_optional(name: str) -> str | None:
    value = str(os.getenv(name, "") or "").strip()
    return value or None


@dataclass(frozen=True)
class HyperGraphRAGSettings:
    """Tunable knobs for HyperGraphRAG indexing + retrieval."""

    enabled: bool = field(default_factory=lambda: _env_bool("HYPERGRAPH_ENABLED", False))

    query_mode: str = field(
        default_factory=lambda: str(os.getenv("HYPERGRAPH_QUERY_MODE", "") or "hybrid").strip()
        or "hybrid"
    )
    top_k: int = field(default_factory=lambda: _env_int("HYPERGRAPH_TOP_K", 60))

    # Official HyperGraphRAG / GraphRAG-Bench aligned chunking
    chunk_token_size: int = field(default_factory=lambda: _env_int("HYPERGRAPH_CHUNK_TOKEN_SIZE", 1200))
    chunk_overlap_token_size: int = field(
        default_factory=lambda: _env_int("HYPERGRAPH_CHUNK_OVERLAP_TOKEN_SIZE", 100)
    )

    llm_model_name: str | None = field(default_factory=lambda: _env_optional("HYPERGRAPH_LLM_MODEL"))
    llm_base_url: str | None = field(default_factory=lambda: _env_optional("HYPERGRAPH_LLM_BASE_URL"))
    llm_api_key: str | None = field(default_factory=lambda: _env_optional("HYPERGRAPH_LLM_API_KEY"))

    # Reuse LightRAG embedding env so GPT-5.1 indexes share one OpenAI embedding config
    embedding_model_name: str = field(
        default_factory=lambda: str(
            os.getenv("HYPERGRAPH_EMBEDDING_MODEL", "")
            or os.getenv("LIGHTRAG_EMBEDDING_MODEL", "")
            or "text-embedding-3-small"
        ).strip()
        or "text-embedding-3-small"
    )
    embedding_dim: int = field(
        default_factory=lambda: _env_int(
            "HYPERGRAPH_EMBEDDING_DIM",
            _env_int("LIGHTRAG_EMBEDDING_DIM", 1536),
        )
    )
    embedding_base_url: str | None = field(
        default_factory=lambda: _env_optional("HYPERGRAPH_EMBEDDING_BASE_URL")
        or _env_optional("LIGHTRAG_EMBEDDING_BASE_URL")
    )
    embedding_api_key: str | None = field(
        default_factory=lambda: _env_optional("HYPERGRAPH_EMBEDDING_API_KEY")
        or _env_optional("LIGHTRAG_EMBEDDING_API_KEY")
    )

    disable_thinking: bool = field(
        default_factory=lambda: _env_bool(
            "HYPERGRAPH_DISABLE_THINKING",
            _env_bool("LIGHTRAG_DISABLE_THINKING", True),
        )
    )

    llm_model_max_async: int = field(default_factory=lambda: _env_int("HYPERGRAPH_LLM_MAX_ASYNC", 4))
    embedding_func_max_async: int = field(
        default_factory=lambda: _env_int("HYPERGRAPH_EMBEDDING_MAX_ASYNC", 8)
    )
    enable_llm_cache: bool = field(default_factory=lambda: _env_bool("HYPERGRAPH_ENABLE_LLM_CACHE", True))
