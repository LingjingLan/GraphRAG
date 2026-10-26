"""LightRAG GraphRAG settings (parallel to HippoRAGSettings)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _backend_dir() -> Path:
    # .../backend/src/esg_encoding/retrieval/lightrag/settings.py -> parents[4] == .../backend
    return Path(__file__).resolve().parents[4]


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "") or default))
    except Exception:
        return default


def _env_optional(name: str) -> str | None:
    value = str(os.getenv(name, "") or "").strip()
    return value or None


@dataclass(frozen=True)
class LightRAGSettings:
    """Tunable knobs for LightRAG indexing + retrieval."""

    # Default OFF: upload / chat must not build LightRAG unless G2 / env enables it.
    enabled: bool = field(default_factory=lambda: _env_bool("LIGHTRAG_ENABLED", False))

    # Query mode: naive | local | global | hybrid | mix
    query_mode: str = field(
        default_factory=lambda: str(os.getenv("LIGHTRAG_QUERY_MODE", "") or "hybrid").strip()
        or "hybrid"
    )

    top_k: int = field(default_factory=lambda: _env_int("LIGHTRAG_TOP_K", 40))
    chunk_top_k: int = field(default_factory=lambda: _env_int("LIGHTRAG_CHUNK_TOP_K", 20))
    max_segment_ids_for_context: int = field(
        default_factory=lambda: _env_int("LIGHTRAG_MAX_SEGMENT_IDS", 12)
    )

    # Index packing (mirror HippoRAG)
    pack_segments: bool = True
    target_chars_per_doc: int = field(
        default_factory=lambda: _env_int("LIGHTRAG_TARGET_CHARS_PER_DOC", 1800)
    )
    max_docs_to_index: int = field(
        default_factory=lambda: _env_int("LIGHTRAG_MAX_DOCS_TO_INDEX", 900)
    )
    min_chars_per_segment: int = field(
        default_factory=lambda: _env_int("LIGHTRAG_MIN_CHARS_PER_SEGMENT", 80)
    )

    # LLM (falls back to LLM_* / ProcessingConfig)
    llm_model_name: str | None = field(default_factory=lambda: _env_optional("LIGHTRAG_LLM_MODEL"))
    llm_base_url: str | None = field(default_factory=lambda: _env_optional("LIGHTRAG_LLM_BASE_URL"))
    llm_api_key: str | None = field(default_factory=lambda: _env_optional("LIGHTRAG_LLM_API_KEY"))

    # Embeddings (DashScope OpenAI-compatible by default)
    embedding_model_name: str = field(
        default_factory=lambda: str(
            os.getenv("LIGHTRAG_EMBEDDING_MODEL", "") or "text-embedding-v3"
        ).strip()
        or "text-embedding-v3"
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("LIGHTRAG_EMBEDDING_DIM", 1024))
    embedding_base_url: str | None = field(
        default_factory=lambda: _env_optional("LIGHTRAG_EMBEDDING_BASE_URL")
    )
    embedding_api_key: str | None = field(
        default_factory=lambda: _env_optional("LIGHTRAG_EMBEDDING_API_KEY")
    )

    # Disable LightRAG's own reranker by default (EulerESG has a separate FlagReranker path).
    enable_rerank: bool = field(default_factory=lambda: _env_bool("LIGHTRAG_ENABLE_RERANK", False))

    # Qwen3 non-thinking mode on DashScope
    disable_thinking: bool = field(
        default_factory=lambda: _env_bool("LIGHTRAG_DISABLE_THINKING", True)
    )

    llm_model_max_async: int = field(default_factory=lambda: _env_int("LIGHTRAG_LLM_MAX_ASYNC", 4))
    embedding_func_max_async: int = field(
        default_factory=lambda: _env_int("LIGHTRAG_EMBED_MAX_ASYNC", 8)
    )

    cache_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("LIGHTRAG_CACHE_ROOT", "") or str(_backend_dir() / "outputs" / "lightrag_cache")
        )
    )
    force_reindex: bool = field(default_factory=lambda: _env_bool("LIGHTRAG_FORCE_REINDEX", False))

    # Cosine threshold for vector retrieval inside LightRAG
    cosine_better_than_threshold: float = field(
        default_factory=lambda: _env_float("LIGHTRAG_COSINE_THRESHOLD", 0.2)
    )

    # Disable when multiple QA processes share the same index working_dir (avoids cache file races).
    enable_llm_cache: bool = field(
        default_factory=lambda: _env_bool("LIGHTRAG_ENABLE_LLM_CACHE", True)
    )
