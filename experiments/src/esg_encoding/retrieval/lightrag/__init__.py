"""LightRAG retrieval integration package (parallel to hipporag/)."""

from .settings import LightRAGSettings

__all__ = [
    "LightRAGSettings",
    "LightRAGRetriever",
    "create_lightrag",
]


def __getattr__(name: str):
    if name == "LightRAGRetriever":
        from .retriever import LightRAGRetriever

        return LightRAGRetriever
    if name == "create_lightrag":
        from .factory import create_lightrag

        return create_lightrag
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
