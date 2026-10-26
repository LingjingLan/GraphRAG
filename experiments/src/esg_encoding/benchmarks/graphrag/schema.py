"""GraphRAG benchmark response contract for hop-style ``reasoning``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


ALLOWED_HOP_KINDS = frozenset({"quote", "compute", "matrix"})


class ReasoningHop(TypedDict, total=False):
    hop: int
    kind: str
    text: str
    loc: Optional[str]


class GraphRAGAnswer(TypedDict, total=False):
    id: str
    answer: Any
    reasoning: List[ReasoningHop]
    status: str
    error: str


class ValidationError(Exception):
    """Raised when a GraphRAG LLM response fails adapter validation."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
