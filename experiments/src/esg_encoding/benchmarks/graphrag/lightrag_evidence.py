"""Build GraphRAG evidence blocks from LightRAG segment retrieval (G2)."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from ...models import MetricCollection, MetricRetrievalResult, ReportContent, RetrievalResult


def build_lightrag_retrieval_results(
    *,
    lightrag_retriever: Any,
    file_id: str,
    report_content: ReportContent,
    metrics: MetricCollection,
) -> List[MetricRetrievalResult]:
    """For each benchmark question, retrieve ordered segment IDs via LightRAG."""
    segments = list(
        getattr(getattr(report_content, "document_content", None), "segments", None) or []
    )
    by_id: Dict[str, Any] = {
        str(getattr(seg, "segment_id", "") or ""): seg
        for seg in segments
        if getattr(seg, "segment_id", None)
    }
    out: List[MetricRetrievalResult] = []

    for metric in list(getattr(metrics, "metrics", None) or []):
        query = str(getattr(metric, "metric_name", "") or "").strip()
        sids: List[str] = []
        if query:
            try:
                sids = list(
                    lightrag_retriever.retrieve_segment_ids(file_id, report_content, query) or []
                )
            except Exception as exc:
                logger.warning(
                    f"[G2/LightRAG] retrieve failed metric={getattr(metric, 'metric_id', '')}: {exc}"
                )
                sids = []

        combined: List[RetrievalResult] = []
        for rank, sid in enumerate(sids):
            seg = by_id.get(str(sid))
            if seg is None:
                continue
            content = str(getattr(seg, "content", "") or "")
            if not content.strip():
                continue
            combined.append(
                RetrievalResult(
                    segment_id=str(sid),
                    content=content,
                    page_number=int(getattr(seg, "page_number", 1) or 1),
                    score=max(0.01, 1.0 - 0.01 * rank),
                    retrieval_type="lightrag",
                    matched_keywords=[],
                    metric_id=str(getattr(metric, "metric_id", "") or ""),
                )
            )

        out.append(
            MetricRetrievalResult(
                metric_id=str(getattr(metric, "metric_id", "") or ""),
                metric_name=str(getattr(metric, "metric_name", "") or ""),
                metric_code=str(getattr(metric, "metric_code", "") or ""),
                keyword_results=[],
                semantic_results=[],
                combined_results=combined,
                total_matches=len(combined),
            )
        )
        logger.info(
            f"[G2/LightRAG] metric={getattr(metric, 'metric_id', '')} "
            f"retrieved={len(combined)} segments"
        )

    return out
