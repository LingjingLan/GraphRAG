"""Metric-centric dual retrieval orchestration.

The retriever follows the project RAG plan:
- exact code index
- exact alias index
- BM25 keyword index
- dense vector index
- weighted RRF fusion
- exact-metric rerank
"""

from __future__ import annotations

from datetime import datetime
import os
import re
import time
from typing import Dict, List, Optional, Sequence

from .scoring import *  # noqa: F401,F403
from .fusion import exact_metric_rerank, rrf_fuse
from .keyword import KeywordRetriever
from .metric_profile import build_metric_retrieval_profile
from .semantic import SemanticRetriever


def _is_exact_code_result(result: RetrievalResult) -> bool:
    return "exact_code" in str(getattr(result, "retrieval_type", "") or "")


class DualChannelRetriever:
    """Metric-centric dual retriever."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.keyword_retriever = KeywordRetriever(config)
        self.semantic_retriever = SemanticRetriever(config)

    def retrieve_for_metric(
        self,
        report_content: ReportContent,
        metric: ESGMetric,
        semantic_expansion: Optional[SemanticExpansion] = None,
    ) -> MetricRetrievalResult:
        """Retrieve evidence for one canonical metric."""
        try:
            retrieval_started = time.perf_counter()
            metric_name = getattr(metric, "metric_name", "") or getattr(metric, "metric_id", "")
            logger.info(f"Starting metric-centric retrieval for metric {metric_name}")
            profile = build_metric_retrieval_profile(metric, semantic_expansion)

            keyword_started = time.perf_counter()
            exact_code_results: List[RetrievalResult] = []
            exact_alias_results: List[RetrievalResult] = []
            bm25_results: List[RetrievalResult] = []
            if getattr(self.config, "use_keyword_retrieval", True):
                exact_code_results = self.keyword_retriever.search_exact_code(report_content, metric, profile)
                exact_alias_results = self.keyword_retriever.search_exact_alias(report_content, metric, profile)
                bm25_results = self.keyword_retriever.search_bm25(report_content, metric, profile)
            keyword_elapsed = time.perf_counter() - keyword_started
            link_code_context_results = self._linked_code_context_results(
                report_content,
                exact_code_results,
            )

            linked_started = time.perf_counter()
            linked_page_results = self._search_linked_pages(
                report_content,
                metric,
                profile,
                exact_code_results + exact_alias_results + bm25_results,
                semantic_expansion,
            )
            linked_category_results = [
                result for result in linked_page_results
                if "linked_page_category" in result.retrieval_type
            ]
            linked_other_results = [
                result for result in linked_page_results
                if "linked_page_category" not in result.retrieval_type
            ]
            linked_elapsed = time.perf_counter() - linked_started

            # Internal links form a second, page-bounded retrieval pass. Once
            # that pass finds evidence, do not run or rerank whole-report dense
            # candidates for the same metric.
            linked_second_pass = bool(linked_page_results)
            semantic_started = time.perf_counter()
            semantic_results: List[RetrievalResult] = []
            if (
                not linked_second_pass
                and getattr(self.config, "use_semantic_retrieval", True)
                and profile.dense_query
            ):
                semantic_results = self.semantic_retriever.search_by_semantic(
                    report_content,
                    metric,
                    semantic_expansion,
                    apply_reranker=False,
                )
            semantic_elapsed = time.perf_counter() - semantic_started

            if linked_second_pass:
                effective_keyword_results = linked_page_results
                channel_results = {
                    "linked_page_category": linked_category_results,
                    "linked_page": linked_other_results,
                }
            else:
                effective_keyword_results = (
                    exact_code_results + exact_alias_results + bm25_results
                )
                channel_results = {
                    "exact_code": exact_code_results,
                    "exact_alias": exact_alias_results,
                    "bm25": bm25_results,
                    "semantic": semantic_results,
                }

            combined_results = self._combine_results(
                keyword_results=effective_keyword_results,
                semantic_results=semantic_results,
                metric=metric,
                report_content=report_content,
                channel_results=channel_results,
                profile=profile,
            )
            if not linked_second_pass and link_code_context_results:
                combined_results = self._retain_link_fallback_code_context(
                    combined_results,
                    link_code_context_results,
                    capacity=max(11, len(combined_results)),
                )

            result = MetricRetrievalResult(
                metric_id=getattr(metric, "metric_id", profile.metric_id),
                metric_name=getattr(metric, "metric_name", profile.metric_name),
                metric_code=getattr(metric, "metric_code", profile.metric_code),
                keyword_results=effective_keyword_results,
                semantic_results=semantic_results,
                combined_results=combined_results,
                total_matches=len(combined_results),
            )
            logger.info(
                f"Metric-centric retrieval completed for {metric_name}, "
                f"found={len(combined_results)}, elapsed={time.perf_counter() - retrieval_started:.2f}s, "
                f"keyword={keyword_elapsed:.2f}s, semantic={semantic_elapsed:.2f}s, "
                f"linked={linked_elapsed:.2f}s, "
                f"mode={'linked_second_pass' if linked_second_pass else 'whole_report'}, "
                f"link_fallback_code_rows={len(link_code_context_results) if not linked_second_pass else 0}, "
                f"discovery_candidates={len(exact_code_results) + len(exact_alias_results) + len(bm25_results)}"
            )
            return result
        except Exception as exc:
            logger.error(f"Metric retrieval failed: {str(exc)}")
            raise ESGEncodingError(f"Metric retrieval failed: {str(exc)}") from exc

    @staticmethod
    def _env_enabled(name: str, default: bool = True) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _row_key(segment) -> Optional[tuple[str, int]]:
        data = getattr(segment, "structured_data", None)
        data = data if isinstance(data, dict) else {}
        table_id = getattr(segment, "source_table_id", None) or data.get("table_id") or data.get("source_table_id")
        row_index = data.get("row_index", data.get("row_idx"))
        if table_id is None or row_index is None:
            return None
        try:
            return str(table_id), int(row_index)
        except Exception:
            return None

    @classmethod
    def _linked_code_context_results(
        cls,
        report_content: ReportContent,
        exact_code_results: Sequence[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Return exact-Code rows that contain link metadata in the same row."""
        segments = list(report_content.document_content.segments or [])
        by_id = {
            str(getattr(segment, "segment_id", "") or ""): segment
            for segment in segments
        }
        by_row: Dict[tuple[str, int], List[object]] = {}
        for segment in segments:
            row_key = cls._row_key(segment)
            if row_key is not None:
                by_row.setdefault(row_key, []).append(segment)

        contexts: List[RetrievalResult] = []
        seen = set()
        for result in exact_code_results:
            segment = by_id.get(str(getattr(result, "segment_id", "") or ""))
            if segment is None:
                continue
            related = [segment]
            row_key = cls._row_key(segment)
            if row_key is not None:
                related.extend(by_row.get(row_key, []))
            has_link = any(
                isinstance(getattr(item, "structured_data", None), dict)
                and any(
                    isinstance(link, dict)
                    for link in ((getattr(item, "structured_data", None) or {}).get("pdf_links") or [])
                )
                for item in related
            )
            if not has_link or result.segment_id in seen:
                continue
            seen.add(result.segment_id)
            contexts.append(result)
        return contexts

    @staticmethod
    def _retain_link_fallback_code_context(
        results: Sequence[RetrievalResult],
        required_contexts: Sequence[RetrievalResult],
        capacity: int,
    ) -> List[RetrievalResult]:
        """Keep unresolved-link Code rows without promoting them as data."""
        values = list(results)
        by_id = {item.segment_id: index for index, item in enumerate(values)}
        missing: List[RetrievalResult] = []
        seen = set()
        for context in required_contexts:
            if context.segment_id in seen:
                continue
            seen.add(context.segment_id)
            existing_index = by_id.get(context.segment_id)
            selected = values[existing_index] if existing_index is not None else context
            retrieval_type = str(selected.retrieval_type or "")
            if "link_fallback_code_context" not in retrieval_type:
                retrieval_type += "+link_fallback_code_context"
            marked = selected.model_copy(update={"retrieval_type": retrieval_type})
            if existing_index is None:
                missing.append(marked)
            else:
                values[existing_index] = marked

        limit = max(1, int(capacity or 1))
        if missing:
            values = values[:max(0, limit - len(missing))] + missing[:limit]
        return values[:limit]

    @classmethod
    def _link_source_rank(cls, segment, anchor_text: str) -> int:
        """Prefer the exact anchor cell/row over a whole-page table segment."""
        segment_type = str(getattr(segment, "segment_type", "") or "").lower()
        type_rank = {
            "table_cell": 6,
            "table_row": 5,
            "link_anchor": 4,
            "text": 3,
            "heading": 3,
            "table": 1,
        }.get(segment_type, 2)
        anchor = cls._normalized_link_text(anchor_text)
        value_text = cls._normalized_link_text(getattr(segment, "value_text", "") or "")
        content = cls._normalized_link_text(getattr(segment, "content", "") or "")
        if anchor and anchor in value_text:
            type_rank += 3
        elif anchor and anchor in content:
            type_rank += 2
        return type_rank

    @staticmethod
    def _normalized_link_text(value: object) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()

    @staticmethod
    def _linked_metric_terms(metric: ESGMetric) -> List[str]:
        """Report-language aliases used only to rank pages reached through an internal link."""
        name = re.sub(r"\s+", " ", str(getattr(metric, "metric_name", "") or "").lower())
        terms: List[str] = []
        if "gender" in name:
            terms.extend(["global female representation", "female representation", "women"])
        if "diversity" in name or "representation" in name:
            terms.extend(["race/ethnicity representation", "race/ethnicity", "underrepresented groups"])

        if "all other" in name:
            terms.extend(["non-technical roles", "nontechnical roles", "non-technical"])
        elif "non-executive" in name:
            terms.extend(["people leader roles", "people leaders", "leadership"])
        elif "executive" in name:
            terms.extend(["people leader roles", "people leaders", "leadership"])
        elif "technical" in name:
            terms.extend(["technical roles", "technical employees", "technical"])
        return list(dict.fromkeys(terms))

    @staticmethod
    def _linked_metric_hit_count(content: str, metric: ESGMetric, terms: Sequence[str]) -> int:
        lowered = str(content or "").lower()
        metric_name = str(getattr(metric, "metric_name", "") or "").lower()
        if (
            "technical" in metric_name
            and "all other" not in metric_name
            and "non-executive" not in metric_name
            and re.search(r"\bnon[\s-]*technical\b", lowered)
        ):
            return 0
        return sum(1 for term in terms if term in lowered)

    @staticmethod
    def _linked_category_adjustment(content: str, metric: ESGMetric) -> float:
        """Prefer the report's matching employee-category section over substring matches."""
        lowered = str(content or "").lower()
        metric_name = str(getattr(metric, "metric_name", "") or "").lower()
        adjustment = 0.0

        is_gender_section = "global female representation" in lowered
        is_diversity_section = "race/ethnicity representation" in lowered
        if "gender" in metric_name:
            adjustment += 0.12 if is_gender_section else (-0.14 if is_diversity_section else 0.0)
        if "diversity" in metric_name:
            adjustment += 0.08 if is_diversity_section else -0.18

        has_non_technical = bool(re.search(r"\bnon[\s-]*technical(?:\s+roles?)?\b", lowered))
        has_technical = bool(re.search(r"\btechnical(?:\s+roles?)?\b", lowered)) and not has_non_technical
        if "all other" in metric_name:
            adjustment += 0.18 if has_non_technical else (-0.08 if has_technical else 0.0)
        elif "non-executive" in metric_name or "executive" in metric_name:
            if "people leader roles" in lowered:
                adjustment += 0.20
        elif "technical" in metric_name:
            if has_non_technical:
                adjustment -= 0.22
            elif has_technical:
                adjustment += 0.18
        return adjustment

    def _linked_page_targets(
        self,
        report_content: ReportContent,
        trigger_results: Sequence[RetrievalResult],
    ) -> Dict[int, Dict[str, object]]:
        if not self._env_enabled("REPORT_LINK_RESOLUTION_ENABLED", True):
            return {}
        try:
            max_depth = max(0, int(os.getenv("REPORT_LINK_MAX_DEPTH", "1") or "1"))
            max_target_pages = max(
                1,
                int(os.getenv("REPORT_LINK_MAX_TARGET_PAGES_PER_METRIC", "5") or "5"),
            )
            forward_page_count = max(
                0,
                int(os.getenv("REPORT_LINK_FORWARD_PAGE_COUNT", "7") or "7"),
            )
        except Exception:
            max_depth, max_target_pages, forward_page_count = 1, 5, 7
        if max_depth < 1:
            return {}

        segments = list(report_content.document_content.segments or [])
        by_id = {getattr(segment, "segment_id", ""): segment for segment in segments}
        by_row: Dict[tuple[str, int], List[object]] = {}
        by_page: Dict[int, List[object]] = {}
        for segment in segments:
            key = self._row_key(segment)
            if key is not None:
                by_row.setdefault(key, []).append(segment)
            try:
                page_number = int(getattr(segment, "page_number", 0) or 0)
            except Exception:
                page_number = 0
            if page_number > 0:
                by_page.setdefault(page_number, []).append(segment)

        direct_links_by_page: Dict[int, Dict[str, object]] = {}
        direct_page_order: List[int] = []
        for result in list(trigger_results)[:32]:
            segment = by_id.get(getattr(result, "segment_id", ""))
            if segment is None:
                continue
            related = [segment]
            row_key = self._row_key(segment)
            if row_key is not None:
                related.extend(by_row.get(row_key, []))
            for related_segment in related:
                data = getattr(related_segment, "structured_data", None)
                if not isinstance(data, dict):
                    continue
                for link in data.get("pdf_links") or []:
                    if (
                        not isinstance(link, dict)
                        or link.get("link_type") != "internal"
                        or bool(link.get("navigation"))
                    ):
                        continue
                    try:
                        target_page = int(link.get("target_page"))
                        source_page = int(link.get("source_page") or getattr(segment, "page_number", 0) or 0)
                    except Exception:
                        continue
                    if target_page < 1:
                        continue
                    if (
                        target_page not in direct_links_by_page
                        and len(direct_page_order) >= max_target_pages
                    ):
                        continue

                    anchor_text = str(link.get("anchor_text") or "").strip()
                    source_segment = related_segment
                    related_type = str(getattr(related_segment, "segment_type", "") or "").lower()
                    if related_type == "table" and self._link_source_rank(segment, anchor_text) > self._link_source_rank(
                        related_segment,
                        anchor_text,
                    ):
                        source_segment = segment
                    candidate = {
                        "target_page": target_page,
                        "source_page": source_page,
                        "anchor_text": anchor_text,
                        "source_segment_id": str(getattr(source_segment, "segment_id", "") or ""),
                        "source_rank": self._link_source_rank(source_segment, anchor_text),
                    }
                    current = direct_links_by_page.get(target_page)
                    if current is None:
                        direct_page_order.append(target_page)
                        direct_links_by_page[target_page] = candidate
                    elif int(candidate["source_rank"]) > int(current.get("source_rank") or 0):
                        direct_links_by_page[target_page] = candidate

        direct_links = [direct_links_by_page[page] for page in direct_page_order]

        targets: Dict[int, Dict[str, object]] = {}
        for link in direct_links:
            target_page = int(link["target_page"])
            targets[target_page] = {
                "source_page": link["source_page"],
                "anchor_text": link["anchor_text"],
                "source_segment_id": link.get("source_segment_id"),
                "root_target_page": target_page,
                "context_offset": 0,
                "continuation": False,
            }

        # A PDF link starts a bounded second pass over the Paddle-extracted
        # target page and the next consecutive pages. Pages before the target
        # are deliberately excluded from linked evidence.
        for link in direct_links:
            root_target_page = int(link["target_page"])
            source_page = int(link.get("source_page") or 0)
            for offset in range(1, forward_page_count + 1):
                candidate_page = root_target_page + offset
                if (
                    candidate_page < 1
                    or candidate_page == source_page
                    or candidate_page not in by_page
                ):
                    continue

                current = targets.get(candidate_page)
                current_offset = int((current or {}).get("context_offset") or 0)
                if current is not None and (
                    current_offset == 0 or current_offset <= offset
                ):
                    continue
                targets[candidate_page] = {
                    "source_page": link["source_page"],
                    "anchor_text": link["anchor_text"],
                    "source_segment_id": link.get("source_segment_id"),
                    "root_target_page": root_target_page,
                    "context_offset": offset,
                    "continuation": True,
                }
        return targets

    def _search_linked_pages(
        self,
        report_content: ReportContent,
        metric: ESGMetric,
        profile,
        trigger_results: Sequence[RetrievalResult],
        semantic_expansion: Optional[SemanticExpansion] = None,
    ) -> List[RetrievalResult]:
        targets = self._linked_page_targets(report_content, trigger_results)
        if not targets:
            return []

        target_segments = [
            segment
            for segment in report_content.document_content.segments
            if int(getattr(segment, "page_number", 0) or 0) in targets
        ]
        if not target_segments:
            return []

        page_table_text: Dict[int, List[str]] = {}
        for segment in target_segments:
            if str(getattr(segment, "segment_type", "") or "").lower() != "table":
                continue
            page_table_text.setdefault(int(segment.page_number), []).append(
                str(getattr(segment, "content", "") or "")
            )
        page_category_adjustments = {
            page: self._linked_category_adjustment("\n".join(contents), metric)
            for page, contents in page_table_text.items()
        }

        target_ids = {getattr(segment, "segment_id", "") for segment in target_segments}
        target_embeddings = [
            embedding for embedding in (report_content.embeddings or [])
            if getattr(embedding, "segment_id", "") in target_ids
        ]
        target_document = report_content.document_content.model_copy(update={"segments": target_segments})
        target_report = report_content.model_copy(
            update={"document_content": target_document, "embeddings": target_embeddings}
        )
        object.__setattr__(target_report, "_semantic_retrieval_embedding_cache", None)

        candidates: List[RetrievalResult] = []
        candidates.extend(self.keyword_retriever.search_exact_code(target_report, metric, profile))
        candidates.extend(self.keyword_retriever.search_exact_alias(target_report, metric, profile))
        candidates.extend(self.keyword_retriever.search_bm25(target_report, metric, profile))
        if (
            getattr(self.config, "use_semantic_retrieval", True)
            and profile.dense_query
            and target_embeddings
        ):
            candidates.extend(
                self.semantic_retriever.search_by_semantic(
                    target_report,
                    metric,
                    semantic_expansion,
                    apply_reranker=False,
                )
            )

        # A linked data page can use a compact heading that does not repeat the
        # framework label. Keep numeric, metric-related target chunks as fallback.
        existing_ids = {item.segment_id for item in candidates}
        anchor_terms = list(getattr(profile, "anchor_terms", None) or [])
        metric_terms = self._linked_metric_terms(metric)
        for segment in target_segments:
            if segment.segment_id in existing_ids:
                continue
            content = str(getattr(segment, "content", "") or "")
            lowered = content.lower()
            anchor_hits = sum(1 for term in anchor_terms if term and str(term).lower() in lowered)
            metric_hits = self._linked_metric_hit_count(lowered, metric, metric_terms)
            category_adjustment = self._linked_category_adjustment(lowered, metric)
            has_number = bool(re.search(r"-?\d[\d,]*(?:\.\d+)?\s*(?:%|[A-Za-z]+)?", content))
            if not has_number or (anchor_hits == 0 and metric_hits == 0):
                continue
            segment_type = str(getattr(segment, "segment_type", "") or "").lower()
            structure_bonus = 0.08 if segment_type == "table" else (0.035 if segment_type == "table_row" else 0.0)
            candidates.append(
                RetrievalResult(
                    segment_id=segment.segment_id,
                    content=content,
                    page_number=segment.page_number,
                    score=min(
                        0.98,
                        0.58
                        + 0.035 * min(anchor_hits, 4)
                        + 0.075 * min(metric_hits, 3)
                        + structure_bonus
                        + category_adjustment,
                    ),
                    retrieval_type="linked_page_fallback",
                    matched_keywords=(
                        [str(term) for term in metric_terms if term in lowered]
                        + [str(term) for term in anchor_terms if str(term).lower() in lowered]
                    )[:12],
                    metric_id=getattr(metric, "metric_id", ""),
                )
            )

        deduped: Dict[str, RetrievalResult] = {}
        for candidate in candidates:
            current = deduped.get(candidate.segment_id)
            if current is None or float(candidate.score or 0.0) > float(current.score or 0.0):
                deduped[candidate.segment_id] = candidate

        segment_by_id = {
            getattr(segment, "segment_id", ""): segment
            for segment in target_segments
        }
        linked_results: List[RetrievalResult] = []
        for candidate in deduped.values():
            target_page = int(candidate.page_number)
            link_meta = targets.get(target_page)
            if link_meta is None:
                continue
            anchor_text = str(link_meta.get("anchor_text") or "").strip()
            matched = list(candidate.matched_keywords or [])
            if anchor_text and anchor_text not in matched:
                matched.append(anchor_text)
            candidate_segment = segment_by_id.get(candidate.segment_id)
            candidate_content = str(getattr(candidate_segment, "content", "") or candidate.content or "").lower()
            metric_hits = self._linked_metric_hit_count(candidate_content, metric, metric_terms)
            category_adjustment = page_category_adjustments.get(
                target_page,
                self._linked_category_adjustment(candidate_content, metric),
            )
            segment_type = str(getattr(candidate_segment, "segment_type", "") or "").lower()
            metric_bonus = min(0.06, 0.02 * metric_hits)
            structure_bonus = 0.03 if segment_type == "table" else (0.015 if segment_type == "table_row" else 0.0)
            base_score = min(0.72, max(0.60, float(candidate.score or 0.0)))
            linked_score = max(
                0.20,
                min(
                    0.97,
                    base_score
                    + metric_bonus
                    + structure_bonus
                    + category_adjustment,
                ),
            )
            linked_kind = (
                "linked_page_category"
                if category_adjustment >= 0.10
                else "linked_page"
            )
            linked_results.append(
                RetrievalResult(
                    segment_id=candidate.segment_id,
                    content=candidate.content,
                    page_number=target_page,
                    score=linked_score,
                    retrieval_type=f"{linked_kind}+{candidate.retrieval_type}",
                    matched_keywords=matched[:18],
                    metric_id=candidate.metric_id,
                    link_source_page=int(link_meta.get("source_page") or 0) or None,
                    link_target_page=int(link_meta.get("root_target_page") or target_page),
                    link_anchor_text=anchor_text or None,
                    link_source_segment_id=str(link_meta.get("source_segment_id") or "") or None,
                )
            )
        linked_results.sort(key=lambda item: item.score, reverse=True)

        # Reserve one high-information result per linked page before filling the
        # remaining slots. This prevents one page's many table cells from hiding
        # later linked context pages in the bounded analysis window.
        page_first: List[RetrievalResult] = []
        remaining: List[RetrievalResult] = []
        represented_pages = set()
        for result in linked_results:
            if result.page_number not in represented_pages:
                represented_pages.add(result.page_number)
                page_first.append(result)
            else:
                remaining.append(result)
        linked_results = page_first + remaining
        logger.info(
            f"Linked-page retrieval for {getattr(metric, 'metric_id', 'unknown')}: "
            f"roots={sorted({int(meta.get('root_target_page') or page) for page, meta in targets.items()})}, "
            f"pages={sorted(targets)}, matches={len(linked_results)}"
        )
        result_limit = max(
            1,
            len(targets),
            int(getattr(self.config, "top_k", 10) or 10),
        )
        return linked_results[:result_limit]

    @staticmethod
    def _is_index_context(content: str) -> bool:
        lowered = str(content or "").lower()
        return any(
            marker in lowered
            for marker in (
                "reference indices",
                "reference index",
                "reporting frameworks index",
                "reporting framework index",
                "sasb index",
                "content index",
                "table of contents",
            )
        )

    @staticmethod
    def _has_non_reference_number(content: str, profile) -> bool:
        cleaned = str(content or "")
        for pattern in getattr(profile, "exact_code_patterns", None) or []:
            cleaned = pattern.sub(" ", cleaned)
        cleaned = re.sub(r"\b(?:fy\s*)?(?:19|20)\d{2}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:page|row|column|col)\s*#?\s*\d+\b", " ", cleaned, flags=re.IGNORECASE)
        return bool(re.search(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*%)?", cleaned))

    @classmethod
    def _segment_has_real_data(
        cls,
        segment,
        profile,
        row_segments: Sequence[object],
    ) -> bool:
        data = getattr(segment, "structured_data", None)
        data = data if isinstance(data, dict) else {}
        value_text = str(
            getattr(segment, "value_text", None)
            or data.get("value_text")
            or ""
        ).strip()
        col_header = str(
            getattr(segment, "col_header", None)
            or data.get("col_header")
            or ""
        ).lower()
        rejected_headers = ("reference", "index", "code", "page", "year", "unit")
        if (
            value_text
            and not any(marker in col_header for marker in rejected_headers)
            and cls._has_non_reference_number(value_text, profile)
        ):
            return True

        for sibling in row_segments:
            if sibling is segment:
                continue
            sibling_data = getattr(sibling, "structured_data", None)
            sibling_data = sibling_data if isinstance(sibling_data, dict) else {}
            sibling_value = str(
                getattr(sibling, "value_text", None)
                or sibling_data.get("value_text")
                or ""
            ).strip()
            sibling_header = str(
                getattr(sibling, "col_header", None)
                or sibling_data.get("col_header")
                or ""
            ).lower()
            if (
                sibling_value
                and not any(marker in sibling_header for marker in rejected_headers)
                and cls._has_non_reference_number(sibling_value, profile)
            ):
                return True

        content = str(getattr(segment, "content", "") or "")
        if not cls._has_non_reference_number(content, profile):
            return False
        if cls._is_index_context(content):
            cleaned = content
            for pattern in getattr(profile, "exact_code_patterns", None) or []:
                cleaned = pattern.sub(" ", cleaned)
            return bool(re.search(r"\d[\d,]*(?:\.\d+)?\s*%", cleaned))
        return True

    def _prepare_unified_rerank_candidates(
        self,
        results: Sequence[RetrievalResult],
        report_content: Optional[ReportContent],
        profile,
        limit: int,
    ) -> List[RetrievalResult]:
        segments = list(
            getattr(getattr(report_content, "document_content", None), "segments", [])
            if report_content is not None
            else []
        )
        by_id = {
            str(getattr(segment, "segment_id", "") or ""): segment
            for segment in segments
        }
        by_row: Dict[tuple[str, int], List[object]] = {}
        for segment in segments:
            row_key = self._row_key(segment)
            if row_key is not None:
                by_row.setdefault(row_key, []).append(segment)

        prepared: List[RetrievalResult] = []
        code_index_count = 0
        seen = set()
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            if result.segment_id in seen:
                continue
            seen.add(result.segment_id)
            retrieval_type = str(result.retrieval_type or "")
            segment = by_id.get(result.segment_id)
            row_key = self._row_key(segment) if segment is not None else None
            row_segments = by_row.get(row_key, []) if row_key is not None else []
            has_real_data = bool(
                segment is not None
                and self._segment_has_real_data(segment, profile, row_segments)
            )
            is_linked = "linked_page" in retrieval_type
            is_code_index = _is_exact_code_result(result) and not is_linked and not has_real_data
            if is_code_index:
                if code_index_count >= 10:
                    continue
                code_index_count += 1

            labels = []
            if has_real_data:
                labels.append("real_data_evidence")
            if is_code_index:
                labels.append("code_index_evidence")
            for label in labels:
                if label not in retrieval_type:
                    retrieval_type += f"+{label}"
            prepared.append(result.model_copy(update={"retrieval_type": retrieval_type}))

        def attention_rank(item: RetrievalResult) -> tuple[int, float]:
            result_type = str(item.retrieval_type or "")
            real_data = "real_data_evidence" in result_type
            linked = "linked_page" in result_type
            rank = 3 if real_data and linked else (2 if real_data else (1 if linked else 0))
            return rank, float(item.score or 0.0)

        prepared.sort(key=attention_rank, reverse=True)
        bounded = prepared[:max(1, int(limit or 1))]
        logger.info(
            f"Unified rerank pool: candidates={len(bounded)}, "
            f"real_data={sum('real_data_evidence' in item.retrieval_type for item in bounded)}, "
            f"linked={sum('linked_page' in item.retrieval_type for item in bounded)}, "
            f"code_index={sum('code_index_evidence' in item.retrieval_type for item in bounded)}"
        )
        return bounded

    @staticmethod
    def _apply_evidence_priority(results: Sequence[RetrievalResult]) -> List[RetrievalResult]:
        prioritized: List[RetrievalResult] = []
        for result in results:
            retrieval_type = str(result.retrieval_type or "")
            has_real_data = "real_data_evidence" in retrieval_type
            is_linked = "linked_page" in retrieval_type
            bonus = 0.0
            if has_real_data:
                bonus += 0.06
            if is_linked and has_real_data:
                bonus += 0.04
            elif is_linked:
                bonus += 0.02
            if "linked_page_category" in retrieval_type:
                bonus += 0.04
            prioritized.append(
                result.model_copy(
                    update={"score": _clamp_score(float(result.score or 0.0) + bonus)}
                )
            )
        has_qwen_scores = any(
            "qwen_unified_rerank" in str(item.retrieval_type or "")
            for item in prioritized
        )
        if has_qwen_scores:
            prioritized.sort(key=lambda item: item.score, reverse=True)
        else:
            def fallback_rank(item: RetrievalResult) -> tuple[int, float]:
                result_type = str(item.retrieval_type or "")
                real_data = "real_data_evidence" in result_type
                linked = "linked_page" in result_type
                rank = 3 if real_data and linked else (2 if real_data else (1 if linked else 0))
                return rank, float(item.score or 0.0)

            prioritized.sort(key=fallback_rank, reverse=True)
        return prioritized

    def _combine_results(
        self,
        keyword_results: List[RetrievalResult],
        semantic_results: List[RetrievalResult],
        metric: Optional[ESGMetric] = None,
        report_content: Optional[ReportContent] = None,
        channel_results: Optional[Dict[str, Sequence[RetrievalResult]]] = None,
        profile=None,
    ) -> List[RetrievalResult]:
        """Fuse retrieval channels and run exact-metric reranking.

        The method keeps the old signature while supporting explicit channel
        groups for weighted RRF.  If called by older code without channel_groups,
        it still fuses keyword and semantic results.
        """
        if channel_results is None:
            channel_results = {
                "keyword": keyword_results,
                "semantic": semantic_results,
            }

        fused = rrf_fuse(channel_results)
        if metric is not None and report_content is not None:
            anchors = profile.anchor_terms if profile is not None else _extract_metric_anchor_terms(metric)
            fused.extend(_synthesize_qualitative_clusters(report_content, metric, fused, anchors))
            deduped: Dict[str, RetrievalResult] = {}
            for result in fused:
                current = deduped.get(result.segment_id)
                if current is None or float(result.score or 0.0) > float(current.score or 0.0):
                    deduped[result.segment_id] = result
            fused = list(deduped.values())

        fused.sort(key=lambda item: item.score, reverse=True)
        if metric is None:
            return fused[: max(1, int(getattr(self.config, "top_k", 10) or 10))]

        final_k = _target_window_size(self.config, metric, observed_matches=len(fused))
        final_k = max(11, final_k)
        if profile is None:
            profile = build_metric_retrieval_profile(metric)
        deterministic = exact_metric_rerank(
            fused,
            metric=metric,
            profile=profile,
            report_content=report_content,
            top_k=len(fused),
        )
        deterministic = _rebalance_qualitative_results(
            metric,
            deterministic,
            report_content=report_content,
        )
        rerank_limit = max(
            final_k,
            int(getattr(self.semantic_retriever, "reranker_top_k", final_k) or final_k),
        )
        candidate_pool = self._prepare_unified_rerank_candidates(
            deterministic,
            report_content,
            profile,
            rerank_limit,
        )
        reranked = self.semantic_retriever.rerank_candidates(candidate_pool, metric)
        reranked = self._apply_evidence_priority(reranked)
        return reranked[:final_k]

    def retrieve_for_collection(
        self,
        report_content: ReportContent,
        metric_collection: MetricCollection,
    ) -> List[MetricRetrievalResult]:
        """Retrieve evidence for all metrics in a collection."""
        results: List[MetricRetrievalResult] = []
        expansions_map = {exp.metric_id: exp for exp in metric_collection.semantic_expansions}
        for metric in metric_collection.metrics:
            logger.info(f"Retrieving metric: {metric.metric_name}")
            results.append(
                self.retrieve_for_metric(
                    report_content,
                    metric,
                    expansions_map.get(metric.metric_id),
                )
            )
        logger.info(f"Completed metric collection retrieval, processed {len(results)} metrics")
        return results

    def generate_retrieval_report(self, retrieval_results: List[MetricRetrievalResult]) -> str:
        """Generate a Markdown retrieval report."""
        report_lines = [
            "# ESG Metric Retrieval Report\n",
            f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Number of Retrieved Metrics**: {len(retrieval_results)}\n",
            f"**Total Matched Segments**: {sum(r.total_matches for r in retrieval_results)}\n",
            "---\n",
        ]
        for result in retrieval_results:
            report_lines.extend([
                f"## {result.metric_name} ({result.metric_id})\n",
                f"**Total Matches**: {result.total_matches}\n",
                f"**Lexical Matches**: {len(result.keyword_results)}\n",
                f"**Semantic Matches**: {len(result.semantic_results)}\n",
                "",
            ])
            if result.combined_results:
                report_lines.append("### Best Matching Segments\n")
                for i, match in enumerate(result.combined_results[:5], 1):
                    report_lines.extend([
                        f"**{i}. Segment {match.segment_id}** (Page: {match.page_number}, Score: {match.score:.3f})\n",
                        f"*Retrieval Type: {match.retrieval_type}*\n",
                        f"*Matched Keywords: {', '.join(match.matched_keywords) if match.matched_keywords else 'None'}*\n",
                        f"```\n{match.content[:200]}...\n```\n",
                        "",
                    ])
            report_lines.append("---\n")
        return "\n".join(report_lines)
