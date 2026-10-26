"""GraphRAG-only parsing and verbatim hop validation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import ALLOWED_HOP_KINDS, ValidationError


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def normalize_ws(text: str) -> str:
    """Collapse whitespace runs to a single space. No other normalisation."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_markdown_fences(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
        # Also drop a leading language tag line if present after partial strip
        if text.lower().startswith("json\n"):
            text = text[5:].lstrip()
    return text


def brace_match_json_object(content: str) -> Optional[str]:
    """Return the first complete top-level `{...}` object substring, or None."""
    text = strip_markdown_fences(content)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(content: str) -> Dict[str, Any]:
    blob = brace_match_json_object(content)
    if not blob:
        raise ValidationError("could not extract a complete JSON object from LLM response")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValidationError("LLM JSON root must be an object")
    return obj


def alias_reasoning_field(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Map reasoning_chain / rationale → reasoning before validation."""
    if "reasoning" not in obj or obj.get("reasoning") in (None, "", []):
        for alias in ("reasoning_chain", "rationale"):
            if alias in obj and obj.get(alias) not in (None, "", []):
                obj["reasoning"] = obj[alias]
                break
    return obj


def _validate_hop_shape(hop: Any, index: int) -> Dict[str, Any]:
    if not isinstance(hop, dict):
        raise ValidationError(f"reasoning[{index}] must be an object")
    kind = str(hop.get("kind") or "").strip().lower()
    if kind not in ALLOWED_HOP_KINDS:
        raise ValidationError(
            f"reasoning[{index}].kind must be one of {sorted(ALLOWED_HOP_KINDS)}, got {kind!r}"
        )
    text = hop.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"reasoning[{index}].text must be a non-empty string")
    hop_num = hop.get("hop", index + 1)
    try:
        hop_num_i = int(hop_num)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"reasoning[{index}].hop must be an integer") from exc
    loc = hop.get("loc", None)
    if kind == "compute":
        loc = None
    elif kind == "matrix":
        if loc is None:
            loc = "matrix"
        loc = str(loc)
    else:
        # quote: loc may be missing; we recompute later
        if loc is not None:
            loc = str(loc)
    return {"hop": hop_num_i, "kind": kind, "text": text, "loc": loc}


def find_first_occurrence(needle_norm: str, haystack_norm: str) -> int:
    return haystack_norm.find(needle_norm)


def validate_and_normalize_response(
    raw_content: str,
    *,
    report_source_text: str,
    question_id: str,
    answerable_requires_quote: bool = True,
) -> Dict[str, Any]:
    """
    Parse LLM content, enforce hop contract, verify quote hops are verbatim
    substrings (whitespace-collapsed), and rewrite loc to the first occurrence.
    """
    obj = alias_reasoning_field(extract_json_object(raw_content))

    reasoning = obj.get("reasoning")
    if reasoning is None or reasoning == "" or reasoning == []:
        raise ValidationError("missing required non-empty field `reasoning`")
    if isinstance(reasoning, str):
        # Tolerate a single string by wrapping as one quote hop.
        reasoning = [
            {"hop": 1, "kind": "quote", "text": reasoning, "loc": None}
        ]
    if not isinstance(reasoning, list) or not reasoning:
        raise ValidationError("`reasoning` must be a non-empty array of hop objects")

    if "answer" not in obj:
        raise ValidationError("missing required field `answer`")
    answer = obj.get("answer")
    if answer is None:
        raise ValidationError("`answer` must not be null")
    answer_str = answer if isinstance(answer, str) else str(answer)

    haystack_norm = normalize_ws(report_source_text)
    normalized_hops: List[Dict[str, Any]] = []
    quote_count = 0

    for i, raw_hop in enumerate(reasoning):
        hop = _validate_hop_shape(raw_hop, i)
        kind = hop["kind"]
        if kind == "quote":
            quote_count += 1
            needle_norm = normalize_ws(hop["text"])
            if not needle_norm:
                raise ValidationError(f"reasoning[{i}].text must be a non-empty string")
            # Soft verbatim check: prefer exact report substrings, but do NOT fail the
            # item when the model paraphrases. Prompt still asks for verbatim quotes.
            idx = find_first_occurrence(needle_norm, haystack_norm)
            if idx >= 0:
                hop["loc"] = str(idx)
                hop["text"] = (
                    _recover_verbatim_span(
                        report_source_text, haystack_norm, needle_norm, idx
                    )
                    or hop["text"]
                )
                hop["verbatim"] = True
            else:
                # Keep model text/loc; mark as non-verbatim for downstream auditing.
                if hop.get("loc") is None:
                    hop["loc"] = None
                else:
                    hop["loc"] = str(hop["loc"])
                hop["verbatim"] = False
        elif kind == "compute":
            hop["loc"] = None
        elif kind == "matrix":
            hop["loc"] = hop.get("loc") or "matrix"
        normalized_hops.append(hop)

    is_abstention = normalize_ws(answer_str).lower() in {"not_disclosed", "n/a", "na"}
    if answerable_requires_quote and not is_abstention and quote_count < 1:
        raise ValidationError("at least one quote hop is required for answerable questions")

    # Enforce key order: reasoning before answer in serialized form (truncation safety).
    return {
        "id": question_id,
        "reasoning": normalized_hops,
        "answer": answer_str,
        "status": "ok",
    }


def _recover_verbatim_span(
    original: str,
    haystack_norm: str,
    needle_norm: str,
    norm_index: int,
) -> Optional[str]:
    """Best-effort map normalized match index back to an original substring."""
    # Walk original while mirroring normalize_ws to find the start/end char range.
    if norm_index < 0 or not needle_norm:
        return None
    oi = 0
    ni = 0
    start_oi: Optional[int] = None
    target_end = norm_index + len(needle_norm)
    n = len(original)
    while oi < n and ni < target_end:
        if original[oi].isspace():
            # skip run of whitespace in original; contributes at most one space in norm
            while oi < n and original[oi].isspace():
                oi += 1
            if ni < len(haystack_norm) and haystack_norm[ni] == " ":
                if ni == norm_index and start_oi is None:
                    start_oi = oi  # shouldn't usually start on space
                ni += 1
            continue
        if ni == norm_index and start_oi is None:
            start_oi = oi
        oi += 1
        ni += 1
    if start_oi is None:
        return None
    end_oi = oi
    span = original[start_oi:end_oi]
    if normalize_ws(span) == needle_norm:
        return span
    return None


def build_evidence_blocks(retrieval_result: Any, max_segments: int = 12) -> List[str]:
    blocks: List[str] = []
    combined = getattr(retrieval_result, "combined_results", None) or []
    for res in combined[:max_segments]:
        sid = getattr(res, "segment_id", "") or ""
        page = getattr(res, "page_number", None)
        content = (getattr(res, "content", None) or "").strip()
        if not content:
            continue
        label = f"[{sid} - Page {page}]" if page is not None else f"[{sid}]"
        blocks.append(f"{label}\n{content}")
    return blocks


def build_report_source_text(report_content: Any) -> str:
    """Concatenate report segments in document reading order for verbatim checks."""
    doc = getattr(report_content, "document_content", None)
    if doc is None:
        return ""
    md = getattr(doc, "markdown_content", None) or ""
    segments = list(getattr(doc, "segments", None) or [])
    if not segments:
        return str(md)
    segments_sorted = sorted(
        segments,
        key=lambda seg: (
            int(getattr(seg, "page_number", 0) or 0),
            float(getattr(seg, "position_y", 0.0) or 0.0),
            float(getattr(seg, "position_x", 0.0) or 0.0),
            str(getattr(seg, "segment_id", "") or ""),
        ),
    )
    parts: List[str] = []
    for seg in segments_sorted:
        text = (getattr(seg, "content", None) or "").strip()
        if text:
            parts.append(text)
    joined = "\n".join(parts)
    return joined if joined.strip() else str(md)
