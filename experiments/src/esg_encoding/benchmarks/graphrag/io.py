"""Atomic checkpoint I/O for GraphRAG compliance JSON."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Set


def empty_result(
    *,
    report_id: str,
    scope_key: str,
    filename: str = "",
) -> Dict[str, Any]:
    return {
        "report_id": report_id,
        "framework": "GraphRAG",
        "scope_key": scope_key,
        "filename": filename,
        "items": [],
        "failed": [],
        "meta": {"total": 0, "ok": 0, "failed": 0},
    }


def load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def completed_ids(data: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for key in ("items", "failed"):
        for row in data.get(key) or []:
            if isinstance(row, dict) and row.get("id") is not None:
                ids.add(str(row["id"]))
    return ids


def _format_reasoning_for_ui(reasoning: Any) -> str:
    """Frontend String(reasoning) expects a string, not a hop array."""
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        lines = []
        for hop in reasoning:
            if not isinstance(hop, dict):
                continue
            kind = hop.get("kind", "")
            text = hop.get("text", "")
            loc = hop.get("loc")
            hop_i = hop.get("hop", "")
            loc_s = f" @{loc}" if loc is not None else ""
            lines.append(f"[{hop_i}:{kind}{loc_s}] {text}")
        return "\n".join(lines)
    return str(reasoning)


def _item_to_metric_analysis(item: Dict[str, Any]) -> Dict[str, Any]:
    """Project a GraphRAG item into the SASB-style row the frontend table expects."""
    answer = item.get("answer")
    answer_s = "" if answer is None else str(answer)
    is_nd = answer_s.strip().lower() in {"not_disclosed", "n/a", "na", ""}
    return {
        "metric_id": item.get("metric_id") or item.get("id") or "",
        "metric_name": item.get("question") or item.get("metric_name") or item.get("id") or "",
        "metric_code": item.get("code") or item.get("metric_code") or "",
        "disclosure_status": "not_disclosed" if is_nd else "fully_disclosed",
        "reasoning": _format_reasoning_for_ui(item.get("reasoning")),
        "value": answer_s if not is_nd else "n/a",
        "unit": item.get("unit") or "",
        "category": item.get("category") or "",
        "topic": item.get("topic") or "",
        "type": item.get("type") or "GraphRAG",
        "definition": "",
        "page": item.get("page"),
        "context": None,
        "evidence_segments": item.get("evidence_segments") or [],
        "improvement_suggestions": [],
        "graphrag_id": item.get("id"),
        "graphrag_status": item.get("status") or "ok",
    }


def _failed_to_metric_analysis(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metric_id": row.get("metric_id") or row.get("id") or "",
        "metric_name": row.get("question") or row.get("id") or "",
        "metric_code": row.get("code") or "",
        "disclosure_status": "not_disclosed",
        "reasoning": str(row.get("error") or "validation_failed"),
        "value": "n/a",
        "unit": row.get("unit") or "",
        "category": "",
        "topic": "",
        "type": "GraphRAG",
        "definition": "",
        "page": None,
        "context": None,
        "evidence_segments": [],
        "improvement_suggestions": [],
        "graphrag_id": row.get("id"),
        "graphrag_status": row.get("status") or "validation_failed",
    }


def project_metric_analyses_for_ui(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep GraphRAG `items`/`failed`/`meta`, and also emit `metric_analyses` so the
    existing Results table (which only reads metric_analyses) can render rows.
    """
    items = list(data.get("items") or [])
    failed = list(data.get("failed") or [])
    analyses = [_item_to_metric_analysis(x) for x in items if isinstance(x, dict)]
    analyses.extend(_failed_to_metric_analysis(x) for x in failed if isinstance(x, dict))
    data["metric_analyses"] = analyses
    total = len(analyses)
    fully = sum(1 for a in analyses if a.get("disclosure_status") == "fully_disclosed")
    not_d = total - fully
    data["total_metrics"] = total
    data["total_metrics_analyzed"] = total
    data["overall_score"] = (fully / total) if total else 0.0
    data["overall_compliance_score"] = data["overall_score"]
    data["disclosure_summary"] = {
        "fully_disclosed": fully,
        "partially_disclosed": 0,
        "not_disclosed": not_d,
    }
    return data


def recompute_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    items = list(data.get("items") or [])
    failed = list(data.get("failed") or [])
    data["items"] = items
    data["failed"] = failed
    data["meta"] = {
        "total": len(items) + len(failed),
        "ok": len(items),
        "failed": len(failed),
    }
    return project_metric_analyses_for_ui(data)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON via temp file + os.replace so a mid-run kill still leaves valid JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(recompute_meta(data), ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def upsert_item(data: Dict[str, Any], item: Dict[str, Any]) -> None:
    qid = str(item.get("id") or "")
    items = [r for r in (data.get("items") or []) if str(r.get("id")) != qid]
    failed = [r for r in (data.get("failed") or []) if str(r.get("id")) != qid]
    items.append(item)
    data["items"] = items
    data["failed"] = failed


def upsert_failed(data: Dict[str, Any], failed_row: Dict[str, Any]) -> None:
    qid = str(failed_row.get("id") or "")
    items = [r for r in (data.get("items") or []) if str(r.get("id")) != qid]
    failed = [r for r in (data.get("failed") or []) if str(r.get("id")) != qid]
    failed.append(failed_row)
    data["items"] = items
    data["failed"] = failed
