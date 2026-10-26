"""Merge ESG QA results.json with gold answers into GraphRAG-Bench eval rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_as_str(x) for x in value]
        return "\n".join(p for p in parts if p)
    return str(value).strip()


def normalize_context(context: Any) -> List[str]:
    """Paper metrics expect List[str] contexts."""
    if context is None:
        return []
    if isinstance(context, list):
        out: List[str] = []
        for c in context:
            s = _as_str(c)
            if s:
                out.append(s)
        return out
    text = _as_str(context)
    if not text:
        return []
    # HippoRAG / LightRAG often return one multi-passage string joined by blank lines.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else [text]


def normalize_evidence(evidence: Any) -> List[str]:
    if evidence is None:
        return []
    if isinstance(evidence, list):
        return [s for s in (_as_str(x) for x in evidence) if s]
    text = _as_str(evidence)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else [text]


def load_gold_index(gold_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    items: List[Dict[str, Any]]
    if isinstance(raw, dict) and "items" in raw:
        items = list(raw["items"])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported gold format in {gold_path}")

    index: Dict[str, Dict[str, Any]] = {}
    for item in items:
        qid = str(item.get("id", "")).strip()
        if qid:
            index[qid] = item
    return index


def load_qa_results(qa_path: Union[str, Path]) -> List[Dict[str, Any]]:
    raw = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "results" in raw:
        raw = raw["results"]
    if not isinstance(raw, list):
        raise ValueError(f"QA results must be a list (or {{results: [...]}}): {qa_path}")
    return raw


def build_eval_rows(
    qa_rows: List[Dict[str, Any]],
    gold_index: Dict[str, Dict[str, Any]],
    *,
    skip_errors: bool = True,
    require_gold: bool = True,
) -> List[Dict[str, Any]]:
    """Produce paper-compatible eval items.

    Fields:
      id, question, question_type, ground_truth, generated_answer,
      context (List[str]), evidence (List[str]), source
    """
    out: List[Dict[str, Any]] = []
    missing_gold = 0
    skipped_error = 0

    for row in qa_rows:
        if skip_errors and row.get("error"):
            skipped_error += 1
            continue

        qid = str(row.get("id", "")).strip()
        gold = gold_index.get(qid, {})
        if require_gold and not gold:
            missing_gold += 1
            continue

        ground_truth = (
            _as_str(row.get("gold_answer"))
            or _as_str(row.get("ground_truth"))
            or _as_str(gold.get("answer"))
        )
        evidence_list = normalize_evidence(row.get("evidence"))
        if not evidence_list:
            evidence_list = normalize_evidence(gold.get("evidence"))

        out.append(
            {
                "id": qid,
                "source": row.get("source") or gold.get("source") or "",
                "question": row.get("question") or gold.get("question") or "",
                "question_type": row.get("question_type")
                or gold.get("question_type")
                or "Uncategorized",
                "ground_truth": ground_truth,
                "generated_answer": _as_str(row.get("generated_answer")),
                "context": normalize_context(row.get("context")),
                "evidence": evidence_list,
            }
        )

    if missing_gold:
        print(f"[adapter] skipped {missing_gold} rows with missing gold")
    if skipped_error:
        print(f"[adapter] skipped {skipped_error} rows with error")
    print(f"[adapter] built {len(out)} eval rows from {len(qa_rows)} QA results")
    return out


def merge_qa_with_gold(
    qa_path: Union[str, Path],
    gold_path: Union[str, Path],
    *,
    skip_errors: bool = True,
) -> List[Dict[str, Any]]:
    return build_eval_rows(
        load_qa_results(qa_path),
        load_gold_index(gold_path),
        skip_errors=skip_errors,
    )


def save_merged_json(
    rows: List[Dict[str, Any]],
    out_path: Union[str, Path],
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
