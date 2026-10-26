"""GraphRAG benchmark runner: per-question isolation, retries, checkpointing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .io import (
    atomic_write_json,
    completed_ids,
    empty_result,
    load_existing,
    upsert_failed,
    upsert_item,
)
from .prompt import build_messages
from .schema import ValidationError
from .validate import (
    build_evidence_blocks,
    build_report_source_text,
    validate_and_normalize_response,
)

MAX_RETRIES = 3


def _question_id_from_metric(metric: Any) -> str:
    mid = str(getattr(metric, "metric_id", "") or "")
    # metric_id looks like graphrag_transportation_automobile_T1-1
    if "_T" in mid:
        return mid.rsplit("_", 1)[-1]
    code = getattr(metric, "metric_code", None)
    if code:
        return str(code)
    return mid or "unknown"


def _call_llm(llm_client: Any, model_name: str, messages: List[dict]) -> str:
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
    }
    # Prefer JSON object mode when the provider supports it; ignore if rejected.
    try:
        kwargs["response_format"] = {"type": "json_object"}
        response = llm_client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("response_format", None)
        response = llm_client.chat.completions.create(**kwargs)
    content = ""
    try:
        content = (response.choices[0].message.content or "").strip()
    except Exception:
        content = ""
    return content


def run_graphrag_benchmark(
    *,
    metrics: Any,
    retrieval_results: List[Any],
    report_content: Any,
    llm_client: Any,
    model_name: str,
    output_path: Path,
    report_id: str,
    scope_key: str,
    filename: str = "",
    max_retries: int = MAX_RETRIES,
    method: str = "GraphRAG",
    retrieval_mode: str = "dual_channel",
) -> Dict[str, Any]:
    """
    Answer each GraphRAG question with hop-style reasoning.

    Never aborts the full run on a single question failure. After every question,
    atomically writes ``output_path`` with ``items`` / ``failed`` / ``meta``.
    """
    output_path = Path(output_path)
    report_source_text = build_report_source_text(report_content)

    existing = load_existing(output_path)
    if existing.get("framework") == "GraphRAG" and isinstance(existing.get("items"), list):
        data = existing
        data.setdefault("report_id", report_id)
        data.setdefault("scope_key", scope_key)
        data.setdefault("filename", filename)
        data.setdefault("failed", [])
    else:
        data = empty_result(report_id=report_id, scope_key=scope_key, filename=filename)
    data["method"] = method
    data["retrieval_mode"] = retrieval_mode

    # Ensure a valid file exists even before the first question finishes.
    atomic_write_json(output_path, data)

    done = completed_ids(data)
    retrieval_by_id = {
        str(getattr(r, "metric_id", "") or ""): r for r in (retrieval_results or [])
    }
    metric_list = list(getattr(metrics, "metrics", None) or metrics or [])

    logger.info(
        f"[GraphRAG] starting benchmark scope={scope_key} questions={len(metric_list)} "
        f"resume_skip={len(done)} out={output_path}"
    )

    for metric in metric_list:
        qid = _question_id_from_metric(metric)
        if qid in done:
            logger.info(f"[GraphRAG] skip already completed id={qid}")
            continue

        question = str(getattr(metric, "metric_name", "") or "")
        unit = getattr(metric, "unit", None)
        code = getattr(metric, "metric_code", None)
        retrieval = retrieval_by_id.get(str(getattr(metric, "metric_id", "") or ""))
        evidence_blocks = build_evidence_blocks(retrieval) if retrieval is not None else []

        feedback: Optional[str] = None
        last_error = "unknown validation error"
        success_item: Optional[Dict[str, Any]] = None

        try:
            for attempt in range(1, max_retries + 1):
                messages = build_messages(
                    question_id=qid,
                    question=question,
                    unit=unit,
                    code=code,
                    evidence_blocks=evidence_blocks,
                    feedback=feedback,
                )
                t0 = time.perf_counter()
                try:
                    raw = _call_llm(llm_client, model_name, messages)
                except Exception as llm_exc:
                    last_error = f"llm_request_failed: {llm_exc}"
                    feedback = last_error
                    logger.warning(
                        f"[GraphRAG] LLM call failed id={qid} attempt={attempt}/{max_retries}: {llm_exc}"
                    )
                    continue
                try:
                    success_item = validate_and_normalize_response(
                        raw,
                        report_source_text=report_source_text,
                        question_id=qid,
                    )
                    success_item["metric_id"] = getattr(metric, "metric_id", None)
                    success_item["code"] = code
                    success_item["question"] = question
                    success_item["unit"] = unit
                    success_item["attempts"] = attempt
                    success_item["latency_seconds"] = round(time.perf_counter() - t0, 3)
                    break
                except ValidationError as vex:
                    last_error = vex.message
                    feedback = vex.message
                    logger.warning(
                        f"[GraphRAG] validation failed id={qid} attempt={attempt}/{max_retries}: {vex.message}"
                    )
                except Exception as exc:
                    last_error = f"unexpected: {exc}"
                    feedback = last_error
                    logger.warning(
                        f"[GraphRAG] unexpected validation error id={qid} attempt={attempt}: {exc}"
                    )

            if success_item is not None:
                upsert_item(data, success_item)
            else:
                upsert_failed(
                    data,
                    {
                        "id": qid,
                        "metric_id": getattr(metric, "metric_id", None),
                        "question": question,
                        "error": last_error,
                        "status": "validation_failed",
                    },
                )
        except Exception as exc:
            # Per-question isolation: never abort the run.
            logger.exception(f"[GraphRAG] isolated failure id={qid}")
            upsert_failed(
                data,
                {
                    "id": qid,
                    "metric_id": getattr(metric, "metric_id", None),
                    "question": question,
                    "error": str(exc),
                    "status": "validation_failed",
                },
            )

        atomic_write_json(output_path, data)
        done.add(qid)

    atomic_write_json(output_path, data)
    logger.info(
        f"[GraphRAG] finished scope={scope_key} meta={data.get('meta')} path={output_path}"
    )
    return data
