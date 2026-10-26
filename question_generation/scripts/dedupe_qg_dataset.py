#!/usr/bin/env python3
"""Rebuild esg_qg_dataset.json from raw packs with question dedupe."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from qg_common import DEFAULT_TARGETS, QUESTION_TYPES, detect_newesg_root


def main() -> int:
    root = detect_newesg_root()
    out_dir = root / "ESG_KG" / "output" / "qg"
    raw_dir = out_dir / "raw"
    dataset_path = out_dir / "esg_qg_dataset.json"

    prev = {}
    if dataset_path.is_file():
        prev = json.loads(dataset_path.read_text(encoding="utf-8"))
    targets = dict(prev.get("targets") or DEFAULT_TARGETS)

    all_q = []
    seen = set()
    for fp in sorted(raw_dir.glob("*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        qt = obj.get("question_type")
        for q in obj.get("questions") or []:
            key = (
                str(q.get("question") or "").strip().lower(),
                str(q.get("answer") or "").strip().lower(),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            item = dict(q)
            item.setdefault("question_type", qt)
            item.setdefault("document_id", obj.get("document_id"))
            item.setdefault("pack_id", obj.get("pack_id"))
            all_q.append(item)

    by: dict[str, list] = defaultdict(list)
    for q in all_q:
        by[str(q.get("question_type"))].append(q)

    final = []
    for qt in QUESTION_TYPES:
        final.extend(by.get(qt, [])[: targets.get(qt, 0)])

    counters: dict[str, int] = defaultdict(int)
    short = {
        "Fact Retrieval": "L1",
        "Complex Reasoning": "L2",
        "Contextual Summarize": "L3",
        "Creative Generation": "L4",
    }
    for q in final:
        qt = str(q.get("question_type"))
        counters[qt] += 1
        doc = str(q.get("document_id") or "doc")
        q["id"] = f"{doc}_{short.get(qt, 'LX')}_{counters[qt]:04d}"

    counts = {qt: sum(1 for q in final if q.get("question_type") == qt) for qt in QUESTION_TYPES}
    out = {
        "dataset": prev.get("dataset", "ESG-GraphReason-QG"),
        "model": prev.get("model"),
        "total": len(final),
        "counts": counts,
        "targets": targets,
        "unique_pool": len(all_q),
        "deduped": True,
        "accepted_before_trim": prev.get("accepted_before_trim"),
        "errors": prev.get("errors") or [],
        "questions": final,
    }
    dataset_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"unique_pool={len(all_q)} final_total={len(final)} counts={counts}")
    shortfall = {k: targets[k] - counts[k] for k in QUESTION_TYPES if counts[k] < targets[k]}
    if shortfall:
        print(f"shortfall={shortfall}")
    print(f"Wrote {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
