#!/usr/bin/env python3
"""Merge Enel QG supplement into main esg_qg_dataset.json and rebuild packs index."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from qg_common import QUESTION_TYPES, detect_newesg_root


def main() -> int:
    root = detect_newesg_root()
    main_p = root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json"
    enel_p = root / "ESG_KG" / "output" / "qg" / "enel_fill" / "esg_qg_dataset.json"
    packs_dir = root / "ESG_KG" / "output" / "qg" / "packs"

    main = json.loads(main_p.read_text(encoding="utf-8"))
    enel = json.loads(enel_p.read_text(encoding="utf-8"))

    kept = [
        q
        for q in (main.get("questions") or [])
        if str(q.get("document_id") or "") != "enel-2024"
    ]
    seen = {(str(q.get("question") or "").strip().lower()) for q in kept}
    uniq_added = []
    for q in enel.get("questions") or []:
        key = str(q.get("question") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq_added.append(dict(q))

    short = {
        "Fact Retrieval": "L1",
        "Complex Reasoning": "L2",
        "Contextual Summarize": "L3",
        "Creative Generation": "L4",
    }
    counters: dict[str, int] = defaultdict(int)
    for q in uniq_added:
        qt = str(q.get("question_type"))
        counters[qt] += 1
        q["id"] = f"enel-2024_{short.get(qt, 'LX')}_{counters[qt]:04d}"

    merged = kept + uniq_added
    counts = {
        qt: sum(1 for q in merged if q.get("question_type") == qt) for qt in QUESTION_TYPES
    }
    main["questions"] = merged
    main["total"] = len(merged)
    main["counts"] = counts
    main["enel_supplement"] = {
        "added": len(uniq_added),
        "enel_counts": {
            qt: sum(1 for q in uniq_added if q.get("question_type") == qt)
            for qt in QUESTION_TYPES
        },
        "model": enel.get("model"),
    }
    main_p.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"merged_total={main['total']} counts={counts}")
    print(f"enel_added={main['enel_supplement']}")

    index = []
    totals: dict[str, int] = defaultdict(int)
    for fp in sorted(packs_dir.glob("*_packs.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        doc = data.get("document_id") or fp.stem.replace("_packs", "")
        for pk in data.get("packs") or []:
            totals[str(pk.get("question_type"))] += 1
            index.append(
                {
                    "pack_id": pk.get("pack_id"),
                    "document_id": doc,
                    "question_type": pk.get("question_type"),
                    "triple_count": pk.get("triple_count"),
                    "est_tokens": pk.get("est_tokens"),
                    "file": fp.name,
                }
            )
    (packs_dir / "index.json").write_text(
        json.dumps(
            {
                "pack_count": len(index),
                "counts_by_type": dict(totals),
                "packs": index,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"packs_index={len(index)} totals={dict(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
