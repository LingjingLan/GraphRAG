#!/usr/bin/env python3
"""Convert ESG QG dataset → GraphRAG-Bench style question set.

For offline eval over HippoRAG-extracted triples (NOT the upload/G1 question set).

Writes:
  Datasets/Questions/esg_questions.json                   # with gold answer/evidence
  Datasets/Questions/esg_questions_by_report/{doc}.json
  Datasets/Questions/esg_questions_blind.json             # answer/evidence blank (for QA)
  Datasets/Questions/esg_questions_blind_by_report/{doc}.json
  ESG_KG/output/qg/esg_qg_gold_for_eval.json              # gold mirror for scoring
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def detect_newesg_root() -> Path:
    here = Path(__file__).resolve()
    euler = here.parents[2]
    sibling = euler.parent
    if (sibling / "Datasets" / "Corpus").is_dir():
        return sibling
    if Path("/workspace/Datasets/Corpus").is_dir():
        return Path("/workspace")
    return sibling


def to_bench_item(q: Dict[str, Any]) -> Dict[str, Any]:
    """GraphRAG-Bench question fields + ESG extras."""
    source = str(q.get("document_id") or q.get("company") or "").strip()
    return {
        "id": str(q.get("id") or ""),
        "question": str(q.get("question") or "").strip(),
        "answer": str(q.get("answer") or "").strip(),
        "question_type": str(q.get("question_type") or "").strip(),
        "evidence": str(q.get("evidence") or "").strip(),
        "source": source,
        # Extras (ignored by GraphRAG-Bench eval if unused)
        "evidence_triple": str(q.get("evidence_triple") or "").strip(),
        "company": str(q.get("company") or "").strip(),
        "pack_id": str(q.get("pack_id") or "").strip(),
    }


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qg",
        default=str(root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json"),
    )
    p.add_argument(
        "--out",
        default=str(root / "Datasets" / "Questions" / "esg_questions.json"),
    )
    p.add_argument(
        "--gold-out",
        default=str(root / "ESG_KG" / "output" / "qg" / "esg_qg_gold_for_eval.json"),
    )
    p.add_argument(
        "--exclude-creative",
        action="store_true",
        help="Skip Creative Generation (L4)",
    )
    args = p.parse_args()

    data = json.loads(Path(args.qg).read_text(encoding="utf-8"))
    questions = list(data.get("questions") or [])
    if args.exclude_creative:
        questions = [
            q for q in questions if q.get("question_type") != "Creative Generation"
        ]

    items = [
        to_bench_item(q)
        for q in questions
        if str(q.get("question") or "").strip() and str(q.get("id") or "").strip()
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_report_dir = out_path.parent / "esg_questions_by_report"
    by_report_dir.mkdir(parents=True, exist_ok=True)
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_doc[item["source"] or "unknown"].append(item)
    for doc, rows in sorted(by_doc.items()):
        (by_report_dir / f"{doc}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # Blind set for answering: no gold answer / evidence leakage.
    blind = [
        {
            "id": item["id"],
            "question": item["question"],
            "answer": "",
            "question_type": item["question_type"],
            "evidence": "",
            "source": item["source"],
            "company": item["company"],
            "pack_id": item["pack_id"],
        }
        for item in items
    ]
    blind_path = out_path.parent / "esg_questions_blind.json"
    blind_path.write_text(
        json.dumps(blind, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    blind_by_dir = out_path.parent / "esg_questions_blind_by_report"
    blind_by_dir.mkdir(parents=True, exist_ok=True)
    blind_by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in blind:
        blind_by[item["source"] or "unknown"].append(item)
    for doc, rows in sorted(blind_by.items()):
        (blind_by_dir / f"{doc}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    gold_path = Path(args.gold_out)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    level_map = {
        "Fact Retrieval": "L1",
        "Complex Reasoning": "L2",
        "Contextual Summarize": "L3",
        "Creative Generation": "L4",
    }
    gold_items = []
    for item in items:
        gold_items.append(
            {
                "id": item["id"],
                "source": item["source"],
                "document_id": item["source"],
                "company": item["company"],
                "question_type": item["question_type"],
                "level": level_map.get(item["question_type"], ""),
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "evidence_triple": item["evidence_triple"],
                "pack_id": item["pack_id"],
            }
        )
    gold_payload = {
        "dataset": "ESG-GraphReason-QG-gold",
        "format": "GraphRAG-Bench",
        "source_qg": str(Path(args.qg)).replace("\\", "/"),
        "questions_file": str(out_path).replace("\\", "/"),
        "total": len(gold_items),
        "counts": dict(Counter(g["question_type"] for g in gold_items)),
        "level_counts": dict(Counter(g["level"] for g in gold_items)),
        "by_document": dict(Counter(g["source"] for g in gold_items)),
        "items": gold_items,
    }
    gold_path.write_text(
        json.dumps(gold_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(items)} questions → {out_path}")
    print(f"Wrote {len(by_doc)} per-report sets → {by_report_dir}")
    print(f"Wrote blind (no answer/evidence) → {blind_path}")
    print(f"Wrote {len(blind_by)} blind per-report sets → {blind_by_dir}")
    print(f"Wrote gold → {gold_path}")
    print("counts:", dict(Counter(i["question_type"] for i in items)))
    print("blind sample:", json.dumps(blind[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
