#!/usr/bin/env python3
"""Generate ESG benchmark questions from packs using L1–L4 prompts.

Reads packs from ESG_KG/output/qg/packs/
Writes:
  ESG_KG/output/qg/raw/{pack_id}.json   (per-pack cache for resume)
  ESG_KG/output/qg/esg_qg_dataset.json  (single final file: all questions)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qg_common import (
    DEFAULT_TARGETS,
    QUESTION_TYPES,
    call_llm,
    detect_newesg_root,
    load_dotenv,
    load_prompt,
    parse_questions_json,
    render_qg_prompt,
)


def count_triples_in_evidence(evidence_triple: str) -> int:
    if not evidence_triple:
        return 0
    return evidence_triple.count("(")


def is_bare_numeric_answer(answer: str) -> bool:
    """Reject gold answers that are only a number (hard to align in QA eval)."""
    a = (answer or "").strip().rstrip(".")
    if not a:
        return False
    return bool(re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", a))


def validate_question(q: Dict[str, Any], question_type: str) -> Tuple[bool, str]:
    if not isinstance(q, dict):
        return False, "not an object"
    question = str(q.get("question") or "").strip()
    answer = str(q.get("answer") or "").strip()
    if not question or not answer:
        return False, "missing question/answer"
    et = str(q.get("evidence_triple") or "")
    n = count_triples_in_evidence(et)
    if question_type == "Complex Reasoning" and n < 3:
        return False, f"need >=3 evidence triples, got {n}"
    if question_type == "Creative Generation" and n < 5:
        return False, f"need >=5 evidence triples, got {n}"
    # soft length checks
    if question_type != "Creative Generation" and len(answer.split()) > 40:
        return False, "answer too long for non-creative type"
    if question_type == "Creative Generation" and len(answer.split()) > 180:
        return False, "creative answer too long"
    if question_type in ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize"):
        if is_bare_numeric_answer(answer):
            return False, "bare numeric answer (need value+metric/unit)"
    return True, "ok"


def load_all_packs(packs_dir: Path, pdf_filter: str = "") -> List[Dict[str, Any]]:
    needle = (pdf_filter or "").strip().lower()
    packs: List[Dict[str, Any]] = []
    for fp in sorted(packs_dir.glob("*_packs.json")):
        if fp.name == "index.json":
            continue
        if needle and needle not in fp.name.lower():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for pk in data.get("packs") or []:
            if isinstance(pk, dict):
                packs.append(pk)
    return packs


def allocate_packs_for_targets(
    packs: List[Dict[str, Any]],
    targets: Dict[str, int],
    *,
    oversample: float = 1.2,
    questions_per_pack: int = 8,
) -> List[Dict[str, Any]]:
    """Pick packs per type so expected questions cover targets * oversample."""
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pk in packs:
        by_type[str(pk.get("question_type"))].append(pk)

    selected: List[Dict[str, Any]] = []
    for qt in QUESTION_TYPES:
        need_q = int(targets.get(qt, 0) * oversample)
        need_packs = max(1, (need_q + questions_per_pack - 1) // questions_per_pack)
        pool = by_type.get(qt) or []
        if not pool:
            print(f"WARN: no packs for {qt}", file=sys.stderr)
            continue
        # round-robin across documents for diversity
        by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pk in pool:
            by_doc[str(pk.get("document_id"))].append(pk)
        docs = sorted(by_doc.keys())
        picked: List[Dict[str, Any]] = []
        i = 0
        while len(picked) < need_packs and docs:
            doc = docs[i % len(docs)]
            if by_doc[doc]:
                picked.append(by_doc[doc].pop(0))
            else:
                docs = [d for d in docs if by_doc[d]]
                if not docs:
                    break
                continue
            i += 1
        if len(picked) < need_packs:
            print(
                f"WARN: {qt} only {len(picked)} unique packs "
                f"(wanted {need_packs}); not wrapping to avoid duplicates",
                file=sys.stderr,
            )
        selected.extend(picked)
        print(f"Allocate {qt}: packs={len(picked)} (need_q~{need_q})")
    return selected


def trim_to_targets(
    questions: List[Dict[str, Any]], targets: Dict[str, int]
) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_type[str(q.get("question_type"))].append(q)
    out: List[Dict[str, Any]] = []
    for qt in QUESTION_TYPES:
        out.extend(by_type.get(qt, [])[: targets.get(qt, 0)])
    return out


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--packs-dir",
        default=str(root / "ESG_KG" / "output" / "qg" / "packs"),
    )
    p.add_argument(
        "--out-dir",
        default=str(root / "ESG_KG" / "output" / "qg"),
    )
    p.add_argument("--pdf", default="", help="Filter packs by document substring")
    p.add_argument("--model", default=None)
    p.add_argument("--num-per-pack", type=int, default=8)
    p.add_argument("--oversample", type=float, default=1.2)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--max-packs",
        type=int,
        default=None,
        help="Limit number of pack LLM calls (for smoke tests)",
    )
    p.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim final dataset to paper target counts",
    )
    p.add_argument(
        "--targets",
        default="",
        help='Override targets JSON, e.g. {"Fact Retrieval":10,...}',
    )
    args = p.parse_args()

    packs_dir = Path(args.packs_dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "esg_qg_dataset.json"

    targets = dict(DEFAULT_TARGETS)
    if args.targets.strip():
        targets.update(json.loads(args.targets))

    all_packs = load_all_packs(packs_dir, pdf_filter=args.pdf)
    if not all_packs:
        print(f"No packs found in {packs_dir}. Run build_qg_packs.py first.", file=sys.stderr)
        return 2

    selected = allocate_packs_for_targets(
        all_packs,
        targets,
        oversample=args.oversample,
        questions_per_pack=args.num_per_pack,
    )
    if args.max_packs is not None:
        selected = selected[: max(0, args.max_packs)]

    cfg = load_dotenv()
    model = args.model or cfg["LLM_MODEL"]
    print(f"Model: {model}")
    print(f"Packs to call: {len(selected)}")
    print(f"Targets: {targets}")

    prompts = {qt: load_prompt(qt) for qt in QUESTION_TYPES}
    accepted: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i, pk in enumerate(selected, 1):
        pack_id = str(pk.get("pack_id") or f"pack_{i}")
        qt = str(pk.get("question_type") or "Fact Retrieval")
        raw_path = raw_dir / f"{pack_id}.json"
        if args.skip_existing and raw_path.is_file():
            try:
                prev = json.loads(raw_path.read_text(encoding="utf-8"))
                for q in prev.get("questions") or []:
                    ok, _ = validate_question(q, qt)
                    if ok:
                        q = dict(q)
                        q.setdefault("question_type", qt)
                        q["document_id"] = pk.get("document_id")
                        q["pack_id"] = pack_id
                        accepted.append(q)
                print(f"[{i}/{len(selected)}] SKIP {pack_id}")
                continue
            except Exception as exc:
                print(f"WARN reload {raw_path}: {exc}", file=sys.stderr)

        evidence = str(pk.get("evidence_block") or "")
        if not evidence:
            print(f"[{i}/{len(selected)}] SKIP empty evidence {pack_id}")
            continue

        prompt = render_qg_prompt(
            prompts[qt],
            num=args.num_per_pack,
            evidence_block=evidence,
            existing_questions="",
        )
        try:
            raw = call_llm(prompt, model=model, temperature=0.3)
            parsed = parse_questions_json(raw)
        except Exception as exc:
            err = str(exc)
            errors.append({"pack_id": pack_id, "error": err})
            print(f"[{i}/{len(selected)}] FAIL {pack_id}: {exc}", file=sys.stderr)
            if "TOKEN_OR_QUOTA_EXHAUSTED" in err:
                print("Stopping due to token/quota exhaustion.", file=sys.stderr)
                break
            continue

        kept: List[Dict[str, Any]] = []
        for q in parsed.get("questions") or []:
            ok, reason = validate_question(q, qt)
            if not ok:
                continue
            item = {
                "question": q.get("question"),
                "answer": q.get("answer"),
                "question_type": qt,
                "evidence": q.get("evidence") or "",
                "evidence_triple": q.get("evidence_triple") or "",
                "document_id": pk.get("document_id"),
                "company": pk.get("company"),
                "pack_id": pack_id,
            }
            if q.get("creative_form"):
                item["creative_form"] = q.get("creative_form")
            kept.append(item)
            accepted.append(item)

        raw_path.write_text(
            json.dumps(
                {
                    "pack_id": pack_id,
                    "document_id": pk.get("document_id"),
                    "question_type": qt,
                    "model": model,
                    "questions": kept,
                    "raw_count": len(parsed.get("questions") or []),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[{i}/{len(selected)}] {pack_id} type={qt.split()[0]} "
            f"kept={len(kept)} total_accepted={len(accepted)}"
        )
        if args.sleep:
            time.sleep(args.sleep)

    final = accepted if args.no_trim else trim_to_targets(accepted, targets)
    counters: Dict[str, int] = defaultdict(int)
    for q in final:
        qt = str(q.get("question_type"))
        counters[qt] += 1
        doc = str(q.get("document_id") or "doc")
        short = {
            "Fact Retrieval": "L1",
            "Complex Reasoning": "L2",
            "Contextual Summarize": "L3",
            "Creative Generation": "L4",
        }.get(qt, "LX")
        q["id"] = f"{doc}_{short}_{counters[qt]:04d}"

    counts = {qt: sum(1 for q in final if q.get("question_type") == qt) for qt in QUESTION_TYPES}
    dataset_path.write_text(
        json.dumps(
            {
                "dataset": "ESG-GraphReason-QG",
                "model": model,
                "total": len(final),
                "counts": counts,
                "targets": targets,
                "accepted_before_trim": len(accepted),
                "errors": errors,
                "questions": final,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {dataset_path} total={len(final)} counts={counts}")
    if errors:
        print(f"Errors: {len(errors)}")
    return 0 if not any("TOKEN_OR_QUOTA_EXHAUSTED" in str(e.get("error")) for e in errors) else 3


if __name__ == "__main__":
    raise SystemExit(main())
