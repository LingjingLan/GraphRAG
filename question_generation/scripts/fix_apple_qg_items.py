#!/usr/bin/env python3
"""Align Apple-2024 questions in esg_qg_dataset.json with the caption-fixed corpus.

Two things happen here:

1. Every Apple evidence line is re-pointed at the current corpus statement. The
   statements now name the table they come from ("for Fiscal year 2023 energy and
   carbon footprint (corporate facilities) - Scope 1, ...").
2. Questions that were only answerable because a reader guessed the table are
   rewritten to name it. "What was Apple's Hydro percentage reported in 2024?" had
   four possible answers (Prineville 51, Reno, Denmark 20, China 17); it now asks
   about the Denmark data center explicitly.

Idempotent: evidence already matching the corpus and questions already rewritten are
left alone.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

CORPUS_NAME = "Apple-2024"
COMPANY = "Apple"

FY23_SCOPE1 = (
    "fiscal year 2023 energy and carbon footprint table for corporate facilities (Scope 1)"
)
FY23_SCOPE1_TRIPLE = "Fiscal year 2023 corporate facilities - Scope 1"
COLO = "Our colocation facilities"


def _gas_edit(qid: str, location: str, value: str) -> Dict[str, Any]:
    metric = f"{FY23_SCOPE1_TRIPLE} - {location} - Total gas (MMBtu)"
    return {
        "id": qid,
        "expect": "total gas consumption",
        "question": (
            f"What total gas consumption (MMBtu) did Apple report for {location} in its "
            f"{FY23_SCOPE1}?"
        ),
        "answer": f"{value} MMBtu.",
        "evidence_triple": (
            f"(Apple,reportsValueFor,{metric}); ({metric},was,{value})"
        ),
    }


def _biogas_edit(qid: str, row: str, value: str) -> Dict[str, Any]:
    metric = f"{FY23_SCOPE1_TRIPLE} - {row} - Renewable biogas (MMBtu)"
    return {
        "id": qid,
        "expect": "biogas",
        "question": (
            f"How much renewable biogas (MMBtu) did Apple report for {row} in its "
            f"{FY23_SCOPE1}?"
        ),
        "answer": f"{value} MMBtu.",
        "evidence_triple": (
            f"(Apple,reportsValueFor,{metric}); ({metric},was,{value})"
        ),
    }


EDITS: List[Dict[str, Any]] = [
    {
        "id": "Apple-2024_L1_0166",
        "expect": "Hydro percentage",
        "question": (
            "What was the hydro share (%) of the default grid mix at Apple's Denmark "
            "data center?"
        ),
        "answer": "20% hydro in the default grid mix.",
        "evidence_triple": (
            "(Apple,reportsValueFor,Denmark data center - default grid mix - Hydro (%)); "
            "(Denmark data center - default grid mix - Hydro (%),was,20)"
        ),
    },
    {
        "id": "Apple-2024_L1_0167",
        "expect": "FY 2016 total energy use",
        "question": (
            "What was total energy use (kWh) at Apple's colocation facilities in FY 2016?"
        ),
        "answer": "145,520,900 kWh.",
        "evidence_triple": (
            f"(Apple,reportsValueFor,{COLO} - FY 2016 - Total energy use (kWh)); "
            f"({COLO} - FY 2016 - Total energy use (kWh),was,145,520,900)"
        ),
    },
    {
        "id": "Apple-2024_L1_0168",
        "expect": "FY 2014 renewable energy",
        "question": (
            "How much renewable energy (kWh) did Apple's colocation facilities use in "
            "FY 2014?"
        ),
        "answer": "88,553,400 kWh.",
        "evidence_triple": (
            f"(Apple,reportsValueFor,{COLO} - FY 2014 - Renewable energy (kWh)); "
            f"({COLO} - FY 2014 - Renewable energy (kWh),was,88,553,400)"
        ),
    },
    _gas_edit("Apple-2024_L1_0169", "Other U.S.", "104,718"),
    _gas_edit("Apple-2024_L1_0170", "Other international", "36,384"),
    _gas_edit("Apple-2024_L1_0171", "International", "23,997"),
    _gas_edit("Apple-2024_L1_0172", "Newark, CA", "0"),
    _gas_edit("Apple-2024_L1_0173", "Elk Grove, CA", "10,175"),
    _gas_edit("Apple-2024_L1_0174", "Prineville, OR", "216"),
    _gas_edit("Apple-2024_L1_0175", "Singapore", "518"),
    _gas_edit("Apple-2024_L1_0176", "all locations combined (Total)", "1,066,257"),
    _biogas_edit("Apple-2024_L1_0177", "all locations combined (Total)", "746,506"),
    _biogas_edit("Apple-2024_L1_0178", "Corporate", "746,506"),
    _biogas_edit("Apple-2024_L1_0179", "Cupertino, CA", "202,306"),
    _biogas_edit("Apple-2024_L1_0180", "Other U.S.", "544,200"),
    _biogas_edit("Apple-2024_L1_0181", "Singapore", "0"),
    _biogas_edit("Apple-2024_L1_0182", "Maiden, NC", "0"),
    _biogas_edit("Apple-2024_L1_0183", "Newark, CA", "0"),
    _biogas_edit("Apple-2024_L1_0184", "Retail stores", "0"),
]


def corpus_statements(path: Path) -> List[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    ctx = next(x for x in items if x["corpus_name"] == CORPUS_NAME)["context"]
    out: List[str] = []
    for m in re.finditer(
        r"## Page \d+\n+### Statements\n(.*?)\n\n### Text\n", ctx, re.DOTALL
    ):
        if m.group(1).strip() == "(none)":
            continue
        out.extend(m.group(1).split("\n"))
    return out


def stmt_key(text: str) -> str:
    t = (text or "").replace("\u2082", "2").replace("\u2083", "3")
    t = re.sub(r"\s*\(page\s+\d+\)\s*\.?\s*$", "", t.strip())
    return re.sub(r"\s+", " ", t).strip().rstrip(".").lower()


def loose_key(text: str) -> str:
    """Key that survives the LaTeX the QG pass stripped out ("$ tCO_{{2}}e $" -> tco2e)."""
    t = (text or "").replace("\u2082", "2").replace("\u2083", "3")
    t = re.sub(r"\s*\(page\s+\d+\)\s*\.?\s*$", "", t.strip())
    # Footnote markers ("$ ^{{5}} $") survive in the corpus but not in the QG evidence.
    t = re.sub(r"\$\s*\^\{+\s*\d+\s*\}+\s*\$", " ", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


CAPTION_RE = re.compile(r"^In \d{4}, ((?:for|regarding) .+?), Apple\b")


def caption_phrase(statement: str) -> str:
    m = CAPTION_RE.match(statement or "")
    return m.group(1) if m else ""


def apply_caption(evidence_part: str, phrase: str) -> str:
    m = re.match(r"^In (\d{4}), ", evidence_part)
    if not m or not phrase or phrase in evidence_part:
        return evidence_part
    return f"In {m.group(1)}, {phrase}, " + evidence_part[m.end() :]


STALE_TITLE_RE = re.compile(r"^In (\d{4}), regarding (.+?), Apple reports: (.+)$")


def row_label(rest: str) -> str:
    """Leading row label of a condensed evidence line ("Paper Used: Metric tonnes")."""
    label = re.split(r":| \(|\.$", rest.strip(), maxsplit=1)[0]
    return label.strip()


# Page headings the statements used before the captions were added, for rows whose
# label alone appears in more than one table.
STALE_TITLES = {
    "Increasing recycled content in our products": (
        "Recycled materials - our transition to certified recycled materials by product line"
    ),
    "Collaborative impact": "Key partnerships and memberships",
}


def caption_for_label(entries: List[tuple], label: str, rest: str = "") -> str:
    """Unique caption of the corpus statements that report this row label."""
    needles = [loose_key(f"reports: {label}"), loose_key(f"Apple's {label}")]
    needles = [n for n in needles if len(n) > 12]
    if not needles:
        return ""
    matches = [(text, cap) for text, cap in entries if any(n in text for n in needles)]
    caps = {cap for _, cap in matches}
    if len(caps) > 1 and rest:
        # Same row label in several tables: fall back to the rest of the line, which
        # usually carries a boundary or unit only one of them uses.
        tail = loose_key(rest)
        lbl = loose_key(label)
        if tail.startswith(lbl):
            tail = tail[len(lbl) :]
        tail = tail[:40]
        if len(tail) >= 12:
            narrowed = {cap for text, cap in matches if tail in text}
            if len(narrowed) == 1:
                caps = narrowed
    return caps.pop() if len(caps) == 1 else ""


def captioned_entries(corpus_path: Path) -> List[tuple]:
    out = []
    for stmt in corpus_statements(corpus_path):
        cap = caption_phrase(stmt)
        if cap:
            out.append((loose_key(stmt), cap.split(" ", 1)[1]))
    return out


def statement_lookup(corpus_path: Path) -> Dict[str, str]:
    """Map both the pre-caption and current wording of a statement to its current text."""
    current = corpus_statements(corpus_path)
    lookup = {stmt_key(s): s for s in current}
    backup = corpus_path.with_suffix(corpus_path.suffix + ".bak_before_apple_captions")
    if backup.exists():
        previous = corpus_statements(backup)
        if len(previous) == len(current):
            for old, new in zip(previous, current):
                lookup.setdefault(stmt_key(old), new)
    return lookup


def caption_lookup(corpus_path: Path) -> Dict[str, str]:
    """Loose key -> caption phrase, for evidence the QG pass reworded slightly."""
    current = corpus_statements(corpus_path)
    backup = corpus_path.with_suffix(corpus_path.suffix + ".bak_before_apple_captions")
    out: Dict[str, str] = {}
    for stmt in current:
        phrase = caption_phrase(stmt)
        if phrase:
            out.setdefault(loose_key(stmt), phrase)
    if backup.exists():
        previous = corpus_statements(backup)
        if len(previous) == len(current):
            for old, new in zip(previous, current):
                phrase = caption_phrase(new)
                if phrase and old != new:
                    out.setdefault(loose_key(old), phrase)
    return out


def split_evidence(evidence: str) -> List[str]:
    """Evidence is a "; "-joined list of statements; keep semicolons inside a statement."""
    parts: List[str] = []
    for chunk in (evidence or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if parts and not re.match(r"^In \d{4}[,)]", chunk):
            parts[-1] = parts[-1] + "; " + chunk
        else:
            parts.append(chunk)
    return parts


def retitle(part: str, entries: List[tuple]) -> str:
    """Swap a stale table heading ("regarding Our Findings") for the current caption."""
    m = STALE_TITLE_RE.match(part)
    if not m:
        return part
    year, stale, rest = m.groups()
    caption = caption_for_label(entries, row_label(rest), rest) or STALE_TITLES.get(
        stale, ""
    )
    if not caption or caption == stale:
        return part
    return f"In {year}, regarding {caption}, Apple reports: {rest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-diff", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    ds_path = args.dataset or (
        root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json"
    )
    corpus_path = args.corpus or (root / "Datasets" / "Corpus" / "esg_ocr.json")

    by_key = statement_lookup(corpus_path)
    by_loose = caption_lookup(corpus_path)
    entries = captioned_entries(corpus_path)

    data = json.loads(ds_path.read_text(encoding="utf-8"))
    questions = data["questions"]
    apple = [q for q in questions if q.get("company") == COMPANY]

    ev_changed = 0
    ev_unmatched: List[str] = []
    for q in apple:
        parts = split_evidence(q.get("evidence") or "")
        if not parts:
            continue
        rebuilt = []
        touched = False
        for p in parts:
            target = by_key.get(stmt_key(p))
            if target is None:
                phrase = by_loose.get(loose_key(p))
                patched = apply_caption(p, phrase) if phrase else p
                if patched == p:
                    patched = retitle(p, entries)
                rebuilt.append(patched)
                if patched != p:
                    touched = True
                elif re.match(r"^In \d{4}, ", p):
                    ev_unmatched.append(p)
                continue
            if target != p:
                touched = True
            rebuilt.append(target)
        if touched:
            if args.show_diff:
                for before, after in zip(parts, rebuilt):
                    if before != after:
                        print(f"  [{q['id']}] {after[:180]}")
            q["evidence"] = "; ".join(rebuilt)
            ev_changed += 1

    by_id = {q.get("id"): q for q in questions}
    edited = 0
    skipped: List[str] = []
    for edit in EDITS:
        q = by_id.get(edit["id"])
        if q is None:
            skipped.append(f"{edit['id']}: not found")
            continue
        expect = edit.get("expect")
        if expect and expect.lower() not in q["question"].lower():
            if q["question"] == edit.get("question"):
                continue  # already rewritten
            skipped.append(f"{edit['id']}: unexpected question {q['question'][:70]!r}")
            continue
        changed = False
        for field in ("question", "answer", "evidence", "evidence_triple"):
            if field in edit and q.get(field) != edit[field]:
                q[field] = edit[field]
                changed = True
        edited += int(changed)

    print(f"Apple questions: {len(apple)}")
    print(f"evidence re-pointed at the corpus: {ev_changed}")
    print(f"question/answer/triple edits applied: {edited}/{len(EDITS)}")
    if skipped:
        print("skipped:")
        for s in skipped:
            print("  ", s)
    if ev_unmatched:
        uniq = sorted(set(ev_unmatched))
        print(f"evidence lines with no corpus match: {len(uniq)}")
        for s in uniq[:10]:
            print("   ", s[:150])

    if args.dry_run:
        print("dry-run: no write")
        return
    if not ev_changed and not edited:
        print("nothing to write")
        return

    bak = ds_path.with_suffix(ds_path.suffix + ".bak_before_apple_qg_fix")
    if not bak.exists():
        shutil.copy2(ds_path, bak)
        print(f"backup: {bak}")
    ds_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote: {ds_path}")


if __name__ == "__main__":
    main()
