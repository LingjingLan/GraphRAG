#!/usr/bin/env python3
"""Name the source table in QG questions that a reader cannot otherwise pin down.

Now that the corpus statements carry table captions, a question such as "What was the
reported value for Ramsay Nordics in FY22?" can say which table it is asking about.
Two kinds of questions are treated:

* vague - the question refers to "the reported value" or a bare "Column 4";
* ambiguous - the same metric and year is reported under several captions with
  different values, so the question as phrased has more than one right answer.

Only Fact Retrieval questions whose evidence is a single captioned statement are
touched, and the caption is appended rather than the question reworded, except for a
few cases listed in EXPLICIT where the wording itself was wrong.

Usage:
    python disambiguate_qg_questions.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

VAGUE = [
    re.compile(r"\breported value\b", re.I),
    re.compile(r"\bwhat value\b", re.I),
    re.compile(r"in the provided evidence", re.I),
    re.compile(r"\bColumn \d+\b"),
    re.compile(r"\bwhat (?:was|is) the value\b", re.I),
    re.compile(r"\bvalue for\b", re.I),
]

# The caption itself often contains commas, so anchor on the company that follows it.
STATEMENT = re.compile(
    r"^In (\d{4}), (?:for|regarding) (.+), ([A-Z][^,]{0,40}?)'s (.+?) was (.+?)"
    r"(?:\s*\(page \d+\))?\.?$"
)

# Questions whose wording, not just their scope, was wrong.
EXPLICIT: Dict[str, Dict[str, str]] = {
    "SHL-2024_L1_0621": {
        "question": (
            "What Scope 1 emissions did SHL report for New Zealand in its restated "
            "FY2023 (previous-year) country breakdown?"
        ),
        "answer": "116 tonnes CO2-e.",
    },
    "SHL-2024_L1_0624": {
        "question": (
            "How much Scope 2 energy use (in GJ) did SHL report for Germany in its "
            "restated FY2023 (previous-year) country breakdown?"
        ),
        "answer": "184,855 GJ.",
    },
    "Snam-2024_L1_0687": {
        "question": (
            "What was the 2024 budget figure Snam reported for the target tracked in "
            "its Sustainability Scorecard?"
        ),
        "answer": "20 Mscm budgeted for 2024.",
    },
    "Snam-2024_L1_0688": {
        "question": (
            "What percentage change did Snam report alongside the 6% figure under "
            "Biodiversity and Regeneration?"
        ),
        "answer": "-29%.",
    },
    "Starbucks-2024_L1_0721": {
        "question": (
            "How large was the Global Community Impact Grants portfolio that The "
            "Starbucks Foundation launched in 2022?"
        ),
        "answer": "$30 million.",
    },
}

MAX_CAPTION_IN_QUESTION = 95


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def split_evidence(evidence: str) -> List[str]:
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


def already_named(question: str, caption: str) -> bool:
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", caption)]
    if not words:
        return True
    present = sum(1 for w in words if w in question.lower())
    return present >= max(1, len(words) // 2)


def add_scope(question: str, caption: str) -> str:
    caption = caption.strip().rstrip(".")
    caption = caption[:1].upper() + caption[1:]
    while len(caption) > MAX_CAPTION_IN_QUESTION and " - " in caption:
        caption = caption.rsplit(" - ", 1)[0]
    if len(caption) > MAX_CAPTION_IN_QUESTION:
        caption = caption[:MAX_CAPTION_IN_QUESTION].rsplit(" ", 1)[0] + "..."
    stem = question.strip()
    trailing = "?" if stem.endswith("?") else ""
    stem = stem[: -1 if trailing else None].rstrip()
    return f'{stem} (as reported in the "{caption}" table){trailing}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    ds_path = args.dataset or (root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json")
    corpus_path = args.corpus or (root / "Datasets" / "Corpus" / "esg_ocr.json")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    metric_captions = defaultdict(set)
    metric_values = defaultdict(set)
    for entry in corpus:
        for m in re.finditer(
            r"## Page \d+\n+### Statements\n(.*?)\n\n### Text\n",
            entry["context"],
            re.DOTALL,
        ):
            if m.group(1).strip() == "(none)":
                continue
            for stmt in m.group(1).split("\n"):
                s = STATEMENT.match(stmt)
                if not s:
                    continue
                year, caption, _company, metric, value = s.groups()
                key = (entry["corpus_name"], year, norm(metric))
                metric_captions[key].add(caption)
                metric_values[key].add(value.strip())

    data = json.loads(ds_path.read_text(encoding="utf-8"))
    scoped = 0
    explicit_done = 0
    skipped: List[str] = []

    for q in data["questions"]:
        if q["id"] in EXPLICIT:
            edit = EXPLICIT[q["id"]]
            if q["question"] != edit["question"]:
                print(f"[reword] {q['id']}\n   - {q['question']}\n   + {edit['question']}")
                q.update(edit)
                explicit_done += 1
            continue
        if q.get("question_type") != "Fact Retrieval":
            continue

        parts = split_evidence(q.get("evidence") or "")
        if len(parts) != 1:
            continue
        s = STATEMENT.match(parts[0])
        if not s:
            continue
        year, caption, _company, metric, _value = s.groups()

        vague = any(p.search(q["question"]) for p in VAGUE)
        key = (q["document_id"], year, norm(metric))
        ambiguous = (
            len(metric_captions.get(key, set())) > 1
            and len(metric_values.get(key, set())) > 1
        )
        if not (vague or ambiguous):
            continue
        if already_named(q["question"], caption):
            continue
        if '" table)' in q["question"]:
            continue

        new_q = add_scope(q["question"], caption)
        print(f"[scope]  {q['id']}\n   - {q['question']}\n   + {new_q}")
        q["question"] = new_q
        scoped += 1

    print(f"\nquestions given a table scope: {scoped}")
    print(f"questions reworded: {explicit_done}")
    for s in skipped:
        print("  skipped:", s)

    if args.dry_run:
        print("dry-run: no write")
        return
    if not scoped and not explicit_done:
        print("nothing to write")
        return

    bak = ds_path.with_suffix(ds_path.suffix + ".bak_before_disambiguation")
    if not bak.exists():
        shutil.copy2(ds_path, bak)
        print(f"backup: {bak}")
    ds_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote: {ds_path}")


if __name__ == "__main__":
    main()
