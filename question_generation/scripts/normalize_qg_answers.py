#!/usr/bin/env python3
"""Normalize gold answers in esg_qg_dataset.json to a canonical QA-alignable form.

Rules:
- Numeric: \"<value> <metric-or-unit>.\" (never bare number / bare symbol alone)
- Yes/No: \"Yes.\" / \"No.\" (optional short clarification)
- Years/ratings: keep report form, end with period
- Missing table cells (\"-\"): \"Not disclosed for <metric>.\"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


BARE_NUM = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$")
BARE_EU = re.compile(r"^\d{1,3}(?:\.\d{3})*,\d+$")  # e.g. 29,1
YEAR = re.compile(r"^(19|20)\d{2}$")
RATING = re.compile(r"^[A-D]{1,3}[+-]?$", re.I)


def is_bare_numeric(answer: str) -> bool:
    """True if answer is only a number with no metric/unit word.

    A trailing % counts as a unit, so \"31%\" is acceptable gold.
    """
    a = (answer or "").strip().rstrip(".")
    if not a:
        return False
    if a.endswith("%"):
        return False
    return bool(BARE_NUM.fullmatch(a) or BARE_EU.fullmatch(a))


def _clean_metric(s: str) -> str:
    s = (s or "").strip()
    s = s.strip("\"'“”‘’")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[?.!\s]+$", "", s)
    s = re.split(r"\s+as of\b|\s+at\b|\s+on\b|\s*\(", s, maxsplit=1, flags=re.I)[0]
    s = re.sub(
        r"\s+(reported|in|for|during|as of|at|by|under|according to|value)\s*$",
        "",
        s,
        flags=re.I,
    )
    # drop column markers often left in MGL questions
    s = re.sub(r"\s*\(Column\s*\d+\)\s*$", "", s, flags=re.I)
    return s.strip(" ,;:")


def infer_metric_from_question(question: str) -> Optional[str]:
    q = (question or "").strip()
    q_norm = q.replace("\u2019", "'").replace("\u2018", "'")

    # fixed-label shortcuts (no capture / awkward phrasing)
    if re.search(r"(?i)capital expenditures value did .+ report", q_norm):
        return "capital expenditures"
    if re.search(r"(?i)MSCI ESG rating", q_norm):
        return "MSCI ESG rating"
    if re.search(r"(?i)^In what year did", q_norm):
        return "year"

    patterns = [
        r"(?i)^(?:In [^,]+,\s*)?how many (.+?) (?:did|does|do|had|have)\b",
        r"(?i)^(?:In [^,]+,\s*)?how many (.+?) (?:were|was|are|is)\b",
        r"(?i)^(?:In [^,]+,\s*)?what percentage of .+? (?:was|were) (.+?)(?:\s+as of\b|\?|$)",
        r"(?i)^(?:In [^,]+,\s*)?what percentage of (.+?)(?:\s+identify|\s+had|\s+came|\s+earned|\s+from\b|\?|$)",
        r"(?i)^To what percentage did \S+ (?:reduce|increase|cut|raise|grow) (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^By what percentage did .+? (?:report )?(?:reducing|reduce|increase|increasing|decrease|decreasing) (.+?)(?:\s+relative|\s+in\b|\?|$)",
        r"(?i)^By what percentage did the (.+?)(?:\s+increase|\s+decrease|\s+rise|\s+fall|\?|$)",
        r"(?i)^What (.+? rate) did\b",
        r"(?i)^(?:In [^,]+,\s*)?what share of (.+?) (?:did|does)\b",
        r"(?i)^(?:In [^,]+,\s*)?what area of (.+?) (?:was|were|is|are)\b",
        r"(?i)^(?:In [^,]+,\s*)?what (data quality score(?:\s*\([^)]*\))?) (?:is|was)\b",
        r"(?i)^What unit(?: marker)? does .+ (?:use for|report for) (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What unit(?: marker)? did .+ (?:use for|report for) (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What value chain (?:indicator )?marker does .+ report for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What value chain (?:indicator )?marker did .+ report for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What value chain indicator \(symbol\) is reported for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What is another value chain indicator \(symbol\) reported for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What aggregation marker does .+ report for (.+?)(?:\s+in\b|\s+for\b|\?|$)",
        r"(?i)^What value is reported for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What was the value(?: reported)? for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What was the number of (.+?)(?:\s+reported|\s+in\b|\s+for\b|\?|$)",
        r"(?i)^What was the total monetary value of (.+?)(?:\s+reported|\s+in\b|\?|$)",
        r"(?i)^What was the total (?:number|amount|volume|value) of (.+?)(?:\s+reported|\s+in\b|\s+for\b|\?|$)",
        r"(?i)^What was the (?:total )?(?:amount|volume|value|rate|share|percentage) of (.+?)(?:\s+reported|\s+in\b|\s+for\b|\?|$)",
        r"(?i)^In [^,]+,\s*what amount of (.+?) (?:did|does)\b",
        r"(?i)^In [^,]+,\s*what was .+?'s value for ['\"](.+?)['\"]",
        r"(?i)^In [^,]+,\s*what was .+?'s (.+?)(?:\?|$)",
        r"(?i)^What (?:was|were) .+?(?:'s|') (.+?)(?:\s+in\b|\s+for\b|\s+during\b|\s+as of\b|\s+as reported\b|\s+according\b|\?|$)",
        r"(?i)^What is .+?(?:'s|') (.+?)(?:\s+in\b|\s+for\b|\s+as of\b|\?|$)",
        r"(?i)^What (.+?) did .+ report\b",
        r"(?i)^(?:In [^,]+,\s*)?what (?:was|were) the (.+?)(?:\s+reported|\s+in\b|\s+for\b|\?|$)",
        r"(?i)^What does .+ report for (.+?)(?:\s+in\b|\?|$)",
        r"(?i)^What does .+ Board Matrix report for (.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, q_norm)
        if not m or not m.groups():
            continue
        metric = _clean_metric(m.group(1))
        if metric and 1 <= len(metric.split()) <= 16:
            return metric
    return None


def _ensure_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith((".", "!", "?")):
        return s
    return s + "."


def _normalize_yes_no(answer: str) -> Optional[str]:
    a = answer.strip()
    # Already canonical: "Yes." / "Yes. Rest..."
    if re.match(r"^(Yes|No)\.\s*", a, flags=re.I):
        return None
    m = re.match(r"^(Yes|No)\b([\s,;:\-—].*)?$", a, flags=re.I)
    if not m:
        return None
    head = m.group(1).capitalize()
    rest = (m.group(2) or "").strip()
    if not rest:
        return f"{head}."
    # "Yes," / "Yes " → "Yes. ..."
    rest = re.sub(r"^[,;:\-—]+\s*", "", rest)
    if rest:
        rest = rest[0].upper() + rest[1:]
        if not rest.endswith((".", "!", "?")):
            rest = rest + "."
    return f"{head}. {rest}" if rest else f"{head}."


def normalize_answer(question: str, answer: str, question_type: str = "") -> Tuple[str, bool]:
    old = (answer or "").strip()
    a = old
    if not a:
        return a, False

    q = question or ""
    metric = infer_metric_from_question(q)

    # missing / blank table cell
    if a in {"-", "-.", "—", "—.", "–", "–."}:
        if metric:
            return f"Not disclosed for {metric}.", True
        return "Not disclosed.", True

    # bare unit / symbol answers
    if len(a) <= 3 and not YEAR.fullmatch(a.rstrip(".")) and not RATING.fullmatch(a.rstrip(".")):
        # e.g. "#", "%", "●", "●●"
        if not re.search(r"[A-Za-z0-9]{2,}", a):
            label = metric or "reported marker"
            return f"{a.rstrip('.')} ({label}).", True

    # years
    if YEAR.fullmatch(a.rstrip(".")):
        return _ensure_period(a.rstrip(".")), True

    # letter ratings like AA
    if RATING.fullmatch(a.rstrip(".")):
        return _ensure_period(a.rstrip(".")), a != _ensure_period(a.rstrip("."))

    # Yes/No head punctuation
    yn = _normalize_yes_no(a)
    if yn is not None and yn != a:
        a = yn

    # bare numeric (incl. european decimal)
    if is_bare_numeric(a):
        num = a.rstrip(".")
        # normalize european decimal comma for single-fraction cases: 29,1 -> 29.1
        if BARE_EU.fullmatch(num) and num.count(",") == 1 and "." not in num:
            num = num.replace(",", ".")
        if re.search(r"(?i)percentage value does .+ report as", q):
            return _ensure_period(num), True
        if metric:
            if metric.lower().startswith(num.lower() + " "):
                return _ensure_period(metric), True
            if num.endswith("%"):
                words = metric.split()
                if len(words) <= 4:
                    return f"{num} {metric}.", True
                return f"{num} of {metric}.", True
            return f"{num} {metric}.", True
        return _ensure_period(num), True

    # non-creative short answers: ensure terminal period
    if question_type != "Creative Generation":
        if a and not a.endswith((".", "!", "?")) and len(a.split()) <= 40:
            a = _ensure_period(a)

    return a, a != old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    path = args.dataset or (root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data.get("questions") or []

    changed = 0
    samples: list[Dict[str, Any]] = []
    remaining_issues = 0

    for q in qs:
        old = str(q.get("answer") or "").strip()
        new, _ = normalize_answer(
            str(q.get("question") or ""),
            old,
            str(q.get("question_type") or ""),
        )
        if new != old:
            changed += 1
            if len(samples) < 20:
                samples.append(
                    {
                        "question": q.get("question"),
                        "before": old,
                        "after": new,
                        "type": q.get("question_type"),
                    }
                )
            if not args.dry_run:
                q["answer"] = new

        # track residual rule breaks
        check = new
        if is_bare_numeric(check) or check in {"-", "#", "%"} or (
            check
            and str(q.get("question_type")) != "Creative Generation"
            and not check.endswith((".", "!", "?"))
            and len(check.split()) <= 40
        ):
            remaining_issues += 1

    print(f"dataset: {path}")
    print(f"questions: {len(qs)}")
    print(f"changed: {changed}")
    print(f"remaining_flagged: {remaining_issues}")
    for s in samples:
        print("---")
        print(s["type"])
        print(s["question"])
        print(f"  {s['before']!r} -> {s['after']!r}")

    if args.dry_run:
        print("dry-run: no write")
        return

    bak = path.with_suffix(path.suffix + ".bak_before_answer_norm_v2")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"backup: {bak}")
    data["answer_normalized"] = True
    data["answer_normalize_version"] = 2
    data["answer_normalize_changed"] = changed
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
