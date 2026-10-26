#!/usr/bin/env python3
"""Align Accenture-2024 questions in esg_qg_dataset.json with the corrected corpus.

The original Statements lost table row groups (Employee Workforce vs Executives vs
Managing Directors, U.S. Workforce vs U.S. Managing Directors) and mislabelled the
year columns, so some gold answers picked the wrong cell and many questions were
ambiguous. Each edit below is keyed by a unique substring of the current question.

Idempotent: entries already matching the target text are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

# key: unique substring of the existing question
# value: fields to overwrite
EDITS: List[Dict[str, Any]] = [
    {
        "match": "What was Accenture's global headcount (rounded) as of December 1, 2024?",
        "evidence": "In 2024, for Workforce, Accenture's Global Headcount (rounded) (As of December 1, 2024) was 798,000 (page 19).",
        "evidence_triple": "(Accenture,reportsValueFor,Workforce - Global Headcount (rounded) (As of December 1, 2024)); (Workforce - Global Headcount (rounded) (As of December 1, 2024),was,798,000)",
    },
    {
        "match": "What percentage of Accenture's Executives was Men as of December 1, 2024?",
        "evidence": "In 2024, for Gender Equality - Executives, Accenture's Men (As of December 1, 2024) was 67% (page 19).",
        "evidence_triple": "(Accenture,reportsValueFor,Gender Equality - Executives - Men (As of December 1, 2024)); (Gender Equality - Executives - Men (As of December 1, 2024),was,67%)",
    },
    {
        "match": "What was the percentage of Hispanic American and Latinx Managing Directors at Accenture",
        "question": "What was the percentage of Hispanic American and Latinx U.S. Managing Directors at Accenture as of December 1, 2024?",
        "answer": "5.0% Hispanic American and Latinx U.S. Managing Directors.",
        "evidence": "In 2024, for Race and Ethnicity - U.S. Managing Directors, Accenture's Hispanic American and Latinx (As of December 1, 2024) was 5.0% (page 19).",
        "evidence_triple": "(Accenture,reportsValueFor,U.S. Managing Directors - Hispanic American and Latinx (As of December 1, 2024)); (U.S. Managing Directors - Hispanic American and Latinx (As of December 1, 2024),was,5.0%)",
    },
    {
        "match": "What was Accenture's Coloured representation as of December 1, 2024?",
        "question": "What was Accenture's Coloured representation in its South Africa Workforce as of December 1, 2024?",
        "answer": "10.5% Coloured representation in the South Africa Workforce.",
        "evidence": "In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's Coloured (As of December 1, 2024) was 10.5% (page 19).",
        "evidence_triple": "(Accenture,reportsValueFor,South Africa Workforce - Coloured (As of December 1, 2024)); (South Africa Workforce - Coloured (As of December 1, 2024),was,10.5%)",
    },
    {
        # gold was 48%, which is the Women row; Men Employee Workforce is 52%
        "match": "What was Accenture's Men Employee Workforce percentage in 2023",
        "question": "What was Accenture's Men Employee Workforce percentage as of December 1, 2023 (as reported in the inclusion and diversity metrics table)?",
        "answer": "52% Men Employee Workforce percentage.",
        "evidence": "In 2023, for Gender Equality - Employee Workforce, Accenture's Men (As of December 1, 2023) was 52% (page 19).",
        "evidence_triple": "(Accenture,reportsValueFor,Gender Equality - Employee Workforce - Men (As of December 1, 2023)); (Gender Equality - Employee Workforce - Men (As of December 1, 2023),was,52%)",
    },
    {
        "match": "What overall inclusion and diversity narrative emerges from Accenture",
        "question": "What overall inclusion and diversity narrative emerges from Accenture’s reporting of women’s representation and multiple ethnicity metrics across its reporting cuts (as of December 1, 2024 and 2023, and the 2020 goal-announcement baseline)?",
        "answer": "Accenture presents a multi-cut diversity snapshot, reporting women’s representation and several ethnicity measures for 2024, 2023 and the 2020 baseline.",
        "evidence": "In 2024, for Gender Equality - Managing Directors, Accenture's Women (As of December 1, 2024) was 30% (page 19).; In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's African American and Black (As of December 1, 2024) was 11.7% (page 19).; In 2024, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (As of December 1, 2024) was 4.1% (page 19).; In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's Hispanic American and Latinx (As of December 1, 2024) was 10.4% (page 19).",
        "evidence_triple": "(Gender Equality - Managing Directors - Women (As of December 1, 2024),was,30%); (U.S. Workforce - African American and Black (As of December 1, 2024),was,11.7%); (U.S. Managing Directors - African American and Black (As of December 1, 2024),was,4.1%); (U.S. Workforce - Hispanic American and Latinx (As of December 1, 2024),was,10.4%)",
    },
    {
        "match": "How does Accenture frame progress and accountability for Black representation",
        "evidence": "In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's African American and Black (As of December 1, 2024) was 11.7% (page 19).; In 2025, for U.S. Workforce, Accenture's African American and Black (2025 Goal) was 12.0% (page 8).; In 2020, for Race and Ethnicity - U.K. Managing Directors, Accenture's Number of Black Managing Directors (At Goal Announcement 2020) was 8 (page 19).",
        "evidence_triple": "(U.S. Workforce - African American and Black (As of December 1, 2024),was,11.7%); (U.S. Workforce - African American and Black (2025 Goal),was,12.0%); (U.K. Managing Directors - Number of Black Managing Directors (At Goal Announcement 2020),was,8); (Accenture,reports,Managing Directors)",
    },
    {
        "match": "How do Accenture’s reported ethnicity metrics for South Africa",
        "evidence": "In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's African Black (As of December 1, 2024) was 46.9% (page 19).; In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's Coloured (As of December 1, 2024) was 10.5% (page 19).; In 2024, for Race and Ethnicity - South Africa Managing Directors, Accenture's African Black, Coloured and Indian (As of December 1, 2024) was 45.7% (page 19).",
        "evidence_triple": "(South Africa Workforce - African Black (As of December 1, 2024),was,46.9%); (South Africa Workforce - Coloured (As of December 1, 2024),was,10.5%); (South Africa Managing Directors - African Black, Coloured and Indian (As of December 1, 2024),was,45.7%)",
    },
    {
        "match": "How does Accenture’s 2024 inclusion and diversity reporting combine workforce composition",
        "evidence": "In 2024, for Gender Equality - Employee Workforce, Accenture's Women (As of December 1, 2024) was 48% (page 19).; In 2024, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (As of December 1, 2024) was 4.1% (page 19).; In 2020, for Race and Ethnicity - U.K. Managing Directors, Accenture's Number of Black Managing Directors (At Goal Announcement 2020) was 8 (page 19).",
        "evidence_triple": "(Gender Equality - Employee Workforce - Women (As of December 1, 2024),was,48%); (U.S. Managing Directors - African American and Black (As of December 1, 2024),was,4.1%); (U.K. Managing Directors - Number of Black Managing Directors (At Goal Announcement 2020),was,8); (Accenture,reports,U.S. Managing Directors)",
    },
    {
        "match": "What overall gender representation picture does Accenture present",
        "question": "What overall gender representation picture does Accenture present across its Employee Workforce, Executives and Managing Directors cuts as of December 1, 2024?",
        "answer": "Women are 48% of Accenture’s employee workforce but only 33% of executives and 30% of managing directors.",
        "evidence": "In 2024, for Gender Equality - Employee Workforce, Accenture's Women (As of December 1, 2024) was 48% (page 19).; In 2024, for Gender Equality - Executives, Accenture's Women (As of December 1, 2024) was 33% (page 19).; In 2024, for Gender Equality - Managing Directors, Accenture's Women (As of December 1, 2024) was 30% (page 19).",
        "evidence_triple": "(Gender Equality - Employee Workforce - Women (As of December 1, 2024),was,48%); (Gender Equality - Executives - Women (As of December 1, 2024),was,33%); (Gender Equality - Managing Directors - Women (As of December 1, 2024),was,30%)",
    },
    {
        "match": "How do Accenture’s 2024 ethnicity-related disclosures synthesize baseline goal-announcement measures",
        "evidence": "In 2023, for Race and Ethnicity - U.S. Workforce, Accenture's African American and Black (As of December 1, 2023) was 11.9% (page 19).; In 2020, for Race and Ethnicity - U.S. Workforce, Accenture's African American and Black (At Goal Announcement 2020) was 9.0% (page 19).; In 2024, for Race and Ethnicity - South Africa Managing Directors, Accenture's African Black, Coloured and Indian (As of December 1, 2024) was 45.7% (page 19).",
        "evidence_triple": "(U.S. Workforce - African American and Black (As of December 1, 2023),was,11.9%); (U.S. Workforce - African American and Black (At Goal Announcement 2020),was,9.0%); (South Africa Managing Directors - African Black, Coloured and Indian (As of December 1, 2024),was,45.7%)",
    },
    {
        "match": "How do Accenture’s 2024 regional and role-based disclosures",
        "evidence": "In 2024, regarding the Consolidated Statements of Inclusion & Diversity Metrics as of and for the years ended December 1, 2024 and 2023, Accenture reports Race and Ethnicity - U.K. Workforce and Race and Ethnicity - U.K. Managing Directors (page 19).; In 2024, for Race and Ethnicity - U.K. Workforce, Accenture's Black (As of December 1, 2024) was 5.5% (page 19).; In 2024, for Race and Ethnicity - U.K. Managing Directors, Accenture's Number of Black Managing Directors (As of December 1, 2024) was 12 (page 19).",
        "evidence_triple": "(Accenture,reports,Race and Ethnicity - U.K. Workforce); (Accenture,reports,Race and Ethnicity - U.K. Managing Directors); (U.K. Workforce - Black (As of December 1, 2024),was,5.5%); (U.K. Managing Directors - Number of Black Managing Directors (As of December 1, 2024),was,12)",
    },
    {
        "match": "How does Accenture’s reporting on U.S. Managing Directors and the U.S. workforce summarize",
        "evidence": "In 2020, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (At Goal Announcement 2020) was 2.8% (page 19).; In 2023, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (As of December 1, 2023) was 4.2% (page 19).; In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's Hispanic American and Latinx (As of December 1, 2024) was 10.4% (page 19).",
        "evidence_triple": "(U.S. Managing Directors - African American and Black (At Goal Announcement 2020),was,2.8%); (U.S. Managing Directors - African American and Black (As of December 1, 2023),was,4.2%); (U.S. Workforce - Hispanic American and Latinx (As of December 1, 2024),was,10.4%); (U.S. Managing Directors - African American and Black (2025 Goal),was,4.4%)",
    },
    {
        "match": "What overall inclusion and diversity storyline emerges from Accenture reporting both baseline and current values",
        "evidence": "In 2023, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (As of December 1, 2023) was 4.2% (page 19).; In 2020, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (At Goal Announcement 2020) was 2.8% (page 19).; In 2025, for U.S. Managing Directors, Accenture's African American and Black (2025 Goal) was 4.4% (page 8).",
        "evidence_triple": "(U.S. Managing Directors - African American and Black (As of December 1, 2023),was,4.2%); (U.S. Managing Directors - African American and Black (At Goal Announcement 2020),was,2.8%); (U.S. Managing Directors - African American and Black (2025 Goal),was,4.4%)",
    },
    {
        "match": "How does Accenture connect its South Africa workforce reporting to demographic targets",
        "evidence": "In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's African Black (As of December 1, 2024) was 46.9% (page 19).; In 2025, for South Africa Workforce, Accenture's Coloured (2025 Goal) was 10.0% (page 8).; In 2025, for South Africa Managing Directors, Accenture's African Black, Coloured and Indian (2025 Goal) was 70.0% (page 8).",
        "evidence_triple": "(Accenture,reports,South Africa Workforce); (South Africa Workforce - African Black (As of December 1, 2024),was,46.9%); (South Africa Workforce - Coloured (2025 Goal),was,10.0%); (South Africa Managing Directors - African Black, Coloured and Indian (2025 Goal),was,70.0%)",
    },
    {
        "match": "How do Accenture’s workforce composition metrics (such as men) and managing director disclosures",
        "evidence": "In 2024, for Gender Equality - Managing Directors, Accenture's Men (As of December 1, 2024) was 70% (page 19).; In 2024, for Gender Equality - Employee Workforce, Accenture's Men (As of December 1, 2024) was 52% (page 19).; In 2024, for Race and Ethnicity - U.S. Managing Directors, Accenture's African American and Black (As of December 1, 2024) was 4.1% (page 19).",
        "evidence_triple": "(Gender Equality - Managing Directors - Men (As of December 1, 2024),was,70%); (Gender Equality - Employee Workforce - Men (As of December 1, 2024),was,52%); (U.S. Managing Directors - African American and Black (As of December 1, 2024),was,4.1%); (Accenture,reports,South Africa Workforce)",
    },
    {
        # gold used 6.0%, which is the 2020 goal-announcement value, not the current one
        "match": "Role-play as an Accenture inclusion & diversity leader presenting to a South Africa leadership forum",
        "answer": "When I speak with our South Africa leaders, I ground the discussion in what we report for both the South Africa Workforce and South Africa Managing Directors. One data point I keep returning to is Coloured representation in the South Africa Workforce: 10.5% as of December 1, 2024, against our stated 2025 goal of 10.0%, up from 6.0% at the 2020 goal announcement date. Alongside it we disclose African Black representation at 46.9% and, for South Africa Managing Directors, African Black, Coloured and Indian representation at 45.7%. I use these figures to keep the conversation specific and measurable.",
        "evidence": "In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's Coloured (As of December 1, 2024) was 10.5% (page 19).; In 2025, for South Africa Workforce, Accenture's Coloured (2025 Goal) was 10.0% (page 8).; In 2020, for Race and Ethnicity - South Africa Workforce, Accenture's Coloured (At Goal Announcement 2020) was 6.0% (page 19).; In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's African Black (As of December 1, 2024) was 46.9% (page 19).; In 2024, for Race and Ethnicity - South Africa Managing Directors, Accenture's African Black, Coloured and Indian (As of December 1, 2024) was 45.7% (page 19).",
        "evidence_triple": "(Accenture,reports,South Africa Workforce); (Accenture,reports,South Africa Managing Directors); (South Africa Workforce - Coloured (As of December 1, 2024),was,10.5%); (South Africa Workforce - Coloured (2025 Goal),was,10.0%); (South Africa Workforce - Coloured (At Goal Announcement 2020),was,6.0%); (South Africa Workforce - African Black (As of December 1, 2024),was,46.9%); (South Africa Managing Directors - African Black, Coloured and Indian (As of December 1, 2024),was,45.7%)",
    },
    {
        # gold used 4.5%, the U.S. Managing Directors 2023 value, not the U.S. Workforce baseline
        "match": "Role-play as an Accenture HR analytics manager in the U.S.",
        "answer": "In my U.S. workforce reporting, I stick to the exact representation figures we disclose. For Hispanic American and Latinx representation in the U.S. Workforce, we report 9.5% at the 2020 goal announcement date and a 2025 goal of 13.0%, with 10.4% as of December 1, 2024. Alongside it, African American and Black representation is 11.7% as of December 1, 2024 against a 12.0% goal. When I share these internally, I frame them as tracked metrics—clear reference points for where we started and what target has been set, without implying progress beyond what is reported.",
        "evidence": "In 2020, for Race and Ethnicity - U.S. Workforce, Accenture's Hispanic American and Latinx (At Goal Announcement 2020) was 9.5% (page 19).; In 2025, for U.S. Workforce, Accenture's Hispanic American and Latinx (2025 Goal) was 13.0% (page 8).; In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's Hispanic American and Latinx (As of December 1, 2024) was 10.4% (page 19).; In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's African American and Black (As of December 1, 2024) was 11.7% (page 19).; In 2025, for U.S. Workforce, Accenture's African American and Black (2025 Goal) was 12.0% (page 8).",
        "evidence_triple": "(Accenture,reports,U.S. Workforce); (U.S. Workforce - Hispanic American and Latinx (At Goal Announcement 2020),was,9.5%); (U.S. Workforce - Hispanic American and Latinx (2025 Goal),was,13.0%); (U.S. Workforce - Hispanic American and Latinx (As of December 1, 2024),was,10.4%); (U.S. Workforce - African American and Black (As of December 1, 2024),was,11.7%); (U.S. Workforce - African American and Black (2025 Goal),was,12.0%)",
    },
    {
        # same two mislabelled cells as above
        "match": "Rewrite the evidence as a brief press release focused on workforce reporting and goals",
        "answer": "Accenture published workforce reporting covering its U.S. Workforce as well as its South Africa Workforce and South Africa Managing Directors. For the U.S. Workforce, the company reported Hispanic American and Latinx representation of 9.5% at the 2020 goal announcement date and a 2025 goal of 13.0%, with 10.4% as of December 1, 2024. In South Africa, Accenture reported Coloured representation of 10.5% as of December 1, 2024 against a 2025 goal of 10.0%, and African Black, Coloured and Indian representation of 45.7% among South Africa Managing Directors. These figures were presented within the company's Inclusion & Diversity reporting.",
        "evidence": "In 2020, for Race and Ethnicity - U.S. Workforce, Accenture's Hispanic American and Latinx (At Goal Announcement 2020) was 9.5% (page 19).; In 2025, for U.S. Workforce, Accenture's Hispanic American and Latinx (2025 Goal) was 13.0% (page 8).; In 2024, for Race and Ethnicity - U.S. Workforce, Accenture's Hispanic American and Latinx (As of December 1, 2024) was 10.4% (page 19).; In 2024, for Race and Ethnicity - South Africa Workforce, Accenture's Coloured (As of December 1, 2024) was 10.5% (page 19).; In 2025, for South Africa Workforce, Accenture's Coloured (2025 Goal) was 10.0% (page 8).; In 2024, for Race and Ethnicity - South Africa Managing Directors, Accenture's African Black, Coloured and Indian (As of December 1, 2024) was 45.7% (page 19).",
        "evidence_triple": "(Accenture,reports,U.S. Workforce); (Accenture,reports,South Africa Workforce); (Accenture,reports,South Africa Managing Directors); (U.S. Workforce - Hispanic American and Latinx (At Goal Announcement 2020),was,9.5%); (U.S. Workforce - Hispanic American and Latinx (2025 Goal),was,13.0%); (South Africa Workforce - Coloured (As of December 1, 2024),was,10.5%); (South Africa Workforce - Coloured (2025 Goal),was,10.0%); (South Africa Managing Directors - African Black, Coloured and Indian (As of December 1, 2024),was,45.7%)",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    path = args.dataset or (root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data.get("questions") or []

    accenture = [
        q for q in qs if str(q.get("document_id") or "").startswith("Accenture")
    ]

    changed = 0
    for edit in EDITS:
        needle = edit["match"]
        hits = [q for q in accenture if needle in str(q.get("question") or "")]
        if len(hits) != 1:
            print(f"WARN {len(hits)} matches for: {needle[:70]}")
            continue
        q = hits[0]
        fields = {k: v for k, v in edit.items() if k != "match"}
        diffs = {k: v for k, v in fields.items() if str(q.get(k) or "") != v}
        if not diffs:
            continue
        changed += 1
        print(f"--- {q.get('id')} [{q.get('question_type')}]")
        for k, v in diffs.items():
            print(f"  {k}:")
            print(f"    - {str(q.get(k))[:180]}")
            print(f"    + {v[:180]}")
        if not args.dry_run:
            q.update(fields)

    print(f"\nedits applied: {changed}/{len(EDITS)}")

    if args.dry_run:
        print("dry-run: no write")
        return
    if not changed:
        print("nothing to write")
        return

    bak = path.with_suffix(path.suffix + ".bak_before_accenture_qg_fix")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"backup: {bak}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
