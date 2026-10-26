#!/usr/bin/env python3
"""Re-point QG evidence at the captioned statements in Datasets/Corpus/esg_ocr.json.

add_table_captions.py rewrites statements in place, one for one, so the backup it
leaves behind gives an exact old -> new mapping. This script walks every question in
ESG_KG/output/qg/esg_qg_dataset.json, splits its evidence into statements and swaps
each one for its current wording.

Evidence is not always a verbatim copy of the corpus line - the QG pass drops LaTeX,
footnote markers and page references - so an exact match is tried first and a loose
match that only injects the caption phrase second.

Usage:
    python sync_qg_to_corpus.py --dry-run
    python sync_qg_to_corpus.py --show-diff --company Ramsay
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

BACKUP_SUFFIX = ".bak_before_all_captions"


def corpus_statements(context: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(
        r"## Page \d+\n+### Statements\n(.*?)\n\n### Text\n", context, re.DOTALL
    ):
        if m.group(1).strip() == "(none)":
            continue
        out.extend(m.group(1).split("\n"))
    return out


def stmt_key(text: str) -> str:
    t = (text or "").replace("\u2082", "2").replace("\u2083", "3")
    t = re.sub(r"\s*\(page\s+\d+\)\s*\.?\s*$", "", t.strip())
    return re.sub(r"\s+", " ", t).strip().rstrip(".").lower()


FOOTNOTE = re.compile(r"\$\s*\^\s*\{+\s*[\d,\s*†‡]+\s*\}+\s*\$|\^\{+[\d,\s]+\}+")


def loose_key(text: str) -> str:
    """Key that survives what the QG pass dropped: LaTeX, footnotes, page references.

    Units matter for matching, so LaTeX is unwrapped rather than deleted: the corpus
    "tonnes $ CO_2 $-e" and the evidence "tonnes CO2-e" have to land on the same key.
    """
    t = (text or "").replace("\u2082", "2").replace("\u2083", "3")
    t = t.replace("\\n", " ").replace("\n", " ")
    t = re.sub(r"\s*\(page\s+\d+\)\s*\.?\s*$", "", t.strip())
    t = FOOTNOTE.sub(" ", t)
    t = re.sub(r"[${}_^\\]", "", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


CAPTION_PREFIX = re.compile(
    r"^(In \d{4}, )(?:for|regarding) .+?, (?=[A-Z][^,]{0,40}?(?:'s |\breports\b))"
)
PARENS = re.compile(r"\([^()]*\)")


def bare_key(text: str) -> str:
    """Loose key of a statement without its caption prefix and without units in parens.

    Some evidence predates both the caption and the unit qualifier the corpus carries
    ("Accenture's Air Travel" vs "Accenture's Air Travel (mTCO2e)").
    """
    t = CAPTION_PREFIX.sub(r"\1", text or "")
    prev = None
    while prev != t:
        prev, t = t, PARENS.sub(" ", t)
    key = loose_key(t)
    return key if len(key) >= 25 else ""


CAPTION_RE = re.compile(r"^In \d{4}, ((?:for|regarding) .+?), (?:.+?)(?:'s|\breports\b)")


def caption_phrase(statement: str) -> str:
    m = CAPTION_RE.match(statement or "")
    return m.group(1) if m else ""


def apply_caption(part: str, phrase: str) -> str:
    m = re.match(r"^In (\d{4}), ", part)
    if not m or not phrase:
        return part
    if re.match(r"^In \d{4}, (?:for|regarding) ", part):
        return part
    return f"In {m.group(1)}, {phrase}, " + part[m.end() :]


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


class CorpusIndex:
    """Ways to recognise a QG evidence line as one of a corpus' statements."""

    def __init__(self, statements: List[str]) -> None:
        self.exact: Dict[str, str] = {stmt_key(s): s for s in statements}
        loose: Dict[str, List[str]] = {}
        bare: Dict[str, List[str]] = {}
        for s in statements:
            loose.setdefault(loose_key(s), []).append(s)
            key = bare_key(s)
            if key:
                bare.setdefault(key, []).append(s)
        self.loose = {k: v[0] for k, v in loose.items() if len(set(v)) == 1}
        self.bare = {k: v[0] for k, v in bare.items() if len(set(v)) == 1}
        self.sorted_loose = sorted(loose.items())

    def add_old_wording(self, old: str, new: str) -> None:
        self.exact.setdefault(stmt_key(old), new)
        self.loose.setdefault(loose_key(old), new)
        key = bare_key(old)
        if key:
            self.bare.setdefault(key, new)

    def by_prefix(self, key: str) -> str:
        """Statement whose key starts with a truncated evidence key, if there is one."""
        if len(key) < 40:
            return ""
        i = bisect_left(self.sorted_loose, (key, []))
        hits = []
        for k, stmts in self.sorted_loose[i : i + 4]:
            if not k.startswith(key):
                break
            hits.extend(stmts)
        return hits[0] if len(set(hits)) == 1 else ""

    def resolve(self, part: str) -> Tuple[str, bool]:
        """(replacement, matched) - the replacement is the part itself when unchanged."""
        target = self.exact.get(stmt_key(part))
        if target is None:
            target = self.loose.get(loose_key(part)) or self.by_prefix(loose_key(part))
        if target is not None and target != "":
            # Statements the caption pass left alone carry no new information, and the
            # corpus wording of those often holds OCR noise the QG pass cleaned up.
            return (target if caption_phrase(target) else part), True
        phrase = self.bare.get(bare_key(part))
        if phrase:
            patched = apply_caption(part, caption_phrase(phrase))
            return patched, True
        return part, False


def build_index(
    current: Dict[str, str], previous: Dict[str, str]
) -> Tuple[Dict[str, CorpusIndex], List[str]]:
    index: Dict[str, CorpusIndex] = {}
    warnings: List[str] = []
    for name, ctx in current.items():
        now = corpus_statements(ctx)
        idx = CorpusIndex(now)
        before_ctx = previous.get(name)
        if before_ctx is not None:
            before = corpus_statements(before_ctx)
            if len(before) != len(now):
                warnings.append(
                    f"{name}: {len(before)} statements before vs {len(now)} now, "
                    "old wording not mapped"
                )
            else:
                for old, new in zip(before, now):
                    if old != new:
                        idx.add_old_wording(old, new)
        index[name] = idx
    return index, warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--company", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-diff", action="store_true")
    ap.add_argument("--show-unmatched", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    ds_path = args.dataset or (root / "ESG_KG" / "output" / "qg" / "esg_qg_dataset.json")
    corpus_path = args.corpus or (root / "Datasets" / "Corpus" / "esg_ocr.json")
    bak_path = corpus_path.with_suffix(corpus_path.suffix + BACKUP_SUFFIX)

    current = {
        e["corpus_name"]: e["context"]
        for e in json.loads(corpus_path.read_text(encoding="utf-8"))
    }
    previous: Dict[str, str] = {}
    if bak_path.exists():
        previous = {
            e["corpus_name"]: e["context"]
            for e in json.loads(bak_path.read_text(encoding="utf-8"))
        }
    else:
        print(f"note: no backup at {bak_path.name}, only current wording is matched")

    index, warnings = build_index(current, previous)
    for w in warnings:
        print("warning:", w)

    data = json.loads(ds_path.read_text(encoding="utf-8"))
    questions = data["questions"]

    changed_by_company: Counter = Counter()
    unmatched_by_company: Counter = Counter()
    unmatched_samples: Dict[str, List[str]] = {}
    total_changed = 0

    for q in questions:
        doc = q.get("document_id")
        if doc not in index:
            continue
        if args.company and q.get("company") not in args.company:
            continue
        parts = split_evidence(q.get("evidence") or "")
        if not parts:
            continue

        rebuilt: List[str] = []
        touched = False
        for p in parts:
            patched, matched = index[doc].resolve(p)
            rebuilt.append(patched)
            if patched != p:
                touched = True
            elif not matched and re.match(r"^In \d{4}, ", p):
                unmatched_by_company[q.get("company")] += 1
                unmatched_samples.setdefault(q.get("company"), []).append(p)

        if touched:
            if args.show_diff:
                for before, after in zip(parts, rebuilt):
                    if before != after:
                        print(f"  [{q['id']}] {after[:200]}")
            q["evidence"] = "; ".join(rebuilt)
            changed_by_company[q.get("company")] += 1
            total_changed += 1

    print(f"\n{'company':22} {'evidence updated':>16} {'unmatched lines':>16}")
    for company in sorted(set(changed_by_company) | set(unmatched_by_company)):
        print(
            f"{company:22} {changed_by_company[company]:16d} "
            f"{unmatched_by_company[company]:16d}"
        )
    print(f"\nquestions with updated evidence: {total_changed}")

    if args.show_unmatched:
        for company, lines in sorted(unmatched_samples.items()):
            print(f"\n-- {company} unmatched --")
            for line in sorted(set(lines))[: args.show_unmatched]:
                print("   ", line[:190])

    if args.dry_run:
        print("dry-run: no write")
        return
    if not total_changed:
        print("nothing to write")
        return

    bak = ds_path.with_suffix(ds_path.suffix + ".bak_before_caption_sync")
    if not bak.exists():
        shutil.copy2(ds_path, bak)
        print(f"backup: {bak}")
    data["evidence_caption_sync"] = {
        "questions_updated": total_changed,
        "corpus_backup": bak_path.name,
    }
    ds_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote: {ds_path}")


if __name__ == "__main__":
    main()
