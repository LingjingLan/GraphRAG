#!/usr/bin/env python3
"""Fill Complex Reasoning shortfall: extra L2 packs + LLM + rebuild dataset.

- Appends new L2 packs (paths / 2-hop / connected samples) into existing pack files
- Adds *_fill packs that reuse prior L2 evidence with existing_questions anti-dup
- Calls LLM only for packs without raw/
- Rebuilds esg_qg_dataset.json from all raw with dedupe
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from build_qg_packs import (
    build_graph,
    dedupe_packs,
    k_hop_pack,
    make_pack,
    sample_paths,
    take_unique,
)
from dedupe_qg_dataset import main as rebuild_dataset
from qg_common import (
    DEFAULT_TARGETS,
    call_llm,
    detect_newesg_root,
    load_dotenv,
    load_prompt,
    parse_questions_json,
    render_qg_prompt,
)
from run_qg_generate import validate_question


def load_report_triples(by_report_dir: Path, document_id: str) -> Optional[Dict[str, Any]]:
    fp = by_report_dir / f"{document_id}_triples.json"
    if not fp.is_file():
        # fuzzy
        hits = list(by_report_dir.glob(f"*{document_id}*_triples.json"))
        if not hits:
            return None
        fp = hits[0]
    return json.loads(fp.read_text(encoding="utf-8"))


def existing_pack_ids(packs: List[Dict[str, Any]]) -> Set[str]:
    return {str(p.get("pack_id")) for p in packs if p.get("pack_id")}


def build_extra_l2_for_report(
    report: Dict[str, Any],
    *,
    used_ids: Set[str],
    rng: random.Random,
    n_extra: int = 16,
    max_edges: int = 28,
    max_tokens: int = 5500,
) -> List[Dict[str, Any]]:
    document_id = str(report.get("document_id") or report.get("company") or "unknown")
    company = str(report.get("company") or document_id.split("-")[0])
    out_adj, in_adj, triples = build_graph(report.get("triples") or [])
    if len(triples) < 3:
        return []

    degree: Dict[str, int] = defaultdict(int)
    for t in triples:
        degree[t["head"]["name"].lower()] += 1
        degree[t["tail"]["name"].lower()] += 1
    seeds = [n for n, _ in sorted(degree.items(), key=lambda x: -x[1])]

    candidates: List[Dict[str, Any]] = []

    # Path-based
    paths = sample_paths(
        triples, out_adj, min_len=3, max_len=7, n_paths=n_extra * 6, rng=rng
    )
    for path in paths:
        idxs = list(path)
        nodes = set()
        for ei in path:
            t = triples[ei]
            nodes.add(t["head"]["name"].lower())
            nodes.add(t["tail"]["name"].lower())
        ctx: List[int] = []
        for n in nodes:
            ctx.extend(out_adj.get(n, [])[:3])
            ctx.extend(in_adj.get(n, [])[:3])
        rng.shuffle(ctx)
        idxs.extend(ctx)
        edges = take_unique(triples, idxs, max_edges)
        if len(edges) < 3:
            continue
        candidates.append(
            make_pack(
                pack_id="tmp",
                document_id=document_id,
                company=company,
                question_type="Complex Reasoning",
                triples=edges,
                method="fill_path",
            )
        )

    # 2-hop neighborhoods
    for seed in seeds:
        edges = k_hop_pack(
            triples, out_adj, in_adj, seed, k=2, max_edges=max_edges, rng=rng
        )
        if len(edges) < 3:
            continue
        candidates.append(
            make_pack(
                pack_id="tmp",
                document_id=document_id,
                company=company,
                question_type="Complex Reasoning",
                triples=edges,
                method="fill_2hop",
            )
        )

    # Connected random walks from random edges
    for _ in range(n_extra * 4):
        if not triples:
            break
        e0 = rng.randrange(len(triples))
        idxs = [e0]
        node = triples[e0]["tail"]["name"].lower()
        used = {e0}
        for _step in range(rng.randint(3, 8)):
            outs = [ei for ei in out_adj.get(node, []) if ei not in used]
            ins = [ei for ei in in_adj.get(node, []) if ei not in used]
            pool = outs or ins
            if not pool:
                break
            ei = rng.choice(pool)
            idxs.append(ei)
            used.add(ei)
            t = triples[ei]
            # continue from the other endpoint when possible
            h, ta = t["head"]["name"].lower(), t["tail"]["name"].lower()
            node = ta if h == node else h
        edges = take_unique(triples, idxs, max_edges)
        if len(edges) < 3:
            continue
        candidates.append(
            make_pack(
                pack_id="tmp",
                document_id=document_id,
                company=company,
                question_type="Complex Reasoning",
                triples=edges,
                method="fill_walk",
            )
        )

    candidates = [c for c in candidates if c and c.get("est_tokens", 0) <= max_tokens]
    candidates = dedupe_packs(candidates, overlap=0.65)

    # assign new ids not colliding with used
    out: List[Dict[str, Any]] = []
    i = 0
    for c in candidates:
        if len(out) >= n_extra:
            break
        while True:
            pid = f"{document_id}_L2x_{i:03d}"
            i += 1
            if pid not in used_ids:
                break
        c["pack_id"] = pid
        used_ids.add(pid)
        out.append(c)
    return out


def make_fill_clones(
    packs: List[Dict[str, Any]],
    raw_dir: Path,
    *,
    used_ids: Set[str],
    max_clones: int,
) -> List[Tuple[Dict[str, Any], str]]:
    """Return (pack, existing_questions_text) for second-pass generation."""
    clones: List[Tuple[Dict[str, Any], str]] = []
    for pk in packs:
        if str(pk.get("question_type")) != "Complex Reasoning":
            continue
        if len(clones) >= max_clones:
            break
        pid = str(pk.get("pack_id") or "")
        if not pid or pid.endswith("_fill"):
            continue
        raw_path = raw_dir / f"{pid}.json"
        if not raw_path.is_file():
            continue
        try:
            prev = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        qs = [str(q.get("question") or "").strip() for q in (prev.get("questions") or [])]
        qs = [q for q in qs if q]
        if not qs:
            continue
        fill_id = f"{pid}_fill"
        if fill_id in used_ids:
            continue
        clone = dict(pk)
        clone["pack_id"] = fill_id
        clone["method"] = f"{pk.get('method', '')}_fill".strip("_")
        used_ids.add(fill_id)
        existing = "\n".join(f"- {q}" for q in qs)
        clones.append((clone, existing))
    return clones


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--by-report-dir",
        default=str(root / "ESG_KG" / "output" / "triples" / "GPT5.2" / "by_report"),
    )
    p.add_argument(
        "--packs-dir",
        default=str(root / "ESG_KG" / "output" / "qg" / "packs"),
    )
    p.add_argument(
        "--out-dir",
        default=str(root / "ESG_KG" / "output" / "qg"),
    )
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--n-extra-per-doc", type=int, default=12)
    p.add_argument("--num-per-pack", type=int, default=8)
    p.add_argument("--fill-clones", type=int, default=40, help="Second-pass packs from prior L2")
    p.add_argument("--need", type=int, default=173, help="Approx Complex questions still needed")
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--model", default=None)
    p.add_argument("--max-packs", type=int, default=None)
    args = p.parse_args()

    by_dir = Path(args.by_report_dir)
    packs_dir = Path(args.packs_dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    need_q = max(0, args.need)
    # oversample a bit
    need_packs = max(1, int((need_q * 1.25 + args.num_per_pack - 1) // args.num_per_pack))

    # 1) Append extra L2 packs into pack files
    new_packs: List[Dict[str, Any]] = []
    fill_jobs: List[Tuple[Dict[str, Any], str]] = []

    for fp in sorted(packs_dir.glob("*_packs.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        packs: List[Dict[str, Any]] = list(data.get("packs") or [])
        used = existing_pack_ids(packs)
        doc = str(data.get("document_id") or fp.stem.replace("_packs", ""))

        report = load_report_triples(by_dir, doc)
        added: List[Dict[str, Any]] = []
        if report is not None:
            added = build_extra_l2_for_report(
                report,
                used_ids=used,
                rng=rng,
                n_extra=args.n_extra_per_doc,
            )
            packs.extend(added)
            new_packs.extend(added)

        # fill clones from original L2 already in file (before this append is fine too)
        clones = make_fill_clones(
            [x for x in packs if not str(x.get("pack_id", "")).endswith("_fill")],
            raw_dir,
            used_ids=used,
            max_clones=10**9,
        )
        for clone, existing in clones:
            packs.append(clone)
            fill_jobs.append((clone, existing))

        data["packs"] = packs
        data["pack_count"] = len(packs)
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Cap fill clones globally for cost control
    rng.shuffle(fill_jobs)
    fill_jobs = fill_jobs[: args.fill_clones]

    # Prefer brand-new graph packs first, then fill clones
    work: List[Tuple[Dict[str, Any], str]] = [(pk, "") for pk in new_packs]
    work.extend(fill_jobs)

    # Drop packs that already have raw
    pending: List[Tuple[Dict[str, Any], str]] = []
    for pk, existing in work:
        pid = str(pk.get("pack_id"))
        if (raw_dir / f"{pid}.json").is_file():
            continue
        pending.append((pk, existing))

    # Round-robin by document for diversity, then cut to need_packs
    by_doc: Dict[str, List[Tuple[Dict[str, Any], str]]] = defaultdict(list)
    for item in pending:
        by_doc[str(item[0].get("document_id"))].append(item)
    docs = sorted(by_doc.keys())
    selected: List[Tuple[Dict[str, Any], str]] = []
    i = 0
    while docs and len(selected) < need_packs:
        doc = docs[i % len(docs)]
        if by_doc[doc]:
            selected.append(by_doc[doc].pop(0))
        else:
            docs = [d for d in docs if by_doc[d]]
            continue
        i += 1
    if args.max_packs is not None:
        selected = selected[: max(0, args.max_packs)]

    print(
        f"Extra L2 packs built: {len(new_packs)}; fill clones available: {len(fill_jobs)}; "
        f"pending LLM: {len(selected)} (need_q~{need_q})"
    )

    cfg = load_dotenv()
    model = args.model or cfg["LLM_MODEL"]
    prompt_t = load_prompt("Complex Reasoning")
    errors: List[Dict[str, Any]] = []
    kept_total = 0

    for i, (pk, existing) in enumerate(selected, 1):
        pack_id = str(pk.get("pack_id"))
        evidence = str(pk.get("evidence_block") or "")
        if not evidence:
            print(f"[{i}/{len(selected)}] SKIP empty {pack_id}")
            continue
        prompt = render_qg_prompt(
            prompt_t,
            num=args.num_per_pack,
            evidence_block=evidence,
            existing_questions=existing,
        )
        try:
            raw = call_llm(prompt, model=model, temperature=0.35)
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
            ok, _ = validate_question(q, "Complex Reasoning")
            if not ok:
                continue
            item = {
                "question": q.get("question"),
                "answer": q.get("answer"),
                "question_type": "Complex Reasoning",
                "evidence": q.get("evidence") or "",
                "evidence_triple": q.get("evidence_triple") or "",
                "document_id": pk.get("document_id"),
                "company": pk.get("company"),
                "pack_id": pack_id,
            }
            kept.append(item)

        (raw_dir / f"{pack_id}.json").write_text(
            json.dumps(
                {
                    "pack_id": pack_id,
                    "document_id": pk.get("document_id"),
                    "question_type": "Complex Reasoning",
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
        kept_total += len(kept)
        print(
            f"[{i}/{len(selected)}] {pack_id} kept={len(kept)} "
            f"session_kept={kept_total}"
        )
        if args.sleep:
            time.sleep(args.sleep)

    # refresh packs index
    index: List[Dict[str, Any]] = []
    totals: Dict[str, int] = defaultdict(int)
    for fp in sorted(packs_dir.glob("*_packs.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for pk in data.get("packs") or []:
            totals[str(pk.get("question_type"))] += 1
            index.append(
                {
                    "pack_id": pk.get("pack_id"),
                    "document_id": data.get("document_id"),
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

    print(f"LLM kept this session: {kept_total}; errors={len(errors)}")
    print("Rebuilding dataset from raw...")
    rc = rebuild_dataset()
    if errors and any("TOKEN_OR_QUOTA_EXHAUSTED" in str(e.get("error")) for e in errors):
        return 3
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
