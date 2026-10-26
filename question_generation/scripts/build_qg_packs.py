#!/usr/bin/env python3
"""Build ESG QG evidence packs from GPT5.2 by_report triples.

Reads:  ESG_KG/output/triples/GPT5.2/by_report/*_triples.json
Writes: ESG_KG/output/qg/packs/{document_id}_packs.json
        ESG_KG/output/qg/packs/index.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from qg_common import (
    detect_newesg_root,
    estimate_tokens,
    format_evidence_block,
    format_triple_line,
    iter_report_files,
    normalize_triple,
    triple_key,
)


def build_graph(
    triples: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List[Dict[str, Any]]]:
    out_adj: Dict[str, List[int]] = defaultdict(list)
    in_adj: Dict[str, List[int]] = defaultdict(list)
    clean: List[Dict[str, Any]] = []
    for t in triples:
        nt = normalize_triple(t)
        if not nt:
            continue
        idx = len(clean)
        clean.append(nt)
        h = nt["head"]["name"].lower()
        ta = nt["tail"]["name"].lower()
        out_adj[h].append(idx)
        in_adj[ta].append(idx)
    return out_adj, in_adj, clean


def pack_text_and_tokens(
    document_id: str,
    company: str,
    triples: Sequence[Dict[str, Any]],
) -> Tuple[str, int]:
    text = format_evidence_block(
        document_id=document_id, company=company, triples=triples
    )
    return text, estimate_tokens(text)


def jaccard_keys(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> float:
    sa = {triple_key(t) for t in a}
    sb = {triple_key(t) for t in b}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def take_unique(
    triples: Sequence[Dict[str, Any]], indices: Sequence[int], max_edges: int
) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for i in indices:
        if i < 0 or i >= len(triples):
            continue
        t = triples[i]
        k = triple_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_edges:
            break
    return out


def k_hop_pack(
    triples: List[Dict[str, Any]],
    out_adj: Dict[str, List[int]],
    in_adj: Dict[str, List[int]],
    seed: str,
    *,
    k: int,
    max_edges: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    seed = seed.lower()
    frontier = {seed}
    visited_nodes = {seed}
    edge_ids: List[int] = []
    seen_e: Set[int] = set()
    for _ in range(k):
        nxt: Set[str] = set()
        cand_edges: List[int] = []
        for n in frontier:
            cand_edges.extend(out_adj.get(n, []))
            cand_edges.extend(in_adj.get(n, []))
        rng.shuffle(cand_edges)
        for ei in cand_edges:
            if ei in seen_e:
                continue
            seen_e.add(ei)
            edge_ids.append(ei)
            t = triples[ei]
            h = t["head"]["name"].lower()
            ta = t["tail"]["name"].lower()
            if h not in visited_nodes:
                nxt.add(h)
                visited_nodes.add(h)
            if ta not in visited_nodes:
                nxt.add(ta)
                visited_nodes.add(ta)
            if len(edge_ids) >= max_edges:
                break
        frontier = nxt
        if len(edge_ids) >= max_edges or not frontier:
            break
    return take_unique(triples, edge_ids, max_edges)


def sample_paths(
    triples: List[Dict[str, Any]],
    out_adj: Dict[str, List[int]],
    *,
    min_len: int,
    max_len: int,
    n_paths: int,
    rng: random.Random,
) -> List[List[int]]:
    """Return lists of edge indices forming directed paths."""
    if not triples:
        return []
    starts = [i for i, t in enumerate(triples) if out_adj.get(t["tail"]["name"].lower())]
    if not starts:
        starts = list(range(len(triples)))
    paths: List[List[int]] = []
    attempts = max(n_paths * 20, 50)
    for _ in range(attempts):
        if len(paths) >= n_paths:
            break
        e0 = rng.choice(starts)
        path = [e0]
        node = triples[e0]["tail"]["name"].lower()
        used = {e0}
        target_len = rng.randint(min_len, max_len)
        while len(path) < target_len:
            outs = [ei for ei in out_adj.get(node, []) if ei not in used]
            if not outs:
                break
            ei = rng.choice(outs)
            path.append(ei)
            used.add(ei)
            node = triples[ei]["tail"]["name"].lower()
        if len(path) >= min_len:
            paths.append(path)
    # dedupe by edge-set
    uniq: List[List[int]] = []
    seen: Set[frozenset] = set()
    for p in paths:
        key = frozenset(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def make_pack(
    *,
    pack_id: str,
    document_id: str,
    company: str,
    question_type: str,
    triples: List[Dict[str, Any]],
    method: str,
) -> Optional[Dict[str, Any]]:
    if not triples:
        return None
    text, toks = pack_text_and_tokens(document_id, company, triples)
    return {
        "pack_id": pack_id,
        "document_id": document_id,
        "company": company,
        "question_type": question_type,
        "method": method,
        "triple_count": len(triples),
        "est_tokens": toks,
        "triples": triples,
        "evidence_block": text,
        "triple_lines": [format_triple_line(t) for t in triples],
    }


def dedupe_packs(
    packs: List[Dict[str, Any]], *, overlap: float = 0.7
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for p in packs:
        if any(jaccard_keys(p["triples"], k["triples"]) >= overlap for k in kept):
            continue
        kept.append(p)
    return kept


def build_packs_for_report(
    report: Dict[str, Any],
    *,
    rng: random.Random,
    max_edges_l1: int = 35,
    max_edges_l2: int = 25,
    max_edges_l3: int = 45,
    max_edges_l4: int = 60,
    min_tokens: int = 400,
    max_tokens: int = 5500,
    n_l1: int = 12,
    n_l2: int = 10,
    n_l3: int = 6,
    n_l4: int = 3,
) -> List[Dict[str, Any]]:
    document_id = str(report.get("document_id") or report.get("company") or "unknown")
    company = str(report.get("company") or document_id.split("-")[0])
    raw = report.get("triples") or []
    out_adj, in_adj, triples = build_graph(raw)
    if not triples:
        return []

    # Small reports: whole graph as multi-type packs
    whole_text, whole_tok = pack_text_and_tokens(document_id, company, triples)
    packs: List[Dict[str, Any]] = []

    if len(triples) <= 120 or whole_tok <= max_tokens:
        for qt, method in (
            ("Fact Retrieval", "whole_report"),
            ("Complex Reasoning", "whole_report"),
            ("Contextual Summarize", "whole_report"),
            ("Creative Generation", "whole_report"),
        ):
            # Creative needs >=5 triples
            if qt == "Creative Generation" and len(triples) < 5:
                continue
            if qt == "Complex Reasoning" and len(triples) < 3:
                continue
            p = make_pack(
                pack_id=f"{document_id}_{qt.replace(' ', '')}_whole",
                document_id=document_id,
                company=company,
                question_type=qt,
                triples=triples[:max_edges_l4],
                method=method,
            )
            if p and min_tokens <= p["est_tokens"] <= max_tokens + 1500:
                packs.append(p)
            elif p and p["est_tokens"] < min_tokens and len(triples) >= 3:
                # keep tiny reports anyway
                packs.append(p)
        if packs:
            # Whole-report packs share the same edges across L1–L4; keep one per type.
            by_qt: Dict[str, Dict[str, Any]] = {}
            for p in packs:
                by_qt.setdefault(str(p["question_type"]), p)
            return list(by_qt.values())

    # Degree-based seeds
    degree: Dict[str, int] = defaultdict(int)
    for t in triples:
        degree[t["head"]["name"].lower()] += 1
        degree[t["tail"]["name"].lower()] += 1
    seeds = [n for n, _ in sorted(degree.items(), key=lambda x: -x[1])]
    if len(seeds) > 80:
        seeds = seeds[:40] + rng.sample(seeds[40:], min(40, len(seeds) - 40))

    # L1 packs: 1-2 hop
    l1: List[Dict[str, Any]] = []
    for seed in seeds:
        if len(l1) >= n_l1:
            break
        for k in (1, 2):
            edges = k_hop_pack(
                triples, out_adj, in_adj, seed, k=k, max_edges=max_edges_l1, rng=rng
            )
            if len(edges) < 2:
                continue
            p = make_pack(
                pack_id=f"{document_id}_L1_{len(l1):03d}",
                document_id=document_id,
                company=company,
                question_type="Fact Retrieval",
                triples=edges,
                method=f"{k}hop",
            )
            if not p:
                continue
            if p["est_tokens"] > max_tokens:
                edges = edges[: max(8, len(edges) // 2)]
                p = make_pack(
                    pack_id=p["pack_id"],
                    document_id=document_id,
                    company=company,
                    question_type="Fact Retrieval",
                    triples=edges,
                    method=f"{k}hop_trim",
                )
            if p and p["est_tokens"] <= max_tokens:
                l1.append(p)
                break
    packs.extend(dedupe_packs(l1))

    # L2 packs: paths >=3 (+ light 1-hop context)
    l2: List[Dict[str, Any]] = []
    paths = sample_paths(
        triples, out_adj, min_len=3, max_len=6, n_paths=n_l2 * 3, rng=rng
    )
    for path in paths:
        if len(l2) >= n_l2:
            break
        idxs = list(path)
        nodes = set()
        for ei in path:
            t = triples[ei]
            nodes.add(t["head"]["name"].lower())
            nodes.add(t["tail"]["name"].lower())
        ctx: List[int] = []
        for n in nodes:
            ctx.extend(out_adj.get(n, [])[:2])
            ctx.extend(in_adj.get(n, [])[:2])
        rng.shuffle(ctx)
        idxs.extend(ctx)
        edges = take_unique(triples, idxs, max_edges_l2)
        if len(edges) < 3:
            continue
        p = make_pack(
            pack_id=f"{document_id}_L2_{len(l2):03d}",
            document_id=document_id,
            company=company,
            question_type="Complex Reasoning",
            triples=edges,
            method="path3plus",
        )
        if p and p["est_tokens"] <= max_tokens:
            l2.append(p)
    packs.extend(dedupe_packs(l2))

    # L3 packs: denser 2-hop around mid-degree seeds
    l3: List[Dict[str, Any]] = []
    mid_seeds = seeds[max(0, len(seeds) // 4) : max(1, (3 * len(seeds)) // 4)] or seeds
    for seed in mid_seeds:
        if len(l3) >= n_l3:
            break
        edges = k_hop_pack(
            triples, out_adj, in_adj, seed, k=2, max_edges=max_edges_l3, rng=rng
        )
        if len(edges) < 4:
            continue
        p = make_pack(
            pack_id=f"{document_id}_L3_{len(l3):03d}",
            document_id=document_id,
            company=company,
            question_type="Contextual Summarize",
            triples=edges,
            method="dense2hop",
        )
        if p and p["est_tokens"] <= max_tokens:
            l3.append(p)
    packs.extend(dedupe_packs(l3))

    # L4 packs: larger neighborhoods
    l4: List[Dict[str, Any]] = []
    for seed in seeds[: max(n_l4 * 4, 8)]:
        if len(l4) >= n_l4:
            break
        edges = k_hop_pack(
            triples, out_adj, in_adj, seed, k=2, max_edges=max_edges_l4, rng=rng
        )
        if len(edges) < 5:
            continue
        p = make_pack(
            pack_id=f"{document_id}_L4_{len(l4):03d}",
            document_id=document_id,
            company=company,
            question_type="Creative Generation",
            triples=edges,
            method="broad2hop",
        )
        if p and p["est_tokens"] <= max_tokens:
            l4.append(p)
    packs.extend(dedupe_packs(l4))

    # filter extremely tiny unless only option
    filtered = [p for p in packs if p["est_tokens"] >= min_tokens or len(triples) < 30]
    return filtered or packs


def main() -> int:
    root = detect_newesg_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--by-report-dir",
        default=str(root / "ESG_KG" / "output" / "triples" / "GPT5.2" / "by_report"),
    )
    p.add_argument(
        "--out-dir",
        default=str(root / "ESG_KG" / "output" / "qg" / "packs"),
    )
    p.add_argument("--pdf", default="", help="Optional document_id filter substring")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=5500)
    p.add_argument("--min-tokens", type=int, default=400)
    args = p.parse_args()

    by_dir = Path(args.by_report_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    files = iter_report_files(by_dir)
    needle = (args.pdf or "").strip().lower()
    if needle:
        files = [f for f in files if needle in f.name.lower()]
    if not files:
        print(f"No report files in {by_dir}", file=sys.stderr)
        return 2

    index: List[Dict[str, Any]] = []
    totals: Dict[str, int] = defaultdict(int)

    for fp in files:
        report = json.loads(fp.read_text(encoding="utf-8"))
        packs = build_packs_for_report(
            report,
            rng=rng,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        doc = str(report.get("document_id") or fp.stem.replace("_triples", ""))
        out_path = out_dir / f"{doc}_packs.json"
        payload = {
            "document_id": doc,
            "company": report.get("company"),
            "source": str(fp).replace("\\", "/"),
            "pack_count": len(packs),
            "packs": packs,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for pk in packs:
            totals[pk["question_type"]] += 1
            index.append(
                {
                    "pack_id": pk["pack_id"],
                    "document_id": doc,
                    "question_type": pk["question_type"],
                    "triple_count": pk["triple_count"],
                    "est_tokens": pk["est_tokens"],
                    "file": out_path.name,
                }
            )
        print(
            f"[{doc}] packs={len(packs)} "
            f"by_type="
            + ",".join(
                f"{qt.split()[0]}:{sum(1 for x in packs if x['question_type']==qt)}"
                for qt in (
                    "Fact Retrieval",
                    "Complex Reasoning",
                    "Contextual Summarize",
                    "Creative Generation",
                )
            )
        )

    index_path = out_dir / "index.json"
    index_path.write_text(
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
    print(f"Wrote {index_path} total_packs={len(index)} counts={dict(totals)}")
    return 0


if __name__ == "__main__":
    # fix accidental bad code path - rewrite L2 section cleanly by re-reading file if needed
    raise SystemExit(main())
