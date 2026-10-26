"""Export LightRAG knowledge-graph edges into HippoRAG-compatible triple JSON."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_PAGE_RE = re.compile(r"(?i)\bpage\s*[:=]?\s*(\d+)\b|\[p(\d+)\]|##\s*Page\s+(\d+)")


def _page_from_text(text: str) -> Optional[str]:
    m = _PAGE_RE.search(text or "")
    if not m:
        return None
    for g in m.groups():
        if g:
            return str(g)
    return None


def triples_from_csv_export(csv_path: Path, *, source: str = "lightrag") -> List[Dict[str, Any]]:
    """Parse LightRAG CSV export (entities/relations) into unified triples."""
    if not csv_path.is_file():
        return []
    triples: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    # Heuristic: relation rows usually have source/target or head/tail columns.
    for row in rows:
        keys = {k.lower(): k for k in row.keys()}
        head = (
            row.get(keys.get("source", ""), "")
            or row.get(keys.get("head", ""), "")
            or row.get(keys.get("src_id", ""), "")
            or row.get(keys.get("entity1", ""), "")
        )
        tail = (
            row.get(keys.get("target", ""), "")
            or row.get(keys.get("tail", ""), "")
            or row.get(keys.get("tgt_id", ""), "")
            or row.get(keys.get("entity2", ""), "")
        )
        rel = (
            row.get(keys.get("relationship", ""), "")
            or row.get(keys.get("relation", ""), "")
            or row.get(keys.get("description", ""), "")
            or row.get(keys.get("keywords", ""), "")
        )
        head, rel, tail = str(head).strip(), str(rel).strip(), str(tail).strip()
        if not head or not tail:
            continue
        if not rel:
            rel = "related_to"
        desc = str(
            row.get(keys.get("description", ""), "")
            or row.get(keys.get("content", ""), "")
            or ""
        )
        page = _page_from_text(desc) or "1"
        triples.append(
            {
                "head": {"type": "Other", "name": head},
                "relation": rel[:200],
                "tail": {"type": "Other", "name": tail},
                "qualifiers": {"page": page, "year": None, "unit": None, "scope": None},
                "evidence_span": desc[:500],
                "chunk_id": "",
                "source": source,
            }
        )
    return triples


def triples_from_graphml(graphml_path: Path, *, source: str = "lightrag") -> List[Dict[str, Any]]:
    """Parse NetworkX GraphML edges written by LightRAG."""
    if not graphml_path.is_file():
        return []
    try:
        import networkx as nx
    except Exception:
        return []
    try:
        g = nx.read_graphml(str(graphml_path))
    except Exception:
        return []
    triples: List[Dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        head = str(u).strip()
        tail = str(v).strip()
        if not head or not tail:
            continue
        rel = (
            str(data.get("description") or data.get("keywords") or data.get("relation") or "")
            .strip()
            or "related_to"
        )
        desc = str(data.get("description") or "")
        page = _page_from_text(desc) or "1"
        triples.append(
            {
                "head": {"type": "Other", "name": head},
                "relation": rel[:200],
                "tail": {"type": "Other", "name": tail},
                "qualifiers": {"page": page, "year": None, "unit": None, "scope": None},
                "evidence_span": desc[:500],
                "chunk_id": str(data.get("source_id") or data.get("chunk_id") or ""),
                "source": source,
            }
        )
    return triples


def export_triples_from_working_dir(
    working_dir: Path,
    *,
    source: str = "lightrag",
) -> List[Dict[str, Any]]:
    """Prefer GraphML; fall back to any CSV export in the working dir."""
    working_dir = Path(working_dir)
    # LightRAG 1.x writes graph_{namespace}.graphml under working_dir[/workspace]
    for graphml in sorted(working_dir.rglob("*.graphml")):
        triples = triples_from_graphml(graphml, source=source)
        if triples:
            return triples
    for csv_path in sorted(working_dir.rglob("*.csv")):
        triples = triples_from_csv_export(csv_path, source=source)
        if triples:
            return triples
    return []


def write_report_triples(
    out_path: Path,
    *,
    document_id: str,
    company: str,
    model: str,
    triples: List[Dict[str, Any]],
    seconds: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "document_id": document_id,
        "company": company,
        "model": model,
        "extractor": "lightrag",
        "source_corpus": "esg_ocr",
        "triple_count": len(triples),
        "seconds": round(float(seconds), 3),
        "triples": triples,
        "errors": [],
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
