"""Write HippoRAG-shaped report triple JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_report_triples(
    out_path: Path,
    *,
    document_id: str,
    company: str,
    model: str,
    triples: List[Dict[str, Any]],
    seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "document_id": document_id,
        "company": company,
        "extractor": "hipporag1",
        "model": model,
        "triple_count": len(triples),
        "triples": triples,
        "seconds": seconds,
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
