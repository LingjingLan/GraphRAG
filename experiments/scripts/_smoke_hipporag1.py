#!/usr/bin/env python3
"""Smoke-test HippoRAG1 package imports + settings."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    backend_src = Path(__file__).resolve().parents[1] / "src"
    if str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))
    vendor = Path(__file__).resolve().parents[1] / "vendor" / "HippoRAG-legacy"
    print("vendor_exists", vendor.is_dir())
    print("vendor_readme", (vendor / "README.md").is_file())

    from esg_encoding.retrieval.hipporag1.settings import HippoRAG1Settings
    from esg_encoding.retrieval.hipporag1 import prompts as P
    from esg_encoding.retrieval.hipporag1.openie import processing_phrases

    s = HippoRAG1Settings()
    print("embedding_model", s.embedding_model_name)
    print("sim_threshold", s.sim_threshold)
    print("damping", s.damping)
    print("top_k", s.top_k)
    print("prompt_ner_ok", bool(P.NER_SYSTEM))
    print("processing", processing_phrases("Hello, World!"))
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
