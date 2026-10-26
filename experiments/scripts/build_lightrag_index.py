#!/usr/bin/env python3
"""Deprecated entrypoint — use extract_lightrag_triples.py.

Kept for compatibility; forwards to extract_lightrag_triples.main().
"""

from extract_lightrag_triples import main

if __name__ == "__main__":
    raise SystemExit(main())
