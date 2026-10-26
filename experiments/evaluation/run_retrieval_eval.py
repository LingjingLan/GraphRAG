#!/usr/bin/env python3
"""Run retrieval evaluation on ESG QA results.

Example (from ESG_KG/output/evaluate):

  python run_retrieval_eval.py \\
    --qa-results ../qa/HippoRAG2-GPT4omini-GPT4omini/results.json \\
    --output_file ./results/HippoRAG2-GPT4omini-GPT4omini/retrieval.json \\
    --num_samples 2
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.retrieval_eval import build_parser, main  # noqa: E402


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.output_file or args.output_file == "retrieval_results.json":
        if args.qa_results:
            name = Path(args.qa_results).parent.name
            args.output_file = str(ROOT / "results" / name / "retrieval.json")
    asyncio.run(main(args))
