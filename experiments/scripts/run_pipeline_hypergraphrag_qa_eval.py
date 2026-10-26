#!/usr/bin/env python3
"""Wait for HyperGraphRAG QA to finish, then run generation eval + summary."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def detect_root() -> Path:
    here = Path(__file__).resolve()
    sibling = here.parents[2].parent  # EulerESG parent = newESG when scripts under backend/scripts
    # __file__ = .../EulerESG/backend/scripts/... -> parents[2]=EulerESG, parent=newESG
    if (sibling / "Datasets" / "Corpus").is_dir():
        return sibling
    if Path("/workspace/Datasets/Corpus").is_dir():
        return Path("/workspace")
    return sibling


def unique_ok(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    ok = {
        str(r.get("id"))
        for r in rows
        if isinstance(r, dict)
        and r.get("id")
        and not r.get("error")
        and (r.get("generated_answer") or "")
    }
    return len(ok), len(rows)


def main() -> int:
    root = detect_root()
    qa_out = root / "ESG_KG" / "output" / "qa" / "HyperGraphRAG-GPT5.1-GPT4omini" / "results.json"
    eval_dir = root / "ESG_KG" / "output" / "evaluate"
    eval_out = eval_dir / "results" / "HyperGraphRAG-GPT5.1-GPT4omini" / "generation_full.json"
    summary_prefix = eval_dir / "results" / "summary_hypergraphrag_gpt51_gpt4omini_generation_full"
    target = 2015
    poll = 120

    print(f"Waiting for QA unique_ok={target}: {qa_out}", flush=True)
    while True:
        ok, rows = unique_ok(qa_out)
        print(f"QA progress: unique_ok={ok}/{target} rows={rows}", flush=True)
        if ok >= target:
            break
        time.sleep(poll)

    print("QA complete. Starting generation eval...", flush=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    # Prefer host python for evaluate (same as previous Hippo/LightRAG pipeline)
    py = sys.executable
    cmd_eval = [
        py,
        "-u",
        str(eval_dir / "run_generation_eval.py"),
        "--qa-results",
        str(qa_out),
        "--model",
        "gpt-4o-mini",
        "--base_url",
        "https://api.openai.com/v1",
        "--max_concurrent",
        "6",
        "--output_file",
        str(eval_out),
    ]
    print(">>>", " ".join(cmd_eval), flush=True)
    r1 = subprocess.run(cmd_eval, cwd=str(eval_dir))
    if r1.returncode != 0:
        print(f"Eval failed exit={r1.returncode}", file=sys.stderr)
        return r1.returncode or 1

    cmd_sum = [
        py,
        "-u",
        str(eval_dir / "summarize_generation_results.py"),
        "--glob",
        "HyperGraphRAG-GPT5.1-GPT4omini/generation_full.json",
        "--out-prefix",
        str(summary_prefix),
    ]
    print(">>>", " ".join(cmd_sum), flush=True)
    r2 = subprocess.run(cmd_sum, cwd=str(eval_dir))
    if r2.returncode != 0:
        return r2.returncode or 1

    print("PIPELINE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
