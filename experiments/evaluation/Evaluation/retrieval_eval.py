"""Retrieval evaluation for ESG GraphReason QA (GraphRAG-Bench metrics).

Metrics:
  context_relevancy, evidence_recall

Expects retrieved `context` + gold `evidence` (merged from gold file).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from Evaluation.adapter import merge_qa_with_gold, build_eval_rows, load_gold_index  # noqa: E402
from Evaluation.clients import add_common_model_args, build_clients  # noqa: E402
from Evaluation.env_utils import detect_newesg_root  # noqa: E402
from Evaluation.metrics import compute_context_relevance, compute_evidence_recall  # noqa: E402


async def evaluate_sample(
    question: str,
    contexts: List[str],
    evidences: List[str],
    llm: Any,
) -> Dict[str, float]:
    relevance_score, recall_score = await asyncio.gather(
        compute_context_relevance(question, contexts, llm),
        compute_evidence_recall(question, contexts, evidences, llm),
    )
    return {
        "context_relevancy": relevance_score,
        "evidence_recall": recall_score,
    }


async def evaluate_dataset(
    dataset: Dict[str, List[Any]],
    llm: Any,
    max_concurrent: int = 4,
    detailed_output: bool = False,
) -> Dict[str, Any]:
    results = {"context_relevancy": [], "evidence_recall": []}
    detailed_results: List[Dict[str, Any]] = [] if detailed_output else []

    ids = dataset["id"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    evidences = dataset["evidences"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_with_semaphore(i: int):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                contexts=contexts_list[i],
                evidences=evidences[i],
                llm=llm,
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "contexts": contexts_list[i],
                    "evidences": evidences[i],
                    "metrics": sample_metrics,
                }
            return sample_metrics

    tasks = [evaluate_with_semaphore(i) for i in range(total_samples)]
    completed = 0
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            if detailed_output:
                detailed_results.append(result)
                metrics_dict = result.get("metrics", {}) if isinstance(result, dict) else {}
                for metric, score in metrics_dict.items():
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        results[metric].append(score)
            else:
                if isinstance(result, dict):
                    for metric, score in result.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            completed += 1
            print(
                f"Completed sample {completed}/{total_samples} - "
                f"{(completed / total_samples) * 100:.1f}%"
            )
        except Exception as e:
            print(f"Sample failed: {e}")
            completed += 1

    avg_results = {
        "context_relevancy": float(np.nanmean(results["context_relevancy"])),
        "evidence_recall": float(np.nanmean(results["evidence_recall"])),
    }
    if detailed_output:
        return {"average_scores": avg_results, "detailed": detailed_results}
    return avg_results


async def run_retrieval_eval(args: argparse.Namespace) -> Dict[str, Any]:
    llm, _embedding, meta = build_clients(args)
    print(f"Judge LLM: {meta['model']} @ {meta['base_url']}")

    if args.qa_results:
        gold_path = args.gold
        if not gold_path:
            root = detect_newesg_root()
            gold_path = str(root / "ESG_KG" / "output" / "qg" / "esg_qg_gold_for_eval.json")
        print(f"Merging QA={args.qa_results} with gold={gold_path}")
        file_data = merge_qa_with_gold(args.qa_results, gold_path)
    else:
        print(f"Loading evaluation data from {args.data_file}...")
        with open(args.data_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        if args.gold:
            file_data = build_eval_rows(file_data, load_gold_index(args.gold))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        grouped.setdefault(q_type, []).append(item)

    all_results: Dict[str, Any] = {
        "_meta": {
            **meta,
            "n_rows": len(file_data),
            "qa_results": args.qa_results,
            "gold": args.gold,
            "data_file": args.data_file,
        }
    }

    for question_type, group_items in grouped.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating question type: {question_type} (n={len(group_items)})")
        print(f"{'=' * 50}")

        dataset = {
            "id": [item["id"] for item in group_items],
            "question": [item["question"] for item in group_items],
            "contexts": [item["context"] for item in group_items],
            "evidences": [item["evidence"] for item in group_items],
        }
        if args.num_samples:
            n = min(args.num_samples, len(dataset["id"]))
            dataset = {k: v[:n] for k, v in dataset.items()}

        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm,
            max_concurrent=args.max_concurrent,
            detailed_output=args.detailed_output,
        )
        all_results[question_type] = results

        print(f"\nResults for {question_type}:")
        scores = results["average_scores"] if args.detailed_output else results
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaving results to {out}...")

    if args.mode == "ollama" and hasattr(llm, "close"):
        await llm.close()
    print("\nEvaluation complete.")
    return all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ESG retrieval evaluation (GraphRAG-Bench metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_model_args(parser)
    parser.add_argument("--qa-results", type=str, default=None)
    parser.add_argument("--gold", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument(
        "--output_file",
        type=str,
        default="retrieval_results.json",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--detailed_output", action="store_true")
    return parser


async def main(args: argparse.Namespace):
    if not args.qa_results and not args.data_file:
        raise SystemExit("Provide --qa-results or --data_file")
    await run_retrieval_eval(args)


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
