"""Generation evaluation for ESG GraphReason QA (GraphRAG-Bench metrics).

Metrics by question type (same as paper):
  Fact Retrieval / Complex Reasoning → ROUGE-L, answer_correctness
  Contextual Summarize → answer_correctness, coverage_score
  Creative Generation → answer_correctness, coverage_score, faithfulness

Coverage uses gold *evidence* when available (else ground_truth).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from langchain_core.embeddings import Embeddings

# Allow `python -m Evaluation.generation_eval` from evaluate/
_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from Evaluation.adapter import (  # noqa: E402
    load_gold_index,
    load_qa_results,
    build_eval_rows,
    merge_qa_with_gold,
)
from Evaluation.clients import add_common_model_args, build_clients  # noqa: E402
from Evaluation.env_utils import detect_newesg_root  # noqa: E402
from Evaluation.metrics import (  # noqa: E402
    compute_answer_correctness,
    compute_coverage_score,
    compute_faithfulness_score,
    compute_rouge_score,
)

METRIC_CONFIG = {
    "Fact Retrieval": ["rouge_score", "answer_correctness"],
    "Complex Reasoning": ["rouge_score", "answer_correctness"],
    "Contextual Summarize": ["answer_correctness", "coverage_score"],
    "Creative Generation": ["answer_correctness", "coverage_score", "faithfulness"],
}


async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    coverage_reference: str,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
) -> Dict[str, float]:
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)

    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )

    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, coverage_reference, answer, llm
        )

    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )

    task_results = await asyncio.gather(*tasks.values())
    return {metric: task_results[i] for i, metric in enumerate(tasks.keys())}


async def evaluate_dataset(
    dataset: Dict[str, List[Any]],
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 4,
    detailed_output: bool = False,
) -> Dict[str, Any]:
    results = {metric: [] for metric in metrics}
    detailed_results: List[Dict[str, Any]] = [] if detailed_output else []

    ids = dataset["id"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    coverage_refs = dataset["coverage_reference"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_with_semaphore(i: int):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                coverage_reference=coverage_refs[i],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "coverage_reference": coverage_refs[i],
                    "generated_answer": answers[i],
                    "contexts": contexts_list[i],
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

    avg_results = {metric: float(np.nanmean(scores)) for metric, scores in results.items()}
    if detailed_output:
        return {"average_scores": avg_results, "detailed": detailed_results}
    return avg_results


def _coverage_reference(item: Dict[str, Any]) -> str:
    """Prefer gold evidence text for coverage; fall back to ground_truth."""
    evidence = item.get("evidence") or []
    if isinstance(evidence, list) and evidence:
        return "\n".join(str(x) for x in evidence if str(x).strip())
    if isinstance(evidence, str) and evidence.strip():
        return evidence.strip()
    return str(item.get("ground_truth") or "")


async def run_generation_eval(args: argparse.Namespace) -> Dict[str, Any]:
    llm, embedding, meta = build_clients(args)
    print(f"Judge LLM: {meta['model']} @ {meta['base_url']}")
    print(f"Embeddings: {meta['embedding_model']} @ {meta['embedding_base_url']}")

    if args.qa_results:
        gold_path = args.gold
        if not gold_path:
            root = detect_newesg_root()
            gold_path = str(root / "ESG_KG" / "output" / "qg" / "esg_qg_gold_for_eval.json")
        print(f"Merging QA={args.qa_results} with gold={gold_path}")
        file_data = merge_qa_with_gold(args.qa_results, gold_path)
        if args.save_merged:
            Path(args.save_merged).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_merged).write_text(
                json.dumps(file_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote merged eval rows → {args.save_merged}")
    else:
        print(f"Loading evaluation data from {args.data_file}...")
        with open(args.data_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        # If caller passed raw QA + separate gold via --gold, merge.
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
        if question_type not in METRIC_CONFIG:
            print(f"Skipping undefined question type: {question_type}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Evaluating question type: {question_type} (n={len(group_items)})")
        print(f"{'=' * 50}")

        dataset = {
            "id": [item["id"] for item in group_items],
            "question": [item["question"] for item in group_items],
            "answer": [item["generated_answer"] for item in group_items],
            "contexts": [item["context"] for item in group_items],
            "ground_truth": [item["ground_truth"] for item in group_items],
            "coverage_reference": [_coverage_reference(item) for item in group_items],
        }
        if args.num_samples:
            n = min(args.num_samples, len(dataset["id"]))
            dataset = {k: v[:n] for k, v in dataset.items()}

        results = await evaluate_dataset(
            dataset=dataset,
            metrics=METRIC_CONFIG[question_type],
            llm=llm,
            embeddings=embedding,
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
        description="ESG generation evaluation (GraphRAG-Bench metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_model_args(parser)
    parser.add_argument(
        "--qa-results",
        type=str,
        default=None,
        help="Path to HippoRAG2/LightRAG results.json (preferred)",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Gold file (default: ESG_KG/output/qg/esg_qg_gold_for_eval.json)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Pre-merged paper-format JSON (alternative to --qa-results)",
    )
    parser.add_argument(
        "--save-merged",
        type=str,
        default=None,
        help="Optional path to dump merged eval rows",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation scores",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Max samples per question type (smoke tests)",
    )
    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Include per-sample metric details",
    )
    return parser


async def main(args: argparse.Namespace):
    if not args.qa_results and not args.data_file:
        raise SystemExit("Provide --qa-results or --data_file")
    await run_generation_eval(args)


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
