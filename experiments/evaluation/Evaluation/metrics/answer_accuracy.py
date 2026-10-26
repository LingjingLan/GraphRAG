import asyncio
import json
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
from langchain_core.language_models import BaseLanguageModel
import re
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler

# Define necessary Pydantic models
class StatementsWithReason(BaseModel):
    statement: str
    reason: str

class ClassificationWithReason(BaseModel):
    TP: List[StatementsWithReason] = []
    FP: List[StatementsWithReason] = []
    FN: List[StatementsWithReason] = []

class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: List[str]
    ground_truth: List[str]

# F-beta score calculation
def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

# Statement generation prompt template (ESG / corporate sustainability QA)
STATEMENT_GENERATOR_PROMPT = """
You are evaluating answers about corporate ESG / sustainability disclosures (e.g. emissions, GRI metrics, workforce, climate targets).
Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Preserve numbers, units, years, and metric names exactly. Format the outputs in JSON.

Example Input:
Question: What were ALS's Scope 1 greenhouse gas emissions in 2024, and did they decrease versus 2023?
Answer: It reported Scope 1 emissions of 12,450 tCO2e in 2024. This was a decrease from the prior year. The company also expanded renewable electricity procurement.

Example Output:
["ALS reported Scope 1 emissions of 12,450 tCO2e in 2024.", "ALS's Scope 1 emissions decreased versus 2023.", "ALS expanded renewable electricity procurement."]

Input Text:
Question:{question}
Answer: {answer}

Generated Statements:
"""

# Correctness classification prompt template
CORRECTNESS_PROMPT_TEMPLATE = """
You are judging factual correctness for ESG report question answering.
Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth, FP (false positive): statements present in the answer but not directly supported by any statement in ground truth, FN (false negative): statements found in the ground truth but not present in answer. Each statement can only belong to one of the categories. Provide a reason for each classification.
Treat numeric values, units, years, entity names, and disclosure labels (e.g. GRI, Scope 1/2/3) strictly: mismatched figures or years are FP / FN, not TP.

Examples:
{examples}

Current Analysis:
Question: {question}
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}
"""

# Pre-defined examples for correctness classification (ESG domain)
CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What was Coles's renewable electricity share in FY2024, and what climate target does it support?",
            "answer": [
                "Coles sourced 80% of electricity from renewable sources in FY2024.",
                "This progress supports Coles's goal to achieve net-zero Scope 1 and 2 emissions by 2030."
            ],
            "ground_truth": [
                "Coles sourced 100% of its electricity from renewable sources in FY2024.",
                "Coles aims to achieve net-zero Scope 1 and 2 greenhouse gas emissions by 2050.",
                "Coles reduced operational emissions by 12% versus FY2023.",
                "Coles reports renewable electricity progress under its climate action program."
            ]
        },
        "output": {
            "TP": [],
            "FP": [
                {
                    "statement": "Coles sourced 80% of electricity from renewable sources in FY2024.",
                    "reason": "The ground truth states 100% renewable electricity in FY2024, so 80% is not supported."
                },
                {
                    "statement": "This progress supports Coles's goal to achieve net-zero Scope 1 and 2 emissions by 2030.",
                    "reason": "The ground truth gives a 2050 net-zero Scope 1 and 2 target, not 2030."
                }
            ],
            "FN": [
                {
                    "statement": "Coles sourced 100% of its electricity from renewable sources in FY2024.",
                    "reason": "The correct renewable share from the ground truth is missing from the answer."
                },
                {
                    "statement": "Coles aims to achieve net-zero Scope 1 and 2 greenhouse gas emissions by 2050.",
                    "reason": "The ground-truth target year is not stated in the answer."
                },
                {
                    "statement": "Coles reduced operational emissions by 12% versus FY2023.",
                    "reason": "This emissions-reduction fact from the ground truth is omitted."
                },
                {
                    "statement": "Coles reports renewable electricity progress under its climate action program.",
                    "reason": "This disclosure context from the ground truth is not present in the answer."
                }
            ]
        }
    },
    {
        "input": {
            "question": "In 2024, what was ALS's value for 'Incident no longer subject to action' (GRI disclosure)?",
            "answer": [
                "ALS reported a value of 4 for 'Incident no longer subject to action' in 2024."
            ],
            "ground_truth": [
                "ALS reported a value of 4 for 'Incident no longer subject to action' in 2024.",
                "The figure appears in the Key data - Full time and part time employees table."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "ALS reported a value of 4 for 'Incident no longer subject to action' in 2024.",
                    "reason": "This statement matches the ground-truth GRI metric value and year."
                }
            ],
            "FP": [],
            "FN": [
                {
                    "statement": "The figure appears in the Key data - Full time and part time employees table.",
                    "reason": "The table/source location from the ground truth is not mentioned in the answer."
                }
            ]
        }
    }
]

async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    callbacks: Callbacks = None
) -> float:
    """Compute answer correctness score combining factuality and semantic similarity"""
    # Generate statements from answer and ground truth
    answer_statements = await generate_statements(llm, question, answer, callbacks)
    gt_statements = await generate_statements(llm, question, ground_truth, callbacks)

    # Calculate factuality score using statement classification
    factuality_score = await calculate_factuality(
        llm, question, answer_statements, gt_statements, callbacks, beta
    ) if weights[0] != 0 else 0.0
    # Calculate semantic similarity
    similarity_score = await calculate_semantic_similarity(
        embeddings, answer, ground_truth
    ) if weights[1] != 0 else 0.0

    # Combine scores using weighted average
    return float(np.average([factuality_score, similarity_score], weights=weights))

async def generate_statements(
    llm: BaseLanguageModel, question: str, answer:str, callbacks: Callbacks
) -> List[str]:
    """Generate concise factual statements from text"""
    handler = JSONHandler()
    prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    parsed = await handler.parse_with_fallbacks(response.content)
    # Ensure we always return List[str]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        # Prefer common keys if present
        for key in ["statements", "answers", "items", "list", "output", "result"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
        # Fallback: flatten dict values
        return [str(v) for v in parsed.values()]
    # Fallback to single string wrapped in list
    return [str(parsed)]

async def calculate_factuality(
    llm: BaseLanguageModel,
    question: str,
    answer_stmts: List[str],
    gt_stmts: List[str],
    callbacks: Callbacks,
    beta: float
) -> float:
    """Classify statements and calculate factuality F-beta score"""
    if not answer_stmts and not gt_stmts:
        return 1.0  # Perfect score if both empty

    # Prepare examples for prompt
    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in CORRECTNESS_EXAMPLES
    )

    # Generate classification
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        examples=examples,
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts)
    )
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    handler = JSONHandler()
    try:
        parsed = await handler.parse_with_fallbacks(response.content)
        if not isinstance(parsed, dict):
            return 0.0
        classification = ClassificationWithReason(**parsed)
        tp = len(classification.TP)
        fp = len(classification.FP)
        fn = len(classification.FN)
        return fbeta_score(tp, fp, fn, beta)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0  # Return minimum score on failure

async def calculate_semantic_similarity(
    embeddings: Embeddings, answer: str, ground_truth: str
) -> float:
    """Compute cosine similarity between answer and ground truth embeddings"""
    a_embed, gt_embed = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth)
    )
    cosine_sim = np.dot(a_embed, gt_embed) / (
        np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
    return (cosine_sim + 1) / 2  # Scale to [0, 1]