import json
import numpy as np
from typing import List, Dict, Optional, Union
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler

STATEMENT_GENERATOR_PROMPT = """
You are evaluating ESG / sustainability QA answers grounded in corporate report context.
Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Preserve numbers, units, years, and metric names exactly. Format the outputs in JSON.

Example Input:
Question: What progress did BHP make on water stewardship in FY2024?
Answer: It reduced freshwater withdrawal intensity at operated assets. It also completed catchment-risk assessments at priority sites and published a new water target for 2030.

Example Output:
["BHP reduced freshwater withdrawal intensity at operated assets.", "BHP completed catchment-risk assessments at priority sites.", "BHP published a new water target for 2030."]

Input Text:
Question:{question}
Answer: {answer}

Generated Statements:
"""

FAITHFULNESS_EXAMPLES = [
    {
        "input": {
            "context": "In FY2024, Telstra reported Scope 1 and 2 emissions of 148 ktCO2e, down 18% from FY2023. Telstra purchased renewable energy certificates covering 100% of Australian electricity consumption. Employee engagement on climate training rose to 92%.",
            "statements": [
                "Telstra's Scope 1 and 2 emissions were 200 ktCO2e in FY2024.",
                "Telstra covered 100% of Australian electricity consumption with renewable energy certificates.",
                "Telstra reduced Scope 1 and 2 emissions versus FY2023.",
                "Telstra achieved net-zero Scope 3 emissions in FY2024.",
            ],
        },
        "output": [
            {"statement": "Telstra's Scope 1 and 2 emissions were 200 ktCO2e in FY2024.", "reason": "The context states 148 ktCO2e, not 200 ktCO2e.", "verdict": 0},
            {"statement": "Telstra covered 100% of Australian electricity consumption with renewable energy certificates.", "reason": "Directly supported by the context about renewable energy certificates.", "verdict": 1},
            {"statement": "Telstra reduced Scope 1 and 2 emissions versus FY2023.", "reason": "The context says emissions were down 18% from FY2023.", "verdict": 1},
            {"statement": "Telstra achieved net-zero Scope 3 emissions in FY2024.", "reason": "Scope 3 net-zero is not mentioned in the context.", "verdict": 0},
        ],
    },
    {
        "input": {
            "context": "Woolworths Group discloses board oversight of sustainability through the Sustainability Committee, which met four times in FY2024.",
            "statements": [
                "Woolworths Group Scope 3 emissions fell by 30% in FY2024.",
            ],
        },
        "output": [
            {"statement": "Woolworths Group Scope 3 emissions fell by 30% in FY2024.", "reason": "The context only discusses board sustainability oversight, not Scope 3 emissions performance.", "verdict": 0},
        ],
    },
]

FAITHFULNESS_EVALUATION_PROMPT = """
You are judging faithfulness of ESG answer statements against retrieved sustainability-report context.
For each statement return verdict 1 if it can be directly inferred from the context, else 0.
Do not use outside knowledge about the company. Wrong numbers, years, scopes, or unsupported targets must be verdict 0.

Examples:
{examples}

Current Analysis:
Context: {context}
Statements: {statements}
"""

async def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate faithfulness score (0.0-1.0) by measuring what percentage of 
    answer statements are supported by the context.
    """
    # Step 1: Generate atomic statements from answer
    statements = await _generate_statements(
        question, answer, llm, callbacks, max_retries
    )
    
    # Handle edge cases
    if not statements:
        return 1.0 if not answer.strip() else np.nan
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0  # No context means no support
    
    # Step 2: Evaluate statement faithfulness
    verdicts = await _evaluate_statements(
        statements, context_str, llm, callbacks, max_retries
    )
    
    # Calculate faithfulness score
    if verdicts:
        supported = [v["verdict"] for v in verdicts]
        return sum(supported) / len(supported)
    return np.nan

async def _generate_statements(
    question: str,
    answer: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[str]:
    """
    Break down the answer into atomic statements using LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = STATEMENT_GENERATOR_PROMPT.format(
        question=question[:500],
        answer=answer[:3000]
    )

    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
            return []
        except Exception:
            continue

    return []


async def _evaluate_statements(
    statements: List[str],
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Evaluate which statements are supported by the context using LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    # Prepare examples for prompt
    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in FAITHFULNESS_EXAMPLES
    )

    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        examples=examples,
        context=context[:10000],
        statements=json.dumps(statements)[:5000]
    )

    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _validate_verdicts(
                parsed if isinstance(parsed, list) else [parsed]
            )        
        except Exception:
            continue

    return []


def _validate_verdicts(verdicts: Union[List, Dict]) -> List[Dict]:
    """
    Ensure verdicts have required fields and correct types.
    Accepts either a list of dicts or a single dict.
    """
    if isinstance(verdicts, dict):
        verdicts = [verdicts]
    elif not isinstance(verdicts, list):
        return []

    valid = []
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        try:
            if (
                "statement" in item
                and "verdict" in item and item["verdict"] in {0, 1}
                and "reason" in item
            ):
                valid.append({
                    "statement": str(item["statement"]),
                    "verdict": int(item["verdict"]),
                    "reason": str(item["reason"])
                })
        except (TypeError, ValueError):
            continue
    return valid
