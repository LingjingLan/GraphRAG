from typing import List, Dict

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel

from Evaluation.metrics.utils import JSONHandler

EVIDENCE_RECALL_PROMPT = """
### Task
You are measuring evidence recall for ESG report retrieval.
Given gold Evidence snippets and retrieved Context from corporate sustainability disclosures, decide for each evidence whether it can be attributed to the Context.
Attribution requires the same substantive ESG fact (company/entity, metric, period, and value when present); paraphrase is allowed.

Respond ONLY with a JSON object containing a "classifications" list. Each item should include:
- "statement": the exact evidence string
- "reason": a brief explanation (1 sentence)
- "attributed": 1 if the evidence can be attributed to the Context, otherwise 0

### Example
Input:
Context: "In FY2024 Coles sourced 100% of electricity from renewable sources and reduced operational emissions by 12% versus FY2023."
Evidence: ["Coles sourced 100% renewable electricity in FY2024", "Coles board met 8 times in FY2024"]

Output:
{{
  "classifications": [
    {{
      "statement": "Coles sourced 100% renewable electricity in FY2024",
      "reason": "Context states 100% renewable electricity in FY2024",
      "attributed": 1
    }},
    {{
      "statement": "Coles board met 8 times in FY2024",
      "reason": "Board meeting frequency is not in the retrieved context",
      "attributed": 0
    }}
  ]
}}

### Actual Input
Context: "{context}"

Evidence: {evidence}

Question: "{question}" (for reference only)

### Your Response:
"""

async def compute_evidence_recall(
    question: str,
    contexts: List[str],
    reference_evidence: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference evidence are supported by the context.
    """
    # Handle edge cases
    if isinstance(contexts, list):
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")

    # No context means no attribution
    if not context_str.strip():
        return 0.0

    prompt = EVIDENCE_RECALL_PROMPT.format(
        question=question,
        context=context_str[:20000],  # Truncate long contexts # TODO : smarter truncation
        evidence=reference_evidence,  # Truncate long answers TODO how?
    )

    # Get LLM classification with retries
    classifications = await _get_classifications(
        prompt, llm, callbacks, max_retries
    )

    # Calculate recall score
    if classifications:
        attributed = [c["attributed"] for c in classifications]
        return sum(attributed) / len(attributed)
    return np.nan  # Return NaN if no valid classifications

async def _get_classifications(
    prompt: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Get valid classifications from LLM with retries using RobustJSONHandler.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            classifications = await parser.parse_with_fallbacks(
                response.content,
                key="classifications",
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _validate_classifications(classifications)
        except Exception:
            continue
    return []

def _validate_classifications(classifications: List) -> List[Dict]:
    """
    Ensure classifications have required fields and proper types.
    """
    valid = []
    for item in classifications:
        try:
            if (
                isinstance(item, dict)
                and "statement" in item
                and "reason" in item
                and "attributed" in item
                and item["attributed"] in {0, 1}
            ):
                valid.append({
                    "statement": str(item["statement"]),
                    "reason": str(item["reason"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError):
            continue
    return valid