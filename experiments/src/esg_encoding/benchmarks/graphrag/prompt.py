"""GraphRAG-benchmark-only prompt template (do not reuse for other frameworks)."""

from __future__ import annotations

from typing import List, Optional


SYSTEM_PROMPT = """You are answering GraphRAG benchmark questions over an ESG report.
Output STRICT JSON only. No markdown fences, no prose outside JSON.

Required JSON object keys (in this order):
1) "reasoning" — non-empty array of hop objects (MUST appear BEFORE "answer")
2) "answer" — the final answer string, or "not_disclosed" if the report does not answer the question

Each hop object:
{
  "hop": <1-based integer>,
  "kind": "quote" | "compute" | "matrix",
  "text": "<string>",
  "loc": "<page/row-or-char-offset of FIRST occurrence, or null for compute, or \\"matrix\\" for matrix>"
}

Rules:
- For every quote hop, preferably copy the sentence/table-row text VERBATIM from the
  report (best effort). Avoid paraphrase / translation when possible, and cite the
  first place a matching span appears. Near-quotes are acceptable if an exact span
  cannot be copied cleanly from tables/OCR.
- kind:"compute" hops are pure arithmetic over numbers introduced by earlier quote hops;
  loc must be null; every operand should appear in some earlier quote hop.
- At least one quote hop is required for answerable questions.
- If answer is "not_disclosed": include quote hops citing any distractor evidence found;
  the final hop may be {"kind":"matrix","text":"<code> not disclosed","loc":"matrix"}.
- Field name must be "reasoning" (not "reasoning_chain").
"""


FEW_SHOT_USER = """Question id: T0-demo
Question: How many vehicles were recalled per 1,000 vehicles sold in 2024?
Expected unit: Number
Evidence segments:
[P003_S001 - Page 3]
The company recalled 426,536 vehicles in 2024.
[P004_S002 - Page 4]
Total vehicles sold in 2024: 2,451,356.
"""


FEW_SHOT_ASSISTANT = """{
  "reasoning": [
    {
      "hop": 1,
      "kind": "quote",
      "text": "The company recalled 426,536 vehicles in 2024.",
      "loc": "P003_S001"
    },
    {
      "hop": 2,
      "kind": "quote",
      "text": "Total vehicles sold in 2024: 2,451,356.",
      "loc": "P004_S002"
    },
    {
      "hop": 3,
      "kind": "compute",
      "text": "426536 / 2451356 * 1000 = 174.0",
      "loc": null
    }
  ],
  "answer": "174.0"
}"""


def build_user_prompt(
    *,
    question_id: str,
    question: str,
    unit: Optional[str],
    code: Optional[str],
    evidence_blocks: List[str],
    feedback: Optional[str] = None,
) -> str:
    parts = [
        f"Question id: {question_id}",
        f"Question: {question}",
    ]
    if code:
        parts.append(f"Metric code: {code}")
    if unit:
        parts.append(f"Expected unit: {unit}")
    parts.append("Evidence segments:")
    if evidence_blocks:
        parts.extend(evidence_blocks)
    else:
        parts.append("(no retrieval hits — answer not_disclosed if unsupported)")
    if feedback:
        parts.append("")
        parts.append("Previous output failed validation:")
        parts.append(feedback)
        parts.append(
            "Prefer VERBATIM quotes from the report when possible; output strict JSON with `reasoning` before `answer`."
        )
    return "\n".join(parts)


def build_messages(
    *,
    question_id: str,
    question: str,
    unit: Optional[str],
    code: Optional[str],
    evidence_blocks: List[str],
    feedback: Optional[str] = None,
) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {
            "role": "user",
            "content": build_user_prompt(
                question_id=question_id,
                question=question,
                unit=unit,
                code=code,
                evidence_blocks=evidence_blocks,
                feedback=feedback,
            ),
        },
    ]
