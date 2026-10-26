Question Generation: Fact Retrieval (ESG)

Instructions:
You are a professional dataset constructor for corporate ESG / sustainability reports.
Generate {num} factual English questions from the provided ESG evidence (report text and/or open information-extraction triples). Follow the rules below.

Generation Guidelines:

1. Focus on explicit ESG information only, such as:
   - Reported metrics and values (e.g. Scope 1/2/3 emissions, renewable electricity share, TRIFR, water use, waste diverted)
   - Targets / commitments and target years (e.g. net-zero by 2040, 100% renewable by 2025)
   - Initiatives, programs, and actions explicitly named in the report
   - Standards, frameworks, and assurance (e.g. GRI, TCFD, ISSB, CDP, limited assurance)
   - Company / subsidiary / site / geography mentioned with a concrete fact
   - Year-over-year changes that are explicitly stated (increased/decreased from X to Y)

2. Basic rules:
   - Each question must have a unique, accurate answer that is directly supported by the provided evidence
   - Prefer facts grounded in triples or explicit sentences; do not invent numbers, years, or claims
   - Cover diverse ESG pillars when possible (E / S / G), without forcing coverage that the evidence does not support
   - Avoid redundant questions that ask for the same fact with different wording
   - Answer field constraints (the "answer" IS the gold label for later QA scoring):
     * The "answer" field is the gold label. Downstream models will be judged by how closely they match this string.
     * Use ONE short English phrase or sentence (≤ 20 words), ending with a period.
     * Canonical numeric form: "<value> <metric-or-unit>" — never a bare number.
       - Good: "0 work-related fatalities."; "42.5% women in leadership."; "13 tonnes of plastic."
       - Bad: "0"; "0."; "42.5"
     * Yes/No: "Yes." / "No." (optionally + ≤8-word clarification).
     * Dates/deadlines: keep report form, e.g. "FY25."; "By the end of FY25."; "2024."
     * Do not include: chain walkthroughs, page numbers, triples, alternatives, or policy commentary.
     * Put supporting triples only in "evidence" / "evidence_triple", never inside "answer".
     * Prefer non-trivial facts; avoid flooding a batch with many zero-value look-ups.

3. Prohibited content:
   - Subjective interpretations or opinions (e.g. “is the company green enough?”)
   - Implied / non-explicit information not stated in the evidence
   - Cross-document or external knowledge not present in the provided ESG report/triples
   - Questions whose answer cannot be uniquely determined from the evidence

4. Output format (STRICT JSON only, no markdown fences):
{
  "questions": [
    {
      "question": string,
      "answer": string,
      "question_type": "Fact Retrieval",
      "evidence": string,
      "evidence_triple": string
    }
  ]
}

Notes for evidence fields:
- evidence: one or more supporting sentences/spans from the report, joined by "; "
- evidence_triple: supporting open triples in the form "(head,relation,tail); (...)", copied or lightly normalized from the input triples

Input ESG evidence:
{evidence_block}
