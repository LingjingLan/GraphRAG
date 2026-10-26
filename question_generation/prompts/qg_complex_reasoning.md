Question Generation: Complex Reasoning (ESG)

Instructions:
You are an expert in constructing reasoning-based datasets from ESG evidence graphs (open information-extraction triples from corporate sustainability reports).
Please generate {num} high-quality English reasoning questions from the following ESG context composed of multiple logical chains.
Each question must span multiple triplets and require reasoning across distant elements within a single chain or across multiple chains.

Generation Guidelines:

1. Generate "Type2" / Complex Reasoning questions for ESG:
   - Each question must require multi-hop reasoning across at least three triplets.
   - Prefer that the relevant triplets are non-adjacent (some logical distance between them), e.g. different report sections/pages or different ESG themes linked by shared entities.
   - You may use a single chain or combine triplets from multiple chains, as long as the reasoning path is clear and explicit.
   - Prefer ESG multi-hop patterns such as:
     * Company → initiative/agenda → outcome/goal
     * Company → stakeholders → material issue → emissions/metric impact
     * Target → reported metric → year/value → achievement check
     * Initiative → contributes to → metric → aligned with → standard

2. The answer must be uniquely supported by the input triplets.
   - Prefer facts grounded only in the provided triples (and optional accompanying report spans if given).
   - Do not invent numbers, years, company claims, or causal links not present in the triples.

3. Do not make up or infer anything that is not explicitly stated in the triplets.
   - No subjective ESG judgments or external knowledge.
   - No answering from general sustainability knowledge outside the input.

4. Answer field constraints (the "answer" IS the gold label for later QA scoring):
   - The "answer" field is the gold label. Downstream models will be judged by how closely they match this string.
   - Write ONE concise English sentence (≤ 20 words), ending with a period.
   - If the conclusion is numeric, use "<value> <metric-or-unit>" — never a bare number (bad: "0" / "0.").
   - Yes/No conclusions: "Yes." / "No." (optionally + short clarification).
   - Do not include: chain walkthroughs, page numbers, extra triples, or policy commentary.
   - Put supporting triples only in "evidence" / "evidence_triple", never inside "answer".

5. Output format (STRICT JSON only, no markdown fences):
{
  "questions": [
    {
      "question": string,
      "answer": string,
      "question_type": "Complex Reasoning",
      "evidence": string,
      "evidence_triple": string
    }
  ]
}

Notes for evidence fields:
- evidence: supporting report sentences/spans if available, joined by "; "; otherwise a short natural-language restatement of the triple chain facts (still without putting the reasoning dump into "answer")
- evidence_triple: at least three supporting open triples in the form "(head,relation,tail); (...)", in an order that reflects the reasoning path

Input ESG evidence (logical chains / triples):
{evidence_block}
