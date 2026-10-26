Question Generation: Contextual Summarize (ESG)

Instructions:
You are a professional dataset constructor for corporate ESG / sustainability reports.
Your task is to generate summarization-style questions based on the following instructions.
Please generate {num} high-level English questions that summarize or synthesize information from the provided ESG logical chains (open information-extraction triples and optional report spans).

Generation Guidelines:

1. Questions must focus on summarizing patterns, themes, or cause-effect structures across the ESG logical chain, such as:
   - How a company's initiatives collectively support a sustainability goal or target
   - Material issues prioritized by stakeholders and how they link to metrics/outcomes
   - The overall emissions / waste / social-performance storyline in the provided chain
   - Governance or assurance arrangements that frame a set of disclosures
   - Cross-theme synthesis (e.g. packaging + waste + Scope 3) when explicitly supported

2. Each question should ideally cover the majority of the triplets in a chain, not just isolated facts.
   - Prefer questions whose gold answer requires integrating several linked triples
   - Avoid single-metric look-ups (those belong to Fact Retrieval)

3. You may combine information from a single chain or compare across chains when appropriate,
   as long as every claim remains explicitly supported by the input triples.

4. Avoid factual recall of a single edge (e.g. "What is X's Scope 1 value?").
   Instead emphasize:
   - disclosure / strategy progression across the chain
   - role of initiatives, stakeholders, or standards in the overall picture
   - explicitly stated cause-effect or contribution chains

5. Do NOT infer or invent unstated details. Use only explicit triplets (and optional provided report spans).
   - No subjective ESG judgments ("is the company sustainable enough?")
   - No external knowledge beyond the input

6. Do NOT generate questions that are semantically similar to each other within this batch,
   or to any existing questions provided in the input (if an existing list is present).

7. Answer field constraints (the "answer" IS the gold label for later QA scoring):
   - The "answer" field is the gold label. Downstream models will be judged by how closely they match this string.
   - Write ONE concise English sentence (≤ 25 words) that states the synthesized conclusion, ending with a period.
   - Keep concrete entities/metrics from the evidence; do not use bare numbers alone (bad: "0" / "0.").
   - Do not include: chain walkthroughs, page numbers, extra triples, or policy commentary.
   - Put supporting triples only in "evidence" / "evidence_triple", never inside "answer".

8. Output format (STRICT JSON only, no markdown fences):
{
  "questions": [
    {
      "question": string,
      "answer": string,
      "question_type": "Contextual Summarize",
      "evidence": string,
      "evidence_triple": string
    }
  ]
}

Notes for evidence fields:
- evidence: supporting report sentences/spans if available, joined by "; "
- evidence_triple: the set of open triples needed for the summary, in the form "(head,relation,tail); (...)", ideally covering most of the chain used

Input ESG evidence (logical chains / triples):
{evidence_block}

Optional existing questions to avoid duplicating (may be empty):
{existing_questions}
