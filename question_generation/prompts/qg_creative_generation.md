Question Generation: Creative Generation (ESG)

Instructions:
You are a professional dataset constructor working on an ESG report understanding benchmark.
You are tasked with generating creative expression questions, which require imaginative re-narration grounded in factual ESG evidence (open information-extraction triples and optional report spans).
Please generate {num} creative-style English questions in the following categories:
- Role-playing narration (first-person): a stakeholder or company actor retells ESG-related facts from their point of view
- Textual rewriting (in diary/memo or news/press-release format): the same ESG facts are re-expressed in another writing style

Generation Guidelines:

1. Role-playing narration questions:
   - Ask the model/user to imagine being an ESG-relevant actor and retelling or expressing the evidence
   - Allowed roles only if supported by the triples/entities, e.g.:
     * company sustainability officer / CEO spokesperson
     * employee / supplier / customer / community stakeholder
     * investor or assurance provider mentioned in the evidence
   - Use explicit entities, metrics, targets, initiatives, and events from the triplets
   - Ground the narrative strictly in observed facts; style may be first-person, but content must not invent claims

2. Text rewriting questions:
   - Ask for retelling the ESG evidence in diary/internal-memo form (subjective voice) or news/press-release form (objective voice)
   - Diary/memo: may reflect a stakeholder or officer perspective, still fact-bound
   - News/press-release: impersonal tone, factually accurate to the triples
   - Do not turn the task into free sustainability opinion writing

3. Evidence breadth:
   - Each question must be supported by at least 5 distinct triplets as evidence
   - The question should synthesize and depend on a broad scope of factual ESG context
   - Do not generate shallow or narrowly grounded questions (single metric / single edge)

4. Faithfulness constraints:
   - Do NOT invent numbers, targets, initiatives, standards, or causal links absent from the input triples
   - Do NOT require external ESG knowledge beyond the provided evidence
   - Creative style is allowed; creative facts are not

5. Answer field constraints (the "answer" IS the gold label for later QA scoring):
   - The "answer" field is the gold label: a short creative passage that correctly reflects the evidence
   - Prefer 2–5 concise English sentences (about 40–120 words), not an essay
   - All numbers/metrics in the passage must keep unit or metric name (never bare "0" / "42.5" alone)
   - Do not include: chain walkthroughs, page numbers, raw triple lists, or policy commentary outside the requested style
   - Put supporting triples only in "evidence" / "evidence_triple", never inside "answer"

6. Output format (STRICT JSON only, no markdown fences):
{
  "questions": [
    {
      "question": string,
      "answer": string,
      "question_type": "Creative Generation",
      "creative_form": "role_play" | "diary_memo" | "news_press",
      "evidence": string,
      "evidence_triple": string
    }
  ]
}

Notes for evidence fields:
- evidence: supporting report sentences/spans if available, joined by "; "
- evidence_triple: at least five supporting open triples in the form "(head,relation,tail); (...)"

Input ESG evidence (logical chains / triples):
{evidence_block}
