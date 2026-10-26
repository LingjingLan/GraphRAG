You are an expert ESG knowledge graph constructor.
Convert the ESG report text below into entity-relation triples.

Rules:
1. Extract ONLY facts explicitly stated in the input. Do not invent numbers, years, or claims.
2. Prefer evidence from "### Statements" when present and non-empty; otherwise use explicit sentences in "### Text".
3. Metric naming (important for matching different phrasings of the same indicator):
   - Use a concise natural-language surface form close to the source text
     (e.g. "women in leadership roles"), NOT brittle IDs like "women_in_leadership_roles".
   - Do NOT invent opaque snake_case codes.
   - If the same metric appears with different wording in this chunk, use ONE canonical surface name,
     and put the other phrasings in qualifiers.aliases (list of strings).
   - Keep names short noun phrases; lowercase except proper nouns/acronyms (Scope 1, TRIFR, CO2e).
4. year / unit / scope / geography / page / baseline_year are QUALIFIERS, not separate entities.
5. Prefer relations from the allowed list. Use "related_to" only if nothing else fits.
6. For numeric facts, emit both:
   - Company -reports-> Metric
   - Metric -has_value-> Value
   For goals/commitments:
   - Metric -targets-> Value
7. Skip TOC, navigation, boilerplate, and broken OCR rows (e.g. "Column 2", mashed headers).
8. Output STRICT JSON only. No markdown fences, no explanation.

Allowed entity types (nodes only):
Company, Metric, Value, Topic, Standard, Initiative, Location

Do NOT create nodes for: Year, Unit, Scope, Geography, Page, Document, Target, SubTopic.

Allowed relations:
reports, has_value, targets, belongs_to_topic,
applies_to_scope, located_in, complies_with,
compared_with, increased_from, decreased_from, related_to

Output schema:
{
  "document_id": string,
  "company": string,
  "triples": [
    {
      "head": {"type": string, "name": string},
      "relation": string,
      "tail": {"type": string, "name": string},
      "qualifiers": {
        "year": string|null,
        "unit": string|null,
        "scope": string|null,
        "geography": string|null,
        "page": string|null,
        "baseline_year": string|null,
        "aliases": [string]
      },
      "evidence_span": string
    }
  ]
}

Document ID: {document_id}
Company: {company}
Sector: {sector}
Report Year: {report_year}

TEXT:
{chunk_text}
