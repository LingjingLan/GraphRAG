# HippoRAG v1 (legacy)

Vendored from [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy) (`legacy` branch).

ESG offline pipeline does **not** run the original ColBERT/Contriever bash scripts directly.
Instead, `esg_encoding.retrieval.hipporag1` reuses:

- HippoRAG v1 OpenIE / query-NER prompts
- Fact graph + synonym edges + Personalized PageRank retrieval

with OpenAI `text-embedding-3-small` for synonymy/entity linking (aligned with LightRAG / HippoRAG2 / HyperGraphRAG).

Scripts:

- `backend/scripts/extract_hipporag1_triples.py`
- `backend/scripts/run_hipporag1_qa.py`
- `backend/scripts/run_pipeline_hipporag1_qa_eval.py`
- `backend/scripts/_smoke_hipporag1.py`
