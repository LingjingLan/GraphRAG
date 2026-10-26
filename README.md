# GraphRAG ESG benchmark

This repository is limited to the reproducible research pipeline:

1. `corpus_generation/` — ESG PDF corpus creation with PaddleOCR-VL, table-to-statement context construction, Vision-LLM KPI recovery, and human review.
2. `question_generation/` — generation of Fact Retrieval, Complex Reasoning, Contextual Summarize, and Creative Generation questions; plus Krippendorff’s α reliability measurement.
3. `experiments/` — comparative HippoRAG, LightRAG, HyperGraphRAG, dual-channel, and triple-QA experiments and evaluation.

The repository intentionally excludes the application frontend, user/upload/authentication flows, raw PDFs, full corpus outputs, indexes, model weights, generated results, and secrets.
