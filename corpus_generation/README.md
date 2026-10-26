# ESG corpus generation

```mermaid
flowchart LR
  A["ESG raw PDF reports<br/>(30 reports, multiple industries)"] --> B["PaddleOCR-VL<br/>layout parsing and text extraction"]
  B --> C["Structured context<br/>pages · tables to statements"]
  C --> D["Vision LLM KPI completion<br/>(Qwen-VL)"]
  D --> E["Manual review and revision"]
  E --> F["GraphRAG-Bench corpus<br/>esg_ocr.json"]
```

`build_esg_ocr_corpus.py` produces page-structured Markdown/context and aggregates `esg_ocr.json`. Raw PDFs, OCR outputs, rendered page images, and the resulting full corpus are deliberately excluded from Git.
