#!/usr/bin/env python3
"""Build GraphRAG-Bench-style corpus JSON from ESG PDFs (OCR/layout only).

Pipeline (step 1 only — no triples / graph / QA / embeddings):
  1) PDF → Markdown via existing PaddleOCR-VL entry:
       esg_encoding.content_extractor.ContentExtractor.extract_pdf
       + ContentExtractor.save_markdown
  2) Tables → natural-language Statements; build page-structured context
  3) Aggregate Datasets/Corpus/esg_ocr.json
       schema: [{ "corpus_name": "...", "context": "..." }, ...]

Per-page context shape:
  ## Page {N}
  ### Statements
  In {year}, {company}'s {metric} was {value} (page {N}).
  ### Text
  {full page text, not truncated}

By default markdown tables are NOT kept in context (use --keep-table-markdown to retain).

Optional vision fill (``--image-fill``):
  Detect pages where OCR likely missed large KPI numbers (infographic cards),
  render the PDF page image, call a vision LLM (see prompts/esg_kpi_image_fill_prompt.md),
  and merge recovered Statements into ``*.context.md`` / ``esg_ocr.json``.

Outputs:
  ESG_KG/output/ocr_md/{corpus_name}.md           # audit OCR markdown (may contain tables)
  ESG_KG/output/ocr_md/{corpus_name}.context.md   # GraphRAG-Bench context body
  ESG_KG/output/ocr_page_images/{corpus_name}/    # page PNGs used for vision fill
  Datasets/Corpus/esg_ocr.json
  ESG_KG/output/ocr_failures.jsonl

Examples:
  python3 scripts/build_esg_ocr_corpus.py --pdf coles --force
  python3 scripts/build_esg_ocr_corpus.py --pdf coles --image-fill-only --image-fill-pages 4
  python3 scripts/build_esg_ocr_corpus.py --aggregate-only --aggregate-all-discovered --image-fill
  python3 scripts/build_esg_ocr_corpus.py --limit 1
  python3 scripts/build_esg_ocr_corpus.py
  python3 scripts/build_esg_ocr_corpus.py --aggregate-only --aggregate-all-discovered
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _detect_newesg_root() -> Path:
    """Resolve newESG/ whether running on host or inside the backend container."""
    here = Path(__file__).resolve()
    euler = here.parents[2]
    sibling = euler.parent
    if (sibling / "ESG_report").is_dir():
        return sibling
    if Path("/workspace/ESG_report").is_dir():
        return Path("/workspace")
    return sibling


NEWESG = _detect_newesg_root()
DEFAULT_PDF_ROOT = NEWESG / "ESG_report"
DEFAULT_MD_DIR = NEWESG / "ESG_KG" / "output" / "ocr_md"
DEFAULT_OUT_JSON = NEWESG / "Datasets" / "Corpus" / "esg_ocr.json"
DEFAULT_FAILURES = NEWESG / "ESG_KG" / "output" / "ocr_failures.jsonl"
DEFAULT_PAGE_IMAGE_DIR = NEWESG / "ESG_KG" / "output" / "ocr_page_images"

_PAGE_HEADING_RE = re.compile(
    r"^(?:##\s*)?(?:第\s*(\d+)\s*页|Page\s+(\d+))\s*$",
    re.IGNORECASE,
)
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")
_VALUE_RE = re.compile(
    r"^[\d,.\s%¢$€£¥₹+\-–—()×xX/~≈<>≤≥]+$"
    r"|^[\d,.]+\s*(?:t|kt|mt|kg|g|lb|%|ppm|mwh|gwh|kwh|gj|tj|co2e?|tco2e|"
    r"hours?|days?|people|employees|sites?|facilities)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdfJob:
    pdf_path: Path
    corpus_name: str
    company: str
    year: str
    sector: str
    md_path: Path

    @property
    def context_path(self) -> Path:
        return self.md_path.with_suffix(".context.md")


def _normalize_token(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\s&/\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def infer_meta(pdf_path: Path) -> Tuple[str, str, str, str]:
    """Infer corpus_name / company / year / sector from path."""
    sector = pdf_path.parent.name.strip() or "Unknown"
    stem = re.sub(r"\s+", " ", pdf_path.stem).strip()
    m = re.search(r"(?:[_\s-])?((?:19|20)\d{2})$", stem)
    if m:
        year = m.group(1)
        company_raw = stem[: m.start()].rstrip(" _-")
    else:
        year = "unknown"
        company_raw = stem

    company_display = re.sub(r"[_\s]+", " ", company_raw).strip() or stem
    company_token = _normalize_token(company_raw) or "Unknown"
    corpus_name = f"{company_token}-{year}"
    return corpus_name, company_display, year, sector


def discover_pdfs(pdf_root: Path) -> List[Path]:
    if not pdf_root.is_dir():
        raise FileNotFoundError(f"PDF root not found: {pdf_root}")
    return sorted(pdf_root.rglob("*.pdf"), key=lambda p: str(p).lower())


def filter_jobs(
    pdfs: Sequence[Path],
    md_dir: Path,
    *,
    limit: Optional[int],
    pdf_filter: Optional[str],
) -> List[PdfJob]:
    needle = (pdf_filter or "").strip().lower()
    jobs: List[PdfJob] = []
    for pdf in pdfs:
        if needle and needle not in pdf.name.lower() and needle not in str(pdf).lower():
            continue
        corpus_name, company, year, sector = infer_meta(pdf)
        jobs.append(
            PdfJob(
                pdf_path=pdf,
                corpus_name=corpus_name,
                company=company,
                year=year,
                sector=sector,
                md_path=md_dir / f"{corpus_name}.md",
            )
        )
        if limit is not None and len(jobs) >= max(0, int(limit)):
            break
    return jobs


# ---------------------------------------------------------------------------
# Table → Statements
# ---------------------------------------------------------------------------


def _clean_cell(text: Any) -> str:
    s = unescape(str(text or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_value(text: str) -> bool:
    t = _clean_cell(text)
    if not t or len(t) > 80:
        return False
    if _VALUE_RE.match(t):
        return True
    # Short numeric-ish tokens
    return bool(re.search(r"\d", t)) and len(t.split()) <= 6


def _is_separator_row(cells: Sequence[str]) -> bool:
    if not cells:
        return True
    return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c)


def parse_markdown_table_rows(table_md: str) -> List[List[str]]:
    """Parse pipe or HTML tables into rows of cells."""
    raw = table_md or ""
    if re.search(r"<\s*(table|tr|td|th)\b", raw, flags=re.IGNORECASE):
        try:
            from html.parser import HTMLParser

            class _P(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self.rows: List[List[str]] = []
                    self._row: List[str] = []
                    self._cell: List[str] = []
                    self._in_cell = False

                def handle_starttag(self, tag, attrs):
                    t = tag.lower()
                    if t == "tr":
                        self._row = []
                    elif t in {"td", "th"}:
                        self._cell = []
                        self._in_cell = True

                def handle_endtag(self, tag):
                    t = tag.lower()
                    if t in {"td", "th"} and self._in_cell:
                        self._row.append(_clean_cell("".join(self._cell)))
                        self._in_cell = False
                    elif t == "tr" and self._row:
                        self.rows.append(self._row)
                        self._row = []

                def handle_data(self, data):
                    if self._in_cell:
                        self._cell.append(data)

            p = _P()
            p.feed(raw)
            p.close()
            rows = [r for r in p.rows if any(c.strip() for c in r)]
            if rows:
                return rows
        except Exception:
            pass

    rows: List[List[str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        if _MD_TABLE_SEP_RE.match(stripped):
            continue
        cells = [_clean_cell(c) for c in stripped.strip("|").split("|")]
        if not cells or _is_separator_row(cells):
            continue
        rows.append(cells)
    return rows


def statements_from_table(
    table_md: str,
    *,
    company: str,
    year: str,
    page: int,
    table_title: str = "",
) -> List[str]:
    """Rewrite a markdown/HTML table into natural-language statements."""
    rows = parse_markdown_table_rows(table_md)
    if len(rows) < 2:
        # Single-row / malformed: one descriptive statement if any text remains.
        blob = _clean_cell(re.sub(r"[|:\-]+", " ", table_md))
        if not blob:
            return []
        topic = table_title or "tabular disclosure"
        return [
            f"In {year}, regarding {topic}, {company} reports: {blob} (page {page})."
        ]

    headers = [_clean_cell(h) or f"Column {i + 1}" for i, h in enumerate(rows[0])]
    data_rows = rows[1:]
    statements: List[str] = []
    seen = set()

    for row in data_rows:
        width = max(len(headers), len(row))
        cells = [_clean_cell(c) for c in row] + [""] * (width - len(row))
        row_label = cells[0] if cells else ""
        metric_base = row_label or table_title or "metric"

        # If first column looks like a year and others are labels/values, swap roles.
        emitted = False
        for idx in range(1, width):
            value = cells[idx] if idx < len(cells) else ""
            if not value:
                continue
            col = headers[idx] if idx < len(headers) else f"Column {idx + 1}"
            # Skip identity cells
            if value.lower() == metric_base.lower() or value.lower() == col.lower():
                continue

            year_hint = year
            col_year = re.search(r"((?:19|20)\d{2})", col)
            if col_year:
                year_hint = col_year.group(1)

            if _looks_like_value(value):
                metric = metric_base
                if col and not re.fullmatch(r"(?:19|20)\d{2}", col.strip()):
                    # Non-year column header adds specificity
                    if col.lower() not in metric.lower():
                        metric = f"{metric_base} ({col})" if metric_base else col
                stmt = (
                    f"In {year_hint}, {company}'s {metric} was {value} (page {page})."
                )
            elif table_title:
                stmt = (
                    f"In {year_hint}, regarding {table_title}, {company} reports: "
                    f"{metric_base}: {col} = {value} (page {page})."
                )
            else:
                stmt = (
                    f"In {year_hint}, {company} reports {value} "
                    f"for {metric_base} / {col} (page {page})."
                )

            key = stmt.lower()
            if key not in seen:
                seen.add(key)
                statements.append(stmt)
                emitted = True

        if not emitted and row_label:
            # Row with only label / narrative cells
            desc = "; ".join(c for c in cells[1:] if c) or row_label
            topic = table_title or row_label
            stmt = (
                f"In {year}, regarding {topic}, {company} reports: {desc} (page {page})."
            )
            key = stmt.lower()
            if key not in seen:
                seen.add(key)
                statements.append(stmt)

    return statements


def _strip_markdown_tables(text: str) -> str:
    """Remove pipe/HTML tables from a text block."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    in_pipe = False
    in_html = False
    pipe_buf: List[str] = []

    def flush_pipe() -> None:
        nonlocal pipe_buf, in_pipe
        # Drop the table entirely
        pipe_buf = []
        in_pipe = False

    for line in lines:
        lower = line.lower()
        if "<table" in lower:
            in_html = True
            if "</table" in lower:
                in_html = False
            continue
        if in_html:
            if "</table" in lower:
                in_html = False
            continue

        stripped = line.strip()
        is_pipe = "|" in stripped and (
            stripped.startswith("|") or stripped.count("|") >= 2
        )
        if is_pipe or _MD_TABLE_SEP_RE.match(stripped or ""):
            in_pipe = True
            pipe_buf.append(line)
            continue
        if in_pipe:
            flush_pipe()
        out.append(line)

    if in_pipe:
        flush_pipe()

    # Collapse excess blank lines
    text_out = "\n".join(out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip()
    return text_out


def _page_num_from_heading(line: str) -> Optional[int]:
    m = _PAGE_HEADING_RE.match(line.strip())
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def pages_from_document(document: Any) -> Dict[int, Dict[str, List[str]]]:
    """Group extractor segments into per-page text + table markdown."""
    pages: Dict[int, Dict[str, List[str]]] = defaultdict(
        lambda: {"text": [], "tables": [], "titles": []}
    )
    segments = list(getattr(document, "segments", None) or [])
    segments.sort(
        key=lambda s: (
            int(getattr(s, "page_number", 1) or 1),
            float(getattr(s, "position_y", 0.0) or 0.0),
            float(getattr(s, "position_x", 0.0) or 0.0),
        )
    )
    for seg in segments:
        stype = str(getattr(seg, "segment_type", "text") or "text").lower()
        content = str(getattr(seg, "content", "") or "").strip()
        if not content:
            continue
        try:
            page = max(1, int(getattr(seg, "page_number", 1) or 1))
        except Exception:
            page = 1

        if stype in {"table_row", "table_cell", "link_anchor"}:
            continue
        if stype == "table":
            pages[page]["tables"].append(content)
            data = getattr(seg, "structured_data", None) or {}
            title = str(data.get("table_title") or "").strip()
            pages[page]["titles"].append(title)
        elif stype == "heading":
            pages[page]["text"].append(content)
            pages[page]["titles"].append(content)  # last heading may caption next table
        else:
            pages[page]["text"].append(content)
    return pages


def pages_from_markdown(markdown: str) -> Dict[int, Dict[str, List[str]]]:
    """Fallback: split audit markdown into pages + extract tables."""
    pages: Dict[int, Dict[str, List[str]]] = defaultdict(
        lambda: {"text": [], "tables": [], "titles": []}
    )
    current = 1
    buf: List[str] = []
    last_heading = ""

    def flush_buf() -> None:
        nonlocal buf, last_heading
        if not buf:
            return
        block = "\n".join(buf).strip()
        buf = []
        if not block:
            return
        # Detect embedded tables inside block
        if re.search(r"<\s*table\b", block, re.I) or (
            block.count("|") >= 4 and sum(1 for ln in block.splitlines() if "|" in ln) >= 2
        ):
            # Split mixed blocks: crude — if whole block is table, else strip tables to text
            rows = parse_markdown_table_rows(block)
            if len(rows) >= 2 and _looks_like_mostly_table(block):
                pages[current]["tables"].append(block)
                pages[current]["titles"].append(last_heading)
                return
        # Drop marker lines like **P001_S000**
        cleaned_lines = []
        for ln in block.splitlines():
            if re.fullmatch(r"\*\*P\d+_[\w]+\*\*", ln.strip()):
                continue
            if ln.strip() == "---":
                continue
            cleaned_lines.append(ln)
        text = "\n".join(cleaned_lines).strip()
        if not text:
            return
        # Extract pipe tables from text
        text_out, extracted = _extract_tables_from_text(text)
        for t in extracted:
            pages[current]["tables"].append(t)
            pages[current]["titles"].append(last_heading)
        if text_out.strip():
            pages[current]["text"].append(text_out.strip())
            if re.match(r"^#{1,6}\s+\S+", text_out.strip()) or (
                len(text_out.strip()) < 120 and "\n" not in text_out.strip()
            ):
                last_heading = re.sub(r"^#{1,6}\s+", "", text_out.strip())

    for line in (markdown or "").replace("\r\n", "\n").split("\n"):
        p = _page_num_from_heading(line)
        if p is not None:
            flush_buf()
            current = p
            continue
        if line.startswith("# ") and not line.startswith("## "):
            # Document title line from save_markdown
            continue
        buf.append(line)
    flush_buf()
    return pages


def _looks_like_mostly_table(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    pipe = sum(1 for ln in lines if "|" in ln or "<tr" in ln.lower())
    return pipe >= max(2, int(0.6 * len(lines)))


def _extract_tables_from_text(text: str) -> Tuple[str, List[str]]:
    """Pull contiguous pipe-table regions out of text; return (text_wo_tables, tables)."""
    lines = text.split("\n")
    kept: List[str] = []
    tables: List[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if "|" in stripped and (stripped.startswith("|") or stripped.count("|") >= 2):
            j = i
            chunk: List[str] = []
            while j < len(lines):
                s = lines[j].strip()
                if "|" in s or _MD_TABLE_SEP_RE.match(s or "") or not s:
                    chunk.append(lines[j])
                    j += 1
                    # stop after blank following table
                    if not s and chunk and "|" in "\n".join(chunk):
                        # peek ahead
                        if j < len(lines) and "|" not in lines[j]:
                            break
                    continue
                break
            table_text = "\n".join(chunk).strip()
            if len(parse_markdown_table_rows(table_text)) >= 2:
                tables.append(table_text)
                i = j
                continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept).strip(), tables


def build_page_context(
    pages: Dict[int, Dict[str, List[str]]],
    *,
    company: str,
    year: str,
    keep_table_markdown: bool = False,
) -> str:
    """Render ## Page / ### Statements / ### Text sections."""
    parts: List[str] = []
    for page in sorted(pages.keys()):
        bucket = pages[page]
        statements: List[str] = []
        titles = bucket.get("titles") or []
        for ti, table_md in enumerate(bucket.get("tables") or []):
            title = ""
            if ti < len(titles):
                title = titles[ti] or ""
            elif titles:
                title = titles[-1]
            statements.extend(
                statements_from_table(
                    table_md,
                    company=company,
                    year=year,
                    page=page,
                    table_title=title,
                )
            )

        text_chunks = []
        for t in bucket.get("text") or []:
            if keep_table_markdown:
                text_chunks.append(t)
            else:
                stripped = _strip_markdown_tables(t)
                if stripped:
                    text_chunks.append(stripped)
        if keep_table_markdown:
            for table_md in bucket.get("tables") or []:
                text_chunks.append(table_md)

        text_body = "\n\n".join(c for c in text_chunks if c and c.strip()).strip()
        # Deduplicate blank
        text_body = re.sub(r"\n{3,}", "\n\n", text_body)

        parts.append(f"## Page {page}")
        parts.append("### Statements")
        if statements:
            parts.extend(statements)
        else:
            parts.append("(none)")
        parts.append("")
        parts.append("### Text")
        parts.append(text_body if text_body else "(empty)")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def build_context(
    job: PdfJob,
    *,
    document: Any = None,
    markdown: str = "",
    keep_table_markdown: bool = False,
) -> str:
    header = (
        f"Company: {job.company}\n"
        f"Report Year: {job.year}\n"
        f"Sector: {job.sector}\n"
    )
    if document is not None:
        pages = pages_from_document(document)
    else:
        pages = pages_from_markdown(markdown or "")
    body = build_page_context(
        pages,
        company=job.company,
        year=job.year,
        keep_table_markdown=keep_table_markdown,
    )
    return f"{header}\n{body}"


def append_failure(failures_path: Path, record: dict) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _import_extractor():
    backend_src = Path(__file__).resolve().parents[1] / "src"
    if str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))
    from esg_encoding.content_extractor import ContentExtractor  # type: ignore
    from esg_encoding.models import ProcessingConfig  # type: ignore

    return ContentExtractor, ProcessingConfig


def ocr_pdf_to_artifacts(
    job: PdfJob,
    *,
    keep_table_markdown: bool = False,
) -> str:
    """OCR → audit MD + context MD. Returns context string (with header)."""
    ContentExtractor, ProcessingConfig = _import_extractor()
    extractor = ContentExtractor(ProcessingConfig())
    document = extractor.extract_pdf(str(job.pdf_path))
    job.md_path.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_markdown(document, output_path=str(job.md_path))
    if not job.md_path.read_text(encoding="utf-8").strip():
        raw = str(getattr(document, "markdown_content", "") or "")
        job.md_path.write_text(raw, encoding="utf-8")

    context = build_context(
        job,
        document=document,
        keep_table_markdown=keep_table_markdown,
    )
    # Store body without re-reading later issues: full context including header
    job.context_path.write_text(context, encoding="utf-8")
    return context


def md_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def load_or_build_context(job: PdfJob, *, keep_table_markdown: bool) -> Optional[str]:
    """Prefer rebuilding from audit .md when flag requires tables; else use .context.md."""
    if keep_table_markdown and md_ready(job.md_path):
        context = build_context(
            job,
            markdown=job.md_path.read_text(encoding="utf-8"),
            keep_table_markdown=True,
        )
        job.context_path.write_text(context, encoding="utf-8")
        return context

    if md_ready(job.context_path):
        text = job.context_path.read_text(encoding="utf-8")
        if text.strip():
            return text

    if not md_ready(job.md_path):
        return None
    context = build_context(
        job,
        markdown=job.md_path.read_text(encoding="utf-8"),
        keep_table_markdown=False,
    )
    job.context_path.write_text(context, encoding="utf-8")
    return context


def aggregate_corpus(
    jobs: Iterable[PdfJob],
    out_json: Path,
    *,
    keep_table_markdown: bool = False,
    rebuild_context: bool = False,
) -> Tuple[int, int, int]:
    items = []
    ok = 0
    missing = 0
    total_chars = 0
    for job in jobs:
        if rebuild_context and md_ready(job.md_path):
            context = build_context(
                job,
                markdown=job.md_path.read_text(encoding="utf-8"),
                keep_table_markdown=keep_table_markdown,
            )
            job.context_path.write_text(context, encoding="utf-8")
        else:
            context = load_or_build_context(job, keep_table_markdown=keep_table_markdown)
        if not context:
            missing += 1
            continue
        items.append({"corpus_name": job.corpus_name, "context": context})
        ok += 1
        total_chars += len(context)

    items.sort(key=lambda x: x["corpus_name"].lower())
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ok, missing, total_chars


def apply_image_fill(
    jobs: Sequence[PdfJob],
    *,
    page_image_root: Path,
    pages_filter: Optional[Sequence[int]] = None,
    max_pages_per_doc: Optional[int] = None,
    vision_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run vision-LLM KPI fill on existing context files; rewrite context in place."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from esg_ocr_image_fill import fill_job_context_from_images

    summary: Dict[str, Any] = {
        "docs": 0,
        "pages_selected": 0,
        "pages_filled": 0,
        "statements_added": 0,
    }

    for job in jobs:
        if not md_ready(job.context_path):
            print(f"[image-fill] skip {job.corpus_name}: missing context")
            continue
        context = job.context_path.read_text(encoding="utf-8")
        img_dir = page_image_root / job.corpus_name
        new_ctx, stats = fill_job_context_from_images(
            pdf_path=job.pdf_path,
            context=context,
            company=job.company,
            year=job.year,
            sector=job.sector,
            image_dir=img_dir,
            pages_filter=pages_filter,
            force_pages=bool(pages_filter),
            model=vision_model,
            max_pages=max_pages_per_doc,
        )
        job.context_path.write_text(new_ctx, encoding="utf-8")
        summary["docs"] += 1
        summary["pages_selected"] += int(stats.get("selected") or 0)
        summary["pages_filled"] += int(stats.get("filled") or 0)
        summary["statements_added"] += int(stats.get("statements_added") or 0)
        print(
            f"[image-fill] {job.corpus_name}: selected={stats.get('selected')} "
            f"filled={stats.get('filled')} +stmts={stats.get('statements_added')}"
        )
        if stats.get("errors"):
            print(f"[image-fill] {job.corpus_name} errors: {stats['errors']}")
    return summary


def run(args: argparse.Namespace) -> int:
    pdf_root = Path(args.pdf_root)
    md_dir = Path(args.md_dir)
    out_json = Path(args.out_json)
    failures = Path(args.failures)
    page_image_root = Path(args.page_image_dir)
    md_dir.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
    keep_tables = bool(args.keep_table_markdown)
    image_fill = bool(args.image_fill or args.image_fill_only)
    pages_filter = None
    if args.image_fill_pages:
        pages_filter = [int(x) for x in str(args.image_fill_pages).replace(";", ",").split(",") if str(x).strip()]

    all_pdfs = discover_pdfs(pdf_root)
    jobs = filter_jobs(all_pdfs, md_dir, limit=args.limit, pdf_filter=args.pdf)
    if not jobs:
        print(f"No PDFs matched under {pdf_root} (filter={args.pdf!r}, limit={args.limit})")
        return 2

    print(f"PDF root : {pdf_root}")
    print(f"MD dir   : {md_dir}")
    print(f"Out JSON : {out_json}")
    print(f"Failures : {failures}")
    print(f"Tables   : {'keep markdown' if keep_tables else 'statements only (default)'}")
    print(f"ImageFill: {'ON' if image_fill else 'off'}")
    print(f"Matched  : {len(jobs)} / discovered {len(all_pdfs)}")
    for j in jobs:
        print(f"  - {j.corpus_name:30s} <- {j.pdf_path.relative_to(pdf_root)}")

    success = 0
    failed = 0
    skipped = 0

    if not args.aggregate_only and not args.image_fill_only:
        for idx, job in enumerate(jobs, 1):
            tag = f"[{idx}/{len(jobs)}] {job.corpus_name}"
            # Resume if both audit md + context exist
            if (
                md_ready(job.md_path)
                and md_ready(job.context_path)
                and not args.force
            ):
                print(f"{tag}: resume skip (exists {job.md_path.name} + context)")
                skipped += 1
                success += 1
                continue
            # If only md exists, rebuild context without re-OCR unless --force
            if md_ready(job.md_path) and not args.force:
                try:
                    ctx = build_context(
                        job,
                        markdown=job.md_path.read_text(encoding="utf-8"),
                        keep_table_markdown=keep_tables,
                    )
                    job.context_path.write_text(ctx, encoding="utf-8")
                    print(f"{tag}: rebuilt context from existing md ({len(ctx)} chars)")
                    skipped += 1
                    success += 1
                    continue
                except Exception as exc:
                    print(f"{tag}: context rebuild failed, will re-OCR: {exc}")

            t0 = time.perf_counter()
            try:
                print(f"{tag}: OCR start -> {job.pdf_path}")
                ctx = ocr_pdf_to_artifacts(job, keep_table_markdown=keep_tables)
                elapsed = time.perf_counter() - t0
                print(
                    f"{tag}: OK ({elapsed:.1f}s, md={job.md_path.stat().st_size}B, "
                    f"context_chars={len(ctx)})"
                )
                success += 1
            except Exception as exc:
                failed += 1
                elapsed = time.perf_counter() - t0
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "corpus_name": job.corpus_name,
                    "pdf": str(job.pdf_path),
                    "md": str(job.md_path),
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 3),
                }
                append_failure(failures, record)
                print(f"{tag}: FAIL {exc}")
                continue

    # Image-fill always respects --pdf / --limit filters (do not expand to all docs).
    if image_fill:
        apply_image_fill(
            jobs,
            page_image_root=page_image_root,
            pages_filter=pages_filter,
            max_pages_per_doc=args.image_fill_max_pages,
            vision_model=args.vision_model,
        )

    agg_jobs = jobs
    if args.aggregate_all_discovered:
        agg_jobs = filter_jobs(all_pdfs, md_dir, limit=None, pdf_filter=None)

    ok_items, missing_md, total_chars = aggregate_corpus(
        agg_jobs,
        out_json,
        keep_table_markdown=keep_tables,
        rebuild_context=bool(args.rebuild_context) and not image_fill,
    )

    print("----")
    print(f"OCR success (incl. resume skips): {success}")
    print(f"OCR failed                      : {failed}")
    print(f"OCR skipped (resume)            : {skipped}")
    print(f"JSON entries written            : {ok_items}")
    print(f"JSON missing md/context         : {missing_md}")
    print(f"Total context chars             : {total_chars}")
    print(f"Wrote                           : {out_json}")
    if failed:
        print(f"Failure log                     : {failures}")
    return 0 if failed == 0 or ok_items > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ESG PDF OCR → GraphRAG-Bench corpus JSON (OCR only; tables→statements)"
    )
    p.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    p.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    p.add_argument("--page-image-dir", type=Path, default=DEFAULT_PAGE_IMAGE_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pdf", type=str, default=None, help="Substring filter (e.g. coles)")
    p.add_argument("--force", action="store_true", help="Re-OCR even if artifacts exist")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--aggregate-all-discovered", action="store_true")
    p.add_argument(
        "--keep-table-markdown",
        action="store_true",
        help="Also keep original markdown tables under ### Text (default: statements only)",
    )
    p.add_argument(
        "--rebuild-context",
        action="store_true",
        help="Force rebuild .context.md from audit .md when aggregating",
    )
    p.add_argument(
        "--image-fill",
        action="store_true",
        help="After OCR/context, use vision LLM to recover missing KPI numbers from page images",
    )
    p.add_argument(
        "--image-fill-only",
        action="store_true",
        help="Skip OCR; only run vision fill on existing context then re-aggregate JSON",
    )
    p.add_argument(
        "--image-fill-pages",
        type=str,
        default=None,
        help="Comma-separated page numbers to force-fill (e.g. 4 or 4,11)",
    )
    p.add_argument(
        "--image-fill-max-pages",
        type=int,
        default=None,
        help="Cap vision calls per document",
    )
    p.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Override LLM_VISION_MODEL (default qwen-vl-max on DashScope)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
