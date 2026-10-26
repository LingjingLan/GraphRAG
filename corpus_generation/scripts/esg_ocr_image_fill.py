"""Vision-LLM fill for KPI numbers missed by OCR on infographic pages.

Used by ``build_esg_ocr_corpus.py --image-fill``.
Renders PDF page images, detects sparse-digit pages, calls an OpenAI-compatible
vision model (DashScope Qwen-VL by default), merges statements into context.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "esg_kpi_image_fill_prompt.md"

_PAGE_SPLIT_RE = re.compile(r"(?m)^## Page (\d+)\s*$")
_DIGIT_RE = re.compile(r"\d")
_KPI_HINT_RE = re.compile(
    r"(leadership|trifr|emissions|waste|landfill|renewable|meals? donated|"
    r"community support|packaging|aboriginal|torres strait|scope\s*[12]|"
    r"reduction|improvement|%|percent)",
    re.IGNORECASE,
)
# Display-style KPI values (ignore bare years in footnotes)
_KPI_VALUE_RE = re.compile(
    r"(?<![\d.])(?:\$\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%"
    r"|(?<![\d.])\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*m\b"
    r"|(?<![\d.])\d{1,3}(?:\.\d+)?\s*m\b"
    r"|(?<![\d.])\d{1,3}(?:\.\d+)?\s*(?:tco2e|kt|mw|gwh)\b",
    re.IGNORECASE,
)


@dataclass
class PageBlock:
    page: int
    statements_raw: str
    text_raw: str
    full: str


def load_system_prompt() -> str:
    if PROMPT_PATH.is_file():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "Recover missing KPI numbers from the page image and return JSON with "
        "key statements (list of English sentences ending with (page N))."
    )


def parse_context_pages(context: str) -> Tuple[str, List[PageBlock]]:
    """Split context into header + page blocks."""
    m = _PAGE_SPLIT_RE.search(context)
    if not m:
        return context.strip(), []
    header = context[: m.start()].strip()
    parts = _PAGE_SPLIT_RE.split(context[m.start() :])
    # parts: ['', num, body, num, body, ...]
    pages: List[PageBlock] = []
    i = 1
    while i + 1 < len(parts):
        page_no = int(parts[i])
        body = parts[i + 1]
        stmt = ""
        text = body
        sm = re.search(r"(?ms)^### Statements\s*\n(.*?)(?=^### Text\s*$|\Z)", body)
        tm = re.search(r"(?ms)^### Text\s*\n(.*)\Z", body)
        if sm:
            stmt = sm.group(1).strip()
        if tm:
            text = tm.group(1).strip()
        pages.append(
            PageBlock(
                page=page_no,
                statements_raw=stmt,
                text_raw=text,
                full=f"## Page {page_no}\n{body.rstrip()}\n",
            )
        )
        i += 2
    return header, pages


def _main_body_without_footnotes(text: str) -> str:
    """Drop dense footnote blocks that inflate digit counts."""
    lines = (text or "").splitlines()
    cut = len(lines)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.match(r"^\d{1,2}[\.\)]\s+\S+", s) and i >= max(3, int(0.4 * len(lines))):
            cut = i
            break
        if re.match(r"^footnotes?\b", s, re.I):
            cut = i
            break
    return "\n".join(lines[:cut])


def page_needs_image_fill(page: PageBlock) -> bool:
    """Heuristic: KPI labels present but few display values → likely missed large KPIs."""
    text = _main_body_without_footnotes(page.text_raw or "")
    if len(text.strip()) < 40:
        return False
    has_hint = bool(_KPI_HINT_RE.search(text))
    kpi_values = _KPI_VALUE_RE.findall(text)
    stmts = (page.statements_raw or "").strip()
    stmts_none = (not stmts) or stmts.lower() in {"(none)", "none"}
    digits = len(_DIGIT_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z]", text))
    footnote_marks = len(
        re.findall(r"\$\s*\^\s*\{?\s*\d+\s*\}?\s*\$|\^\s*\{\s*\d+\s*\}|\^\d+", text)
    )

    # Infographic cards: many footnote markers / captions, but few recovered display KPIs
    if stmts_none and footnote_marks >= 3 and len(kpi_values) < max(3, footnote_marks // 2):
        return True
    if has_hint and stmts_none and len(kpi_values) <= 1:
        return True
    if has_hint and len(kpi_values) == 0 and letters >= 60:
        return True
    if has_hint and digits <= 12 and letters >= 80 and len(kpi_values) <= 1:
        return True
    return False


def render_pdf_page_png(pdf_path: Path, page_number: int, out_path: Path, *, dpi: int = 160) -> Path:
    import fitz  # PyMuPDF

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        idx = max(0, int(page_number) - 1)
        if idx >= len(doc):
            raise IndexError(f"page {page_number} out of range (pdf has {len(doc)} pages)")
        page = doc[idx]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        out_path.write_bytes(pix.tobytes("png"))
    finally:
        doc.close()
    return out_path


def _load_dotenv_llm() -> Dict[str, str]:
    """Best-effort load LLM_* from backend/config/.env if not already in os.environ."""
    env: Dict[str, str] = {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", "") or "",
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "") or "",
        "LLM_MODEL": os.getenv("LLM_MODEL", "") or "",
        "LLM_VISION_MODEL": os.getenv("LLM_VISION_MODEL", "") or "",
    }
    candidates = [
        Path(__file__).resolve().parents[1] / "config" / ".env",
        Path("/workspace/backend/config/.env"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in env and not env[k]:
                env[k] = v
        break
    if not env["LLM_VISION_MODEL"]:
        # DashScope OpenAI-compatible vision default
        env["LLM_VISION_MODEL"] = os.getenv("LLM_VISION_MODEL", "qwen-vl-max")
    return env


def call_vision_llm(
    *,
    system_prompt: str,
    user_text: str,
    image_path: Path,
    model: Optional[str] = None,
) -> str:
    import openai

    cfg = _load_dotenv_llm()
    api_key = cfg["LLM_API_KEY"]
    base_url = cfg["LLM_BASE_URL"] or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vision_model = model or cfg["LLM_VISION_MODEL"] or "qwen-vl-max"
    if not api_key:
        raise RuntimeError("LLM_API_KEY missing; cannot run image fill")

    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=vision_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_statements_json(raw: str) -> List[str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # try extract first {...}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            logger.warning("image-fill: LLM returned non-JSON: {}", text[:300])
            return []
        obj = json.loads(m.group(0))
    stmts = obj.get("statements") if isinstance(obj, dict) else None
    if not isinstance(stmts, list):
        return []
    out: List[str] = []
    seen = set()
    for s in stmts:
        line = re.sub(r"\s+", " ", str(s or "")).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def build_user_payload(
    *,
    company: str,
    year: str,
    sector: str,
    page: int,
    ocr_text: str,
    existing_statements: str,
) -> str:
    return (
        f"Company: {company}\n"
        f"Report Year: {year}\n"
        f"Sector: {sector}\n"
        f"Page: {page}\n\n"
        f"## Existing Statements\n{existing_statements or '(none)'}\n\n"
        f"## OCR Text (may miss large numbers)\n{ocr_text or '(empty)'}\n\n"
        "Read the attached page image and return JSON only."
    )


def merge_statements_into_page(page: PageBlock, new_statements: Sequence[str]) -> PageBlock:
    existing = []
    for ln in (page.statements_raw or "").splitlines():
        s = ln.strip()
        if not s or s.lower() in {"(none)", "none"}:
            continue
        existing.append(s)
    seen = {e.lower() for e in existing}
    merged = list(existing)
    for s in new_statements:
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        merged.append(s)
    stmt_body = "\n".join(merged) if merged else "(none)"
    full = (
        f"## Page {page.page}\n"
        f"### Statements\n{stmt_body}\n\n"
        f"### Text\n{page.text_raw.strip() if page.text_raw.strip() else '(empty)'}\n"
    )
    return PageBlock(
        page=page.page,
        statements_raw=stmt_body,
        text_raw=page.text_raw,
        full=full,
    )


def rebuild_context(header: str, pages: Sequence[PageBlock]) -> str:
    parts = [header.strip(), ""]
    for p in sorted(pages, key=lambda x: x.page):
        parts.append(p.full.rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def fill_job_context_from_images(
    *,
    pdf_path: Path,
    context: str,
    company: str,
    year: str,
    sector: str,
    image_dir: Path,
    pages_filter: Optional[Sequence[int]] = None,
    force_pages: bool = False,
    model: Optional[str] = None,
    dpi: int = 160,
    max_pages: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return updated context + stats."""
    header, pages = parse_context_pages(context)
    if not pages:
        return context, {"selected": 0, "filled": 0, "statements_added": 0}

    system_prompt = load_system_prompt()
    selected: List[PageBlock] = []
    for p in pages:
        if pages_filter is not None and p.page not in set(int(x) for x in pages_filter):
            continue
        if pages_filter is not None or force_pages or page_needs_image_fill(p):
            selected.append(p)
    if max_pages is not None:
        selected = selected[: max(0, int(max_pages))]

    stats: Dict[str, Any] = {
        "selected": len(selected),
        "filled": 0,
        "statements_added": 0,
        "pages": [],
        "model": model or _load_dotenv_llm().get("LLM_VISION_MODEL"),
    }
    if not selected:
        return context, stats

    by_page = {p.page: p for p in pages}
    for p in selected:
        img_path = image_dir / f"page_{p.page:04d}.png"
        try:
            render_pdf_page_png(pdf_path, p.page, img_path, dpi=dpi)
            user_text = build_user_payload(
                company=company,
                year=year,
                sector=sector,
                page=p.page,
                ocr_text=p.text_raw,
                existing_statements=p.statements_raw,
            )
            raw = call_vision_llm(
                system_prompt=system_prompt,
                user_text=user_text,
                image_path=img_path,
                model=model,
            )
            stmts = _parse_statements_json(raw)
            before = len(
                [
                    ln
                    for ln in (p.statements_raw or "").splitlines()
                    if ln.strip() and ln.strip().lower() not in {"(none)", "none"}
                ]
            )
            updated = merge_statements_into_page(p, stmts)
            after = len(
                [
                    ln
                    for ln in (updated.statements_raw or "").splitlines()
                    if ln.strip() and ln.strip().lower() not in {"(none)", "none"}
                ]
            )
            by_page[p.page] = updated
            stats["filled"] += 1
            stats["statements_added"] += max(0, after - before)
            stats["pages"].append(
                {"page": p.page, "new_statements": stmts, "image": str(img_path)}
            )
            logger.info(
                "[image-fill] {} page={} +{} statements",
                pdf_path.name,
                p.page,
                max(0, after - before),
            )
        except Exception as exc:
            logger.exception("[image-fill] failed page={} pdf={}", p.page, pdf_path)
            stats.setdefault("errors", []).append({"page": p.page, "error": str(exc)})

    new_pages = [by_page[k] for k in sorted(by_page.keys())]
    return rebuild_context(header, new_pages), stats
