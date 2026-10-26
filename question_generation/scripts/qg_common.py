"""Shared helpers for ESG question-generation pack building and LLM calls."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

QUESTION_TYPES = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)

DEFAULT_TARGETS = {
    "Fact Retrieval": 971,
    "Complex Reasoning": 610,
    "Contextual Summarize": 362,
    "Creative Generation": 67,
}

PROMPT_FILES = {
    "Fact Retrieval": "qg_fact_retrieval.md",
    "Complex Reasoning": "qg_complex_reasoning.md",
    "Contextual Summarize": "qg_contextual_summarize.md",
    "Creative Generation": "qg_creative_generation.md",
}


def detect_newesg_root() -> Path:
    here = Path(__file__).resolve()
    euler = here.parents[2]
    sibling = euler.parent
    if (sibling / "Datasets" / "Corpus").is_dir():
        return sibling
    if Path("/workspace/Datasets/Corpus").is_dir():
        return Path("/workspace")
    return sibling


def load_dotenv() -> Dict[str, str]:
    env = {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", "") or "",
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "") or "",
        "LLM_MODEL": os.getenv("LLM_MODEL", "") or "",
    }
    for path in (
        Path(__file__).resolve().parents[1] / "config" / ".env",
        Path("/workspace/backend/config/.env"),
    ):
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
    if not env["LLM_BASE_URL"]:
        env["LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not env["LLM_MODEL"]:
        env["LLM_MODEL"] = "qwen-vl-max"
    return env


def node_name(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("name") or "").strip()
    return str(node or "").strip()


def node_type(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        t = node.get("type")
        return str(t) if t else None
    return None


def normalize_triple(t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(t, dict):
        return None
    h = node_name(t.get("head"))
    ta = node_name(t.get("tail"))
    rel = str(t.get("relation") or "").strip()
    if not h or not ta or not rel:
        return None
    q = t.get("qualifiers") if isinstance(t.get("qualifiers"), dict) else {}
    return {
        "head": {"type": node_type(t.get("head")), "name": h},
        "relation": rel,
        "tail": {"type": node_type(t.get("tail")), "name": ta},
        "qualifiers": {
            "year": q.get("year"),
            "page": q.get("page"),
            "unit": q.get("unit"),
            "scope": q.get("scope"),
        },
        "evidence_span": (t.get("evidence_span") or "").strip(),
    }


def triple_key(t: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        t["head"]["name"].lower(),
        t["relation"].lower(),
        t["tail"]["name"].lower(),
    )


def format_triple_line(t: Dict[str, Any]) -> str:
    return f"({t['head']['name']},{t['relation']},{t['tail']['name']})"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def format_evidence_block(
    *,
    document_id: str,
    company: str,
    triples: Sequence[Dict[str, Any]],
    include_spans: bool = True,
    max_spans: int = 12,
) -> str:
    lines = [
        f"Document: {document_id}",
        f"Company: {company}",
        "",
        "Triples:",
    ]
    for t in triples:
        lines.append(format_triple_line(t))
    if include_spans:
        spans: List[str] = []
        seen = set()
        for t in triples:
            span = (t.get("evidence_span") or "").strip()
            if not span or span in seen:
                continue
            seen.add(span)
            spans.append(span)
            if len(spans) >= max_spans:
                break
        if spans:
            lines.append("")
            lines.append("Optional spans:")
            for s in spans:
                lines.append(f"- {s}")
    return "\n".join(lines)


def load_prompt(question_type: str) -> str:
    name = PROMPT_FILES[question_type]
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


def render_qg_prompt(
    template: str,
    *,
    num: int,
    evidence_block: str,
    existing_questions: str = "",
) -> str:
    return (
        template.replace("{num}", str(num))
        .replace("{evidence_block}", evidence_block)
        .replace("{existing_questions}", existing_questions or "(none)")
    )


def parse_questions_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    qs = obj.get("questions")
    if not isinstance(qs, list):
        obj["questions"] = []
    return obj


def call_llm(prompt: str, *, model: Optional[str] = None, temperature: float = 0.2) -> str:
    import openai

    cfg = load_dotenv()
    if not cfg["LLM_API_KEY"]:
        raise RuntimeError("LLM_API_KEY missing")
    client = openai.OpenAI(api_key=cfg["LLM_API_KEY"], base_url=cfg["LLM_BASE_URL"])
    try:
        resp = client.chat.completions.create(
            model=model or cfg["LLM_MODEL"],
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        msg = str(exc).lower()
        if any(
            k in msg
            for k in (
                "insufficient_quota",
                "exceeded your current quota",
                "billing_hard_limit",
                "no credits remaining",
                "payment required",
                "quota exceeded",
                "you have insufficient",
            )
        ):
            raise RuntimeError(f"TOKEN_OR_QUOTA_EXHAUSTED: {exc}") from exc
        raise
    return (resp.choices[0].message.content or "").strip()


def iter_report_files(by_report_dir: Path) -> List[Path]:
    return sorted(by_report_dir.glob("*_triples.json"))
