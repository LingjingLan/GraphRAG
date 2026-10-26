"""OpenAI OpenIE + query NER using HippoRAG v1 legacy prompts."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import prompts as P

_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}")


def processing_phrases(phrase: str) -> str:
    return re.sub("[^A-Za-z0-9 ]", " ", str(phrase).lower()).strip()


def _extract_json_dict(text: str) -> Any:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def _chat_json(
    client,
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    # gpt-5* / o-series prefer max_completion_tokens
    mlow = (model or "").lower()
    if mlow.startswith("gpt-5") or mlow.startswith("o1") or mlow.startswith("o3") or mlow.startswith("o4"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_json_dict(content)
    return data if isinstance(data, dict) else {}


def passage_ner(client, model: str, passage: str) -> List[str]:
    messages = [
        {"role": "system", "content": P.NER_SYSTEM},
        {
            "role": "user",
            "content": f"Paragraph:\n```\n{P.ONE_SHOT_PASSAGE}\n```",
        },
        {"role": "assistant", "content": P.ONE_SHOT_PASSAGE_ENTITIES},
        {"role": "user", "content": f"Paragraph:```\n{passage}\n```"},
    ]
    data = _chat_json(client, model=model, messages=messages, max_tokens=1024)
    ents = data.get("named_entities") or []
    if not isinstance(ents, list):
        return []
    return [str(x).strip() for x in ents if str(x).strip()]


def passage_openie(client, model: str, passage: str, entities: List[str]) -> List[List[str]]:
    named_entity_json = json.dumps({"named_entities": entities}, ensure_ascii=False)
    frame = (
        "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
        "Paragraph:\n"
        "```\n"
        f"{passage}\n"
        "```\n\n"
        f"{named_entity_json}"
    )
    one_shot_user = (
        "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
        "Paragraph:\n"
        "```\n"
        f"{P.ONE_SHOT_PASSAGE}\n"
        "```\n\n"
        f"{P.ONE_SHOT_PASSAGE_ENTITIES}"
    )
    messages = [
        {"role": "system", "content": P.OPENIE_SYSTEM},
        {"role": "user", "content": one_shot_user},
        {"role": "assistant", "content": P.ONE_SHOT_PASSAGE_TRIPLES},
        {"role": "user", "content": frame},
    ]
    data = _chat_json(client, model=model, messages=messages, max_tokens=4096)
    triples_raw = data.get("triples") or []
    out: List[List[str]] = []
    if not isinstance(triples_raw, list):
        return out
    for t in triples_raw:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            out.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])
        elif isinstance(t, dict):
            h = t.get("head") or t.get("subject") or t.get("s")
            r = t.get("relation") or t.get("predicate") or t.get("p")
            o = t.get("tail") or t.get("object") or t.get("o")
            if h and r and o:
                out.append([str(h).strip(), str(r).strip(), str(o).strip()])
    return out


def query_ner(client, model: str, question: str) -> List[str]:
    messages = [
        {"role": "system", "content": P.QUERY_NER_SYSTEM},
        {"role": "user", "content": P.QUERY_NER_ONE_SHOT_Q},
        {"role": "assistant", "content": P.QUERY_NER_ONE_SHOT_A},
        {"role": "user", "content": f"\nQuestion: {question}\n\n"},
    ]
    data = _chat_json(client, model=model, messages=messages, max_tokens=300)
    ents = data.get("named_entities") or []
    if not isinstance(ents, list):
        return []
    return [processing_phrases(x) for x in ents if str(x).strip()]


def openie_one_passage(
    client,
    model: str,
    *,
    passage: str,
    title: str = "",
    page: int = 1,
    chunk_id: str = "",
) -> Dict[str, Any]:
    full = f"{title}\n{passage}".strip() if title else passage
    entities = passage_ner(client, model, full)
    triples = passage_openie(client, model, full, entities)
    return {
        "title": title,
        "passage": full,
        "page": page,
        "chunk_id": chunk_id,
        "extracted_entities": entities,
        "extracted_triples": triples,
    }
