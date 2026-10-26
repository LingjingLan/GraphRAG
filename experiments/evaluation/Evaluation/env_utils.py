"""Load LLM / embedding settings from EulerESG backend config/.env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def detect_newesg_root() -> Path:
    here = Path(__file__).resolve()
    # .../ESG_KG/output/evaluate/Evaluation/env_utils.py → newESG
    candidates = [
        here.parents[4],  # evaluate → output → ESG_KG → newESG
        Path("/workspace"),
    ]
    for root in candidates:
        if (root / "Datasets" / "Corpus").is_dir() or (root / "ESG_KG").is_dir():
            return root
    return candidates[0]


def load_dotenv() -> Dict[str, str]:
    keys = [
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LIGHTRAG_EMBEDDING_MODEL",
        "LIGHTRAG_EMBEDDING_BASE_URL",
        "LIGHTRAG_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ]
    env = {k: "" for k in keys}
    root = detect_newesg_root()
    for path in (
        root / "EulerESG" / "backend" / "config" / ".env",
        Path("/workspace/backend/config/.env"),
        Path("/workspace/EulerESG/backend/config/.env"),
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in env and v:
                env[k] = v
        break
    for k in list(env):
        if not env[k]:
            env[k] = os.getenv(k, "") or ""
    return env


def apply_env_to_os(cfg: Dict[str, str]) -> None:
    """Export keys so langchain/openai clients can pick them up."""
    api_key = cfg.get("LLM_API_KEY") or cfg.get("OPENAI_API_KEY") or ""
    base_url = cfg.get("LLM_BASE_URL") or cfg.get("OPENAI_BASE_URL") or ""
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
