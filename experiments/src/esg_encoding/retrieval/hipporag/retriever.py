"""HippoRAG retriever wrapper for EulerESG.

Put this file at:
  backend/src/esg_encoding/retrieval/hipporag/retriever.py

Goals
- **No breaking changes**: return *segment IDs* so your existing ESGChatbot
  pipeline (IDs -> content) keeps working.
- **Fast**: cache per file_id, pack short segments into fewer docs, avoid
  reindexing via a meta fingerprint, and do background indexing optionally.
- **Safe**: if HippoRAG isn't installed or fails, we return empty results so
  the app falls back to the current keyword search.

Requires
  pip install hipporag
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ...models import ProcessingConfig, ReportContent
from .settings import HippoRAGSettings
import math


# Segment-id markers embedded into docs during indexing.
# NOTE: HippoRAG may strip or reformat "metadata-like" lines, so we support multiple encodings.
_SEG_IDS_RE = re.compile(r"__HIPPO_SEGMENT_IDS__:\s*([^\n\r]+)", re.IGNORECASE)
_SEG_IDS_RE2 = re.compile(r"Segment IDs:\s*([^\n\r]+)", re.IGNORECASE)
# Our docs also include per-segment labels like: [<segment_id> - Page <n>]
_SEG_ID_BRACKET_RE = re.compile(r"\[\s*([^\]\s]+)\s*-\s*Page\s*[^\]]*\]", re.IGNORECASE)
# Fallback: common ID token pattern (e.g., P001_S023)
_SEG_ID_TOKEN_RE = re.compile(r"\bP\d{1,4}_S\d{1,4}\b")

def _parse_segment_ids_from_text(text: str) -> List[str]:
    """Extract segment IDs from a retrieved text blob (best-effort)."""
    if not text:
        return []
    # 1) header formats
    for rgx in (_SEG_IDS_RE, _SEG_IDS_RE2):
        m = rgx.search(text)
        if m:
            segs = [x.strip() for x in m.group(1).split(",") if x.strip()]
            if segs:
                return segs
    # 2) bracket labels
    segs = [m.group(1).strip() for m in _SEG_ID_BRACKET_RE.finditer(text)]
    if segs:
        # de-dup preserving order
        seen=set(); out=[]
        for s in segs:
            if s not in seen:
                seen.add(s); out.append(s)
        return out
    # 3) token pattern
    segs = [m.group(0) for m in _SEG_ID_TOKEN_RE.finditer(text)]
    if segs:
        seen=set(); out=[]
        for s in segs:
            if s not in seen:
                seen.add(s); out.append(s)
        return out
    return []



def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _temporary_env(overrides: Dict[str, str | None]):
    """Temporarily set environment variables (restores original values)."""

    class _Ctx:
        def __enter__(self):
            self._old: Dict[str, Optional[str]] = {}
            for k, v in overrides.items():
                self._old[k] = os.environ.get(k)
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, old in self._old.items():
                if old is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = old
            return False

    return _Ctx()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_docs(docs: List[str], settings_sig: str) -> str:
    """Fast stable fingerprint for doc list + relevant settings."""
    h = blake2b(digest_size=16)
    h.update(settings_sig.encode("utf-8"))
    for d in docs:
        b = d.encode("utf-8", errors="ignore")
        h.update(len(b).to_bytes(8, "little", signed=False))
        h.update(b[:512])
        h.update(b[-512:])
    return h.hexdigest()


def _settings_signature(settings: HippoRAGSettings, config: ProcessingConfig) -> str:
    # Anything that changes the index should be in this signature.
    return json.dumps(
        {
            "embedding_model": settings.embedding_model_name,
            "pack": settings.pack_segments,
            "target_chars": settings.target_chars_per_doc,
            "max_docs": settings.max_docs_to_index,
            "min_seg_chars": settings.min_chars_per_segment,
            "llm_model": settings.llm_model_name or config.llm_model,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _segment_text(seg) -> str:
    return (getattr(seg, "content", "") or "").strip()


def _pack_segments(report_content: ReportContent, settings: HippoRAGSettings) -> List[str]:
    """Pack many short segments into fewer medium docs.

    We embed segment IDs into a header line so we can return IDs later.
    """
    segs = getattr(report_content.document_content, "segments", []) or []
    docs: List[str] = []
    if not segs:
        return docs

    current_ids: List[str] = []
    current_pages: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush():
        nonlocal current_ids, current_pages, current_parts, current_len
        if not current_parts:
            return
        ids = ",".join(current_ids)
        pages = ",".join(current_pages)
        body = "\n\n".join(current_parts)
        doc = f"__HIPPO_SEGMENT_IDS__: {ids}\n__HIPPO_PAGES__: {pages}\n\n{body}".strip()
        docs.append(doc)
        current_ids, current_pages, current_parts, current_len = [], [], [], 0

    for seg in segs:
        txt = _segment_text(seg)
        if len(txt) < settings.min_chars_per_segment:
            continue
        sid = str(getattr(seg, "segment_id", ""))
        page = str(getattr(seg, "page_number", ""))

        if current_len > 0 and (current_len + len(txt) + 2) > settings.target_chars_per_doc:
            flush()

        current_ids.append(sid)
        current_pages.append(page)
        current_parts.append(f"[{sid} - Page {page}]\n{txt}")
        current_len += len(txt) + 2


    flush()

    # If we have more docs than the index budget, sample evenly across the whole report
    # to avoid indexing only the beginning of the document.
    if settings.max_docs_to_index and len(docs) > settings.max_docs_to_index:
        stride = max(1, math.ceil(len(docs) / settings.max_docs_to_index))
        sampled = docs[::stride]
        # Ensure we do not exceed the budget.
        docs = sampled[: settings.max_docs_to_index]

    return docs


def _one_segment_per_doc(report_content: ReportContent, settings: HippoRAGSettings) -> List[str]:
    segs = getattr(report_content.document_content, "segments", []) or []
    docs: List[str] = []
    for seg in segs:
        txt = _segment_text(seg)
        if len(txt) < settings.min_chars_per_segment:
            continue
        sid = str(getattr(seg, "segment_id", ""))
        page = str(getattr(seg, "page_number", ""))
        docs.append(
            f"__HIPPO_SEGMENT_IDS__: {sid}\n__HIPPO_PAGES__: {page}\n\n[{sid} - Page {page}]\n{txt}".strip()
        )
    return docs


def _extract_text_blobs(obj: Any) -> List[str]:
    """Best-effort extraction of retrieved doc texts from HippoRAG outputs."""
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        for k in ("retrieved_docs", "docs", "passages", "contexts", "results", "retrieval_results"):
            if k in obj:
                return _extract_text_blobs(obj[k])
        if "text" in obj and isinstance(obj["text"], str):
            return [obj["text"]]
        if "content" in obj and isinstance(obj["content"], str):
            return [obj["content"]]
        return [json.dumps(obj, ensure_ascii=False)]
    if isinstance(obj, list):
        out: List[str] = []
        for x in obj:
            out.extend(_extract_text_blobs(x))
        return out
    return [str(obj)]


@dataclass
class _IndexMeta:
    fingerprint: str
    doc_count: int
    created_at: str
    settings_sig: str


class HippoRAGRetriever:
    """Per-file HippoRAG index + retrieval."""

    def __init__(self, config: ProcessingConfig, settings: Optional[HippoRAGSettings] = None):
        self.config = config
        self.settings = settings or HippoRAGSettings()
        self._rag_cache: Dict[str, Any] = {}
        self._docmap_cache: Dict[str, Dict[str, List[str]]] = {}  # file_id -> doc_idx(str) -> [segment_ids]
        self._locks: Dict[str, threading.Lock] = {}
        self._indexing: Dict[str, bool] = {}
        self._stats: Dict[str, int] = {
            "hipporag_calls": 0,     # 走 HippoRAG 的次数
            "base_calls": 0,         # fallback 到 BaseRAG 的次数（由 patch 增加）
            "index_builds": 0,       # 真实发生重建索引次数
        }

    def is_enabled(self) -> bool:
        return bool(self.settings.enabled)

    def is_indexing(self, file_id: str) -> bool:
        return bool(self._indexing.get(file_id))

    def _lock_for(self, file_id: str) -> threading.Lock:
        if file_id not in self._locks:
            self._locks[file_id] = threading.Lock()
        return self._locks[file_id]

    def _save_dir(self, file_id: str) -> Path:
        return Path(self.settings.cache_root) / str(file_id)


    def _docmap_path(self, save_dir: Path) -> Path:
        return Path(save_dir) / "docmap.json"

    def _load_docmap(self, save_dir: Path) -> Dict[str, List[str]]:
        p = self._docmap_path(save_dir)
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    # ensure list-of-str
                    out: Dict[str, List[str]] = {}
                    for k, v in obj.items():
                        if isinstance(v, list):
                            out[str(k)] = [str(x) for x in v if str(x)]
                    return out
        except Exception:
            pass
        return {}

    def _write_docmap(self, save_dir: Path, docmap: Dict[str, List[str]]) -> None:
        p = self._docmap_path(save_dir)
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(docmap, f, ensure_ascii=False, indent=2)
        except Exception:
            # docmap is best-effort; do not fail indexing
            return

    def _collect_doc_indices(self, obj: Any) -> List[int]:
        """Best-effort extraction of doc indices/ids from HippoRAG outputs."""
        out: List[int] = []
        if obj is None:
            return out
        if isinstance(obj, int):
            return [obj]
        if isinstance(obj, str):
            # common patterns: "doc_12", "12"
            out.extend([int(m.group(1)) for m in re.finditer(r"\bdoc_(\d+)\b", obj)])
            return out
        if isinstance(obj, (list, tuple)):
            for x in obj:
                out.extend(self._collect_doc_indices(x))
            return out
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in {"doc_id", "docids", "doc_ids", "doc_idx", "doc_index", "doc_indices",
                          "document_id", "document_ids", "document_index", "document_indices",
                          "retrieved_doc_ids", "retrieved_docs", "docs", "documents"}:
                    out.extend(self._collect_doc_indices(v))
                else:
                    out.extend(self._collect_doc_indices(v))
            return out
        # generic object: look for common attributes
        for attr in ("doc_id", "doc_ids", "doc_indices", "documents", "docs"):
            if hasattr(obj, attr):
                out.extend(self._collect_doc_indices(getattr(obj, attr)))
        return out
    def _meta_path(self, save_dir: Path) -> Path:
        return save_dir / "index_meta.json"

    def _load_meta(self, save_dir: Path) -> Optional[_IndexMeta]:
        p = self._meta_path(save_dir)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return _IndexMeta(
                fingerprint=str(data.get("fingerprint", "")),
                doc_count=int(data.get("doc_count", 0)),
                created_at=str(data.get("created_at", "")),
                settings_sig=str(data.get("settings_sig", "")),
            )
        except Exception:
            return None

    def _write_meta(self, save_dir: Path, meta: _IndexMeta) -> None:
        p = self._meta_path(save_dir)
        try:
            p.write_text(
                json.dumps(
                    {
                        "fingerprint": meta.fingerprint,
                        "doc_count": meta.doc_count,
                        "created_at": meta.created_at,
                        "settings_sig": meta.settings_sig,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[HippoRAG] write meta failed: {e}")

    def _import_hipporag(self):
        try:
            from hipporag import HippoRAG  # type: ignore
            return HippoRAG
        except Exception as e:
            raise RuntimeError("HippoRAG is not installed. Run: pip install hipporag") from e

    def _init_rag(self, save_dir: Path):
        HippoRAG = self._import_hipporag()

        llm_model_name = self.settings.llm_model_name or self.config.llm_model
        llm_base_url = self.settings.llm_base_url or (self.config.llm_base_url or None)
        llm_api_key = self.settings.llm_api_key or (self.config.llm_api_key or None)

        kwargs: Dict[str, Any] = {
            "save_dir": str(save_dir),
            "llm_model_name": llm_model_name,
            "embedding_model_name": self.settings.embedding_model_name,
        }
        # 用户要求：不要使用 NVIDIA / NV-Embed 作为 HippoRAG 的 embedding。
        if isinstance(kwargs.get("embedding_model_name"), str) and kwargs["embedding_model_name"].lower().startswith("nvidia/"):
            logger.warning("[HippoRAG] embedding_model_name 指向 NVIDIA 模型，已自动切换为 fallback_embedding_model_name。")
            kwargs["embedding_model_name"] = getattr(self.settings, "fallback_embedding_model_name", "facebook/contriever")
        if llm_base_url:
            kwargs["llm_base_url"] = llm_base_url
        if self.settings.embedding_base_url:
            kwargs["embedding_base_url"] = self.settings.embedding_base_url

        # Extra HippoRAG init kwargs (optional)
        # This lets you tune HippoRAG without changing code. For example:
        #   HIPPO_EXTRA_KWARGS_JSON='{"max_phrases_per_doc": 256, "bm25_k1": 1.2}'
        extra_json = os.getenv("HIPPO_EXTRA_KWARGS_JSON", "").strip()
        if extra_json:
            try:
                extra = json.loads(extra_json)
                if isinstance(extra, dict):
                    # Avoid letting env override the essentials
                    for k in ("save_dir", "llm_model_name", "embedding_model_name"):
                        extra.pop(k, None)
                    kwargs.update(extra)
            except Exception as e:
                logger.warning(f"[HippoRAG] invalid HIPPO_EXTRA_KWARGS_JSON: {e}")

        env = {
            "OPENAI_API_KEY": llm_api_key,
            "OPENAI_BASE_URL": llm_base_url,
            "OPENAI_API_BASE": llm_base_url,
        }
        return kwargs, env, HippoRAG

    def _build_docs(self, report_content: ReportContent) -> List[str]:
        if not report_content or not getattr(report_content, "document_content", None):
            return []
        if self.settings.pack_segments:
            return _pack_segments(report_content, self.settings)
        return _one_segment_per_doc(report_content, self.settings)

    def ensure_index(self, file_id: str, report_content: ReportContent) -> bool:
        """Ensure an index exists on disk for this file_id."""
        if not self.is_enabled():
            return False

        docs = self._build_docs(report_content)
        if not docs:
            return False

        save_dir = self._save_dir(file_id)
        _safe_mkdir(save_dir)

        # Persist a mapping from HippoRAG doc index -> segment_id(s) so we can
        # recover segment IDs even if HippoRAG returns doc IDs rather than raw text.
        docmap: Dict[str, List[str]] = {}
        for i, d in enumerate(docs):
            segs = _parse_segment_ids_from_text(d)
            if segs:
                docmap[str(i)] = segs
        self._docmap_cache[file_id] = docmap
        self._write_docmap(save_dir, docmap)

        settings_sig = _settings_signature(self.settings, self.config)
        fingerprint = _hash_docs(docs, settings_sig)
        meta = self._load_meta(save_dir)

        # A "ready" marker prevents treating half-built cache dirs as valid.
        # We only short-circuit if both the meta matches AND the marker exists.
        ready_path = save_dir / ".ready"

        if (
            not self.settings.force_reindex
            and meta
            and meta.fingerprint == fingerprint
            and meta.doc_count == len(docs)
            and meta.settings_sig == settings_sig
            and ready_path.exists()
        ):
            if file_id not in self._rag_cache:
                try:
                    kwargs, env, HippoRAG = self._init_rag(save_dir)
                    with _temporary_env(env):
                        try:
                            try:
                                self._rag_cache[file_id] = HippoRAG(**kwargs)
                            except TypeError as te:
                                # Extra kwargs (e.g. HIPPO_EXTRA_KWARGS_JSON) might not be supported by
                                # the installed HippoRAG version. Retry with a minimal safe subset.
                                logger.warning(f"[HippoRAG] init kwargs rejected ({te}); retrying with minimal kwargs")
                                minimal = {k: kwargs[k] for k in ("save_dir", "llm_model_name", "embedding_model_name") if k in kwargs}
                                for opt in ("llm_base_url", "llm_api_key", "embedding_base_url", "embedding_api_key"):
                                    if opt in kwargs:
                                        minimal[opt] = kwargs[opt]
                                self._rag_cache[file_id] = HippoRAG(**minimal)
                        except AssertionError as e:
                            # HippoRAG v2.x asserts if `embedding_model_name` is not one of its
                            # supported local embedding backends. If the user configured a HuggingFace
                            # model id (e.g., "BAAI/bge-m3") without providing an OpenAI-compatible
                            # embedding_base_url, fall back to the upstream default.
                            msg = str(e)
                            if "Unknown embedding model name" in msg:
                                logger.warning(
                                    f"[HippoRAG] {msg}. Falling back embedding_model_name='{getattr(self.settings, 'fallback_embedding_model_name', 'facebook/contriever')}'."
                                )
                                kwargs = dict(kwargs)
                                kwargs["embedding_model_name"] = getattr(self.settings, "fallback_embedding_model_name", "facebook/contriever")
                                self._rag_cache[file_id] = HippoRAG(**kwargs)
                            else:
                                raise
                except Exception as e:
                    logger.warning(f"[HippoRAG] init existing index failed: {e}")
                    return False
            return True

        lock = self._lock_for(file_id)
        with lock:
            meta = self._load_meta(save_dir)
            if (
                not self.settings.force_reindex
                and meta
                and meta.fingerprint == fingerprint
                and meta.doc_count == len(docs)
                and meta.settings_sig == settings_sig
                and ready_path.exists()
            ):
                return True

            # We are (re)building the index. Remove stale ready marker first so
            # a half-built cache isn't treated as valid by concurrent requests.
            try:
                ready_path.unlink()
            except FileNotFoundError:
                pass

            try:
                kwargs, env, HippoRAG = self._init_rag(save_dir)
                with _temporary_env(env):
                    try:
                        try:
                            rag = HippoRAG(**kwargs)
                        except TypeError as te:
                            logger.warning(f"[HippoRAG] init kwargs rejected ({te}); retrying with minimal kwargs")
                            minimal = {k: kwargs[k] for k in ("save_dir", "llm_model_name", "embedding_model_name") if k in kwargs}
                            for opt in ("llm_base_url", "llm_api_key", "embedding_base_url", "embedding_api_key"):
                                if opt in kwargs:
                                    minimal[opt] = kwargs[opt]
                            rag = HippoRAG(**minimal)
                    except AssertionError as e:
                        msg = str(e)
                        if "Unknown embedding model name" in msg:
                            logger.warning(
                                f"[HippoRAG] {msg}. Falling back embedding_model_name='{getattr(self.settings, 'fallback_embedding_model_name', 'facebook/contriever')}'."
                            )
                            kwargs = dict(kwargs)
                            kwargs["embedding_model_name"] = getattr(self.settings, "fallback_embedding_model_name", "facebook/contriever")
                            rag = HippoRAG(**kwargs)
                        else:
                            raise
                    logger.info(f"[HippoRAG] indexing file_id={file_id} docs={len(docs)}")
                    rag.index(docs)
                    self._rag_cache[file_id] = rag
                    self._stats["index_builds"] += 1

                self._write_meta(
                    save_dir,
                    _IndexMeta(
                        fingerprint=fingerprint,
                        doc_count=len(docs),
                        created_at=_now_iso(),
                        settings_sig=settings_sig,
                    ),
                )
                # Marker file to avoid treating partially built cache dirs as valid.
                try:
                    ready_path.write_text(_now_iso(), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"[HippoRAG] failed to write ready marker: {e}")
                return True
            except Exception as e:
                # Keep full traceback - the typical failure here is inside HippoRAG OpenIE/NER
                # parsing (e.g., entities returned as dict -> unhashable), or LLM config.
                logger.exception(f"[HippoRAG] indexing failed (fallback to keyword). file_id={file_id}")
                try:
                    ready_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return False

    def schedule_index(self, file_id: str, report_content: ReportContent) -> None:
        """Optionally warm-index in a background thread (non-blocking)."""
        if not self.is_enabled() or not self.settings.warm_index_in_background:
            return
        if self._indexing.get(file_id):
            return

        def _worker():
            try:
                self._indexing[file_id] = True
                self.ensure_index(file_id, report_content)
            finally:
                self._indexing[file_id] = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def retrieve_segment_ids(self, file_id: str, report_content: ReportContent, query: str) -> List[str]:
        """Retrieve segment IDs for the current query. Return [] on failure -> fallback."""
        if not self.is_enabled():
            return []

        if self._indexing.get(file_id):
            return []

        save_dir = self._save_dir(file_id)
        meta = self._load_meta(save_dir)
        ready_path = save_dir / ".ready"
        if not meta or self.settings.force_reindex or not ready_path.exists():
            ok = self.ensure_index(file_id, report_content)
            if not ok:
                return []

        rag = self._rag_cache.get(file_id)
        if rag is None:
            try:
                kwargs, env, HippoRAG = self._init_rag(save_dir)
                with _temporary_env(env):
                    rag = HippoRAG(**kwargs)
                self._rag_cache[file_id] = rag
            except Exception:
                return []

        llm_base_url = self.settings.llm_base_url or (self.config.llm_base_url or None)
        llm_api_key = self.settings.llm_api_key or (self.config.llm_api_key or None)
        env = {"OPENAI_API_KEY": llm_api_key, "OPENAI_BASE_URL": llm_base_url, "OPENAI_API_BASE": llm_base_url}

        try:
            self._stats["hipporag_calls"] += 1
            with _temporary_env(env):
                raw = rag.retrieve(queries=[query], num_to_retrieve=int(self.settings.top_k_docs))

            texts = _extract_text_blobs(raw)

            ids: List[str] = []
            seen = set()

            # 1) Preferred: parse segment IDs directly from returned text blobs.
            for t in texts:
                for sid in _parse_segment_ids_from_text(t):
                    if sid in seen:
                        continue
                    seen.add(sid)
                    ids.append(sid)
                    if len(ids) >= int(self.settings.max_segment_ids_for_context):
                        return ids

            if ids:
                return ids

            # 2) If HippoRAG returns doc indices/ids rather than raw text, recover via docmap.
            docmap = self._docmap_cache.get(file_id) or {}
            if not docmap:
                docmap = self._load_docmap(save_dir)
                # If the index exists but docmap is missing (old runs), rebuild docmap from
                # current report_content without forcing a re-index.
                if not docmap:
                    try:
                        rebuilt_docs = self._build_docs(report_content)
                        tmp: Dict[str, List[str]] = {}
                        for i, d in enumerate(rebuilt_docs):
                            segs = _parse_segment_ids_from_text(d)
                            if segs:
                                tmp[str(i)] = segs
                        if tmp:
                            docmap = tmp
                            self._write_docmap(save_dir, docmap)
                    except Exception:
                        pass
                self._docmap_cache[file_id] = docmap

            for di in self._collect_doc_indices(raw):
                segs = docmap.get(str(di))
                if not segs:
                    continue
                for sid in segs:
                    if sid in seen:
                        continue
                    seen.add(sid)
                    ids.append(sid)
                    if len(ids) >= int(self.settings.max_segment_ids_for_context):
                        return ids

            if ids:
                return ids

            # 3) Debug breadcrumbs (kept short) -> caller will fallback.
            if isinstance(raw, dict):
                keys = list(raw.keys())[:12]
            else:
                keys = None
            preview = ""
            try:
                preview = str(raw)
            except Exception:
                preview = repr(raw)
            preview = preview[:600]
            tprev = " | ".join([(t or "")[:120].replace("\n", " ") for t in (texts or [])[:3]])
            logger.info(f"[HippoRAG] empty results after parse. raw_type={type(raw).__name__} keys={keys} texts_preview='{tprev}' raw_preview='{preview}'")
            return []

            return ids
        except Exception as e:
            logger.warning(f"[HippoRAG] retrieve failed (fallback). Error: {e}")
            return []

    def get_status(self, file_id: Optional[str] = None) -> Dict[str, object]:
        enabled = self.is_enabled()
        indexing = self.is_indexing(file_id) if (file_id and enabled) else False

        last_index_time = None
        ready = False
        doc_count = 0

        if file_id and enabled:
            save_dir = self._save_dir(file_id)
            meta = self._load_meta(save_dir)
            if meta:
                last_index_time = meta.created_at or None
                doc_count = int(meta.doc_count or 0)
                # 有 meta + doc_count>0 + 当前不在 indexing => ready
                ready = (doc_count > 0) and (not indexing)

        return {
            "enabled": enabled,
            "ready": ready,
            "indexing": indexing,
            "last_index_time": last_index_time,
            "doc_count": doc_count,
            "stats": dict(self._stats),
        }