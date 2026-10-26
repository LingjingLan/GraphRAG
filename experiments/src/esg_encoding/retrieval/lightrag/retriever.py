"""LightRAG retriever wrapper for EulerESG (parallel to HippoRAGRetriever).

Returns segment IDs so the existing ESGChatbot / GraphRAG benchmark pipeline
(IDs → content) keeps working. Safe no-op when LightRAG is missing or disabled.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ...models import ProcessingConfig, ReportContent
from .factory import create_lightrag
from .settings import LightRAGSettings

_SEG_IDS_RE = re.compile(r"__LIGHTRAG_SEGMENT_IDS__:\s*([^\n\r]+)", re.IGNORECASE)
_SEG_IDS_RE2 = re.compile(r"__HIPPO_SEGMENT_IDS__:\s*([^\n\r]+)", re.IGNORECASE)
_SEG_ID_BRACKET_RE = re.compile(r"\[\s*([^\]\s]+)\s*-\s*Page\s*[^\]]*\]", re.IGNORECASE)
_SEG_ID_TOKEN_RE = re.compile(r"\bP\d{1,4}_S\d{1,4}\b")


def _parse_segment_ids_from_text(text: str) -> List[str]:
    if not text:
        return []
    for rgx in (_SEG_IDS_RE, _SEG_IDS_RE2):
        m = rgx.search(text)
        if m:
            segs = [x.strip() for x in m.group(1).split(",") if x.strip()]
            if segs:
                return segs
    segs = [m.group(1).strip() for m in _SEG_ID_BRACKET_RE.finditer(text)]
    if segs:
        seen: set[str] = set()
        out: List[str] = []
        for s in segs:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    segs = [m.group(0) for m in _SEG_ID_TOKEN_RE.finditer(text)]
    if segs:
        seen = set()
        out = []
        for s in segs:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return []


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _hash_docs(docs: List[str], settings_sig: str) -> str:
    h = blake2b(digest_size=16)
    h.update(settings_sig.encode("utf-8"))
    for d in docs:
        b = d.encode("utf-8", errors="ignore")
        h.update(len(b).to_bytes(8, "little", signed=False))
        h.update(b[:512])
    h.update(str(len(docs)).encode("utf-8"))
    return h.hexdigest()


def _segment_text(seg: Any) -> str:
    return str(getattr(seg, "content", "") or "").strip()


def _pack_docs(report_content: ReportContent, settings: LightRAGSettings) -> List[str]:
    segs = getattr(report_content.document_content, "segments", []) or []
    units: List[tuple[str, str, str]] = []
    for seg in segs:
        txt = _segment_text(seg)
        if len(txt) < settings.min_chars_per_segment:
            continue
        sid = str(getattr(seg, "segment_id", "") or "")
        page = str(getattr(seg, "page_number", "") or "")
        if not sid:
            continue
        units.append((sid, page, txt))

    if not settings.pack_segments:
        docs = [
            f"__LIGHTRAG_SEGMENT_IDS__: {sid}\n__LIGHTRAG_PAGES__: {page}\n\n"
            f"[{sid} - Page {page}]\n{txt}".strip()
            for sid, page, txt in units
        ]
        return docs[: settings.max_docs_to_index]

    docs: List[str] = []
    buf_sids: List[str] = []
    buf_pages: List[str] = []
    buf_parts: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf_sids, buf_pages, buf_parts, buf_len
        if not buf_parts:
            return
        header = (
            f"__LIGHTRAG_SEGMENT_IDS__: {','.join(buf_sids)}\n"
            f"__LIGHTRAG_PAGES__: {','.join(buf_pages)}\n\n"
        )
        docs.append(header + "\n\n".join(buf_parts))
        buf_sids, buf_pages, buf_parts, buf_len = [], [], [], 0

    for sid, page, txt in units:
        block = f"[{sid} - Page {page}]\n{txt}"
        if buf_len and buf_len + len(block) + 2 > settings.target_chars_per_doc:
            flush()
            if len(docs) >= settings.max_docs_to_index:
                break
        buf_sids.append(sid)
        buf_pages.append(page)
        buf_parts.append(block)
        buf_len += len(block) + 2
    flush()
    return docs[: settings.max_docs_to_index]


def _settings_sig(settings: LightRAGSettings) -> str:
    return "|".join(
        [
            settings.query_mode,
            settings.embedding_model_name,
            str(settings.embedding_dim),
            str(settings.target_chars_per_doc),
            str(settings.max_docs_to_index),
            str(settings.pack_segments),
            str(settings.llm_model_name or ""),
        ]
    )


def _run_async(coro):
    """Run async coroutine from sync code, including when a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Nested / already-running loop: use a dedicated thread.
        result: dict[str, Any] = {}

        def _worker():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")
    return asyncio.run(coro)


@dataclass
class _IndexMeta:
    fingerprint: str
    doc_count: int
    created_at: str
    settings_sig: str


class LightRAGRetriever:
    """Per-file LightRAG index + retrieval."""

    def __init__(self, config: ProcessingConfig, settings: Optional[LightRAGSettings] = None):
        self.config = config
        self.settings = settings or LightRAGSettings()
        self._rag_cache: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._indexing: Dict[str, bool] = {}

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
        self._meta_path(save_dir).write_text(
            json.dumps(
                {
                    "fingerprint": meta.fingerprint,
                    "doc_count": meta.doc_count,
                    "created_at": meta.created_at,
                    "settings_sig": meta.settings_sig,
                    "backend": "lightrag",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    async def _get_or_create_rag(self, file_id: str):
        if file_id in self._rag_cache:
            return self._rag_cache[file_id]
        save_dir = self._save_dir(file_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        rag = create_lightrag(
            save_dir,
            settings=self.settings,
            config=self.config,
            workspace=str(file_id),
        )
        await rag.initialize_storages()
        self._rag_cache[file_id] = rag
        return rag

    def ensure_index(self, file_id: str, report_content: ReportContent) -> bool:
        if not self.is_enabled():
            return False
        try:
            import lightrag  # noqa: F401
        except Exception as exc:
            logger.warning(f"[LightRAG] package missing: {exc}")
            return False

        with self._lock_for(file_id):
            self._indexing[file_id] = True
            try:
                return bool(_run_async(self._aensure_index(file_id, report_content)))
            except Exception as exc:
                logger.exception(f"[LightRAG] ensure_index failed file_id={file_id}: {exc}")
                return False
            finally:
                self._indexing[file_id] = False

    async def _aensure_index(self, file_id: str, report_content: ReportContent) -> bool:
        docs = _pack_docs(report_content, self.settings)
        if not docs:
            logger.warning(f"[LightRAG] no docs to index file_id={file_id}")
            return False
        sig = _settings_sig(self.settings)
        fp = _hash_docs(docs, sig)
        save_dir = self._save_dir(file_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = self._load_meta(save_dir)
        has_graph = any(save_dir.rglob("*.graphml"))
        if (
            meta
            and meta.fingerprint == fp
            and not self.settings.force_reindex
            and has_graph
        ):
            # Still warm the in-memory instance.
            await self._get_or_create_rag(file_id)
            logger.info(f"[LightRAG] reuse index file_id={file_id} docs={meta.doc_count}")
            return True

        # Fresh insert: clear previous store if force / fingerprint mismatch.
        if self.settings.force_reindex or (meta and meta.fingerprint != fp):
            for p in save_dir.glob("*"):
                if p.name == "index_meta.json":
                    continue
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        import shutil

                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            self._rag_cache.pop(file_id, None)

        rag = await self._get_or_create_rag(file_id)
        logger.info(f"[LightRAG] indexing file_id={file_id} docs={len(docs)}")
        await rag.ainsert(docs)
        self._write_meta(
            save_dir,
            _IndexMeta(
                fingerprint=fp,
                doc_count=len(docs),
                created_at=_now_iso(),
                settings_sig=sig,
            ),
        )
        return True

    def retrieve_segment_ids(
        self,
        file_id: str,
        report_content: ReportContent,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[str]:
        if not self.is_enabled() or not str(query or "").strip():
            return []
        try:
            return list(
                _run_async(
                    self._aretrieve_segment_ids(
                        file_id, report_content, query, top_k=top_k
                    )
                )
                or []
            )
        except Exception as exc:
            logger.warning(f"[LightRAG] retrieve failed file_id={file_id}: {exc}")
            return []

    async def _aretrieve_segment_ids(
        self,
        file_id: str,
        report_content: ReportContent,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[str]:
        from lightrag import QueryParam

        ok = await self._aensure_index(file_id, report_content)
        if not ok:
            return []
        rag = await self._get_or_create_rag(file_id)
        param = QueryParam(
            mode=self.settings.query_mode,
            only_need_context=True,
            top_k=int(top_k or self.settings.top_k),
            chunk_top_k=self.settings.chunk_top_k,
            enable_rerank=self.settings.enable_rerank,
        )
        ctx = await rag.aquery(query, param=param)
        text = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False)
        ids = _parse_segment_ids_from_text(text)
        # de-dup preserve order, cap
        seen: set[str] = set()
        out: List[str] = []
        for sid in ids:
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
            if len(out) >= self.settings.max_segment_ids_for_context:
                break
        return out

    def retrieve_context(self, file_id: str, query: str, *, only_need_context: bool = True) -> str:
        """Raw LightRAG context / answer string (offline QA helper)."""
        try:
            return str(
                _run_async(self._aretrieve_context(file_id, query, only_need_context=only_need_context))
                or ""
            )
        except Exception as exc:
            logger.warning(f"[LightRAG] retrieve_context failed: {exc}")
            return ""

    async def _aretrieve_context(
        self, file_id: str, query: str, *, only_need_context: bool = True
    ) -> str:
        from lightrag import QueryParam

        rag = await self._get_or_create_rag(file_id)
        param = QueryParam(
            mode=self.settings.query_mode,
            only_need_context=only_need_context,
            top_k=self.settings.top_k,
            chunk_top_k=self.settings.chunk_top_k,
            enable_rerank=self.settings.enable_rerank,
        )
        ctx = await rag.aquery(query, param=param)
        return ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False)
