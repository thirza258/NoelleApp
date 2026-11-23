"""Local document collections for retrieval-augmented generation (RAG).

Everything lives in one SQLite database, ``<store_dir>/rag.db`` (WAL mode),
created on the first ingest:

* ``collections``      name, embedding model + dimension (fixed at creation), counters
* ``documents``        one row per source file (absolute path, sha256 for change detection)
* ``chunks``           chunk text, 1-based page (PDF) and the float32 embedding as a BLOB
* ``vec_chunks_<id>``  a sqlite-vec ``vec0`` cosine index per collection

The BLOB column is the source of truth for vectors. The numpy brute-force
fallback searches it when the sqlite-vec extension cannot be loaded, and it
lets a ``vec0`` index be rebuilt when it is missing (collection built without
the extension) or stale (changed by a process without the extension). Both
search paths return identical hit dictionaries.

Threading: every thread gets its own connection, so the worker's I/O thread
can list collections (WAL readers never wait for the writer) while the GPU
lane ingests. Writes are serialised by a lock that is never held while files
are extracted or embedded. Document text is never logged.
"""

from __future__ import annotations

import bisect
import codecs
import hashlib
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from noelle.runtime.errors import BadRequest, Cancelled

log = logging.getLogger("noelle.rag")

DB_FILENAME = "rag.db"
SCHEMA_VERSION = 1
EMBED_BATCH = 32
WORDS_PER_TOKEN = 0.75
MIN_CHUNK_TOKENS = 16
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOP_K = 200
MIN_TRUNCATED_CHARS = 120

TEXT_SUFFIXES = frozenset({
    ".txt", ".text", ".md", ".markdown", ".mdx", ".rst", ".adoc", ".org", ".tex", ".csv", ".tsv", ".json",
    ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log", ".xml", ".srt", ".vtt",
    ".py", ".pyi", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".scala",
    ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs",
    ".fs", ".lua", ".pl", ".r", ".jl", ".dart", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".sql",
    ".graphql", ".proto", ".css", ".scss", ".less", ".vue", ".svelte",
})
HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})

_VEC_NONE, _VEC_SYNCED, _VEC_STALE = 0, 1, 2
_WORD_RE = re.compile(r"\S+")
_VEC_TABLE_RE = re.compile(r"vec_chunks_(\d+)")
_SENTENCE_END = frozenset(".!?…。！？")
_CLOSERS = frozenset("\"')]}»”’")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding_model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    documents INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 0,
    vec_state INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunks INTEGER NOT NULL,
    added_at REAL NOT NULL,
    UNIQUE (collection_id, path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""

ProgressFn = Callable[[dict[str, Any]], None]


# ============================================================================
# chunking (pure)
# ============================================================================


@dataclass(frozen=True)
class TextChunk:
    """A window of source text; ``text == source[start:end]``."""

    text: str
    start: int
    end: int
    tokens: int


def chunk_text(text: str, tokenizer: Any = None, *, chunk_tokens: int = 512, overlap: int = 64) -> list[TextChunk]:
    """Split ``text`` into overlapping windows of at most ``chunk_tokens`` tokens.

    Token positions come from the tokenizer's offset mapping (Hugging Face fast
    tokenizers), so every chunk is an exact slice of the source. Inside a window
    the cut prefers a paragraph break, then a sentence end, then a line break,
    provided the chunk keeps at least half the window. The next window starts
    ``overlap`` tokens earlier, moved forward to a sentence start when one lies
    in the overlap. Without a usable tokenizer, whitespace words stand in for
    tokens at ~0.75 words per token.
    """
    if not text or not text.strip():
        return []
    spans, exact = _token_spans(text, tokenizer)
    if not spans:
        return []
    size = max(1, int(chunk_tokens))
    lap = max(0, int(overlap))
    if not exact:
        size = max(1, round(size * WORDS_PER_TOKEN))
        lap = round(lap * WORDS_PER_TOKEN)
    lap = min(lap, size // 2)
    chunks: list[TextChunk] = []
    for first, last in _windows(text, spans, size, lap):
        start, end = spans[first][0], max(spans[last - 1][1], spans[first][0])
        piece = text[start:end]
        stripped = piece.strip()
        if not stripped:
            continue
        start += len(piece) - len(piece.lstrip())
        chunks.append(TextChunk(stripped, start, start + len(stripped), last - first))
    return chunks


def chunk_segments(segments: Sequence[tuple[str, int | None]], tokenizer: Any = None, *, chunk_tokens: int = 512,
                   overlap: int = 64) -> list[tuple[TextChunk, int | None]]:
    """Chunk extracted segments (e.g. PDF pages) as one text, labelling each chunk with a page.

    Segments are joined with blank lines, so page breaks count as paragraph
    breaks; ``TextChunk`` offsets refer to that joined text. A chunk's page is
    where its new text begins: the overlap it repeats from the previous chunk
    is ignored, so a chunk that reaches back onto the previous page only for
    the overlap is labelled with the next page.
    """
    parts: list[str] = []
    starts: list[int] = []
    pages: list[int | None] = []
    offset = 0
    for text, page in segments:
        text = (text or "").strip()
        if not text:
            continue
        if parts:
            offset += 2
        parts.append(text)
        starts.append(offset)
        pages.append(page)
        offset += len(text)
    if not parts:
        return []
    joined = "\n\n".join(parts)
    out: list[tuple[TextChunk, int | None]] = []
    previous_end = 0
    for chunk in chunk_text(joined, tokenizer, chunk_tokens=chunk_tokens, overlap=overlap):
        anchor = chunk.start if chunk.start >= previous_end else min(previous_end, chunk.end - 1)
        while anchor < chunk.end - 1 and joined[anchor].isspace():
            anchor += 1
        out.append((chunk, pages[max(bisect.bisect_right(starts, anchor) - 1, 0)]))
        previous_end = chunk.end
    return out


def effective_chunk_tokens(embedder: Any, chunk_tokens: int) -> int:
    """Cap ``chunk_tokens`` so a chunk plus special tokens fits the embedder's ``max_seq_length``."""
    size = int(chunk_tokens) if chunk_tokens else 512
    limit = _safe_attr(embedder, "max_seq_length")
    if limit:
        tokenizer = _safe_attr(embedder, "tokenizer")
        special = 0
        if tokenizer is not None:
            try:
                special = int(tokenizer.num_special_tokens_to_add(pair=False))
            except Exception:
                special = 2
        size = min(size, int(limit) - special)
    return max(MIN_CHUNK_TOKENS, size)


def _token_spans(text: str, tokenizer: Any) -> tuple[list[tuple[int, int]], bool]:
    """Character spans of each token; ``exact`` is False for the whitespace-word approximation."""
    if tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
            spans = [(int(s), int(e)) for s, e in encoded["offset_mapping"] if e > s]
            if spans:
                return spans, True
        except Exception:  # slow (pure Python) tokenizers cannot return offsets
            pass
        words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        try:  # count tokens per word; each word repeats its span once per token
            ids = tokenizer([text[s:e] for s, e in words], add_special_tokens=False, verbose=False)["input_ids"]
            spans = [span for span, tokens in zip(words, ids) for _ in range(max(1, len(tokens)))]
            if spans:
                return spans, True
        except Exception:
            pass
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)], False


def _windows(text: str, spans: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Token index ranges ``[first, last)`` of each window; always moves forward."""
    count = len(spans)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + size, count)
        if end < count:
            end = _best_cut(text, spans, start + max(1, size // 2), end)
        windows.append((start, end))
        if end >= count:
            break
        following = max(end - overlap, start + 1)
        if overlap:
            following = _snap_to_sentence(text, spans, following, end)
        start = following
    return windows


def _best_cut(text: str, spans: list[tuple[int, int]], low: int, high: int) -> int:
    """Strongest boundary in ``[low, high]``; the latest one wins ties; ``high`` when none."""
    best, best_strength = high, 0
    for index in range(high, low - 1, -1):
        strength = _break_strength(text, spans, index)
        if strength > best_strength:
            best, best_strength = index, strength
            if strength == 3:
                break
    return best


def _snap_to_sentence(text: str, spans: list[tuple[int, int]], low: int, high: int) -> int:
    """First sentence/paragraph start in ``[low, high)``, else ``low``."""
    for index in range(low, high):
        if _break_strength(text, spans, index) >= 2:
            return index
    return low


def _break_strength(text: str, spans: list[tuple[int, int]], index: int) -> int:
    """Quality of a cut before token ``index``: 3 paragraph, 2 sentence, 1 line break, 0 none."""
    if index <= 0 or index >= len(spans):
        return 0
    prev_end = spans[index - 1][1]
    next_start = spans[index][0]
    if next_start <= prev_end:
        return 0
    gap = text[prev_end:next_start]
    newlines = gap.count("\n")
    if newlines >= 2:
        return 3
    last = text[prev_end - 1]
    ends_sentence = last in _SENTENCE_END or (
        last in _CLOSERS and prev_end >= 2 and text[prev_end - 2] in _SENTENCE_END)
    if ends_sentence and gap.isspace():
        return 2
    return 1 if newlines else 0


# ============================================================================
# extraction
# ============================================================================


class ExtractionError(Exception):
    """A file that cannot be indexed; ``reason`` is shown to the user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def extract_segments(path: Path, stop_event: threading.Event | None = None) -> list[tuple[str, int | None]]:
    """Return ``[(text, page)]`` for a supported file; ``page`` is 1-based for PDFs, else ``None``.

    Raises :class:`ExtractionError` (with a user-facing reason) for anything
    that cannot be read, so one odd file never aborts a batch.
    """
    try:
        return _extract(path, stop_event)
    except (ExtractionError, Cancelled, OSError):
        raise
    except Exception as exc:
        raise ExtractionError(f"could not read file ({type(exc).__name__})") from exc


def _extract(path: Path, stop_event: threading.Event | None) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, stop_event)
    if suffix == ".docx":
        return [(_extract_docx(path), None)]
    if suffix in HTML_SUFFIXES:
        try:
            from noelle.runtime.web_search_engine import html_to_text
        except ImportError as exc:  # pragma: no cover - httpx ships with huggingface_hub
            raise ExtractionError("HTML support is unavailable in this environment") from exc
        return [(html_to_text(_read_text(path)), None)]
    if suffix in TEXT_SUFFIXES or (not suffix and _looks_like_text(path)):
        return [(_read_text(path), None)]
    raise ExtractionError(f"unsupported file type ({suffix or 'no extension'})")


def _extract_pdf(path: Path, stop_event: threading.Event | None) -> list[tuple[str, int | None]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError('PDF support needs pypdf (pip install "noelle[rag]")') from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not _try_decrypt(reader):
            raise ExtractionError("encrypted PDF (password required)")
        pages: list[tuple[str, int | None]] = []
        for number, page in enumerate(reader.pages, start=1):
            _check_stop(stop_event)
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # one damaged page should not lose the document
                log.debug("page %d of %s is unreadable (%s)", number, path.name, type(exc).__name__)
                text = ""
            pages.append((text, number))
        return pages
    except (ExtractionError, Cancelled):
        raise
    except Exception as exc:
        raise ExtractionError(f"could not read PDF ({type(exc).__name__})") from exc


def _try_decrypt(reader: Any) -> bool:
    try:
        return bool(reader.decrypt(""))
    except Exception:
        return False


def _extract_docx(path: Path) -> str:
    try:
        import docx
        from docx.table import Table
    except ImportError as exc:
        raise ExtractionError('DOCX support needs python-docx (pip install "noelle[rag]")') from exc
    try:
        document = docx.Document(str(path))
        blocks: list[str] = []
        for item in document.iter_inner_content():  # paragraphs and tables in document order
            text = _docx_table_text(item) if isinstance(item, Table) else item.text.strip()
            if text:
                blocks.append(text)
        return "\n\n".join(blocks)
    except Exception as exc:
        raise ExtractionError(f"could not read DOCX ({type(exc).__name__})") from exc


def _docx_table_text(table: Any) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen: list[Any] = []  # merged cells repeat the same <w:tc>; keep references so identity is stable
        for cell in row.cells:
            element = getattr(cell, "_tc", None)
            if element is not None and any(element is other for other in seen):
                continue
            seen.append(element)
            text = " ".join(cell.text.split())
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16", errors="replace")
    else:
        text = data.decode("utf-8-sig", errors="replace")
    return text.replace("\x00", "")


def _looks_like_text(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(8192)
    return b"\x00" not in head or head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))


# ============================================================================
# engine
# ============================================================================


class RagEngine:
    """Collections of embedded document chunks stored under ``store_dir``."""

    def __init__(self, store_dir: Path | str, *, use_sqlite_vec: bool = True) -> None:
        self.store_dir = Path(store_dir)
        self.db_path = self.store_dir / DB_FILENAME
        self._use_vec = use_sqlite_vec
        self._vec_probe: bool | None = None
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._schema_ready = False
        self._write_lock = threading.RLock()
        self._matrices: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
        self._matrices_lock = threading.Lock()

    # ----------------------------------------------------------- public API
    @property
    def backend(self) -> str:
        """``"sqlite-vec"`` when the extension loads in this environment, else ``"numpy"``."""
        return "sqlite-vec" if self._vec_available() else "numpy"

    def close(self) -> None:
        """Close every connection opened by this engine (threads reconnect lazily)."""
        with self._conn_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._connections.clear()
            self._local = threading.local()

    def list_collections(self) -> list[dict[str, Any]]:
        conn = self._connect(create=False)
        if conn is None:
            return []
        rows = conn.execute("SELECT * FROM collections ORDER BY name COLLATE NOCASE, name").fetchall()
        return [self._describe(row) for row in rows]

    def collection_info(self, name: str) -> dict[str, Any] | None:
        row = self._collection_row(str(name or "").strip())
        return self._describe(row) if row is not None else None

    def list_documents(self, collection: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """Documents per collection (all collections when ``collection`` is None)."""
        conn = self._connect(create=False)
        if conn is None:
            return {}
        sql = ("SELECT c.name AS collection, d.path, d.name, d.size_bytes, d.chunks, d.added_at FROM documents d "
               "JOIN collections c ON c.id = d.collection_id")
        params: tuple[Any, ...] = ()
        if collection:
            sql += " WHERE c.name = ?"
            params = (str(collection).strip(),)
        catalog: dict[str, list[dict[str, Any]]] = {}
        for row in conn.execute(sql + " ORDER BY d.name COLLATE NOCASE, d.path", params).fetchall():
            catalog.setdefault(str(row["collection"]), []).append({
                "path": str(row["path"]), "name": str(row["name"]), "size_bytes": int(row["size_bytes"] or 0),
                "chunks": int(row["chunks"] or 0), "added_at": row["added_at"]})
        return catalog

    def ingest(self, collection: str, files: Sequence[str], embedder: Any, *, embedding_model: str,
               chunk_tokens: int = 512, overlap: int = 64, progress: ProgressFn | None = None,
               stop_event: threading.Event | None = None) -> dict[str, Any]:
        """Extract, chunk, embed and store ``files`` in ``collection`` (created on first store).

        Progress dicts are ``{"stage", "file", "current", "total"}``: for
        ``extract``/``chunk``/``store`` the counters are file number / file
        count, for ``embed`` chunks embedded / chunks in the file. Each file is
        stored atomically, so a cancel keeps finished files and leaves the one
        in flight (including any previous version of it) untouched.
        """
        started = time.perf_counter()
        name = _clean_name(collection)
        model = str(embedding_model or "").strip()
        if not model:
            raise BadRequest("An embedding model is required to build a collection.")
        dimension = int(embedder.dimension)
        existing = self._collection_row(name)
        if existing is not None:
            _check_compatible(existing, model, dimension)
        tokenizer = _safe_attr(embedder, "tokenizer")
        size = effective_chunk_tokens(embedder, chunk_tokens)
        lap = max(0, min(int(overlap or 0), size // 2))
        emit = progress or _no_progress
        added = replaced = chunks_added = 0
        skipped: list[dict[str, str]] = []
        total = len(files)
        for number, raw in enumerate(files, start=1):
            _check_stop(stop_event)
            path = _resolve(raw)
            label = path.name or str(raw)
            emit({"stage": "extract", "file": label, "current": number, "total": total})
            try:
                if path.is_dir():
                    raise ExtractionError("is a folder; add the files inside it")
                if not path.is_file():
                    raise ExtractionError("file not found")
                size_bytes = path.stat().st_size
                if size_bytes > MAX_FILE_BYTES:
                    raise ExtractionError(f"file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
                digest = _sha256(path)
                if self._stored_digest(name, path) == digest:
                    raise ExtractionError("unchanged")
                segments = extract_segments(path, stop_event)
                emit({"stage": "chunk", "file": label, "current": number, "total": total})
                pieces = chunk_segments(segments, tokenizer, chunk_tokens=size, overlap=lap)
                if not pieces:
                    scanned = " (scanned PDF?)" if path.suffix.lower() == ".pdf" else ""
                    raise ExtractionError(f"no extractable text{scanned}")
            except ExtractionError as exc:
                skipped.append({"path": str(raw), "reason": exc.reason})
                continue
            except OSError as exc:
                reason = f"could not read file ({exc.strerror or type(exc).__name__})"
                skipped.append({"path": str(raw), "reason": reason})
                continue
            vectors = self._embed([chunk.text for chunk, _ in pieces], embedder, dimension, label, emit, stop_event)
            _check_stop(stop_event)
            emit({"stage": "store", "file": label, "current": number, "total": total})
            if self._store(name, model, dimension, path, digest, size_bytes, pieces, vectors):
                replaced += 1
            added += 1
            chunks_added += len(pieces)
            log.info("indexed %s into collection '%s' (%d chunks)", label, name, len(pieces))
        info = self.collection_info(name)
        return {
            "collection": name,
            "documents_added": added,
            "documents_replaced": replaced,
            "chunks_added": chunks_added,
            "skipped": skipped,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "backend": info["backend"] if info else self.backend,
        }

    def query(self, collection: str, text: str, embedder: Any, *, top_k: int = 4) -> list[dict[str, Any]]:
        """Return the ``top_k`` chunks most similar to ``text`` (cosine similarity, best first)."""
        question = str(text or "").strip()
        if not question:
            return []
        name = str(collection or "").strip()
        conn = self._connect(create=False)
        row = None
        if conn is not None:
            row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise BadRequest(f"Collection '{name}' does not exist.")
        vector = np.asarray(embedder.encode([question], normalize=True), dtype=np.float32).reshape(-1)
        if vector.shape[0] != int(row["dimension"]):
            raise BadRequest(f"The embedding model does not match collection '{name}'.",
                             hint=f"It was built with {row['embedding_model']} ({row['dimension']}-d vectors).")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0 or int(row["chunks"]) == 0:
            return []
        vector = vector / norm
        k = max(1, min(int(top_k or 1), MAX_TOP_K))
        if self._vec_on_thread() and self._index_ready(conn, row):
            scored = self._search_index(conn, int(row["id"]), vector, k)
        else:
            scored = self._search_numpy(conn, row, vector, k)
        return self._hits(conn, scored)[:k]

    def delete_collection(self, name: str) -> bool:
        cleaned = str(name or "").strip()
        with self._write_lock:
            conn = self._connect(create=False)
            if conn is None or not cleaned:
                return False
            with _transaction(conn):
                row = conn.execute("SELECT id FROM collections WHERE name = ?", (cleaned,)).fetchone()
                if row is None:
                    return False
                cid = int(row["id"])
                if self._vec_on_thread():  # otherwise the orphan index is dropped on a later open
                    conn.execute(f"DROP TABLE IF EXISTS {_vec_table(cid)}")
                conn.execute("DELETE FROM chunks WHERE collection_id = ?", (cid,))
                conn.execute("DELETE FROM documents WHERE collection_id = ?", (cid,))
                conn.execute("DELETE FROM collections WHERE id = ?", (cid,))
        with self._matrices_lock:
            self._matrices.pop(cid, None)
        log.info("deleted collection '%s'", cleaned)
        return True

    def delete_document(self, collection: str, path: str) -> bool:
        name = str(collection or "").strip()
        candidates = (str(_resolve(path)), str(path))
        with self._write_lock:
            conn = self._connect(create=False)
            if conn is None or not name:
                return False
            with _transaction(conn):
                row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
                if row is None:
                    return False
                cid = int(row["id"])
                doc = conn.execute("SELECT id FROM documents WHERE collection_id = ? AND path IN (?, ?)",
                                   (cid, *candidates)).fetchone()
                if doc is None:
                    return False
                indexed = self._index_for_write(conn, row, build=False)
                self._remove_document(conn, cid, int(doc["id"]), indexed)
                self._bump(conn, cid)
        return True

    @staticmethod
    def format_context(hits: Sequence[dict[str, Any]], budget_chars: int, start_index: int = 1) -> str:
        """Numbered ``[n] (source, p. X)`` blocks; block ``i`` is numbered ``start_index + i``.

        Whole blocks are added while they fit in ``budget_chars``; the first one
        that does not fit is truncated when a useful amount of room is left, and
        the rest are dropped (never skipped, so numbering stays positional).
        """
        budget = int(budget_chars or 0)
        blocks: list[str] = []
        used = 0
        for position, hit in enumerate(hits):
            source = str(hit.get("source") or Path(str(hit.get("path") or "")).name or "document")
            page = hit.get("page")
            label = f"{source}, p. {page}" if page not in (None, "") else source
            header = f"[{start_index + position}] ({label})"
            body = str(hit.get("text") or "").strip()
            separator = 2 if blocks else 0
            block = f"{header}\n{body}"
            if used + separator + len(block) <= budget:
                blocks.append(block)
                used += separator + len(block)
                continue
            room = budget - used - separator - len(header) - 1
            if room >= MIN_TRUNCATED_CHARS:
                blocks.append(f"{header}\n{_truncate(body, room)}")
            break
        return "\n\n".join(blocks)

    # ------------------------------------------------------------ ingestion
    def _embed(self, texts: list[str], embedder: Any, dimension: int, label: str, emit: ProgressFn,
               stop_event: threading.Event | None) -> np.ndarray:
        vectors = np.empty((len(texts), dimension), dtype=np.float32)
        for begin in range(0, len(texts), EMBED_BATCH):
            _check_stop(stop_event)
            batch = texts[begin:begin + EMBED_BATCH]
            encoded = np.asarray(embedder.encode(batch, normalize=True), dtype=np.float32)
            if encoded.shape != (len(batch), dimension):
                raise BadRequest(f"The embedding model returned vectors of shape {tuple(encoded.shape)}; "
                                 f"expected ({len(batch)}, {dimension}).")
            vectors[begin:begin + len(batch)] = encoded
            emit({"stage": "embed", "file": label, "current": begin + len(batch), "total": len(texts)})
        return _normalize_rows(vectors)

    def _store(self, name: str, model: str, dimension: int, path: Path, digest: str, size_bytes: int,
               pieces: list[tuple[TextChunk, int | None]], vectors: np.ndarray) -> bool:
        """Insert one document atomically, replacing an older version of the same path."""
        blobs = [row.astype("<f4", copy=False).tobytes() for row in vectors]
        now = time.time()
        with self._write_lock:
            conn = self._connect(create=True)
            assert conn is not None
            with _transaction(conn):
                row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO collections (name, embedding_model, dimension, created_at, updated_at) "
                                 "VALUES (?, ?, ?, ?, ?)", (name, model, dimension, now, now))
                    row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
                else:
                    _check_compatible(row, model, dimension)
                cid = int(row["id"])
                indexed = self._index_for_write(conn, row, build=True)
                old = conn.execute("SELECT id FROM documents WHERE collection_id = ? AND path = ?",
                                   (cid, str(path))).fetchone()
                if old is not None:
                    self._remove_document(conn, cid, int(old["id"]), indexed)
                cursor = conn.execute(
                    "INSERT INTO documents (collection_id, path, name, sha256, size_bytes, chunks, added_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", (cid, str(path), path.name, digest, size_bytes, len(pieces), now))
                doc_id = int(cursor.lastrowid or 0)
                conn.executemany(
                    "INSERT INTO chunks (collection_id, document_id, chunk_index, page, text, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [(cid, doc_id, index, page, chunk.text, blob)
                     for index, ((chunk, page), blob) in enumerate(zip(pieces, blobs))])
                if indexed:
                    ids = [r[0] for r in conn.execute(
                        "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,))]
                    conn.executemany(f"INSERT INTO {_vec_table(cid)} (rowid, embedding) VALUES (?, ?)",
                                     zip(ids, blobs))
                self._bump(conn, cid, now)
        return old is not None

    def _remove_document(self, conn: sqlite3.Connection, cid: int, doc_id: int, indexed: bool) -> None:
        if indexed:
            ids = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,)).fetchall()
            conn.executemany(f"DELETE FROM {_vec_table(cid)} WHERE rowid = ?", [(r[0],) for r in ids])
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    @staticmethod
    def _bump(conn: sqlite3.Connection, cid: int, now: float | None = None) -> None:
        conn.execute(
            "UPDATE collections SET "
            "documents = (SELECT COUNT(*) FROM documents WHERE collection_id = :cid), "
            "chunks = (SELECT COUNT(*) FROM chunks WHERE collection_id = :cid), "
            "revision = revision + 1, updated_at = :now WHERE id = :cid",
            {"cid": cid, "now": now if now is not None else time.time()})

    def _stored_digest(self, name: str, path: Path) -> str | None:
        conn = self._connect(create=False)
        if conn is None:
            return None
        row = conn.execute(
            "SELECT d.sha256 FROM documents d JOIN collections c ON c.id = d.collection_id "
            "WHERE c.name = ? AND d.path = ?", (name, str(path))).fetchone()
        return str(row[0]) if row else None

    # --------------------------------------------------------------- search
    def _index_for_write(self, conn: sqlite3.Connection, row: sqlite3.Row, *, build: bool) -> bool:
        """Whether the collection's vec0 index must be updated in this transaction.

        Without the extension a synced index is marked stale (rebuilt later from
        the BLOBs). With it, a missing or stale index is rebuilt when ``build``.
        """
        cid, state = int(row["id"]), int(row["vec_state"])
        if not self._vec_on_thread():
            if state == _VEC_SYNCED:
                conn.execute("UPDATE collections SET vec_state = ? WHERE id = ?", (_VEC_STALE, cid))
            return False
        if state == _VEC_SYNCED:
            return True
        if not build:
            return False
        table = _vec_table(cid)
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0("
                         f"embedding float[{int(row['dimension'])}] distance_metric=cosine)")
            conn.execute(f"INSERT INTO {table} (rowid, embedding) "
                         "SELECT id, embedding FROM chunks WHERE collection_id = ?", (cid,))
        except sqlite3.Error as exc:
            log.warning("sqlite-vec index for collection %d unavailable (%s); using numpy search", cid, exc)
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            except sqlite3.Error:
                pass
            return False
        conn.execute("UPDATE collections SET vec_state = ? WHERE id = ?", (_VEC_SYNCED, cid))
        return True

    def _index_ready(self, conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
        if int(row["vec_state"]) == _VEC_SYNCED:
            return True
        with self._write_lock:
            with _transaction(conn):
                fresh = conn.execute("SELECT * FROM collections WHERE id = ?", (int(row["id"]),)).fetchone()
                return fresh is not None and self._index_for_write(conn, fresh, build=True)

    @staticmethod
    def _search_index(conn: sqlite3.Connection, cid: int, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        rows = conn.execute(f"SELECT rowid, distance FROM {_vec_table(cid)} WHERE embedding MATCH ? AND k = ?",
                            (vector.astype("<f4").tobytes(), k)).fetchall()
        return [(int(r[0]), 1.0 - float(r[1]) if r[1] is not None else 0.0) for r in rows]

    def _search_numpy(self, conn: sqlite3.Connection, row: sqlite3.Row, vector: np.ndarray,
                      k: int) -> list[tuple[int, float]]:
        cid, revision, dimension = int(row["id"]), int(row["revision"]), int(row["dimension"])
        with self._matrices_lock:
            cached = self._matrices.get(cid)
        if cached is None or cached[0] != revision:
            ids: list[int] = []
            blobs: list[bytes] = []
            for chunk_id, blob in conn.execute("SELECT id, embedding FROM chunks WHERE collection_id = ? ORDER BY id",
                                               (cid,)):
                ids.append(int(chunk_id))
                blobs.append(blob)
            matrix = np.frombuffer(b"".join(blobs), dtype="<f4").astype(np.float32).reshape(len(ids), dimension)
            cached = (revision, np.asarray(ids, dtype=np.int64), _normalize_rows(matrix))
            with self._matrices_lock:
                self._matrices[cid] = cached
        _, ids_array, matrix = cached
        if not len(ids_array):
            return []
        scores = matrix @ vector
        k = min(k, len(ids_array))
        top = np.argpartition(-scores, k - 1)[:k]
        return [(int(ids_array[i]), float(scores[i])) for i in top]

    @staticmethod
    def _hits(conn: sqlite3.Connection, scored: list[tuple[int, float]]) -> list[dict[str, Any]]:
        if not scored:
            return []
        placeholders = ",".join("?" * len(scored))
        details = {int(r["id"]): r for r in conn.execute(
            "SELECT c.id, c.chunk_index, c.page, c.text, d.path, d.name FROM chunks c "
            f"JOIN documents d ON d.id = c.document_id WHERE c.id IN ({placeholders})",
            [chunk_id for chunk_id, _ in scored])}
        hits: list[dict[str, Any]] = []
        for chunk_id, score in scored:
            detail = details.get(chunk_id)
            if detail is None:
                continue
            hits.append({
                "source": str(detail["name"]),
                "path": str(detail["path"]),
                "chunk_id": int(chunk_id),
                "chunk_index": int(detail["chunk_index"]),
                "page": int(detail["page"]) if detail["page"] is not None else None,
                "text": str(detail["text"]),
                "score": round(float(score), 6),
            })
        hits.sort(key=lambda hit: (-hit["score"], hit["chunk_id"]))
        return hits

    # ---------------------------------------------------------- connections
    def _vec_available(self) -> bool:
        if self._vec_probe is None:
            available = False
            if self._use_vec:
                probe = sqlite3.connect(":memory:")
                try:
                    available = _load_sqlite_vec(probe)
                finally:
                    probe.close()
                if not available:
                    log.info("sqlite-vec is unavailable; RAG search uses the numpy fallback")
            self._vec_probe = available
        return self._vec_probe

    def _vec_on_thread(self) -> bool:
        return bool(getattr(self._local, "vec", False))

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not create and not self.db_path.exists():
            return None
        self.store_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        vec = self._vec_available() and _load_sqlite_vec(conn)
        with self._conn_lock:
            if not self._schema_ready:
                self._init_schema(conn, vec)
                self._schema_ready = True
            self._connections.append(conn)
            self._local.conn = conn
            self._local.vec = vec
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection, vec: bool) -> None:
        version = None
        try:
            found = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            version = found[0] if found else None
        except sqlite3.OperationalError:
            pass
        if version != str(SCHEMA_VERSION):
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                         (str(SCHEMA_VERSION),))
        if vec:  # drop indexes of collections deleted by a process that could not load the extension
            live = {int(r[0]) for r in conn.execute("SELECT id FROM collections")}
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'").fetchall()
            for (table,) in tables:
                match = _VEC_TABLE_RE.fullmatch(table)
                if match and int(match.group(1)) not in live:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")

    def _collection_row(self, name: str) -> sqlite3.Row | None:
        conn = self._connect(create=False)
        if conn is None or not name:
            return None
        return conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()

    def _describe(self, row: sqlite3.Row) -> dict[str, Any]:
        indexed = self._vec_available() and int(row["vec_state"]) == _VEC_SYNCED
        return {
            "name": str(row["name"]),
            "embedding_model": str(row["embedding_model"]),
            "dimension": int(row["dimension"]),
            "documents": int(row["documents"]),
            "chunks": int(row["chunks"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "backend": "sqlite-vec" if indexed else "numpy",
        }


# ============================================================================
# helpers
# ============================================================================


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        conn.execute("COMMIT")


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception as exc:  # ImportError, missing enable_load_extension, load failure
        log.debug("sqlite-vec could not be loaded: %s", exc)
        return False


def _vec_table(cid: int) -> str:
    return f"vec_chunks_{int(cid)}"


def _check_compatible(row: sqlite3.Row, model: str, dimension: int) -> None:
    if str(row["embedding_model"]) != model or int(row["dimension"]) != int(dimension):
        raise BadRequest(
            f"Collection '{row['name']}' was built with {row['embedding_model']} ({row['dimension']}-d vectors), "
            f"not {model} ({dimension}-d).",
            hint="Index with the same embedding model, or create a new collection for the other model.")


def _clean_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        raise BadRequest("A collection name is required.")
    if len(cleaned) > 200:
        raise BadRequest("Collection names are limited to 200 characters.")
    return cleaned


def _resolve(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:max(0, limit - 1)]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _check_stop(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise Cancelled("Indexing was cancelled.")


def _no_progress(_data: dict[str, Any]) -> None:
    return None
