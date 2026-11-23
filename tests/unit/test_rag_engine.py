"""RagEngine: chunking, extraction, sqlite-vec/numpy storage, querying and context formatting.

No network and no model downloads: a deterministic hashed bag-of-words embedder
and a tiny offset-aware tokenizer stand in for sentence-transformers.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from noelle.runtime.errors import BadRequest, Cancelled
from noelle.runtime.rag_engine import RagEngine, chunk_segments, chunk_text, effective_chunk_tokens

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class FakeTokenizer:
    """Mimics a Hugging Face fast tokenizer: one token per word or punctuation mark, with char offsets."""

    def __call__(self, text: Any, add_special_tokens: bool = False, return_offsets_mapping: bool = False,
                 **_kwargs: Any) -> dict[str, Any]:
        if isinstance(text, list):
            return {"input_ids": [self(item)["input_ids"] for item in text]}
        spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
        encoded: dict[str, Any] = {"input_ids": [zlib.crc32(text[s:e].encode()) % 30_000 for s, e in spans]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = spans
        return encoded

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors (zlib.crc32, never the per-process salted ``hash``)."""

    def __init__(self, dim: int = 1024, tokenizer: Any = None, max_seq_length: int = 512) -> None:
        self.dim = dim
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @property
    def dimension(self) -> int:
        return self.dim

    def encode(self, texts: list[str], normalize: bool = True, **_kwargs: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        if normalize:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal valid PDF (Helvetica text, correct xref) with one text page per entry."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(len(pages)))}] "
               f"/Count {len(pages)} >>".encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    for index, text in enumerate(pages):
        lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in text.split("\n")]
        stream = "\n".join(["BT", "/F1 12 Tf", "14 TL", "72 720 Td", *(f"({line}) Tj T*" for line in lines),
                            "ET"]).encode("latin-1")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> "
                       f">> /Contents {5 + 2 * index} 0 R >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def write_docx(path: Path) -> Path:
    import docx

    document = docx.Document()
    document.add_heading("Quarterly report", level=1)
    document.add_paragraph("Revenue from turbines grew strongly in the third quarter.")
    table = document.add_table(rows=2, cols=2)
    for row, values in zip(table.rows, (("Region", "Sales"), ("Nordics", "Windmills"))):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.save(str(path))
    return path


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(tokenizer=FakeTokenizer())


@pytest.fixture
def engine(tmp_path: Path):
    rag = RagEngine(tmp_path / "vector_store")
    yield rag
    rag.close()


@pytest.fixture
def corpus(tmp_path: Path) -> list[str]:
    docs = tmp_path / "docs"
    docs.mkdir()
    texts = {
        "capybara.txt": "The capybara is the largest living rodent. It lives near rivers in South America.",
        "sourdough.md": "# Sourdough\n\nSourdough bread rises with a fermented starter of flour and water.",
        "volcano.txt": "Volcanoes erupt molten lava and ash when magma rises from the mantle.",
        "penguin.txt": "Emperor penguins huddle together through the long Antarctic winter.",
    }
    for name, text in texts.items():
        (docs / name).write_text(text, encoding="utf-8")
    return [str(docs / name) for name in texts]


def need_sqlite_vec() -> None:
    pytest.importorskip("sqlite_vec")
    if not RagEngine(Path("unused")).backend == "sqlite-vec":
        pytest.skip("sqlite-vec cannot be loaded in this interpreter")


# ============================================================================
# chunking
# ============================================================================


def test_chunks_are_exact_slices_within_window_with_overlap() -> None:
    tokenizer = FakeTokenizer()
    text = " ".join(f"Sentence number {i} talks about topic{i} in detail." for i in range(40))
    chunks = chunk_text(text, tokenizer, chunk_tokens=30, overlap=8)
    assert len(chunks) > 5
    for chunk in chunks:
        assert text[chunk.start:chunk.end] == chunk.text
        assert len(tokenizer(chunk.text)["input_ids"]) <= 30
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.start < current.start < previous.end  # forward progress with overlap
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    assert all(chunk.text.endswith(".") for chunk in chunks)  # cuts land on sentence ends


def test_chunks_prefer_paragraph_breaks() -> None:
    paragraphs = [" ".join([word] * 12) + "." for word in ("alpha", "beta", "gamma")]
    chunks = chunk_text("\n\n".join(paragraphs), FakeTokenizer(), chunk_tokens=20, overlap=0)
    assert [chunk.text for chunk in chunks] == paragraphs


def test_whitespace_fallback_without_tokenizer() -> None:
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), None, chunk_tokens=20, overlap=4)  # ~0.75 words per token
    assert all(len(chunk.text.split()) <= 15 for chunk in chunks)
    assert chunks[0].text.split() == words[:15]
    assert chunks[1].text.split()[:3] == words[12:15]
    assert chunks[-1].text.split()[-1] == "w99"


def test_chunker_always_moves_forward() -> None:
    chunks = chunk_text("word " * 50, FakeTokenizer(), chunk_tokens=1, overlap=5)
    assert len(chunks) == 50
    assert chunk_text("   \n  ", FakeTokenizer()) == []


def test_segments_keep_their_page_numbers() -> None:
    segments = [("Introduction to astronomy and telescopes.", 1), ("", 2),
                ("Neutron stars are extremely dense stellar remnants.", 3)]
    pieces = chunk_segments(segments, FakeTokenizer(), chunk_tokens=12, overlap=2)
    assert [page for _, page in pieces] == [1, 3]
    assert pieces[1][0].text.startswith("telescopes.")  # overlap reaches back onto page 1; label follows new text

    tiny_pages = [(f"Page {number} discusses topic {number}.", number) for number in range(1, 21)]
    pieces = chunk_segments(tiny_pages, FakeTokenizer(), chunk_tokens=30, overlap=5)
    labels = [page for _, page in pieces]
    assert labels[0] == 1 and labels == sorted(set(labels)) and len(labels) > 3
    for (previous, _), (chunk, page) in zip(pieces, pieces[1:]):
        assert f"Page {page} discusses" in chunk.text and f"Page {page} discusses" not in previous.text


def test_chunk_size_is_capped_by_the_embedder() -> None:
    assert effective_chunk_tokens(FakeEmbedder(tokenizer=FakeTokenizer(), max_seq_length=256), 512) == 254
    assert effective_chunk_tokens(FakeEmbedder(max_seq_length=128), 512) == 128
    assert effective_chunk_tokens(FakeEmbedder(tokenizer=FakeTokenizer()), 100) == 100


# ============================================================================
# ingest + query
# ============================================================================


def test_ingest_supported_formats_and_query(engine: RagEngine, embedder: FakeEmbedder, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The capybara is the largest living rodent on earth.", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Sourdough\n\nSourdough bread rises with a fermented starter.",
                                       encoding="utf-8")
    (tmp_path / "page.html").write_text(
        "<html><body><nav>Menu Login</nav><article><h1>Coral reefs</h1><p>Coral reefs shelter a quarter of "
        "marine species.</p></article><footer>Copyright</footer></body></html>", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    write_docx(tmp_path / "report.docx")
    write_pdf(tmp_path / "paper.pdf", ["Introduction to astronomy and telescopes. Optics gather faint light "
                                       "from distant galaxies.",
                                       "Neutron stars are extremely dense stellar remnants."])
    names = ["notes.txt", "guide.md", "page.html", "report.docx", "paper.pdf", "missing.txt", "image.png"]
    events: list[dict[str, Any]] = []

    result = engine.ingest("docs", [str(tmp_path / n) for n in names], embedder, embedding_model="fake-model",
                           chunk_tokens=16, overlap=2, progress=events.append)

    assert result["collection"] == "docs"
    assert result["documents_added"] == 5
    assert result["chunks_added"] >= 5
    reasons = {Path(item["path"]).name: item["reason"] for item in result["skipped"]}
    assert reasons["missing.txt"] == "file not found"
    assert "unsupported" in reasons["image.png"]
    assert {event["stage"] for event in events} == {"extract", "chunk", "embed", "store"}
    assert all(set(event) == {"stage", "file", "current", "total"} for event in events)

    hit = engine.query("docs", "neutron stars dense remnants", embedder, top_k=3)[0]
    assert set(hit) == {"source", "path", "chunk_id", "chunk_index", "page", "text", "score"}
    assert hit["source"] == "paper.pdf" and hit["page"] == 2 and "Neutron" in hit["text"]
    assert hit["path"] == str((tmp_path / "paper.pdf").resolve())
    assert engine.query("docs", "capybara rodent", embedder)[0]["source"] == "notes.txt"
    assert engine.query("docs", "sourdough starter", embedder)[0]["source"] == "guide.md"
    html_hit = engine.query("docs", "coral reefs marine species", embedder)[0]
    assert html_hit["source"] == "page.html" and "Menu" not in html_hit["text"]
    docx_hit = engine.query("docs", "Nordics windmills", embedder)[0]
    assert docx_hit["source"] == "report.docx" and "Nordics | Windmills" in docx_hit["text"]

    hits = engine.query("docs", "capybara rodent", embedder, top_k=4)
    assert len(hits) == 4
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert hits[0]["page"] is None and isinstance(hits[0]["score"], float)
    info = engine.collection_info("docs")
    assert info is not None and info["documents"] == 5 and info["embedding_model"] == "fake-model"
    assert info["dimension"] == embedder.dim and info["created_at"] <= info["updated_at"]


def test_reingest_skips_unchanged_and_replaces_changed(engine: RagEngine, embedder: FakeEmbedder,
                                                       tmp_path: Path) -> None:
    doc = tmp_path / "facts.txt"
    doc.write_text("Volcanoes erupt molten lava from the mantle.", encoding="utf-8")
    assert engine.ingest("c", [str(doc)], embedder, embedding_model="fake")["documents_added"] == 1

    again = engine.ingest("c", [str(doc)], embedder, embedding_model="fake")
    assert again["documents_added"] == 0
    assert again["skipped"] == [{"path": str(doc), "reason": "unchanged"}]

    doc.write_text("Glaciers carve deep valleys out of solid granite.", encoding="utf-8")
    changed = engine.ingest("c", [str(doc)], embedder, embedding_model="fake")
    assert changed["documents_added"] == 1 and changed["documents_replaced"] == 1
    info = engine.collection_info("c")
    assert info is not None and info["documents"] == 1 and info["chunks"] == changed["chunks_added"]
    assert all("Volcanoes" not in hit["text"] for hit in engine.query("c", "volcanoes lava", embedder, top_k=10))
    assert engine.query("c", "glaciers granite", embedder)[0]["text"].startswith("Glaciers")


def test_sqlite_vec_and_numpy_backends_agree(tmp_path: Path, embedder: FakeEmbedder, corpus: list[str]) -> None:
    need_sqlite_vec()
    store = tmp_path / "store"
    vec_engine = RagEngine(store)
    numpy_engine = RagEngine(store, use_sqlite_vec=False)
    try:
        result = vec_engine.ingest("kb", corpus, embedder, embedding_model="fake", chunk_tokens=16, overlap=4)
        assert result["backend"] == "sqlite-vec"
        assert vec_engine.backend == "sqlite-vec" and numpy_engine.backend == "numpy"
        assert vec_engine.list_collections()[0]["backend"] == "sqlite-vec"
        assert numpy_engine.collection_info("kb")["backend"] == "numpy"  # type: ignore[index]
        for question, source in (("capybara rodent rivers", "capybara.txt"), ("fermented starter", "sourdough.md"),
                                 ("magma mantle lava", "volcano.txt"), ("antarctic penguins", "penguin.txt")):
            via_vec = vec_engine.query("kb", question, embedder, top_k=3)
            via_numpy = numpy_engine.query("kb", question, embedder, top_k=3)
            assert via_vec[0]["source"] == via_numpy[0]["source"] == source
            assert via_vec[0]["chunk_id"] == via_numpy[0]["chunk_id"]
            assert via_vec[0]["score"] == pytest.approx(via_numpy[0]["score"], abs=1e-4)
            assert [set(hit) for hit in via_vec] == [set(hit) for hit in via_numpy]

        # A process without the extension edits the collection: the vec0 index goes stale and is rebuilt.
        extra = tmp_path / "extra.txt"
        extra.write_text("Koalas sleep most of the day in eucalyptus trees.", encoding="utf-8")
        numpy_engine.ingest("kb", [str(extra)], embedder, embedding_model="fake")
        assert vec_engine.collection_info("kb")["backend"] == "numpy"  # type: ignore[index]
        assert vec_engine.query("kb", "koalas eucalyptus", embedder)[0]["source"] == "extra.txt"
        assert vec_engine.collection_info("kb")["backend"] == "sqlite-vec"  # type: ignore[index]

        # A collection built without the extension gets an index on first use.
        numpy_engine.ingest("plain", corpus[:2], embedder, embedding_model="fake")
        assert vec_engine.collection_info("plain")["backend"] == "numpy"  # type: ignore[index]
        assert vec_engine.query("plain", "capybara", embedder)[0]["source"] == "capybara.txt"
        assert vec_engine.collection_info("plain")["backend"] == "sqlite-vec"  # type: ignore[index]

        # Deleting without the extension leaves an orphan vec0 table; the next engine to open drops it.
        assert numpy_engine.delete_collection("plain") is True
        def vec_tables() -> set[str]:
            with closing(sqlite3.connect(store / "rag.db")) as raw:
                return {row[0] for row in raw.execute(
                    "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%'")}
        assert len(vec_tables()) == 2
        reopened = RagEngine(store)
        assert [c["name"] for c in reopened.list_collections()] == ["kb"]
        assert len(vec_tables()) == 1
        reopened.close()
    finally:
        vec_engine.close()
        numpy_engine.close()


def test_numpy_fallback_stands_alone(tmp_path: Path, embedder: FakeEmbedder, corpus: list[str]) -> None:
    engine = RagEngine(tmp_path / "store", use_sqlite_vec=False)
    try:
        result = engine.ingest("kb", corpus, embedder, embedding_model="fake")
        assert result["backend"] == "numpy" and result["documents_added"] == 4
        assert engine.query("kb", "penguins antarctic winter", embedder)[0]["source"] == "penguin.txt"
        with closing(sqlite3.connect(tmp_path / "store" / "rag.db")) as raw:
            assert not raw.execute("SELECT name FROM sqlite_master WHERE name LIKE 'vec_chunks%'").fetchall()
    finally:
        engine.close()


def test_backend_is_sqlite_vec_in_this_environment(engine: RagEngine, embedder: FakeEmbedder,
                                                   corpus: list[str]) -> None:
    need_sqlite_vec()
    assert engine.ingest("kb", corpus, embedder, embedding_model="fake")["backend"] == "sqlite-vec"
    assert [c["backend"] for c in engine.list_collections()] == ["sqlite-vec"]


def test_delete_document_and_collection(engine: RagEngine, embedder: FakeEmbedder, corpus: list[str]) -> None:
    engine.ingest("kb", corpus, embedder, embedding_model="fake")
    assert engine.delete_document("kb", corpus[0]) is True
    assert engine.delete_document("kb", corpus[0]) is False
    info = engine.collection_info("kb")
    assert info is not None and info["documents"] == 3
    assert all(hit["source"] != "capybara.txt" for hit in engine.query("kb", "capybara rodent", embedder, top_k=10))
    assert engine.delete_collection("kb") is True
    assert engine.collection_info("kb") is None and engine.list_collections() == []
    assert engine.delete_collection("kb") is False
    with pytest.raises(BadRequest):
        engine.query("kb", "capybara", embedder)


def test_embedding_model_is_fixed_per_collection(engine: RagEngine, corpus: list[str]) -> None:
    engine.ingest("kb", corpus[:1], FakeEmbedder(dim=64), embedding_model="model-a")
    with pytest.raises(BadRequest, match="model-a"):
        engine.ingest("kb", corpus[1:2], FakeEmbedder(dim=64), embedding_model="model-b")
    with pytest.raises(BadRequest):
        engine.ingest("kb", corpus[1:2], FakeEmbedder(dim=32), embedding_model="model-a")
    with pytest.raises(BadRequest):
        engine.query("kb", "capybara", FakeEmbedder(dim=32))
    info = engine.collection_info("kb")
    assert info is not None and info["documents"] == 1 and info["dimension"] == 64


def test_cancel_mid_file_keeps_previous_state(engine: RagEngine, embedder: FakeEmbedder, corpus: list[str],
                                              tmp_path: Path) -> None:
    stop = threading.Event()

    def cancel_on_embed(event: dict[str, Any]) -> None:
        if event["stage"] == "embed":
            stop.set()

    with pytest.raises(Cancelled):
        engine.ingest("kb", corpus, embedder, embedding_model="fake", progress=cancel_on_embed, stop_event=stop)
    assert engine.collection_info("kb") is None  # nothing stored, collection not created

    doc = Path(corpus[0])
    engine.ingest("kb", [str(doc)], embedder, embedding_model="fake")
    doc.write_text("Completely different text about meteorites.", encoding="utf-8")
    stop.clear()
    with pytest.raises(Cancelled):
        engine.ingest("kb", [str(doc)], embedder, embedding_model="fake", progress=cancel_on_embed, stop_event=stop)
    assert engine.query("kb", "capybara rodent", embedder)[0]["text"].startswith("The capybara")

    stop.set()
    with pytest.raises(Cancelled):
        engine.ingest("kb", corpus, embedder, embedding_model="fake", stop_event=stop)


def test_reads_are_not_blocked_by_writes(engine: RagEngine, embedder: FakeEmbedder, corpus: list[str]) -> None:
    listings: list[list[dict[str, Any]]] = []

    def list_from_other_thread() -> None:
        worker = threading.Thread(target=lambda: listings.append(engine.list_collections()))
        worker.start()
        worker.join(5)
        assert not worker.is_alive(), "list_collections blocked"

    engine.ingest("kb", corpus, embedder, embedding_model="fake",
                  progress=lambda event: list_from_other_thread() if event["stage"] == "embed" else None)
    assert listings[0] == [] and listings[-1][0]["name"] == "kb"

    with closing(sqlite3.connect(engine.db_path, timeout=0.1, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")  # another writer holds the database lock
        writer.execute("UPDATE collections SET updated_at = updated_at")
        list_from_other_thread()
        writer.execute("ROLLBACK")
    assert listings[-1][0]["documents"] == 4


def test_lazy_store_and_blank_queries(tmp_path: Path, embedder: FakeEmbedder, corpus: list[str]) -> None:
    store = tmp_path / "lazy"
    engine = RagEngine(store)
    try:
        assert engine.list_collections() == [] and engine.collection_info("kb") is None
        assert engine.delete_collection("kb") is False and engine.delete_document("kb", corpus[0]) is False
        assert not store.exists()
        with pytest.raises(BadRequest):
            engine.query("kb", "anything", embedder)
        engine.ingest("kb", corpus, embedder, embedding_model="fake")
        assert (store / "rag.db").exists()
        assert engine.query("kb", "   ", embedder) == []
        assert engine.query("kb", "?!", embedder) == []  # zero query vector
        with pytest.raises(BadRequest):
            engine.ingest("  ", corpus, embedder, embedding_model="fake")
    finally:
        engine.close()


def test_format_context_numbering_and_budget() -> None:
    hits = [
        {"source": "a.pdf", "path": "/x/a.pdf", "chunk_id": 1, "chunk_index": 0, "page": 3, "text": "Alpha text.",
         "score": 0.9},
        {"source": "b.txt", "path": "/x/b.txt", "chunk_id": 2, "chunk_index": 0, "page": None, "text": "Beta text.",
         "score": 0.8},
        {"source": "c.txt", "path": "/x/c.txt", "chunk_id": 3, "chunk_index": 0, "page": None, "text": "C" * 1000,
         "score": 0.7},
    ]
    full = RagEngine.format_context(hits, 10_000, start_index=4)
    assert full.startswith("[4] (a.pdf, p. 3)\nAlpha text.\n\n[5] (b.txt)\nBeta text.\n\n[6] (c.txt)\nCCC")
    assert RagEngine.format_context(hits, 60) == "[1] (a.pdf, p. 3)\nAlpha text.\n\n[2] (b.txt)\nBeta text."
    truncated = RagEngine.format_context(hits, 300, start_index=2)
    assert len(truncated) <= 300 and "\n\n[4] (c.txt)\nCCC" in truncated and truncated.endswith("…")
    assert RagEngine.format_context([], 1000) == ""
    assert RagEngine.format_context(hits, 0) == ""
