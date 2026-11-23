"""Knowledge page (RAG Studio) offscreen: bases list, documents, search, ingestion settings and evaluation.

The runtime is a fake that answers RAG commands asynchronously, the way worker events arrive. One test
builds a real RAG store (numpy search, hashed embeddings) under the isolated NOELLE_HOME.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QObject, QTimer, Signal

from noelle.runtime.protocol import Command
from noelle.runtime.retrieval_eval import EvalCase

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"


class FakeHandle(QObject):
    status = Signal(dict)
    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(dict)

    def __init__(self, cmd: Command, payload: dict[str, Any], parent: QObject) -> None:
        super().__init__(parent)
        self.cmd = cmd
        self.payload = payload
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeRuntime(QObject):
    """Stands in for RuntimeClient. RAG_QUERY ignores cancel (like the worker); RAG_INGEST honours it."""

    state_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.state = "ready"
        self.collections: list[dict[str, Any]] = []
        self.hits: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.requests: list[FakeHandle] = []
        self.hold = False  # leave requests unanswered until answer() is called

    @property
    def is_ready(self) -> bool:
        return self.state == "ready"

    def set_state(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)

    def request(self, cmd: Command, payload: dict[str, Any] | None = None) -> FakeHandle:
        handle = FakeHandle(cmd, dict(payload or {}), self)
        self.requests.append(handle)
        if not self.hold:
            QTimer.singleShot(0, lambda: self.answer(handle))
        return handle

    def sent(self, cmd: Command) -> list[dict[str, Any]]:
        return [handle.payload for handle in self.requests if handle.cmd == cmd]

    def answer(self, handle: FakeHandle) -> None:
        payload = handle.payload
        if handle.cmd == Command.RAG_LIST_COLLECTIONS:
            handle.finished.emit({"collections": [dict(item) for item in self.collections]})
        elif handle.cmd == Command.RAG_QUERY:
            hits = self.hits.get((payload["collection"], payload["query"]), [])
            handle.finished.emit({"hits": hits[:int(payload["top_k"])], "latency_ms": 7.0})
        elif handle.cmd == Command.RAG_INGEST:
            if handle.cancelled:
                handle.failed.emit({"code": "CANCELLED", "message": "Indexing was cancelled.", "hint": None})
                return
            files = payload["files"]
            handle.status.emit({"stage": "load", "message": "Loading the embedding model…"})
            for number, path in enumerate(files, start=1):
                handle.progress.emit({"stage": "extract", "file": Path(path).name, "current": number,
                                      "total": len(files)})
            base = next((item for item in self.collections if item["name"] == payload["collection"]), None)
            if base is None:
                base = _base(payload["collection"], 0, 0, payload["embedding_model"])
                self.collections.append(base)
            base["documents"] += len(files)
            base["chunks"] += 3 * len(files)
            handle.finished.emit({"collection": payload["collection"], "documents_added": len(files),
                                  "documents_replaced": 0, "chunks_added": 3 * len(files), "elapsed_ms": 420.0,
                                  "skipped": [{"path": str(Path("/elsewhere") / "scan.pdf"),
                                               "reason": "no extractable text (scanned PDF?)"}],
                                  "backend": "numpy"})
        elif handle.cmd == Command.RAG_DELETE_COLLECTION:
            self.collections = [item for item in self.collections if item["name"] != payload["collection"]]
            handle.finished.emit({"deleted": True})
        elif handle.cmd == Command.RAG_LIST_DOCUMENTS:
            handle.finished.emit({"documents": _store().list_documents(payload.get("collection"))})
        elif handle.cmd == Command.RAG_DELETE_DOCUMENT:
            engine = _store()
            removed = sum(1 for path in payload["paths"] if engine.delete_document(payload["collection"], path))
            handle.finished.emit({"collection": payload["collection"], "removed": removed})
        else:
            handle.failed.emit({"code": "UNSUPPORTED", "message": f"{handle.cmd} is not faked", "hint": None})


def _store() -> Any:
    """The worker answers document commands from the real store; so does the fake."""
    from noelle import paths
    from noelle.runtime.rag_engine import RagEngine

    return RagEngine(paths.vector_store_dir(), use_sqlite_vec=False)


class HashEmbedder:
    """Deterministic hashed bag-of-words vectors for a real RagEngine store."""

    dimension = 64

    def encode(self, texts: list[str], normalize: bool = True, **_kwargs: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dimension] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _base(name: str, documents: int, chunks: int, model: str = MINILM) -> dict[str, Any]:
    now = time.time()
    return {"name": name, "embedding_model": model, "dimension": 384, "documents": documents, "chunks": chunks,
            "created_at": now - 7200, "updated_at": now - 90, "backend": "sqlite-vec"}


def _hit(path: str, page: int | None = None, score: float = 0.5, text: str = "A passage.") -> dict[str, Any]:
    return {"source": Path(path).name, "path": path, "chunk_id": 1, "chunk_index": 0, "page": page, "text": text,
            "score": score}


@pytest.fixture()
def runtime(qapp: Any) -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture()
def make_page(qtbot: Any, runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-hub"))  # never look into the real Hugging Face cache
    from noelle.app import main as appmain
    from noelle.app.ui.pages import knowledge_page

    monkeypatch.setattr(knowledge_page, "is_downloaded", lambda *_args, **_kwargs: True)

    def make() -> Any:
        ctx = appmain.build_context()
        ctx.runtime = runtime
        toasts: list[tuple[str, str]] = []
        ctx.toast = lambda text, kind="info": toasts.append((text, kind))
        ctx.extras["toasts"] = toasts
        widget = knowledge_page.KnowledgePage(ctx)
        qtbot.addWidget(widget)
        widget.resize(1280, 820)
        widget.show()
        return widget

    return make


@pytest.fixture()
def page(make_page: Any) -> Any:
    return make_page()


def _activate(page: Any, qtbot: Any, count: int) -> None:
    page.on_activated()
    qtbot.waitUntil(lambda: page.list.count() == count and page.catalog_loaded, timeout=3000)


# ---------------------------------------------------------------------------- tests


def test_runtime_not_ready_shows_an_empty_state_until_it_is(page: Any, runtime: FakeRuntime, qtbot: Any) -> None:
    runtime.collections = [_base("manuals", 2, 6)]
    runtime.set_state("crashed")
    page.on_activated()
    assert page.stack.currentWidget() is page.offline
    assert page.offline.title.text() == "The model runtime crashed"
    assert not runtime.requests  # nothing is sent to a runtime that is not ready
    runtime.set_state("warming")
    assert page.offline.title.text().startswith("Starting the model runtime")
    runtime.set_state("ready")  # the visible page refreshes by itself
    qtbot.waitUntil(lambda: page.list.count() == 1, timeout=3000)
    assert page.stack.currentWidget() is page.body and page.title.text() == "manuals"


def test_bases_list_shows_counts_models_and_details(page: Any, runtime: FakeRuntime, qtbot: Any) -> None:
    runtime.collections = [_base("manuals", 2, 6), _base("Notes", 1, 3, BGE)]
    _activate(page, qtbot, 2)
    first, second = page.list.item(0).text(), page.list.item(1).text()
    assert first == "manuals\n2 docs · 6 chunks\nall-MiniLM-L6-v2 · updated 1m ago"
    assert second.startswith("Notes\n1 docs · 3 chunks\nbge-small-en-v1.5")
    assert BGE in page.list.item(1).toolTip() and "384-d vectors · sqlite-vec search" in page.list.item(1).toolTip()
    assert page.current_base() == "manuals" and page.title.text() == "manuals"
    assert f"embedding model {MINILM}" in page.info.text() and "updated 1m ago" in page.info.text()
    assert page.embedder_pill.text() == "Embedder ready" and page.embed_banner.isHidden()
    assert page.evaluate.compare.findData("Notes") > 0  # the other base can be compared
    page.select_base("Notes")
    assert page.title.text() == "Notes" and page.evaluate.base_label.text() == "A: Notes"
    assert page.evaluate.compare.findData("manuals") > 0 and page.evaluate.compare.findData("Notes") < 0


def test_new_base_ingests_a_folder_with_its_chunk_settings(page: Any, runtime: FakeRuntime, qtbot: Any,
                                                           tmp_path: Path) -> None:
    _activate(page, qtbot, 0)
    assert page.detail_stack.currentWidget() is page.no_bases
    changed: list[bool] = []
    page.collections_changed.connect(lambda: changed.append(True))
    assert page.create_base("Research", BGE)
    assert not page.create_base("Research", MINILM)  # names are unique
    assert page.current_base() == "Research" and "New · add documents" in page.list.item(0).text()
    assert page.tabs.currentWidget() is page.documents and "No documents yet" in page.documents.hint.text()
    page.tabs.setCurrentWidget(page.settings_tab)
    page.settings_tab.chunk.setValue(256)
    page.settings_tab.overlap.setValue(200)  # capped at half the chunk size
    assert page.base_options("Research") == {"chunk_tokens": 256, "chunk_overlap": 128}
    page.settings_tab.overlap.setValue(32)

    folder = tmp_path / "papers"
    (folder / "sub").mkdir(parents=True)
    (folder / ".git").mkdir()
    for name in ("b.md", "a.pdf", "sub/c.txt", "image.png", ".git/notes.md", ".hidden.md"):
        (folder / name).write_text("text", encoding="utf-8")
    assert page.documents.ingest([folder])
    assert not page.documents.add_files_btn.isEnabled()  # one ingest at a time
    qtbot.waitUntil(lambda: "Added 3 document(s) to 'Research'" in page.documents.status.text(), timeout=3000)
    assert "skipped 1: scan.pdf (no extractable text (scanned PDF?))" in page.documents.status.text()
    [payload] = runtime.sent(Command.RAG_INGEST)
    assert payload["files"] == sorted(str(folder / name) for name in ("a.pdf", "b.md", "sub/c.txt"))
    assert payload["collection"] == "Research" and payload["embedding_model"] == BGE
    assert payload["chunk_tokens"] == 256 and payload["chunk_overlap"] == 32
    qtbot.waitUntil(lambda: not page.drafts and "3 docs · 9 chunks" in page.list.item(0).text(), timeout=3000)
    assert changed and page.documents.add_files_btn.isEnabled()
    assert page.ctx.settings.get("rag.knowledge_bases") == {"Research": {"chunk_tokens": 256, "chunk_overlap": 32}}


def test_ingest_can_be_cancelled(page: Any, runtime: FakeRuntime, qtbot: Any, tmp_path: Path) -> None:
    runtime.collections = [_base("manuals", 2, 6)]
    _activate(page, qtbot, 1)
    doc = tmp_path / "extra.md"
    doc.write_text("more", encoding="utf-8")
    runtime.hold = True
    assert page.documents.ingest([doc])
    assert not page.documents.progress_row.isHidden()
    page.documents.cancel_btn.click()
    [handle] = [item for item in runtime.requests if item.cmd == Command.RAG_INGEST]
    assert handle.cancelled
    runtime.answer(handle)
    assert "was cancelled" in page.documents.status.text() and page.documents.progress_row.isHidden()


def test_search_shows_ranked_hits_and_copies_a_citation(page: Any, runtime: FakeRuntime, qtbot: Any) -> None:
    from PySide6.QtGui import QGuiApplication

    runtime.collections = [_base("manuals", 2, 6)]
    runtime.hits[("manuals", "reset the router")] = [
        _hit("/docs/guide.pdf", 3, 0.91, "Hold the reset button\nfor ten seconds."),
        _hit("/docs/faq.md", None, 0.72, "Resetting clears every setting."),
    ]
    _activate(page, qtbot, 1)
    page.tabs.setCurrentWidget(page.search)
    page.search.top_k.setValue(7)
    page.search.query.setText("reset the router")
    page.search.search_btn.click()
    results = page.search.results
    qtbot.waitUntil(lambda: results.topLevelItemCount() == 2, timeout=3000)
    assert [results.topLevelItem(0).text(column) for column in range(5)] == [
        "1", "0.910", "guide.pdf", "3", "Hold the reset button for ten seconds."]
    assert results.topLevelItem(1).text(3) == "—"
    assert runtime.sent(Command.RAG_QUERY) == [{"collection": "manuals", "query": "reset the router", "top_k": 7}]
    assert page.search.status.text().startswith("2 passage(s) in 7 ms")
    page.search.copy_btn.click()
    assert QGuiApplication.clipboard().text() == (
        "“Hold the reset button for ten seconds.” — guide.pdf, p. 3 (/docs/guide.pdf)")
    assert ("Citation copied", "success") in page.ctx.extras["toasts"]


def test_evaluation_computes_metrics_and_compares_two_bases(page: Any, runtime: FakeRuntime, qtbot: Any,
                                                            tmp_path: Path) -> None:
    from noelle.runtime.retrieval_eval import load_cases

    runtime.collections = [_base("small-chunks", 4, 40), _base("large-chunks", 4, 12, BGE)]
    cases = [{"query": "reset", "relevant": ["guide.pdf"]},
             {"query": "warranty", "relevant": ["warranty.md", "terms"]},
             {"query": "battery", "relevant": ["battery.txt"]}]
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")
    runtime.hits.update({
        ("small-chunks", "reset"): [_hit("/d/faq.md"), _hit("/d/guide.pdf"), _hit("/d/guide.pdf")],
        ("small-chunks", "warranty"): [_hit("/d/notes.txt"), _hit("/d/faq.md"), _hit("/d/warranty.md")],
        ("small-chunks", "battery"): [_hit("/d/faq.md"), _hit("/d/notes.txt"), _hit("/d/guide.pdf")],
        ("large-chunks", "reset"): [_hit("/d/guide.pdf"), _hit("/d/faq.md"), _hit("/d/notes.txt"),
                                    _hit("/d/guide.pdf")],  # beyond k: ignored
        ("large-chunks", "warranty"): [_hit("/d/legal/terms-of-sale.pdf"), _hit("/d/warranty.md"), _hit("/d/faq.md")],
        ("large-chunks", "battery"): [_hit("/d/faq.md"), _hit("/d/BATTERY.TXT")],  # fewer hits than k
    })
    _activate(page, qtbot, 2)
    page.select_base("small-chunks")
    evaluate = page.evaluate
    page.tabs.setCurrentWidget(evaluate)
    assert evaluate.import_cases(case_file) and evaluate.table.rowCount() == 3
    assert evaluate.table.item(1, 1).text() == "warranty.md; terms"
    evaluate.k.setValue(3)
    evaluate.compare.setCurrentIndex(evaluate.compare.findData("large-chunks"))
    evaluate.run_btn.click()
    assert evaluate.running and not evaluate.cancel_btn.isHidden()
    qtbot.waitUntil(lambda: not evaluate.running, timeout=5000)

    queries = runtime.sent(Command.RAG_QUERY)
    assert [(q["collection"], q["query"]) for q in queries] == [
        ("small-chunks", "reset"), ("small-chunks", "warranty"), ("small-chunks", "battery"),
        ("large-chunks", "reset"), ("large-chunks", "warranty"), ("large-chunks", "battery")]
    assert {q["top_k"] for q in queries} == {3}

    log2 = math.log2
    a, b = evaluate.reports["small-chunks"], evaluate.reports["large-chunks"]
    assert a.hit_rate == pytest.approx(2 / 3) and b.hit_rate == pytest.approx(1.0)
    assert a.recall == pytest.approx((1 + 1 / 2 + 0) / 3) and b.recall == pytest.approx(1.0)
    assert a.precision == pytest.approx((2 / 3 + 1 / 3 + 0) / 3)
    assert b.precision == pytest.approx((1 / 3 + 2 / 3 + 1 / 3) / 3)
    assert a.mrr == pytest.approx((1 / 2 + 1 / 3 + 0) / 3) and b.mrr == pytest.approx((1 + 1 + 1 / 2) / 3)
    ndcg_a = (1 / log2(3) + (1 / log2(4)) / (1 + 1 / log2(3)) + 0) / 3
    ndcg_b = (1 + 1 + 1 / log2(3)) / 3
    assert a.ndcg == pytest.approx(ndcg_a) and b.ndcg == pytest.approx(ndcg_b)

    view_a, view_b = evaluate.view_a, evaluate.view_b
    assert not evaluate.results.isHidden() and not view_b.isHidden()
    assert view_a.title.text() == "small-chunks" and view_b.title.text() == "large-chunks"
    assert view_a.tiles["hit_rate"].value.text() == "0.67" and view_a.tiles["hit_rate"].caption.text() == (
        "2 of 3 queries")
    assert view_a.tiles["ndcg"].title.text() == "NDCG@3" and view_a.tiles["mrr"].title.text() == "MRR"
    assert view_b.tiles["mrr"].value.text() == "0.83" and view_b.tiles["mrr"].caption.text() == "+0.56 vs A"
    assert view_b.tiles["hit_rate"].caption.text() == "3 of 3 queries · +0.33 vs A"
    assert [view_a.table.item(row, 6).text() for row in range(3)] == ["#2", "#3", "—"]
    assert [view_b.table.item(row, 1).text() for row in range(3)] == ["yes", "yes", "yes"]
    assert "(relevant)" in view_a.table.item(0, 0).toolTip() and "Not found: battery.txt" in view_a.table.item(
        2, 0).toolTip()
    view_a.table.selectRow(2)
    assert view_b.table.currentRow() == 2  # the two per-case tables stay aligned
    assert evaluate.status.text() == "Evaluated 3 test case(s) at k = 3 on 'small-chunks' and 'large-chunks'."

    exported = tmp_path / "out" / "cases.jsonl"
    exported.parent.mkdir()
    assert evaluate.export_cases(exported)
    assert load_cases(exported) == load_cases(case_file)


def test_evaluation_can_be_cancelled_and_ignores_late_answers(page: Any, runtime: FakeRuntime, qtbot: Any) -> None:
    runtime.collections = [_base("manuals", 2, 6)]
    _activate(page, qtbot, 1)
    evaluate = page.evaluate
    evaluate.set_cases([EvalCase.create("first", ["a.md"]), EvalCase.create("second", ["b.md"])])
    runtime.hold = True
    assert evaluate.run()
    [pending] = [handle for handle in runtime.requests if handle.cmd == Command.RAG_QUERY]
    evaluate.cancel_btn.click()
    assert pending.cancelled and not evaluate.running and evaluate.status.text() == "Evaluation cancelled."
    runtime.answer(pending)  # the runtime still answers the query it had started
    assert len(runtime.sent(Command.RAG_QUERY)) == 1 and evaluate.results.isHidden() and not evaluate.reports


def test_evaluation_failure_keeps_the_finished_base(page: Any, runtime: FakeRuntime, qtbot: Any) -> None:
    runtime.collections = [_base("manuals", 2, 6), _base("other", 1, 2)]
    runtime.hits[("manuals", "q")] = [_hit("/d/a.md")]
    _activate(page, qtbot, 2)
    evaluate = page.evaluate
    evaluate.set_cases([EvalCase.create("q", ["a.md"])])
    evaluate.compare.setCurrentIndex(evaluate.compare.findData("other"))
    runtime.hold = True
    assert evaluate.run()
    runtime.answer(runtime.requests[-1])
    failing = runtime.requests[-1]
    assert failing.payload["collection"] == "other"
    failing.failed.emit({"code": "MODEL_NOT_FOUND", "message": "BAAI/x is not downloaded", "hint": None})
    assert not evaluate.running and "Evaluation stopped: BAAI/x is not downloaded" in evaluate.error.text.text()
    assert set(evaluate.reports) == {"manuals"} and not evaluate.results.isHidden() and evaluate.view_b.isHidden()
    assert evaluate.status.text().endswith("Showing the knowledge base that finished.")


def test_case_problems_are_reported_with_their_line_or_row(page: Any, runtime: FakeRuntime, qtbot: Any,
                                                           tmp_path: Path) -> None:
    from PySide6.QtWidgets import QTableWidgetItem

    runtime.collections = [_base("manuals", 2, 6)]
    _activate(page, qtbot, 1)
    evaluate = page.evaluate
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "ok", "relevant": ["a.md"]}\n\n{"query": "", "relevant": ["b.md"]}\n', encoding="utf-8")
    assert not evaluate.import_cases(bad)
    assert "Line 3: 'query' must be a non-empty string" in evaluate.error.text.text()
    assert evaluate.table.rowCount() == 0
    assert not evaluate.run()
    assert "Add at least one test case" in evaluate.error.text.text()
    evaluate.add_case()
    evaluate.add_case()
    evaluate.table.setItem(1, 0, QTableWidgetItem("orphan query"))
    assert not evaluate.run()  # row 1 is blank and skipped; row 2 has no relevant documents
    assert "Row 2: 'relevant' needs at least one document" in evaluate.error.text.text()
    assert not runtime.sent(Command.RAG_QUERY)


def test_documents_come_from_the_store_and_can_be_removed(page: Any, runtime: FakeRuntime, qtbot: Any,
                                                          tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from noelle import paths
    from noelle.runtime.rag_engine import RagEngine

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    files = []
    for name, text in (("beta.txt", "Beta decay emits electrons."), ("Alpha.md", "Alpha particles are helium.")):
        (corpus / name).write_text(text, encoding="utf-8")
        files.append(str(corpus / name))
    engine = RagEngine(paths.vector_store_dir(), use_sqlite_vec=False)
    try:
        engine.ingest("physics", files, HashEmbedder(), embedding_model=MINILM)
        runtime.collections = engine.list_collections()
    finally:
        engine.close()
    _activate(page, qtbot, 1)
    tree = page.documents.tree
    assert [tree.topLevelItem(row).text(0) for row in range(tree.topLevelItemCount())] == ["Alpha.md", "beta.txt"]
    assert tree.topLevelItem(0).text(1) == "1" and tree.topLevelItem(0).toolTip(0).endswith("Alpha.md")
    assert "2 docs · 2 chunks · 54 B" in page.list.item(0).text()

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)
    assert not page.documents.remove_btn.isEnabled()
    tree.topLevelItem(0).setSelected(True)
    page.documents.remove_btn.click()
    qtbot.waitUntil(lambda: tree.topLevelItemCount() == 1, timeout=5000)
    assert tree.topLevelItem(0).text(0) == "beta.txt"
    assert page.documents.status.text() == "Removed 1 document(s) from 'physics'."
    assert [doc["name"] for doc in _store().list_documents("physics")["physics"]] == ["beta.txt"]
    assert runtime.sent(Command.RAG_DELETE_DOCUMENT)[0]["collection"] == "physics"


def test_deleting_a_base_asks_first(page: Any, runtime: FakeRuntime, qtbot: Any,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    runtime.collections = [_base("manuals", 2, 6), _base("old", 1, 2)]
    _activate(page, qtbot, 2)
    page.set_base_options("old", 128, 16)
    page.select_base("old")
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: answers.pop(0))
    page.documents.delete_btn.click()
    assert not runtime.sent(Command.RAG_DELETE_COLLECTION)
    page.documents.delete_btn.click()
    qtbot.waitUntil(lambda: page.list.count() == 1, timeout=3000)
    assert runtime.sent(Command.RAG_DELETE_COLLECTION) == [{"collection": "old"}]
    assert "old" not in (page.ctx.settings.get("rag.knowledge_bases") or {})
    assert ("Deleted the knowledge base 'old'.", "success") in page.ctx.extras["toasts"]


def test_missing_embedder_is_offered_for_download(page: Any, runtime: FakeRuntime, qtbot: Any,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from noelle.app.ui.pages import knowledge_page

    monkeypatch.setattr(knowledge_page, "is_downloaded", lambda *_args, **_kwargs: False)
    started: list[tuple[str, str]] = []
    monkeypatch.setattr(page.ctx.downloads, "start",
                        lambda repo_id, **kwargs: started.append((repo_id, kwargs["summary"].pipeline_tag)))
    runtime.collections = [_base("manuals", 2, 6, BGE)]
    _activate(page, qtbot, 1)
    assert page.embed_banner.isVisible() and page.embedder_pill.text() == "Embedder missing"
    assert BGE in page.embed_banner.text.text()
    page.download_btn.click()
    assert started == [(BGE, "sentence-similarity")]
    page.search.query.setText("anything")
    page.search.run_search()
    assert "Download the embedding model" in page.search.status.text() and not runtime.sent(Command.RAG_QUERY)

    dialog = knowledge_page.NewBaseDialog(page)
    qtbot.addWidget(dialog)
    assert dialog.pill.text() == "Not downloaded" and not dialog.download_btn.isHidden()
    dialog.name.setText("manuals")
    assert not dialog.create_btn.isEnabled() and "already exists" in dialog.problem.text()
    dialog.name.setText("  fresh ")
    assert dialog.create_btn.isEnabled() and dialog.values() == ("fresh", MINILM)
    dialog.download_btn.click()
    assert started[-1] == (MINILM, "sentence-similarity")


@pytest.mark.parametrize("palette", ["light", "dark"])
def test_page_uses_the_active_palette(make_page: Any, runtime: FakeRuntime, qtbot: Any, palette: str) -> None:
    from noelle.app import theme

    before = theme.NAME
    theme.use(palette)
    try:
        runtime.collections = [_base("manuals", 2, 6)]
        page = make_page()
        _activate(page, qtbot, 1)
        assert theme.SUCCESS in page.embedder_pill.styleSheet()
    finally:
        theme.use(before)


def test_store_helpers() -> None:
    from noelle.app.ui.pages import knowledge_page

    assert knowledge_page.split_relevant(" a.md ; ;b/c.pdf\nd.txt ") == ["a.md", "b/c.pdf", "d.txt"]
    assert knowledge_page.format_citation({"source": "notes.md", "text": "x" * 400}).endswith("…” — notes.md")
    assert knowledge_page.snippet("  a \n b  ") == "a b"
