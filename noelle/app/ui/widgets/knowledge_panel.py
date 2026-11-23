"""Documents tab: local RAG collections (ingest, inspect, test retrieval)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import Banner, SearchField, button, label, time_ago
from noelle.runtime.hub_cache import is_downloaded
from noelle.runtime.protocol import Command

FILE_FILTER = "Documents (*.pdf *.docx *.md *.markdown *.txt *.rst *.csv *.json *.html *.py);;All files (*)"
EMBEDDING_CHOICES = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-m3",
                     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".html", ".htm", ".py"}


class KnowledgePanel(QWidget):
    collections_changed = Signal()

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._collections: list[dict[str, Any]] = []
        self._ingest: Any = None
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)
        root.addWidget(label("Local documents", "h2"))
        root.addWidget(label("Index PDFs, Word, Markdown and text files for grounded answers. Parsing, embedding "
                             "and search all run on this machine; nothing is uploaded.", "muted", wrap=True))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.addWidget(label("COLLECTIONS", "eyebrow"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        left_layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        new_btn = button("New", "plus")
        new_btn.clicked.connect(self._new_collection)
        row.addWidget(new_btn)
        self.delete_btn = button("Delete", "trash", "danger")
        self.delete_btn.clicked.connect(self._delete_collection)
        row.addWidget(self.delete_btn)
        row.addStretch(1)
        left_layout.addLayout(row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(10)
        self.title = label("Create a collection to get started", "h3")
        right_layout.addWidget(self.title)
        self.info = label("", "faint", wrap=True)
        right_layout.addWidget(self.info)

        embed_row = QHBoxLayout()
        embed_row.addWidget(label("Embedding model", "muted"))
        self.embedder = QComboBox()
        self.embedder.setEditable(True)
        default = str(ctx.settings.get("rag.embedding_model"))
        for choice in [default] + [c for c in EMBEDDING_CHOICES if c != default]:
            self.embedder.addItem(choice)
        self.embedder.currentTextChanged.connect(lambda _t: self._check_embedder())
        embed_row.addWidget(self.embedder, 1)
        right_layout.addLayout(embed_row)
        self.embed_banner = Banner("warning", "")
        self.embed_banner.add_action("Download", self._download_embedder, "primary")
        self.embed_banner.hide()
        right_layout.addWidget(self.embed_banner)

        add_row = QHBoxLayout()
        self.add_btn = button("Add files…", "upload", "primary")
        self.add_btn.clicked.connect(self._pick_files)
        add_row.addWidget(self.add_btn)
        add_row.addWidget(label("or drop files here", "faint"))
        add_row.addStretch(1)
        right_layout.addLayout(add_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        right_layout.addWidget(self.progress)
        self.progress_text = label("", "faint", wrap=True)
        right_layout.addWidget(self.progress_text)

        right_layout.addWidget(label("TEST RETRIEVAL", "eyebrow"))
        self.query = SearchField("Ask something your documents answer…")
        self.query.returnPressed.connect(self._run_query)
        right_layout.addWidget(self.query)
        self.results = QListWidget()
        self.results.setWordWrap(True)
        right_layout.addWidget(self.results, 1)
        splitter.addWidget(right)
        splitter.setSizes([260, 640])
        ctx.downloads.job_finished.connect(lambda *_: self._check_embedder())

    # ------------------------------------------------------------------ data
    def on_activated(self) -> None:
        self.refresh()
        self._check_embedder()

    def refresh(self) -> None:
        handle = self.ctx.runtime.request(Command.RAG_LIST_COLLECTIONS)
        handle.finished.connect(self._set_collections)

    def _set_collections(self, payload: dict[str, Any]) -> None:
        current = self.current_name()
        self._collections = payload.get("collections") or []
        self.list.blockSignals(True)
        self.list.clear()
        for item in self._collections:
            entry = QListWidgetItem(f"{item['name']}\n{item.get('documents', 0)} docs · {item.get('chunks', 0)} chunks")
            entry.setData(Qt.ItemDataRole.UserRole, item["name"])
            self.list.addItem(entry)
        self.list.blockSignals(False)
        names = [c["name"] for c in self._collections]
        if current in names:
            self.list.setCurrentRow(names.index(current))
        elif names:
            self.list.setCurrentRow(0)
        else:
            self._on_select(-1)

    def current_name(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _info_for(self, name: str | None) -> dict[str, Any] | None:
        return next((c for c in self._collections if c["name"] == name), None)

    def _on_select(self, _row: int) -> None:
        info = self._info_for(self.current_name())
        has = info is not None
        self.delete_btn.setEnabled(has)
        self.add_btn.setEnabled(has)
        self.query.setEnabled(has and bool(info and info.get("chunks")))
        if info is None:
            self.title.setText("Create a collection to get started")
            self.info.setText("")
            return
        self.title.setText(info["name"])
        self.info.setText(f"{info.get('documents', 0)} documents · {info.get('chunks', 0)} chunks · "
                          f"{info.get('dimension', '?')}-d vectors ({info.get('backend', '?')}) · "
                          f"embedding model {info.get('embedding_model')} · updated {time_ago(info.get('updated_at'))}")
        index = self.embedder.findText(info.get("embedding_model", ""))
        if index < 0 and info.get("embedding_model"):
            self.embedder.addItem(info["embedding_model"])
            index = self.embedder.count() - 1
        self.embedder.setCurrentIndex(index)
        self.embedder.setEnabled(not info.get("chunks"))

    def _check_embedder(self) -> None:
        model = self.embedder.currentText().strip()
        ok = bool(model) and is_downloaded(model)
        self.embed_banner.set_text(f"The embedding model <b>{model}</b> is not downloaded yet." if not ok else "")
        self.embed_banner.setVisible(not ok)

    def _download_embedder(self) -> None:
        model = self.embedder.currentText().strip()
        if not model:
            return
        from noelle.app.services.hf_client import ModelSummary

        self.ctx.downloads.start(model, summary=ModelSummary(id=model, pipeline_tag="sentence-similarity"))
        self.ctx.toast(f"Downloading {model}…", "info")

    # --------------------------------------------------------------- actions
    def _new_collection(self) -> None:
        name, ok = QInputDialog.getText(self, "New collection", "Collection name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if self._info_for(name):
            self.ctx.toast("A collection with that name already exists.", "warning")
            return
        self._collections.append({"name": name, "documents": 0, "chunks": 0,
                                  "embedding_model": self.embedder.currentText().strip()})
        item = QListWidgetItem(f"{name}\nempty")
        item.setData(Qt.ItemDataRole.UserRole, name)
        self.list.addItem(item)
        self.list.setCurrentItem(item)

    def _delete_collection(self) -> None:
        name = self.current_name()
        if not name:
            return
        if QMessageBox.question(self, "Delete collection", f"Delete the index '{name}'? Your original files are not "
                                "touched.") != QMessageBox.StandardButton.Yes:
            return
        handle = self.ctx.runtime.request(Command.RAG_DELETE_COLLECTION, {"collection": name})
        handle.finished.connect(lambda _p: (self.refresh(), self.collections_changed.emit()))

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add documents", "", FILE_FILTER)
        if files:
            self.ingest(files)

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and self.current_name():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.ingest(files)

    def ingest(self, files: list[str]) -> None:
        expanded: list[str] = []
        for item in files:
            path = Path(item)
            if path.is_dir():  # dropped folders: index the supported files inside them
                expanded += sorted(str(p) for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
            else:
                expanded.append(item)
        files = expanded
        if not files:
            self.ctx.toast("No supported documents found.", "info")
            return
        name = self.current_name()
        if not name or self._ingest is not None:
            return
        model = self.embedder.currentText().strip()
        if not is_downloaded(model):
            self._check_embedder()
            self.ctx.toast("Download the embedding model first.", "warning")
            return
        self.progress.show()
        self.progress.setValue(0)
        self.progress_text.setText(f"Indexing {len(files)} file(s)…")
        self.add_btn.setEnabled(False)
        handle = self.ctx.runtime.request(Command.RAG_INGEST, {"collection": name, "files": files,
                                                               "embedding_model": model})
        handle.progress.connect(self._on_progress)
        handle.status.connect(lambda p: self.progress_text.setText(p.get("message", "")))
        handle.finished.connect(self._on_ingested)
        handle.failed.connect(self._on_ingest_failed)
        self._ingest = handle

    def _on_progress(self, data: dict[str, Any]) -> None:
        total = data.get("total") or 0
        current = data.get("current") or 0
        stage = data.get("stage", "")
        name = Path(str(data.get("file", ""))).name
        if stage == "embed":  # counts chunks inside the current file; keep the bar on the file scale
            self.progress_text.setText(f"Embedding {name} · chunk {current}/{total}")
            return
        if total:
            self.progress.setValue(int(current / total * 1000))
        self.progress_text.setText(f"{stage.capitalize()} · {name} ({current}/{total} files)")

    def _on_ingested(self, result: dict[str, Any]) -> None:
        self._ingest = None
        self.progress.hide()
        skipped = result.get("skipped") or []
        note = f" · skipped {len(skipped)}: " + ", ".join(f"{s.get('path', '').split('/')[-1]} ({s.get('reason')})"
                                                          for s in skipped[:3]) if skipped else ""
        self.progress_text.setText(f"Added {result.get('documents_added', 0)} document(s), "
                                   f"{result.get('chunks_added', 0)} chunks in {result.get('elapsed_ms', 0) / 1000:.1f} s"
                                   f"{note}")
        self.add_btn.setEnabled(True)
        self.refresh()
        self.collections_changed.emit()

    def _on_ingest_failed(self, payload: dict[str, Any]) -> None:
        self._ingest = None
        self.progress.hide()
        self.add_btn.setEnabled(True)
        self.progress_text.setText(f"Indexing failed: {payload.get('message')}")

    def _run_query(self) -> None:
        name = self.current_name()
        text = self.query.text().strip()
        if not name or not text:
            return
        self.results.clear()
        self.results.addItem("Searching…")
        handle = self.ctx.runtime.request(Command.RAG_QUERY, {"collection": name, "query": text,
                                                              "top_k": int(self.ctx.settings.get("rag.top_k", 4))})
        handle.finished.connect(self._show_results)
        handle.failed.connect(lambda p: (self.results.clear(), self.results.addItem(p.get("message", "Failed"))))

    def _show_results(self, payload: dict[str, Any]) -> None:
        self.results.clear()
        hits = payload.get("hits") or []
        if not hits:
            self.results.addItem("No matches.")
            return
        latency = payload.get("latency_ms")
        for hit in hits:
            page = f" · p. {hit['page']}" if hit.get("page") else ""
            snippet = " ".join(str(hit.get("text", "")).split())[:260]
            item = QListWidgetItem(f"{hit.get('score', 0):.3f}  {hit.get('source')}{page} · chunk {hit.get('chunk_id')}"
                                   f"\n{snippet}…")
            item.setToolTip(str(hit.get("text", ""))[:1200])
            self.results.addItem(item)
        if latency is not None:
            self.results.addItem(f"Retrieved in {latency:.0f} ms")
