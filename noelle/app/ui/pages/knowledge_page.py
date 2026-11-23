"""Knowledge page (RAG Studio): local knowledge bases with documents, search, retrieval evaluation and settings.

Everything that touches the store runs in the model runtime (``RAG_*`` commands): listing knowledge
bases and their documents, indexing, search, and removing documents or whole bases. The shell never
opens the vector store itself.
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app import icons, theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import (
    Banner,
    EmptyState,
    Panel,
    Pill,
    SearchField,
    Spinner,
    StatTile,
    button,
    human_bytes,
    label,
    time_ago,
)
from noelle.app.ui.widgets.knowledge_panel import EMBEDDING_CHOICES, FILE_FILTER, SUPPORTED_SUFFIXES
from noelle.runtime import retrieval_eval
from noelle.runtime.hub_cache import is_downloaded
from noelle.runtime.protocol import Command, ErrorCode

OPTIONS_KEY = "rag.knowledge_bases"  # settings: {base name: {"chunk_tokens": int, "chunk_overlap": int}}
CHUNK_RANGE = (64, 2048)  # the bounds of Settings › Grounding
OVERLAP_RANGE = (1, 512)  # the runtime reads an overlap of 0 as "use the global default"
CASE_FILTER = "JSON Lines (*.jsonl *.ndjson);;All files (*)"
ROLE = Qt.ItemDataRole.UserRole
STARTING = ("Starting the model runtime…", "Knowledge bases appear as soon as it is ready.")
RUNTIME_STATES = {
    "stopped": ("The model runtime is not running",
                "Knowledge bases are indexed and searched by the local runtime. Restart it from the status bar."),
    "crashed": ("The model runtime crashed",
                "Restart it from the status bar. Your knowledge bases are safe on disk."),
    "unresponsive": ("The model runtime is not responding",
                     "It may be busy with a long job. This page comes back when the runtime answers again."),
}


# ============================================================================
# store access and formatting (no Qt)
# ============================================================================


def expand_documents(items: Iterable[str | Path]) -> list[str]:
    """Files to index: a folder stands for the supported files inside it, skipping hidden folders and files."""
    files: list[str] = []
    for item in items:
        path = Path(item)
        if not path.is_dir():
            files.append(str(path))
            continue
        found = (candidate for candidate in path.rglob("*")
                 if candidate.suffix.lower() in SUPPORTED_SUFFIXES and candidate.is_file()
                 and not any(part.startswith(".") for part in candidate.relative_to(path).parts))
        files += sorted(str(candidate) for candidate in found)
    return files


def snippet(text: Any, limit: int = 280) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def format_citation(hit: dict[str, Any]) -> str:
    """A pasteable quote of a search hit: passage, document, page and location."""
    path = str(hit.get("path") or "")
    source = str(hit.get("source") or Path(path).name or "document")
    page = hit.get("page")
    where = f"{source}, p. {page}" if page not in (None, "") else source
    location = f" ({path})" if path and path != source else ""
    return f"“{snippet(hit.get('text'))}” — {where}{location}"


def split_relevant(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ";").split(";") if part.strip()]


def _bounded(value: Any, bounds: tuple[int, int], fallback: Any) -> int:
    for candidate in (value, fallback):
        try:
            return min(max(int(candidate), bounds[0]), bounds[1])
        except (TypeError, ValueError):
            continue
    return bounds[0]


def _short_model(model: Any) -> str:
    return str(model or "").split("/")[-1]


# ============================================================================
# page
# ============================================================================


class KnowledgePage(QWidget):
    """Knowledge bases (local RAG collections): documents, search, retrieval evaluation and ingestion settings."""

    collections_changed = Signal()  # a knowledge base was filled, emptied or deleted

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.bases: list[dict[str, Any]] = []  # as RAG_LIST_COLLECTIONS reports them
        self.drafts: dict[str, str] = {}  # created here, no documents yet: name -> embedding model
        self.catalog: dict[str, list[dict[str, Any]]] = {}
        self.catalog_loaded = False
        self.catalog_error: str | None = None
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Knowledge", "h2"))
        titles.addWidget(label("Local knowledge bases for grounded answers: index documents, search them and measure "
                               "how well retrieval finds the right ones. Everything runs on this machine.", "muted",
                               wrap=True))
        head.addLayout(titles, 1)
        self.spinner = Spinner(16)
        head.addWidget(self.spinner)
        self.refresh_btn = button("Refresh", "refresh", "ghost")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        root.addLayout(head)
        self.error = Banner("danger", "")
        self.error.hide()
        root.addWidget(self.error)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.offline = EmptyState("cpu", *STARTING)
        self.stack.addWidget(self.offline)
        self.body = QSplitter(Qt.Orientation.Horizontal)
        self.stack.addWidget(self.body)

        side = Panel()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)
        side_layout.addWidget(label("KNOWLEDGE BASES", "eyebrow"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _row: self._show_detail())
        side_layout.addWidget(self.list, 1)
        self.new_btn = button("New knowledge base", "plus", "primary")
        self.new_btn.clicked.connect(self.new_base)
        side_layout.addWidget(self.new_btn)
        self.body.addWidget(side)

        self.detail_stack = QStackedWidget()
        self.no_bases = EmptyState("database", "No knowledge bases yet",
                                   "Create one, add documents, then search it and measure its retrieval quality.",
                                   action="New knowledge base")
        self.no_bases.action_clicked.connect(self.new_base)
        self.detail_stack.addWidget(self.no_bases)
        self.detail = QWidget()
        detail_layout = QVBoxLayout(self.detail)
        detail_layout.setContentsMargins(16, 0, 0, 0)
        detail_layout.setSpacing(8)
        title_row = QHBoxLayout()
        self.title = label("", "h3")
        title_row.addWidget(self.title)
        self.embedder_pill = Pill("", "unknown")
        title_row.addWidget(self.embedder_pill)
        title_row.addStretch(1)
        detail_layout.addLayout(title_row)
        self.info = label("", "faint", wrap=True)
        detail_layout.addWidget(self.info)
        self.embed_banner = Banner("warning", "")
        self.download_btn = self.embed_banner.add_action("Download", lambda: self.download_embedder(self._model()),
                                                         "primary")
        self.embed_banner.hide()
        detail_layout.addWidget(self.embed_banner)
        self.tabs = QTabWidget()
        self.documents = DocumentsTab(self)
        self.search = SearchTab(self)
        self.evaluate = EvaluateTab(self)
        self.settings_tab = SettingsTab(self)
        for widget, icon_name, text in ((self.documents, "file", "Documents"), (self.search, "search", "Search"),
                                        (self.evaluate, "gauge", "Evaluate"),
                                        (self.settings_tab, "sliders", "Settings")):
            self.tabs.addTab(widget, icons.icon(icon_name, theme.TEXT_MUTED, 14), text)
        detail_layout.addWidget(self.tabs, 1)
        self.detail_stack.addWidget(self.detail)
        self.body.addWidget(self.detail_stack)
        self.body.setStretchFactor(1, 1)
        self.body.setSizes([290, 910])

        ctx.runtime.state_changed.connect(self._sync_runtime)
        ctx.downloads.job_updated.connect(self._on_download)
        ctx.downloads.job_finished.connect(self._on_download)
        self._sync_runtime()
        self._render_bases()

    # ------------------------------------------------------------- lifecycle
    def on_activated(self) -> None:
        self._sync_runtime()
        self.refresh()

    def shutdown(self) -> None:
        self.documents.cancel()
        self.evaluate.cancel()

    def _sync_runtime(self, _state: str = "") -> None:
        ready = bool(self.ctx.runtime.is_ready)
        if not ready:
            title, message = RUNTIME_STATES.get(str(self.ctx.runtime.state), STARTING)
            self.offline.title.setText(title)
            self.offline.message.setText(message)
        was_ready = self.stack.currentWidget() is self.body
        self.stack.setCurrentWidget(self.body if ready else self.offline)
        self.refresh_btn.setEnabled(ready)
        if ready and not was_ready and self.isVisible():
            self.refresh()

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        """Ask the runtime for the knowledge bases and re-read the document catalogue."""
        if not self.ctx.runtime.is_ready:
            return
        self.spinner.start()
        handle = self.ctx.runtime.request(Command.RAG_LIST_COLLECTIONS)
        handle.finished.connect(self._on_bases)
        handle.failed.connect(self._on_bases_failed)
        documents = self.ctx.runtime.request(Command.RAG_LIST_DOCUMENTS)
        documents.finished.connect(lambda payload: self._on_catalog(payload.get("documents") or {}))
        documents.failed.connect(lambda payload: self._on_catalog_failed(payload.get("message") or "unknown error"))

    def after_change(self) -> None:
        self.refresh()
        self.collections_changed.emit()

    def _on_bases(self, payload: dict[str, Any]) -> None:
        self.spinner.stop()
        self.error.hide()
        self.bases = [dict(item) for item in payload.get("collections") or [] if item.get("name")]
        built = {item["name"] for item in self.bases}
        self.drafts = {name: model for name, model in self.drafts.items() if name not in built}
        self._render_bases()

    def _on_bases_failed(self, payload: dict[str, Any]) -> None:
        self.spinner.stop()
        self.show_error(f"Could not list the knowledge bases: {payload.get('message') or 'unknown error'}")

    def _on_catalog(self, catalog: dict[str, list[dict[str, Any]]]) -> None:
        self.catalog, self.catalog_loaded, self.catalog_error = catalog, True, None
        self._render_bases()

    def _on_catalog_failed(self, message: str) -> None:
        self.catalog, self.catalog_loaded, self.catalog_error = {}, True, message
        self._render_bases()

    def show_error(self, text: str) -> None:
        self.error.set_text(html.escape(text, quote=False))
        self.error.show()

    def base_names(self) -> set[str]:
        return {item["name"] for item in self.bases} | set(self.drafts)

    def all_bases(self) -> list[dict[str, Any]]:
        merged = [dict(item) for item in self.bases]
        merged += [{"name": name, "embedding_model": model, "documents": 0, "chunks": 0, "draft": True}
                   for name, model in self.drafts.items()]
        return sorted(merged, key=lambda item: (str(item["name"]).casefold(), str(item["name"])))

    def base_info(self, name: str | None) -> dict[str, Any] | None:
        return next((item for item in self.all_bases() if item["name"] == name), None)

    def current_base(self) -> str | None:
        item = self.list.currentItem()
        return item.data(ROLE) if item is not None else None

    def current_info(self) -> dict[str, Any] | None:
        return self.base_info(self.current_base())

    def select_base(self, name: str) -> None:
        for row in range(self.list.count()):
            if self.list.item(row).data(ROLE) == name:
                self.list.setCurrentRow(row)
                return

    def _source_bytes(self, name: str) -> int | None:
        documents = self.catalog.get(name)
        return sum(doc["size_bytes"] for doc in documents) if documents else None

    def _render_bases(self, select: str | None = None) -> None:
        current = select or self.current_base()
        bases = self.all_bases()
        self.list.blockSignals(True)
        self.list.clear()
        for info in bases:
            item = QListWidgetItem(self._describe(info))
            item.setData(ROLE, info["name"])
            item.setToolTip(self._tooltip(info))
            self.list.addItem(item)
        names = [info["name"] for info in bases]
        self.list.setCurrentRow(names.index(current) if current in names else (0 if names else -1))
        self.list.blockSignals(False)
        self._show_detail()

    def _describe(self, info: dict[str, Any]) -> str:
        model = _short_model(info.get("embedding_model"))
        if info.get("draft"):
            return f"{info['name']}\nNew · add documents to build it\n{model}"
        counts = f"{info.get('documents', 0)} docs · {info.get('chunks', 0)} chunks"
        size = self._source_bytes(info["name"])
        if size is not None:
            counts += f" · {human_bytes(size)}"
        return f"{info['name']}\n{counts}\n{model} · updated {time_ago(info.get('updated_at'))}"

    def _tooltip(self, info: dict[str, Any]) -> str:
        lines = [str(info["name"]), f"Embedding model: {info.get('embedding_model')}"]
        if not info.get("draft"):
            lines.append(f"{info.get('dimension', '?')}-d vectors · {info.get('backend', '?')} search")
            lines.append(f"Created {time_ago(info.get('created_at'))} · updated {time_ago(info.get('updated_at'))}")
            size = self._source_bytes(info["name"])
            if size is not None:
                lines.append(f"Source files: {human_bytes(size)}")
        return "\n".join(lines)

    def _summary(self, info: dict[str, Any]) -> str:
        model = info.get("embedding_model")
        if info.get("draft"):
            return f"Not indexed yet · embedding model {model} · add files or a folder to build it"
        parts = [f"{info.get('documents', 0)} documents", f"{info.get('chunks', 0)} chunks",
                 f"{info.get('dimension', '?')}-d vectors ({info.get('backend', '?')})", f"embedding model {model}"]
        size = self._source_bytes(info["name"])
        if size is not None:
            parts.append(f"{human_bytes(size)} of source files")
        parts.append(f"updated {time_ago(info.get('updated_at'))}")
        return " · ".join(parts)

    def _show_detail(self) -> None:
        info = self.current_info()
        self.detail_stack.setCurrentWidget(self.detail if info is not None else self.no_bases)
        documents: list[dict[str, Any]] | None = None
        if info is not None:
            self.title.setText(str(info["name"]))
            self.info.setText(self._summary(info))
            if info.get("draft"):
                documents = []
            elif self.catalog_loaded and self.catalog_error is None:
                documents = self.catalog.get(info["name"], [])
        self._update_embedder()
        self.documents.set_base(info, documents, self.catalog_error)
        self.search.set_base(info)
        self.evaluate.set_base(info, self.all_bases())
        self.settings_tab.set_base(info)

    # ------------------------------------------------------------- embedders
    def default_embedder(self) -> str:
        return str(self.ctx.settings.get("rag.embedding_model") or EMBEDDING_CHOICES[0]).strip()

    def embedder_ready(self, model: str) -> bool:
        model = model.strip()
        if not model:
            return False
        if model in self.ctx.downloaded_ids():
            return True
        return is_downloaded(model, cache_dir=paths.hf_hub_cache(self.ctx.settings.get("storage.hf_home")))

    def download_embedder(self, model: str) -> None:
        model = model.strip()
        if not model:
            return
        from noelle.app.services.hf_client import ModelSummary

        self.ctx.downloads.start(model, summary=ModelSummary(id=model, pipeline_tag="sentence-similarity"))
        self.ctx.toast(f"Downloading {model}…", "info")
        self._update_embedder()

    def _model(self) -> str:
        info = self.current_info()
        return str(info.get("embedding_model") or "") if info is not None else ""

    def _update_embedder(self) -> None:
        model = self._model()
        if not model or self.embedder_ready(model):
            self.embedder_pill.set_level("green")
            self.embedder_pill.set_text("Embedder ready")
            self.embedder_pill.setToolTip(model)
            self.embed_banner.hide()
            return
        job = self.ctx.downloads.job(model)
        downloading = job is not None and job.active
        self.embedder_pill.set_level("yellow")
        self.embedder_pill.set_text("Embedder downloading" if downloading else "Embedder missing")
        self.embedder_pill.setToolTip(model)
        shown = html.escape(model, quote=False)
        if downloading:
            self.embed_banner.set_text(f"Downloading the embedding model <b>{shown}</b>… {job.fraction:.0%}")
        else:
            self.embed_banner.set_text(f"The embedding model <b>{shown}</b> is not downloaded yet. Adding documents, "
                                       "search and evaluation need it.")
        self.download_btn.setVisible(not downloading)
        self.embed_banner.show()

    def _on_download(self, repo_id: str, *_args: Any) -> None:
        if repo_id == self._model():
            self._update_embedder()

    # --------------------------------------------------------------- actions
    def new_base(self) -> None:
        dialog = NewBaseDialog(self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.create_base(*dialog.values())
        finally:
            dialog.deleteLater()

    def create_base(self, name: str, model: str) -> bool:
        """Add an empty knowledge base; the runtime creates it with the first documents added to it."""
        name, model = name.strip(), model.strip()
        if not name or not model or name in self.base_names():
            return False
        self.drafts[name] = model
        self._render_bases(select=name)
        self.tabs.setCurrentWidget(self.documents)
        return True

    def delete_current_base(self) -> None:
        info = self.current_info()
        if info is None:
            return
        name = str(info["name"])
        if QMessageBox.question(self, "Delete knowledge base", f"Delete the knowledge base '{name}' and its index? "
                                "Your original files are not touched.") != QMessageBox.StandardButton.Yes:
            return
        if info.get("draft"):
            self.drafts.pop(name, None)
            self.forget_base_options(name)
            self._render_bases()
            return
        handle = self.ctx.runtime.request(Command.RAG_DELETE_COLLECTION, {"collection": name})
        handle.finished.connect(lambda _payload: self._on_deleted(name))
        handle.failed.connect(lambda payload: self.show_error(
            f"Could not delete '{name}': {payload.get('message') or 'unknown error'}"))

    def _on_deleted(self, name: str) -> None:
        self.forget_base_options(name)
        self.ctx.toast(f"Deleted the knowledge base '{name}'.", "success")
        self.after_change()

    # ------------------------------------------------------ ingestion options
    def base_options(self, name: str) -> dict[str, int]:
        """Chunk size and overlap for documents added to ``name`` (its own values, else the global ones)."""
        stored = self._stored_options().get(name)
        stored = stored if isinstance(stored, dict) else {}
        return {"chunk_tokens": _bounded(stored.get("chunk_tokens"), CHUNK_RANGE,
                                         self.ctx.settings.get("rag.chunk_tokens", 512)),
                "chunk_overlap": _bounded(stored.get("chunk_overlap"), OVERLAP_RANGE,
                                          self.ctx.settings.get("rag.chunk_overlap", 64))}

    def set_base_options(self, name: str, chunk_tokens: int, chunk_overlap: int) -> None:
        stored = self._stored_options()
        stored[name] = {"chunk_tokens": int(chunk_tokens), "chunk_overlap": int(chunk_overlap)}
        self.ctx.settings.set(OPTIONS_KEY, stored)

    def forget_base_options(self, name: str) -> None:
        stored = self._stored_options()
        if stored.pop(name, None) is not None:
            self.ctx.settings.set(OPTIONS_KEY, stored)

    def _stored_options(self) -> dict[str, Any]:
        stored = self.ctx.settings.get(OPTIONS_KEY)
        return dict(stored) if isinstance(stored, dict) else {}

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and self.ctx.runtime.is_ready and self.current_info() is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.tabs.setCurrentWidget(self.documents)
            self.documents.ingest(files)


# ============================================================================
# new knowledge base
# ============================================================================


class NewBaseDialog(QDialog):
    """Name and embedding model of a new knowledge base; offers to download a missing embedding model."""

    def __init__(self, page: KnowledgePage) -> None:
        super().__init__(page)
        self.page = page
        self.setWindowTitle("New knowledge base")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.name = QLineEdit()
        self.name.setMaxLength(200)
        self.name.setPlaceholderText("e.g. Product manuals")
        form.addRow("Name", self.name)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(28)
        default = page.default_embedder()
        for choice in [default] + [c for c in EMBEDDING_CHOICES if c != default]:
            self.model.addItem(choice)
        form.addRow("Embedding model", self.model)
        layout.addLayout(form)
        status = QHBoxLayout()
        self.pill = Pill("", "unknown")
        status.addWidget(self.pill)
        self.note = label("", "faint", wrap=True)
        status.addWidget(self.note, 1)
        self.download_btn = button("Download", "download")
        self.download_btn.clicked.connect(self._download)
        status.addWidget(self.download_btn)
        layout.addLayout(status)
        layout.addWidget(label("The embedding model is fixed once the base has documents. Bases can use different "
                               "models, and Evaluate › Compare measures which one retrieves better.", "faint",
                               wrap=True))
        self.problem = label("", "error", wrap=True)
        self.problem.hide()
        layout.addWidget(self.problem)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.create_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.create_btn.setText("Create")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.name.textChanged.connect(lambda _text: self._validate())
        self.model.currentTextChanged.connect(lambda _text: self._update_status())
        page.ctx.downloads.job_updated.connect(self._on_download)
        page.ctx.downloads.job_finished.connect(self._on_download)
        self._update_status()

    def values(self) -> tuple[str, str]:
        return self.name.text().strip(), self.model.currentText().strip()

    def _validate(self) -> None:
        name, model = self.values()
        taken = bool(name) and name in self.page.base_names()
        self.problem.setText("A knowledge base with this name already exists." if taken else "")
        self.problem.setVisible(taken)
        self.create_btn.setEnabled(bool(name) and bool(model) and not taken)

    def _update_status(self) -> None:
        model = self.model.currentText().strip()
        job = self.page.ctx.downloads.job(model) if model else None
        if model and self.page.embedder_ready(model):
            level, text, note = "green", "Downloaded", "Ready to index on this machine."
        elif job is not None and job.active:
            level, text, note = "info", "Downloading", f"{job.fraction:.0%} downloaded…"
        else:
            level, text, note = "yellow", "Not downloaded", "Download it before adding documents (one-time download)."
        self.pill.set_level(level)
        self.pill.set_text(text)
        self.note.setText(note)
        self.download_btn.setVisible(level == "yellow" and bool(model))
        self._validate()

    def _download(self) -> None:
        self.page.download_embedder(self.model.currentText())
        self._update_status()

    def _on_download(self, repo_id: str, *_args: Any) -> None:
        if repo_id == self.model.currentText().strip():
            self._update_status()


# ============================================================================
# documents
# ============================================================================


class DocumentsTab(QWidget):
    def __init__(self, page: KnowledgePage) -> None:
        super().__init__()
        self.page = page
        self.ctx = page.ctx
        self.base: dict[str, Any] | None = None
        self._ingest: Any = None
        self._removing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)
        bar = QHBoxLayout()
        self.add_files_btn = button("Add files…", "upload", "primary")
        self.add_files_btn.clicked.connect(self._pick_files)
        bar.addWidget(self.add_files_btn)
        self.add_folder_btn = button("Add folder…", "folder")
        self.add_folder_btn.clicked.connect(self._pick_folder)
        bar.addWidget(self.add_folder_btn)
        bar.addWidget(label("or drop files and folders here", "faint"))
        bar.addStretch(1)
        self.remove_btn = button("Remove", "trash", "ghost",
                                 tooltip="Remove the selected documents from this knowledge base (files stay on disk)")
        self.remove_btn.clicked.connect(self.remove_selected)
        bar.addWidget(self.remove_btn)
        self.delete_btn = button("Delete knowledge base…", "trash", "danger")
        self.delete_btn.clicked.connect(self.page.delete_current_base)
        bar.addWidget(self.delete_btn)
        layout.addLayout(bar)

        self.progress_row = QWidget()
        progress_layout = QHBoxLayout(self.progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        progress_layout.addWidget(self.progress, 1)
        self.cancel_btn = button("Cancel", "x", "ghost")
        self.cancel_btn.clicked.connect(self.cancel)
        progress_layout.addWidget(self.cancel_btn)
        self.progress_row.hide()
        layout.addWidget(self.progress_row)
        self.status = label("", "faint", wrap=True)
        layout.addWidget(self.status)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Document", "Chunks", "Size", "Added"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 70), (2, 90), (3, 110)):
            self.tree.setColumnWidth(column, width)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.tree, 1)
        self.hint = label("", "muted", wrap=True)
        layout.addWidget(self.hint)

    @property
    def busy(self) -> bool:
        return self._ingest is not None or self._removing

    def set_base(self, info: dict[str, Any] | None, documents: list[dict[str, Any]] | None,
                 error: str | None) -> None:
        self.base = info
        self.tree.clear()
        for doc in documents or []:
            item = QTreeWidgetItem([doc["name"], str(doc["chunks"]), human_bytes(doc["size_bytes"]),
                                    time_ago(doc["added_at"])])
            item.setData(0, ROLE, doc["path"])
            item.setToolTip(0, doc["path"])
            for column in (1, 2):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.tree.addTopLevelItem(item)
        if info is None or documents:
            hint = ""
        elif documents is None:
            hint = f"The document list is unavailable: {error}" if error else "Reading the document list…"
        elif info.get("documents"):
            hint = (f"The runtime reports {info['documents']} document(s) but listed none for this base. Refresh to "
                    "read it again.")
        else:
            hint = ("No documents yet. Add files or a whole folder: PDF, Word, Markdown, text, HTML, CSV, JSON and "
                    "code files are indexed on this machine.")
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))
        self._update_buttons()

    def _update_buttons(self) -> None:
        idle = self.base is not None and not self.busy
        self.add_files_btn.setEnabled(idle)
        self.add_folder_btn.setEnabled(idle)
        self.delete_btn.setEnabled(idle)
        self.remove_btn.setEnabled(idle and bool(self.tree.selectedItems()))

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add documents", "", FILE_FILTER)
        if files:
            self.ingest(files)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of documents")
        if folder:
            self.ingest([folder])

    # --------------------------------------------------------------- ingest
    def ingest(self, items: Iterable[str | Path]) -> bool:
        """Index files (folders are expanded) into the selected knowledge base with its chunk settings."""
        if self.base is None or self.busy:
            return False
        files = expand_documents(items)
        if not files:
            self.ctx.toast("No supported documents found.", "info")
            return False
        name = str(self.base["name"])
        model = str(self.base.get("embedding_model") or "")
        if not self.page.embedder_ready(model):
            self.ctx.toast("Download the embedding model first.", "warning")
            return False
        options = self.page.base_options(name)
        handle = self.ctx.runtime.request(Command.RAG_INGEST, {"collection": name, "files": files,
                                                               "embedding_model": model, **options})
        handle.progress.connect(self._on_progress)
        handle.status.connect(lambda payload: self.status.setText(str(payload.get("message") or "")))
        handle.finished.connect(lambda result: self._on_ingested(name, result))
        handle.failed.connect(lambda payload: self._on_ingest_failed(name, payload))
        self._ingest = handle
        self.progress.setValue(0)
        self.progress_row.show()
        self.status.setText(f"Indexing {len(files)} file(s) into '{name}'…")
        self._update_buttons()
        return True

    def cancel(self) -> None:
        if self._ingest is not None:
            self._ingest.cancel()
            self.status.setText("Cancelling… files already indexed are kept.")

    def _on_progress(self, data: dict[str, Any]) -> None:
        total = int(data.get("total") or 0)
        current = int(data.get("current") or 0)
        stage = str(data.get("stage") or "")
        name = str(data.get("file") or "")
        if stage == "embed":  # counts chunks inside the current file; the bar stays on the file scale
            self.status.setText(f"Embedding {name} · chunk {current}/{total}")
            return
        if total:
            self.progress.setValue(int(current / total * 1000))
        self.status.setText(f"{stage.capitalize()} · {name} ({current}/{total} files)")

    def _finish_ingest(self) -> None:
        self._ingest = None
        self.progress_row.hide()
        self._update_buttons()
        self.page.after_change()

    def _on_ingested(self, name: str, result: dict[str, Any]) -> None:
        skipped = result.get("skipped") or []
        note = ""
        if skipped:
            shown = ", ".join(f"{Path(str(item.get('path') or '')).name} ({item.get('reason')})"
                              for item in skipped[:3])
            more = f" and {len(skipped) - 3} more" if len(skipped) > 3 else ""
            note = f" · skipped {len(skipped)}: {shown}{more}"
        self.status.setText(f"Added {result.get('documents_added', 0)} document(s) to '{name}', "
                            f"{result.get('chunks_added', 0)} chunks in "
                            f"{float(result.get('elapsed_ms') or 0) / 1000:.1f} s{note}")
        self._finish_ingest()

    def _on_ingest_failed(self, name: str, payload: dict[str, Any]) -> None:
        if payload.get("code") == ErrorCode.CANCELLED:
            self.status.setText(f"Indexing into '{name}' was cancelled. Files finished before that were kept.")
        else:
            self.status.setText(f"Indexing into '{name}' failed: {payload.get('message') or 'unknown error'}")
        self._finish_ingest()

    # --------------------------------------------------------------- remove
    def remove_selected(self) -> None:
        if self.base is None or self.busy:
            return
        selected = [str(item.data(0, ROLE)) for item in self.tree.selectedItems()]
        if not selected:
            return
        name = str(self.base["name"])
        what = f"'{Path(selected[0]).name}'" if len(selected) == 1 else f"{len(selected)} documents"
        if QMessageBox.question(self, "Remove documents", f"Remove {what} from '{name}'? The files on disk are not "
                                "touched.") != QMessageBox.StandardButton.Yes:
            return
        self._removing = True
        self._update_buttons()
        self.status.setText(f"Removing {what}…")
        handle = self.page.ctx.runtime.request(Command.RAG_DELETE_DOCUMENT, {"collection": name, "paths": selected})
        handle.finished.connect(lambda payload: self._on_removed(name, int(payload.get("removed") or 0)))
        handle.failed.connect(lambda payload: self._on_remove_failed(payload.get("message") or "unknown error"))

    def _on_removed(self, name: str, count: int) -> None:
        self._removing = False
        self.status.setText(f"Removed {count} document(s) from '{name}'.")
        self._update_buttons()
        self.page.after_change()

    def _on_remove_failed(self, message: str) -> None:
        self._removing = False
        self.status.setText(f"Could not remove the documents: {message}")
        self._update_buttons()


# ============================================================================
# search
# ============================================================================


class SearchTab(QWidget):
    def __init__(self, page: KnowledgePage) -> None:
        super().__init__()
        self.page = page
        self.ctx = page.ctx
        self.base: dict[str, Any] | None = None
        self.hits: list[dict[str, Any]] = []
        self._token = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)
        row = QHBoxLayout()
        self.query = SearchField("Ask something your documents answer…")
        self.query.returnPressed.connect(self.run_search)
        row.addWidget(self.query, 1)
        row.addWidget(label("Top k", "muted"))
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 50)
        self.top_k.setValue(_bounded(self.ctx.settings.get("rag.top_k"), (1, 50), 4))
        row.addWidget(self.top_k)
        self.search_btn = button("Search", "search", "primary")
        self.search_btn.clicked.connect(self.run_search)
        row.addWidget(self.search_btn)
        layout.addLayout(row)
        self.results = QTreeWidget()
        self.results.setHeaderLabels(["#", "Score", "Document", "Page", "Passage"])
        self.results.setRootIsDecorated(False)
        self.results.header().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 36), (1, 64), (2, 200), (3, 50)):
            self.results.setColumnWidth(column, width)
        self.results.itemSelectionChanged.connect(self._update_copy)
        self.results.itemDoubleClicked.connect(lambda *_args: self.copy_citation())
        layout.addWidget(self.results, 1)
        foot = QHBoxLayout()
        self.status = label("", "faint", wrap=True)
        foot.addWidget(self.status, 1)
        self.copy_btn = button("Copy citation", "copy", tooltip="Copy the passage with its document and page")
        self.copy_btn.clicked.connect(self.copy_citation)
        foot.addWidget(self.copy_btn)
        layout.addLayout(foot)
        self._update_copy()

    def set_base(self, info: dict[str, Any] | None) -> None:
        if (info or {}).get("name") != (self.base or {}).get("name"):
            self._token += 1  # drop answers for the previous base
            self.results.clear()
            self.hits = []
            self.status.setText("")
        self.base = info
        searchable = info is not None and bool(info.get("chunks"))
        for widget in (self.query, self.top_k, self.search_btn):
            widget.setEnabled(searchable)
        if info is not None and not searchable:
            self.status.setText("Add documents to this knowledge base to search it.")
        self._update_copy()

    def run_search(self) -> None:
        text = self.query.text().strip()
        if self.base is None or not self.base.get("chunks") or not text:
            return
        if not self.page.embedder_ready(str(self.base.get("embedding_model") or "")):
            self.status.setText("Download the embedding model of this knowledge base first.")
            return
        self._token += 1
        token = self._token
        self.results.clear()
        self.hits = []
        self.status.setText("Searching…")
        handle = self.ctx.runtime.request(Command.RAG_QUERY, {"collection": self.base["name"], "query": text,
                                                              "top_k": self.top_k.value()})
        handle.finished.connect(lambda payload: self._show(token, payload))
        handle.failed.connect(lambda payload: self._failed(token, payload))
        self._update_copy()

    def _show(self, token: int, payload: dict[str, Any]) -> None:
        if token != self._token:
            return
        self.hits = [dict(hit) for hit in payload.get("hits") or []]
        for rank, hit in enumerate(self.hits, start=1):
            page = hit.get("page")
            item = QTreeWidgetItem([str(rank), f"{float(hit.get('score') or 0.0):.3f}", str(hit.get("source") or ""),
                                    str(page) if page not in (None, "") else "—", snippet(hit.get("text"), 220)])
            item.setData(0, ROLE, rank - 1)
            item.setToolTip(2, str(hit.get("path") or ""))
            item.setToolTip(4, str(hit.get("text") or "")[:1500])
            self.results.addTopLevelItem(item)
        latency = payload.get("latency_ms")
        if not self.hits:
            self.status.setText("No matches.")
        else:
            took = f" in {float(latency):.0f} ms" if latency is not None else ""
            self.status.setText(f"{len(self.hits)} passage(s){took}. Select one to copy it as a citation.")
            self.results.setCurrentItem(self.results.topLevelItem(0))
        self._update_copy()

    def _failed(self, token: int, payload: dict[str, Any]) -> None:
        if token == self._token:
            self.status.setText(f"Search failed: {payload.get('message') or 'unknown error'}")

    def _update_copy(self) -> None:
        self.copy_btn.setEnabled(self.results.currentItem() is not None and bool(self.hits))

    def copy_citation(self) -> None:
        item = self.results.currentItem()
        if item is None or not self.hits:
            return
        QGuiApplication.clipboard().setText(format_citation(self.hits[int(item.data(0, ROLE))]))
        self.ctx.toast("Citation copied", "success")


# ============================================================================
# evaluation
# ============================================================================


@dataclass
class _EvalRun:
    token: int
    cases: list[retrieval_eval.EvalCase]
    k: int
    bases: list[str]
    queue: list[tuple[str, int]]  # (base, case index) still to query: every case on A, then on B
    hits: dict[str, list[list[dict[str, Any]] | None]]
    handle: Any = None

    @property
    def total(self) -> int:
        return len(self.cases) * len(self.bases)


class RunView(QWidget):
    """Mean metrics and per-case results of one knowledge base."""

    COLUMNS = ["Query", "Hit", "Recall", "Precision", "RR", "nDCG", "Rank"]

    def __init__(self, tag: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(Pill(tag, "accent"))
        self.title = label("", "h3")
        head.addWidget(self.title, 1)
        layout.addLayout(head)
        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.tiles = {key: StatTile(retrieval_eval.metric_label(key, 5)) for key, _template in retrieval_eval.METRICS}
        self.set_columns(len(self.tiles))
        layout.addLayout(self.grid)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeaderItem(4).setToolTip("Reciprocal rank: 1 / rank of the first relevant hit")
        self.table.horizontalHeaderItem(6).setToolTip("Rank of the first relevant hit")
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.table, 1)

    def set_columns(self, columns: int) -> None:
        """Lay the metric tiles out in ``columns`` columns (one row alone, a narrower grid side by side)."""
        for index, tile in enumerate(self.tiles.values()):
            self.grid.removeWidget(tile)
            self.grid.addWidget(tile, index // columns, index % columns)
        for column in range(len(self.tiles)):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)

    def show_report(self, base: str, report: retrieval_eval.EvalReport,
                    baseline: retrieval_eval.EvalReport | None = None) -> None:
        self.title.setText(base)
        for key, tile in self.tiles.items():
            tile.title.setText(retrieval_eval.metric_label(key, report.k).upper())
            value = report.metric(key)
            notes = [f"{report.hits} of {len(report.cases)} queries"] if key == "hit_rate" else []
            if baseline is not None:
                notes.append(f"{value - baseline.metric(key):+.2f} vs A")
            tile.set(f"{value:.2f}", " · ".join(notes))
        self.table.setRowCount(len(report.cases))
        for row, result in enumerate(report.cases):
            cells = [result.case.query, "yes" if result.hit else "no", f"{result.recall:.2f}",
                     f"{result.precision:.2f}", f"{result.reciprocal_rank:.2f}", f"{result.ndcg:.2f}",
                     f"#{result.first_rank}" if result.first_rank else "—"]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setToolTip(_case_tooltip(result))
            self.table.item(row, 1).setForeground(QColor(theme.SUCCESS if result.hit else theme.DANGER))


def _case_tooltip(result: retrieval_eval.CaseResult) -> str:
    relevant = set(result.relevant_ranks)
    lines = [result.case.query, "", "Retrieved:"]
    lines += [f"{rank}. {name}{'  (relevant)' if rank in relevant else ''}"
              for rank, name in enumerate(result.retrieved, start=1)] or ["nothing"]
    if result.missed:
        lines += ["", "Not found: " + ", ".join(result.missed)]
    return "\n".join(lines)


class EvaluateTab(QWidget):
    """Test cases, a run over one or two knowledge bases, and their metrics side by side."""

    def __init__(self, page: KnowledgePage) -> None:
        super().__init__()
        self.page = page
        self.ctx = page.ctx
        self.base: dict[str, Any] | None = None
        self.reports: dict[str, retrieval_eval.EvalReport] = {}
        self._run: _EvalRun | None = None
        self._token = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 8)
        editor_layout.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label("TEST CASES", "eyebrow"))
        head.addWidget(label("A query per row and the documents that answer it: file names or path fragments, "
                             "separated by ;", "faint"), 1)
        self.add_btn = button("Add", "plus", "ghost")
        self.add_btn.clicked.connect(self.add_case)
        head.addWidget(self.add_btn)
        self.remove_btn = button("Remove", "trash", "ghost")
        self.remove_btn.clicked.connect(self.remove_cases)
        head.addWidget(self.remove_btn)
        self.import_btn = button("Import…", "upload", "ghost", tooltip="Load test cases from a JSON Lines file")
        self.import_btn.clicked.connect(self._pick_import)
        head.addWidget(self.import_btn)
        self.export_btn = button("Export…", "download", "ghost", tooltip="Save the test cases as JSON Lines")
        self.export_btn.clicked.connect(self._pick_export)
        head.addWidget(self.export_btn)
        editor_layout.addLayout(head)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Query", "Relevant documents"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        editor_layout.addWidget(self.table, 1)

        controls = QHBoxLayout()
        self.base_label = label("", "muted")
        controls.addWidget(self.base_label)
        controls.addSpacing(12)
        controls.addWidget(label("Compare with", "muted"))
        self.compare = QComboBox()
        self.compare.setMinimumWidth(200)
        controls.addWidget(self.compare)
        controls.addSpacing(12)
        controls.addWidget(label("k", "muted"))
        self.k = QSpinBox()
        self.k.setRange(1, 50)
        self.k.setValue(5)
        controls.addWidget(self.k)
        controls.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(160)
        self.progress.hide()
        controls.addWidget(self.progress)
        self.cancel_btn = button("Cancel", "x", "ghost")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.hide()
        controls.addWidget(self.cancel_btn)
        self.run_btn = button("Run evaluation", "play", "primary")
        self.run_btn.clicked.connect(self.run)
        controls.addWidget(self.run_btn)
        editor_layout.addLayout(controls)
        self.status = label("", "faint", wrap=True)
        editor_layout.addWidget(self.status)
        self.error = Banner("danger", "")
        self.error.hide()
        editor_layout.addWidget(self.error)
        splitter.addWidget(editor)

        self.results = QWidget()
        results_layout = QHBoxLayout(self.results)
        results_layout.setContentsMargins(0, 8, 0, 0)
        results_layout.setSpacing(16)
        self.view_a = RunView("A")
        self.view_b = RunView("B")
        results_layout.addWidget(self.view_a, 1)
        results_layout.addWidget(self.view_b, 1)
        self.view_a.table.currentCellChanged.connect(lambda row, *_args: self._sync_row(self.view_b, row))
        self.view_b.table.currentCellChanged.connect(lambda row, *_args: self._sync_row(self.view_a, row))
        self.view_b.hide()
        self.results.hide()
        splitter.addWidget(self.results)
        splitter.setSizes([300, 460])
        self._update_buttons()

    @property
    def running(self) -> bool:
        return self._run is not None

    def set_base(self, info: dict[str, Any] | None, bases: list[dict[str, Any]]) -> None:
        self.base = info
        self.base_label.setText(f"A: {info['name']}" if info is not None else "")
        chosen = self.compare.currentData()
        self.compare.blockSignals(True)
        self.compare.clear()
        self.compare.addItem("No comparison", None)
        for other in bases:
            if info is not None and other["name"] != info["name"] and other.get("chunks"):
                self.compare.addItem(f"B: {other['name']}", other["name"])
        index = self.compare.findData(chosen) if chosen else 0
        self.compare.setCurrentIndex(max(index, 0))
        self.compare.blockSignals(False)
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self.running
        for widget in (self.table, self.add_btn, self.remove_btn, self.import_btn, self.compare, self.k):
            widget.setEnabled(not running)
        self.export_btn.setEnabled(not running and self.table.rowCount() > 0)
        self.run_btn.setEnabled(not running and self.base is not None and bool(self.base.get("chunks")))
        self.run_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.progress.setVisible(running)

    # ----------------------------------------------------------------- cases
    def add_case(self) -> None:
        row = self._append_row("", "")
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def remove_cases(self) -> None:
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
        self._update_buttons()

    def _append_row(self, query: str, relevant: str) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(query))
        self.table.setItem(row, 1, QTableWidgetItem(relevant))
        self._update_buttons()
        return row

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text() if item is not None else ""

    def cases(self) -> list[retrieval_eval.EvalCase]:
        """The table's cases (blank rows skipped); raises ValueError naming the first bad row."""
        cases: list[retrieval_eval.EvalCase] = []
        for row in range(self.table.rowCount()):
            query, relevant = self._cell(row, 0), self._cell(row, 1)
            if not query.strip() and not relevant.strip():
                continue
            try:
                cases.append(retrieval_eval.EvalCase.create(query, split_relevant(relevant)))
            except ValueError as exc:
                raise ValueError(f"Row {row + 1}: {exc}") from None
        return cases

    def set_cases(self, cases: Iterable[retrieval_eval.EvalCase]) -> None:
        self.table.setRowCount(0)
        for case in cases:
            self._append_row(case.query, "; ".join(case.relevant))
        self._update_buttons()

    def import_cases(self, path: str | Path) -> bool:
        try:
            cases = retrieval_eval.load_cases(path)
        except (OSError, ValueError) as exc:
            self._show_error(f"Could not import {Path(path).name}: {exc}")
            return False
        self.set_cases(cases)
        self.error.hide()
        self.status.setText(f"Imported {len(cases)} test case(s) from {Path(path).name}.")
        return True

    def export_cases(self, path: str | Path) -> bool:
        try:
            retrieval_eval.save_cases(path, self.cases())
        except (OSError, ValueError) as exc:
            self._show_error(f"Could not export the test cases: {exc}")
            return False
        self.error.hide()
        self.status.setText(f"Saved the test cases to {path}.")
        return True

    def _pick_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import test cases", "", CASE_FILTER)
        if path:
            self.import_cases(path)

    def _pick_export(self) -> None:
        name = str(self.base["name"]) if self.base is not None else "knowledge"
        default = paths.exports_dir() / f"{name}-retrieval-cases.jsonl"
        path, _ = QFileDialog.getSaveFileName(self, "Export test cases", str(default), CASE_FILTER)
        if path:
            self.export_cases(path)

    def _show_error(self, text: str) -> None:
        self.error.set_text(html.escape(text, quote=False))
        self.error.show()

    # ------------------------------------------------------------------- run
    def run(self) -> bool:
        """Query every case on the selected base (A), then on the comparison base (B), one query at a time."""
        if self.running or self.base is None:
            return False
        self.error.hide()
        try:
            cases = self.cases()
        except ValueError as exc:
            self._show_error(str(exc))
            return False
        if not cases:
            self._show_error("Add at least one test case: a query and the documents that answer it.")
            return False
        bases = [str(self.base["name"])] + ([self.compare.currentData()] if self.compare.currentData() else [])
        for name in bases:
            info = self.page.base_info(name)
            model = str((info or {}).get("embedding_model") or "")
            if info is None or not info.get("chunks"):
                self._show_error(f"'{name}' has no indexed documents yet.")
                return False
            if not self.page.embedder_ready(model):
                self._show_error(f"The embedding model of '{name}' ({model}) is not downloaded.")
                return False
        self._token += 1
        self._run = _EvalRun(token=self._token, cases=cases, k=self.k.value(), bases=bases,
                             queue=[(name, index) for name in bases for index in range(len(cases))],
                             hits={name: [None] * len(cases) for name in bases})
        self.reports = {}
        self.results.hide()
        self.progress.setValue(0)
        self._update_buttons()
        self._next()
        return True

    def cancel(self) -> None:
        run = self._run
        if run is None:
            return
        if run.handle is not None:
            run.handle.cancel()  # the runtime finishes a query it has started; its answer is ignored
        self._stop(run, "Evaluation cancelled.")

    def _next(self) -> None:
        run = self._run
        if run is None:
            return
        if not run.queue:
            names = " and ".join(f"'{name}'" for name in run.bases)
            self._stop(run, f"Evaluated {len(run.cases)} test case(s) at k = {run.k} on {names}.")
            return
        name, index = run.queue[0]
        done = run.total - len(run.queue)
        self.progress.setValue(int(done / run.total * 1000))
        self.status.setText(f"Query {done + 1} of {run.total} · {name}")
        handle = self.ctx.runtime.request(Command.RAG_QUERY, {"collection": name, "query": run.cases[index].query,
                                                              "top_k": run.k})
        handle.finished.connect(lambda payload: self._on_hits(run.token, payload))
        handle.failed.connect(lambda payload: self._on_failed(run.token, payload))
        run.handle = handle

    def _on_hits(self, token: int, payload: dict[str, Any]) -> None:
        run = self._run
        if run is None or run.token != token:
            return
        name, index = run.queue.pop(0)
        run.hits[name][index] = [dict(hit) for hit in payload.get("hits") or []]
        self._next()

    def _on_failed(self, token: int, payload: dict[str, Any]) -> None:
        run = self._run
        if run is None or run.token != token:
            return
        message = f"Evaluation stopped: {payload.get('message') or 'unknown error'}"
        self._stop(run, message)
        self._show_error(message)

    def _stop(self, run: _EvalRun, message: str) -> None:
        """End ``run`` and show every knowledge base whose queries all finished."""
        self._run = None
        self._update_buttons()
        for name in run.bases:
            answered = [hits for hits in run.hits[name] if hits is not None]
            if len(answered) == len(run.cases):
                self.reports[name] = retrieval_eval.evaluate(run.cases, answered, run.k)
        first = self.reports.get(run.bases[0])
        second = self.reports.get(run.bases[1]) if len(run.bases) > 1 else None
        columns = 3 if second is not None else len(retrieval_eval.METRICS)
        if first is not None:
            self.view_a.set_columns(columns)
            self.view_a.show_report(run.bases[0], first)
        if second is not None:
            self.view_b.set_columns(columns)
            self.view_b.show_report(run.bases[1], second, baseline=first)
        self.view_b.setVisible(second is not None)
        self.results.setVisible(first is not None)
        partial = first is not None and len(self.reports) < len(run.bases)
        self.status.setText(message + (" Showing the knowledge base that finished." if partial else ""))

    @staticmethod
    def _sync_row(view: RunView, row: int) -> None:
        if 0 <= row < view.table.rowCount() and view.table.currentRow() != row:
            view.table.selectRow(row)


# ============================================================================
# settings
# ============================================================================


class SettingsTab(QWidget):
    """Chunking used when documents are added to the selected base, and facts about its index."""

    def __init__(self, page: KnowledgePage) -> None:
        super().__init__()
        self.page = page
        self.base: dict[str, Any] | None = None
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(label("INGESTION", "eyebrow"))
        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.chunk = QSpinBox()
        self.chunk.setRange(*CHUNK_RANGE)
        self.chunk.setSingleStep(64)
        self.chunk.setSuffix(" tokens")
        self.chunk.valueChanged.connect(lambda _value: self._on_chunk())
        form.addRow("Chunk size", self.chunk)
        self.overlap = QSpinBox()
        self.overlap.setRange(*OVERLAP_RANGE)
        self.overlap.setSingleStep(16)
        self.overlap.setSuffix(" tokens")
        self.overlap.valueChanged.connect(lambda _value: self._save())
        form.addRow("Overlap", self.overlap)
        for spin in (self.chunk, self.overlap):
            spin.setMinimumWidth(150)
        layout.addLayout(form)
        reset_row = QHBoxLayout()
        self.reset_btn = button("Use the global defaults", "refresh", "ghost")
        self.reset_btn.clicked.connect(self._reset)
        reset_row.addWidget(self.reset_btn)
        self.globals = label("", "faint")
        reset_row.addWidget(self.globals, 1)
        layout.addLayout(reset_row)
        layout.addWidget(label(
            "These values apply when documents are added to this knowledge base. Documents already indexed keep "
            "their chunks, and adding an unchanged file again is skipped. To compare chunk sizes, build a second "
            "base from the same files and use Evaluate › Compare with.", "faint", wrap=True))
        layout.addWidget(label(
            "The chunk size is capped by the embedding model's maximum sequence length, and the overlap by half the "
            "chunk size. The runtime reads an overlap of 0 as \"use the global default\", so the smallest overlap "
            "here is 1 token.", "faint", wrap=True))
        layout.addSpacing(6)
        layout.addWidget(label("INDEX", "eyebrow"))
        self.details = label("", "muted", wrap=True, selectable=True)
        layout.addWidget(self.details)
        layout.addStretch(1)

    def set_base(self, info: dict[str, Any] | None) -> None:
        self.base = info
        settings = self.page.ctx.settings
        self.globals.setText(f"Global defaults (Settings › Grounding): "
                             f"{_bounded(settings.get('rag.chunk_tokens'), CHUNK_RANGE, 512)}-token chunks, "
                             f"{settings.get('rag.chunk_overlap', 64)}-token overlap.")
        for widget in (self.chunk, self.overlap, self.reset_btn):
            widget.setEnabled(info is not None)
        if info is None:
            self.details.setText("")
            return
        options = self.page.base_options(str(info["name"]))
        self._loading = True
        self.chunk.setValue(options["chunk_tokens"])
        self._limit_overlap()
        self.overlap.setValue(options["chunk_overlap"])
        self._loading = False
        model = info.get("embedding_model")
        if info.get("draft"):
            self.details.setText(f"Not indexed yet. The first documents added are embedded with {model}.")
            return
        self.details.setText(
            f"Embedding model: {model} (fixed now that the base has documents)\n"
            f"Vectors: {info.get('dimension', '?')} dimensions · {info.get('backend', '?')} search\n"
            f"Created {time_ago(info.get('created_at'))} · updated {time_ago(info.get('updated_at'))}\n"
            f"Store: {paths.vector_store_dir()}")

    def _limit_overlap(self) -> None:
        self.overlap.setMaximum(max(OVERLAP_RANGE[0], min(OVERLAP_RANGE[1], self.chunk.value() // 2)))

    def _on_chunk(self) -> None:
        self._limit_overlap()
        self._save()

    def _save(self) -> None:
        if self._loading or self.base is None:
            return
        self.page.set_base_options(str(self.base["name"]), self.chunk.value(), self.overlap.value())

    def _reset(self) -> None:
        if self.base is not None:
            self.page.forget_base_options(str(self.base["name"]))
            self.set_base(self.base)
