"""Text playground: chat with streaming, sampling controls, document/web grounding and citations."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.chat_canvas import ChatCanvas, MessageView
from noelle.app.ui.widgets.common import button, divider, label, open_external_url
from noelle.app.ui.widgets.param_panel import ParamPanel
from noelle.runtime.protocol import Command


class Composer(QPlainTextEdit):
    submitted = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setPlaceholderText("Message the model…  (Enter to send, Shift+Enter for a new line)")
        self.setFixedHeight(76)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class SourcesPanel(QWidget):
    """Citation inspector: file name / URL, chunk id, page and similarity score."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(label("SOURCES", "eyebrow"))
        self.empty = label("Turn on Documents or Web search to ground answers. Retrieved sources appear here.",
                           "faint", wrap=True)
        layout.addWidget(self.empty)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.itemDoubleClicked.connect(self._open)
        self.list.hide()
        layout.addWidget(self.list, 1)

    def set_sources(self, sources: list[dict[str, Any]]) -> None:
        self.list.clear()
        for source in sources:
            index = source.get("index", "?")
            if source.get("kind") == "web":
                title = source.get("title") or source.get("url")
                text = f"[{index}] {title}\n{source.get('url', '')}"
            else:
                page = f" · p. {source['page']}" if source.get("page") else ""
                score = f" · score {source['score']:.2f}" if isinstance(source.get("score"), (int, float)) else ""
                snippet = " ".join(str(source.get("text", "")).split())[:180]
                text = (f"[{index}] {source.get('source', 'document')}{page}\nchunk {source.get('chunk_id', '?')}"
                        f"{score}\n{snippet}…")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, source.get("url"))
            item.setToolTip(str(source.get("text") or source.get("snippet") or "")[:800])
            self.list.addItem(item)
        has = bool(sources)
        self.list.setVisible(has)
        self.empty.setVisible(not has)

    def _open(self, item: QListWidgetItem) -> None:
        url = item.data(Qt.ItemDataRole.UserRole)
        if url:
            open_external_url(url)


class TextPlayground(QWidget):
    def __init__(self, ctx: AppContext, payload: Callable[[], dict[str, Any]], display_name: Callable[[], str]) -> None:
        super().__init__()
        self.ctx = ctx
        self.payload = payload
        self.display_name = display_name
        self.history: list[dict[str, str]] = []
        self._handle: Any = None
        self._current: MessageView | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

        # ---------------------------------------------------- chat column
        chat = QWidget()
        column = QVBoxLayout(chat)
        column.setContentsMargins(18, 14, 14, 14)
        column.setSpacing(10)

        top = QHBoxLayout()
        self.system_toggle = button("System prompt", "chevron-down", "ghost", checkable=True)
        self.system_toggle.toggled.connect(lambda on: self.system_prompt.setVisible(on))
        top.addWidget(self.system_toggle)
        top.addStretch(1)
        self.export_btn = button("Export chat", "save", "ghost")
        self.export_btn.clicked.connect(self.export_chat)
        top.addWidget(self.export_btn)
        self.clear_btn = button("New chat", "plus", "ghost")
        self.clear_btn.clicked.connect(self.new_chat)
        top.addWidget(self.clear_btn)
        column.addLayout(top)

        self.system_prompt = QPlainTextEdit(str(ctx.settings.get("generation.system_prompt", "")))
        self.system_prompt.setFixedHeight(70)
        self.system_prompt.setPlaceholderText("System prompt (optional)")
        self.system_prompt.hide()
        column.addWidget(self.system_prompt)

        self.canvas = ChatCanvas()
        self.canvas.set_placeholder("Load the model and say hello. Everything runs on this machine.")
        column.addWidget(self.canvas, 1)

        grounding = QHBoxLayout()
        grounding.setSpacing(10)
        self.use_docs = QCheckBox("Documents")
        self.use_docs.setToolTip("Retrieve passages from a local document collection (RAG)")
        grounding.addWidget(self.use_docs)
        self.collection = QComboBox()
        self.collection.setMinimumWidth(170)
        self.collection.setPlaceholderText("No collections")
        grounding.addWidget(self.collection)
        self.use_web = QCheckBox("Web search")
        self.use_web.setToolTip("Search the web through your configured SearXNG instance and cite pages")
        grounding.addWidget(self.use_web)
        grounding.addStretch(1)
        self.adapter_label = label("", "faint")
        grounding.addWidget(self.adapter_label)
        column.addLayout(grounding)

        composer_row = QHBoxLayout()
        composer_row.setSpacing(8)
        self.composer = Composer()
        self.composer.submitted.connect(self.send)
        composer_row.addWidget(self.composer, 1)
        self.send_btn = button("Send", "send", "primary")
        self.send_btn.setFixedHeight(76)
        self.send_btn.setMinimumWidth(96)
        self.send_btn.clicked.connect(self._send_or_stop)
        composer_row.addWidget(self.send_btn)
        column.addLayout(composer_row)
        splitter.addWidget(chat)

        # ---------------------------------------------------- side panel
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setMinimumWidth(270)
        side_scroll.setMaximumWidth(360)
        side = QFrame()
        side.setStyleSheet(f"QFrame#ParamSide {{ background: {theme.BG_ELEV}; border-left: 1px solid {theme.BORDER}; }}")
        side.setObjectName("ParamSide")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)
        side_layout.addWidget(label("SAMPLING", "eyebrow"))
        defaults = {k: ctx.settings.get(f"generation.{k}") for k in
                    ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")}
        self.params = ParamPanel(defaults)
        side_layout.addWidget(self.params)
        side_layout.addWidget(divider())
        self.sources = SourcesPanel()
        side_layout.addWidget(self.sources, 1)
        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([900, 300])

    # ---------------------------------------------------------- lifecycle
    def on_activated(self) -> None:
        self.refresh_collections()

    def refresh_collections(self) -> None:
        handle = self.ctx.runtime.request(Command.RAG_LIST_COLLECTIONS)
        handle.finished.connect(self._set_collections)

    def _set_collections(self, payload: dict[str, Any]) -> None:
        current = self.collection.currentText()
        self.collection.clear()
        for item in payload.get("collections", []):
            self.collection.addItem(f"{item['name']} ({item.get('documents', 0)} docs)", item["name"])
        index = self.collection.findData(current.split(" (")[0])
        if index >= 0:
            self.collection.setCurrentIndex(index)
        self.use_docs.setEnabled(self.collection.count() > 0)

    def set_adapter(self, name: str | None) -> None:
        self.adapter_label.setText(f"LoRA: {name}" if name else "")

    def export_config(self) -> dict[str, Any]:
        return {"system_prompt": self.system_prompt.toPlainText().strip() or None, "params": self.params.values()}

    # -------------------------------------------------------------- chat
    def new_chat(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
        self.history.clear()
        self.canvas.clear()
        self.sources.set_sources([])

    def _send_or_stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self.send_btn.setEnabled(False)
        else:
            self.send()

    def send(self) -> None:
        if self._handle is not None:
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        self.composer.clear()
        self.canvas.add_message("user", text)
        self.history.append({"role": "user", "content": text})
        view = self.canvas.add_message("assistant", "", self.display_name())
        view.set_busy(True, "waiting for the runtime…")
        self._current = view
        grounding: dict[str, Any] = {}
        if self.use_docs.isChecked() and self.collection.currentData():
            grounding["rag"] = {"collection": self.collection.currentData(),
                                "top_k": int(self.ctx.settings.get("rag.top_k", 4))}
        if self.use_web.isChecked():
            grounding["web"] = {"enabled": True, "searxng_url": self.ctx.settings.get("web_search.searxng_url")}
        payload = {**self.payload(), "messages": list(self.history),
                   "system_prompt": self.system_prompt.toPlainText().strip() or None,
                   "params": self.params.values(), "grounding": grounding}
        self._started = time.monotonic()
        handle = self.ctx.runtime.request(Command.GENERATE_STREAM, payload)
        handle.status.connect(lambda p: self._on_status(view, p))
        handle.token.connect(self._on_token)
        handle.context.connect(self._on_context)
        handle.finished.connect(self._on_finished)
        handle.failed.connect(self._on_failed)
        self._handle = handle
        self._set_running(True)

    def _on_status(self, view: MessageView, payload: dict[str, Any]) -> None:
        if payload.get("warning"):
            self.ctx.toast(payload.get("message", ""), "warning")
            return
        view.set_busy(True, payload.get("message", ""))

    def _on_token(self, text: str) -> None:
        if self._current is not None:
            if self._current.spinner.isVisible():
                self._current.set_busy(False, "")
            self._current.append_text(text)

    def _on_context(self, payload: dict[str, Any]) -> None:
        sources = payload.get("sources") or []
        self.sources.set_sources(sources)
        if self._current is not None:
            self._current.set_sources(sources)

    def _on_finished(self, payload: dict[str, Any]) -> None:
        view = self._current
        if view is not None:
            view.set_busy(False)
            view.set_text(payload.get("text", view.text))
            view.set_metrics(payload.get("metrics") or {})
        self.history.append({"role": "assistant", "content": payload.get("text", "")})
        self._done()

    def _on_failed(self, payload: dict[str, Any]) -> None:
        view = self._current
        code = payload.get("code")
        if code == "CANCELLED" and view is not None and view.text:
            view.set_busy(False)
            self.history.append({"role": "assistant", "content": view.text})
        elif view is not None:
            message = payload.get("message", "Generation failed.")
            hint = payload.get("hint")
            if code == "OOM":
                hint = (hint or "") + "\n\nOpen **Optimize** to build a 4-bit version that fits."
            view.set_error(message, hint)
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            if payload.get("component") == "llama.cpp":
                from noelle.app.ui.component_dialog import ensure_llama_cpp

                ensure_llama_cpp(self, lambda: self.ctx.toast("llama.cpp tools installed — send your message again.",
                                                              "success"), reason=message)
        self._done()

    def _done(self) -> None:
        self._handle = None
        self._current = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.send_btn.setEnabled(True)
        if running:
            self.send_btn.setText("Stop")
            self.send_btn.setProperty("variant", "danger")
        else:
            self.send_btn.setText("Send")
            self.send_btn.setProperty("variant", "primary")
        theme.set_prop(self.send_btn, "variant", self.send_btn.property("variant"))

    def export_chat(self) -> None:
        if not self.history:
            self.ctx.toast("Nothing to export yet.", "info")
            return
        default = str(paths.exports_dir() / f"chat-{time.strftime('%Y%m%d-%H%M%S')}.md")
        path, _ = QFileDialog.getSaveFileName(self, "Export chat", default, "Markdown (*.md);;JSON (*.json)")
        if not path:
            return
        if path.endswith(".json"):
            data = {"model": self.payload().get("model_id"), "system_prompt": self.system_prompt.toPlainText(),
                    "params": self.params.values(), "messages": self.history}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        else:
            lines = [f"# Chat with {self.payload().get('model_id')}", ""]
            for turn in self.history:
                lines += [f"**{turn['role'].capitalize()}:**", "", turn["content"], ""]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        self.ctx.toast(f"Saved {path}", "success")

    def shutdown(self) -> None:
        if self._handle is not None:
            self._handle.cancel()

