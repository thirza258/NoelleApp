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
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.i18n import tr
from noelle.app.services.provider_stream import ConsentGate, ProviderStream
from noelle.app.services.voice_session import VoiceSession
from noelle.app.ui.widgets.audio_visualizer import MicRecorder, request_microphone
from noelle.app.ui.widgets.chat_canvas import ChatCanvas, MessageView
from noelle.app.ui.widgets.common import SearchField, button, divider, label, open_external_url
from noelle.app.ui.widgets.param_panel import ParamPanel
from noelle.runtime import tools as tool_registry
from noelle.runtime.protocol import Command


def _group_label(timestamp: float) -> str:
    from datetime import datetime

    moment = datetime.fromtimestamp(timestamp or time.time()).date()
    today = datetime.now().date()
    age = (today - moment).days
    if age <= 0:
        return "Today"
    if age == 1:
        return "Yesterday"
    if age < 7:
        return "Last 7 days"
    return "Older"


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
        self._current: MessageView | None = None
        # Chat history (F2): a separate chats.db; private mode never touches it.
        self._store: Any = None
        self._conversation_id: int | None = None
        self._restored = False
        self._last_sources: list[dict[str, Any]] = []
        self._stream: ProviderStream | None = None
        self._tool_rounds = 0
        self._pending_tools = False
        self._voice_view: MessageView | None = None

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
        self.system_toggle = button(tr("chat.system_prompt"), "chevron-down", "ghost", checkable=True)
        self.system_toggle.toggled.connect(lambda on: self.system_prompt.setVisible(on))
        top.addWidget(self.system_toggle)
        top.addStretch(1)
        self.route_label = label(f"● {tr('provider.local')}", "faint")
        self.route_label.setToolTip("Which provider answered the last message")
        top.addWidget(self.route_label)
        self.export_btn = button(tr("chat.export"), "save", "ghost")
        self.export_btn.clicked.connect(self.export_chat)
        top.addWidget(self.export_btn)
        self.clear_btn = button(tr("chat.new"), "plus", "ghost")
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
        self.use_tools = QCheckBox(tr("chat.tools"))
        self.use_tools.setToolTip("Let the model call read-only tools (calculator, date, knowledge/web search)")
        self.use_tools.setChecked(bool(ctx.settings.get("tools.enabled") or []))
        grounding.addWidget(self.use_tools)
        grounding.addStretch(1)
        self.voice_status = label("", "faint")
        grounding.addWidget(self.voice_status)
        self.adapter_label = label("", "faint")
        grounding.addWidget(self.adapter_label)
        column.addLayout(grounding)

        composer_row = QHBoxLayout()
        composer_row.setSpacing(8)
        self.composer = Composer()
        self.composer.submitted.connect(self.send)
        composer_row.addWidget(self.composer, 1)
        self.voice_btn = button("", "mic", "ghost", tooltip=tr("voice.input"))
        self.voice_btn.setFixedHeight(76)
        self.voice_btn.setMinimumWidth(52)
        self.voice_btn.setCheckable(True)
        self.voice_btn.clicked.connect(self._toggle_voice)
        composer_row.addWidget(self.voice_btn)
        self.send_btn = button(tr("chat.send"), "send", "primary")
        self.send_btn.setFixedHeight(76)
        self.send_btn.setMinimumWidth(96)
        self.send_btn.clicked.connect(self._send_or_stop)
        composer_row.addWidget(self.send_btn)
        column.addLayout(composer_row)
        splitter.addWidget(chat)

        # ---------------------------------------------------- conversation sidebar
        sidebar = QFrame()
        sidebar.setObjectName("ChatSidebar")
        sidebar.setStyleSheet(f"QFrame#ChatSidebar {{ background: {theme.BG_ELEV};"
                              f" border-right: 1px solid {theme.BORDER}; }}")
        side_box = QVBoxLayout(sidebar)
        side_box.setContentsMargins(12, 14, 12, 12)
        side_box.setSpacing(8)
        heading = QHBoxLayout()
        heading.addWidget(label(tr("chat.title").upper(), "eyebrow"))
        heading.addStretch(1)
        new_btn = button("", "plus", "ghost", tooltip=tr("chat.new"))
        new_btn.clicked.connect(self.new_chat)
        heading.addWidget(new_btn)
        side_box.addLayout(heading)
        self.conv_search = SearchField(tr("chat.search"))
        self.conv_search.textChanged.connect(lambda _text: self._refresh_conversations())
        side_box.addWidget(self.conv_search)
        self.conv_list = QListWidget()
        self.conv_list.setWordWrap(True)
        self.conv_list.currentItemChanged.connect(self._on_conversation_selected)
        side_box.addWidget(self.conv_list, 1)
        conv_actions = QHBoxLayout()
        rename = button(tr("chat.rename"), "save", "ghost")
        rename.clicked.connect(self._rename_conversation)
        conv_actions.addWidget(rename)
        pin = button(tr("chat.pin"), "check", "ghost")
        pin.clicked.connect(self._toggle_pin)
        conv_actions.addWidget(pin)
        delete = button(tr("chat.delete"), "trash", "ghost")
        delete.clicked.connect(self._delete_conversation)
        conv_actions.addWidget(delete)
        side_box.addLayout(conv_actions)
        splitter.insertWidget(0, sidebar)
        splitter.setStretchFactor(1, 1)

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
        side_layout.addWidget(label("SPEED-UP", "eyebrow"))
        self.speedup = QComboBox()
        for text, value in (("Off", "off"), ("Prompt lookup", "prompt_lookup"), ("Draft model", "draft")):
            self.speedup.addItem(text, value)
        current_mode = str(ctx.settings.get("generation.speculative.mode", "off"))
        self.speedup.setCurrentIndex(max(self.speedup.findData(current_mode), 0))
        side_layout.addWidget(self.speedup)
        self.draft_model = QComboBox()
        self.draft_model.addItem("No draft model", None)
        try:
            for record in ctx.registry.list_models(family="text"):
                if record.exists_on_disk():
                    self.draft_model.addItem(record.repo_id, record.repo_id)
        except Exception:  # noqa: BLE001
            pass
        self.draft_model.setCurrentIndex(max(self.draft_model.findData(
            ctx.settings.get("generation.speculative.draft_model")), 0))
        self.draft_model.setVisible(current_mode == "draft")
        self.speedup.currentIndexChanged.connect(
            lambda _i: self.draft_model.setVisible(self.speedup.currentData() == "draft"))
        side_layout.addWidget(self.draft_model)
        side_layout.addWidget(divider())
        self.sources = SourcesPanel()
        side_layout.addWidget(self.sources, 1)
        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([260, 900, 300])

        self.recorder = MicRecorder(self)
        self.recorder.error.connect(lambda message: self.ctx.toast(message, "error"))
        self.consent_gate = ConsentGate(self)
        self.consent_gate.requested.connect(self._ask_cloud_consent)
        self.voice = VoiceSession(ctx, self)
        self.voice.state_changed.connect(self._on_voice_state)
        self.voice.transcript.connect(self._on_voice_transcript)
        self.voice.token.connect(self._on_voice_token)
        self.voice.finished.connect(self._on_voice_finished)
        self.voice.failed.connect(self._on_voice_failed)

    # ---------------------------------------------------------- lifecycle
    def on_activated(self) -> None:
        self.refresh_collections()
        self._restore_last_conversation()
        self._refresh_conversations()

    # ---------------------------------------------------------- chat history
    def _saving(self) -> bool:
        return bool(self.ctx.settings.get("chats.save", True))

    def _conversations(self) -> Any:
        if self._store is None:
            from noelle.app.services.conversation_store import ConversationStore

            self._store = ConversationStore()
        return self._store

    def _restore_last_conversation(self) -> None:
        if self._restored or not self._saving() or self.history:
            self._restored = True
            return
        self._restored = True
        try:
            recent = self._conversations().list_conversations(limit=1)
        except Exception:  # noqa: BLE001 - history is a convenience, never a startup blocker
            return
        if recent:
            self._load_conversation(recent[0]["id"])

    def _refresh_conversations(self) -> None:
        if not self._saving():
            self.conv_list.clear()
            return
        try:
            rows = self._conversations().list_conversations(search=self.conv_search.text().strip() or None)
        except Exception:  # noqa: BLE001
            return
        self.conv_list.blockSignals(True)
        self.conv_list.clear()
        last_group = None
        for row in rows:
            moment = row.get("updated_at") or 0
            group = time.strftime("%Y-%m-%d", time.localtime(moment))
            if group != last_group:
                header = QListWidgetItem(_group_label(moment))
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                header.setForeground(theme.TEXT_FAINT)
                self.conv_list.addItem(header)
                last_group = group
            title = ("📌 " if row.get("pinned") else "") + str(row.get("title") or "Untitled")
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            item.setToolTip(f"{row.get('message_count', 0)} messages · {row.get('model_id') or 'model unknown'}")
            self.conv_list.addItem(item)
            if row["id"] == self._conversation_id:
                self.conv_list.setCurrentItem(item)
        self.conv_list.blockSignals(False)

    def _load_conversation(self, conversation_id: int) -> None:
        if self._stream is not None:
            self._stream.cancel()
        self._current = None
        try:
            data = self._conversations().get_conversation(conversation_id)
        except Exception:  # noqa: BLE001
            return
        if not data:
            return
        self._conversation_id = conversation_id
        self.history.clear()
        self.canvas.clear()
        self.sources.set_sources([])
        if data.get("system_prompt"):
            self.system_prompt.setPlainText(data["system_prompt"])
        for message in data.get("messages") or []:
            role = message.get("role")
            if role not in ("user", "assistant") or not message.get("content"):
                continue
            self.history.append({"role": role, "content": message["content"]})
            view = self.canvas.add_message(role, message["content"], "" if role == "user" else self.display_name())
            if message.get("sources"):
                view.set_sources(message["sources"])
                self._last_sources = message["sources"]
            if message.get("metrics"):
                view.set_metrics(message["metrics"])
        self._set_running(False)

    def _on_conversation_selected(self, current: QListWidgetItem | None, _previous: Any = None) -> None:
        if current is None or current.data(Qt.ItemDataRole.UserRole) is None:
            return
        conversation_id = int(current.data(Qt.ItemDataRole.UserRole))
        if conversation_id != self._conversation_id:
            self._load_conversation(conversation_id)

    def _selected_conversation(self) -> int | None:
        item = self.conv_list.currentItem()
        value = item.data(Qt.ItemDataRole.UserRole) if item else None
        return int(value) if value is not None else None

    def _rename_conversation(self) -> None:
        conversation_id = self._selected_conversation()
        if conversation_id is None:
            return
        current = self.conv_list.currentItem().text().removeprefix("📌 ")
        title, ok = QInputDialog.getText(self, "Rename chat", "Title:", text=current)
        if ok and title.strip():
            self._conversations().rename_conversation(conversation_id, title.strip())
            self._refresh_conversations()

    def _toggle_pin(self) -> None:
        conversation_id = self._selected_conversation()
        if conversation_id is None:
            return
        data = self._conversations().get_conversation(conversation_id)
        if data:
            self._conversations().pin_conversation(conversation_id, not data.get("pinned"))
            self._refresh_conversations()

    def _delete_conversation(self) -> None:
        conversation_id = self._selected_conversation()
        if conversation_id is None:
            return
        if QMessageBox.question(self, "Delete chat", "Delete this conversation and its messages?") != \
                QMessageBox.StandardButton.Yes:
            return
        self._conversations().delete_conversation(conversation_id)
        if conversation_id == self._conversation_id:
            self._conversation_id = None
            self.history.clear()
            self.canvas.clear()
        self._refresh_conversations()

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
        if self._stream is not None:
            self._stream.cancel()
        self.history.clear()
        self.canvas.clear()
        self.sources.set_sources([])
        self._conversation_id = None
        self._restored = True
        self._last_sources = []
        self._tool_rounds = 0
        self.conv_list.blockSignals(True)
        self.conv_list.clearSelection()
        self.conv_list.blockSignals(False)

    def _send_or_stop(self) -> None:
        if self._stream is not None and self._stream.running:
            self._stream.cancel()
            self.send_btn.setEnabled(False)
        else:
            self.send()

    def _toggle_voice(self) -> None:
        if self.voice.state not in ("idle", "recording"):
            self.voice.stop()
            self.voice_btn.setChecked(False)
            return
        if self.recorder.recording:
            path = self.recorder.stop()
            self.voice_btn.setChecked(False)
            if not path:
                self.ctx.toast("No audio was recorded.", "warning")
                return
            if self._stream is not None and self._stream.running:
                self._stream.cancel()
            self._stream_messages_voice(path)
            return

        def granted(ok: bool) -> None:
            if not ok:
                self.voice_btn.setChecked(False)
                self.ctx.toast("Microphone access is blocked. Allow it in the system privacy settings.", "error")
                return
            self.recorder.start()
            if self.recorder.recording:
                self.voice_btn.setChecked(True)
                self.voice.set_recording(True)
                self.voice_status.setText(tr("voice.listening"))
            else:
                self.voice_btn.setChecked(False)

        request_microphone(granted)

    def _stream_messages_voice(self, audio_path: str) -> None:
        self.voice_status.setText(tr("voice.transcribing"))
        system_prompt = self.system_prompt.toPlainText().strip() or None
        if not self.voice.start_turn(audio_path, list(self.history), llm_payload=self.payload(),
                                     system_prompt=system_prompt, params=self.params.values(),
                                     language=None if self.ctx.settings.get("voice.language") in (None, "auto")
                                     else str(self.ctx.settings.get("voice.language"))):
            self.voice_status.setText("")

    def _on_voice_state(self, state: str) -> None:
        labels = {"idle": "", "recording": tr("voice.listening"),
                  "transcribing": tr("voice.transcribing"), "generating": tr("voice.generating"),
                  "speaking": tr("voice.speaking")}
        self.voice_status.setText(labels.get(state, ""))
        if state == "recording":
            self.voice_btn.setChecked(True)
        elif state == "idle":
            self.voice_btn.setChecked(False)

    def _on_voice_transcript(self, text: str) -> None:
        if not text.strip():
            return
        self.canvas.add_message("user", text)
        self.history.append({"role": "user", "content": text})
        if self._saving():
            try:
                store = self._conversations()
                if self._conversation_id is None:
                    self._conversation_id = store.create_conversation(
                        model_id=str(self.payload().get("model_id") or ""),
                        system_prompt=self.system_prompt.toPlainText().strip() or None,
                        params=self.params.values())
                store.add_message(self._conversation_id, "user", text, audio_path=None)
            except Exception:  # noqa: BLE001
                pass
        self._voice_view = self.canvas.add_message("assistant", "", self.display_name())
        self._voice_view.set_busy(True, "thinking…")

    def _on_voice_token(self, text: str) -> None:
        if self._voice_view is not None:
            if self._voice_view.spinner.isVisible():
                self._voice_view.set_busy(False, "")
            self._voice_view.append_text(text)

    def _on_voice_finished(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "")
        if self._voice_view is not None:
            self._voice_view.set_busy(False)
            self._voice_view.set_text(text)
            self._voice_view.set_metrics(payload.get("metrics") or {})
        if text:
            self.history.append({"role": "assistant", "content": text})
            self._persist("assistant", text, metrics=payload.get("metrics"))
        self._voice_view = None

    def _on_voice_failed(self, payload: dict[str, Any]) -> None:
        message = str(payload.get("message") or "Voice turn failed.")
        hint = payload.get("hint")
        if self._voice_view is not None:
            self._voice_view.set_error(message, hint)
            self._voice_view = None
        else:
            self.ctx.toast(message + (f" — {hint}" if hint else ""), "error")
        self.voice_status.setText("")

    def send(self) -> None:
        if self.voice.state not in ("idle",):
            self.ctx.toast("Stop the voice turn before typing.", "info")
            return
        if self._stream is not None and self._stream.running:
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        self.composer.clear()
        self.canvas.add_message("user", text)
        self.history.append({"role": "user", "content": text})
        self._tool_rounds = 0
        if self._saving():
            try:
                store = self._conversations()
                if self._conversation_id is None:
                    self._conversation_id = store.create_conversation(
                        model_id=str(self.payload().get("model_id") or ""),
                        system_prompt=self.system_prompt.toPlainText().strip() or None,
                        params=self.params.values())
                store.add_message(self._conversation_id, "user", text)
                self._refresh_conversations()
            except Exception:  # noqa: BLE001 - saving must not block sending
                self._conversation_id = None
        self._stream_messages()

    def _enabled_tools(self) -> list[str]:
        names: list[str] = []
        if self.use_tools.isChecked():
            names += ["calculator", "current_datetime"]
        if self.use_docs.isChecked() and self.collection.currentData():
            names.append("knowledge_search")
        if self.use_web.isChecked():
            names.append("web_search")
        return names

    def _ask_cloud_consent(self, reason: str, destination: str) -> None:
        self.consent_gate.answer(self._cloud_consent(reason, destination))

    def _cloud_consent(self, reason: str, destination: str) -> bool:
        if self.ctx.settings.get("privacy.local_only", False):
            return False
        answer = QMessageBox.question(
            self, "Use the cloud instead?",
            f"The local model could not answer:\n\n{reason}\n\n"
            + tr("provider.cloud_confirm", destination=destination),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _stream_messages(self) -> None:
        view = self.canvas.add_message("assistant", "", self.display_name())
        view.set_busy(True, "waiting for the runtime…")
        self._current = view
        provider = self.ctx.hybrid_provider(consent=self.consent_gate.decide)
        system_prompt = self.system_prompt.toPlainText().strip() or None
        enabled = self._enabled_tools()
        if enabled:
            system_prompt = ((system_prompt + "\n\n") if system_prompt else "") + tool_registry.tool_instructions(enabled)
        options = {key: value for key, value in self.payload().items() if key not in ("model_id", "task")}
        mode = self.speedup.currentData() or "off"
        if mode != "off":
            options["speculative"] = {"mode": mode, "draft_model": self.draft_model.currentData(),
                                      "draft_tokens": 10}
            self.ctx.settings.update({"generation.speculative.mode": mode,
                                      "generation.speculative.draft_model": self.draft_model.currentData()})
        if self._stream is None:
            self._stream = ProviderStream(self)
            self._stream.token.connect(self._on_token)
            self._stream.finished.connect(self._on_finished)
            self._stream.failed.connect(self._on_failed)
        self.route_label.setText("…")
        self._stream.start(provider, list(self.history), model=self.payload().get("model_id"),
                           system_prompt=system_prompt, params=self.params.values(),
                           task=self.payload().get("task") or "text-generation", **options)
        self._set_running(True)

    def _on_token(self, text: str) -> None:
        if self._current is not None:
            if self._current.spinner.isVisible():
                self._current.set_busy(False, "")
            self._current.append_text(text)

    def _on_finished(self, text: str, metrics: dict[str, Any]) -> None:
        view = self._current
        if view is not None:
            view.set_busy(False)
            view.set_text(text)
            view.set_metrics(metrics or {})
        self.history.append({"role": "assistant", "content": text})
        self._persist("assistant", text, metrics=metrics)
        provider = self._stream.provider if self._stream is not None else None
        route = getattr(provider, "last_route", "local")
        reason = getattr(provider, "last_reason", "")
        self.route_label.setText(f"☁ {tr('provider.cloud')}" if route == "cloud" else f"● {tr('provider.local')}")
        self.route_label.setToolTip(reason or "Answered by the local runtime")
        call = tool_registry.parse_tool_call(text) if self._enabled_tools() else None
        max_rounds = int(self.ctx.settings.get("tools.max_rounds", 5))
        if call is not None and self._tool_rounds < max_rounds:
            self._run_tool(call)
            return
        if call is not None:
            self.ctx.toast(tr("tools.limit"), "warning")
        self._last_sources = []
        self._done()

    def _persist(self, role: str, text: str, *, sources: Any = None, metrics: Any = None) -> None:
        if not self._saving() or self._conversation_id is None or not text:
            return
        try:
            self._conversations().add_message(self._conversation_id, role, text, sources=sources,
                                              metrics=metrics)
            self._refresh_conversations()
        except Exception:  # noqa: BLE001
            pass

    def _run_tool(self, call: Any) -> None:
        try:
            tool_registry.validate_call(call)
        except tool_registry.ToolError as exc:
            self.canvas.add_message("assistant", f"🔧 Tool call could not be used: {exc}")
            self._done()
            return
        self._tool_rounds += 1
        card = self.canvas.add_message("assistant", f"🔧 Calling **{call.name}**…")
        self._persist("tool", f"tool_call {call.name} {json.dumps(call.arguments)}")
        provider = self._stream.provider if self._stream is not None else None
        if provider is None:
            self._done()
            return
        arguments = dict(call.arguments)
        if call.name == "knowledge_search" and not arguments.get("collection") and self.collection.currentData():
            arguments["collection"] = self.collection.currentData()

        def work() -> Any:
            return provider.execute_tool(call.name, arguments)

        from noelle.app.services.task_runner import runner

        def done(result: Any) -> None:
            data = result if isinstance(result, dict) else {"summary": str(result), "ok": True}
            summary = str(data.get("summary") or data.get("data") or "done")
            ok = data.get("ok", True)
            card.set_text(f"🔧 **{call.name}** {self._tool_arguments(arguments)}\n\n"
                          f"{'✓' if ok else '✗'} {summary}")
            self.history.append({"role": "user", "content": f"Tool result ({call.name}): {summary}"})
            self._persist("tool", f"tool_result {call.name}: {summary}")
            self._stream_messages()

        def failed(message: str) -> None:
            card.set_text(f"🔧 **{call.name}** failed: {message}")
            self.history.append({"role": "user", "content": f"Tool error ({call.name}): {message}"})
            self._stream_messages()

        runner().submit(work, on_result=done, on_error=failed)

    @staticmethod
    def _tool_arguments(arguments: dict[str, Any]) -> str:
        return " · ".join(f"{key}={value}" for key, value in arguments.items())[:200]

    def _on_failed(self, payload: dict[str, Any]) -> None:
        view = self._current
        code = payload.get("code")
        if code == "CANCELLED" and view is not None:
            view.set_busy(False)
            if view.text:
                self.history.append({"role": "assistant", "content": view.text})
                self._persist("assistant", view.text)
        elif view is not None:
            message = payload.get("message", "Generation failed.")
            hint = payload.get("hint") or ""
            actions = payload.get("suggested_actions") or []
            if code == "OOM":
                hint = (hint + "\n\n" if hint else "") + "Open **Optimize** to build a 4-bit version that fits."
            if actions:
                hint = (hint + "\n\n" if hint else "") + "Try: " + ", ".join(str(a).replace("_", " ") for a in actions)
            view.set_error(message, hint or None)
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            if payload.get("component") == "llama.cpp":
                from noelle.app.ui.component_dialog import ensure_llama_cpp

                ensure_llama_cpp(self, lambda: self.ctx.toast("llama.cpp tools installed — send your message again.",
                                                              "success"), reason=message)
        self._done()

    def _done(self) -> None:
        self._current = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.send_btn.setEnabled(True)
        if running:
            self.send_btn.setText(tr("chat.stop"))
            self.send_btn.setProperty("variant", "danger")
        else:
            self.send_btn.setText(tr("chat.send"))
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
        if self.recorder.recording:
            self.recorder.stop()
        self.voice.shutdown()
        if self._stream is not None:
            self._stream.cancel()

