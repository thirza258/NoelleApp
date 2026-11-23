"""API server page: local OpenAI-compatible endpoint for Cursor, Continue, LangChain and scripts."""

from __future__ import annotations

import secrets

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import Banner, Panel, Pill, button, label

STATE_LEVEL = {"running": "green", "starting": "yellow", "error": "red", "stopped": "unknown"}
ENDPOINTS = [("GET", "/v1/models", "Downloaded models ready to serve"),
             ("POST", "/v1/chat/completions", "Chat completions, streaming via Server-Sent Events"),
             ("POST", "/v1/embeddings", "Dense vectors from a local sentence-transformer"),
             ("POST", "/v1/audio/transcriptions", "Whisper-style transcription of an uploaded file"),
             ("GET", "/health", "Liveness probe")]


class ServerPage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Local API server", "h2"))
        titles.addWidget(label("An OpenAI-compatible endpoint on 127.0.0.1 so coding assistants and scripts can use "
                               "your local models. It never listens on other network interfaces.", "muted", wrap=True))
        head.addLayout(titles, 1)
        self.state_pill = Pill("Stopped", "unknown")
        head.addWidget(self.state_pill)
        self.toggle_btn = button("Start server", "play", "primary")
        self.toggle_btn.clicked.connect(self._toggle)
        head.addWidget(self.toggle_btn)
        root.addLayout(head)

        self.error = Banner("danger", "")
        self.error.hide()
        root.addWidget(self.error)

        config = Panel()
        config_layout = QHBoxLayout(config)
        config_layout.setContentsMargins(16, 12, 16, 12)
        config_layout.setSpacing(12)
        config_layout.addWidget(label("Port", "muted"))
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(int(ctx.settings.get("server.port", 8765)))
        self.port.valueChanged.connect(lambda v: ctx.settings.set("server.port", int(v)))
        config_layout.addWidget(self.port)
        config_layout.addWidget(label("API key", "muted"))
        self.api_key = QLineEdit(str(ctx.settings.get("server.api_key") or ""))
        self.api_key.setPlaceholderText("optional — required as Bearer token when set")
        self.api_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.api_key.editingFinished.connect(lambda: ctx.settings.set("server.api_key", self.api_key.text().strip() or None))
        config_layout.addWidget(self.api_key, 1)
        generate = button("Generate", "key", "ghost")
        generate.clicked.connect(self._generate_key)
        config_layout.addWidget(generate)
        self.autostart = QCheckBox("Start with the runtime")
        self.autostart.setChecked(bool(ctx.settings.get("server.autostart")))
        self.autostart.toggled.connect(lambda on: ctx.settings.set("server.autostart", bool(on)))
        config_layout.addWidget(self.autostart)
        root.addWidget(config)

        row = QHBoxLayout()
        row.setSpacing(14)
        left = QVBoxLayout()
        left.addWidget(label("ENDPOINTS", "eyebrow"))
        self.base_url = label("", "mono", selectable=True)
        left.addWidget(self.base_url)
        endpoints = QListWidget()
        for method, path, text in ENDPOINTS:
            endpoints.addItem(f"{method:5} {path}\n      {text}")
        endpoints.setMaximumHeight(200)
        left.addWidget(endpoints)
        left.addWidget(label("MODELS SERVED", "eyebrow"))
        self.models = QListWidget()
        left.addWidget(self.models, 1)
        row.addLayout(left, 2)

        right = QVBoxLayout()
        right.addWidget(label("CONNECT YOUR TOOLS", "eyebrow"))
        self.snippets = QTabWidget()
        self.snippets.setDocumentMode(True)
        self.snippet_editors: dict[str, QPlainTextEdit] = {}
        for name in ("curl", "Python", "Cursor", "Continue", "LangChain"):
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            editor.setProperty("code", True)
            self.snippet_editors[name] = editor
            self.snippets.addTab(editor, name)
        right.addWidget(self.snippets, 2)
        copy = button("Copy snippet", "copy", "ghost")
        copy.clicked.connect(self._copy_snippet)
        right.addWidget(copy)
        right.addWidget(label("ACCESS LOG", "eyebrow"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("code", True)
        self.log.setMaximumBlockCount(400)
        right.addWidget(self.log, 1)
        row.addLayout(right, 3)
        root.addLayout(row, 1)

        ctx.server.state_changed.connect(self._on_state)
        ctx.server.log_line.connect(self.log.appendPlainText)
        self._access_log = paths.logs_dir() / "server-access.log"
        self._access_offset = self._access_log.stat().st_size if self._access_log.exists() else 0
        self._tail_timer = QTimer(self)
        self._tail_timer.setInterval(1000)
        self._tail_timer.timeout.connect(self._tail_access_log)
        self._tail_timer.start()
        ctx.runtime.state_changed.connect(lambda _s: self._on_state(ctx.server.state))
        self._on_state(ctx.server.state)

    def on_activated(self) -> None:
        self._refresh_models()
        self._render_snippets()

    def _tail_access_log(self) -> None:
        """Append requests written by the server process (method, path, status, latency — never content)."""
        try:
            size = self._access_log.stat().st_size
        except OSError:
            return
        if size < self._access_offset:  # rotated
            self._access_offset = 0
        if size == self._access_offset:
            return
        with self._access_log.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self._access_offset)
            chunk = fh.read(64_000)
            self._access_offset = fh.tell()
        for line in chunk.splitlines():
            if line.strip():
                self.log.appendPlainText(line)

    def _refresh_models(self) -> None:
        self.models.clear()
        for record in self.ctx.registry.list_models(only_existing=True):
            self.models.addItem(f"{record.api_name}   ·   {record.pipeline_tag or '?'}")
        if not self.models.count():
            self.models.addItem("No downloaded models yet.")

    def _toggle(self) -> None:
        if self.ctx.server.state in ("running", "starting"):
            self.ctx.server.stop()
        else:
            self.ctx.settings.set("server.api_key", self.api_key.text().strip() or None)
            self.ctx.server.start()

    def _generate_key(self) -> None:
        self.api_key.setText("nl-" + secrets.token_urlsafe(24))
        self.ctx.settings.set("server.api_key", self.api_key.text())
        self._render_snippets()

    def _on_state(self, state: str) -> None:
        self.state_pill.set_level(STATE_LEVEL.get(state, "unknown"))
        self.state_pill.set_text(state.capitalize())
        running = state in ("running", "starting")
        self.toggle_btn.setText("Stop server" if running else "Start server")
        self.toggle_btn.setEnabled(self.ctx.runtime.is_ready or running)
        self.port.setEnabled(not running)
        if state == "error" and self.ctx.server.last_error:
            self.error.set_text(self.ctx.server.last_error)
            self.error.show()
        else:
            self.error.hide()
        self._render_snippets()

    def _render_snippets(self) -> None:
        port = self.port.value()
        base = f"http://127.0.0.1:{port}/v1"
        self.base_url.setText(f"Base URL  {base}")
        records = self.ctx.registry.list_models(only_existing=True)
        model = next((r.api_name for r in records if r.pipeline_tag == "text-generation"),
                     "HuggingFaceTB/SmolLM2-135M-Instruct")
        key = self.api_key.text().strip() or "not-needed"
        auth = f' \\\n  -H "Authorization: Bearer {key}"' if self.api_key.text().strip() else ""
        snippets: dict[str, str] = {
            "curl": (f"curl {base}/chat/completions{auth} \\\n  -H 'Content-Type: application/json' \\\n"
                     f"  -d '{{\"model\": \"{model}\", \"stream\": true,\n"
                     "       \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"),
            "Python": ("from openai import OpenAI\n\n"
                       f"client = OpenAI(base_url=\"{base}\", api_key=\"{key}\")\n"
                       f"stream = client.chat.completions.create(\n    model=\"{model}\",\n"
                       "    messages=[{\"role\": \"user\", \"content\": \"Hello!\"}],\n    stream=True,\n)\n"
                       "for chunk in stream:\n    print(chunk.choices[0].delta.content or \"\", end=\"\")\n"),
            "Cursor": ("Cursor › Settings › Models:\n"
                       f"  • Override OpenAI Base URL:  {base}\n  • OpenAI API Key:  {key}\n"
                       f"  • Add custom model:  {model}\n\n"
                       "Cursor may route requests through its own servers; tools that call the URL directly from "
                       "your machine (Continue, scripts) keep traffic local."),
            "Continue": ("# ~/.continue/config.yaml\nmodels:\n"
                         f"  - name: Noelle · {model.split('/')[-1]}\n    provider: openai\n    model: {model}\n"
                         f"    apiBase: {base}\n    apiKey: {key}\n"),
            "LangChain": ("from langchain_openai import ChatOpenAI\n\n"
                          f"llm = ChatOpenAI(base_url=\"{base}\", api_key=\"{key}\", model=\"{model}\", streaming=True)\n"
                          "print(llm.invoke(\"Hello!\").content)\n"),
        }
        for name, text in snippets.items():
            self.snippet_editors[name].setPlainText(text)

    def _copy_snippet(self) -> None:
        editor = self.snippets.currentWidget()
        if isinstance(editor, QPlainTextEdit):
            QGuiApplication.clipboard().setText(editor.toPlainText())
            self.ctx.toast("Snippet copied", "success")

    def start(self) -> None:
        self._refresh_models()
        self._render_snippets()

