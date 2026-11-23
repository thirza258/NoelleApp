"""API server page: local OpenAI-compatible endpoint for Cursor, Continue, LangChain and scripts."""

from __future__ import annotations

import json
import re
import secrets

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import Banner, Panel, Pill, button, label
from noelle.runtime import tasks

STATE_LEVEL = {"running": "green", "starting": "yellow", "error": "red", "stopped": "unknown"}
ENDPOINTS = [("GET", "/v1/models", "Downloaded models, your aliases and auto"),
             ("POST", "/v1/chat/completions", "Chat completions, streaming via Server-Sent Events"),
             ("POST", "/v1/embeddings", "Dense vectors from a local sentence-transformer"),
             ("POST", "/v1/audio/transcriptions", "Whisper-style transcription of an uploaded file"),
             ("POST", "/v1/images/generations", "Text-to-image with a local diffusion model"),
             ("GET", "/v1/knowledge", "Document collections (Documents tab)"),
             ("POST", "/v1/knowledge/query", "Best passages from one collection"),
             ("GET", "/health", "Liveness probe")]
# Same rule as noelle.runtime.server_service.ALIAS_PATTERN (full match); the shell never imports the server.
ALIAS_PATTERN = r"[a-z0-9][a-z0-9._-]{0,63}"
AUTO_NOTE = ('Send "model": "auto" to let Noelle choose. Chat that looks like code goes to noelle-code, other chat '
             "to noelle-chat, then to the server's default model, a text model already loaded in the runtime, or "
             "your most recently used text model. Embeddings use noelle-embed, a loaded embedding model, the "
             "Documents embedding model, then the most recent one; images use noelle-image or your most recent "
             "diffusion model. The x-noelle-routed-model and x-noelle-route-reason response headers show the choice.")


def alias_error(name: str, target: str | None, *, aliases: dict[str, str], models: set[str],
                editing: str | None) -> str | None:
    """Why ``name -> target`` cannot be saved (``editing`` is the alias being changed), or None."""
    if name == "auto":
        return "'auto' is reserved: it asks the server to choose a model."
    if not re.fullmatch(ALIAS_PATTERN, name):
        return "Use 1-64 lower-case letters, digits, '.', '_' or '-', starting with a letter or digit."
    if name in models:
        return f"'{name}' is already the name of a downloaded model."
    if name in aliases and name != editing:
        return f"The alias '{name}' already exists. Select it in the list to change it."
    if not target:
        return "Choose the model this alias points to."
    return None


def gateway_snippets(base: str, image_model: str, *, keyed: bool) -> dict[str, str]:
    """curl examples for images and knowledge. With an API key they read $NOELLE_API_KEY: never the key itself."""
    auth = ' \\\n  -H "Authorization: Bearer $NOELLE_API_KEY"' if keyed else ""
    setup = "# First: export NOELLE_API_KEY=<the API key above>\n" if keyed else ""
    image_body = json.dumps({"model": image_model, "prompt": "a lighthouse at dawn, watercolor", "size": "512x512",
                             "steps": 25, "seed": 42})
    query_body = json.dumps({"collection": "my-docs", "query": "When does the contract renew?", "top_k": 4})
    images = (f"{setup}# One PNG, returned as base64 and saved as image.png\n"
              f"curl {base}/images/generations{auth} \\\n  -H 'Content-Type: application/json' \\\n"
              f"  -d '{image_body}' \\\n"
              "  | python3 -c 'import base64, json, sys; open(\"image.png\", \"wb\").write("
              "base64.b64decode(json.load(sys.stdin)[\"data\"][0][\"b64_json\"]))'\n")
    knowledge = (f"{setup}# Document collections (built in a model's Documents tab)\n"
                 f"curl {base}/knowledge{auth}\n\n"
                 "# The best passages from one collection\n"
                 f"curl {base}/knowledge/query{auth} \\\n  -H 'Content-Type: application/json' \\\n"
                 f"  -d '{query_body}'\n")
    return {"Images": images, "Knowledge": knowledge}


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
        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)
        self.models = QListWidget()
        self.left_tabs.addTab(self.models, "Models served")
        self.left_tabs.addTab(self._build_aliases(), "Aliases")
        left.addWidget(self.left_tabs, 1)
        row.addLayout(left, 2)

        right = QVBoxLayout()
        right.addWidget(label("CONNECT YOUR TOOLS", "eyebrow"))
        self.snippets = QTabWidget()
        self.snippets.setDocumentMode(True)
        self.snippet_editors: dict[str, QPlainTextEdit] = {}
        for name in ("curl", "Python", "Cursor", "Continue", "LangChain", "Images", "Knowledge"):
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

    def _build_aliases(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(label("Stable names for your models: clients keep the alias while you change the model "
                               "behind it. Changes reach the running server at once.", "muted", wrap=True))
        self.alias_table = QTableWidget(0, 2)
        self.alias_table.setHorizontalHeaderLabels(["Alias", "Model"])
        self.alias_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.alias_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.alias_table.verticalHeader().hide()
        self.alias_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.alias_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.alias_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.alias_table.itemSelectionChanged.connect(self._on_alias_selected)
        layout.addWidget(self.alias_table, 1)
        form = QHBoxLayout()
        form.setSpacing(8)
        self.alias_name = QLineEdit()
        self.alias_name.setPlaceholderText("alias, e.g. noelle-chat")
        self.alias_name.setMinimumWidth(160)
        self.alias_name.returnPressed.connect(self._save_alias)
        form.addWidget(self.alias_name, 2)
        self.alias_target = QComboBox()
        self.alias_target.setPlaceholderText("Choose a downloaded model")
        self.alias_target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.alias_target.setMinimumContentsLength(18)
        form.addWidget(self.alias_target, 3)
        layout.addLayout(form)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.alias_save = button("Save alias", "save", "primary")
        self.alias_save.clicked.connect(self._save_alias)
        actions.addWidget(self.alias_save)
        self.alias_new = button("New", "plus", "ghost")
        self.alias_new.clicked.connect(self._new_alias)
        actions.addWidget(self.alias_new)
        self.alias_remove = button("Remove", "trash", "ghost")
        self.alias_remove.clicked.connect(self._remove_alias)
        actions.addWidget(self.alias_remove)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.alias_error = label("", "error", wrap=True)
        self.alias_error.hide()
        layout.addWidget(self.alias_error)
        layout.addWidget(label(AUTO_NOTE, "faint", wrap=True))
        self._editing: str | None = None  # alias loaded into the form; saving under another name renames it
        self._targets: dict[str, str] = {}  # downloaded api_name -> pipeline tag
        self._refresh_aliases()
        return page

    def _saved_aliases(self) -> dict[str, str]:
        raw = self.ctx.settings.get("server.aliases")
        if not isinstance(raw, dict):
            return {}
        return {name: target for name, target in raw.items() if isinstance(name, str) and isinstance(target, str)}

    def _refresh_aliases(self) -> None:
        """Rebuild the model choices and the alias table; the alias being edited stays selected."""
        self._targets = {}
        for record in self.ctx.registry.list_models(only_existing=True):
            self._targets.setdefault(record.api_name, record.pipeline_tag or "?")
        chosen = self.alias_target.currentData()
        self.alias_target.clear()
        for api_name, tag in self._targets.items():
            self.alias_target.addItem(f"{api_name}  ·  {tag}", api_name)
        self.alias_target.setCurrentIndex(self.alias_target.findData(chosen) if chosen else -1)
        self.alias_table.blockSignals(True)
        self.alias_table.setRowCount(0)
        selected = -1
        for row, (alias, target) in enumerate(sorted(self._saved_aliases().items())):
            self.alias_table.insertRow(row)
            self.alias_table.setItem(row, 0, QTableWidgetItem(alias))
            item = QTableWidgetItem(target)
            if target not in self._targets:
                item.setText(f"{target}  (not downloaded)")
                item.setForeground(QColor(theme.WARNING))
                item.setToolTip("Requests for this alias fail until you choose a downloaded model for it.")
            self.alias_table.setItem(row, 1, item)
            if alias == self._editing:
                selected = row
        if selected >= 0:
            self.alias_table.selectRow(selected)
        else:
            self._editing = None
        self.alias_table.blockSignals(False)
        self.alias_remove.setEnabled(self._editing is not None)

    def _selected_alias(self) -> str | None:
        rows = self.alias_table.selectionModel().selectedRows()
        item = self.alias_table.item(rows[0].row(), 0) if rows else None
        return item.text() if item is not None else None

    def _on_alias_selected(self) -> None:
        self._editing = self._selected_alias()
        self.alias_remove.setEnabled(self._editing is not None)
        self.alias_error.hide()
        if self._editing is not None:
            self.alias_name.setText(self._editing)
            target = self._saved_aliases().get(self._editing)
            self.alias_target.setCurrentIndex(self.alias_target.findData(target))  # -1 if not downloaded

    def _new_alias(self) -> None:
        self.alias_table.clearSelection()
        self._editing = None
        self.alias_remove.setEnabled(False)
        self.alias_name.clear()
        self.alias_target.setCurrentIndex(-1)
        self.alias_error.hide()
        self.alias_name.setFocus()

    def _save_alias(self) -> None:
        name = self.alias_name.text().strip()
        target = self.alias_target.currentData()
        aliases = self._saved_aliases()
        error = alias_error(name, target, aliases=aliases, models=set(self._targets), editing=self._editing)
        if error:
            self.alias_error.setText(error)
            self.alias_error.show()
            return
        if self._editing is not None and self._editing != name:
            aliases.pop(self._editing, None)  # renamed
        aliases[name] = str(target)
        self.ctx.settings.set("server.aliases", aliases)
        self._editing = name
        self.alias_error.hide()
        self._refresh_aliases()
        self.ctx.toast(f"Alias '{name}' now points to {target}", "success")

    def _remove_alias(self) -> None:
        alias = self._selected_alias()
        if alias is None:
            return
        aliases = self._saved_aliases()
        aliases.pop(alias, None)
        self.ctx.settings.set("server.aliases", aliases)
        self._new_alias()
        self._refresh_aliases()
        self.ctx.toast(f"Alias '{alias}' removed", "info")

    def on_activated(self) -> None:
        self._refresh_models()
        self._refresh_aliases()
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
        image_model = next((r.api_name for r in records if r.family == tasks.DIFFUSION), "auto")
        snippets.update(gateway_snippets(base, image_model, keyed=bool(self.api_key.text().strip())))
        for name, text in snippets.items():
            self.snippet_editors[name].setPlainText(text)

    def _copy_snippet(self) -> None:
        editor = self.snippets.currentWidget()
        if isinstance(editor, QPlainTextEdit):
            QGuiApplication.clipboard().setText(editor.toPlainText())
            self.ctx.toast("Snippet copied", "success")

    def start(self) -> None:
        self._refresh_models()
        self._refresh_aliases()
        self._render_snippets()

