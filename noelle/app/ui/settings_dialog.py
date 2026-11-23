"""Settings: Hugging Face account, storage, runtime and environments, grounding."""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noelle import __version__, paths
from noelle.app.context import AppContext
from noelle.app.services import hf_client
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import Banner, button, label

RUNTIME_KEYS = ("runtime.device", "runtime.memory_budget_gb", "runtime.safety_headroom", "runtime.keep_models_loaded",
                "runtime.offline", "runtime.environment")


class SettingsDialog(QDialog):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.runtime_restart_needed = False
        self._before = {key: ctx.settings.get(key) for key in RUNTIME_KEYS}
        self._env_stop: threading.Event | None = None
        self.setWindowTitle("Settings")
        self.setMinimumSize(760, 560)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._account_tab(), "Account")
        tabs.addTab(self._runtime_tab(), "Runtime")
        tabs.addTab(self._environments_tab(), "Environments")
        tabs.addTab(self._storage_tab(), "Storage")
        tabs.addTab(self._grounding_tab(), "Grounding")
        tabs.addTab(self._about_tab(), "About")
        root.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---------------------------------------------------------------- tabs
    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 14, 6, 6)
        layout.setSpacing(12)
        return page, layout

    def _account_tab(self) -> QWidget:
        page, layout = self._page()
        layout.addWidget(label("Hugging Face access token", "h3"))
        layout.addWidget(label("Needed for gated or private repositories. It is stored in your operating system's "
                               "credential vault (Keychain / Credential Manager / Secret Service), sent only to the "
                               "Hub for metadata and downloads, and never passed to inference workers or prompts.",
                               "muted", wrap=True))
        row = QHBoxLayout()
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("hf_…  (leave empty to keep the current token)")
        row.addWidget(self.token, 1)
        verify = button("Verify", "check")
        verify.clicked.connect(self._verify_token)
        row.addWidget(verify)
        remove = button("Remove", "trash", "danger")
        remove.clicked.connect(self._remove_token)
        row.addWidget(remove)
        layout.addLayout(row)
        self.token_status = label("A token is stored." if hf_client.get_token() else "No token stored.", "faint")
        layout.addWidget(self.token_status)
        create = button("Create a token on huggingface.co", "external", "link")
        create.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens")))
        layout.addWidget(create)
        layout.addStretch(1)
        return page

    def _runtime_tab(self) -> QWidget:
        page, layout = self._page()
        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.device = QComboBox()
        for text, value in (("Auto", "auto"), ("NVIDIA CUDA", "cuda"), ("Apple Metal (MPS)", "mps"), ("CPU only", "cpu")):
            self.device.addItem(text, value)
        self.device.setCurrentIndex(max(self.device.findData(self.ctx.settings.get("runtime.device", "auto")), 0))
        form.addRow("Preferred device", self.device)
        self.budget = QDoubleSpinBox()
        self.budget.setRange(0, 1024)
        self.budget.setDecimals(1)
        self.budget.setSuffix(" GB")
        self.budget.setSpecialValueText("Auto (detected)")
        self.budget.setValue(float(self.ctx.settings.get("runtime.memory_budget_gb") or 0))
        form.addRow("Memory budget override", self.budget)
        self.headroom = QSpinBox()
        self.headroom.setRange(5, 50)
        self.headroom.setSuffix(" %")
        self.headroom.setValue(int(float(self.ctx.settings.get("runtime.safety_headroom", 0.15)) * 100))
        form.addRow("Safety headroom", self.headroom)
        self.keep_loaded = QCheckBox("Keep models resident between requests (LRU eviction when memory is needed)")
        self.keep_loaded.setChecked(bool(self.ctx.settings.get("runtime.keep_models_loaded", True)))
        form.addRow("", self.keep_loaded)
        self.offline = QCheckBox("Inference runtime never contacts the Hub (HF_HUB_OFFLINE)")
        self.offline.setChecked(bool(self.ctx.settings.get("runtime.offline", True)))
        form.addRow("", self.offline)
        layout.addLayout(form)
        layout.addWidget(Banner("info", "Changing these restarts the model runtime; loaded models are unloaded."))
        layout.addStretch(1)
        return page

    def _environments_tab(self) -> QWidget:
        page, layout = self._page()
        layout.addWidget(label("Isolated runtime environments", "h3"))
        layout.addWidget(label("Create separate virtual environments (under ~/.noelle/environments) when model families "
                               "need conflicting dependencies. The runtime can run from any verified environment.",
                               "muted", wrap=True))
        row = QHBoxLayout()
        row.addWidget(label("Run the runtime from", "muted"))
        self.environment = QComboBox()
        row.addWidget(self.environment, 1)
        layout.addLayout(row)
        self.env_list = QListWidget()
        self.env_list.setMaximumHeight(130)
        layout.addWidget(self.env_list)
        buttons = QHBoxLayout()
        create = button("Create…", "plus")
        create.clicked.connect(self._create_env)
        buttons.addWidget(create)
        verify = button("Verify", "check")
        verify.clicked.connect(self._verify_env)
        buttons.addWidget(verify)
        remove = button("Remove", "trash", "danger")
        remove.clicked.connect(self._remove_env)
        buttons.addWidget(remove)
        self.env_cancel = button("Cancel install", "x", "ghost")
        self.env_cancel.clicked.connect(lambda: self._env_stop and self._env_stop.set())
        self.env_cancel.hide()
        buttons.addWidget(self.env_cancel)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.env_log = QPlainTextEdit()
        self.env_log.setReadOnly(True)
        self.env_log.setProperty("code", True)
        self.env_log.setMaximumBlockCount(2000)
        layout.addWidget(self.env_log, 1)
        self._refresh_envs()
        return page

    def _storage_tab(self) -> QWidget:
        page, layout = self._page()
        form = QFormLayout()
        form.setVerticalSpacing(10)
        hf_home = self.ctx.settings.get("storage.hf_home")
        form.addRow("Hugging Face cache", label(str(paths.hf_hub_cache(hf_home)), "mono", selectable=True))
        form.addRow("Noelle data", label(str(paths.home()), "mono", selectable=True))
        self.download_mode = QComboBox()
        self.download_mode.addItem("Hugging Face cache (deduplicated, shared with other tools)", "hf_cache")
        self.download_mode.addItem("Explicit folders under ~/.noelle/models", "local_dir")
        self.download_mode.setCurrentIndex(max(self.download_mode.findData(self.ctx.settings.get("storage.download_mode")), 0))
        form.addRow("Download into", self.download_mode)
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(int(self.ctx.settings.get("storage.max_download_workers", 8)))
        form.addRow("Parallel file downloads", self.workers)
        self.verify = QCheckBox("Verify SHA-256 of every downloaded weight file")
        self.verify.setChecked(bool(self.ctx.settings.get("storage.verify_checksums", True)))
        form.addRow("", self.verify)
        layout.addLayout(form)
        layout.addWidget(label("To move the cache to another drive, use Storage › Relocate cache.", "faint"))
        layout.addStretch(1)
        return page

    def _grounding_tab(self) -> QWidget:
        page, layout = self._page()
        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.searxng = QLineEdit(str(self.ctx.settings.get("web_search.searxng_url") or ""))
        self.searxng.setPlaceholderText("http://127.0.0.1:8888 (a SearXNG instance with JSON output enabled)")
        form.addRow("Web search (SearXNG)", self.searxng)
        self.embed_model = QLineEdit(str(self.ctx.settings.get("rag.embedding_model")))
        form.addRow("Embedding model", self.embed_model)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 20)
        self.top_k.setValue(int(self.ctx.settings.get("rag.top_k", 4)))
        form.addRow("Passages per answer", self.top_k)
        self.chunk = QSpinBox()
        self.chunk.setRange(64, 2048)
        self.chunk.setSingleStep(64)
        self.chunk.setValue(int(self.ctx.settings.get("rag.chunk_tokens", 512)))
        form.addRow("Chunk size (tokens)", self.chunk)
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 512)
        self.overlap.setValue(int(self.ctx.settings.get("rag.chunk_overlap", 64)))
        form.addRow("Chunk overlap (tokens)", self.overlap)
        self.web_budget = QSpinBox()
        self.web_budget.setRange(1000, 32000)
        self.web_budget.setSingleStep(1000)
        self.web_budget.setValue(int(self.ctx.settings.get("web_search.context_chars", 8000)))
        form.addRow("Web context budget (chars)", self.web_budget)
        layout.addLayout(form)
        layout.addWidget(label("Document grounding never leaves this machine. Web search sends only the generated "
                               "search queries to your configured search provider.", "faint", wrap=True))
        layout.addStretch(1)
        return page

    def _about_tab(self) -> QWidget:
        page, layout = self._page()
        layout.addWidget(label(f"Noelle Desktop AI Studio {__version__}", "h3"))
        layout.addWidget(label("Local-first workbench for open-weight models. Inference, document indexing and the "
                               "API server run on this machine.", "muted", wrap=True))
        for caption, path in (("Data folder", paths.home()), ("Logs", paths.logs_dir()),
                              ("Exports", paths.exports_dir())):
            btn = button(f"Open {caption.lower()} — {path}", "folder", "link")
            btn.clicked.connect(lambda _=False, p=path: QDesktopServices.openUrl(QUrl.fromLocalFile(str(p))))
            layout.addWidget(btn)
        layout.addStretch(1)
        return page

    # --------------------------------------------------------------- token
    def _verify_token(self) -> None:
        token = self.token.text().strip() or hf_client.get_token()
        if not token:
            self.token_status.setText("Enter a token first.")
            return
        self.token_status.setText("Checking…")
        runner().submit(self.ctx.hf.whoami, token,
                        on_result=lambda info: self.token_status.setText(f"Valid — signed in as {info.get('name')}"),
                        on_error=lambda msg: self.token_status.setText(f"Token rejected: {msg}"))

    def _remove_token(self) -> None:
        hf_client.clear_token()
        self.token.clear()
        self.token_status.setText("Token removed from the credential store.")

    # -------------------------------------------------------- environments
    def _refresh_envs(self) -> None:
        from noelle.runtime import environment_manager

        current = self.ctx.settings.get("runtime.environment")
        self.environment.clear()
        self.environment.addItem("This application's Python", None)
        self.env_list.clear()
        for env in environment_manager.list_environments():
            verified = "verified" if env.get("verified") else "not verified"
            item = QListWidgetItem(f"{env['name']} · {', '.join(env.get('profiles') or [])} · {verified}"
                                   + (f" · torch {env['torch_version']} on {env.get('device')}" if env.get("torch_version")
                                      else ""))
            item.setData(256, env["name"])
            self.env_list.addItem(item)
            self.environment.addItem(env["name"], env["name"])
        self.environment.setCurrentIndex(max(self.environment.findData(current), 0))

    def _selected_env(self) -> str | None:
        item = self.env_list.currentItem()
        return item.data(256) if item else None

    def _create_env(self) -> None:
        from noelle.runtime import environment_manager

        name, ok = QInputDialog.getText(self, "New environment", "Name (letters, digits, - _ .):")
        if not ok or not name.strip():
            return
        options = sorted(environment_manager.PROFILES)
        profiles, ok = QInputDialog.getText(self, "Profiles", f"Profiles to install ({', '.join(options)}):",
                                            text="base,rag")
        if not ok:
            return
        chosen = [p.strip() for p in profiles.split(",") if p.strip()]
        self._env_stop = threading.Event()
        self.env_cancel.show()
        self.env_log.appendPlainText(f"$ creating {name} with {', '.join(chosen)}")

        def work(progress: Any) -> dict[str, Any]:
            return environment_manager.create_environment(name.strip(), chosen, progress=progress,
                                                          stop_event=self._env_stop)

        runner().submit(work, on_progress=lambda line: self.env_log.appendPlainText(str(line)),
                        on_result=lambda _r: (self.env_log.appendPlainText("✓ environment ready"), self._refresh_envs()),
                        on_error=lambda msg: self.env_log.appendPlainText(f"✗ {msg}"),
                        on_finished=self.env_cancel.hide)

    def _verify_env(self) -> None:
        from noelle.runtime import environment_manager

        name = self._selected_env()
        if not name:
            return
        self.env_log.appendPlainText(f"$ verifying {name}")
        runner().submit(environment_manager.verify_environment, name,
                        on_result=lambda r: (self.env_log.appendPlainText(str(r)), self._refresh_envs()),
                        on_error=lambda msg: self.env_log.appendPlainText(f"✗ {msg}"))

    def _remove_env(self) -> None:
        from noelle.runtime import environment_manager

        name = self._selected_env()
        if not name:
            return
        if QMessageBox.question(self, "Remove environment", f"Delete the environment '{name}' and its packages?") \
                != QMessageBox.StandardButton.Yes:
            return
        environment_manager.remove_environment(name)
        self._refresh_envs()

    # ---------------------------------------------------------------- save
    def _save(self) -> None:
        token = self.token.text().strip()
        if token:
            try:
                hf_client.set_token(token)
            except Exception as exc:
                QMessageBox.warning(self, "Token not saved", f"The credential store refused the token: {exc}")
                return
        s = self.ctx.settings
        s.update({
            "runtime.device": self.device.currentData(),
            "runtime.memory_budget_gb": self.budget.value() or None,
            "runtime.safety_headroom": self.headroom.value() / 100,
            "runtime.keep_models_loaded": self.keep_loaded.isChecked(),
            "runtime.offline": self.offline.isChecked(),
            "runtime.environment": self.environment.currentData(),
            "storage.download_mode": self.download_mode.currentData(),
            "storage.max_download_workers": self.workers.value(),
            "storage.verify_checksums": self.verify.isChecked(),
            "web_search.searxng_url": self.searxng.text().strip() or None,
            "web_search.context_chars": self.web_budget.value(),
            "rag.embedding_model": self.embed_model.text().strip(),
            "rag.top_k": self.top_k.value(),
            "rag.chunk_tokens": self.chunk.value(),
            "rag.chunk_overlap": self.overlap.value(),
        })
        self.runtime_restart_needed = any(s.get(k) != v for k, v in self._before.items())
        self.accept()
