"""Page 2: model detail, download/runtime control and the multi-modal workspace."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noelle.app import icons, theme
from noelle.app.context import AppContext, summary_from_record
from noelle.app.services.hf_client import ModelSummary
from noelle.app.services.model_registry import ModelRecord
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import (
    Avatar,
    Banner,
    EmptyState,
    ErrorLine,
    FlowLayout,
    Pill,
    Spinner,
    button,
    clear_layout,
    error_html,
    human_count,
    icon_button,
    label,
    open_external_url,
    time_ago,
)
from noelle.app.ui.widgets.analysis_panel import AnalysisPanel, AnalysisSummary, RemoteCodeNotice
from noelle.app.ui.widgets.download_panel import DownloadPanel
from noelle.runtime import advisor, tasks
from noelle.runtime.preflight import (
    PRECISIONS,
    context_limit,
    format_bytes,
    format_params,
)
from noelle.runtime.protocol import Command

FAMILY_LABEL = {tasks.TEXT: "Text playground", tasks.VISION: "Vision playground", tasks.DETECTION: "Detection",
                tasks.CLASSIFICATION: "Classification", tasks.ASR: "Speech recognition", tasks.TTS: "Speech synthesis",
                tasks.DIFFUSION: "Image generation", tasks.EMBEDDING: "Embeddings"}
# precision choices for the original checkpoint, per family
FAMILY_PRECISIONS = {
    tasks.TEXT: ["auto", "fp16", "bf16", "fp32", "int8", "nf4"],
    tasks.VISION: ["auto", "fp16", "bf16", "fp32", "int8", "nf4"],
    tasks.DIFFUSION: ["auto", "fp16", "fp32"],
}


def _section(title: str) -> tuple[QWidget, QVBoxLayout]:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(label(title.upper(), "eyebrow"))
    return host, layout


class ModelDetailPage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.summary: ModelSummary | None = None
        self.config: dict[str, Any] = {}
        self.variants: list[ModelRecord | None] = []
        self._load_handle: Any = None
        self._playground: Any = None
        self._generation = 0
        self.tokenizer_config: dict[str, Any] | None = None
        self.assessment: advisor.Assessment | None = None
        self._remote_once = False  # "Allow once": trusted for this session only, never persisted
        self._profile = str(ctx.settings.get("runtime.profile") or "balanced")

        self.stack = QStackedWidget()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.stack)

        # ------------------------------------------------ empty state
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        self.empty_state = EmptyState("cpu", "No model selected",
                                      "Pick a model on Discover, or open one you already downloaded.", "Browse models")
        self.empty_state.action_clicked.connect(lambda: self.ctx.navigate("discover"))
        empty_layout.addWidget(self.empty_state)
        self.recent_host = QWidget()
        self.recent = FlowLayout(self.recent_host, spacing=8)
        recent_wrap = QHBoxLayout()
        recent_wrap.addStretch(1)
        recent_wrap.addWidget(self.recent_host, 2)
        recent_wrap.addStretch(1)
        empty_layout.addLayout(recent_wrap)
        empty_layout.addStretch(1)
        self.stack.addWidget(empty)

        # ------------------------------------------------ content
        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(372)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QFrame()
        left.setObjectName("Sidebar")
        self.left_layout = QVBoxLayout(left)
        self.left_layout.setContentsMargins(18, 18, 18, 18)
        self.left_layout.setSpacing(16)
        left_scroll.setWidget(left)
        row.addWidget(left_scroll)

        # identity
        ident = QHBoxLayout()
        ident.setSpacing(12)
        self.avatar = Avatar("", 46, radius=12)
        ident.addWidget(self.avatar)
        names = QVBoxLayout()
        names.setSpacing(1)
        self.name = label("", "h2", wrap=True, selectable=True)
        self.author = label("", "muted")
        names.addWidget(self.name)
        names.addWidget(self.author)
        ident.addLayout(names, 1)
        self.hub_btn = icon_button("external", "Open on huggingface.co")
        self.hub_btn.clicked.connect(self._open_hub)
        ident.addWidget(self.hub_btn, 0, Qt.AlignmentFlag.AlignTop)
        self.left_layout.addLayout(ident)
        self.pills_host = QWidget()
        self.pills = FlowLayout(self.pills_host, spacing=6)
        self.left_layout.addWidget(self.pills_host)
        self.remote_notice = RemoteCodeNotice()
        self.remote_notice.allow_once.connect(self._allow_once)
        self.remote_notice.trust_repository.connect(lambda: self._trust("repo"))
        self.remote_notice.trust_publisher.connect(lambda: self._trust("publisher"))
        self.remote_notice.revoke.connect(self._revoke_trust)
        self.remote_notice.review.connect(self._review_remote_code)
        self.left_layout.addWidget(self.remote_notice)

        # overview
        overview, overview_layout = _section("Overview")
        self.facts = QFormLayout()
        self.facts.setHorizontalSpacing(14)
        self.facts.setVerticalSpacing(6)
        overview_layout.addLayout(self.facts)
        self.left_layout.addWidget(overview)

        # hardware fit
        fit, fit_layout = _section("Hardware fit")
        self.analysis_summary = AnalysisSummary()
        self.analysis_summary.details_requested.connect(lambda: self.tabs.setCurrentWidget(self.analysis_tab))
        fit_layout.addWidget(self.analysis_summary)
        self.left_layout.addWidget(fit)

        # download
        download, download_layout = _section("Download")
        self.runnability_notice = Banner("warning", "")
        self.runnability_notice.hide()
        download_layout.addWidget(self.runnability_notice)
        self.download_panel = DownloadPanel(ctx)
        self.download_panel.downloaded.connect(self._on_downloaded)
        download_layout.addWidget(self.download_panel)
        self.left_layout.addWidget(download)

        # runtime
        runtime, runtime_layout = _section("Runtime")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.profile = QComboBox()
        for key, text in advisor.PROFILES.items():
            self.profile.addItem(text, key)
            self.profile.setItemData(self.profile.count() - 1, advisor.PROFILE_HINTS[key],
                                     Qt.ItemDataRole.ToolTipRole)
        self.profile.setCurrentIndex(max(self.profile.findData(self._profile), 0))
        self.profile.currentIndexChanged.connect(lambda _i: self._set_profile(self.profile.currentData()))
        form.addRow(label("Profile", "muted"), self.profile)
        self.variant = QComboBox()
        self.variant.currentIndexChanged.connect(self._on_variant)
        form.addRow(label("Build", "muted"), self.variant)
        self.precision = QComboBox()
        form.addRow(label("Precision", "muted"), self.precision)
        self.device = QComboBox()
        form.addRow(label("Device", "muted"), self.device)
        runtime_layout.addLayout(form)
        load_row = QHBoxLayout()
        self.load_btn = button("Load model", "zap", "primary")
        self.load_btn.clicked.connect(self._load_or_unload)
        load_row.addWidget(self.load_btn)
        self.force_btn = button("Load anyway", None, "ghost")
        self.force_btn.setToolTip("Skip the memory pre-flight check")
        self.force_btn.clicked.connect(lambda: self._load(force=True))
        self.force_btn.hide()
        load_row.addWidget(self.force_btn)
        self.load_spinner = Spinner(16)
        load_row.addWidget(self.load_spinner)
        load_row.addStretch(1)
        runtime_layout.addLayout(load_row)
        self.load_status = label("", "faint", wrap=True)
        runtime_layout.addWidget(self.load_status)
        self.load_error = ErrorLine()
        runtime_layout.addWidget(self.load_error)
        self.left_layout.addWidget(runtime)
        self.left_layout.addStretch(1)

        # ------------------------------------------------ workspace tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        tabs_host = QWidget()
        tabs_layout = QVBoxLayout(tabs_host)
        tabs_layout.setContentsMargins(12, 8, 0, 0)
        tabs_layout.addWidget(self.tabs)
        row.addWidget(tabs_host, 1)
        self.stack.addWidget(content)
        self.playground_host = QWidget()
        self.playground_layout = QVBoxLayout(self.playground_host)
        self.playground_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_tab = AnalysisPanel(self._profile)
        self.analysis_tab.profile_changed.connect(self._set_profile)
        self.analysis_tab.action_requested.connect(self._on_analysis_action)
        self.knowledge_tab = QWidget()
        self.optimize_tab = QWidget()
        self.code_tab = QWidget()
        self.notebook_tab = QWidget()
        self.docs_tab = QWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._populate_devices()
        self._budget_seen: int | None = None
        ctx.hardware.hardware_telemetry_updated.connect(self._on_hardware)
        ctx.runtime.models_changed.connect(lambda _m: self._update_load_state())
        ctx.runtime.state_changed.connect(lambda _s: self._update_load_state())
        self.stack.setCurrentIndex(0)

    # ================================================================= model
    def on_activated(self) -> None:
        if self.summary is None:
            self._populate_recent()
        elif self._playground is not None and hasattr(self._playground, "on_activated"):
            self._playground.on_activated()

    def set_model(self, model: Any) -> None:
        self._generation += 1
        if isinstance(model, ModelRecord):
            summary = summary_from_record(model)
        elif isinstance(model, str):
            record = self.ctx.registry.get_model(model)
            summary = summary_from_record(record) if record else ModelSummary(id=model)
        else:
            summary = model
        self.summary = summary
        self.config = {}
        self.tokenizer_config = None
        self.assessment = None
        self._remote_once = False
        self.stack.setCurrentIndex(1)
        self._render_identity()
        self._build_tabs()
        self._refresh_variants()
        self._render_fit()
        self.download_panel.set_model(summary)
        self._update_load_state()
        self._render_runnability()
        self._fetch_details()

    def _fetch_details(self) -> None:
        summary = self.summary
        if summary is None:
            return
        generation = self._generation
        record = self.ctx.registry.get_model(summary.id, source="hub")
        if record and record.local_path and (Path(record.local_path) / "config.json").is_file():
            try:
                self.config = json.loads((Path(record.local_path) / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.config = {}
            self._render_facts()
            self._render_fit()
            self._render_runnability()

        local_dir = Path(record.local_path) if record and record.local_path else None

        def fetch() -> tuple[ModelSummary | None, dict[str, Any], dict[str, Any] | None]:
            info = None
            try:
                info = self.ctx.hf.model_info(summary.id)
            except Exception:
                info = None
            config = self.config or self.ctx.hf.fetch_config(summary.id)
            return info, config, _tokenizer_config(self.ctx, summary, local_dir)

        def done(result: tuple[ModelSummary | None, dict[str, Any], dict[str, Any] | None]) -> None:
            if generation != self._generation:
                return
            info, config, tokenizer_config = result
            self.tokenizer_config = tokenizer_config
            if info is not None:
                # keep list-level fields the detail call may omit
                info.trending_score = info.trending_score or summary.trending_score
                self.summary = info
                self.download_panel.set_model(info)
            if config:
                self.config = config
            self._render_identity()
            self._render_fit()
            self._render_runnability()
            self._apply_profile()
            self._refresh_docs()

        runner().submit(fetch, on_result=done)

    # =============================================================== render
    def _render_identity(self) -> None:
        s = self.summary
        if s is None:
            return
        self.avatar.set_name(s.namespace)
        self.name.setText(s.name)
        self.author.setText(s.namespace)
        self.pills.clear()
        if s.pipeline_tag:
            self.pills.addWidget(Pill(s.pipeline_tag, "info"))
        if s.param_count:
            self.pills.addWidget(Pill(f"{format_params(s.param_count)} params", "accent"))
        if s.license:
            self.pills.addWidget(Pill(str(s.license)[:24], "unknown"))
        if s.gated:
            self.pills.addWidget(Pill("gated", "yellow"))
        if s.prequantized:
            self.pills.addWidget(Pill("pre-quantized", "unknown"))
        if s.is_gguf:
            self.pills.addWidget(Pill("GGUF", "unknown"))
        if s.custom_code:
            self.pills.addWidget(Pill("custom code", "yellow"))
        self._render_remote_code()
        self._render_facts()

    def _render_facts(self) -> None:
        s = self.summary
        if s is None:
            return
        while self.facts.rowCount():
            self.facts.removeRow(0)
        ctx_len = context_limit(self.config) or s.context_length
        arch = None
        if self.config.get("architectures"):
            arch = self.config["architectures"][0]
        arch = arch or s.architecture
        rows = [
            ("Task", s.pipeline_tag or "—"),
            ("Architecture", arch or "—"),
            ("Parameters", format_params(s.param_count) if s.param_count else "—"),
            ("Context window", f"{ctx_len:,} tokens" if ctx_len else "—"),
            ("License", str(s.license or "—")),
            ("Library", s.library_name or "—"),
            ("Updated", time_ago(s.last_modified) if s.last_modified else "—"),
            ("Popularity", f"{human_count(s.downloads)} downloads · {human_count(s.likes)} likes"),
        ]
        record = self.ctx.registry.get_model(s.id, source="hub")
        if record is not None and record.exists_on_disk():  # provenance of the copy on this machine
            commit = record.commit_hash or record.revision
            rows += [
                ("Revision", commit[:12] if commit and len(commit) >= 12 else (commit or "—")),
                ("Downloaded", time_ago(record.downloaded_at) if record.downloaded_at else "—"),
                ("Last used", time_ago(record.last_used_at) if record.last_used_at else "Not loaded yet"),
                # zero-width spaces after separators let long paths wrap inside the sidebar
                ("Stored at", (record.local_path or "—").replace("/", "/\u200b").replace("\\", "\\\u200b")),
            ]
        for key, value in rows:
            value_label = label(str(value), selectable=True)
            value_label.setWordWrap(True)
            self.facts.addRow(label(key, "muted"), value_label)

    def _render_fit(self) -> None:
        if self.summary is None:
            return
        self._assess()

    def _render_runnability(self) -> None:
        """Exact verdict beside the Download button, using the worker's runtime facts."""
        if self.summary is None:
            return
        verdict = self.ctx.runnability(self.summary, config=self.config or None)
        self._runnability = verdict
        self.runnability_notice.clear_actions()
        if verdict.supported:
            self.runnability_notice.hide()
            return
        body = f"<b>This model can't run in Noelle yet.</b> {html.escape(verdict.reason or verdict.label)}"
        if verdict.hint:
            body += f"<br><span style='color:{theme.TEXT_FAINT}'>{html.escape(verdict.hint)}</span>"
        self.runnability_notice.set_text(body)
        if verdict.verdict == "GATED_NO_ACCESS":
            self.runnability_notice.add_action("Open model page", self._open_hub, "primary")
        elif verdict.remediation and verdict.remediation.get("action") in ("install_component", "update_runtime"):
            self.runnability_notice.add_action("Fix runtime environment…",
                                               lambda: self._fix_runtime_environment(verdict), "primary")
        self.runnability_notice.show()

    # ================================================================ tabs
    def _build_tabs(self) -> None:
        s = self.summary
        assert s is not None
        if self._playground is not None and hasattr(self._playground, "shutdown"):
            self._playground.shutdown()
        self.tabs.blockSignals(True)
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if widget not in (self.playground_host, self.analysis_tab):  # release web views, kernels, timers
                if hasattr(widget, "shutdown"):
                    widget.shutdown()
                widget.deleteLater()
        clear_layout(self.playground_layout)
        self._playground = self._make_playground(s)
        self.playground_layout.addWidget(self._playground)
        family = s.family
        self.tabs.addTab(self.playground_host, icons.icon("play", theme.TEXT_MUTED, 14),
                         FAMILY_LABEL.get(family or "", "Playground"))
        self.tabs.addTab(self.analysis_tab, icons.icon("gauge", theme.TEXT_MUTED, 14), "Analysis")
        if family == tasks.TEXT:
            from noelle.app.ui.widgets.knowledge_panel import KnowledgePanel

            self.knowledge_tab = KnowledgePanel(self.ctx)
            self.knowledge_tab.collections_changed.connect(
                lambda: getattr(self._playground, "refresh_collections", lambda: None)())
            self.tabs.addTab(self.knowledge_tab, icons.icon("book", theme.TEXT_MUTED, 14), "Documents")
        from noelle.app.ui.widgets.optimize_panel import OptimizePanel

        self.optimize_tab = OptimizePanel(self.ctx, self)
        self.tabs.addTab(self.optimize_tab, icons.icon("layers", theme.TEXT_MUTED, 14), "Optimize")
        from noelle.app.ui.widgets.code_view import CodeExportPanel

        self.code_tab = CodeExportPanel(self.ctx, self)
        self.tabs.addTab(self.code_tab, icons.icon("code", theme.TEXT_MUTED, 14), "Code")
        from noelle.app.ui.widgets.jupyter_bridge import NotebookPanel

        self.notebook_tab = NotebookPanel(self.ctx, self)
        self.tabs.addTab(self.notebook_tab, icons.icon("terminal", theme.TEXT_MUTED, 14), "Notebook")
        from noelle.app.ui.widgets.docs_viewer import DocsViewer

        self.docs_tab = DocsViewer(self.ctx)
        self.tabs.addTab(self.docs_tab, icons.icon("file", theme.TEXT_MUTED, 14), "Model card")
        self.tabs.setCurrentIndex(0)
        self.tabs.blockSignals(False)

    def _make_playground(self, s: ModelSummary) -> QWidget:
        family = s.family
        payload = self.worker_payload
        name = lambda: self.summary.name if self.summary else "Model"  # noqa: E731
        if family == tasks.TEXT:
            from noelle.app.ui.widgets.playgrounds.text_playground import TextPlayground

            return TextPlayground(self.ctx, payload, name)
        if family in (tasks.VISION, tasks.DETECTION, tasks.CLASSIFICATION):
            from noelle.app.ui.widgets.playgrounds.vision_playground import VisionPlayground

            return VisionPlayground(self.ctx, payload, family)
        if family in (tasks.ASR, tasks.TTS):
            from noelle.app.ui.widgets.playgrounds.audio_playground import AudioPlayground

            return AudioPlayground(self.ctx, payload, family)
        if family == tasks.DIFFUSION:
            from noelle.app.ui.widgets.playgrounds.diffusion_playground import DiffusionPlayground

            return DiffusionPlayground(self.ctx, payload)
        if family == tasks.EMBEDDING:
            from noelle.app.ui.widgets.playgrounds.embedding_playground import EmbeddingPlayground

            return EmbeddingPlayground(self.ctx, payload)
        return EmptyState("info", "No playground for this task yet",
                          f"Noelle can download '{s.pipeline_tag or 'unknown'}' models, but has no interactive "
                          "playground for this task.")

    def _on_tab_changed(self, _index: int) -> None:
        widget = self.tabs.currentWidget()
        if widget is self.playground_host and self._playground is not None and hasattr(self._playground, "on_activated"):
            self._playground.on_activated()
        elif hasattr(widget, "on_activated"):
            widget.on_activated()

    def _refresh_docs(self) -> None:
        if hasattr(self.docs_tab, "set_model"):
            self.docs_tab.set_model(self.summary)

    # ============================================================= runtime
    def _populate_devices(self) -> None:
        self.device.clear()
        self.device.addItem("Auto", "auto")
        kind = self.ctx.hardware.gpu.kind
        if kind == "cuda":
            self.device.addItem("NVIDIA CUDA", "cuda")
        if kind == "mps":
            self.device.addItem("Apple Metal (MPS)", "mps")
        self.device.addItem("CPU", "cpu")

    def _refresh_variants(self) -> None:
        s = self.summary
        assert s is not None
        self.variant.blockSignals(True)
        self.variant.clear()
        self.variants = []
        original = self.ctx.registry.get_model(s.id, source="hub")
        self.variant.addItem("Original checkpoint", None)
        self.variants.append(original)
        for record in self.ctx.registry.list_models():
            if record.base_model == s.id and record.source == "quantized" and record.exists_on_disk():
                spec = PRECISIONS.get(record.precision)
                self.variant.addItem(f"{spec.label if spec else record.precision} · local", record.repo_id)
                self.variants.append(record)
        self.variant.blockSignals(False)
        self.variant.setCurrentIndex(0)
        self._on_variant(0)

    def _on_variant(self, index: int) -> None:
        self.precision.clear()
        record = self.variants[index] if 0 <= index < len(self.variants) else None
        if index > 0 and record is not None:
            spec = PRECISIONS.get(record.precision)
            self.precision.addItem(spec.label if spec else record.precision, record.precision)
            self.precision.setEnabled(False)
            return
        self.precision.setEnabled(True)
        family = self.summary.family if self.summary else None
        gpu = self.ctx.hardware.gpu.kind
        if record is not None and record.gguf_file:
            self.precision.addItem(f"GGUF ({record.gguf_file})", record.precision)
            return
        for key in FAMILY_PRECISIONS.get(family or "", ["auto"]):
            spec = PRECISIONS[key]
            self.precision.addItem(spec.label, key)
            if spec.requires == "cuda" and gpu != "cuda":
                item = self.precision.model().item(self.precision.count() - 1)
                item.setEnabled(False)
                item.setToolTip("Requires an NVIDIA CUDA GPU")

    def selected_record(self) -> ModelRecord | None:
        index = self.variant.currentIndex()
        return self.variants[index] if 0 <= index < len(self.variants) else None

    def worker_payload(self) -> dict[str, Any]:
        s = self.summary
        assert s is not None
        record = self.selected_record()
        if record is not None:
            payload = record.worker_payload(self.ctx.settings)
            payload["task"] = s.pipeline_tag or payload.get("task")
        else:
            payload = {"model_id": s.id, "task": s.pipeline_tag or "text-generation",
                       "library_name": s.library_name, "gated": bool(s.gated)}
        if self.variant.currentIndex() == 0:
            payload["precision"] = self.precision.currentData() or "auto"
        payload["device"] = self.device.currentData() or "auto"
        payload["trust_remote_code"] = self._remote_once or self.ctx.is_trusted(s.id)
        return payload

    def _loaded_entry(self) -> dict[str, Any] | None:
        if self.summary is None:
            return None
        record = self.selected_record()
        model_id = record.repo_id if (record is not None and self.variant.currentIndex() > 0) else self.summary.id
        for entry in self.ctx.runtime.loaded_models:
            if entry.get("model_id") == model_id:
                return entry
        return None

    def _update_load_state(self) -> None:
        if self.summary is None:
            return
        downloaded = self.ctx.registry.get_model(self.summary.id) is not None or self.variant.currentIndex() > 0
        entry = self._loaded_entry()
        ready = self.ctx.runtime.is_ready
        if self._load_handle is not None:
            return
        self.load_spinner.stop()
        if entry is not None:
            self.load_btn.setText("Unload")
            theme.set_prop(self.load_btn, "variant", "")
            self.load_status.setText(f"Loaded on {entry['device']} · {format_bytes(entry['memory_bytes'])} · "
                                     f"{entry['engine']} · {entry['load_ms'] / 1000:.1f} s load")
        else:
            self.load_btn.setText("Load model")
            theme.set_prop(self.load_btn, "variant", "primary")
            if not ready:
                self.load_status.setText("Waiting for the model runtime…" if self.ctx.runtime.state != "crashed"
                                         else "The runtime crashed — restart it from the status bar.")
            elif not downloaded:
                self.load_status.setText("Download the model to run it locally.")
            else:
                self.load_status.setText("Not loaded. It also loads automatically on first use.")
        self.load_btn.setEnabled(ready and downloaded)

    def _load_or_unload(self) -> None:
        entry = self._loaded_entry()
        if entry is not None:
            handle = self.ctx.runtime.request(Command.UNLOAD_MODEL, {"key": entry["key"]})
            handle.finished.connect(lambda _p: self._update_load_state())
            return
        self._load(force=False)

    def _load(self, force: bool) -> None:
        if self.summary is None:
            return
        self.force_btn.hide()
        self.load_error.clear()
        payload = {**self.worker_payload(), "force": force}
        self.load_spinner.start()
        self.load_btn.setEnabled(False)
        self.load_status.setText("Loading…")
        handle = self.ctx.runtime.request(Command.LOAD_MODEL, payload)
        handle.status.connect(lambda p: self.load_status.setText(p.get("message", "")))
        handle.finished.connect(self._on_loaded)
        handle.failed.connect(self._on_load_failed)
        self._load_handle = handle

    def _on_loaded(self, payload: dict[str, Any]) -> None:
        self._load_handle = None
        self._update_load_state()
        model = payload.get("model") or {}
        self.ctx.toast(f"{self.summary.name if self.summary else 'Model'} loaded on {model.get('device', '?')}", "success")

    def _on_load_failed(self, payload: dict[str, Any]) -> None:
        self._load_handle = None
        self._update_load_state()
        message = payload.get("message") or "Loading failed."
        hint = payload.get("hint")
        self.load_status.setText("Load failed — the full reason is below.")
        self.load_error.set_error(str(message), str(hint) if hint else None)
        if payload.get("code") == "OOM":
            self.force_btn.show()
        if payload.get("code") == "REMOTE_CODE_BLOCKED":
            self._render_remote_code(force=True)
        if payload.get("component") == "llama.cpp":
            from noelle.app.ui.component_dialog import ensure_llama_cpp

            ensure_llama_cpp(self, lambda: self._load(force=False), reason=payload.get("message"))
        if payload.get("code") in ("NEEDS_NEWER_RUNTIME", "DEPENDENCY_MISSING"):
            verdict = getattr(self, "_runnability", None)
            if verdict is not None and verdict.remediation:
                self.runnability_notice.clear_actions()
                self.runnability_notice.add_action("Fix runtime environment…",
                                                   lambda: self._fix_runtime_environment(verdict), "primary")
                self.runnability_notice.show()

    def _fix_runtime_environment(self, verdict: Any) -> None:
        """Agent Runtime remediation: build an isolated env with the packages the model needs.

        ``NEEDS_NEWER_RUNTIME`` gets a fresh base environment with the latest transformers;
        ``NEEDS_COMPONENT`` gets the profile that ships the missing engine.
        """
        from noelle.runtime import environment_manager
        from noelle.runtime.agent.environment_planner import plan_for_verdict

        plan = plan_for_verdict(verdict)
        if plan is None or not plan.actionable:
            return
        detail = ", ".join(plan.packages) if plan.packages else ", ".join(plan.profiles)
        answer = QMessageBox.question(
            self, "Create a compatible runtime environment?",
            f"Noelle can build an isolated environment with:\n\n{detail}\n\n"
            f"Profiles: {', '.join(plan.profiles)}\n\nThis downloads packages once and leaves the "
            "application's own Python untouched. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        existing = [env["name"] for env in environment_manager.list_environments()]
        name = plan.suggested_name(existing)
        self.runnability_notice.set_text("<b>Building the runtime environment…</b> This can take several minutes.")

        def work(progress: Any) -> dict[str, Any]:
            return environment_manager.create_environment(name, plan.profiles, progress=progress)

        def progress(line: Any) -> None:
            self.runnability_notice.set_text(f"<b>Building {html.escape(name)}…</b> {html.escape(str(line))}")

        def done(info: dict[str, Any]) -> None:
            self.ctx.settings.set("runtime.environment", info.get("name") or name)
            self.ctx.toast(f"Runtime environment '{name}' is ready — restarting the model runtime.", "success")
            self.load_status.setText("Runtime environment updated. Press Load model again once it is ready.")
            self.ctx.runtime.restart()

        runner().submit(work, on_progress=progress, on_result=done,
                        on_error=lambda message: self.runnability_notice.set_text(
                            error_html(f"Could not build the environment: {message}")))

    # ======================================================== remote code
    def _render_remote_code(self, force: bool = False) -> None:
        s = self.summary
        if s is None:
            return
        from noelle.runtime import capabilities

        cap = capabilities.analyze(s, self.config or None)
        if not (cap.remote_code.required or s.custom_code or force):
            self.remote_notice.hide()
            return
        repos = self.ctx.settings.get("runtime.trusted_remote_code") or []
        if s.id in repos:
            state = "repo"
        elif self.ctx.is_trusted(s.id):
            state = "publisher"
        elif self._remote_once:
            state = "once"
        else:
            state = "blocked"
        self.remote_notice.set_state(cap.remote_code.files, cap.remote_code.auto_map, s.namespace, state)

    def _allow_once(self) -> None:
        self._remote_once = True
        self._after_trust_change()

    def _trust(self, scope: str) -> None:
        s = self.summary
        if s is None:
            return
        key, value = ("runtime.trusted_remote_code", s.id) if scope == "repo" else (
            "runtime.trusted_publishers", s.namespace)
        if not value:
            return
        trusted = list(self.ctx.settings.get(key) or [])
        if value not in trusted:
            trusted.append(value)
        self.ctx.settings.set(key, trusted)
        self._after_trust_change()

    def _revoke_trust(self) -> None:
        s = self.summary
        if s is None:
            return
        self._remote_once = False
        repos = [r for r in (self.ctx.settings.get("runtime.trusted_remote_code") or []) if r != s.id]
        self.ctx.settings.set("runtime.trusted_remote_code", repos)
        publishers = list(self.ctx.settings.get("runtime.trusted_publishers") or [])
        if s.namespace in publishers and QMessageBox.question(
                self, "Stop trusting this publisher?",
                f"You trust every repository from {s.namespace}. Remove that trust as well?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            publishers.remove(s.namespace)
            self.ctx.settings.set("runtime.trusted_publishers", publishers)
        self._after_trust_change()

    def _review_remote_code(self) -> None:
        if self.summary is not None:
            open_external_url(f"https://huggingface.co/{self.summary.id}/tree/main")

    def _after_trust_change(self) -> None:
        self._render_remote_code()
        self._render_fit()

    # ========================================================== analysis
    def _assess(self) -> None:
        s = self.summary
        if s is None:
            return
        try:
            self.assessment = self.ctx.assess(s, config=self.config or None, profile=self._profile,
                                              tokenizer_config=self.tokenizer_config,
                                              session_trusted=self._remote_once)
        except Exception:  # analysis must never break the workspace
            import logging

            logging.getLogger("noelle.ui").exception("analysis failed for %s", s.id)
            self.assessment = None
        self.analysis_summary.set_assessment(self.assessment)
        self.analysis_tab.set_assessment(self.assessment)

    def _set_profile(self, profile: str | None) -> None:
        profile = profile if profile in advisor.PROFILES else "balanced"
        if profile == self._profile and self.assessment is not None and self.assessment.profile == profile:
            return
        self._profile = profile
        self.ctx.settings.set("runtime.profile", profile)
        self.profile.blockSignals(True)
        self.profile.setCurrentIndex(max(self.profile.findData(profile), 0))
        self.profile.blockSignals(False)
        self.analysis_tab.set_profile(profile)
        self._render_fit()
        self._apply_profile()

    def _apply_profile(self) -> None:
        """Point the Build/Precision selectors at the profile's recommendation when it can load as-is."""
        a = self.assessment
        option = a.recommended if a is not None else None
        params = advisor.profile_params(self._profile)
        panel = getattr(self._playground, "params", None)
        if params and panel is not None and hasattr(panel, "set_values"):
            panel.set_values(params)
        if option is None:
            return
        if option.local_variant:
            self._select_variant(option.local_variant)
        elif option.status == advisor.READY and self.variant.currentIndex() == 0:
            index = self.precision.findData(_load_precision(option))
            if index >= 0 and self.precision.model().item(index).isEnabled():
                self.precision.setCurrentIndex(index)
        if option.status != advisor.READY:
            self.load_status.setText(f"{advisor.PROFILES[self._profile]} profile: {option.label} — "
                                     f"{option.reason or 'see Analysis'}")

    def _select_variant(self, repo_id: str) -> bool:
        index = self.variant.findData(repo_id)
        if index >= 0:
            self.variant.setCurrentIndex(index)
        return index >= 0

    def _on_analysis_action(self, action: str, option: advisor.RunOption) -> None:
        if action == "use" and option.local_variant:
            self._select_variant(option.local_variant)
            self.tabs.setCurrentIndex(0)
        elif action == "load":
            self.variant.setCurrentIndex(0)
            index = self.precision.findData(_load_precision(option))
            if index >= 0:
                self.precision.setCurrentIndex(index)
            self._load(force=False)
        elif action == "quantize":
            self.tabs.setCurrentWidget(self.optimize_tab)
            self.ctx.toast(f"Choose “{option.label}” and press Quantize — it builds a local copy once.", "info")
        elif action == "install":
            if option.runtime == "llama.cpp":
                from noelle.app.ui.component_dialog import ensure_llama_cpp

                ensure_llama_cpp(self, self._render_fit, reason=option.reason)
            else:
                QMessageBox.information(self, "Optional component", option.reason or "Install the missing package "
                                        "in the runtime environment, then restart the runtime.")

    def _on_downloaded(self) -> None:
        self._refresh_variants()
        self._update_load_state()
        self._fetch_details()

    def refresh_variants(self) -> None:
        if self.summary is not None:
            self._refresh_variants()

    def _on_hardware(self, _metrics: Any) -> None:
        usable = self.ctx.hardware.memory_budget().usable_bytes
        if self._budget_seen is None or abs(usable - self._budget_seen) > 0.01 * max(usable, 1):
            self._budget_seen = usable
            if self.summary is not None:
                self._render_fit()

    # ================================================================ misc
    def _open_hub(self) -> None:
        if self.summary is not None:
            QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{self.summary.id}"))

    def _populate_recent(self) -> None:
        self.recent.clear()
        for record in self.ctx.registry.list_models(only_existing=True)[:12]:
            chip = button(record.display_name, "box", "chip")
            chip.clicked.connect(lambda _=False, r=record: self.set_model(r))
            self.recent.addWidget(chip)

    def current_export_config(self) -> dict[str, Any]:
        """Settings the code exporter and notebook reproduce."""
        data: dict[str, Any] = {"payload": self.worker_payload() if self.summary else {}}
        if self._playground is not None and hasattr(self._playground, "export_config"):
            data.update(self._playground.export_config())
        return data

    def shutdown(self) -> None:
        if self._playground is not None and hasattr(self._playground, "shutdown"):
            self._playground.shutdown()
        for tab in (self.notebook_tab,):
            if hasattr(tab, "shutdown"):
                tab.shutdown()


def _tokenizer_config(ctx: AppContext, summary: ModelSummary, local_dir: Path | None) -> dict[str, Any] | None:
    """tokenizer_config.json (local copy first) — only used to detect chat templates and tool calling."""
    if summary.family not in (tasks.TEXT, tasks.VISION):
        return None
    text = None
    if local_dir is not None and (local_dir / "tokenizer_config.json").is_file():
        try:
            text = (local_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        except OSError:
            text = None
    if text is None:
        text = ctx.hf.fetch_file_text(summary.id, "tokenizer_config.json", max_bytes=1_000_000)
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _load_precision(option: advisor.RunOption) -> str:
    """Full-precision Transformers options load as "auto" (the device's best float: BF16 on recent CUDA GPUs)."""
    if option.quality == advisor.REFERENCE and option.runtime == "transformers":
        return "auto"
    return option.precision
