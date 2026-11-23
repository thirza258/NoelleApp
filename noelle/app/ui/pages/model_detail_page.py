"""Page 2: model detail, download/runtime control and the multi-modal workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
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
    FlowLayout,
    Pill,
    Spinner,
    button,
    clear_layout,
    human_count,
    icon_button,
    label,
    time_ago,
)
from noelle.app.ui.widgets.download_panel import DownloadPanel
from noelle.runtime import tasks
from noelle.runtime.preflight import (
    PRECISIONS,
    context_limit,
    estimate_runtime_bytes,
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
        self.remote_code_banner = Banner("warning", "This repository ships custom Python code that runs when the model "
                                                    "loads. Only allow it for publishers you trust.")
        self.remote_code_banner.hide()
        self.left_layout.addWidget(self.remote_code_banner)

        # overview
        overview, overview_layout = _section("Overview")
        self.facts = QFormLayout()
        self.facts.setHorizontalSpacing(14)
        self.facts.setVerticalSpacing(6)
        overview_layout.addLayout(self.facts)
        self.left_layout.addWidget(overview)

        # hardware fit
        fit, fit_layout = _section("Hardware fit")
        self.fit_rows = QVBoxLayout()
        self.fit_rows.setSpacing(5)
        fit_layout.addLayout(self.fit_rows)
        self.fit_banner = Banner("warning", "")
        self.fit_banner.add_action("Optimize", lambda: self.tabs.setCurrentWidget(self.optimize_tab))
        self.fit_banner.hide()
        fit_layout.addWidget(self.fit_banner)
        self.left_layout.addWidget(fit)

        # download
        download, download_layout = _section("Download")
        self.download_panel = DownloadPanel(ctx)
        self.download_panel.downloaded.connect(self._on_downloaded)
        download_layout.addWidget(self.download_panel)
        self.left_layout.addWidget(download)

        # runtime
        runtime, runtime_layout = _section("Runtime")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.variant = QComboBox()
        self.variant.currentIndexChanged.connect(self._on_variant)
        form.addRow(label("Build", "muted"), self.variant)
        self.precision = QComboBox()
        form.addRow(label("Precision", "muted"), self.precision)
        self.device = QComboBox()
        form.addRow(label("Device", "muted"), self.device)
        runtime_layout.addLayout(form)
        self.allow_remote = QCheckBox("Allow remote code for this model")
        self.allow_remote.toggled.connect(self._on_trust_toggled)
        self.allow_remote.hide()
        runtime_layout.addWidget(self.allow_remote)
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
        self.stack.setCurrentIndex(1)
        self._render_identity()
        self._build_tabs()
        self._refresh_variants()
        self._render_fit()
        self.download_panel.set_model(summary)
        self._update_load_state()
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

        def fetch() -> tuple[ModelSummary | None, dict[str, Any]]:
            info = None
            try:
                info = self.ctx.hf.model_info(summary.id)
            except Exception:
                info = None
            config = self.config or self.ctx.hf.fetch_config(summary.id)
            return info, config

        def done(result: tuple[ModelSummary | None, dict[str, Any]]) -> None:
            if generation != self._generation:
                return
            info, config = result
            if info is not None:
                # keep list-level fields the detail call may omit
                info.trending_score = info.trending_score or summary.trending_score
                self.summary = info
                self.download_panel.set_model(info)
            if config:
                self.config = config
            self._render_identity()
            self._render_fit()
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
        trusted = s.id in (self.ctx.settings.get("runtime.trusted_remote_code") or [])
        self.remote_code_banner.setVisible(s.custom_code and not trusted)
        self.allow_remote.setVisible(s.custom_code)
        self.allow_remote.blockSignals(True)
        self.allow_remote.setChecked(trusted)
        self.allow_remote.blockSignals(False)
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
        for key, value in rows:
            value_label = label(str(value), selectable=True)
            value_label.setWordWrap(True)
            self.facts.addRow(label(key, "muted"), value_label)

    def _render_fit(self) -> None:
        clear_layout(self.fit_rows)
        s = self.summary
        if s is None:
            return
        budget = self.ctx.hardware.memory_budget(float(self.ctx.settings.get("runtime.safety_headroom", 0.15)))
        usable = budget.usable_bytes
        compat = self.ctx.compat(s)
        header = QHBoxLayout()
        header.addWidget(Pill(compat.label, compat.level))
        header.addStretch(1)
        header.addWidget(label(f"{format_bytes(usable)} usable", "faint"))
        self.fit_rows.addLayout(header)
        if s.param_count:
            for key, caption in (("fp16", "FP16 / BF16"), ("int8", "8-bit"), ("mlx-4bit" if budget.device == "mps"
                                                                                 else "nf4", "4-bit")):
                need = estimate_runtime_bytes(s.param_count, key, config=self.config or None) or 0
                ok = need <= usable
                row = QHBoxLayout()
                mark = label("✓" if ok else "✗")
                mark.setStyleSheet(f"color: {theme.SUCCESS if ok else theme.DANGER}; font-weight: 700;")
                mark.setFixedWidth(14)
                row.addWidget(mark)
                row.addWidget(label(caption, "muted"))
                row.addStretch(1)
                row.addWidget(label(f"~{format_bytes(need)}", "mono"))
                self.fit_rows.addLayout(row)
        detail = label(compat.detail, "faint", wrap=True)
        self.fit_rows.addWidget(detail)
        if compat.level in ("yellow", "red") and compat.fp16_bytes:
            target = "Quantize to 4-bit MLX" if budget.device == "mps" else "Quantize to 4-bit"
            self.fit_banner.set_text(
                f"<b>Hardware notice:</b> this model needs ~{format_bytes(compat.fp16_bytes)} at FP16, above your "
                f"{format_bytes(usable)} budget. Open <i>Optimize</i> and choose “{target}” to build a local "
                f"~{format_bytes(compat.int4_bytes)} checkpoint.")
            self.fit_banner.show()
        else:
            self.fit_banner.hide()

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
            if widget is not self.playground_host:  # release web views, kernels and timers of the old model
                if hasattr(widget, "shutdown"):
                    widget.shutdown()
                widget.deleteLater()
        clear_layout(self.playground_layout)
        self._playground = self._make_playground(s)
        self.playground_layout.addWidget(self._playground)
        family = s.family
        self.tabs.addTab(self.playground_host, icons.icon("play", theme.TEXT_MUTED, 14),
                         FAMILY_LABEL.get(family or "", "Playground"))
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
            payload = record.worker_payload()
            payload["task"] = s.pipeline_tag or payload.get("task")
        else:
            payload = {"model_id": s.id, "task": s.pipeline_tag or "text-generation"}
        if self.variant.currentIndex() == 0:
            payload["precision"] = self.precision.currentData() or "auto"
        payload["device"] = self.device.currentData() or "auto"
        payload["trust_remote_code"] = bool(self.allow_remote.isChecked())
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
        message = payload.get("message", "Loading failed.")
        hint = payload.get("hint")
        self.load_status.setText(f"<span style='color:{theme.DANGER}'>{message}</span>" + (f"<br>{hint}" if hint else ""))
        if payload.get("code") == "OOM":
            self.force_btn.show()
        if payload.get("code") == "REMOTE_CODE_BLOCKED":
            self.allow_remote.show()
        if payload.get("component") == "llama.cpp":
            from noelle.app.ui.component_dialog import ensure_llama_cpp

            ensure_llama_cpp(self, lambda: self._load(force=False), reason=payload.get("message"))

    def _on_trust_toggled(self, checked: bool) -> None:
        if self.summary is None:
            return
        trusted = list(self.ctx.settings.get("runtime.trusted_remote_code") or [])
        if checked and self.summary.id not in trusted:
            trusted.append(self.summary.id)
        if not checked and self.summary.id in trusted:
            trusted.remove(self.summary.id)
        self.ctx.settings.set("runtime.trusted_remote_code", trusted)
        self.remote_code_banner.setVisible(self.summary.custom_code and not checked)

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

