"""Optimize tab: one-click quantization studio and LoRA adapter hot-swapping."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from noelle.app.context import AppContext
from noelle.app.services.model_registry import ModelRecord
from noelle.app.ui.widgets.common import Banner, Card, Pill, Spinner, button, clear_layout, label, widget_alive
from noelle.runtime import tasks
from noelle.runtime.hub_cache import is_downloaded
from noelle.runtime.preflight import format_bytes
from noelle.runtime.protocol import Command

ORDER = ["mlx-4bit", "gguf-q4_k_m", "awq-4bit", "gptq-4bit", "nf4"]


class OptimizePanel(QScrollArea):
    def __init__(self, ctx: AppContext, page: Any) -> None:
        super().__init__()
        self.ctx = ctx
        self.page = page
        self._quant: Any = None
        self._quant_method: str | None = None
        self._lora_handle: Any = None
        self.setWidgetResizable(True)
        host = QWidget()
        self.setWidget(host)
        root = QVBoxLayout(host)
        root.setContentsMargins(18, 14, 18, 18)
        root.setSpacing(14)

        root.addWidget(label("Quantization studio", "h2"))
        root.addWidget(label("Convert the original weights into a compact 4-bit build stored under ~/.noelle/models. "
                             "Methods your hardware cannot run are shown with the reason.", "muted", wrap=True))
        self.notice = Banner("warning", "")
        self.notice.hide()
        root.addWidget(self.notice)
        self.methods = QVBoxLayout()
        self.methods.setSpacing(8)
        root.addLayout(self.methods)
        self.methods_status = label("Checking which backends this machine supports…", "faint")
        root.addWidget(self.methods_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        root.addWidget(self.progress)
        self.progress_text = label("", "faint", wrap=True)
        root.addWidget(self.progress_text)
        self.result_row = QHBoxLayout()
        root.addLayout(self.result_row)

        self.lora_host = QWidget()
        lora = QVBoxLayout(self.lora_host)
        lora.setContentsMargins(0, 12, 0, 0)
        lora.setSpacing(10)
        lora.addWidget(label("LoRA adapters", "h2"))
        lora.addWidget(label("Attach low-rank adapters to the resident base model and switch between them without "
                             "reloading the base weights.", "muted", wrap=True))
        add = QHBoxLayout()
        self.adapter_input = QLineEdit()
        self.adapter_input.setPlaceholderText("Adapter repo id, e.g. org/my-lora")
        add.addWidget(self.adapter_input, 1)
        self.adapter_dl = button("Download adapter", "download")
        self.adapter_dl.clicked.connect(self._download_adapter)
        add.addWidget(self.adapter_dl)
        lora.addLayout(add)
        self.adapters = QListWidget()
        self.adapters.setMaximumHeight(140)
        lora.addWidget(self.adapters)
        attach_row = QHBoxLayout()
        self.attach_btn = button("Attach selected", "plug", "primary")
        self.attach_btn.clicked.connect(self._attach)
        attach_row.addWidget(self.attach_btn)
        attach_row.addWidget(label("Active", "muted"))
        self.active = QComboBox()
        self.active.setMinimumWidth(200)
        self.active.activated.connect(lambda _i: self._switch())
        attach_row.addWidget(self.active)
        self.lora_spinner = Spinner(16)
        attach_row.addWidget(self.lora_spinner)
        attach_row.addStretch(1)
        lora.addLayout(attach_row)
        self.lora_status = label("", "faint", wrap=True)
        lora.addWidget(self.lora_status)
        root.addWidget(self.lora_host)
        root.addStretch(1)
        # A bound method of this QObject: PySide disconnects it automatically when the panel
        # (rebuilt per model by ModelDetailPage._build_tabs) is deleted. A lambda would keep
        # firing into the dead widget once per previously opened model.
        ctx.downloads.job_finished.connect(self._on_download_finished)

    # ------------------------------------------------------------ activation
    def on_activated(self) -> None:
        summary = self.page.summary
        if summary is None:
            return
        compat = self.ctx.compat(summary)
        if compat.level in ("yellow", "red") and compat.fp16_bytes:
            budget = self.ctx.hardware.memory_budget()
            self.notice.set_text(f"<b>Hardware notice:</b> the selected model requires ~{format_bytes(compat.fp16_bytes)}"
                                 f" at FP16, exceeding your accelerator budget ({format_bytes(budget.usable_bytes)}). "
                                 f"A 4-bit build needs about {format_bytes(compat.int4_bytes)}.")
            self.notice.show()
        else:
            self.notice.hide()
        handle = self.ctx.runtime.request(Command.QUANTIZE_CAPABILITIES)
        handle.finished.connect(self._render_methods)
        handle.failed.connect(self._on_methods_failed)
        self.lora_host.setVisible(summary.family == tasks.TEXT)
        self._refresh_adapters()

    def _render_methods(self, payload: dict[str, Any]) -> None:
        clear_layout(self.methods)
        summary = self.page.summary
        downloaded = summary is not None and self.ctx.registry.get_model(summary.id, source="hub") is not None
        methods = payload.get("methods") or {}
        is_text = summary is not None and summary.family == tasks.TEXT
        self.methods_status.setText(f"Runtime device: {payload.get('device', '?')}" +
                                    ("" if downloaded else " · download the model first"))
        for key in ORDER:
            info = methods.get(key)
            if not info:
                continue
            card = Card()
            row = QHBoxLayout(card)
            row.setContentsMargins(14, 12, 14, 12)
            text = QVBoxLayout()
            text.setSpacing(3)
            head = QHBoxLayout()
            head.addWidget(label(info.get("label", key), "h3"))
            available = bool(info.get("available")) and is_text
            reason = info.get("reason") or ("Only text-generation models are supported." if not is_text else None)
            head.addWidget(Pill("available" if available else "unavailable", "green" if available else "unknown",
                                tooltip=reason or ""))
            head.addStretch(1)
            text.addLayout(head)
            text.addWidget(label(info.get("description", ""), "muted", wrap=True))
            if not available and reason:
                text.addWidget(label(reason, "faint", wrap=True))
            row.addLayout(text, 1)
            needs_tools = key == "gguf-q4_k_m" and not available and is_text and "llama.cpp tools" in (reason or "")
            if needs_tools:
                go = button("Download tools…", "download", "primary")
                go.setToolTip("Asks before downloading anything")
                go.clicked.connect(self._get_llama_cpp)
            else:
                go = button("Quantize", "layers", "primary" if available else None)
                go.setEnabled(available and downloaded and self._quant is None)
                go.clicked.connect(lambda _=False, k=key: self._quantize(k))
            row.addWidget(go, 0, Qt.AlignmentFlag.AlignVCenter)
            self.methods.addWidget(card)

    def _get_llama_cpp(self) -> None:
        from noelle.app.ui.component_dialog import ensure_llama_cpp

        ensure_llama_cpp(self, self.on_activated,
                         reason="GGUF quantization uses llama.cpp's converter and llama-quantize.")

    # ---------------------------------------------------------- quantization
    def _quantize(self, method: str) -> None:
        summary = self.page.summary
        record = self.ctx.registry.get_model(summary.id, source="hub") if summary else None
        if summary is None or record is None:
            return
        payload = {**record.worker_payload(self.ctx.settings), "method": method}
        clear_layout(self.result_row)
        self.progress.show()
        self.progress_text.setText("Starting…")
        handle = self.ctx.runtime.request(Command.QUANTIZE, payload)
        handle.progress.connect(self._on_progress)
        handle.status.connect(self._on_quant_status)
        handle.finished.connect(self._on_quantized_result)
        handle.failed.connect(self._on_failed)
        self._quant = handle
        self._quant_method = method
        self.on_activated()

    def _on_progress(self, data: dict[str, Any]) -> None:
        if data.get("total"):
            self.progress.setRange(0, int(data["total"]))
            self.progress.setValue(int(data.get("current") or 0))
        self.progress_text.setText(f"{data.get('stage', '')}: {data.get('message', '')}".strip(": "))

    def _on_methods_failed(self, payload: dict[str, Any]) -> None:
        self.methods_status.setText(str(payload.get("message") or "Could not read quantization support."))

    def _on_quant_status(self, payload: dict[str, Any]) -> None:
        self.progress_text.setText(str(payload.get("message") or ""))

    def _on_quantized_result(self, result: dict[str, Any]) -> None:
        self._on_quantized(self._quant_method or "auto", result)
        self._quant_method = None

    def _on_lora_status(self, payload: dict[str, Any]) -> None:
        self.lora_status.setText(str(payload.get("message") or ""))

    def _on_quantized(self, method: str, result: dict[str, Any]) -> None:
        self._quant = None
        self.progress.hide()
        summary = self.page.summary
        record = ModelRecord(repo_id=result.get("derived_id") or f"local/{summary.name}-{method}",
                             pipeline_tag=summary.pipeline_tag, local_path=result.get("output_dir"),
                             precision=result.get("precision") or method, engine=result.get("engine"),
                             gguf_file=result.get("gguf_file"), source="quantized", base_model=summary.id,
                             size_bytes=result.get("size_bytes"), param_count=summary.param_count,
                             license=summary.license, architecture=summary.architecture)
        self.ctx.registry.upsert_model(record)
        self.progress_text.setText(f"Saved {format_bytes(result.get('size_bytes'))} to {result.get('output_dir')} in "
                                   f"{(result.get('elapsed_ms') or 0) / 1000:.1f} s.")
        use = button("Use this build", "zap", "primary")
        use.clicked.connect(lambda: self._use_build(record.repo_id))
        self.result_row.addWidget(use)
        reveal = button("Show folder", "folder", "ghost")
        reveal.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.get("output_dir")))))
        self.result_row.addWidget(reveal)
        self.result_row.addStretch(1)
        self.page.refresh_variants()
        self.ctx.toast("Quantized build ready", "success")
        self.on_activated()

    def _use_build(self, repo_id: str) -> None:
        combo = self.page.variant
        index = combo.findData(repo_id)
        if index >= 0:
            combo.setCurrentIndex(index)
        self.page.tabs.setCurrentIndex(0)

    def _on_failed(self, payload: dict[str, Any]) -> None:
        self._quant = None
        self.progress.hide()
        self.progress_text.setText(f"Quantization failed: {payload.get('message')}"
                                   + (f" — {payload['hint']}" if payload.get("hint") else ""))
        self.on_activated()

    # ------------------------------------------------------------------ LoRA
    def _on_download_finished(self, *_args: Any) -> None:
        if not widget_alive(self):
            return
        self._refresh_adapters()

    def shutdown(self) -> None:
        """Detach signals and cancel in-flight work before model-tab teardown."""
        try:
            self.ctx.downloads.job_finished.disconnect(self._on_download_finished)
        except (RuntimeError, TypeError):
            pass
        for handle in (self._quant, self._lora_handle):
            if handle is not None:
                try:
                    handle.cancel()
                except Exception:  # noqa: BLE001 - cancellation is best effort
                    pass
        self._quant = None
        self._lora_handle = None

    def _refresh_adapters(self) -> None:
        if not widget_alive(self):
            return
        summary = self.page.summary
        if summary is None:
            return
        self.adapters.clear()
        for record in self.ctx.registry.list_adapters(base_model_id=summary.id):
            item = QListWidgetItem(f"{record.adapter_id}  ·  {format_bytes(record.size_bytes)}")
            item.setData(Qt.ItemDataRole.UserRole, record.adapter_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, record.local_path)
            self.adapters.addItem(item)
        handle = self.ctx.runtime.request(Command.LORA_LIST, self.page.worker_payload())
        handle.finished.connect(self._set_active_list)

    def _set_active_list(self, payload: dict[str, Any]) -> None:
        self.active.blockSignals(True)
        self.active.clear()
        self.active.addItem("Base model (no adapter)", None)
        for name in payload.get("adapters") or []:
            self.active.addItem(name, name)
        index = self.active.findData(payload.get("active_adapter"))
        self.active.setCurrentIndex(max(index, 0))
        self.active.blockSignals(False)

    def _download_adapter(self) -> None:
        repo = self.adapter_input.text().strip()
        if not repo or "/" not in repo:
            self.ctx.toast("Enter an adapter repository id like org/name.", "warning")
            return
        from noelle.app.services.hf_client import ModelSummary

        base = self.page.summary.id if self.page.summary else None
        summary = ModelSummary(id=repo, library_name="peft", tags=[f"base_model:adapter:{base}"] if base else [],
                               base_models=[base] if base else [])
        self.ctx.downloads.start(repo, summary=summary)
        self.lora_status.setText(f"Downloading {repo}…")

    def _attach(self) -> None:
        item = self.adapters.currentItem()
        if item is None:
            self.ctx.toast("Select an adapter first.", "info")
            return
        adapter_id = item.data(Qt.ItemDataRole.UserRole)
        local_path = item.data(Qt.ItemDataRole.UserRole + 1)
        if not local_path and not is_downloaded(adapter_id):
            self.ctx.toast("That adapter is not downloaded.", "warning")
            return
        self.lora_spinner.start()
        payload = {**self.page.worker_payload(), "adapter_id": adapter_id, "adapter_local_path": local_path}
        handle = self.ctx.runtime.request(Command.LORA_LOAD, payload)
        handle.status.connect(self._on_lora_status)
        handle.finished.connect(self._on_attached)
        handle.failed.connect(self._on_lora_failed)
        self._lora_handle = handle

    def _on_attached(self, result: dict[str, Any]) -> None:
        self.lora_spinner.stop()
        warn = "" if result.get("base_matches", True) else (
            f" Warning: it was trained for {result.get('base_model')}.")
        self.lora_status.setText(f"Attached {result.get('adapter')} in {result.get('load_ms', 0):.0f} ms; switched in "
                                 f"{(result.get('switch_ms') or 0):.2f} ms.{warn}")
        self._set_active_list(result)
        if hasattr(self.page._playground, "set_adapter"):
            self.page._playground.set_adapter(result.get("active_adapter"))

    def _switch(self) -> None:
        name = self.active.currentData()
        self.lora_spinner.start()
        handle = self.ctx.runtime.request(Command.LORA_SET_ADAPTER, {**self.page.worker_payload(),
                                                                     "adapter_name": name})
        handle.finished.connect(self._on_switched)
        handle.failed.connect(self._on_lora_failed)
        self._lora_handle = handle

    def _on_switched(self, result: dict[str, Any]) -> None:
        self.lora_spinner.stop()
        active = result.get("active_adapter")
        self.lora_status.setText(f"Active: {active or 'base model'} — switched in {result.get('switch_ms', 0):.2f} ms "
                                 "without reloading base weights.")
        if hasattr(self.page._playground, "set_adapter"):
            self.page._playground.set_adapter(active)

    def _on_lora_failed(self, payload: dict[str, Any]) -> None:
        self.lora_spinner.stop()
        self.lora_status.setText(f"{payload.get('message')}" + (f" — {payload['hint']}" if payload.get("hint") else ""))

