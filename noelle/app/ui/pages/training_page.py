"""Training Studio: local dataset preparation, hardware planning and fine-tuning runs."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import (
    ErrorLine,
    Panel,
    StatTile,
    button,
    human_bytes,
    human_duration,
    label,
    time_ago,
)
from noelle.runtime import environment_manager, tasks
from noelle.training.config import (
    ConfigError,
    DatasetConfig,
    HardwareInfo,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    TrainingParams,
    estimate_plan,
    load_recipe,
    save_recipe,
)
from noelle.training.dataset import DatasetReport, analyze_dataset, prepare_dataset

STATUS_LABELS = {
    "preparing": "Preparing", "queued": "Queued", "training": "Training", "stopping": "Stopping",
    "completed": "Completed", "stopped": "Stopped", "interrupted": "Interrupted", "failed": "Failed",
    "oom": "Out of memory",
}

MODES = [("LoRA (recommended)", "lora"), ("QLoRA (4-bit, CUDA)", "qlora"),
         ("Full fine-tuning", "full"), ("Continued pretraining", "continued_pretrain")]


def _dataset_summary(report: DatasetReport) -> str:
    warnings = "".join(f"<br>⚠ {item}" for item in report.warnings[:4])
    errors = "".join(f"<br>✕ {item}" for item in report.errors[:4])
    return (f"<b>{report.valid:,}</b> valid rows · {report.invalid:,} invalid · {report.duplicates:,} duplicates<br>"
            f"~{report.avg_tokens:.0f} tokens average · {report.max_tokens:,} max · "
            f"{report.very_long:,} over the sequence limit<br>"
            f"schema: {report.schema} · split: {report.train:,} train / {report.validation:,} validation"
            f"{warnings}{errors}")


class LossChart(QWidget):
    """Small dependency-free loss curve (no QtCharts requirement)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float]] = []
        self.setMinimumHeight(190)
        self.setObjectName("LossChart")

    def set_metrics(self, metrics: list[dict[str, Any]]) -> None:
        points: list[tuple[float, float]] = []
        for index, metric in enumerate(metrics):
            loss = metric.get("loss")
            if not isinstance(loss, (int, float)) or loss != loss:
                continue
            step = metric.get("step")
            points.append((float(step) if isinstance(step, (int, float)) else float(index), float(loss)))
        self._points = points[-3000:]
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.SURFACE_2))
        area = QRectF(self.rect()).adjusted(46, 14, -14, -26)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawRect(area)
        painter.setPen(QPen(QColor(theme.TEXT_FAINT), 1))
        for index in range(1, 4):
            y = area.top() + area.height() * index / 4
            painter.drawLine(area.left(), y, area.right(), y)
        if len(self._points) < 2:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "The loss curve appears after the first steps.")
            return
        losses = [point[1] for point in self._points]
        low, high = min(losses), max(losses)
        if high <= low:
            high = low + 1.0
        span = high - low
        low -= span * 0.08
        high += span * 0.08
        x0, x1 = self._points[0][0], self._points[-1][0]
        if x1 <= x0:
            x1 = x0 + 1
        painter.setPen(QPen(QColor(theme.ACCENT), 2))
        previous: Any = None
        for step, loss in self._points:
            x = area.left() + (step - x0) / (x1 - x0) * area.width()
            y = area.bottom() - (loss - low) / (high - low) * area.height()
            point = (x, y)
            if previous is not None:
                painter.drawLine(previous[0], previous[1], x, y)
            previous = point
        if previous is not None:
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawEllipse(QRectF(previous[0] - 3, previous[1] - 3, 6, 6))
        painter.setPen(QColor(theme.TEXT_FAINT))
        font = QFont(painter.font())
        font.setPointSizeF(max(font.pointSizeF() * 0.85, 7))
        painter.setFont(font)
        painter.drawText(QRectF(0, area.bottom() + 4, area.left() - 6, 18),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{low:.3f}")
        painter.drawText(QRectF(0, area.top() - 4, area.left() - 6, 18),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{high:.3f}")
        painter.drawText(QRectF(area.left(), area.bottom() + 4, area.width(), 18),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"step {x0:.0f}")
        painter.drawText(QRectF(area.left(), area.bottom() + 4, area.width(), 18),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"step {x1:.0f}")


def _table(headers: list[str], stretch: int | None = None) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch is not None:
        header.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    return table


class TrainingPage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.manager = ctx.training
        self._selected_run_id: str | None = None
        self._plan: Any = None
        self._dataset_report: DatasetReport | None = None
        self._studio_report: DatasetReport | None = None
        self._starting = False
        self._render_pending = False
        self._log_run_id: str | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(300)
        self._render_timer.timeout.connect(self._render_later)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Training Studio", "h2"))
        titles.addWidget(label("Prepare data, estimate the hardware budget, fine-tune locally and keep every "
                               "checkpoint. Data and models never leave this machine unless you export them.",
                               "muted", wrap=True))
        root.addLayout(titles)
        self.banner = ErrorLine()
        root.addWidget(self.banner)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_runs_tab(), "Runs")
        self.tabs.addTab(self._build_new_run_tab(), "New training run")
        self.tabs.addTab(self._build_dataset_tab(), "Dataset Studio")
        root.addWidget(self.tabs, 1)

        if self.manager is not None:
            self.manager.job_added.connect(self._on_job_added)
            self.manager.job_updated.connect(self._on_job_updated)
            self.manager.job_finished.connect(self._on_job_finished)
            self.manager.log_line.connect(self._on_log_line)
        self.refresh_runs()
        self.refresh_environments()
        self.refresh_models()

    # ------------------------------------------------------------- runs tab
    def _build_runs_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        side = Panel()
        side.setMinimumWidth(250)
        side.setMaximumWidth(320)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 12, 14, 12)
        side_layout.setSpacing(8)
        new_btn = button("New training run", "wand", "primary")
        new_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        side_layout.addWidget(new_btn)
        refresh_btn = button("", "refresh", "ghost", tooltip="Refresh runs")
        refresh_btn.clicked.connect(self.refresh_runs)
        head = QHBoxLayout()
        head.addWidget(label("RUNS", "eyebrow"), 1)
        head.addWidget(refresh_btn)
        side_layout.addLayout(head)
        self.run_list = QListWidget()
        self.run_list.setSpacing(2)
        self.run_list.itemSelectionChanged.connect(self._on_run_selected)
        side_layout.addWidget(self.run_list, 1)
        self.run_hint = label("", "faint", wrap=True)
        side_layout.addWidget(self.run_hint)
        splitter.addWidget(side)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(16, 4, 0, 0)
        detail_layout.setSpacing(8)
        head_row = QHBoxLayout()
        self.run_title = label("No training run selected", "h2")
        head_row.addWidget(self.run_title, 1)
        detail_layout.addLayout(head_row)
        self.run_meta = label("Select a run to see its progress, checkpoints and logs.", "faint", wrap=True)
        detail_layout.addWidget(self.run_meta)
        self.run_error = ErrorLine()
        detail_layout.addWidget(self.run_error)

        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tile_status = StatTile("Status", "—")
        self.tile_step = StatTile("Progress", "—")
        self.tile_loss = StatTile("Loss", "—")
        self.tile_eta = StatTile("Estimated remaining", "—")
        for tile in (self.tile_status, self.tile_step, self.tile_loss, self.tile_eta):
            tiles.addWidget(tile, 1)
        detail_layout.addLayout(tiles)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        detail_layout.addWidget(self.progress)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.stop_btn = button("Stop", "stop", "ghost", tooltip="Save a checkpoint and stop after this step")
        self.stop_btn.clicked.connect(lambda: self._stop_selected(False))
        self.force_btn = button("Force stop", "trash", "ghost", tooltip="Terminate the worker now")
        self.force_btn.clicked.connect(lambda: self._stop_selected(True))
        self.resume_btn = button("Resume", "play", "ghost", tooltip="Resume from the last checkpoint")
        self.resume_btn.clicked.connect(self._resume_selected)
        self.open_btn = button("Open folder", "folder", "ghost")
        self.open_btn.clicked.connect(self._open_output)
        self.merge_btn = button("Merge adapter", "layers", "ghost",
                                tooltip="Merge the LoRA adapter into a full checkpoint for inference")
        self.merge_btn.clicked.connect(self._merge_selected)
        self.eval_btn = button("Evaluate", "activity", "ghost", tooltip="Open the Evaluation Lab for this model")
        self.eval_btn.clicked.connect(self._evaluate_selected)
        self.delete_btn = button("", "trash", "ghost", tooltip="Delete this run record")
        self.delete_btn.clicked.connect(self._delete_selected)
        for widget in (self.stop_btn, self.force_btn, self.resume_btn, self.open_btn, self.merge_btn,
                       self.eval_btn, self.delete_btn):
            actions.addWidget(widget)
        actions.addStretch(1)
        detail_layout.addLayout(actions)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setDocumentMode(True)
        progress_page = QWidget()
        progress_layout = QVBoxLayout(progress_page)
        progress_layout.setContentsMargins(0, 8, 0, 0)
        progress_layout.setSpacing(8)
        self.chart = LossChart()
        progress_layout.addWidget(self.chart, 3)
        progress_layout.addWidget(label("LOGS", "eyebrow"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setPlaceholderText("Training output appears here.")
        progress_layout.addWidget(self.log_view, 2)
        self.detail_tabs.addTab(progress_page, "Progress")

        checkpoints_page = QWidget()
        checkpoints_layout = QVBoxLayout(checkpoints_page)
        checkpoints_layout.setContentsMargins(0, 8, 0, 0)
        self.checkpoints = _table(["Step", "Size", "Created", "Path"], 3)
        self.checkpoints.doubleClicked.connect(lambda _i: self._open_checkpoint())
        checkpoints_layout.addWidget(self.checkpoints)
        self.detail_tabs.addTab(checkpoints_page, "Checkpoints")

        details_page = QWidget()
        details_layout = QVBoxLayout(details_page)
        details_layout.setContentsMargins(0, 8, 0, 0)
        self.details_view = QPlainTextEdit()
        self.details_view.setReadOnly(True)
        self.details_view.setPlaceholderText("Run manifest and recipe.")
        details_layout.addWidget(self.details_view)
        self.detail_tabs.addTab(details_page, "Manifest")
        detail_layout.addWidget(self.detail_tabs, 1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 1)
        return page

    def _build_new_run_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(10)
        scroll.setWidget(container)

        # environment
        env_panel = Panel()
        env_layout = QVBoxLayout(env_panel)
        env_layout.setContentsMargins(16, 12, 16, 12)
        env_layout.setSpacing(6)
        env_layout.addWidget(label("TRAINING ENVIRONMENT", "eyebrow"))
        row = QHBoxLayout()
        self.env_combo = QComboBox()
        self.env_combo.setMinimumWidth(220)
        self.env_combo.currentIndexChanged.connect(self._on_environment_changed)
        row.addWidget(self.env_combo, 1)
        manage_btn = button("Manage environments…", "sliders", "ghost")
        manage_btn.clicked.connect(self.ctx.open_settings)
        row.addWidget(manage_btn)
        check_btn = button("Check dependencies", "refresh", "ghost")
        check_btn.clicked.connect(self._check_dependencies)
        row.addWidget(check_btn)
        env_layout.addLayout(row)
        self.env_status = label("", "faint", wrap=True)
        env_layout.addWidget(self.env_status)
        layout.addWidget(env_panel)

        # dataset
        dataset_panel = Panel()
        dataset_layout = QVBoxLayout(dataset_panel)
        dataset_layout.setContentsMargins(16, 12, 16, 12)
        dataset_layout.setSpacing(6)
        dataset_layout.addWidget(label("DATASET", "eyebrow"))
        row = QHBoxLayout()
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("JSONL, JSON, CSV, Parquet, TXT or Markdown")
        self.dataset_path.textChanged.connect(self._on_dataset_path_changed)
        row.addWidget(self.dataset_path, 1)
        browse = button("", "folder", "ghost", tooltip="Choose a dataset file")
        browse.clicked.connect(self._browse_dataset)
        row.addWidget(browse)
        inspect = button("Inspect", "eye", "ghost")
        inspect.clicked.connect(self._inspect_dataset)
        row.addWidget(inspect)
        dataset_layout.addLayout(row)
        self.dataset_report_label = label("Choose a dataset to see row counts, duplicates and previews.", "faint",
                                          wrap=True)
        dataset_layout.addWidget(self.dataset_report_label)
        self.dataset_preview = QPlainTextEdit()
        self.dataset_preview.setReadOnly(True)
        self.dataset_preview.setMaximumHeight(110)
        self.dataset_preview.setPlaceholderText("Short previews stay on this machine.")
        dataset_layout.addWidget(self.dataset_preview)
        layout.addWidget(dataset_panel)

        # model
        model_panel = Panel()
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(16, 12, 16, 12)
        model_layout.setSpacing(6)
        model_layout.addWidget(label("BASE MODEL", "eyebrow"))
        row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo)
        row.addWidget(self.model_combo, 1)
        self.model_id = QLineEdit()
        self.model_id.setPlaceholderText("Hub repo id or local model folder")
        self.model_id.textEdited.connect(lambda: self.model_combo.setCurrentIndex(0))
        row.addWidget(self.model_id, 2)
        browse_model = button("", "folder", "ghost", tooltip="Choose a local model folder")
        browse_model.clicked.connect(self._browse_model)
        row.addWidget(browse_model)
        model_layout.addLayout(row)
        advanced = QFormLayout()
        advanced.setVerticalSpacing(6)
        self.revision = QLineEdit()
        self.revision.setPlaceholderText("main")
        advanced.addRow("Revision", self.revision)
        self.trust_remote = QCheckBox("Allow remote code from this model")
        advanced.addRow("", self.trust_remote)
        model_layout.addLayout(advanced)
        layout.addWidget(model_panel)

        # method + parameters
        methods = QHBoxLayout()
        methods.setSpacing(10)
        method_panel = Panel()
        method_layout = QFormLayout(method_panel)
        method_layout.setContentsMargins(16, 12, 16, 12)
        method_layout.setVerticalSpacing(6)
        method_layout.addRow(label("METHOD", "eyebrow"))
        self.mode_combo = QComboBox()
        for text, value in MODES:
            self.mode_combo.addItem(text, value)
        self.mode_combo.currentIndexChanged.connect(self._update_mode_fields)
        method_layout.addRow("Training mode", self.mode_combo)
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 512)
        self.lora_rank.setValue(16)
        method_layout.addRow("LoRA rank", self.lora_rank)
        self.lora_alpha = QSpinBox()
        self.lora_alpha.setRange(1, 1024)
        self.lora_alpha.setValue(32)
        method_layout.addRow("LoRA alpha", self.lora_alpha)
        self.lora_dropout = QDoubleSpinBox()
        self.lora_dropout.setRange(0.0, 0.9)
        self.lora_dropout.setSingleStep(0.05)
        self.lora_dropout.setValue(0.05)
        method_layout.addRow("LoRA dropout", self.lora_dropout)
        self.lora_targets = QLineEdit(", ".join(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                 "down_proj")))
        method_layout.addRow("Target modules", self.lora_targets)
        methods.addWidget(method_panel, 1)

        params_panel = Panel()
        params_layout = QFormLayout(params_panel)
        params_layout.setContentsMargins(16, 12, 16, 12)
        params_layout.setVerticalSpacing(6)
        params_layout.addRow(label("PARAMETERS", "eyebrow"))
        self.output_name = QLineEdit("noelle-lora")
        params_layout.addRow("Output name", self.output_name)
        self.epochs = QDoubleSpinBox()
        self.epochs.setRange(0.1, 100.0)
        self.epochs.setSingleStep(0.5)
        self.epochs.setValue(3.0)
        params_layout.addRow("Epochs", self.epochs)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(8)
        self.learning_rate.setRange(1e-7, 1e-2)
        self.learning_rate.setSingleStep(1e-5)
        self.learning_rate.setValue(2e-4)
        params_layout.addRow("Learning rate", self.learning_rate)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        params_layout.addRow("Batch size", self.batch_size)
        self.grad_accum = QSpinBox()
        self.grad_accum.setRange(1, 512)
        self.grad_accum.setValue(8)
        params_layout.addRow("Gradient accumulation", self.grad_accum)
        self.max_seq = QSpinBox()
        self.max_seq.setRange(128, 32768)
        self.max_seq.setSingleStep(256)
        self.max_seq.setValue(1024)
        params_layout.addRow("Max sequence length", self.max_seq)
        self.precision = QComboBox()
        for text, value in (("Auto", "auto"), ("bf16", "bf16"), ("fp16", "fp16"), ("fp32", "fp32")):
            self.precision.addItem(text, value)
        params_layout.addRow("Precision", self.precision)
        self.save_steps = QSpinBox()
        self.save_steps.setRange(1, 100000)
        self.save_steps.setValue(100)
        params_layout.addRow("Save every N steps", self.save_steps)
        self.validation_ratio = QDoubleSpinBox()
        self.validation_ratio.setRange(0.0, 0.5)
        self.validation_ratio.setSingleStep(0.05)
        self.validation_ratio.setValue(0.1)
        params_layout.addRow("Validation split", self.validation_ratio)
        self.dedupe = QCheckBox("Remove duplicate rows")
        self.dedupe.setChecked(True)
        self.trim_long = QCheckBox("Trim samples longer than the sequence limit")
        params_layout.addRow("", self.dedupe)
        params_layout.addRow("", self.trim_long)
        self.grad_checkpoint = QCheckBox("Gradient checkpointing (saves memory)")
        self.grad_checkpoint.setChecked(True)
        params_layout.addRow("", self.grad_checkpoint)
        methods.addWidget(params_panel, 1)
        layout.addLayout(methods)

        # plan + start
        plan_panel = Panel()
        plan_layout = QVBoxLayout(plan_panel)
        plan_layout.setContentsMargins(16, 12, 16, 12)
        plan_layout.setSpacing(6)
        plan_head = QHBoxLayout()
        plan_head.addWidget(label("HARDWARE PLAN", "eyebrow"), 1)
        plan_btn = button("Refresh plan", "refresh", "ghost")
        plan_btn.clicked.connect(self._refresh_plan)
        plan_head.addWidget(plan_btn)
        self.recommend_btn = button("Apply recommendation", "check", "ghost")
        self.recommend_btn.clicked.connect(self._apply_recommendation)
        self.recommend_btn.setEnabled(False)
        plan_head.addWidget(self.recommend_btn)
        plan_layout.addLayout(plan_head)
        self.plan_risk = label("Choose a model and dataset to estimate memory and time.", "faint", wrap=True)
        plan_layout.addWidget(self.plan_risk)
        self.plan_notes = label("", "faint", wrap=True)
        plan_layout.addWidget(self.plan_notes)
        self.form_error = ErrorLine()
        plan_layout.addWidget(self.form_error)
        row = QHBoxLayout()
        self.load_recipe_btn = button("Load recipe", "upload", "ghost")
        self.load_recipe_btn.clicked.connect(self._load_recipe)
        save_recipe_btn = button("Save recipe", "download", "ghost")
        save_recipe_btn.clicked.connect(self._save_recipe)
        row.addWidget(self.load_recipe_btn)
        row.addWidget(save_recipe_btn)
        row.addStretch(1)
        self.start_btn = button("Start training", "play", "primary")
        self.start_btn.clicked.connect(self._start_training)
        row.addWidget(self.start_btn)
        plan_layout.addLayout(row)
        self.start_status = label("", "faint", wrap=True)
        plan_layout.addWidget(self.start_status)
        layout.addWidget(plan_panel)
        layout.addStretch(1)
        self._update_mode_fields()
        return scroll

    def _build_dataset_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)
        panel = Panel()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 12, 16, 12)
        panel_layout.setSpacing(8)
        panel_layout.addWidget(label("DATASET STUDIO", "eyebrow"))
        panel_layout.addWidget(label("Normalize ChatML, instruction, query and plain-text rows locally. Names are "
                                     "hashed and counts are saved with the run; dataset contents are not written to logs.",
                                     "faint", wrap=True))
        row = QHBoxLayout()
        self.studio_path = QLineEdit()
        self.studio_path.setPlaceholderText("Choose a dataset file")
        row.addWidget(self.studio_path, 1)
        browse = button("", "folder", "ghost", tooltip="Choose a dataset file")
        browse.clicked.connect(self._browse_studio_dataset)
        row.addWidget(browse)
        inspect = button("Inspect", "eye", "ghost")
        inspect.clicked.connect(self._inspect_studio_dataset)
        row.addWidget(inspect)
        prepare = button("Prepare split", "layers", "ghost")
        prepare.clicked.connect(self._prepare_studio_dataset)
        row.addWidget(prepare)
        use_btn = button("Use in new run", "wand", "primary")
        use_btn.clicked.connect(self._use_studio_dataset)
        row.addWidget(use_btn)
        panel_layout.addLayout(row)
        self.studio_report_label = label("", "faint", wrap=True)
        panel_layout.addWidget(self.studio_report_label)
        self.studio_preview = QPlainTextEdit()
        self.studio_preview.setReadOnly(True)
        panel_layout.addWidget(self.studio_preview, 1)
        layout.addWidget(panel, 1)
        return page

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.refresh_runs()
        self.refresh_environments()
        self.refresh_models()
        if not self._selected_run_id and self.run_list.count():
            self.run_list.setCurrentRow(0)

    def on_activated(self) -> None:
        self.start()

    def on_library_changed(self) -> None:
        self.refresh_models()

    def shutdown(self) -> None:
        if self.manager is not None:
            self.manager.shutdown()

    # --------------------------------------------------------------- runs
    def refresh_runs(self, select: str | None = None) -> None:
        if self.manager is None:
            self.run_hint.setText("The training service is unavailable.")
            return
        runs = self.manager.list_runs(200)
        chosen = select or self._selected_run_id
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for run in runs:
            item = QListWidgetItem(self._run_item_text(run))
            item.setData(Qt.ItemDataRole.UserRole, run["id"])
            item.setToolTip(f"{run['id']}\n{run.get('model_id') or ''}")
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)
        self.run_hint.setText("" if runs else "No runs yet. Prepare a dataset and start your first local fine-tuning run.")
        if runs:
            index = next((i for i in range(self.run_list.count())
                          if self.run_list.item(i).data(Qt.ItemDataRole.UserRole) == chosen), 0)
            self.run_list.blockSignals(True)
            self.run_list.setCurrentRow(index)
            self.run_list.blockSignals(False)
            self._selected_run_id = self.run_list.item(index).data(Qt.ItemDataRole.UserRole)
            self._render_run()
        else:
            self._selected_run_id = None
            self.run_title.setText("No training run selected")
            self.run_meta.setText("Select a run to see its progress, checkpoints and logs.")
            self.run_error.clear()
            self._render_empty_details()

    @staticmethod
    def _run_item_text(run: dict[str, Any]) -> str:
        status = STATUS_LABELS.get(str(run.get("status")), str(run.get("status") or "unknown"))
        name = run.get("name") or run.get("output_name") or run["id"]
        return f"{name}\n{status} · {run.get('mode') or ''} · {time_ago(run.get('created_at'))}"

    def _update_run_item(self, run_id: str) -> None:
        run = self.manager.get(run_id) if self.manager is not None else None
        if run is None:
            return
        for index in range(self.run_list.count()):
            item = self.run_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == run_id:
                item.setText(self._run_item_text(run))
                break

    def _on_job_added(self, run_id: str) -> None:
        self.refresh_runs(select=run_id)
        self.ctx.toast(f"Training run {run_id} started.", "info")

    def _on_job_updated(self, run_id: str) -> None:
        self._update_run_item(run_id)
        if run_id == self._selected_run_id:
            self._render_pending = True
            if not self._render_timer.isActive():
                self._render_timer.start()

    def _render_later(self) -> None:
        if self._render_pending:
            self._render_pending = False
            self._render_run()

    def _on_job_finished(self, run_id: str, ok: bool) -> None:
        self._update_run_item(run_id)
        if run_id == self._selected_run_id:
            self._render_pending = False
            self._render_timer.stop()
            self._render_run()
        run = self.manager.get(run_id) if self.manager is not None else None
        if ok:
            self.ctx.toast(f"Training completed: {run_id}", "success")
        elif run is not None and run.get("status") in ("failed", "oom"):
            self.ctx.toast(f"Training failed: {run.get('error') or run_id}", "error")

    def _on_log_line(self, run_id: str, line: str) -> None:
        if run_id == self._selected_run_id:
            self.log_view.appendPlainText(line)

    def _on_run_selected(self) -> None:
        item = self.run_list.currentItem()
        if item is None:
            return
        self._selected_run_id = item.data(Qt.ItemDataRole.UserRole)
        self._render_run()

    def _render_empty_details(self) -> None:
        self.tile_status.set("—")
        self.tile_step.set("—")
        self.tile_loss.set("—")
        self.tile_eta.set("—")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.chart.set_metrics([])
        self.log_view.clear()
        self._log_run_id = None
        self.checkpoints.setRowCount(0)
        self.details_view.clear()
        for widget in (self.stop_btn, self.force_btn, self.resume_btn, self.open_btn, self.merge_btn,
                       self.eval_btn, self.delete_btn):
            widget.setEnabled(False)

    def _render_run(self) -> None:
        if self.manager is None or not self._selected_run_id:
            return
        run = self.manager.get(self._selected_run_id)
        if run is None:
            return
        status = str(run.get("status") or "")
        self.run_title.setText(str(run.get("name") or run.get("output_name") or run["id"]))
        active = self.manager.is_active(run["id"])
        mode = run.get("mode") or ""
        created = time_ago(run.get("created_at"))
        self.run_meta.setText(f"{STATUS_LABELS.get(status, status)} · {mode} · {run.get('model_id') or '—'} · "
                              f"started {created}")
        if run.get("error") and status in ("failed", "oom", "interrupted"):
            self.run_error.set_error(str(run["error"]),
                                     "Resume from the latest checkpoint, or lower batch size / sequence length.")
        else:
            self.run_error.clear()

        step = int(run.get("step") or 0)
        total = int(run.get("total_steps") or 0)
        progress_text = f"{step:,} / {total:,}" if total else f"{step:,}"
        self.tile_status.set(STATUS_LABELS.get(status, status), f"updated {time_ago(run.get('finished_at') or run.get('started_at'))}")
        self.tile_step.set(progress_text, f"epoch {run.get('epoch') or 0:.2f}" if run.get("epoch") else "")
        loss = run.get("loss")
        self.tile_loss.set(f"{loss:.4f}" if isinstance(loss, (int, float)) else "—",
                           f"{run.get('tokens_per_second'):.1f} tok/s" if run.get("tokens_per_second") else "")
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(min(step, max(total, 1)))
        self.progress.setFormat(f"{min(step, total):,} / {total:,} steps" if total else "Preparing…")

        if status in ("training", "stopping") and total and 0 < step < total and run.get("started_at"):
            elapsed = max(time.time() - float(run["started_at"]), 1.0)
            remaining = elapsed / step * (total - step)
            self.tile_eta.set(human_duration(remaining))
        elif status == "completed":
            self.tile_eta.set("Done")
        else:
            self.tile_eta.set("—")

        self.chart.set_metrics(self.manager.metrics(run["id"]) if self.manager else [])
        if self._log_run_id != run["id"]:
            self.log_view.setPlainText(self.manager.logs(run["id"]) if self.manager else "")
            self._log_run_id = run["id"]
        self._render_checkpoints(run)
        manifest = run.get("manifest") or {}
        details = {"run": {k: v for k, v in run.items() if k not in ("config", "manifest")},
                   "config": run.get("config") or {}, "manifest": manifest}
        self.details_view.setPlainText(json.dumps(details, indent=2, default=str))

        self.stop_btn.setEnabled(active and status != "stopping")
        self.force_btn.setEnabled(active)
        checkpoints = self.manager.checkpoints(run["id"]) if self.manager else []
        self.resume_btn.setEnabled(not active and bool(checkpoints)
                                   and status in ("stopped", "interrupted", "failed", "oom"))
        output = str(run.get("output_path") or "")
        self.open_btn.setEnabled(bool(output) and Path(output).exists())
        self.merge_btn.setEnabled(bool(output) and mode in ("lora", "qlora") and Path(output).exists())
        self.eval_btn.setEnabled(bool(output) and Path(output).exists() and mode not in ("lora", "qlora"))
        self.delete_btn.setEnabled(not active)

    def _render_checkpoints(self, run: dict[str, Any]) -> None:
        rows = self.manager.checkpoints(run["id"]) if self.manager else []
        self.checkpoints.setRowCount(len(rows))
        for index, checkpoint in enumerate(rows):
            self.checkpoints.setItem(index, 0, QTableWidgetItem(str(checkpoint.get("step") or "—")))
            self.checkpoints.setItem(index, 1, QTableWidgetItem(human_bytes(checkpoint.get("size_bytes"))))
            self.checkpoints.setItem(index, 2, QTableWidgetItem(time_ago(checkpoint.get("created_at"))))
            item = QTableWidgetItem(str(checkpoint.get("path") or ""))
            item.setData(Qt.ItemDataRole.UserRole, checkpoint.get("path"))
            self.checkpoints.setItem(index, 3, item)

    def _stop_selected(self, force: bool) -> None:
        if not self._selected_run_id or self.manager is None:
            return
        self.manager.stop(self._selected_run_id, force=force)
        self.ctx.toast("Stopping the training worker…" if not force else "Terminating the training worker…", "info")
        self._render_run()

    def _resume_selected(self) -> None:
        if not self._selected_run_id or self.manager is None:
            return
        try:
            self.manager.resume(self._selected_run_id)
        except ConfigError as exc:
            self.banner.set_error(str(exc))
            return
        self.ctx.toast("Resuming from the latest checkpoint.", "info")
        self.refresh_runs(select=self._selected_run_id)

    def _open_output(self) -> None:
        if not self._selected_run_id or self.manager is None:
            return
        run = self.manager.get(self._selected_run_id) or {}
        path = Path(str(run.get("output_path") or ""))
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_checkpoint(self) -> None:
        row = self.checkpoints.currentRow()
        if row < 0:
            return
        item = self.checkpoints.item(row, 3)
        path = Path(str(item.data(Qt.ItemDataRole.UserRole) or "")) if item else Path()
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _delete_selected(self) -> None:
        if not self._selected_run_id or self.manager is None:
            return
        answer = QMessageBox.question(
            self, "Delete training run?",
            "Remove this run from Training Studio? Checkpoints and the trained model stay on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.manager.store.delete(self._selected_run_id)
        self._selected_run_id = None
        self.refresh_runs()

    # ------------------------------------------------------------ merge/eval
    def _merge_selected(self) -> None:
        if not self._selected_run_id or self.manager is None:
            return
        run = self.manager.get(self._selected_run_id) or {}
        config = run.get("config") or {}
        model = config.get("model") or {}
        base = str(model.get("local_path") or "")
        if not base:
            from noelle.runtime.hub_cache import cached_snapshot_path

            snapshot = cached_snapshot_path(str(model.get("id") or ""))
            base = str(snapshot) if snapshot else ""
        if not base or not Path(base).exists():
            self.banner.set_error("The base model is not available locally.",
                                  "Download it on Discover first so the merge stays offline.")
            return
        output = str(run.get("output_path") or "")
        target = Path(self.manager.store.run_dir(run["id"])) / "exports" / f"{run.get('output_name')}-merged"
        self.merge_btn.setEnabled(False)
        self.ctx.toast("Merging the adapter into a full checkpoint…", "info")

        def work() -> str:
            return self.manager.merge(base, output, str(target))

        def done(path: str) -> None:
            from noelle.app.services.model_registry import ModelRecord

            repo_id = f"trained/{Path(path).name}"
            size = sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
            self.ctx.registry.upsert_model(ModelRecord(
                repo_id=repo_id, pipeline_tag="text-generation", local_path=path, source="local",
                base_model=str(model.get("id") or "") or None, size_bytes=size, engine="transformers",
                metadata={"training_run": run["id"], "merged": True}))
            self.ctx.toast(f"Merged model added to the library as {repo_id}", "success")
            self._render_run()

        runner().submit(work, on_result=done,
                        on_error=lambda message: self.banner.set_error(f"Merge failed: {message}"),
                        on_finished=lambda: self._render_run())

    def _evaluate_selected(self) -> None:
        if not self._selected_run_id or self.manager is None:
            return
        run = self.manager.get(self._selected_run_id) or {}
        name = f"trained/{run.get('output_name') or run['id']}"
        self.ctx.navigate("evaluate")
        page = self.ctx.extras.get("evaluate_page")
        if page is None:
            self.ctx.toast("Open Evaluate and select the trained model.", "info")
            return
        page.set_checked([name])

    # --------------------------------------------------------------- new run
    def refresh_environments(self) -> None:
        if not hasattr(self, "env_combo"):
            return
        current = str(self.ctx.settings.get("training.environment") or "")
        self.env_combo.blockSignals(True)
        self.env_combo.clear()
        self.env_combo.addItem("App Python (bundled runtime)", "")
        try:
            environments = environment_manager.list_environments()
        except Exception:  # noqa: BLE001
            environments = []
        for environment in environments:
            if environment.get("status") == "incomplete":
                continue
            profiles = ", ".join(environment.get("profiles") or [])
            self.env_combo.addItem(f"{environment['name']}  ({profiles or 'base'})", environment["name"])
        index = self.env_combo.findData(current)
        self.env_combo.setCurrentIndex(index if index >= 0 else 0)
        self.env_combo.blockSignals(False)
        self._on_environment_changed()

    def _on_environment_changed(self) -> None:
        selected = self.env_combo.currentData() or ""
        if selected != str(self.ctx.settings.get("training.environment") or ""):
            self.ctx.settings.set("training.environment", selected or None)
        if selected:
            self.env_status.setText(f"Runs use the isolated environment “{selected}”. Its packages are checked by the "
                                    "training worker; QLoRA also needs the cuda-quant profile.")
        else:
            self.env_status.setText("Runs use the app's Python. For a clean install, create an environment with the "
                                    "training profile in Settings › Runtime environments.")

    def refresh_models(self) -> None:
        if not hasattr(self, "model_combo"):
            return
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Choose from the library…", None)
        try:
            records = [record for record in self.ctx.registry.list_models(only_existing=True)
                       if (record.family or tasks.TEXT) == tasks.TEXT]
        except Exception:  # noqa: BLE001
            records = []
        for record in records:
            payload = {"id": record.repo_id, "local_path": record.local_path, "param_count": record.param_count}
            self.model_combo.addItem(f"{record.display_name}  ·  {record.precision}", payload)
        self.model_combo.blockSignals(False)

    def _on_model_combo(self, index: int) -> None:
        payload = self.model_combo.itemData(index)
        if isinstance(payload, dict):
            self.model_id.setText(str(payload.get("local_path") or payload.get("id") or ""))
            self.model_combo.setToolTip(str(payload.get("id") or ""))

    def _browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a training dataset", str(Path.home()),
            "Datasets (*.jsonl *.json *.csv *.parquet *.txt *.md *.markdown);;All files (*)")
        if path:
            self.dataset_path.setText(path)
            self._inspect_dataset()

    def _browse_studio_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a dataset", str(Path.home()),
            "Datasets (*.jsonl *.json *.csv *.parquet *.txt *.md *.markdown);;All files (*)")
        if path:
            self.studio_path.setText(path)
            self._inspect_studio_dataset()

    def _browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a local model folder", str(Path.home()))
        if path:
            self.model_id.setText(path)

    def _inspect_dataset(self) -> None:
        path = self.dataset_path.text().strip()
        if not path:
            self.form_error.set_error("Choose a dataset file first.")
            return
        if not Path(path).expanduser().is_file():
            self.form_error.set_error(f"Dataset not found: {path}")
            return
        self.dataset_report_label.setText("Inspecting dataset locally…")
        self.form_error.clear()

        def done(report: DatasetReport) -> None:
            self._dataset_report = report
            self.dataset_report_label.setText(_dataset_summary(report))
            preview = []
            for row in report.preview:
                preview.append(f"[{row.get('schema')}, {row.get('tokens')} tokens] {row.get('preview')}")
            self.dataset_preview.setPlainText("\n\n".join(preview) or "(no preview)")
            self._refresh_plan()

        runner().submit(analyze_dataset, path, max_seq_length=self.max_seq.value(),
                        validation_ratio=self.validation_ratio.value(), on_result=done,
                        on_error=lambda message: self.dataset_report_label.setText(f"Could not read the dataset: {message}"))

    def _inspect_studio_dataset(self) -> None:
        path = self.studio_path.text().strip()
        if not path:
            self.studio_report_label.setText("Choose a dataset file first.")
            return
        if not Path(path).expanduser().is_file():
            self.studio_report_label.setText(f"Dataset not found: {path}")
            return
        self.studio_report_label.setText("Inspecting dataset locally…")

        def done(report: DatasetReport) -> None:
            self._studio_report = report
            self.studio_report_label.setText(_dataset_summary(report))
            preview = [f"[{row.get('schema')}, {row.get('tokens')} tokens] {row.get('preview')}"
                       for row in report.preview]
            self.studio_preview.setPlainText("\n\n".join(preview) or "(no preview)")

        runner().submit(analyze_dataset, path, on_result=done,
                        on_error=lambda message: self.studio_report_label.setText(f"Could not read the dataset: {message}"))

    def _use_studio_dataset(self) -> None:
        path = self.studio_path.text().strip()
        if path:
            self.dataset_path.setText(path)
            self.dataset_report_label.setText(self.studio_report_label.text())
            self.dataset_preview.setPlainText(self.studio_preview.toPlainText())
            self._dataset_report = self._studio_report
            self.tabs.setCurrentIndex(1)
            self._refresh_plan()

    def _prepare_studio_dataset(self) -> None:
        path = self.studio_path.text().strip()
        if not path or not Path(path).expanduser().is_file():
            self.studio_report_label.setText("Choose a dataset file first.")
            return
        output = QFileDialog.getExistingDirectory(self, "Choose a folder for the prepared dataset",
                                                  str(Path.home()))
        if not output:
            return
        self.studio_report_label.setText("Normalizing and splitting the dataset locally…")

        def done(prepared: Any) -> None:
            self._studio_report = prepared.report
            self.studio_report_label.setText(
                _dataset_summary(prepared.report)
                + f"<br>Prepared: {prepared.train_path} and {prepared.validation_path}")

        runner().submit(prepare_dataset, path, str(Path(output) / "noelle-prepared"), max_seq_length=1024,
                        validation_ratio=0.1, on_result=done,
                        on_error=lambda message: self.studio_report_label.setText(
                            f"Could not prepare the dataset: {message}"))

    def _update_mode_fields(self) -> None:
        lora = self.mode_combo.currentData() in ("lora", "qlora")
        for widget in (self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_targets):
            widget.setEnabled(lora)
        if self.mode_combo.currentData() == "qlora":
            self.precision.setCurrentIndex(0)

    # -------------------------------------------------------------- planning
    def _hardware(self) -> HardwareInfo:
        budget = self.ctx.hardware.memory_budget(0.0)
        gpu = getattr(self.ctx.hardware, "gpu", None)
        try:
            disk = int(shutil.disk_usage(paths.home()).free)
        except OSError:
            disk = 0
        return HardwareInfo(device=budget.device, gpu_name=getattr(gpu, "name", "") or "",
                            vram_total_bytes=int(getattr(gpu, "total_memory", 0) or 0),
                            vram_usable_bytes=int(budget.usable_bytes) if budget.device != "cpu" else 0,
                            ram_total_bytes=int(budget.system_ram_bytes or 0), disk_free_bytes=disk)

    def _config(self) -> TrainingConfig:
        model_text = self.model_id.text().strip()
        if not model_text:
            raise ConfigError(["Choose a base model or local model folder."])
        if Path(model_text).expanduser().is_dir():
            local_path = str(Path(model_text).expanduser())
            record = next((item for item in self.ctx.registry.list_models()
                           if item.local_path and Path(item.local_path).expanduser() == Path(local_path)), None)
            model = ModelConfig(id=record.repo_id if record else Path(local_path).name, local_path=local_path,
                                trust_remote_code=self.trust_remote.isChecked())
        else:
            model = ModelConfig(id=model_text, revision=self.revision.text().strip() or None,
                                trust_remote_code=self.trust_remote.isChecked())
        name = self.output_name.text().strip()
        if not name:
            raise ConfigError(["Give the output a name."])
        targets = [item.strip() for item in self.lora_targets.text().split(",") if item.strip()]
        config = TrainingConfig(
            name=name, output_name=name, mode=str(self.mode_combo.currentData()),
            model=model,
            dataset=DatasetConfig(path=str(Path(self.dataset_path.text().strip()).expanduser()),
                                  validation_ratio=self.validation_ratio.value(),
                                  max_seq_length=self.max_seq.value(), dedupe=self.dedupe.isChecked(),
                                  trim_long=self.trim_long.isChecked()),
            lora=LoRAConfig(rank=self.lora_rank.value(), alpha=self.lora_alpha.value(),
                            dropout=self.lora_dropout.value(), target_modules=targets),
            params=TrainingParams(epochs=self.epochs.value(), learning_rate=self.learning_rate.value(),
                                  batch_size=self.batch_size.value(),
                                  gradient_accumulation_steps=self.grad_accum.value(),
                                  precision=str(self.precision.currentData()),
                                  gradient_checkpointing=self.grad_checkpoint.isChecked(),
                                  save_steps=self.save_steps.value()),
        )
        return config

    def _refresh_plan(self) -> None:
        if not self.model_id.text().strip() or not self.dataset_path.text().strip():
            self._plan = None
            self.recommend_btn.setEnabled(False)
            self.plan_risk.setText("Choose a model and dataset to estimate memory and time.")
            self.plan_notes.clear()
            self.form_error.clear()
            return
        try:
            config = self._config()
            config.validate()
        except ConfigError as exc:
            self.form_error.set_error("; ".join(exc.problems))
            self.recommend_btn.setEnabled(False)
            self.plan_risk.setText("Complete the model and dataset fields to estimate the plan.")
            self.plan_notes.clear()
            return
        param_count = None
        payload = self.model_combo.currentData()
        if isinstance(payload, dict):
            param_count = payload.get("param_count")
        report = self._dataset_report
        plan = estimate_plan(config, self._hardware(), param_count=param_count,
                             dataset_samples=report.valid if report else 0,
                             dataset_avg_tokens=int(report.avg_tokens) if report else 0)
        self._plan = plan
        self.recommend_btn.setEnabled(bool(plan.recommended))
        color = {"SAFE": theme.SUCCESS, "WARNING": theme.WARNING, "HIGH RISK": theme.DANGER,
                 "UNSUPPORTED": theme.DANGER}.get(plan.risk, theme.TEXT_MUTED)
        vram = f"{plan.estimated_vram_bytes / 1e9:.1f} GB VRAM" if plan.estimated_vram_bytes else "CPU only"
        self.plan_risk.setText(
            f"<span style='color:{color}'><b>{plan.risk}</b></span> · {plan.mode.upper()} · {plan.precision} · "
            f"{vram} · {plan.estimated_ram_bytes / 1e9:.1f} GB RAM · {plan.disk_bytes / 1e9:.1f} GB disk · "
            f"~{human_duration(plan.estimated_seconds)} · {plan.expected_steps:,} steps")
        notes = list(plan.notes)
        recommendation = plan.recommended or {}
        if recommendation.get("note"):
            notes.append(str(recommendation["note"]))
        self.plan_notes.setText("<br>".join(f"• {note}" for note in notes))
        self.form_error.clear()

    def _apply_recommendation(self) -> None:
        if self._plan is None or not self._plan.recommended:
            return
        recommendation = self._plan.recommended
        index = self.mode_combo.findData(recommendation.get("mode"))
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        self.max_seq.setValue(int(recommendation.get("max_seq_length") or self.max_seq.value()))
        self.batch_size.setValue(int(recommendation.get("batch_size") or self.batch_size.value()))
        self.grad_accum.setValue(int(recommendation.get("gradient_accumulation_steps") or self.grad_accum.value()))
        precision_index = self.precision.findData(recommendation.get("precision"))
        if precision_index >= 0:
            self.precision.setCurrentIndex(precision_index)
        self._update_mode_fields()
        self._refresh_plan()
        self.start_status.setText("Applied the conservative recommendation for this hardware.")

    def _check_dependencies(self) -> None:
        if self.manager is None:
            return
        self.env_status.setText("Checking training packages…")
        runner().submit(self.manager.dependencies, on_result=self._dependencies_done,
                        on_error=lambda message: self.env_status.setText(f"Could not check packages: {message}"))

    def _dependencies_done(self, report: dict[str, Any]) -> None:
        missing = report.get("missing") or []
        if self.mode_combo.currentData() == "qlora" and not report.get("bitsandbytes"):
            missing = [*missing, "bitsandbytes"]
        if report.get("error"):
            self.env_status.setText(f"Training environment error: {report['error']}")
            return
        if missing:
            self.env_status.setText("Missing packages: " + ", ".join(missing)
                                    + ". Create an environment with the training profile in Settings, or install them.")
            return
        versions = report.get("versions") or {}
        bits = " · bitsandbytes installed" if report.get("bitsandbytes") else ""
        torch_version = versions.get("torch") or "?"
        self.env_status.setText(f"Training packages are ready (torch {torch_version}){bits}.")

    def _start_training(self) -> None:
        if self.manager is None or self._starting:
            return
        try:
            config = self._config()
            config.validate()
        except ConfigError as exc:
            self.form_error.set_error("; ".join(exc.problems))
            return
        self._starting = True
        self.start_btn.setEnabled(False)
        self.start_status.setText("Checking the training environment…")
        self.form_error.clear()

        def proceed() -> None:
            try:
                run_id = self.manager.start(config)
            except ConfigError as exc:
                self.form_error.set_error("; ".join(exc.problems))
                self.start_status.setText("")
                return
            except Exception as exc:  # noqa: BLE001
                self.form_error.set_error(f"Could not start training: {exc}")
                self.start_status.setText("")
                return
            finally:
                self._starting = False
                self.start_btn.setEnabled(True)
            self.start_status.setText(f"Run {run_id} started in its own worker process.")
            self.tabs.setCurrentIndex(0)
            self.refresh_runs(select=run_id)

        def checked(report: dict[str, Any]) -> None:
            missing = list(report.get("missing") or [])
            if self.mode_combo.currentData() == "qlora" and not report.get("bitsandbytes"):
                missing.append("bitsandbytes")
            if report.get("error"):
                self.form_error.set_error(f"Training environment error: {report['error']}")
            elif missing:
                self.form_error.set_error("Install the training packages first: " + ", ".join(missing),
                                          "Settings › Runtime environments › Create with the training profile.")
            else:
                proceed()
                return
            self._starting = False
            self.start_btn.setEnabled(True)
            self.start_status.setText("")

        if config.mode == "qlora" and self._hardware().device != "cuda":
            self.form_error.set_error("QLoRA needs an NVIDIA CUDA GPU (bitsandbytes NF4).",
                                      "Choose LoRA or full fine-tuning on this machine.")
            self._starting = False
            self.start_btn.setEnabled(True)
            self.start_status.setText("")
            return
        runner().submit(self.manager.dependencies, on_result=checked,
                        on_error=lambda message: self._start_error(message))

    def _start_error(self, message: str) -> None:
        self._starting = False
        self.start_btn.setEnabled(True)
        self.start_status.setText("")
        self.form_error.set_error(f"Could not check the training environment: {message}")

    # --------------------------------------------------------------- recipes
    def _load_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load a training recipe", str(Path.home()),
                                              "Recipes (*.yaml *.yml *.json);;All files (*)")
        if not path:
            return
        try:
            config = load_recipe(path)
        except Exception as exc:  # noqa: BLE001 - hand-edited recipes can fail in many parser/library ways
            self.form_error.set_error(f"Could not load the recipe: {exc}")
            return
        self._populate_form(config)
        self._refresh_plan()

    def _save_recipe(self) -> None:
        try:
            config = self._config()
        except ConfigError as exc:
            self.form_error.set_error("; ".join(exc.problems))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save training recipe",
                                              str(Path.home() / f"{config.output_name}.yaml"),
                                              "Recipe (*.yaml);;JSON (*.json)")
        if not path:
            return
        try:
            saved_path = Path(path)
            if saved_path.suffix.lower() == ".json":
                saved_path.write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
                saved = saved_path
            else:
                saved = save_recipe(config, path)
        except OSError as exc:
            self.form_error.set_error(f"Could not save the recipe: {exc}")
            return
        self.ctx.toast(f"Recipe saved to {saved}", "success")

    def _populate_form(self, config: TrainingConfig) -> None:
        self.model_id.setText(config.model.local_path or config.model.id)
        self.revision.setText(config.model.revision or "")
        self.trust_remote.setChecked(config.model.trust_remote_code)
        self.dataset_path.setText(config.dataset.path)
        index = self.mode_combo.findData(config.mode)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.lora_rank.setValue(config.lora.rank)
        self.lora_alpha.setValue(config.lora.alpha)
        self.lora_dropout.setValue(config.lora.dropout)
        self.lora_targets.setText(", ".join(config.lora.target_modules))
        self.output_name.setText(config.output_name)
        self.epochs.setValue(config.params.epochs)
        self.learning_rate.setValue(config.params.learning_rate)
        self.batch_size.setValue(config.params.batch_size)
        self.grad_accum.setValue(config.params.gradient_accumulation_steps)
        self.max_seq.setValue(config.dataset.max_seq_length)
        precision_index = self.precision.findData(config.params.precision)
        self.precision.setCurrentIndex(precision_index if precision_index >= 0 else 0)
        self.save_steps.setValue(config.params.save_steps)
        self.validation_ratio.setValue(config.dataset.validation_ratio)
        self.dedupe.setChecked(config.dataset.dedupe)
        self.trim_long.setChecked(config.dataset.trim_long)
        self.grad_checkpoint.setChecked(config.params.gradient_checkpointing)
        self._update_mode_fields()

    def _on_dataset_path_changed(self) -> None:
        self._dataset_report = None