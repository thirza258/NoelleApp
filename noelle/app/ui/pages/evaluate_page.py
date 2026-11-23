"""Evaluation Lab: run datasets against local models, track experiments, see measured performance."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
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
from noelle.app.services import experiments
from noelle.app.ui.widgets.common import Banner, Panel, button, label
from noelle.runtime import evaluation, tasks
from noelle.runtime.preflight import format_bytes
from noelle.runtime.protocol import Command, Event

LEADERBOARD = ["Model", "Precision", "Engine", "Accuracy", "Mean score", "TTFT p50", "tok/s", "Peak memory", "Load",
               "Status"]
CASES = ["Case", "Result", "Score", "TTFT", "tok/s", "Output", "Detail"]
HISTORY = ["Date", "Name", "Kind", "Dataset", "Models", "Best", "Status"]
MATRIX = ["Model", "Precision", "Engine", "Device", "Runs", "tok/s", "TTFT", "Peak memory", "Accuracy by dataset"]


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


def _item(text: Any, *, align_right: bool = False, color: str | None = None, data: Any = None) -> QTableWidgetItem:
    item = QTableWidgetItem("—" if text in (None, "") else str(text))
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color:
        item.setForeground(QColor(color))
    if data is not None:
        item.setData(Qt.ItemDataRole.UserRole, data)
    return item


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def _tps(value: float | None) -> str:
    return "—" if not value else f"{value:.1f}"


class EvaluatePage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.records: list[Any] = []
        self.dataset: evaluation.EvalDataset | None = None
        self.custom_datasets: dict[str, str] = {}  # label -> file path
        self.experiment: dict[str, Any] | None = None
        self._handle: Any = None
        self._live: dict[int, list[dict[str, Any]]] = {}
        self._run: dict[str, Any] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Evaluation Lab", "h2"))
        titles.addWidget(label("Run a dataset against several local models. Every case is scored on this machine, "
                               "and each run is saved with the dataset digest, model revisions and hardware so it can "
                               "be reproduced.", "muted", wrap=True))
        root.addLayout(titles)
        self.banner = Banner("danger", "")
        self.banner.hide()
        root.addWidget(self.banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # ---------------------------------------------------------- setup
        setup = Panel()
        setup.setMinimumWidth(320)
        setup.setMaximumWidth(400)
        side = QVBoxLayout(setup)
        side.setContentsMargins(16, 14, 16, 14)
        side.setSpacing(10)
        side.addWidget(label("MODELS", "eyebrow"))
        self.models = QListWidget()
        self.models.setMinimumHeight(140)
        self.models.itemChanged.connect(lambda _i: self._update_run_state())
        side.addWidget(self.models, 1)
        self.models_hint = label("", "faint", wrap=True)
        side.addWidget(self.models_hint)

        side.addWidget(label("DATASET", "eyebrow"))
        row = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset)
        row.addWidget(self.dataset_combo, 1)
        import_btn = button("", "upload", "ghost", tooltip="Import a JSONL dataset")
        import_btn.clicked.connect(self._import_dataset)
        row.addWidget(import_btn)
        side.addLayout(row)
        self.dataset_info = label("", "faint", wrap=True)
        side.addWidget(self.dataset_info)

        side.addWidget(label("GENERATION", "eyebrow"))
        form = QFormLayout()
        form.setVerticalSpacing(8)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(8, 4096)
        self.max_tokens.setValue(256)
        form.addRow("Max new tokens", self.max_tokens)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.0)
        self.temperature.setToolTip("0 = greedy decoding: the same model and dataset give the same answers.")
        form.addRow("Temperature", self.temperature)
        self.system_prompt = QLineEdit()
        self.system_prompt.setPlaceholderText("Optional system prompt")
        form.addRow("System", self.system_prompt)
        side.addLayout(form)

        self.run_btn = button("Run evaluation", "play", "primary")
        self.run_btn.clicked.connect(self._run_or_stop)
        side.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.hide()
        side.addWidget(self.progress)
        self.status = label("", "faint", wrap=True)
        side.addWidget(self.status)
        splitter.addWidget(setup)

        # ---------------------------------------------------------- results
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        results = QWidget()
        res = QVBoxLayout(results)
        res.setContentsMargins(0, 10, 0, 0)
        res.setSpacing(8)
        head = QHBoxLayout()
        self.result_title = label("No results yet", "h3")
        head.addWidget(self.result_title, 1)
        self.export_json_btn = button("JSON", "download", "ghost", tooltip="Export the experiment as JSON")
        self.export_json_btn.clicked.connect(lambda: self._export("json"))
        self.export_md_btn = button("Markdown", "download", "ghost", tooltip="Export a Markdown report")
        self.export_md_btn.clicked.connect(lambda: self._export("md"))
        for widget in (self.export_json_btn, self.export_md_btn):
            widget.setEnabled(False)
            head.addWidget(widget)
        res.addLayout(head)
        self.result_meta = label("", "faint", wrap=True)
        res.addWidget(self.result_meta)
        self.leaderboard = _table(LEADERBOARD, 0)
        self.leaderboard.itemSelectionChanged.connect(self._render_cases)
        res.addWidget(self.leaderboard, 2)
        res.addWidget(label("CASES", "eyebrow"))
        self.cases = _table(CASES, 5)
        self.cases.setWordWrap(False)
        res.addWidget(self.cases, 3)
        self.tabs.addTab(results, "Results")

        history = QWidget()
        hist = QVBoxLayout(history)
        hist.setContentsMargins(0, 10, 0, 0)
        hist_head = QHBoxLayout()
        hist_head.addWidget(label("Saved experiments (Evaluation Lab, CLI and Compare runs)", "muted"), 1)
        open_btn = button("Open", "eye", "ghost")
        open_btn.clicked.connect(self._open_selected)
        delete_btn = button("Delete", "trash", "ghost")
        delete_btn.clicked.connect(self._delete_selected)
        refresh_btn = button("", "refresh", "ghost", tooltip="Refresh")
        refresh_btn.clicked.connect(self.refresh_history)
        for widget in (open_btn, delete_btn, refresh_btn):
            hist_head.addWidget(widget)
        hist.addLayout(hist_head)
        self.history = _table(HISTORY, 1)
        self.history.doubleClicked.connect(lambda _i: self._open_selected())
        hist.addWidget(self.history)
        self.tabs.addTab(history, "History")

        matrix = QWidget()
        mat = QVBoxLayout(matrix)
        mat.setContentsMargins(0, 10, 0, 0)
        mat.addWidget(label("Measured on this machine — medians over every finished run. The Workspace analysis "
                            "uses these numbers instead of estimates.", "muted", wrap=True))
        self.matrix = _table(MATRIX, 8)
        mat.addWidget(self.matrix)
        self.tabs.addTab(matrix, "Performance matrix")
        self.tabs.currentChanged.connect(self._on_tab)

        self._load_datasets()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.refresh_models()

    def on_activated(self) -> None:
        self.refresh_models()
        if self.tabs.currentIndex() == 1:
            self.refresh_history()
        elif self.tabs.currentIndex() == 2:
            self.refresh_matrix()

    def on_library_changed(self) -> None:
        self.refresh_models()

    def shutdown(self) -> None:
        if self._handle is not None:
            self._handle.cancel()

    def _on_tab(self, index: int) -> None:
        if index == 1:
            self.refresh_history()
        elif index == 2:
            self.refresh_matrix()

    # ---------------------------------------------------------------- models
    def refresh_models(self) -> None:
        checked = {self.models.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.models.count())
                   if self.models.item(i).checkState() == Qt.CheckState.Checked}
        self.records = [r for r in self.ctx.registry.list_models(only_existing=True)
                        if (r.family or tasks.TEXT) == tasks.TEXT]
        self.models.blockSignals(True)
        self.models.clear()
        for record in self.records:
            item = QListWidgetItem(f"{record.display_name}  ·  {record.precision}")
            item.setToolTip(record.api_name)
            item.setData(Qt.ItemDataRole.UserRole, record.api_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if record.api_name in checked else Qt.CheckState.Unchecked)
            self.models.addItem(item)
        self.models.blockSignals(False)
        self.models_hint.setText("" if self.records else "Download a text-generation model on Discover first.")
        self._update_run_state()

    def selected_records(self) -> list[Any]:
        names = [self.models.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.models.count())
                 if self.models.item(i).checkState() == Qt.CheckState.Checked]
        by_name = {r.api_name: r for r in self.records}
        return [by_name[n] for n in names if n in by_name]

    def set_checked(self, api_names: list[str]) -> None:
        for i in range(self.models.count()):
            item = self.models.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in api_names
                               else Qt.CheckState.Unchecked)

    # -------------------------------------------------------------- datasets
    def _load_datasets(self) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for meta in evaluation.builtin_catalog():
            self.dataset_combo.addItem(f"{meta['title']}  ({meta['cases']})", meta["source"])
        for text, path in self.custom_datasets.items():
            self.dataset_combo.addItem(text, path)
        self.dataset_combo.blockSignals(False)
        if self.dataset_combo.count():
            self._on_dataset(self.dataset_combo.currentIndex())

    def select_dataset(self, ref: str) -> bool:
        index = self.dataset_combo.findData(ref)
        if index >= 0:
            self.dataset_combo.setCurrentIndex(index)
        return index >= 0

    def _on_dataset(self, index: int) -> None:
        ref = self.dataset_combo.itemData(index)
        if not ref:
            return
        try:
            self.dataset = evaluation.load_dataset(ref)
        except (OSError, evaluation.DatasetError) as exc:
            self.dataset = None
            self.dataset_info.setText(f"<span style='color:{theme.DANGER}'>{exc}</span>")
        else:
            scorers = sorted({c.scorer for c in self.dataset.cases})
            self.dataset_info.setText(f"{self.dataset.description}<br>{len(self.dataset.cases)} cases · scored by "
                                      f"{', '.join(scorers)} · {self.dataset.category}")
        self._update_run_state()

    def _import_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import evaluation dataset", str(Path.home()),
                                              "JSON Lines (*.jsonl *.json);;All files (*)")
        if path:
            self.add_dataset_file(path)

    def add_dataset_file(self, path: str) -> bool:
        try:
            dataset = evaluation.load_dataset(path)
        except (OSError, evaluation.DatasetError) as exc:
            QMessageBox.warning(self, "Dataset not loaded", str(exc))
            return False
        text = f"{dataset.title}  ({len(dataset.cases)}) · file"
        self.custom_datasets[text] = path
        self.dataset_combo.addItem(text, path)
        self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)
        return True

    # ------------------------------------------------------------------- run
    def _update_run_state(self) -> None:
        running = self._handle is not None
        self.run_btn.setEnabled(running or (bool(self.selected_records()) and self.dataset is not None))

    def _run_or_stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self.run_btn.setEnabled(False)
            self.status.setText("Stopping after the current case…")
            return
        records = self.selected_records()
        if not records or self.dataset is None:
            return
        models = [{**r.worker_payload(self.ctx.settings), "task": r.pipeline_tag or "text-generation"}
                  for r in records]
        params = {"max_new_tokens": self.max_tokens.value(), "temperature": self.temperature.value()}
        payload = {"models": models, "cases": [c.to_dict() for c in self.dataset.cases], "params": params,
                   "system_prompt": self.system_prompt.text().strip() or None}
        self._run = {"records": records, "models": models, "params": params, "dataset": self.dataset,
                     "started": time.time(),
                     "hardware": experiments.hardware_snapshot(self.ctx.hardware_profile(),
                                                               getattr(self.ctx.runtime, "last_heartbeat_payload", None))}
        self._live = {}
        self.experiment = None
        self.banner.hide()
        self._render_live()
        self.progress.setRange(0, len(models) * len(self.dataset.cases))
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Queued on the model runtime…")
        handle = self.ctx.runtime.request(Command.RUN_EVAL, payload)
        handle.event.connect(self._on_event)
        handle.finished.connect(self._on_finished)
        handle.failed.connect(self._on_failed)
        self._handle = handle
        self.run_btn.setText("Stop")
        theme.set_prop(self.run_btn, "variant", "danger")
        self.tabs.setCurrentIndex(0)

    def _on_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == Event.QUEUE_PLAN:
            mode = "all models stay loaded" if payload.get("mode") == "retain" else \
                "models are swapped one at a time to fit memory"
            self.status.setText(f"Plan: {mode}.")
        elif event == Event.STATUS and payload.get("message"):
            self.status.setText(payload["message"])
        elif event == Event.PROGRESS:
            self.progress.setMaximum(int(payload.get("total") or self.progress.maximum()))
            self.progress.setValue(int(payload.get("current") or 0))
            case = payload.get("case")
            if case:
                self._live.setdefault(int(case.get("view") or 1), []).append(case)
                self.status.setText(f"{payload.get('model_id')} · {case.get('case_id')}")
                self._render_live()

    def _on_finished(self, result: dict[str, Any]) -> None:
        run = self._run or {}
        self._reset_run_button()
        try:
            experiment_id = experiments.record_evaluation(
                self.ctx.registry, result, dataset=run["dataset"].meta(), params=run["params"],
                hardware=run["hardware"], models=run["models"], records=run["records"])
            self.show_experiment(self.ctx.registry.get_experiment(experiment_id))
        except Exception as exc:  # storage problems must not lose the on-screen results
            self.banner.set_text(f"<b>Results could not be saved:</b> {exc}")
            self.banner.show()
        self.status.setText(f"Finished in {(result.get('total_ms') or 0) / 1000:.1f} s.")
        self.ctx.toast("Evaluation finished", "success")

    def _on_failed(self, payload: dict[str, Any]) -> None:
        self._reset_run_button()
        if payload.get("code") == "CANCELLED":
            self.status.setText("Cancelled.")
            return
        hint = f" {payload['hint']}" if payload.get("hint") else ""
        self.banner.set_text(f"<b>Evaluation failed:</b> {payload.get('message')}{hint}")
        self.banner.show()
        self.status.setText("")

    def _reset_run_button(self) -> None:
        self._handle = None
        self.progress.hide()
        self.run_btn.setText("Run evaluation")
        theme.set_prop(self.run_btn, "variant", "primary")
        self._update_run_state()

    # --------------------------------------------------------------- render
    def _render_live(self) -> None:
        run = self._run or {}
        records = run.get("records") or []
        self.result_title.setText(f"{run['dataset'].title} — running" if run.get("dataset") else "Running")
        self.result_meta.setText("")
        self.leaderboard.setRowCount(0)
        for view, record in enumerate(records, start=1):
            rows = self._live.get(view, [])
            summary = evaluation.summarize(rows)
            row = self.leaderboard.rowCount()
            self.leaderboard.insertRow(row)
            values = [record.api_name, record.precision, "", _pct(summary["accuracy"]),
                      "—" if summary["mean_score"] is None else f"{summary['mean_score']:.2f}",
                      _ms(summary["ttft_ms_p50"]), _tps(summary["tokens_per_second"]), "", "",
                      f"{len(rows)}/{len(run['dataset'].cases)}"]
            for col, value in enumerate(values):
                self.leaderboard.setItem(row, col, _item(value, align_right=col in (3, 4, 5, 6), data=view))
        self._render_cases()

    def show_experiment(self, experiment: dict[str, Any] | None) -> None:
        self.experiment = experiment
        self._run = None
        if experiment is None:
            return
        dataset = experiment.get("dataset") or {}
        hardware = experiment.get("hardware") or {}
        params = experiment.get("params") or {}
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(experiment.get("created_at") or time.time()))
        self.result_title.setText(experiment.get("name") or "Experiment")
        self.result_meta.setText(
            f"{created} · {dataset.get('cases')} cases · dataset sha256 {str(dataset.get('digest') or '')[:12]} · "
            f"{hardware.get('name') or hardware.get('device') or ''} · temperature {params.get('temperature')} · "
            f"max {params.get('max_new_tokens')} tokens · experiment {experiment.get('id')}")
        self.leaderboard.setRowCount(0)
        runs = sorted(experiment.get("runs") or [], key=lambda r: -((r.get("summary") or {}).get("accuracy") or -1))
        for run in runs:
            summary, metrics = run.get("summary") or {}, run.get("metrics") or {}
            row = self.leaderboard.rowCount()
            self.leaderboard.insertRow(row)
            status = run.get("status")
            color = theme.SUCCESS if status == "finished" else theme.DANGER if status == "error" else None
            values = [run.get("api_name") or run.get("model_id"), run.get("precision"), metrics.get("engine"),
                      _pct(summary.get("accuracy")),
                      "—" if summary.get("mean_score") is None else f"{summary['mean_score']:.2f}",
                      _ms(summary.get("ttft_ms_p50")), _tps(summary.get("tokens_per_second")),
                      format_bytes(metrics.get("peak_bytes")) if metrics.get("peak_bytes") else "—",
                      _ms(metrics.get("load_ms")), status]
            for col, value in enumerate(values):
                item = _item(value, align_right=col in (3, 4, 5, 6, 7, 8), data=run.get("id"),
                             color=color if col == 9 else None)
                if col == 9 and run.get("error"):
                    item.setToolTip((run["error"] or {}).get("message", ""))
                self.leaderboard.setItem(row, col, item)
        if self.leaderboard.rowCount():
            self.leaderboard.selectRow(0)
        for widget in (self.export_json_btn, self.export_md_btn):
            widget.setEnabled(True)
        self.tabs.setCurrentIndex(0)

    def _render_cases(self) -> None:
        self.cases.setRowCount(0)
        rows = self.leaderboard.selectionModel().selectedRows() if self.leaderboard.selectionModel() else []
        if not rows:
            return
        key = self.leaderboard.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if self.experiment is not None:
            run = next((r for r in self.experiment.get("runs") or [] if r.get("id") == key), None)
            cases = (run or {}).get("cases") or []
        else:
            cases = self._live.get(int(key or 0), [])
        for case in cases:
            row = self.cases.rowCount()
            self.cases.insertRow(row)
            passed, error = case.get("passed"), case.get("error")
            verdict = "error" if error else "—" if passed is None else ("pass" if passed else "fail")
            color = theme.DANGER if error or passed is False else theme.SUCCESS if passed else theme.TEXT_MUTED
            output = (case.get("output") or "").replace("\n", " ⏎ ")
            values = [case.get("case_id"), verdict, "—" if case.get("score") is None else f"{case['score']:.2f}",
                      _ms(case.get("ttft_ms")), _tps(case.get("tokens_per_second")), output[:200],
                      error or case.get("detail")]
            for col, value in enumerate(values):
                item = _item(value, align_right=col in (2, 3, 4), color=color if col == 1 else None)
                if col == 5:
                    item.setToolTip((case.get("output") or "")[:2000])
                if col == 0 and case.get("prompt"):
                    item.setToolTip(case["prompt"])
                self.cases.setItem(row, col, item)

    # -------------------------------------------------------------- history
    def refresh_history(self) -> None:
        self.history.setRowCount(0)
        for experiment in self.ctx.registry.list_experiments():
            row = self.history.rowCount()
            self.history.insertRow(row)
            dataset = experiment.get("dataset") or {}
            summary = experiment.get("summary") or {}
            best = summary.get("best_accuracy")
            models = ", ".join((r.get("api_name") or r.get("model_id") or "").split("/")[-1]
                               for r in experiment.get("runs") or [])
            values = [time.strftime("%Y-%m-%d %H:%M", time.localtime(experiment.get("created_at") or 0)),
                      experiment.get("name"), experiment.get("kind"), dataset.get("title") or dataset.get("name"),
                      models, _pct(best) if best is not None else "—", experiment.get("status")]
            for col, value in enumerate(values):
                self.history.setItem(row, col, _item(value, data=experiment.get("id")))

    def _selected_history_id(self) -> str | None:
        rows = self.history.selectionModel().selectedRows()
        if not rows:
            return None
        return self.history.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def _open_selected(self) -> None:
        experiment_id = self._selected_history_id()
        if experiment_id:
            self.show_experiment(self.ctx.registry.get_experiment(experiment_id))

    def _delete_selected(self) -> None:
        experiment_id = self._selected_history_id()
        if not experiment_id:
            return
        if QMessageBox.question(self, "Delete experiment?", "Delete this experiment and its per-case results?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) \
                == QMessageBox.StandardButton.Yes:
            self.ctx.registry.delete_experiment(experiment_id)
            if self.experiment and self.experiment.get("id") == experiment_id:
                self.experiment = None
            self.refresh_history()

    # --------------------------------------------------------------- matrix
    def refresh_matrix(self) -> None:
        self.matrix.setRowCount(0)
        for entry in self.ctx.registry.performance_matrix():
            row = self.matrix.rowCount()
            self.matrix.insertRow(row)
            accuracy = ", ".join(f"{name} {_pct(value)}" for name, value in sorted(entry["accuracy"].items()))
            values = [entry["model_id"], entry["precision"], entry["engine"], entry["device"], entry["runs"],
                      _tps(entry["tokens_per_second"]), _ms(entry["ttft_ms"]),
                      format_bytes(entry["peak_memory_bytes"]) if entry["peak_memory_bytes"] else "—", accuracy]
            for col, value in enumerate(values):
                self.matrix.setItem(row, col, _item(value, align_right=col in (4, 5, 6, 7)))

    # --------------------------------------------------------------- export
    def _export(self, kind: str) -> None:
        if self.experiment is None:
            return
        folder = paths.exports_dir()
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "json" if kind == "json" else "md"
        default = folder / f"experiment-{self.experiment.get('id')}.{suffix}"
        path, _ = QFileDialog.getSaveFileName(self, "Export experiment", str(default),
                                              "JSON (*.json)" if kind == "json" else "Markdown (*.md)")
        if not path:
            return
        text = experiments.export_json(self.experiment) if kind == "json" else experiments.export_markdown(
            self.experiment)
        Path(path).write_text(text, encoding="utf-8")
        self.ctx.toast(f"Saved {Path(path).name}", "success")
