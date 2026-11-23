"""Page 3: comparative bench — 1/2/4 views, broadcast prompt, sequential GPU queue, benchmark matrix."""

from __future__ import annotations

import json
import platform
import time
import uuid
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle import __version__, paths
from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import Banner, Panel, button, label
from noelle.app.ui.widgets.comparison_pane import ComparisonPane
from noelle.app.ui.widgets.param_panel import ParamPanel
from noelle.runtime import tasks
from noelle.runtime.preflight import format_bytes
from noelle.runtime.protocol import Command, Event

LAYOUTS = {1: [(0, 0)], 2: [(0, 0), (0, 1)], 4: [(0, 0), (0, 1), (1, 0), (1, 1)]}
METRIC_ROWS = [
    ("Status", "status"),
    ("Time to first token", "ttft_ms"),
    ("Generation speed", "tokens_per_second"),
    ("Peak memory", "model_peak_bytes"),
    ("Precision", "precision"),
    ("Engine / device", "engine"),
    ("Load time", "load_ms"),
    ("Tokens generated", "completion_tokens"),
]


class ComparativePage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.view_count = 2
        self._handle: Any = None
        self._run: dict[str, Any] | None = None
        self.records: list[Any] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(12)

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Compare models", "h2"))
        titles.addWidget(label("One prompt, several local models. Views run in order on a single GPU lane and swap "
                               "weights when they don't all fit.", "muted", wrap=True))
        head.addLayout(titles, 1)
        self._layout_group = QButtonGroup(self)
        for count, text in ((1, "1 view"), (2, "2 views"), (4, "4 views")):
            btn = button(text, {1: "square", 2: "columns", 4: "grid"}[count], "segment", checkable=True)
            btn.clicked.connect(lambda _=False, c=count: self.set_view_count(c))
            self._layout_group.addButton(btn, count)
            head.addWidget(btn)
        self._layout_group.button(2).setChecked(True)
        head.addSpacing(10)
        self.export_btn = button("Export", "save", "ghost")
        menu = QMenu(self)
        menu.addAction("Markdown report…", lambda: self.export("md"))
        menu.addAction("JSON…", lambda: self.export("json"))
        self.export_btn.setMenu(menu)
        self.export_btn.setEnabled(False)
        head.addWidget(self.export_btn)
        self.run_btn = button("Run all", "play", "primary")
        self.run_btn.setMinimumWidth(110)
        self.run_btn.clicked.connect(self._run_or_stop)
        head.addWidget(self.run_btn)
        root.addLayout(head)

        prompt_panel = Panel()
        prompt_layout = QVBoxLayout(prompt_panel)
        prompt_layout.setContentsMargins(14, 12, 14, 12)
        prompt_layout.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(label("System", "muted"))
        self.system_prompt = QLineEdit(str(ctx.settings.get("generation.system_prompt", "")))
        top.addWidget(self.system_prompt, 1)
        self.broadcast = QCheckBox("Broadcast prompt and parameters to all views")
        self.broadcast.setChecked(True)
        self.broadcast.toggled.connect(self._on_broadcast)
        top.addWidget(self.broadcast)
        prompt_layout.addLayout(top)
        self.prompt = QPlainTextEdit("Explain what a KV cache is and why it matters for long prompts, in four sentences.")
        self.prompt.setFixedHeight(58)
        prompt_layout.addWidget(self.prompt)
        defaults = {k: ctx.settings.get(f"generation.{k}") for k in
                    ("temperature", "top_p", "top_k", "repetition_penalty")}
        defaults["max_new_tokens"] = 256
        defaults["temperature"] = 0.0
        self.params = ParamPanel(defaults, columns=5, show_seed=False)
        prompt_layout.addWidget(self.params)
        root.addWidget(prompt_panel)

        self.plan_banner = Banner("info", "")
        self.plan_banner.hide()
        root.addWidget(self.plan_banner)

        splitter = QSplitter(Qt.Orientation.Vertical)
        grid_host = QWidget()
        self.grid = QGridLayout(grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        self.panes = [ComparisonPane(i) for i in range(1, 5)]
        splitter.addWidget(grid_host)
        table_host = QWidget()
        table_layout = QVBoxLayout(table_host)
        table_layout.setContentsMargins(0, 6, 0, 0)
        table_layout.addWidget(label("BENCHMARK", "eyebrow"))
        self.table = QTableWidget(len(METRIC_ROWS), 2)
        self.table.setVerticalHeaderLabels([name for name, _ in METRIC_ROWS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_host)
        splitter.setSizes([520, 230])
        root.addWidget(splitter, 1)
        self.set_view_count(2)
        ctx.downloads.job_finished.connect(lambda *_: self.refresh_models())

    # ---------------------------------------------------------------- layout
    def on_activated(self) -> None:
        self.refresh_models()

    def start(self) -> None:
        self.refresh_models()

    def refresh_models(self) -> None:
        self.records = [r for r in self.ctx.registry.list_models(only_existing=True)
                        if tasks.task_family(r.pipeline_tag) == tasks.TEXT]
        taken: set[str] = set()
        for pane in self.panes:
            pane.set_models(self.records)
            if pane.selected_key() is None and self.records:
                # default to a model no earlier view uses, falling back to the first one
                choice = next((i for i, r in enumerate(self.records)
                               if f"{r.repo_id}|{r.precision}" not in taken), 0)
                pane.select_index(choice + 1)
            if pane.selected_key():
                taken.add(pane.selected_key())

    def on_library_changed(self) -> None:
        self.refresh_models()

    def set_view_count(self, count: int) -> None:
        self.view_count = count
        for pane in self.panes:
            self.grid.removeWidget(pane)
            pane.hide()
        for pane, (row, col) in zip(self.panes, LAYOUTS[count]):
            self.grid.addWidget(pane, row, col)
            pane.show()
        self._layout_group.button(count).setChecked(True)
        self._reset_table()

    def active_panes(self) -> list[ComparisonPane]:
        return self.panes[: self.view_count]

    def _on_broadcast(self, broadcast: bool) -> None:
        for pane in self.panes:
            pane.set_broadcast(broadcast)

    # ------------------------------------------------------------------- run
    def _record_for(self, key: str) -> Any:
        repo_id, precision = key.rsplit("|", 1)
        return next((r for r in self.records if r.repo_id == repo_id and r.precision == precision), None)

    def _run_or_stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self.run_btn.setEnabled(False)
            return
        views = []
        prompt = self.prompt.toPlainText().strip()
        for pane in self.active_panes():
            key = pane.selected_key()
            pane.reset()
            if not key:
                continue
            record = self._record_for(key)
            if record is None:
                continue
            entry = {**record.worker_payload(self.ctx.settings), "view": pane.view}
            if not self.broadcast.isChecked():
                entry["prompt"] = pane.prompt.text().strip() or prompt
            views.append(entry)
        if not views:
            self.ctx.toast("Pick a downloaded text model in at least one view.", "info")
            return
        if self.broadcast.isChecked() and not prompt:
            self.ctx.toast("Write a prompt first.", "info")
            return
        payload = {"views": views, "prompt": prompt, "system_prompt": self.system_prompt.text().strip() or None,
                   "params": self.params.values()}
        self._run = {"run_id": uuid.uuid4().hex[:12], "started": time.time(), "payload": payload, "results": {},
                     "plan": None}
        self._reset_table()
        self.export_btn.setEnabled(False)
        handle = self.ctx.runtime.request(Command.RUN_BENCHMARK, payload)
        handle.event.connect(self._on_event)
        handle.finished.connect(self._on_done)
        handle.failed.connect(self._on_failed)
        self._handle = handle
        self.run_btn.setText("Stop")
        theme.set_prop(self.run_btn, "variant", "danger")

    def _pane(self, view: Any) -> ComparisonPane | None:
        try:
            return self.panes[int(view) - 1]
        except (TypeError, ValueError, IndexError):
            return None

    def _on_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == Event.QUEUE_PLAN:
            self._run["plan"] = payload
            mode = payload.get("mode")
            total, usable = payload.get("total_bytes", 0), payload.get("usable_bytes", 0)
            if mode == "retain":
                self.plan_banner.set_text(f"<b>Retain mode</b> — all selected models fit together "
                                          f"({format_bytes(total)} of {format_bytes(usable)} usable on "
                                          f"{payload.get('device')}). They stay loaded; views run one after another.")
            else:
                self.plan_banner.set_text(f"<b>Hot-swap mode</b> — the selected models need {format_bytes(total)}, more "
                                          f"than the {format_bytes(usable)} usable. Each view loads, runs, then frees "
                                          "memory (gc + cache flush) before the next one.")
            self.plan_banner.show()
        elif event == Event.VIEW_STARTED:
            pane = self._pane(payload.get("view"))
            if pane:
                pane.set_status("executing" if payload.get("reuse_loaded") else "loading",
                                "Executing" if payload.get("reuse_loaded") else "Loading")
        elif event == Event.STATUS:
            pane = self._pane(payload.get("view"))
            if pane and payload.get("stage") == "load":
                pane.set_status("loading", "Loading")
                pane.footer.setText(payload.get("message", ""))
            elif pane and payload.get("stage") == "evict":
                pane.footer.setText(payload.get("message", ""))
        elif event == Event.TOKEN_STREAM:
            pane = self._pane(payload.get("view"))
            if pane:
                if pane.status != "executing":
                    pane.set_status("executing", "Executing")
                pane.append(payload.get("text", ""))
        elif event == Event.VIEW_COMPLETE:
            pane = self._pane(payload.get("view"))
            if pane:
                pane.finish(payload)
            self._run["results"][int(payload.get("view", 0))] = payload
            self._update_table()

    def _on_done(self, summary: dict[str, Any]) -> None:
        self._finish_run()
        run = self._run or {}
        for view, result in sorted(run.get("results", {}).items()):
            pane = self._pane(view)
            self.ctx.registry.add_evaluation(
                run["run_id"], view, result.get("model_id", ""), prompt=(pane.prompt.text() if pane and
                                                                        not self.broadcast.isChecked() else
                                                                        run["payload"]["prompt"]),
                system_prompt=run["payload"].get("system_prompt"), params=run["payload"]["params"],
                output=result.get("text", ""), status=result.get("status", ""), metrics=result.get("metrics") or {})
        self.export_btn.setEnabled(bool(run.get("results")))
        self._record_experiment(run)
        self.ctx.toast(f"Benchmark finished in {summary.get('total_ms', 0) / 1000:.1f} s", "success")

    def _record_experiment(self, run: dict[str, Any]) -> None:
        """Also keep the run in the Evaluation Lab history (with model provenance and hardware)."""
        from noelle.app.services import experiments

        try:
            records = {}
            for view, result in run.get("results", {}).items():
                key = f"{result.get('model_id')}|{(result.get('metrics') or {}).get('precision') or 'auto'}"
                records[int(view)] = self._record_for(key)
            payload = run.get("payload") or {}
            experiments.record_compare(
                self.ctx.registry, prompt=payload.get("prompt") or "", system_prompt=payload.get("system_prompt"),
                params=payload.get("params") or {}, results=list(run.get("results", {}).values()), records=records,
                hardware=experiments.hardware_snapshot(self.ctx.hardware_profile(),
                                                       getattr(self.ctx.runtime, "last_heartbeat_payload", None)))
        except Exception:  # history is a convenience; never fail the bench for it
            import logging

            logging.getLogger("noelle.ui").exception("could not record the compare run as an experiment")

    def _on_failed(self, payload: dict[str, Any]) -> None:
        self._finish_run()
        for pane in self.active_panes():
            if pane.status in ("queued", "loading", "executing"):
                pane.set_status("cancelled" if payload.get("code") == "CANCELLED" else "error")
        if payload.get("code") != "CANCELLED":
            self.plan_banner.set_text(f"<b>Benchmark failed:</b> {payload.get('message')}")
            self.plan_banner.show()
        self.export_btn.setEnabled(bool(self._run and self._run.get("results")))

    def _finish_run(self) -> None:
        self._handle = None
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run all")
        theme.set_prop(self.run_btn, "variant", "primary")

    # ----------------------------------------------------------------- table
    def _reset_table(self) -> None:
        panes = self.active_panes()
        self.table.setColumnCount(len(panes))
        self.table.setHorizontalHeaderLabels([f"View {p.view}" for p in panes])
        for row in range(self.table.rowCount()):
            for col in range(len(panes)):
                self.table.setItem(row, col, QTableWidgetItem("—"))

    def _update_table(self) -> None:
        results = (self._run or {}).get("results", {})
        panes = self.active_panes()
        headers = []
        for pane in panes:
            result = results.get(pane.view)
            name = (result or {}).get("model_id") or (pane.model.currentText() if pane.selected_key() else "")
            headers.append(f"View {pane.view}" + (f" · {name.split('/')[-1]}" if name else ""))
        self.table.setHorizontalHeaderLabels(headers)
        best: dict[str, tuple[int, float]] = {}
        for col, pane in enumerate(panes):
            result = results.get(pane.view)
            if not result:
                continue
            metrics = result.get("metrics") or {}
            for row, (_name, key) in enumerate(METRIC_ROWS):
                value = result.get("status") if key == "status" else metrics.get(key)
                self.table.setItem(row, col, QTableWidgetItem(self._format(key, value, metrics)))
                if isinstance(value, (int, float)) and result.get("status") == "finished":
                    if key not in best:
                        best[key] = (col, float(value))
                    else:
                        higher_is_better = key == "tokens_per_second"
                        current = best[key][1]
                        if (value > current) if higher_is_better else (value < current):
                            best[key] = (col, float(value))
        tags = {"ttft_ms": "fastest", "tokens_per_second": "fastest", "model_peak_bytes": "lightest"}
        finished = sum(1 for r in results.values() if r.get("status") == "finished")
        if finished > 1:
            for key, tag in tags.items():
                if key in best:
                    row = [k for _n, k in METRIC_ROWS].index(key)
                    item = self.table.item(row, best[key][0])
                    if item is not None:
                        item.setText(f"{item.text()}  ({tag})")
                        item.setForeground(QColor(theme.SUCCESS))

    @staticmethod
    def _format(key: str, value: Any, metrics: dict[str, Any]) -> str:
        if value is None:
            return "—"
        if key == "ttft_ms" or key == "load_ms":
            return f"{value:.0f} ms"
        if key == "tokens_per_second":
            return f"{value:.1f} tok/s"
        if key == "model_peak_bytes":
            return format_bytes(value)
        if key == "engine":
            return f"{value} / {metrics.get('device', '?')}"
        if key == "status":
            return str(value).capitalize()
        return str(value)

    # ---------------------------------------------------------------- export
    def report(self) -> dict[str, Any]:
        run = self._run or {}
        metrics = self.ctx.hardware.latest
        budget = self.ctx.hardware.memory_budget()
        return {
            "run_id": run.get("run_id"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run.get("started", time.time()))),
            "noelle_version": __version__,
            "hardware": {"gpu": metrics.gpu.name if metrics else None, "platform": platform.platform(),
                         "memory_budget_bytes": budget.total_bytes, "usable_bytes": budget.usable_bytes},
            "prompt": run.get("payload", {}).get("prompt"),
            "system_prompt": run.get("payload", {}).get("system_prompt"),
            "params": run.get("payload", {}).get("params"),
            "broadcast": self.broadcast.isChecked(),
            "queue_plan": run.get("plan"),
            "views": [run["results"][v] for v in sorted(run.get("results", {}))],
        }

    def export(self, kind: str) -> None:
        report = self.report()
        if not report.get("views"):
            return
        default = paths.exports_dir() / f"benchmark-{report['run_id']}.{kind}"
        path, _ = QFileDialog.getSaveFileName(self, "Export benchmark", str(default),
                                              "Markdown (*.md)" if kind == "md" else "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2, ensure_ascii=False) if kind == "json" else markdown_report(report))
        self.ctx.toast(f"Saved {path}", "success")

    def shutdown(self) -> None:
        if self._handle is not None:
            self._handle.cancel()


def markdown_report(report: dict[str, Any]) -> str:
    views = report.get("views") or []
    lines = [f"# Noelle benchmark {report.get('run_id')}", "",
             f"- Date: {report.get('created_at')}", f"- Hardware: {report['hardware'].get('gpu')} "
             f"({format_bytes(report['hardware'].get('usable_bytes'))} usable)",
             f"- Queue mode: {(report.get('queue_plan') or {}).get('mode', '?')}", "",
             "## Prompt", "", f"> {report.get('prompt')}", ""]
    if report.get("system_prompt"):
        lines += [f"System: *{report['system_prompt']}*", ""]
    lines += ["## Results", "", "| Metric | " + " | ".join(f"View {v['view']} ({v.get('model_id', '').split('/')[-1]})"
                                                          for v in views) + " |",
              "|---|" + "---|" * len(views)]
    for name, key in METRIC_ROWS:
        cells = []
        for v in views:
            metrics = v.get("metrics") or {}
            value = v.get("status") if key == "status" else metrics.get(key)
            cells.append(ComparativePage._format(key, value, metrics))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Outputs", ""]
    for v in views:
        lines += [f"### View {v['view']} — {v.get('model_id')}", "", v.get("text") or f"*{v.get('status')}*", ""]
    lines += ["## Parameters", "", "```json", json.dumps(report.get("params"), indent=2), "```", ""]
    return "\n".join(lines)
