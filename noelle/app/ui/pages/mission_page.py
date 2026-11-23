"""Mission Mode: describe a goal, get a local stack, then run the plan step by step with your approval."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app.context import AppContext
from noelle.app.services import experiments, mission
from noelle.app.services.model_registry import ModelRecord
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import Banner, FlowLayout, Panel, Pill, Spinner, button, clear_layout, label
from noelle.runtime import advisor, evaluation
from noelle.runtime.preflight import format_bytes
from noelle.runtime.protocol import Command

EXAMPLES = ("A local Indonesian legal assistant for my PDF contracts", "A fast coding assistant for Python",
            "Transcribe my meeting recordings", "Chat with my notes, English only, low memory",
            "I need Indonesian document QA with 16 GB RAM and an RTX 3060")
STATUS_LEVEL = {"pending": "unknown", "running": "info", "done": "green", "skipped": "unknown", "failed": "red",
                "manual": "yellow"}
STATUS_TEXT = {"pending": "To do", "running": "Running", "done": "Done", "skipped": "Skipped", "failed": "Failed",
               "manual": "Manual"}


class MissionPage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.plan: mission.MissionPlan | None = None
        self._queue: list[mission.Step] = []
        self._current: mission.Step | None = None
        self._handle: Any = None
        self._planning = False

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(14)

        root.addWidget(label("MISSION", "eyebrow"))
        root.addWidget(label("What do you want to build?", "h1"))
        root.addWidget(label("Describe the goal in your own words — the task, languages, and anything about your "
                             "hardware. Noelle picks models that fit, then downloads, converts, indexes, tests and "
                             "serves them one step at a time, asking before anything is downloaded or started.",
                             "muted", wrap=True))
        row = QHBoxLayout()
        self.goal = QLineEdit()
        self.goal.setPlaceholderText("e.g. A local Indonesian legal assistant for my PDF contracts")
        self.goal.setMinimumHeight(40)
        self.goal.returnPressed.connect(self.start_planning)
        row.addWidget(self.goal, 1)
        self.plan_btn = button("Plan mission", "sparkles", "primary")
        self.plan_btn.setMinimumHeight(40)
        self.plan_btn.clicked.connect(self.start_planning)
        row.addWidget(self.plan_btn)
        root.addLayout(row)
        chips_host = QWidget()
        chips = FlowLayout(chips_host, spacing=6)
        for example in EXAMPLES:
            chip = button(example, None, "chip")
            chip.clicked.connect(lambda _=False, text=example: (self.goal.setText(text), self.start_planning()))
            chips.addWidget(chip)
        root.addWidget(chips_host)
        status_row = QHBoxLayout()
        self.spinner = Spinner(16)
        status_row.addWidget(self.spinner)
        self.status = label("", "faint", wrap=True)
        status_row.addWidget(self.status, 1)
        root.addLayout(status_row)
        self.banner = Banner("warning", "")
        self.banner.hide()
        root.addWidget(self.banner)

        self.plan_host = QWidget()
        self.plan_layout = QVBoxLayout(self.plan_host)
        self.plan_layout.setContentsMargins(0, 0, 0, 0)
        self.plan_layout.setSpacing(14)
        root.addWidget(self.plan_host)
        root.addStretch(1)
        ctx.downloads.job_finished.connect(self._on_download_finished)

    # ----------------------------------------------------------- planning
    def start_planning(self) -> None:
        text = self.goal.text().strip()
        if not text or self._planning:
            return
        self._planning = True
        self.plan_btn.setEnabled(False)
        self.spinner.start()
        self.status.setText("Reading your goal, checking this machine and searching the Hub…")
        self.banner.hide()
        hardware = self.ctx.hardware_profile()
        registry = self.ctx.registry
        planner = mission.Planner(search=mission.hub_search(self.ctx.hf), info=self.ctx.hf.model_info,
                                  hardware=hardware, local_models=registry.list_models(only_existing=True),
                                  measured=registry.measurements)
        runner().submit(planner.plan, text, on_result=self.set_plan, on_error=self._plan_failed,
                        on_finished=self._planning_done)

    def _planning_done(self) -> None:
        self._planning = False
        self.plan_btn.setEnabled(True)
        self.spinner.stop()

    def _plan_failed(self, message: str) -> None:
        self.status.setText("")
        self.banner.set_text(f"<b>Could not plan this mission:</b> {message}")
        self.banner.show()

    def set_plan(self, plan: mission.MissionPlan) -> None:
        self.plan = plan
        self._queue = []
        self._current = None
        self.status.setText(f"Plan ready — {plan.goal.title}.")
        self._render()

    # ------------------------------------------------------------ render
    def _render(self) -> None:
        clear_layout(self.plan_layout)
        plan = self.plan
        if plan is None:
            return
        summary = Panel()
        box = QVBoxLayout(summary)
        box.setContentsMargins(18, 16, 18, 16)
        box.setSpacing(6)
        box.addWidget(label("MISSION PLAN", "eyebrow"))
        box.addWidget(label(plan.goal.title, "h2", wrap=True))
        hw = plan.hardware_note or f"{plan.hardware.name} · {format_bytes(plan.hardware.usable_bytes)} usable"
        box.addWidget(label(f"Hardware: {hw} · profile {advisor.PROFILES.get(plan.goal.profile, plan.goal.profile)}"
                            + (f" · estimated memory ~{format_bytes(plan.total_memory)}" if plan.total_memory else ""),
                            "muted", wrap=True))
        for warning in plan.warnings:
            box.addWidget(label(f"⚠ {warning}", "faint", wrap=True))
        actions = QHBoxLayout()
        self.run_all_btn = button("Run all steps", "play", "primary")
        self.run_all_btn.clicked.connect(self.run_all)
        actions.addWidget(self.run_all_btn)
        export = button("Export plan", "download", "ghost")
        export.clicked.connect(self._export)
        actions.addWidget(export)
        actions.addStretch(1)
        box.addLayout(actions)
        self.plan_layout.addWidget(summary)

        self.plan_layout.addWidget(label("RECOMMENDED STACK", "eyebrow"))
        for component in plan.components:
            self.plan_layout.addWidget(self._component_card(component))
        if plan.goal.task == "rag":
            store = Panel()
            row = QHBoxLayout(store)
            row.setContentsMargins(16, 12, 16, 12)
            row.addWidget(label("Vector store", "muted"))
            row.addWidget(label("SQLite + sqlite-vec on this machine", "h3"), 1)
            self.plan_layout.addWidget(store)

        self.plan_layout.addWidget(label("STEPS", "eyebrow"))
        self.steps_host = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_host)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(8)
        self.plan_layout.addWidget(self.steps_host)
        self._render_steps()

    def _component_card(self, component: mission.Component) -> QWidget:
        card = QFrame()
        card.setProperty("card", True)
        col = QVBoxLayout(card)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label(component.title.upper(), "eyebrow"))
        head.addStretch(1)
        col.addLayout(head)
        candidate = component.candidate
        if candidate is None:
            col.addWidget(label(f"Nothing suitable found. {component.note}", "muted", wrap=True))
            return card
        title_row = QHBoxLayout()
        title_row.addWidget(label(candidate.model_id, "h3", wrap=True), 1)
        assessment = candidate.assessment
        title_row.addWidget(Pill(f"{assessment.score}/100 · {assessment.grade}", assessment.level))
        col.addLayout(title_row)
        rec = assessment.recommended
        if rec is not None:
            memory = f" · ~{format_bytes(rec.memory_bytes)}" if rec.memory_bytes else ""
            col.addWidget(label(f"Runs as {rec.label} with {rec.runtime}{memory} · {assessment.speed_text()}",
                                "muted", wrap=True))
        for reason in candidate.reasons[:4]:
            col.addWidget(label(f"• {reason}", "faint", wrap=True))
        if component.alternatives:
            alt_row = QHBoxLayout()
            alt_row.addWidget(label("Alternatives", "faint"))
            combo = QComboBox()
            combo.addItem("Keep this choice", None)
            for alt in component.alternatives:
                combo.addItem(f"{alt.model_id} · {alt.assessment.score}/100", alt.model_id)
            combo.currentIndexChanged.connect(lambda _i, c=combo, comp=component: self._swap(comp, c.currentData()))
            alt_row.addWidget(combo, 1)
            open_btn = button("Open in Workspace", None, "link")
            open_btn.clicked.connect(lambda _=False, s=candidate.summary: self.ctx.open_model(s))
            alt_row.addWidget(open_btn)
            col.addLayout(alt_row)
        return card

    def _swap(self, component: mission.Component, model_id: str | None) -> None:
        if not model_id or self.plan is None or self._current is not None:
            return
        chosen = next((a for a in component.alternatives if a.model_id == model_id), None)
        if chosen is None or component.candidate is None:
            return
        component.alternatives = [a for a in component.alternatives if a is not chosen] + [component.candidate]
        component.candidate = chosen
        self.plan.steps = mission.build_steps(self.plan.goal, self.plan.components, self.plan.eval_dataset)
        self._render()

    def _render_steps(self) -> None:
        if self.plan is None:
            return
        clear_layout(self.steps_layout)
        for index, step in enumerate(self.plan.steps, start=1):
            row = QFrame()
            row.setProperty("card", True)
            line = QHBoxLayout(row)
            line.setContentsMargins(14, 10, 14, 10)
            line.setSpacing(12)
            line.addWidget(label(str(index), "h3"))
            text = QVBoxLayout()
            text.setSpacing(2)
            text.addWidget(label(step.title, wrap=True))
            text.addWidget(label(step.message or step.detail, "faint", wrap=True))
            line.addLayout(text, 1)
            line.addWidget(Pill(STATUS_TEXT.get(step.status, step.status), STATUS_LEVEL.get(step.status, "unknown")))
            if step.status in ("pending", "failed", "manual"):
                go = button("Run" if step.status != "failed" else "Retry", "play", "ghost")
                go.setEnabled(self._current is None)
                go.clicked.connect(lambda _=False, s=step: self.run_steps([s]))
                line.addWidget(go)
            self.steps_layout.addWidget(row)
        if hasattr(self, "run_all_btn"):
            pending = any(s.status in ("pending", "failed") for s in self.plan.steps)
            self.run_all_btn.setEnabled(self._current is None and pending)

    # --------------------------------------------------------- execution
    def run_all(self) -> None:
        if self.plan is not None:
            self.run_steps([s for s in self.plan.steps if s.status in ("pending", "failed")])

    def run_steps(self, steps: list[mission.Step]) -> None:
        if self._current is not None:
            return
        self._queue = list(steps)
        self._advance()

    def _advance(self) -> None:
        self._current = None
        while self._queue:
            step = self._queue.pop(0)
            if step.approval and not self._approve(step):
                step.status, step.message = "skipped", "Skipped — you declined this step."
                self._queue = []  # later steps usually depend on this one
                break
            self._current = step
            step.status, step.message = "running", ""
            self._render_steps()
            try:
                getattr(self, f"_step_{step.key}")(step)
            except Exception as exc:  # a broken step must not leave the page stuck in "running"
                self._finish(step, False, str(exc))
            return
        self._render_steps()

    def _approve(self, step: mission.Step) -> bool:
        answer = QMessageBox.question(self, "Allow this step?", step.approval or step.title,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _finish(self, step: mission.Step, ok: bool, message: str = "") -> None:
        if step is not self._current:
            return
        step.status = "done" if ok else "failed"
        step.message = message
        if not ok:
            self._queue = []
        self._handle = None
        self._advance()

    def _component_for(self, step: mission.Step) -> mission.Candidate | None:
        component = self.plan.component(step.role or "llm") if self.plan else None
        return component.candidate if component else None

    # each _step_<key> starts work and later calls _finish(step, ok, message)
    def _step_download(self, step: mission.Step) -> None:
        candidate = self._component_for(step)
        if candidate is None:
            self._finish(step, False, "Nothing to download.")
            return
        if self.ctx.registry.is_downloaded(candidate.model_id):
            self._finish(step, True, "Already downloaded.")
            return
        rec = candidate.assessment.recommended
        include = None
        if rec is not None and rec.gguf_file:
            include = [rec.gguf_file, "*.json", "*.md", "tokenizer*"]
        self.ctx.downloads.start(candidate.model_id, summary=candidate.summary, include=include)
        step.message = "Downloading — progress is shown in the top bar."
        self._render_steps()

    def _on_download_finished(self, repo_id: str, ok: bool) -> None:
        step = self._current
        if step is None or step.key != "download":
            return
        candidate = self._component_for(step)
        if candidate is None or candidate.model_id != repo_id:
            return
        job = self.ctx.downloads.job(repo_id)
        self._finish(step, ok, "Downloaded." if ok else f"Download failed: {getattr(job, 'error', '') or 'see Storage'}")

    def _step_build(self, step: mission.Step) -> None:
        candidate = self._component_for(step)
        record = self.ctx.registry.get_model(candidate.model_id, source="hub") if candidate else None
        rec = candidate.assessment.recommended if candidate else None
        if record is None or rec is None:
            self._finish(step, False, "Download the model first.")
            return
        handle = self.ctx.runtime.request(Command.QUANTIZE, {**record.worker_payload(self.ctx.settings), "method": rec.precision})
        handle.status.connect(lambda p: self._progress(step, p.get("message", "")))
        handle.progress.connect(lambda p: self._progress(step, f"{p.get('stage', '')}: {p.get('message', '')}"))
        handle.finished.connect(lambda result: self._on_built(step, candidate, rec, result))
        handle.failed.connect(lambda p: self._finish(step, False, p.get("message", "Conversion failed.")))
        self._handle = handle

    def _on_built(self, step: mission.Step, candidate: mission.Candidate, rec: advisor.RunOption,
                  result: dict[str, Any]) -> None:
        summary = candidate.summary
        derived = ModelRecord(repo_id=result.get("derived_id") or f"local/{summary.name}-{rec.precision}",
                              pipeline_tag=summary.pipeline_tag, local_path=result.get("output_dir"),
                              precision=result.get("precision") or rec.precision, engine=result.get("engine"),
                              gguf_file=result.get("gguf_file"), source="quantized", base_model=summary.id,
                              size_bytes=result.get("size_bytes"), param_count=summary.param_count,
                              license=summary.license)
        self.ctx.registry.upsert_model(derived)
        self._finish(step, True, f"Built {derived.repo_id} ({format_bytes(result.get('size_bytes'))}).")

    def _step_install(self, step: mission.Step) -> None:
        candidate = self._component_for(step)
        rec = candidate.assessment.recommended if candidate else None
        if rec is not None and rec.runtime == "llama.cpp":
            from noelle.app.ui.component_dialog import ensure_llama_cpp

            installed = {"ok": False}
            ready = ensure_llama_cpp(self, lambda: installed.update(ok=True), reason=rec.reason)
            ok = ready or installed["ok"]
            self._finish(step, ok, "llama.cpp tools are installed." if ok else "Install them, then retry.")
            return
        QMessageBox.information(self, "Install a component", (rec.reason if rec else None) or step.detail)
        step.status, step.message = "manual", "Install the package, restart the runtime, then retry."
        self._current = None
        self._queue = []
        self._render_steps()

    def _step_knowledge(self, step: mission.Step) -> None:
        candidate = self._component_for(step)
        files, _ = QFileDialog.getOpenFileNames(self, "Documents for the knowledge base", str(Path.home()),
                                                "Documents (*.pdf *.docx *.md *.txt *.html *.csv);;All files (*)")
        if not files:
            self._finish(step, False, "No documents selected.")
            return
        slug = re.sub(r"[^a-z0-9]+", "-", (self.plan.goal.text if self.plan else "mission").lower()).strip("-")[:40]
        collection = f"mission-{slug or 'docs'}"
        payload: dict[str, Any] = {"collection": collection, "files": files}
        if candidate is not None:
            payload["embedding_model"] = candidate.model_id
            record = self.ctx.registry.get_model(candidate.model_id)
            if record is not None and record.local_path:
                payload["embedding_local_path"] = record.local_path
        handle = self.ctx.runtime.request(Command.RAG_INGEST, payload)
        handle.progress.connect(lambda p: self._progress(step, f"Indexing {p.get('file') or ''} "
                                                               f"{p.get('current', '')}/{p.get('total', '')}"))
        handle.finished.connect(lambda r: self._finish(step, True, f"Indexed {len(files)} file(s) into "
                                                                   f"'{collection}' ({r.get('chunks', '?')} chunks)."))
        handle.failed.connect(lambda p: self._finish(step, False, p.get("message", "Indexing failed.")))
        self._handle = handle

    def _llm_record(self) -> tuple[ModelRecord | None, dict[str, Any] | None]:
        component = self.plan.component("llm") if self.plan else None
        candidate = component.candidate if component else None
        if candidate is None:
            return None, None
        rec = candidate.assessment.recommended
        local = {r.precision: r for r in self.ctx.registry.list_models()
                 if r.base_model == candidate.model_id and r.source == "quantized" and r.exists_on_disk()}
        if rec is not None and rec.precision in local:
            record = local[rec.precision]
            return record, record.worker_payload(self.ctx.settings)
        record = self.ctx.registry.get_model(candidate.model_id, source="hub")
        if record is None:
            return None, None
        payload = record.worker_payload(self.ctx.settings)
        if rec is not None and rec.status == advisor.READY and rec.precision not in ("auto",) and \
                rec.runtime == "transformers" and rec.quality != advisor.REFERENCE:
            payload["precision"] = rec.precision  # load-time quantization (bitsandbytes)
        return record, payload

    def _step_test(self, step: mission.Step) -> None:
        record, payload = self._llm_record()
        if record is None or payload is None or not self.plan or not self.plan.eval_dataset:
            self._finish(step, False, "Download the language model first.")
            return
        dataset = evaluation.load_dataset(f"builtin:{self.plan.eval_dataset}")
        params = {"max_new_tokens": 192, "temperature": 0.0}
        request = {"models": [payload], "cases": [c.to_dict() for c in dataset.cases], "params": params}
        handle = self.ctx.runtime.request(Command.RUN_EVAL, request)
        handle.progress.connect(lambda p: self._progress(step, f"Case {p.get('current')}/{p.get('total')}"))
        handle.finished.connect(lambda result: self._on_tested(step, result, dataset, params, [payload], [record]))
        handle.failed.connect(lambda p: self._finish(step, False, p.get("message", "Evaluation failed.")))
        self._handle = handle

    def _on_tested(self, step: mission.Step, result: dict[str, Any], dataset: evaluation.EvalDataset,
                   params: dict[str, Any], models: list[dict[str, Any]], records: list[Any]) -> None:
        hardware = experiments.hardware_snapshot(self.ctx.hardware_profile(),
                                                 getattr(self.ctx.runtime, "last_heartbeat_payload", None))
        experiments.record_evaluation(self.ctx.registry, result, dataset=dataset.meta(), params=params,
                                      hardware=hardware, models=models, records=records, kind="mission",
                                      name=f"Mission · {dataset.title}")
        run = (result.get("models") or [{}])[0]
        summary = run.get("summary") or {}
        accuracy = summary.get("accuracy")
        speed = summary.get("tokens_per_second")
        text = (f"Accuracy {accuracy * 100:.0f}%" if accuracy is not None else "Done") + \
            (f" · {speed:.1f} tok/s" if speed else "") + " · saved in the Evaluation Lab."
        self._finish(step, run.get("status") == "finished", text)

    def _step_serve(self, step: mission.Step) -> None:
        record, _payload = self._llm_record()
        if record is None:
            self._finish(step, False, "Download the language model first.")
            return
        alias = re.search(r"'([^']+)'", step.title)
        name = alias.group(1) if alias else "noelle-chat"
        aliases = dict(self.ctx.settings.get("server.aliases") or {})
        aliases[name] = record.api_name
        self.ctx.settings.set("server.aliases", aliases)
        if self.ctx.server.state not in ("running", "starting"):
            self.ctx.server.start()
        self._finish(step, True, f"'{name}' → {record.api_name} on {self.ctx.server.base_url or 'the local API'}.")

    def _progress(self, step: mission.Step, message: str) -> None:
        if step is self._current and message.strip(": "):
            step.message = message.strip(": ")[:200]
            self._render_steps()

    # ------------------------------------------------------------ export
    def _export(self) -> None:
        if self.plan is None:
            return
        folder = paths.exports_dir()
        folder.mkdir(parents=True, exist_ok=True)
        default = folder / f"mission-{time.strftime('%Y%m%d-%H%M')}.md"
        path, _ = QFileDialog.getSaveFileName(self, "Export mission plan", str(default), "Markdown (*.md)")
        if path:
            Path(path).write_text(self.plan.to_markdown(), encoding="utf-8")
            self.ctx.toast(f"Saved {Path(path).name}", "success")

    def shutdown(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
        self._queue = []
