"""Evaluation Lab page: model/dataset selection, live progress, saved experiment, history, matrix, export."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QObject, QTimer, Signal

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeHandle(QObject):
    event = Signal(str, dict)
    finished = Signal(dict)
    failed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def fake_eval(payload: dict[str, Any], handle: FakeHandle) -> None:
    """Answer like the worker: model 1 answers correctly, model 2 always says 0."""
    from noelle.runtime import evaluation

    cases = [evaluation.EvalCase.from_dict(c, i) for i, c in enumerate(payload["cases"])]
    handle.event.emit("QUEUE_PLAN", {"mode": "retain"})
    models, done, total = [], 0, len(cases) * len(payload["models"])
    for view, model in enumerate(payload["models"], start=1):
        rows = []
        for case in cases:
            output = str(case.expected) if view == 1 else "0"
            score = evaluation.score_case(case, output)
            row = {"view": view, "case_id": case.id, "prompt": case.display_prompt(), "expected": case.expected,
                   "output": output, "score": score.score, "passed": score.passed, "detail": score.detail,
                   "ttft_ms": 100.0 * view, "tokens_per_second": 40.0 / view, "completion_tokens": 3,
                   "latency_ms": 300.0, "tags": [], "error": None}
            rows.append(row)
            done += 1
            handle.event.emit("PROGRESS", {"current": done, "total": total, "view": view,
                                           "model_id": model["model_id"], "case": row})
        models.append({"view": view, "model_id": model["model_id"], "precision": model.get("precision", "auto"),
                       "status": "finished", "summary": evaluation.summarize(rows),
                       "metrics": {"engine": "transformers", "device": "mps", "peak_bytes": 2**30, "load_ms": 500},
                       "error": None, "cases": rows})
    handle.finished.emit({"models": models, "cases": len(cases), "total_ms": 1234.0})


@pytest.fixture()
def page(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from noelle.app import main as appmain
    from noelle.app.services.model_registry import ModelRecord
    from noelle.app.ui.pages.evaluate_page import EvaluatePage

    ctx = appmain.build_context()
    for name in ("org/alpha-1B", "org/beta-1B"):
        folder = tmp_path / name.replace("/", "_")
        folder.mkdir()
        ctx.registry.upsert_model(ModelRecord(repo_id=name, pipeline_tag="text-generation", local_path=str(folder),
                                              commit_hash="c0ffee"))
    ctx.registry.upsert_model(ModelRecord(repo_id="org/embedder", pipeline_tag="feature-extraction",
                                          local_path=str(tmp_path)))
    sent: list[dict[str, Any]] = []

    def request(cmd: Any, payload: dict[str, Any] | None = None) -> FakeHandle:
        handle = FakeHandle()
        sent.append({"cmd": str(cmd), "payload": payload})
        QTimer.singleShot(10, lambda: fake_eval(payload or {}, handle))
        return handle

    monkeypatch.setattr(ctx.runtime, "request", request)
    widget = EvaluatePage(ctx)
    widget.sent = sent
    qtbot.addWidget(widget)
    widget.show()
    widget.start()
    return widget


def test_run_records_an_experiment(page: Any, qtbot: Any) -> None:
    assert page.models.count() == 2  # the embedding model is not offered
    assert not page.run_btn.isEnabled()
    page.set_checked(["org/alpha-1B", "org/beta-1B"])
    assert page.select_dataset("builtin:reasoning-math") and page.run_btn.isEnabled()
    page.max_tokens.setValue(32)
    page.run_btn.click()
    qtbot.waitUntil(lambda: page.experiment is not None, timeout=5000)
    (request,) = page.sent
    assert request["cmd"] == "RUN_EVAL" and len(request["payload"]["models"]) == 2
    assert request["payload"]["params"] == {"max_new_tokens": 32, "temperature": 0.0}
    assert len(request["payload"]["cases"]) == 10

    experiment = page.experiment
    assert experiment["status"] == "finished" and experiment["dataset"]["name"] == "reasoning-math"
    assert page.leaderboard.rowCount() == 2
    first = request["payload"]["models"][0]["model_id"]  # the fake answers correctly for the first model only
    assert page.leaderboard.item(0, 0).text() == first and page.leaderboard.item(0, 3).text() == "100.0%"
    assert page.leaderboard.item(1, 3).text() == "0.0%"
    assert page.cases.rowCount() == 10 and page.cases.item(0, 1).text() == "pass"
    run = experiment["runs"][0]
    assert run["provenance"]["commit_hash"] == "c0ffee"
    assert page.run_btn.text() == "Run evaluation"

    page.tabs.setCurrentIndex(1)
    assert page.history.rowCount() == 1 and page.history.item(0, 6).text() == "finished"
    page.tabs.setCurrentIndex(2)
    assert page.matrix.rowCount() == 2


def test_trusted_publisher_reaches_the_runtime(page: Any, qtbot: Any) -> None:
    page.ctx.settings.set("runtime.trusted_publishers", ["org"])
    page.set_checked(["org/alpha-1B"])
    page.select_dataset("builtin:sentiment")
    page.run_btn.click()
    qtbot.waitUntil(lambda: page.experiment is not None, timeout=5000)
    (request,) = page.sent
    assert request["payload"]["models"][0]["trust_remote_code"] is True


def test_export_and_custom_dataset(page: Any, qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QFileDialog

    dataset = tmp_path / "mine.jsonl"
    dataset.write_text('{"_dataset": {"name": "mine", "title": "Mine"}}\n'
                       '{"id": "x", "prompt": "Say 7", "expected": 7, "scorer": "numeric"}\n', encoding="utf-8")
    assert page.add_dataset_file(str(dataset)) and page.dataset.name == "mine"
    page.set_checked(["org/alpha-1B"])
    page.run_btn.click()
    qtbot.waitUntil(lambda: page.experiment is not None, timeout=5000)
    target = tmp_path / "report.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    page._export("md")
    text = target.read_text(encoding="utf-8")
    assert "| Model |" in text and "Mine" in text


def test_invalid_dataset_is_reported(page: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json", encoding="utf-8")
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    assert not page.add_dataset_file(str(bad))
    assert shown and "line 1" in shown[0]
