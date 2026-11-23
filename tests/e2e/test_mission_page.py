"""Mission page: a plan runs step by step — approval, download, test (Evaluation Lab), serve (API alias)."""

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
    status = Signal(dict)
    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(dict)

    def cancel(self) -> None:
        pass


@pytest.fixture()
def page(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from PySide6.QtWidgets import QMessageBox

    from noelle.app import main as appmain
    from noelle.app.services.model_registry import ModelRecord
    from noelle.app.ui.pages.mission_page import MissionPage

    ctx = appmain.build_context()
    answers = {"value": QMessageBox.StandardButton.Yes}
    asked: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or answers["value"])

    def start_download(repo_id: str, **_kw: Any) -> None:
        folder = tmp_path / repo_id.replace("/", "_")
        folder.mkdir(exist_ok=True)
        ctx.registry.upsert_model(ModelRecord(repo_id=repo_id, pipeline_tag="text-generation", local_path=str(folder),
                                              commit_hash="feedbee"))
        QTimer.singleShot(10, lambda: ctx.downloads.job_finished.emit(repo_id, True))

    monkeypatch.setattr(ctx.downloads, "start", start_download)
    requests: list[tuple[str, dict[str, Any]]] = []

    def request(cmd: Any, payload: dict[str, Any] | None = None) -> FakeHandle:
        handle = FakeHandle()
        requests.append((str(cmd), payload or {}))
        if str(cmd) == "RUN_EVAL":
            rows = [{"case_id": c.get("id"), "score": 1.0, "passed": True, "tokens_per_second": 42.0, "ttft_ms": 90.0}
                    for c in payload["cases"]]
            result = {"models": [{"view": 1, "model_id": payload["models"][0]["model_id"], "status": "finished",
                                  "precision": "auto", "summary": {"accuracy": 1.0, "tokens_per_second": 42.0},
                                  "metrics": {"engine": "transformers", "device": "mps"}, "cases": rows}],
                      "cases": len(rows), "total_ms": 10.0}
            QTimer.singleShot(10, lambda: handle.finished.emit(result))
        return handle

    monkeypatch.setattr(ctx.runtime, "request", request)
    started: list[bool] = []
    monkeypatch.setattr(ctx.server, "start", lambda: started.append(True))
    widget = MissionPage(ctx)
    widget.answers, widget.asked, widget.requests, widget.started = answers, asked, requests, started
    qtbot.addWidget(widget)
    widget.show()
    return widget


def small_plan(text: str = "Indonesian chat assistant") -> Any:
    from noelle.app.services import mission
    from noelle.app.services.hf_client import ModelSummary
    from noelle.runtime import advisor
    from noelle.runtime.preflight import GB, MemoryBudget

    summary = ModelSummary(id="Qwen/Qwen2.5-1.5B-Instruct", pipeline_tag="text-generation", tags=["conversational"],
                           license="apache-2.0", param_count=1_540_000_000, downloads=1_000_000)
    hw = advisor.HardwareProfile("mps", "Apple M2 Pro GPU", MemoryBudget(int(12 * GB), 0.15, int(16 * GB), "mps"),
                                 {"transformers": True, "mlx": True}, 200)
    return mission.Planner(search=lambda **_q: [summary], info=None, hardware=hw).plan(text)


def test_run_all_steps(page: Any, qtbot: Any) -> None:
    plan = small_plan()
    assert [s.key for s in plan.steps] == ["download", "test", "serve"]
    page.set_plan(plan)
    page.run_all_btn.click()
    qtbot.waitUntil(lambda: all(s.status == "done" for s in plan.steps), timeout=5000)
    assert len(page.asked) == 2 and "Download Qwen/Qwen2.5-1.5B-Instruct" in page.asked[0]
    (cmd, payload), = page.requests
    assert cmd == "RUN_EVAL" and payload["models"][0]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "Accuracy 100%" in plan.steps[1].message
    experiments = page.ctx.registry.list_experiments(kind="mission")
    assert len(experiments) == 1 and experiments[0]["dataset"]["name"] == "indonesian-qa"
    assert page.ctx.settings.get("server.aliases") == {"noelle-chat": "Qwen/Qwen2.5-1.5B-Instruct"}
    assert page.started == [True]


def test_declining_a_download_stops_the_plan(page: Any) -> None:
    from PySide6.QtWidgets import QMessageBox

    plan = small_plan()
    page.set_plan(plan)
    page.answers["value"] = QMessageBox.StandardButton.No
    page.run_all()
    assert plan.steps[0].status == "skipped" and all(s.status == "pending" for s in plan.steps[1:])
    assert not page.requests and page.run_all_btn.isEnabled()


def test_test_step_needs_the_model(page: Any) -> None:
    plan = small_plan()
    page.set_plan(plan)
    page.run_steps([plan.steps[1]])  # skip the download
    assert plan.steps[1].status == "failed" and "Download the language model first" in plan.steps[1].message
