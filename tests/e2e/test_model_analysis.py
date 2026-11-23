"""Workspace analysis: score card + Analysis tab, runtime profiles and remote-code trust levels."""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CFG = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "num_hidden_layers": 28,
       "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584, "max_position_embeddings": 32768,
       "auto_map": {"AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"}}


@pytest.fixture()
def page(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from noelle.app import main as appmain
    from noelle.app.services.hf_client import HFClient, ModelSummary
    from noelle.app.ui.main_window import MainWindow
    from noelle.runtime.preflight import GB, MemoryBudget

    summary = ModelSummary(id="Qwen/Qwen-7B-Chat", author="Qwen", pipeline_tag="text-generation",
                           library_name="transformers", tags=["custom_code", "safetensors"], license="apache-2.0",
                           param_count=7_700_000_000, config=CFG,
                           siblings=[{"name": "model.safetensors", "size": 1}, {"name": "modeling_qwen.py", "size": 1},
                                     {"name": "configuration_qwen.py", "size": 1}])
    monkeypatch.setattr(HFClient, "pager", lambda self, **kw: type("P", (), {"exhausted": True,
                                                                             "fetch_next": lambda s: []})())
    monkeypatch.setattr(HFClient, "trending", lambda self, count=5, tag=None: [])
    monkeypatch.setattr(HFClient, "model_info", lambda self, repo: summary)
    monkeypatch.setattr(HFClient, "fetch_config", lambda self, repo: CFG)
    monkeypatch.setattr(HFClient, "fetch_file_text", lambda self, repo, name, max_bytes=0: (
        '{"chat_template": "{% for m in messages %}{% endfor %}{% if tools %}{% endif %}"}'
        if name == "tokenizer_config.json" else ""))
    monkeypatch.setattr(HFClient, "fetch_readme", lambda self, repo: "")
    monkeypatch.setattr(HFClient, "whoami", lambda self, token=None: {"name": "tester"})
    ctx = appmain.build_context()
    ctx.hardware.memory_budget = lambda headroom=0.15: MemoryBudget(int(12 * GB), headroom, int(16 * GB), "mps")
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.show()
    window.open_model(summary)
    workspace = window.pages["workspace"]
    qtbot.waitUntil(lambda: workspace.tokenizer_config is not None and workspace.assessment is not None, timeout=5000)
    yield workspace
    window.close()


def test_analysis_tab_and_summary(page: Any) -> None:
    a = page.assessment
    assert page.tabs.indexOf(page.analysis_tab) == 1
    assert page.analysis_summary.dial._score == a.score == page.analysis_tab.dial._score
    assert a.capability.tool_calling is True and a.capability.context_length == 32768
    assert page.analysis_tab.options.rowCount() == len(a.options) > 0
    assert page.analysis_tab.title.text().startswith(f"{a.score} / 100")
    page.analysis_summary.details_requested.emit()
    assert page.tabs.currentWidget() is page.analysis_tab


def test_profiles_persist_and_sync(page: Any) -> None:
    page.analysis_tab._profile_buttons["low_memory"].click()
    assert page.ctx.settings.get("runtime.profile") == "low_memory"
    assert page.profile.currentData() == "low_memory" and page.assessment.profile == "low_memory"
    page.profile.setCurrentIndex(page.profile.findData("quality"))
    assert page.analysis_tab._profile_buttons["quality"].isChecked() and page.assessment.profile == "quality"


def test_remote_code_trust_levels(page: Any) -> None:
    ctx, notice = page.ctx, page.remote_notice
    assert notice.isVisible() and notice.state == "blocked"
    assert "modeling_qwen.py" in notice.files.text()
    assert page.worker_payload()["trust_remote_code"] is False
    trust_before = {i.key: i.points for i in page.assessment.items}["trust"]

    notice.allow_once.emit()
    assert notice.state == "once" and page.worker_payload()["trust_remote_code"] is True
    assert not ctx.settings.get("runtime.trusted_remote_code")  # session only, nothing persisted
    assert {i.key: i.points for i in page.assessment.items}["trust"] > trust_before

    notice.trust_repository.emit()
    assert notice.state == "repo" and "Qwen/Qwen-7B-Chat" in ctx.settings.get("runtime.trusted_remote_code")

    notice.revoke.emit()
    assert notice.state == "blocked" and page.worker_payload()["trust_remote_code"] is False
    assert "Qwen/Qwen-7B-Chat" not in (ctx.settings.get("runtime.trusted_remote_code") or [])

    notice.trust_publisher.emit()
    assert notice.state == "publisher" and ctx.settings.get("runtime.trusted_publishers") == ["Qwen"]
    assert ctx.is_trusted("Qwen/another-model") and page.worker_payload()["trust_remote_code"] is True
