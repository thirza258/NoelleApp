"""E2E UI test for WorkflowPage in offscreen PySide6 environment."""

from __future__ import annotations

import os
from typing import Any
import pytest
from PySide6.QtCore import Qt

from noelle.app.ui.pages.workflow_page import WorkflowPage
from noelle.workflow.graph import PrivacyMode

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def ctx(qapp: Any) -> Any:
    from noelle.app import main as appmain
    return appmain.build_context()


def test_workflow_page_instantiation_and_templates(ctx: Any, qtbot: Any) -> None:
    page = WorkflowPage(ctx)
    qtbot.addWidget(page)
    page.show()

    # Initial template loaded
    assert page.node_list.count() > 0

    # Test template switching
    idx = page.template_combo.findData("embedding_lab")
    page.template_combo.setCurrentIndex(idx)
    qtbot.wait(50)
    assert page.graph.name == "embedding_lab"

    # Test privacy mode selector
    p_idx = page.privacy_combo.findData(PrivacyMode.LOCAL_ONLY)
    page.privacy_combo.setCurrentIndex(p_idx)
    qtbot.wait(50)
    assert page.graph.privacy_mode == PrivacyMode.LOCAL_ONLY

    # Test decision advisor tab recalculation
    page.queries_spin.setValue(100_000)
    qtbot.wait(50)
    assert "Dedicated Local Workstation" in page.decision_banner.text.text()


def test_workflow_page_execution(ctx: Any, qtbot: Any) -> None:
    from noelle.providers.base import NodeResponse, ProviderCapabilities, TelemetryRecord
    from noelle.providers.registry import _registry, ensure_default_providers, register_provider

    ensure_default_providers()
    original = _registry.get("local")

    class FakeLocal:
        id = "local"
        name = "Fake local"
        is_local = True

        def get_capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(provider_id="local", display_name=self.name, is_local=True,
                                        modalities={"llm", "tts", "stt", "embedding", "vision", "rag"})

        def estimate_cost(self, _request: Any) -> float:
            return 0.0

        def execute(self, request: Any) -> NodeResponse:
            output = b"RIFF" if request.modality_str == "tts" else "jawaban lokal"
            return NodeResponse(output=output, telemetry=TelemetryRecord(latency_ms=1.0),
                                provider_id="local", model_id=request.model or "local")

    register_provider(FakeLocal())
    page = WorkflowPage(ctx)
    qtbot.addWidget(page)
    page.show()

    # Load a purely local template to avoid confirmation dialogs
    idx = page.template_combo.findData("embedding_lab")
    page.template_combo.setCurrentIndex(idx)
    # Ensure all nodes are local
    for node in page.graph.nodes.values():
        node.provider_id = "local"

    qtbot.mouseClick(page.run_btn, Qt.MouseButton.LeftButton)
    try:
        qtbot.waitUntil(lambda: page.telemetry_table.rowCount() > 0, timeout=5000)
        assert "Latency" in page.stat_latency.text()
    finally:
        if original is not None:
            register_provider(original)
