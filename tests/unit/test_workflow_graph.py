"""Unit tests for Workflow Graph data model, topological sorting, and execution runner."""

from __future__ import annotations

import pytest

from noelle.workflow.graph import PrivacyMode, WorkflowGraph
from noelle.workflow.nodes import create_node
from noelle.workflow.runner import PrivacyViolationError, WorkflowRunner


def test_workflow_graph_construction_and_sort() -> None:
    graph = WorkflowGraph(id="wf_test", name="Test Graph")
    n1 = create_node("input", "in_1", title="Input Text", parameters={"value": "Hello world"})
    n2 = create_node("llm", "llm_1", title="Local LLM", provider_id="local")
    n3 = create_node("tts", "tts_1", title="TTS", provider_id="local")

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    graph.connect("in_1", "output", "llm_1", "prompt")
    graph.connect("llm_1", "text", "tts_1", "text")

    order = graph.topological_sort()
    assert order == ["in_1", "llm_1", "tts_1"]


def test_cycle_detection() -> None:
    graph = WorkflowGraph()
    n1 = create_node("llm", "n1")
    n2 = create_node("llm", "n2")
    graph.add_node(n1)
    graph.add_node(n2)

    graph.connect("n1", "text", "n2", "prompt")
    graph.connect("n2", "text", "n1", "prompt")  # Directed Cycle

    with pytest.raises(ValueError, match="Directed cycle"):
        graph.topological_sort()


def test_workflow_serialization() -> None:
    graph = WorkflowGraph(id="wf_ser", name="Serialization Test", privacy_mode=PrivacyMode.HYBRID)
    graph.add_node(create_node("input", "in_1", parameters={"value": "test"}))
    graph.add_node(create_node("llm", "llm_1", provider_id="local", model_id="Qwen2.5-7B"))
    graph.connect("in_1", "output", "llm_1", "prompt")

    json_str = graph.to_json()
    reloaded = WorkflowGraph.from_json(json_str)

    assert reloaded.id == "wf_ser"
    assert len(reloaded.nodes) == 2
    assert len(reloaded.edges) == 1
    assert reloaded.edges[0].from_node == "in_1"


def test_local_only_privacy_mode_enforcement() -> None:
    graph = WorkflowGraph(privacy_mode=PrivacyMode.LOCAL_ONLY)
    graph.add_node(create_node("input", "in_1", parameters={"value": "Confidential data"}))
    # Add a cloud node in local-only mode
    graph.add_node(create_node("llm", "cloud_llm", provider_id="openai", model_id="gpt-4o"))
    graph.connect("in_1", "output", "cloud_llm", "prompt")

    issues = graph.validate()
    assert any("Privacy Mode is Local-Only" in iss for iss in issues)

    runner = WorkflowRunner()
    with pytest.raises(PrivacyViolationError, match="Zero Egress Violation"):
        runner.execute(graph)


def test_workflow_execution_pipeline() -> None:
    from noelle.providers.base import NodeResponse, ProviderCapabilities, TelemetryRecord
    from noelle.providers.registry import _registry, ensure_default_providers, register_provider

    ensure_default_providers()
    original = _registry.get("local")

    class FakeLocal:
        id = "local"
        name = "Fake local provider"
        is_local = True

        def get_capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(provider_id="local", display_name=self.name, is_local=True,
                                        modalities={"llm", "tts", "stt", "embedding", "vision"})

        def estimate_cost(self, _request) -> float:  # type: ignore[no-untyped-def]
            return 0.0

        def execute(self, request):  # type: ignore[no-untyped-def]
            output = b"RIFF-fake-audio" if request.modality_str == "tts" else "Jawaban lokal sintesis suara."
            return NodeResponse(output=output,
                                telemetry=TelemetryRecord(latency_ms=1.0, cost_usd=0.0, output_tokens=4),
                                provider_id="local", model_id=request.model or "local")

    register_provider(FakeLocal())
    try:
        _run_workflow_pipeline()
    finally:
        if original is not None:
            register_provider(original)


def _run_workflow_pipeline() -> None:
    graph = WorkflowGraph(privacy_mode=PrivacyMode.HYBRID)
    graph.add_node(create_node("input", "in_1", parameters={"value": "Jelaskan sintesis suara"}))
    graph.add_node(create_node("llm", "llm_1", provider_id="local", model_id="Qwen2.5-7B"))
    graph.add_node(create_node("tts", "tts_1", provider_id="local", model_id="SpeechT5"))

    graph.connect("in_1", "output", "llm_1", "prompt")
    graph.connect("llm_1", "text", "tts_1", "text")

    runner = WorkflowRunner()
    result = runner.execute(graph)

    assert result.is_success
    assert result.total_latency_ms > 0.0
    assert result.total_cost_usd == 0.0  # All local nodes
    assert "llm_1" in result.node_outputs
    assert "tts_1" in result.node_outputs
    assert isinstance(result.node_outputs["tts_1"]["audio"], bytes)


def test_cli_forwards_cloud_consent(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from noelle.workflow import cli
    from noelle.workflow.runner import WorkflowResult

    graph = WorkflowGraph(privacy_mode=PrivacyMode.HYBRID)
    graph.add_node(create_node("input", "in_1", parameters={"value": "hello"}))
    graph.add_node(create_node("llm", "llm_1", provider_id="openai", model_id="gpt-4o"))
    graph.connect("in_1", "output", "llm_1", "prompt")
    manifest = tmp_path / "workflow.json"
    manifest.write_text(graph.to_json(), encoding="utf-8")
    calls: list[bool] = []

    class FakeRunner:
        def execute(self, _graph, initial_inputs=None, consent_given=False):  # type: ignore[no-untyped-def]
            calls.append(consent_given)
            return WorkflowResult(workflow_id="w", workflow_name="w", is_success=True, total_latency_ms=1.0,
                                  total_cost_usd=0.0, total_tokens=0)

    monkeypatch.setattr(cli, "WorkflowRunner", lambda: FakeRunner())
    args = SimpleNamespace(manifest=str(manifest), dataset=None, input="hello", output=None, privacy_mode=None,
                           allow_cloud=True)
    assert cli._run_workflow_cli(args) == 0
    assert calls == [True]


def test_cli_rejects_cloud_without_consent(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from noelle.workflow import cli

    graph = WorkflowGraph(privacy_mode=PrivacyMode.HYBRID)
    graph.add_node(create_node("input", "in_1", parameters={"value": "hello"}))
    graph.add_node(create_node("llm", "llm_1", provider_id="openai", model_id="gpt-4o"))
    graph.connect("in_1", "output", "llm_1", "prompt")
    manifest = tmp_path / "workflow.json"
    manifest.write_text(graph.to_json(), encoding="utf-8")
    args = SimpleNamespace(manifest=str(manifest), dataset=None, input="hello", output=None, privacy_mode=None,
                           allow_cloud=False)
    assert cli._run_workflow_cli(args) == 1
    assert "--allow-cloud" in capsys.readouterr().err


def test_rag_node_execution() -> None:
    graph = WorkflowGraph()
    rag_node = create_node(
        "rag",
        "rag_1",
        parameters={
            "query": "kebijakan moneter",
            "documents": [
                "Laporan fiskal kuartal 3 dan kebijakan moneter nasional.",
                "Peternakan sapi dan domba di Jawa Tengah.",
            ],
            "top_k": 1,
        },
    )
    graph.add_node(rag_node)

    runner = WorkflowRunner()
    res = runner.execute(graph)
    assert res.is_success
    context_out = res.node_outputs["rag_1"]["context"]
    assert "kebijakan moneter" in context_out
