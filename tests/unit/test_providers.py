"""Unit tests for AI Providers, pricing, and credential storage."""

from __future__ import annotations

import pytest
import httpx

from noelle.providers.base import Modality, NodeRequest
from noelle.providers.credentials import delete_api_key, get_api_key, has_api_key, set_api_key
from noelle.providers.local_provider import LocalProvider
from noelle.providers.openai_provider import OpenAIProvider
from noelle.providers.openrouter_provider import OpenRouterProvider
from noelle.providers.pricing import calculate_cost
from noelle.providers.registry import (
    get_capability_matrix,
    validate_node_target,
)


def test_credentials_management(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test setting, getting, checking, deleting in memory
    set_api_key("test_provider", "secret_key_123")
    assert has_api_key("test_provider")
    assert get_api_key("test_provider") == "secret_key_123"

    delete_api_key("test_provider")
    assert not has_api_key("test_provider")
    assert get_api_key("test_provider") is None


def test_pricing_calculations() -> None:
    # GPT-4o pricing: $2.50 input / $10.00 output per 1M tokens
    cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    expected = (1000 / 1e6 * 2.50) + (500 / 1e6 * 10.00)
    assert abs(cost - expected) < 1e-6

    # Whisper-1 pricing: $0.006 per minute
    audio_cost = calculate_cost("whisper-1", audio_duration_seconds=120.0)
    assert abs(audio_cost - 0.012) < 1e-6

    # TTS pricing: $0.015 per 1k characters
    tts_cost = calculate_cost("tts-1", text_chars=2000)
    assert abs(tts_cost - 0.030) < 1e-6


def test_local_provider_uses_the_worker(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from noelle.providers import local_provider as module_provider
    from noelle.runtime.protocol import Event

    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF-local-audio")

    class FakeWorker:
        def __init__(self, *_args, **_kwargs) -> None:
            self.calls: list[tuple[str, dict]] = []

        def stream(self, cmd, payload, timeout=None):  # type: ignore[no-untyped-def]
            self.calls.append((str(cmd), payload))
            yield {"event": Event.TOKEN_STREAM, "payload": {"text": "Halo "}}
            yield {"event": Event.TOKEN_STREAM, "payload": {"text": "Noelle"}}
            yield {"event": Event.GENERATION_COMPLETE, "payload": {"text": "Halo Noelle", "metrics": {"tokens_per_second": 42.0}}}

        def call(self, cmd, payload, timeout=None):  # type: ignore[no-untyped-def]
            self.calls.append((str(cmd), payload))
            if str(cmd) == "EMBED":
                return {"embeddings": [[0.1] * 768 for _ in payload["inputs"]], "dimension": 768}
            if str(cmd) == "TRANSCRIBE":
                return {"text": "transkripsi lokal", "duration_s": 1.0}
            if str(cmd) == "SYNTHESIZE_SPEECH":
                return {"audio_path": str(wav), "sampling_rate": 16000, "duration_s": 1.0}
            if str(cmd) == "VISION_GENERATE":
                return {"text": "a local vision answer"}
            if str(cmd) == "PING":
                return {"pid": 123, "runtime_ready": True, "uptime_s": 1.0}
            return {}

        def stop(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

    fake = FakeWorker()
    monkeypatch.setattr(module_provider, "WorkerClient", lambda *_a, **_k: fake)
    runtime = type("Runtime", (), {"state": "ready", "endpoint": "tcp://127.0.0.1:1", "token": None,
                                   "loaded_models": [], "runtime_caps": {}})()
    provider = LocalProvider(runtime_client=runtime)
    caps = provider.get_capabilities()
    assert caps.is_local and caps.supports_modality(Modality.LLM)
    assert provider.estimate_cost(NodeRequest(modality=Modality.LLM, prompt="Hello")) == 0.0

    res = provider.execute(NodeRequest(modality=Modality.LLM, prompt="Halo Noelle", model="org/model"))
    assert res.is_success and res.output == "Halo Noelle"
    assert res.telemetry.cost_usd == 0.0

    emb_res = provider.execute(NodeRequest(modality=Modality.EMBEDDING, model="org/embed",
                                           texts=["Test vector A", "Test vector B"]))
    assert emb_res.is_success and len(emb_res.output) == 2 and len(emb_res.output[0]) == 768

    stt_res = provider.execute(NodeRequest(modality=Modality.STT, model="org/asr",
                                           audio_bytes=b"\x00" * 32000))
    assert stt_res.is_success and stt_res.output == "transkripsi lokal"

    tts_res = provider.execute(NodeRequest(modality=Modality.TTS, model="org/tts", prompt="Suara lokal Noelle"))
    assert tts_res.is_success and tts_res.output == b"RIFF-local-audio"

    assert provider.health()["reachable"] is True
    assert any(cmd == "GENERATE_STREAM" for cmd, _payload in fake.calls)


def test_openai_provider_mocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "chat/completions" in url:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Halo dari Cloud AI"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15},
            })
        elif "embeddings" in url:
            return httpx.Response(200, json={
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 5},
            })
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-key", client=client)

    # Chat execution
    res = provider.execute(NodeRequest(modality=Modality.LLM, prompt="Halo"))
    assert res.is_success
    assert res.output == "Halo dari Cloud AI"
    assert res.telemetry.input_tokens == 10
    assert res.telemetry.output_tokens == 15
    assert res.telemetry.cost_usd > 0.0

    # Embedding execution
    emb = provider.execute(NodeRequest(modality=Modality.EMBEDDING, prompt="Teks dokumen"))
    assert emb.is_success
    assert emb.output == [0.1, 0.2, 0.3]


def test_openrouter_provider_mocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Jawaban dari OpenRouter Qwen"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
            "total_cost": 0.00015,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(api_key="sk-or-test", client=client)

    res = provider.execute(NodeRequest(modality=Modality.LLM, model="qwen/qwen-2.5-72b-instruct", prompt="Uji OpenRouter"))
    assert res.is_success
    assert res.output == "Jawaban dari OpenRouter Qwen"
    assert res.telemetry.cost_usd == 0.00015


def test_registry_and_capability_matrix() -> None:
    matrix = get_capability_matrix()
    assert "local" in matrix
    assert "openai" in matrix
    assert "openrouter" in matrix
    assert matrix["local"]["llm"] is True
    assert matrix["local"]["stt"] is True

    valid, reason = validate_node_target("local", "llm")
    assert valid

    invalid, reason = validate_node_target("nonexistent_provider", "llm")
    assert not invalid
