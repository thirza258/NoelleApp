"""Provider IPC contract, hybrid fallback, tools, voice chunking, speculation, i18n and native engines."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from noelle.app import i18n
from noelle.providers.hybrid import CloudProvider, HybridProvider
from noelle.providers.local_provider import LocalProvider
from noelle.providers.provider import ProviderError
from noelle.runtime import speculative, tools, voice


# ----------------------------------------------------------------- providers
def test_provider_error_is_structured() -> None:
    error = ProviderError.from_worker_payload({"code": "OOM", "message": "No VRAM", "model": "org/m"})
    assert error.code == "OOM" and error.info.recoverable
    assert "use_quantized_model" in error.info.suggested_actions
    data = error.to_dict()
    assert data["category"] == "runtime" and data["model_id"] == "org/m"


def test_local_provider_without_runtime_fails_structured() -> None:
    provider = LocalProvider()
    with pytest.raises(ProviderError) as info:
        provider.chat([{"role": "user", "content": "hi"}], model="org/model")
    assert info.value.code == "WORKER_UNAVAILABLE"
    assert "enable_cloud_fallback" in info.value.info.suggested_actions


class _LocalFails:
    is_local = True
    id = "local"

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        raise ProviderError("OOM", "No VRAM", suggested_actions=("use_quantized_model",))

    def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):  # type: ignore[no-untyped-def]
        raise ProviderError("OOM", "No VRAM", suggested_actions=("use_quantized_model",))
        yield ""  # pragma: no cover - keeps the generator contract explicit

    def health(self) -> dict[str, Any]:
        return {"state": "RUNTIME_ERROR"}

    def capabilities(self) -> Any:
        return SimpleNamespace(vision=False, audio_input=False, audio_output=False, tools=False, embedding=True,
                               image_generation=False, speculative_decoding=True, models=())


class _Cloud:
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "cloud answer"

    def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):  # type: ignore[no-untyped-def]
        yield "cloud "
        yield "answer"

    def health(self) -> dict[str, Any]:
        return {"configured": True}


def test_hybrid_fallback_requires_consent() -> None:
    denied = HybridProvider(_LocalFails(), _Cloud(), consent=lambda _reason, _destination: False)
    with pytest.raises(ProviderError):
        denied.chat([{"role": "user", "content": "hi"}], model="org/model")
    assert denied.last_route == "local"

    seen: list[tuple[str, str]] = []
    allowed = HybridProvider(_LocalFails(), _Cloud(),
                             consent=lambda reason, destination: seen.append((reason, destination)) or True)
    assert allowed.chat([{"role": "user", "content": "hi"}], model="org/model") == "cloud answer"
    assert allowed.last_route == "cloud" and seen and "OpenAI" in seen[0][1]

    local_only = HybridProvider(_LocalFails(), _Cloud(), local_only=True,
                                consent=lambda _reason, _destination: True)
    with pytest.raises(ProviderError):
        local_only.chat([{"role": "user", "content": "hi"}], model="org/model")
    assert local_only.last_route == "local"


def test_hybrid_stream_does_not_mix_partial_local_and_cloud() -> None:
    class PartialLocal:
        is_local = True
        id = "local"

        def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):  # type: ignore[no-untyped-def]
            yield "partial"
            raise ProviderError("OOM", "out of memory")

        def health(self) -> dict[str, Any]:
            return {}

        def capabilities(self) -> Any:
            return SimpleNamespace(vision=False, audio_input=False, audio_output=False, tools=False,
                                   embedding=True, image_generation=False, speculative_decoding=True, models=())

    provider = HybridProvider(PartialLocal(), _Cloud(), consent=lambda *_a: True)
    with pytest.raises(ProviderError):
        list(provider.stream_chat([{"role": "user", "content": "hi"}], model="org/model"))
    assert provider.last_route == "local"  # an error after tokens must not append a cloud answer


def test_consent_gate_bridges_worker_decision() -> None:
    from noelle.app.services.provider_stream import ConsentGate

    gate = ConsentGate()
    gate.requested.connect(lambda _reason, _destination: gate.answer(True))
    assert gate.decide("no VRAM", "OpenAI") is True


def test_cloud_provider_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from noelle.providers import hybrid as hybrid_module

    monkeypatch.setattr(hybrid_module, "has_api_key", lambda _provider: False)
    cloud = CloudProvider()
    with pytest.raises(ProviderError) as info:
        cloud.chat([{"role": "user", "content": "hi"}], model="gpt-4o-mini")
    assert info.value.code == "MISSING_API_KEY"


# --------------------------------------------------------------------- tools
def test_calculator_is_safe_and_exact() -> None:
    result = tools.calculator("1234*5678")
    assert result.ok and result.summary == "7006652"
    assert tools.calculator("3.5 + 2").summary == "5.5"
    for expression in ("__import__('os')", "x + 1", "2**999", "open('x')"):
        assert not tools.calculator(expression).ok


def test_datetime_and_argument_validation() -> None:
    assert tools.current_datetime().ok
    assert tools.current_datetime("Asia/Jakarta").ok
    assert not tools.current_datetime("Nowhere/Invalid").ok
    with pytest.raises(tools.ToolError):
        tools.validate_call(tools.ToolCall("calculator", {}))
    with pytest.raises(tools.ToolError):
        tools.validate_call(tools.ToolCall("rm_rf", {"path": "/"}))


@pytest.mark.parametrize("payload", [
    '<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>',
    '```json\n{"name":"calculator","arguments":{"expression":"2+2"}}\n```',
    '{"name":"calculator","arguments":{"expression":"2+2"}}',
    '[TOOL_CALLS][{"name":"calculator","parameters":{"expression":"2+2"}}]',
])
def test_tool_call_formats(payload: str) -> None:
    call = tools.parse_tool_call(f"answer first {payload} then done")
    assert call == tools.ToolCall("calculator", {"expression": "2+2"})


def test_unknown_tool_and_instructions() -> None:
    assert tools.parse_tool_call('<tool_call>{"name":"shell","arguments":{}}</tool_call>') is None
    text = tools.tool_instructions(["calculator", "current_datetime"])
    assert "calculator" in text and "<tool_call>" in text and "web_search" not in text


# --------------------------------------------------------------------- voice
def test_sentence_chunker_keeps_abbreviations_and_decimals() -> None:
    text = ("Dr. Smith arrived at 3.5 seconds. It costs Rp 12.500,00! Halo dunia, ini uji coba… "
            "dst. semuanya berjalan.")
    chunks = voice.split_sentences(text, min_chars=10, max_chars=200)
    assert any("Dr. Smith" in chunk and "3.5" in chunk for chunk in chunks)
    assert any("Halo dunia" in chunk for chunk in chunks)
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_long_chunks_split_and_silence_detection() -> None:
    chunks = voice.split_sentences("word " * 200, min_chars=10, max_chars=80)
    assert chunks and all(len(chunk) <= 80 for chunk in chunks)
    assert voice.is_silence([0.0] * 12)
    assert not voice.is_silence([0.2] * 12)


# --------------------------------------------------------------- speculation
def test_speculative_compatibility_checks() -> None:
    target = {"vocab_size": 1000, "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float16"}
    ok, reason = speculative.compatibility(target, dict(target))
    assert ok and not reason
    bad, reason = speculative.compatibility(target, {**target, "vocab_size": 2000})
    assert not bad and "Vocabulary" in reason
    bad, reason = speculative.compatibility(target, {**target, "eos_token_id": 3})
    assert not bad and "EOS" in reason
    unknown, reason = speculative.compatibility({}, {})
    assert not unknown and "metadata" in reason
    config = speculative.SpeculativeConfig.from_mapping({"mode": "draft", "draft_model": "org/draft"})
    assert config.enabled and config.draft_model == "org/draft" and config.to_dict()["mode"] == "draft"


def test_speculative_draft_suggestions() -> None:
    records = [SimpleNamespace(repo_id="org/target", pipeline_tag="text-generation", source="hub"),
               SimpleNamespace(repo_id="org/small", pipeline_tag="text-generation", source="hub"),
               SimpleNamespace(repo_id="other/one", pipeline_tag="text-generation", source="hub"),
               SimpleNamespace(repo_id="org/embed", pipeline_tag="feature-extraction", source="hub")]
    drafts = speculative.suggest_drafts("org/target", records)
    assert drafts[0] == "org/small" and "org/target" not in drafts


# ---------------------------------------------------------------------- i18n
def test_i18n_catalogs_and_fallback() -> None:
    assert i18n.set_language("en") == "en"
    assert i18n.tr("chat.send") == "Send"
    assert i18n.set_language("id") == "id"
    assert i18n.tr("chat.send") == "Kirim"
    assert i18n.tr("tools.calling", name="calculator") == "Memanggil calculator…"
    assert i18n.tr("missing.key", "Fallback text") == "Fallback text"
    assert i18n.current_language() == "id"
    i18n.set_language("en")


# --------------------------------------------------------------- native engines
def test_voice_worker_turn_emits_transcript_tokens_and_audio(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import threading

    from noelle.runtime import worker_service
    from noelle.runtime.protocol import Event

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    out = tmp_path / "out.wav"
    out.write_bytes(b"RIFF")
    events: list[tuple[str, dict]] = []

    class Engine:
        def __init__(self, task: str) -> None:
            self.task = task
            self.engine_name = f"fake-{task}"
            self.device = "cpu"

        def transcribe(self, path: str, **_kwargs: Any) -> dict[str, Any]:
            return {"text": "halo noelle", "chunks": [], "metrics": {"real_time_factor": 0.1}}

        def stream_chat(self, messages, params, *, stop_event, on_text, system_prompt=None):  # type: ignore[no-untyped-def]
            on_text("Halo! ")
            on_text("Dunia.")
            return SimpleNamespace(text="Halo! Dunia.", metrics=lambda: {"tokens_per_second": 10.0})

        def synthesize(self, text: str, **_kwargs: Any) -> dict[str, Any]:
            return {"audio_path": str(out), "sampling_rate": 16000, "duration_s": 0.5}

    class Models:
        def ensure(self, spec: dict[str, Any], on_status=None):  # type: ignore[no-untyped-def]
            return SimpleNamespace(engine=Engine(spec["task"]), engine_name=f"fake-{spec['task']}", device="cpu")

    class Worker:
        models = Models()

        @staticmethod
        def spec(payload: dict[str, Any], default: str) -> dict[str, Any]:
            return {**payload, "task": payload.get("task") or default}

    class Ctx:
        payload = {"audio_path": str(audio), "messages": [], "asr": {"model_id": "org/asr"},
                   "tts": {"model_id": "org/tts"}, "llm": {"model_id": "org/llm"}, "voice": 0, "speed": 1.0}
        stop_event = threading.Event()

        @staticmethod
        def status(_stage: str, _message: str) -> None:
            pass

        @staticmethod
        def emit(event: str, payload: dict[str, Any]) -> None:
            events.append((str(event), payload))

    worker_service._cmd_voice_turn(Worker(), Ctx())  # type: ignore[arg-type]
    kinds = [event for event, _payload in events]
    assert Event.TRANSCRIPT in kinds and Event.TOKEN_STREAM in kinds and Event.AUDIO_CHUNK in kinds and Event.RESULT in kinds
    result = next(payload for event, payload in events if event == Event.RESULT)
    assert result["transcript"] == "halo noelle" and result["audio_chunks"]


def test_worker_run_tool_executes_calculator() -> None:
    from noelle.runtime import worker_service
    from noelle.runtime.protocol import Event

    sent: list[tuple[str, dict]] = []

    class Ctx:
        payload = {"name": "calculator", "arguments": {"expression": "6*7"}}

        @staticmethod
        def emit(event: str, payload: dict[str, Any]) -> None:
            sent.append((str(event), payload))

    class Worker:
        pass

    worker_service._cmd_run_tool(Worker(), Ctx())  # type: ignore[arg-type]
    assert sent and sent[0][0] == Event.RESULT and sent[0][1]["summary"] == "42"


def test_native_engine_registry_does_not_import_modules() -> None:
    from noelle.runtime.native import detect_native_engines

    statuses = {status.engine_id: status for status in detect_native_engines()}
    assert {"kokoro", "faster-whisper", "model2vec", "diffusion-single-file"} <= set(statuses)
    assert statuses["diffusion-single-file"].capabilities == ("diffusion",)