from __future__ import annotations

import json
from pathlib import Path

import pytest

from noelle.app.services.model_registry import AdapterRecord, ModelRecord, ModelRegistry
from noelle.config.settings import Settings
from noelle.runtime import tasks
from noelle.runtime.protocol import (
    Command,
    Event,
    GenerationParams,
    ProtocolError,
    decode,
    error_payload,
    make_event,
    make_request,
    parse_request,
)


# ------------------------------------------------------------------ protocol
def test_request_round_trip() -> None:
    rid, frame = make_request(Command.GENERATE_STREAM, {"prompt": "hi"}, auth="tok")
    parsed_id, cmd, payload, auth = parse_request(frame)
    assert parsed_id == rid and cmd == Command.GENERATE_STREAM and payload == {"prompt": "hi"} and auth == "tok"


def test_event_frame_and_errors() -> None:
    frame = make_event("abc", Event.ERROR, error_payload("OOM", "boom", "use 4-bit"))
    data = decode(frame)
    assert data["event"] == "ERROR" and data["payload"]["code"] == "OOM" and data["payload"]["hint"] == "use 4-bit"
    with pytest.raises(ProtocolError):
        parse_request(b"not json")
    with pytest.raises(ProtocolError):
        parse_request(json.dumps({"id": "x", "cmd": "NOPE"}).encode())


def test_generation_params_are_clamped() -> None:
    params = GenerationParams.from_dict({"temperature": 9, "top_p": 0, "max_new_tokens": -5, "stop": "###", "seed": 7})
    assert params.temperature == 2.0
    assert params.top_p == 0.01
    assert params.max_new_tokens == 1
    assert params.stop == ["###"]
    assert params.seed == 7


# ------------------------------------------------------------------ settings
def test_settings_defaults_overrides_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = Settings(path)
    assert settings.get("rag.chunk_tokens") == 512
    settings.set("rag.chunk_tokens", 256)
    settings.update({"server.port": 9999, "runtime.device": "cpu"})
    again = Settings(path)
    assert again.get("rag.chunk_tokens") == 256
    assert again.get("server.port") == 9999
    again.reset("rag.chunk_tokens")
    assert Settings(path).get("rag.chunk_tokens") == 512
    assert settings.get("missing.key", "fallback") == "fallback"


def test_corrupt_settings_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    assert Settings(path).get("server.port") == 8765


# ------------------------------------------------------------------ registry
def test_registry_models_variants_and_payload(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.db")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    base = registry.upsert_model(ModelRecord("org/model-1B", "text-generation", str(snapshot), param_count=10**9))
    assert base.id is not None and base.api_name == "org/model-1B" and base.family == tasks.TEXT
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    registry.upsert_model(ModelRecord("local/model-1B-mlx-4bit", "text-generation", str(quant_dir),
                                      precision="mlx-4bit", source="quantized", base_model="org/model-1B", engine="mlx"))
    assert registry.get_model("org/model-1B").source == "hub"
    payload = registry.get_model("local/model-1B-mlx-4bit").worker_payload()
    assert payload["precision"] == "mlx-4bit" and payload["engine"] == "mlx" and payload["local_path"] == str(quant_dir)
    assert len(registry.list_models(family=tasks.TEXT)) == 2
    assert registry.find_by_api_name("local/model-1B-mlx-4bit") is not None
    # upsert keeps one row per (repo, precision, source)
    registry.upsert_model(ModelRecord("org/model-1B", "text-generation", str(snapshot), size_bytes=123))
    assert len(registry.list_models()) == 2
    quant_dir.rmdir()
    assert registry.prune_missing() == ["local/model-1B-mlx-4bit"]
    assert registry.is_downloaded("org/model-1B")


def test_registry_adapters_and_evaluations(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.db")
    registry.add_adapter(AdapterRecord("org/lora", "org/base", str(tmp_path), 1000))
    assert [a.adapter_id for a in registry.list_adapters(base_model_id="org/base")] == ["org/lora"]
    registry.add_evaluation("run1", 1, "org/base", prompt="p", system_prompt=None, params={"t": 0}, output="o",
                            status="finished", metrics={"ttft_ms": 12.5, "tokens_per_second": 40.0,
                                                        "model_peak_bytes": 99})
    rows = registry.list_evaluations("run1")
    assert rows[0]["ttft_ms"] == 12.5 and rows[0]["peak_memory_bytes"] == 99 and rows[0]["params"] == {"t": 0}


def test_task_families() -> None:
    assert tasks.task_family("text-generation") == tasks.TEXT
    assert tasks.task_family("text-to-speech") == tasks.TTS
    assert tasks.modality_for_tag("image-text-to-text") == "vision"
    assert not tasks.is_supported("robotics")
