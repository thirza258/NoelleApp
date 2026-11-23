"""Real MLX 4-bit conversion of the cached SmolLM2-135M-Instruct snapshot."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from noelle.runtime import quantization_engine as qe
from noelle.runtime.errors import Cancelled

pytestmark = pytest.mark.integration

TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(autouse=True)
def _require_mlx() -> None:
    cap = qe.capabilities("mps")["mlx-4bit"]
    if not cap["available"]:
        pytest.skip(cap["reason"])


def _safetensors_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.glob("*.safetensors"))  # stat() follows HF cache symlinks


def test_mlx_4bit_conversion_produces_a_working_smaller_checkpoint(text_model_path: Path, tmp_path: Path) -> None:
    from noelle.runtime.generation import MlxTextEngine, is_mlx_checkpoint
    from noelle.runtime.protocol import GenerationParams

    out = tmp_path / "smol-mlx"
    events: list[dict] = []
    result = qe.quantize(str(text_model_path), "mlx-4bit", model_id=TEXT_MODEL, output_dir=str(out), device="mps",
                         progress=events.append, stop_event=threading.Event())

    assert result["output_dir"] == str(out)
    assert is_mlx_checkpoint(out)
    source_bytes = _safetensors_bytes(text_model_path)
    assert 0 < _safetensors_bytes(out) < source_bytes / 2
    assert 0 < result["size_bytes"] < source_bytes
    assert result["method"] == result["precision"] == "mlx-4bit"
    assert result["engine"] == "mlx"
    assert result["gguf_file"] is None
    assert result["derived_id"] == "local/SmolLM2-135M-Instruct-mlx-4bit"
    assert result["base_model"] == TEXT_MODEL
    assert result["elapsed_ms"] > 0
    assert qe.read_marker(out)["method"] == "mlx-4bit"
    assert not list(tmp_path.glob(".smol-mlx.*")), "staging folder left behind"

    stages = [e["stage"] for e in events]
    assert stages[0] == "prepare" and "convert" in stages and stages[-1] == "finalize"
    assert any("Quantiz" in e["message"] for e in events if e["stage"] == "convert"), events
    assert all({"stage", "message", "current", "total"} <= set(e) for e in events)

    engine = MlxTextEngine.load(str(out))
    try:
        pieces: list[str] = []
        generated = engine.stream_chat([{"role": "user", "content": "Say hello in one short sentence."}],
                                       GenerationParams(temperature=0.0, max_new_tokens=16),
                                       stop_event=threading.Event(), on_text=pieces.append)
    finally:
        engine.release()
    assert generated.completion_tokens > 0
    assert "".join(pieces).strip()


def test_worker_quantize_commands(worker_process, text_model_path: Path, tmp_path: Path) -> None:
    """The real worker exposes capabilities on its I/O thread and runs QUANTIZE on the GPU lane."""
    from noelle.runtime.protocol import Command, Event

    client = worker_process.client()
    caps = client.call(Command.QUANTIZE_CAPABILITIES, timeout=30)
    assert set(caps["methods"]) == {"mlx-4bit", "gguf-q4_k_m", "awq-4bit", "gptq-4bit", "nf4"}
    assert caps["methods"]["mlx-4bit"]["available"] is True

    out = tmp_path / "via-worker"
    events = list(client.stream(Command.QUANTIZE, {"model_id": TEXT_MODEL, "local_path": str(text_model_path),
                                                   "method": "mlx-4bit", "output_dir": str(out)}, timeout=300))
    assert events[-1]["event"] == Event.RESULT, events[-1]
    result = events[-1]["payload"]
    assert result["output_dir"] == str(out) and result["engine"] == "mlx"
    assert (out / "config.json").is_file()
    progress = [e["payload"] for e in events if e["event"] == Event.PROGRESS]
    assert progress and {p["stage"] for p in progress} >= {"prepare", "convert", "finalize"}

    refused = list(client.stream(Command.QUANTIZE, {"model_id": TEXT_MODEL, "local_path": str(text_model_path),
                                                    "method": "awq-4bit", "output_dir": str(tmp_path / "awq")},
                                 timeout=60))
    assert refused[-1]["event"] == Event.ERROR
    assert refused[-1]["payload"]["code"] == "UNSUPPORTED"
    assert "NVIDIA CUDA" in refused[-1]["payload"]["message"]


def test_mlx_conversion_can_be_cancelled_mid_run(text_model_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "cancelled"
    stop = threading.Event()

    def progress(event: dict) -> None:
        if event["stage"] == "convert" and event["message"].startswith("[INFO]"):  # mlx_lm itself is running
            stop.set()

    with pytest.raises(Cancelled):
        qe.quantize(str(text_model_path), "mlx-4bit", model_id=TEXT_MODEL, output_dir=str(out), device="mps",
                    progress=progress, stop_event=stop)
    assert not out.exists()
    assert not list(tmp_path.glob(".cancelled.*"))
