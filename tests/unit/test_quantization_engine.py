"""quantization_engine: capability gating, output-folder rules and the llama.cpp pipeline (fake tools)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from noelle import paths
from noelle.runtime import quantization_engine as qe
from noelle.runtime.errors import BadRequest, Cancelled, ModelNotFound, Unsupported

METHOD_KEYS = {"mlx-4bit", "gguf-q4_k_m", "awq-4bit", "gptq-4bit", "nf4"}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    folder = tmp_path / "src-model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "llama", "torch_dtype": "bfloat16"}))
    (folder / "model.safetensors").write_bytes(b"\0" * 4096)
    return folder


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the MLX runner with one that writes a tiny quantized checkpoint."""
    calls: list[Path] = []

    def runner(source: Path, staging: Path, *, name: str, device: str, report: Any,
               stop_event: threading.Event) -> None:
        calls.append(staging)
        report.step("convert", "fake convert")
        staging.mkdir(parents=True)
        (staging / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
        (staging / "model.safetensors").write_bytes(b"\1" * 1000)
        return None

    monkeypatch.setitem(qe._RUNNERS, "mlx-4bit", runner)
    monkeypatch.setattr(qe, "unavailable_reason", lambda method, device: None)
    return calls


def run(source: Path, method: str = "mlx-4bit", *, output_dir: Path | None = None,
        stop: threading.Event | None = None, events: list | None = None, device: str = "mps") -> dict:
    return qe.quantize(str(source), method, model_id="org/Tiny-Model", output_dir=str(output_dir) if output_dir else None,
                       device=device, progress=(events.append if events is not None else lambda _e: None),
                       stop_event=stop or threading.Event())


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_shape() -> None:
    caps = qe.capabilities("mps")
    assert set(caps) == METHOD_KEYS
    engines = {"mlx-4bit": "mlx", "gguf-q4_k_m": "llama_cpp", "awq-4bit": "transformers",
               "gptq-4bit": "transformers", "nf4": "transformers"}
    for key, cap in caps.items():
        assert set(cap) == {"label", "description", "available", "reason", "engine", "precision"}
        assert cap["precision"] == key
        assert cap["engine"] == engines[key]
        assert isinstance(cap["available"], bool)
        assert (cap["reason"] is None) == cap["available"]


def test_precision_keys_exist_in_registry_precisions() -> None:
    from noelle.runtime.preflight import PRECISIONS

    assert METHOD_KEYS <= set(PRECISIONS)


@pytest.mark.parametrize("method", ["awq-4bit", "gptq-4bit", "nf4"])
def test_cuda_methods_gated_on_device_and_package(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qe, "_has_module", lambda name: True)
    assert qe.capabilities("cuda")[method]["available"] is True
    mps = qe.capabilities("mps")[method]
    assert mps["available"] is False
    assert mps["reason"] == "Requires an NVIDIA CUDA GPU (this runtime: Apple Metal)."
    assert "NVIDIA CUDA GPU (this runtime: CPU)" in qe.capabilities("cpu")[method]["reason"]

    monkeypatch.setattr(qe, "_has_module", lambda name: False)
    missing = qe.capabilities("cuda")[method]
    assert missing["available"] is False
    assert "noelle[cuda-quant]" in missing["reason"]
    assert qe._CUDA_PACKAGES[method][1] in missing["reason"]


def test_cuda_methods_unavailable_on_this_machine() -> None:
    # No CUDA here (and none of autoawq / gptqmodel / bitsandbytes installed).
    for method in ("awq-4bit", "gptq-4bit", "nf4"):
        assert qe.capabilities("mps")[method]["available"] is False


def test_mlx_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qe, "is_apple_silicon", lambda: False)
    reason = qe.capabilities("cuda")["mlx-4bit"]["reason"]
    assert reason.startswith("Requires an Apple Silicon Mac (this runtime: NVIDIA CUDA")

    monkeypatch.setattr(qe, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(qe, "_has_module", lambda name: name != "mlx_lm")
    reason = qe.capabilities("mps")["mlx-4bit"]["reason"]
    assert "mlx_lm missing" in reason and 'pip install "noelle[mlx]"' in reason

    monkeypatch.setattr(qe, "_has_module", lambda name: True)
    assert qe.capabilities("mps")["mlx-4bit"]["available"] is True


def test_capabilities_is_light_and_never_imports_heavy_libraries(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "from noelle.runtime import quantization_engine as q\n"
        "caps = q.capabilities('mps')\n"
        "assert set(caps) == {'mlx-4bit', 'gguf-q4_k_m', 'awq-4bit', 'gptq-4bit', 'nf4'}\n"
        "heavy = [m for m in ('torch', 'mlx', 'mlx_lm', 'transformers', 'awq', 'gptqmodel', 'bitsandbytes',"
        " 'numpy') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "print('LIGHT_OK')\n"
    )
    env = dict(os.environ, NOELLE_HOME=str(tmp_path / "home"))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "LIGHT_OK" in proc.stdout


# ---------------------------------------------------------------------------
# llama.cpp discovery
# ---------------------------------------------------------------------------


def _no_system_llama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(qe, "_common_llama_dirs", lambda: [tmp_path / "nowhere"])


def _make_fake_llama(folder: Path, *, quantize_exit: int = 0) -> Path:
    """A llama.cpp-like folder: convert_hf_to_gguf.py, gguf-py/ and an executable llama-quantize."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gguf-py").mkdir(exist_ok=True)
    (folder / "convert_hf_to_gguf.py").write_text(
        "import argparse, sys, time\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('model'); p.add_argument('--outfile'); p.add_argument('--outtype')\n"
        "a = p.parse_args()\n"
        "assert a.outtype == 'f16'\n"
        "print('INFO:hf-to-gguf:Loading model: ' + a.model, flush=True)\n"
        "for i in range(3):\n"
        "    sys.stdout.write(f'Writing: {33 * (i + 1)}%|###| {i}/3\\r'); sys.stdout.flush()\n"
        "print('', flush=True)\n"
        "open(a.outfile, 'wb').write(b'GGUF' + b'\\0' * 4000)\n"
        "print('INFO:hf-to-gguf:Model successfully exported to ' + a.outfile, flush=True)\n")
    binary = folder / "build" / "bin" / "llama-quantize"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "src, dst, kind = sys.argv[1:4]\n"
        "assert kind == 'Q4_K_M', kind\n"
        "print('main: quantizing ' + src + ' to ' + dst + ' as ' + kind, flush=True)\n"
        "print('[   1/   2]  token_embd.weight - converting to q4_K', flush=True)\n"
        f"if {quantize_exit}:\n"
        "    print('llama_model_quantize: failed to quantize: unsupported tensor', flush=True)\n"
        f"    sys.exit({quantize_exit})\n"
        "open(dst, 'wb').write(b'GGUF' + b'\\0' * 1000)\n"
        "print('main: quantize time = 1.00 ms', flush=True)\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return folder


def test_gguf_unavailable_reason_names_missing_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_system_llama(monkeypatch, tmp_path)
    cap = qe.capabilities("mps")["gguf-q4_k_m"]
    assert cap["available"] is False
    assert "convert_hf_to_gguf.py and llama-quantize" in cap["reason"]
    assert "quantization.llama_cpp_dir" in cap["reason"]
    with pytest.raises(Unsupported):
        qe.quantize(str(tmp_path), "gguf-q4_k_m", model_id="x/y", output_dir=None, device="mps",
                    progress=lambda _e: None, stop_event=threading.Event())


def test_gguf_tools_found_via_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from noelle.config import get_settings

    _no_system_llama(monkeypatch, tmp_path)
    llama = _make_fake_llama(tmp_path / "llama.cpp")
    get_settings().set("quantization.llama_cpp_dir", str(llama))
    tools = qe.find_llama_cpp()
    assert tools.convert_script == llama / "convert_hf_to_gguf.py"
    assert tools.quantize_binary == llama / "build" / "bin" / "llama-quantize"
    assert tools.gguf_package is True
    assert qe.capabilities("mps")["gguf-q4_k_m"]["available"] is True


def test_gguf_requires_gguf_python_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_system_llama(monkeypatch, tmp_path)
    llama = _make_fake_llama(tmp_path / "llama.cpp")
    (llama / "gguf-py").rmdir()
    monkeypatch.setattr(qe, "_has_module", lambda name: False)
    reason = qe.find_llama_cpp(llama).missing_reason()
    assert reason is not None and "'gguf' Python package" in reason


def test_gguf_pipeline_with_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: Path) -> None:
    from noelle.config import get_settings

    _no_system_llama(monkeypatch, tmp_path)
    get_settings().set("quantization.llama_cpp_dir", str(_make_fake_llama(tmp_path / "llama.cpp")))
    events: list[dict] = []
    result = run(source, "gguf-q4_k_m", events=events)

    out = Path(result["output_dir"])
    assert out == paths.models_dir() / "Tiny-Model-gguf-q4_k_m"
    assert result["engine"] == "llama_cpp"
    assert result["precision"] == "gguf-q4_k_m"
    assert result["gguf_file"] == "Tiny-Model-Q4_K_M.gguf"
    assert (out / result["gguf_file"]).is_file()
    assert not (out / "Tiny-Model-f16.gguf").exists(), "FP16 intermediate must be deleted"
    assert result["derived_id"] == "local/Tiny-Model-gguf-q4_k_m"
    assert result["size_bytes"] == qe.folder_size(out)
    assert qe.read_marker(out)["gguf_file"] == result["gguf_file"]

    by_stage: dict[str, list[str]] = {}
    for event in events:
        by_stage.setdefault(event["stage"], []).append(event["message"])
        assert event["total"] == 4
    assert any("Model successfully exported" in m for m in by_stage["convert"])
    assert any("token_embd.weight" in m for m in by_stage["quantize"])
    assert [e["stage"] for e in events][0] == "prepare"
    assert events[-1]["stage"] == "finalize" and events[-1]["current"] == 4


def test_gguf_tool_failure_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: Path) -> None:
    from noelle.config import get_settings

    _no_system_llama(monkeypatch, tmp_path)
    get_settings().set("quantization.llama_cpp_dir", str(_make_fake_llama(tmp_path / "llama.cpp", quantize_exit=3)))
    out = tmp_path / "out"
    with pytest.raises(qe.QuantizationFailed, match="exit code 3.*unsupported tensor"):
        run(source, "gguf-q4_k_m", output_dir=out)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.*"))


# ---------------------------------------------------------------------------
# quantize(): validation and output folder rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["awq-4bit", "gptq-4bit", "nf4"])
def test_cuda_methods_raise_unsupported_here(method: str, source: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(Unsupported) as info:
        run(source, method, output_dir=out)
    assert "NVIDIA CUDA" in info.value.message
    assert not out.exists()


def test_unknown_method_is_a_bad_request(source: Path) -> None:
    with pytest.raises(BadRequest):
        run(source, "int3")


def test_source_validation(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    with pytest.raises(ModelNotFound):
        run(tmp_path / "missing")
    (source / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    with pytest.raises(BadRequest, match="already quantized"):
        run(source)
    assert fake_mlx == []


def test_default_output_dir_and_result(fake_mlx: list, source: Path) -> None:
    events: list[dict] = []
    result = run(source, events=events)
    out = paths.models_dir() / "Tiny-Model-mlx-4bit"
    assert result["output_dir"] == str(out)
    assert result["derived_id"] == "local/Tiny-Model-mlx-4bit"
    assert result["base_model"] == "org/Tiny-Model"
    assert result["engine"] == "mlx" and result["gguf_file"] is None
    assert (out / "model.safetensors").is_file()
    assert qe.read_marker(out)["method"] == "mlx-4bit"
    assert [e["stage"] for e in events] == ["prepare", "convert", "finalize", "finalize"]
    assert fake_mlx[0].name.startswith(".Tiny-Model-mlx-4bit.partial-")


def test_refuses_non_empty_foreign_folder(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    out = tmp_path / "busy"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(BadRequest, match="not empty"):
        run(source, output_dir=out)
    assert (out / "notes.txt").read_text() == "keep me"
    assert fake_mlx == []


def test_accepts_existing_empty_folder(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    out = tmp_path / "empty"
    out.mkdir()
    result = run(source, output_dir=out)
    assert result["output_dir"] == str(out)
    assert (out / "config.json").is_file()


def test_replaces_previous_output_of_same_method(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    out = tmp_path / "prev"
    run(source, output_dir=out)
    (out / "stale.bin").write_bytes(b"old")
    run(source, output_dir=out)
    assert not (out / "stale.bin").exists()
    assert (out / "model.safetensors").is_file()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".prev")) == []


def test_refuses_previous_output_of_other_method(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    out = tmp_path / "other"
    out.mkdir()
    (out / qe.MARKER_FILE).write_text(json.dumps({"method": "gguf-q4_k_m"}))
    with pytest.raises(BadRequest):
        run(source, output_dir=out)


def test_cancel_before_start(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    stop = threading.Event()
    stop.set()
    with pytest.raises(Cancelled):
        run(source, output_dir=tmp_path / "never", stop=stop)
    assert not (tmp_path / "never").exists()
    assert fake_mlx == []


def test_cancel_between_stages_discards_output(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    stop = threading.Event()
    out = tmp_path / "late"
    events: list[dict] = []

    def progress(event: dict) -> None:
        events.append(event)
        if event["stage"] == "convert":
            stop.set()

    with pytest.raises(Cancelled):
        qe.quantize(str(source), "mlx-4bit", model_id="org/Tiny-Model", output_dir=str(out), device="mps",
                    progress=progress, stop_event=stop)
    assert not out.exists()
    assert not list(tmp_path.glob(".late.*"))


def test_runner_failure_removes_partial_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: Path) -> None:
    def broken(source: Path, staging: Path, **_kwargs: Any) -> None:
        staging.mkdir(parents=True)
        (staging / "half.bin").write_bytes(b"x")
        raise RuntimeError("disk full")

    monkeypatch.setitem(qe._RUNNERS, "mlx-4bit", broken)
    monkeypatch.setattr(qe, "unavailable_reason", lambda method, device: None)
    with pytest.raises(RuntimeError, match="disk full"):
        run(source, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.*"))


def test_failing_progress_callback_does_not_abort(fake_mlx: list, tmp_path: Path, source: Path) -> None:
    def progress(_event: dict) -> None:
        raise ConnectionError("ui gone")

    result = qe.quantize(str(source), "mlx-4bit", model_id="org/Tiny-Model", output_dir=str(tmp_path / "o"),
                         device="mps", progress=progress, stop_event=threading.Event())
    assert Path(result["output_dir"]).is_dir()


@pytest.mark.parametrize(("model_id", "expected"), [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct"),
    ("/Users/me/models/My Model v2/", "My-Model-v2"),
    ("weird:name*", "weird-name"),
    ("", "model"),
])
def test_derived_name(model_id: str, expected: str) -> None:
    assert qe.derived_name(model_id) == expected


def test_calibration_texts_are_built_in() -> None:
    assert len(qe.CALIBRATION_TEXTS) >= 16
    assert all(isinstance(t, str) and t.strip() for t in qe.CALIBRATION_TEXTS)
