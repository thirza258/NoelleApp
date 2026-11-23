"""In-app quantization: turn a downloaded FP16/BF16 checkpoint into a 4-bit build.

Methods (each key doubles as the registry precision key, see
:data:`noelle.runtime.preflight.PRECISIONS`):

* ``mlx-4bit``     Apple Silicon: ``mlx_lm.convert`` (affine 4-bit, group size 64)
* ``gguf-q4_k_m``  llama.cpp: ``convert_hf_to_gguf.py`` to FP16, then ``llama-quantize`` Q4_K_M
* ``awq-4bit``     AutoAWQ (NVIDIA CUDA only)
* ``gptq-4bit``    GPTQModel (NVIDIA CUDA only)
* ``nf4``          bitsandbytes NF4 load + ``save_pretrained`` (NVIDIA CUDA only)

:func:`capabilities` runs on the worker's I/O thread, so it only inspects the
platform, ``PATH`` and import specs; it never imports torch, MLX or a
quantization library. :func:`quantize` runs on the GPU lane. Every method writes
into a fresh staging folder beside the target and moves it into place only
when it succeeds, so a failed or cancelled run never leaves a half-written
checkpoint behind. MLX and llama.cpp conversions run in child processes, which
keeps their memory out of the worker and lets Stop interrupt them mid-way.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from noelle import __version__, paths
from noelle.runtime.errors import BadRequest, Cancelled, ModelNotFound, NoelleRuntimeError, Unsupported

log = logging.getLogger("noelle.quantize")

Progress = Callable[[dict[str, Any]], None]

MARKER_FILE = "noelle-quantization.json"
MLX_BITS = 4
MLX_GROUP_SIZE = 64
GGUF_QUANT_TYPE = "Q4_K_M"
CUDA_GROUP_SIZE = 128
_BAR_RE = re.compile(r"\d+%\|")  # tqdm-style progress bars are throttled
_SCRIPT_NAMES = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
_QUANTIZE_NAMES = ("llama-quantize.exe",) if os.name == "nt" else ("llama-quantize",)
_DEVICE_LABELS = {"cuda": "NVIDIA CUDA", "mps": "Apple Metal", "cpu": "CPU"}


class QuantizationFailed(NoelleRuntimeError):
    """A conversion tool exited with an error."""


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    description: str
    engine: str  # mlx | llama_cpp | transformers
    steps: tuple[str, ...]


METHODS: dict[str, Method] = {
    m.key: m
    for m in (
        Method("mlx-4bit", "4-bit MLX",
               "Apple Silicon 4-bit (affine, group size 64) with mlx-lm; runs on the Apple GPU.",
               "mlx", ("prepare", "convert", "finalize")),
        Method("gguf-q4_k_m", "GGUF Q4_K_M",
               "llama.cpp GGUF: FP16 conversion with convert_hf_to_gguf.py, then llama-quantize Q4_K_M.",
               "llama_cpp", ("prepare", "convert", "quantize", "finalize")),
        Method("awq-4bit", "4-bit AWQ",
               "Activation-aware weight quantization (AutoAWQ, GEMM kernels, group size 128).",
               "transformers", ("prepare", "load", "quantize", "save", "finalize")),
        Method("gptq-4bit", "4-bit GPTQ",
               "GPTQ post-training quantization with GPTQModel (group size 128).",
               "transformers", ("prepare", "load", "quantize", "save", "finalize")),
        Method("nf4", "4-bit NF4 (bitsandbytes)",
               "bitsandbytes NF4 with double quantization, saved as a pre-quantized checkpoint.",
               "transformers", ("prepare", "quantize", "save", "finalize")),
    )
}

# method -> (top-level import name, pip distribution)
_CUDA_PACKAGES: dict[str, tuple[str, str]] = {
    "awq-4bit": ("awq", "autoawq"),
    "gptq-4bit": ("gptqmodel", "gptqmodel"),
    "nf4": ("bitsandbytes", "bitsandbytes"),
}

# Generic calibration text for AWQ / GPTQ. Small on purpose: nothing is downloaded.
CALIBRATION_TEXTS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "The committee will meet on Tuesday to review the budget and approve the new hiring plan.",
    "To install the package, open a terminal, create a virtual environment and run pip install.",
    "Water boils at one hundred degrees Celsius at sea level, but at a lower temperature in the mountains.",
    "She opened the window, let the cold morning air in, and started writing the first chapter of her novel.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "The French Revolution began in 1789 and transformed the political landscape of Europe.",
    "Regular exercise improves cardiovascular health, sleep quality and mood.",
    "A neural network learns by adjusting its weights to minimise a loss function with gradient descent.",
    "Please confirm your reservation by replying to this email before Friday at noon.",
    "The recipe calls for two cups of flour, one egg, a pinch of salt and a tablespoon of butter.",
    "Mount Everest, on the border between Nepal and China, is the highest mountain above sea level.",
    "The stock market closed higher today as investors reacted to better than expected earnings.",
    "In mathematics, a prime number has exactly two distinct divisors: one and itself.",
    "SELECT name, email FROM customers WHERE created_at > '2024-01-01' ORDER BY name;",
    "The orchestra tuned their instruments as the audience slowly settled into their seats.",
    "Climate change is driven largely by greenhouse gas emissions from burning fossil fuels.",
    "He forgot his umbrella at the station, so he walked home through the rain.",
    "The museum's new exhibition features paintings from the Dutch Golden Age.",
    "Customer support is available twenty-four hours a day through chat, phone and email.",
    "Light travels at roughly three hundred thousand kilometres per second in a vacuum.",
    "Q: What is the capital of Japan?\nA: The capital of Japan is Tokyo.",
    "The children built a sandcastle with towers, a moat and a small flag made from a leaf.",
    "Version control systems such as Git record every change made to the source code.",
    "The doctor recommended rest, plenty of fluids and a follow-up visit in two weeks.",
    "Bees pollinate flowers, which helps plants reproduce and produce fruit.",
    "The train to Berlin departs from platform nine at a quarter past eight.",
    "An operating system manages hardware resources and provides services to applications.",
    "The detective examined the footprints in the snow and realised the thief had walked backwards.",
    "Good documentation explains not only what the code does but also why it was written that way.",
    "Saving a small amount every month can grow into a substantial sum thanks to compound interest.",
)


# ---------------------------------------------------------------------------
# capabilities (I/O thread: no heavy imports)
# ---------------------------------------------------------------------------


def _has_module(name: str) -> bool:
    """``find_spec`` on a top-level name; never imports a package (dotted names would)."""
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def runtime_label(device: str) -> str:
    return _DEVICE_LABELS.get(device, device or "unknown")


@dataclass
class LlamaCppTools:
    convert_script: Path | None
    quantize_binary: Path | None
    python: str | None
    gguf_package: bool
    searched: list[str] = field(default_factory=list)
    python_args: list[str] = field(default_factory=list)  # e.g. ["pyrun"] for the frozen app binary

    def missing_reason(self) -> str | None:
        missing = [name for name, found in (("convert_hf_to_gguf.py", self.convert_script),
                                            ("llama-quantize", self.quantize_binary)) if found is None]
        if missing:
            where = ", ".join(self.searched)
            return (f"llama.cpp tools not found: {' and '.join(missing)}. Noelle can download them for you "
                    f"(Settings › Components), or set quantization.llama_cpp_dir to a llama.cpp folder. "
                    f"Searched quantization.llama_cpp_dir, PATH and {where}.")
        if self.python is None:
            return "No Python interpreter is available to run convert_hf_to_gguf.py."
        if not self.gguf_package:
            return (f"{self.convert_script} needs the 'gguf' Python package: install it with "
                    f'pip install "noelle[gguf]" or point quantization.llama_cpp_dir at a llama.cpp checkout '
                    f"that contains gguf-py/.")
        return None


def _configured_llama_dir() -> str | None:
    try:
        from noelle.config import get_settings

        value = get_settings().get("quantization.llama_cpp_dir")
    except Exception:  # a broken settings file must not break capability probing
        return None
    return str(value) if value else None


def _common_llama_dirs() -> list[Path]:
    home = Path.home()
    return [Path("/opt/homebrew/bin"), Path("/opt/homebrew/opt/llama.cpp/bin"), Path("/usr/local/bin"),
            Path("/usr/local/opt/llama.cpp/bin"), Path("/home/linuxbrew/.linuxbrew/bin"),
            home / "llama.cpp", home / "llama.cpp" / "build" / "bin"]


def _find_in(dirs: list[Path], names: tuple[str, ...], *, executable: bool) -> Path | None:
    for folder in dirs:
        for name in names:
            candidate = folder / name
            if candidate.is_file() and (not executable or os.access(candidate, os.X_OK)):
                return candidate
    return None


def _which(names: tuple[str, ...]) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def find_llama_cpp(llama_cpp_dir: str | os.PathLike[str] | None = None) -> LlamaCppTools:
    """Locate ``convert_hf_to_gguf.py`` and ``llama-quantize``.

    Order: ``llama_cpp_dir`` (default: settings ``quantization.llama_cpp_dir``),
    then ``PATH``, then common Homebrew / checkout locations.
    """
    configured = str(llama_cpp_dir) if llama_cpp_dir is not None else _configured_llama_dir()
    preferred: list[Path] = []
    if configured:
        root = Path(configured).expanduser()
        preferred = [root, root / "bin", root / "build" / "bin", root / "build" / "bin" / "Release"]
    if llama_cpp_dir is None:  # the in-app download, read from disk so a running worker sees it immediately
        from noelle.runtime.llama_tools import installed_info

        info = installed_info()
        if info is not None:
            preferred = [Path(info["path"]), Path(info["path"]) / "bin", *preferred]
    common = _common_llama_dirs()
    script = (_find_in(preferred, _SCRIPT_NAMES, executable=False) or _which(_SCRIPT_NAMES)
              or _find_in(common, _SCRIPT_NAMES, executable=False))
    binary = (_find_in(preferred, _QUANTIZE_NAMES, executable=True) or _which(_QUANTIZE_NAMES)
              or _find_in(common, _QUANTIZE_NAMES, executable=True))
    # A frozen app has no separate python; its own binary runs scripts via the hidden `pyrun` subcommand,
    # which also gives the converter the bundled torch/numpy/transformers.
    python = sys.executable
    python_args = ["pyrun"] if getattr(sys, "frozen", False) else []
    gguf_ok = script is not None and ((script.parent / "gguf-py").is_dir() or _has_module("gguf"))
    return LlamaCppTools(script, binary, python, gguf_ok, [str(p) for p in preferred + common], python_args)


def unavailable_reason(method: str, device: str) -> str | None:
    """Why ``method`` cannot run on this runtime, or None when it can."""
    if method == "mlx-4bit":
        if not is_apple_silicon():
            return f"Requires an Apple Silicon Mac (this runtime: {runtime_label(device)} on {platform.machine()})."
        missing = [name for name in ("mlx", "mlx_lm") if not _has_module(name)]
        if missing:
            return (f"MLX is not installed in the runtime environment ({', '.join(missing)} missing). "
                    f'Install it with: pip install "noelle[mlx]"')
        return None
    if method == "gguf-q4_k_m":
        return find_llama_cpp().missing_reason()
    module, dist = _CUDA_PACKAGES[method]
    if device != "cuda":
        return f"Requires an NVIDIA CUDA GPU (this runtime: {runtime_label(device)})."
    if not _has_module(module):
        return (f"Python package '{dist}' is not installed in the runtime environment. "
                f'Install it with: pip install "noelle[cuda-quant]"')
    return None


def capabilities(device: str) -> dict[str, dict[str, Any]]:
    """Every method with ``available`` / ``reason`` for ``device``. Cheap: safe on the I/O thread."""
    result: dict[str, dict[str, Any]] = {}
    for key, method in METHODS.items():
        reason = unavailable_reason(key, device)
        result[key] = {"label": method.label, "description": method.description, "available": reason is None,
                       "reason": reason, "engine": method.engine, "precision": key}
    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Reporter:
    """Turns stages and tool output into ``{"stage", "message", "current", "total"}`` dicts."""

    def __init__(self, progress: Progress | None, steps: tuple[str, ...]) -> None:
        self._progress = progress
        self.steps = steps
        self.current = 0
        self._last_bar = 0.0

    def step(self, stage: str, message: str) -> None:
        if stage in self.steps:
            self.current = self.steps.index(stage) + 1
        self.emit(stage, message)

    def line(self, stage: str, line: str) -> None:
        text = line.strip()
        if not text:
            return
        if _BAR_RE.search(text):
            now = time.monotonic()
            if now - self._last_bar < 0.25:
                return
            self._last_bar = now
        self.emit(stage, text[:500])

    def emit(self, stage: str, message: str) -> None:
        if self._progress is None:
            return
        try:
            self._progress({"stage": stage, "message": message, "current": self.current, "total": len(self.steps)})
        except Exception:  # progress delivery must never abort a conversion
            log.debug("progress callback failed", exc_info=True)


def derived_name(model_id: str) -> str:
    """Folder-safe short name: ``HuggingFaceTB/SmolLM2-135M`` -> ``SmolLM2-135M``."""
    tail = re.split(r"[\\/]", str(model_id).rstrip("/\\"))[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.") or "model"


def folder_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def read_marker(folder: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((folder / MARKER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _check_stop(stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise Cancelled("Quantization was cancelled.")


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise ModelNotFound(f"Model folder not found: {source}", hint="Download the model first.")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise BadRequest(f"{source} is not a Hugging Face checkpoint (config.json is missing).")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BadRequest(f"Unreadable config.json in {source}: {exc}") from exc
    if isinstance(config.get("quantization"), dict) or config.get("quantization_config"):
        raise BadRequest("This checkpoint is already quantized.",
                         hint="Quantize the original FP16/BF16 model instead.")
    if not any(source.glob("*.safetensors")) and not any(source.glob("*.bin")):
        raise BadRequest(f"No weights (*.safetensors or *.bin) found in {source}.")


def check_output_dir(target: Path, method: str) -> None:
    """Refuse a non-empty ``target`` unless it holds a previous output of ``method``."""
    if not target.exists():
        return
    if not target.is_dir():
        raise BadRequest(f"Output path {target} exists and is not a folder.")
    if not any(target.iterdir()):
        return
    marker = read_marker(target)
    if marker is not None and marker.get("method") == method:
        return
    raise BadRequest(f"Output folder {target} is not empty.",
                     hint="Pick an empty folder, or delete the existing one first.")


def _install(staging: Path, target: Path, method: str) -> None:
    """Move the finished staging folder to ``target``, replacing a previous output of the same method."""
    check_output_dir(target, method)
    if target.is_dir() and any(target.iterdir()):
        backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex[:8]}"
        os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return
    if target.is_dir():
        target.rmdir()
    os.replace(staging, target)


def _tool_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run_tool(cmd: list[str], *, stage: str, tool: str, report: _Reporter, stop_event: threading.Event) -> None:
    from noelle.runtime.environment_manager import run_streaming

    code, tail = run_streaming(cmd, on_line=lambda line: report.line(stage, line), stop_event=stop_event,
                               env=_tool_env())
    if code != 0:
        detail = next((line.strip() for line in reversed(tail) if line.strip()), "no output")
        raise QuantizationFailed(f"{tool} failed (exit code {code}): {detail}")


def _is_mlx_quantized(folder: Path) -> bool:
    try:
        quant = json.loads((folder / "config.json").read_text(encoding="utf-8")).get("quantization")
    except (OSError, ValueError, AttributeError):
        return False
    return isinstance(quant, dict) and "bits" in quant and "group_size" in quant


# ---------------------------------------------------------------------------
# method runners: (source, staging, ...) -> gguf file name or None
# ---------------------------------------------------------------------------

_MLX_CONVERT = "import json, sys\nfrom mlx_lm import convert\nconvert(**json.loads(sys.argv[1]))\n"


def _quantize_mlx(source: Path, staging: Path, *, name: str, device: str, report: _Reporter,
                  stop_event: threading.Event) -> str | None:
    report.step("convert", f"Converting with mlx-lm ({MLX_BITS}-bit, group size {MLX_GROUP_SIZE})…")
    kwargs = {"hf_path": str(source), "mlx_path": str(staging), "quantize": True, "q_bits": MLX_BITS,
              "q_group_size": MLX_GROUP_SIZE}
    if getattr(sys, "frozen", False):  # no separate interpreter in a frozen build: convert in-process
        from mlx_lm import convert

        convert(**kwargs)
    else:
        _run_tool([sys.executable, "-c", _MLX_CONVERT, json.dumps(kwargs)], stage="convert",
                  tool="mlx_lm.convert", report=report, stop_event=stop_event)
    if not _is_mlx_quantized(staging):
        raise QuantizationFailed("mlx_lm.convert finished but the output has no quantization config.")
    return None


def _quantize_gguf(source: Path, staging: Path, *, name: str, device: str, report: _Reporter,
                   stop_event: threading.Event) -> str | None:
    tools = find_llama_cpp()
    reason = tools.missing_reason()
    if reason:
        raise Unsupported(reason)
    assert tools.convert_script and tools.quantize_binary and tools.python
    staging.mkdir(parents=True)
    f16 = staging / f"{name}-f16.gguf"
    final = staging / f"{name}-{GGUF_QUANT_TYPE}.gguf"
    report.step("convert", f"Converting to an FP16 GGUF with {tools.convert_script.name}…")
    _run_tool([tools.python, *tools.python_args, str(tools.convert_script), str(source), "--outfile", str(f16),
               "--outtype", "f16"],
              stage="convert", tool=tools.convert_script.name, report=report, stop_event=stop_event)
    if not f16.is_file():
        raise QuantizationFailed(f"{tools.convert_script.name} did not write {f16.name}.")
    _check_stop(stop_event)
    report.step("quantize", f"Quantizing to {GGUF_QUANT_TYPE} with llama-quantize…")
    _run_tool([str(tools.quantize_binary), str(f16), str(final), GGUF_QUANT_TYPE], stage="quantize",
              tool="llama-quantize", report=report, stop_event=stop_event)
    if not final.is_file():
        raise QuantizationFailed(f"llama-quantize did not write {final.name}.")
    f16.unlink(missing_ok=True)  # the FP16 intermediate is several times larger than the result
    return final.name


def _calibration_seq_len(tokenizer: Any) -> int:
    total = sum(len(tokenizer(text)["input_ids"]) for text in CALIBRATION_TEXTS)
    return max(16, min(512, total // 8))


def _quantize_awq(source: Path, staging: Path, *, name: str, device: str, report: _Reporter,
                  stop_event: threading.Event) -> str | None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    report.step("load", "Loading the model for AWQ calibration…")
    model = AutoAWQForCausalLM.from_pretrained(str(source), low_cpu_mem_usage=True, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    _check_stop(stop_event)
    report.step("quantize", f"Calibrating on {len(CALIBRATION_TEXTS)} built-in samples and quantizing (AWQ)…")
    quant_config = {"zero_point": True, "q_group_size": CUDA_GROUP_SIZE, "w_bit": 4, "version": "GEMM"}
    model.quantize(tokenizer, quant_config=quant_config, calib_data=list(CALIBRATION_TEXTS),
                   max_calib_samples=len(CALIBRATION_TEXTS), max_calib_seq_len=_calibration_seq_len(tokenizer))
    _check_stop(stop_event)
    report.step("save", "Saving the AWQ checkpoint…")
    staging.mkdir(parents=True)
    model.save_quantized(str(staging))
    tokenizer.save_pretrained(str(staging))
    return None


def _quantize_gptq(source: Path, staging: Path, *, name: str, device: str, report: _Reporter,
                   stop_event: threading.Event) -> str | None:
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    report.step("load", "Loading the model for GPTQ…")
    model = GPTQModel.load(str(source), QuantizeConfig(bits=4, group_size=CUDA_GROUP_SIZE))
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    _check_stop(stop_event)
    report.step("quantize", f"Calibrating on {len(CALIBRATION_TEXTS)} built-in samples and quantizing (GPTQ)…")
    model.quantize(list(CALIBRATION_TEXTS), batch_size=1)
    _check_stop(stop_event)
    report.step("save", "Saving the GPTQ checkpoint…")
    staging.mkdir(parents=True)
    model.save(str(staging))
    tokenizer.save_pretrained(str(staging))
    return None


def _quantize_nf4(source: Path, staging: Path, *, name: str, device: str, report: _Reporter,
                  stop_event: threading.Event) -> str | None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    report.step("quantize", "Loading the model in 4-bit NF4 (bitsandbytes)…")
    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=compute)
    model = AutoModelForCausalLM.from_pretrained(str(source), quantization_config=config, device_map="auto",
                                                 dtype=compute)
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    try:
        _check_stop(stop_event)
        report.step("save", "Saving the NF4 checkpoint…")
        staging.mkdir(parents=True)
        model.save_pretrained(str(staging))
        tokenizer.save_pretrained(str(staging))
    finally:
        del model
        torch.cuda.empty_cache()
    return None


_RUNNERS: dict[str, Callable[..., str | None]] = {
    "mlx-4bit": _quantize_mlx,
    "gguf-q4_k_m": _quantize_gguf,
    "awq-4bit": _quantize_awq,
    "gptq-4bit": _quantize_gptq,
    "nf4": _quantize_nf4,
}

_HINTS = {
    "mlx-4bit": 'On Apple Silicon install the MLX extra: pip install "noelle[mlx]"',
    "gguf-q4_k_m": "Install llama.cpp (brew install llama.cpp) or set quantization.llama_cpp_dir.",
    "awq-4bit": "On Apple Silicon use 'mlx-4bit' instead.",
    "gptq-4bit": "On Apple Silicon use 'mlx-4bit' instead.",
    "nf4": "On Apple Silicon use 'mlx-4bit' instead.",
}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def quantize(source_dir: str, method: str, *, model_id: str, output_dir: str | None, device: str,
             progress: Progress, stop_event: threading.Event) -> dict[str, Any]:
    """Quantize the checkpoint in ``source_dir`` with ``method``.

    Returns ``{output_dir, method, precision, engine, size_bytes, gguf_file,
    derived_id, base_model, elapsed_ms}``. Raises :class:`Unsupported` when the
    method cannot run here and :class:`Cancelled` when ``stop_event`` is set.
    """
    started = time.perf_counter()
    spec = METHODS.get(method)
    if spec is None:
        raise BadRequest(f"Unknown quantization method '{method}'. Choose one of: {', '.join(METHODS)}.")
    reason = unavailable_reason(method, device)
    if reason:
        raise Unsupported(reason, hint=_HINTS.get(method))
    source = Path(source_dir).expanduser()
    _validate_source(source)
    name = derived_name(model_id or source.name)
    target = (Path(output_dir).expanduser() if output_dir else paths.models_dir() / f"{name}-{method}").absolute()
    check_output_dir(target, method)
    report = _Reporter(progress, spec.steps)
    report.step("prepare", f"Quantizing {model_id} to {spec.label} in {target}")
    _check_stop(stop_event)
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(f".{target.name}.partial-*"):  # left behind by a crashed worker
        shutil.rmtree(stale, ignore_errors=True)
    staging = target.parent / f".{target.name}.partial-{uuid.uuid4().hex[:8]}"
    try:
        gguf_file = _RUNNERS[method](source, staging, name=name, device=device, report=report,
                                     stop_event=stop_event)
        _check_stop(stop_event)
        report.step("finalize", "Finalizing the checkpoint…")
        marker = {"method": method, "precision": method, "engine": spec.engine, "base_model": model_id,
                  "source_dir": str(source), "gguf_file": gguf_file, "created_at": time.time(),
                  "noelle_version": __version__}
        (staging / MARKER_FILE).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        _install(staging, target, method)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    size = folder_size(target)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    report.emit("finalize", f"Saved {size / 1024**2:.1f} MB to {target} in {elapsed_ms / 1000:.1f}s")
    return {
        "output_dir": str(target),
        "method": method,
        "precision": method,
        "engine": spec.engine,
        "size_bytes": size,
        "gguf_file": gguf_file,
        "derived_id": f"local/{name}-{method}",
        "base_model": model_id,
        "elapsed_ms": elapsed_ms,
    }
