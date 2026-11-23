"""Runtime and quantization advisor: "how well will this model run here, and how should I run it?"

Builds on :mod:`noelle.runtime.capabilities` (what a model is) and
:mod:`noelle.runtime.preflight` (memory arithmetic) and adds:

* **run options** — every precision/runtime route for the model on this machine,
  with its memory need, whether it fits, and whether it is ready, needs a local
  build (Optimize tab), needs an optional component, or cannot run on this device;
* a **recommendation** per runtime profile (Balanced, Quality, Fast, Low memory,
  Max context);
* a **0–100 compatibility score** made of visible parts (memory fit, runtime,
  precision, context, trust) rather than one opaque number;
* the **largest context** whose KV cache still fits;
* **speed**: measured on this machine when the Compare page or the Evaluation Lab
  has run the model, otherwise an explicitly labelled bandwidth estimate.

Pure Python (no torch, no Qt).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from noelle.runtime import tasks
from noelle.runtime.capabilities import ModelCapability
from noelle.runtime.preflight import (
    FIXED_RUNTIME_OVERHEAD,
    Compatibility,
    MemoryBudget,
    assess_compatibility,
    estimate_runtime_bytes,
    format_bytes,
    kv_cache_bytes_per_token,
    precision_spec,
    weights_bytes,
)

READY, BUILD, INSTALL, UNSUPPORTED = "ready", "build", "install", "unsupported"
_STATUS_RANK = {READY: 0, BUILD: 1, INSTALL: 2, UNSUPPORTED: 3}

REFERENCE, NEAR_LOSSLESS, SMALL_LOSS, NOTICEABLE_LOSS = "Reference", "Near-lossless", "Small loss", "Noticeable loss"
_QUALITY_RANK = {REFERENCE: 3, NEAR_LOSSLESS: 2, SMALL_LOSS: 1, NOTICEABLE_LOSS: 0}

PROFILES: dict[str, str] = {
    "balanced": "Balanced",
    "quality": "Quality",
    "fast": "Fast",
    "low_memory": "Low memory",
    "max_context": "Max context",
}
PROFILE_HINTS: dict[str, str] = {
    "balanced": "Full precision when it fits comfortably, otherwise the best 4-bit build for this device.",
    "quality": "Highest-precision build that fits, even if it is slower.",
    "fast": "Smallest weights on the fastest engine for this device; shorter replies.",
    "low_memory": "Smallest memory footprint, leaving room for other apps and models.",
    "max_context": "Small weights so the KV cache can hold the longest context that fits.",
}
PROFILE_PARAMS: dict[str, dict[str, Any]] = {
    "balanced": {},
    "quality": {"max_new_tokens": 1024},
    "fast": {"max_new_tokens": 256},
    "low_memory": {"max_new_tokens": 256},
    "max_context": {"max_new_tokens": 2048},
}
CONTEXT_STEPS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)

# Peak memory bandwidth in GB/s (vendor figures). Only used for labelled speed estimates;
# entries are matched against the device name, longest key first.
MEMORY_BANDWIDTH_GBPS: dict[str, float] = {
    "M1 Ultra": 800, "M1 Max": 400, "M1 Pro": 200, "M1": 68,
    "M2 Ultra": 800, "M2 Max": 400, "M2 Pro": 200, "M2": 100,
    "M3 Ultra": 819, "M3 Max": 300, "M3 Pro": 150, "M3": 100,
    "M4 Max": 410, "M4 Pro": 273, "M4": 120, "M5": 153,
    "RTX 5090": 1792, "RTX 5080": 960, "RTX 5070 Ti": 896, "RTX 5070": 672, "RTX 5060 Ti": 448,
    "RTX 4090": 1008, "RTX 4080 SUPER": 736, "RTX 4080": 717, "RTX 4070 Ti SUPER": 672, "RTX 4070 Ti": 504,
    "RTX 4070 SUPER": 504, "RTX 4070": 504, "RTX 4060 Ti": 288, "RTX 4060": 272,
    "RTX 3090 Ti": 1008, "RTX 3090": 936, "RTX 3080 Ti": 912, "RTX 3080": 760, "RTX 3070 Ti": 608,
    "RTX 3070": 448, "RTX 3060 Ti": 448, "RTX 3060": 360, "RTX 2080 Ti": 616, "RTX 2080": 448, "RTX 2070": 448,
    "RTX 2060": 336, "GTX 1080 Ti": 484, "GTX 1660": 192,
    "H100": 2000, "A100": 1555, "A10G": 600, "A10": 600, "L40S": 864, "L4": 300, "T4": 320, "V100": 900,
}
# (low, high) fraction of peak bandwidth a decode step reaches, and fixed seconds per token.
_EFFICIENCY: dict[tuple[str, str], tuple[float, float, float]] = {
    ("mps", "mlx"): (0.55, 0.8, 0.003),
    ("mps", "llama.cpp"): (0.5, 0.75, 0.003),
    ("mps", "transformers"): (0.2, 0.45, 0.02),
    ("cuda", "transformers"): (0.45, 0.7, 0.012),
    ("cuda", "transformers-bnb"): (0.15, 0.35, 0.02),
}


@dataclass(frozen=True)
class HardwareProfile:
    device: str  # cuda | mps | cpu
    name: str
    budget: MemoryBudget
    backends: Mapping[str, bool] = field(default_factory=dict)
    bandwidth_gbps: float | None = None

    @property
    def usable_bytes(self) -> int:
        return self.budget.usable_bytes

    def has(self, backend: str) -> bool:
        return bool(self.backends.get(backend))


def memory_bandwidth(name: str | None, device: str) -> float | None:
    """Peak bandwidth for a known accelerator name, else None (no estimate is better than a wrong one)."""
    if not name or device not in ("mps", "cuda"):
        return None
    upper = name.upper()
    for key in sorted(MEMORY_BANDWIDTH_GBPS, key=len, reverse=True):
        token = key.upper()
        index = upper.find(token)
        if index < 0:
            continue
        after = upper[index + len(token):index + len(token) + 1]
        if after.isalnum():  # "M2" must not match "M2 PRO" or "M20"; "RTX 4070" not "RTX 40700"
            continue
        return MEMORY_BANDWIDTH_GBPS[key]
    return None


@dataclass(frozen=True)
class Measured:
    runs: int
    tokens_per_second: float | None
    ttft_ms: float | None
    precision: str | None = None
    engine: str | None = None
    device: str | None = None

    def describe(self) -> str:
        parts = []
        if self.tokens_per_second:
            parts.append(f"{self.tokens_per_second:.1f} tok/s")
        if self.ttft_ms:
            parts.append(f"TTFT {self.ttft_ms:.0f} ms")
        where = " · ".join(p for p in (self.precision, self.engine, self.device) if p)
        runs = f"{self.runs} run{'s' if self.runs != 1 else ''}"
        return f"{' · '.join(parts) or 'no timing'} ({runs}{', ' + where if where else ''})"


def summarize_measurements(rows: Iterable[Mapping[str, Any]]) -> dict[str, Measured]:
    """Median speed per precision from Compare / Evaluation Lab rows for one model."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in (None, "finished", "ok"):
            continue
        if not row.get("tokens_per_second") and not row.get("ttft_ms"):
            continue
        groups.setdefault(str(row.get("precision") or "auto"), []).append(row)
    result: dict[str, Measured] = {}
    for precision, items in groups.items():
        tps = [float(r["tokens_per_second"]) for r in items if r.get("tokens_per_second")]
        ttft = [float(r["ttft_ms"]) for r in items if r.get("ttft_ms")]
        metrics = items[0].get("metrics") if isinstance(items[0].get("metrics"), Mapping) else {}
        result[precision] = Measured(len(items), statistics.median(tps) if tps else None,
                                     statistics.median(ttft) if ttft else None, precision,
                                     items[0].get("engine") or metrics.get("engine"),
                                     items[0].get("device") or metrics.get("device"))
    return result


@dataclass(frozen=True)
class RunOption:
    key: str
    label: str
    runtime: str
    precision: str
    memory_bytes: int | None
    fits: bool | None
    status: str  # ready | build | install | unsupported
    reason: str | None
    quality: str
    speed: str
    weight_bytes: int | None = None
    tokens_per_second: tuple[float, float] | None = None  # estimate range
    local_variant: str | None = None  # repo_id of an existing local build
    gguf_file: str | None = None

    @property
    def runnable_now(self) -> bool:
        return self.status == READY and self.fits is not False

    @property
    def action(self) -> str | None:
        if self.status == BUILD:
            return "quantize"
        if self.status == INSTALL:
            return "install"
        if self.status == READY:
            return "use" if self.local_variant else "load"
        return None

    def estimate_text(self) -> str | None:
        if not self.tokens_per_second:
            return None
        low, high = self.tokens_per_second
        return f"~{low:.0f}–{high:.0f} tok/s (estimate)"


@dataclass(frozen=True)
class ScoreItem:
    key: str
    label: str
    points: int
    max_points: int
    note: str


@dataclass
class Assessment:
    capability: ModelCapability
    hardware: HardwareProfile
    compatibility: Compatibility
    options: list[RunOption]
    recommended: RunOption | None
    profile: str
    score: int
    grade: str
    items: list[ScoreItem]
    cap_note: str | None
    context_tokens: int | None
    measured: dict[str, Measured]
    notes: list[str]

    @property
    def level(self) -> str:
        if self.score >= 70:
            return "green"
        if self.score >= 45:
            return "yellow"
        return "red"

    def measured_for(self, option: RunOption | None) -> Measured | None:
        """Measurements for ``option``; "auto" runs count for the device's default full-precision build."""
        if option is None:
            return None
        found = self.measured.get(option.precision)
        if found is None and option.quality == REFERENCE and option.runtime == "transformers":
            found = self.measured.get("auto")
        return found

    @property
    def measured_for_recommended(self) -> Measured | None:
        return self.measured_for(self.recommended)

    def speed_text(self) -> str:
        measured = self.measured_for_recommended
        if measured and measured.tokens_per_second:
            return f"{measured.tokens_per_second:.1f} tok/s measured on this machine"
        if self.recommended is not None and self.recommended.estimate_text():
            return self.recommended.estimate_text() or ""
        return "Not measured yet — run it in Compare or the Evaluation Lab."

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.capability.model_id,
            "score": self.score,
            "grade": self.grade,
            "level": self.level,
            "profile": self.profile,
            "verdict": self.compatibility.label,
            "breakdown": [item.__dict__ for item in self.items],
            "cap_note": self.cap_note,
            "recommended": _option_dict(self.recommended),
            "options": [_option_dict(o) for o in self.options],
            "context_tokens": self.context_tokens,
            "speed": self.speed_text(),
            "notes": list(self.notes),
            "hardware": {"device": self.hardware.device, "name": self.hardware.name,
                         "usable_bytes": self.hardware.usable_bytes},
        }


def _option_dict(option: RunOption | None) -> dict[str, Any] | None:
    if option is None:
        return None
    data = dict(option.__dict__)
    data["action"] = option.action
    data["estimate"] = option.estimate_text()
    return data


# ---------------------------------------------------------------------------
# run options
# ---------------------------------------------------------------------------


def run_options(cap: ModelCapability, hw: HardwareProfile, *, config: dict[str, Any] | None = None,
                local_variants: Mapping[str, str] | None = None) -> list[RunOption]:
    """Every way to run ``cap`` on ``hw``. ``local_variants`` maps precision -> repo_id of local builds."""
    local = dict(local_variants or {})
    params = cap.param_count
    options: list[RunOption] = []
    if not cap.runtimes and not cap.conversions:
        return options
    if cap.quantization == "gguf":
        return _gguf_options(cap, hw, config)
    if cap.quantization is not None:
        return [_prequantized_option(cap, hw, config)]

    def sized(precision: str) -> tuple[int | None, int | None]:
        if not params:
            if cap.weight_bytes and not precision_spec(precision).quantized:
                # parameter count unknown: use the published file size (conservative if stored as FP32)
                return estimate_runtime_bytes(None, config=config, weight_bytes=cap.weight_bytes), cap.weight_bytes
            return None, None
        weights = weights_bytes(params, precision)
        return estimate_runtime_bytes(params, precision, config=config), weights

    runtime = cap.runtimes[0] if cap.runtimes else "transformers"
    if "fp16" in cap.precisions or "fp32" in cap.precisions:
        floats = ["fp32"] if hw.device == "cpu" else ["fp16", "bf16"] if hw.device == "cuda" else ["fp16"]
        if cap.family not in (tasks.TEXT, tasks.VISION):
            floats = floats[:1]
        for key in floats:
            memory, weights = sized(key)
            label = {"fp32": "FP32", "fp16": "FP16", "bf16": "BF16"}[key]
            options.append(_option(key, label, runtime, key, memory, weights, hw, READY, None, REFERENCE,
                                   _speed(hw.device, runtime, key)))
    elif cap.runtimes:  # task families without a precision choice load at the engine's default
        memory, weights = sized("fp32" if hw.device == "cpu" else "fp16")
        options.append(_option("auto", "Default precision", runtime, "auto", memory, weights, hw, READY, None,
                               REFERENCE, _speed(hw.device, runtime, "auto")))
    if cap.family in (tasks.TEXT, tasks.VISION) and "int8" in cap.precisions:
        for key, quality in (("int8", NEAR_LOSSLESS), ("nf4", SMALL_LOSS)):
            memory, weights = sized(key)
            status, reason = _bnb_status(hw)
            options.append(_option(key, precision_spec(key).label, "transformers", key, memory, weights, hw, status,
                                   reason, quality, _speed(hw.device, "transformers-bnb", key),
                                   speed_key="transformers-bnb", local_variant=local.get(key)))
    for key in cap.conversions:
        if key == "nf4":
            continue  # offered above as a load-time precision
        memory, weights = sized(key)
        status, reason, runtime_name = _conversion_status(key, hw)
        if key in local and status == BUILD:
            status, reason = READY, None
        options.append(_option(key, precision_spec(key).label, runtime_name, key, memory, weights, hw, status, reason,
                               SMALL_LOSS, _speed(hw.device, runtime_name, key), local_variant=local.get(key)))
    return options


def _option(key: str, label: str, runtime: str, precision: str, memory: int | None, weights: int | None,
            hw: HardwareProfile, status: str, reason: str | None, quality: str, speed: str,
            local_variant: str | None = None, gguf_file: str | None = None, speed_key: str | None = None) -> RunOption:
    fits = None if memory is None else memory <= hw.usable_bytes
    if memory is not None and not fits and status != UNSUPPORTED:
        reason = reason or f"Needs ~{format_bytes(memory)}; {format_bytes(hw.usable_bytes)} usable."
    estimate = _estimate_tps(hw, speed_key or runtime, weights) if status != UNSUPPORTED else None
    return RunOption(key, label, runtime, precision, memory, fits, status, reason, quality, speed, weights, estimate,
                     local_variant, gguf_file)


def _bnb_status(hw: HardwareProfile) -> tuple[str, str | None]:
    if hw.device != "cuda":
        return UNSUPPORTED, "bitsandbytes needs an NVIDIA CUDA GPU."
    if not hw.has("bitsandbytes"):
        return INSTALL, 'Install bitsandbytes: pip install "noelle[cuda-quant]".'
    return READY, None


def _conversion_status(key: str, hw: HardwareProfile) -> tuple[str, str | None, str]:
    if key == "mlx-4bit":
        if hw.device != "mps":
            return UNSUPPORTED, "MLX runs only on Apple Silicon.", "mlx"
        if not hw.has("mlx"):
            return INSTALL, 'Install MLX: pip install "noelle[mlx]".', "mlx"
        return BUILD, "Build it once in the Optimize tab.", "mlx"
    if key == "gguf-q4_k_m":
        if not hw.has("llama.cpp-tools"):
            return INSTALL, "Needs the llama.cpp tools (Settings › Components asks before downloading).", "llama.cpp"
        if not hw.has("llama.cpp"):
            return INSTALL, "Needs llama-server or llama-cpp-python to run GGUF files.", "llama.cpp"
        return BUILD, "Build it once in the Optimize tab.", "llama.cpp"
    package = {"awq-4bit": "awq", "gptq-4bit": "gptq"}.get(key, key)
    if hw.device != "cuda":
        return UNSUPPORTED, f"{precision_spec(key).label} needs an NVIDIA CUDA GPU.", "transformers"
    if not hw.has(package):
        return INSTALL, f'Install the {package.upper()} package: pip install "noelle[cuda-quant]".', "transformers"
    return BUILD, "Build it once in the Optimize tab.", "transformers"


def _prequantized_option(cap: ModelCapability, hw: HardwareProfile, config: dict[str, Any] | None) -> RunOption:
    method = cap.quantization or ""
    memory = None
    if cap.weight_bytes or cap.param_count:
        memory = estimate_runtime_bytes(cap.param_count, config=config, weight_bytes=cap.weight_bytes) \
            if cap.weight_bytes else estimate_runtime_bytes(cap.param_count, "nf4", config=config)
    weights = cap.weight_bytes or (weights_bytes(cap.param_count, "nf4") if cap.param_count else None)
    label = {"mlx": "MLX (pre-quantized)", "awq": "AWQ (pre-quantized)", "gptq": "GPTQ (pre-quantized)",
             "bnb": "bitsandbytes (pre-quantized)", "fp8": "FP8 (pre-quantized)"}.get(method,
                                                                                   f"{method.upper()} (pre-quantized)")
    if method == "mlx":
        status, reason = (READY, None) if hw.device == "mps" and hw.has("mlx") else (
            (INSTALL, 'Install MLX: pip install "noelle[mlx]".') if hw.device == "mps" else
            (UNSUPPORTED, "MLX checkpoints run only on Apple Silicon."))
        runtime = "mlx"
    elif method in ("awq", "gptq", "bnb", "fp8", "compressed", "hqq", "eetq", "quanto", "torchao"):
        runtime = "transformers"
        if hw.device != "cuda":
            status, reason = UNSUPPORTED, f"{method.upper()} checkpoints need an NVIDIA CUDA GPU."
        else:
            package = {"bnb": "bitsandbytes"}.get(method, method)
            status, reason = (READY, None) if hw.has(package) or method in ("fp8", "compressed") else (
                INSTALL, f"Install the {package} package in the runtime environment.")
    else:
        runtime, status, reason = "transformers", UNSUPPORTED, f"Noelle has no engine for {method} checkpoints."
    quality = NEAR_LOSSLESS if method == "fp8" else SMALL_LOSS
    return _option("auto", label, runtime, "auto", memory, weights, hw, status, reason, quality,
                   _speed(hw.device, runtime, "auto" if method != "bnb" else "nf4"))


def _gguf_options(cap: ModelCapability, hw: HardwareProfile, config: dict[str, Any] | None) -> list[RunOption]:
    status, reason = (READY, None) if hw.has("llama.cpp") else (
        INSTALL, "Needs llama.cpp (Settings › Components asks before downloading).")
    variants = cap.gguf_variants
    options = []
    for variant in variants:
        weights = variant.size_bytes
        if weights is None and cap.param_count:
            weights = int(cap.param_count * _gguf_bits(variant.quant) / 8)
        memory = estimate_runtime_bytes(cap.param_count, config=config, weight_bytes=weights) if weights else None
        options.append(_option(f"gguf:{variant.quant}", f"GGUF {variant.quant}", "llama.cpp", "auto", memory, weights,
                               hw, status, reason, _gguf_quality(variant.quant), _speed(hw.device, "llama.cpp", "gguf"),
                               gguf_file=variant.first_file))
    if not options:  # file list not fetched yet
        weights = cap.weight_bytes
        memory = estimate_runtime_bytes(cap.param_count, config=config, weight_bytes=weights) if weights else (
            estimate_runtime_bytes(cap.param_count, "gguf-q4_k_m", config=config) if cap.param_count else None)
        options.append(_option("auto", "GGUF", "llama.cpp", "auto", memory, weights, hw, status, reason, SMALL_LOSS,
                               _speed(hw.device, "llama.cpp", "gguf")))
    return options


def _gguf_bits(quant: str) -> float:
    q = quant.upper()
    if q in ("F32",):
        return 32
    if q in ("F16", "BF16"):
        return 16
    for prefix, bits in (("Q8", 8.5), ("Q6", 6.6), ("Q5", 5.7), ("Q4", 4.85), ("IQ4", 4.5), ("Q3", 3.9),
                         ("IQ3", 3.5), ("Q2", 3.0), ("IQ2", 2.5), ("IQ1", 1.8), ("TQ", 2.0)):
        if q.startswith(prefix):
            return bits
    return 4.85


def _gguf_quality(quant: str) -> str:
    q = quant.upper()
    if q in ("F16", "BF16", "F32"):
        return REFERENCE
    if q.startswith(("Q8", "Q6")):
        return NEAR_LOSSLESS
    if q.startswith(("Q5", "Q4", "IQ4")):
        return SMALL_LOSS
    return NOTICEABLE_LOSS


def _speed(device: str, runtime: str, precision: str) -> str:
    if runtime == "mlx":
        return "Fastest on Apple Silicon"
    if runtime == "llama.cpp":
        return "Fast on the Apple GPU (Metal)" if device == "mps" else "Best choice for CPU" if device == "cpu" else \
            "Runs on the CPU with the bundled llama.cpp build"
    if runtime == "transformers-bnb":
        return "Slower than FP16 (dequantizes on the fly) but saves memory"
    if device == "cuda":
        return "Fast 4-bit kernels" if precision in ("awq-4bit", "gptq-4bit", "auto") else "Fast"
    if device == "mps":
        return "Moderate"
    return "Slow (CPU)"


def _estimate_tps(hw: HardwareProfile, runtime: str, weights: int | None) -> tuple[float, float] | None:
    if not weights or not hw.bandwidth_gbps:
        return None
    eff = _EFFICIENCY.get((hw.device, runtime))
    if eff is None:
        return None
    low_eff, high_eff, fixed = eff
    bandwidth = hw.bandwidth_gbps * 1e9
    low = 1.0 / (weights / (low_eff * bandwidth) + fixed * 1.5)
    high = 1.0 / (weights / (high_eff * bandwidth) + fixed * 0.75)
    return round(low, 1), round(high, 1)


# ---------------------------------------------------------------------------
# recommendation, context and score
# ---------------------------------------------------------------------------


# Fraction of peak memory bandwidth each engine reaches while decoding, by device. Only used to rank
# options against each other (weights are read once per token, so speed ~ factor / weight bytes).
_RELATIVE_FACTOR: dict[tuple[str, str], float] = {
    ("mps", "mlx"): 0.7, ("mps", "llama.cpp"): 0.62, ("mps", "transformers"): 0.32,
    ("cuda", "transformers"): 0.6, ("cuda", "transformers-bnb"): 0.25, ("cuda", "llama.cpp"): 0.15,
    ("cpu", "llama.cpp"): 0.5, ("cpu", "transformers"): 0.2, ("cpu", "mlx"): 0.0,
}


def relative_speed(option: RunOption, device: str) -> float:
    runtime = "transformers-bnb" if option.precision in ("int8", "nf4") and option.runtime == "transformers" \
        else option.runtime
    factor = _RELATIVE_FACTOR.get((device, runtime), 0.3)
    size = option.weight_bytes or option.memory_bytes
    return factor / size if size else factor * 1e-12


def recommend(options: list[RunOption], profile: str = "balanced", *, param_count: int | None = None,
              device: str = "cpu", usable_bytes: int | None = None) -> RunOption | None:
    viable = [o for o in options if o.status != UNSUPPORTED and o.fits is not False]
    if not viable:
        return None

    def status(o: RunOption) -> int:
        return _STATUS_RANK[o.status]

    def memory(o: RunOption) -> int:
        return o.memory_bytes if o.memory_bytes is not None else 1 << 62

    def speed(o: RunOption) -> float:
        return relative_speed(o, device)

    # options that work without installing anything (a one-time local build is fine)
    local = [o for o in viable if o.status in (READY, BUILD)] or viable
    if profile == "quality":
        return min(viable, key=lambda o: (-_QUALITY_RANK[o.quality], status(o), memory(o)))
    if profile == "fast":
        return min(local, key=lambda o: (-speed(o), status(o)))
    if profile in ("low_memory", "max_context"):
        return min(local, key=lambda o: (memory(o), status(o)))
    # balanced: full precision when it fits with room to spare (and the model is small enough to be quick),
    # otherwise the fastest 4-bit build that is ready or buildable here.
    usable = usable_bytes or 0
    roomy = [o for o in viable if o.quality == REFERENCE and o.status == READY and o.memory_bytes is not None
             and o.memory_bytes <= usable * 0.75]
    small = param_count is not None and param_count <= 4_000_000_000
    if roomy and (small or device == "cuda"):
        return roomy[0]
    four_bit = [o for o in viable if o.quality in (SMALL_LOSS, NEAR_LOSSLESS) and o.status in (READY, BUILD)]
    if four_bit:
        return min(four_bit, key=lambda o: (status(o), -speed(o)))
    return min(viable, key=lambda o: (status(o), -_QUALITY_RANK[o.quality], memory(o)))


def max_context(cap: ModelCapability, option: RunOption | None, hw: HardwareProfile,
                config: dict[str, Any] | None) -> int | None:
    """Largest standard context whose KV cache still fits next to ``option``'s weights."""
    if option is None or option.weight_bytes is None:
        return None
    per_token = kv_cache_bytes_per_token(config)
    if not per_token:
        return None
    room = hw.usable_bytes - option.weight_bytes - FIXED_RUNTIME_OVERHEAD
    if room <= 0:
        return None
    limit = cap.context_length
    if not limit:
        return None  # unknown maximum: do not suggest a context the model may not support
    best = None
    for steps in CONTEXT_STEPS:
        if steps > limit:
            break
        if per_token * steps <= room:
            best = steps
    if per_token * limit <= room:
        return limit  # the model's own maximum fits
    return best


def assess(cap: ModelCapability, hw: HardwareProfile, *, config: dict[str, Any] | None = None,
           profile: str = "balanced", measured: Mapping[str, Measured] | None = None,
           local_variants: Mapping[str, str] | None = None, trusted: bool = False) -> Assessment:
    compat = assess_compatibility(cap.param_count, hw.budget, config=config, weight_bytes=cap.weight_bytes,
                                  prequantized=cap.prequantized)
    options = run_options(cap, hw, config=config, local_variants=local_variants)
    recommended = recommend(options, profile, param_count=cap.param_count, device=hw.device,
                            usable_bytes=hw.usable_bytes)
    notes: list[str] = []
    items = [
        _memory_item(cap, compat, options, hw, recommended),
        _runtime_item(cap, options, hw),
        _precision_item(recommended),
        _context_item(cap),
        _trust_item(cap, trusted),
    ]
    score = sum(item.points for item in items)
    cap_note = None
    if not cap.supported and not cap.conversions:
        score, cap_note = min(score, 10), "Capped: Noelle has no engine for this task or file format."
    elif options and all(o.status == UNSUPPORTED for o in options):
        score, cap_note = min(score, 15), "Capped: this checkpoint format cannot run on this device."
    elif options and not any(o.fits is not False and o.status != UNSUPPORTED for o in options):
        if compat.label == "CPU fallback":
            score, cap_note = min(score, 45), "Capped: only a slow CPU fallback fits in memory."
        else:
            score, cap_note = min(score, 20), "Capped: no build of this model fits in this machine's memory."
    context = max_context(cap, recommended, hw, config)
    if cap.context_length and context and context < cap.context_length:
        notes.append(f"KV cache memory limits the context to ~{context:,} of {cap.context_length:,} tokens here.")
    if recommended is not None and recommended.status == BUILD:
        notes.append(f"Build {recommended.label} once in the Optimize tab, then load it from the Build list.")
    if recommended is not None and recommended.status == INSTALL and recommended.reason:
        notes.append(recommended.reason)
    if cap.remote_code.required and not trusted:
        notes.append("Loading runs Python code from the repository; review the files before allowing it.")
    return Assessment(cap, hw, compat, options, recommended, profile, score, grade(score), items, cap_note,
                      context, dict(measured or {}), notes)


def grade(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    if score >= 30:
        return "Poor"
    return "Not recommended"


def _memory_item(cap: ModelCapability, compat: Compatibility, options: list[RunOption], hw: HardwareProfile,
                 recommended: RunOption | None) -> ScoreItem:
    label, top = "Memory fit", 40
    if not options or all(o.memory_bytes is None for o in options):
        return ScoreItem("memory", label, 16, top, "Size not published; Noelle checks again before loading.")
    if recommended is None or recommended.memory_bytes is None:
        if compat.label == "CPU fallback":
            return ScoreItem("memory", label, 8, top, "Only fits in system RAM; runs on the CPU.")
        return ScoreItem("memory", label, 0, top, compat.detail)
    need = format_bytes(recommended.memory_bytes)
    usable = format_bytes(hw.usable_bytes)
    if recommended.status == BUILD:
        return ScoreItem("memory", label, 28, top, f"Fits after a local {recommended.label} build (~{need}).")
    if recommended.status == INSTALL:
        return ScoreItem("memory", label, 22, top, f"{recommended.label} fits (~{need}) once its runtime is installed.")
    spare = 1 - recommended.memory_bytes / max(hw.usable_bytes, 1)
    if spare >= 0.25:
        points = top if recommended.quality == REFERENCE else 36
        return ScoreItem("memory", label, points, top, f"{recommended.label} needs ~{need} of {usable} usable.")
    return ScoreItem("memory", label, 32, top, f"{recommended.label} fits, tightly (~{need} of {usable} usable).")


def _runtime_item(cap: ModelCapability, options: list[RunOption], hw: HardwareProfile) -> ScoreItem:
    label, top = "Runtime", 20
    if not options:
        return ScoreItem("runtime", label, 0, top, "No Noelle engine for this task or file format.")
    statuses = {o.status for o in options}
    where = {"mps": "the Apple GPU", "cuda": "the NVIDIA GPU", "cpu": "the CPU"}.get(hw.device, hw.device)
    if READY in statuses:
        runtime = next(o.runtime for o in options if o.status == READY)
        return ScoreItem("runtime", label, top, top, f"Runs with {runtime} on {where}.")
    if BUILD in statuses:
        return ScoreItem("runtime", label, 16, top, "Needs a one-time local build.")
    if INSTALL in statuses:
        reason = next(o.reason for o in options if o.status == INSTALL)
        return ScoreItem("runtime", label, 10, top, reason or "Needs an optional component.")
    return ScoreItem("runtime", label, 0, top, options[0].reason or "Not supported on this device.")


def _precision_item(recommended: RunOption | None) -> ScoreItem:
    label, top = "Precision", 15
    if recommended is None:
        return ScoreItem("precision", label, 0, top, "No precision fits on this device.")
    points = {READY: top, BUILD: 10, INSTALL: 6}.get(recommended.status, 0)
    return ScoreItem("precision", label, points, top, f"{recommended.label} · {recommended.quality.lower()}.")


def _context_item(cap: ModelCapability) -> ScoreItem:
    label, top = "Context", 10
    ctx = cap.context_length
    if not ctx:
        return ScoreItem("context", label, 5, top, "Context length not published.")
    points = 10 if ctx >= 32768 else 8 if ctx >= 8192 else 6 if ctx >= 4096 else 3
    return ScoreItem("context", label, points, top, f"{ctx:,} tokens.")


def _trust_item(cap: ModelCapability, trusted: bool) -> ScoreItem:
    label, top = "Trust & access", 15
    points, notes = top, []
    if cap.remote_code.required and not trusted:
        points -= 9
        notes.append("runs repository code on load")
    if cap.gated:
        points -= 3
        notes.append("gated: request access on huggingface.co")
    if not cap.license:
        points -= 3
        notes.append("no license declared")
    note = "; ".join(notes).capitalize() + "." if notes else "Open weights, no custom code."
    return ScoreItem("trust", label, max(points, 0), top, note)


def with_score(compat: Compatibility, assessment: Assessment) -> Compatibility:
    """The memory verdict, annotated with the overall score (kept for badges and tests)."""
    return replace(compat, score=assessment.score, grade=assessment.grade)


def profile_params(profile: str) -> dict[str, Any]:
    return dict(PROFILE_PARAMS.get(profile, {}))

