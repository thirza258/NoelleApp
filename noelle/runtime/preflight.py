"""Memory budgeting, hardware compatibility and queue planning.

Pure Python (no torch) so the desktop shell can use the same arithmetic as the
worker when it tags model cards, and so it can be unit-tested in isolation.

The model follows the spec: weights cost ``params x bytes_per_param`` (FP16 = 2,
INT4 = 0.5 plus group scales), the KV cache costs
``2 x layers x kv_heads x head_dim x 2 bytes`` per token, and a 15 % safety
headroom is kept free for the OS and driver.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

GB = 1024**3
MB = 1024**2
DEFAULT_HEADROOM = 0.15
FIXED_RUNTIME_OVERHEAD = 384 * MB  # allocator slack, CUDA/Metal context, activations at batch 1
KV_HEURISTIC_FRACTION = 0.12  # KV cache for ~4k tokens as a fraction of FP16 weights, when config unknown
DEFAULT_PLANNING_CONTEXT = 4096


@dataclass(frozen=True)
class PrecisionSpec:
    key: str
    label: str
    bits_per_param: float
    family: str  # float | bnb | awq | gptq | gguf | mlx
    requires: str | None = None  # None | "cuda" | "apple_silicon"
    quantized: bool = False

    @property
    def bytes_per_param(self) -> float:
        return self.bits_per_param / 8.0


PRECISIONS: dict[str, PrecisionSpec] = {
    spec.key: spec
    for spec in (
        PrecisionSpec("auto", "Auto (best for device)", 16, "float"),
        PrecisionSpec("fp32", "FP32", 32, "float"),
        PrecisionSpec("fp16", "FP16", 16, "float"),
        PrecisionSpec("bf16", "BF16", 16, "float"),
        PrecisionSpec("int8", "8-bit (bitsandbytes)", 8, "bnb", "cuda", True),
        PrecisionSpec("nf4", "4-bit NF4 (bitsandbytes)", 4.5, "bnb", "cuda", True),
        PrecisionSpec("awq-4bit", "4-bit AWQ", 4.25, "awq", "cuda", True),
        PrecisionSpec("gptq-4bit", "4-bit GPTQ", 4.25, "gptq", "cuda", True),
        PrecisionSpec("gguf-q4_k_m", "GGUF Q4_K_M", 4.85, "gguf", None, True),
        PrecisionSpec("mlx-4bit", "4-bit MLX", 4.5, "mlx", "apple_silicon", True),
    )
}

SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 0.5, "I4": 0.5, "U4": 0.5,
}


def precision_spec(key: str | None) -> PrecisionSpec:
    return PRECISIONS.get(key or "auto", PRECISIONS["auto"])


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

_MOE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", re.IGNORECASE)
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", re.IGNORECASE)


def parse_param_count(name: str | None) -> int | None:
    """Best-effort parameter count from a repo name, e.g. ``Qwen2.5-0.5B`` -> 5e8.

    Mixture-of-experts names (``8x7B``) return the total; active-parameter
    markers (``A3B``) are ignored because memory must hold every expert.
    """
    if not name:
        return None
    text = name.split("/")[-1]
    candidates: list[float] = []
    for count, size, unit in _MOE_RE.findall(text):
        candidates.append(int(count) * float(size) * (1e9 if unit.lower() == "b" else 1e6))
    stripped = _MOE_RE.sub(" ", text)
    for size, unit in _SIZE_RE.findall(stripped):
        value = float(size) * (1e9 if unit.lower() == "b" else 1e6)
        if value >= 1e6:
            candidates.append(value)
    if not candidates:
        return None
    return int(max(candidates))


def params_from_safetensors(parameters: dict[str, int] | None) -> tuple[int | None, int | None]:
    """Return ``(param_count, weight_bytes)`` from Hub safetensors metadata."""
    if not parameters:
        return None, None
    count = 0
    size = 0.0
    for dtype, n in parameters.items():
        count += int(n)
        size += int(n) * SAFETENSORS_DTYPE_BYTES.get(str(dtype).upper(), 2)
    return count, int(size)


def format_params(count: int | None) -> str:
    if not count:
        return "?"
    if count >= 1e12:
        return f"{count / 1e12:.1f}T"
    if count >= 1e9:
        value = count / 1e9
        return f"{value:.0f}B" if value >= 100 else f"{value:.1f}B".replace(".0B", "B")
    if count >= 1e6:
        return f"{count / 1e6:.0f}M"
    return f"{count / 1e3:.0f}K"


def format_bytes(num: float | None) -> str:
    if num is None:
        return "?"
    num = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            if unit in ("B", "KB"):
                return f"{num:.0f} {unit}"
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


# ---------------------------------------------------------------------------
# Memory estimates
# ---------------------------------------------------------------------------


def weights_bytes(param_count: int, precision: str = "fp16") -> int:
    spec = precision_spec(precision)
    return int(math.ceil(param_count * spec.bytes_per_param))


def text_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    for key in ("text_config", "llm_config", "language_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and nested:
            merged = dict(config)
            merged.update(nested)
            return merged
    return config


def kv_cache_bytes_per_token(config: dict[str, Any] | None, dtype_bytes: int = 2) -> int | None:
    cfg = text_config(config)
    layers = _first_int(cfg, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    heads = _first_int(cfg, "num_attention_heads", "n_head", "num_heads")
    if not layers or not heads:
        return None
    kv_heads = _first_int(cfg, "num_key_value_heads", "num_kv_heads", "n_kv_heads") or heads
    head_dim = _first_int(cfg, "head_dim")
    if not head_dim:
        hidden = _first_int(cfg, "hidden_size", "n_embd", "d_model")
        if not hidden:
            return None
        head_dim = hidden // heads
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def context_limit(config: dict[str, Any] | None) -> int | None:
    cfg = text_config(config)
    return _first_int(cfg, "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length",
                      "max_seq_len", "n_ctx")


def estimate_runtime_bytes(param_count: int | None, precision: str = "fp16", *, config: dict[str, Any] | None = None,
                           context_tokens: int = DEFAULT_PLANNING_CONTEXT, weight_bytes: int | None = None) -> int | None:
    """Weights + KV cache for ``context_tokens`` + fixed runtime overhead."""
    if weight_bytes is None:
        if not param_count:
            return None
        weight_bytes = weights_bytes(param_count, precision)
    per_token = kv_cache_bytes_per_token(config)
    if per_token:
        kv = per_token * context_tokens
    elif param_count:
        kv = int(param_count * 2 * KV_HEURISTIC_FRACTION)
    else:
        kv = int(weight_bytes * KV_HEURISTIC_FRACTION)
    return int(weight_bytes + kv + FIXED_RUNTIME_OVERHEAD)


@dataclass(frozen=True)
class MemoryBudget:
    total_bytes: int
    headroom: float = DEFAULT_HEADROOM
    system_ram_bytes: int | None = None
    device: str = "cpu"

    @property
    def usable_bytes(self) -> int:
        return int(self.total_bytes * (1.0 - self.headroom))


@dataclass(frozen=True)
class Compatibility:
    level: str  # green | yellow | red | unknown
    label: str
    detail: str
    fp16_bytes: int | None = None
    int4_bytes: int | None = None
    recommended_precision: str | None = None
    score: int | None = None  # 0-100 overall score from noelle.runtime.advisor, when computed
    grade: str | None = None


def assess_compatibility(param_count: int | None, budget: MemoryBudget, *, config: dict[str, Any] | None = None,
                         weight_bytes: int | None = None, prequantized: bool = False) -> Compatibility:
    """Green: FP16 fits. Yellow: only a 4-bit variant fits. Red: CPU fallback or too large."""
    if not param_count and not weight_bytes:
        return Compatibility("unknown", "Unknown size", "Parameter count not published for this repository.")
    usable = budget.usable_bytes
    if prequantized and weight_bytes:
        need = estimate_runtime_bytes(param_count, config=config, weight_bytes=weight_bytes) or weight_bytes
        if need <= usable:
            return Compatibility("green", "Fits (quantized)", f"Needs ~{format_bytes(need)} of {format_bytes(usable)} usable.",
                                 need, need, "auto")
        return _red(need, budget)
    count = param_count or int((weight_bytes or 0) / 2)
    fp16 = estimate_runtime_bytes(count, "fp16", config=config) or 0
    int4 = estimate_runtime_bytes(count, "nf4", config=config) or 0
    if fp16 <= usable:
        return Compatibility("green", "Runs natively",
                             f"FP16 needs ~{format_bytes(fp16)} of {format_bytes(usable)} usable.", fp16, int4, "fp16")
    if int4 <= usable:
        target = "mlx-4bit" if budget.device == "mps" else "nf4" if budget.device == "cuda" else "gguf-q4_k_m"
        return Compatibility("yellow", "Needs 4-bit",
                             f"FP16 needs ~{format_bytes(fp16)}; a 4-bit build needs ~{format_bytes(int4)} "
                             f"of {format_bytes(usable)} usable.", fp16, int4, target)
    return _red(int4, budget, fp16)


def _red(need: int, budget: MemoryBudget, fp16: int | None = None) -> Compatibility:
    ram = budget.system_ram_bytes
    if ram and need <= ram * 0.85:
        return Compatibility("red", "CPU fallback",
                             f"Even 4-bit needs ~{format_bytes(need)}, above the {format_bytes(budget.usable_bytes)} "
                             "accelerator budget. Runs on CPU with high latency.", fp16, need, "gguf-q4_k_m")
    return Compatibility("red", "Too large",
                         f"Needs ~{format_bytes(need)} even at 4-bit, above this machine's memory.", fp16, need, None)


# ---------------------------------------------------------------------------
# Generation auto-tuning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TunedGeneration:
    max_new_tokens: int
    batch_size: int
    clamped: bool
    reason: str | None = None


def tune_generation(requested_max_new_tokens: int, prompt_tokens: int, *, context_window: int | None,
                    kv_bytes_per_token: int | None, free_bytes: int | None, batch_size: int = 1,
                    min_new_tokens: int = 16) -> TunedGeneration:
    """Clamp ``max_new_tokens`` (and batch size) so KV cache growth stays in budget."""
    requested = max(int(requested_max_new_tokens), 1)
    limit = requested
    reasons: list[str] = []
    if context_window:
        room = context_window - prompt_tokens
        if room < limit:
            limit = max(room, 1)
            reasons.append(f"context window is {context_window} tokens")
    batch = max(batch_size, 1)
    if kv_bytes_per_token and free_bytes is not None:
        while batch > 1 and kv_bytes_per_token * batch * (prompt_tokens + min(limit, min_new_tokens)) > free_bytes:
            batch -= 1
        affordable_total = int(free_bytes // (kv_bytes_per_token * batch)) if kv_bytes_per_token > 0 else limit
        affordable_new = affordable_total - prompt_tokens
        if affordable_new < limit:
            limit = max(affordable_new, 1)
            reasons.append(f"free memory {format_bytes(free_bytes)} limits KV cache")
        if batch != batch_size:
            reasons.append(f"batch reduced to {batch}")
    clamped = limit != requested or batch != batch_size
    return TunedGeneration(limit, batch, clamped, "; ".join(reasons) or None)


# ---------------------------------------------------------------------------
# Sequential queue planning (comparative bench)
# ---------------------------------------------------------------------------


@dataclass
class QueueStep:
    view: int
    key: str
    estimated_bytes: int
    evict_before: list[str] = field(default_factory=list)
    reuse_loaded: bool = False


@dataclass
class QueuePlan:
    mode: str  # "retain" | "hot_swap"
    total_bytes: int
    usable_bytes: int
    steps: list[QueueStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_bytes": self.total_bytes,
            "usable_bytes": self.usable_bytes,
            "steps": [step.__dict__ for step in self.steps],
        }


def plan_queue(jobs: Iterable[tuple[int, str, int]], usable_bytes: int,
               resident: dict[str, int] | None = None) -> QueuePlan:
    """Plan View 1..N execution.

    ``jobs`` are ``(view, model_key, estimated_bytes)`` in execution order and
    ``resident`` maps already-loaded keys to their size. If every distinct model
    fits at once the plan retains them all ("retain"); otherwise each step evicts
    least-recently-used models until the next one fits ("hot_swap"), which for
    large models means unloading the previous view before loading the next.
    """
    ordered = sorted(jobs, key=lambda job: job[0])
    resident = dict(resident or {})
    sizes: dict[str, int] = {}
    for _, key, size in ordered:
        sizes[key] = max(size, sizes.get(key, 0))
    total = sum(sizes.values())
    mode = "retain" if total <= usable_bytes else "hot_swap"

    loaded: dict[str, int] = dict(resident)  # insertion order doubles as LRU order
    steps: list[QueueStep] = []
    for index, (view, key, size) in enumerate(ordered):
        step = QueueStep(view=view, key=key, estimated_bytes=size)
        if key in loaded:
            step.reuse_loaded = True
            loaded[key] = loaded.pop(key)
            steps.append(step)
            continue
        upcoming = {k for _, k, _ in ordered[index + 1:]}
        while loaded and sum(loaded.values()) + size > usable_bytes:
            victim = _pick_victim(loaded, upcoming, mode)
            step.evict_before.append(victim)
            loaded.pop(victim)
        loaded[key] = size
        steps.append(step)
    return QueuePlan(mode=mode, total_bytes=total, usable_bytes=usable_bytes, steps=steps)


def _pick_victim(loaded: dict[str, int], upcoming: set[str], mode: str) -> str:
    # Prefer evicting models that no later view needs; fall back to plain LRU.
    for key in loaded:
        if key not in upcoming:
            return key
    return next(iter(loaded))


def _first_int(cfg: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None
