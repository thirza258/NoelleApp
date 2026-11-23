"""Unified capability registry: one description of what a model is and how Noelle can run it.

Discover badges, the Workspace analysis, the runtime/quantization advisor and the
Evaluation Lab all read the same :class:`ModelCapability` instead of re-deriving
facts from tags, configs and file lists in several places.

Pure Python (no torch, no Qt) so the desktop shell, the API server and tests can
import it. Facts the Hub does not publish stay ``None`` or empty: the analyzer
never guesses.
"""

from __future__ import annotations

import importlib.util
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from noelle.runtime import tasks
from noelle.runtime.preflight import context_limit

# pipeline tag -> (input modalities, output modalities)
_IO: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "text-generation": (("text",), ("text",)),
    "text2text-generation": (("text",), ("text",)),
    "conversational": (("text",), ("text",)),
    "summarization": (("text",), ("text",)),
    "translation": (("text",), ("text",)),
    "image-text-to-text": (("text", "image"), ("text",)),
    "image-to-text": (("image",), ("text",)),
    "visual-question-answering": (("text", "image"), ("text",)),
    "document-question-answering": (("text", "image"), ("text",)),
    "video-text-to-text": (("text", "video"), ("text",)),
    "object-detection": (("image",), ("detections",)),
    "image-classification": (("image",), ("labels",)),
    "automatic-speech-recognition": (("audio",), ("text",)),
    "text-to-speech": (("text",), ("audio",)),
    "text-to-audio": (("text",), ("audio",)),
    "text-to-image": (("text",), ("image",)),
    "image-to-image": (("text", "image"), ("image",)),
    "unconditional-image-generation": ((), ("image",)),
    "sentence-similarity": (("text",), ("embedding",)),
    "feature-extraction": (("text",), ("embedding",)),
}

# Engines Noelle's worker loads directly, per task family.
_FAMILY_RUNTIMES: dict[str, tuple[str, ...]] = {
    tasks.TEXT: ("transformers",),
    tasks.VISION: ("transformers",),
    tasks.DETECTION: ("transformers",),
    tasks.CLASSIFICATION: ("transformers",),
    tasks.ASR: ("transformers",),
    tasks.TTS: ("transformers",),
    tasks.DIFFUSION: ("diffusers",),
    tasks.EMBEDDING: ("sentence-transformers",),
}

# Load-time precisions offered for the original checkpoint, per family (bitsandbytes needs CUDA).
FAMILY_PRECISIONS: dict[str, tuple[str, ...]] = {
    tasks.TEXT: ("auto", "fp16", "bf16", "fp32", "int8", "nf4"),
    tasks.VISION: ("auto", "fp16", "bf16", "fp32", "int8", "nf4"),
    tasks.DIFFUSION: ("auto", "fp16", "fp32"),
}

# Local conversions the Optimize tab can build (text-generation checkpoints only).
CONVERSIONS: tuple[str, ...] = ("mlx-4bit", "gguf-q4_k_m", "nf4", "awq-4bit", "gptq-4bit")

_QUANT_METHODS = {"awq": "awq", "gptq": "gptq", "bitsandbytes": "bnb", "fp8": "fp8", "compressed-tensors": "compressed",
                  "hqq": "hqq", "eetq": "eetq", "aqlm": "aqlm", "exl2": "exl2", "quanto": "quanto", "mlx": "mlx",
                  "torchao": "torchao", "bitnet": "bitnet"}
_QUANT_TAGS = {"gguf": "gguf", "awq": "awq", "gptq": "gptq", "mlx": "mlx", "exl2": "exl2", "bitsandbytes": "bnb",
               "4-bit": "bnb", "8-bit": "bnb", "fp8": "fp8", "compressed-tensors": "compressed"}
_TOOL_TAGS = {"tool-use", "tool_use", "function-calling", "function_calling", "tools", "agent"}
# quantization label at the end of a GGUF file stem: "...-Q4_K_M", "....Q8_0", "...-IQ4_XS", "...-f16"
_GGUF_QUANT = re.compile(r"(?:^|[-._])((?:i?q|tq)\d+(?:_[a-z0-9]+)*|bf16|f16|f32|fp16|fp32)$", re.IGNORECASE)
_SHARD = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)


@dataclass(frozen=True)
class GgufVariant:
    quant: str  # e.g. Q4_K_M
    files: tuple[str, ...]
    size_bytes: int | None

    @property
    def first_file(self) -> str:
        return self.files[0]


@dataclass(frozen=True)
class RemoteCode:
    required: bool
    files: tuple[str, ...] = ()
    auto_map: tuple[str, ...] = ()


@dataclass
class ModelCapability:
    model_id: str
    pipeline_tag: str | None
    family: str | None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    architecture: str | None = None
    model_type: str | None = None
    param_count: int | None = None
    weight_bytes: int | None = None
    context_length: int | None = None
    library: str | None = None
    formats: tuple[str, ...] = ()  # safetensors | pytorch | gguf | mlx | onnx
    quantization: str | None = None  # pre-quantized checkpoint: gguf | mlx | awq | gptq | bnb | fp8 | ...
    gguf_variants: tuple[GgufVariant, ...] = ()
    runtimes: tuple[str, ...] = ()  # engines that load the repository as published
    precisions: tuple[str, ...] = ()  # load-time precisions for the original checkpoint
    conversions: tuple[str, ...] = ()  # local builds the Optimize tab can make
    chat_template: bool | None = None
    tool_calling: bool | None = None
    languages: tuple[str, ...] = ()
    license: str | None = None
    gated: bool = False
    remote_code: RemoteCode = field(default_factory=lambda: RemoteCode(False))
    is_adapter: bool = False
    base_models: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return bool(self.runtimes)

    @property
    def prequantized(self) -> bool:
        return self.quantization is not None

    @property
    def publisher(self) -> str:
        return self.model_id.split("/")[0] if "/" in self.model_id else ""

    @property
    def modalities(self) -> tuple[str, ...]:
        seen: list[str] = []
        for item in (*self.inputs, *self.outputs):
            if item not in seen:
                seen.append(item)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supported"] = self.supported
        data["prequantized"] = self.prequantized
        data["publisher"] = self.publisher
        data["modalities"] = list(self.modalities)
        return data


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyze(model: Any, config: dict[str, Any] | None = None, *, tokenizer_config: dict[str, Any] | None = None,
            chat_template: str | None = None) -> ModelCapability:
    """Build a :class:`ModelCapability` from a Hub summary (or any object with the same attributes).

    ``config`` overrides the summary's own config (the detail page fetches the full
    ``config.json``); ``tokenizer_config`` / ``chat_template`` are optional and only
    used to detect chat and tool-calling support.
    """
    model_id = str(getattr(model, "id", None) or getattr(model, "repo_id", "") or "")
    tags = [str(t) for t in (getattr(model, "tags", None) or [])]
    lowered = {t.lower() for t in tags}
    cfg = dict(config or getattr(model, "config", None) or {})
    siblings = list(getattr(model, "siblings", None) or [])
    names = [str(s.get("name") if isinstance(s, dict) else s) for s in siblings]
    sizes = {str(s.get("name")): s.get("size") for s in siblings if isinstance(s, dict)}
    library = getattr(model, "library_name", None)
    pipeline_tag = getattr(model, "pipeline_tag", None)
    family = tasks.task_family(pipeline_tag)
    inputs, outputs = _IO.get(pipeline_tag or "", ((), ()))

    formats = _formats(names, lowered, library, cfg)
    quantization = _quantization(cfg, lowered, library, formats)
    gguf_variants = gguf_variants_from_files(names, sizes) if "gguf" in formats else ()
    runtimes = _runtimes(family, quantization, formats)
    precisions: tuple[str, ...] = ()
    conversions: tuple[str, ...] = ()
    if family and quantization is None:
        precisions = FAMILY_PRECISIONS.get(family, ("auto",))
        if family == tasks.TEXT and (formats & {"safetensors", "pytorch"} or not formats):
            conversions = CONVERSIONS
    elif family:
        precisions = ("auto",)

    template = chat_template or _template_text(tokenizer_config)
    has_template: bool | None = None  # unknown until a tokenizer config or template file is seen
    if template or "chat_template.jinja" in names:
        has_template = True
    elif tokenizer_config is not None:
        has_template = False
    tool_calling: bool | None = None
    if lowered & _TOOL_TAGS:
        tool_calling = True
    elif template:
        tool_calling = bool(re.search(r"\btools\b", template))

    gguf = getattr(model, "gguf", None) or {}
    context = None
    if isinstance(gguf, dict) and gguf.get("context_length"):
        context = int(gguf["context_length"])
    context = context or context_limit(cfg)
    architecture = None
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs:
        architecture = str(archs[0])
    elif isinstance(gguf, dict) and gguf.get("architecture"):
        architecture = str(gguf["architecture"])

    py_files = tuple(sorted(n for n in names if n.endswith(".py")))
    auto_map = cfg.get("auto_map") if isinstance(cfg.get("auto_map"), dict) else {}
    auto_refs = tuple(sorted({str(v) if not isinstance(v, (list, tuple)) else str(v[0]) for v in auto_map.values()}))
    needs_code = "custom_code" in lowered or bool(auto_refs)

    weight_bytes = getattr(model, "weight_bytes", None) or _weight_file_bytes(names, sizes, formats)
    gated = getattr(model, "gated", False)
    return ModelCapability(
        model_id=model_id,
        pipeline_tag=pipeline_tag,
        family=family,
        inputs=inputs,
        outputs=outputs,
        architecture=architecture,
        model_type=cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else None,
        param_count=getattr(model, "param_count", None),
        weight_bytes=weight_bytes,
        context_length=context,
        library=library,
        formats=tuple(sorted(formats)),
        quantization=quantization,
        gguf_variants=gguf_variants,
        runtimes=runtimes,
        precisions=precisions,
        conversions=conversions,
        chat_template=has_template,
        tool_calling=tool_calling,
        languages=_languages(getattr(model, "card_data", None) or {}),
        license=_license(getattr(model, "license", None)),
        gated=bool(gated) and gated not in ("false", "False"),
        remote_code=RemoteCode(needs_code, py_files, auto_refs),
        is_adapter=bool(getattr(model, "is_adapter", False)) or library == "peft",
        base_models=tuple(getattr(model, "base_models", None) or ()),
    )


def _weight_file_bytes(names: list[str], sizes: dict[str, Any], formats: set[str]) -> int | None:
    """Size of the weights Noelle would load (one format only), when the Hub lists file sizes."""
    for suffixes in ((".safetensors",), (".bin", ".pt", ".pth")):
        files = [n for n in names if n.lower().endswith(suffixes) and "/" not in n.strip("/")
                 and not n.lower().startswith(("training_args", "optimizer", "scheduler"))]
        if files and all(sizes.get(f) for f in files):
            return int(sum(int(sizes[f]) for f in files))
    return None


def gguf_variants_from_files(names: Iterable[str], sizes: dict[str, Any] | None = None) -> tuple[GgufVariant, ...]:
    """Group ``*.gguf`` files by quantization label (multi-part shards are summed)."""
    groups: dict[str, list[str]] = {}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        if not base.lower().endswith(".gguf") or base.lower().startswith("mmproj"):
            continue
        match = _GGUF_QUANT.search(_SHARD.sub("", base[:-5]))
        quant = match.group(1).upper() if match else "GGUF"
        quant = {"FP16": "F16", "FP32": "F32"}.get(quant, quant)
        groups.setdefault(quant, []).append(name)
    variants = []
    for quant, files in groups.items():
        files.sort()
        total: int | None = 0
        for f in files:
            size = (sizes or {}).get(f)
            if size is None:
                total = None
                break
            total += int(size)
        variants.append(GgufVariant(quant, tuple(files), total))
    variants.sort(key=lambda v: (_quant_bits(v.quant), v.quant))
    return tuple(variants)


def _quant_bits(quant: str) -> float:
    match = re.search(r"(\d+)", quant)
    if quant in ("F16", "BF16"):
        return 16.0
    if quant == "F32":
        return 32.0
    return float(match.group(1)) if match else 99.0


def _formats(names: list[str], tags: set[str], library: str | None, cfg: dict[str, Any]) -> set[str]:
    formats: set[str] = set()
    for name in names:
        lower = name.lower()
        if lower.endswith(".safetensors"):
            formats.add("safetensors")
        elif lower.endswith((".bin", ".pt", ".pth")) and ("pytorch_model" in lower or "diffusion_pytorch" in lower
                                                           or lower.endswith((".pt", ".pth"))):
            formats.add("pytorch")
        elif lower.endswith(".gguf"):
            formats.add("gguf")
        elif lower.endswith(".onnx"):
            formats.add("onnx")
        elif lower.endswith(".npz"):
            formats.add("mlx")
    if "gguf" in tags or library == "gguf":
        formats.add("gguf")
    if "mlx" in tags or library == "mlx" or _mlx_quantized(cfg):
        formats.add("mlx")
    return formats


def _mlx_quantized(cfg: dict[str, Any]) -> bool:
    quant = cfg.get("quantization")
    return isinstance(quant, dict) and "bits" in quant and "group_size" in quant


def _quantization(cfg: dict[str, Any], tags: set[str], library: str | None, formats: set[str]) -> str | None:
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method") or "").lower()
        if method:
            return _QUANT_METHODS.get(method, method)
        if qc.get("load_in_4bit") or qc.get("load_in_8bit"):
            return "bnb"
    if _mlx_quantized(cfg) or library == "mlx" or ("mlx" in formats and "safetensors" not in formats):
        return "mlx"
    if "gguf" in formats and not formats & {"safetensors", "pytorch"}:
        return "gguf"
    if library == "gguf":
        return "gguf"
    for tag, method in _QUANT_TAGS.items():
        if tag in tags and method != "gguf":
            if method == "mlx" and library not in (None, "mlx"):
                continue
            return method
    return None


def _runtimes(family: str | None, quantization: str | None, formats: set[str]) -> tuple[str, ...]:
    if family is None:
        return ()
    if quantization == "gguf":
        return ("llama.cpp",) if family == tasks.TEXT else ()
    if quantization == "mlx":
        return ("mlx",) if family == tasks.TEXT else ()
    if quantization in ("exl2", "aqlm", "bitnet"):
        return ()  # no Noelle engine for these formats
    if formats == {"onnx"}:
        return ()
    return _FAMILY_RUNTIMES.get(family, ())


def _template_text(tokenizer_config: dict[str, Any] | None) -> str | None:
    if not tokenizer_config:
        return None
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str):
        return template
    if isinstance(template, list):  # named templates: [{"name": "default", "template": "..."}, ...]
        return "\n".join(str(t.get("template", "")) for t in template if isinstance(t, dict)) or None
    return None


def _languages(card: dict[str, Any]) -> tuple[str, ...]:
    value = card.get("language") if isinstance(card, dict) else None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value if v)
    return ()


def _license(value: Any) -> str | None:
    if value is None:
        return None
    text = ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    return text.strip() or None


# ---------------------------------------------------------------------------
# runtime backends (what this machine can run)
# ---------------------------------------------------------------------------

BACKEND_LABELS = {
    "mlx": "MLX (mlx-lm)",
    "llama.cpp": "llama.cpp (llama-server or llama-cpp-python)",
    "llama.cpp-tools": "llama.cpp GGUF tools",
    "bitsandbytes": "bitsandbytes",
    "awq": "AutoAWQ",
    "gptq": "GPTQModel",
    "diffusers": "diffusers",
    "sentence-transformers": "sentence-transformers",
    "transformers": "transformers",
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_backends() -> dict[str, bool]:
    """Which optional engines this Python environment can use. Never imports them."""
    from noelle.runtime import llama_tools
    from noelle.runtime.quantization_engine import find_llama_cpp

    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    try:
        tools_ok = find_llama_cpp().missing_reason() is None
    except Exception:
        tools_ok = False
    try:
        server = llama_tools.find_llama_server() is not None
    except Exception:
        server = False
    return {
        "transformers": _has_module("transformers"),
        "mlx": apple and _has_module("mlx") and _has_module("mlx_lm"),
        "llama.cpp": server or _has_module("llama_cpp"),
        "llama.cpp-tools": tools_ok,
        "bitsandbytes": _has_module("bitsandbytes"),
        "awq": _has_module("awq"),
        "gptq": _has_module("gptqmodel"),
        "diffusers": _has_module("diffusers"),
        "sentence-transformers": _has_module("sentence_transformers"),
    }


@lru_cache(maxsize=1)
def detect_backends_cached() -> dict[str, bool]:
    return detect_backends()
