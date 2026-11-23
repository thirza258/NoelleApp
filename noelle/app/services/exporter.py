"""Reproduce a playground configuration outside the app (task T-702).

:func:`export_bundle` writes a self-contained folder:

* ``inference.py``     runnable CLI (transformers, diffusers, sentence-transformers, mlx-lm or llama.cpp)
* ``server.py``        minimal FastAPI server around ``inference.py`` (OpenAI-style routes, ``GET /health``)
* ``requirements.txt`` versions pinned from the exporting environment
* ``Dockerfile``       NVIDIA CUDA runtime image; weights fetched at build time through a BuildKit secret
* ``README.md``        run instructions

Every user-provided value is embedded as a Python literal (``repr``), never pasted
into code, shell or Dockerfile syntax. Qt-free and torch-free.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import math
import os
import pprint
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noelle.runtime import tasks
from noelle.runtime.preflight import PRECISIONS
from noelle.runtime.protocol import GenerationParams

DOCKER_BASE_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"  # Ubuntu 24.04 -> Python 3.12
CONTAINER_UID = 10001
APPLE_SILICON = 'sys_platform == "darwin" and platform_machine == "arm64"'
NOT_APPLE_SILICON = 'sys_platform != "darwin" or platform_machine != "arm64"'

_FAMILY_PRECISIONS: dict[str, tuple[str, ...]] = {
    tasks.TEXT: tuple(PRECISIONS),
    tasks.VISION: ("auto", "fp32", "fp16", "bf16", "nf4", "int8", "awq-4bit", "gptq-4bit"),
    tasks.ASR: ("auto", "fp32", "fp16", "bf16"),
    tasks.TTS: ("auto",),  # the playground runs speech synthesis at the pipeline default
    tasks.DIFFUSION: ("auto", "fp32", "fp16", "bf16"),
    tasks.EMBEDDING: ("auto",),  # sentence-transformers default, as in the playground
}
_FAMILY_BACKEND = {tasks.VISION: "vision", tasks.ASR: "asr", tasks.TTS: "tts", tasks.DIFFUSION: "diffusers",
                   tasks.EMBEDDING: "sentence_transformers"}
_BACKEND_PACKAGES: dict[str, tuple[str, ...]] = {
    "transformers": ("torch", "transformers", "accelerate", "safetensors", "sentencepiece", "huggingface-hub"),
    "vision": ("torch", "transformers", "accelerate", "safetensors", "pillow", "huggingface-hub"),
    "asr": ("torch", "transformers", "accelerate", "numpy", "scipy", "soundfile", "huggingface-hub"),
    "tts": ("torch", "transformers", "accelerate", "numpy", "scipy", "soundfile", "huggingface-hub"),
    "diffusers": ("torch", "diffusers", "transformers", "accelerate", "safetensors", "sentencepiece", "pillow",
                  "huggingface-hub"),
    "sentence_transformers": ("torch", "sentence-transformers", "transformers", "numpy", "huggingface-hub"),
    "llama_cpp": ("llama-cpp-python", "huggingface-hub"),
    "mlx": ("mlx", "mlx-lm", "huggingface-hub"),
}
_PRECISION_PACKAGES = {"nf4": ("bitsandbytes",), "int8": ("bitsandbytes",), "awq-4bit": ("gptqmodel",),
                       "gptq-4bit": ("gptqmodel", "optimum")}
_SERVER_PACKAGES = ("fastapi", "uvicorn")
_ADAPTER_FAMILIES = (tasks.TEXT, tasks.VISION, tasks.DIFFUSION)
_REPO_ID_RE = re.compile(r"^[^\s\x00-\x1f\x7f]+$")
# Mirrors noelle.runtime.downloader.default_patterns for the build-time download.
_ALWAYS_SKIP = ["onnx/*", "openvino/*", "coreml/*", "*.mlpackage/*", "*.onnx", "*.onnx_data", "*.tflite",
                "*.mlmodel", "*.msgpack", "*.h5", "*.ot", "flax_model*", "tf_model*", "rust_model*"]
_REDUNDANT_WITH_SAFETENSORS = ["*.bin", "*.pt", "*.pth", "*.ckpt"]


@dataclass
class ExportConfig:
    """A playground configuration to reproduce outside the app."""

    model_id: str
    task: str = "text-generation"  # Hub pipeline tag
    precision: str = "auto"  # key of noelle.runtime.preflight.PRECISIONS
    device: str = "auto"
    system_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # temperature, top_p, top_k, repetition_penalty, ...
    local_path: str | None = None  # snapshot / converted folder used by the app
    gguf_file: str | None = None
    adapter_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # steps, guidance, width, height, language, speed, ...


@dataclass(frozen=True)
class _Plan:
    family: str
    backend: str  # transformers | mlx | llama_cpp | vision | asr | tts | diffusers | sentence_transformers
    precision: str
    notes: tuple[str, ...]


def supported_precisions(task: str) -> tuple[str, ...]:
    """Precisions that change the exported code for ``task``; others fall back to ``auto``."""
    family = tasks.task_family(task)
    if family not in _FAMILY_PRECISIONS:
        raise ValueError(f"Export is not available for task {task!r}.")
    return _FAMILY_PRECISIONS[family]


def _resolve(config: ExportConfig) -> _Plan:
    if not isinstance(config.model_id, str) or not _REPO_ID_RE.match(config.model_id):
        raise ValueError(f"Invalid model id: {config.model_id!r}")
    for name in ("gguf_file", "adapter_id"):
        value = getattr(config, name)
        if value is not None and (not isinstance(value, str) or not _REPO_ID_RE.match(value)):
            raise ValueError(f"Invalid {name}: {value!r}")
    family = tasks.task_family(config.task)
    if family not in _FAMILY_PRECISIONS:
        raise ValueError(f"Export is not available for task {config.task!r}.")
    precision = str(config.precision or "auto").lower()
    notes: list[str] = []
    if precision not in PRECISIONS:
        notes.append(f"Unknown precision {precision!r}; exported with auto.")
        precision = "auto"
    if family == tasks.TEXT:
        if precision == "mlx-4bit":
            backend = "mlx"
        elif precision == "gguf-q4_k_m" or config.gguf_file:
            if precision != "gguf-q4_k_m":
                notes.append(f"A GGUF file is selected, so llama.cpp runs it as stored ({precision} ignored).")
            backend, precision = "llama_cpp", "gguf-q4_k_m"
        else:
            backend = "transformers"
    else:
        backend = _FAMILY_BACKEND[family]
        if precision not in _FAMILY_PRECISIONS[family]:
            notes.append(f"{PRECISIONS[precision].label} is not available for {config.task}; exported with auto "
                         "(the playground default).")
            precision = "auto"
    if config.adapter_id and family not in _ADAPTER_FAMILIES:
        notes.append("Adapters are not applied for this task (the playground does not either).")
    if backend == "mlx" and not (config.extra or {}).get("base_model") and config.model_id.startswith("mlx-community/"):
        notes.append("MODEL_ID is an MLX conversion: set BASE_MODEL_ID to its original FP16 repo for the "
                     "non-Apple fallback (Docker).")
    return _Plan(family, backend, precision, tuple(_one_line(n) for n in notes))


def _one_line(text: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text)).strip()


def _plain(value: Any) -> Any:
    """JSON-like Python values only, so ``repr`` always yields a valid literal."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _literal(value: Any) -> str:
    return pprint.pformat(_plain(value), sort_dicts=False, width=100)


def _generation(config: ExportConfig) -> dict[str, Any]:
    """Playground sampling parameters with the same defaults and clamping as the app."""
    params = GenerationParams.from_dict(_plain(config.params or {}))
    return {"temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k,
            "repetition_penalty": params.repetition_penalty, "max_new_tokens": params.max_new_tokens,
            "seed": params.seed}


def _base_model(config: ExportConfig) -> str:
    base = (config.extra or {}).get("base_model") if isinstance(config.extra or {}, dict) else None
    if isinstance(base, str) and _REPO_ID_RE.match(base):
        return base
    return config.model_id


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


def installed_version(package: str) -> str | None:
    """Version in the exporting environment (local ``+cu124`` suffixes dropped), or None."""
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # broken metadata must not break an export
        return None
    return version.split("+", 1)[0] or None


def _pin(package: str) -> str:
    version = installed_version(package)
    return f"{package}=={version}" if version else package


def _inference_packages(plan: _Plan, config: ExportConfig) -> list[tuple[str, str | None]]:
    """(package, environment marker) pairs needed by inference.py."""
    if plan.backend == "mlx":
        fallback = [p for p in _BACKEND_PACKAGES["transformers"] if p != "huggingface-hub"]
        pairs = [(p, APPLE_SILICON) for p in ("mlx", "mlx-lm")] + [("huggingface-hub", None)]
        pairs += [(p, NOT_APPLE_SILICON) for p in fallback]
        if config.adapter_id:
            pairs.append(("peft", NOT_APPLE_SILICON))
        return pairs
    names = list(_BACKEND_PACKAGES[plan.backend]) + list(_PRECISION_PACKAGES.get(plan.precision, ()))
    if config.adapter_id and plan.family in _ADAPTER_FAMILIES and plan.backend != "llama_cpp":
        names.append("peft")
    return [(name, None) for name in dict.fromkeys(names)]


def _requirement(package: str, marker: str | None) -> str:
    return f"{_pin(package)} ; {marker}" if marker else _pin(package)


def requirements_txt(config: ExportConfig) -> str:
    """requirements.txt pinned from the exporting environment (unpinned when a package is not installed)."""
    plan = _resolve(config)
    inference = _inference_packages(plan, config)
    lines = [f"# Noelle export of {_one_line(config.model_id)} ({config.task}, {plan.precision}).",
             f"# Pinned from the exporting environment on {_today()}; unpinned packages were not installed there."]
    if plan.backend == "mlx":
        lines.append("# mlx-lm runs on Apple Silicon; other platforms (Docker) use the PyTorch FP16 fallback.")
    lines += [_requirement(name, marker) for name, marker in inference]
    lines.append("# HTTP server (server.py)")
    lines += [_requirement(name, None) for name in _SERVER_PACKAGES if name not in {n for n, _ in inference}]
    return "\n".join(lines) + "\n"


def _install_lines(plan: _Plan, config: ExportConfig) -> list[str]:
    pairs = _inference_packages(plan, config)
    if plan.backend == "mlx":
        apple = " ".join(f'"{_pin(n)}"' for n, m in pairs if m != NOT_APPLE_SILICON)
        other = " ".join(f'"{_pin(n)}"' for n, m in pairs if m != APPLE_SILICON)
        return [f"Apple Silicon:  pip install {apple}", f"Elsewhere (FP16 fallback):  pip install {other}"]
    return ["pip install " + " ".join(f'"{_pin(n)}"' for n, _m in pairs)]


def _today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# inference.py
# ---------------------------------------------------------------------------

_COMMON = r'''

class ExportError(RuntimeError):
    """A configuration this machine cannot run (reported without a traceback)."""


def model_source(model_id: str | None = None, local_path: str | None = LOCAL_PATH) -> str:
    """The snapshot folder the app used when it exists on this machine, else the Hub id."""
    if local_path:
        folder = os.path.expanduser(local_path)
        if os.path.isdir(folder):
            return folder
    return model_id or MODEL_ID


def read_prompt(value: str | None, demo: str) -> str:
    """Prompt from the command line, else from a pipe, else a demo prompt."""
    if value:
        return value
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return demo
'''

_TORCH = r'''

def pick_device(requested: str | None = None) -> str:
    """CUDA, then Apple Metal (MPS), then CPU: the order the Noelle runtime uses."""
    import torch

    preference = (requested or DEVICE or "auto").lower()
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if preference in ("cuda", "gpu") and cuda:
        return "cuda"
    if preference in ("mps", "metal") and mps:
        return "mps"
    if preference == "cpu":
        return "cpu"
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def pick_dtype(device: str, precision: str = PRECISION):
    """Same dtype policy as the Noelle runtime."""
    import torch

    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16 if device != "cpu" else torch.float32
    if precision == "auto":
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float16 if device == "mps" else torch.float32
    return torch.float16 if device != "cpu" else torch.float32  # quantized weights compute in half precision
'''

_HF_MODEL = r'''

def model_kwargs(device: str, precision: str = PRECISION) -> dict:
    """from_pretrained() arguments for the exported precision (transformers 5 API: dtype=)."""
    import torch

    if precision in ("nf4", "int8"):
        if device != "cuda":
            raise ExportError(f"{precision} (bitsandbytes) needs an NVIDIA CUDA GPU; this machine has {device}.")
        from transformers import BitsAndBytesConfig

        if precision == "int8":
            quantization = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        return {"quantization_config": quantization, "device_map": "auto"}
    if precision in ("awq-4bit", "gptq-4bit") and device != "cuda":
        raise ExportError(f"{precision} kernels need an NVIDIA CUDA GPU; this machine has {device}.")
    kwargs = {"dtype": pick_dtype(device, precision)}
    if device != "cpu":
        kwargs["device_map"] = device
    return kwargs


def attach_adapter(model):
    """Apply the PEFT (LoRA) adapter selected in the playground."""
    if not ADAPTER_ID:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, ADAPTER_ID)
'''

_SAMPLING = r'''

def sampling(overrides: dict | None = None) -> dict:
    """Playground sampling settings, clamped like Noelle's GenerationParams."""
    values = dict(GENERATION)
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    seed = values.get("seed")
    return {
        "temperature": min(max(float(values.get("temperature", 0.7)), 0.0), 2.0),
        "top_p": min(max(float(values.get("top_p", 0.9)), 0.01), 1.0),
        "top_k": max(int(values.get("top_k", 50)), 0),
        "repetition_penalty": min(max(float(values.get("repetition_penalty", 1.1)), 0.5), 2.5),
        "max_new_tokens": max(int(values.get("max_new_tokens", 512)), 1),
        "seed": None if seed is None else int(seed),
    }
'''

_CHAT_HELPERS = r'''

def build_messages(prompt: str, system: str | None = None) -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def merge_system(messages: list) -> list:
    """Fold system text into the first user turn for chat templates that reject a system role."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system:
        if rest and rest[0]["role"] == "user":
            rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
        else:
            rest.insert(0, {"role": "user", "content": system})
    return rest


def plain_prompt(messages: list) -> str:
    """Prompt for models without a chat template (same format as Noelle)."""
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    lines.append("Assistant:")
    return "\n\n".join(lines)
'''

_HF_STREAM = r'''

def generation_kwargs(settings: dict) -> dict:
    """generate() arguments; temperature 0 means greedy decoding."""
    kwargs = {"max_new_tokens": settings["max_new_tokens"], "repetition_penalty": settings["repetition_penalty"]}
    if settings["temperature"] > 0:
        kwargs.update(do_sample=True, temperature=settings["temperature"], top_p=settings["top_p"],
                      top_k=settings["top_k"] if settings["top_k"] > 0 else None)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kwargs


def stream_generate(model, tokenizer, inputs: dict, settings: dict):
    """Run generate() on a worker thread and yield text as it is decoded."""
    import threading

    import torch
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(inputs)
    kwargs.update(generation_kwargs(settings))
    kwargs["streamer"] = streamer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if settings["seed"] is not None:
        torch.manual_seed(settings["seed"])
    errors = []

    def run() -> None:
        try:
            model.generate(**kwargs)
        except BaseException as exc:  # surface the failure instead of blocking the consumer
            errors.append(exc)
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    for text in streamer:
        if text:
            yield text
    worker.join()
    if errors:
        raise errors[0]
'''

_TRANSFORMERS_CHAT = r'''

class TransformersChat:
    """PyTorch backend: CUDA, Apple Metal (MPS) or CPU."""

    def __init__(self, device: str | None = None, model_id: str | None = None, precision: str = PRECISION,
                 local_path: str | None = LOCAL_PATH) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = pick_device(device)
        source = model_source(model_id, local_path)
        config = AutoConfig.from_pretrained(source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        model_class = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
        self.model = attach_adapter(model_class.from_pretrained(source, **model_kwargs(self.device, precision)))
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, messages: list):
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                     return_dict=True)
            except Exception:
                return tokenizer.apply_chat_template(merge_system(messages), add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        return tokenizer(plain_prompt(messages), return_tensors="pt")

    def stream(self, messages: list, **overrides):
        """Yield the reply to ``messages`` as text pieces."""
        inputs = self.encode(messages).to(self.model.device)
        return stream_generate(self.model, self.tokenizer, inputs, sampling(overrides))
'''

_MLX_CHAT = r'''

MLX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-4bit")


def is_apple_silicon() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_mlx_checkpoint(path: str) -> bool:
    """MLX conversions record their quantization in config.json."""
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
            quantization = json.load(handle).get("quantization")
    except (OSError, ValueError):
        return False
    return isinstance(quantization, dict) and "bits" in quantization


def mlx_source() -> str:
    """The app's MLX build, an earlier conversion, an MLX repo, or a one-time 4-bit conversion."""
    local = model_source(None, LOCAL_PATH)
    if local != MODEL_ID:
        return local
    if is_mlx_checkpoint(MLX_DIR):
        return MLX_DIR
    from huggingface_hub import snapshot_download

    if is_mlx_checkpoint(snapshot_download(MODEL_ID, allow_patterns=["config.json"])):
        return MODEL_ID
    if os.path.exists(MLX_DIR):
        raise ExportError(f"{MLX_DIR} exists but is not an MLX checkpoint; delete it and run again.")
    from mlx_lm import convert

    sys.stderr.write(f"Converting {MODEL_ID} to 4-bit MLX in {MLX_DIR} (first run only)...\n")
    convert(hf_path=MODEL_ID, mlx_path=MLX_DIR, quantize=True, q_bits=4, q_group_size=64)
    return MLX_DIR


class MlxChat:
    """Apple Silicon backend (mlx-lm, Metal)."""

    def __init__(self, device: str | None = None) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(mlx_source())

    def encode(self, messages: list) -> list:
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            except Exception:
                return list(tokenizer.apply_chat_template(merge_system(messages), add_generation_prompt=True))
        return list(tokenizer.encode(plain_prompt(messages)))

    def stream(self, messages: list, **overrides):
        """Yield the reply to ``messages`` as text pieces."""
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        settings = sampling(overrides)
        greedy = settings["temperature"] <= 0
        sampler = make_sampler(temp=settings["temperature"], top_p=0.0 if greedy else settings["top_p"],
                               top_k=0 if greedy else settings["top_k"])
        penalty = settings["repetition_penalty"]
        processors = make_logits_processors(repetition_penalty=penalty if penalty != 1.0 else None)
        if settings["seed"] is not None:
            mx.random.seed(settings["seed"])
        for response in stream_generate(self.model, self.tokenizer, self.encode(messages),
                                        max_tokens=settings["max_new_tokens"], sampler=sampler,
                                        logits_processors=processors):
            if response.text:
                yield response.text


def load_runner(device: str | None = None):
    """MLX on Apple Silicon. Containers, Linux and Intel Macs cannot use Metal: they run
    BASE_MODEL_ID in FP16 with PyTorch instead."""
    if is_apple_silicon():
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            sys.stderr.write("mlx-lm is not installed; using the PyTorch FP16 fallback.\n")
        else:
            return MlxChat(device)
    return TransformersChat(device, model_id=BASE_MODEL_ID, precision="fp16", local_path=None)
'''

_LLAMA_CHAT = r'''

GGUF_PREFERENCE = ("Q4_K_M", "q4_k_m", "Q4_K_S", "Q5_K_M", "Q4_0")


def cached_gguf() -> str | None:
    """A GGUF file of MODEL_ID already in the Hugging Face cache (works offline, e.g. in Docker)."""
    try:
        from huggingface_hub import scan_cache_dir

        repos = scan_cache_dir().repos
    except Exception:
        return None
    for repo in repos:
        if repo.repo_id == MODEL_ID:
            names = sorted(str(f.file_path) for rev in repo.revisions for f in rev.files if f.file_name.endswith(".gguf"))
            if names:
                return next((n for tag in GGUF_PREFERENCE for n in names if tag in os.path.basename(n)), names[0])
    return None


def gguf_path() -> str:
    """The GGUF file the app used, else the selected quantization (downloaded once)."""
    if LOCAL_PATH:
        local = os.path.expanduser(LOCAL_PATH)
        candidate = os.path.join(local, GGUF_FILE) if GGUF_FILE and os.path.isdir(local) else local
        if os.path.isfile(candidate):
            return candidate
    from huggingface_hub import HfApi, hf_hub_download

    if GGUF_FILE:
        return hf_hub_download(MODEL_ID, GGUF_FILE)
    cached = cached_gguf()
    if cached:
        return cached
    names = [n for n in HfApi().list_repo_files(MODEL_ID) if n.endswith(".gguf")]
    if not names:
        raise ExportError(f"{MODEL_ID} has no .gguf file.")
    return hf_hub_download(MODEL_ID, next((n for tag in GGUF_PREFERENCE for n in names if tag in n), sorted(names)[0]))


class LlamaCppChat:
    """GGUF backend (llama-cpp-python): Metal on macOS, CUDA when built for it, else CPU."""

    def __init__(self, device: str | None = None) -> None:
        from llama_cpp import Llama

        gpu_layers = 0 if (device or DEVICE or "auto").lower() == "cpu" else -1
        self.llm = Llama(model_path=gguf_path(), n_ctx=int(EXTRA.get("n_ctx") or 4096), n_gpu_layers=gpu_layers,
                         verbose=False)

    def stream(self, messages: list, **overrides):
        """Yield the reply to ``messages`` as text pieces."""
        settings = sampling(overrides)
        chunks = self.llm.create_chat_completion(
            messages=messages, stream=True, temperature=settings["temperature"], top_p=settings["top_p"],
            top_k=settings["top_k"], repeat_penalty=settings["repetition_penalty"],
            max_tokens=settings["max_new_tokens"], seed=settings["seed"] if settings["seed"] is not None else -1)
        for chunk in chunks:
            text = chunk["choices"][0].get("delta", {}).get("content")
            if text:
                yield text


def load_runner(device: str | None = None):
    return LlamaCppChat(device)
'''

_TRANSFORMERS_RUNNER = r'''

def load_runner(device: str | None = None):
    return TransformersChat(device)
'''

_TEXT_MAIN = r'''

def main() -> None:
    defaults = sampling()
    parser = argparse.ArgumentParser(description=f"Chat with {MODEL_ID} (exported from Noelle).")
    parser.add_argument("prompt", nargs="?", help="user message (default: stdin, else a demo prompt)")
    parser.add_argument("--system", default=SYSTEM_PROMPT, help="system prompt")
    parser.add_argument("--device", default=DEVICE, help="auto, cuda, mps or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=defaults["repetition_penalty"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    args = parser.parse_args()
    prompt = read_prompt(args.prompt, "Introduce yourself in one sentence.")
    try:
        runner = load_runner(args.device)
        pieces = runner.stream(build_messages(prompt, args.system), temperature=args.temperature, top_p=args.top_p,
                               top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                               max_new_tokens=args.max_new_tokens, seed=args.seed)
        for text in pieces:
            sys.stdout.write(text)
            sys.stdout.flush()
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    sys.stdout.write("\n")
'''

_VISION = r'''

class VisionChat:
    """Image + prompt -> streamed text (VLMs, captioners and OCR models)."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        self.device = pick_device(device)
        source = model_source()
        config = AutoConfig.from_pretrained(source)
        self.processor = AutoProcessor.from_pretrained(source)
        self.encoder_decoder = getattr(config, "model_type", "") == "vision-encoder-decoder"
        if self.encoder_decoder:
            from transformers import VisionEncoderDecoderModel

            model = VisionEncoderDecoderModel.from_pretrained(source, **model_kwargs(self.device))
        else:
            model = AutoModelForImageTextToText.from_pretrained(source, **model_kwargs(self.device))
        self.model = attach_adapter(model)
        self.model.eval()
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def encode(self, image, prompt: str, system: str | None = None) -> dict:
        if self.encoder_decoder:
            batch = self.processor(images=image, return_tensors="pt")
        else:
            prompt = prompt or "Describe this image in detail."
            text = prompt
            if getattr(self.processor, "chat_template", None):
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                if system:
                    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
                try:
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                except Exception:
                    text = self.processor.apply_chat_template(messages[-1:], add_generation_prompt=True,
                                                              tokenize=False)
            batch = self.processor(text=[text], images=[image], return_tensors="pt")
        dtype = getattr(self.model, "dtype", None)
        inputs = {}
        for key, value in dict(batch).items():
            if hasattr(value, "to"):
                value = value.to(self.model.device)
                if dtype is not None and value.is_floating_point():
                    value = value.to(dtype)
            inputs[key] = value
        return inputs

    def stream(self, image, prompt: str, system: str | None = None, **overrides):
        """Yield the answer about ``image`` as text pieces."""
        return stream_generate(self.model, self.tokenizer, self.encode(image, prompt, system), sampling(overrides))


def load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def load_runner(device: str | None = None):
    return VisionChat(device)


def main() -> None:
    defaults = sampling()
    parser = argparse.ArgumentParser(description=f"Ask {MODEL_ID} about an image (exported from Noelle).")
    parser.add_argument("prompt", nargs="?", default="", help="question (default: describe the image)")
    parser.add_argument("--image", required=True, help="path to a PNG/JPEG/WebP image")
    parser.add_argument("--system", default=SYSTEM_PROMPT, help="system prompt (when the template supports it)")
    parser.add_argument("--device", default=DEVICE, help="auto, cuda, mps or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=defaults["repetition_penalty"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    args = parser.parse_args()
    try:
        runner = load_runner(args.device)
        pieces = runner.stream(load_image(args.image), args.prompt, args.system or None,
                               temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                               repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens,
                               seed=args.seed)
        for text in pieces:
            sys.stdout.write(text)
            sys.stdout.flush()
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    sys.stdout.write("\n")
'''

_AUDIO_IO = r'''

def resample(audio, orig_sr: int, target_sr: int):
    """Polyphase resampling (scipy) with a linear fallback; no ffmpeg or torchaudio needed."""
    import numpy as np

    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly
    except ImportError:
        duration = audio.shape[0] / float(orig_sr)
        target = np.linspace(0.0, duration, int(round(duration * target_sr)), endpoint=False)
        source = np.linspace(0.0, duration, audio.shape[0], endpoint=False)
        return np.interp(target, source, audio).astype(np.float32)
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)
'''

_ASR = r'''

def load_audio(source, target_sr: int):
    """Decode WAV/FLAC/OGG/MP3 with libsndfile (soundfile), downmix to mono float32 and resample."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(source, dtype="float32", always_2d=True)
    return resample(data.mean(axis=1).astype(np.float32), int(rate), int(target_sr))


class SpeechRecognizer:
    """transformers ASR pipeline fed with soundfile-decoded audio."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import pipeline

        self.device = pick_device(device)
        self.pipe = pipeline("automatic-speech-recognition", model=model_source(), device=self.device,
                             dtype=pick_dtype(self.device))
        self.model_type = str(getattr(self.pipe.model.config, "model_type", "") or "")
        self.sampling_rate = int(getattr(self.pipe.feature_extractor, "sampling_rate", 16000))

    def transcribe(self, audio, language: str | None = None, translate: bool = False,
                   timestamps: bool = True) -> dict:
        """``audio``: a path, a file object or a mono float32 array at ``sampling_rate``."""
        if not hasattr(audio, "shape"):
            audio = load_audio(audio, self.sampling_rate)
        whisper = "whisper" in self.model_type
        kwargs = {}
        if timestamps:
            kwargs["return_timestamps"] = True if whisper else "word"
        if whisper:
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            if translate:
                generate_kwargs["task"] = "translate"
            if generate_kwargs:
                kwargs["generate_kwargs"] = generate_kwargs
            if audio.shape[0] > 30 * self.sampling_rate:
                kwargs["chunk_length_s"] = 30
        out = self.pipe({"raw": audio, "sampling_rate": self.sampling_rate}, **kwargs)
        chunks = []
        for chunk in out.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            chunks.append({"start": start, "end": end, "text": str(chunk.get("text", "")).strip()})
        return {"text": str(out.get("text", "")).strip(), "chunks": chunks,
                "duration_s": audio.shape[0] / float(self.sampling_rate)}


def load_runner(device: str | None = None):
    return SpeechRecognizer(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Transcribe audio with {MODEL_ID} (exported from Noelle).")
    parser.add_argument("audio", help="audio file (WAV, FLAC, OGG or MP3)")
    parser.add_argument("--language", default=EXTRA.get("language"), help="spoken language, e.g. en (Whisper)")
    parser.add_argument("--translate", action="store_true", help="translate to English (Whisper)")
    parser.add_argument("--timestamps", action="store_true", help="print timestamped segments")
    parser.add_argument("--device", default=DEVICE, help="auto, cuda, mps or cpu")
    args = parser.parse_args()
    try:
        result = load_runner(args.device).transcribe(args.audio, language=args.language, translate=args.translate,
                                                     timestamps=args.timestamps)
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    if args.timestamps and result["chunks"]:
        for chunk in result["chunks"]:
            print(f"[{chunk['start']} - {chunk['end']}] {chunk['text']}")
    else:
        print(result["text"])
'''

_TTS = r'''

def time_stretch(audio, rate: float, frame: int = 1024, hop: int = 256):
    """Overlap-add time stretch: rate > 1 plays faster with the pitch preserved (as in Noelle)."""
    import numpy as np

    if abs(rate - 1.0) < 1e-3 or audio.size < frame:
        return audio.astype(np.float32)
    window = np.hanning(frame).astype(np.float32)
    hop_in = max(1, int(round(hop * rate)))
    n_frames = max(1, (len(audio) - frame) // hop_in + 1)
    out_len = (n_frames - 1) * hop + frame
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        piece = audio[i * hop_in: i * hop_in + frame]
        if len(piece) < frame:
            piece = np.pad(piece, (0, frame - len(piece)))
        out[i * hop: i * hop + frame] += piece * window
        norm[i * hop: i * hop + frame] += window
    norm[norm < 1e-6] = 1.0
    return out / norm


def pitch_shift(audio, rate: int, semitones: float):
    """Shift pitch without changing duration: resample, then stretch back."""
    factor = 2.0 ** (semitones / 12.0)
    shifted = resample(audio, int(round(rate * factor)), rate)
    return time_stretch(shifted, 1.0 / factor)[: len(audio)]


def peak_normalize(audio, peak: float = 0.95):
    import numpy as np

    top = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio.astype(np.float32) if top < 1e-9 else (audio * (peak / top)).astype(np.float32)


class SpeechSynthesizer:
    """transformers text-to-audio pipeline with the playground's speed and pitch controls."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import pipeline

        self.device = pick_device(device)
        self.pipe = pipeline("text-to-audio", model=model_source(), device=self.device)
        self.model_type = str(getattr(self.pipe.model.config, "model_type", "") or "")

    def synthesize(self, text: str, speed: float | None = None, pitch: float | None = None,
                   voice: int | None = None):
        """Return ``(mono float32 samples, sample rate)``."""
        import numpy as np

        forward_params = {}
        if self.model_type == "speecht5":
            import torch

            generator = torch.Generator().manual_seed(int(voice if voice is not None else EXTRA.get("voice") or 0))
            forward_params["speaker_embeddings"] = torch.randn((1, 512), generator=generator) * 0.05
        out = self.pipe(text, forward_params=forward_params) if forward_params else self.pipe(text)
        audio = np.asarray(out["audio"], dtype=np.float32)
        rate = int(out["sampling_rate"])
        if audio.ndim > 1:
            audio = audio.squeeze()
            if audio.ndim > 1:  # (channels, samples) -> mono
                audio = audio.mean(axis=0)
        speed = float(speed if speed is not None else EXTRA.get("speed") or 1.0)
        pitch = float(pitch if pitch is not None else EXTRA.get("pitch") or 0.0)
        if speed > 0 and abs(speed - 1.0) > 1e-3:
            audio = time_stretch(audio, speed)
        if abs(pitch) > 1e-3:
            audio = pitch_shift(audio, rate, pitch)
        return peak_normalize(audio), rate


def load_runner(device: str | None = None):
    return SpeechSynthesizer(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Synthesize speech with {MODEL_ID} (exported from Noelle).")
    parser.add_argument("text", nargs="?", help="text to speak (default: stdin, else a demo sentence)")
    parser.add_argument("--output", default="speech.wav", help="WAV file to write")
    parser.add_argument("--speed", type=float, default=None, help="tempo factor, pitch preserved")
    parser.add_argument("--pitch", type=float, default=None, help="pitch shift in semitones")
    parser.add_argument("--voice", type=int, default=None, help="voice seed (SpeechT5)")
    parser.add_argument("--device", default=DEVICE, help="auto, cuda, mps or cpu")
    args = parser.parse_args()
    text = read_prompt(args.text, "Hello from Noelle Desktop AI Studio.")
    try:
        audio, rate = load_runner(args.device).synthesize(text, speed=args.speed, pitch=args.pitch, voice=args.voice)
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    import soundfile as sf

    sf.write(args.output, audio, rate)
    print(f"Wrote {args.output} ({len(audio) / rate:.1f} s at {rate} Hz)")
'''

_DIFFUSION = r'''

class ImageGenerator:
    """diffusers text-to-image pipeline with the playground's sampler settings."""

    def __init__(self, device: str | None = None) -> None:
        from diffusers import AutoPipelineForText2Image, DiffusionPipeline

        self.device = pick_device(device)
        dtype = pick_dtype(self.device)
        source = model_source()
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(source, dtype=dtype)
        except Exception:
            pipe = DiffusionPipeline.from_pretrained(source, dtype=dtype)
        if ADAPTER_ID:
            pipe.load_lora_weights(ADAPTER_ID)
        self.pipe = pipe.to(self.device)
        if self.device == "mps" and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, negative_prompt: str | None = None, steps: int | None = None,
                 guidance: float | None = None, width: int | None = None, height: int | None = None,
                 seed: int | None = None):
        """Return ``(PIL image, seed)``; a negative or missing seed picks a random one."""
        import inspect

        import torch

        def pick(value, key, fallback):
            return value if value is not None else (EXTRA.get(key) if EXTRA.get(key) is not None else fallback)

        seed = pick(seed, "seed", -1)
        seed = int(seed) if int(seed) >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        call = {
            "prompt": prompt,
            "num_inference_steps": max(1, int(pick(steps, "steps", 25))),
            "guidance_scale": float(pick(guidance, "guidance", 7.0)),
            "width": max(64, int(pick(width, "width", 512)) // 8 * 8),
            "height": max(64, int(pick(height, "height", 512)) // 8 * 8),
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }
        negative_prompt = pick(negative_prompt, "negative_prompt", None)
        if negative_prompt and "negative_prompt" in inspect.signature(self.pipe.__call__).parameters:
            call["negative_prompt"] = negative_prompt
        return self.pipe(**call).images[0], seed


def load_runner(device: str | None = None):
    return ImageGenerator(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate an image with {MODEL_ID} (exported from Noelle).")
    parser.add_argument("prompt", nargs="?", help="what to draw (default: stdin, else a demo prompt)")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--steps", type=int, default=None, help=f"default {EXTRA.get('steps', 25)}")
    parser.add_argument("--guidance", type=float, default=None, help=f"default {EXTRA.get('guidance', 7.0)}")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="-1 = random")
    parser.add_argument("--output", default="image.png", help="PNG file to write")
    parser.add_argument("--device", default=DEVICE, help="auto, cuda, mps or cpu")
    args = parser.parse_args()
    prompt = read_prompt(args.prompt, "A watercolor lighthouse at dawn, soft light")
    try:
        image, seed = load_runner(args.device).generate(prompt, negative_prompt=args.negative_prompt,
                                                        steps=args.steps, guidance=args.guidance, width=args.width,
                                                        height=args.height, seed=args.seed)
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    image.save(args.output)
    print(f"Saved {args.output} ({image.width}x{image.height}, seed {seed})")
'''

_EMBEDDINGS = r'''

class Embedder:
    """sentence-transformers model producing unit-length embeddings."""

    def __init__(self, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = pick_device(device)
        self.model = SentenceTransformer(model_source(), device=self.device)

    def embed(self, texts: list):
        import numpy as np

        vectors = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def similarity(self, query: str, sentences: list) -> list:
        """Cosine similarity of each sentence to ``query`` (dot product of unit vectors)."""
        vectors = self.embed([query, *sentences])
        return [float(score) for score in vectors[1:] @ vectors[0]]


def load_runner(device: str | None = None):
    return Embedder(device)


DEMO_SENTENCES = [
    "The cat sits on the mat.",
    "A feline is resting on a rug.",
    "Stock markets fell sharply today.",
    "Quantum computers use qubits.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Semantic similarity with {MODEL_ID} (exported from Noelle).")
    parser.add_argument("query", nargs="?", default="A cat lying on a carpet.", help="reference sentence")
    parser.add_argument("sentences", nargs="*", help="sentences to rank (default: a small demo set)")
    parser.add_argument("--device", default=DEVICE, help="auto, cuda, mps or cpu")
    args = parser.parse_args()
    sentences = args.sentences or DEMO_SENTENCES
    try:
        scores = load_runner(args.device).similarity(args.query, sentences)
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    print(f"Query: {args.query}")
    for score, sentence in sorted(zip(scores, sentences), reverse=True):
        print(f"{score:7.4f}  {sentence}")
'''

_MAIN_GUARD = r'''

if __name__ == "__main__":
    main()
'''

_BACKEND_SECTIONS: dict[str, tuple[str, ...]] = {
    "transformers": (_TORCH, _HF_MODEL, _SAMPLING, _CHAT_HELPERS, _HF_STREAM, _TRANSFORMERS_CHAT,
                     _TRANSFORMERS_RUNNER, _TEXT_MAIN),
    "mlx": (_TORCH, _HF_MODEL, _SAMPLING, _CHAT_HELPERS, _HF_STREAM, _TRANSFORMERS_CHAT, _MLX_CHAT, _TEXT_MAIN),
    "llama_cpp": (_SAMPLING, _CHAT_HELPERS, _LLAMA_CHAT, _TEXT_MAIN),
    "vision": (_TORCH, _HF_MODEL, _SAMPLING, _HF_STREAM, _VISION),
    "asr": (_TORCH, _AUDIO_IO, _ASR),
    "tts": (_TORCH, _AUDIO_IO, _TTS),
    "diffusers": (_TORCH, _DIFFUSION),
    "sentence_transformers": (_TORCH, _EMBEDDINGS),
}
_USAGE = {
    "transformers": 'python inference.py "Explain KV caching in two sentences."',
    "mlx": 'python inference.py "Explain KV caching in two sentences."',
    "llama_cpp": 'python inference.py "Explain KV caching in two sentences."',
    "vision": 'python inference.py --image photo.jpg "What is in this picture?"',
    "asr": "python inference.py recording.wav --timestamps",
    "tts": 'python inference.py "Hello there." --output hello.wav',
    "diffusers": 'python inference.py "A lighthouse at dawn" --seed 42 --output lighthouse.png',
    "sentence_transformers": 'python inference.py "A cat on a carpet" "The cat sits on the mat." "Stocks fell."',
}


def _backend_label(plan: _Plan) -> str:
    return {"transformers": "transformers", "mlx": "mlx-lm, PyTorch FP16 fallback", "llama_cpp": "llama.cpp",
            "vision": "transformers", "asr": "transformers pipeline", "tts": "transformers pipeline",
            "diffusers": "diffusers", "sentence_transformers": "sentence-transformers"}[plan.backend]


def python_script(config: ExportConfig) -> str:
    """Self-contained ``inference.py`` reproducing the playground configuration."""
    plan = _resolve(config)
    header = [
        "#!/usr/bin/env python3",
        "# Noelle Desktop AI Studio export: reproduces a playground configuration outside the app.",
        "#",
        f"#   Model:     {_one_line(config.model_id)}",
        f"#   Task:      {_one_line(config.task)}",
        f"#   Precision: {plan.precision} ({_backend_label(plan)})",
        f"#   Exported:  {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "#",
        "# Requirements (pinned from the exporting environment):",
        *[f"#   {line}" for line in _install_lines(plan, config)],
        "#",
        "# Usage:",
        f"#   {_USAGE[plan.backend]}",
        "#   python inference.py --help",
    ]
    if plan.notes:
        header += ["#", *[f"# Note: {note}" for note in plan.notes]]
    imports = ["import argparse", "import os", "import sys"]
    if plan.backend == "mlx":
        imports.insert(1, "import json")
    constants = [
        "# --- Exported configuration -------------------------------------------------",
        f"MODEL_ID = {_literal(config.model_id)}",
        f"LOCAL_PATH = {_literal(config.local_path)}  # folder the app used; ignored when missing here",
        f"ADAPTER_ID = {_literal(config.adapter_id if plan.family in _ADAPTER_FAMILIES else None)}",
        f"TASK = {_literal(config.task)}",
        f"PRECISION = {_literal(plan.precision)}",
        f"DEVICE = {_literal(config.device or 'auto')}",
        f"SYSTEM_PROMPT = {_literal(config.system_prompt or '')}",
        f"GENERATION = {_literal(_generation(config))}",
        f"EXTRA = {_literal(config.extra or {})}",
    ]
    if plan.backend == "mlx":
        constants.append(f"BASE_MODEL_ID = {_literal(_base_model(config))}  # FP16 model used where MLX cannot run")
    if plan.backend == "llama_cpp":
        constants.append(f"GGUF_FILE = {_literal(config.gguf_file)}")
    body = "".join((_COMMON, *_BACKEND_SECTIONS[plan.backend], _MAIN_GUARD))
    doc = repr(f"Standalone inference for {_one_line(config.model_id)} ({_one_line(config.task)}).")
    return "\n".join([*header, doc, "", "from __future__ import annotations", "", *imports, "", *constants]) + \
        "\n" + body


# ---------------------------------------------------------------------------
# server.py
# ---------------------------------------------------------------------------

_SERVER_COMMON = r'''

STATE = {"runner": None}
LOCK = threading.Lock()  # one request at a time on the accelerator


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE["runner"] = inference.load_runner(os.environ.get("NOELLE_DEVICE") or inference.DEVICE)
    yield
    STATE["runner"] = None


app = FastAPI(title=f"{inference.MODEL_ID} (Noelle export)", lifespan=lifespan)


def engine():
    current = STATE["runner"]
    if current is None:
        raise HTTPException(status_code=503, detail="The model is still loading.")
    return current


@app.get("/health")
def health() -> JSONResponse:
    ready = STATE["runner"] is not None
    return JSONResponse({"status": "ok" if ready else "loading", "model": inference.MODEL_ID,
                         "task": inference.TASK}, status_code=200 if ready else 503)
'''

_SERVER_CHAT_COMMON = r'''

class ChatMessage(BaseModel):
    role: str = "user"
    content: str | list | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    seed: int | None = None
    stream: bool = False


def text_of(content) -> str:
    """Text of a message: a string or the text parts of OpenAI-style content arrays."""
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content
                       if isinstance(part, dict) and part.get("type", "text") == "text")
    return "" if content is None else str(content)


def overrides(req: ChatRequest) -> dict:
    return {"temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
            "max_new_tokens": req.max_completion_tokens or req.max_tokens, "seed": req.seed}


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def respond(make_pieces, stream: bool):
    """OpenAI chat.completion (or chat.completion.chunk events when ``stream`` is true)."""
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": inference.MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    if stream:
        def events():
            with LOCK:
                yield sse(chunk({"role": "assistant", "content": ""}))
                for text in make_pieces():
                    yield sse(chunk({"content": text}))
            yield sse(chunk({}, "stop"))
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    with LOCK:
        text = "".join(make_pieces())
    return {"id": completion_id, "object": "chat.completion", "created": created, "model": inference.MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
'''

_SERVER_TEXT = r'''

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    runner = engine()
    messages = [{"role": m.role, "content": text_of(m.content)} for m in req.messages if m.content is not None]
    if not messages:
        raise HTTPException(status_code=400, detail="messages is empty")
    if inference.SYSTEM_PROMPT and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": inference.SYSTEM_PROMPT})
    return respond(lambda: runner.stream(messages, **overrides(req)), req.stream)
'''

_SERVER_VISION = r'''

MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Fetching arbitrary URLs from a server is an SSRF risk, so remote images are off by default:
# clients send base64 data: URLs. To allow https images, set ALLOW_REMOTE_IMAGE_URLS=1 and list the
# permitted hosts in REMOTE_IMAGE_HOSTS (comma-separated). Private, loopback, link-local and reserved
# addresses are always refused and redirects are never followed.
import ipaddress
import os
import socket
import urllib.parse

ALLOW_REMOTE_IMAGES = os.environ.get("ALLOW_REMOTE_IMAGE_URLS") == "1"
REMOTE_IMAGE_HOSTS = {h.strip().lower() for h in os.environ.get("REMOTE_IMAGE_HOSTS", "").split(",") if h.strip()}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # a redirect could point at an internal address


def _resolves_to_public_addresses(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified):
            return False
    return bool(infos)


def load_image_url(url: str):
    """Decode a base64 data: URL, or fetch an allow-listed https URL when explicitly enabled."""
    from PIL import Image

    try:
        if url.startswith("data:"):
            raw = base64.b64decode(url.partition(",")[2], validate=False)
        elif url.startswith("https://"):
            host = (urllib.parse.urlsplit(url).hostname or "").lower()
            if not ALLOW_REMOTE_IMAGES or host not in REMOTE_IMAGE_HOSTS:
                raise HTTPException(status_code=400, detail="remote image URLs are disabled on this server; send a "
                                    "base64 data: URL (or set ALLOW_REMOTE_IMAGE_URLS=1 and REMOTE_IMAGE_HOSTS)")
            if not _resolves_to_public_addresses(host):
                raise HTTPException(status_code=400, detail="image host resolves to a non-public address")
            opener = urllib.request.build_opener(_NoRedirect)
            with opener.open(url, timeout=30) as response:
                raw = response.read(MAX_IMAGE_BYTES + 1)
        else:
            raise HTTPException(status_code=400, detail="image_url must be a base64 data: URL")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image is larger than 20 MB")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"unreadable image: {exc}") from exc


def first_image(content):
    if not isinstance(content, list):
        return None
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if isinstance(url, str) and url:
                return load_image_url(url)
    return None


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-style vision chat: the last user message carries the text and an image_url part."""
    runner = engine()
    system, prompt, image = None, "", None
    for message in req.messages:
        if message.role == "system":
            system = text_of(message.content)
        elif message.role == "user":
            prompt = text_of(message.content)
            image = first_image(message.content) or image
    if image is None:
        raise HTTPException(status_code=400, detail="Send the image as an image_url content part.")
    system = system or inference.SYSTEM_PROMPT or None
    return respond(lambda: runner.stream(image, prompt, system, **overrides(req)), req.stream)
'''

_SERVER_ASR = r'''

@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request, language: str | None = None, translate: bool = False,
                         timestamps: bool = False):
    """Body: the raw audio file (WAV, FLAC, OGG or MP3). Query: ?language=en&translate=false."""
    runner = engine()
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Send the audio file as the request body.")
    try:
        audio = inference.load_audio(io.BytesIO(data), runner.sampling_rate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"unreadable audio: {exc}") from exc

    def work() -> dict:
        with LOCK:
            return runner.transcribe(audio, language=language or inference.EXTRA.get("language"),
                                     translate=translate, timestamps=timestamps)

    return await run_in_threadpool(work)
'''

_SERVER_TTS = r'''

class SpeechRequest(BaseModel):
    input: str
    speed: float | None = None
    pitch: float | None = None
    voice: int | None = None


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    """Returns audio/wav."""
    runner = engine()
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input is empty")
    with LOCK:
        audio, rate = runner.synthesize(req.input, speed=req.speed, pitch=req.pitch, voice=req.voice)
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")
'''

_SERVER_DIFFUSION = r'''

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    size: str | None = None  # OpenAI style, e.g. "768x512"
    seed: int | None = None


@app.post("/v1/images/generations")
def generate_image(req: ImageRequest) -> dict:
    """Returns base64 PNG data plus the seed used."""
    runner = engine()
    width, height = req.width, req.height
    if req.size:
        try:
            width, height = (int(v) for v in req.size.lower().split("x", 1))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="size must look like 512x512") from exc
    with LOCK:
        image, seed = runner.generate(req.prompt, negative_prompt=req.negative_prompt, steps=req.steps,
                                      guidance=req.guidance, width=width, height=height, seed=req.seed)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"), "seed": seed}]}
'''

_SERVER_EMBEDDINGS = r'''

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


class SimilarityRequest(BaseModel):
    source: str
    sentences: list[str]


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest) -> dict:
    runner = engine()
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input is empty")
    with LOCK:
        vectors = runner.embed(texts)
    return {"object": "list", "model": inference.MODEL_ID,
            "data": [{"object": "embedding", "index": i, "embedding": vector.tolist()}
                     for i, vector in enumerate(vectors)]}


@app.post("/v1/similarity")
def similarity(req: SimilarityRequest) -> dict:
    """Cosine similarity of each sentence to ``source``."""
    runner = engine()
    with LOCK:
        scores = runner.similarity(req.source, req.sentences)
    return {"model": inference.MODEL_ID, "scores": scores}
'''

_SERVER_MAIN = r'''

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
'''

_SERVER_ROUTES = {
    tasks.TEXT: ("POST /v1/chat/completions (OpenAI-compatible, stream supported)",
                 ["import json", "import os", "import threading", "import time", "import uuid"],
                 ["from fastapi.responses import JSONResponse, StreamingResponse", "from pydantic import BaseModel"],
                 (_SERVER_CHAT_COMMON, _SERVER_TEXT)),
    tasks.VISION: ("POST /v1/chat/completions (OpenAI-style image_url parts)",
                   ["import base64", "import io", "import json", "import os", "import threading", "import time",
                    "import urllib.request", "import uuid"],
                   ["from fastapi.responses import JSONResponse, StreamingResponse", "from pydantic import BaseModel"],
                   (_SERVER_CHAT_COMMON, _SERVER_VISION)),
    tasks.ASR: ("POST /v1/audio/transcriptions (raw audio body)",
                ["import io", "import os", "import threading"],
                ["from fastapi import Request", "from fastapi.responses import JSONResponse",
                 "from starlette.concurrency import run_in_threadpool"],
                (_SERVER_ASR,)),
    tasks.TTS: ("POST /v1/audio/speech -> audio/wav",
                ["import io", "import os", "import threading"],
                ["from fastapi.responses import JSONResponse, Response", "from pydantic import BaseModel"],
                (_SERVER_TTS,)),
    tasks.DIFFUSION: ("POST /v1/images/generations -> base64 PNG",
                      ["import base64", "import io", "import os", "import threading", "import time"],
                      ["from fastapi.responses import JSONResponse", "from pydantic import BaseModel"],
                      (_SERVER_DIFFUSION,)),
    tasks.EMBEDDING: ("POST /v1/embeddings and POST /v1/similarity",
                      ["import os", "import threading"],
                      ["from fastapi.responses import JSONResponse", "from pydantic import BaseModel"],
                      (_SERVER_EMBEDDINGS,)),
}


def server_script(config: ExportConfig) -> str:
    """Minimal FastAPI ``server.py`` (imports ``inference.py`` from the same folder)."""
    plan = _resolve(config)
    route, stdlib, third_party, sections = _SERVER_ROUTES[plan.family]
    lines = [
        "#!/usr/bin/env python3",
        f"# Noelle Desktop AI Studio export: HTTP server for {_one_line(config.model_id)} ({_one_line(config.task)}).",
        "#",
        "#   uvicorn server:app --host 127.0.0.1 --port 8000",
        "#   GET  /health",
        f"#   {route}",
        "#",
        "# Keep inference.py next to this file. NOELLE_DEVICE=cuda|mps|cpu overrides the device.",
        repr("Minimal FastAPI server around inference.py."),
        "",
        "from __future__ import annotations",
        "",
        *stdlib,
        "from contextlib import asynccontextmanager",
        "",
        "from fastapi import FastAPI, HTTPException",
        *third_party,
        "",
        "import inference",
    ]
    return "\n".join(lines) + "\n" + _SERVER_COMMON + "".join(sections) + _SERVER_MAIN


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


def _download_script(config: ExportConfig, plan: _Plan) -> str:
    """Python run at build time (inside the BuildKit secret mount) to cache the weights."""
    repo = _base_model(config) if plan.backend == "mlx" else config.model_id
    lines = ["import os", "", "from huggingface_hub import HfApi, hf_hub_download, snapshot_download", "",
             'token = os.environ.get("HF_TOKEN") or None', f"repo_id = {repo!r}"]
    if plan.backend == "llama_cpp":
        lines += [
            f"filename = {config.gguf_file!r}",
            "if not filename:",
            '    names = [n for n in HfApi(token=token).list_repo_files(repo_id) if n.endswith(".gguf")]',
            '    preferred = ("Q4_K_M", "q4_k_m", "Q4_K_S", "Q5_K_M", "Q4_0")',
            "    filename = next((n for tag in preferred for n in names if tag in n), sorted(names)[0])",
            "hf_hub_download(repo_id, filename, token=token)  # one quantization, not the whole repo",
        ]
    else:
        lines += [
            "names = HfApi(token=token).list_repo_files(repo_id)",
            f"ignore = {_ALWAYS_SKIP!r}",
            'if any(n.endswith(".safetensors") for n in names):',
            f"    ignore += {_REDUNDANT_WITH_SAFETENSORS!r}",
            '    if any(".fp16." in n for n in names) and any(n.endswith(".safetensors") and ".fp16." not in n',
            "                                                 for n in names):",
            '        ignore.append("*.fp16.*")',
            "snapshot_download(repo_id, token=token, ignore_patterns=ignore)",
        ]
    if config.adapter_id and plan.family in _ADAPTER_FAMILIES and plan.backend != "llama_cpp":
        lines.append(f"snapshot_download({config.adapter_id!r}, token=token)")
    lines.append('print("Weights cached in", os.environ.get("HF_HOME"))')
    return "\n".join(lines)


def _docker_tag(config: ExportConfig) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", config.model_id.lower()).strip("-")[:48] or "model"
    return f"noelle-{slug}"


def dockerfile(config: ExportConfig) -> str:
    """CUDA runtime image serving ``server.py``; weights downloaded at build time via a BuildKit secret."""
    plan = _resolve(config)
    tag = _docker_tag(config)
    lines = [
        "# syntax=docker/dockerfile:1",
        f"# Noelle Desktop AI Studio export: {_one_line(config.model_id)} ({_one_line(config.task)}, {plan.precision})",
        "#",
        "# Build with BuildKit. The Hugging Face token is a build secret: it is mounted only while the",
        "# weights download and is never stored in an image layer, ENV or ARG.",
        "#   export HF_TOKEN=...   # only for gated or private repositories",
        f"#   docker build --secret id=hf_token,env=HF_TOKEN -t {tag} .",
        "# Run (NVIDIA Container Toolkit):",
        f"#   docker run --rm --gpus all -p 8000:8000 {tag}",
        "#",
        "# Base image: Ubuntu 24.04 provides Python 3.12. The pinned requirements need Python >= 3.12",
        "# (numpy, scipy), which the Ubuntu 22.04 CUDA images (Python 3.10) cannot install.",
    ]
    if plan.backend == "mlx":
        lines += [
            "#",
            "# MLX (mlx-4bit) runs only on Apple Silicon through Metal. Containers cannot use Metal, so this",
            f"# image serves the base model {_one_line(_base_model(config))} in FP16 with PyTorch on the GPU",
            "# (inference.load_runner picks that fallback automatically). Set BASE_MODEL_ID in inference.py",
            "# if the exported id is an MLX conversion.",
        ]
    extra_apt = ""
    if plan.backend == "llama_cpp":
        extra_apt = " build-essential cmake"
        lines += [
            "#",
            "# llama-cpp-python is compiled here for the CPU. For GPU offload build it with",
            '# CMAKE_ARGS="-DGGML_CUDA=on" on an nvidia/cuda:*-devel image (needs the CUDA compiler).',
        ]
    if plan.precision in ("nf4", "int8", "awq-4bit", "gptq-4bit"):
        lines += ["#", f"# {plan.precision} kernels need an NVIDIA GPU at run time (docker run --gpus all)."]
    lines += [
        f"FROM {DOCKER_BASE_IMAGE}",
        "",
        "ENV DEBIAN_FRONTEND=noninteractive \\",
        "    PYTHONUNBUFFERED=1 \\",
        "    PIP_NO_CACHE_DIR=1 \\",
        "    PIP_DISABLE_PIP_VERSION_CHECK=1 \\",
        "    VIRTUAL_ENV=/opt/venv \\",
        "    PATH=/opt/venv/bin:$PATH \\",
        "    HF_HOME=/opt/hf \\",
        "    HF_HUB_DISABLE_TELEMETRY=1",
        "",
        "RUN apt-get update \\",
        f"    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates{extra_apt} \\",
        "    && rm -rf /var/lib/apt/lists/* \\",
        "    && python3 -m venv /opt/venv \\",
        f"    && useradd --create-home --uid {CONTAINER_UID} --shell /usr/sbin/nologin noelle \\",
        "    && mkdir -p /opt/hf /app \\",
        "    && chown noelle:noelle /opt/hf /app",
        "",
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install --upgrade pip && pip install -r requirements.txt",
        "",
        "USER noelle",
        "# Download the weights at build time so the container starts offline. The secret file is",
        "# readable by the noelle user only during this step.",
        f"RUN --mount=type=secret,id=hf_token,uid={CONTAINER_UID} "
        "HF_TOKEN=\"$(cat /run/secrets/hf_token 2>/dev/null || true)\" python - <<'PY'",
        _download_script(config, plan),
        "PY",
        "",
        "ENV HF_HUB_OFFLINE=1",
        "COPY inference.py server.py ./",
        "EXPOSE 8000",
        "HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \\",
        "    CMD [\"python\", \"-c\", \"import sys, urllib.request; sys.exit(0 if urllib.request.urlopen("
        "'http://127.0.0.1:8000/health', timeout=8).status == 200 else 1)\"]",
        'ENTRYPOINT ["uvicorn", "server:app"]',
        'CMD ["--host", "0.0.0.0", "--port", "8000"]',
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# README.md and bundle
# ---------------------------------------------------------------------------

_CURL = {
    tasks.TEXT: "curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \\\n"
                "  -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 128}'",
    tasks.VISION: "curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\": "
                  "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Describe it\"}, "
                  "{\"type\": \"image_url\", \"image_url\": {\"url\": \"https://example.com/cat.jpg\"}}]}]}'",
    tasks.ASR: "curl http://127.0.0.1:8000/v1/audio/transcriptions?language=en --data-binary @recording.wav",
    tasks.TTS: "curl http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \\\n"
               "  -d '{\"input\": \"Hello there\"}' -o speech.wav",
    tasks.DIFFUSION: "curl http://127.0.0.1:8000/v1/images/generations -H 'Content-Type: application/json' \\\n"
                     "  -d '{\"prompt\": \"A lighthouse at dawn\", \"seed\": 42}'",
    tasks.EMBEDDING: "curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' \\\n"
                     "  -d '{\"input\": [\"first sentence\", \"second sentence\"]}'",
}


def readme_markdown(config: ExportConfig) -> str:
    """README.md with setup, CLI, server and Docker instructions."""
    plan = _resolve(config)
    tag = _docker_tag(config)
    model = _one_line(config.model_id)
    lines = [
        f"# {model} - Noelle export",
        "",
        "Reproduces a Noelle Desktop AI Studio playground configuration outside the app.",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Model | `{model}` |",
        f"| Task | `{_one_line(config.task)}` |",
        f"| Precision | `{plan.precision}` ({_backend_label(plan)}) |",
        f"| Device | `{_one_line(config.device or 'auto')}` (auto = CUDA, then Apple Metal, then CPU) |",
    ]
    if config.adapter_id and plan.family in _ADAPTER_FAMILIES:
        lines.append(f"| Adapter | `{_one_line(config.adapter_id)}` |")
    if plan.backend == "llama_cpp":
        lines.append(f"| GGUF file | `{_one_line(config.gguf_file or 'picked automatically (Q4_K_M preferred)')}` |")
    lines += [
        "",
        "Sampling settings, the system prompt and task options are stored at the top of `inference.py`.",
    ]
    if plan.notes:
        lines += ["", *[f"> {note}" for note in plan.notes]]
    lines += [
        "",
        "## Files",
        "",
        "- `inference.py`: command-line inference (`python inference.py --help`)",
        "- `server.py`: minimal FastAPI server built on `inference.py`",
        "- `requirements.txt`: package versions pinned from the exporting environment",
        "- `Dockerfile`: NVIDIA CUDA image that downloads the weights at build time",
        "",
        "## Run locally",
        "",
        "```bash",
        "python3 -m venv .venv && source .venv/bin/activate",
        "pip install -r requirements.txt",
        _USAGE[plan.backend],
        "```",
        "",
        "The first run downloads the weights into the Hugging Face cache. For gated or private models, run",
        "`hf auth login` or export `HF_TOKEN` first. If the snapshot the app used (`LOCAL_PATH` in",
        "`inference.py`) exists on this machine, it is used directly.",
        "",
        "## HTTP server",
        "",
        "```bash",
        "uvicorn server:app --host 127.0.0.1 --port 8000",
        "curl http://127.0.0.1:8000/health",
        _CURL[plan.family],
        "```",
        "",
        "## Docker (NVIDIA GPU)",
        "",
        "```bash",
        "export HF_TOKEN=...   # only for gated or private models",
        f"docker build --secret id=hf_token,env=HF_TOKEN -t {tag} .",
        f"docker run --rm --gpus all -p 8000:8000 {tag}",
        "```",
        "",
        "The token is passed as a BuildKit secret: it is used only while the weights download and never",
        "ends up in an image layer. With an older Docker CLI, use",
        "`--secret id=hf_token,src=$HOME/.cache/huggingface/token` instead.",
    ]
    if plan.backend == "mlx":
        lines += ["", "MLX needs Apple Silicon (Metal). The container runs the base model "
                      f"`{_one_line(_base_model(config))}` in FP16 with PyTorch instead."]
    if plan.precision in ("nf4", "int8", "awq-4bit", "gptq-4bit"):
        lines += ["", f"`{plan.precision}` needs an NVIDIA CUDA GPU; it does not run on Apple Silicon or CPU."]
    return "\n".join(lines) + "\n"


def export_bundle(config: ExportConfig, out_dir: str | os.PathLike[str]) -> list[Path]:
    """Write inference.py, requirements.txt, Dockerfile, server.py and README.md; return their paths."""
    contents = {
        "inference.py": python_script(config),
        "requirements.txt": requirements_txt(config),
        "Dockerfile": dockerfile(config),
        "server.py": server_script(config),
        "README.md": readme_markdown(config),
    }
    folder = Path(out_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in contents.items():
        path = folder / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    try:
        (folder / "inference.py").chmod(0o755)
    except OSError:
        pass
    return written
