"""Loaded-model cache with memory pre-flight checks and LRU eviction.

All mutating methods run on the worker's single GPU lane thread. ``snapshot``
and ``resident_bytes`` may be called from the I/O thread and take the lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from noelle.runtime import devices, tasks
from noelle.runtime import runnability
from noelle.runtime.errors import (
    BadRequest,
    DependencyMissing,
    Gated,
    LoadFailed,
    ModelNotFound,
    NeedsNewerRuntime,
    NoelleRuntimeError,
    PreflightRejected,
    RemoteCodeBlocked,
    Unsupported,
)
from noelle.runtime.generation import (
    LlamaCppTextEngine,
    LlamaServerEngine,
    MlxTextEngine,
    TransformersTextEngine,
    VisionLanguageEngine,
    is_mlx_checkpoint,
)
from noelle.runtime.hub_cache import cached_snapshot_path, is_downloaded
from noelle.runtime.pipelines import AsrEngine, DiffusionEngine, EmbeddingEngine, TtsEngine, VisionPipelineEngine
from noelle.runtime.preflight import (
    FIXED_RUNTIME_OVERHEAD,
    MemoryBudget,
    format_bytes,
    kv_cache_bytes_per_token,
    precision_spec,
)

PLANNING_CONTEXT_TOKENS = 2048
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".npz")
log = logging.getLogger("noelle.models")


@dataclass
class ModelSpec:
    model_id: str
    task: str = "text-generation"
    precision: str = "auto"
    device: str = "auto"
    local_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    gguf_file: str | None = None
    engine: str | None = None
    library_name: str | None = None
    gated: bool = False
    language: str | None = None
    speculative: dict[str, Any] = field(default_factory=dict)
    force: bool = False

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> ModelSpec:
        model_id = data.get("model_id") or data.get("model")
        if not isinstance(model_id, str) or not model_id.strip():
            raise BadRequest("model_id is required")
        return cls(
            model_id=model_id.strip(),
            task=str(data.get("task") or "text-generation"),
            precision=str(data.get("precision") or "auto"),
            device=str(data.get("device") or "auto"),
            local_path=data.get("local_path") or None,
            revision=data.get("revision") or None,
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            gguf_file=data.get("gguf_file") or None,
            engine=data.get("engine") or None,
            library_name=data.get("library_name") or None,
            gated=bool(data.get("gated", False)),
            language=data.get("language") or data.get("voice_language") or None,
            speculative=dict(data.get("speculative") or {}),
            force=bool(data.get("force", False)),
        )

    @property
    def family(self) -> str:
        family = tasks.task_family(self.task)
        if family is None:
            raise BadRequest(f"Unsupported task '{self.task}'.")
        return family

    def key(self, device: str) -> str:
        parts = [self.model_id, self.precision, device]
        if self.local_path:
            parts.append(hashlib.sha1(self.local_path.encode()).hexdigest()[:8])
        if self.gguf_file:
            parts.append(self.gguf_file)
        if self.family in (tasks.DETECTION, tasks.CLASSIFICATION):
            parts.append(self.task)
        return "|".join(parts)


@dataclass
class LoadedModel:
    key: str
    spec: ModelSpec
    family: str
    device: str
    engine: Any
    engine_name: str
    source: str
    memory_bytes: int
    estimated_bytes: int
    load_ms: float
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    adapters: list[str] = field(default_factory=list)
    active_adapter: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model_id": self.spec.model_id,
            "task": self.spec.task,
            "family": self.family,
            "precision": self.spec.precision,
            "device": self.device,
            "engine": self.engine_name,
            "memory_bytes": self.memory_bytes,
            "estimated_bytes": self.estimated_bytes,
            "load_ms": self.load_ms,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "adapters": list(self.adapters),
            "active_adapter": self.active_adapter,
            "context_window": getattr(self.engine, "context_window", None),
        }


class ModelManager:
    def __init__(self, device_preference: str = "auto", *, memory_override_gb: float | None = None,
                 headroom: float = 0.15, keep_loaded: bool = True,
                 on_change: Callable[[], None] | None = None) -> None:
        self.device = devices.resolve_device(device_preference)
        self.memory_override_gb = memory_override_gb
        self.headroom = headroom
        self.keep_loaded = keep_loaded
        self._on_change = on_change
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._pinned: dict[str, int] = {}

    # -- pinning: models used by the running job are never evicted mid-job ---
    @contextmanager
    def pinned(self, *keys: str) -> Iterator[None]:
        with self._lock:
            for key in keys:
                self._pinned[key] = self._pinned.get(key, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                for key in keys:
                    count = self._pinned.get(key, 0) - 1
                    if count <= 0:
                        self._pinned.pop(key, None)
                    else:
                        self._pinned[key] = count

    # -- queries -----------------------------------------------------------
    def budget(self) -> MemoryBudget:
        return devices.memory_budget(self.device, self.memory_override_gb, self.headroom)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [m.describe() for m in self._models.values()]

    def resident_bytes(self, exclude: set[str] | None = None) -> int:
        with self._lock:
            return sum(m.memory_bytes or m.estimated_bytes for k, m in self._models.items()
                       if not exclude or k not in exclude)

    def resident_sizes(self) -> dict[str, int]:
        with self._lock:
            return {k: (m.memory_bytes or m.estimated_bytes) for k, m in self._models.items()}

    def get(self, key: str) -> LoadedModel | None:
        with self._lock:
            return self._models.get(key)

    def find(self, model_id: str, family: str | None = None) -> LoadedModel | None:
        with self._lock:
            for loaded in reversed(self._models.values()):
                if loaded.spec.model_id == model_id and (family is None or loaded.family == family):
                    return loaded
        return None

    def resolve_device(self, spec: ModelSpec) -> str:
        return devices.resolve_device(spec.device) if spec.device != "auto" else self.device

    # -- sources and estimates ----------------------------------------------
    def resolve_source(self, spec: ModelSpec) -> str:
        if spec.local_path:
            path = Path(spec.local_path).expanduser()
            if not path.exists():
                raise ModelNotFound(f"Local model folder not found: {path}",
                                    hint="Download the model again from its detail page.")
            return str(path)
        snapshot = cached_snapshot_path(spec.model_id, spec.revision)
        if snapshot is None or not is_downloaded(spec.model_id, spec.revision):
            raise ModelNotFound(f"'{spec.model_id}' is not downloaded.",
                                hint="Open the model page and click Download first.")
        return str(snapshot)

    def estimate_bytes(self, spec: ModelSpec, source: str, device: str) -> int:
        root = Path(source)
        if spec.gguf_file:
            target = root / spec.gguf_file
            weights = target.stat().st_size if target.is_file() else 0
            return int(weights + FIXED_RUNTIME_OVERHEAD)
        weights = _weight_bytes(root)
        config = _read_config(root)
        stored = _stored_bytes_per_param(config)
        prec = precision_spec(spec.precision)
        if prec.quantized or is_mlx_checkpoint(root) or config.get("quantization_config"):
            if spec.precision in ("int8", "nf4") and stored:
                scaled = weights * prec.bytes_per_param / stored
            else:
                scaled = weights  # already quantized on disk
        elif spec.precision == "fp32" or (spec.precision == "auto" and device == "cpu"):
            scaled = weights * 4 / stored if stored else weights
        else:
            scaled = weights * 2 / stored if stored else weights
        kv = 0
        if spec.family in (tasks.TEXT, tasks.VISION):
            per_token = kv_cache_bytes_per_token(config)
            kv = per_token * PLANNING_CONTEXT_TOKENS if per_token else int(scaled * 0.05)
        return int(scaled + kv + FIXED_RUNTIME_OVERHEAD)

    # -- loading -------------------------------------------------------------
    def ensure(self, spec: ModelSpec, *, on_status: Callable[[str], None] | None = None) -> LoadedModel:
        device = self.resolve_device(spec)
        key = spec.key(device)
        with self._lock:
            loaded = self._models.get(key)
            if loaded is not None:
                self._models.move_to_end(key)
                loaded.last_used = time.time()
                return loaded
        source = self.resolve_source(spec)
        estimate = self.estimate_bytes(spec, source, device)
        budget = devices.memory_budget(device, self.memory_override_gb, self.headroom)
        if estimate > budget.usable_bytes and not spec.force:
            raise PreflightRejected(
                f"{spec.model_id} needs ~{format_bytes(estimate)} but only {format_bytes(budget.usable_bytes)} "
                f"is usable on {device} after the {int(self.headroom * 100)}% safety headroom.",
                hint="Quantize it to 4-bit, pick a smaller model, or use 'Load anyway'.")
        evicted = self.make_room(estimate, budget.usable_bytes, keep={key})
        if evicted:
            log.info("evicted %d model(s) to make room for %s: %s", len(evicted), spec.model_id, evicted)
        if evicted and on_status:
            on_status(f"Evicted {len(evicted)} model(s) to free memory")
        log.info("loading %s (task=%s precision=%s device=%s estimate=%s)",
                 spec.model_id, spec.task, spec.precision, device, format_bytes(estimate))
        if on_status:
            on_status(f"Loading {spec.model_id} on {device} (~{format_bytes(estimate)})")
        before = self._allocated(device)
        t0 = time.perf_counter()
        try:
            engine, engine_name = self._load_engine(spec, source, device)
        except NoelleRuntimeError:
            devices.empty_cache(device)
            raise
        except Exception as exc:
            devices.empty_cache(device)
            if devices.is_oom(exc):
                raise
            # Keep the full library message and traceback in models.log; the UI gets the
            # reason with the cache path replaced by the model id (the path is not actionable).
            log.warning("load failed for %s; full traceback follows", spec.model_id, exc_info=True)
            message = str(exc)
            if "trust_remote_code" in message:
                raise RemoteCodeBlocked(f"{spec.model_id} ships custom Python code.",
                                        hint="Review the repository, then allow remote code for this model.") from exc
            raise LoadFailed(f"Could not load {spec.model_id}: {_scrub_source(message, source, spec.model_id)}") from exc
        load_ms = (time.perf_counter() - t0) * 1000.0
        measured = max(self._allocated(device, engine_name) - before, 0)
        loaded = LoadedModel(key=key, spec=spec, family=spec.family, device=device, engine=engine,
                             engine_name=engine_name, source=source, memory_bytes=measured or estimate,
                             estimated_bytes=estimate, load_ms=load_ms)
        with self._lock:
            self._models[key] = loaded
        log.info("loaded %s in %.0f ms using %s (%s)", spec.model_id, load_ms, engine_name,
                 format_bytes(loaded.memory_bytes))
        self._changed()
        return loaded

    def make_room(self, needed: int, usable: int, keep: set[str] | None = None) -> list[str]:
        keep = keep or set()
        evicted: list[str] = []
        while True:
            with self._lock:
                candidates = [k for k in self._models if k not in keep and k not in self._pinned]
                resident = sum((m.memory_bytes or m.estimated_bytes) for m in self._models.values())
            if not candidates:
                break
            if self.keep_loaded and resident + needed <= usable:
                break
            victim = candidates[0]  # least recently used
            self.unload(victim)
            evicted.append(victim)
        return evicted

    def unload(self, key: str) -> bool:
        with self._lock:
            loaded = self._models.pop(key, None)
        if loaded is None:
            return False
        log.info("unloading %s (%s on %s)", loaded.spec.model_id, loaded.engine_name, loaded.device)
        device = loaded.device
        try:
            release = getattr(loaded.engine, "release", None)
            if release:
                release()
        except Exception:  # noqa: BLE001 - the model is already gone; a faulty release must not wedge the lane
            log.warning("Releasing %s raised; the model was unloaded anyway.", key, exc_info=True)
        finally:
            loaded.engine = None
            del loaded
            devices.empty_cache(device)
            self._changed()
        return True

    def unload_matching(self, model_id: str | None = None) -> list[str]:
        with self._lock:
            keys = [k for k, m in self._models.items() if model_id is None or m.spec.model_id == model_id]
        for key in keys:
            self.unload(key)
        return keys

    def touch(self, key: str) -> None:
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                self._models[key].last_used = time.time()

    # -- internals -------------------------------------------------------------
    def _changed(self) -> None:
        if self._on_change:
            try:
                self._on_change()
            except Exception:  # noqa: BLE001 - a telemetry/broadcast failure must not abort a load or unload
                log.warning("Model-change callback raised", exc_info=True)

    @staticmethod
    def _allocated(device: str, engine_name: str | None = None) -> int:
        if engine_name == "mlx":
            try:
                import mlx.core as mx

                return int(mx.get_active_memory())
            except Exception:
                return 0
        snap = devices.memory_snapshot(device)
        return int(snap.get("allocated") or 0)

    def _load_engine(self, spec: ModelSpec, source: str, device: str) -> tuple[Any, str]:
        self._check_runnability(spec, source)
        family = spec.family
        trc, rev = spec.trust_remote_code, spec.revision
        if family == tasks.TEXT:
            engine = spec.engine
            if engine is None:
                if spec.precision == "mlx-4bit" or is_mlx_checkpoint(source):
                    engine = "mlx"
                elif spec.gguf_file:
                    engine = "gguf"
                else:
                    engine = "transformers"
            if engine == "mlx":
                return MlxTextEngine.load(source, trust_remote_code=trc), "mlx"
            if engine in ("gguf", "llama_cpp", "llama_server"):
                return self._load_gguf(spec, source, device)
            return (TransformersTextEngine.load(source, device=device, precision=spec.precision, trust_remote_code=trc,
                                                revision=None, gguf_file=spec.gguf_file,
                                                speculative=spec.speculative), "transformers")
        if family == tasks.VISION:
            return VisionLanguageEngine.load(source, device=device, precision=spec.precision,
                                             trust_remote_code=trc), "transformers-vlm"
        if family in (tasks.DETECTION, tasks.CLASSIFICATION):
            return VisionPipelineEngine.load(source, task=spec.task, device=device,
                                             trust_remote_code=trc), "transformers-pipeline"
        if family == tasks.ASR:
            if (spec.library_name or "").lower() in ("faster-whisper", "ctranslate2"):
                from noelle.runtime.native import FasterWhisperEngine

                return FasterWhisperEngine.load(source, device=device, precision=spec.precision), "faster-whisper"
            return AsrEngine.load(source, device=device, precision=spec.precision,
                                  trust_remote_code=trc), "transformers-asr"
        if family == tasks.TTS:
            if (spec.library_name or "").lower() == "kokoro":
                from noelle.runtime.native import KokoroEngine

                return KokoroEngine.load(source, device=device, language=spec.language or "en"), "kokoro"
            return TtsEngine.load(source, device=device, precision=spec.precision,
                                  trust_remote_code=trc), "transformers-tts"
        if family == tasks.DIFFUSION:
            if not (Path(source) / "model_index.json").is_file():
                from noelle.runtime.native import SingleFileDiffusionEngine

                return SingleFileDiffusionEngine.load(source, device=device, precision=spec.precision,
                                                      trust_remote_code=trc), "diffusers-single-file"
            return DiffusionEngine.load(source, device=device, precision=spec.precision,
                                        trust_remote_code=trc), "diffusers"
        if family == tasks.EMBEDDING:
            if (spec.library_name or "").lower() == "model2vec":
                from noelle.runtime.native import Model2VecEngine

                return Model2VecEngine.load(source, device=device), "model2vec"
            return EmbeddingEngine.load(source, device=device, trust_remote_code=trc), "sentence-transformers"
        raise BadRequest(f"No engine for task family '{family}' (rev={rev})")

    @staticmethod
    def _check_runnability(spec: ModelSpec, source: str) -> None:
        """Refuse a checkpoint Noelle has no engine for, before importing any heavy library."""
        root = Path(source)
        if spec.gguf_file:
            # Ollama stores GGUF blobs without an extension; the extension is not what makes
            # a file a GGUF checkpoint, the explicit gguf_file selection is.
            files = [spec.gguf_file]
            config: dict[str, Any] = {}
            library = "gguf"  # an explicitly selected GGUF file always goes through llama.cpp
        else:
            files = runnability.file_names(root)
            config = _read_config(root)
            library = spec.library_name
        verdict = runnability.assess(
            family=spec.family,
            library=library,
            files=files,
            config=config,
            gated=spec.gated,
            facts=runnability.collect_runtime_facts(),
        )
        extra = {"model": spec.model_id, "verdict": verdict.verdict, "library": verdict.library}
        if verdict.verdict == runnability.UNSUPPORTED_LIBRARY:
            raise Unsupported(verdict.reason or f"{spec.model_id} cannot be loaded by Noelle.",
                              hint=verdict.hint, **extra)
        if verdict.verdict == runnability.NEEDS_NEWER_RUNTIME:
            raise NeedsNewerRuntime(verdict.reason or f"{spec.model_id} needs a newer transformers version.",
                                    hint=verdict.hint, **extra)
        if verdict.verdict == runnability.REMOTE_CODE and not spec.trust_remote_code:
            raise RemoteCodeBlocked(f"{spec.model_id} ships custom Python code.",
                                    hint="Review the repository, then allow remote code for this model.", **extra)
        if verdict.verdict == runnability.NEEDS_COMPONENT:
            raise DependencyMissing(verdict.reason or f"Loading {spec.model_id} needs an uninstalled component.",
                                    hint=verdict.hint, component=verdict.engine, **extra)
        if verdict.verdict == runnability.GATED_NO_ACCESS:
            raise Gated(verdict.reason or f"Access to {spec.model_id} has not been approved.",
                        hint=verdict.hint, **extra)

    @staticmethod
    def _load_gguf(spec: ModelSpec, source: str, device: str) -> tuple[Any, str]:
        """llama-cpp-python when installed, else the llama-server binary from the llama.cpp component."""
        import importlib.util

        if not spec.gguf_file:
            raise BadRequest("GGUF models need a gguf_file.")
        gguf_path = str(Path(source) / spec.gguf_file)
        if importlib.util.find_spec("llama_cpp") is not None:
            return LlamaCppTextEngine.load(gguf_path, device=device), "llama_cpp"
        from noelle.runtime.llama_tools import COMPONENT, find_llama_server

        server = find_llama_server()
        if server is None:
            raise DependencyMissing("Running GGUF models needs the llama.cpp tools, which are not installed.",
                                    hint="Noelle can download them for you after you approve it.",
                                    component=COMPONENT)
        return LlamaServerEngine.load(gguf_path, device=device, binary=str(server)), "llama_server"


def _scrub_source(message: str, source: str, model_id: str) -> str:
    """Replace the HF cache path (one unbreakable token, not actionable) with the model id."""
    if source and source in message:
        message = message.replace(source, model_id)
    return message


def _read_config(root: Path) -> dict[str, Any]:
    for name in ("config.json", "model_index.json"):
        path = root / name
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


def _stored_bytes_per_param(config: dict[str, Any]) -> float | None:
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "").lower()
    if not dtype:
        nested = config.get("text_config")
        if isinstance(nested, dict):
            dtype = str(nested.get("torch_dtype") or nested.get("dtype") or "").lower()
    if "float32" in dtype:
        return 4.0
    if dtype in ("float16", "bfloat16", "torch.float16", "torch.bfloat16"):
        return 2.0
    return None


def _weight_bytes(root: Path) -> int:
    """Bytes of the weight files transformers/diffusers would actually read."""
    if root.is_file():
        return root.stat().st_size
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in _WEIGHT_SUFFIXES]
    safetensors = [p for p in files if p.suffix == ".safetensors"]
    chosen = safetensors or [p for p in files if p.suffix in (".bin", ".pt", ".pth", ".npz")] or files
    fp16_variants = [p for p in chosen if ".fp16." in p.name]
    if fp16_variants:
        twins = {p.name.replace(".fp16.", ".") for p in fp16_variants}
        chosen = fp16_variants + [p for p in chosen if ".fp16." not in p.name and p.name not in twins
                                  and not any(p.parent == f.parent for f in fp16_variants)]
    return sum(p.stat().st_size for p in chosen)
