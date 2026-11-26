"""One runnability verdict per model, shared by the desktop shell and the worker.

Pure Python (no torch, no Qt) so both processes import exactly the same rules.
The shell has no ``transformers``: the worker publishes the model types its
bundled libraries understand (``RUNTIME_CAPS`` in the ``HELLO`` result), and the
shell evaluates the same :func:`assess` rules with those facts.

Verdicts:

``RUNNABLE``             an engine exists and the repository layout matches
``NEEDS_COMPONENT``      the engine exists but its package/tool is not installed
``NEEDS_NEWER_RUNTIME``  ``model_type`` is unknown to the bundled transformers
``REMOTE_CODE``          ``auto_map`` is present (existing consent flow applies)
``UNSUPPORTED_LIBRARY``  library/layout belongs to a library Noelle has no engine for
``GATED_NO_ACCESS``      gated and the account has not been approved
``UNKNOWN``              not enough metadata (list view without config)
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from noelle.runtime import tasks

RUNNABLE = "RUNNABLE"
NEEDS_COMPONENT = "NEEDS_COMPONENT"
NEEDS_NEWER_RUNTIME = "NEEDS_NEWER_RUNTIME"
REMOTE_CODE = "REMOTE_CODE"
UNSUPPORTED_LIBRARY = "UNSUPPORTED_LIBRARY"
GATED_NO_ACCESS = "GATED_NO_ACCESS"
UNKNOWN = "UNKNOWN"

SUPPORTED_VERDICTS = frozenset({RUNNABLE, NEEDS_COMPONENT, REMOTE_CODE, UNKNOWN})
BLOCKED_VERDICTS = frozenset({UNSUPPORTED_LIBRARY, NEEDS_NEWER_RUNTIME, GATED_NO_ACCESS})

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".nemo", ".onnx", ".npz", ".gguf",
                    ".h5", ".msgpack", ".mlmodel")
_GGUF_SUFFIX = ".gguf"
_MLX_SUFFIX = ".npz"

# ``library_name`` values that map to a real Noelle engine, per family.
_LIBRARY_ENGINES: dict[str, dict[str, str]] = {
    tasks.TEXT: {"transformers": "transformers", "gguf": "llama.cpp", "mlx": "mlx", "peft": "peft"},
    tasks.VISION: {"transformers": "transformers"},
    tasks.DETECTION: {"transformers": "transformers"},
    tasks.CLASSIFICATION: {"transformers": "transformers"},
    tasks.ASR: {"transformers": "transformers", "faster-whisper": "faster-whisper", "ctranslate2": "faster-whisper"},
    tasks.TTS: {"transformers": "transformers", "kokoro": "kokoro"},
    tasks.DIFFUSION: {"diffusers": "diffusers"},
    tasks.EMBEDDING: {"sentence-transformers": "sentence-transformers", "transformers": "transformers",
                      "model2vec": "model2vec", "peft": "peft"},
}

# engine -> backend key reported by ``capabilities.detect_backends``
_ENGINE_BACKENDS = {
    "transformers": "transformers",
    "diffusers": "diffusers",
    "sentence-transformers": "sentence-transformers",
    "llama.cpp": "llama.cpp",
    "mlx": "mlx",
    "peft": "transformers",
    "kokoro": "kokoro",
    "faster-whisper": "faster-whisper",
    "model2vec": "model2vec",
    "diffusion-single-file": "diffusion-single-file",
}

_MODEL_TYPE_FAMILIES = (tasks.TEXT, tasks.VISION, tasks.DETECTION, tasks.CLASSIFICATION, tasks.ASR, tasks.TTS,
                        tasks.EMBEDDING)
_LAYOUT_FAMILIES = (tasks.TEXT, tasks.VISION, tasks.DETECTION, tasks.CLASSIFICATION, tasks.ASR, tasks.TTS)


@dataclass(frozen=True)
class RuntimeFacts:
    """What the worker's bundled libraries can load. Empty when the libraries are absent."""

    versions: dict[str, str] = field(default_factory=dict)
    model_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    all_model_types: tuple[str, ...] = ()
    backends: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, data: Any) -> RuntimeFacts | None:
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            return None
        versions = {str(k): str(v) for k, v in (data.get("versions") or {}).items() if v}
        model_types = {str(k): tuple(str(t) for t in (v or ())) for k, v in (data.get("model_types") or {}).items()
                       if isinstance(v, (list, tuple))}
        all_types = tuple(str(t) for t in (data.get("all_model_types") or ()) if isinstance(t, str))
        backends = {str(k): bool(v) for k, v in (data.get("backends") or {}).items()}
        return cls(versions, model_types, all_types, backends)

    def to_payload(self) -> dict[str, Any]:
        return {"versions": dict(self.versions),
                "model_types": {key: list(value) for key, value in self.model_types.items()},
                "all_model_types": list(self.all_model_types),
                "backends": dict(self.backends)}


@dataclass(frozen=True)
class RunnabilityResult:
    verdict: str
    family: str | None = None
    engine: str | None = None
    library: str | None = None
    model_type: str | None = None
    reason: str = ""
    hint: str | None = None
    gated: bool = False

    @property
    def supported(self) -> bool:
        return self.verdict in SUPPORTED_VERDICTS

    @property
    def remediation(self) -> dict[str, Any] | None:
        """Machine-readable fix for the Agent Runtime (environment planning / consent / browser)."""
        if self.verdict == NEEDS_COMPONENT:
            return {"action": "install_component", "engine": self.engine}
        if self.verdict == NEEDS_NEWER_RUNTIME:
            return {"action": "update_runtime", "package": "transformers",
                    "reason": f"model_type '{self.model_type}' is newer than the bundled transformers"}
        if self.verdict == REMOTE_CODE:
            return {"action": "allow_remote_code"}
        if self.verdict == GATED_NO_ACCESS:
            return {"action": "open_model_page"}
        return None

    @property
    def label(self) -> str:
        return {
            RUNNABLE: "Runs in Noelle",
            NEEDS_COMPONENT: "Needs a component",
            NEEDS_NEWER_RUNTIME: "Needs newer runtime",
            REMOTE_CODE: "Needs remote-code consent",
            UNSUPPORTED_LIBRARY: "Not supported in Noelle",
            GATED_NO_ACCESS: "Gated — access not approved",
            UNKNOWN: "Runnability unknown",
        }.get(self.verdict, self.verdict)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "family": self.family, "engine": self.engine, "library": self.library,
                "model_type": self.model_type, "reason": self.reason, "hint": self.hint,
                "gated": self.gated, "supported": self.supported, "label": self.label,
                "remediation": self.remediation}


def _result(verdict: str, family: str | None, *, engine: str | None = None, library: str | None = None,
            model_type: str | None = None, reason: str = "", hint: str | None = None,
            gated: bool = False) -> RunnabilityResult:
    return RunnabilityResult(verdict, family, engine, library, model_type, reason, hint, gated)


def supported(verdict: str) -> bool:
    return verdict in SUPPORTED_VERDICTS


def _normalise_names(files: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for item in files or ():
        name = str(item).replace("\\", "/").strip("/")
        if name:
            names.append(name)
    return names


def _has_config(names: list[str], config: Mapping[str, Any]) -> bool:
    return "config.json" in names or bool(config)


def _has_weights(names: list[str]) -> bool:
    return any(name.lower().endswith(_WEIGHT_SUFFIXES) for name in names)


def _has_gguf(names: list[str]) -> bool:
    return any(name.lower().endswith(_GGUF_SUFFIX) for name in names)


def _has_mlx(names: list[str], library: str | None, config: Mapping[str, Any]) -> bool:
    if library == "mlx":
        return True
    if any(name.lower().endswith(_MLX_SUFFIX) for name in names):
        return True
    return _mlx_quantized(config)


def _mlx_quantized(config: Mapping[str, Any]) -> bool:
    quant = config.get("quantization")
    return isinstance(quant, Mapping) and "bits" in quant and "group_size" in quant


def _backend_available(engine: str | None, facts: RuntimeFacts | None, backends: Mapping[str, bool] | None) -> bool | None:
    """True/False when a backend report is known, None when nobody reported it."""
    if engine is None:
        return None
    key = _ENGINE_BACKENDS.get(engine)
    if not key:
        return True
    available: Mapping[str, bool] | None = None
    if facts is not None and facts.backends:
        available = facts.backends
    elif backends is not None:
        available = backends
    return None if available is None else bool(available.get(key))


def _component(engine: str, family: str | None, facts: RuntimeFacts | None, backends: Mapping[str, bool] | None,
               *, library: str | None, model_type: str | None, gated: bool) -> RunnabilityResult:
    available = _backend_available(engine, facts, backends)
    if engine == "llama.cpp":
        detail = "the llama.cpp tools"
    elif engine == "mlx":
        detail = "the MLX runtime"
    elif engine == "diffusion-single-file":
        detail = "the diffusers package"
    else:
        detail = f"the '{engine}' package"
    if available is False:
        return _result(NEEDS_COMPONENT, family, engine=engine, library=library, model_type=model_type,
                       reason=f"Loading this model needs {detail}, which is not installed in the runtime.",
                       hint="Install the component from Noelle's component dialog, then load the model again.",
                       gated=gated)
    return _result(RUNNABLE, family, engine=engine, library=library, model_type=model_type, gated=gated)


def assess(*, family: str | None, library: str | None = None, files: Iterable[Any] = (), config: Any = None,
           gated: bool = False, has_access: bool | None = None, facts: RuntimeFacts | None = None,
           backends: Mapping[str, bool] | None = None) -> RunnabilityResult:
    """Return one verdict for a model. ``facts`` comes from the worker (see module docstring)."""
    names = _normalise_names(files)
    cfg: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    lib = str(library).strip().lower() or None if library else None
    model_type = cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else None
    auto_map = cfg.get("auto_map")
    remote_code = bool(auto_map)
    gated = bool(gated)

    if gated and has_access is False:
        return _result(GATED_NO_ACCESS, family, library=lib, model_type=model_type,
                       reason="This gated model's licence has not been accepted for your account.",
                       hint="Open the model page on huggingface.co, accept the licence, then try again.",
                       gated=True)

    if family is None:
        return _result(UNSUPPORTED_LIBRARY, family, library=lib, model_type=model_type, gated=gated,
                       reason="Noelle has no playground or engine for this task family.")

    allowed = _LIBRARY_ENGINES.get(family, {})
    engine = allowed.get(lib or "")
    if lib and engine is None:
        return _result(UNSUPPORTED_LIBRARY, family, library=lib, model_type=model_type, gated=gated,
                       reason=f"This model is published for the '{lib}' library, which Noelle has no engine for.",
                       hint="You can still download it, but Noelle cannot run it yet.")

    has_config = _has_config(names, cfg)
    has_gguf = _has_gguf(names)
    has_mlx = _has_mlx(names, lib, cfg)
    has_index = "model_index.json" in names
    has_modules = "modules.json" in names

    # Explicit formats win over the family's generic engine.
    if lib == "gguf" or (family == tasks.TEXT and not has_config and has_gguf):
        return _component("llama.cpp", family, facts, backends, library=lib, model_type=model_type, gated=gated)
    if lib == "mlx" or (family == tasks.TEXT and not has_config and has_mlx):
        return _component("mlx", family, facts, backends, library=lib, model_type=model_type, gated=gated)
    if lib == "peft":
        return _component("peft", family, facts, backends, library=lib, model_type=model_type, gated=gated)
    if engine in ("kokoro", "faster-whisper", "model2vec"):
        if engine == "faster-whisper" and names and not any(
                name.lower().endswith((".bin", ".ct2", ".safetensors")) for name in names):
            return _result(UNSUPPORTED_LIBRARY, family, engine=engine, library=lib, model_type=model_type,
                           gated=gated, reason="No CTranslate2/faster-whisper model file was found.")
        return _component(engine, family, facts, backends, library=lib, model_type=model_type, gated=gated)

    if family == tasks.DIFFUSION:
        if not has_index:
            if not names:
                return _result(UNKNOWN, family, library=lib, model_type=model_type, gated=gated,
                               reason="The repository's file layout is not known yet.")
            if any(name.lower().endswith((".safetensors", ".ckpt")) for name in names):
                return _component("diffusion-single-file", family, facts, backends, library=lib,
                                  model_type=model_type, gated=gated)
            return _result(UNSUPPORTED_LIBRARY, family, library=lib, model_type=model_type, gated=gated,
                           reason="This is not a diffusers pipeline: no model_index.json or checkpoint file.")
        return _component("diffusers", family, facts, backends, library=lib, model_type=model_type, gated=gated)

    if family == tasks.EMBEDDING:
        if has_modules:
            return _component("sentence-transformers", family, facts, backends, library=lib,
                              model_type=model_type, gated=gated)
        if not has_config:
            if not names:
                return _result(UNKNOWN, family, library=lib, model_type=model_type, gated=gated,
                               reason="The repository's file layout is not known yet.")
            return _result(UNSUPPORTED_LIBRARY, family, library=lib, model_type=model_type, gated=gated,
                           reason="This is not a sentence-transformers repository (no modules.json/config.json).",
                           hint="ONNX-only, Model2Vec and multi-vector embedding layouts are not supported yet.")
        engine = engine or "sentence-transformers"

    elif family in _LAYOUT_FAMILIES:
        if not has_config:
            if not names:
                return _result(UNKNOWN, family, library=lib, model_type=model_type, gated=gated,
                               reason="The repository's file layout is not known yet.")
            return _result(UNSUPPORTED_LIBRARY, family, library=lib, model_type=model_type, gated=gated,
                           reason="This repository is not a transformers checkpoint (no config.json).",
                           hint="PyTorch/transformers checkpoints, GGUF and MLX repositories are supported.")
        engine = engine or "transformers"

    if engine not in ("transformers", "sentence-transformers"):
        return _result(UNSUPPORTED_LIBRARY, family, engine=engine, library=lib, model_type=model_type, gated=gated,
                       reason="Noelle has no engine for this repository layout.")

    if remote_code:
        return _result(REMOTE_CODE, family, engine=engine, library=lib, model_type=model_type, gated=gated,
                       reason="This repository ships custom Python code (auto_map).",
                       hint="Review the repository, then allow remote code for this model.")

    if not model_type:
        # A quantization_config without a model_type is still a transformers checkpoint
        # (the test fixtures and some pre-quantized repos omit model_type).
        if isinstance(cfg.get("quantization_config"), Mapping):
            return _component("transformers", family, facts, backends, library=lib, model_type=None,
                              gated=gated)
        return _result(UNSUPPORTED_LIBRARY, family, engine=engine, library=lib, gated=gated,
                       reason="The config has no model_type, so this is not a standard transformers checkpoint.",
                       hint="Repositories that need their own package (Kokoro, OmniVoice, Stable Audio) are not "
                            "supported by this runtime yet.")

    family_types = set(facts.model_types.get(family, ())) if facts is not None else set()
    all_types = set(facts.all_model_types) if facts is not None else set()
    if not family_types and facts is not None:
        # The worker reported its libraries but not this loader: the package is missing.
        return _component(engine, family, facts, backends, library=lib, model_type=model_type, gated=gated)

    if facts is None or (not family_types and not all_types):
        return _result(UNKNOWN, family, engine=engine, library=lib, model_type=model_type, gated=gated,
                       reason="The runtime has not reported which model types it supports yet.")

    if model_type in family_types:
        return _component(engine, family, facts, backends, library=lib, model_type=model_type, gated=gated)

    if model_type in all_types:
        return _result(UNSUPPORTED_LIBRARY, family, engine=engine, library=lib, model_type=model_type, gated=gated,
                       reason=f"'{model_type}' is a transformers model type, but not one this task's loader "
                              "can run.",
                       hint="The repository's pipeline tag may not match its actual architecture.")

    if lib in ("transformers", "sentence-transformers"):
        version = facts.versions.get("transformers", "bundled")
        return _result(NEEDS_NEWER_RUNTIME, family, engine=engine, library=lib, model_type=model_type, gated=gated,
                       reason=f"The model type '{model_type}' is not supported by the bundled transformers "
                              f"({version}).",
                       hint="A newer version of Noelle may support it.")
    return _result(UNSUPPORTED_LIBRARY, family, engine=engine, library=lib, model_type=model_type, gated=gated,
                   reason=f"'{model_type}' is not a transformers model type; this repository needs its own "
                          "library.",
                   hint="A future transformers release might add it, but Noelle cannot run it today.")


# ---------------------------------------------------------------------------
# convenience wrappers for Hub summaries and local snapshots
# ---------------------------------------------------------------------------

def _summary_names(model: Any) -> list[str]:
    names: list[str] = []
    for item in getattr(model, "siblings", None) or ():
        name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
        if name:
            names.append(str(name))
    return names


def assess_model(model: Any, *, config: Any = None, facts: RuntimeFacts | None = None,
                 has_access: bool | None = None, backends: Mapping[str, bool] | None = None) -> RunnabilityResult:
    """Assess a Hub summary / registry record / capability input object."""
    cfg = config or getattr(model, "config", None) or {}
    gated_value = getattr(model, "gated", False)
    gated = bool(gated_value) and gated_value not in ("false", "False")
    return assess(family=tasks.task_family(getattr(model, "pipeline_tag", None)),
                  library=getattr(model, "library_name", None), files=_summary_names(model), config=cfg,
                  gated=gated, has_access=has_access, facts=facts, backends=backends)


def file_names(source: str | Path, *, limit: int = 50_000) -> list[str]:
    """Relative file names in a model folder (or the file's own name)."""
    root = Path(source).expanduser()
    if root.is_file():
        return [root.name]
    if not root.is_dir():
        return []
    names: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in (".git", ".cache")]
        relative = Path(dirpath).relative_to(root)
        for filename in filenames:
            names.append(filename if str(relative) == "." else str(relative / filename))
            if len(names) >= limit:
                return names
    return names


def read_config(source: str | Path) -> dict[str, Any]:
    path = Path(source).expanduser()
    if path.is_file():
        path = path.parent
    try:
        data = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def assess_local(source: str | Path, *, family: str | None, library: str | None = None,
                 facts: RuntimeFacts | None = None, gated: bool = False,
                 has_access: bool | None = None) -> RunnabilityResult:
    return assess(family=family, library=library, files=file_names(source), config=read_config(source),
                  gated=gated, has_access=has_access, facts=facts)


# ---------------------------------------------------------------------------
# worker-side facts
# ---------------------------------------------------------------------------

def _mapping_keys(module: Any, *names: str) -> set[str]:
    keys: set[str] = set()
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, Mapping):
            keys.update(str(key) for key in value)
    return keys


@lru_cache(maxsize=1)
def collect_runtime_facts() -> RuntimeFacts:
    """Detect the worker's libraries and the model types they can load (cached)."""
    versions: dict[str, str] = {}
    for distribution, key in (("transformers", "transformers"), ("diffusers", "diffusers"),
                              ("sentence-transformers", "sentence-transformers"), ("torch", "torch"),
                              ("huggingface_hub", "huggingface_hub")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - broken metadata must not break runnability
            continue

    model_types: dict[str, tuple[str, ...]] = {}
    all_types: set[str] = set()
    try:
        from transformers.models.auto import modeling_auto as auto

        text = _mapping_keys(auto, "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES")
        vision = _mapping_keys(auto, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES") | {"vision-encoder-decoder"}
        detection = _mapping_keys(auto, "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES")
        classification = _mapping_keys(auto, "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES")
        asr = _mapping_keys(auto, "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "MODEL_FOR_CTC_MAPPING_NAMES")
        tts = _mapping_keys(auto, "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES",
                            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES")
        embedding = _mapping_keys(auto, "MODEL_MAPPING_NAMES")
        model_types = {tasks.TEXT: tuple(sorted(text)), tasks.VISION: tuple(sorted(vision)),
                       tasks.DETECTION: tuple(sorted(detection)),
                       tasks.CLASSIFICATION: tuple(sorted(classification)), tasks.ASR: tuple(sorted(asr)),
                       tasks.TTS: tuple(sorted(tts)), tasks.EMBEDDING: tuple(sorted(embedding))}
        all_types = set().union(*model_types.values())
        # Keep the full configuration registry too: a type can be loadable by an auto class that
        # lives outside the task mappings above.
        from transformers.models.auto import configuration_auto
        all_types |= _mapping_keys(configuration_auto, "CONFIG_MAPPING", "CONFIG_MAPPING_NAMES")
        model_types[tasks.EMBEDDING] = tuple(sorted(set(model_types[tasks.EMBEDDING]) | all_types))
    except Exception:  # noqa: BLE001 - transformers may be missing in the shell/CI
        model_types = {}
        all_types = set()

    backends: dict[str, bool] = {}
    try:
        from noelle.runtime.capabilities import detect_backends

        backends = detect_backends()
    except Exception:  # noqa: BLE001
        backends = {}

    return RuntimeFacts(versions=versions, model_types=model_types, all_model_types=tuple(sorted(all_types)),
                        backends=backends)


def clear_runtime_facts_cache() -> None:
    collect_runtime_facts.cache_clear()