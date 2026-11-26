"""Polymorphic adapter dispatch: one mapping from a model's metadata to its engine.

Kept pure (engine classes are imported lazily) so the desktop shell, tests and the
worker can all resolve the same adapter without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from noelle.runtime import tasks
from noelle.runtime.adapters.base_adapter import UnsupportedArchitectureError


@dataclass(frozen=True)
class AdapterSpec:
    family: str
    engine: str
    module: str
    class_name: str
    description: str = ""

    def adapter_class(self) -> type:
        import importlib

        module = importlib.import_module(self.module)
        return getattr(module, self.class_name)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "engine": self.engine, "module": self.module,
                "class_name": self.class_name, "description": self.description}


_SPECS: dict[tuple[str, str | None, str | None], AdapterSpec] = {}


def _register(family: str, engine: str, module: str, class_name: str, description: str = "",
              *, model_type: str | None = None) -> None:
    _SPECS[(family, engine, model_type)] = AdapterSpec(family, engine, module, class_name, description)


_register(tasks.TEXT, "transformers", "noelle.runtime.generation", "TransformersTextEngine",
          "AutoModelForCausalLM / AutoModelForSeq2SeqLM")
_register(tasks.TEXT, "mlx", "noelle.runtime.generation", "MlxTextEngine", "Apple MLX 4-bit")
_register(tasks.TEXT, "llama.cpp", "noelle.runtime.generation", "LlamaCppTextEngine", "GGUF via llama-cpp-python")
_register(tasks.TEXT, "llama_server", "noelle.runtime.generation", "LlamaServerEngine",
          "GGUF via the llama-server component")
_register(tasks.VISION, "transformers", "noelle.runtime.generation", "VisionLanguageEngine",
          "AutoModelForImageTextToText / VisionEncoderDecoderModel")
_register(tasks.DETECTION, "transformers", "noelle.runtime.pipelines", "VisionPipelineEngine", "transformers pipeline")
_register(tasks.CLASSIFICATION, "transformers", "noelle.runtime.pipelines", "VisionPipelineEngine",
          "transformers pipeline")
_register(tasks.ASR, "transformers", "noelle.runtime.pipelines", "AsrEngine", "automatic-speech-recognition")
_register(tasks.TTS, "transformers", "noelle.runtime.pipelines", "TtsEngine", "text-to-audio")
_register(tasks.DIFFUSION, "diffusers", "noelle.runtime.pipelines", "DiffusionEngine", "DiffusionPipeline")
_register(tasks.EMBEDDING, "sentence-transformers", "noelle.runtime.pipelines", "EmbeddingEngine",
          "SentenceTransformer")
_register(tasks.EMBEDDING, "transformers", "noelle.runtime.pipelines", "EmbeddingEngine", "plain encoder")


def get_adapter_spec(*, family: str | None, library: str | None = None, engine: str | None = None,
                     model_type: str | None = None, files: Any = (), target_tool: str | None = None) -> AdapterSpec:
    """Resolve the engine class for a family, honouring GGUF/MLX layouts and explicit engines."""
    if family is None:
        raise UnsupportedArchitectureError("Unknown task family: no adapter exists.")
    if target_tool == "timm":
        raise UnsupportedArchitectureError(
            "This repository needs timm (a PyTorch image model library Noelle has no adapter for yet).")
    names = [str(item).replace("\\", "/") for item in (files or ())]
    lib = (library or "").strip().lower() or None

    if family == tasks.TEXT:
        if engine in ("llama_server", "llama_cpp"):
            return _SPECS[(tasks.TEXT, "llama_server" if engine == "llama_server" else "llama.cpp", None)]
        if engine == "mlx" or lib == "mlx" or any(name.lower().endswith(".npz") for name in names):
            return _SPECS[(tasks.TEXT, "mlx", None)]
        if engine in ("gguf", "llama.cpp") or lib == "gguf" or any(name.lower().endswith(".gguf") for name in names):
            return _SPECS[(tasks.TEXT, "llama.cpp", None)]
        return _SPECS[(tasks.TEXT, "transformers", None)]

    chosen = engine or ("diffusers" if family == tasks.DIFFUSION else
                        "sentence-transformers" if family == tasks.EMBEDDING else "transformers")
    for key in ((family, chosen, model_type), (family, chosen, None)):
        if key in _SPECS:
            return _SPECS[key]
    raise UnsupportedArchitectureError(
        f"No adapter for family '{family}' with engine '{chosen}' (library={lib!r}, model_type={model_type!r}).")


def get_adapter_for_model(report: Any) -> AdapterSpec:
    """Factory entry point for an :class:`~noelle.runtime.agent.ModelInspectionReport`."""
    return get_adapter_spec(
        family=getattr(report, "family", None),
        library=getattr(report, "library_name", None),
        model_type=getattr(report, "model_type", None),
        files=getattr(report, "architectures", ()),
        target_tool=getattr(report, "target_tool", None),
    )