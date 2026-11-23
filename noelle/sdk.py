"""Tiny Python API for notebooks and scripts: ``noelle.load_model(...)``.

Runs models in the *current* process with the same engines the worker uses::

    import noelle.sdk as noelle
    model = noelle.load_model("HuggingFaceTB/SmolLM2-135M-Instruct")
    model.chat("Explain KV caching in one paragraph.")

The desktop shell never imports this module (it imports torch); the Notebook tab
runs it inside a separate Jupyter kernel process.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from noelle.runtime.protocol import GenerationParams

_manager: Any = None


def _get_manager(device: str = "auto") -> Any:
    global _manager
    if _manager is None:
        from noelle.runtime.model_manager import ModelManager

        _manager = ModelManager(device)
    return _manager


def _registry_record(model_id: str) -> Any:
    try:
        from noelle.app.services.model_registry import get_registry

        return get_registry().get_model(model_id)
    except Exception:
        return None


class NoelleModel:
    def __init__(self, loaded: Any) -> None:
        self._loaded = loaded

    @property
    def engine(self) -> Any:
        return self._loaded.engine

    @property
    def model(self) -> Any:
        return getattr(self.engine, "model", None)

    @property
    def tokenizer(self) -> Any:
        return getattr(self.engine, "tokenizer", None)

    @property
    def info(self) -> dict[str, Any]:
        return self._loaded.describe()

    def chat(self, prompt: str | list[dict[str, Any]], *, system_prompt: str | None = None, stream: bool = True,
             **params: Any) -> str:
        """Generate a reply; streams to stdout by default and returns the full text."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        on_text = (lambda t: (sys.stdout.write(t), sys.stdout.flush())) if stream else (lambda _t: None)
        result = self.engine.stream_chat(messages, GenerationParams.from_dict(params), stop_event=threading.Event(),
                                         on_text=on_text, system_prompt=system_prompt)
        if stream:
            sys.stdout.write("\n")
        self.last_metrics = result.metrics()
        return result.text

    __call__ = chat

    def embed(self, texts: list[str] | str) -> Any:
        return self.engine.encode([texts] if isinstance(texts, str) else texts)

    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        return self.engine.transcribe(audio_path, **kwargs)

    def speak(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return self.engine.synthesize(text, **kwargs)

    def generate_image(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return self.engine.generate(prompt=prompt, **kwargs)

    def unload(self) -> None:
        _get_manager().unload(self._loaded.key)

    def __repr__(self) -> str:
        info = self.info
        return f"<NoelleModel {info['model_id']} {info['precision']} on {info['device']} ({info['engine']})>"


def load_model(model_id: str, *, task: str | None = None, precision: str = "auto", device: str = "auto",
               local_path: str | None = None, trust_remote_code: bool = False) -> NoelleModel:
    from noelle.runtime.model_manager import ModelSpec

    record = _registry_record(model_id)
    if record is not None:
        task = task or record.pipeline_tag
        local_path = local_path or record.local_path
        if precision == "auto" and record.precision:
            precision = record.precision
    spec = ModelSpec(model_id=model_id, task=task or "text-generation", precision=precision, device=device,
                     local_path=local_path, trust_remote_code=trust_remote_code,
                     gguf_file=getattr(record, "gguf_file", None), engine=getattr(record, "engine", None))
    return NoelleModel(_get_manager(device).ensure(spec))


def loaded_models() -> list[dict[str, Any]]:
    return _get_manager().snapshot() if _manager is not None else []
