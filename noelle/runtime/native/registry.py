"""Native engine registry: what is installed, what it can do, and why it is unavailable."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeEngineStatus:
    engine_id: str
    name: str
    capabilities: tuple[str, ...]
    module: str
    component: str
    installed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"id": self.engine_id, "name": self.name, "capabilities": list(self.capabilities),
                "module": self.module, "component": self.component, "installed": self.installed,
                "reason": self.reason}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def detect_native_engines() -> list[NativeEngineStatus]:
    definitions = (
        ("kokoro", "Kokoro TTS", ("tts",), "kokoro", "kokoro"),
        ("faster-whisper", "faster-whisper ASR", ("asr",), "faster_whisper", "faster-whisper"),
        ("model2vec", "Model2Vec embeddings", ("embedding",), "model2vec", "model2vec"),
        ("diffusion-single-file", "Single-file diffusion", ("diffusion",), "diffusers", "diffusion"),
    )
    out = []
    for engine_id, name, capabilities, module, component in definitions:
        installed = _installed(module)
        out.append(NativeEngineStatus(engine_id, name, capabilities, module, component, installed,
                                      None if installed else f"Python package '{module}' is not installed."))
    return out


def native_backends() -> dict[str, bool]:
    return {status.engine_id: status.installed for status in detect_native_engines()}