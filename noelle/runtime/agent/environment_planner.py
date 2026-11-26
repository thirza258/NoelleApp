"""Turn a runnability verdict into an isolated-environment plan.

The Agent Runtime never mutates the application's Python. ``NEEDS_COMPONENT`` maps to
the profile that installs the missing engine; ``NEEDS_NEWER_RUNTIME`` maps to a fresh
base environment with the latest transformers. Unsupported libraries stay unsupported:
no environment can add an engine Noelle does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from noelle.runtime import runnability

_ENGINE_PROFILES = {
    "transformers": "base",
    "diffusers": "diffusion",
    "sentence-transformers": "rag",
    "llama.cpp": "gguf",
    "mlx": "mlx",
    "peft": "peft",
    "kokoro": "kokoro",
    "faster-whisper": "faster-whisper",
    "model2vec": "model2vec",
    "diffusion-single-file": "diffusion",
}

_ENGINE_PACKAGES = {
    "transformers": ("transformers",),
    "diffusers": ("diffusers",),
    "sentence-transformers": ("sentence-transformers",),
    "llama.cpp": ("llama-cpp-python", "gguf"),
    "mlx": ("mlx", "mlx-lm"),
    "peft": ("peft",),
    "kokoro": ("kokoro", "misaki"),
    "faster-whisper": ("faster-whisper",),
    "model2vec": ("model2vec",),
    "diffusion-single-file": ("diffusers",),
}


@dataclass(frozen=True)
class EnvironmentPlan:
    action: str  # create_environment | update_runtime | none
    engine: str | None
    profiles: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    reason: str = ""
    #: True when an existing environment cannot be repaired in place and a fresh one is needed.
    recreate: bool = False

    @property
    def actionable(self) -> bool:
        return self.action in ("create_environment", "update_runtime") and bool(self.profiles)

    def suggested_name(self, existing: Iterable[str] = ()) -> str:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.engine or "runtime").strip("-") or "runtime"
        name = f"auto-{base}"[:64]
        taken = set(existing)
        if name not in taken:
            return name
        index = 2
        while f"{name[:58]}-{index}" in taken:
            index += 1
        return f"{name[:58]}-{index}"

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "engine": self.engine, "profiles": list(self.profiles),
                "packages": list(self.packages), "reason": self.reason, "recreate": self.recreate,
                "actionable": self.actionable}


def plan_for_verdict(verdict: runnability.RunnabilityResult) -> EnvironmentPlan | None:
    if verdict.verdict == runnability.NEEDS_COMPONENT:
        engine = verdict.engine or "transformers"
        profile = _ENGINE_PROFILES.get(engine, "base")
        return EnvironmentPlan("create_environment", engine, (profile,), _ENGINE_PACKAGES.get(engine, ()),
                               reason=verdict.reason or f"{engine} is missing")
    if verdict.verdict == runnability.NEEDS_NEWER_RUNTIME:
        return EnvironmentPlan(
            "update_runtime", "transformers", ("base",), ("transformers",),
            reason=(verdict.reason or "The model needs a newer transformers") +
                   " — a fresh environment installs the latest release",
            recreate=True,
        )
    return None


def plan_for_payload(payload: dict[str, Any]) -> EnvironmentPlan | None:
    """Plan from a worker error payload (``NEEDS_NEWER_RUNTIME`` / ``DEPENDENCY_MISSING``)."""
    code = str(payload.get("code") or "")
    if code == "NEEDS_NEWER_RUNTIME":
        verdict = runnability.RunnabilityResult(runnability.NEEDS_NEWER_RUNTIME, engine="transformers",
                                                model_type=payload.get("model_type"))
        return plan_for_verdict(verdict)
    if code == "DEPENDENCY_MISSING":
        component = payload.get("component") or payload.get("engine") or "transformers"
        engine = "llama.cpp" if component in ("llama.cpp", "llama-cpp", "llama_cpp") else str(component)
        verdict = runnability.RunnabilityResult(runnability.NEEDS_COMPONENT, engine=engine)
        return plan_for_verdict(verdict)
    return None