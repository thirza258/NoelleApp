"""Find GGUF models already installed by Ollama or LM Studio.

Read-only: the other application's files are never moved, rewritten or deleted. Imported
records reference the blob/GGUF in place and carry the source in ``metadata``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from noelle.app.services.model_registry import ModelRecord, ModelRegistry

OLLAMA_MANIFEST = "application/vnd.ollama.image.model"
_MAX_MANIFEST_BYTES = 4_000_000


@dataclass(frozen=True)
class ExternalModel:
    source: str  # ollama | lmstudio
    name: str
    path: str  # GGUF file or directory to load
    gguf_file: str | None
    size_bytes: int | None = None
    quant: str | None = None

    @property
    def repo_id(self) -> str:
        if self.source == "ollama":
            return "ollama--" + self.name.replace(":", "--").replace("/", "--")
        return "lmstudio--" + self.name.replace("/", "--")

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "name": self.name, "path": self.path, "gguf_file": self.gguf_file,
                "size_bytes": self.size_bytes, "quant": self.quant, "repo_id": self.repo_id}


def ollama_root(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    if os.environ.get("OLLAMA_MODELS"):
        return Path(os.environ["OLLAMA_MODELS"]).expanduser()
    return Path.home() / ".ollama" / "models"


def lmstudio_roots(override: str | Path | None = None) -> list[Path]:
    if override:
        return [Path(override).expanduser()]
    roots = [Path.home() / ".lmstudio" / "models", Path.home() / ".cache" / "lm-studio" / "models"]
    if sys.platform == "win32":
        roots.insert(0, Path.home() / ".lmstudio" / "models")
    return roots


def _quant_from_name(name: str) -> str | None:
    import re

    match = re.search(r"(?:^|[-._])((?:i?q|tq)\d+(?:_[a-z0-9]+)*|f16|bf16|f32)$", Path(name).stem, re.IGNORECASE)
    return match.group(1).upper() if match else None


def scan_ollama(root: str | Path | None = None) -> list[ExternalModel]:
    base = ollama_root(root)
    manifests = base / "manifests"
    blobs = base / "blobs"
    found: list[ExternalModel] = []
    if not manifests.is_dir():
        return found
    for path in sorted(manifests.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > _MAX_MANIFEST_BYTES:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        layers = data.get("layers") if isinstance(data, dict) else None
        if not isinstance(layers, list):
            continue
        digest = next((str(layer.get("digest") or "") for layer in layers
                       if isinstance(layer, dict) and layer.get("mediaType") == OLLAMA_MANIFEST), "")
        if not digest:
            continue
        blob = blobs / digest.replace(":", "-")
        if not blob.is_file():
            continue
        relative = path.relative_to(manifests).parts
        name = "/".join(relative[:-1]) + ":" + path.name if len(relative) >= 2 else path.name
        found.append(ExternalModel("ollama", name, str(blob.parent), blob.name, blob.stat().st_size,
                                   _quant_from_name(str(data.get("config", {}).get("model_format") or ""))))
    return found


def scan_lmstudio(root: str | Path | None = None) -> list[ExternalModel]:
    found: list[ExternalModel] = []
    for base in lmstudio_roots(root):
        if not base.is_dir():
            continue
        for gguf in sorted(base.rglob("*.gguf")):
            if not gguf.is_file():
                continue
            relative = gguf.relative_to(base)
            name = str(relative.with_suffix("")).replace("\\", "/")
            found.append(ExternalModel("lmstudio", name, str(gguf.parent), gguf.name, gguf.stat().st_size,
                                       _quant_from_name(gguf.name)))
    return found


def discover_external_models(*, ollama: str | Path | None = None,
                             lmstudio: str | Path | None = None) -> list[ExternalModel]:
    return [*scan_ollama(ollama), *scan_lmstudio(lmstudio)]


def register_external_models(registry: ModelRegistry, models: Iterable[ExternalModel]) -> list[str]:
    added: list[str] = []
    for model in models:
        record = ModelRecord(
            repo_id=model.repo_id, pipeline_tag="text-generation", local_path=model.path, source="local",
            engine="gguf", gguf_file=model.gguf_file, size_bytes=model.size_bytes,
            license=None, metadata={"external": model.source, "display_name": model.name,
                                    "original_path": model.path, "quant": model.quant},
        )
        registry.upsert_model(record)
        added.append(record.repo_id)
    return added