"""Register models that are already in the Hugging Face cache, offline.

Used on first launch (empty registry) and by Storage › "Add cached models". It only
reads the cache: the task is inferred from local config files, the license from the
README front matter and the parameter count from safetensors headers. Nothing is
downloaded, and repositories that are incomplete or of an unsupported task are skipped.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from noelle.paths import hf_hub_cache
from noelle.runtime import runnability, tasks
from noelle.runtime.hub_cache import cached_snapshot_path, is_downloaded

from noelle.app.services.model_registry import ModelRecord, ModelRegistry

_ARCH_RULES: tuple[tuple[str, str], ...] = (
    ("ForCausalLM", "text-generation"),
    ("LMHeadModel", "text-generation"),
    ("ForSeq2SeqLM", "text-generation"),
    ("ForSpeechSeq2Seq", "automatic-speech-recognition"),
    ("WhisperForConditionalGeneration", "automatic-speech-recognition"),
    ("ForCTC", "automatic-speech-recognition"),
    ("VitsModel", "text-to-speech"),
    ("ForTextToSpeech", "text-to-speech"),
    ("ForTextToWaveform", "text-to-speech"),
    ("ForObjectDetection", "object-detection"),
    ("ForImageClassification", "image-classification"),
    ("ForImageTextToText", "image-text-to-text"),
    ("ForVision2Seq", "image-text-to-text"),
    ("VisionEncoderDecoderModel", "image-text-to-text"),
    ("ForSequenceClassification", "text-classification"),
)
_VISION_HINTS = ("vision_config", "visual", "image_token_index", "image_token_id")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def infer_pipeline_tag(snapshot: Path) -> str | None:
    if (snapshot / "model_index.json").is_file():
        index = _read_json(snapshot / "model_index.json")
        name = str(index.get("_class_name", ""))
        return "image-to-image" if "Img2Img" in name or "Image2Image" in name else "text-to-image"
    if (snapshot / "modules.json").is_file() or (snapshot / "config_sentence_transformers.json").is_file():
        return "sentence-similarity"
    config = _read_json(snapshot / "config.json")
    architectures = [str(a) for a in config.get("architectures") or []]
    for arch in architectures:
        if arch.endswith("ForConditionalGeneration") and any(hint in config for hint in _VISION_HINTS):
            return "image-text-to-text"
        for suffix, tag in _ARCH_RULES:
            if suffix in arch:
                return tag
    if config.get("model_type") == "vits":
        return "text-to-speech"
    return None


def license_from_readme(snapshot: Path) -> str | None:
    readme = snapshot / "README.md"
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    for line in text[3:text.find("\n---", 3)].splitlines():
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"") or None
    return None


def count_safetensors_params(snapshot: Path) -> int | None:
    """Sum tensor element counts from safetensors headers (8-byte length + JSON), without loading weights."""
    total = 0
    files = sorted(snapshot.rglob("*.safetensors"))
    if not files:
        return None
    for path in files:
        try:
            with path.open("rb") as fh:
                (length,) = struct.unpack("<Q", fh.read(8))
                if length > 100_000_000:
                    return None
                header = json.loads(fh.read(length))
        except (OSError, ValueError, struct.error):
            return None
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            count = 1
            for dim in meta.get("shape") or []:
                count *= int(dim)
            total += count
    return total or None


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".npz")


def has_weights(snapshot: Path) -> bool:
    return any(p.suffix in _WEIGHT_SUFFIXES and p.is_file() for p in snapshot.rglob("*"))


def _library(snapshot: Path) -> str:
    if (snapshot / "model_index.json").is_file():
        return "diffusers"
    if (snapshot / "modules.json").is_file():
        return "sentence-transformers"
    return "transformers"


def discover_cached_models(cache_dir: Path | None = None) -> list[ModelRecord]:
    cache = Path(cache_dir) if cache_dir else hf_hub_cache()
    if not cache.is_dir():
        return []
    records: list[ModelRecord] = []
    for folder in sorted(cache.glob("models--*")):
        repo_id = folder.name[len("models--"):].replace("--", "/", 1)
        snapshot = cached_snapshot_path(repo_id, cache_dir=cache)
        if snapshot is None or not is_downloaded(repo_id, cache_dir=cache):
            continue
        if (snapshot / "adapter_config.json").is_file():
            continue  # adapters are listed per base model, not as standalone models
        if not has_weights(snapshot):
            continue  # config/tokenizer-only snapshots cannot be loaded
        tag = infer_pipeline_tag(snapshot)
        if not tasks.is_supported(tag):
            continue
        config = _read_json(snapshot / "config.json")
        architectures = config.get("architectures") or []
        # Flag models the bundled runtimes cannot load, before the user clicks Load.
        verdict = runnability.assess_local(snapshot, family=tasks.task_family(tag), library=_library(snapshot))
        records.append(ModelRecord(
            repo_id=repo_id, pipeline_tag=tag, local_path=str(snapshot), commit_hash=snapshot.name,
            library_name=_library(snapshot), architecture=str(architectures[0]) if architectures else None,
            param_count=count_safetensors_params(snapshot), license=license_from_readme(snapshot),
            size_bytes=sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file()),
            metadata={"imported_from_cache": True, "runnability": verdict.verdict,
                      "runnability_reason": verdict.reason or verdict.hint},
        ))
    return records


def import_cached_models(registry: ModelRegistry, cache_dir: Path | None = None) -> list[str]:
    """Register cached models the registry does not know yet; returns the repo ids added."""
    known = {r.repo_id for r in registry.list_models()}
    added = []
    for record in discover_cached_models(cache_dir):
        if record.repo_id in known:
            continue
        registry.upsert_model(record)
        added.append(record.repo_id)
    return added
