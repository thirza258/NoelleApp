"""Static repository inspection: architecture, dependencies and an adapter target.

Pure Python. Network access is injected (``api``/``fetch_file``) so tests and the CLI can
run offline, and the desktop shell can pass its existing :class:`HFClient`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from noelle.runtime import runnability, tasks

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*))", re.MULTILINE)
_REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_KNOWN_IMPORTS = {"timm", "diffusers", "transformers", "torch", "torchaudio", "sentence_transformers",
                  "peft", "accelerate", "soundfile", "librosa", "PIL"}

# architectures suffix -> target adapter class (README/PRD naming)
_ARCH_TARGETS: tuple[tuple[str, str], ...] = (
    ("ForCausalLM", "AutoModelForCausalLM"),
    ("ForSeq2SeqLM", "AutoModelForSeq2SeqLM"),
    ("ForConditionalGeneration", "AutoModelForConditionalGeneration"),
    ("ForImageTextToText", "AutoModelForImageTextToText"),
    ("ForVision2Seq", "AutoModelForVision2Seq"),
    ("VisionEncoderDecoderModel", "VisionEncoderDecoderModel"),
    ("ForObjectDetection", "AutoModelForObjectDetection"),
    ("ForImageClassification", "AutoModelForImageClassification"),
    ("ForSpeechSeq2Seq", "AutoModelForSpeechSeq2Seq"),
    ("ForCTC", "AutoModelForCTC"),
    ("ForTextToSpectrogram", "AutoModelForTextToSpectrogram"),
    ("ForTextToWaveform", "AutoModelForTextToWaveform"),
    ("ForMaskedLM", "AutoModelForMaskedLM"),
    ("Model", "AutoModel"),
)


@dataclass(frozen=True)
class ModelInspectionReport:
    repo_id: str
    pipeline_tag: str | None
    library_name: str | None
    family: str | None
    architectures: tuple[str, ...] = ()
    model_type: str | None = None
    torch_dtype: str | None = None
    required_libraries: tuple[str, ...] = ()
    readme_imports: tuple[str, ...] = ()
    target_class: str | None = None
    target_tool: str | None = None
    remote_code: bool = False
    verdict: str | None = None
    verdict_reason: str | None = None
    weight_bytes: int | None = None
    estimated_vram_gb: float | None = None

    @property
    def supported(self) -> bool:
        return runnability.supported(self.verdict or runnability.UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id, "pipeline_tag": self.pipeline_tag, "library_name": self.library_name,
            "family": self.family, "architectures": list(self.architectures), "model_type": self.model_type,
            "torch_dtype": self.torch_dtype, "required_libraries": list(self.required_libraries),
            "readme_imports": list(self.readme_imports), "target_class": self.target_class,
            "target_tool": self.target_tool, "remote_code": self.remote_code, "verdict": self.verdict,
            "verdict_reason": self.verdict_reason, "weight_bytes": self.weight_bytes,
            "estimated_vram_gb": self.estimated_vram_gb, "supported": self.supported,
        }


def _read_json(fetch: Callable[[str], str | None], name: str) -> dict[str, Any]:
    try:
        text = fetch(name)
    except Exception:  # noqa: BLE001 - metadata is optional
        return {}
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _requirements(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    names: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or line.startswith("git+"):
            continue
        match = _REQUIREMENT_RE.match(line)
        if match:
            names.append(match.group(1).lower())
    return tuple(dict.fromkeys(names))


def _readme_imports(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    found = set()
    for match in _IMPORT_RE.finditer(text):
        name = (match.group(1) or match.group(2) or "").split(".")[0]
        if name in _KNOWN_IMPORTS:
            found.add(name)
    return tuple(sorted(found))


_ARCH_PIPELINES: tuple[tuple[str, str], ...] = (
    ("ForCausalLM", "text-generation"),
    ("ForSeq2SeqLM", "text-generation"),
    ("ForImageTextToText", "image-text-to-text"),
    ("ForVision2Seq", "image-text-to-text"),
    ("VisionEncoderDecoderModel", "image-text-to-text"),
    ("ForObjectDetection", "object-detection"),
    ("ForImageClassification", "image-classification"),
    ("ForSpeechSeq2Seq", "automatic-speech-recognition"),
    ("ForCTC", "automatic-speech-recognition"),
    ("ForTextToSpectrogram", "text-to-speech"),
    ("ForTextToWaveform", "text-to-speech"),
    ("ForMaskedLM", "feature-extraction"),
)


def _infer_pipeline_tag(architectures: tuple[str, ...]) -> str | None:
    architecture = architectures[0] if architectures else ""
    for suffix, tag in _ARCH_PIPELINES:
        if architecture.endswith(suffix):
            return tag
    return None


def _target(architecture: str | None, imports: tuple[str, ...],
            pipeline_tag: str | None) -> tuple[str | None, str | None]:
    family = tasks.task_family(pipeline_tag)
    vision_family = family in (tasks.VISION, tasks.DETECTION, tasks.CLASSIFICATION)
    if "timm" in imports and vision_family:
        return None, "timm"
    if family == tasks.DIFFUSION or (pipeline_tag or "").startswith(("text-to-image", "image-to-image")) or (
            family is None and "diffusers" in imports):
        return "DiffusionPipeline", "diffusers"
    for suffix, target in _ARCH_TARGETS:
        if architecture and architecture.endswith(suffix):
            return target, "transformers"
    return {
        tasks.TEXT: "AutoModelForCausalLM",
        tasks.VISION: "AutoModelForImageTextToText",
        tasks.DETECTION: "AutoModelForObjectDetection",
        tasks.CLASSIFICATION: "AutoModelForImageClassification",
        tasks.ASR: "AutoModelForSpeechSeq2Seq",
        tasks.TTS: "AutoModelForTextToWaveform",
        tasks.EMBEDDING: "SentenceTransformer",
    }.get(family or ""), None


def _vram_gb(weight_bytes: int | None, torch_dtype: str | None) -> float | None:
    """Rough residency estimate; the stored weight size already reflects its dtype."""
    if not weight_bytes:
        return None
    return round(weight_bytes / 1e9 * 1.15, 2)


class RepositoryAnalyzer:
    """Inspect a Hub repository with an injected API (no implicit network)."""

    def __init__(self, *, api: Any = None, fetch_file: Callable[[str, str], str | None] | None = None,
                 facts: runnability.RuntimeFacts | None = None) -> None:
        self.api = api
        self.fetch_file = fetch_file
        self.facts = facts

    def inspect_repository(self, repo_id: str, token: str | None = None) -> ModelInspectionReport:
        info = None
        if self.api is not None:
            try:
                info = self.api.model_info(repo_id, files_metadata=not callable(self.fetch_file))
            except TypeError:
                info = self.api.model_info(repo_id)
            except Exception:  # noqa: BLE001 - report what metadata is available
                info = None
        fetch = self.fetch_file
        if fetch is None:
            if self.api is None:
                return self._report(repo_id, info=None, config={}, requirements=None, readme=None)
            def fetch(_repo: str, name: str) -> str | None:  # noqa: ARG001
                return self._api_file(_repo, name)
        config = _read_json(lambda name: fetch(repo_id, name), "config.json")
        requirements = fetch(repo_id, "requirements.txt")
        readme = fetch(repo_id, "README.md")
        return self._report(repo_id, info=info, config=config, requirements=requirements, readme=readme)

    def _api_file(self, repo_id: str, name: str) -> str | None:
        try:
            path = self.api.hf_hub_download(repo_id, name) if hasattr(self.api, "hf_hub_download") else None
            return Path(path).read_text(encoding="utf-8", errors="replace") if path else None
        except Exception:  # noqa: BLE001
            return None

    def _report(self, repo_id: str, *, info: Any, config: Mapping[str, Any], requirements: str | None,
                readme: str | None) -> ModelInspectionReport:
        architectures = tuple(str(a) for a in (config.get("architectures") or []) if a)
        pipeline_tag = getattr(info, "pipeline_tag", None) or _infer_pipeline_tag(architectures)
        library = getattr(info, "library_name", None)
        tags = [str(t) for t in (getattr(info, "tags", None) or ())]
        siblings = []
        for item in getattr(info, "siblings", None) or ():
            name = getattr(item, "rfilename", None) or (item.get("name") if isinstance(item, Mapping) else None)
            if name:
                siblings.append(name)
        model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else None
        imports = tuple(sorted(set(_readme_imports(readme)) | set(_requirements(requirements))))
        target_class, target_tool = _target(architectures[0] if architectures else None, imports, pipeline_tag)
        family = tasks.task_family(pipeline_tag)
        gated = bool(getattr(info, "gated", False))
        verdict = runnability.assess(family=family, library=library, files=siblings, config=config, gated=gated,
                                     facts=self.facts)
        weight_bytes = None
        for item in getattr(info, "siblings", None) or ():
            size = getattr(item, "size", None)
            name = getattr(item, "rfilename", "") or ""
            if size and name.endswith((".safetensors", ".bin")):
                weight_bytes = (weight_bytes or 0) + int(size)
        return ModelInspectionReport(
            repo_id=repo_id, pipeline_tag=pipeline_tag, library_name=library, family=family,
            architectures=architectures, model_type=model_type,
            torch_dtype=str(config.get("torch_dtype") or config.get("dtype") or "") or None,
            required_libraries=imports, readme_imports=_readme_imports(readme),
            target_class=target_class, target_tool=target_tool,
            remote_code=bool(config.get("auto_map")) or "custom_code" in tags,
            verdict=verdict.verdict, verdict_reason=verdict.reason or verdict.hint,
            weight_bytes=weight_bytes, estimated_vram_gb=_vram_gb(weight_bytes, str(config.get("torch_dtype") or "")),
        )


def analyze_local(source: str | Path, *, family: str | None = None, library: str | None = None,
                  facts: runnability.RuntimeFacts | None = None) -> ModelInspectionReport:
    root = Path(source).expanduser()
    config = runnability.read_config(root)
    names = runnability.file_names(root)
    architectures = tuple(str(a) for a in (config.get("architectures") or []) if a)
    readme = ""
    try:
        readme = (root / "README.md").read_text(encoding="utf-8", errors="replace")[:200_000]
    except OSError:
        pass
    requirements = ""
    try:
        requirements = (root / "requirements.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    imports = tuple(sorted(set(_readme_imports(readme)) | set(_requirements(requirements))))
    family = family or tasks.task_family(config.get("pipeline_tag"))
    target_class, target_tool = _target(architectures[0] if architectures else None, imports, None)
    verdict = runnability.assess_local(root, family=family, library=library, facts=facts)
    weight_bytes = sum((root / name).stat().st_size for name in names
                       if name.endswith((".safetensors", ".bin")) and (root / name).is_file())
    return ModelInspectionReport(
        repo_id=str(root), pipeline_tag=None, library_name=library, family=family, architectures=architectures,
        model_type=config.get("model_type") if isinstance(config.get("model_type"), str) else None,
        torch_dtype=str(config.get("torch_dtype") or config.get("dtype") or "") or None,
        required_libraries=imports, readme_imports=_readme_imports(readme), target_class=target_class,
        target_tool=target_tool, remote_code=bool(config.get("auto_map")),
        verdict=verdict.verdict, verdict_reason=verdict.reason or verdict.hint,
        weight_bytes=weight_bytes or None,
        estimated_vram_gb=_vram_gb(weight_bytes or None, str(config.get("torch_dtype") or "")),
    )