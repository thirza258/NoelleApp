"""Shared services handed to every page."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from noelle.config import Settings
from noelle.runtime.preflight import Compatibility, assess_compatibility

from noelle.app.services.hf_client import HFClient, ModelSummary
from noelle.app.services.model_registry import ModelRecord, ModelRegistry


@dataclass
class AppContext:
    settings: Settings
    registry: ModelRegistry
    hf: HFClient
    runtime: Any  # RuntimeClient
    hardware: Any  # HardwareManager
    downloads: Any  # DownloadManager
    server: Any  # ServerManager
    toast: Callable[[str, str], None] = lambda _text, _kind="info": None
    open_model: Callable[[Any], None] = lambda _model: None
    navigate: Callable[[str], None] = lambda _page: None
    extras: dict[str, Any] = field(default_factory=dict)

    def compat(self, model: Any) -> Compatibility:
        budget = self.hardware.memory_budget(float(self.settings.get("runtime.safety_headroom", 0.15)))
        return assess_compatibility(getattr(model, "param_count", None), budget,
                                    config=getattr(model, "config", None) or None,
                                    weight_bytes=getattr(model, "weight_bytes", None),
                                    prequantized=bool(getattr(model, "prequantized", False)))

    def downloaded_ids(self) -> set[str]:
        return {r.repo_id for r in self.registry.list_models() if r.exists_on_disk()}


def summary_from_record(record: ModelRecord) -> ModelSummary:
    tags = list((record.metadata or {}).get("tags") or [])
    moment = datetime.fromtimestamp(record.downloaded_at, tz=timezone.utc) if record.downloaded_at else None
    return ModelSummary(
        id=record.repo_id,
        author=record.repo_id.split("/")[0] if "/" in record.repo_id else "local",
        pipeline_tag=record.pipeline_tag,
        library_name=record.library_name,
        tags=tags,
        license=record.license,
        param_count=record.param_count,
        last_modified=moment,
        config={"architectures": [record.architecture]} if record.architecture else {},
    )
