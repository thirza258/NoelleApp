"""Shared services handed to every page."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from noelle.config import Settings
from noelle.runtime import advisor, capabilities, runnability
from noelle.runtime.preflight import Compatibility

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
        """Memory verdict plus the overall 0-100 score (cheap: no network, used for every Discover card)."""
        assessment = self.assess(model, measured=False)
        return advisor.with_score(assessment.compatibility, assessment)

    # ------------------------------------------------------------ model intelligence
    def hardware_profile(self) -> advisor.HardwareProfile:
        budget = self.hardware.memory_budget(float(self.settings.get("runtime.safety_headroom", 0.15)))
        gpu = getattr(self.hardware, "gpu", None)
        name = getattr(gpu, "name", "") or ""
        beat = getattr(self.runtime, "last_heartbeat_payload", None) or {}
        backends = beat.get("backends") or capabilities.detect_backends_cached()
        return advisor.HardwareProfile(budget.device, name, budget, dict(backends),
                                       advisor.memory_bandwidth(name, budget.device))

    def is_trusted(self, repo_id: str) -> bool:
        """Remote code allowed for this repository, by repository or by publisher."""
        from noelle.app.services.model_registry import remote_code_trusted

        return remote_code_trusted(self.settings, repo_id)

    def local_variants(self, repo_id: str) -> dict[str, str]:
        """precision -> repo_id of local builds (Optimize tab) derived from ``repo_id``."""
        return {r.precision: r.repo_id for r in self.registry.list_models()
                if r.base_model == repo_id and r.source == "quantized" and r.exists_on_disk()}

    def local_provider(self) -> Any:
        """The real worker-backed local provider (cached for this runtime client)."""
        from noelle.providers.local_provider import LocalProvider

        provider = self.extras.get("local_provider")
        if provider is None or getattr(provider, "runtime_client", None) is not self.runtime:
            provider = LocalProvider(runtime_client=self.runtime)
            self.extras["local_provider"] = provider
        return provider

    def hybrid_provider(self, consent: Any = None) -> Any:
        """Local-first provider honouring ``privacy.local_only`` and the fallback setting."""
        from noelle.providers.hybrid import hybrid_from_settings

        return hybrid_from_settings(self.settings, self.local_provider(), consent=consent)

    def runtime_facts(self) -> runnability.RuntimeFacts | None:
        """Model types/backends reported by the worker (None until its HELLO result arrives)."""
        payload = getattr(self.runtime, "runtime_caps", None)
        return runnability.RuntimeFacts.from_payload(payload) if payload else None

    def runnability(self, model: Any, *, config: dict[str, Any] | None = None,
                    has_access: bool | None = None) -> runnability.RunnabilityResult:
        """One verdict for a Hub summary/registry record using the worker's runtime facts."""
        return runnability.assess_model(model, config=config, facts=self.runtime_facts(), has_access=has_access)

    def assess(self, model: Any, *, config: dict[str, Any] | None = None, profile: str = "balanced",
               tokenizer_config: dict[str, Any] | None = None, measured: bool = True,
               session_trusted: bool = False) -> advisor.Assessment:
        cfg = config or getattr(model, "config", None) or None
        cap = capabilities.analyze(model, cfg, tokenizer_config=tokenizer_config,
                                  runtime_facts=self.runtime_facts())
        rows = self.registry.measurements(cap.model_id) if measured else []
        return advisor.assess(cap, self.hardware_profile(), config=cfg, profile=profile,
                              measured=advisor.summarize_measurements(rows),
                              local_variants=self.local_variants(cap.model_id) if measured else None,
                              trusted=session_trusted or self.is_trusted(cap.model_id))

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
