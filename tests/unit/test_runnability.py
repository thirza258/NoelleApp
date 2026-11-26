"""Shared runnability rules: fixture matrix for every family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from noelle.runtime import capabilities, runnability
from noelle.runtime.model_manager import ModelManager, ModelSpec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "checkpoints"


def _fixture(family: str, case: str) -> tuple[Path, list[str], dict[str, Any]]:
    root = FIXTURES / family / case
    files = runnability.file_names(root)
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config = {}
    return root, files, config


def _verdict(family: str, case: str, *, library: str | None, gated: bool = False,
             has_access: bool | None = None) -> runnability.RunnabilityResult:
    root, files, config = _fixture(family, case)
    return runnability.assess(family=family, library=library, files=files, config=config, gated=gated,
                              has_access=has_access, facts=runnability.collect_runtime_facts())


@pytest.mark.parametrize("family, case, library, expected", [
    ("text", "good", "transformers", runnability.RUNNABLE),
    ("text", "unknown-model-type", "transformers", runnability.NEEDS_NEWER_RUNTIME),
    ("text", "remote-code", "transformers", runnability.REMOTE_CODE),
    ("text", "onnx-only", "onnx", runnability.UNSUPPORTED_LIBRARY),
    ("vision", "good", "transformers", runnability.RUNNABLE),
    ("vision", "not-vlm", "transformers", runnability.UNSUPPORTED_LIBRARY),
    ("vision", "ultralytics", "ultralytics", runnability.UNSUPPORTED_LIBRARY),
    ("asr", "good", "transformers", runnability.RUNNABLE),
    ("asr", "nemo", "nemo", runnability.UNSUPPORTED_LIBRARY),
    ("asr", "ctranslate2", "ctranslate2", runnability.UNSUPPORTED_LIBRARY),
    ("tts", "good", "transformers", runnability.RUNNABLE),
    ("tts", "kokoro", None, runnability.UNSUPPORTED_LIBRARY),
    ("tts", "omnivoice", None, runnability.UNSUPPORTED_LIBRARY),
    ("tts", "stable-audio-3", "stable-audio-3", runnability.UNSUPPORTED_LIBRARY),
    ("diffusion", "good", "diffusers", runnability.RUNNABLE),
    ("diffusion", "single-file", "diffusers", runnability.UNSUPPORTED_LIBRARY),
    ("embedding", "good", "sentence-transformers", runnability.RUNNABLE),
    ("embedding", "model2vec", "model2vec", runnability.UNSUPPORTED_LIBRARY),
    ("embedding", "unknown-st", "sentence-transformers", runnability.NEEDS_NEWER_RUNTIME),
])
def test_fixture_matrix(family: str, case: str, library: str | None, expected: str) -> None:
    verdict = _verdict(family, case, library=library)
    assert verdict.verdict == expected, verdict.to_dict()


def test_gguf_only_is_runnable_or_needs_the_component() -> None:
    verdict = _verdict("text", "gguf-only", library="gguf")
    assert verdict.verdict in (runnability.RUNNABLE, runnability.NEEDS_COMPONENT)
    assert verdict.engine == "llama.cpp"


def test_unknown_metadata_stays_supported_until_the_detail_page() -> None:
    verdict = runnability.assess(family="text", library="transformers", files=[], config={}, facts=None)
    assert verdict.verdict == runnability.UNKNOWN and verdict.supported


def test_gated_without_access_is_blocked() -> None:
    root, files, config = _fixture("diffusion", "good")
    verdict = runnability.assess(family="diffusion", library="diffusers", files=files, config=config,
                                 gated=True, has_access=False, facts=runnability.collect_runtime_facts())
    assert verdict.verdict == runnability.GATED_NO_ACCESS and not verdict.supported


def test_capabilities_use_the_same_verdict() -> None:
    facts = runnability.collect_runtime_facts()
    root, files, config = _fixture("text", "good")
    summary = SimpleNamespace(id="org/good", pipeline_tag="text-generation", library_name="transformers",
                              tags=[], siblings=[{"name": name} for name in files], config=config, gated=False,
                              card_data={}, gguf=None, base_models=[], is_adapter=False, param_count=None,
                              weight_bytes=None, license=None)
    cap = capabilities.analyze(summary, config, runtime_facts=facts)
    assert cap.verdict == runnability.RUNNABLE and cap.supported and cap.runtimes == ("transformers",)

    _, kokoro_files, kokoro_config = _fixture("tts", "kokoro")
    kokoro = SimpleNamespace(id="hexgrad/Kokoro-82M", pipeline_tag="text-to-speech", library_name=None, tags=[],
                             siblings=[{"name": name} for name in kokoro_files], config=kokoro_config, gated=False,
                             card_data={}, gguf=None, base_models=[], is_adapter=False, param_count=None,
                             weight_bytes=None, license=None)
    cap = capabilities.analyze(kokoro, kokoro_config, runtime_facts=facts)
    assert cap.verdict == runnability.UNSUPPORTED_LIBRARY and not cap.supported


def test_worker_check_raises_typed_errors() -> None:
    from noelle.runtime.errors import NeedsNewerRuntime, RemoteCodeBlocked, Unsupported

    facts = runnability.collect_runtime_facts()
    runnability.collect_runtime_facts.cache_clear()  # keep the next call deterministic after monkeypatching if needed
    try:
        ModelManager._check_runnability(ModelSpec(model_id="hexgrad/Kokoro-82M", task="text-to-speech"),
                                        str(FIXTURES / "tts" / "kokoro"))
        raise AssertionError("Kokoro must not pass the worker check")
    except Unsupported:
        pass
    try:
        ModelManager._check_runnability(ModelSpec(model_id="k2-fsa/OmniVoice", task="text-to-speech"),
                                        str(FIXTURES / "tts" / "omnivoice"))
        raise AssertionError("OmniVoice must not pass the worker check")
    except Unsupported:
        pass
    with pytest.raises(NeedsNewerRuntime):
        ModelManager._check_runnability(ModelSpec(model_id="org/new", task="text-generation",
                                                  library_name="transformers"),
                                        str(FIXTURES / "text" / "unknown-model-type"))
    with pytest.raises(RemoteCodeBlocked):
        ModelManager._check_runnability(ModelSpec(model_id="org/custom", task="text-generation",
                                                  library_name="transformers"),
                                        str(FIXTURES / "text" / "remote-code"))
    assert facts  # collected (and cached) without raising


def test_downloader_gated_preflight_emits_one_terminal_event(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    from huggingface_hub.errors import GatedRepoError

    from noelle.runtime import downloader

    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/org/gated"))
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(downloader, "emit", events.append)

    class FakeApi:
        def __init__(self, token: Any = None) -> None:
            pass

        def model_info(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(gated=True, sha="abc123", siblings=[])

        def auth_check(self, *args: Any, **kwargs: Any) -> None:
            raise GatedRepoError("gated", response=response)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    args = argparse.Namespace(repo_id="stabilityai/stable-audio-3-small-sfx", revision=None, include=None,
                              exclude=None, local_dir=None, dry_run=False, max_workers=1, retries=1,
                              no_verify=True)
    assert downloader.run(args) == 1
    assert [event.get("code") for event in events] == ["GATED"]
    assert events[0]["url"] == "https://huggingface.co/stabilityai/stable-audio-3-small-sfx"
    assert "not approved" in events[0]["message"] or "gated" in events[0]["message"].lower()


def test_environment_plan_targets_the_missing_engine() -> None:
    from noelle.runtime.agent.environment_planner import plan_for_verdict

    component = runnability.RunnabilityResult(runnability.NEEDS_COMPONENT, family="diffusion",
                                              engine="diffusers", reason="missing")
    plan = plan_for_verdict(component)
    assert plan is not None and plan.actionable and plan.profiles == ("diffusion",)
    newer = runnability.RunnabilityResult(runnability.NEEDS_NEWER_RUNTIME, engine="transformers",
                                          model_type="brand_new")
    plan = plan_for_verdict(newer)
    assert plan is not None and plan.recreate and plan.packages == ("transformers",)
    unsupported = runnability.RunnabilityResult(runnability.UNSUPPORTED_LIBRARY, library="kokoro")
    assert plan_for_verdict(unsupported) is None