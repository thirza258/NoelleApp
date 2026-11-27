"""Backend services for the v1.2 daily-driver features and the Agent Runtime foundation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from noelle.app.services import diagnostics, external_models, update_checker
from noelle.app.services.conversation_store import ConversationStore
from noelle.app.services.cost_ledger import CostLedger
from noelle.app.services.model_router import ModelRouter
from noelle.providers import credentials
from noelle.runtime.agent.environment_planner import plan_for_verdict
from noelle.runtime.agent.repository_analyzer import RepositoryAnalyzer
from noelle.runtime.adapters.factory import get_adapter_spec
from noelle.runtime.adapters.base_adapter import UnsupportedArchitectureError
from noelle.runtime import runnability


# ---------------------------------------------------------------- credentials
def test_provider_manager_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtCore import QObject

    from noelle.app.services.provider_manager import ProviderManager

    class FakeKeyring:
        def __init__(self) -> None:
            self.store: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, user: str) -> str | None:
            return self.store.get((service, user))

        def set_password(self, service: str, user: str, value: str) -> None:
            self.store[(service, user)] = value

        def delete_password(self, service: str, user: str) -> None:
            self.store.pop((service, user), None)

    import keyring

    fake = FakeKeyring()
    monkeypatch.setattr(keyring, "get_password", fake.get_password)
    monkeypatch.setattr(keyring, "set_password", fake.set_password)
    monkeypatch.setattr(keyring, "delete_password", fake.delete_password)
    credentials._memory_keys.clear()
    manager = ProviderManager()
    seen: list[tuple[str, bool]] = []
    manager.credentials_updated.connect(lambda provider, configured: seen.append((provider, configured)))
    assert manager.set_api_key("openai", "sk-test-12345678901234567890")
    assert manager.has_valid_key("openai")
    assert manager.get_api_key("openai").startswith("sk-test")
    assert manager.delete_api_key("openai")
    assert not manager.has_valid_key("openai")
    assert seen == [("openai", True), ("openai", False)]
    assert isinstance(manager, QObject)


# ------------------------------------------------------------------ cost ledger
def test_cost_ledger_records_and_summarises(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "cost.db")
    try:
        row = ledger.record_transaction("openai", "gpt-4o", 1_000_000, 500_000, 1200.0)
        assert row["cost_usd"] == pytest.approx(2.5 + 5.0)
        ledger.record_transaction("openrouter", "deepseek/deepseek-chat", 1_000_000, 0, 300.0)
        summary = ledger.get_session_summary()
        assert summary["calls"] == 2 and summary["total_tokens"] == 2_500_000
        assert summary["cost_usd"] > 0
        assert ledger.get_monthly_burn_rate() == pytest.approx(summary["cost_usd"])
        assert ledger.by_provider()[0]["provider"] in ("openai", "openrouter")
        unknown = ledger.record_transaction("custom", "unknown-model-xyz", 10, 10, 5.0)
        assert unknown["pricing_missing"] is True
    finally:
        ledger.close()


# ----------------------------------------------------------------- model router
def test_router_heuristics_and_fallbacks() -> None:
    router = ModelRouter(available_providers={"openrouter"}, local_model="local/qwen", local_ready=True)
    assert router.resolve("auto", [{"role": "user", "content": "hello there"}]).model_id == "deepseek/deepseek-chat"
    assert router.resolve("auto", [{"role": "user", "content": "solve this integral step by step"}]).model_id == \
        "deepseek/deepseek-r1"
    long = "word " * 70_000
    assert router.resolve("auto", [{"role": "user", "content": long}]).model_id == "google/gemini-pro-1.5"
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    assert router.resolve("auto", image).model_id == "anthropic/claude-3.5-sonnet"
    fast = router.resolve("frontier-fast", [])
    assert fast.model_id == "anthropic/claude-3.5-haiku"
    assert any(provider == "local" for provider, _model in fast.chain())
    chain = fast.chain()
    assert router.next_fallback(fast, chain[0]) == (chain[1] if len(chain) > 1 else None)


def test_router_respects_missing_provider() -> None:
    router = ModelRouter(available_providers=set())
    route = router.resolve("gpt-4o", [])
    assert route.provider_id == "openai" and route.fallbacks == ()


# ---------------------------------------------------------------- diagnostics
def test_diagnostics_redacts_secrets_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOELLE_HOME", str(tmp_path / "home"))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "app.log").write_text(
        "token=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 user=alice@example.com\n"
        f"path={tmp_path}/home/models secret=sk-live-abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8")
    target = diagnostics.build_report(
        tmp_path / "report.zip", description="Broke at C:\\Users\\alice\\Noelle",
        settings={"server": {"api_key": "top-secret-value"}, "ui": {"theme": "dark"}},
        loaded_models=[{"model_id": "org/model"}], log_dir=logs)
    with zipfile.ZipFile(target) as archive:
        blob = "".join(archive.read(name).decode("utf-8", errors="replace") for name in archive.namelist())
    assert "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in blob
    assert "sk-live-abcdefghijklmnopqrstuvwxyz" not in blob
    assert "alice@example.com" not in blob
    assert "top-secret-value" not in blob
    assert "<redacted>" in blob
    assert "org/model" in blob


# ---------------------------------------------------------------- update checker
def test_update_checker_semver_throttle_and_local_only() -> None:
    assert update_checker.is_newer("v1.2.0", "1.1.1")
    assert not update_checker.is_newer("1.1.1", "1.1.1")
    assert update_checker.parse_version("v2.0.3-beta") == (2, 0, 3)

    release = {"tag_name": "v1.2.0", "html_url": "https://example/release", "body": "notes"}
    update = update_checker.check_app(current="1.1.1", get_json=lambda _url: release)
    assert update is not None and update.version == "v1.2.0"
    assert update_checker.check_app(current="1.1.1", get_json=lambda _url: release,
                                    skipped_version="1.2.0") is None

    class Settings(dict):
        def get(self, key: str, default: Any = None) -> Any:
            return super().get(key, default)

        def set(self, key: str, value: Any) -> None:
            self[key] = value

    settings = Settings({"privacy.local_only": True})
    scheduler = update_checker.UpdateScheduler(settings, now=lambda: 1000.0)
    assert scheduler.run()["skipped"] is True
    settings["privacy.local_only"] = False
    settings["updates.check_models"] = False
    settings["updates.check_app"] = True
    result = scheduler.run(app_check=lambda **_kw: SimpleNamespace(version="v9.9.9"))
    assert result["app"].version == "v9.9.9"
    assert "updates.last_check.app" in settings


def test_model_update_compares_commit_hashes() -> None:
    class Api:
        def model_info(self, repo_id: str, revision: str = "main") -> Any:
            return SimpleNamespace(sha="new" if repo_id == "org/old" else "same")

    records = [SimpleNamespace(repo_id="org/old", source="hub", commit_hash="old", revision="main"),
               SimpleNamespace(repo_id="org/same", source="hub", commit_hash="same", revision="main"),
               SimpleNamespace(repo_id="local/x", source="local", commit_hash="old", revision=None)]
    updates = update_checker.check_models(records, api=Api())
    assert [update.repo_id for update in updates] == ["org/old"]


# --------------------------------------------------------------- external models
def test_external_model_scan_and_register(tmp_path: Path) -> None:
    from noelle.app.services.model_registry import ModelRegistry

    ollama = tmp_path / "ollama"
    manifest = ollama / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": external_models.OLLAMA_MANIFEST, "digest": "sha256:abc123"}]}), encoding="utf-8")
    blob = ollama / "blobs" / "sha256-abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF")
    lmstudio = tmp_path / "lmstudio" / "publisher" / "repo"
    lmstudio.mkdir(parents=True)
    (lmstudio / "model-Q4_K_M.gguf").write_bytes(b"GGUF")

    found = external_models.discover_external_models(ollama=ollama, lmstudio=tmp_path / "lmstudio")
    assert {model.source for model in found} == {"ollama", "lmstudio"}
    assert any(model.gguf_file == "sha256-abc123" for model in found)
    registry = ModelRegistry(tmp_path / "registry.db")
    try:
        added = external_models.register_external_models(registry, found)
        assert len(added) == 2
        record = registry.get_model(added[0], source="local")
        assert record is not None and record.engine == "gguf" and record.local_path
        assert record.metadata["external"] in ("ollama", "lmstudio")
    finally:
        registry.close()


# ------------------------------------------------------------ conversation store
def test_conversation_store_crud_search_and_prompts(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "chats.db")
    try:
        conversation = store.create_conversation(model_id="org/model", system_prompt="Be brief.")
        store.add_message(conversation, "user", "How do I quantize a model?")
        store.add_message(conversation, "assistant", "Use the Optimize tab.")
        listed = store.list_conversations()
        assert listed[0]["message_count"] == 2
        assert listed[0]["title"].startswith("How do I quantize")
        assert store.list_conversations(search="Optimize")[0]["id"] == conversation
        assert store.rename_conversation(conversation, "Quantization")
        assert store.pin_conversation(conversation, True)
        assert store.get_conversation(conversation)["messages"][0]["role"] == "user"
        assert "Quantization" in store.export(conversation, "markdown")
        assert json.loads(store.export(conversation, "json"))["id"] == conversation
        prompts = store.list_prompts()
        assert any(prompt["builtin"] for prompt in prompts)
        prompt_id = store.save_prompt("My prompt", "Answer in Indonesian.")
        assert any(prompt["id"] == prompt_id for prompt in store.list_prompts())
        assert store.delete_prompt(prompt_id)
        assert not store.delete_prompt(next(p["id"] for p in store.list_prompts() if p["builtin"]))
        assert isinstance(store.list_conversations(search='"""'), list)  # malformed FTS must not raise
        assert store.delete_conversation(conversation)
        assert store.messages(conversation) == []
        store.create_conversation(title="keep")
        assert store.delete_all() == 1
    finally:
        store.close()


# ------------------------------------------------------- agent runtime foundation
def test_repository_analyzer_detects_target_and_verdict() -> None:
    files = {
        "config.json": json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        "requirements.txt": "transformers>=4.50\naccelerate\n",
        "README.md": "```python\nimport timm\nfrom diffusers import DiffusionPipeline\n```",
    }
    report = RepositoryAnalyzer(fetch_file=lambda _repo, name: files.get(name)).inspect_repository(
        "org/model", token=None)
    assert report.family == "text" and report.target_class == "AutoModelForCausalLM"
    assert "timm" in report.required_libraries
    assert report.verdict in (runnability.RUNNABLE, runnability.UNKNOWN)


def test_adapter_factory_maps_formats() -> None:
    assert get_adapter_spec(family="text", library="gguf").engine == "llama.cpp"
    assert get_adapter_spec(family="text", library="mlx").engine == "mlx"
    assert get_adapter_spec(family="text", library="transformers").class_name == "TransformersTextEngine"
    assert get_adapter_spec(family="diffusion", library="diffusers").engine == "diffusers"
    with pytest.raises(UnsupportedArchitectureError):
        get_adapter_spec(family="image-classification", library="transformers", target_tool="timm")


def test_environment_plan_for_new_runtime_recreates_base() -> None:
    verdict = runnability.RunnabilityResult(runnability.NEEDS_NEWER_RUNTIME, engine="transformers")
    plan = plan_for_verdict(verdict)
    assert plan is not None and plan.recreate and plan.actionable
    assert plan.suggested_name([]) == "auto-transformers"
    assert plan.suggested_name(["auto-transformers"]) == "auto-transformers-2"