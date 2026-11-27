"""Cloud and hybrid providers.

``CloudProvider`` presents OpenAI + OpenRouter behind the same contract as the local
worker. ``HybridProvider`` routes per task and only falls back to the cloud when the
caller supplied a consent callback (or explicitly allowed it); it never sends data
silently and never mixes partial local output with a cloud answer.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Mapping

from noelle.providers.credentials import has_api_key
from noelle.providers.openai_provider import OpenAIProvider
from noelle.providers.openrouter_provider import OpenRouterProvider
from noelle.providers.provider import AsyncProviderMixin, ProviderCapabilities, ProviderError

log = logging.getLogger("noelle.providers.hybrid")

FALLBACK_CODES = frozenset({"MODEL_NOT_FOUND", "DEPENDENCY_MISSING", "UNSUPPORTED", "UNSUPPORTED_LIBRARY",
                            "NEEDS_NEWER_RUNTIME", "REMOTE_CODE_BLOCKED", "LOAD_FAILED", "INTERNAL", "OOM",
                            "WORKER_CRASHED", "WORKER_NOT_READY", "WORKER_UNAVAILABLE", "TIMEOUT",
                            "MODEL_REQUIRED"})

DEFAULT_POLICY: dict[str, str] = {
    "chat": "local", "vision": "local", "embedding": "local", "tts": "local", "stt": "local",
    "image_generation": "local", "fallback": "cloud",
}


def _openai_model(model: str | None) -> bool:
    return bool(model and (model.startswith(("gpt-", "o1", "o3", "text-embedding", "whisper", "tts-"))
                           or model in ("gpt-4o",)))


class CloudProvider(AsyncProviderMixin):
    id = "cloud"
    name = "Cloud AI Gateway"
    is_local = False

    def __init__(self, *, openai: OpenAIProvider | None = None, openrouter: OpenRouterProvider | None = None,
                 default_model: str = "anthropic/claude-3.5-sonnet") -> None:
        self.openai = openai or OpenAIProvider()
        self.openrouter = openrouter or OpenRouterProvider()
        self.default_model = default_model

    def _pick(self, model: str | None, *, modality: str = "chat") -> Any:
        if modality in ("embedding", "stt", "tts"):
            if not has_api_key("openai"):
                raise ProviderError("MISSING_API_KEY", "OpenAI is required for this modality.",
                                    category="auth", recoverable=False,
                                    suggested_actions=("configure_api_key",), provider_id="openai")
            return self.openai
        if _openai_model(model):
            return self.openai
        if has_api_key("openrouter"):
            return self.openrouter
        if has_api_key("openai"):
            return self.openai
        raise ProviderError("NO_CLOUD_PROVIDER", "No cloud API key is configured.", category="auth",
                            recoverable=False, suggested_actions=("configure_api_key",))

    def _effective_model(self, provider: Any, model: str | None) -> str:
        """A provider must never receive a model id from the other provider's catalogue."""
        model_id = model or self.default_model
        if provider is self.openai and not _openai_model(model_id):
            return "gpt-4o-mini"
        if provider is self.openrouter and _openai_model(model_id):
            return self.default_model if not _openai_model(self.default_model) else "deepseek/deepseek-chat"
        return model_id

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, **params: Any) -> str:
        provider = self._pick(model or self.default_model)
        return provider.chat(messages, model=self._effective_model(provider, model), **params)

    def stream_chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
                    **params: Any) -> Iterator[str]:
        provider = self._pick(model or self.default_model)
        yield from provider.stream_chat(messages, model=self._effective_model(provider, model), **params)

    def embed(self, texts: list[str], *, model: str | None = None, **params: Any) -> list[list[float]]:
        return self._pick(model, modality="embedding").embed(texts, model=model, **params)

    def transcribe(self, audio_path: str, *, model: str | None = None, language: str | None = None,
                   **params: Any) -> dict[str, Any]:
        return self._pick(model, modality="stt").transcribe(audio_path, model=model, language=language, **params)

    def synthesize(self, text: str, *, model: str | None = None, voice: int | str = "alloy",
                   **params: Any) -> dict[str, Any]:
        return self._pick(model, modality="tts").synthesize(text, model=model, voice=voice, **params)

    def generate_image(self, prompt: str, *, model: str | None = None, **params: Any) -> dict[str, Any]:
        raise ProviderError("UNSUPPORTED_TASK", "No cloud image provider is configured.", category="request",
                            suggested_actions=("use_local_diffusion",))

    def vision(self, image_path: str, prompt: str = "", *, model: str | None = None,
               **params: Any) -> str:
        provider = self._pick(model)
        return provider.vision(image_path, prompt, model=self._effective_model(provider, model), **params)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from noelle.runtime.tools import execute_tool

        return execute_tool(name, arguments).to_dict()

    def health(self) -> dict[str, Any]:
        return {"provider_id": self.id, "is_local": False,
                "openai": has_api_key("openai"), "openrouter": has_api_key("openrouter")}

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(chat=True, streaming=True, vision=True, audio_input=has_api_key("openai"),
                                    audio_output=has_api_key("openai"), tools=True,
                                    embedding=has_api_key("openai"), image_generation=False, is_local=False,
                                    models=(self.default_model, "gpt-4o-mini", "deepseek/deepseek-chat"))


class HybridProvider(AsyncProviderMixin):
    """Local-first router with explicit, consented cloud fallback."""

    id = "hybrid"
    name = "Hybrid (Local + Cloud)"
    is_local = False

    def __init__(self, local: Any, cloud: CloudProvider | None = None, *,
                 policy: Mapping[str, str] | None = None, local_only: bool = False,
                 consent: Callable[[str, str], bool] | None = None) -> None:
        self.local = local
        self.cloud = cloud or CloudProvider()
        self.policy = {**DEFAULT_POLICY, **dict(policy or {})}
        self.local_only = bool(local_only)
        self.consent = consent
        self.last_route = "local"
        self.last_reason = ""
        self.last_error: ProviderError | None = None

    @property
    def indicator(self) -> str:
        return "local" if self.last_route == "local" else "cloud"

    def _may_fallback(self, error: ProviderError) -> bool:
        if self.local_only or self.policy.get("fallback") == "none":
            return False
        if error.code not in FALLBACK_CODES:
            return False
        if self.consent is not None:
            destination = "OpenAI / OpenRouter"
            return bool(self.consent(error.info.message, destination))
        return False

    def _note(self, route: str, reason: str, error: ProviderError | None = None) -> None:
        self.last_route = route
        self.last_reason = reason
        self.last_error = error

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, **params: Any) -> str:
        prefer = self.policy.get("chat", "local")
        if prefer == "cloud" and not self.local_only:
            self._note("cloud", "policy: cloud preferred")
            return self.cloud.chat(messages, model=model, **params)
        try:
            text = self.local.chat(messages, model=model, **params)
            self._note("local", "local runtime")
            return text
        except ProviderError as error:
            if not self._may_fallback(error):
                self._note("local", "local failed; fallback denied", error)
                raise
            self._note("cloud", f"cloud fallback: {error.code}", error)
            return self.cloud.chat(messages, model=model, **params)

    def stream_chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
                    **params: Any) -> Iterator[str]:
        prefer = self.policy.get("chat", "local")
        if prefer == "cloud" and not self.local_only:
            self._note("cloud", "policy: cloud preferred")
            yield from self.cloud.stream_chat(messages, model=model, **params)
            return
        yielded = False
        try:
            for token in self.local.stream_chat(messages, model=model, **params):
                yielded = True
                yield token
            self._note("local", "local runtime")
            return
        except ProviderError as error:
            if yielded or not self._may_fallback(error):
                self._note("local", "local failed after streaming; fallback denied", error)
                raise
            self._note("cloud", f"cloud fallback: {error.code}", error)
        yield from self.cloud.stream_chat(messages, model=model, **params)

    def _route(self, task: str, method: str, *args: Any, **kwargs: Any) -> Any:
        prefer = self.policy.get(task, "local")
        if prefer == "cloud" and not self.local_only:
            self._note("cloud", f"policy: {task} on cloud")
            return getattr(self.cloud, method)(*args, **kwargs)
        try:
            result = getattr(self.local, method)(*args, **kwargs)
            self._note("local", f"{task} on local runtime")
            return result
        except ProviderError as error:
            if not self._may_fallback(error):
                self._note("local", f"{task} failed; fallback denied", error)
                raise
            self._note("cloud", f"cloud fallback: {error.code}", error)
            return getattr(self.cloud, method)(*args, **kwargs)

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self._route("embedding", "embed", texts, **kwargs)

    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        return self._route("stt", "transcribe", audio_path, **kwargs)

    def synthesize(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return self._route("tts", "synthesize", text, **kwargs)

    def vision(self, image_path: str, prompt: str = "", **kwargs: Any) -> str:
        return self._route("vision", "vision", image_path, prompt, **kwargs)

    def generate_image(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self._note("local", "image generation is local-only")
        return self.local.generate_image(prompt, **kwargs)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self.local.execute_tool(name, arguments)

    def cancel(self) -> None:
        target = self.cloud if self.last_route == "cloud" else self.local
        cancel = getattr(target, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001 - cancellation is best effort
                pass

    def health(self) -> dict[str, Any]:
        local = self.local.health() if hasattr(self.local, "health") else {}
        return {"provider_id": self.id, "local": local, "cloud": self.cloud.health(),
                "route": self.last_route, "reason": self.last_reason, "local_only": self.local_only}

    def capabilities(self) -> ProviderCapabilities:
        local = self.local.capabilities()
        return ProviderCapabilities(chat=True, streaming=True, vision=local.vision, audio_input=local.audio_input,
                                    audio_output=local.audio_output, tools=local.tools,
                                    embedding=local.embedding, image_generation=local.image_generation,
                                    speculative_decoding=local.speculative_decoding, is_local=False,
                                    models=local.models)


def hybrid_from_settings(settings: Any, local: Any, *, consent: Callable[[str, str], bool] | None = None,
                         cloud: CloudProvider | None = None) -> HybridProvider:
    policy = {
        "chat": str(settings.get("providers.chat", "local")),
        "vision": str(settings.get("providers.vision", "local")),
        "embedding": str(settings.get("providers.embedding", "local")),
        "tts": str(settings.get("providers.tts", "local")),
        "stt": str(settings.get("providers.stt", "local")),
        "image_generation": str(settings.get("providers.image_generation", "local")),
        "fallback": "none" if settings.get("privacy.local_only", False) else
                    ("cloud" if settings.get("cloud_fallback.enabled", False) else "none"),
    }
    return HybridProvider(local, cloud, policy=policy, local_only=bool(settings.get("privacy.local_only", False)),
                          consent=consent)