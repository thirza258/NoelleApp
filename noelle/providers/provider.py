"""Strict provider contract shared by local, cloud and hybrid execution.

Blocking methods are the primitive (the Qt shell runs them on a task thread); ``a*``
variants are convenience wrappers for async callers such as the API server.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Mapping, Protocol, runtime_checkable


@dataclass
class ProviderErrorInfo:
    code: str
    message: str
    category: str = "runtime"
    recoverable: bool = True
    suggested_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "category": self.category, "message": self.message,
                "recoverable": self.recoverable, "suggested_actions": list(self.suggested_actions)}


class ProviderError(RuntimeError):
    """Structured, user-actionable provider failure."""

    def __init__(self, code: str, message: str, *, category: str = "runtime", recoverable: bool = True,
                 suggested_actions: tuple[str, ...] | list[str] = (), detail: str | None = None,
                 provider_id: str | None = None, model_id: str | None = None) -> None:
        super().__init__(message)
        self.info = ProviderErrorInfo(code, message, category, recoverable, tuple(suggested_actions))
        self.detail = detail
        self.provider_id = provider_id
        self.model_id = model_id

    @property
    def code(self) -> str:
        return self.info.code

    def to_dict(self) -> dict[str, Any]:
        data = self.info.to_dict()
        data.update({"detail": self.detail, "provider_id": self.provider_id, "model_id": self.model_id})
        return data

    @classmethod
    def from_worker_payload(cls, payload: Mapping[str, Any], *, provider_id: str | None = None) -> ProviderError:
        code = str(payload.get("code") or "RUNTIME_ERROR")
        actions: tuple[str, ...] = ()
        if code == "OOM":
            actions = ("use_quantized_model", "reduce_context", "enable_cloud_fallback")
        elif code in ("MODEL_NOT_FOUND", "DEPENDENCY_MISSING"):
            actions = ("download_or_install", "change_model", "enable_cloud_fallback")
        elif code in ("UNSUPPORTED", "NEEDS_NEWER_RUNTIME", "UNSUPPORTED_LIBRARY", "REMOTE_CODE_BLOCKED"):
            actions = ("change_model", "fix_runtime_environment", "enable_cloud_fallback")
        elif code == "GATED":
            actions = ("open_model_page", "enable_cloud_fallback")
        return cls(code, str(payload.get("message") or "The runtime request failed."),
                   recoverable=code not in ("CANCELLED", "UNSUPPORTED", "GATED"),
                   suggested_actions=actions, detail=payload.get("detail"), provider_id=provider_id,
                   model_id=payload.get("model"))


@dataclass
class ProviderCapabilities:
    chat: bool = True
    streaming: bool = True
    vision: bool = False
    audio_input: bool = False
    audio_output: bool = False
    tools: bool = False
    embedding: bool = False
    image_generation: bool = False
    speculative_decoding: bool = False
    is_local: bool = False
    models: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chat": self.chat, "streaming": self.streaming, "vision": self.vision,
                "audio_input": self.audio_input, "audio_output": self.audio_output, "tools": self.tools,
                "embedding": self.embedding, "image_generation": self.image_generation,
                "speculative_decoding": self.speculative_decoding, "is_local": self.is_local,
                "models": list(self.models), **self.extra}

    def supports(self, feature: str) -> bool:
        return bool(getattr(self, feature, False))


@runtime_checkable
class Provider(Protocol):
    id: str
    name: str
    is_local: bool

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, **params: Any) -> str: ...

    def stream_chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
                    **params: Any) -> Iterator[str]: ...

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...

    def transcribe(self, audio_path: str, *, model: str | None = None, language: str | None = None,
                   **params: Any) -> dict[str, Any]: ...

    def synthesize(self, text: str, *, model: str | None = None, voice: int | str = 0,
                   **params: Any) -> dict[str, Any]: ...

    def generate_image(self, prompt: str, *, model: str | None = None, **params: Any) -> dict[str, Any]: ...

    def vision(self, image_path: str, prompt: str = "", *, model: str | None = None,
               **params: Any) -> str: ...

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    def health(self) -> dict[str, Any]: ...

    def capabilities(self) -> ProviderCapabilities: ...


class AsyncProviderMixin:
    """Async wrappers; callers that need true streaming should use the sync iterator in a thread."""

    async def achat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return await asyncio.to_thread(self.chat, messages, **kwargs)  # type: ignore[attr-defined]

    async def aembed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, texts, **kwargs)  # type: ignore[attr-defined]

    async def atranscribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.transcribe, audio_path, **kwargs)  # type: ignore[attr-defined]

    async def asynthesize(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.synthesize, text, **kwargs)  # type: ignore[attr-defined]

    async def agenerate_image(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.generate_image, prompt, **kwargs)  # type: ignore[attr-defined]

    async def avision(self, image_path: str, prompt: str = "", **kwargs: Any) -> str:
        return await asyncio.to_thread(self.vision, image_path, prompt, **kwargs)  # type: ignore[attr-defined]

    async def ahealth(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.health)  # type: ignore[attr-defined]

    async def acapabilities(self) -> ProviderCapabilities:
        return await asyncio.to_thread(self.capabilities)  # type: ignore[attr-defined]

    async def astream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        iterator = await asyncio.to_thread(lambda: iter(self.stream_chat(messages, **kwargs)))  # type: ignore[attr-defined]
        loop = asyncio.get_running_loop()
        sentinel = object()

        def take() -> Any:
            try:
                return next(iterator)
            except StopIteration:
                return sentinel

        while True:
            text = await loop.run_in_executor(None, take)
            if text is sentinel:
                return
            yield text