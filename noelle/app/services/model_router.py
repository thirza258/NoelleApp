"""Heuristic virtual-model resolver with fallback chaining.

Resolves ``auto`` / ``frontier-fast`` / ``frontier-quality`` to a concrete
(provider, model) pair in pure Python (well under the 2 ms budget). Fallback order:
primary cloud model -> alternate cloud model -> local model (only when the local
runtime is ready). No network calls; availability is injected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

FAST_MODELS = ("anthropic/claude-3.5-haiku", "google/gemini-2.0-flash", "gpt-4o-mini")
REASONING_MODELS = ("deepseek/deepseek-r1", "o3-mini", "anthropic/claude-3.5-sonnet")
QUALITY_MODELS = ("anthropic/claude-3.5-sonnet", "gpt-4o")
LONG_CONTEXT_MODELS = ("google/gemini-pro-1.5", "anthropic/claude-3.5-sonnet")
CODE_MODELS = ("qwen/qwen-2.5-coder-32b-instruct", "deepseek/deepseek-chat", "anthropic/claude-3.5-sonnet")
CHEAP_MODELS = ("deepseek/deepseek-chat", "gpt-4o-mini", "google/gemini-2.0-flash")
MULTIMODAL_MODELS = ("anthropic/claude-3.5-sonnet", "gpt-4o")

_CODE_RE = re.compile(r"\b(code|python|javascript|typescript|sql|function|debug|traceback|refactor|regex|api)\b",
                      re.IGNORECASE)
_MATH_RE = re.compile(r"\b(solve|prove|derive|integral|equation|calculate|math|reason|logic|step by step)\b",
                      re.IGNORECASE)
_REASONING_PREFIXES = ("o1", "o3", "deepseek/deepseek-r1")

Message = Mapping[str, Any]


@dataclass(frozen=True)
class Route:
    provider_id: str
    model_id: str
    alias: str
    reason: str
    fallbacks: tuple[tuple[str, str], ...] = ()  # (provider_id, model_id)

    def chain(self) -> list[tuple[str, str]]:
        return [(self.provider_id, self.model_id), *self.fallbacks]

    def to_dict(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "model_id": self.model_id, "alias": self.alias,
                "reason": self.reason, "fallbacks": list(self.fallbacks)}


def _text(messages: Iterable[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return "\n".join(parts)


def _has_image(messages: Iterable[Message]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list) and any(isinstance(part, Mapping) and part.get("type") in
                                              ("image_url", "image", "input_image") for part in content):
            return True
    return False


def estimate_tokens(messages: Iterable[Message]) -> int:
    """Cheap token estimate (~4 chars/token); the router only needs a magnitude."""
    return max(len(_text(messages)) // 4, 0)


def _provider_for(model_id: str) -> str:
    return "openai" if model_id.startswith(("gpt-", "o1", "o3", "text-embedding", "whisper", "tts-")) else "openrouter"


class ModelRouter:
    def __init__(self, available_providers: Iterable[str] | None = None,
                 local_model: str | None = None, local_ready: bool = False) -> None:
        self.available = (set(available_providers) if available_providers is not None
                          else {"openai", "openrouter"})
        self.local_model = local_model
        self.local_ready = local_ready

    def _usable(self, model_id: str) -> bool:
        if self.local_model and model_id == self.local_model:
            return self.local_ready
        return _provider_for(model_id) in self.available

    def _first(self, candidates: Iterable[str]) -> str | None:
        return next((model for model in candidates if self._usable(model)), None)

    def resolve(self, model: str | None, messages: Iterable[Message] = (), *,
                modality: str = "llm") -> Route:
        messages = list(messages)
        alias = (model or "auto").strip()
        text = _text(messages)
        tokens = estimate_tokens(messages)
        image = _has_image(messages)

        if alias == "auto":
            if modality in ("embedding", "stt", "tts"):
                chosen = {"embedding": "text-embedding-3-large", "stt": "whisper-1", "tts": "tts-1-hd"}[modality]
                return self._route(chosen, alias, f"{modality} requests use OpenAI by default", local=False)
            if image:
                chosen = self._first(MULTIMODAL_MODELS) or "gpt-4o"
                return self._route(chosen, alias, "image input needs a multimodal model")
            if tokens > 64_000:
                chosen = self._first(LONG_CONTEXT_MODELS) or "gpt-4o"
                return self._route(chosen, alias, "long context (> 64k tokens)")
            if _MATH_RE.search(text):
                chosen = self._first(REASONING_MODELS) or "o3-mini"
                return self._route(chosen, alias, "reasoning/maths intent")
            if _CODE_RE.search(text):
                chosen = self._first(CODE_MODELS) or "deepseek/deepseek-chat"
                return self._route(chosen, alias, "code intent")
            chosen = self._first(CHEAP_MODELS) or "gpt-4o-mini"
            return self._route(chosen, alias, "general chat routes to a cost-efficient model")
        if alias == "frontier-fast":
            chosen = self._first(FAST_MODELS) or "gpt-4o-mini"
            return self._route(chosen, alias, "lowest latency tier")
        if alias == "frontier-quality":
            chosen = self._first(QUALITY_MODELS) or "gpt-4o"
            return self._route(chosen, alias, "maximum capability tier")
        provider = _provider_for(alias)
        return self._route(alias, alias, "explicit model", provider=provider)

    def _route(self, model_id: str, alias: str, reason: str, *, provider: str | None = None,
               local: bool = False) -> Route:
        provider = provider or _provider_for(model_id)
        fallbacks: list[tuple[str, str]] = []
        # 1) alternate cloud model, 2) the other provider, 3) local when ready
        for candidate in (*QUALITY_MODELS, *CHEAP_MODELS):
            if candidate != model_id and self._usable(candidate):
                pair = (_provider_for(candidate), candidate)
                if pair not in fallbacks:
                    fallbacks.append(pair)
                break
        other = "openai" if provider != "openai" else "openrouter"
        if other in self.available:
            fallback_model = "gpt-4o-mini" if other == "openai" else "deepseek/deepseek-chat"
            fallbacks.append((other, fallback_model))
        if self.local_ready and self.local_model:
            fallbacks.append(("local", self.local_model))
        return Route(provider, model_id, alias, reason, tuple(fallbacks))

    def next_fallback(self, route: Route, failed: tuple[str, str]) -> tuple[str, str] | None:
        chain = route.chain()
        try:
            index = chain.index(failed)
        except ValueError:
            return chain[0] if chain else None
        return chain[index + 1] if index + 1 < len(chain) else None