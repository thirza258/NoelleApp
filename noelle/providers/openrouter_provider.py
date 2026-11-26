"""OpenRouter API Gateway Provider Adapter.

Provides access to 100+ frontier and open-weight models (Claude 3.5, Gemini 2.0,
Qwen 2.5, Llama 3.3, DeepSeek R1) over a normalized API with live cost tracking.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Iterator

import httpx

from noelle.providers.base import (
    AIProvider,
    Modality,
    NodeRequest,
    NodeResponse,
    ProviderCapabilities,
    TelemetryRecord,
)
from noelle.providers.credentials import get_api_key
from noelle.providers.pricing import calculate_cost
from noelle.providers.provider import AsyncProviderMixin, ProviderCapabilities as StrictCapabilities, ProviderError

log = logging.getLogger("noelle.providers.openrouter")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=10.0)


class OpenRouterProvider(AIProvider, AsyncProviderMixin):
    id: str = "openrouter"
    name: str = "OpenRouter Gateway"
    is_local: bool = False

    def __init__(self, api_key: str | None = None, base_url: str = OPENROUTER_BASE_URL,
                 client: httpx.Client | None = None) -> None:
        self._explicit_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._cached_models: list[str] = []

    @property
    def api_key(self) -> str | None:
        return self._explicit_key or get_api_key("openrouter")

    def get_capabilities(self) -> ProviderCapabilities:
        models = self._cached_models or [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "google/gemini-2.0-flash",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwen-2.5-72b-instruct",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-r1",
        ]
        return ProviderCapabilities(
            provider_id=self.id,
            display_name=self.name,
            is_local=False,
            modalities={Modality.LLM.value, Modality.VISION.value},
            supported_models=models,
            supports_streaming=True,
            supports_json_schema=True,
            max_context_window=200_000,
        )

    def fetch_live_models(self) -> list[str]:
        """Query OpenRouter for all live models available."""
        try:
            resp = self._request("GET", f"{self.base_url}/models")
            if resp.is_success:
                    data = resp.json()
                    self._cached_models = [m["id"] for m in data.get("data", [])]
                    return self._cached_models
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch OpenRouter models: %s", exc)
        return self.get_capabilities().supported_models

    def estimate_cost(self, request: NodeRequest) -> float:
        model = request.model or "qwen/qwen-2.5-72b-instruct"
        prompt_tokens = len(request.prompt.split()) * 2
        max_tokens = request.max_tokens or 512
        return calculate_cost(model, input_tokens=prompt_tokens, output_tokens=max_tokens // 2)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            return self._client.request(method, url, **kwargs)
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            return client.request(method, url, **kwargs)

    def execute(self, request: NodeRequest) -> NodeResponse:
        key = self.api_key
        if not key:
            return NodeResponse(
                output="",
                telemetry=TelemetryRecord(latency_ms=0.0),
                provider_id=self.id,
                model_id=request.model,
                is_error=True,
                error_message="Missing OpenRouter API Key. Configure it in Settings › AI Providers or set OPENROUTER_API_KEY.",
            )

        model = request.model or "qwen/qwen-2.5-72b-instruct"
        start_time = time.perf_counter()
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        content: Any = request.prompt
        if request.image_path or request.image_bytes:
            data = request.image_bytes
            mime = "image/jpeg"
            if data is None and request.image_path:
                with open(request.image_path, "rb") as handle:
                    data = handle.read()
                mime = "image/png" if request.image_path.lower().endswith(".png") else "image/jpeg"
            content = [
                {"type": "text", "text": request.prompt or "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,"
                                                              f"{base64.b64encode(data or b'').decode('ascii')}"}},
            ]
        messages.append({"role": "user", "content": content})

        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://noelle.nevatal.id",
            "X-Title": "Noelle AI Studio",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        try:
            resp = self._request("POST", f"{self.base_url}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            output_text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Check if OpenRouter provided explicit cost in response
            cost = 0.0
            if "total_cost" in data:
                cost = float(data["total_cost"])
            else:
                cost = calculate_cost(model, input_tokens=prompt_tokens, output_tokens=completion_tokens)

            tps = (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

            return NodeResponse(
                output=output_text,
                telemetry=TelemetryRecord(
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    tokens_per_second=tps,
                ),
                provider_id=self.id,
                model_id=model,
                raw_payload=data,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            log.exception("OpenRouter request failed: %s", exc)
            return NodeResponse(
                output="",
                telemetry=TelemetryRecord(latency_ms=latency_ms),
                provider_id=self.id,
                model_id=model,
                is_error=True,
                error_message=str(exc),
            )

    # ------------------------------------------------ strict provider contract
    def _require_key(self) -> str:
        key = self.api_key
        if not key:
            raise ProviderError("MISSING_API_KEY", "Missing OpenRouter API key.", category="auth",
                                recoverable=False, suggested_actions=("configure_api_key",),
                                provider_id=self.id)
        return key

    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://noelle.nevatal.id",
                "X-Title": "Noelle AI Studio", "Content-Type": "application/json"}

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, **params: Any) -> str:
        key = self._require_key()
        body = {"model": model or "deepseek/deepseek-chat", "messages": messages,
                "temperature": float(params.get("temperature", 0.7)),
                "max_tokens": int(params.get("max_tokens", 1024)), "stream": False}
        response = self._request("POST", f"{self.base_url}/chat/completions", headers=self._headers(key), json=body)
        if response.status_code >= 400:
            code = {401: "UNAUTHORIZED", 429: "RATE_LIMITED"}.get(response.status_code, "CLOUD_ERROR")
            raise ProviderError(code, f"OpenRouter returned HTTP {response.status_code}: {response.text[:300]}",
                                category="cloud", recoverable=response.status_code in (429, 500, 502, 503, 504),
                                suggested_actions=("retry", "enable_cloud_fallback"), provider_id=self.id)
        data = response.json()
        return str(data["choices"][0]["message"].get("content") or "")

    def stream_chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
                    **params: Any) -> Iterator[str]:
        key = self._require_key()
        body = {"model": model or "deepseek/deepseek-chat", "messages": messages,
                "temperature": float(params.get("temperature", 0.7)),
                "max_tokens": int(params.get("max_tokens", 1024)), "stream": True}
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            with client.stream("POST", f"{self.base_url}/chat/completions", headers=self._headers(key),
                               json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ProviderError("CLOUD_ERROR",
                                        f"OpenRouter returned HTTP {response.status_code}: {response.text[:300]}",
                                        category="cloud", suggested_actions=("retry", "enable_cloud_fallback"),
                                        provider_id=self.id)
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except ValueError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        text = (choices[0].get("delta") or {}).get("content")
                        if text:
                            yield str(text)

    def embed(self, texts: list[str], *, model: str | None = None, **params: Any) -> list[list[float]]:
        raise ProviderError("UNSUPPORTED_TASK", "OpenRouter does not expose embeddings here; use OpenAI or a "
                            "local embedding model.", category="request",
                            suggested_actions=("use_openai", "use_local_embedding"))

    def transcribe(self, audio_path: str, *, model: str | None = None, language: str | None = None,
                   **params: Any) -> dict[str, Any]:
        raise ProviderError("UNSUPPORTED_TASK", "OpenRouter does not provide speech recognition.",
                            category="request", suggested_actions=("use_openai", "use_local_asr"))

    def synthesize(self, text: str, *, model: str | None = None, voice: int | str = 0,
                   **params: Any) -> dict[str, Any]:
        raise ProviderError("UNSUPPORTED_TASK", "OpenRouter does not provide speech synthesis.",
                            category="request", suggested_actions=("use_openai", "use_local_tts"))

    def generate_image(self, prompt: str, *, model: str | None = None, **params: Any) -> dict[str, Any]:
        raise ProviderError("UNSUPPORTED_TASK", "Image generation is not offered by this provider.",
                            category="request", suggested_actions=("use_local_diffusion",))

    def vision(self, image_path: str, prompt: str = "", *, model: str | None = None,
               **params: Any) -> str:
        request = NodeRequest(modality=Modality.VISION, image_path=image_path, prompt=prompt,
                              model=model or "anthropic/claude-3.5-sonnet")
        response = self.execute(request)
        if response.is_error:
            raise ProviderError("CLOUD_ERROR", response.error_message or "OpenRouter vision failed.",
                                category="cloud", suggested_actions=("retry", "enable_cloud_fallback"))
        return str(response.output)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from noelle.runtime.tools import execute_tool

        return execute_tool(name, arguments).to_dict()

    def health(self) -> dict[str, Any]:
        return {"provider_id": self.id, "configured": bool(self.api_key), "reachable": bool(self.api_key),
                "is_local": False}

    def capabilities(self) -> StrictCapabilities:
        return StrictCapabilities(chat=True, streaming=True, vision=True, audio_input=False, audio_output=False,
                                  tools=True, embedding=False, image_generation=False, is_local=False,
                                  models=tuple(self.get_capabilities().supported_models))
