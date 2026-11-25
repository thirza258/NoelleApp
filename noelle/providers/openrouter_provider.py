"""OpenRouter API Gateway Provider Adapter.

Provides access to 100+ frontier and open-weight models (Claude 3.5, Gemini 2.0,
Qwen 2.5, Llama 3.3, DeepSeek R1) over a normalized API with live cost tracking.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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

log = logging.getLogger("noelle.providers.openrouter")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=10.0)


class OpenRouterProvider(AIProvider):
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
        messages.append({"role": "user", "content": request.prompt})

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
