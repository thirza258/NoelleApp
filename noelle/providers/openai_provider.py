"""OpenAI API Provider Adapter.

Supports:
- LLM: Chat completions (gpt-4o, gpt-4o-mini, o1, o3-mini)
- Embeddings: text-embedding-3-small, text-embedding-3-large
- STT: whisper-1 audio transcription
- TTS: tts-1, tts-1-hd text-to-speech synthesis
- Vision: Multimodal image comprehension
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
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

log = logging.getLogger("noelle.providers.openai")

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class OpenAIProvider(AIProvider, AsyncProviderMixin):
    id: str = "openai"
    name: str = "OpenAI Direct API"
    is_local: bool = False

    def __init__(self, api_key: str | None = None, base_url: str = OPENAI_BASE_URL,
                 client: httpx.Client | None = None) -> None:
        self._explicit_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client

    @property
    def api_key(self) -> str | None:
        return self._explicit_key or get_api_key("openai")

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.id,
            display_name=self.name,
            is_local=False,
            modalities={Modality.LLM.value, Modality.EMBEDDING.value, Modality.STT.value,
                        Modality.TTS.value, Modality.VISION.value},
            supported_models=[
                "gpt-4o",
                "gpt-4o-mini",
                "o3-mini",
                "o1-mini",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "whisper-1",
                "tts-1",
                "tts-1-hd",
            ],
            supports_streaming=True,
            supports_json_schema=True,
            max_context_window=128_000,
        )

    def estimate_cost(self, request: NodeRequest) -> float:
        model = request.model or "gpt-4o-mini"
        prompt_tokens = len(request.prompt.split()) * 2
        max_tokens = request.max_tokens or 512
        text_chars = len(request.prompt)
        return calculate_cost(
            model,
            input_tokens=prompt_tokens,
            output_tokens=max_tokens // 2,
            text_chars=text_chars if request.modality_str == Modality.TTS.value else 0,
        )

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
                error_message="Missing OpenAI API Key. Set it in Settings › AI Providers or via OPENAI_API_KEY.",
            )

        modality = request.modality_str
        start_time = time.perf_counter()
        try:
            if modality == Modality.LLM.value or modality == Modality.VISION.value:
                return self._execute_chat(request, key, start_time)
            elif modality == Modality.EMBEDDING.value:
                return self._execute_embedding(request, key, start_time)
            elif modality == Modality.STT.value:
                return self._execute_stt(request, key, start_time)
            elif modality == Modality.TTS.value:
                return self._execute_tts(request, key, start_time)
            else:
                return NodeResponse(
                    output="",
                    telemetry=TelemetryRecord(latency_ms=0.0),
                    provider_id=self.id,
                    model_id=request.model,
                    is_error=True,
                    error_message=f"Unsupported modality '{modality}' for OpenAI provider",
                )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start_time) * 1000.0
            log.exception("OpenAI execution failure: %s", exc)
            return NodeResponse(
                output="",
                telemetry=TelemetryRecord(latency_ms=latency),
                provider_id=self.id,
                model_id=request.model,
                is_error=True,
                error_message=str(exc),
            )

    def _execute_chat(self, req: NodeRequest, key: str, start: float) -> NodeResponse:
        model = req.model or "gpt-4o-mini"
        messages: list[dict[str, Any]] = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})

        user_content: Any = req.prompt
        if req.image_bytes or req.image_path:
            # Multimodal Vision Payload
            b64_data = ""
            mime_type = "image/jpeg"
            if req.image_bytes:
                b64_data = base64.b64encode(req.image_bytes).decode("ascii")
            elif req.image_path:
                with open(req.image_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("ascii")
                if req.image_path.lower().endswith(".png"):
                    mime_type = "image/png"
            user_content = [
                {"type": "text", "text": req.prompt or "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
            ]

        messages.append({"role": "user", "content": user_content})

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

        resp = self._request("POST", f"{self.base_url}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        latency_ms = (time.perf_counter() - start) * 1000.0
        output_text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

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

    def _execute_embedding(self, req: NodeRequest, key: str, start: float) -> NodeResponse:
        model = req.model or "text-embedding-3-small"
        texts = req.texts if req.texts else ([req.prompt] if req.prompt else [""])
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "input": texts if len(texts) > 1 else texts[0],
        }

        resp = self._request("POST", f"{self.base_url}/embeddings", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        latency_ms = (time.perf_counter() - start) * 1000.0
        embeddings = [item["embedding"] for item in data["data"]]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        cost = calculate_cost(model, input_tokens=prompt_tokens)

        return NodeResponse(
            output=embeddings if len(embeddings) > 1 else embeddings[0],
            telemetry=TelemetryRecord(
                latency_ms=latency_ms,
                cost_usd=cost,
                input_tokens=prompt_tokens,
            ),
            provider_id=self.id,
            model_id=model,
            raw_payload=data,
        )

    def _execute_stt(self, req: NodeRequest, key: str, start: float) -> NodeResponse:
        model = req.model or "whisper-1"
        headers = {"Authorization": f"Bearer {key}"}

        audio_content = req.audio_bytes
        filename = "audio.wav"
        if not audio_content and req.audio_path:
            with open(req.audio_path, "rb") as f:
                audio_content = f.read()
            filename = Path(req.audio_path).name

        if not audio_content:
            return NodeResponse(
                output="",
                telemetry=TelemetryRecord(latency_ms=0.0),
                provider_id=self.id,
                model_id=model,
                is_error=True,
                error_message="No audio input provided for STT transcription",
            )

        files = {"file": (filename, audio_content, "audio/wav")}
        data_fields = {"model": model}
        if req.language:
            data_fields["language"] = req.language

        resp = self._request("POST", f"{self.base_url}/audio/transcriptions", headers=headers,
                             files=files, data=data_fields)
        resp.raise_for_status()
        data = resp.json()

        latency_ms = (time.perf_counter() - start) * 1000.0
        transcript = data.get("text", "")
        # Estimate duration roughly from WAV or default
        duration_s = max(1.0, len(audio_content) / 32000.0)
        cost = calculate_cost(model, audio_duration_seconds=duration_s)
        rtf = (latency_ms / 1000.0) / duration_s

        return NodeResponse(
            output=transcript,
            telemetry=TelemetryRecord(
                latency_ms=latency_ms,
                cost_usd=cost,
                audio_duration_s=duration_s,
                real_time_factor=rtf,
            ),
            provider_id=self.id,
            model_id=model,
            raw_payload=data,
        )

    def _execute_tts(self, req: NodeRequest, key: str, start: float) -> NodeResponse:
        model = req.model or "tts-1"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "input": req.prompt,
            "voice": req.voice or "alloy",
            "response_format": "mp3",
        }

        resp = self._request("POST", f"{self.base_url}/audio/speech", headers=headers, json=body)
        resp.raise_for_status()
        audio_bytes = resp.content

        latency_ms = (time.perf_counter() - start) * 1000.0
        text_chars = len(req.prompt)
        cost = calculate_cost(model, text_chars=text_chars)
        # Approximate audio duration from mp3 size (128kbps = 16KB/s)
        duration_s = max(0.5, len(audio_bytes) / 16000.0)
        rtf = (latency_ms / 1000.0) / duration_s

        return NodeResponse(
            output=audio_bytes,
            telemetry=TelemetryRecord(
                latency_ms=latency_ms,
                cost_usd=cost,
                audio_duration_s=duration_s,
                real_time_factor=rtf,
            ),
            provider_id=self.id,
            model_id=model,
        )

    # ------------------------------------------------ strict provider contract
    def _require_key(self) -> str:
        key = self.api_key
        if not key:
            raise ProviderError("MISSING_API_KEY", "Missing OpenAI API key.",
                                category="auth", recoverable=False,
                                suggested_actions=("configure_api_key",), provider_id=self.id)
        return key

    @staticmethod
    def _raise_for_http(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        code = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 429: "RATE_LIMITED"}.get(response.status_code, "CLOUD_ERROR")
        raise ProviderError(code, f"OpenAI returned HTTP {response.status_code}: {response.text[:300]}",
                            category="cloud", recoverable=response.status_code in (429, 500, 502, 503, 504),
                            suggested_actions=("retry", "enable_cloud_fallback"), provider_id="openai")

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, **params: Any) -> str:
        key = self._require_key()
        body = {"model": model or "gpt-4o-mini", "messages": messages,
                "temperature": float(params.get("temperature", 0.7)),
                "max_tokens": int(params.get("max_tokens", 1024)), "stream": False}
        if params.get("tools"):
            body["tools"] = params["tools"]
        response = self._request("POST", f"{self.base_url}/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body)
        self._raise_for_http(response)
        data = response.json()
        return str(data["choices"][0]["message"].get("content") or "")

    def stream_chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
                    **params: Any) -> Iterator[str]:
        key = self._require_key()
        body = {"model": model or "gpt-4o-mini", "messages": messages,
                "temperature": float(params.get("temperature", 0.7)),
                "max_tokens": int(params.get("max_tokens", 1024)), "stream": True,
                "stream_options": {"include_usage": True}}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        timeout = params.get("timeout") or DEFAULT_TIMEOUT
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_http(response)
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
        response = self.execute(NodeRequest(modality=Modality.EMBEDDING, texts=list(texts),
                                            model=model or "text-embedding-3-small"))
        self._raise_node(response)
        embeddings = response.output
        return embeddings if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list) \
            else [embeddings]

    def transcribe(self, audio_path: str, *, model: str | None = None, language: str | None = None,
                   **params: Any) -> dict[str, Any]:
        response = self.execute(NodeRequest(modality=Modality.STT, audio_path=audio_path,
                                            model=model or "whisper-1", language=language or "id"))
        self._raise_node(response)
        return {"text": str(response.output), "metrics": response.telemetry.to_dict()}

    def synthesize(self, text: str, *, model: str | None = None, voice: int | str = "alloy",
                   **params: Any) -> dict[str, Any]:
        response = self.execute(NodeRequest(modality=Modality.TTS, prompt=text,
                                            model=model or "tts-1", voice=str(voice)))
        self._raise_node(response)
        return {"audio": response.output, "metrics": response.telemetry.to_dict()}

    def generate_image(self, prompt: str, *, model: str | None = None, **params: Any) -> dict[str, Any]:
        raise ProviderError("UNSUPPORTED_TASK", "OpenAI image generation is not wired in this release.",
                            category="request", suggested_actions=("use_local_diffusion",))

    def vision(self, image_path: str, prompt: str = "", *, model: str | None = None,
               **params: Any) -> str:
        response = self.execute(NodeRequest(modality=Modality.VISION, image_path=image_path, prompt=prompt,
                                            model=model or "gpt-4o-mini"))
        self._raise_node(response)
        return str(response.output)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from noelle.runtime.tools import execute_tool

        return execute_tool(name, arguments).to_dict()

    def health(self) -> dict[str, Any]:
        return {"provider_id": self.id, "configured": bool(self.api_key), "reachable": bool(self.api_key),
                "is_local": False}

    def capabilities(self) -> StrictCapabilities:
        caps = self.get_capabilities()
        return StrictCapabilities(chat=True, streaming=True, vision=caps.supports_modality(Modality.VISION),
                                  audio_input=caps.supports_modality(Modality.STT),
                                  audio_output=caps.supports_modality(Modality.TTS), tools=True,
                                  embedding=caps.supports_modality(Modality.EMBEDDING),
                                  image_generation=False, is_local=False, models=tuple(caps.supported_models))

    @staticmethod
    def _raise_node(response: NodeResponse) -> None:
        if response.is_error:
            raise ProviderError("CLOUD_ERROR", response.error_message or "OpenAI request failed.",
                                category="cloud", suggested_actions=("retry", "enable_cloud_fallback"))
