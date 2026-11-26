"""Local provider backed by the real out-of-process Noelle worker (ZeroMQ IPC).

No simulation: every call goes through ``WorkerClient`` to the worker that owns the
model runtime. When the worker is unavailable the provider raises a structured
:class:`ProviderError`, so the UI can offer "start runtime" or cloud fallback.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from noelle.providers.base import (
    AIProvider,
    Modality,
    NodeRequest,
    NodeResponse,
    ProviderCapabilities as LegacyCapabilities,
    TelemetryRecord,
)
from noelle.providers.provider import AsyncProviderMixin, ProviderCapabilities, ProviderError
from noelle.runtime.client import WorkerClient, WorkerError, WorkerProcess, read_discovery_file, spawn_worker
from noelle.runtime.protocol import Command, Event

log = logging.getLogger("noelle.providers.local")

_STATE_IDLE = "IDLE"
_STATE_STARTING = "STARTING"
_STATE_LOADING = "LOADING_MODEL"
_STATE_READY = "READY"
_STATE_GENERATING = "GENERATING"
_STATE_FAILED = "LOADING_FAILED"
_STATE_ERROR = "RUNTIME_ERROR"
_STATE_CRASHED = "WORKER_CRASHED"
_STATE_TIMEOUT = "TIMEOUT"
_STATE_CANCELLED = "CANCELLED"


class LocalProvider(AIProvider, AsyncProviderMixin):
    id: str = "local"
    name: str = "Local On-Device Engine"
    is_local: bool = True

    def __init__(self, runtime_client: Any | None = None, *, endpoint: str | None = None,
                 token: str | None = None, auto_spawn: bool = False, request_timeout: float = 900.0,
                 default_model: str | None = None) -> None:
        self.runtime_client = runtime_client
        self._endpoint = endpoint
        self._token = token
        self._auto_spawn = auto_spawn
        self._timeout = request_timeout
        self._default_model = default_model
        self._owned: WorkerProcess | None = None
        self._lock = threading.RLock()
        self.state = _STATE_IDLE
        self.last_error: ProviderError | None = None
        self.last_route = "local"
        self.last_request_id: str | None = None
        self.last_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------ connection
    def _connection(self) -> tuple[str, str | None]:
        if self.runtime_client is not None:
            state = getattr(self.runtime_client, "state", "ready")
            endpoint = getattr(self.runtime_client, "endpoint", None)
            token = getattr(self.runtime_client, "token", None)
            if state in ("crashed", "unresponsive"):
                raise ProviderError("WORKER_CRASHED", "The model runtime is not running.",
                                    suggested_actions=("restart_runtime", "enable_cloud_fallback"))
            if not endpoint:
                raise ProviderError("WORKER_NOT_READY", "The model runtime is still starting.",
                                    recoverable=True, suggested_actions=("wait", "restart_runtime"))
            return str(endpoint), token
        if self._endpoint:
            return self._endpoint, self._token
        if self._owned is not None:
            return self._owned.endpoint, self._owned.token
        if self._auto_spawn:
            with self._lock:
                if self._owned is None:
                    self.state = _STATE_STARTING
                    self._owned = spawn_worker(parent_pid=os.getpid())
                    self._endpoint = self._owned.endpoint
                    self._token = self._owned.token
            return self._owned.endpoint, self._owned.token
        discovered = read_discovery_file()
        if discovered and discovered.get("port"):
            token = os.environ.get("NOELLE_WORKER_TOKEN")
            if token:
                return f"tcp://127.0.0.1:{discovered['port']}", token
        raise ProviderError(
            "WORKER_UNAVAILABLE", "The model runtime is not connected.",
            category="runtime", recoverable=True,
            suggested_actions=("start_runtime", "enable_cloud_fallback", "check_runtime_settings"))

    def _client(self) -> WorkerClient:
        endpoint, token = self._connection()
        return WorkerClient(endpoint, token)

    def _call(self, cmd: Command, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        try:
            self.state = _STATE_READY
            return self._client().call(cmd, payload, timeout=timeout or self._timeout)
        except TimeoutError as exc:
            self.state = _STATE_TIMEOUT
            raise ProviderError("TIMEOUT", f"The runtime did not answer within {timeout or self._timeout:.0f}s.",
                                recoverable=True, suggested_actions=("retry", "restart_runtime")) from exc
        except WorkerError as exc:
            payload_error = exc.payload if isinstance(exc.payload, Mapping) else {}
            error = ProviderError.from_worker_payload(payload_error, provider_id=self.id)
            self.last_error = error
            self.state = _STATE_LOADING if error.code in ("LOAD_FAILED", "NEEDS_NEWER_RUNTIME") else _STATE_ERROR
            raise error from exc
        except (ConnectionError, OSError) as exc:
            self.state = _STATE_CRASHED
            raise ProviderError("WORKER_CRASHED", "Lost the connection to the model runtime.",
                                suggested_actions=("restart_runtime", "enable_cloud_fallback")) from exc

    def _payload(self, model: str | None, task: str, **extra: Any) -> dict[str, Any]:
        model_id = model or self._default_model
        if not model_id:
            raise ProviderError("MODEL_REQUIRED", "Choose a model before sending a request.", category="request",
                                recoverable=True, suggested_actions=("select_model",))
        payload: dict[str, Any] = {"model_id": model_id, "task": task}
        payload.update({key: value for key, value in extra.items() if value is not None})
        return payload

    # ---------------------------------------------------------------- chat
    def stream_chat(self, messages: list[dict[str, Any]], *, model: str | None = None,
                    system_prompt: str | None = None, params: dict[str, Any] | None = None,
                    task: str = "text-generation", **options: Any) -> Iterator[str]:
        payload = self._payload(model, task, messages=messages, system_prompt=system_prompt,
                                params=params or {}, **options)
        self.state = _STATE_GENERATING
        client = self._client()
        try:
            for message in client.stream(Command.GENERATE_STREAM, payload, timeout=self._timeout):
                event = message.get("event")
                body = message.get("payload") or {}
                if event == Event.TOKEN_STREAM:
                    yield str(body.get("text") or "")
                elif event == Event.ERROR:
                    error = ProviderError.from_worker_payload(body, provider_id=self.id)
                    self.last_error = error
                    self.state = _STATE_ERROR
                    raise error
                elif event in (Event.RESULT, Event.GENERATION_COMPLETE):
                    self.last_metrics = dict(body.get("metrics") or {})
            self.state = _STATE_READY
        except TimeoutError as exc:
            self.state = _STATE_TIMEOUT
            raise ProviderError("TIMEOUT", "The generation did not finish in time.", recoverable=True,
                                suggested_actions=("retry", "reduce_max_tokens")) from exc
        except WorkerError as exc:
            raise ProviderError.from_worker_payload(exc.payload if isinstance(exc.payload, Mapping) else {},
                                                   provider_id=self.id) from exc
        finally:
            self.last_request_id = None

    def chat(self, messages: list[dict[str, Any]], *, model: str | None = None, **params: Any) -> str:
        return "".join(self.stream_chat(messages, model=model, **params))

    # ---------------------------------------------------------- modalities
    def embed(self, texts: list[str], *, model: str | None = None, **options: Any) -> list[list[float]]:
        payload = self._payload(model, "sentence-similarity", inputs=list(texts), **options)
        result = self._call(Command.EMBED, payload)
        return [list(vector) for vector in (result.get("embeddings") or [])]

    def transcribe(self, audio_path: str, *, model: str | None = None, language: str | None = None,
                   translate: bool = False, timestamps: bool = True, **options: Any) -> dict[str, Any]:
        payload = self._payload(model, "automatic-speech-recognition", audio_path=audio_path,
                                language=language, translate=translate, timestamps=timestamps, **options)
        return self._call(Command.TRANSCRIBE, payload)

    def synthesize(self, text: str, *, model: str | None = None, voice: int | str = 0, speed: float = 1.0,
                   pitch: float = 0.0, **options: Any) -> dict[str, Any]:
        payload = self._payload(model, "text-to-speech", text=text, voice=int(voice or 0), speed=speed,
                                pitch=pitch, **options)
        result = self._call(Command.SYNTHESIZE_SPEECH, payload)
        path = result.get("audio_path")
        if path and Path(path).is_file():
            result["audio"] = Path(path).read_bytes()
        return result

    def generate_image(self, prompt: str, *, model: str | None = None, **options: Any) -> dict[str, Any]:
        payload = self._payload(model, "text-to-image", prompt=prompt, **options)
        return self._call(Command.GENERATE_IMAGE, payload)

    def vision(self, image_path: str, prompt: str = "", *, model: str | None = None,
               **options: Any) -> str:
        payload = self._payload(model, "image-text-to-text", image_path=image_path, prompt=prompt, **options)
        result = self._call(Command.VISION_GENERATE, payload)
        return str(result.get("text") or "")

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from noelle.runtime.tools import ToolError, execute_tool

        try:
            return execute_tool(name, arguments)
        except ToolError as exc:
            raise ProviderError("TOOL_INVALID", str(exc), category="tool", recoverable=True,
                                suggested_actions=("fix_tool_arguments",)) from exc

    # --------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        data: dict[str, Any] = {"provider_id": self.id, "state": self.state, "is_local": True,
                                "last_error": self.last_error.to_dict() if self.last_error else None}
        try:
            payload = self._call(Command.PING, {}, timeout=5.0)
            data.update({"reachable": True, "pid": payload.get("pid"), "runtime_ready": payload.get("runtime_ready"),
                         "uptime_s": payload.get("uptime_s")})
            if self.runtime_client is not None:
                data["loaded_models"] = list(getattr(self.runtime_client, "loaded_models", []))
                data["runtime_caps"] = dict(getattr(self.runtime_client, "runtime_caps", {}) or {})
        except ProviderError as exc:
            data.update({"reachable": False, "error": exc.to_dict()})
        return data

    def cancel(self) -> None:
        try:
            self._client().stop()
            self.state = _STATE_CANCELLED
        except Exception as exc:  # noqa: BLE001 - cancellation is best effort
            log.debug("cancel failed: %s", exc)

    def capabilities(self) -> ProviderCapabilities:
        loaded = getattr(self.runtime_client, "loaded_models", []) if self.runtime_client is not None else []
        return ProviderCapabilities(
            chat=True, streaming=True, vision=True, audio_input=True, audio_output=True, tools=True,
            embedding=True, image_generation=True, speculative_decoding=True, is_local=True,
            models=tuple(entry.get("model_id", "") for entry in loaded if isinstance(entry, Mapping)),
        )

    # ---------------------------------------------------- legacy workflow API
    def get_capabilities(self) -> LegacyCapabilities:
        return LegacyCapabilities(
            provider_id=self.id, display_name=self.name, is_local=True,
            modalities={m.value for m in Modality}, supported_models=[],
            supports_streaming=True, max_context_window=32_768)

    def estimate_cost(self, request: NodeRequest) -> float:
        return 0.0

    def execute(self, request: NodeRequest) -> NodeResponse:
        start = time.perf_counter()
        modality = request.modality_str
        model = request.model or self._default_model
        try:
            if modality == Modality.LLM.value:
                messages = ([{"role": "system", "content": request.system_prompt}] if request.system_prompt else [])
                messages.append({"role": "user", "content": request.prompt})
                output = self.chat(messages, model=model, params={"max_new_tokens": request.max_tokens,
                                                                  "temperature": request.temperature})
                telemetry = self._telemetry(start, request, output)
            elif modality == Modality.EMBEDDING.value:
                texts = request.texts or [request.prompt]
                output = self.embed(texts, model=model)
                telemetry = self._telemetry(start, request, "")
            elif modality == Modality.STT.value:
                path = request.audio_path
                temporary = None
                if not path and request.audio_bytes:
                    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    handle.write(request.audio_bytes)
                    handle.close()
                    temporary = handle.name
                    path = temporary
                if not path:
                    raise ProviderError("MISSING_AUDIO", "No audio was provided.", category="request")
                try:
                    result = self.transcribe(path, model=model, language=request.language)
                finally:
                    if temporary:
                        Path(temporary).unlink(missing_ok=True)
                output = result.get("text", "")
                telemetry = self._telemetry(start, request, output)
                telemetry.audio_duration_s = float(result.get("duration_s") or 0.0)
            elif modality == Modality.TTS.value:
                result = self.synthesize(request.prompt, model=model)
                output = result.get("audio") or b""
                telemetry = self._telemetry(start, request, "")
                telemetry.audio_duration_s = float(result.get("duration_s") or 0.0)
            elif modality == Modality.VISION.value:
                if not request.image_path:
                    raise ProviderError("MISSING_IMAGE", "No image was provided.", category="request")
                output = self.vision(request.image_path, request.prompt, model=model)
                telemetry = self._telemetry(start, request, str(output))
            else:
                raise ProviderError("UNSUPPORTED_TASK", f"Unsupported local modality: {modality}",
                                    category="request")
            return NodeResponse(output=output, telemetry=telemetry, provider_id=self.id,
                                model_id=model or "", raw_payload={"engine": self.last_metrics.get("engine")})
        except ProviderError as exc:
            return NodeResponse(output="", telemetry=TelemetryRecord(latency_ms=(time.perf_counter() - start) * 1000),
                                provider_id=self.id, model_id=model or "", is_error=True,
                                error_message=exc.info.message, raw_payload=exc.to_dict())
        except Exception as exc:  # noqa: BLE001
            log.exception("Local execution failed")
            return NodeResponse(output="", telemetry=TelemetryRecord(latency_ms=(time.perf_counter() - start) * 1000),
                                provider_id=self.id, model_id=model or "", is_error=True, error_message=str(exc))

    @staticmethod
    def _telemetry(start: float, request: NodeRequest, output: str) -> TelemetryRecord:
        latency_ms = (time.perf_counter() - start) * 1000.0
        output_tokens = max(len(output.split()), 0) if output else 0
        return TelemetryRecord(latency_ms=latency_ms, cost_usd=0.0,
                               input_tokens=max(len(request.prompt.split()), 0), output_tokens=output_tokens,
                               tokens_per_second=(output_tokens / (latency_ms / 1000.0)) if latency_ms else 0.0)

    def close(self) -> None:
        if self._owned is not None:
            try:
                self._owned.stop()
            except Exception:  # noqa: BLE001
                pass
            self._owned = None