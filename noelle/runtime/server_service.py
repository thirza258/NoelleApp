"""OpenAI-compatible local REST server (``noelle server``).

Endpoints (everything under ``/v1`` requires ``Authorization: Bearer <key>``
when ``NOELLE_SERVER_API_KEY`` is set):

* ``GET  /health``                   server and worker liveness
* ``GET  /v1/models``                registry models (OpenAI list shape plus ``task``)
* ``GET  /v1/models/{id}``           one model
* ``POST /v1/chat/completions``      GENERATE_STREAM, streamed as server-sent events or not
* ``POST /v1/embeddings``            EMBED
* ``POST /v1/audio/transcriptions``  TRANSCRIBE (multipart upload)

The server never loads a model itself: each request becomes one worker command
over ZeroMQ (:class:`noelle.runtime.client.AsyncWorkerClient`), so API traffic
queues on the same GPU lane as the desktop UI. Worker errors before the first
token keep their HTTP status even for streaming requests; if the HTTP client
disconnects, the server sends ``STOP_GENERATION`` for its request. The server
binds to loopback only. The access log (``logs/server-access.log``) records
method, path, status and duration, never prompt or completion content.
"""

from __future__ import annotations

import argparse
import base64
import email.parser
import email.policy
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

import anyio
import uvicorn
import zmq
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from noelle import __version__, paths
from noelle.app.services.model_registry import ModelRecord, ModelRegistry, get_registry
from noelle.runtime import tasks
from noelle.runtime.client import AsyncWorkerClient, WorkerError, read_discovery_file
from noelle.runtime.protocol import AUTH_ENV, TERMINAL_EVENTS, Command, ErrorCode, Event, new_id

log = logging.getLogger("noelle.server")
T = TypeVar("T")

READY_PREFIX = "NOELLE_SERVER_READY"
API_KEY_ENV = "NOELLE_SERVER_API_KEY"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WORKER_ACK_TIMEOUT_S = 15.0  # every queued command is ACKed at once; silence means no worker
WORKER_EVENT_TIMEOUT_S = 600.0  # longest gap between two events (model loading)
HEALTH_TIMEOUT_S = 3.0
STOP_TIMEOUT_S = 5.0
DISCONNECT_POLL_S = 0.25
ACCESS_LOG_MAX_BYTES = 10 * 1024 * 1024
TRANSCRIPTION_FORMATS = ("json", "text", "verbose_json")
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """Rendered as an OpenAI error object ``{"error": {"message", "type", "param", "code"}}``."""

    def __init__(self, status: int, message: str, *, type: str = "invalid_request_error", code: str | None = None,
                 param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.param = param

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}

    def response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status)


class ClientGone(Exception):
    """The HTTP client disconnected before the answer was ready."""


_WORKER_STATUS: dict[str, tuple[int, str]] = {
    ErrorCode.MODEL_NOT_FOUND: (404, "invalid_request_error"),
    ErrorCode.BAD_REQUEST: (400, "invalid_request_error"),
    ErrorCode.OOM: (503, "server_error"),
}


def worker_error(payload: dict[str, Any]) -> ApiError:
    """Map a worker ``ERROR`` payload to an HTTP error (MODEL_NOT_FOUND 404, BAD_REQUEST 400, OOM 503, else 500)."""
    code = str(payload.get("code") or ErrorCode.INTERNAL)
    message = str(payload.get("message") or "The model worker reported an error.")
    if payload.get("hint"):
        message = f"{message} {payload['hint']}"
    if code == ErrorCode.UNAUTHORIZED:
        # The worker rejected this *server's* token: a deployment problem, not the API client's key.
        return ApiError(500, f"The model worker rejected the API server's credentials: {message}",
                        type="server_error", code="worker_unauthorized")
    status, kind = _WORKER_STATUS.get(code, (500, "server_error"))
    return ApiError(status, message, type=kind, code=code.lower())


# ---------------------------------------------------------------------------
# worker requests
# ---------------------------------------------------------------------------


class WorkerCall:
    """One worker request. ``ERROR`` events and transport failures surface as :class:`ApiError`."""

    def __init__(self, client: AsyncWorkerClient, cmd: Command, payload: dict[str, Any]) -> None:
        self.client = client
        self.request_id = new_id()
        self._events = client.stream(cmd, payload, timeout=WORKER_EVENT_TIMEOUT_S, request_id=self.request_id)
        self._acked = False
        self.finished = False  # a terminal event arrived, or the request was abandoned
        self.client_gone = False

    async def next(self) -> dict[str, Any]:
        try:
            if self._acked:
                event = await self._events.__anext__()
            else:
                with anyio.fail_after(WORKER_ACK_TIMEOUT_S):
                    event = await self._events.__anext__()
                self._acked = True
        except StopAsyncIteration:
            self.finished = True
            raise ApiError(502, "The model worker closed the request without a result.", type="server_error",
                           code="worker_error") from None
        except TimeoutError:
            self.finished = True
            if not self._acked:
                raise ApiError(503, "The Noelle model worker is not responding. Is the runtime running?",
                               type="server_error", code="worker_unavailable") from None
            raise ApiError(504, "The model worker stopped sending events.", type="server_error",
                           code="timeout") from None
        except zmq.ZMQError as exc:
            self.finished = True
            raise ApiError(503, f"Cannot reach the model worker: {exc}", type="server_error",
                           code="worker_unavailable") from None
        kind = event.get("event")
        if kind in TERMINAL_EVENTS:
            self.finished = True
        if kind == Event.ERROR:
            raise worker_error(event.get("payload") or {})
        return event

    async def first_output(self) -> dict[str, Any]:
        """Skip ACK/STATUS/... until the first token or terminal event."""
        while True:
            event = await self.next()
            if event.get("event") == Event.TOKEN_STREAM or event.get("event") in TERMINAL_EVENTS:
                return event

    async def generation(self) -> tuple[str, dict[str, Any]]:
        """Collect a whole generation: ``(streamed text, GENERATION_COMPLETE payload)``."""
        parts: list[str] = []
        while True:
            event = await self.next()
            payload = event.get("payload") or {}
            if event.get("event") == Event.TOKEN_STREAM:
                parts.append(str(payload.get("text") or ""))
            elif event.get("event") in TERMINAL_EVENTS:
                return "".join(parts), payload

    async def result(self) -> dict[str, Any]:
        """Payload of the terminal ``RESULT`` event."""
        while True:
            event = await self.next()
            if event.get("event") in TERMINAL_EVENTS:
                return event.get("payload") or {}

    async def close(self) -> None:
        """Release the request; stops it on the worker if it has not finished. Safe in cancelled tasks."""
        with anyio.CancelScope(shield=True):
            if not self.finished:
                self.finished = True
                with anyio.move_on_after(STOP_TIMEOUT_S):
                    try:
                        await self.client.call(Command.STOP_GENERATION, {"target_id": self.request_id},
                                               timeout=STOP_TIMEOUT_S)
                    except Exception as exc:  # the worker may be gone; nothing else to do
                        log.warning("could not stop worker request %s: %s", self.request_id, exc)
            try:
                await self._events.aclose()
            except Exception:
                pass


async def watch_disconnect(request: Request, call: WorkerCall, work: Callable[[], Awaitable[T]]) -> T:
    """Run ``work`` while polling for an HTTP client disconnect; raises :class:`ClientGone` if it happens."""
    outcome: dict[str, Any] = {}
    async with anyio.create_task_group() as group:

        async def watcher() -> None:
            while True:
                await anyio.sleep(DISCONNECT_POLL_S)
                if await request.is_disconnected():
                    call.client_gone = True
                    group.cancel_scope.cancel()
                    return

        group.start_soon(watcher)
        try:
            outcome["value"] = await work()
        except Exception as exc:  # re-raised below, outside the task group (no ExceptionGroup wrapping)
            outcome["error"] = exc
        finally:
            group.cancel_scope.cancel()
    if call.client_gone:
        raise ClientGone()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class _EventStreamResponse(StreamingResponse):
    """StreamingResponse that always closes its generator (so cleanup runs even when cancelled mid-send)."""

    async def stream_response(self, send: Any) -> None:
        try:
            await super().stream_response(send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# request parsing
# ---------------------------------------------------------------------------


async def _json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    try:
        data = json.loads(raw or b"null")
    except ValueError as exc:
        raise ApiError(400, f"The request body is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ApiError(400, "The request body must be a JSON object.")
    return data


def _number(body: dict[str, Any], key: str, low: float, high: float) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ApiError(400, f"'{key}' must be a number between {low:g} and {high:g}.", param=key)
    return float(value)


def _integer(body: dict[str, Any], key: str, low: int | None = None) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or (low is not None and value < low):
        raise ApiError(400, f"'{key}' must be an integer" + (f" >= {low}." if low is not None else "."), param=key)
    return value


def chat_messages(raw: Any) -> list[dict[str, Any]]:
    """Validate OpenAI messages; keep text only (``developer`` becomes ``system``)."""
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, "'messages' must be a non-empty array.", param="messages")
    out: list[dict[str, Any]] = []
    for index, message in enumerate(raw):
        where = f"messages[{index}]"
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ApiError(400, f"{where} must be an object with a string 'role'.", param=where)
        role = "system" if message["role"] == "developer" else message["role"]
        content = message.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                    texts.append(str(part.get("text") or ""))
                else:
                    raise ApiError(400, f"{where}: only text content parts are supported by this server.",
                                   param=where)
            content = "".join(texts)
        elif content is not None and not isinstance(content, str):
            raise ApiError(400, f"{where}.content must be a string or an array of text parts.", param=where)
        if content is None:  # e.g. an assistant tool-call stub
            continue
        out.append({"role": role, "content": content})
    if not out:
        raise ApiError(400, "'messages' contains no text.", param="messages")
    return out


def sampling_params(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI sampling fields -> worker ``params`` (see :class:`noelle.runtime.protocol.GenerationParams`)."""
    max_tokens = _integer(body, "max_completion_tokens", 1)
    if max_tokens is None:
        max_tokens = _integer(body, "max_tokens", 1)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif stop is not None and not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        raise ApiError(400, "'stop' must be a string or an array of strings.", param="stop")
    params: dict[str, Any] = {
        "temperature": _number(body, "temperature", 0.0, 2.0),
        "top_p": _number(body, "top_p", 0.0, 1.0),
        "max_new_tokens": max_tokens,
        "stop": stop,
        "seed": _integer(body, "seed"),
    }
    top_k = _integer(body, "top_k", 0)  # common extensions (vLLM / llama.cpp servers)
    if top_k is not None:
        params["top_k"] = top_k
    repetition = _number(body, "repetition_penalty", 0.5, 2.5)
    if repetition is not None:
        params["repetition_penalty"] = repetition
    return params


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parse ``multipart/form-data`` with the standard library (python-multipart is not required).

    Returns ``(fields, files)`` where ``files`` maps a field name to ``(filename, data)``.
    """
    try:
        header = b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n"
    except UnicodeEncodeError:
        raise ApiError(400, "Malformed Content-Type header.") from None
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(header + body)
    if not message.is_multipart():
        raise ApiError(400, "Malformed multipart/form-data body.")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            files[str(name)] = (str(filename), data)
        else:
            fields[str(name)] = data.decode(part.get_content_charset() or "utf-8", errors="replace")
    return fields, files


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise ApiError(413, f"The upload is larger than {limit // 2**20} MB.", code="file_too_large")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise ApiError(413, f"The upload is larger than {limit // 2**20} MB.", code="file_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _save_upload(filename: str, data: bytes) -> Path:
    folder = paths.projects_dir() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ".bin"
    fd, name = tempfile.mkstemp(prefix="transcribe-", suffix=suffix, dir=folder)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return Path(name)


def _encode_vector(vector: list[float], encoding: str) -> list[float] | str:
    if encoding == "base64":  # little-endian float32, what the official clients decode
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
    return [float(x) for x in vector]


def _usage(metrics: dict[str, Any]) -> dict[str, int]:
    prompt = int(metrics.get("prompt_tokens") or 0)
    completion = int(metrics.get("completion_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _finish_reason(metrics: dict[str, Any]) -> str:
    return "length" if metrics.get("finish_reason") == "length" else "stop"  # cancelled -> stop


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@dataclass
class _ChatMeta:
    id: str
    model: str
    created: int
    include_usage: bool = False

    def chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "system_fingerprint": None,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
        }
        if self.include_usage:
            data["usage"] = None
        return data

    def usage_chunk(self, usage: dict[str, int]) -> dict[str, Any]:
        return {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                "system_fingerprint": None, "choices": [], "usage": usage}

    def completion(self, text: str, finish_reason: str, usage: dict[str, int]) -> dict[str, Any]:
        return {
            "id": self.id, "object": "chat.completion", "created": self.created, "model": self.model,
            "system_fingerprint": None,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "refusal": None},
                         "logprobs": None, "finish_reason": finish_reason}],
            "usage": usage,
        }


async def _chat_events(call: WorkerCall, first: dict[str, Any], meta: _ChatMeta) -> AsyncIterator[str]:
    try:
        yield _sse(meta.chunk({"role": "assistant"}))
        event = first
        while True:
            payload = event.get("payload") or {}
            if event.get("event") == Event.TOKEN_STREAM:
                text = str(payload.get("text") or "")
                if text:
                    yield _sse(meta.chunk({"content": text}))
            elif event.get("event") in TERMINAL_EVENTS:
                metrics = payload.get("metrics") or {}
                yield _sse(meta.chunk({}, _finish_reason(metrics)))
                if meta.include_usage:
                    yield _sse(meta.usage_chunk(_usage(metrics)))
                break
            event = await call.next()
        yield "data: [DONE]\n\n"
    except ApiError as exc:  # failure after the stream started: OpenAI clients raise on an error event
        yield _sse(exc.body())
        yield "data: [DONE]\n\n"
    finally:
        await call.close()  # stops the worker request when the client went away mid-stream


# ---------------------------------------------------------------------------
# access log
# ---------------------------------------------------------------------------


class AccessLogMiddleware:
    """Pure ASGI middleware: one line per request (method, path, status, duration). Never logs bodies."""

    def __init__(self, app: Any, log_path: Path) -> None:
        self.app = app
        self.log_path = log_path
        self._lock = threading.Lock()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.write(str(scope.get("method", "?")), str(scope.get("path", "?")), status,
                       (time.perf_counter() - started) * 1000.0)

    def write(self, method: str, path: str, status: int, duration_ms: float) -> None:
        safe_path = urllib.parse.quote(path, safe="/:@-._~!$&'()*+,;=")
        stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        line = f"{stamp} {method} {safe_path} {status} {duration_ms:.1f}ms\n"
        try:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                if self.log_path.exists() and self.log_path.stat().st_size > ACCESS_LOG_MAX_BYTES:
                    os.replace(self.log_path, self.log_path.with_name(self.log_path.name + ".1"))
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except OSError:
            log.debug("could not write the access log", exc_info=True)


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------


class _ServerState:
    def __init__(self, worker_endpoint: str | None, token: str | None, registry: ModelRegistry | None,
                 api_key: str | None) -> None:
        self._endpoint = worker_endpoint
        self.token = token
        self._registry = registry
        self.api_key = api_key

    def endpoint(self) -> str | None:
        """The fixed endpoint, or the live worker's discovery file (follows worker restarts)."""
        if self._endpoint:
            return self._endpoint
        info = read_discovery_file()
        if info and info.get("port"):
            return f"tcp://{info.get('host') or '127.0.0.1'}:{int(info['port'])}"
        return None

    def client(self) -> AsyncWorkerClient:
        endpoint = self.endpoint()
        if not endpoint:
            raise ApiError(503, "The Noelle model worker is not running. Start the runtime from the desktop app.",
                           type="server_error", code="worker_unavailable")
        return AsyncWorkerClient(endpoint, self.token)

    def call(self, cmd: Command, payload: dict[str, Any]) -> WorkerCall:
        return WorkerCall(self.client(), cmd, payload)

    def registry(self) -> ModelRegistry:
        return self._registry if self._registry is not None else get_registry()

    def find(self, name: str) -> ModelRecord | None:
        """Exact ``api_name`` first: the registry's lookup also matches ``repo_id`` and may pick another precision."""
        registry = self.registry()
        for record in registry.list_models():
            if record.api_name == name:
                return record
        return registry.find_by_api_name(name)

    def listable(self) -> list[ModelRecord]:
        seen: set[str] = set()
        records = []
        for record in self.registry().list_models():
            if record.local_path and not Path(record.local_path).exists():
                continue  # deleted outside Noelle
            if record.api_name in seen:
                continue
            seen.add(record.api_name)
            records.append(record)
        return records

    async def resolve(self, name: Any, families: tuple[str, ...], purpose: str) -> ModelRecord:
        if not isinstance(name, str) or not name.strip():
            raise ApiError(400, "'model' is required.", param="model")
        record = await run_in_threadpool(self.find, name.strip())
        if record is None:
            available = [r.api_name for r in await run_in_threadpool(self.listable)]
            raise ApiError(404, f"The model '{name}' does not exist. Available models: "
                                f"{', '.join(available) if available else '(none)'}.",
                           code="model_not_found", param="model")
        family = record.family or (tasks.TEXT if not record.pipeline_tag else None)
        if family not in families:
            raise ApiError(400, f"'{record.api_name}' is a {record.pipeline_tag or 'unknown'} model and cannot "
                                f"serve {purpose}.", code="model_not_supported", param="model")
        return record


def _model_card(record: ModelRecord) -> dict[str, Any]:
    return {"id": record.api_name, "object": "model", "created": int(record.downloaded_at or 0),
            "owned_by": "local", "task": record.pipeline_tag}


def create_app(worker_endpoint: str | None, token: str | None, *, registry: ModelRegistry | None = None,
               api_key: str | None = None) -> FastAPI:
    """Build the FastAPI app. ``worker_endpoint=None`` follows the worker discovery file."""
    state = _ServerState(worker_endpoint, token, registry, api_key)
    app = FastAPI(title="Noelle API", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.noelle = state
    app.add_middleware(AccessLogMiddleware, log_path=paths.logs_dir() / "server-access.log")

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(ClientGone)
    async def _client_gone(_request: Request, _exc: ClientGone) -> Response:
        return Response(status_code=499)  # nginx's "client closed request"; nobody reads it

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse({"error": {"message": str(exc.detail), "type": kind, "param": None, "code": None}},
                            status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return ApiError(400, f"Invalid request: {exc.errors()}").response()

    async def require_api_key(request: Request) -> None:
        if not state.api_key:
            return
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip().encode(), state.api_key.encode()):
            raise ApiError(401, "Invalid or missing API key. Send 'Authorization: Bearer <key>'.",
                           code="invalid_api_key")

    @app.get("/health")
    async def health() -> JSONResponse:
        worker: dict[str, Any] = {"reachable": False}
        endpoint = state.endpoint()
        if endpoint is None:
            worker["error"] = "worker not running"
        else:
            try:
                with anyio.fail_after(HEALTH_TIMEOUT_S):
                    info = await AsyncWorkerClient(endpoint, state.token).call(Command.PING, timeout=HEALTH_TIMEOUT_S)
                worker.update(reachable=True, runtime_ready=info.get("runtime_ready"), pid=info.get("pid"))
            except WorkerError as exc:
                worker["error"] = str(exc)
            except Exception as exc:
                worker["error"] = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        ok = bool(worker["reachable"])
        return JSONResponse({"status": "ok" if ok else "degraded", "version": __version__, "worker": worker},
                            status_code=200 if ok else 503)

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    @v1.get("/models")
    async def list_models() -> dict[str, Any]:
        records = await run_in_threadpool(state.listable)
        return {"object": "list", "data": [_model_card(r) for r in records]}

    @v1.get("/models/{model_id:path}")
    async def get_model(model_id: str) -> dict[str, Any]:
        record = await run_in_threadpool(state.find, model_id)
        if record is None:
            raise ApiError(404, f"The model '{model_id}' does not exist.", code="model_not_found", param="model")
        return _model_card(record)

    @v1.post("/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await _json_body(request)
        messages = chat_messages(body.get("messages"))
        params = sampling_params(body)
        if body.get("n") not in (None, 1):
            raise ApiError(400, "Only n=1 is supported.", param="n")
        stream_options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        record = await state.resolve(body.get("model"), (tasks.TEXT,), "chat completions")
        call = state.call(Command.GENERATE_STREAM, {**record.worker_payload(), "messages": messages,
                                                    "params": params})
        meta = _ChatMeta(id=f"chatcmpl-{call.request_id}", model=record.api_name, created=int(time.time()),
                         include_usage=bool(stream_options.get("include_usage")))
        handed_off = False
        try:
            if body.get("stream"):
                # Wait for the first token so errors before it (unknown model, OOM, load failure) keep their status.
                first = await watch_disconnect(request, call, call.first_output)
                handed_off = True
                return _EventStreamResponse(_chat_events(call, first, meta), media_type="text/event-stream",
                                            headers=_SSE_HEADERS)
            streamed, done = await watch_disconnect(request, call, call.generation)
            metrics = done.get("metrics") or {}
            text = done.get("text") if isinstance(done.get("text"), str) else streamed
            return JSONResponse(meta.completion(text, _finish_reason(metrics), _usage(metrics)))
        finally:
            if not handed_off:
                await call.close()

    @v1.post("/embeddings")
    async def embeddings(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs or not all(isinstance(t, str) for t in inputs):
            raise ApiError(400, "'input' must be a string or a non-empty array of strings "
                                "(token arrays are not supported).", param="input")
        encoding = body.get("encoding_format") or "float"
        if encoding not in ("float", "base64"):
            raise ApiError(400, "'encoding_format' must be 'float' or 'base64'.", param="encoding_format")
        record = await state.resolve(body.get("model"), (tasks.EMBEDDING,), "embeddings")
        call = state.call(Command.EMBED, {**record.worker_payload(), "inputs": inputs})
        try:
            result = await watch_disconnect(request, call, call.result)
        finally:
            await call.close()
        tokens = int(result.get("prompt_tokens") or 0)
        data = [{"object": "embedding", "index": i, "embedding": _encode_vector(vector, encoding)}
                for i, vector in enumerate(result.get("embeddings") or [])]
        return {"object": "list", "data": data, "model": record.api_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    @v1.post("/audio/transcriptions")
    async def transcriptions(request: Request) -> Response:
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ApiError(400, "Send the audio as multipart/form-data with a 'file' field.", param="file")
        body = await _read_body(request, MAX_UPLOAD_BYTES)
        fields, files = await run_in_threadpool(parse_multipart, content_type, body)
        if "file" not in files:
            raise ApiError(400, "The 'file' field (audio upload) is required.", param="file")
        response_format = fields.get("response_format") or "json"
        if response_format not in TRANSCRIPTION_FORMATS:
            raise ApiError(400, f"'response_format' must be one of: {', '.join(TRANSCRIPTION_FORMATS)}.",
                           param="response_format")
        language = (fields.get("language") or "").strip() or None
        record = await state.resolve(fields.get("model"), (tasks.ASR,), "transcriptions")
        filename, data = files["file"]
        audio = await run_in_threadpool(_save_upload, filename, data)
        try:
            call = state.call(Command.TRANSCRIBE, {**record.worker_payload(), "audio_path": str(audio),
                                                   "language": language,
                                                   "timestamps": response_format == "verbose_json"})
            try:
                result = await watch_disconnect(request, call, call.result)
            finally:
                await call.close()
        finally:
            audio.unlink(missing_ok=True)
        text = str(result.get("text") or "")
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "json":
            return JSONResponse({"text": text})
        segments = [{"id": i, "seek": 0, "start": chunk.get("start") or 0.0, "end": chunk.get("end") or 0.0,
                     "text": chunk.get("text", ""), "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                     "compression_ratio": 0.0, "no_speech_prob": 0.0}
                    for i, chunk in enumerate(result.get("chunks") or [])]
        return JSONResponse({"task": "transcribe", "language": language or "", "text": text,
                             "duration": float(result.get("duration_s") or 0.0), "segments": segments})

    app.include_router(v1)
    return app


# ---------------------------------------------------------------------------
# process entry points
# ---------------------------------------------------------------------------


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class NoelleServer(uvicorn.Server):
    """uvicorn.Server that reports the bound port once it accepts connections (``--port 0`` works)."""

    def __init__(self, config: uvicorn.Config, on_ready: Callable[[int], None] | None = None) -> None:
        super().__init__(config)
        self._on_ready = on_ready
        self.port: int | None = None

    async def startup(self, sockets: list[Any] | None = None) -> None:
        await super().startup(sockets=sockets)
        servers = getattr(self, "servers", None)
        if self.should_exit or not servers:
            return
        self.port = int(servers[0].sockets[0].getsockname()[1])
        if self._on_ready is not None:
            self._on_ready(self.port)


def _uvicorn_config(app: FastAPI, host: str, port: int) -> uvicorn.Config:
    return uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False, server_header=False,
                          proxy_headers=False, timeout_graceful_shutdown=5)


@dataclass
class BackgroundServer:
    server: NoelleServer
    thread: threading.Thread
    host: str

    @property
    def port(self) -> int:
        assert self.server.port is not None
        return self.server.port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self, timeout: float = 10.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)


def serve_in_thread(app: FastAPI, host: str = "127.0.0.1", port: int = 0, *, timeout: float = 15.0) -> BackgroundServer:
    """Run ``app`` with uvicorn on a daemon thread (tests, scripts); returns once it is listening."""
    ready = threading.Event()
    server = NoelleServer(_uvicorn_config(app, host, port), on_ready=lambda _port: ready.set())
    thread = threading.Thread(target=server.run, name="noelle-api-server", daemon=True)
    thread.start()
    if not ready.wait(timeout):
        server.should_exit = True
        thread.join(5)
        raise RuntimeError(f"the API server did not start on {host}:{port}")
    return BackgroundServer(server, thread, host)


def main(args: argparse.Namespace) -> int:
    """``noelle server --host --port --worker-endpoint``; prints ``NOELLE_SERVER_READY port=<p>`` when listening."""
    host = str(getattr(args, "host", None) or "127.0.0.1")
    if not is_loopback(host):
        print(f"Refusing to listen on {host}: the Noelle API server only binds to loopback "
              "(127.0.0.1, ::1 or localhost).", file=sys.stderr)
        return 2
    bind_host = "127.0.0.1" if host == "localhost" else host
    port = int(getattr(args, "port", 8765))
    paths.ensure_dirs()
    from noelle.runtime.worker_service import configure_logging

    configure_logging("server")
    endpoint = getattr(args, "worker_endpoint", None) or None
    if endpoint is None and read_discovery_file() is None:
        log.warning("no worker discovery file yet; requests will fail until the runtime starts")
    app = create_app(endpoint, os.environ.get(AUTH_ENV) or None, api_key=os.environ.get(API_KEY_ENV) or None)

    def ready(bound: int) -> None:
        print(f"{READY_PREFIX} port={bound}", flush=True)
        log.info("Noelle API server listening on http://%s:%s/v1 (worker: %s)", bind_host, bound,
                 endpoint or "discovery file")

    server = NoelleServer(_uvicorn_config(app, bind_host, port), on_ready=ready)
    parent_pid = getattr(args, "parent_pid", None)
    if parent_pid:
        threading.Thread(target=_watch_parent, args=(int(parent_pid), server), name="noelle-server-parent",
                         daemon=True).start()
    try:
        server.run()
    except SystemExit as exc:  # uvicorn exits during startup, e.g. "address already in use"
        reason = _bind_problem(bind_host, port) or "see the log above"
        print(f"The API server could not start on {bind_host}:{port}: {reason}.", file=sys.stderr)
        return exc.code if isinstance(exc.code, int) and exc.code else 1
    return 0


def _watch_parent(parent_pid: int, server: NoelleServer) -> None:
    """Exit when the desktop app that spawned us is gone, so the port is not held by an orphan."""
    import psutil

    while not server.should_exit:
        time.sleep(2.0)
        if not psutil.pid_exists(parent_pid):
            log.warning("parent process %s exited; stopping the API server", parent_pid)
            server.should_exit = True
            return


def _bind_problem(host: str, port: int) -> str | None:
    """Why ``host:port`` cannot be bound right now (e.g. "Address already in use"), or None."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            return exc.strerror or str(exc)
    return None
