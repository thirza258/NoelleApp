"""Wire contract between the desktop shell, the API server and the worker.

Transport: ZeroMQ over loopback TCP. Clients use a DEALER socket, the worker a
ROUTER socket. Every frame is one UTF-8 JSON object.

Request (client -> worker)::

    {"v": 1, "id": "<uuid>", "cmd": "GENERATE_STREAM", "auth": "<token>", "payload": {...}}

Event (worker -> client)::

    {"v": 1, "id": "<request id or null>", "event": "TOKEN_STREAM", "payload": {...}}

A request is finished when it receives one of :data:`TERMINAL_EVENTS`.
Events with ``id == null`` are broadcasts (heartbeats, model load/unload
notices) delivered to every client that sent ``HELLO``.

This module is pure Python on purpose: the UI imports it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 1
READY_PREFIX = "NOELLE_WORKER_READY"
AUTH_ENV = "NOELLE_WORKER_TOKEN"


class Command(StrEnum):
    HELLO = "HELLO"
    PING = "PING"
    HARDWARE_STATS = "HARDWARE_STATS"
    LIST_LOADED = "LIST_LOADED"
    LOAD_MODEL = "LOAD_MODEL"
    UNLOAD_MODEL = "UNLOAD_MODEL"
    GENERATE_STREAM = "GENERATE_STREAM"
    STOP_GENERATION = "STOP_GENERATION"
    VISION_GENERATE = "VISION_GENERATE"
    DETECT_OBJECTS = "DETECT_OBJECTS"
    CLASSIFY_IMAGE = "CLASSIFY_IMAGE"
    TRANSCRIBE = "TRANSCRIBE"
    SYNTHESIZE_SPEECH = "SYNTHESIZE_SPEECH"
    GENERATE_IMAGE = "GENERATE_IMAGE"
    EMBED = "EMBED"
    RAG_INGEST = "RAG_INGEST"
    RAG_QUERY = "RAG_QUERY"
    RAG_LIST_COLLECTIONS = "RAG_LIST_COLLECTIONS"
    RAG_DELETE_COLLECTION = "RAG_DELETE_COLLECTION"
    RAG_LIST_DOCUMENTS = "RAG_LIST_DOCUMENTS"
    RAG_DELETE_DOCUMENT = "RAG_DELETE_DOCUMENT"
    WEB_SEARCH = "WEB_SEARCH"
    QUANTIZE_CAPABILITIES = "QUANTIZE_CAPABILITIES"
    QUANTIZE = "QUANTIZE"
    LORA_LOAD = "LORA_LOAD"
    LORA_SET_ADAPTER = "LORA_SET_ADAPTER"
    LORA_LIST = "LORA_LIST"
    RUN_BENCHMARK = "RUN_BENCHMARK"
    RUN_EVAL = "RUN_EVAL"  # Evaluation Lab: dataset cases x models, scored locally
    SHUTDOWN = "SHUTDOWN"


class Event(StrEnum):
    ACK = "ACK"  # request accepted; payload: {"queued": bool, "position": int}
    STATUS = "STATUS"  # human-readable stage, e.g. {"stage": "loading", "message": "..."}
    PROGRESS = "PROGRESS"  # numeric progress, e.g. {"current": 3, "total": 20}
    CONTEXT = "CONTEXT"  # grounding sources, e.g. {"sources": [...]}
    TOKEN_STREAM = "TOKEN_STREAM"  # {"text": "...", "view": optional int}
    QUEUE_PLAN = "QUEUE_PLAN"  # benchmark scheduling decision
    VIEW_STARTED = "VIEW_STARTED"
    VIEW_COMPLETE = "VIEW_COMPLETE"
    RESULT = "RESULT"  # terminal for non-streaming commands
    GENERATION_COMPLETE = "GENERATION_COMPLETE"  # terminal for streaming commands
    ERROR = "ERROR"  # terminal; {"code": ErrorCode, "message": str, "hint": str|None}
    HEARTBEAT = "HEARTBEAT"  # broadcast; memory + loaded models
    MODELS_CHANGED = "MODELS_CHANGED"  # broadcast after load/unload/eviction


class ErrorCode(StrEnum):
    OOM = "OOM"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    LOAD_FAILED = "LOAD_FAILED"
    BAD_REQUEST = "BAD_REQUEST"
    CANCELLED = "CANCELLED"
    UNSUPPORTED = "UNSUPPORTED"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    REMOTE_CODE_BLOCKED = "REMOTE_CODE_BLOCKED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL = "INTERNAL"
    WORKER_CRASHED = "WORKER_CRASHED"  # synthesized client-side when the process dies
    TIMEOUT = "TIMEOUT"  # synthesized client-side


TERMINAL_EVENTS = frozenset({Event.RESULT, Event.GENERATION_COMPLETE, Event.ERROR})

# Commands answered immediately by the worker's I/O thread. Everything else is
# queued on the single GPU lane, which is what serialises GPU work.
LIGHT_COMMANDS = frozenset(
    {
        Command.HELLO,
        Command.PING,
        Command.HARDWARE_STATS,
        Command.LIST_LOADED,
        Command.STOP_GENERATION,
        Command.RAG_LIST_COLLECTIONS,
        Command.RAG_LIST_DOCUMENTS,
        Command.QUANTIZE_CAPABILITIES,
        Command.SHUTDOWN,
    }
)


class ProtocolError(ValueError):
    pass


def new_id() -> str:
    return uuid.uuid4().hex


def encode(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_json_default).encode("utf-8")


def decode(data: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid frame: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("frame must be a JSON object")
    return obj


def make_request(cmd: Command | str, payload: dict[str, Any] | None = None, *, auth: str | None = None,
                 request_id: str | None = None) -> tuple[str, bytes]:
    rid = request_id or new_id()
    frame = {"v": PROTOCOL_VERSION, "id": rid, "cmd": str(cmd), "auth": auth, "payload": payload or {}}
    return rid, encode(frame)


def make_event(request_id: str | None, event: Event | str, payload: dict[str, Any] | None = None) -> bytes:
    return encode({"v": PROTOCOL_VERSION, "id": request_id, "event": str(event), "payload": payload or {}})


def parse_request(data: bytes) -> tuple[str, Command, dict[str, Any], str | None]:
    obj = decode(data)
    rid = obj.get("id")
    if not isinstance(rid, str) or not rid:
        raise ProtocolError("request id missing")
    try:
        cmd = Command(obj.get("cmd"))
    except ValueError as exc:
        raise ProtocolError(f"unknown command {obj.get('cmd')!r}") from exc
    payload = obj.get("payload") or {}
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    auth = obj.get("auth")
    return rid, cmd, payload, auth if isinstance(auth, str) else None


def error_payload(code: ErrorCode | str, message: str, hint: str | None = None, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"code": str(code), "message": message, "hint": hint}
    data.update(extra)
    return data


@dataclass
class GenerationParams:
    """Sampling parameters shared by the playground, the bench and the API server."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    seed: int | None = None
    stop: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GenerationParams:
        data = data or {}
        params = cls()
        if "temperature" in data and data["temperature"] is not None:
            params.temperature = min(max(float(data["temperature"]), 0.0), 2.0)
        if "top_p" in data and data["top_p"] is not None:
            params.top_p = min(max(float(data["top_p"]), 0.01), 1.0)
        if "top_k" in data and data["top_k"] is not None:
            params.top_k = max(int(data["top_k"]), 0)
        if "repetition_penalty" in data and data["repetition_penalty"] is not None:
            params.repetition_penalty = min(max(float(data["repetition_penalty"]), 0.5), 2.5)
        if "max_new_tokens" in data and data["max_new_tokens"] is not None:
            params.max_new_tokens = max(int(data["max_new_tokens"]), 1)
        if data.get("seed") is not None:
            params.seed = int(data["seed"])
        stop = data.get("stop")
        if isinstance(stop, str):
            params.stop = [stop]
        elif isinstance(stop, list):
            params.stop = [str(s) for s in stop if s]
        return params

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, StrEnum):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__fspath__"):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
