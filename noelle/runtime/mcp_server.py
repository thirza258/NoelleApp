"""Model Context Protocol server on stdio (``noelle mcp``).

Lets coding agents (Claude Code, Cursor, ...) use Noelle as a set of tools:

* ``hardware_info``         accelerator, memory budget, memory bandwidth and installed engines
* ``list_local_models``     models downloaded or built in Noelle
* ``search_models``         Hugging Face Hub search with a fit score for this machine
* ``inspect_model``         capabilities of one Hub repository and how to run it here
* ``chat`` / ``embed``      inference through Noelle's local OpenAI-compatible API server
* ``list_knowledge_bases`` / ``query_knowledge_base``  document search through the same server

Transport: JSON-RPC 2.0, one message per line on stdin/stdout (MCP stdio). stdout carries
protocol messages only; logs go to ``logs/mcp.log`` and stderr. Messages are handled one at a
time, in order. Arguments that break a tool's input schema are rejected with -32602; failures
while a tool runs (server not running, Hub errors, ...) are tool results with ``isError``.

Privacy: inference and knowledge-base requests only go to an API server on 127.0.0.1, localhost
or ::1, so prompts never leave the machine; the Hub tools send only the search text or the
repository id. Prompts, tool arguments and the API key are never logged, and the key never
appears in a result or an error. There are deliberately no download, load, delete, file or
shell tools.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping, TextIO

import httpx

from noelle import __version__
from noelle.app.services import hf_client
from noelle.app.services.model_registry import get_registry
from noelle.config import Settings, get_settings
from noelle.runtime import advisor, capabilities, tasks
from noelle.runtime.preflight import format_bytes, format_params

log = logging.getLogger("noelle.mcp")

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first
MAX_MESSAGE_CHARS = 4 * 1024 * 1024
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
HTTP_TIMEOUT = httpx.Timeout(300.0, connect=3.0)
NOT_RUNNING = "Noelle's API server is not running. Start it from the API Server page in Noelle."
UNEXPECTED_RESPONSE = "Noelle's API server sent an unexpected response."
REDACTED = "[redacted]"
# Shorter keys are not scrubbed from output: replacing a common string would corrupt answers,
# and the key is never put into output in the first place.
MIN_SCRUB_LENGTH = 8
_HEADER_SAFE = re.compile(r"[\x21-\x7e]+")
REPO_ID_PATTERN = r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

INSTRUCTIONS = (
    "Noelle runs open-weight models on this machine. Use hardware_info, search_models and inspect_model to find "
    "models that fit this hardware, and list_local_models to see what is downloaded. chat, embed and the "
    "knowledge-base tools run locally through Noelle's API server, which must be running (API Server page in the "
    "Noelle app); prompts never leave this machine. Downloading, loading and deleting models is done in the app."
)


class RpcError(Exception):
    """A JSON-RPC error response (unknown method, invalid params, ...)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolError(Exception):
    """A tool failed in an expected way; the message is shown to the calling agent."""


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def _object(properties: dict[str, Any], *required: str) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = list(required)
    return schema


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    open_world: bool

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description,
                "inputSchema": self.input_schema,
                "annotations": {"readOnlyHint": self.read_only, "destructiveHint": False,
                                "openWorldHint": self.open_world}}


_LOCAL_MODEL = {"type": "string", "minLength": 1, "maxLength": 300,
                "description": "api_name from list_local_models, or 'auto' to let Noelle choose."}

TOOLS: tuple[Tool, ...] = (
    Tool("hardware_info", "Hardware info",
         "Describe this machine for running local models: the accelerator (cuda, mps or cpu) and its name, the "
         "memory budget a model may use, peak memory bandwidth, and which inference engines are installed.",
         _object({}), read_only=True, open_world=False),
    Tool("list_local_models", "List local models",
         "List the models downloaded or built in Noelle on this machine. Pass a model's api_name as the 'model' "
         "argument of chat or embed; only models with on_disk true can run.",
         _object({}), read_only=True, open_world=False),
    Tool("search_models", "Search Hugging Face models",
         "Search the Hugging Face Hub and rate how well each model would run on this machine: a 0-100 fit score, a "
         "grade, a memory verdict and the recommended way to run it. Only the search text is sent to the Hub.",
         _object({
             "query": {"type": "string", "maxLength": 200,
                       "description": "Model name, publisher or keywords. An empty string lists the top models."},
             "task": {"type": "string", "enum": sorted(tasks.TASK_FAMILIES),
                      "description": "Hub pipeline tag to filter by, e.g. text-generation."},
             "sort": {"type": "string", "enum": ["trending", "downloads", "likes", "recent"], "default": "trending"},
             "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
         }, "query"), read_only=True, open_world=True),
    Tool("inspect_model", "Inspect a Hugging Face model",
         "Analyze one Hugging Face Hub repository and advise how to run it on this machine. Returns its capability "
         "(task, architecture, parameters, context length, file formats, quantization, chat template, tool "
         "calling, license, remote code) and an assessment: every run option with its memory need and status, the "
         "recommendation for the chosen profile, a 0-100 score with its breakdown, the largest context that fits "
         "and the expected speed.",
         _object({
             "repo_id": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": REPO_ID_PATTERN,
                         "description": "Repository id, e.g. Qwen/Qwen3-4B."},
             "profile": {"type": "string", "enum": list(advisor.PROFILES),
                         "description": "Runtime profile for the recommendation (default: the profile chosen in "
                                        "Noelle's settings, normally balanced)."},
         }, "repo_id"), read_only=True, open_world=True),
    Tool("chat", "Chat with a local model",
         "Generate a reply with a local model through Noelle's API server; prompts stay on this machine. Pass "
         "'prompt' for a single question or 'messages' for a conversation (with both, the prompt is appended as "
         "the last user message). Needs Noelle's API server to be running; the first call to a model is slower "
         "because the model worker loads it.",
         _object({
             "model": _LOCAL_MODEL,
             "prompt": {"type": "string", "minLength": 1},
             "messages": {"type": "array", "minItems": 1, "items": _object({
                 "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                 "content": {"type": "string"},
             }, "role", "content")},
             "system": {"type": "string", "description": "System prompt placed before the conversation."},
             "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
             "temperature": {"type": "number", "minimum": 0, "maximum": 2},
         }, "model"), read_only=False, open_world=False),
    Tool("embed", "Embed text locally",
         "Compute embedding vectors with a local embedding model through Noelle's API server; the text stays on "
         "this machine. Needs Noelle's API server to be running.",
         _object({
             "model": {**_LOCAL_MODEL, "description": "api_name of an embedding model from list_local_models."},
             "input": {"anyOf": [{"type": "string"},
                                 {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 16}],
                       "description": "One text, or up to 16 texts."},
         }, "model", "input"), read_only=False, open_world=False),
    Tool("list_knowledge_bases", "List knowledge bases",
         "List the document collections indexed in Noelle for retrieval (RAG). Needs Noelle's API server to be "
         "running.",
         _object({}), read_only=True, open_world=False),
    Tool("query_knowledge_base", "Search a knowledge base",
         "Find the passages of a Noelle knowledge base that best match a query (local vector search) and return "
         "them with their source and score. Needs Noelle's API server to be running.",
         _object({
             "collection": {"type": "string", "minLength": 1, "maxLength": 200,
                            "description": "Knowledge base name from list_knowledge_bases."},
             "query": {"type": "string", "minLength": 1},
             "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 4},
         }, "collection", "query"), read_only=False, open_world=False),
)


# ---------------------------------------------------------------------------
# argument validation (the JSON Schema subset the tool schemas use)
# ---------------------------------------------------------------------------

_TYPE_NAMES = {"object": "an object", "array": "an array", "string": "a string", "integer": "an integer",
               "number": "a number", "boolean": "true or false"}
_PY_TYPES: dict[str, type | tuple[type, ...]] = {"object": dict, "array": list, "string": str, "boolean": bool}


def _is_type(value: Any, kind: str) -> bool:
    if kind in ("integer", "number"):
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value) and (kind == "number" or value.is_integer())
    return isinstance(value, _PY_TYPES[kind])


def _bounds(low: Any, high: Any) -> str:
    if low is not None and high is not None:
        return f"between {low:g} and {high:g}"
    return f"at least {low:g}" if low is not None else f"at most {high:g}"


def validate(value: Any, schema: Mapping[str, Any], where: str = "arguments") -> str | None:
    """The first way ``value`` breaks ``schema``, or None when it conforms."""
    options = schema.get("anyOf")
    if options is not None:
        for option in options:
            if _is_type(value, option["type"]):
                return validate(value, option, where)
        return f"{where} must be " + " or ".join(_TYPE_NAMES[option["type"]] for option in options)
    kind = schema.get("type")
    if kind is not None and not _is_type(value, kind):
        return f"{where} must be {_TYPE_NAMES[kind]}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be one of: {', '.join(map(str, schema['enum']))}"
    if kind == "string":
        if len(value) < schema.get("minLength", 0):
            return f"{where} must not be empty"
        if len(value) > schema.get("maxLength", len(value)):
            return f"{where} must be at most {schema['maxLength']} characters"
        if "pattern" in schema and not re.search(schema["pattern"], value):
            return f"{where} has an invalid format"
    elif kind in ("integer", "number"):
        low, high = schema.get("minimum"), schema.get("maximum")
        if (low is not None and value < low) or (high is not None and value > high):
            return f"{where} must be {_bounds(low, high)}"
    elif kind == "array":
        if len(value) < schema.get("minItems", 0):
            return f"{where} must not be empty"
        if len(value) > schema.get("maxItems", len(value)):
            return f"{where} must have at most {schema['maxItems']} items"
        for index, item in enumerate(value):
            problem = validate(item, schema.get("items", {}), f"{where}[{index}]")
            if problem:
                return problem
    elif kind == "object":
        top = where == "arguments"
        for key in schema.get("required", ()):
            if key not in value:
                return f"missing required argument: {key}" if top else f"{where} is missing {key}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    return f"unexpected argument: {key[:64]}" if top else f"{where} has unexpected property: {key[:64]}"
        for key, sub in properties.items():
            if key in value:
                problem = validate(value[key], sub, key if top else f"{where}.{key}")
                if problem:
                    return problem
    return None


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _compact(value: Any, *, ascii_only: bool = False) -> str:
    return json.dumps(value, ensure_ascii=ascii_only, separators=(",", ":"), default=str)


def _log_failure(what: str, exc: BaseException) -> None:
    """Log the traceback but not the exception message, which can quote prompts or arguments."""
    frames = "".join(traceback.format_tb(exc.__traceback__)).rstrip()
    log.error("%s failed with %s\n%s", what, type(exc).__name__, frames)


def _hub_problem(exc: Exception, repo_id: str | None = None) -> str:
    from huggingface_hub.errors import GatedRepoError, OfflineModeIsEnabled, RepositoryNotFoundError

    if isinstance(exc, GatedRepoError):
        return (f"{repo_id} is gated: accept its terms on huggingface.co, then add your Hugging Face token in "
                "Noelle's settings.")
    if isinstance(exc, RepositoryNotFoundError):
        return (f"{repo_id} was not found on the Hugging Face Hub (private repositories need your Hugging Face "
                "token in Noelle's settings).")
    if isinstance(exc, OfflineModeIsEnabled):
        return "Hugging Face Hub access is disabled in this environment (HF_HUB_OFFLINE is set)."
    detail = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__
    return f"The Hugging Face Hub request failed: {detail}"


def _server_message(payload: Any) -> str | None:
    """The message of an OpenAI-style error body ``{"error": {"message": ...}}`` (or FastAPI's ``detail``)."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:500]
    if isinstance(payload.get("detail"), str):
        return payload["detail"][:500]
    return None


def _trusted(settings: Settings, repo_id: str) -> bool:
    """Remote code allowed for this repository in Noelle's settings, by repository or by publisher."""
    repos = settings.get("runtime.trusted_remote_code") or []
    publishers = settings.get("runtime.trusted_publishers") or []
    return repo_id in repos or ("/" in repo_id and repo_id.split("/")[0] in publishers)


def _json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


class McpServer:
    """One MCP session. ``transport`` replaces the HTTP transport to the API server (tests)."""

    def __init__(self, *, api_url: str | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self._api_url = api_url
        self._transport = transport
        self._tools = {tool.name: tool for tool in TOOLS}
        self._profile: advisor.HardwareProfile | None = None
        self._secrets: set[str] = set()

    # ------------------------------------------------------------- transport
    def serve(self, stdin: TextIO, stdout: TextIO) -> int:
        """Answer messages from ``stdin`` until it ends (or the client stops reading); returns 0."""
        while True:
            try:
                line = stdin.readline(MAX_MESSAGE_CHARS + 1)
            except (OSError, ValueError):  # stdin closed under us
                return 0
            if not line:
                return 0
            if len(line) > MAX_MESSAGE_CHARS and not line.endswith("\n"):
                self._discard_line(stdin)
                reply = _error(None, INVALID_REQUEST, f"Message too large (limit {MAX_MESSAGE_CHARS >> 20} MiB).")
            elif not line.strip():
                continue
            else:
                reply = self.handle_line(line)
            if reply is not None and not self._send(stdout, reply):
                return 0

    @staticmethod
    def _discard_line(stdin: TextIO) -> None:
        while True:
            try:
                chunk = stdin.readline(MAX_MESSAGE_CHARS)
            except (OSError, ValueError):
                return
            if not chunk or chunk.endswith("\n"):
                return

    def _send(self, stdout: TextIO, message: dict[str, Any]) -> bool:
        try:
            # ASCII-only on the wire: no encoding surprises on Windows pipes and no exotic line separators.
            line = _compact(self._scrub(message), ascii_only=True)
        except (ValueError, RecursionError) as exc:
            _log_failure("serializing a reply", exc)
            line = _compact(_error(message.get("id"), INTERNAL_ERROR, "Internal error: the reply could not be "
                                                                      "serialized."), ascii_only=True)
        try:
            stdout.write(line + "\n")
            stdout.flush()
        except (OSError, ValueError):  # the client closed the pipe
            return False
        return True

    def _scrub(self, value: Any) -> Any:
        """Replace the API key wherever it appears (it never should; this is a last line of defence)."""
        if not self._secrets:
            return value
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, REDACTED)
            return value
        if isinstance(value, dict):
            return {self._scrub(key): self._scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._scrub(item) for item in value]
        return value

    # ------------------------------------------------------------- JSON-RPC
    def handle_line(self, line: str) -> dict[str, Any] | None:
        try:
            message = json.loads(line)
        except (ValueError, RecursionError):
            return _error(None, PARSE_ERROR, "Parse error: each line must be one JSON-RPC message.")
        return self.handle(message)

    def handle(self, message: Any) -> dict[str, Any] | None:
        """The reply to one decoded message, or None (notifications and client responses get none)."""
        if isinstance(message, list):
            return _error(None, INVALID_REQUEST, "Batches are not supported: send one JSON-RPC message per line.")
        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST, "Invalid Request: a JSON-RPC message must be an object.")
        has_id = "id" in message
        msg_id = message.get("id")
        if has_id and not _valid_id(msg_id):
            return _error(None, INVALID_REQUEST, "Invalid Request: 'id' must be a string or an integer.")
        method = message.get("method")
        if method is None and ("result" in message or "error" in message):
            return None  # a response to a server request; this server sends none
        if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return _error(msg_id, INVALID_REQUEST, "Invalid Request: expected jsonrpc '2.0' and a string 'method'.")
        if not has_id:
            return None  # notifications (initialized, cancelled, ...) never get a reply
        params = message.get("params")
        try:
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise RpcError(INVALID_PARAMS, "'params' must be an object.")
            return _result(msg_id, self._dispatch(method, params))
        except RpcError as exc:
            return _error(msg_id, exc.code, exc.message)
        except Exception as exc:
            _log_failure(method[:80], exc)
            return _error(msg_id, INTERNAL_ERROR, "Internal error; see logs/mcp.log in Noelle's data folder.")

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [tool.definition() for tool in TOOLS]}
        if method == "tools/call":
            return self._call_tool(params)
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method[:80]}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        client = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        log.info("initialize: %s %s, protocol %s", str(client.get("name"))[:60], str(client.get("version"))[:30],
                 version)
        return {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "noelle", "version": __version__}, "instructions": INSTRUCTIONS}

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError(INVALID_PARAMS, "tools/call needs the tool's 'name'.")
        tool = self._tools.get(name)
        if tool is None:
            raise RpcError(INVALID_PARAMS, f"Unknown tool: {name[:80]}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        problem = validate(arguments, tool.input_schema)
        if problem:
            raise RpcError(INVALID_PARAMS, f"Invalid arguments for {name}: {problem}.")
        started = time.monotonic()
        try:
            data = getattr(self, f"_tool_{name}")(arguments)
        except RpcError:
            raise
        except ToolError as exc:
            result = self._tool_error(str(exc))
        except Exception as exc:
            _log_failure(f"tool {name}", exc)
            result = self._tool_error(f"{name} failed: {type(exc).__name__}: {exc}"[:600])
        else:
            data = self._scrub(data)
            result = {"content": [{"type": "text", "text": _compact(data)}], "structuredContent": data}
        log.info("tools/call %s: %s in %.0f ms", name, "error" if result.get("isError") else "ok",
                 (time.monotonic() - started) * 1000)
        return result

    def _tool_error(self, message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": self._scrub(message)}], "isError": True}

    # ------------------------------------------------------------- context
    @staticmethod
    def _settings() -> Settings:
        settings = get_settings()
        settings.reload()  # the desktop app may have changed the port, the API key or the Hub endpoint
        return settings

    def _hardware(self) -> advisor.HardwareProfile:
        if self._profile is None:
            from noelle.app.services import hardware_manager  # imports PySide6.QtCore; no QApplication needed

            headroom = float(self._settings().get("runtime.safety_headroom", 0.15))
            self._profile = hardware_manager.probe_profile(headroom=headroom)
        return self._profile

    @staticmethod
    def _hub(settings: Settings) -> hf_client.HFClient:
        return hf_client.HFClient(endpoint=settings.get("hub.endpoint") or None)

    # ------------------------------------------------------------- local tools
    def _tool_hardware_info(self, _args: dict[str, Any]) -> dict[str, Any]:
        hw = self._hardware()
        budget = hw.budget
        return {
            "device": hw.device,
            "name": hw.name,
            "memory_budget": {"total_bytes": budget.total_bytes, "usable_bytes": budget.usable_bytes,
                              "total": format_bytes(budget.total_bytes), "usable": format_bytes(budget.usable_bytes),
                              "headroom": budget.headroom, "system_ram_bytes": budget.system_ram_bytes},
            "bandwidth_gbps": hw.bandwidth_gbps,
            "backends": dict(hw.backends),
        }

    def _tool_list_local_models(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"models": [{"api_name": r.api_name, "repo_id": r.repo_id, "task": r.pipeline_tag, "family": r.family,
                            "precision": r.precision, "source": r.source, "size_bytes": r.size_bytes,
                            "on_disk": r.exists_on_disk()} for r in get_registry().list_models()]}

    # ------------------------------------------------------------- Hub tools
    def _tool_search_models(self, args: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        limit = int(args.get("limit", 10))
        pager = self._hub(settings).pager(pipeline_tag=args.get("task"), search=args["query"],
                                          sort=args.get("sort", "trending"), page_size=limit)
        try:
            found = pager.fetch_next()[:limit]
        except Exception as exc:
            raise ToolError(_hub_problem(exc)) from None
        hw = self._hardware()
        models = []
        for summary in found:
            assessment = advisor.assess(capabilities.analyze(summary), hw, trusted=_trusted(settings, summary.id))
            models.append({
                "id": summary.id,
                "task": summary.pipeline_tag,
                "params": format_params(summary.param_count) if summary.param_count else None,
                "downloads": summary.downloads,
                "likes": summary.likes,
                "license": summary.license,
                "fit": {"score": assessment.score, "grade": assessment.grade,
                        "verdict": assessment.compatibility.label,
                        "recommended": assessment.recommended.label if assessment.recommended else None},
            })
        return {"models": models, "hardware": {"device": hw.device, "name": hw.name, "usable_bytes": hw.usable_bytes}}

    def _tool_inspect_model(self, args: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        profile = args.get("profile") or settings.get("runtime.profile")
        if profile not in advisor.PROFILES:
            profile = "balanced"
        client = self._hub(settings)
        try:
            summary = client.model_info(args["repo_id"])
        except Exception as exc:
            raise ToolError(_hub_problem(exc, args["repo_id"])) from None
        repo_id = summary.id or args["repo_id"]  # the Hub's canonical spelling, as stored in the registry
        config = client.fetch_config(repo_id) or summary.config or None
        tokenizer_config = None
        if summary.family in (tasks.TEXT, tasks.VISION):  # only used to detect chat templates and tool calling
            tokenizer_config = _json_object(client.fetch_file_text(repo_id, "tokenizer_config.json",
                                                                   max_bytes=1_000_000))
        cap = capabilities.analyze(summary, config, tokenizer_config=tokenizer_config)
        registry = get_registry()
        local_variants = {r.precision: r.repo_id for r in registry.list_models()
                          if r.base_model == repo_id and r.source == "quantized" and r.exists_on_disk()}
        assessment = advisor.assess(cap, self._hardware(), config=config, profile=profile,
                                    measured=advisor.summarize_measurements(registry.measurements(repo_id)),
                                    local_variants=local_variants, trusted=_trusted(settings, repo_id))
        return {"capability": cap.to_dict(), "assessment": assessment.to_dict()}

    # ------------------------------------------------------------- API server tools
    def _tool_chat(self, args: dict[str, Any]) -> dict[str, Any]:
        if "prompt" not in args and "messages" not in args:
            raise RpcError(INVALID_PARAMS, "Invalid arguments for chat: pass 'prompt' or 'messages'.")
        messages = [{"role": "system", "content": args["system"]}] if args.get("system") else []
        messages += [{"role": m["role"], "content": m["content"]} for m in args.get("messages") or []]
        if "prompt" in args:
            messages.append({"role": "user", "content": args["prompt"]})
        body: dict[str, Any] = {"model": args["model"], "messages": messages, "stream": False}
        if "max_tokens" in args:
            body["max_tokens"] = int(args["max_tokens"])
        if "temperature" in args:
            body["temperature"] = float(args["temperature"])
        data = self._request("POST", "/v1/chat/completions", body)
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ToolError(UNEXPECTED_RESPONSE)
        return {"model": data.get("model") or args["model"], "content": message.get("content") or "",
                "finish_reason": choice.get("finish_reason"), "usage": data.get("usage")}

    def _tool_embed(self, args: dict[str, Any]) -> dict[str, Any]:
        data = self._request("POST", "/v1/embeddings", {"model": args["model"], "input": args["input"]})
        items = data.get("data")
        vectors = [item.get("embedding") for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        if not vectors or not all(isinstance(vector, list) for vector in vectors):
            raise ToolError(UNEXPECTED_RESPONSE)
        return {"model": data.get("model") or args["model"], "dimensions": len(vectors[0]), "vectors": vectors}

    def _tool_list_knowledge_bases(self, _args: dict[str, Any]) -> dict[str, Any]:
        items = self._request("GET", "/v1/knowledge").get("data")
        if not isinstance(items, list):
            raise ToolError(UNEXPECTED_RESPONSE)
        return {"knowledge_bases": items}

    def _tool_query_knowledge_base(self, args: dict[str, Any]) -> dict[str, Any]:
        body = {"collection": args["collection"], "query": args["query"], "top_k": int(args.get("top_k", 4))}
        data = self._request("POST", "/v1/knowledge/query", body)
        hits = data.get("data")
        if not isinstance(hits, list):
            raise ToolError(UNEXPECTED_RESPONSE)
        return {"collection": data.get("collection") or args["collection"], "hits": hits}

    # ------------------------------------------------------------- HTTP
    def _base_url(self, settings: Settings) -> str:
        """The API server's origin (plus any path prefix, without ``/v1``); loopback only."""
        raw = self._api_url
        if not raw:
            host = str(settings.get("server.host") or "127.0.0.1")
            raw = f"http://{f'[{host}]' if ':' in host else host}:{settings.get('server.port') or 8765}"
        try:
            url = httpx.URL(raw)
        except (httpx.InvalidURL, TypeError, ValueError):
            url = None
        if url is None or not url.scheme or not url.host:
            raise ToolError("The API server address must look like http://127.0.0.1:8765.")
        host = url.host.lower()
        if url.scheme not in ("http", "https") or host not in LOOPBACK_HOSTS:
            raise ToolError(f"Refusing to send requests to {host}: Noelle's MCP server only talks to an API server "
                            "on this machine (127.0.0.1, localhost or ::1), so prompts never leave it.")
        path = url.path.rstrip("/")
        path = path[:-3] if path.endswith("/v1") else path  # accept the OpenAI-style ".../v1" base too
        port = f":{url.port}" if url.port else ""
        return f"{url.scheme}://{f'[{host}]' if ':' in host else host}{port}{path}"

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = self._settings()
        base = self._base_url(settings)
        headers = {"Accept": "application/json"}
        key = str(settings.get("server.api_key") or "").strip()
        if key:
            if not _HEADER_SAFE.fullmatch(key):
                raise ToolError("The API key in Noelle's settings contains characters that cannot be sent in an "
                                "HTTP header. Set a new key on the API Server page.")
            if len(key) >= MIN_SCRUB_LENGTH:
                self._secrets.add(key)
            headers["Authorization"] = f"Bearer {key}"
        try:
            # trust_env=False: proxy variables must never route local prompts through a proxy.
            with httpx.Client(transport=self._transport, timeout=HTTP_TIMEOUT, trust_env=False,
                              follow_redirects=False) as client:
                response = client.request(method, base + path, json=body, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise ToolError(f"{NOT_RUNNING} (Tried {base}.)") from None
        except httpx.TimeoutException:
            raise ToolError(f"Noelle's API server did not answer within {HTTP_TIMEOUT.read:.0f} s.") from None
        except httpx.HTTPError as exc:  # exception text can quote request headers: report only the type
            raise ToolError(f"The request to Noelle's API server failed ({type(exc).__name__}).") from None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code == 401:
            raise ToolError("Noelle's API server rejected the request: its API key does not match server.api_key "
                            "in Noelle's settings (API Server page).")
        if not response.is_success:
            detail = _server_message(payload) or response.reason_phrase or "no details"
            raise ToolError(f"Noelle's API server answered HTTP {response.status_code}: {detail}")
        if not isinstance(payload, dict):
            raise ToolError(UNEXPECTED_RESPONSE)
        return payload


def serve(stdin: TextIO, stdout: TextIO, *, api_url: str | None = None,
          transport: httpx.BaseTransport | None = None) -> int:
    """Run one MCP session over text streams until ``stdin`` ends; returns the exit code."""
    return McpServer(api_url=api_url, transport=transport).serve(stdin, stdout)


# ---------------------------------------------------------------------------
# process entry point
# ---------------------------------------------------------------------------


def client_config(api_url: str | None = None) -> dict[str, Any]:
    """The ``mcpServers`` entry that starts this installation's MCP server."""
    from noelle.runtime.client import worker_command

    command = worker_command()  # frozen app: [exe]; source checkout: [python, "-m", "noelle"]
    args = [*command[1:], "mcp"]
    if api_url:
        args += ["--api-url", api_url]
    return {"mcpServers": {"noelle": {"command": command[0], "args": args}}}


def _configure_logging() -> None:
    """MCP protocol frames own stdout; logs go to ``logs/mcp.log`` and (warnings+) stderr."""
    from noelle import logging_setup

    logging_setup.configure_mcp_logging()
    log.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> int:
    """``noelle mcp [--api-url URL] [--print-config]``."""
    api_url = getattr(args, "api_url", None) or None
    if getattr(args, "print_config", False):
        print(json.dumps(client_config(api_url), indent=2))
        return 0
    _configure_logging()
    # Split lines on "\n" only (a "\r" before it is JSON whitespace) and write bare "\n" on every OS.
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="\n")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n", write_through=True)
    sys.stdout = sys.stderr  # stray print() output from libraries must never reach the protocol stream
    log.info("Noelle MCP server %s started (pid %s)", __version__, os.getpid())
    try:
        return serve(stdin, stdout, api_url=api_url)
    except KeyboardInterrupt:
        return 130
    finally:
        log.info("Noelle MCP server stopped")
