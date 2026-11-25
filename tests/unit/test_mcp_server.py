"""noelle mcp: JSON-RPC framing, tool schemas and every tool against fakes (no network, no real API server)."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import socket
import socketserver
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from noelle import __version__, paths
from noelle.app.services import hardware_manager, hf_client
from noelle.app.services.hf_client import ModelSummary
from noelle.app.services.model_registry import ModelRecord, get_registry
from noelle.config import get_settings
from noelle.runtime import advisor
from noelle.runtime import mcp_server as mcp
from noelle.runtime.preflight import GB, MemoryBudget

ROOT = Path(__file__).resolve().parents[2]
KEY = "nl-test-key-0123456789abcdefghijkl"
CANARY = "PROMPT-CANARY-7f3a"
EXPECTED_TOOLS = {"hardware_info", "list_local_models", "search_models", "inspect_model", "chat", "embed",
                  "list_knowledge_bases", "query_knowledge_base"}
PROFILE = advisor.HardwareProfile("mps", "Apple M2 Pro GPU", MemoryBudget(24 * GB, 0.15, 32 * GB, "mps"),
                                  {"transformers": True, "mlx": True, "llama.cpp": False, "llama.cpp-tools": False},
                                  200.0)
CONFIG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "max_position_embeddings": 32768,
          "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def req(msg_id: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def session(*messages: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Run serve() over in-memory streams; strings are sent verbatim, anything else as one JSON line."""
    text = "".join(m if isinstance(m, str) else json.dumps(m) + "\n" for m in messages)
    out = io.StringIO()
    assert mcp.serve(io.StringIO(text), out, **kwargs) == 0
    raw = out.getvalue()
    assert raw.isascii() and (raw == "" or raw.endswith("\n"))
    return [json.loads(line) for line in raw.splitlines()]


def call(name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """One tools/call; returns its result (it must not be a protocol error)."""
    [reply] = session(req(1, "tools/call", {"name": name, "arguments": arguments or {}}), **kwargs)
    assert "result" in reply, reply
    return reply["result"]


def ok(result: dict[str, Any]) -> dict[str, Any]:
    """Structured content of a successful result; its text block must carry the same JSON."""
    assert not result.get("isError"), result
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    return result["structuredContent"]


def failed(result: dict[str, Any]) -> str:
    assert result.get("isError") is True, result
    assert "structuredContent" not in result
    return result["content"][0]["text"]


class FakeApi:
    """Stands in for Noelle's API server through httpx.MockTransport and records each request."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        self.transport = httpx.MockTransport(record)


def completion(content: str = "Hello!", model: str = "acme/chat-1B") -> dict[str, Any]:
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


def summary(repo_id: str, params: int, **extra: Any) -> ModelSummary:
    return ModelSummary(id=repo_id, pipeline_tag="text-generation", downloads=1000, likes=50, license="apache-2.0",
                        param_count=params, siblings=[{"name": "model.safetensors", "size": params * 2}], **extra)


class FakePager:
    def __init__(self, results: list[ModelSummary] | Exception) -> None:
        self.results = results

    def fetch_next(self) -> list[ModelSummary]:
        if isinstance(self.results, Exception):
            raise self.results
        return list(self.results)


class Gated(GatedRepoError):
    def __init__(self) -> None:  # the real constructor needs an HTTP response
        Exception.__init__(self, "403")


class Missing(RepositoryNotFoundError):
    def __init__(self) -> None:
        Exception.__init__(self, "404")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def hardware(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A fixed Apple Silicon profile instead of probing this machine; records the requested headroom."""
    headrooms: list[float] = []

    def probe(headroom: float = 0.15, backends: dict[str, bool] | None = None) -> advisor.HardwareProfile:
        headrooms.append(headroom)
        return PROFILE

    monkeypatch.setattr(hardware_manager, "probe_profile", probe)
    return headrooms


# ---------------------------------------------------------------------------
# JSON-RPC framing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("requested", "expected"), [("2025-06-18", "2025-06-18"), ("2025-03-26", "2025-03-26"),
                                                     ("2024-11-05", "2024-11-05"), ("2099-01-01", "2025-06-18"),
                                                     (None, "2025-06-18")])
def test_initialize_negotiates_the_protocol_version(requested: str | None, expected: str) -> None:
    params: dict[str, Any] = {"capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}}
    if requested:
        params["protocolVersion"] = requested
    [reply] = session(req(1, "initialize", params))
    result = reply["result"]
    assert reply["id"] == 1 and result["protocolVersion"] == expected
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"] == {"name": "noelle", "version": __version__}
    assert "API server" in result["instructions"]


def test_notifications_and_client_responses_get_no_reply() -> None:
    replies = session({"jsonrpc": "2.0", "method": "notifications/initialized"},
                      {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 7}},
                      {"jsonrpc": "2.0", "method": "no/such/notification", "params": [1]},
                      {"jsonrpc": "2.0", "id": 5, "result": {}},
                      req(0, "ping"), req("abc", "ping", {"_meta": {"progressToken": 1}}))
    assert replies == [{"jsonrpc": "2.0", "id": 0, "result": {}}, {"jsonrpc": "2.0", "id": "abc", "result": {}}]


def test_protocol_errors_keep_the_session_alive() -> None:
    replies = session("this is not json\n",
                      "[" * 100_000 + "\n",  # nesting deep enough to raise RecursionError in json.loads
                      '[{"jsonrpc": "2.0", "id": 1, "method": "ping"}]\n',
                      "42\n",
                      {"jsonrpc": "2.0", "id": {"x": 1}, "method": "ping"},
                      {"jsonrpc": "2.0", "id": True, "method": "ping"},
                      {"id": 3, "method": "ping"},
                      {"jsonrpc": "2.0", "id": 4, "method": 7},
                      req(5, "resources/list"),
                      {"jsonrpc": "2.0", "id": 6, "method": "ping", "params": [1]},
                      req(7, "ping"))
    assert [(r["id"], r.get("error", {}).get("code")) for r in replies] == [
        (None, -32700), (None, -32700), (None, -32600), (None, -32600), (None, -32600), (None, -32600),
        (3, -32600), (4, -32600), (5, -32601), (6, -32602), (7, None)]


def test_oversized_message_is_rejected_and_skipped() -> None:
    big = json.dumps(req(1, "ping", {"pad": "x" * mcp.MAX_MESSAGE_CHARS}))
    first, second = session(big + "\n", req(2, "ping"))
    assert first["id"] is None and first["error"]["code"] == -32600 and "too large" in first["error"]["message"]
    assert second == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_crlf_blank_lines_and_a_last_line_without_newline() -> None:
    text = json.dumps(req(1, "ping")) + "\r\n\r\n   \n" + json.dumps(req(2, "ping"))
    out = io.StringIO()
    assert mcp.serve(io.StringIO(text), out) == 0
    assert [json.loads(line)["id"] for line in out.getvalue().splitlines()] == [1, 2]


def test_serve_returns_quietly_when_the_client_stops_reading() -> None:
    class ClosedPipe(io.StringIO):
        def write(self, _text: str) -> int:
            raise BrokenPipeError

    messages = json.dumps(req(1, "ping")) + "\n" + json.dumps(req(2, "ping")) + "\n"
    assert mcp.serve(io.StringIO(messages), ClosedPipe()) == 0


# ---------------------------------------------------------------------------
# tool catalogue and argument validation
# ---------------------------------------------------------------------------


def test_tools_list_describes_every_tool() -> None:
    [reply] = session(req(1, "tools/list", {}))
    tools = {tool["name"]: tool for tool in reply["result"]["tools"]}
    assert set(tools) == EXPECTED_TOOLS
    for name, tool in tools.items():
        schema = tool["inputSchema"]
        assert tool["description"] and tool["title"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert isinstance(schema["properties"], dict)
        hints = tool["annotations"]
        assert hints["destructiveHint"] is False and isinstance(hints["readOnlyHint"], bool)
        assert hints["openWorldHint"] is (name in {"search_models", "inspect_model"})
    assert tools["hardware_info"]["annotations"]["readOnlyHint"] is True
    assert tools["chat"]["annotations"]["readOnlyHint"] is False  # it may load a model into the worker
    assert tools["search_models"]["inputSchema"]["required"] == ["query"]
    assert tools["search_models"]["inputSchema"]["properties"]["limit"]["maximum"] == 20
    assert tools["inspect_model"]["inputSchema"]["properties"]["profile"]["enum"] == list(advisor.PROFILES)
    assert tools["chat"]["inputSchema"]["required"] == ["model"]
    assert tools["chat"]["inputSchema"]["properties"]["max_tokens"]["maximum"] == 4096
    assert tools["embed"]["inputSchema"]["required"] == ["model", "input"]
    assert tools["query_knowledge_base"]["inputSchema"]["required"] == ["collection", "query"]


@pytest.mark.parametrize(("params", "message"), [
    ({"name": "download_model", "arguments": {}}, "Unknown tool: download_model"),
    ({"arguments": {}}, "'name'"),
    ({"name": "hardware_info", "arguments": [1]}, "arguments must be an object"),
    ({"name": "hardware_info", "arguments": {"verbose": True}}, "unexpected argument: verbose"),
    ({"name": "search_models", "arguments": {}}, "missing required argument: query"),
    ({"name": "search_models", "arguments": {"query": "x", "limit": 50}}, "limit must be between 1 and 20"),
    ({"name": "search_models", "arguments": {"query": "x", "limit": True}}, "limit must be an integer"),
    ({"name": "search_models", "arguments": {"query": "x", "sort": "newest"}}, "sort must be one of"),
    ({"name": "inspect_model", "arguments": {"repo_id": "../etc/passwd"}}, "repo_id has an invalid format"),
    ({"name": "inspect_model", "arguments": {"repo_id": "a/b", "profile": "turbo"}}, "profile must be one of"),
    ({"name": "chat", "arguments": {"model": "m"}}, "pass 'prompt' or 'messages'"),
    ({"name": "chat", "arguments": {"model": "m", "prompt": "hi", "temperature": 3}},
     "temperature must be between 0 and 2"),
    ({"name": "chat", "arguments": {"model": "m", "prompt": "hi", "max_tokens": 5000}},
     "max_tokens must be between 1 and 4096"),
    ({"name": "chat", "arguments": {"model": "m", "messages": [{"role": "tool", "content": "x"}]}},
     "messages[0].role must be one of"),
    ({"name": "chat", "arguments": {"model": "m", "messages": [{"role": "user"}]}}, "messages[0] is missing content"),
    ({"name": "embed", "arguments": {"model": "m", "input": ["t"] * 17}}, "input must have at most 16 items"),
    ({"name": "embed", "arguments": {"model": "m", "input": 5}}, "input must be a string or an array"),
    ({"name": "query_knowledge_base", "arguments": {"collection": "docs", "query": ""}}, "query must not be empty"),
])
def test_invalid_tool_calls_are_rejected_with_invalid_params(params: dict[str, Any], message: str) -> None:
    api = FakeApi(lambda request: httpx.Response(500))
    [reply] = session(req(9, "tools/call", params), transport=api.transport)
    assert reply["id"] == 9 and reply["error"]["code"] == -32602
    assert message in reply["error"]["message"]
    assert api.requests == []


# ---------------------------------------------------------------------------
# local tools
# ---------------------------------------------------------------------------


def test_hardware_info_reports_the_probed_profile(hardware: list[float]) -> None:
    get_settings().set("runtime.safety_headroom", 0.25)
    # no "arguments" at all, plus the _meta clients attach for progress tokens
    [reply] = session(req(1, "tools/call", {"name": "hardware_info", "_meta": {"progressToken": "p1"}}))
    data = ok(reply["result"])
    assert hardware == [0.25]
    assert data["device"] == "mps" and data["name"] == "Apple M2 Pro GPU"
    assert data["memory_budget"]["total_bytes"] == 24 * GB
    assert data["memory_budget"]["usable_bytes"] == PROFILE.usable_bytes
    assert data["memory_budget"]["usable"].endswith("GB")
    assert data["bandwidth_gbps"] == 200.0 and data["backends"]["mlx"] is True


def test_list_local_models_reads_the_registry(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    registry = get_registry()
    registry.upsert_model(ModelRecord(repo_id="acme/chat-1B", pipeline_tag="text-generation",
                                      local_path=str(snapshot), size_bytes=2_000_000))
    registry.upsert_model(ModelRecord(repo_id="acme/chat-1B-mlx-4bit", pipeline_tag="text-generation",
                                      local_path=str(tmp_path / "deleted"), precision="mlx-4bit", source="quantized",
                                      base_model="acme/chat-1B"))
    registry.upsert_model(ModelRecord(repo_id="acme/embedder", pipeline_tag="feature-extraction",
                                      local_path=str(snapshot), precision="fp16"))
    models = {m["api_name"]: m for m in ok(call("list_local_models"))["models"]}
    assert models["acme/chat-1B"] == {"api_name": "acme/chat-1B", "repo_id": "acme/chat-1B",
                                      "task": "text-generation", "family": "text", "precision": "auto",
                                      "source": "hub", "size_bytes": 2_000_000, "on_disk": True}
    assert models["acme/chat-1B-mlx-4bit"]["on_disk"] is False
    assert models["acme/embedder:fp16"]["family"] == "embedding"


# ---------------------------------------------------------------------------
# Hub tools
# ---------------------------------------------------------------------------


def test_search_models_scores_each_result_for_this_machine(hardware: list[float],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, dict[str, Any]]] = []
    results = [summary("acme/tiny-chat-0.5B", 500_000_000), summary("acme/huge-405B", 405_000_000_000),
               summary("acme/third-1B", 1_000_000_000)]

    def pager(self: hf_client.HFClient, **kwargs: Any) -> FakePager:
        calls.append((self.endpoint, kwargs))
        return FakePager(results)

    monkeypatch.setattr(hf_client.HFClient, "pager", pager)
    get_settings().set("hub.endpoint", "https://hub.example")
    data = ok(call("search_models", {"query": "chat", "task": "text-generation", "sort": "downloads", "limit": 2}))
    assert calls == [("https://hub.example", {"pipeline_tag": "text-generation", "search": "chat",
                                              "sort": "downloads", "page_size": 2})]
    tiny, huge = data["models"]
    assert (tiny["id"], tiny["task"], tiny["params"]) == ("acme/tiny-chat-0.5B", "text-generation", "500M")
    assert (tiny["downloads"], tiny["likes"], tiny["license"]) == (1000, 50, "apache-2.0")
    assert tiny["fit"]["grade"] in {"Excellent", "Good", "Fair", "Poor", "Not recommended"}
    assert tiny["fit"]["verdict"] and tiny["fit"]["recommended"]
    assert tiny["fit"]["score"] > huge["fit"]["score"]
    assert data["hardware"] == {"device": "mps", "name": "Apple M2 Pro GPU", "usable_bytes": PROFILE.usable_bytes}

    ok(call("search_models", {"query": ""}))
    assert calls[-1][1] == {"pipeline_tag": None, "search": "", "sort": "trending", "page_size": 10}


def test_search_models_reports_hub_failures(hardware: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_client.HFClient, "pager",
                        lambda self, **kwargs: FakePager(ConnectionError("Name or service not known")))
    assert failed(call("search_models", {"query": "x"})) == \
        "The Hugging Face Hub request failed: Name or service not known"


def test_inspect_model_combines_capability_and_advice(hardware: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[tuple[str, str]] = []
    template = "{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"

    def model_info(self: hf_client.HFClient, repo_id: str) -> ModelSummary:
        fetched.append(("info", repo_id))
        return summary("Acme/Tiny-Chat-1.5B", 1_500_000_000, tags=["text-generation", "conversational"])

    def fetch_config(self: hf_client.HFClient, repo_id: str) -> dict[str, Any]:
        fetched.append(("config.json", repo_id))
        return dict(CONFIG)

    def fetch_file_text(self: hf_client.HFClient, repo_id: str, filename: str, max_bytes: int = 0) -> str:
        fetched.append((filename, repo_id))
        return json.dumps({"chat_template": template})

    monkeypatch.setattr(hf_client.HFClient, "model_info", model_info)
    monkeypatch.setattr(hf_client.HFClient, "fetch_config", fetch_config)
    monkeypatch.setattr(hf_client.HFClient, "fetch_file_text", fetch_file_text)
    get_registry().add_evaluation("run-1", 1, "Acme/Tiny-Chat-1.5B", prompt="p", system_prompt=None, params={},
                                  output="o", status="finished",
                                  metrics={"precision": "fp16", "tokens_per_second": 42.0, "ttft_ms": 120.0})

    data = ok(call("inspect_model", {"repo_id": "acme/tiny-chat-1.5b"}))
    # later calls use the Hub's canonical spelling, which is also how the registry stores measurements
    assert fetched == [("info", "acme/tiny-chat-1.5b"), ("config.json", "Acme/Tiny-Chat-1.5B"),
                       ("tokenizer_config.json", "Acme/Tiny-Chat-1.5B")]
    capability, assessment = data["capability"], data["assessment"]
    assert capability["model_id"] == "Acme/Tiny-Chat-1.5B" and capability["architecture"] == "LlamaForCausalLM"
    assert capability["context_length"] == 32768
    assert capability["chat_template"] is True and capability["tool_calling"] is True
    assert assessment["profile"] == "balanced" and 0 <= assessment["score"] <= 100
    assert assessment["recommended"]["precision"] == "fp16"
    assert assessment["speed"] == "42.0 tok/s measured on this machine"
    assert assessment["hardware"]["device"] == "mps"

    get_settings().set("runtime.profile", "low_memory")
    assert ok(call("inspect_model", {"repo_id": "Acme/Tiny-Chat-1.5B"}))["assessment"]["profile"] == "low_memory"
    assert ok(call("inspect_model", {"repo_id": "Acme/Tiny-Chat-1.5B", "profile": "fast"}))["assessment"][
        "profile"] == "fast"


@pytest.mark.parametrize(("error", "message"), [
    (Gated(), "acme/private is gated"),
    (Missing(), "acme/private was not found on the Hugging Face Hub"),
    (OSError("connection reset by peer"), "The Hugging Face Hub request failed: connection reset by peer"),
])
def test_inspect_model_reports_hub_errors(hardware: list[float], monkeypatch: pytest.MonkeyPatch, error: Exception,
                                          message: str) -> None:
    def model_info(self: hf_client.HFClient, repo_id: str) -> ModelSummary:
        raise error

    monkeypatch.setattr(hf_client.HFClient, "model_info", model_info)
    assert message in failed(call("inspect_model", {"repo_id": "acme/private"}))


# ---------------------------------------------------------------------------
# API server tools
# ---------------------------------------------------------------------------


def test_chat_posts_to_the_local_api_server() -> None:
    api = FakeApi(lambda request: httpx.Response(200, json=completion("Grüß dich 👋", model="acme/routed")))
    result = call("chat", {"model": "auto", "prompt": "Hi", "system": "Be brief.", "max_tokens": 64,
                           "temperature": 0.2}, transport=api.transport)
    assert ok(result) == {"model": "acme/routed", "content": "Grüß dich 👋", "finish_reason": "stop",
                          "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    assert "Grüß dich 👋" in result["content"][0]["text"]  # readable for the model, not \u escapes
    [request] = api.requests
    assert request.method == "POST" and str(request.url) == "http://127.0.0.1:8765/v1/chat/completions"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {"model": "auto", "stream": False, "max_tokens": 64, "temperature": 0.2,
                                           "messages": [{"role": "system", "content": "Be brief."},
                                                        {"role": "user", "content": "Hi"}]}


def test_chat_uses_the_settings_port_and_key_and_appends_the_prompt() -> None:
    settings = get_settings()
    settings.set("server.port", 9123)
    settings.set("server.api_key", KEY)
    api = FakeApi(lambda request: httpx.Response(200, json=completion()))
    history = [{"role": "user", "content": "Name a colour."}, {"role": "assistant", "content": "Blue."}]
    ok(call("chat", {"model": "acme/chat-1B", "messages": history, "prompt": "Another?"}, transport=api.transport))
    [request] = api.requests
    assert str(request.url) == "http://127.0.0.1:9123/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content)["messages"] == [*history, {"role": "user", "content": "Another?"}]


def test_chat_explains_when_the_api_server_is_not_running() -> None:
    get_settings().set("server.port", free_port())  # nothing listens there: a real connection refusal
    assert failed(call("chat", {"model": "auto", "prompt": "Hi"})).startswith(mcp.NOT_RUNNING)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert failed(call("list_knowledge_bases", transport=FakeApi(refuse).transport)).startswith(mcp.NOT_RUNNING)


@pytest.mark.parametrize("url", ["http://example.com:8765", "http://192.168.1.10:8765", "http://127.0.0.2:8765",
                                 "http://127.0.0.1.nip.io:8765", "http://127.0.0.1@evil.example:8765",
                                 "http://[::ffff:127.0.0.1]:8765", "ftp://127.0.0.1:8765", "127.0.0.1:8765"])
def test_non_loopback_api_urls_are_refused(url: str) -> None:
    api = FakeApi(lambda request: httpx.Response(200, json=completion()))
    text = failed(call("chat", {"model": "auto", "prompt": "secret plans"}, api_url=url, transport=api.transport))
    assert text.startswith(("Refusing to send requests", "The API server address must look like"))
    assert api.requests == [] and "secret plans" not in text


@pytest.mark.parametrize(("url", "expected"), [("http://localhost:8765/v1", "http://localhost:8765/v1/embeddings"),
                                               ("http://[::1]:9000/", "http://[::1]:9000/v1/embeddings"),
                                               ("http://127.0.0.1:8765", "http://127.0.0.1:8765/v1/embeddings")])
def test_loopback_api_urls_are_accepted(url: str, expected: str) -> None:
    body = {"object": "list", "model": "acme/embedder",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, -0.5]}]}
    api = FakeApi(lambda request: httpx.Response(200, json=body))
    ok(call("embed", {"model": "acme/embedder", "input": "hello"}, api_url=url, transport=api.transport))
    assert [str(request.url) for request in api.requests] == [expected]


@pytest.mark.parametrize(("response", "message"), [
    (httpx.Response(404, json={"error": {"message": "The model 'nope' does not exist.", "code": "model_not_found"}}),
     "Noelle's API server answered HTTP 404: The model 'nope' does not exist."),
    (httpx.Response(401, json={"error": {"message": "Invalid or missing API key."}}),
     "does not match server.api_key"),
    (httpx.Response(307, headers={"location": "http://example.com/v1/chat/completions"}), "answered HTTP 307"),
    (httpx.Response(200, text="<html>"), mcp.UNEXPECTED_RESPONSE),
    (httpx.Response(200, json={"choices": []}), mcp.UNEXPECTED_RESPONSE),
])
def test_api_server_errors_become_tool_errors(response: httpx.Response, message: str) -> None:
    api = FakeApi(lambda request: response)
    assert message in failed(call("chat", {"model": "nope", "prompt": "Hi"}, transport=api.transport))
    assert len(api.requests) == 1  # redirects are never followed


def test_api_server_timeouts_are_reported() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    text = failed(call("embed", {"model": "m", "input": "x"}, transport=FakeApi(slow).transport))
    assert text == "Noelle's API server did not answer within 300 s."


def test_embed_returns_vectors() -> None:
    body = {"object": "list", "model": "acme/embedder", "usage": {"prompt_tokens": 2, "total_tokens": 2},
            "data": [{"object": "embedding", "index": i, "embedding": [float(i), 0.5, -0.5]} for i in range(2)]}
    api = FakeApi(lambda request: httpx.Response(200, json=body))
    data = ok(call("embed", {"model": "acme/embedder", "input": ["a", "b"]}, transport=api.transport))
    assert data == {"model": "acme/embedder", "dimensions": 3, "vectors": [[0.0, 0.5, -0.5], [1.0, 0.5, -0.5]]}
    assert json.loads(api.requests[0].content) == {"model": "acme/embedder", "input": ["a", "b"]}


def test_knowledge_base_tools_follow_the_api_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/knowledge":
            return httpx.Response(200, json={"object": "list", "data": [{"name": "handbook", "documents": 3}]})
        if request.method == "POST" and request.url.path == "/v1/knowledge/query":
            body = json.loads(request.content)
            hits = [{"text": "Vacation is 25 days.", "score": 0.91, "source": "handbook.pdf", "page": 4}]
            return httpx.Response(200, json={"object": "list", "collection": body["collection"], "data": hits})
        return httpx.Response(404, json={"error": {"message": "Not Found"}})

    api = FakeApi(handler)
    assert ok(call("list_knowledge_bases", transport=api.transport)) == {
        "knowledge_bases": [{"name": "handbook", "documents": 3}]}
    hits = ok(call("query_knowledge_base", {"collection": "handbook", "query": "vacation days"},
                   transport=api.transport))
    assert hits["collection"] == "handbook" and hits["hits"][0]["source"] == "handbook.pdf"
    assert json.loads(api.requests[1].content) == {"collection": "handbook", "query": "vacation days", "top_k": 4}
    ok(call("query_knowledge_base", {"collection": "handbook", "query": "q", "top_k": 3}, transport=api.transport))
    assert json.loads(api.requests[2].content)["top_k"] == 3


def test_api_key_never_appears_in_output_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    get_settings().set("server.api_key", KEY)

    def echo(request: httpx.Request) -> httpx.Response:  # a misbehaving server that echoes credentials back
        auth = request.headers.get("authorization", "")
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=completion(f"you sent {auth}"))
        if request.url.path == "/v1/embeddings":
            return httpx.Response(400, json={"error": {"message": f"bad credentials {auth}"}})
        if request.url.path == "/v1/knowledge/query":
            raise httpx.LocalProtocolError(f"Illegal header value {auth!r}")
        return httpx.Response(401, json={"error": {"message": f"Invalid key {auth}"}})

    api = FakeApi(echo)
    replies = session(req(1, "tools/call", {"name": "chat", "arguments": {"model": "auto", "prompt": CANARY}}),
                      req(2, "tools/call", {"name": "embed", "arguments": {"model": "m", "input": CANARY}}),
                      req(3, "tools/call", {"name": "query_knowledge_base",
                                            "arguments": {"collection": "c", "query": CANARY}}),
                      req(4, "tools/call", {"name": "list_knowledge_bases"}), transport=api.transport)
    assert [request.headers["authorization"] for request in api.requests] == [f"Bearer {KEY}"] * 4
    assert KEY not in json.dumps(replies, ensure_ascii=False)
    assert ok(replies[0]["result"])["content"] == f"you sent Bearer {mcp.REDACTED}"
    assert all(reply["result"]["isError"] for reply in replies[1:])
    assert KEY not in caplog.text and CANARY not in caplog.text
    assert "tools/call chat: ok" in caplog.text


def test_api_key_with_unsendable_characters_is_not_echoed() -> None:
    get_settings().set("server.api_key", "bad key\twith-0123456789")
    api = FakeApi(lambda request: httpx.Response(200, json=completion()))
    text = failed(call("chat", {"model": "auto", "prompt": "hi"}, transport=api.transport))
    assert "cannot be sent in an HTTP header" in text and "bad key" not in text
    assert api.requests == []


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def test_print_config_points_at_this_installation(capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from noelle.runtime.client import worker_command

    expected_cmd = worker_command()[0]
    assert mcp.main(argparse.Namespace(print_config=True, api_url=None)) == 0
    assert json.loads(capsys.readouterr().out) == {
        "mcpServers": {"noelle": {"command": expected_cmd, "args": ["-m", "noelle", "mcp"]}}}
    monkeypatch.setattr(sys, "frozen", True, raising=False)  # PyInstaller build: the executable is the CLI
    assert mcp.main(argparse.Namespace(print_config=True, api_url="http://127.0.0.1:9000")) == 0
    assert json.loads(capsys.readouterr().out)["mcpServers"]["noelle"] == {
        "command": sys.executable, "args": ["mcp", "--api-url", "http://127.0.0.1:9000"]}


def run_cli(payload: bytes, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")  # NOELLE_HOME is the test's temporary home
    return subprocess.run([sys.executable, "-m", "noelle", "mcp", *args], input=payload, capture_output=True,
                          cwd=ROOT, env=env, timeout=120)


def test_cli_speaks_mcp_over_stdio() -> None:
    messages = [req(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "pytest", "version": "1"}}),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                req(2, "tools/list")]
    proc = run_cli(b"".join(json.dumps(m).encode() + b"\r\n" for m in messages))  # Windows-style line endings
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"\r" not in proc.stdout
    replies = [json.loads(line) for line in proc.stdout.decode().split("\n") if line]
    assert [reply["id"] for reply in replies] == [1, 2]
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert {tool["name"] for tool in replies[1]["result"]["tools"]} == EXPECTED_TOOLS
    assert b"Traceback" not in proc.stderr
    assert "initialize: pytest 1" in (paths.logs_dir() / "mcp.log").read_text(encoding="utf-8")


class _EchoHandler(BaseHTTPRequestHandler):
    """A fake API server that answers chat and echoes the Authorization header back everywhere."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length") or 0))
        auth = self.headers.get("authorization", "")
        if self.path == "/v1/chat/completions":
            self._reply(200, completion(f"you sent {auth}"))
        else:
            self._reply(400, {"error": {"message": f"rejected {auth}"}})

    def do_GET(self) -> None:
        self._reply(401, {"error": {"message": f"Invalid key {self.headers.get('authorization')}"}})

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args: Any) -> None:
        pass


class _LoopbackServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), a reverse-DNS lookup that can stall on CI runners.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


@pytest.fixture
def echo_server() -> Iterator[str]:
    server = _LoopbackServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_cli_never_writes_the_api_key_or_prompts_anywhere(echo_server: str) -> None:
    get_settings().set("server.api_key", KEY)  # the child reads the same settings.json
    calls = [("chat", {"model": "auto", "prompt": CANARY}), ("embed", {"model": "m", "input": [CANARY]}),
             ("list_knowledge_bases", {})]
    payload = b"".join(json.dumps(req(i, "tools/call", {"name": name, "arguments": arguments})).encode() + b"\n"
                       for i, (name, arguments) in enumerate(calls))
    proc = run_cli(payload, "--api-url", echo_server)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    replies = [json.loads(line) for line in proc.stdout.decode().splitlines()]
    assert ok(replies[0]["result"])["content"] == f"you sent Bearer {mcp.REDACTED}"
    assert replies[1]["result"]["isError"] and replies[2]["result"]["isError"]
    log_text = (paths.logs_dir() / "mcp.log").read_text(encoding="utf-8")
    stderr = proc.stderr.decode(errors="replace")
    for blob in (proc.stdout.decode(), stderr, log_text):
        assert KEY not in blob
    for blob in (stderr, log_text):
        assert CANARY not in blob
    assert "tools/call chat: ok" in log_text
