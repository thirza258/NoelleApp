"""server_service against a fake ZeroMQ worker (no models, no torch)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
import zmq
from fastapi.testclient import TestClient

from noelle import paths
from noelle.app.services.model_registry import ModelRecord, ModelRegistry
from noelle.runtime import server_service as srv
from noelle.runtime.protocol import Command, ErrorCode, Event, error_payload, make_event, parse_request

TOKEN = "worker-secret"


LOCAL = "http://127.0.0.1"  # the API only answers loopback Host headers


class FakeWorker:
    """ROUTER socket on a thread answering like the real worker, keyed on ``model_id``."""

    def __init__(self, token: str = TOKEN) -> None:
        self.token = token
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.requests: list[dict[str, Any]] = []
        self.stops: list[str | None] = []
        self.uploads: list[dict[str, Any]] = []
        self._slow: dict[str, dict[str, Any]] = {}
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"tcp://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._running = False
        self._thread.join(5)

    def of(self, cmd: Command) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["cmd"] == cmd]

    # -- loop ---------------------------------------------------------------
    def _send(self, identity: bytes, rid: str | None, event: Event, payload: dict[str, Any] | None = None) -> None:
        self.sock.send_multipart([identity, make_event(rid, event, payload)])

    def _loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        try:
            while self._running:
                if dict(poller.poll(10)).get(self.sock):
                    while True:
                        try:
                            frames = self.sock.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self._handle(frames[0], frames[-1])
                self._tick()
        finally:
            self.sock.close(linger=0)
            self.ctx.term()

    def _tick(self) -> None:
        now = time.monotonic()
        for rid, stream in list(self._slow.items()):
            if stream["stopped"] or stream["sent"] >= 400:
                reason = "cancelled" if stream["stopped"] else "length"
                self._send(stream["identity"], rid, Event.GENERATION_COMPLETE,
                           {"text": "x" * stream["sent"], "metrics": {"prompt_tokens": 2,
                                                                       "completion_tokens": stream["sent"],
                                                                       "finish_reason": reason}})
                del self._slow[rid]
            elif stream["interval"] is not None and now >= stream["next"]:
                stream["sent"] += 1
                stream["next"] = now + stream["interval"]
                self._send(stream["identity"], rid, Event.TOKEN_STREAM, {"text": "x"})

    def _handle(self, identity: bytes, data: bytes) -> None:
        rid, cmd, payload, auth = parse_request(data)
        if auth != self.token:
            self._send(identity, rid, Event.ERROR, error_payload(ErrorCode.UNAUTHORIZED, "Invalid worker token."))
            return
        self.requests.append({"cmd": cmd, "payload": payload, "id": rid})
        if cmd == Command.PING:
            self._send(identity, rid, Event.RESULT, {"pong": True, "pid": 4242, "runtime_ready": True})
            return
        if cmd == Command.STOP_GENERATION:
            target = payload.get("target_id")
            self.stops.append(target)
            if target in self._slow:
                self._slow[target]["stopped"] = True
            self._send(identity, rid, Event.RESULT, {"stopped": [target] if target in self._slow else []})
            return
        self._send(identity, rid, Event.ACK, {"queued": False, "position": 0})
        model = payload.get("model_id")
        errors = {"fake/missing": (ErrorCode.MODEL_NOT_FOUND, "'fake/missing' is not downloaded."),
                  "fake/bad": (ErrorCode.BAD_REQUEST, "No messages to respond to."),
                  "fake/oom": (ErrorCode.OOM, "The model ran out of accelerator memory."),
                  "fake/crash": (ErrorCode.INTERNAL, "RuntimeError: boom")}
        if model in errors:
            code, message = errors[model]
            self._send(identity, rid, Event.ERROR, error_payload(code, message, "Try a 4-bit build."))
            return
        if cmd == Command.GENERATE_STREAM:
            self._send(identity, rid, Event.STATUS, {"stage": "generate", "message": "Generating…"})
            if model in ("fake/slow", "fake/stall"):  # tokens every 20 ms / no tokens, until STOP_GENERATION
                self._slow[rid] = {"identity": identity, "sent": 0, "next": 0.0, "stopped": False,
                                   "interval": 0.02 if model == "fake/slow" else None}
                return
            self._send(identity, rid, Event.TOKEN_STREAM, {"text": "Hel"})
            if model == "fake/midstream":
                self._send(identity, rid, Event.ERROR, error_payload(ErrorCode.OOM, "ran out of memory"))
                return
            for text in ("lo", " world", "!"):
                self._send(identity, rid, Event.TOKEN_STREAM, {"text": text})
            max_new = (payload.get("params") or {}).get("max_new_tokens")
            self._send(identity, rid, Event.GENERATION_COMPLETE, {
                "text": "Hello world!", "sources": [],
                "metrics": {"prompt_tokens": 7, "completion_tokens": 4,
                            "finish_reason": "length" if max_new == 4 else "stop"}})
        elif cmd == Command.EMBED:
            inputs = payload["inputs"]
            self._send(identity, rid, Event.RESULT, {
                "embeddings": [[float(len(t)), 0.5, -1.25] for t in inputs], "dimension": 3,
                "model": model, "prompt_tokens": sum(len(t.split()) for t in inputs)})
        elif cmd == Command.TRANSCRIBE:
            path = Path(payload["audio_path"])
            self.uploads.append({"path": path, "exists": path.is_file(),
                                 "data": path.read_bytes() if path.is_file() else None})
            self._send(identity, rid, Event.RESULT, {
                "text": "hello there", "duration_s": 1.5, "metrics": {"total_ms": 10.0},
                "chunks": [{"start": 0.0, "end": 0.7, "text": "hello"}, {"start": 0.7, "end": 1.5, "text": "there"}]})
        else:
            self._send(identity, rid, Event.ERROR, error_payload(ErrorCode.BAD_REQUEST, f"unhandled {cmd}"))


@pytest.fixture
def worker() -> Iterator[FakeWorker]:
    fake = FakeWorker()
    yield fake
    fake.close()


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[ModelRegistry]:
    reg = ModelRegistry(tmp_path / "registry.db")
    for record in (
        ModelRecord(repo_id="fake/chat", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/chat", pipeline_tag="text-generation", precision="fp16"),
        ModelRecord(repo_id="local/chat-mlx-4bit", pipeline_tag="text-generation", precision="mlx-4bit",
                    source="quantized", engine="mlx", local_path=str(tmp_path)),
        ModelRecord(repo_id="fake/gone", pipeline_tag="text-generation", local_path=str(tmp_path / "deleted")),
        ModelRecord(repo_id="fake/slow", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/stall", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/midstream", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/missing", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/bad", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/oom", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/crash", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/embed", pipeline_tag="sentence-similarity"),
        ModelRecord(repo_id="fake/whisper", pipeline_tag="automatic-speech-recognition"),
    ):
        reg.upsert_model(record)
    yield reg
    reg.close()


@pytest.fixture
def client(worker: FakeWorker, registry: ModelRegistry) -> Iterator[TestClient]:
    with TestClient(srv.create_app(worker.endpoint, TOKEN, registry=registry), base_url=LOCAL) as test_client:
        yield test_client


def chat(client: TestClient, model: str = "fake/chat", **extra: Any) -> httpx.Response:
    body = {"model": model, "messages": [{"role": "user", "content": "Hi there"}], **extra}
    return client.post("/v1/chat/completions", json=body)


def sse_events(text: str) -> list[Any]:
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), block
        data = block[len("data: "):]
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


# ---------------------------------------------------------------------------
# health and models
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["worker"] == {"reachable": True, "runtime_ready": True, "pid": 4242}


def test_health_degraded_without_worker(registry: ModelRegistry) -> None:
    with TestClient(srv.create_app(None, TOKEN, registry=registry), base_url=LOCAL) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["worker"]["reachable"] is False


def test_models_list(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert {"fake/chat", "fake/chat:fp16", "local/chat-mlx-4bit", "fake/embed", "fake/whisper"} <= set(ids)
    assert "fake/gone" not in ids  # local folder deleted
    assert len(ids) == len(set(ids))
    by_id = {m["id"]: m for m in data["data"]}
    assert by_id["fake/embed"] == {"id": "fake/embed", "object": "model", "created": by_id["fake/embed"]["created"],
                                   "owned_by": "local", "task": "sentence-similarity"}
    assert by_id["fake/chat:fp16"]["task"] == "text-generation"
    assert isinstance(by_id["fake/chat"]["created"], int) and by_id["fake/chat"]["created"] > 0

    single = client.get("/v1/models/local/chat-mlx-4bit")
    assert single.status_code == 200 and single.json()["id"] == "local/chat-mlx-4bit"
    missing = client.get("/v1/models/nope/nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# chat completions
# ---------------------------------------------------------------------------


def test_chat_non_stream(client: TestClient, worker: FakeWorker) -> None:
    response = chat(client, temperature=0.2, top_p=0.5, max_tokens=99, max_completion_tokens=50, stop="\n", seed=7,
                    messages=[{"role": "developer", "content": "Be brief."},
                              {"role": "user", "content": [{"type": "text", "text": "Hi "},
                                                           {"type": "text", "text": "there"}]},
                              {"role": "assistant", "content": None, "tool_calls": []}])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "fake/chat"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "Hello world!", "refusal": None}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}

    (request,) = worker.of(Command.GENERATE_STREAM)
    record = ModelRecord(repo_id="fake/chat", pipeline_tag="text-generation")
    assert request["payload"] == {
        **record.worker_payload(),
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}],
        "params": {"temperature": 0.2, "top_p": 0.5, "max_new_tokens": 50, "stop": ["\n"], "seed": 7},
    }
    assert data["id"] == f"chatcmpl-{request['id']}"


def test_chat_defaults_and_length_finish(client: TestClient, worker: FakeWorker) -> None:
    data = chat(client, model="fake/chat:fp16", max_tokens=4).json()
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["model"] == "fake/chat:fp16"
    payload = worker.of(Command.GENERATE_STREAM)[0]["payload"]
    assert payload["precision"] == "fp16"
    assert payload["params"] == {"temperature": None, "top_p": None, "max_new_tokens": 4, "stop": None,
                                 "seed": None}


def test_chat_stream_with_usage(client: TestClient, worker: FakeWorker) -> None:
    response = chat(client, stream=True, stream_options={"include_usage": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert [c["choices"][0]["delta"].get("content") for c in chunks[1:5]] == ["Hel", "lo", " world", "!"]
    final = chunks[5]
    assert final["choices"][0]["delta"] == {} and final["choices"][0]["finish_reason"] == "stop"
    usage = chunks[6]
    assert usage["choices"] == [] and usage["usage"] == {"prompt_tokens": 7, "completion_tokens": 4,
                                                         "total_tokens": 11}
    assert all(c["usage"] is None for c in chunks[:6])
    assert len(chunks) == 7
    assert chunks[0]["id"] == f"chatcmpl-{worker.of(Command.GENERATE_STREAM)[0]['id']}"


def test_chat_stream_without_usage(client: TestClient) -> None:
    events = sse_events(chat(client, stream=True, max_tokens=4).text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert len(chunks) == 6
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all("usage" not in c for c in chunks)


@pytest.mark.parametrize(("model", "status", "code", "kind"), [
    ("fake/missing", 404, "model_not_found", "invalid_request_error"),
    ("fake/bad", 400, "bad_request", "invalid_request_error"),
    ("fake/oom", 503, "oom", "server_error"),
    ("fake/crash", 500, "internal", "server_error"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_worker_error_mapping(client: TestClient, model: str, status: int, code: str, kind: str,
                              stream: bool) -> None:
    response = chat(client, model=model, stream=stream)
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["code"] == code and error["type"] == kind
    assert error["message"] and "Try a 4-bit build." in error["message"]


def test_error_after_stream_started_is_sent_as_event(client: TestClient) -> None:
    response = chat(client, model="fake/midstream", stream=True)
    assert response.status_code == 200
    events = sse_events(response.text)
    assert events[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert events[-2] == {"error": {"message": "ran out of memory", "type": "server_error", "param": None,
                                    "code": "oom"}}
    assert events[-1] == "[DONE]"


def test_unknown_model_lists_available(client: TestClient, worker: FakeWorker) -> None:
    response = chat(client, model="nobody/nothing")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found" and error["param"] == "model"
    assert "fake/chat" in error["message"] and "fake/embed" in error["message"]
    assert worker.of(Command.GENERATE_STREAM) == []


@pytest.mark.parametrize(("body", "param"), [
    ({"model": "fake/chat"}, "messages"),
    ({"model": "fake/chat", "messages": []}, "messages"),
    ({"model": "fake/chat", "messages": [{"content": "no role"}]}, "messages[0]"),
    ({"model": "fake/chat", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "messages[0]"),
    ({"model": "fake/chat", "messages": [{"role": "user", "content": "x"}], "temperature": "hot"}, "temperature"),
    ({"model": "fake/chat", "messages": [{"role": "user", "content": "x"}], "temperature": 3}, "temperature"),
    ({"model": "fake/chat", "messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
    ({"model": "fake/chat", "messages": [{"role": "user", "content": "x"}], "stop": [1]}, "stop"),
    ({"model": "fake/chat", "messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
    ({"messages": [{"role": "user", "content": "x"}]}, "model"),
])
def test_chat_validation(client: TestClient, worker: FakeWorker, body: dict[str, Any], param: str) -> None:
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert worker.of(Command.GENERATE_STREAM) == []


def test_invalid_json_and_wrong_family(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 400 and response.json()["error"]["type"] == "invalid_request_error"
    response = chat(client, model="fake/embed")
    assert response.status_code == 400 and "sentence-similarity" in response.json()["error"]["message"]


def test_unknown_route_is_openai_shaped(client: TestClient) -> None:
    response = client.get("/v1/nothing-here")
    assert response.status_code == 404
    assert set(response.json()["error"]) == {"message", "type", "param", "code"}


# ---------------------------------------------------------------------------
# worker availability and credentials
# ---------------------------------------------------------------------------


def test_wrong_worker_token_is_a_server_error(worker: FakeWorker, registry: ModelRegistry) -> None:
    with TestClient(srv.create_app(worker.endpoint, "not-the-token", registry=registry), base_url=LOCAL) as client:
        response = chat(client)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "worker_unauthorized"


def test_worker_not_running(registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(srv.create_app(None, TOKEN, registry=registry), base_url=LOCAL) as client:
        response = chat(client)
    assert response.status_code == 503 and response.json()["error"]["code"] == "worker_unavailable"

    monkeypatch.setattr(srv, "WORKER_ACK_TIMEOUT_S", 0.3)
    silent = zmq.Context.instance().socket(zmq.ROUTER)  # bound but never answers
    port = silent.bind_to_random_port("tcp://127.0.0.1")
    try:
        with TestClient(srv.create_app(f"tcp://127.0.0.1:{port}", TOKEN, registry=registry), base_url=LOCAL) as client:
            response = chat(client)
    finally:
        silent.close(linger=0)
    assert response.status_code == 503 and "not responding" in response.json()["error"]["message"]


def test_endpoint_from_discovery_file(worker: FakeWorker, registry: ModelRegistry) -> None:
    paths.runtime_dir().mkdir(parents=True, exist_ok=True)
    (paths.runtime_dir() / "worker.json").write_text(json.dumps({"host": "127.0.0.1", "port": worker.port}))
    with TestClient(srv.create_app(None, TOKEN, registry=registry), base_url=LOCAL) as client:
        response = chat(client)
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "Hello world!"


def test_api_key(worker: FakeWorker, registry: ModelRegistry) -> None:
    app = srv.create_app(worker.endpoint, TOKEN, registry=registry, api_key="sk-local-123")
    with TestClient(app, base_url=LOCAL) as client:
        assert client.get("/health").status_code == 200  # liveness stays open
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic sk-local-123"}):
            response = client.get("/v1/models", headers=headers)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_api_key"
        response = client.post("/v1/chat/completions", json={"model": "fake/chat", "messages": []})
        assert response.status_code == 401
        ok = client.get("/v1/models", headers={"Authorization": "Bearer sk-local-123"})
        assert ok.status_code == 200
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer sk-local-123"},
                               json={"model": "fake/chat", "messages": [{"role": "user", "content": "x"}]})
        assert response.status_code == 200
    assert len(worker.of(Command.GENERATE_STREAM)) == 1


# ---------------------------------------------------------------------------
# embeddings and transcriptions
# ---------------------------------------------------------------------------


def test_embeddings(client: TestClient, worker: FakeWorker) -> None:
    response = client.post("/v1/embeddings", json={"model": "fake/embed", "input": ["hello world", "a"]})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "list" and data["model"] == "fake/embed"
    assert data["data"] == [{"object": "embedding", "index": 0, "embedding": [11.0, 0.5, -1.25]},
                            {"object": "embedding", "index": 1, "embedding": [1.0, 0.5, -1.25]}]
    assert data["usage"] == {"prompt_tokens": 3, "total_tokens": 3}
    assert worker.of(Command.EMBED)[0]["payload"]["inputs"] == ["hello world", "a"]
    assert worker.of(Command.EMBED)[0]["payload"]["task"] == "sentence-similarity"

    single = client.post("/v1/embeddings", json={"model": "fake/embed", "input": "abc", "encoding_format": "base64"})
    encoded = single.json()["data"][0]["embedding"]
    assert list(struct.unpack("<3f", base64.b64decode(encoded))) == [3.0, 0.5, -1.25]


@pytest.mark.parametrize("body", [
    {"model": "fake/embed", "input": [1, 2, 3]},
    {"model": "fake/embed", "input": []},
    {"model": "fake/embed", "input": "x", "encoding_format": "int8"},
    {"model": "fake/chat", "input": "x"},
])
def test_embeddings_validation(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/v1/embeddings", json=body).status_code == 400


def _uploads() -> list[Path]:
    folder = paths.projects_dir() / "uploads"
    return list(folder.iterdir()) if folder.is_dir() else []


@pytest.mark.parametrize("response_format", ["json", "text", "verbose_json"])
def test_transcriptions(client: TestClient, worker: FakeWorker, response_format: str) -> None:
    audio = bytes(range(256)) * 64 + b"\r\n--boundary-lookalike\r\n"
    response = client.post("/v1/audio/transcriptions",
                           files={"file": ("clip.WAV", audio, "audio/wav")},
                           data={"model": "fake/whisper", "language": "en", "response_format": response_format})
    assert response.status_code == 200, response.text
    if response_format == "text":
        assert response.text == "hello there"
    elif response_format == "json":
        assert response.json() == {"text": "hello there"}
    else:
        data = response.json()
        assert data["text"] == "hello there" and data["language"] == "en" and data["duration"] == 1.5
        assert [s["text"] for s in data["segments"]] == ["hello", "there"]
        assert data["segments"][1]["start"] == 0.7
    upload = worker.uploads[0]
    assert upload["exists"] and upload["data"] == audio
    assert upload["path"].suffix == ".wav"
    assert upload["path"].parent == paths.projects_dir() / "uploads"
    payload = worker.of(Command.TRANSCRIBE)[0]["payload"]
    assert payload["language"] == "en" and payload["timestamps"] == (response_format == "verbose_json")
    assert _uploads() == [], "temporary upload must be deleted"


def test_transcription_errors_never_leave_uploads(client: TestClient, worker: FakeWorker) -> None:
    bad_format = client.post("/v1/audio/transcriptions", files={"file": ("a.mp3", b"123", "audio/mpeg")},
                             data={"model": "fake/whisper", "response_format": "srt"})
    assert bad_format.status_code == 400
    no_file = client.post("/v1/audio/transcriptions", data={"model": "fake/whisper"},
                          files={"other": ("a.mp3", b"123", "audio/mpeg")})
    assert no_file.status_code == 400 and no_file.json()["error"]["param"] == "file"
    not_multipart = client.post("/v1/audio/transcriptions", json={"model": "fake/whisper"})
    assert not_multipart.status_code == 400
    wrong_model = client.post("/v1/audio/transcriptions", files={"file": ("a.mp3", b"123", "audio/mpeg")},
                              data={"model": "fake/chat"})
    assert wrong_model.status_code == 400
    assert _uploads() == []
    assert worker.of(Command.TRANSCRIBE) == []


def test_parse_multipart_roundtrip() -> None:
    request = httpx.Request("POST", "http://x/", files={"file": ("x.bin", os.urandom(5000), "application/octet-stream")},
                            data={"model": "m", "language": "fr"})
    body = request.read()
    fields, files = srv.parse_multipart(request.headers["content-type"], body)
    assert fields == {"model": "m", "language": "fr"}
    assert files["file"][0] == "x.bin" and len(files["file"][1]) == 5000


# ---------------------------------------------------------------------------
# access log
# ---------------------------------------------------------------------------


def test_access_log_has_no_content(client: TestClient) -> None:
    secret = "my secret prompt 1234"
    chat(client, messages=[{"role": "user", "content": secret}])
    client.get("/v1/models")
    chat(client, model="nobody/nothing")
    log_text = (paths.logs_dir() / "server-access.log").read_text()
    lines = log_text.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].split()[1:4] == ["POST", "/v1/chat/completions", "200"]
    assert lines[0].endswith("ms")
    assert lines[1].split()[1:4] == ["GET", "/v1/models", "200"]
    assert lines[2].split()[1:4] == ["POST", "/v1/chat/completions", "404"]
    assert secret not in log_text and "Hello world" not in log_text


# ---------------------------------------------------------------------------
# real uvicorn: client disconnect mid-stream sends STOP_GENERATION
# ---------------------------------------------------------------------------


def test_disconnect_mid_stream_stops_generation(worker: FakeWorker, registry: ModelRegistry) -> None:
    server = srv.serve_in_thread(srv.create_app(worker.endpoint, TOKEN, registry=registry))
    try:
        with httpx.Client(base_url=server.base_url, timeout=10) as http:
            with http.stream("POST", "/v1/chat/completions", json={
                    "model": "fake/slow", "stream": True,
                    "messages": [{"role": "user", "content": "go"}]}) as response:
                assert response.status_code == 200
                received = 0
                for line in response.iter_lines():
                    if line.startswith("data: ") and '"content":"x"' in line:
                        received += 1
                        if received == 3:
                            break
        (generate,) = worker.of(Command.GENERATE_STREAM)
        deadline = time.monotonic() + 5
        while generate["id"] not in worker.stops and time.monotonic() < deadline:
            time.sleep(0.05)
        assert worker.stops == [generate["id"]]
    finally:
        server.stop()
    log_lines = (paths.logs_dir() / "server-access.log").read_text().splitlines()
    assert log_lines[-1].split()[1:4] == ["POST", "/v1/chat/completions", "200"]


@pytest.mark.parametrize("stream", [True, False])
def test_disconnect_before_first_token_stops_generation(worker: FakeWorker, registry: ModelRegistry,
                                                        monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    monkeypatch.setattr(srv, "DISCONNECT_POLL_S", 0.05)
    server = srv.serve_in_thread(srv.create_app(worker.endpoint, TOKEN, registry=registry))
    try:
        # "fake/stall" never produces a token: the client gives up while the server is still waiting.
        with httpx.Client(base_url=server.base_url, timeout=0.5) as http:
            with pytest.raises(httpx.ReadTimeout):
                http.post("/v1/chat/completions", json={"model": "fake/stall", "stream": stream,
                                                        "messages": [{"role": "user", "content": "go"}]})
        deadline = time.monotonic() + 5
        while not worker.stops and time.monotonic() < deadline:
            time.sleep(0.05)
        (generate,) = worker.of(Command.GENERATE_STREAM)
        assert worker.stops == [generate["id"]]
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("host", "ok"), [("127.0.0.1", True), ("localhost", True), ("::1", True),
                                          ("127.0.0.2", True), ("0.0.0.0", False), ("192.168.1.10", False),
                                          ("example.com", False), ("::", False)])
def test_is_loopback(host: str, ok: bool) -> None:
    assert srv.is_loopback(host) is ok


def test_main_refuses_public_host(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(host="0.0.0.0", port=0, worker_endpoint=None)
    assert srv.main(args) == 2
    assert "loopback" in capsys.readouterr().err


def test_cli_port_in_use_explains_itself() -> None:
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        proc = subprocess.run([sys.executable, "-m", "noelle", "server", "--port", str(port), "--worker-endpoint",
                               "tcp://127.0.0.1:9"], capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "NOELLE_SERVER_READY" not in proc.stdout
    assert "address already in use" in proc.stderr.strip().splitlines()[-1].lower()


def test_cli_prints_ready_line_and_serves(worker: FakeWorker, tmp_path: Path) -> None:
    env = dict(os.environ, NOELLE_WORKER_TOKEN=TOKEN, NOELLE_SERVER_API_KEY="cli-key", PYTHONUNBUFFERED="1",
               NOELLE_HOME=os.environ["NOELLE_HOME"])
    proc = subprocess.Popen([sys.executable, "-m", "noelle", "server", "--port", "0", "--worker-endpoint",
                             worker.endpoint], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        assert line.startswith("NOELLE_SERVER_READY port="), line
        port = int(line.split("port=")[1])
        base = f"http://127.0.0.1:{port}"
        assert httpx.get(f"{base}/health", timeout=5).json()["worker"]["reachable"] is True
        assert httpx.get(f"{base}/v1/models", timeout=5).status_code == 401
        response = httpx.get(f"{base}/v1/models", headers={"Authorization": "Bearer cli-key"}, timeout=5)
        assert response.status_code == 200
    finally:
        proc.terminate()
        proc.wait(timeout=10)
