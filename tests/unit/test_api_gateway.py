"""API gateway features of server_service: aliases, "auto" routing, images and knowledge endpoints.

Uses the fake ZeroMQ worker from test_server_service (no models, no torch, no network).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from noelle import paths
from noelle.app.services.model_registry import ModelRecord, ModelRegistry
from noelle.config import Settings
from noelle.runtime import server_service as srv
from noelle.runtime.protocol import Command, ErrorCode, Event, error_payload, parse_request
from tests.unit.test_server_service import TOKEN, FakeWorker, sse_events

PNG_HEAD = b"\x89PNG\r\n\x1a\n"
FAILING_MODELS = {"fake/missing", "fake/bad", "fake/oom", "fake/crash"}  # FakeWorker answers these with ERROR
GATEWAY_COMMANDS = {Command.LIST_LOADED, Command.RAG_LIST_COLLECTIONS, Command.GENERATE_IMAGE, Command.RAG_QUERY}
COLLECTIONS = [
    {"name": "docs", "embedding_model": "fake/embed", "dimension": 3, "documents": 2, "chunks": 5,
     "created_at": 1.0, "updated_at": 2.0, "backend": "numpy"},
    {"name": "notes", "embedding_model": "fake/embed", "dimension": 3, "documents": 1, "chunks": 1,
     "created_at": 3.0, "updated_at": 3.0, "backend": "sqlite-vec"},
]
HITS = [{"source": f"doc{i}.md", "path": str(Path("docs") / f"doc{i}.md"), "chunk_id": i, "chunk_index": 0,
         "page": None, "text": f"passage {i}", "score": round(1.0 - i / 10, 6)} for i in range(6)]


LOCAL = "http://127.0.0.1"  # the API only answers loopback Host headers


class GatewayWorker(FakeWorker):
    """FakeWorker plus LIST_LOADED, RAG_LIST_COLLECTIONS (light: no ACK), GENERATE_IMAGE and RAG_QUERY."""

    def __init__(self) -> None:
        self.loaded: list[dict[str, Any]] = []
        self.loaded_fails = False
        self.collections = [dict(c) for c in COLLECTIONS]
        self.images: list[Path] = []
        super().__init__()

    def _handle(self, identity: bytes, data: bytes) -> None:
        rid, cmd, payload, auth = parse_request(data)
        model = payload.get("model_id")
        if auth != self.token or cmd not in GATEWAY_COMMANDS or model in FAILING_MODELS:
            super()._handle(identity, data)
            return
        self.requests.append({"cmd": cmd, "payload": payload, "id": rid})
        if cmd == Command.LIST_LOADED:
            if self.loaded_fails:
                self._send(identity, rid, Event.ERROR, error_payload(ErrorCode.INTERNAL, "boom"))
            else:
                self._send(identity, rid, Event.RESULT, {"models": self.loaded})
        elif cmd == Command.RAG_LIST_COLLECTIONS:
            self._send(identity, rid, Event.RESULT, {"collections": self.collections})
        elif cmd == Command.GENERATE_IMAGE:
            self._send(identity, rid, Event.ACK, {"queued": False, "position": 0})
            path = paths.outputs_dir() / f"image-{rid}.png"
            if model != "fake/sd-blank":  # returns a path but never writes the file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(PNG_HEAD + rid.encode())
                self.images.append(path)
            self._send(identity, rid, Event.RESULT, {"image_path": str(path), "seed": payload.get("seed", 99),
                                                     "width": payload.get("width", 512),
                                                     "height": payload.get("height", 512),
                                                     "metrics": {"total_ms": 5.0}})
        else:
            self._send(identity, rid, Event.ACK, {"queued": False, "position": 0})
            self._send(identity, rid, Event.RESULT, {"hits": HITS[:int(payload["top_k"])], "latency_ms": 1.5})


@pytest.fixture
def worker() -> Iterator[GatewayWorker]:
    fake = GatewayWorker()
    yield fake
    fake.close()


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[ModelRegistry]:
    """Most recently used first: fake/chat, fake/coder, ... (list_models orders by last use, then download)."""
    reg = ModelRegistry(tmp_path / "registry.db")
    now = time.time()
    records = [
        ModelRecord(repo_id="fake/chat", pipeline_tag="text-generation"),
        ModelRecord(repo_id="fake/coder", pipeline_tag="text-generation"),
        ModelRecord(repo_id="tiny", pipeline_tag="text-generation", source="local"),
        ModelRecord(repo_id="gpt2", pipeline_tag="text-generation", precision="fp16"),
        ModelRecord(repo_id="fake/embed", pipeline_tag="sentence-similarity"),
        ModelRecord(repo_id="fake/embed-old", pipeline_tag="feature-extraction"),
        ModelRecord(repo_id="fake/sd", pipeline_tag="text-to-image"),
        ModelRecord(repo_id="fake/sd-old", pipeline_tag="text-to-image"),
        ModelRecord(repo_id="fake/sd-blank", pipeline_tag="text-to-image"),
        ModelRecord(repo_id="fake/oom", pipeline_tag="text-to-image"),
        ModelRecord(repo_id="fake/gone", pipeline_tag="text-generation", local_path=str(tmp_path / "deleted")),
        ModelRecord(repo_id="fake/whisper", pipeline_tag="automatic-speech-recognition"),
    ]
    for age, record in enumerate(records):
        record.downloaded_at = now - 60 * (age + 1)
        reg.upsert_model(record)
    yield reg
    reg.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(tmp_path / "settings.json")


@pytest.fixture
def client(worker: GatewayWorker, registry: ModelRegistry, settings: Settings) -> Iterator[TestClient]:
    with TestClient(srv.create_app(worker.endpoint, TOKEN, registry=registry, settings=settings), base_url=LOCAL) as test_client:
        yield test_client


def chat(client: TestClient, model: str, content: str = "Hi there", **extra: Any) -> httpx.Response:
    return client.post("/v1/chat/completions",
                       json={"model": model, "messages": [{"role": "user", "content": content}], **extra})


def cards(client: TestClient) -> dict[str, dict[str, Any]]:
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [card["id"] for card in response.json()["data"]]
    assert len(ids) == len(set(ids))
    return {card["id"]: card for card in response.json()["data"]}


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_alias_serves_its_target(client: TestClient, worker: GatewayWorker, settings: Settings,
                                 stream: bool) -> None:
    settings.set("server.aliases", {"noelle-chat": "fake/coder"})
    response = chat(client, "noelle-chat", stream=stream)
    assert response.status_code == 200, response.text
    assert response.headers["x-noelle-model"] == "fake/coder"
    assert "x-noelle-routed-model" not in response.headers and "x-noelle-route-reason" not in response.headers
    if stream:
        chunks = [e for e in sse_events(response.text) if e != "[DONE]"]
        assert {c["model"] for c in chunks} == {"fake/coder"}
    else:
        assert response.json()["model"] == "fake/coder"
    (request,) = worker.of(Command.GENERATE_STREAM)
    assert request["payload"]["model_id"] == "fake/coder"


def test_resolution_order_exact_alias_then_fallbacks(client: TestClient, worker: GatewayWorker,
                                                     settings: Settings) -> None:
    assert chat(client, "gpt2").json()["model"] == "gpt2:fp16"  # registry fallback: repo id
    settings.set("server.aliases", {"gpt2": "fake/chat", "tiny": "fake/coder"})
    assert chat(client, "gpt2").json()["model"] == "fake/chat"  # the alias beats the fallback
    assert chat(client, "tiny").json()["model"] == "tiny"  # an exact model name beats the alias
    assert [r["payload"]["model_id"] for r in worker.of(Command.GENERATE_STREAM)] == ["gpt2", "fake/chat", "tiny"]
    listed = cards(client)
    assert listed["tiny"]["owned_by"] == "local"  # the shadowed alias is not listed over the model


def test_alias_to_a_missing_model(client: TestClient, worker: GatewayWorker, settings: Settings) -> None:
    settings.set("server.aliases", {"broken": "nobody/nothing", "stale": "fake/gone"})
    response = chat(client, "broken")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found" and error["param"] == "model"
    assert "'broken'" in error["message"] and "'nobody/nothing'" in error["message"]
    assert worker.of(Command.GENERATE_STREAM) == []
    listed = cards(client)
    assert "broken" not in listed and "stale" not in listed  # only aliases whose target is downloaded


def test_aliases_are_read_live(client: TestClient, settings: Settings) -> None:
    assert "live" not in cards(client)
    desktop = Settings(settings.path)  # the desktop app's own instance writes the shared settings.json
    desktop.set("server.aliases", {"live": "fake/coder"})
    assert cards(client)["live"]["parent"] == "fake/coder"
    assert chat(client, "live").headers["x-noelle-model"] == "fake/coder"
    desktop.set("server.aliases", {"live": "fake/chat"})
    assert chat(client, "live").headers["x-noelle-model"] == "fake/chat"
    desktop.set("server.aliases", {})
    assert "live" not in cards(client)
    assert chat(client, "live").status_code == 404


def test_malformed_aliases_are_ignored(client: TestClient, settings: Settings) -> None:
    settings.set("server.aliases", {"auto": "fake/coder", "Bad Name": "fake/chat", "num": 5, "empty": "",
                                    "abc\n": "fake/chat", "fine": "fake/chat"})
    listed = cards(client)
    assert "fine" in listed
    assert not {"Bad Name", "num", "empty", "abc\n"} & set(listed)
    assert listed["auto"]["owned_by"] == "noelle-router"  # "auto" stays the router, never an alias
    assert srv.clean_aliases(["not", "a", "dict"]) == {}
    assert srv.clean_aliases({"x" * 64: "a/b", "y" * 65: "a/b", "-dash": "a/b", "ok.v1_x-y": "a/b"}) == {
        "x" * 64: "a/b", "ok.v1_x-y": "a/b"}


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_models_list_aliases_and_auto(client: TestClient, settings: Settings) -> None:
    settings.set("server.aliases", {"noelle-chat": "fake/chat", "noelle-image": "fake/sd"})
    listed = cards(client)
    chat_card = listed["fake/chat"]
    assert chat_card == {"id": "fake/chat", "object": "model", "created": chat_card["created"], "owned_by": "local",
                         "task": "text-generation"}
    assert listed["noelle-chat"] == {"id": "noelle-chat", "object": "model", "created": chat_card["created"],
                                     "owned_by": "noelle-alias", "task": "text-generation", "parent": "fake/chat"}
    assert listed["noelle-image"]["parent"] == "fake/sd" and listed["noelle-image"]["task"] == "text-to-image"
    assert listed["auto"] == {"id": "auto", "object": "model", "created": 0, "owned_by": "noelle-router",
                              "task": None}
    assert client.get("/v1/models/noelle-chat").json() == listed["noelle-chat"]
    assert client.get("/v1/models/auto").json() == listed["auto"]
    assert client.get("/v1/models/fake/chat").json() == chat_card


def test_auto_is_listed_only_with_a_text_model(worker: GatewayWorker, tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "embed-only.db")
    reg.upsert_model(ModelRecord(repo_id="fake/embed", pipeline_tag="sentence-similarity"))
    app = srv.create_app(worker.endpoint, TOKEN, registry=reg, settings=Settings(tmp_path / "s.json"))
    with TestClient(app, base_url=LOCAL) as client:
        assert "auto" not in cards(client)
        assert client.get("/v1/models/auto").status_code == 404
        response = chat(client, "auto")
        assert response.status_code == 404 and "no downloaded text model" in response.json()["error"]["message"]
        response = client.post("/v1/images/generations", json={"model": "auto", "prompt": "a cat"})
        assert response.status_code == 404 and "diffusion" in response.json()["error"]["message"]
    reg.close()


# ---------------------------------------------------------------------------
# "auto" routing: the pure decision
# ---------------------------------------------------------------------------


def rec(repo_id: str, tag: str | None = "text-generation", precision: str = "auto") -> ModelRecord:
    return ModelRecord(repo_id=repo_id, pipeline_tag=tag, precision=precision)


def loaded(model_id: str, precision: str = "auto", last_used: float = 1.0) -> dict[str, Any]:
    return {"key": f"{model_id}|{precision}|mps", "model_id": model_id, "precision": precision, "family": "text",
            "last_used": last_used}


RECORDS = [rec("org/chat"), rec("org/coder"), rec("org/coder", precision="q4"),
           rec("org/embed", "sentence-similarity"), rec("org/embed-b", "feature-extraction"),
           rec("org/sd", "text-to-image"), rec("org/sd-xl", "text-to-image"),
           rec("org/whisper", "automatic-speech-recognition"), rec("org/untagged", None)]


@pytest.mark.parametrize("text", [
    "```python\nprint('hi')\n```",
    "Why does this fail?\ndef parse(line):\n    return line.split()",
    "    async def handler(request):",
    "class Parser(Base):\n    pass",
    "export class Store extends Base {",
    "const add = function (a, b) { return a + b }",
    "function main() {}",
    "SELECT id, name FROM users WHERE id = 1",
    "#include <stdio.h>",
    "Write a Python function that removes duplicates",
    "please write me a rust program for this",
    "Write some Go scripts to rotate logs",
])
def test_looks_like_code(text: str) -> None:
    assert srv.looks_like_code(text)


@pytest.mark.parametrize("text", [
    "What is the capital of France?",
    "class starts at 9 tomorrow, remind me",
    "Explain the function of the liver.",
    "Please select the best option from the list",
    "Write a poem about a trip to Java",
    "I'd like to define my goals for next year",
])
def test_does_not_look_like_code(text: str) -> None:
    assert not srv.looks_like_code(text)


@pytest.mark.parametrize(("text", "aliases", "loaded_models", "settings", "expected"), [
    ("def f(x):\n  return x", {"noelle-code": "org/coder", "noelle-chat": "org/chat"}, [], {},
     ("org/coder", "code-alias")),
    ("def f(x):\n  return x", {"noelle-chat": "org/coder:q4"}, [], {}, ("org/coder:q4", "chat-alias")),
    ("Hello there", {"noelle-code": "org/coder", "noelle-chat": "org/coder:q4"}, [], {},
     ("org/coder:q4", "chat-alias")),
    ("```js\nx()\n```", {"noelle-code": "org/embed"}, [], {}, ("org/chat", "recent-model")),  # wrong family
    ("Hello", {"noelle-chat": "nobody/missing"}, [], {"server.default_model": "org/coder"},
     ("org/coder", "default-model")),
    ("Hello", {}, [], {"server.default_model": "org/coder:q4"}, ("org/coder:q4", "default-model")),
    ("Hello", {}, [], {"server.default_model": "org/embed"}, ("org/chat", "recent-model")),
    ("Hello", {}, [loaded("org/coder", "q4", 10.0), loaded("org/chat", "auto", 5.0)], {},
     ("org/coder:q4", "loaded-model")),
    ("Hello", {}, [loaded("org/coder", "fp16", 10.0), loaded("org/embed", "auto", 20.0)], {},
     ("org/chat", "recent-model")),  # a build that is not downloaded, and a non-text model
    ("Hello", {}, [loaded("org/untagged")], {"server.default_model": "org/coder"}, ("org/coder", "default-model")),
    ("Hello", {}, [loaded("org/untagged")], {}, ("org/untagged", "loaded-model")),  # no pipeline tag: text
    ("Hello", {}, [], {}, ("org/chat", "recent-model")),
])
def test_route_chat(text: str, aliases: dict[str, str], loaded_models: list[dict[str, Any]],
                    settings: dict[str, Any], expected: tuple[str, str]) -> None:
    routed = srv.route_auto("chat", text, aliases, RECORDS, loaded_models, settings)
    assert routed is not None
    assert (routed[0].api_name, routed[1]) == expected


@pytest.mark.parametrize(("aliases", "loaded_models", "settings", "expected"), [
    ({"noelle-embed": "org/embed-b"}, [loaded("org/embed")], {}, ("org/embed-b", "embed-alias")),
    ({"noelle-embed": "org/chat"}, [loaded("org/embed-b", last_used=3.0), loaded("org/embed", last_used=1.0)],
     {"rag.embedding_model": "org/embed"}, ("org/embed-b", "loaded-model")),
    ({}, [loaded("org/chat")], {"rag.embedding_model": "org/embed-b"}, ("org/embed-b", "rag-embedding-model")),
    ({}, [], {"rag.embedding_model": "nobody/nothing"}, ("org/embed", "recent-model")),
])
def test_route_embeddings(aliases: dict[str, str], loaded_models: list[dict[str, Any]], settings: dict[str, Any],
                          expected: tuple[str, str]) -> None:
    routed = srv.route_auto("embeddings", "", aliases, RECORDS, loaded_models, settings)
    assert routed is not None
    assert (routed[0].api_name, routed[1]) == expected


@pytest.mark.parametrize(("aliases", "expected"), [
    ({"noelle-image": "org/sd-xl"}, ("org/sd-xl", "image-alias")),
    ({"noelle-image": "org/chat"}, ("org/sd", "recent-model")),
    ({}, ("org/sd", "recent-model")),
])
def test_route_images(aliases: dict[str, str], expected: tuple[str, str]) -> None:
    routed = srv.route_auto("images", "a cat", aliases, RECORDS, [loaded("org/sd-xl")], {})
    assert routed is not None
    assert (routed[0].api_name, routed[1]) == expected


def test_route_default_model_by_repo_id_and_nothing_found() -> None:
    routed = srv.route_auto("chat", "hi", {}, [rec("org/chat"), rec("org/coder", precision="q4")], [],
                            {"server.default_model": "org/coder"})
    assert routed is not None and (routed[0].api_name, routed[1]) == ("org/coder:q4", "default-model")
    assert srv.route_auto("chat", "hi", {}, [rec("org/sd", "text-to-image")], [], {}) is None
    assert srv.route_auto("embeddings", "", {}, [rec("org/chat")], [loaded("org/chat")], {}) is None
    assert srv.route_auto("images", "", {"noelle-image": "org/chat"}, [rec("org/chat")], [], {}) is None


# ---------------------------------------------------------------------------
# "auto" routing over HTTP
# ---------------------------------------------------------------------------


def test_auto_chat_routes_code_to_the_code_alias(client: TestClient, worker: GatewayWorker,
                                                 settings: Settings) -> None:
    settings.set("server.aliases", {"noelle-code": "fake/coder"})
    response = chat(client, "auto", "Why is this slow?\ndef total(rows):\n    return sum(rows)")
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "fake/coder"
    assert response.headers["x-noelle-model"] == "fake/coder"
    assert response.headers["x-noelle-routed-model"] == "fake/coder"
    assert response.headers["x-noelle-route-reason"] == "code-alias"
    assert worker.of(Command.LIST_LOADED) == []  # decided before the loaded-model rule
    assert worker.of(Command.GENERATE_STREAM)[0]["payload"]["model_id"] == "fake/coder"


def test_auto_chat_reads_every_user_turn(client: TestClient, settings: Settings) -> None:
    settings.set("server.aliases", {"noelle-code": "fake/coder", "noelle-chat": "fake/chat"})
    messages = [{"role": "user", "content": "Write a Python function that parses dates"},
                {"role": "assistant", "content": "Here it is."}, {"role": "user", "content": "Thanks, now explain it"}]
    response = client.post("/v1/chat/completions", json={"model": "auto", "messages": messages})
    assert response.headers["x-noelle-route-reason"] == "code-alias"
    response = chat(client, "auto", "Thanks, now explain it")
    assert (response.headers["x-noelle-routed-model"], response.headers["x-noelle-route-reason"]) == (
        "fake/chat", "chat-alias")


def test_auto_chat_prefers_loaded_then_recent(client: TestClient, worker: GatewayWorker,
                                              settings: Settings) -> None:
    response = chat(client, "auto", stream=True)
    assert response.status_code == 200
    assert (response.headers["x-noelle-routed-model"], response.headers["x-noelle-route-reason"]) == (
        "fake/chat", "recent-model")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert len(worker.of(Command.LIST_LOADED)) == 1

    worker.loaded = [loaded("fake/coder", last_used=5.0), loaded("fake/embed", last_used=9.0)]
    response = chat(client, "auto")
    assert (response.json()["model"], response.headers["x-noelle-route-reason"]) == ("fake/coder", "loaded-model")

    settings.set("server.default_model", "tiny")
    response = chat(client, "auto")
    assert (response.json()["model"], response.headers["x-noelle-route-reason"]) == ("tiny", "default-model")


def test_auto_tolerates_list_loaded_errors(client: TestClient, worker: GatewayWorker) -> None:
    worker.loaded_fails = True
    response = chat(client, "auto")
    assert response.status_code == 200
    assert response.headers["x-noelle-route-reason"] == "recent-model"
    assert len(worker.of(Command.LIST_LOADED)) == 1


def test_auto_embeddings(client: TestClient, worker: GatewayWorker, settings: Settings) -> None:
    response = client.post("/v1/embeddings", json={"model": "auto", "input": "hello"})
    assert response.status_code == 200
    assert response.json()["model"] == "fake/embed"
    assert response.headers["x-noelle-route-reason"] == "recent-model"  # rag.embedding_model is not downloaded
    settings.set("rag.embedding_model", "fake/embed-old")
    response = client.post("/v1/embeddings", json={"model": "auto", "input": "hello"})
    assert (response.json()["model"], response.headers["x-noelle-route-reason"]) == (
        "fake/embed-old", "rag-embedding-model")
    worker.loaded = [loaded("fake/embed", last_used=2.0)]
    response = client.post("/v1/embeddings", json={"model": "auto", "input": "hello"})
    assert (response.json()["model"], response.headers["x-noelle-route-reason"]) == ("fake/embed", "loaded-model")
    assert [r["payload"]["model_id"] for r in worker.of(Command.EMBED)] == ["fake/embed", "fake/embed-old",
                                                                           "fake/embed"]


def test_auto_without_a_worker_still_routes(registry: ModelRegistry, settings: Settings) -> None:
    with TestClient(srv.create_app(None, TOKEN, registry=registry, settings=settings), base_url=LOCAL) as client:
        response = chat(client, "auto")  # LIST_LOADED cannot run; the request then fails like any other
    assert response.status_code == 503 and response.json()["error"]["code"] == "worker_unavailable"


def test_model_header_on_every_model_endpoint(client: TestClient, settings: Settings) -> None:
    settings.set("server.aliases", {"noelle-embed": "fake/embed-old", "noelle-asr": "fake/whisper"})
    response = client.post("/v1/embeddings", json={"model": "noelle-embed", "input": "x"})
    assert response.json()["model"] == "fake/embed-old" and response.headers["x-noelle-model"] == "fake/embed-old"
    response = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF0000", "audio/wav")},
                           data={"model": "noelle-asr", "response_format": "text"})
    assert response.status_code == 200 and response.headers["x-noelle-model"] == "fake/whisper"
    response = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF0000", "audio/wav")},
                           data={"model": "auto"})
    assert response.status_code == 404  # "auto" only routes chat, embeddings and images


def test_header_values_are_latin1_safe() -> None:
    record = ModelRecord(repo_id="local/modèle ü", pipeline_tag="text-generation", source="local")
    headers = srv.Resolved(record, "recent-model").headers()
    assert headers == {"x-noelle-model": "local/mod%C3%A8le%20%C3%BC",
                       "x-noelle-routed-model": "local/mod%C3%A8le%20%C3%BC", "x-noelle-route-reason": "recent-model"}
    assert all(value.isascii() for value in headers.values())


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


def test_image_generation(client: TestClient, worker: GatewayWorker, registry: ModelRegistry) -> None:
    response = client.post("/v1/images/generations", json={
        "model": "fake/sd", "prompt": "a cat in a hat", "size": "640x384", "seed": 3, "steps": 4, "n": 1,
        "response_format": "b64_json", "quality": "standard"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {"created", "data"} and isinstance(data["created"], int) and data["created"] > 0
    (item,) = data["data"]
    assert item["revised_prompt"] is None
    (request,) = worker.of(Command.GENERATE_IMAGE)
    assert base64.b64decode(item["b64_json"]) == PNG_HEAD + request["id"].encode()
    record = registry.find_by_api_name("fake/sd")
    assert record is not None
    assert request["payload"] == {**record.worker_payload(), "prompt": "a cat in a hat", "preview_every": 0,
                                  "width": 640, "height": 384, "seed": 3, "steps": 4}
    assert response.headers["x-noelle-model"] == "fake/sd"
    assert not worker.images[0].exists(), "the server deletes the generated file once it is encoded"


def test_image_generation_defaults_and_routing(client: TestClient, worker: GatewayWorker,
                                               settings: Settings) -> None:
    response = client.post("/v1/images/generations", json={"model": "auto", "prompt": "a lighthouse"})
    assert response.status_code == 200, response.text
    assert (response.headers["x-noelle-routed-model"], response.headers["x-noelle-route-reason"]) == (
        "fake/sd", "recent-model")
    payload = worker.of(Command.GENERATE_IMAGE)[0]["payload"]
    assert not {"width", "height", "seed", "steps"} & set(payload)  # the worker's defaults (512x512)
    settings.set("server.aliases", {"noelle-image": "fake/sd-old"})
    response = client.post("/v1/images/generations", json={"model": "auto", "prompt": "a lighthouse"})
    assert (response.headers["x-noelle-routed-model"], response.headers["x-noelle-route-reason"]) == (
        "fake/sd-old", "image-alias")
    response = client.post("/v1/images/generations", json={"model": "noelle-image", "prompt": "a lighthouse"})
    assert response.headers["x-noelle-model"] == "fake/sd-old" and "x-noelle-route-reason" not in response.headers
    assert worker.of(Command.LIST_LOADED) == []  # images never ask which models are loaded


@pytest.mark.parametrize(("body", "param"), [
    ({"model": "fake/sd"}, "prompt"),
    ({"model": "fake/sd", "prompt": "  "}, "prompt"),
    ({"model": "fake/sd", "prompt": ["a cat"]}, "prompt"),
    ({"model": "fake/sd", "prompt": "a cat", "n": 2}, "n"),
    ({"model": "fake/sd", "prompt": "a cat", "n": True}, "n"),
    ({"model": "fake/sd", "prompt": "a cat", "response_format": "url"}, "response_format"),
    ({"model": "fake/sd", "prompt": "a cat", "response_format": "png"}, "response_format"),
    ({"model": "fake/sd", "prompt": "a cat", "size": "512"}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "size": "500x512"}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "size": "512x4096"}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "size": "248x512"}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "size": "512x512x2"}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "size": " 512x512"}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "size": 512}, "size"),
    ({"model": "fake/sd", "prompt": "a cat", "seed": -1}, "seed"),
    ({"model": "fake/sd", "prompt": "a cat", "seed": 2**32}, "seed"),
    ({"model": "fake/sd", "prompt": "a cat", "seed": "7"}, "seed"),
    ({"model": "fake/sd", "prompt": "a cat", "steps": 0}, "steps"),
    ({"model": "fake/sd", "prompt": "a cat", "steps": 151}, "steps"),
    ({"model": "fake/sd", "prompt": "a cat", "steps": True}, "steps"),
    ({"prompt": "a cat"}, "model"),
    ({"model": "fake/chat", "prompt": "a cat"}, "model"),
])
def test_image_validation(client: TestClient, worker: GatewayWorker, body: dict[str, Any], param: str) -> None:
    response = client.post("/v1/images/generations", json=body)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == param and error["type"] == "invalid_request_error"
    if body.get("response_format") == "url":
        assert "not supported" in error["message"] and "b64_json" in error["message"]
    assert worker.of(Command.GENERATE_IMAGE) == []


def test_image_size_bounds() -> None:
    assert srv.image_size(None) is None
    assert srv.image_size("256x2048") == (256, 2048)
    assert srv.image_size("1024x768") == (1024, 768)


def test_image_errors(client: TestClient, worker: GatewayWorker) -> None:
    response = client.post("/v1/images/generations", json={"model": "fake/oom", "prompt": "a cat"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "oom"
    response = client.post("/v1/images/generations", json={"model": "fake/sd-blank", "prompt": "a cat"})
    assert response.status_code == 502 and response.json()["error"]["code"] == "worker_error"
    response = client.post("/v1/images/generations", json={"model": "nobody/nothing", "prompt": "a cat"})
    assert response.status_code == 404


def test_take_image_only_deletes_worker_outputs(tmp_path: Path) -> None:
    outside = tmp_path / "keep.png"
    outside.write_bytes(PNG_HEAD)
    assert base64.b64decode(srv._take_image(str(outside))) == PNG_HEAD
    assert outside.exists()
    inside = paths.outputs_dir() / "image-x.png"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(PNG_HEAD + b"x")
    assert base64.b64decode(srv._take_image(str(inside))) == PNG_HEAD + b"x"
    assert not inside.exists()


# ---------------------------------------------------------------------------
# knowledge
# ---------------------------------------------------------------------------


def test_knowledge_list(client: TestClient, worker: GatewayWorker) -> None:
    response = client.get("/v1/knowledge")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": COLLECTIONS}
    assert worker.of(Command.RAG_LIST_COLLECTIONS)[0]["payload"] == {}
    worker.collections = []
    assert client.get("/v1/knowledge").json() == {"object": "list", "data": []}


def test_knowledge_query(client: TestClient, worker: GatewayWorker) -> None:
    response = client.post("/v1/knowledge/query", json={"collection": " docs ", "query": "renewal terms",
                                                         "top_k": 2})
    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "collection": "docs", "data": HITS[:2]}
    assert worker.of(Command.RAG_QUERY)[0]["payload"] == {"collection": "docs", "query": "renewal terms", "top_k": 2}
    response = client.post("/v1/knowledge/query", json={"collection": "notes", "query": "x"})
    assert response.json()["data"] == HITS[:4]  # default top_k
    assert worker.of(Command.RAG_QUERY)[1]["payload"]["top_k"] == 4
    response = client.post("/v1/knowledge/query", json={"collection": "notes", "query": "x", "top_k": 20})
    assert response.json()["data"] == HITS


@pytest.mark.parametrize(("body", "param"), [
    ({"query": "x"}, "collection"),
    ({"collection": "", "query": "x"}, "collection"),
    ({"collection": 5, "query": "x"}, "collection"),
    ({"collection": "docs"}, "query"),
    ({"collection": "docs", "query": "   "}, "query"),
    ({"collection": "docs", "query": "x", "top_k": 0}, "top_k"),
    ({"collection": "docs", "query": "x", "top_k": 21}, "top_k"),
    ({"collection": "docs", "query": "x", "top_k": True}, "top_k"),
    ({"collection": "docs", "query": "x", "top_k": "3"}, "top_k"),
])
def test_knowledge_query_validation(client: TestClient, worker: GatewayWorker, body: dict[str, Any],
                                    param: str) -> None:
    response = client.post("/v1/knowledge/query", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert worker.of(Command.RAG_QUERY) == []


def test_knowledge_unknown_collection(client: TestClient, worker: GatewayWorker) -> None:
    response = client.post("/v1/knowledge/query", json={"collection": "nope", "query": "x"})
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "collection_not_found" and error["param"] == "collection"
    assert "docs, notes" in error["message"]
    assert worker.of(Command.RAG_QUERY) == []


def test_knowledge_and_images_need_the_api_key(worker: GatewayWorker, registry: ModelRegistry,
                                               settings: Settings) -> None:
    app = srv.create_app(worker.endpoint, TOKEN, registry=registry, settings=settings, api_key="sk-gw")
    auth = {"Authorization": "Bearer sk-gw"}
    query = {"collection": "docs", "query": "x"}
    image = {"model": "fake/sd", "prompt": "a cat"}
    with TestClient(app, base_url=LOCAL) as client:
        assert client.get("/v1/knowledge").status_code == 401
        assert client.post("/v1/knowledge/query", json=query).status_code == 401
        assert client.post("/v1/images/generations", json=image).status_code == 401
        assert client.get("/v1/knowledge", headers=auth).status_code == 200
        assert client.post("/v1/knowledge/query", json=query, headers=auth).status_code == 200
        assert client.post("/v1/images/generations", json=image, headers=auth).status_code == 200
    assert len(worker.of(Command.RAG_QUERY)) == 1 and len(worker.of(Command.GENERATE_IMAGE)) == 1


def test_knowledge_without_worker(registry: ModelRegistry, settings: Settings) -> None:
    with TestClient(srv.create_app(None, TOKEN, registry=registry, settings=settings), base_url=LOCAL) as client:
        response = client.get("/v1/knowledge")
    assert response.status_code == 503 and response.json()["error"]["code"] == "worker_unavailable"


def test_access_log_has_no_gateway_content(client: TestClient) -> None:
    client.post("/v1/knowledge/query", json={"collection": "docs", "query": "secret question 42"})
    client.post("/v1/images/generations", json={"model": "fake/sd", "prompt": "secret prompt 42"})
    log_text = (paths.logs_dir() / "server-access.log").read_text(encoding="utf-8")
    assert "secret" not in log_text
    assert [line.split()[1:4] for line in log_text.strip().splitlines()] == [
        ["POST", "/v1/knowledge/query", "200"], ["POST", "/v1/images/generations", "200"]]


def test_trust_given_in_the_app_applies_to_api_requests(client: TestClient, worker: GatewayWorker,
                                                        settings: Settings) -> None:
    assert chat(client, "fake/coder").status_code == 200
    settings.set("runtime.trusted_publishers", ["fake"])  # "Trust publisher" in the Workspace
    assert chat(client, "fake/coder").status_code == 200
    first, second = worker.of(Command.GENERATE_STREAM)
    assert first["payload"]["trust_remote_code"] is False and second["payload"]["trust_remote_code"] is True


@pytest.mark.parametrize("host, allowed", [
    ("127.0.0.1:8765", True), ("localhost:8765", True), ("LOCALHOST", True), ("[::1]:8765", True),
    ("evil.example:8765", False), ("127.0.0.1.evil.example", False), ("", False), ("[::1", False),
])
def test_dns_rebinding_is_blocked_by_the_host_check(client: TestClient, host: str, allowed: bool) -> None:
    for path in ("/v1/models", "/v1/knowledge", "/health"):
        response = client.get(path, headers={"host": host})
        if allowed:
            assert response.status_code != 421, (path, response.text)
        else:
            assert response.status_code == 421 and response.json()["error"]["code"] == "invalid_host", path


def test_json_endpoints_require_a_json_content_type(client: TestClient, worker: GatewayWorker) -> None:
    body = '{"collection": "docs", "query": "reset", "top_k": 2}'
    for path in ("/v1/knowledge/query", "/v1/chat/completions"):
        response = client.post(path, content=body, headers={"content-type": "text/plain"})
        assert response.status_code == 415 and response.json()["error"]["code"] == "unsupported_media_type"
    assert not worker.of(Command.RAG_QUERY) and not worker.of(Command.GENERATE_STREAM)  # nothing reached the worker
    ok = client.post("/v1/knowledge/query", content=body, headers={"content-type": "application/json; charset=utf-8"})
    assert ok.status_code == 200, ok.text
