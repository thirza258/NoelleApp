"""Integration tests against a real worker process (models read from the local HF cache)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pytest

from noelle.runtime.client import WorkerClient, WorkerError, spawn_worker
from noelle.runtime.hub_cache import cached_snapshot_path
from noelle.runtime.protocol import Command, Event, make_request

pytestmark = pytest.mark.integration

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MEDIUM = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _require(*repos: str) -> None:
    for repo in repos:
        if cached_snapshot_path(repo) is None:
            pytest.skip(f"{repo} is not in the local Hugging Face cache")


def _generate(client: WorkerClient, model: str, prompt: str, **extra: Any) -> tuple[list[str], dict[str, Any]]:
    tokens: list[str] = []
    final: dict[str, Any] = {}
    for message in client.stream(Command.GENERATE_STREAM, {"model_id": model, "messages": [
            {"role": "user", "content": prompt}], "params": {"temperature": 0, "max_new_tokens": 24}, **extra}):
        if message["event"] == Event.TOKEN_STREAM:
            tokens.append(message["payload"]["text"])
        elif message["event"] in (Event.GENERATION_COMPLETE, Event.ERROR):
            final = message
    return tokens, final


def test_stream_metrics_and_stop(worker_process: Any) -> None:
    client = worker_process.client()
    hello = client.call(Command.HELLO)
    assert hello["pid"] == worker_process.process.pid
    tokens, final = _generate(client, SMALL, "Say hello in five words.")
    assert final["event"] == Event.GENERATION_COMPLETE
    assert len(tokens) > 1 and "".join(tokens) == final["payload"]["text"]
    metrics = final["payload"]["metrics"]
    assert metrics["ttft_ms"] > 0 and metrics["tokens_per_second"] > 0 and metrics["completion_tokens"] > 0
    assert metrics["peak_memory_bytes"] and metrics["engine"] == "transformers"

    events: list[dict[str, Any]] = []

    def run() -> None:
        events.extend(client.stream(Command.GENERATE_STREAM, {
            "model_id": SMALL, "messages": [{"role": "user", "content": "Write a very long story about the sea."}],
            "params": {"temperature": 0.8, "max_new_tokens": 600}}, request_id="long-run"))

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(1.0)
    assert client.stop("long-run")["stopped"] == ["long-run"]
    thread.join(timeout=60)
    assert events[-1]["event"] == Event.GENERATION_COMPLETE
    assert events[-1]["payload"]["metrics"]["finish_reason"] == "cancelled"
    assert events[-1]["payload"]["metrics"]["completion_tokens"] < 600


def test_typed_errors_and_auth(worker_process: Any) -> None:
    client = worker_process.client()
    with pytest.raises(WorkerError) as missing:
        client.call(Command.LOAD_MODEL, {"model_id": "nobody/not-downloaded-1B"})
    assert missing.value.code == "MODEL_NOT_FOUND" and missing.value.hint
    with pytest.raises(WorkerError) as bad:
        client.call(Command.GENERATE_STREAM, {"model_id": SMALL, "messages": []})
    assert bad.value.code == "BAD_REQUEST"
    intruder = WorkerClient(worker_process.endpoint, "wrong-token")
    with pytest.raises(WorkerError) as denied:
        intruder.call(Command.PING, timeout=10)
    assert denied.value.code == "UNAUTHORIZED"


def test_heartbeat_broadcast_reports_memory(worker_process: Any) -> None:
    import zmq

    sock = zmq.Context.instance().socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(worker_process.endpoint)
    try:
        _, frame = make_request(Command.HELLO, {}, auth=worker_process.token)
        sock.send(frame)
        deadline = time.time() + 15
        beat = None
        from noelle.runtime.protocol import decode

        while time.time() < deadline and beat is None:
            if sock.poll(1000):
                message = decode(sock.recv())
                if message.get("event") == Event.HEARTBEAT and message["payload"].get("runtime_ready"):
                    beat = message["payload"]
        assert beat is not None
        assert beat["memory"]["total"] > 0 and beat["budget"]["usable"] < beat["budget"]["total"]
    finally:
        sock.close(linger=0)


def test_injected_oom_is_contained_and_worker_keeps_serving() -> None:
    _require(SMALL)
    proc = spawn_worker(parent_pid=os.getpid(), extra_env={"NOELLE_FAULT_INJECTION": "1"})
    try:
        client = proc.client()
        _, final = _generate(client, SMALL, "hi", _inject="oom")
        assert final["event"] == Event.ERROR
        assert final["payload"]["code"] == "OOM" and final["payload"]["recovered"] is True
        assert "4-bit" in final["payload"]["hint"]
        assert proc.process.poll() is None  # still alive
        tokens, ok = _generate(client, SMALL, "Say hi.")
        assert ok["event"] == Event.GENERATION_COMPLETE and tokens
    finally:
        proc.stop()


def test_hot_swap_queue_under_a_small_budget() -> None:
    """Two models that do not fit together: the queue must unload View 1 before loading View 2."""
    _require(SMALL, MEDIUM)
    proc = spawn_worker(parent_pid=os.getpid(), memory_budget_gb=1.5)
    try:
        client = proc.client()
        events = list(client.stream(Command.RUN_BENCHMARK, {
            "views": [{"view": 1, "model_id": SMALL}, {"view": 2, "model_id": MEDIUM}],
            "prompt": "Name a color.", "params": {"temperature": 0, "max_new_tokens": 8}}))
        plan = next(e["payload"] for e in events if e["event"] == Event.QUEUE_PLAN)
        assert plan["mode"] == "hot_swap"
        started = [e["payload"] for e in events if e["event"] == Event.VIEW_STARTED]
        assert [s["view"] for s in started] == [1, 2]
        assert started[1]["evict_before"] and started[1]["evict_before"][0].startswith(SMALL)
        done = [e["payload"] for e in events if e["event"] == Event.VIEW_COMPLETE]
        assert [d["status"] for d in done] == ["finished", "finished"]
        assert all(d["metrics"]["model_peak_bytes"] > 0 for d in done)
        assert events[-1]["event"] == Event.GENERATION_COMPLETE
        loaded = [m["model_id"] for m in client.call(Command.LIST_LOADED)["models"]]
        assert loaded == [MEDIUM]
    finally:
        proc.stop()


def test_retain_queue_reuses_resident_models(worker_process: Any) -> None:
    client = worker_process.client()
    events = list(client.stream(Command.RUN_BENCHMARK, {
        "views": [{"view": 1, "model_id": SMALL}, {"view": 2, "model_id": SMALL, "prompt": "Count to three."}],
        "prompt": "Name a fruit.", "params": {"temperature": 0, "max_new_tokens": 8}}))
    plan = next(e["payload"] for e in events if e["event"] == Event.QUEUE_PLAN)
    assert plan["mode"] == "retain"
    started = [e["payload"] for e in events if e["event"] == Event.VIEW_STARTED]
    assert started[1]["reuse_loaded"] is True and not started[1]["evict_before"]
    tokens_by_view: dict[int, str] = {}
    for event in events:
        if event["event"] == Event.TOKEN_STREAM:
            view = event["payload"]["view"]
            tokens_by_view[view] = tokens_by_view.get(view, "") + event["payload"]["text"]
    assert set(tokens_by_view) == {1, 2}
