"""The Qt runtime client against a real worker: streaming, crash containment and restart."""

from __future__ import annotations

import os
from typing import Any

import pytest

from noelle.runtime.hub_cache import cached_snapshot_path
from noelle.runtime.protocol import Command

pytestmark = [pytest.mark.e2e, pytest.mark.integration]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture()
def runtime(qtbot: Any) -> Any:
    if cached_snapshot_path(MODEL) is None:
        pytest.skip(f"{MODEL} is not cached")
    from noelle.app.ipc.runtime_client import RuntimeClient
    from noelle.config import get_settings

    client = RuntimeClient(get_settings())
    client.start()
    qtbot.waitUntil(lambda: client.state == "ready", timeout=90_000)
    yield client
    client.stop()


def test_streaming_through_qt_signals(runtime: Any, qtbot: Any) -> None:
    tokens: list[str] = []
    done: list[dict[str, Any]] = []
    handle = runtime.request(Command.GENERATE_STREAM, {"model_id": MODEL, "messages": [
        {"role": "user", "content": "Say hello."}], "params": {"temperature": 0, "max_new_tokens": 16}})
    handle.token.connect(tokens.append)
    handle.finished.connect(done.append)
    qtbot.waitUntil(lambda: bool(done), timeout=60_000)
    assert tokens and "".join(tokens) == done[0]["text"]


def test_worker_crash_is_contained_and_restartable(runtime: Any, qtbot: Any) -> None:
    failures: list[dict[str, Any]] = []
    crashes: list[tuple[int, str]] = []
    runtime.worker_crashed.connect(lambda code, why: crashes.append((code, why)))
    handle = runtime.request(Command.GENERATE_STREAM, {"model_id": MODEL, "messages": [
        {"role": "user", "content": "Write a long essay about rivers."}], "params": {"max_new_tokens": 800}})
    handle.failed.connect(failures.append)
    qtbot.wait(800)
    runtime.kill_for_test()  # SIGKILL, like the OS killing the process under memory pressure
    qtbot.waitUntil(lambda: runtime.state == "crashed" and bool(failures), timeout=15_000)
    assert failures[0]["code"] == "WORKER_CRASHED"
    assert crashes and "memory" in crashes[0][1].lower()
    runtime.restart()
    qtbot.waitUntil(lambda: runtime.state == "ready", timeout=90_000)
    done: list[dict[str, Any]] = []
    again = runtime.request(Command.PING)
    again.finished.connect(done.append)
    qtbot.waitUntil(lambda: bool(done), timeout=10_000)
    assert done[0]["pong"] is True
