"""Release acceptance checks that can be verified on a developer machine."""

from __future__ import annotations

import ipaddress
import statistics
import threading
import time
from typing import Any

import psutil
import pytest

from noelle.runtime.client import build_worker_env
from noelle.runtime.protocol import AUTH_ENV, Command, Event

TEXT = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.network
def test_hub_pages_arrive_within_800ms() -> None:
    """Discovery responsiveness: a page of model cards from the Hub in under 800 ms (median of 5)."""
    from noelle.app.services.hf_client import HFClient

    client = HFClient()
    client.pager(modality="text").fetch_next()  # warm up DNS/TLS like a running app would have
    timings = []
    for sort in ("trending", "downloads", "likes", "recent", "trending"):
        pager = client.pager(modality="text", sort=sort)
        started = time.perf_counter()
        models = pager.fetch_next()
        timings.append(time.perf_counter() - started)
        assert len(models) >= 20
    assert statistics.median(timings) < 0.8, timings


def test_worker_environment_has_no_credentials_and_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak")
    env = build_worker_env("worker-token", offline=True)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert env[AUTH_ENV] == "worker-token"


@pytest.mark.integration
def test_worker_only_uses_loopback_sockets_while_generating(worker_process: Any) -> None:
    """Zero data egress (process-level proxy for packet inspection): every socket the worker
    holds during a generation is bound to or connected to a loopback address."""
    client = worker_process.client()
    proc = psutil.Process(worker_process.process.pid)
    seen: list[Any] = []
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            try:
                seen.extend(proc.net_connections(kind="inet"))
            except psutil.AccessDenied:
                return
            time.sleep(0.05)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        events = list(client.stream(Command.GENERATE_STREAM, {
            "model_id": TEXT, "messages": [{"role": "user", "content": "Tell me a fact about the moon."}],
            "params": {"max_new_tokens": 48}}))
    finally:
        done.set()
        sampler.join()
    assert events[-1]["event"] == Event.GENERATION_COMPLETE
    if not seen:
        pytest.skip("socket table not readable for this process on this OS")
    for conn in seen:
        for address in (conn.laddr, conn.raddr):
            if address:
                assert ipaddress.ip_address(address.ip).is_loopback, conn
