from __future__ import annotations

import time

from noelle.runtime.protocol import Command, Event
from noelle.runtime.worker_service import MAX_SUBSCRIBERS, Worker


def test_idle_subscribers_are_not_expired_and_table_is_capped() -> None:
    worker = Worker(port=0)
    try:
        worker.subscribers[b"idle-ui"] = time.time() - 3600  # silent for an hour
        worker.broadcast(Event.HEARTBEAT, {"ok": True})  # unknown peers are dropped by ROUTER, no error
        assert b"idle-ui" in worker.subscribers
        for i in range(MAX_SUBSCRIBERS + 5):
            worker._handle_light(f"client-{i}".encode(), f"r{i}", Command.HELLO, {})
        assert len(worker.subscribers) == MAX_SUBSCRIBERS
        assert b"idle-ui" not in worker.subscribers  # oldest goes first when the cap is hit
    finally:
        worker.router.close(linger=0)
        worker.outbox.close(linger=0)
