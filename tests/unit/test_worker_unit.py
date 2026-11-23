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


def test_document_listing_and_removal_go_through_the_worker(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    from noelle.runtime import worker_service
    from noelle.runtime.errors import BadRequest

    class Embedder:
        dimension = 8

        def encode(self, texts, normalize=True, **_kw):  # type: ignore[no-untyped-def]
            return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    docs = []
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(f"{name} is a short document about testing.", encoding="utf-8")
        docs.append(str(tmp_path / name))
    worker = Worker(port=0)
    sent: list[tuple[str, dict]] = []
    worker.send = lambda _identity, _rid, event, payload: sent.append((str(event), payload))  # type: ignore[method-assign]
    try:
        worker.rag().ingest("notes", docs, Embedder(), embedding_model="test/embedder")
        worker._handle_light(b"ui", "r1", Command.RAG_LIST_DOCUMENTS, {})
        listed = sent[-1][1]["documents"]
        assert [d["name"] for d in listed["notes"]] == ["a.txt", "b.txt"]

        class Ctx:
            def __init__(self, payload: dict) -> None:
                self.payload = payload
                self.events: list[tuple[str, dict]] = []

            def emit(self, event, payload=None):  # type: ignore[no-untyped-def]
                self.events.append((str(event), payload or {}))

        ctx = Ctx({"collection": "notes", "paths": [docs[0], str(tmp_path / "missing.txt")]})
        worker_service._cmd_rag_delete_document(worker, ctx)  # type: ignore[arg-type]
        assert ctx.events == [("RESULT", {"collection": "notes", "removed": 1})]
        worker._handle_light(b"ui", "r2", Command.RAG_LIST_DOCUMENTS, {"collection": "notes"})
        assert [d["name"] for d in sent[-1][1]["documents"]["notes"]] == ["b.txt"]
        try:
            worker_service._cmd_rag_delete_document(worker, Ctx({"collection": "notes", "paths": []}))  # type: ignore[arg-type]
        except BadRequest:
            pass
        else:
            raise AssertionError("an empty path list must be rejected")
    finally:
        worker.rag().close()
        worker.router.close(linger=0)
        worker.outbox.close(linger=0)
