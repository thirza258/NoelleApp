"""Out-of-process inference daemon.

Topology::

    clients (DEALER) ──tcp://127.0.0.1:<port>──▶ ROUTER  [I/O thread]
                                                   │  light commands answered inline
                                                   ▼
                                           job queue (FIFO)
                                                   │
                                                   ▼
                                          GPU lane thread  ──inproc PUSH──▶ PULL ─▶ ROUTER

Only the I/O thread touches the ROUTER socket. The single GPU lane executes
every model operation in arrival order, which is what keeps concurrent
playground, bench and API-server traffic from oversubscribing the GPU. A
memory error inside a job is caught, the allocator cache is flushed and the
client gets an ``ERROR{code: OOM}``; if the OS kills the process instead, the
desktop shell notices the exit and offers a restart.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import psutil
import zmq

from noelle import __version__, paths
from noelle.runtime.protocol import (
    AUTH_ENV,
    LIGHT_COMMANDS,
    READY_PREFIX,
    Command,
    ErrorCode,
    Event,
    GenerationParams,
    ProtocolError,
    error_payload,
    make_event,
    parse_request,
)

log = logging.getLogger("noelle.worker")
OUTBOX = "inproc://noelle-worker-outbox"
HEARTBEAT_INTERVAL_S = 1.0
MAX_SUBSCRIBERS = 64


@dataclass
class Job:
    identity: bytes
    request_id: str
    cmd: Command
    payload: dict[str, Any]
    stop_event: threading.Event = field(default_factory=threading.Event)
    enqueued_at: float = field(default_factory=time.time)


class JobContext:
    def __init__(self, worker: Worker, job: Job) -> None:
        self.worker = worker
        self.job = job
        self.payload = job.payload

    @property
    def stop_event(self) -> threading.Event:
        return self.job.stop_event

    def emit(self, event: Event, payload: dict[str, Any] | None = None) -> None:
        self.worker.send(self.job.identity, self.job.request_id, event, payload)

    def status(self, stage: str, message: str, **extra: Any) -> None:
        self.emit(Event.STATUS, {"stage": stage, "message": message, **extra})

    def progress(self, current: float, total: float | None = None, **extra: Any) -> None:
        self.emit(Event.PROGRESS, {"current": current, "total": total, **extra})


class Worker:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, auth_token: str | None = None,
                 device: str = "auto", memory_budget_gb: float | None = None, headroom: float = 0.15,
                 keep_loaded: bool = True, parent_pid: int | None = None) -> None:
        self.ctx = zmq.Context.instance()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.setsockopt(zmq.SNDHWM, 0)
        self.router.setsockopt(zmq.RCVHWM, 0)
        if port:
            self.router.bind(f"tcp://{host}:{port}")
            self.port = port
        else:
            self.port = self.router.bind_to_random_port(f"tcp://{host}")
        self.host = host
        self.outbox = self.ctx.socket(zmq.PULL)
        self.outbox.setsockopt(zmq.RCVHWM, 0)
        self.outbox.bind(OUTBOX)
        self._local = threading.local()
        self._push_sockets: list[zmq.Socket] = []
        self._push_lock = threading.Lock()
        self._main_ident = threading.get_ident()

        self.auth_token = auth_token
        self.device_pref = device
        self.memory_budget_gb = memory_budget_gb
        self.headroom = headroom
        self.keep_loaded = keep_loaded
        self.parent_pid = parent_pid

        self.jobs: queue.Queue[Job | None] = queue.Queue()
        self._jobs_lock = threading.Lock()
        self.pending: dict[str, Job] = {}
        self.current: Job | None = None
        self.subscribers: dict[bytes, float] = {}
        self.started = time.time()
        self._backends: dict[str, bool] | None = None
        self._backends_at = 0.0
        self.running = True
        self.runtime_ready = threading.Event()
        self.init_error: str | None = None
        self.models: Any = None  # ModelManager, created on the GPU lane after torch imports
        self.device: str | None = None
        self._rag_engine: Any = None
        self._gpu_thread = threading.Thread(target=self._gpu_loop, name="noelle-gpu-lane", daemon=True)

    # ------------------------------------------------------------------ I/O
    def send(self, identity: bytes, request_id: str | None, event: Event, payload: dict[str, Any] | None = None) -> None:
        frames = [identity, make_event(request_id, event, payload)]
        if threading.get_ident() == self._main_ident:
            self.router.send_multipart(frames)
            return
        sock = getattr(self._local, "push", None)
        if sock is None:
            sock = self.ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.setsockopt(zmq.LINGER, 1000)
            sock.connect(OUTBOX)
            self._local.push = sock
            with self._push_lock:
                self._push_sockets.append(sock)
        sock.send_multipart(frames)

    def broadcast(self, event: Event, payload: dict[str, Any]) -> None:
        # Idle clients send nothing, so subscribers are never expired by time; ROUTER silently
        # drops frames for peers that have disconnected, and HELLO caps the table size.
        for identity in list(self.subscribers):
            self.send(identity, None, event, payload)

    def serve(self) -> int:
        self._write_discovery_file()
        print(f"{READY_PREFIX} port={self.port} pid={os.getpid()}", flush=True)
        log.info("worker %s listening on %s:%s", __version__, self.host, self.port)
        self._gpu_thread.start()
        if self.parent_pid:
            threading.Thread(target=self._watch_parent, name="noelle-parent-watch", daemon=True).start()
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)
        poller.register(self.outbox, zmq.POLLIN)
        next_heartbeat = time.monotonic()
        try:
            while self.running:
                ready = dict(poller.poll(timeout=200))
                if self.outbox in ready:
                    self._drain_outbox()
                if self.router in ready:
                    while True:
                        try:
                            frames = self.router.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        if len(frames) >= 2:
                            self._handle(frames[0], frames[-1])
                now = time.monotonic()
                if now >= next_heartbeat:
                    next_heartbeat = now + HEARTBEAT_INTERVAL_S
                    if self.subscribers:
                        self.broadcast(Event.HEARTBEAT, self.heartbeat_payload())
        finally:
            self._shutdown()
        return 0

    def _drain_outbox(self) -> None:
        while True:
            try:
                frames = self.outbox.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            self.router.send_multipart(frames)

    def _handle(self, identity: bytes, data: bytes) -> None:
        try:
            request_id, cmd, payload, auth = parse_request(data)
        except ProtocolError as exc:
            self.send(identity, None, Event.ERROR, error_payload(ErrorCode.BAD_REQUEST, str(exc)))
            return
        if self.auth_token and auth != self.auth_token:
            self.send(identity, request_id, Event.ERROR, error_payload(ErrorCode.UNAUTHORIZED, "Invalid worker token."))
            return
        if identity in self.subscribers:
            self.subscribers[identity] = time.time()
        if cmd in LIGHT_COMMANDS:
            try:
                self._handle_light(identity, request_id, cmd, payload)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("light command %s failed", cmd)
                self.send(identity, request_id, Event.ERROR, error_payload(ErrorCode.INTERNAL, str(exc)))
            return
        job = Job(identity, request_id, cmd, payload)
        with self._jobs_lock:
            position = len(self.pending)  # jobs ahead of this one, excluding the running job
            busy = self.current is not None
            self.pending[request_id] = job
        self.send(identity, request_id, Event.ACK, {"queued": busy or position > 0, "position": position})
        self.jobs.put(job)

    # --------------------------------------------------------- light commands
    def _handle_light(self, identity: bytes, request_id: str, cmd: Command, payload: dict[str, Any]) -> None:
        if cmd == Command.HELLO:
            self.subscribers[identity] = time.time()
            while len(self.subscribers) > MAX_SUBSCRIBERS:  # forget the oldest reconnected clients
                self.subscribers.pop(min(self.subscribers, key=self.subscribers.__getitem__), None)
            self.send(identity, request_id, Event.RESULT, self.hello_payload())
        elif cmd == Command.PING:
            self.send(identity, request_id, Event.RESULT, {"pong": True, "pid": os.getpid(),
                                                           "uptime_s": time.time() - self.started,
                                                           "runtime_ready": self.runtime_ready.is_set()})
        elif cmd == Command.HARDWARE_STATS:
            self.send(identity, request_id, Event.RESULT, self.heartbeat_payload())
        elif cmd == Command.LIST_LOADED:
            models = self.models.snapshot() if self.models is not None else []
            self.send(identity, request_id, Event.RESULT, {"models": models})
        elif cmd == Command.STOP_GENERATION:
            stopped = self.stop(payload.get("target_id"))
            self.send(identity, request_id, Event.RESULT, {"stopped": stopped})
        elif cmd == Command.RAG_LIST_COLLECTIONS:
            self.send(identity, request_id, Event.RESULT, {"collections": self.rag().list_collections()})
        elif cmd == Command.RAG_LIST_DOCUMENTS:
            collection = payload.get("collection") if isinstance(payload.get("collection"), str) else None
            self.send(identity, request_id, Event.RESULT, {"documents": self.rag().list_documents(collection)})
        elif cmd == Command.QUANTIZE_CAPABILITIES:
            from noelle.runtime import quantization_engine

            device = self.device or ("mps" if sys.platform == "darwin" else "cpu")
            self.send(identity, request_id, Event.RESULT,
                      {"device": device, "methods": quantization_engine.capabilities(device)})
        elif cmd == Command.SHUTDOWN:
            self.send(identity, request_id, Event.RESULT, {"shutting_down": True})
            self.running = False

    def stop(self, target_id: str | None) -> list[str]:
        stopped: list[str] = []
        with self._jobs_lock:
            jobs = list(self.pending.values())
            if self.current is not None:
                jobs.append(self.current)
        for job in jobs:
            if target_id in (None, "all", job.request_id):
                job.stop_event.set()
                stopped.append(job.request_id)
        return stopped

    def hello_payload(self) -> dict[str, Any]:
        return {"version": __version__, "pid": os.getpid(), "port": self.port,
                "runtime_ready": self.runtime_ready.is_set(), "device": self.device, "init_error": self.init_error}

    def heartbeat_payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "pid": os.getpid(),
            "uptime_s": time.time() - self.started,
            "runtime_ready": self.runtime_ready.is_set(),
            "init_error": self.init_error,
            "device": self.device,
            "queue_depth": self.jobs.qsize(),
            "busy": str(self.current.cmd) if self.current is not None else None,
            "loaded": [],
        }
        if self.models is not None and self.device is not None:
            from noelle.runtime import devices

            data["memory"] = devices.memory_snapshot(self.device)
            budget = self.models.budget()
            data["budget"] = {"total": budget.total_bytes, "usable": budget.usable_bytes, "headroom": budget.headroom}
            data["loaded"] = self.models.snapshot()
            if "mlx.core" in sys.modules:
                try:
                    import mlx.core as mx

                    data["memory"]["mlx_active"] = int(mx.get_active_memory())
                except Exception:
                    pass
        else:
            vm = psutil.virtual_memory()
            data["memory"] = {"system_total": vm.total, "system_available": vm.available,
                              "process_rss": psutil.Process().memory_info().rss}
        data["backends"] = self.backends()
        return data

    def backends(self) -> dict[str, bool]:
        """Optional engines available in *this* environment (the UI may run in another one).

        Re-checked every 30 s so a component installed while the app runs shows up.
        """
        now = time.time()
        if self._backends is None or now - self._backends_at > 30:
            from noelle.runtime.capabilities import detect_backends

            try:
                self._backends = detect_backends()
            except Exception:  # telemetry must never take the worker down
                log.debug("backend detection failed", exc_info=True)
                self._backends = {}
            self._backends_at = now
        return self._backends

    # --------------------------------------------------------------- GPU lane
    def _init_runtime(self) -> None:
        from noelle.runtime import devices
        from noelle.runtime.model_manager import ModelManager

        self.models = ModelManager(self.device_pref, memory_override_gb=self.memory_budget_gb,
                                   headroom=self.headroom, keep_loaded=self.keep_loaded,
                                   on_change=self._models_changed)
        self.device = self.models.device
        log.info("runtime ready on %s (%s)", self.device, devices.device_name(self.device))

    def _models_changed(self) -> None:
        if self.models is not None:
            self.broadcast(Event.MODELS_CHANGED, {"loaded": self.models.snapshot()})

    def _gpu_loop(self) -> None:
        try:
            self._init_runtime()
        except Exception as exc:
            self.init_error = f"{type(exc).__name__}: {exc}"
            log.exception("runtime initialisation failed")
        finally:
            self.runtime_ready.set()
        while True:
            job = self.jobs.get()
            if job is None:
                return
            with self._jobs_lock:
                self.pending.pop(job.request_id, None)
                self.current = job
            started = time.perf_counter()
            try:
                if job.stop_event.is_set():
                    from noelle.runtime.errors import Cancelled

                    raise Cancelled("The request was cancelled before it started.")
                if self.init_error:
                    from noelle.runtime.errors import DependencyMissing

                    raise DependencyMissing(f"The runtime failed to start: {self.init_error}",
                                            hint='Install the runtime extras: pip install "noelle[runtime]"')
                self._run_job(JobContext(self, job))
            except BaseException as exc:  # noqa: BLE001 - report everything to the client
                self._report_failure(job, exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                with self._jobs_lock:
                    self.current = None
                log.info("%s finished in %.0f ms", job.cmd, (time.perf_counter() - started) * 1000)

    def _report_failure(self, job: Job, exc: BaseException) -> None:
        from noelle.runtime.errors import NoelleRuntimeError

        oom = False
        if self.models is not None:
            from noelle.runtime import devices

            oom = devices.is_oom(exc)
            if oom:
                devices.empty_cache(self.device)
        if oom:
            snapshot: dict[str, Any] = {}
            if self.models is not None and self.device:
                from noelle.runtime import devices

                snapshot = devices.memory_snapshot(self.device)
            payload = error_payload(
                ErrorCode.OOM, "The model ran out of accelerator memory. The allocator cache was flushed.",
                hint="Lower max tokens, unload other models, or switch to a 4-bit build.", memory=snapshot,
                recovered=True)
            log.warning("%s: out of memory (%s)", job.cmd, exc)
        elif isinstance(exc, NoelleRuntimeError):
            payload = error_payload(exc.code, exc.message, exc.hint, **getattr(exc, "extra", {}))
        else:
            log.error("%s failed:\n%s", job.cmd, "".join(traceback.format_exception(exc)))
            payload = error_payload(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}",
                                    detail="".join(traceback.format_exception(exc))[-4000:])
        self.send(job.identity, job.request_id, Event.ERROR, payload)

    def _run_job(self, ctx: JobContext) -> None:
        handler = _HANDLERS.get(ctx.job.cmd)
        if handler is None:
            from noelle.runtime.errors import BadRequest

            raise BadRequest(f"Command {ctx.job.cmd} is not handled by the GPU lane.")
        handler(self, ctx)

    # --------------------------------------------------------------- helpers
    def rag(self) -> Any:
        if self._rag_engine is None:
            from noelle.runtime.rag_engine import RagEngine

            self._rag_engine = RagEngine(paths.vector_store_dir())
        return self._rag_engine

    def spec(self, payload: dict[str, Any], default_task: str) -> Any:
        from noelle.runtime.model_manager import ModelSpec

        data = dict(payload)
        data.setdefault("task", default_task)
        if not data.get("task"):
            data["task"] = default_task
        return ModelSpec.from_payload(data)

    # ---------------------------------------------------------- lifecycle
    def _watch_parent(self) -> None:
        while self.running:
            time.sleep(2.0)
            if not psutil.pid_exists(self.parent_pid or 0):
                log.warning("parent process %s exited; shutting down", self.parent_pid)
                self.running = False
                self.stop(None)
                time.sleep(3.0)
                os._exit(0)

    def _write_discovery_file(self) -> None:
        try:
            paths.runtime_dir().mkdir(parents=True, exist_ok=True)
            (paths.runtime_dir() / "worker.json").write_text(
                json.dumps({"host": self.host, "port": self.port, "pid": os.getpid(), "started": self.started,
                            "version": __version__}), encoding="utf-8")
        except OSError:
            pass

    def _shutdown(self) -> None:
        self.running = False
        self.stop(None)
        self.jobs.put(None)
        self._gpu_thread.join(timeout=5.0)
        try:
            discovery = paths.runtime_dir() / "worker.json"
            data = json.loads(discovery.read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid():
                discovery.unlink()
        except (OSError, ValueError):
            pass
        with self._push_lock:
            for sock in self._push_sockets:
                sock.close(linger=0)
        self.outbox.close(linger=0)
        self.router.close(linger=0)
        log.info("worker stopped")


# ============================================================================
# GPU lane command handlers
# ============================================================================


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "")
    return ""


def compose_system_prompt(system_prompt: str | None, blocks: list[str]) -> str | None:
    if not blocks:
        return system_prompt
    guide = ("Answer using the context below when it is relevant. Cite sources inline with their bracketed "
             "numbers, like [1]. If the context does not contain the answer, say so.")
    context = "\n\n".join(blocks)
    base = f"{system_prompt}\n\n" if system_prompt else ""
    return f"{base}{guide}\n\n### Context\n{context}"


def _cmd_load(worker: Worker, ctx: JobContext) -> None:
    spec = worker.spec(ctx.payload, "text-generation")
    already = worker.models.get(spec.key(worker.models.resolve_device(spec))) is not None
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    ctx.emit(Event.RESULT, {"model": loaded.describe(), "already_loaded": already})


def _cmd_unload(worker: Worker, ctx: JobContext) -> None:
    key = ctx.payload.get("key")
    if key:
        unloaded = [key] if worker.models.unload(key) else []
    elif ctx.payload.get("all"):
        unloaded = worker.models.unload_matching(None)
    else:
        unloaded = worker.models.unload_matching(ctx.payload.get("model_id"))
    ctx.emit(Event.RESULT, {"unloaded": unloaded})


def _embedder_for(worker: Worker, model_id: str, ctx: JobContext) -> Any:
    from noelle.runtime.model_manager import ModelSpec

    spec = ModelSpec(model_id=model_id, task="sentence-similarity",
                     local_path=ctx.payload.get("embedding_local_path"))
    return worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))


def _web_engine(payload: dict[str, Any]) -> Any:
    from noelle.config import get_settings
    from noelle.runtime.web_search_engine import WebSearchEngine

    settings = get_settings()
    web = payload or {}
    return WebSearchEngine(
        web.get("searxng_url") or settings.get("web_search.searxng_url"),
        timeout=float(web.get("timeout_s") or settings.get("web_search.timeout_s", 10)),
        results_per_query=int(web.get("results_per_query") or settings.get("web_search.results_per_query", 4)),
        context_chars=int(web.get("context_chars") or settings.get("web_search.context_chars", 8000)),
    )


def _ground(worker: Worker, ctx: JobContext, loaded: Any, messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    from noelle.config import get_settings

    grounding = ctx.payload.get("grounding") or {}
    question = _last_user_text(messages)
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    settings = get_settings()
    rag_cfg = grounding.get("rag") or {}
    if rag_cfg.get("collection") and question:
        rag = worker.rag()
        info = rag.collection_info(rag_cfg["collection"])
        if info is None:
            from noelle.runtime.errors import BadRequest

            raise BadRequest(f"Document collection '{rag_cfg['collection']}' does not exist.")
        ctx.status("retrieve", f"Searching '{rag_cfg['collection']}'…")
        embedder = _embedder_for(worker, info["embedding_model"], ctx)
        with worker.models.pinned(embedder.key):
            hits = rag.query(rag_cfg["collection"], question, embedder.engine,
                             top_k=int(rag_cfg.get("top_k") or settings.get("rag.top_k", 4)))
        budget = int(rag_cfg.get("context_chars") or settings.get("rag.context_chars", 6000))
        start = len(sources) + 1
        if hits:
            context = rag.format_context(hits, budget, start_index=start)
            if context.strip():
                blocks.append(context)
            for offset, hit in enumerate(hits):
                index = start + offset
                if f"[{index}]" in context:  # only sources that fit the context budget
                    sources.append({"index": index, "kind": "document", **hit})
        else:
            ctx.status("retrieve", "No matching passages in the collection.")
    web_cfg = grounding.get("web") or {}
    if web_cfg.get("enabled") and question:
        engine = _web_engine(web_cfg)

        def generate_queries(prompt: str) -> list[str]:
            ctx.status("web", "Writing search queries…")
            text = loaded.engine.complete(
                "Write up to 3 short web search queries (one per line, no numbering, no quotes) that would "
                f"find information to answer this request:\n\n{prompt}", max_new_tokens=64)
            queries = []
            for line in text.splitlines():
                cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("\"'")
                if cleaned:
                    queries.append(cleaned)
            return queries[:3]

        embedder = None
        embed_model = web_cfg.get("embedding_model") or settings.get("rag.embedding_model")
        try:
            embedder = _embedder_for(worker, embed_model, ctx)
        except Exception as exc:  # reranking is optional; fall back to lexical order
            ctx.status("web", f"Embedding model unavailable ({exc}); using lexical ranking.")
        from noelle.runtime.errors import BadRequest

        try:
            with worker.models.pinned(*([embedder.key] if embedder else [])):
                outcome = engine.ground(question, generate_queries=generate_queries,
                                        embedder=embedder.engine if embedder else None,
                                        max_queries=int(web_cfg.get("max_queries")
                                                        or settings.get("web_search.max_queries", 3)),
                                        on_status=lambda msg: ctx.status("web", msg), stop_event=ctx.stop_event,
                                        start_index=len(sources) + 1)
        except BadRequest as exc:  # search provider down or misconfigured: answer without web context
            ctx.status("web", f"Web search unavailable: {exc.message}", warning=True, hint=exc.hint)
            outcome = {}
        if outcome.get("context"):
            blocks.append(outcome["context"])
        for source in outcome.get("sources", []):
            sources.append({"kind": "web", **source})
    return blocks, sources


def _maybe_inject_fault(payload: dict[str, Any]) -> None:
    """Test hook: only active when the worker was started with NOELLE_FAULT_INJECTION=1."""
    if os.environ.get("NOELLE_FAULT_INJECTION") != "1":
        return
    fault = payload.get("_inject")
    if fault == "oom":
        import torch

        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB (injected by test)")
    if fault == "crash":
        os._exit(137)


def _cmd_generate(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime import lora_manager
    from noelle.runtime.errors import BadRequest

    spec = worker.spec(ctx.payload, "text-generation")
    messages = ctx.payload.get("messages") or []
    if not messages and ctx.payload.get("prompt"):
        messages = [{"role": "user", "content": str(ctx.payload["prompt"])}]
    if not messages:
        raise BadRequest("No messages to respond to.")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    if not hasattr(loaded.engine, "stream_chat"):
        raise BadRequest(f"{spec.model_id} is not a text-generation model.")
    with worker.models.pinned(loaded.key):
        _maybe_inject_fault(ctx.payload)
        if "adapter" in ctx.payload and ctx.payload["adapter"] != loaded.active_adapter:
            lora_manager.set_adapter(loaded, ctx.payload["adapter"])
        blocks, sources = _ground(worker, ctx, loaded, messages)
        if sources:
            ctx.emit(Event.CONTEXT, {"sources": sources})
        system_prompt = compose_system_prompt(ctx.payload.get("system_prompt"), blocks)
        params = GenerationParams.from_dict(ctx.payload.get("params"))
        ctx.status("generate", "Generating…")
        result = loaded.engine.stream_chat(messages, params, stop_event=ctx.stop_event,
                                           on_text=lambda text: ctx.emit(Event.TOKEN_STREAM, {"text": text}),
                                           system_prompt=system_prompt)
    worker.models.touch(loaded.key)
    metrics = result.metrics()
    metrics.update(model_key=loaded.key, engine=loaded.engine_name, device=loaded.device,
                   precision=spec.precision, adapter=loaded.active_adapter)
    ctx.emit(Event.GENERATION_COMPLETE, {"text": result.text, "metrics": metrics, "sources": sources})


def _cmd_vision(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.errors import BadRequest

    image_path = ctx.payload.get("image_path")
    if not image_path or not os.path.isfile(image_path):
        raise BadRequest("Pick an image first.")
    spec = worker.spec(ctx.payload, "image-text-to-text")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    params = GenerationParams.from_dict(ctx.payload.get("params"))
    with worker.models.pinned(loaded.key):
        result = loaded.engine.stream(image_path, str(ctx.payload.get("prompt") or ""), params,
                                      stop_event=ctx.stop_event,
                                      on_text=lambda text: ctx.emit(Event.TOKEN_STREAM, {"text": text}))
    metrics = result.metrics()
    metrics.update(model_key=loaded.key, engine=loaded.engine_name, device=loaded.device)
    ctx.emit(Event.GENERATION_COMPLETE, {"text": result.text, "metrics": metrics})


def _cmd_vision_pipeline(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.errors import BadRequest

    image_path = ctx.payload.get("image_path")
    if not image_path or not os.path.isfile(image_path):
        raise BadRequest("Pick an image first.")
    default = "object-detection" if ctx.job.cmd == Command.DETECT_OBJECTS else "image-classification"
    spec = worker.spec({**ctx.payload, "task": ctx.payload.get("task") or default}, default)
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    result = loaded.engine.run(image_path, threshold=float(ctx.payload.get("threshold", 0.5)),
                               top_k=int(ctx.payload.get("top_k", 5)))
    ctx.emit(Event.RESULT, result)


def _cmd_transcribe(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.errors import BadRequest

    audio_path = ctx.payload.get("audio_path")
    if not audio_path or not os.path.isfile(audio_path):
        raise BadRequest("Pick or record an audio file first.")
    spec = worker.spec(ctx.payload, "automatic-speech-recognition")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    ctx.status("transcribe", "Transcribing…")
    result = loaded.engine.transcribe(audio_path, language=ctx.payload.get("language") or None,
                                      translate=bool(ctx.payload.get("translate")),
                                      timestamps=bool(ctx.payload.get("timestamps", True)))
    ctx.emit(Event.RESULT, result)


def _cmd_tts(worker: Worker, ctx: JobContext) -> None:
    spec = worker.spec(ctx.payload, "text-to-speech")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    ctx.status("synthesize", "Synthesizing…")
    result = loaded.engine.synthesize(str(ctx.payload.get("text") or ""), voice=int(ctx.payload.get("voice", 0)),
                                      speed=float(ctx.payload.get("speed", 1.0)),
                                      pitch=float(ctx.payload.get("pitch", 0.0)))
    ctx.emit(Event.RESULT, result)


def _cmd_image(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.pipelines import encode_png_b64

    spec = worker.spec(ctx.payload, "text-to-image")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    p = ctx.payload
    preview_every = int(p.get("preview_every", 1))

    def on_step(step: int, total: int, preview: bytes | None) -> None:
        ctx.progress(step, total, preview_png=encode_png_b64(preview))

    with worker.models.pinned(loaded.key):
        result = loaded.engine.generate(
            prompt=str(p.get("prompt") or ""), negative_prompt=p.get("negative_prompt") or None,
            steps=int(p.get("steps", 25)), guidance=float(p.get("guidance", 7.0)),
            seed=p.get("seed"), width=int(p.get("width", 512)), height=int(p.get("height", 512)),
            init_image=p.get("init_image") or None, strength=float(p.get("strength", 0.6)),
            stop_event=ctx.stop_event, on_step=on_step, preview_every=preview_every)
    ctx.emit(Event.RESULT, result)


def _cmd_embed(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.errors import BadRequest

    inputs = ctx.payload.get("inputs")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not inputs or not all(isinstance(t, str) for t in inputs):
        raise BadRequest("inputs must be a string or a list of strings")
    spec = worker.spec(ctx.payload, "sentence-similarity")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    vectors = loaded.engine.encode(inputs, normalize=bool(ctx.payload.get("normalize", True)))
    tokens = 0
    try:
        tok = loaded.engine.tokenizer
        tokens = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in inputs)
    except Exception:
        tokens = sum(len(t.split()) for t in inputs)
    ctx.emit(Event.RESULT, {"embeddings": vectors.tolist(), "dimension": int(vectors.shape[1]),
                            "model": spec.model_id, "prompt_tokens": tokens})


def _cmd_rag_ingest(worker: Worker, ctx: JobContext) -> None:
    from noelle.config import get_settings
    from noelle.runtime.errors import BadRequest

    settings = get_settings()
    collection = str(ctx.payload.get("collection") or "").strip()
    files = [f for f in ctx.payload.get("files") or [] if isinstance(f, str)]
    if not collection or not files:
        raise BadRequest("A collection name and at least one file are required.")
    rag = worker.rag()
    info = rag.collection_info(collection)
    model_id = (info or {}).get("embedding_model") or ctx.payload.get("embedding_model") or settings.get(
        "rag.embedding_model")
    embedder = _embedder_for(worker, model_id, ctx)
    with worker.models.pinned(embedder.key):
        result = rag.ingest(collection, files, embedder.engine, embedding_model=model_id,
                            chunk_tokens=int(ctx.payload.get("chunk_tokens") or settings.get("rag.chunk_tokens", 512)),
                            overlap=int(ctx.payload["chunk_overlap"] if ctx.payload.get("chunk_overlap") is not None
                                        else settings.get("rag.chunk_overlap", 64)),
                            progress=lambda data: ctx.emit(Event.PROGRESS, data), stop_event=ctx.stop_event)
    ctx.emit(Event.RESULT, result)


def _cmd_rag_query(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.errors import BadRequest

    collection = str(ctx.payload.get("collection") or "")
    rag = worker.rag()
    info = rag.collection_info(collection)
    if info is None:
        raise BadRequest(f"Collection '{collection}' does not exist.")
    embedder = _embedder_for(worker, info["embedding_model"], ctx)
    t0 = time.perf_counter()
    hits = rag.query(collection, str(ctx.payload.get("query") or ""), embedder.engine,
                     top_k=int(ctx.payload.get("top_k", 4)))
    ctx.emit(Event.RESULT, {"hits": hits, "latency_ms": (time.perf_counter() - t0) * 1000.0})


def _cmd_rag_delete(worker: Worker, ctx: JobContext) -> None:
    ctx.emit(Event.RESULT, {"deleted": worker.rag().delete_collection(str(ctx.payload.get("collection") or ""))})


def _cmd_rag_delete_document(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.errors import BadRequest

    collection = str(ctx.payload.get("collection") or "").strip()
    paths_ = [p for p in ctx.payload.get("paths") or [] if isinstance(p, str)]
    if not collection or not paths_:
        raise BadRequest("A collection name and at least one document path are required.")
    rag = worker.rag()
    removed = sum(1 for path in paths_ if rag.delete_document(collection, path))
    ctx.emit(Event.RESULT, {"collection": collection, "removed": removed})


def _cmd_web_search(worker: Worker, ctx: JobContext) -> None:
    engine = _web_engine(ctx.payload)
    query = str(ctx.payload.get("query") or "")
    ctx.emit(Event.RESULT, {"results": engine.search(query)})


def _cmd_quantize(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime import quantization_engine

    spec = worker.spec(ctx.payload, "text-generation")
    source = worker.models.resolve_source(spec)
    worker.models.unload_matching(spec.model_id)  # free memory for the conversion
    result = quantization_engine.quantize(
        source, str(ctx.payload.get("method") or "mlx-4bit"), model_id=spec.model_id,
        output_dir=ctx.payload.get("output_dir"), device=worker.device or "cpu",
        progress=lambda data: ctx.emit(Event.PROGRESS, data), stop_event=ctx.stop_event)
    ctx.emit(Event.RESULT, result)


def _cmd_lora_load(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime import lora_manager

    spec = worker.spec(ctx.payload, "text-generation")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    result = lora_manager.load_adapter(loaded, str(ctx.payload.get("adapter_id") or ""),
                                       name=ctx.payload.get("adapter_name"),
                                       local_path=ctx.payload.get("adapter_local_path"),
                                       activate=bool(ctx.payload.get("activate", True)))
    worker._models_changed()
    ctx.emit(Event.RESULT, result)


def _cmd_lora_set(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime import lora_manager

    spec = worker.spec(ctx.payload, "text-generation")
    loaded = worker.models.ensure(spec, on_status=lambda msg: ctx.status("load", msg))
    result = lora_manager.set_adapter(loaded, ctx.payload.get("adapter_name"))
    worker._models_changed()
    ctx.emit(Event.RESULT, result)


def _cmd_lora_list(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime import lora_manager

    spec = worker.spec(ctx.payload, "text-generation")
    loaded = worker.models.get(spec.key(worker.models.resolve_device(spec)))
    ctx.emit(Event.RESULT, lora_manager.list_adapters(loaded) if loaded else {"adapters": [], "active_adapter": None})


def _cmd_benchmark(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.sequential_queue import run_benchmark

    summary = run_benchmark(worker.models, ctx.payload, stop_event=ctx.stop_event, emit=ctx.emit)
    ctx.emit(Event.GENERATION_COMPLETE, summary)


def _cmd_eval(worker: Worker, ctx: JobContext) -> None:
    from noelle.runtime.evaluation_runner import run_evaluation

    summary = run_evaluation(worker.models, ctx.payload, stop_event=ctx.stop_event, emit=ctx.emit)
    ctx.emit(Event.GENERATION_COMPLETE, summary)


_HANDLERS: dict[Command, Callable[[Worker, JobContext], None]] = {
    Command.LOAD_MODEL: _cmd_load,
    Command.UNLOAD_MODEL: _cmd_unload,
    Command.GENERATE_STREAM: _cmd_generate,
    Command.VISION_GENERATE: _cmd_vision,
    Command.DETECT_OBJECTS: _cmd_vision_pipeline,
    Command.CLASSIFY_IMAGE: _cmd_vision_pipeline,
    Command.TRANSCRIBE: _cmd_transcribe,
    Command.SYNTHESIZE_SPEECH: _cmd_tts,
    Command.GENERATE_IMAGE: _cmd_image,
    Command.EMBED: _cmd_embed,
    Command.RAG_INGEST: _cmd_rag_ingest,
    Command.RAG_QUERY: _cmd_rag_query,
    Command.RAG_DELETE_COLLECTION: _cmd_rag_delete,
    Command.RAG_DELETE_DOCUMENT: _cmd_rag_delete_document,
    Command.WEB_SEARCH: _cmd_web_search,
    Command.QUANTIZE: _cmd_quantize,
    Command.LORA_LOAD: _cmd_lora_load,
    Command.LORA_SET_ADAPTER: _cmd_lora_set,
    Command.LORA_LIST: _cmd_lora_list,
    Command.RUN_BENCHMARK: _cmd_benchmark,
    Command.RUN_EVAL: _cmd_eval,
}


# ============================================================================
# entry point
# ============================================================================


def configure_logging(name: str) -> None:
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(paths.logs_dir() / f"{name}.log", maxBytes=2_000_000,
                                                   backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(stream)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument("--headroom", type=float, default=0.15)
    parser.add_argument("--no-keep-loaded", action="store_true")


def main(args: argparse.Namespace) -> int:
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("The worker only binds to the loopback interface.", file=sys.stderr)
        return 2
    paths.ensure_dirs()
    configure_logging("worker")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    worker = Worker(host=args.host, port=args.port, auth_token=os.environ.get(AUTH_ENV) or None, device=args.device,
                    memory_budget_gb=args.memory_budget_gb, headroom=args.headroom,
                    keep_loaded=not args.no_keep_loaded, parent_pid=args.parent_pid)

    def _terminate(signum: int, _frame: Any) -> None:
        worker.running = False
        worker.stop(None)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    return worker.serve()
