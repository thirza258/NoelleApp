"""Qt-free worker clients and process helpers.

Used by the API server, the download child, tests and scripts. The desktop
shell has its own Qt client in :mod:`noelle.app.ipc.runtime_client` but reuses
:func:`worker_command` and :func:`build_worker_env` from here.

Safe to import from the UI: no torch, no Qt.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, AsyncIterator, Iterator

import zmq

from noelle import paths
from noelle.runtime.protocol import (
    AUTH_ENV,
    READY_PREFIX,
    TERMINAL_EVENTS,
    Command,
    Event,
    decode,
    make_request,
)

PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])


class WorkerError(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("message", "worker error"))
        self.payload = payload
        self.code = payload.get("code")
        self.hint = payload.get("hint")


def worker_command(python: str | None = None) -> list[str]:
    """Command prefix that runs ``noelle <subcommand>`` in a child process."""
    if getattr(sys, "frozen", False):  # PyInstaller bundle: the executable is the dispatcher
        return [sys.executable]
    return [python or sys.executable, "-m", "noelle"]


def build_worker_env(token: str | None = None, *, hf_home: str | None = None, offline: bool = True,
                     extra: dict[str, str] | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    if token:
        env[AUTH_ENV] = token
    if hf_home:
        env["HF_HOME"] = str(hf_home)
        env.pop("HF_HUB_CACHE", None)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"  # the inference plane never talks to the Hub
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HF_TOKEN", None)  # credentials never reach inference workers
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    if not getattr(sys, "frozen", False):
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = PACKAGE_PARENT + (os.pathsep + existing if existing else "")
    if extra:
        env.update(extra)
    return env


@dataclass
class WorkerProcess:
    process: subprocess.Popen[str]
    port: int
    token: str

    @property
    def endpoint(self) -> str:
        return f"tcp://127.0.0.1:{self.port}"

    def client(self) -> WorkerClient:
        return WorkerClient(self.endpoint, self.token)

    def stop(self, timeout: float = 10.0) -> int | None:
        if self.process.poll() is None:
            try:
                self.client().call(Command.SHUTDOWN, timeout=3.0)
            except Exception:
                pass
            try:
                return self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return self.process.wait(timeout=5)
        return self.process.returncode


def spawn_worker(*, python: str | None = None, port: int = 0, device: str | None = None,
                 memory_budget_gb: float | None = None, parent_pid: int | None = None, hf_home: str | None = None,
                 offline: bool = True, extra_env: dict[str, str] | None = None, log_path: Path | None = None,
                 ready_timeout: float = 120.0, keep_loaded: bool = True) -> WorkerProcess:
    """Start a worker subprocess and block until it reports its port."""
    token = secrets.token_hex(16)
    args = worker_command(python) + ["worker", "--port", str(port)]
    if device:
        args += ["--device", device]
    if memory_budget_gb:
        args += ["--memory-budget-gb", str(memory_budget_gb)]
    if parent_pid:
        args += ["--parent-pid", str(parent_pid)]
    if not keep_loaded:
        args.append("--no-keep-loaded")
    env = build_worker_env(token, hf_home=hf_home, offline=offline, extra=extra_env)
    log_path = log_path or (paths.logs_dir() / "worker-stderr.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr: IO[str] = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - owned by the child
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr, env=env, text=True, bufsize=1)
    stderr.close()
    deadline = time.monotonic() + ready_timeout
    assert proc.stdout is not None
    while True:
        if time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError("worker did not report ready in time")
        line = proc.stdout.readline()
        if not line:
            code = proc.wait(timeout=5)
            raise RuntimeError(f"worker exited during startup (code {code}); see {log_path}")
        if line.startswith(READY_PREFIX):
            fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
            port = int(fields["port"])
            break
    threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
    return WorkerProcess(proc, port, token)


def _drain(stream: IO[str]) -> None:
    try:
        for _ in stream:
            pass
    except (OSError, ValueError):
        pass


class WorkerClient:
    """Blocking client. Each call opens its own DEALER socket, so it is thread-safe."""

    def __init__(self, endpoint: str, token: str | None = None, *, context: zmq.Context | None = None) -> None:
        self.endpoint = endpoint
        self.token = token
        self.ctx = context or zmq.Context.instance()

    def stream(self, cmd: Command | str, payload: dict[str, Any] | None = None, *,
               timeout: float | None = 600.0, request_id: str | None = None) -> Iterator[dict[str, Any]]:
        sock = self.ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.endpoint)
        rid, frame = make_request(cmd, payload, auth=self.token, request_id=request_id)
        try:
            sock.send(frame)
            while True:
                wait_ms = None if timeout is None else int(timeout * 1000)
                if not sock.poll(wait_ms):
                    raise TimeoutError(f"no reply to {cmd} within {timeout}s")
                message = decode(sock.recv())
                if message.get("id") not in (rid, None):
                    continue
                if message.get("id") is None:
                    if message.get("event") == Event.ERROR:  # protocol-level rejection
                        yield message
                        return
                    continue
                yield message
                if message.get("event") in TERMINAL_EVENTS:
                    return
        finally:
            sock.close(linger=0)

    def call(self, cmd: Command | str, payload: dict[str, Any] | None = None, *,
             timeout: float | None = 600.0) -> dict[str, Any]:
        last: dict[str, Any] | None = None
        for message in self.stream(cmd, payload, timeout=timeout):
            last = message
        if last is None:
            raise WorkerError({"code": "INTERNAL", "message": "no reply"})
        if last.get("event") == Event.ERROR:
            raise WorkerError(last.get("payload") or {})
        return last.get("payload") or {}

    def stop(self, target_id: str | None = None) -> dict[str, Any]:
        return self.call(Command.STOP_GENERATION, {"target_id": target_id}, timeout=10)


class AsyncWorkerClient:
    """asyncio client for the FastAPI server (one socket per request)."""

    def __init__(self, endpoint: str, token: str | None = None) -> None:
        import zmq.asyncio

        self.endpoint = endpoint
        self.token = token
        self.ctx = zmq.asyncio.Context.instance()

    async def stream(self, cmd: Command | str, payload: dict[str, Any] | None = None, *,
                     timeout: float | None = 600.0, request_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        sock = self.ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.endpoint)
        rid, frame = make_request(cmd, payload, auth=self.token, request_id=request_id)
        try:
            await sock.send(frame)
            while True:
                wait_ms = None if timeout is None else int(timeout * 1000)
                if not await sock.poll(wait_ms):
                    raise TimeoutError(f"no reply to {cmd} within {timeout}s")
                message = decode(await sock.recv())
                if message.get("id") not in (rid, None):
                    continue
                if message.get("id") is None:
                    if message.get("event") == Event.ERROR:
                        yield message
                        return
                    continue
                yield message
                if message.get("event") in TERMINAL_EVENTS:
                    return
        finally:
            sock.close(linger=0)

    async def call(self, cmd: Command | str, payload: dict[str, Any] | None = None, *,
                   timeout: float | None = 600.0) -> dict[str, Any]:
        last: dict[str, Any] | None = None
        async for message in self.stream(cmd, payload, timeout=timeout):
            last = message
        if last is None:
            raise WorkerError({"code": "INTERNAL", "message": "no reply"})
        if last.get("event") == Event.ERROR:
            raise WorkerError(last.get("payload") or {})
        return last.get("payload") or {}


def read_discovery_file() -> dict[str, Any] | None:
    import json

    try:
        return json.loads((paths.runtime_dir() / "worker.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
