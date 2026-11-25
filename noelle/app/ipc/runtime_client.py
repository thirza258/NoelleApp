"""Supervises the worker process and multiplexes requests over one DEALER socket.

States::

    stopped -> starting -> warming (process up, torch importing) -> ready
                     \\-> crashed (process died)      ready -> unresponsive (no heartbeat)

When the worker dies (CUDA/Metal OOM kill, segfault, ``kill -9``), every
pending request fails with ``WORKER_CRASHED``, :attr:`worker_crashed` fires with
a diagnosis, and the UI offers a restart. The desktop shell keeps running.
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import time
from collections import deque
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from noelle.config import Settings
from noelle.runtime.client import build_worker_env, worker_command
from noelle.runtime.protocol import (
    READY_PREFIX,
    TERMINAL_EVENTS,
    Command,
    ErrorCode,
    Event,
    error_payload,
    make_request,
    parse_ready_port,
)

from noelle.app.ipc.stream_listener import StreamListener

log = logging.getLogger("noelle.runtime-client")


class RequestHandle(QObject):
    """Per-request signals. Emitted on the GUI thread."""

    event = Signal(str, dict)
    token = Signal(str)
    status = Signal(dict)
    progress = Signal(dict)
    context = Signal(dict)
    finished = Signal(dict)
    failed = Signal(dict)

    def __init__(self, request_id: str, cmd: Command, client: RuntimeClient) -> None:
        super().__init__(client)
        self.request_id = request_id
        self.cmd = cmd
        self._client = client
        self.done = False
        self.started_at = time.monotonic()

    def cancel(self) -> None:
        if not self.done:
            self._client.stop_request(self.request_id)

    def _dispatch(self, event: str, payload: dict[str, Any]) -> None:
        self.event.emit(event, payload)
        if event == Event.TOKEN_STREAM:
            self.token.emit(payload.get("text", ""))
        elif event == Event.STATUS:
            self.status.emit(payload)
        elif event == Event.PROGRESS:
            self.progress.emit(payload)
        elif event == Event.CONTEXT:
            self.context.emit(payload)
        if event in TERMINAL_EVENTS:
            self.done = True
            if event == Event.ERROR:
                self.failed.emit(payload)
            else:
                self.finished.emit(payload)
            self.deleteLater()


class RuntimeClient(QObject):
    state_changed = Signal(str)
    heartbeat = Signal(dict)
    models_changed = Signal(list)
    worker_crashed = Signal(int, str)
    log_line = Signal(str)

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._state = "stopped"
        self._process: QProcess | None = None
        self._listener: StreamListener | None = None
        self._token = ""
        self._port: int | None = None
        self._pending: dict[str, RequestHandle] = {}
        self._outbox: list[bytes] = []
        self._stdout_buffer = ""
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stopping = False
        self._last_heartbeat = 0.0
        self._exit_handled = False
        self._init_error_reported = False
        self.last_heartbeat_payload: dict[str, Any] = {}
        self.loaded_models: list[dict[str, Any]] = []
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(2000)
        self._watchdog.timeout.connect(self._check_heartbeat)

    # ----------------------------------------------------------- properties
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == "ready"

    @property
    def endpoint(self) -> str | None:
        return f"tcp://127.0.0.1:{self._port}" if self._port else None

    @property
    def token(self) -> str:
        return self._token

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        pid = int(self._process.processId())
        return pid or None

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self._stopping = False
        self._exit_handled = False
        self._init_error_reported = False
        self._token = secrets.token_hex(16)
        self._port = None
        self._stdout_buffer = ""
        python = self.settings.get("runtime.python") or None
        env_name = self.settings.get("runtime.environment")
        if env_name and not python:
            try:
                from noelle.runtime import environment_manager

                python = str(environment_manager.python_for(env_name))
            except Exception as exc:  # noqa: BLE001 - fall back, but make the misconfiguration visible
                log.warning("Could not resolve runtime environment %r; using the application interpreter: %s",
                            env_name, exc)
                self._stderr_tail.append(f"Runtime environment '{env_name}' could not be resolved: {exc}")
                python = None
        command = worker_command(python)
        args = command[1:] + ["worker", "--port", "0", "--parent-pid", str(os.getpid()),
                              "--device", str(self.settings.get("runtime.device", "auto")),
                              "--headroom", str(self.settings.get("runtime.safety_headroom", 0.15))]
        budget = self.settings.get("runtime.memory_budget_gb")
        if budget:
            args += ["--memory-budget-gb", str(budget)]
        if not self.settings.get("runtime.keep_models_loaded", True):
            args.append("--no-keep-loaded")
        env = build_worker_env(self._token, hf_home=self.settings.get("storage.hf_home"),
                               offline=bool(self.settings.get("runtime.offline", True)))
        qenv = QProcessEnvironment()
        for key, value in env.items():
            qenv.insert(key, value)
        process = QProcess(self)
        process.setProcessEnvironment(qenv)
        process.setProgram(command[0])
        process.setArguments(args)
        # Bind the process into every slot: a queued signal from an older worker must never be
        # mistaken for the current one after a restart.
        process.readyReadStandardOutput.connect(lambda p=process: self._on_stdout(p))
        process.readyReadStandardError.connect(lambda p=process: self._on_stderr(p))
        process.finished.connect(lambda code, status, p=process: self._on_finished(p, code, status))
        process.errorOccurred.connect(lambda error, p=process: self._on_process_error(p, error))
        previous, self._process = self._process, process
        if previous is not None:
            previous.deleteLater()  # a dead worker replaced by this one; its late signals are ignored by identity
        self._set_state("starting")
        process.start()

    def stop(self, timeout_ms: int = 5000) -> None:
        self._stopping = True
        self._watchdog.stop()
        process, self._process = self._process, None  # ignore any signal this process emits while stopping
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            if self._listener is not None and self._port:
                _, frame = make_request(Command.SHUTDOWN, {}, auth=self._token)
                self._listener.send(frame)
            if not process.waitForFinished(timeout_ms):
                process.kill()
                process.waitForFinished(2000)
        if process is not None:
            process.deleteLater()
        self._teardown_listener()
        self._fail_pending(error_payload(ErrorCode.CANCELLED, "The runtime was stopped."))
        self._set_state("stopped")

    def restart(self) -> None:
        self.stop()
        self.start()

    def kill_for_test(self) -> None:
        """Simulate a hard crash (used by the OOM-resilience test)."""
        if self.pid:
            os.kill(self.pid, signal.SIGKILL)

    # -------------------------------------------------------------- requests
    def request(self, cmd: Command, payload: dict[str, Any] | None = None) -> RequestHandle:
        request_id, frame = make_request(cmd, payload or {}, auth=self._token)
        handle = RequestHandle(request_id, cmd, self)
        if self._state == "stopped" or self._process is None:
            self.start()
        if self._state in ("crashed", "unresponsive"):
            message = ("The model runtime crashed. Restart it from the status bar."
                       if self._state == "crashed" else
                       "The model runtime is not responding. Restart it from the status bar.")
            self._refuse_later(handle, message)
            return handle
        self._pending[request_id] = handle
        if self._listener is None:
            self._outbox.append(frame)
        else:
            self._listener.send(frame)
        return handle

    def stop_request(self, request_id: str | None = None) -> None:
        _, frame = make_request(Command.STOP_GENERATION, {"target_id": request_id}, auth=self._token)
        if self._listener is not None:
            self._listener.send(frame)

    # ------------------------------------------------------------- internals
    def _on_stdout(self, process: QProcess) -> None:
        if process is not self._process:
            return
        self._stdout_buffer += bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if line.startswith(READY_PREFIX) and self._port is None:
                port = parse_ready_port(line)
                if port is None:
                    # A malformed handshake would otherwise leave the app in "starting" forever.
                    self._stderr_tail.append(f"The runtime sent an invalid ready line: {line[:200]}")
                    self.log_line.emit(self._stderr_tail[-1])
                    process.kill()
                    self._handle_exit(-1, crashed=True)
                    continue
                self._port = port
                self._connect()
            elif line:
                self.log_line.emit(line)

    def _on_stderr(self, process: QProcess) -> None:
        if process is not self._process:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self._stderr_tail.append(line)
                self.log_line.emit(line)

    def _connect(self) -> None:
        listener = StreamListener(self.endpoint or "", self)
        listener.message_received.connect(self._on_message)
        listener.connection_error.connect(self._on_ipc_error)
        self._listener = listener
        listener.start()
        _, hello = make_request(Command.HELLO, {}, auth=self._token)
        listener.send(hello)
        for frame in self._outbox:
            listener.send(frame)
        self._outbox.clear()
        self._last_heartbeat = time.monotonic()
        self._watchdog.start()
        self._set_state("warming")

    def _on_message(self, message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            return
        request_id = message.get("id")
        event = message.get("event", "")
        payload = message.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if request_id is None:
            if event == Event.HEARTBEAT:
                self._last_heartbeat = time.monotonic()
                self.last_heartbeat_payload = payload
                loaded = payload.get("loaded")
                self.loaded_models = loaded if isinstance(loaded, list) else []
                init_error = payload.get("init_error")
                if isinstance(init_error, str) and init_error and not self._init_error_reported:
                    self._init_error_reported = True
                    diagnosis = f"The runtime could not initialise: {init_error}"
                    self._fail_pending(error_payload(ErrorCode.WORKER_CRASHED, diagnosis,
                                                     hint="Check the runtime Python and installed extras in Settings."))
                    self._set_state("crashed")
                    self.worker_crashed.emit(-1, diagnosis)
                    return
                self._set_state("ready" if payload.get("runtime_ready") else "warming")
                self.heartbeat.emit(payload)
            elif event == Event.MODELS_CHANGED:
                loaded = payload.get("loaded")
                self.loaded_models = loaded if isinstance(loaded, list) else []
                self.models_changed.emit(self.loaded_models)
            return
        if not isinstance(request_id, str):
            return
        handle = self._pending.get(request_id)
        if handle is None:
            return
        if event in TERMINAL_EVENTS:
            self._pending.pop(request_id, None)
        handle._dispatch(event, payload)

    def _check_heartbeat(self) -> None:
        if self._state not in ("ready", "warming"):
            return
        timeout = float(self.settings.get("runtime.heartbeat_timeout_s", 20))
        if time.monotonic() - self._last_heartbeat > timeout:
            self._set_state("unresponsive")
            # Do not leave the UI stuck on "Generating…" behind a wedged worker.
            self._fail_pending(error_payload(
                ErrorCode.WORKER_CRASHED,
                "The model runtime stopped responding (no heartbeat).",
                hint="Restart the runtime from the status bar. Your work in the app is safe."))

    def _on_process_error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if process is not self._process:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._stderr_tail.append("The worker process failed to start (is the runtime Python installed?).")
            self._handle_exit(-1, crashed=True)

    def _on_finished(self, process: QProcess, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if process is not self._process:
            return
        abrupt = exit_status == QProcess.ExitStatus.CrashExit
        code = CRASH_EXIT if abrupt else exit_code
        self._handle_exit(code, crashed=(abrupt or exit_code != 0) and not self._stopping)

    def _on_ipc_error(self, message: str) -> None:
        """The listener died: pending requests must fail instead of waiting forever."""
        self.log_line.emit(f"IPC error: {message}")
        if self._stopping or self._state in ("stopped", "crashed"):
            return
        self._exit_handled = True
        self._watchdog.stop()
        self._teardown_listener()
        self._port = None
        diagnosis = ("Noelle lost its connection to the model runtime. The runtime may still be running; "
                     "restart it from the status bar.")
        self._fail_pending(error_payload(ErrorCode.WORKER_CRASHED, diagnosis,
                                         hint="Restart the runtime. If this repeats, check the runtime log."))
        self._set_state("crashed")
        self.worker_crashed.emit(CRASH_EXIT, diagnosis)

    def _handle_exit(self, exit_code: int, *, crashed: bool) -> None:
        if self._exit_handled:
            return
        self._exit_handled = True
        self._watchdog.stop()
        self._teardown_listener()
        self._port = None
        if self._stopping or not crashed:
            self._fail_pending(error_payload(ErrorCode.CANCELLED, "The runtime stopped."))
            self._set_state("stopped")
            return
        diagnosis = diagnose_exit(exit_code, self.stderr_tail())
        self._fail_pending(error_payload(ErrorCode.WORKER_CRASHED, diagnosis,
                                         hint="Restart the runtime. Try a smaller or 4-bit model if it repeats."))
        self._set_state("crashed")
        self.worker_crashed.emit(exit_code, diagnosis)

    def _teardown_listener(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener.deleteLater()
            self._listener = None

    def _refuse_later(self, handle: RequestHandle, message: str) -> None:
        """Deliver an immediate refusal on the next event-loop turn, so the caller can connect.

        The timer is owned by the handle: if the handle or its client is destroyed first,
        Qt cancels the callback instead of calling into a deleted object.
        """
        payload = error_payload(ErrorCode.WORKER_CRASHED, message)
        timer = QTimer(handle)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: handle._dispatch(Event.ERROR, payload))
        timer.timeout.connect(timer.deleteLater)
        timer.start(0)

    def _fail_pending(self, payload: dict[str, Any]) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        self._outbox.clear()
        for handle in pending:
            handle._dispatch(Event.ERROR, payload)


CRASH_EXIT = -1000  # QProcess reports CrashExit without the signal number


def diagnose_exit(exit_code: int, stderr_tail: str) -> str:
    tail = stderr_tail.lower()
    if "out of memory" in tail or "mps backend out of memory" in tail:
        return "The worker ran out of accelerator memory and exited."
    if "segmentation fault" in tail or "fatal python error" in tail:
        return "The worker crashed in native code."
    if exit_code == CRASH_EXIT:
        return ("The worker was terminated abruptly — usually the operating system killing it under memory "
                "pressure, or a crash in a native library.")
    if exit_code in (-9, 137, 9) or "killed" in tail:
        return "The worker was killed by the operating system, most likely for using too much memory."
    if exit_code in (-11, 139):
        return "The worker crashed in native code (segmentation fault)."
    if exit_code == -1:
        return "The worker process could not be started."
    return f"The worker exited unexpectedly (exit code {exit_code})."
