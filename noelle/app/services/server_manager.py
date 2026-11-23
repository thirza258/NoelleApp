"""Starts/stops the OpenAI-compatible API server child process (``noelle server``).

The server talks to the same worker as the desktop UI; when the worker restarts
(new port and token) a running server is restarted to follow it.
"""

from __future__ import annotations

import os
from collections import deque

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from noelle.config import Settings
from noelle.runtime.client import build_worker_env, worker_command

READY_PREFIX = "NOELLE_SERVER_READY"


class ServerManager(QObject):
    state_changed = Signal(str)  # stopped | starting | running | error
    log_line = Signal(str)

    def __init__(self, settings: Settings, runtime: object, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.runtime = runtime
        self._process: QProcess | None = None
        self._state = "stopped"
        self._buffer = ""
        self._endpoint: str | None = None
        self.port: int | None = None
        self.log: deque[str] = deque(maxlen=500)
        self.last_error: str | None = None
        runtime.state_changed.connect(self._on_runtime_state)  # type: ignore[attr-defined]

    @property
    def state(self) -> str:
        return self._state

    @property
    def base_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}/v1" if self.port and self._state == "running" else None

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def start(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            return
        endpoint = getattr(self.runtime, "endpoint", None)
        token = getattr(self.runtime, "token", "")
        if not endpoint:
            self.last_error = "The model runtime is not running yet."
            self._set_state("error")
            return
        port = int(self.settings.get("server.port", 8765))
        command = worker_command(None)
        args = command[1:] + ["server", "--host", "127.0.0.1", "--port", str(port), "--worker-endpoint", endpoint,
                              "--parent-pid", str(os.getpid())]
        env = build_worker_env(token, offline=True)
        api_key = self.settings.get("server.api_key")
        qenv = QProcessEnvironment()
        for key, value in env.items():
            qenv.insert(key, value)
        if api_key:
            qenv.insert("NOELLE_SERVER_API_KEY", str(api_key))
        process = QProcess(self)
        process.setProcessEnvironment(qenv)
        process.setProgram(command[0])
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        self._process = process
        self._endpoint = endpoint
        self.port = port
        self.last_error = None
        self._set_state("starting")
        process.start()

    def stop(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(4000):
                self._process.kill()
                self._process.waitForFinished(2000)
        self._process = None
        self._set_state("stopped")

    def _on_output(self) -> None:
        assert self._process is not None
        self._buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(READY_PREFIX):
                fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
                if fields.get("port"):
                    self.port = int(fields["port"])
                self._set_state("running")
            self.log.append(line)
            self.log_line.emit(line)

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._state != "stopped":
            tail = list(self.log)[-5:]
            if any("address already in use" in l.lower() or "errno 48" in l.lower() for l in tail):
                self.last_error = f"Port {self.port} is already in use. Choose another port."
            else:
                self.last_error = f"The API server exited (code {exit_code})."
            self._set_state("error")
        self._process = None

    def _on_runtime_state(self, state: str) -> None:
        if state == "ready" and self._state == "running" and self._endpoint != getattr(self.runtime, "endpoint", None):
            self.stop()
            self.start()
        elif state == "ready" and self._state == "stopped" and self.settings.get("server.autostart"):
            self.start()
