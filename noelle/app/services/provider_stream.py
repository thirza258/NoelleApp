"""Run a blocking provider stream off the GUI thread and deliver tokens as Qt signals."""

from __future__ import annotations

import logging
import threading
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from shiboken6 import isValid as _qt_is_valid

from noelle.providers.provider import ProviderError

log = logging.getLogger("noelle.providers.stream")


class _Signals(QObject):
    token = Signal(str)
    finished = Signal(str, dict)
    failed = Signal(dict)


class _Task(QRunnable):
    def __init__(self, provider: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.provider = provider
        self.messages = messages
        self.kwargs = kwargs
        self.signals = _Signals()
        self.cancelled = False
        self.setAutoDelete(False)

    def run(self) -> None:
        parts: list[str] = []
        try:
            for token in self.provider.stream_chat(self.messages, **self.kwargs):
                if self.cancelled:
                    break
                parts.append(token)
                self.signals.token.emit(token)
            if self.cancelled:
                self.signals.failed.emit({"code": "CANCELLED", "message": "Generation stopped.",
                                          "recoverable": True, "suggested_actions": []})
            else:
                self.signals.finished.emit("".join(parts), dict(getattr(self.provider, "last_metrics", {}) or {}))
        except ProviderError as exc:
            self.signals.failed.emit(exc.to_dict())
        except Exception as exc:  # noqa: BLE001 - a provider bug must not crash the GUI
            log.exception("provider stream failed")
            self.signals.failed.emit({"code": "RUNTIME_ERROR", "message": str(exc), "recoverable": True,
                                      "suggested_actions": ["retry"]})


class ConsentGate(QObject):
    """Ask the GUI for cloud-egress consent from a worker thread without touching widgets there.

    ``decide`` blocks the provider task; the signal is delivered to the GUI thread (queued),
    where a dialog calls :meth:`answer`.
    """

    requested = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending: tuple[dict[str, Any], threading.Event] | None = None

    def decide(self, reason: str, destination: str, timeout: float = 180.0) -> bool:
        try:
            if not _qt_is_valid(self):
                return False
        except RuntimeError:
            return False
        event = threading.Event()
        box: dict[str, Any] = {"allowed": False}
        with self._lock:
            self._pending = (box, event)
        self.requested.emit(reason, destination)
        event.wait(timeout)
        with self._lock:
            self._pending = None
        return bool(box.get("allowed"))

    def answer(self, allowed: bool) -> None:
        with self._lock:
            pending = self._pending
        if pending is not None:
            box, event = pending
            box["allowed"] = bool(allowed)
            event.set()


class ProviderStream(QObject):
    """Handle for one provider request; token/finished/failed are re-emitted on the GUI thread."""

    token = Signal(str)
    finished = Signal(str, dict)
    failed = Signal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._task: _Task | None = None
        self.provider: Any = None

    def start(self, provider: Any, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        if self._task is not None:
            self.cancel()
        self.provider = provider
        task = _Task(provider, messages, kwargs)
        # Bind the task into each slot: a cancelled task's late signal must not clear or
        # append to a newer request.
        task.signals.token.connect(lambda text, t=task: self._forward_token(t, text))
        task.signals.finished.connect(lambda text, metrics, t=task: self._forward_finished(t, text, metrics))
        task.signals.failed.connect(lambda payload, t=task: self._forward_failed(t, payload))
        self._task = task
        QThreadPool.globalInstance().start(task)

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancelled = True
            provider = self.provider
            if provider is not None and hasattr(provider, "cancel"):
                try:
                    provider.cancel()
                except Exception:  # noqa: BLE001
                    pass
            self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None

    def _alive(self) -> bool:
        try:
            return bool(_qt_is_valid(self))
        except RuntimeError:
            return False

    def _forward_token(self, task: _Task, text: str) -> None:
        if self._alive() and self._task is task:
            self.token.emit(text)

    def _forward_finished(self, task: _Task, text: str, metrics: dict) -> None:
        if self._alive() and self._task is task:
            self._task = None
            self.finished.emit(text, metrics)

    def _forward_failed(self, task: _Task, payload: dict) -> None:
        if self._alive() and self._task is task:
            self._task = None
            self.failed.emit(payload)