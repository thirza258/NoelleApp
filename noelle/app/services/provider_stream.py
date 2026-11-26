"""Run a blocking provider stream off the GUI thread and deliver tokens as Qt signals."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

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
        task.signals.token.connect(self.token)
        task.signals.finished.connect(self._on_finished)
        task.signals.failed.connect(self._on_failed)
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

    def _on_finished(self, text: str, metrics: dict) -> None:
        self._task = None
        self.finished.emit(text, metrics)

    def _on_failed(self, payload: dict) -> None:
        self._task = None
        self.failed.emit(payload)