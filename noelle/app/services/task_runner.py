"""QThreadPool helpers: run blocking I/O off the GUI thread, deliver results via signals."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(object)
    finished = Signal()


class Task(QRunnable):
    """Wraps ``fn(*args, **kwargs)``. If ``fn`` accepts ``progress``, it receives an emitter."""

    def __init__(self, fn: Callable[..., Any], *args: Any, with_progress: bool = False, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.with_progress = with_progress
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            if self.with_progress:
                self.kwargs["progress"] = self.signals.progress.emit
            value = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            message = f"{exc}" if str(exc) else type(exc).__name__
            try:
                self.signals.error.emit(message)
            except RuntimeError:
                pass  # receiver already deleted
            _log_debug(traceback.format_exc())
        else:
            try:
                self.signals.result.emit(value)
            except RuntimeError:
                pass
        finally:
            try:
                self.signals.finished.emit()
            except RuntimeError:
                pass


def _log_debug(text: str) -> None:
    import logging

    logging.getLogger("noelle.tasks").debug(text)


class TaskRunner(QObject):
    """Submits tasks and keeps them (and their signal objects) alive until ``finished``
    has been delivered on the GUI thread; otherwise queued signals to plain-function
    slots are dropped when the sender is garbage-collected."""

    def __init__(self, pool: QThreadPool | None = None) -> None:
        super().__init__()
        self.pool = pool or QThreadPool.globalInstance()
        self._live: set[Task] = set()

    def submit(self, fn: Callable[..., Any], *args: Any, on_result: Callable[[Any], None] | None = None,
               on_error: Callable[[str], None] | None = None, on_progress: Callable[[Any], None] | None = None,
               on_finished: Callable[[], None] | None = None, **kwargs: Any) -> Task:
        task = Task(fn, *args, with_progress=on_progress is not None, **kwargs)
        task.setAutoDelete(False)
        if on_result:
            task.signals.result.connect(on_result)
        if on_error:
            task.signals.error.connect(on_error)
        if on_progress:
            task.signals.progress.connect(on_progress)
        if on_finished:
            task.signals.finished.connect(on_finished)
        task.signals.finished.connect(lambda t=task: self._live.discard(t))
        self._live.add(task)
        self.pool.start(task)
        return task

    @property
    def active_count(self) -> int:
        return len(self._live)


_runner: TaskRunner | None = None


def runner() -> TaskRunner:
    global _runner
    if _runner is None:
        _runner = TaskRunner()
    return _runner
