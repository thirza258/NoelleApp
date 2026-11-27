"""UI lifecycle regressions: dead toasts, dead per-model panels, clipped errors, Discover filter."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

LONG = ("LOAD_FAILED: Could not load hexgrad/Kokoro-82M: Unrecognized model in "
        "C:\\Users\\someone\\models--hexgrad--Kokoro-82M\\snapshots\\"
        "0123456789abcdef0123456789abcdef01234567. Should have a 'model_type' key in its config.json.")


@pytest.fixture()
def qt_app(qtbot: Any) -> Any:
    return qtbot


def test_toast_after_dismissal_does_not_touch_a_deleted_wrapper(qtbot: Any) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

    from noelle.app.ui.main_window import MainWindow

    class Host(QMainWindow):
        toast = MainWindow.toast
        _forget_toast = MainWindow._forget_toast

        def __init__(self) -> None:
            super().__init__()
            self.setCentralWidget(QWidget())
            self._toasts: list[Any] = []

    host = Host()
    host.resize(1200, 800)
    qtbot.addWidget(host)
    host.show()
    host.toast("first", "error")
    QApplication.processEvents()
    assert len(host._toasts) == 1
    first = host._toasts[0]
    first.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    # The old code kept the stale wrapper in _toasts and raised RuntimeError here.
    host.toast("second", "success")
    QApplication.processEvents()
    assert len(host._toasts) == 1
    assert host._toasts[0] is not first


@pytest.mark.parametrize("panel_name", ["optimize", "knowledge"])
def test_deleted_per_model_panel_ignores_download_signal(qtbot: Any, panel_name: str) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
    from PySide6.QtWidgets import QApplication

    class FakeDownloads(QObject):
        job_finished = Signal(str, bool)

    class FakeSettings:
        def get(self, _key: str, default: Any = None) -> Any:
            return "sentence-transformers/all-MiniLM-L6-v2" if default is None else default

    ctx = SimpleNamespace(downloads=FakeDownloads(), settings=FakeSettings(), toast=lambda *_a, **_k: None,
                          runtime=SimpleNamespace(request=lambda *_a, **_k: None))
    if panel_name == "optimize":
        from noelle.app.ui.widgets.optimize_panel import OptimizePanel

        panel = OptimizePanel(ctx, SimpleNamespace(summary=None))
    else:
        from noelle.app.ui.widgets.knowledge_panel import KnowledgePanel

        panel = KnowledgePanel(ctx)
    qtbot.addWidget(panel)
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    # Must not raise libshiboken "C++ object already deleted" from a stale connection.
    ctx.downloads.job_finished.emit("org/model", True)


def _assert_error_line_visible(error: Any, panel: Any) -> None:
    assert error.isVisible()
    assert LONG in error.message()
    assert "model_type" in error.message()
    assert error.sizeHint().width() <= 680
    assert error.sizeHint().width() <= max(panel.width(), 680)


def test_vision_error_renders_in_full(qtbot: Any) -> None:
    from noelle.app.ui.widgets.playgrounds.vision_playground import VisionPlayground
    from noelle.runtime import tasks

    ctx = SimpleNamespace(toast=lambda *_a, **_k: None, runtime=SimpleNamespace())
    page = VisionPlayground(ctx, lambda: {}, tasks.VISION)
    qtbot.addWidget(page)
    page.resize(1100, 700)
    page.show()
    page._on_failed({"message": LONG, "hint": "Open the model page or pick a transformers checkpoint."})
    _assert_error_line_visible(page.error, page)
    assert "Open the model page" in page.error.message()


def test_audio_error_renders_in_full(qtbot: Any) -> None:
    from PySide6.QtCore import QObject, Signal

    from noelle.app.ui.widgets.playgrounds.audio_playground import AudioPlayground
    from noelle.runtime import tasks
    from noelle.runtime.protocol import Command

    class Handle(QObject):
        status = Signal(dict)
        finished = Signal(dict)
        failed = Signal(dict)
        token = Signal(str)

        def cancel(self) -> None:
            pass

    class Runtime:
        def request(self, *_args: Any, **_kwargs: Any) -> Handle:
            self.handle = Handle()
            return self.handle

    runtime = Runtime()
    ctx = SimpleNamespace(toast=lambda *_a, **_k: None, runtime=runtime)
    page = AudioPlayground(ctx, lambda: {}, tasks.TTS)
    qtbot.addWidget(page)
    page.resize(1100, 700)
    page.show()
    page._run(Command.SYNTHESIZE_SPEECH, {}, lambda _r: None)
    runtime.handle.failed.emit({"message": LONG, "hint": "Pick a supported TTS model."})
    _assert_error_line_visible(page.error, page)


def test_diffusion_error_renders_in_full(qtbot: Any) -> None:
    from noelle.app.ui.widgets.playgrounds.diffusion_playground import DiffusionPlayground

    ctx = SimpleNamespace(toast=lambda *_a, **_k: None, runtime=SimpleNamespace())
    page = DiffusionPlayground(ctx, lambda: {})
    qtbot.addWidget(page)
    page.resize(1100, 700)
    page.show()
    page._on_failed({"message": LONG, "hint": "This is not a diffusers repository."})
    _assert_error_line_visible(page.error, page)


def test_embedding_error_renders_in_full(qtbot: Any) -> None:
    from noelle.app.ui.widgets.playgrounds.embedding_playground import EmbeddingPlayground

    ctx = SimpleNamespace(toast=lambda *_a, **_k: None, runtime=SimpleNamespace())
    page = EmbeddingPlayground(ctx, lambda: {})
    qtbot.addWidget(page)
    page.resize(1100, 700)
    page.show()
    page._failed({"message": LONG, "hint": "The embedding model is not downloaded."})
    _assert_error_line_visible(page.error, page)


def test_deleted_audio_playground_ignores_runtime_signals(qtbot: Any) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
    from PySide6.QtWidgets import QApplication

    from noelle.app.ui.widgets.playgrounds.audio_playground import AudioPlayground
    from noelle.runtime import tasks
    from noelle.runtime.protocol import Command

    class Handle(QObject):
        status = Signal(dict)
        finished = Signal(dict)
        failed = Signal(dict)
        token = Signal(str)

        def cancel(self) -> None:
            pass

    class Runtime:
        def request(self, *_args: Any, **_kwargs: Any) -> Handle:
            self.handle = Handle()
            return self.handle

    runtime = Runtime()
    page = AudioPlayground(SimpleNamespace(toast=lambda *_a, **_k: None, runtime=runtime), lambda: {}, tasks.TTS)
    qtbot.addWidget(page)
    page._run(Command.SYNTHESIZE_SPEECH, {}, lambda _r: None)
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    runtime.handle.status.emit({"message": "late"})
    runtime.handle.finished.emit({"audio_path": "/tmp/x.wav"})
    runtime.handle.failed.emit({"message": "late error"})


def test_camera_dialog_is_off_until_opened_and_survives_no_camera(qtbot: Any) -> None:
    from noelle.app.ui.widgets.camera_dialog import CameraDialog

    dialog = CameraDialog()
    qtbot.addWidget(dialog)
    assert dialog.image_path is None
    assert dialog._camera is None  # the device has not been opened
    assert dialog.take_btn.isEnabled() is False


def test_error_html_escapes_and_breaks_long_tokens() -> None:
    from noelle.app.ui.widgets.common import error_html

    rendered = error_html("<script>alert(1)</script>", hint="see C:\\Users\\x\\" + "a" * 90)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "\u200b" in rendered


def test_runs_filter_hides_unsupported_cards(qtbot: Any) -> None:
    from PySide6.QtWidgets import QScrollArea

    from noelle.app.services.hf_client import ModelSummary
    from noelle.app.ui.widgets.model_card_grid import ModelCardGrid
    from noelle.runtime import runnability

    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    grid = ModelCardGrid(scroll)
    qtbot.addWidget(grid)
    grid.set_compat_function(lambda _m: None)
    grid.set_runnability_function(
        lambda model: runnability.RunnabilityResult(
            runnability.UNSUPPORTED_LIBRARY if model.id == "hexgrad/Kokoro-82M" else runnability.UNKNOWN,
            library="kokoro" if model.id == "hexgrad/Kokoro-82M" else None))
    good = ModelSummary(id="org/good", author="org", pipeline_tag="text-generation")
    bad = ModelSummary(id="hexgrad/Kokoro-82M", author="hexgrad", pipeline_tag="text-to-speech")
    grid.set_models([good, bad])
    grid.set_runs_filter(True)
    by_id = {card.model.id: card for card in grid.cards}
    assert by_id["hexgrad/Kokoro-82M"].isHidden()
    assert not by_id["org/good"].isHidden()
    grid.set_runs_filter(False)
    assert not by_id["hexgrad/Kokoro-82M"].isHidden()