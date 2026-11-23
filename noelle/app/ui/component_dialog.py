"""Ask before downloading an optional component, then install it with progress."""

from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from noelle.app import icons, theme
from noelle.app.services import components
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import Spinner, button, label
from noelle.runtime.preflight import format_bytes


class ComponentInstallDialog(QDialog):
    """Resolves the release (metadata only), shows exactly what would be downloaded and
    from where, and downloads nothing until the user presses *Download and install*."""

    def __init__(self, parent: QWidget | None = None, *, reason: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Download llama.cpp tools?")
        self.setMinimumWidth(560)
        self.release: components.ComponentRelease | None = None
        self.result_info: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._installing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(12)
        head = QHBoxLayout()
        art = QLabel()
        art.setPixmap(icons.pixmap("download", theme.ACCENT, 28))
        head.addWidget(art, 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.addWidget(label("Download llama.cpp tools?", "h2"))
        titles.addWidget(label(reason or "GGUF quantization and GGUF models need the llama.cpp tools. Noelle can "
                                         "download them for you.", "muted", wrap=True))
        head.addLayout(titles, 1)
        root.addLayout(head)
        self.details = label("Checking the latest llama.cpp release…", wrap=True, selectable=True)
        root.addWidget(self.details)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        root.addWidget(self.progress)
        self.status = label("", "faint", wrap=True)
        root.addWidget(self.status)
        buttons = QHBoxLayout()
        self.spinner = Spinner(16)
        buttons.addWidget(self.spinner)
        buttons.addStretch(1)
        self.cancel_btn = button("Not now", None, "ghost")
        self.cancel_btn.clicked.connect(self._cancel)
        buttons.addWidget(self.cancel_btn)
        self.install_btn = button("Download and install", "download", "primary")
        self.install_btn.setEnabled(False)
        self.install_btn.clicked.connect(self._install)
        buttons.addWidget(self.install_btn)
        root.addLayout(buttons)
        self.spinner.start()
        runner().submit(components.resolve_llama_cpp, on_result=self._resolved, on_error=self._resolve_failed)

    def _resolved(self, release: components.ComponentRelease) -> None:
        self.spinner.stop()
        self.release = release
        self.details.setText(
            f"<b>What:</b> llama.cpp {release.tag} — <code>{release.asset_name}</code> "
            f"({format_bytes(release.asset_size)}) plus the GGUF converter script from the same release.<br>"
            f"<b>From:</b> {release.homepage} (official releases, {release.license} licence)<br>"
            f"<b>To:</b> {components.install_root()}<br>"
            "<b>Checks:</b> binaries against the SHA-256 GitHub publishes, converter files against the release "
            "tag's git hashes; nothing outside that folder is changed.")
        if not release.asset_sha256:
            self.details.setText(self.details.text() + f"<br><span style='color:{theme.DANGER}'>GitHub did not "
                                 "publish a checksum for this build, so it cannot be installed.</span>")
            return
        self.install_btn.setEnabled(True)

    def _resolve_failed(self, message: str) -> None:
        self.spinner.stop()
        self.details.setText(f"<span style='color:{theme.DANGER}'>Could not look up llama.cpp releases: "
                             f"{message}</span>")
        self.cancel_btn.setText("Close")

    def _install(self) -> None:
        if self.release is None or self._installing:
            return
        self._installing = True
        self.install_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")
        self.progress.show()
        self.spinner.start()

        def work(progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            return components.install_llama_cpp(self.release, progress=progress, stop_event=self._stop)

        runner().submit(work, on_progress=self._on_progress, on_result=self._done, on_error=self._failed)

    def _on_progress(self, data: dict[str, Any]) -> None:
        stage = data.get("stage")
        total, current = data.get("total"), data.get("current")
        if stage == "download" and total:
            self.progress.setValue(int(current / total * 1000))
            self.status.setText(f"Downloading {data.get('file')} — {format_bytes(current)} of {format_bytes(total)}")
        elif stage == "download":
            self.progress.setRange(0, 0)
            self.status.setText(f"Downloading {data.get('file')} — {format_bytes(current)}")
        elif stage in ("verified", "downloaded"):
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
            self.status.setText(f"{data.get('file')} {'verified (SHA-256)' if stage == 'verified' else 'downloaded'}")
        elif stage == "extract":
            self.status.setText("Installing…")

    def _done(self, info: dict[str, Any]) -> None:
        from noelle.runtime import capabilities

        capabilities.detect_backends_cached.cache_clear()  # the advisor should see the new tools right away
        self.spinner.stop()
        self.result_info = info
        self.status.setText(f"Installed llama.cpp {info.get('tag')}.")
        self.accept()

    def _failed(self, message: str) -> None:
        self.spinner.stop()
        self._installing = False
        self.progress.hide()
        cancelled = self._stop.is_set()
        self.status.setText("Cancelled — nothing was installed." if cancelled else
                            f"<span style='color:{theme.DANGER}'>Install failed: {message}</span>")
        self.cancel_btn.setText("Close")
        self.install_btn.setEnabled(not cancelled and self.release is not None)
        self._stop.clear()

    def _cancel(self) -> None:
        if self._installing:
            self._stop.set()
            return
        self.reject()


def ensure_llama_cpp(parent: QWidget | None, on_ready: Callable[[], None] | None = None,
                     reason: str | None = None) -> bool:
    """Run ``on_ready`` if the tools are installed; otherwise ask the user first. Returns True if ready now."""
    if components.installed_llama_cpp() is not None:
        if on_ready:
            on_ready()
        return True
    dialog = ComponentInstallDialog(parent, reason=reason)
    if dialog.exec() == QDialog.DialogCode.Accepted and on_ready:
        on_ready()
    return False
