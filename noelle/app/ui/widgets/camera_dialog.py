"""Explicit webcam capture dialog for the vision playground.

The camera is opened only while this dialog is visible; closing it stops the device.
Captured frames are temporary PNGs and are deleted when the playground replaces them.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout

from noelle.app.ui.widgets.common import button, label


class CameraDialog(QDialog):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera")
        self.setMinimumSize(720, 560)
        self.image_path: str | None = None
        self._camera: Any = None
        self._session: Any = None
        self._capture: Any = None
        self._video: Any = None

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(label("Camera", "muted"))
        self.devices = QComboBox()
        top.addWidget(self.devices, 1)
        self.start_btn = button("Start", "play", "primary")
        self.start_btn.clicked.connect(self._start)
        top.addWidget(self.start_btn)
        root.addLayout(top)

        self.preview_host = QLabel("The camera is off.")
        self.preview_host.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_host.setMinimumSize(640, 400)
        self.preview_host.setStyleSheet("QLabel { background: #111; color: #888; border-radius: 8px; }")
        root.addWidget(self.preview_host, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = button("Cancel", "x", "ghost")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        self.take_btn = button("Capture", "image", "primary")
        self.take_btn.setEnabled(False)
        self.take_btn.clicked.connect(self._take)
        actions.addWidget(self.take_btn)
        root.addLayout(actions)

        self._load_devices()

    def _load_devices(self) -> None:
        try:
            from PySide6.QtMultimedia import QMediaDevices

            for camera in QMediaDevices.videoInputs():
                self.devices.addItem(camera.description(), camera)
        except Exception:  # noqa: BLE001 - no QtMultimedia backend on this system
            pass
        if self.devices.count() == 0:
            self.devices.addItem("No camera found", None)
            self.start_btn.setEnabled(False)
            self.preview_host.setText("No camera was found. Check the system camera permission.")

    def _start(self) -> None:
        device = self.devices.currentData()
        if device is None:
            return
        try:
            from PySide6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession
            from PySide6.QtMultimediaWidgets import QVideoWidget

            if self._camera is not None:
                self._camera.stop()
            self._camera = QCamera(device, self)
            self._session = QMediaCaptureSession(self)
            self._session.setCamera(self._camera)
            self._video = QVideoWidget(self)
            self._session.setVideoOutput(self._video)
            self._capture = QImageCapture(self)
            self._session.setImageCapture(self._capture)
            self._capture.imageCaptured.connect(self._captured)
            self._capture.errorOccurred.connect(self._error)
            self.preview_host.hide()
            self.layout().insertWidget(1, self._video, 1)
            self._camera.start()
            self.take_btn.setEnabled(True)
            self.start_btn.setText("Restart")
        except Exception as exc:  # noqa: BLE001
            self.preview_host.setText(f"Could not open the camera: {exc}")

    def _take(self) -> None:
        if self._capture is None:
            return
        try:
            self._capture.capture()
        except Exception as exc:  # noqa: BLE001
            self._error(-1, str(exc))

    def _captured(self, _id: int, image: Any) -> None:
        path = Path(tempfile.gettempdir()) / f"noelle-camera-{int(time.time() * 1000)}.png"
        if image.save(str(path), "PNG"):
            self.image_path = str(path)
            self._stop()
            self.accept()
        else:
            self._error(-1, "Could not save the captured frame.")

    def _error(self, _id: int, message: str) -> None:
        self.preview_host.setText(f"Camera error: {message}")

    def _stop(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:  # noqa: BLE001
                pass

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self._stop()
        super().closeEvent(event)

    def reject(self) -> None:
        self._stop()
        super().reject()