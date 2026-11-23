"""Waveform display and microphone capture (16 kHz mono WAV) for the audio playground."""

from __future__ import annotations

import io
import time
import wave
from array import array
from typing import Any

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

from noelle import paths
from noelle.app import theme


class WaveformView(QWidget):
    seek_requested = Signal(float)

    def __init__(self, height: int = 96) -> None:
        super().__init__()
        self.setMinimumHeight(height)
        self._envelope: list[float] = []
        self._playhead: float | None = None
        self._live = False

    def set_envelope(self, envelope: list[float], live: bool = False) -> None:
        peak = max(envelope) if envelope else 0.0
        self._envelope = [v / peak for v in envelope] if peak > 0 else list(envelope)
        self._live = live
        self.update()

    def set_playhead(self, fraction: float | None) -> None:
        self._playhead = fraction
        self.update()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if self._envelope and self.width() > 0:
            self.seek_requested.emit(max(0.0, min(event.position().x() / self.width(), 1.0)))

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        frame = QPainterPath()
        frame.addRoundedRect(rect, 10, 10)
        painter.fillPath(frame, QColor(theme.SURFACE))
        if not self._envelope:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No audio yet")
            return
        mid = rect.center().y()
        usable = rect.height() / 2 - 8
        n = len(self._envelope)
        bar = max(rect.width() / n, 1.0)
        color = QColor(theme.DANGER if self._live else theme.ACCENT)
        played = QColor(color)
        rest = QColor(color)
        rest.setAlpha(110)
        for i, value in enumerate(self._envelope):
            x = rect.x() + i * bar
            h = max(value * usable, 1.0)
            fraction = i / n
            painter.fillRect(QRectF(x, mid - h, max(bar - 1, 1), 2 * h),
                             played if (self._playhead is None or fraction <= self._playhead) else rest)
        if self._playhead is not None:
            x = rect.x() + rect.width() * self._playhead
            painter.setPen(QColor(theme.TEXT))
            painter.drawLine(int(x), int(rect.top() + 4), int(x), int(rect.bottom() - 4))


class MicRecorder(QObject):
    """Captures the default microphone as 16-bit PCM and writes a WAV file on stop."""

    level = Signal(float)
    envelope = Signal(list)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source: Any = None
        self._io: Any = None
        self._buffer = bytearray()
        self._format: Any = None
        self._levels: list[float] = []
        self.recording = False

    def start(self) -> None:
        from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            self.error.emit("No microphone was found.")
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
        self._format = fmt
        self._buffer = bytearray()
        self._levels = []
        self._source = QAudioSource(device, fmt, self)
        self._io = self._source.start()
        if self._io is None:
            self.error.emit("Could not open the microphone. Check the microphone permission in System Settings.")
            return
        self._io.readyRead.connect(self._read)
        self.recording = True

    def _read(self) -> None:
        data = bytes(self._io.readAll())
        if not data:
            return
        self._buffer.extend(data)
        samples = self._to_int16(data)
        if samples:
            peak = max(abs(s) for s in samples) / 32768.0
            self._levels.append(peak)
            self.level.emit(peak)
            self.envelope.emit(self._levels[-400:])

    def _to_int16(self, data: bytes) -> array:
        from PySide6.QtMultimedia import QAudioFormat

        sample_format = self._format.sampleFormat()
        if sample_format == QAudioFormat.SampleFormat.Int16:
            values = array("h")
            values.frombytes(data[: len(data) - len(data) % 2])
            return values
        if sample_format == QAudioFormat.SampleFormat.Float:
            floats = array("f")
            floats.frombytes(data[: len(data) - len(data) % 4])
            return array("h", (int(max(-1.0, min(1.0, f)) * 32767) for f in floats))
        if sample_format == QAudioFormat.SampleFormat.Int32:
            ints = array("i")
            ints.frombytes(data[: len(data) - len(data) % 4])
            return array("h", (i >> 16 for i in ints))
        return array("h", ((b - 128) << 8 for b in data))  # UInt8

    def stop(self) -> str | None:
        if self._source is None:
            return None
        self._source.stop()
        self.recording = False
        samples = self._to_int16(bytes(self._buffer))
        channels = self._format.channelCount()
        rate = self._format.sampleRate()
        self._source = None
        self._io = None
        if not samples:
            return None
        paths.outputs_dir().mkdir(parents=True, exist_ok=True)
        path = paths.outputs_dir() / f"recording-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples.tobytes())
        path.write_bytes(buffer.getvalue())
        return str(path)


def request_microphone(callback: Any) -> None:
    """Ask for microphone access where the platform requires it (macOS), then call ``callback(granted)``."""
    try:
        from PySide6.QtCore import QMicrophonePermission, Qt as QtNs
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        permission = QMicrophonePermission()
        status = app.checkPermission(permission)
        if status == QtNs.PermissionStatus.Granted:
            callback(True)
            return
        if status == QtNs.PermissionStatus.Denied:
            callback(False)
            return
        app.requestPermission(permission, app, lambda p: callback(app.checkPermission(p) == QtNs.PermissionStatus.Granted))
    except Exception:
        callback(True)  # platforms without a permission model
