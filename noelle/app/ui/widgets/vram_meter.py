"""Bottom status bar: runtime state, GPU/VRAM, RAM and CPU telemetry."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from noelle.app import icons, theme
from noelle.app.ui.widgets.common import button, label
from noelle.runtime.preflight import format_bytes

STATE_TEXT = {
    "stopped": ("Runtime idle", theme.TEXT_FAINT),
    "starting": ("Starting runtime…", theme.WARNING),
    "warming": ("Runtime warming up…", theme.WARNING),
    "ready": ("Runtime ready", theme.SUCCESS),
    "crashed": ("Runtime crashed", theme.DANGER),
    "unresponsive": ("Runtime not responding", theme.DANGER),
}


def level_color(fraction: float) -> str:
    if fraction >= 0.9:
        return theme.DANGER
    if fraction >= 0.7:
        return theme.WARNING
    return theme.SUCCESS


class MeterBar(QWidget):
    def __init__(self, width: int = 72, height: int = 6) -> None:
        super().__init__()
        self._fraction = 0.0
        self._marker: float | None = None
        self.setFixedSize(width, height)

    def set_fraction(self, fraction: float, marker: float | None = None) -> None:
        self._fraction = max(0.0, min(fraction, 1.0))
        self._marker = marker
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        track = QPainterPath()
        track.addRoundedRect(rect, radius, radius)
        painter.fillPath(track, QColor(theme.SURFACE_4))
        if self._fraction > 0:
            fill = QPainterPath()
            fill.addRoundedRect(QRectF(rect.x(), rect.y(), max(rect.width() * self._fraction, rect.height()),
                                       rect.height()), radius, radius)
            painter.fillPath(fill, QColor(level_color(self._fraction)))
        if self._marker is not None:
            x = rect.x() + rect.width() * max(0.0, min(self._marker, 1.0))
            painter.setPen(QColor(theme.TEXT_2))
            painter.drawLine(int(x), 0, int(x), int(rect.height()))


class Meter(QWidget):
    """Caption + bar + value, e.g. ``VRAM ▓▓▓░░ 3.1 / 12.7 GB``."""

    def __init__(self, caption: str, icon_name: str | None = None, bar_width: int = 72) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        if icon_name:
            art = QLabel()
            art.setPixmap(icons.pixmap(icon_name, theme.TEXT_FAINT, 13))
            layout.addWidget(art)
        self.caption = label(caption, "faint")
        layout.addWidget(self.caption)
        self.bar = MeterBar(bar_width)
        layout.addWidget(self.bar, 0, Qt.AlignmentFlag.AlignVCenter)
        self.value = label("—", "faint")
        layout.addWidget(self.value)

    def set(self, used: float | None, total: float | None, text: str | None = None, marker: float | None = None) -> None:
        if used is None or not total:
            self.bar.set_fraction(0)
            self.value.setText(text or "—")
            return
        self.bar.set_fraction(used / total, marker)
        self.value.setText(text or f"{format_bytes(used)} / {format_bytes(total)}")


class TelemetryBar(QFrame):
    restart_requested = Signal()
    server_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(16)
        self.state_dot = QLabel("●")
        layout.addWidget(self.state_dot)
        self.state_text = label("Runtime stopped", "faint")
        layout.addWidget(self.state_text)
        self.restart_btn = button("Restart", "refresh", "ghost")
        self.restart_btn.setStyleSheet("QPushButton { padding: 2px 8px; }")
        self.restart_btn.clicked.connect(self.restart_requested)
        self.restart_btn.hide()
        layout.addWidget(self.restart_btn)
        self.models_text = label("", "faint")
        layout.addWidget(self.models_text)
        layout.addStretch(1)
        self.gpu_name = label("", "faint")
        layout.addWidget(self.gpu_name)
        self.gpu_util = label("", "faint")
        layout.addWidget(self.gpu_util)
        self.vram = Meter("VRAM", "cpu")
        layout.addWidget(self.vram)
        self.ram = Meter("RAM", None, 56)
        layout.addWidget(self.ram)
        self.cpu = label("CPU —", "faint")
        layout.addWidget(self.cpu)
        self.server = button("API off", "server", "ghost")
        self.server.setStyleSheet("QPushButton { padding: 2px 8px; color: %s; }" % theme.TEXT_FAINT)
        self.server.clicked.connect(self.server_clicked)
        layout.addWidget(self.server)

    def set_runtime_state(self, state: str) -> None:
        text, color = STATE_TEXT.get(state, (state, theme.TEXT_FAINT))
        self.state_text.setText(text)
        self.state_dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.restart_btn.setVisible(state in ("crashed", "unresponsive", "stopped"))
        self.restart_btn.setText("Start" if state == "stopped" else "Restart")

    def set_server_state(self, state: str, port: int | None) -> None:
        if state == "running" and port:
            self.server.setText(f"API :{port}")
            self.server.setStyleSheet("QPushButton { padding: 2px 8px; color: %s; }" % theme.SUCCESS)
        else:
            self.server.setText({"starting": "API starting…", "error": "API error"}.get(state, "API off"))
            color = theme.DANGER if state == "error" else theme.TEXT_FAINT
            self.server.setStyleSheet("QPushButton { padding: 2px 8px; color: %s; }" % color)

    def update_metrics(self, metrics: Any) -> None:
        self.gpu_name.setText(metrics.gpu.name)
        self.gpu_util.setText(f"GPU {metrics.gpu_util:.0f}%" if metrics.gpu_util is not None else "")
        total = metrics.vram_total
        used = metrics.vram_used
        marker = (metrics.budget_usable / total) if (metrics.budget_usable and total) else None
        if metrics.gpu.kind == "cpu":
            self.vram.set(None, None, "CPU mode")
        else:
            self.vram.set(used, total, marker=marker)
            tip = [f"Accelerator memory in use: {format_bytes(used)}"]
            if metrics.worker_allocated is not None:
                tip.append(f"Allocated by models: {format_bytes(metrics.worker_allocated)}")
            if metrics.worker_peak is not None:
                tip.append(f"Peak since last reset: {format_bytes(metrics.worker_peak)}")
            if metrics.budget_usable:
                tip.append(f"Usable budget (after headroom): {format_bytes(metrics.budget_usable)}")
            self.vram.setToolTip("\n".join(tip))
        self.ram.set(metrics.ram_used, metrics.ram_total)
        self.cpu.setText(f"CPU {metrics.cpu_percent:.0f}%")
        loaded = metrics.loaded_models
        if loaded:
            self.models_text.setText(f"· {len(loaded)} model{'s' if len(loaded) != 1 else ''} loaded")
            self.models_text.setToolTip("\n".join(f"{m['model_id']} ({m['precision']}, {format_bytes(m['memory_bytes'])})"
                                                  for m in loaded))
        else:
            self.models_text.setText("")

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1200, 32)
