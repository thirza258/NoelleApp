"""One viewport of the comparative bench."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLineEdit, QScrollArea, QVBoxLayout, QWidget

from noelle.app.ui.widgets.chat_canvas import MarkdownView
from noelle.app.ui.widgets.common import Pill, Spinner, icon_button, label
from noelle.runtime.preflight import format_bytes

STATUS_LEVEL = {"idle": "unknown", "queued": "unknown", "loading": "yellow", "executing": "accent",
                "finished": "green", "error": "red", "cancelled": "unknown"}


class ComparisonPane(QFrame):
    model_changed = Signal()

    def __init__(self, view: int) -> None:
        super().__init__()
        self.view = view
        self.setProperty("card", True)
        self._text = ""
        self.metrics: dict[str, Any] = {}
        self.status = "idle"
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(label(f"View {view}", "h3"))
        self.model = QComboBox()
        self.model.setMinimumWidth(180)
        self.model.currentIndexChanged.connect(lambda _i: self.model_changed.emit())
        head.addWidget(self.model, 1)
        self.pill = Pill("Idle", "unknown")
        head.addWidget(self.pill)
        self.spinner = Spinner(14)
        head.addWidget(self.spinner)
        copy = icon_button("copy", "Copy output", size=26, icon_size=13)
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self._text))
        head.addWidget(copy)
        root.addLayout(head)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Prompt for this view (broadcast is off)")
        self.prompt.hide()
        root.addWidget(self.prompt)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(2, 2, 6, 2)
        self.output = MarkdownView()
        body_layout.addWidget(self.output)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        self._scroll = scroll
        root.addWidget(scroll, 1)
        self.footer = label("", "faint")
        root.addWidget(self.footer)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._render)

    # ------------------------------------------------------------------- models
    def set_models(self, records: list[Any]) -> None:
        current = self.model.currentData()
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem("Select a model…", None)
        for record in records:
            self.model.addItem(record.display_name if record.source == "hub" else
                               f"{record.repo_id.split('/')[-1]} ({record.precision})", record.repo_id + "|" + record.precision)
        index = self.model.findData(current)
        self.model.setCurrentIndex(index if index >= 0 else 0)
        self.model.blockSignals(False)

    def selected_key(self) -> str | None:
        return self.model.currentData()

    def select_index(self, index: int) -> None:
        if 0 <= index < self.model.count():
            self.model.setCurrentIndex(index)

    def set_broadcast(self, broadcast: bool) -> None:
        self.prompt.setVisible(not broadcast)

    # ------------------------------------------------------------------- state
    def reset(self) -> None:
        self._text = ""
        self.metrics = {}
        self.output.set_markdown("")
        self.footer.setText("")
        self.set_status("queued" if self.selected_key() else "idle")

    def set_status(self, status: str, text: str | None = None) -> None:
        self.status = status
        self.pill.set_level(STATUS_LEVEL.get(status, "unknown"))
        self.pill.set_text(text or status.capitalize())
        if status in ("loading", "executing"):
            self.spinner.start()
        else:
            self.spinner.stop()

    def append(self, text: str) -> None:
        self._text += text
        if not self._timer.isActive():
            self._timer.start()

    def _render(self) -> None:
        self.output.set_markdown(self._text)
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def finish(self, result: dict[str, Any]) -> None:
        status = result.get("status", "finished")
        if status == "error":
            error = result.get("error") or {}
            self.set_status("error", "Error")
            self.output.set_markdown(f"**{error.get('message', 'Failed')}**\n\n{error.get('hint') or ''}")
            return
        self._text = result.get("text", self._text)
        self._render()
        self.metrics = result.get("metrics") or {}
        self.set_status("cancelled" if status == "cancelled" else "finished")
        m = self.metrics
        parts = []
        if m.get("ttft_ms") is not None:
            parts.append(f"TTFT {m['ttft_ms']:.0f} ms")
        if m.get("tokens_per_second"):
            parts.append(f"{m['tokens_per_second']:.1f} tok/s")
        if m.get("model_peak_bytes"):
            parts.append(f"peak {format_bytes(m['model_peak_bytes'])}")
        if m.get("load_ms"):
            parts.append(f"load {m['load_ms'] / 1000:.1f} s")
        self.footer.setText(" · ".join(parts))

    @property
    def text(self) -> str:
        return self._text
