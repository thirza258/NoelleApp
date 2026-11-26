"""Vision playground: VLM question answering, object detection and image classification."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QMargins, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.chat_canvas import MarkdownView
from noelle.app.ui.widgets.common import ErrorLine, Spinner, button, label
from noelle.runtime import tasks
from noelle.runtime.protocol import Command

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff)"
BOX_COLORS = ["#8B7CF6", "#3DBE84", "#E9B949", "#4FA3F7", "#F0645F", "#E07BD0", "#5CC8C8"]


class ImageCanvas(QWidget):
    """Shows an image scaled to fit, draws detection boxes, accepts drag-and-drop."""

    image_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 260)
        self._pixmap: QPixmap | None = None
        self._boxes: list[dict[str, Any]] = []

    def set_image(self, path: str | None) -> None:
        self._pixmap = QPixmap(path) if path else None
        self._boxes = []
        self.update()

    def set_boxes(self, boxes: list[dict[str, Any]]) -> None:
        self._boxes = boxes
        self.update()

    def image_size(self) -> tuple[int, int] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return self._pixmap.width(), self._pixmap.height()

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.image_dropped.emit(url.toLocalFile())
                return

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1, Qt.PenStyle.DashLine))
        painter.setBrush(QColor(theme.SURFACE))
        painter.drawRoundedRect(rect, 12, 12)
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Drop an image here\nor use “Open image…”")
            return
        scaled = self._pixmap.size().scaled(rect.size().shrunkBy(QMargins(12, 12, 12, 12)),
                                            Qt.AspectRatioMode.KeepAspectRatio)
        x = rect.x() + (rect.width() - scaled.width()) / 2
        y = rect.y() + (rect.height() - scaled.height()) / 2
        target = QRectF(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        sx = scaled.width() / self._pixmap.width()
        sy = scaled.height() / self._pixmap.height()
        font = QFont(self.font())
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        for index, det in enumerate(self._boxes):
            box = det["box"]
            color = QColor(BOX_COLORS[index % len(BOX_COLORS)])
            r = QRectF(x + box["xmin"] * sx, y + box["ymin"] * sy, (box["xmax"] - box["xmin"]) * sx,
                       (box["ymax"] - box["ymin"]) * sy)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)
            text = f"{det['label']} {det['score']:.2f}"
            width = painter.fontMetrics().horizontalAdvance(text) + 8
            tag = QRectF(r.x(), max(r.y() - 16, y), width, 16)
            painter.fillRect(tag, color)
            painter.setPen(QColor("#0C0A1A"))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, text)


class VisionPlayground(QWidget):
    def __init__(self, ctx: AppContext, payload: Callable[[], dict[str, Any]], family: str) -> None:
        super().__init__()
        self.ctx = ctx
        self.payload = payload
        self.family = family
        self.image_path: str | None = None
        self._handle: Any = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        self.canvas = ImageCanvas()
        self.canvas.image_dropped.connect(self.set_image)
        left_layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        open_btn = button("Open image…", "image")
        open_btn.clicked.connect(self._pick)
        row.addWidget(open_btn)
        paste_btn = button("Paste", "copy")
        paste_btn.setToolTip("Use an image from the clipboard (Ctrl/Cmd+V screenshots)")
        paste_btn.clicked.connect(self._paste)
        row.addWidget(paste_btn)
        self.image_info = label("", "faint")
        row.addWidget(self.image_info, 1)
        left_layout.addLayout(row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)
        titles = {tasks.VISION: "Ask about the image", tasks.DETECTION: "Detect objects",
                  tasks.CLASSIFICATION: "Classify the image"}
        right_layout.addWidget(label(titles.get(family, "Vision"), "h2"))
        self.prompt = QPlainTextEdit("Describe this image in detail.")
        self.prompt.setFixedHeight(70)
        self.prompt.setVisible(family == tasks.VISION)
        right_layout.addWidget(self.prompt)
        self.threshold_row = QWidget()
        threshold_layout = QHBoxLayout(self.threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(label("Confidence threshold", "muted"))
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(5, 95)
        self.threshold.setValue(50)
        threshold_layout.addWidget(self.threshold, 1)
        self.threshold_value = label("0.50", "mono")
        self.threshold.valueChanged.connect(lambda v: self.threshold_value.setText(f"{v / 100:.2f}"))
        threshold_layout.addWidget(self.threshold_value)
        self.threshold_row.setVisible(family == tasks.DETECTION)
        right_layout.addWidget(self.threshold_row)
        run_row = QHBoxLayout()
        self.run_btn = button({tasks.VISION: "Ask", tasks.DETECTION: "Detect"}.get(family, "Classify"), "play",
                              "primary")
        self.run_btn.clicked.connect(self.run)
        run_row.addWidget(self.run_btn)
        self.spinner = Spinner(16)
        run_row.addWidget(self.spinner)
        self.status = label("", "faint")
        run_row.addWidget(self.status, 1)
        right_layout.addLayout(run_row)
        # Errors get their own full-width, wrapped, selectable line: the status row is too
        # narrow for a message carrying a snapshot path or a library traceback.
        self.error = ErrorLine()
        right_layout.addWidget(self.error)
        self.answer = MarkdownView()
        self.answer.setVisible(family == tasks.VISION)
        right_layout.addWidget(self.answer)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "Score", "Box" if family == tasks.DETECTION else ""])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setVisible(family != tasks.VISION)
        right_layout.addWidget(self.table, 1)
        right_layout.addStretch(0 if family != tasks.VISION else 1)
        splitter.addWidget(right)
        splitter.setSizes([560, 420])

    def _pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", IMAGE_FILTER)
        if path:
            self.set_image(path)

    def _paste(self) -> None:
        image = QGuiApplication.clipboard().image()
        if image.isNull():
            self.ctx.toast("The clipboard does not contain an image.", "info")
            return
        path = Path(tempfile.gettempdir()) / f"noelle-paste-{int(time.time() * 1000)}.png"
        if not image.save(str(path), "PNG"):
            self.ctx.toast("Could not save the pasted image.", "warning")
            return
        self.set_image(str(path))

    def set_image(self, path: str) -> None:
        if not Path(path).is_file():
            return
        self.image_path = path
        self.canvas.set_image(path)
        size = self.canvas.image_size()
        self.image_info.setText(f"{Path(path).name} · {size[0]}×{size[1]} px" if size else Path(path).name)
        self.table.setRowCount(0)
        self.answer.set_markdown("")

    def run(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            return
        if not self.image_path:
            self.ctx.toast("Open an image first.", "info")
            return
        payload = {**self.payload(), "image_path": self.image_path}
        self.error.clear()
        self.spinner.start()
        self.status.setText("Running…")
        if self.family == tasks.VISION:
            payload.update(prompt=self.prompt.toPlainText().strip(), params={"max_new_tokens": 384, "temperature": 0.2})
            self._text = ""
            self.answer.set_markdown("")
            handle = self.ctx.runtime.request(Command.VISION_GENERATE, payload)
            handle.token.connect(self._on_token)
            self.run_btn.setText("Stop")
        elif self.family == tasks.DETECTION:
            payload["threshold"] = self.threshold.value() / 100
            handle = self.ctx.runtime.request(Command.DETECT_OBJECTS, payload)
        else:
            payload["top_k"] = 5
            handle = self.ctx.runtime.request(Command.CLASSIFY_IMAGE, payload)
        handle.status.connect(lambda p: self.status.setText(p.get("message", "")))
        handle.finished.connect(self._on_finished)
        handle.failed.connect(self._on_failed)
        self._handle = handle

    def _on_token(self, text: str) -> None:
        self._text += text
        self.answer.set_markdown(self._text)

    def _on_finished(self, payload: dict[str, Any]) -> None:
        self._reset()
        metrics = payload.get("metrics") or {}
        if self.family == tasks.VISION:
            self.answer.set_markdown(payload.get("text", self._text))
            tps = metrics.get("tokens_per_second")
            self.status.setText(f"TTFT {metrics.get('ttft_ms', 0):.0f} ms" + (f" · {tps:.1f} tok/s" if tps else ""))
            return
        rows = payload.get("detections") if self.family == tasks.DETECTION else payload.get("labels")
        rows = rows or []
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(row["label"])))
            self.table.setItem(i, 1, QTableWidgetItem(f"{row['score']:.3f}"))
            if "box" in row:
                b = row["box"]
                self.table.setItem(i, 2, QTableWidgetItem(f"{b['xmin']:.0f},{b['ymin']:.0f} → {b['xmax']:.0f},{b['ymax']:.0f}"))
        if self.family == tasks.DETECTION:
            self.canvas.set_boxes(rows)
        self.status.setText(f"{len(rows)} result(s) in {metrics.get('total_ms', 0):.0f} ms")

    def _on_failed(self, payload: dict[str, Any]) -> None:
        self._reset()
        self.status.setText("Failed.")
        self.error.set_error(str(payload.get("message") or "The request failed."),
                             str(payload["hint"]) if payload.get("hint") else None)

    def _reset(self) -> None:
        self._handle = None
        self.spinner.stop()
        self.run_btn.setText({tasks.VISION: "Ask", tasks.DETECTION: "Detect"}.get(self.family, "Classify"))

    def shutdown(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
