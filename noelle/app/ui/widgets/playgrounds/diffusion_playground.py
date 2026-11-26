"""Diffusion playground: prompt/negative prompt, steps, CFG, seed lock, size, img2img, live latent preview."""

from __future__ import annotations

import base64
import random
import shutil
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import ErrorLine, button, icon_button, label
from noelle.runtime.protocol import Command

SIZES = [("512 × 512", (512, 512)), ("768 × 768", (768, 768)), ("1024 × 1024", (1024, 1024)),
         ("512 × 768 (portrait)", (512, 768)), ("768 × 512 (landscape)", (768, 512))]


class ImagePreview(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 360)
        self.setStyleSheet(f"QLabel {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER};"
                           f" border-radius: 12px; color: {theme.TEXT_FAINT}; }}")
        self._pixmap: QPixmap | None = None
        self.setText("Your image appears here")

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        target = self.size() - QSize(16, 16)
        self.setPixmap(self._pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))


class DiffusionPlayground(QWidget):
    def __init__(self, ctx: AppContext, payload: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.ctx = ctx
        self.payload = payload
        self._handle: Any = None
        self.init_image: str | None = None
        self.last_image: str | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.addWidget(splitter)

        form_host = QWidget()
        form = QVBoxLayout(form_host)
        form.setContentsMargins(0, 0, 10, 0)
        form.setSpacing(9)
        form.addWidget(label("Prompt", "muted"))
        self.prompt = QPlainTextEdit("A cozy reading nook by a rainy window, warm lamp light, highly detailed")
        self.prompt.setFixedHeight(84)
        form.addWidget(self.prompt)
        form.addWidget(label("Negative prompt", "muted"))
        self.negative = QPlainTextEdit("blurry, low quality, distorted")
        self.negative.setFixedHeight(54)
        form.addWidget(self.negative)
        self.steps, self.steps_value = self._slider(form, "Inference steps", 1, 100, 25, lambda v: f"{v}")
        self.guidance, self.guidance_value = self._slider(form, "Guidance scale (CFG)", 0, 200, 70,
                                                          lambda v: f"{v / 10:.1f}")
        size_row = QHBoxLayout()
        size_row.addWidget(label("Size", "muted"))
        self.size = QComboBox()
        for text, value in SIZES:
            self.size.addItem(text, value)
        size_row.addWidget(self.size, 1)
        form.addLayout(size_row)
        seed_row = QHBoxLayout()
        seed_row.addWidget(label("Seed", "muted"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(random.randint(0, 999_999))
        seed_row.addWidget(self.seed, 1)
        dice = icon_button("refresh", "New random seed")
        dice.clicked.connect(lambda: self.seed.setValue(random.randint(0, 2_147_483_647)))
        seed_row.addWidget(dice)
        self.lock_seed = QCheckBox("Lock")
        self.lock_seed.setToolTip("Keep the same seed between runs for reproducible images")
        seed_row.addWidget(self.lock_seed)
        form.addLayout(seed_row)
        init_row = QHBoxLayout()
        self.init_btn = button("Init image…", "image", "ghost")
        self.init_btn.clicked.connect(self._pick_init)
        init_row.addWidget(self.init_btn)
        self.init_label = label("text-to-image", "faint")
        init_row.addWidget(self.init_label, 1)
        clear_init = icon_button("x", "Clear init image", size=26, icon_size=13)
        clear_init.clicked.connect(lambda: self._set_init(None))
        init_row.addWidget(clear_init)
        form.addLayout(init_row)
        self.strength, self.strength_value = self._slider(form, "Strength (img2img)", 5, 100, 60,
                                                          lambda v: f"{v / 100:.2f}")
        self.generate_btn = button("Generate", "sparkles", "primary")
        self.generate_btn.setMinimumHeight(38)
        self.generate_btn.clicked.connect(self._generate_or_stop)
        form.addWidget(self.generate_btn)
        form.addStretch(1)
        splitter.addWidget(form_host)

        view_host = QWidget()
        view = QVBoxLayout(view_host)
        view.setContentsMargins(10, 0, 0, 0)
        view.setSpacing(8)
        self.preview = ImagePreview()
        view.addWidget(self.preview, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        view.addWidget(self.progress)
        status_row = QHBoxLayout()
        self.status = label("Latent previews stream while the image denoises.", "faint")
        status_row.addWidget(self.status, 1)
        self.save_btn = button("Save as…", "save", "ghost")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        status_row.addWidget(self.save_btn)
        self.reuse_btn = button("Use as init", "image", "ghost")
        self.reuse_btn.clicked.connect(lambda: self._set_init(self.last_image))
        self.reuse_btn.setEnabled(False)
        status_row.addWidget(self.reuse_btn)
        view.addLayout(status_row)
        self.error = ErrorLine()
        view.addWidget(self.error)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(72, 72))
        self.gallery.setFixedHeight(96)
        self.gallery.setFlow(QListWidget.Flow.LeftToRight)
        self.gallery.itemClicked.connect(self._show_from_gallery)
        self.gallery.itemDoubleClicked.connect(lambda item: QDesktopServices.openUrl(
            QUrl.fromLocalFile(item.data(Qt.ItemDataRole.UserRole))))
        view.addWidget(self.gallery)
        splitter.addWidget(view_host)
        splitter.setSizes([360, 640])

    def _slider(self, layout: QVBoxLayout, caption: str, lo: int, hi: int, value: int,
                fmt: Callable[[int], str]) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        row.addWidget(label(caption, "muted"))
        row.addStretch(1)
        text = label(fmt(value), "mono")
        row.addWidget(text)
        layout.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v: text.setText(fmt(v)))
        layout.addWidget(slider)
        return slider, text

    def _pick_init(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Init image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self._set_init(path)

    def _set_init(self, path: str | None) -> None:
        self.init_image = path
        self.init_label.setText(Path(path).name if path else "text-to-image")

    def export_config(self) -> dict[str, Any]:
        width, height = self.size.currentData()
        return {"extra": {"steps": self.steps.value(), "guidance": self.guidance.value() / 10, "width": width,
                          "height": height, "negative_prompt": self.negative.toPlainText().strip(),
                          "seed": self.seed.value()},
                "system_prompt": None, "params": {}}

    def _generate_or_stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self.status.setText("Stopping…")
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            return
        if not self.lock_seed.isChecked() and self.last_image is not None:
            self.seed.setValue(random.randint(0, 2_147_483_647))
        width, height = self.size.currentData()
        payload = {**self.payload(), "prompt": prompt, "negative_prompt": self.negative.toPlainText().strip() or None,
                   "steps": self.steps.value(), "guidance": self.guidance.value() / 10, "seed": self.seed.value(),
                   "width": width, "height": height, "init_image": self.init_image,
                   "strength": self.strength.value() / 100, "preview_every": 1}
        self.error.clear()
        self.progress.setValue(0)
        self.status.setText("Loading pipeline…")
        handle = self.ctx.runtime.request(Command.GENERATE_IMAGE, payload)
        handle.status.connect(lambda p: self.status.setText(p.get("message", "")))
        handle.progress.connect(self._on_progress)
        handle.finished.connect(self._on_finished)
        handle.failed.connect(self._on_failed)
        self._handle = handle
        self.generate_btn.setText("Stop")
        theme.set_prop(self.generate_btn, "variant", "danger")

    def _on_progress(self, data: dict[str, Any]) -> None:
        current, total = int(data.get("current") or 0), int(data.get("total") or 1)
        self.progress.setValue(int(current / max(total, 1) * 100))
        self.status.setText(f"Denoising step {current}/{total}")
        preview = data.get("preview_png")
        if preview:
            pixmap = QPixmap()
            pixmap.loadFromData(base64.b64decode(preview), "PNG")
            self.preview.set_pixmap(pixmap)

    def _on_finished(self, result: dict[str, Any]) -> None:
        self._reset()
        path = result.get("image_path")
        if not path:
            return
        self.last_image = path
        pixmap = QPixmap(path)
        self.preview.set_pixmap(pixmap)
        self.progress.setValue(100)
        metrics = result.get("metrics") or {}
        self.status.setText(f"{result.get('width')}×{result.get('height')} · seed {result.get('seed')} · "
                            f"{metrics.get('total_ms', 0) / 1000:.1f} s ({metrics.get('seconds_per_step', 0):.2f} s/step)")
        self.save_btn.setEnabled(True)
        self.reuse_btn.setEnabled(True)
        item = QListWidgetItem(QIcon(pixmap), "")
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(f"seed {result.get('seed')}")
        self.gallery.insertItem(0, item)

    def _on_failed(self, payload: dict[str, Any]) -> None:
        self._reset()
        if payload.get("code") == "CANCELLED":
            self.status.setText("Stopped.")
            return
        self.status.setText("Failed.")
        self.error.set_error(str(payload.get("message") or "Image generation failed."),
                             str(payload["hint"]) if payload.get("hint") else None)

    def _reset(self) -> None:
        self._handle = None
        self.generate_btn.setText("Generate")
        theme.set_prop(self.generate_btn, "variant", "primary")

    def _show_from_gallery(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        self.last_image = path
        self.preview.set_pixmap(QPixmap(path))

    def _save(self) -> None:
        if not self.last_image:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save image", Path(self.last_image).name, "PNG (*.png)")
        if path:
            shutil.copyfile(self.last_image, path)

    def shutdown(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
