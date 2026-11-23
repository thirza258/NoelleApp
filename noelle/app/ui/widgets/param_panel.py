"""Sampling hyper-parameter controls shared by the playground and the comparative bench."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QSlider, QSpinBox, QVBoxLayout, QWidget

from noelle.app.ui.widgets.common import label

# key, label, minimum, maximum, step, decimals, tooltip
SPECS: list[tuple[str, str, float, float, float, int, str]] = [
    ("temperature", "Temperature", 0.0, 2.0, 0.05, 2, "0 = greedy decoding; higher = more random"),
    ("top_p", "Top-p", 0.05, 1.0, 0.01, 2, "Nucleus sampling: keep the smallest token set with this probability mass"),
    ("top_k", "Top-k", 0, 200, 1, 0, "Keep only the k most likely tokens (0 = disabled)"),
    ("repetition_penalty", "Repetition penalty", 0.8, 2.0, 0.01, 2, "Values above 1 discourage repeating tokens"),
    ("max_new_tokens", "Max new tokens", 16, 8192, 16, 0,
     "Upper bound; the runtime clamps it to the context window and free memory"),
]


class SliderRow(QWidget):
    value_changed = Signal(float)

    def __init__(self, caption: str, minimum: float, maximum: float, step: float, decimals: int, tooltip: str) -> None:
        super().__init__()
        self._minimum = minimum
        self._step = step
        self._decimals = decimals
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        top = QHBoxLayout()
        name = label(caption, "muted")
        name.setToolTip(tooltip)
        top.addWidget(name)
        top.addStretch(1)
        self.value_label = label("", "mono")
        top.addWidget(self.value_label)
        layout.addLayout(top)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((maximum - minimum) / step)))
        self.slider.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

    def value(self) -> float:
        raw = self._minimum + self.slider.value() * self._step
        return round(raw, self._decimals) if self._decimals else float(int(round(raw)))

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round((float(value) - self._minimum) / self._step)))
        self._render()

    def _on_slider(self, _v: int) -> None:
        self._render()
        self.value_changed.emit(self.value())

    def _render(self) -> None:
        value = self.value()
        self.value_label.setText(f"{value:.{self._decimals}f}" if self._decimals else f"{int(value)}")


class ParamPanel(QWidget):
    changed = Signal()

    def __init__(self, defaults: dict[str, Any] | None = None, *, columns: int = 1, show_seed: bool = True) -> None:
        super().__init__()
        defaults = defaults or {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        self.rows: dict[str, SliderRow] = {}
        for index, (key, caption, lo, hi, step, decimals, tip) in enumerate(SPECS):
            row = SliderRow(caption, lo, hi, step, decimals, tip)
            row.set_value(defaults.get(key, (lo + hi) / 2))
            row.value_changed.connect(lambda _v: self.changed.emit())
            self.rows[key] = row
            grid.addWidget(row, index // columns, index % columns)
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2_147_483_647)
        self.seed.setValue(int(defaults.get("seed", -1) if defaults.get("seed") is not None else -1))
        self.seed.setSpecialValueText("random")
        self.seed.setToolTip("Fix the sampling seed for reproducible outputs (-1 = random)")
        if show_seed:
            seed_row = QWidget()
            seed_layout = QHBoxLayout(seed_row)
            seed_layout.setContentsMargins(0, 0, 0, 0)
            seed_layout.addWidget(label("Seed", "muted"))
            seed_layout.addStretch(1)
            seed_layout.addWidget(self.seed)
            grid.addWidget(seed_row, len(SPECS) // columns + (1 if len(SPECS) % columns else 0), 0)

    def values(self) -> dict[str, Any]:
        data: dict[str, Any] = {key: row.value() for key, row in self.rows.items()}
        data["top_k"] = int(data["top_k"])
        data["max_new_tokens"] = int(data["max_new_tokens"])
        data["seed"] = None if self.seed.value() < 0 else int(self.seed.value())
        return data

    def set_values(self, values: dict[str, Any]) -> None:
        for key, row in self.rows.items():
            if values.get(key) is not None:
                row.set_value(values[key])
        if "seed" in values:
            self.seed.setValue(-1 if values["seed"] is None else int(values["seed"]))
