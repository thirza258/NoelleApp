"""Scroll-driven hero collapse for the discovery page (task T-204).

Scrolling the catalog past ``threshold`` px animates the hero column's
``maximumWidth`` from its full width to 0 (``OutQuad``, 250 ms) and hides it,
so the grid expands to the full width. Scrolling back to the very top reverses
the animation. The two trigger points (``> threshold`` to collapse, ``== 0`` to
expand) form a hysteresis band, so the reflow caused by the width change can
never bounce the state back and forth.

The animated container has an explicit zero minimum width (so it really reaches
0 px instead of stopping at its minimum size hint); its child keeps a fixed width and is
clipped instead of re-wrapping text on every frame. The grid is told to defer
its reflow until the animation settles.
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, Signal
from PySide6.QtWidgets import QScrollArea, QSizePolicy, QWidget


class Reflowable(Protocol):
    def suspend_reflow(self, suspended: bool) -> None: ...


class HeroCollapseController(QObject):
    collapsed_changed = Signal(bool)
    animation_finished = Signal(bool)

    def __init__(self, hero_column: QWidget, scroll_area: QScrollArea, *, threshold: int = 120,
                 width: int = 380, duration_ms: int = 250, grid: Reflowable | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent or hero_column)
        self.hero = hero_column
        self.scroll_area = scroll_area
        self.threshold = threshold
        self.width = width
        self.grid = grid
        self._collapsed = False
        self._enabled = True
        hero_column.setMinimumWidth(0)
        hero_column.setMaximumWidth(width)
        policy = hero_column.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
        hero_column.setSizePolicy(policy)
        self.animation = QPropertyAnimation(hero_column, b"maximumWidth", self)
        self.animation.setDuration(duration_ms)
        self.animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.animation.finished.connect(self._on_finished)
        scroll_area.verticalScrollBar().valueChanged.connect(self.on_scroll)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def set_enabled(self, enabled: bool) -> None:
        """Pages without a hero (e.g. the local library) keep it collapsed."""
        self._enabled = enabled
        if not enabled:
            self.collapse(animated=False)
        elif self.scroll_area.verticalScrollBar().value() <= 0:
            self.expand()

    def on_scroll(self, value: int) -> None:
        if not self._enabled:
            return
        if not self._collapsed and value > self.threshold:
            self.collapse()
        elif self._collapsed and value <= 0:
            self.expand()

    def collapse(self, animated: bool = True) -> None:
        if self._collapsed and self.animation.state() != QPropertyAnimation.State.Running:
            return
        self._collapsed = True
        self.collapsed_changed.emit(True)
        self._run(0, animated)

    def expand(self, animated: bool = True) -> None:
        if not self._enabled:
            return
        if not self._collapsed and self.hero.isVisible():
            return
        self._collapsed = False
        self.hero.show()
        self.collapsed_changed.emit(False)
        self._run(self.width, animated)

    def _run(self, target: int, animated: bool) -> None:
        self.animation.stop()
        if self.grid is not None:
            self.grid.suspend_reflow(True)
        if not animated:
            self.hero.setMaximumWidth(target)
            self._on_finished()
            return
        self.animation.setStartValue(self.hero.maximumWidth() if self.hero.isVisible() else 0)
        self.animation.setEndValue(target)
        self.animation.start()

    def _on_finished(self) -> None:
        if self._collapsed:
            self.hero.setMaximumWidth(0)
            self.hero.hide()
        else:
            self.hero.setMaximumWidth(self.width)
        if self.grid is not None:
            self.grid.suspend_reflow(False)
        self.animation_finished.emit(self._collapsed)
