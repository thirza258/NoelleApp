"""Responsive model card grid with hardware compatibility badges (task T-203)."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from noelle.app import icons, theme
from noelle.app.ui.widgets.common import Avatar, Card, ElidedLabel, Pill, Spinner, human_count, label, time_ago
from noelle.runtime.preflight import Compatibility, format_params

CARD_MIN_WIDTH = 286
CARD_HEIGHT = 140
SPACING = 12
_BADGE_TEXT = {"green": "Runs natively", "yellow": "Needs 4-bit", "red": "CPU fallback", "unknown": "Unknown size"}


class ModelCard(Card):
    selected = Signal(object)

    def __init__(self, model: Any, compat: Compatibility | None, downloaded: bool) -> None:
        super().__init__(clickable=True)
        self.model = model
        self.setMinimumWidth(CARD_MIN_WIDTH - 40)
        self.setFixedHeight(CARD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.clicked.connect(lambda: self.selected.emit(self.model))

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 13, 14, 12)
        root.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(Avatar(model.namespace, 34, radius=9))
        names = QVBoxLayout()
        names.setSpacing(1)
        self.name = ElidedLabel(model.name, "h3")
        names.addWidget(self.name)
        self.author = ElidedLabel(model.namespace, "muted")
        names.addWidget(self.author)
        head.addLayout(names, 1)
        root.addLayout(head)

        pills = QHBoxLayout()
        pills.setSpacing(6)
        if model.pipeline_tag:
            pills.addWidget(Pill(model.pipeline_tag, "info"))
        if model.param_count:
            pills.addWidget(Pill(format_params(model.param_count), "accent"))
        if downloaded:
            pills.addWidget(Pill("downloaded", "green"))
        elif model.license:
            pills.addWidget(Pill(str(model.license)[:16], "unknown"))
        pills.addStretch(1)
        root.addLayout(pills)

        root.addStretch(1)
        foot = QHBoxLayout()
        foot.setSpacing(12)
        if model.downloads or model.likes:  # local-only library entries have no Hub stats
            foot.addWidget(self._stat("arrow-down", human_count(model.downloads)))
            foot.addWidget(self._stat("heart", human_count(model.likes)))
        if model.last_modified:
            foot.addWidget(label(time_ago(model.last_modified), "faint"))
        foot.addStretch(1)
        self.badge = Pill("", "unknown")
        foot.addWidget(self.badge)
        root.addLayout(foot)
        self.set_compat(compat)

    @staticmethod
    def _stat(icon_name: str, text: str) -> QWidget:
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        art = QLabel()
        art.setPixmap(icons.pixmap(icon_name, theme.TEXT_FAINT, 13))
        layout.addWidget(art)
        layout.addWidget(label(text, "faint"))
        return host

    def set_compat(self, compat: Compatibility | None) -> None:
        if compat is None:
            self.badge.hide()
            return
        self.badge.show()
        self.badge.set_level(compat.level)
        self.badge.set_text(f"● {_BADGE_TEXT.get(compat.level, compat.label)}" if compat.level != "red"
                            else f"● {compat.label}")
        self.badge.setToolTip(compat.detail)


class ModelCardGrid(QWidget):
    """Cards laid out in as many columns as fit; asks for more when scrolled near the end."""

    model_selected = Signal(object)
    load_more_requested = Signal()

    def __init__(self, scroll_area: QScrollArea) -> None:
        super().__init__()
        self.scroll_area = scroll_area
        self._cards: list[ModelCard] = []
        self._columns = 0
        self._suspended = False
        self._compat: Callable[[Any], Compatibility | None] = lambda _m: None
        self._downloaded: set[str] = set()
        self._has_more = False
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(SPACING)
        self.grid.setVerticalSpacing(SPACING)
        self.grid.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.grid)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.spinner = Spinner(18)
        footer.addWidget(self.spinner)
        self.footer_text = label("", "faint")
        footer.addWidget(self.footer_text)
        footer.addStretch(1)
        root.addLayout(footer)
        root.addStretch(1)

        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.setInterval(60)
        self._reflow_timer.timeout.connect(self._reflow)
        scroll_area.verticalScrollBar().valueChanged.connect(self._check_near_end)
        scroll_area.verticalScrollBar().rangeChanged.connect(lambda *_: self._check_near_end())

    # ----------------------------------------------------------------- API
    @property
    def cards(self) -> list[ModelCard]:
        return list(self._cards)

    @property
    def columns(self) -> int:
        return self._columns

    def set_compat_function(self, fn: Callable[[Any], Compatibility | None]) -> None:
        self._compat = fn

    def set_downloaded(self, repo_ids: set[str]) -> None:
        self._downloaded = set(repo_ids)

    def clear(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._columns = 0
        self.footer_text.setText("")

    def set_models(self, models: list[Any]) -> None:
        self.clear()
        self.append_models(models)

    def append_models(self, models: list[Any]) -> None:
        for model in models:
            card = ModelCard(model, self._compat(model), model.id in self._downloaded)
            card.selected.connect(self.model_selected)
            self._cards.append(card)
        self._reflow(force=True)

    def refresh_badges(self) -> None:
        for card in self._cards:
            card.set_compat(self._compat(card.model))

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
        if loading:
            self.spinner.start()
            self.footer_text.setText("Loading models…" if not self._cards else "Loading more…")
        else:
            self.spinner.stop()
            self._update_footer()

    def set_has_more(self, has_more: bool) -> None:
        self._has_more = has_more
        self._update_footer()

    def set_message(self, text: str) -> None:
        self.footer_text.setText(text)

    def suspend_reflow(self, suspended: bool) -> None:
        self._suspended = suspended
        if not suspended:
            self._reflow(force=True)

    # ------------------------------------------------------------ layout
    def _update_footer(self) -> None:
        if self._loading:
            return
        if not self._cards:
            self.footer_text.setText("No models match these filters.")
        elif not self._has_more:
            self.footer_text.setText(f"{len(self._cards)} models · end of results")
        else:
            self.footer_text.setText(f"{len(self._cards)} models")

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._suspended:
            self._reflow_timer.start()

    def column_count_for(self, width: int) -> int:
        return max(1, (width + SPACING) // (CARD_MIN_WIDTH + SPACING))

    def _reflow(self, force: bool = False) -> None:
        columns = self.column_count_for(self.width() or self.scroll_area.viewport().width())
        if not force and columns == self._columns:
            return
        self._columns = columns
        while self.grid.count():
            self.grid.takeAt(0)
        for index, card in enumerate(self._cards):
            self.grid.addWidget(card, index // columns, index % columns)
        for column in range(max(columns, self.grid.columnCount())):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)

    def _check_near_end(self, *_args: Any) -> None:
        bar = self.scroll_area.verticalScrollBar()
        if self._has_more and not self._loading and self._cards and bar.value() >= bar.maximum() - 500:
            self.load_more_requested.emit()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(900, 600)
