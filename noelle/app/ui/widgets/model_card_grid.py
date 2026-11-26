"""Responsive model card grid with hardware compatibility badges (task T-203)."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from noelle.app import icons, theme
from noelle.app.ui.widgets.common import Avatar, Card, ElidedLabel, Pill, Spinner, human_count, label, time_ago
from noelle.runtime import runnability
from noelle.runtime.preflight import Compatibility, format_params

CARD_MIN_WIDTH = 286
CARD_HEIGHT = 140
SPACING = 12
_BADGE_TEXT = {"green": "Runs natively", "yellow": "Needs 4-bit", "red": "CPU fallback", "unknown": "Unknown size"}


def _verdict_badge(verdict: runnability.RunnabilityResult) -> str:
    if verdict.verdict == runnability.UNSUPPORTED_LIBRARY:
        return f"needs {verdict.library}"[:24] if verdict.library else "not supported"
    if verdict.verdict == runnability.NEEDS_NEWER_RUNTIME:
        return "needs newer runtime"
    if verdict.verdict == runnability.GATED_NO_ACCESS:
        return "gated"
    return verdict.label[:24]


class ModelCard(Card):
    selected = Signal(object)

    def __init__(self, model: Any, compat: Compatibility | None, downloaded: bool,
                 verdict: runnability.RunnabilityResult | None = None) -> None:
        super().__init__(clickable=True)
        self.model = model
        self.verdict = verdict
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
        elif self.verdict is not None and not self.verdict.supported:
            pills.addWidget(Pill(_verdict_badge(self.verdict), "red", tooltip=self.verdict.reason))
        elif model.license:
            pills.addWidget(Pill(str(model.license)[:16], "unknown"))
        if self.verdict is not None and self.verdict.verdict == runnability.NEEDS_COMPONENT:
            pills.addWidget(Pill(f"needs {self.verdict.engine or 'component'}", "yellow",
                                 tooltip=self.verdict.reason))
        if bool(getattr(model, "gated", False)):
            pills.addWidget(Pill("gated", "yellow",
                                 tooltip="Accept the licence on huggingface.co and add your token"))
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

    def set_runnability(self, verdict: runnability.RunnabilityResult | None) -> None:
        self.verdict = verdict
        if verdict is not None and not verdict.supported:
            self.badge.setToolTip(verdict.reason or verdict.label)

    def set_compat(self, compat: Compatibility | None) -> None:
        if compat is None:
            self.badge.hide()
            return
        self.badge.show()
        self.badge.set_level(compat.level)
        verdict = _BADGE_TEXT.get(compat.level, compat.label) if compat.level != "red" else compat.label
        score = f"{compat.score} · " if compat.score is not None else ""
        self.badge.set_text(f"● {score}{verdict}")
        tip = compat.detail
        if compat.score is not None:
            tip = f"Compatibility {compat.score}/100 ({compat.grade}) on this machine. {tip} Open the model for the " \
                  "full breakdown."
        self.badge.setToolTip(tip)


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
        self._runnability: Callable[[Any], runnability.RunnabilityResult | None] = lambda _m: None
        self._downloaded: set[str] = set()
        self._has_more = False
        self._loading = False
        self._spec_filter: str = "all"
        self._runs_filter: bool = True

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

    @property
    def spec_filter(self) -> str:
        return self._spec_filter

    def set_spec_filter(self, filter_key: str) -> None:
        if self._spec_filter != filter_key:
            self._spec_filter = filter_key
            self._reflow(force=True)
            self._update_footer()

    def _card_matches_spec(self, card: ModelCard) -> bool:
        if self._runs_filter and card.verdict is not None and not card.verdict.supported:
            return False
        if self._spec_filter == "all":
            return True
        compat = self._compat(card.model)
        if compat is None:
            return False
        if self._spec_filter == "supported":
            return compat.level in ("green", "yellow")
        if self._spec_filter == "green":
            return compat.level == "green"
        if self._spec_filter == "yellow":
            return compat.level == "yellow"
        if self._spec_filter == "red":
            return compat.level == "red"
        return True

    def set_compat_function(self, fn: Callable[[Any], Compatibility | None]) -> None:
        self._compat = fn

    def set_runnability_function(self, fn: Callable[[Any], runnability.RunnabilityResult | None]) -> None:
        self._runnability = fn

    def set_runs_filter(self, enabled: bool) -> None:
        if self._runs_filter != enabled:
            self._runs_filter = enabled
            self._reflow(force=True)
            self._update_footer()

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
            verdict = self._runnability(model)
            card = ModelCard(model, self._compat(model), model.id in self._downloaded, verdict)
            card.selected.connect(self.model_selected)
            self._cards.append(card)
        self._reflow(force=True)

    def refresh_badges(self) -> None:
        for card in self._cards:
            card.set_compat(self._compat(card.model))
        if self._spec_filter != "all" or self._runs_filter:
            self._reflow(force=True)
            self._update_footer()

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
        visible = [c for c in self._cards if self._card_matches_spec(c)]
        if not visible:
            self.footer_text.setText("No models match these filters.")
        elif not self._has_more:
            count_str = f"{len(visible)} models" if len(visible) != len(self._cards) else f"{len(self._cards)} models"
            self.footer_text.setText(f"{count_str} · end of results")
        else:
            count_str = f"{len(visible)} models" if len(visible) != len(self._cards) else f"{len(self._cards)} models"
            self.footer_text.setText(count_str)

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
        visible_cards = [c for c in self._cards if self._card_matches_spec(c)]
        for index, card in enumerate(visible_cards):
            self.grid.addWidget(card, index // columns, index % columns)
            card.show()
        for card in self._cards:
            if card not in visible_cards:
                card.hide()
        for column in range(max(columns, self.grid.columnCount())):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)

    def _check_near_end(self, *_args: Any) -> None:
        bar = self.scroll_area.verticalScrollBar()
        if self._has_more and not self._loading and self._cards and bar.value() >= bar.maximum() - 500:
            self.load_more_requested.emit()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(900, 600)
