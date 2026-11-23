"""Page 1: model discovery with a trending hero and a hardware-aware catalog grid."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from noelle.app import icons, theme
from noelle.app.context import AppContext, summary_from_record
from noelle.app.services.task_runner import runner
from noelle.app.ui.animations.scroll_transitions import HeroCollapseController
from noelle.app.ui.widgets.common import Banner, SearchField, button, icon_button, label
from noelle.app.ui.widgets.hero_card import HERO_WIDTH, HeroCard
from noelle.app.ui.widgets.model_card_grid import ModelCardGrid
from noelle.runtime import tasks

CATEGORIES: list[tuple[str, str, str, str | None]] = [
    ("trending", "Trending", "flame", None),
    ("text", "Text & Language", "type", "text"),
    ("vision", "Vision & Multimodal", "eye", "vision"),
    ("audio", "Audio & Speech", "wave", "audio"),
    ("diffusion", "Diffusion & Synthesis", "brush", "diffusion"),
    ("embeddings", "Embeddings", "nodes", "embeddings"),
    ("library", "Downloaded", "box", None),
]
SORT_OPTIONS = [("Trending", "trending"), ("Most downloads", "downloads"), ("Most likes", "likes"),
                ("Recently updated", "recent")]


class DiscoveryPage(QWidget):
    model_selected = Signal(object)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._category = "trending"
        self._pager: Any = None
        self._generation = 0
        self._hero_generation = 0

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ------------------------------------------------ modality sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(212)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 18, 12, 14)
        side.setSpacing(3)
        side.addWidget(label("BROWSE", "eyebrow"))
        side.addSpacing(6)
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: dict[str, Any] = {}
        for key, text, icon_name, _modality in CATEGORIES:
            if key == "library":
                side.addSpacing(10)
                side.addWidget(label("YOUR MACHINE", "eyebrow"))
                side.addSpacing(6)
            btn = button(text, icon_name, "nav", checkable=True)
            btn.setIcon(icons.icon(icon_name, theme.TEXT_MUTED, 16, active_color=theme.ACCENT))
            btn.clicked.connect(lambda _=False, k=key: self.set_category(k))
            self._nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            side.addWidget(btn)
        side.addStretch(1)
        self.hw_hint = label("", "faint", wrap=True)
        side.addWidget(self.hw_hint)
        root.addWidget(sidebar)

        # ------------------------------------------------ hero column
        self.hero_column = QWidget()
        hero_layout = QVBoxLayout(self.hero_column)
        hero_layout.setContentsMargins(20, 18, 0, 18)
        hero_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetNoConstraint)
        self.hero = HeroCard()
        self.hero.model_selected.connect(self.model_selected)
        self.hero.fetch_blurb = self._fetch_blurb
        hero_layout.addWidget(self.hero, 1, Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.hero_column)

        # ------------------------------------------------ catalog column
        main = QWidget()
        main.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        column = QVBoxLayout(main)
        column.setContentsMargins(20, 18, 20, 0)
        column.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title = label("Trending", "h2")
        self.subtitle = label("What the community is running this week", "muted")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        header.addLayout(titles, 1)
        self.search = SearchField("Search models on the Hub")
        self.search.setFixedWidth(300)
        header.addWidget(self.search)
        self.sort = QComboBox()
        for text, key in SORT_OPTIONS:
            self.sort.addItem(text, key)
        self.sort.setFixedWidth(160)
        header.addWidget(self.sort)
        self.refresh_btn = icon_button("refresh", "Refresh")
        header.addWidget(self.refresh_btn)
        column.addLayout(header)

        self.error_banner = Banner("danger", "")
        self.error_banner.add_action("Retry", self.reload)
        self.error_banner.hide()
        column.addWidget(self.error_banner)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.grid = ModelCardGrid(self.scroll)
        self.grid.set_compat_function(self.ctx.compat)
        self.grid.model_selected.connect(self.model_selected)
        self.grid.load_more_requested.connect(self.load_more)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 2, 6, 20)
        container_layout.addWidget(self.grid)
        self.scroll.setWidget(container)
        column.addWidget(self.scroll, 1)
        root.addWidget(main, 1)

        self.collapse = HeroCollapseController(
            self.hero_column, self.scroll, threshold=int(ctx.settings.get("ui.hero_collapse_threshold", 120)),
            width=HERO_WIDTH, duration_ms=int(ctx.settings.get("ui.hero_animation_ms", 250)), grid=self.grid)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(400)
        self._search_timer.timeout.connect(self.reload)
        self.search.textChanged.connect(lambda _t: self._search_timer.start())
        self.search.returnPressed.connect(self.reload)
        self.sort.currentIndexChanged.connect(lambda _i: self.reload())
        self.refresh_btn.clicked.connect(self.reload)

        ctx.hardware.hardware_telemetry_updated.connect(self._on_hardware)
        ctx.downloads.job_finished.connect(lambda *_: self._on_library_changed())
        self._nav_buttons["trending"].setChecked(True)
        self._hint_set = False

    # ---------------------------------------------------------------- state
    def start(self) -> None:
        self.reload()
        self.load_hero()

    def set_category(self, key: str) -> None:
        if key == self._category:
            return
        self._category = key
        self._nav_buttons[key].setChecked(True)
        title = next(text for k, text, *_ in CATEGORIES if k == key)
        self.title.setText(title)
        modality = tasks.MODALITY_BY_KEY.get(key)
        if key == "trending":
            self.subtitle.setText("What the community is running this week")
        elif key == "library":
            self.subtitle.setText("Models already on this machine — available offline")
        elif modality:
            self.subtitle.setText(modality.description)
        is_library = key == "library"
        self.search.setEnabled(not is_library)
        self.sort.setEnabled(not is_library)
        self.collapse.set_enabled(not is_library)
        self.scroll.verticalScrollBar().setValue(0)
        self.reload()
        if not is_library:
            self.load_hero()

    def reload(self) -> None:
        self._generation += 1
        generation = self._generation
        self.error_banner.hide()
        self.grid.set_downloaded(self.ctx.downloaded_ids())
        self.grid.clear()
        if self._category == "library":
            records = [r for r in self.ctx.registry.list_models() if r.exists_on_disk()]
            self.grid.set_models([summary_from_record(r) for r in records])
            self.grid.set_has_more(False)
            self.grid.set_loading(False)
            if not records:
                self.grid.set_message("Nothing downloaded yet. Pick a model and press Download.")
            return
        modality = None if self._category == "trending" else self._category
        self._pager = self.ctx.hf.pager(modality=modality, search=self.search.text(),
                                        sort=self.sort.currentData() or "trending",
                                        page_size=int(self.ctx.settings.get("hub.page_size", 30)))
        self.grid.set_has_more(True)
        self._fetch(generation)

    def load_more(self) -> None:
        if self._pager is None or self._pager.exhausted or self._category == "library":
            return
        self._fetch(self._generation)

    def _fetch(self, generation: int) -> None:
        self.grid.set_loading(True)
        pager = self._pager
        supported_only = self._category == "trending"

        def work() -> list[Any]:
            models = pager.fetch_next()
            if supported_only:
                models = [m for m in models if tasks.is_supported(m.pipeline_tag)]
            return models

        def done(models: list[Any]) -> None:
            if generation != self._generation:
                return
            self.grid.append_models(models)
            self.grid.set_has_more(not pager.exhausted)
            self.grid.set_loading(False)

        def failed(message: str) -> None:
            if generation != self._generation:
                return
            self.grid.set_loading(False)
            self.error_banner.set_text(f"<b>Could not reach the Hugging Face Hub.</b> {message}<br>"
                                       "Downloaded models are still available under <i>Downloaded</i>.")
            self.error_banner.show()

        runner().submit(work, on_result=done, on_error=failed)

    def load_hero(self) -> None:
        self._hero_generation += 1
        generation = self._hero_generation
        modality = tasks.MODALITY_BY_KEY.get(self._category)
        tag = modality.pipeline_tags[0] if modality else "text-generation"
        count = int(self.ctx.settings.get("hub.hero_count", 5))
        self.hero.set_loading(True)

        def done(models: list[Any]) -> None:
            if generation == self._hero_generation:
                self.hero.set_models(models, self.ctx.compat)

        def failed(message: str) -> None:
            if generation == self._hero_generation:
                self.hero.set_error(message)

        runner().submit(self.ctx.hf.trending, count, tag, on_result=done, on_error=failed)

    def _fetch_blurb(self, repo_id: str, callback: Any) -> None:
        from noelle.app.ui.widgets.hero_card import readme_blurb

        runner().submit(lambda: readme_blurb(self.ctx.hf.fetch_readme(repo_id)), on_result=callback)

    # --------------------------------------------------------------- events
    def _on_hardware(self, metrics: Any) -> None:
        budget = self.ctx.hardware.memory_budget()
        from noelle.runtime.preflight import format_bytes

        text = (f"Badges reflect {metrics.gpu.name} · {format_bytes(budget.usable_bytes)} usable "
                f"after {int(budget.headroom * 100)}% headroom")
        if text != self.hw_hint.text():
            changed = self._hint_set
            self.hw_hint.setText(text)
            self._hint_set = True
            if changed:
                self.grid.refresh_badges()
                self.hero.refresh_compat()

    def on_library_changed(self) -> None:
        self._on_library_changed()

    def _on_library_changed(self) -> None:
        self.grid.set_downloaded(self.ctx.downloaded_ids())
        if self._category == "library":
            self.reload()
