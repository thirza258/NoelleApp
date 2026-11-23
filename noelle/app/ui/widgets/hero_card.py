"""Left-side trending showcase on the discovery page (task T-202)."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from noelle.app import icons, theme
from noelle.app.ui.widgets.common import Avatar, FlowLayout, Pill, Spinner, button, human_count, icon_button, label, time_ago
from noelle.runtime.preflight import Compatibility, format_params

HERO_WIDTH = 380
_LEVEL_TITLE = {"green": "Runs natively on your hardware", "yellow": "Needs a 4-bit build",
                "red": "Too heavy for your accelerator", "unknown": "Size not published"}


class HeroCard(QFrame):
    model_selected = Signal(object)  # ModelSummary

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("hero", True)
        self.setFixedWidth(HERO_WIDTH - 20)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._models: list[Any] = []
        self._index = 0
        self._compat: Callable[[Any], Compatibility] | None = None
        self._blurbs: dict[str, str] = {}
        self.fetch_blurb: Callable[[str, Callable[[str], None]], None] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(12)

        top = QHBoxLayout()
        flame = QLabel()
        flame.setPixmap(icons.pixmap("flame", theme.WARNING, 15))
        top.addWidget(flame)
        self.eyebrow = label("TRENDING", "eyebrow")
        top.addWidget(self.eyebrow)
        top.addStretch(1)
        self.prev_btn = icon_button("chevron-left", "Previous", size=26, icon_size=14)
        self.next_btn = icon_button("chevron-right", "Next", size=26, icon_size=14)
        self.prev_btn.clicked.connect(lambda: self.show_index(self._index - 1))
        self.next_btn.clicked.connect(lambda: self.show_index(self._index + 1))
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        root.addLayout(top)

        ident = QHBoxLayout()
        ident.setSpacing(12)
        self.avatar = Avatar("", 52, radius=14)
        ident.addWidget(self.avatar)
        names = QVBoxLayout()
        names.setSpacing(2)
        self.author = label("", "muted")
        self.updated = label("", "faint")
        names.addWidget(self.author)
        names.addWidget(self.updated)
        ident.addLayout(names, 1)
        root.addLayout(ident)

        self.title = label("Loading trending models…", "h1", wrap=True)
        self.title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.title)

        self.badges_host = QWidget()
        self.badges = FlowLayout(self.badges_host, spacing=6)
        root.addWidget(self.badges_host)

        self.blurb = label("", "muted", wrap=True)
        self.blurb.setMaximumHeight(120)
        root.addWidget(self.blurb)

        stats = QHBoxLayout()
        stats.setSpacing(18)
        self.downloads = self._stat("arrow-down")
        self.likes = self._stat("heart")
        self.trend = self._stat("activity")
        for widget in (self.downloads, self.likes, self.trend):
            stats.addWidget(widget)
        stats.addStretch(1)
        root.addLayout(stats)

        self.compat_box = QFrame()
        compat_layout = QVBoxLayout(self.compat_box)
        compat_layout.setContentsMargins(12, 10, 12, 10)
        compat_layout.setSpacing(3)
        self.compat_title = label("", "h3")
        self.compat_detail = label("", "muted", wrap=True)
        compat_layout.addWidget(self.compat_title)
        compat_layout.addWidget(self.compat_detail)
        root.addWidget(self.compat_box)

        self.notice = label("", "faint", wrap=True)
        root.addWidget(self.notice)
        root.addStretch(1)

        actions = QHBoxLayout()
        self.try_btn = button("Try This Model", "play", "primary")
        self.try_btn.setMinimumHeight(38)
        self.try_btn.clicked.connect(self._emit_selected)
        actions.addWidget(self.try_btn, 1)
        self.hub_btn = icon_button("external", "Open on huggingface.co", size=38)
        self.hub_btn.clicked.connect(self._open_hub)
        actions.addWidget(self.hub_btn)
        root.addLayout(actions)

        dots_row = QHBoxLayout()
        dots_row.addStretch(1)
        self.dots = QLabel()
        dots_row.addWidget(self.dots)
        self.spinner = Spinner(16)
        dots_row.addWidget(self.spinner)
        dots_row.addStretch(1)
        root.addLayout(dots_row)

        self._rotate = QTimer(self)
        self._rotate.setInterval(9000)
        self._rotate.timeout.connect(lambda: self.show_index(self._index + 1))
        self.spinner.start()

    def _stat(self, icon_name: str) -> QWidget:
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        art = QLabel()
        art.setPixmap(icons.pixmap(icon_name, theme.TEXT_MUTED, 14))
        layout.addWidget(art)
        text = label("—", "muted")
        layout.addWidget(text)
        host.text = text  # type: ignore[attr-defined]
        return host

    # ----------------------------------------------------------------- API
    def set_loading(self, loading: bool) -> None:
        if loading:
            self.spinner.start()
        else:
            self.spinner.stop()

    def set_error(self, message: str) -> None:
        self.spinner.stop()
        self.title.setText("Trending is unavailable")
        self.notice.setText(message)
        self.try_btn.setEnabled(False)

    def set_models(self, models: list[Any], compat: Callable[[Any], Compatibility]) -> None:
        self._models = models
        self._compat = compat
        self.spinner.stop()
        self.try_btn.setEnabled(bool(models))
        if not models:
            self.title.setText("Nothing trending here yet")
            return
        self.show_index(0)
        if len(models) > 1:
            self._rotate.start()

    def refresh_compat(self) -> None:
        if self._models:
            self.show_index(self._index)

    def current(self) -> Any | None:
        return self._models[self._index] if self._models else None

    def show_index(self, index: int) -> None:
        if not self._models:
            return
        self._index = index % len(self._models)
        model = self._models[self._index]
        self.eyebrow.setText(f"TRENDING #{self._index + 1}")
        self.avatar.set_name(model.namespace)
        self.author.setText(model.namespace)
        self.updated.setText(f"Updated {time_ago(model.last_modified)}" if model.last_modified else "")
        self.title.setText(model.name)
        self.badges.clear()
        if model.pipeline_tag:
            self.badges.addWidget(Pill(model.pipeline_tag, "info"))
        if model.param_count:
            self.badges.addWidget(Pill(f"{format_params(model.param_count)} params", "accent"))
        if model.context_length:
            self.badges.addWidget(Pill(f"{model.context_length // 1024}K context", "info"))
        if model.license:
            self.badges.addWidget(Pill(str(model.license)[:24], "unknown"))
        if model.library_name:
            self.badges.addWidget(Pill(model.library_name, "unknown"))
        if model.gated:
            self.badges.addWidget(Pill("gated", "yellow", tooltip="Accept the license on the Hub and add a token"))
        if model.custom_code:
            self.badges.addWidget(Pill("custom code", "yellow", tooltip="Runs Python code from the repository"))
        self.downloads.text.setText(human_count(model.downloads))  # type: ignore[attr-defined]
        self.likes.text.setText(human_count(model.likes))  # type: ignore[attr-defined]
        self.trend.text.setText(f"{int(model.trending_score or 0)} trend")  # type: ignore[attr-defined]
        compat = self._compat(model) if self._compat else None
        if compat is not None:
            fg, bg = theme.LEVEL_COLORS.get(compat.level, theme.LEVEL_COLORS["unknown"])
            self.compat_box.setStyleSheet(f"QFrame {{ background: {bg}; border-radius: 9px; border: none; }}"
                                          f" QLabel {{ background: transparent; }}")
            self.compat_title.setText(f"<span style='color:{fg}'>●</span>  {_LEVEL_TITLE.get(compat.level, compat.label)}")
            self.compat_detail.setText(compat.detail)
        self.blurb.setText(self._blurbs.get(model.id, ""))
        if model.id not in self._blurbs and self.fetch_blurb is not None:
            self._blurbs[model.id] = ""
            self.fetch_blurb(model.id, lambda text, repo=model.id: self._set_blurb(repo, text))
        supported = model.family is not None
        self.notice.setText("" if supported else "This task has no playground in Noelle yet.")
        self.dots.setText("  ".join("●" if i == self._index else "○" for i in range(len(self._models))))
        self.dots.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 9px;")

    def _set_blurb(self, repo_id: str, text: str) -> None:
        self._blurbs[repo_id] = text
        current = self.current()
        if current is not None and current.id == repo_id:
            self.blurb.setText(text)

    def _emit_selected(self) -> None:
        model = self.current()
        if model is not None:
            self.model_selected.emit(model)

    def _open_hub(self) -> None:
        model = self.current()
        if model is not None:
            QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{model.id}"))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(HERO_WIDTH - 20, 560)


def readme_blurb(markdown: str, limit: int = 260) -> str:
    """First prose paragraph of a model card (front matter, headings, tables, HTML and images skipped)."""
    import re

    text = markdown
    if text.startswith("---"):
        end = text.find("\n---", 3)
        text = text[end + 4:] if end != -1 else text
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith(("#", "|", "<", "!", "```", ">", "-", "*", "[!", "+")):
            continue
        if re.match(r"^\d+[.)]\s", block):  # numbered list / table of contents
            continue
        plain = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", block)
        plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", plain)
        plain = re.sub(r"[*_`]{1,3}", "", plain)
        plain = re.sub(r"<[^>]+>", "", plain)
        plain = " ".join(plain.split())
        if len(plain) < 40:
            continue
        return plain if len(plain) <= limit else plain[: limit - 1].rsplit(" ", 1)[0] + "…"
    return ""
