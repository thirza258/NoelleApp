"""Top navigation bar."""

from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QWidget

from noelle.app import icons, theme
from noelle.app.ui.widgets.common import button, icon_button, label

PAGES = [
    ("discover", "Discover", "compass"),
    ("mission", "Mission", "sparkles"),
    ("workspace", "Workspace", "cpu"),
    ("knowledge", "Knowledge", "book"),
    ("evaluate", "Evaluate", "activity"),
    ("training", "Training", "wand"),
    ("workflow", "Workflow", "layers"),
    ("compare", "Compare", "columns"),
    ("storage", "Storage", "database"),
    ("server", "API", "server"),
]


class NavigationBar(QFrame):
    page_requested = Signal(str)
    settings_requested = Signal()
    downloads_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(54)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(6)

        logo = QLabel()
        logo.setPixmap(icons.app_icon().pixmap(QSize(26, 26)))
        layout.addWidget(logo)
        brand = label("Noelle", "h3")
        brand.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(brand)
        studio = label("AI Studio", "faint")
        layout.addWidget(studio)
        layout.addSpacing(22)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons = {}
        for key, text, icon_name in PAGES:
            btn = button(text, icon_name, "nav", checkable=True)
            btn.setIcon(icons.icon(icon_name, theme.TEXT_MUTED, 16, active_color=theme.ACCENT))
            btn.clicked.connect(lambda _=False, k=key: self.page_requested.emit(k))
            self._group.addButton(btn)
            self._buttons[key] = btn
            layout.addWidget(btn)
        layout.addStretch(1)

        self.downloads_btn = button("", "download", "ghost", tooltip="Downloads")
        self.downloads_btn.clicked.connect(self.downloads_requested)
        self.downloads_btn.hide()
        layout.addWidget(self.downloads_btn)
        self.account = label("", "faint")
        layout.addWidget(self.account)
        self.settings_btn = icon_button("sliders", "Settings")
        self.settings_btn.clicked.connect(self.settings_requested)
        layout.addWidget(self.settings_btn)

    def set_current(self, key: str) -> None:
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

    def set_active_downloads(self, count: int, fraction: float | None = None) -> None:
        if count <= 0:
            self.downloads_btn.hide()
            return
        pct = f" · {fraction * 100:.0f}%" if fraction is not None else ""
        self.downloads_btn.setText(f"{count} downloading{pct}")
        self.downloads_btn.show()

    def set_account(self, name: str | None) -> None:
        if name:
            self.account.setText(f"Signed in as {name}")
            self.account.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        else:
            self.account.setText("")

