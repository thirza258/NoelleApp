"""Design tokens and the application style sheet.

Two palettes: **light** (white) and **dark** (neutral grey). ``ui.theme`` in the
settings picks one, or ``"system"`` to follow the operating system's appearance.
The palette is chosen once at start-up, before any widget exists, because icons
and a few inline styles are rendered with the colours of the active palette.
"""

from __future__ import annotations

import sys
from typing import Any

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

THEMES = {"system": "Match system", "light": "Light", "dark": "Dark"}

LIGHT: dict[str, str] = {
    "BG": "#FFFFFF", "BG_ELEV": "#F7F7F8", "SURFACE": "#FFFFFF", "SURFACE_2": "#F7F7F8", "SURFACE_3": "#F1F1F3",
    "SURFACE_4": "#E6E6E9", "BORDER": "#E4E4E7", "BORDER_STRONG": "#D0D0D5",
    "TEXT": "#1C1C1E", "TEXT_2": "#3C3C43", "TEXT_MUTED": "#6B6B73", "TEXT_FAINT": "#8E8E96",
    "ACCENT": "#FFC700", "ACCENT_HOVER": "#FFD026", "ACCENT_PRESSED": "#E5B300",
    "ACCENT_SOFT": "rgba(255, 199, 0, 0.16)", "ON_ACCENT": "#0B0B0D",
    "SUCCESS": "#16A061", "SUCCESS_SOFT": "rgba(22, 160, 97, 0.12)", "SUCCESS_BORDER": "rgba(22, 160, 97, 0.35)",
    "WARNING": "#B7791F", "WARNING_SOFT": "rgba(214, 158, 46, 0.15)", "WARNING_BORDER": "rgba(183, 121, 31, 0.35)",
    "DANGER": "#D93D38", "DANGER_SOFT": "rgba(217, 61, 56, 0.10)", "DANGER_BORDER": "rgba(217, 61, 56, 0.35)",
    "INFO": "#2D7FD3", "INFO_SOFT": "rgba(45, 127, 211, 0.10)", "INFO_BORDER": "rgba(45, 127, 211, 0.35)",
    "UNKNOWN_SOFT": "rgba(107, 107, 115, 0.11)", "DANGER_TEXT": "#B42318",
    "CARD_HOVER": "#F1F1F4", "PRESSED": "#DCDCE0", "SCROLL_HOVER": "#BDBDC4", "SLIDER_HANDLE": "#FFFFFF",
    "PRIMARY_DISABLED_BG": "#FDF2CC", "PRIMARY_DISABLED_TEXT": "#A69566",
    "CODE_BG": "#F6F7F9", "CODE_TEXT": "#24292F",
    "HERO_START": "#FFF9E6", "HERO_MID": "#FFFDF7", "HERO_END": "#FFFFFF", "HERO_BORDER": "#FFEAA3",
    "SYNTAX_KEYWORD": "#8250DF", "SYNTAX_NUMBER": "#B35900", "SYNTAX_STRING": "#0A7F45", "SYNTAX_COMMENT": "#8C959F",
    "SYNTAX_FUNCTION": "#0550AE",
}

DARK: dict[str, str] = {
    "BG": "#1E1E1E", "BG_ELEV": "#252525", "SURFACE": "#232323", "SURFACE_2": "#2A2A2A", "SURFACE_3": "#323232",
    "SURFACE_4": "#3D3D3D", "BORDER": "#353535", "BORDER_STRONG": "#474747",
    "TEXT": "#ECECEC", "TEXT_2": "#C8C8C8", "TEXT_MUTED": "#9D9D9D", "TEXT_FAINT": "#707070",
    "ACCENT": "#FFC700", "ACCENT_HOVER": "#FFD53D", "ACCENT_PRESSED": "#E5B300",
    "ACCENT_SOFT": "rgba(255, 199, 0, 0.20)", "ON_ACCENT": "#0B0B0D",
    "SUCCESS": "#3DBE84", "SUCCESS_SOFT": "rgba(61, 190, 132, 0.16)", "SUCCESS_BORDER": "rgba(61, 190, 132, 0.35)",
    "WARNING": "#E9B949", "WARNING_SOFT": "rgba(233, 185, 73, 0.16)", "WARNING_BORDER": "rgba(233, 185, 73, 0.35)",
    "DANGER": "#F0645F", "DANGER_SOFT": "rgba(240, 100, 95, 0.16)", "DANGER_BORDER": "rgba(240, 100, 95, 0.40)",
    "INFO": "#5AA9F8", "INFO_SOFT": "rgba(90, 169, 248, 0.16)", "INFO_BORDER": "rgba(90, 169, 248, 0.35)",
    "UNKNOWN_SOFT": "rgba(157, 157, 157, 0.16)", "DANGER_TEXT": "#FFB3B0",
    "CARD_HOVER": "#303030", "PRESSED": "#454545", "SCROLL_HOVER": "#555555", "SLIDER_HANDLE": "#ECECEC",
    "PRIMARY_DISABLED_BG": "#4D4012", "PRIMARY_DISABLED_TEXT": "#8C7D4A",
    "CODE_BG": "#181818", "CODE_TEXT": "#D9D9D9",
    "HERO_START": "#383010", "HERO_MID": "#28251E", "HERO_END": "#222222", "HERO_BORDER": "#544818",
    "SYNTAX_KEYWORD": "#C792EA", "SYNTAX_NUMBER": "#F78C6C", "SYNTAX_STRING": "#C3E88D", "SYNTAX_COMMENT": "#6A737D",
    "SYNTAX_FUNCTION": "#82AAFF",
}

PALETTES = {"light": LIGHT, "dark": DARK}

# ---------------------------------------------------------------- tokens (replaced by use())
NAME = "dark"
IS_DARK = True
BG = BG_ELEV = SURFACE = SURFACE_2 = SURFACE_3 = SURFACE_4 = BORDER = BORDER_STRONG = ""
TEXT = TEXT_2 = TEXT_MUTED = TEXT_FAINT = ""
ACCENT = ACCENT_HOVER = ACCENT_PRESSED = ACCENT_SOFT = ON_ACCENT = ""
SUCCESS = SUCCESS_SOFT = SUCCESS_BORDER = WARNING = WARNING_SOFT = WARNING_BORDER = ""
DANGER = DANGER_SOFT = DANGER_BORDER = INFO = INFO_SOFT = INFO_BORDER = UNKNOWN_SOFT = DANGER_TEXT = ""
CARD_HOVER = PRESSED = SCROLL_HOVER = SLIDER_HANDLE = PRIMARY_DISABLED_BG = PRIMARY_DISABLED_TEXT = ""
CODE_BG = CODE_TEXT = HERO_START = HERO_MID = HERO_END = HERO_BORDER = ""
SYNTAX_KEYWORD = SYNTAX_NUMBER = SYNTAX_STRING = SYNTAX_COMMENT = SYNTAX_FUNCTION = ""
LEVEL_COLORS: dict[str, tuple[str, str]] = {}

RADIUS = 10
RADIUS_SM = 7
MONO = '"SF Mono", "JetBrains Mono", Menlo, Consolas, monospace'

# Distinct, readable avatar hues for organisations.
AVATAR_COLORS = ["#8B7CF6", "#4FA3F7", "#3DBE84", "#E9B949", "#F0645F", "#E07BD0", "#5CC8C8", "#F29B4B"]


def use(name: str) -> str:
    """Make ``name`` ("light" or "dark") the active palette. Returns the name actually used."""
    name = name if name in PALETTES else "dark"
    tokens: dict[str, Any] = dict(PALETTES[name])
    tokens["NAME"] = name
    tokens["IS_DARK"] = name == "dark"
    tokens["LEVEL_COLORS"] = {
        "green": (tokens["SUCCESS"], tokens["SUCCESS_SOFT"]),
        "yellow": (tokens["WARNING"], tokens["WARNING_SOFT"]),
        "red": (tokens["DANGER"], tokens["DANGER_SOFT"]),
        "unknown": (tokens["TEXT_MUTED"], tokens["UNKNOWN_SOFT"]),
        "info": (tokens["INFO"], tokens["INFO_SOFT"]),
        "accent": (tokens["ACCENT"], tokens["ACCENT_SOFT"]),
    }
    globals().update(tokens)
    clear = getattr(sys.modules.get("noelle.app.icons"), "clear_cache", None)
    if callable(clear):  # icons are cached per colour; drop renders made with a previous palette
        clear()
    return name


def resolve(preference: str | None, app: QApplication | None = None) -> str:
    """Map a ``ui.theme`` setting to "light" or "dark" ("system" asks the OS, light when it cannot tell)."""
    choice = (preference or "system").lower()
    if choice in PALETTES:
        return choice
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        hints = (app or QGuiApplication.instance()).styleHints()  # type: ignore[union-attr]
        if hints.colorScheme() == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:
        pass
    return "light"


use("dark")


def avatar_color(name: str) -> str:
    total = sum(ord(c) * (i + 1) for i, c in enumerate(name or "?"))
    return AVATAR_COLORS[total % len(AVATAR_COLORS)]


def stylesheet() -> str:
    return f"""
* {{ outline: none; }}
QWidget {{ background: transparent; color: {TEXT}; font-size: 13px; }}
QMainWindow, QDialog, #AppRoot {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 6px 8px;
            border-radius: 6px; }}

/* ---- text roles ---- */
QLabel[role="h1"] {{ font-size: 24px; font-weight: 650; color: {TEXT}; }}
QLabel[role="h2"] {{ font-size: 18px; font-weight: 600; color: {TEXT}; }}
QLabel[role="h3"] {{ font-size: 14px; font-weight: 600; color: {TEXT}; }}
QLabel[role="eyebrow"] {{ font-size: 11px; font-weight: 700; color: {TEXT_MUTED}; letter-spacing: 1px; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; }}
QLabel[role="faint"] {{ color: {TEXT_FAINT}; font-size: 12px; }}
QLabel[role="mono"] {{ font-family: {MONO}; font-size: 12px; color: {TEXT_2}; }}
QLabel[role="metric"] {{ font-size: 20px; font-weight: 650; }}
QLabel[role="error"] {{ color: {DANGER}; }}
QLabel[role="link"] {{ color: {ACCENT}; }}

/* ---- surfaces ---- */
#TopBar {{ background: {BG_ELEV}; border-bottom: 1px solid {BORDER}; }}
#StatusBar {{ background: {BG_ELEV}; border-top: 1px solid {BORDER}; }}
#Sidebar {{ background: {BG_ELEV}; border-right: 1px solid {BORDER}; }}
QFrame[card="true"] {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}
QFrame[card="true"][hover="true"] {{ border-color: {BORDER_STRONG}; background: {CARD_HOVER}; }}
QFrame[panel="true"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}
QFrame[hero="true"] {{ background: {HERO_START}; border: 1px solid {HERO_BORDER}; border-radius: 14px; }}
QFrame[divider="true"] {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame[banner="warning"] {{ background: {WARNING_SOFT}; border: 1px solid {WARNING_BORDER}; border-radius: 8px; }}
QFrame[banner="danger"] {{ background: {DANGER_SOFT}; border: 1px solid {DANGER_BORDER}; border-radius: 8px; }}
QFrame[banner="info"] {{ background: {INFO_SOFT}; border: 1px solid {INFO_BORDER}; border-radius: 8px; }}
QFrame[banner="success"] {{ background: {SUCCESS_SOFT}; border: 1px solid {SUCCESS_BORDER}; border-radius: 8px; }}

/* ---- buttons ---- */
QPushButton, QToolButton {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_SM}px; padding: 7px 14px; font-weight: 500; }}
QPushButton:hover, QToolButton:hover {{ background: {SURFACE_4}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {PRESSED}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE_2}; border-color: {BORDER}; }}
QPushButton[variant="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: {ON_ACCENT}; font-weight: 600; }}
QPushButton[variant="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[variant="primary"]:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton[variant="primary"]:disabled {{ background: {PRIMARY_DISABLED_BG}; border-color: {PRIMARY_DISABLED_BG};
    color: {PRIMARY_DISABLED_TEXT}; }}
QPushButton[variant="danger"] {{ background: {DANGER_SOFT}; border-color: {DANGER_BORDER}; color: {DANGER_TEXT}; }}
QPushButton[variant="danger"]:hover {{ border-color: {DANGER}; }}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{ background: transparent; border-color: transparent;
    color: {TEXT_2}; }}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{ background: {SURFACE_3}; color: {TEXT}; }}
QPushButton[variant="ghost"]:checked, QToolButton[variant="ghost"]:checked {{ background: {ACCENT_SOFT};
    color: {TEXT}; }}
QPushButton[variant="ghost"]:disabled, QToolButton[variant="ghost"]:disabled {{ background: transparent;
    border-color: transparent; color: {TEXT_FAINT}; }}
QPushButton[variant="nav"] {{ background: transparent; border: none; border-radius: 8px; padding: 7px 12px;
    color: {TEXT_MUTED}; font-weight: 550; text-align: left; }}
QPushButton[variant="nav"]:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
QPushButton[variant="nav"]:checked {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QPushButton[variant="segment"] {{ background: transparent; border: 1px solid transparent; border-radius: 6px;
    padding: 5px 10px; color: {TEXT_MUTED}; }}
QPushButton[variant="segment"]:checked {{ background: {SURFACE_4}; color: {TEXT}; border-color: {BORDER_STRONG}; }}
QPushButton[variant="chip"] {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 5px 12px; color: {TEXT_2}; }}
QPushButton[variant="chip"]:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {TEXT}; }}
QPushButton[variant="link"] {{ background: transparent; border: none; padding: 0; color: {ACCENT}; text-align: left; }}
QPushButton[variant="link"]:hover {{ color: {ACCENT_HOVER}; text-decoration: underline; }}

/* ---- inputs ---- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE_3}; border: 1px solid {BORDER_STRONG}; border-radius: {RADIUS_SM}px; padding: 6px 9px;
    selection-background-color: {ACCENT}; selection-color: {ON_ACCENT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {TEXT_FAINT}; }}
QLineEdit[search="true"] {{ padding-left: 30px; border-radius: 9px; background: {SURFACE_2}; }}
QPlainTextEdit[code="true"], QTextEdit[code="true"] {{ font-family: {MONO}; font-size: 12px; background: {CODE_BG};
    color: {CODE_TEXT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER_STRONG}; selection-background-color:
    {ACCENT_SOFT}; selection-color: {TEXT}; padding: 4px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px;
    border: none; }}
QCheckBox {{ spacing: 8px; color: {TEXT_2}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {BORDER_STRONG};
    background: {SURFACE_3}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QRadioButton {{ spacing: 8px; color: {TEXT_2}; }}

QSlider::groove:horizontal {{ height: 4px; background: {SURFACE_4}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {SLIDER_HANDLE}; border: 1px solid {BORDER_STRONG}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ border-color: {ACCENT}; }}

QProgressBar {{ background: {SURFACE_4}; border: none; border-radius: 3px; height: 6px; text-align: center;
    color: transparent; max-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QProgressBar[level="green"]::chunk {{ background: {SUCCESS}; }}
QProgressBar[level="yellow"]::chunk {{ background: {WARNING}; }}
QProgressBar[level="red"]::chunk {{ background: {DANGER}; }}

/* ---- tabs ---- */
QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {TEXT_MUTED}; padding: 9px 14px; margin-right: 2px;
    border-bottom: 2px solid transparent; font-weight: 550; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}

/* ---- lists & tables ---- */
QListWidget, QTreeWidget, QTableWidget, QTableView, QTreeView {{ background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}px; alternate-background-color: {SURFACE_2}; gridline-color: {BORDER};
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; }}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {{ background: {ACCENT_SOFT};
    color: {TEXT}; }}
QListWidget::item, QTreeWidget::item {{ padding: 6px 6px; border-radius: 5px; }}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {ACCENT_SOFT};
    color: {TEXT}; }}
QHeaderView::section {{ background: {SURFACE_2}; color: {TEXT_MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 7px 8px; font-weight: 600; font-size: 12px; }}
QTableCornerButton::section {{ background: {SURFACE_2}; border: none; }}

/* ---- scrollbars ---- */
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_4}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {SCROLL_HOVER}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {SURFACE_4}; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER_STRONG}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: {RADIUS}px; margin-top: 14px; padding: 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT_MUTED}; }}
"""


def _indicator_images() -> str:
    """Render arrow/check images for combo boxes, spin boxes and check boxes.

    Qt style sheets only accept file URLs, so the PNGs are written once per run into the
    Noelle runtime folder. Returns extra QSS (empty when rendering is impossible)."""
    from noelle import paths
    from noelle.app import icons

    folder = paths.runtime_dir() / "qss"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        images = {}
        for name, icon_name, color, size in (("down", "chevron-down", TEXT_MUTED, 12), ("up", "chevron-up", TEXT_MUTED, 10),
                                             ("down-small", "chevron-down", TEXT_MUTED, 10),
                                             ("check", "check", ON_ACCENT, 12)):
            target = folder / f"{name}-{NAME}.png"
            icons.pixmap(icon_name, color, size).save(str(target), "PNG")
            images[name] = target.as_posix()
    except Exception:
        return ""
    return f"""
QComboBox::down-arrow {{ image: url({images['down']}); width: 12px; height: 12px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({images['up']}); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({images['down-small']}); width: 9px; height: 9px; }}
QCheckBox::indicator:checked {{ image: url({images['check']}); }}
"""


def apply(app: QApplication, preference: str | None = None) -> str:
    """Choose the palette (``preference``, else the ``ui.theme`` setting) and style the application.

    Returns the palette used ("light" or "dark"). Call it before creating windows.
    """
    if preference is None:
        try:
            from noelle.config import get_settings

            preference = get_settings().get("ui.theme", "system")
        except Exception:
            preference = "system"
    name = use(resolve(preference, app))
    try:  # native title bars and dialogs follow the app's palette instead of the OS (Qt 6.8+)
        from PySide6.QtCore import Qt

        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if IS_DARK else Qt.ColorScheme.Light)
    except Exception:
        pass
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE_3))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_2))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_3))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(ON_ACCENT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_3))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Light, QColor(SURFACE_4 if IS_DARK else "#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(SURFACE_3))
    palette.setColor(QPalette.ColorRole.Mid, QColor(BORDER_STRONG))
    palette.setColor(QPalette.ColorRole.Dark, QColor(BG if IS_DARK else "#A0A0A5"))
    palette.setColor(QPalette.ColorRole.Shadow, QColor("#000000" if IS_DARK else "#7A7A80"))
    app.setPalette(palette)
    font = QFont(app.font())
    font.setPointSizeF(max(font.pointSizeF(), 13.0) if font.pointSizeF() > 0 else 13.0)
    app.setFont(font)
    app.setStyleSheet(stylesheet() + _indicator_images())
    return name


def set_prop(widget: QWidget, name: str, value: object) -> None:
    """Set a dynamic property used by the style sheet and re-polish the widget."""
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
