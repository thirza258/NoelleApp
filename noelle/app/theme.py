"""Design tokens and the application style sheet (dark theme)."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

# ---------------------------------------------------------------- tokens
BG = "#0D1015"
BG_ELEV = "#11151C"
SURFACE = "#151A22"
SURFACE_2 = "#1A2029"
SURFACE_3 = "#222935"
SURFACE_4 = "#2B3341"
BORDER = "#252D39"
BORDER_STRONG = "#323C4B"
TEXT = "#E8ECF2"
TEXT_2 = "#B6BFCB"
TEXT_MUTED = "#8B96A6"
TEXT_FAINT = "#5E6A7A"
ACCENT = "#8B7CF6"
ACCENT_HOVER = "#9C8FF8"
ACCENT_PRESSED = "#7867EC"
ACCENT_SOFT = "rgba(139, 124, 246, 0.16)"
SUCCESS = "#3DBE84"
SUCCESS_SOFT = "rgba(61, 190, 132, 0.15)"
WARNING = "#E9B949"
WARNING_SOFT = "rgba(233, 185, 73, 0.15)"
DANGER = "#F0645F"
DANGER_SOFT = "rgba(240, 100, 95, 0.15)"
INFO = "#4FA3F7"
INFO_SOFT = "rgba(79, 163, 247, 0.15)"

RADIUS = 10
RADIUS_SM = 7
MONO = '"SF Mono", "JetBrains Mono", Menlo, Consolas, monospace'

LEVEL_COLORS = {
    "green": (SUCCESS, SUCCESS_SOFT),
    "yellow": (WARNING, WARNING_SOFT),
    "red": (DANGER, DANGER_SOFT),
    "unknown": (TEXT_MUTED, "rgba(139, 150, 166, 0.14)"),
    "info": (INFO, INFO_SOFT),
    "accent": (ACCENT, ACCENT_SOFT),
}

# Distinct, readable avatar hues for organisations.
AVATAR_COLORS = ["#8B7CF6", "#4FA3F7", "#3DBE84", "#E9B949", "#F0645F", "#E07BD0", "#5CC8C8", "#F29B4B"]


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
QFrame[card="true"][hover="true"] {{ border-color: {BORDER_STRONG}; background: #1D2430; }}
QFrame[panel="true"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}
QFrame[hero="true"] {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #231D45, stop:0.55 #171B2A,
                       stop:1 #121722); border: 1px solid #2F2A57; border-radius: 14px; }}
QFrame[divider="true"] {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame[banner="warning"] {{ background: {WARNING_SOFT}; border: 1px solid rgba(233,185,73,0.35); border-radius: 8px; }}
QFrame[banner="danger"] {{ background: {DANGER_SOFT}; border: 1px solid rgba(240,100,95,0.35); border-radius: 8px; }}
QFrame[banner="info"] {{ background: {INFO_SOFT}; border: 1px solid rgba(79,163,247,0.35); border-radius: 8px; }}
QFrame[banner="success"] {{ background: {SUCCESS_SOFT}; border: 1px solid rgba(61,190,132,0.35); border-radius: 8px; }}

/* ---- buttons ---- */
QPushButton, QToolButton {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_SM}px; padding: 7px 14px; font-weight: 500; }}
QPushButton:hover, QToolButton:hover {{ background: {SURFACE_4}; }}
QPushButton:pressed, QToolButton:pressed {{ background: #333D4D; }}
QPushButton:disabled, QToolButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE_2}; border-color: {BORDER}; }}
QPushButton[variant="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #0C0A1A; font-weight: 600; }}
QPushButton[variant="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[variant="primary"]:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton[variant="primary"]:disabled {{ background: #3A3566; border-color: #3A3566; color: #8F89B8; }}
QPushButton[variant="danger"] {{ background: {DANGER_SOFT}; border-color: rgba(240,100,95,0.45); color: #FFB3B0; }}
QPushButton[variant="danger"]:hover {{ background: rgba(240,100,95,0.25); }}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{ background: transparent; border-color: transparent;
    color: {TEXT_2}; }}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{ background: {SURFACE_3}; color: {TEXT}; }}
QPushButton[variant="ghost"]:checked, QToolButton[variant="ghost"]:checked {{ background: {ACCENT_SOFT};
    color: {TEXT}; }}
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
    selection-background-color: {ACCENT}; selection-color: #0C0A1A; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {TEXT_FAINT}; }}
QLineEdit[search="true"] {{ padding-left: 30px; border-radius: 9px; background: {SURFACE_2}; }}
QPlainTextEdit[code="true"], QTextEdit[code="true"] {{ font-family: {MONO}; font-size: 12px; background: #0F131A; }}
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
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}

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
    border-radius: {RADIUS_SM}px; alternate-background-color: {SURFACE_2}; gridline-color: {BORDER}; }}
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
QScrollBar::handle:vertical:hover {{ background: #3D4859; }}
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
                                             ("check", "check", "#0C0A1A", 12)):
            target = folder / f"{name}.png"
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


def apply(app: QApplication) -> None:
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
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0C0A1A"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_3))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    font = QFont(app.font())
    font.setPointSizeF(max(font.pointSizeF(), 13.0) if font.pointSizeF() > 0 else 13.0)
    app.setFont(font)
    app.setStyleSheet(stylesheet() + _indicator_images())


def set_prop(widget: QWidget, name: str, value: object) -> None:
    """Set a dynamic property used by the style sheet and re-polish the widget."""
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
