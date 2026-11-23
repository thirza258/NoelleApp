"""Small reusable widgets and formatting helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from noelle.app import icons, theme


# --------------------------------------------------------------- formatting
def human_count(value: int | float | None) -> str:
    if value is None:
        return "—"
    value = float(value)
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= size:
            text = f"{value / size:.1f}".rstrip("0").rstrip(".")
            return f"{text}{unit}"
    return f"{int(value)}"


def human_bytes(value: int | float | None) -> str:
    from noelle.runtime.preflight import format_bytes

    return format_bytes(value)


def time_ago(moment: datetime | float | None) -> str:
    if moment is None:
        return "—"
    if isinstance(moment, (int, float)):
        moment = datetime.fromtimestamp(moment, tz=timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    for unit, size in (("y", 31_536_000), ("mo", 2_592_000), ("d", 86_400), ("h", 3_600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


# ------------------------------------------------------------------ widgets
def open_external_url(url: Any) -> bool:
    """Open a link from untrusted content (model cards, model output, search results) in the browser.

    Only http(s) is allowed: other schemes (file:, smb:, app-specific handlers) could open local files
    or launch other applications."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    qurl = url if isinstance(url, QUrl) else QUrl(str(url))
    if qurl.scheme().lower() not in ("http", "https") or not qurl.host():
        return False
    return QDesktopServices.openUrl(qurl)


def label(text: str = "", role: str | None = None, *, wrap: bool = False, selectable: bool = False) -> QLabel:
    widget = QLabel(text)
    if role:
        widget.setProperty("role", role)
    widget.setWordWrap(wrap)
    if selectable:
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


def button(text: str = "", icon_name: str | None = None, variant: str | None = None, *,
           tooltip: str | None = None, checkable: bool = False, icon_color: str | None = None) -> QPushButton:
    widget = QPushButton(text.replace("&", "&&"))
    if variant:
        widget.setProperty("variant", variant)
    if icon_name:
        color = icon_color or (theme.ON_ACCENT if variant == "primary" else theme.TEXT_2)
        widget.setIcon(icons.icon(icon_name, color, 16))
        widget.setIconSize(QSize(16, 16))
    if tooltip:
        widget.setToolTip(tooltip)
    widget.setCheckable(checkable)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    return widget


def icon_button(icon_name: str, tooltip: str, *, size: int = 32, icon_size: int = 17,
                color: str | None = None, checkable: bool = False) -> QToolButton:
    widget = QToolButton()
    widget.setProperty("variant", "ghost")
    widget.setIcon(icons.icon(icon_name, color, icon_size))
    widget.setIconSize(QSize(icon_size, icon_size))
    widget.setToolTip(tooltip)
    widget.setFixedSize(size, size)
    widget.setCheckable(checkable)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.setStyleSheet("QToolButton { padding: 0; }")
    return widget


class Pill(QLabel):
    """Rounded badge. ``level`` picks colours from theme.LEVEL_COLORS."""

    def __init__(self, text: str = "", level: str = "unknown", *, icon_name: str | None = None,
                 tooltip: str | None = None, mono: bool = False) -> None:
        super().__init__()
        self._icon_name = icon_name
        self._mono = mono
        self.set_level(level)
        self.set_text(text)
        if tooltip:
            self.setToolTip(tooltip)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_level(self, level: str) -> None:
        fg, bg = theme.LEVEL_COLORS.get(level, theme.LEVEL_COLORS["unknown"])
        family = f"font-family: {theme.MONO};" if self._mono else ""
        self.setStyleSheet(f"QLabel {{ color: {fg}; background: {bg}; border-radius: 9px; padding: 2px 8px;"
                           f" font-size: 11px; font-weight: 600; {family} }}")
        self._fg = fg

    def set_text(self, text: str) -> None:
        if self._icon_name:
            self.setText(f"● {text}" if self._icon_name == "dot" else text)
        else:
            self.setText(text)


class Avatar(QWidget):
    def __init__(self, name: str = "", size: int = 36, radius: int | None = None) -> None:
        super().__init__()
        self._name = name
        self._radius = radius if radius is not None else max(6, size // 4)
        self.setFixedSize(size, size)

    def set_name(self, name: str) -> None:
        self._name = name
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        color = QColor(theme.avatar_color(self._name))
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        bg = QColor(color)
        bg.setAlpha(46)
        painter.fillPath(path, bg)
        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 120), 1))
        painter.drawPath(path)
        initials = "".join(part[:1] for part in self._name.replace("-", " ").replace("_", " ").split()[:2]).upper()
        initials = initials or (self._name[:1].upper() if self._name else "?")
        font = QFont(self.font())
        font.setPixelSize(max(10, int(self.height() * 0.38)))
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(color.lighter(130) if theme.IS_DARK else color.darker(135))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, initials[:2])


class Card(QFrame):
    clicked = Signal()

    def __init__(self, *, clickable: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self._clickable = clickable
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        if self._clickable:
            theme.set_prop(self, "hover", True)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        if self._clickable:
            theme.set_prop(self, "hover", False)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if self._clickable and event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class Panel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)


def divider() -> QFrame:
    line = QFrame()
    line.setProperty("divider", True)
    return line


class Banner(QFrame):
    """Inline notice with optional action buttons."""

    def __init__(self, kind: str = "info", text: str = "", *, icon_name: str | None = None) -> None:
        super().__init__()
        self.setProperty("banner", kind)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 9, 10, 9)
        layout.setSpacing(10)
        color = {"warning": theme.WARNING, "danger": theme.DANGER, "success": theme.SUCCESS}.get(kind, theme.INFO)
        self._icon = QLabel()
        self._icon.setPixmap(icons.pixmap(icon_name or ("alert" if kind in ("warning", "danger") else "info"), color, 16))
        self._icon.setFixedWidth(18)
        layout.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        self.text = QLabel(text)
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        self.text.setOpenExternalLinks(False)
        layout.addWidget(self.text, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        layout.addLayout(self.actions)

    def set_text(self, text: str) -> None:
        self.text.setText(text)

    def add_action(self, text: str, callback: Callable[[], None], variant: str | None = None) -> QPushButton:
        widget = button(text, variant=variant)
        widget.clicked.connect(callback)
        self.actions.addWidget(widget)
        return widget

    def clear_actions(self) -> None:
        while self.actions.count():
            item = self.actions.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class StatTile(QFrame):
    def __init__(self, title: str, value: str = "—", caption: str = "") -> None:
        super().__init__()
        self.setProperty("card", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)
        self.title = label(title.upper(), "eyebrow")
        self.value = label(value, "metric")
        self.caption = label(caption, "faint")
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.caption)

    def set(self, value: str, caption: str | None = None) -> None:
        self.value.setText(value)
        if caption is not None:
            self.caption.setText(caption)


class EmptyState(QWidget):
    action_clicked = Signal()

    def __init__(self, icon_name: str, title: str, message: str = "", action: str | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 40, 24, 40)
        layout.setSpacing(8)
        layout.addStretch(1)
        art = QLabel()
        art.setPixmap(icons.pixmap(icon_name, theme.TEXT_FAINT, 40))
        art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(art)
        self.title = label(title, "h3")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        self.message = label(message, "muted", wrap=True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message)
        if action:
            row = QHBoxLayout()
            row.addStretch(1)
            btn = button(action, variant="primary")
            btn.clicked.connect(self.action_clicked)
            row.addWidget(btn)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addStretch(2)


class SearchField(QLineEdit):
    def __init__(self, placeholder: str = "Search") -> None:
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setProperty("search", True)
        self.setClearButtonEnabled(True)
        self._icon = QLabel(self)
        self._icon.setPixmap(icons.pixmap("search", theme.TEXT_FAINT, 15))
        self._icon.setFixedSize(16, 16)
        self._icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._icon.move(11, (self.height() - 16) // 2)


class Spinner(QWidget):
    def __init__(self, size: int = 18, color: str | None = None) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self._angle = 0
        self._color = QColor(color or theme.ACCENT)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance)
        self.hide()

    def start(self) -> None:
        self.show()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _advance(self) -> None:
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self._color, 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        margin = 2
        rect = self.rect().adjusted(margin, margin, -margin, -margin)
        painter.drawArc(rect, -self._angle * 16, 270 * 16)


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", role: str | None = None) -> None:
        super().__init__()
        self._full = text
        if role:
            self.setProperty("role", role)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, max(self.width(), 20)))


class FlowLayout(QLayout):
    """Wrapping layout for chips/tags (adapted from the Qt flow layout example)."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._layout(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, dry=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def clear(self) -> None:
        while self._items:
            item = self._items.pop()
            widget = item.widget()
            if widget is not None:
                # detach immediately: deleteLater alone leaves the old widget painted at its last
                # geometry until the event loop runs, which overlaps the new items
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.invalidate()

    def _layout(self, rect: QRect, *, dry: bool) -> int:
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > rect.right() and line_height > 0:
                x = rect.x()
                y += line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not dry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class Toast(QFrame):
    """Transient notification anchored to the bottom-right of its parent window."""

    def __init__(self, parent: QWidget, text: str, kind: str = "info", timeout_ms: int = 3800) -> None:
        super().__init__(parent)
        color = {"success": theme.SUCCESS, "warning": theme.WARNING, "error": theme.DANGER}.get(kind, theme.ACCENT)
        self.setStyleSheet(f"QFrame {{ background: {theme.SURFACE_3}; border: 1px solid {theme.BORDER_STRONG};"
                           f" border-left: 3px solid {color}; border-radius: 8px; }}"
                           f" QLabel {{ background: transparent; border: none; }}")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 14, 10)
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setMaximumWidth(380)
        layout.addWidget(text_label)
        self.adjustSize()
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self._fade = QPropertyAnimation(effect, b"opacity", self)
        self._fade.setDuration(220)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        QTimer.singleShot(timeout_ms, self._dismiss)

    def show_at_corner(self, offset: int = 0) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x = parent.width() - self.width() - 20
        y = parent.height() - self.height() - 46 - offset
        self.move(max(x, 10), max(y, 10))
        self.show()
        self.raise_()
        self._fade.start()

    def _dismiss(self) -> None:
        self._fade.setDirection(QPropertyAnimation.Direction.Backward)
        self._fade.finished.connect(self.deleteLater)
        self._fade.start()


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())
