"""Streaming chat transcript with markdown rendering and per-turn metrics."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QGuiApplication, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from noelle.app import icons, theme
from noelle.app.ui.widgets.common import Spinner, icon_button, label, open_external_url
from noelle.runtime.preflight import format_bytes

def doc_css() -> str:
    """Rich-text style for rendered markdown, in the active palette."""
    return f"""
code {{ font-family: Menlo, monospace; background-color: {theme.CODE_BG}; color: {theme.CODE_TEXT}; }}
pre {{ font-family: Menlo, monospace; background-color: {theme.CODE_BG}; color: {theme.CODE_TEXT}; padding: 8px; }}
a {{ color: {theme.ACCENT}; }}
h1, h2, h3 {{ color: {theme.TEXT}; }}
blockquote {{ color: {theme.TEXT_MUTED}; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid {theme.BORDER_STRONG}; padding: 4px 8px; }}
"""


def markdown_html(text: str) -> str | None:
    """CommonMark (+ tables, strikethrough) to HTML so ``doc_css`` styles code blocks; raw HTML stays escaped."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return None
    return MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough").render(text)


class MarkdownView(QTextBrowser):
    """Read-only markdown block that grows to fit its content (no inner scrollbar)."""

    def __init__(self) -> None:
        super().__init__()
        self.setOpenLinks(False)
        self.anchorClicked.connect(open_external_url)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("QTextBrowser { background: transparent; border: none; padding: 0; }")
        self.document().setDocumentMargin(0)
        self.document().setDefaultStyleSheet(doc_css())
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().contentsChanged.connect(self._fit)
        self.setFixedHeight(22)

    def set_markdown(self, text: str) -> None:
        html = markdown_html(text or "")
        if html is None:
            self.document().setMarkdown(text or "")  # Qt's importer ignores the style sheet (no code styling)
        else:
            self.document().setHtml(html)
        self._fit()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        self.document().setTextWidth(max(self.viewport().width(), 50))
        height = int(self.document().size().height()) + 4
        if height != self.height():
            self.setFixedHeight(max(height, 20))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(400, self.height())


class MessageView(QFrame):
    """One turn. Assistant turns stream: ``append_text`` buffers and re-renders at most every 50 ms."""

    def __init__(self, role: str, name: str = "", text: str = "") -> None:
        super().__init__()
        self.role = role
        self._text = text
        is_user = role == "user"
        background = theme.SURFACE_2 if is_user else "transparent"
        border = theme.BORDER if is_user else "transparent"
        self.setStyleSheet(f"MessageView {{ background: {background}; border: 1px solid {border}; border-radius: 12px; }}")
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        badge = QLabel()
        badge.setPixmap(icons.pixmap("user" if is_user else ("alert" if role == "error" else "sparkles"),
                                     theme.TEXT_MUTED if is_user else theme.ACCENT, 15))
        head.addWidget(badge)
        head.addWidget(label("You" if is_user else (name or "Assistant"), "h3"))
        self.status = label("", "faint")
        head.addWidget(self.status)
        self.spinner = Spinner(14)
        head.addWidget(self.spinner)
        head.addStretch(1)
        self.metrics = label("", "faint")
        head.addWidget(self.metrics)
        self.copy_btn = icon_button("copy", "Copy", size=24, icon_size=13)
        self.copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(self._text))
        head.addWidget(self.copy_btn)
        root.addLayout(head)
        self.body = MarkdownView()
        root.addWidget(self.body)
        self.sources = label("", "faint", wrap=True)
        self.sources.hide()
        root.addWidget(self.sources)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._render)
        if text:
            self._render()

    @property
    def text(self) -> str:
        return self._text

    def append_text(self, delta: str) -> None:
        self._text += delta
        if not self._timer.isActive():
            self._timer.start()

    def set_text(self, text: str) -> None:
        self._text = text
        self._render()

    def _render(self) -> None:
        self.body.set_markdown(self._text if self._text else "")

    def set_busy(self, busy: bool, status: str = "") -> None:
        self.status.setText(status)
        if busy:
            self.spinner.start()
        else:
            self.spinner.stop()

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        parts = []
        if metrics.get("ttft_ms") is not None:
            parts.append(f"TTFT {metrics['ttft_ms']:.0f} ms")
        if metrics.get("tokens_per_second"):
            parts.append(f"{metrics['tokens_per_second']:.1f} tok/s")
        if metrics.get("completion_tokens") is not None:
            parts.append(f"{metrics['completion_tokens']} tokens")
        if metrics.get("peak_memory_bytes"):
            parts.append(f"peak {format_bytes(metrics['peak_memory_bytes'])}")
        if metrics.get("finish_reason") and metrics["finish_reason"] != "stop":
            parts.append(metrics["finish_reason"])
        tuning = metrics.get("tuning") or {}
        if tuning.get("clamped"):
            parts.append(f"max tokens clamped to {tuning.get('max_new_tokens')}")
            self.metrics.setToolTip(str(tuning.get("reason") or ""))
        self.metrics.setText(" · ".join(parts))

    def set_sources(self, sources: list[dict[str, Any]]) -> None:
        if not sources:
            self.sources.hide()
            return
        names = []
        for source in sources[:8]:
            title = source.get("source") or source.get("title") or source.get("url") or "source"
            names.append(f"[{source.get('index', '?')}] {title}")
        self.sources.setText("Sources: " + "  ·  ".join(names))
        self.sources.show()

    def set_error(self, message: str, hint: str | None = None) -> None:
        self.set_busy(False)
        text = f"**{message}**" + (f"\n\n{hint}" if hint else "")
        self.setStyleSheet(f"MessageView {{ background: {theme.DANGER_SOFT}; border: 1px solid {theme.DANGER_BORDER};"
                           " border-radius: 12px; }")
        self.set_text(text)


class ChatCanvas(QScrollArea):
    """Vertical list of turns; keeps the view pinned to the bottom while streaming."""

    link_clicked = Signal(QUrl)

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        self._layout = QVBoxLayout(host)
        self._layout.setContentsMargins(4, 8, 10, 8)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)
        self.setWidget(host)
        self._stick = True
        self.verticalScrollBar().rangeChanged.connect(self._on_range)
        self.verticalScrollBar().valueChanged.connect(self._on_value)
        self.messages: list[MessageView] = []
        self.placeholder = label("", "muted", wrap=True)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.insertWidget(0, self.placeholder)

    def set_placeholder(self, text: str) -> None:
        self.placeholder.setText(text)
        self.placeholder.setVisible(not self.messages and bool(text))

    def add_message(self, role: str, text: str = "", name: str = "") -> MessageView:
        view = MessageView(role, name, text)
        self._layout.insertWidget(self._layout.count() - 1, view)
        self.messages.append(view)
        self.placeholder.hide()
        self._stick = True
        return view

    def clear(self) -> None:
        for view in self.messages:
            view.setParent(None)
            view.deleteLater()
        self.messages.clear()
        self.placeholder.setVisible(bool(self.placeholder.text()))

    def _on_range(self, _minimum: int, maximum: int) -> None:
        if self._stick:
            self.verticalScrollBar().setValue(maximum)

    def _on_value(self, value: int) -> None:
        self._stick = value >= self.verticalScrollBar().maximum() - 24
