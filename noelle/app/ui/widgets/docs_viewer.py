"""Model card tab: README and license rendered in a sandboxed viewer.

With QtWebEngine the page uses an off-the-record profile with JavaScript,
plugins and local-file access disabled; a request interceptor only lets
images/styles load from Hugging Face hosts, and clicked links open in the
system browser instead of navigating the embedded view. Without QtWebEngine
(or when rendering offscreen) a plain QTextBrowser shows the markdown.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QTextBrowser, QVBoxLayout, QWidget

from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.chat_canvas import doc_css
from noelle.app.ui.widgets.common import Pill, Spinner, button, label, open_external_url

ALLOWED_HOSTS = ("huggingface.co", "cdn-uploads.huggingface.co", "cdn-avatars.huggingface.co",
                 "cas-bridge.xethub.hf.co")

def page_css() -> str:
    """Model-card page style, in the active palette."""
    return f"""
body {{ background: {theme.BG}; color: {theme.TEXT_2}; font: 14px -apple-system, "Segoe UI", sans-serif;
       line-height: 1.55; margin: 0 auto; padding: 8px 18px 40px; max-width: 920px; }}
h1, h2, h3, h4 {{ color: {theme.TEXT}; }} a {{ color: {theme.ACCENT}; }}
code, pre {{ font-family: Menlo, monospace; background: {theme.CODE_BG}; color: {theme.CODE_TEXT}; border-radius: 6px; }}
code {{ padding: 1px 5px; }} pre {{ padding: 12px; overflow-x: auto; }}
table {{ border-collapse: collapse; }} th, td {{ border: 1px solid {theme.BORDER_STRONG}; padding: 5px 9px; }}
img {{ max-width: 100%; }} blockquote {{ border-left: 3px solid {theme.BORDER_STRONG}; margin: 0; padding-left: 12px;
       color: {theme.TEXT_MUTED}; }}
"""


def strip_front_matter(markdown: str) -> tuple[str, dict[str, str]]:
    meta: dict[str, str] = {}
    if markdown.startswith("---"):
        end = markdown.find("\n---", 3)
        if end != -1:
            for line in markdown[3:end].splitlines():
                if ":" in line and not line.startswith(" "):
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            markdown = markdown[end + 4:]
    return markdown, meta


def markdown_to_html(markdown: str) -> str:
    try:
        from markdown_it import MarkdownIt

        body = MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough").render(markdown)
    except Exception:  # markdown-it not installed: fall back to Qt's converter
        from PySide6.QtGui import QTextDocument

        doc = QTextDocument()
        doc.setMarkdown(markdown)
        body = doc.toHtml()
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{page_css()}</style></head><body>{body}</body></html>"


def _webengine_usable() -> bool:
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    try:
        importlib.import_module("PySide6.QtWebEngineWidgets")
    except Exception:
        return False
    return True


def _make_web_view() -> Any:
    from PySide6.QtWebEngineCore import (
        QWebEnginePage,
        QWebEngineProfile,
        QWebEngineSettings,
        QWebEngineUrlRequestInterceptor,
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView

    class Interceptor(QWebEngineUrlRequestInterceptor):
        def interceptRequest(self, info: Any) -> None:  # noqa: N802
            url = info.requestUrl()
            if url.scheme() in ("data", "about"):
                return
            host = url.host().lower()
            if url.scheme() != "https" or not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
                info.block(True)

    class SandboxPage(QWebEnginePage):
        def acceptNavigationRequest(self, url: QUrl, nav_type: Any, is_main_frame: bool) -> bool:  # noqa: N802
            if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
                open_external_url(url)
                return False
            return True

    profile = QWebEngineProfile()  # off-the-record: no cookies, cache or storage persisted
    interceptor = Interceptor(profile)
    profile.setUrlRequestInterceptor(interceptor)
    profile._noelle_interceptor = interceptor  # keep a Python reference alive
    view = QWebEngineView()
    page = SandboxPage(profile, view)
    settings = page.settings()
    for attribute, value in (
        (QWebEngineSettings.WebAttribute.JavascriptEnabled, False),
        (QWebEngineSettings.WebAttribute.PluginsEnabled, False),
        (QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False),
        (QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False),
        (QWebEngineSettings.WebAttribute.LocalStorageEnabled, False),
        (QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False),
        (QWebEngineSettings.WebAttribute.AutoLoadImages, True),
    ):
        settings.setAttribute(attribute, value)
    view.setPage(page)
    view._noelle_profile = profile
    return view


class DocsViewer(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._repo: str | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 0)
        root.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(label("Model card", "h2"))
        self.license = Pill("", "unknown")
        head.addWidget(self.license)
        self.sandbox = label("", "faint")
        head.addWidget(self.sandbox)
        head.addStretch(1)
        self.spinner = Spinner(16)
        head.addWidget(self.spinner)
        self.license_btn = button("License on the Hub", "shield", "ghost")
        self.license_btn.clicked.connect(self._open_license)
        head.addWidget(self.license_btn)
        root.addLayout(head)
        self.web = _make_web_view() if _webengine_usable() else None
        if self.web is not None:
            self.sandbox.setText("sandboxed · scripts disabled")
            root.addWidget(self.web, 1)
            self.text = None
        else:
            self.sandbox.setText("plain-text renderer")
            self.text = QTextBrowser()
            self.text.setOpenLinks(False)
            self.text.anchorClicked.connect(open_external_url)
            self.text.document().setDefaultStyleSheet(doc_css())
            root.addWidget(self.text, 1)

    def set_model(self, summary: Any) -> None:
        if summary is None or summary.id == self._repo:
            return
        self._repo = summary.id
        self.license.set_text(str(summary.license or "license unknown"))
        self.spinner.start()
        record = self.ctx.registry.get_model(summary.id, source="hub")
        local = Path(record.local_path) / "README.md" if record and record.local_path else None

        def fetch() -> str:
            if local is not None and local.is_file():
                return local.read_text(encoding="utf-8", errors="replace")
            return self.ctx.hf.fetch_readme(summary.id)

        runner().submit(fetch, on_result=lambda text, repo=summary.id: self._render(repo, text),
                        on_error=lambda msg: self._render(summary.id, f"Could not load the model card: {msg}"))

    def _render(self, repo: str, markdown: str) -> None:
        if repo != self._repo:
            return
        self.spinner.stop()
        body, meta = strip_front_matter(markdown or "*This repository has no README.*")
        if meta.get("license") and not self.license.text().strip("● ").strip():
            self.license.set_text(meta["license"])
        if self.web is not None:
            self.web.setHtml(markdown_to_html(body), QUrl(f"https://huggingface.co/{repo}/"))
        elif self.text is not None:
            self.text.setMarkdown(re.sub(r"<[^>]+>", "", body))

    def _open_license(self) -> None:
        if self._repo:
            QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{self._repo}/blob/main/README.md"))

