"""Code tab: standalone inference script / container exporter (task T-702)."""

from __future__ import annotations

import re
import time
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QGuiApplication, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget

from noelle import paths
from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import button, label

_KEYWORDS = r"\b(def|class|import|from|as|return|if|elif|else|for|while|in|not|and|or|with|try|except|finally|raise|" \
            r"None|True|False|lambda|yield|async|await|pass|break|continue|FROM|RUN|COPY|ENV|ARG|EXPOSE|CMD|" \
            r"ENTRYPOINT|WORKDIR|USER|HEALTHCHECK)\b"


class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, document: Any) -> None:
        super().__init__(document)
        self.rules: list[tuple[re.Pattern[str], QTextCharFormat]] = []
        for pattern, color, bold in ((_KEYWORDS, theme.SYNTAX_KEYWORD, True),
                                     (r"\b\d+(\.\d+)?\b", theme.SYNTAX_NUMBER, False),
                                     (r"(\"[^\"\n]*\"|'[^'\n]*')", theme.SYNTAX_STRING, False),
                                     (r"#[^\n]*", theme.SYNTAX_COMMENT, False),
                                     (r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()", theme.SYNTAX_FUNCTION, False)):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if bold:
                fmt.setFontWeight(QFont.Weight.DemiBold)
            self.rules.append((re.compile(pattern), fmt))

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class CodeExportPanel(QWidget):
    FILES = [("inference.py", "python_script"), ("requirements.txt", "requirements_txt"), ("Dockerfile", "dockerfile"),
             ("server.py", "server_script")]

    def __init__(self, ctx: AppContext, page: Any) -> None:
        super().__init__()
        self.ctx = ctx
        self.page = page
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("Export to code", "h2"))
        titles.addWidget(label("Reproduce this playground outside Noelle: a pinned standalone script and a CUDA "
                               "container that serves the model over HTTP.", "muted", wrap=True))
        head.addLayout(titles, 1)
        regen = button("Regenerate", "refresh", "ghost")
        regen.clicked.connect(self.regenerate)
        head.addWidget(regen)
        copy = button("Copy", "copy")
        copy.clicked.connect(self._copy)
        head.addWidget(copy)
        save = button("Save bundle…", "save", "primary")
        save.clicked.connect(self._save)
        head.addWidget(save)
        root.addLayout(head)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.editors: dict[str, QPlainTextEdit] = {}
        for name, _ in self.FILES:
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            editor.setProperty("code", True)
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            CodeHighlighter(editor.document())
            self.editors[name] = editor
            self.tabs.addTab(editor, name)
        root.addWidget(self.tabs, 1)
        self.note = label("", "faint", wrap=True)
        root.addWidget(self.note)

    def on_activated(self) -> None:
        self.regenerate()

    def _config(self) -> Any:
        from noelle.app.services.exporter import ExportConfig

        data = self.page.current_export_config()
        payload = data.get("payload") or {}
        params = dict(data.get("params") or {})
        extra = dict(data.get("extra") or {})
        record = self.page.selected_record() if hasattr(self.page, "selected_record") else None
        if record is not None and record.base_model:
            extra.setdefault("base_model", record.base_model)  # containers fall back to the original weights
        return ExportConfig(
            model_id=payload.get("model_id", ""),
            task=payload.get("task") or "text-generation",
            precision=payload.get("precision") or "auto",
            device=payload.get("device") or "auto",
            system_prompt=data.get("system_prompt"),
            params=params,
            local_path=None,  # exported code downloads from the Hub instead of pointing at this machine's cache
            gguf_file=payload.get("gguf_file"),
            adapter_id=data.get("adapter_id"),
            extra=extra,
        )

    def regenerate(self) -> None:
        try:
            from noelle.app.services import exporter

            config = self._config()
            for name, fn in self.FILES:
                self.editors[name].setPlainText(getattr(exporter, fn)(config))
            base_note = ""
            if config.precision == "mlx-4bit":
                base_note = " MLX builds only run on Apple Silicon; the Dockerfile falls back to the FP16 model."
            self.note.setText(f"Generated {time.strftime('%H:%M:%S')} for {config.model_id} ({config.task}, "
                              f"{config.precision}).{base_note}")
        except Exception as exc:  # exporter missing or config incomplete
            self.note.setText(f"Could not generate code: {exc}")

    def _copy(self) -> None:
        editor = self.tabs.currentWidget()
        if isinstance(editor, QPlainTextEdit):
            QGuiApplication.clipboard().setText(editor.toPlainText())
            self.ctx.toast("Copied to clipboard", "success")

    def _save(self) -> None:
        from pathlib import Path

        from noelle.app.services import exporter

        config = self._config()
        default = paths.exports_dir() / f"{config.model_id.split('/')[-1]}-export"
        folder = QFileDialog.getExistingDirectory(self, "Export bundle to folder", str(default.parent))
        if not folder:
            return
        target = Path(folder) / default.name
        target.mkdir(parents=True, exist_ok=True)
        files = exporter.export_bundle(config, target)
        self.ctx.toast(f"Wrote {len(files)} files to {target}", "success")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

