"""Embeddings playground: rank passages against a query by cosine similarity."""

from __future__ import annotations

import math
from typing import Any, Callable

from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import ErrorLine, SearchField, Spinner, button, label
from noelle.runtime.protocol import Command

SAMPLE_DOCS = """Noelle runs open-weight models locally on your GPU.
The cat sat quietly on the warm windowsill.
Quantization stores weights in 4 bits to save memory.
Paris is the capital of France.
Sentence embeddings map text to vectors for semantic search."""


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingPlayground(QWidget):
    def __init__(self, ctx: AppContext, payload: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.ctx = ctx
        self.payload = payload
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)
        root.addWidget(label("Semantic similarity", "h2"))
        root.addWidget(label("Embed a query and a set of passages (one per line) and rank them by cosine similarity.",
                             "muted"))
        query_row = QHBoxLayout()
        self.query = SearchField("Query")
        self.query.setText("How do I make a model use less memory?")
        self.query.returnPressed.connect(self.run)
        query_row.addWidget(self.query, 1)
        self.run_btn = button("Embed & rank", "nodes", "primary")
        self.run_btn.clicked.connect(self.run)
        query_row.addWidget(self.run_btn)
        self.spinner = Spinner(16)
        query_row.addWidget(self.spinner)
        root.addLayout(query_row)
        self.docs = QPlainTextEdit(SAMPLE_DOCS)
        self.docs.setFixedHeight(130)
        root.addWidget(self.docs)
        self.status = label("", "faint", wrap=True)
        root.addWidget(self.status)
        self.error = ErrorLine()
        root.addWidget(self.error)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Score", "", "Passage"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 150)
        self.table.verticalHeader().hide()
        root.addWidget(self.table, 1)

    def run(self) -> None:
        query = self.query.text().strip()
        docs = [line.strip() for line in self.docs.toPlainText().splitlines() if line.strip()]
        if not query or not docs:
            return
        self.error.clear()
        self.spinner.start()
        self.run_btn.setEnabled(False)
        handle = self.ctx.runtime.request(Command.EMBED, {**self.payload(), "inputs": [query, *docs]})
        handle.status.connect(self._on_status)
        handle.finished.connect(lambda result: self._show(docs, result))
        handle.failed.connect(self._failed)

    def _on_status(self, payload: dict[str, Any]) -> None:
        self.status.setText(str(payload.get("message") or ""))

    def _show(self, docs: list[str], result: dict[str, Any]) -> None:
        self.spinner.stop()
        self.run_btn.setEnabled(True)
        vectors = result.get("embeddings") or []
        if len(vectors) != len(docs) + 1:
            return
        query = vectors[0]
        scored = sorted(((cosine(query, vec), doc) for vec, doc in zip(vectors[1:], docs)), reverse=True)
        self.table.setRowCount(len(scored))
        for row, (score, doc) in enumerate(scored):
            self.table.setItem(row, 0, QTableWidgetItem(f"{score:.3f}"))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(max(score, 0) * 1000))
            self.table.setCellWidget(row, 1, bar)
            self.table.setItem(row, 2, QTableWidgetItem(doc))
        preview = ", ".join(f"{v:+.3f}" for v in query[:6])
        self.status.setText(f"{result.get('dimension')}-dimensional vectors · {result.get('prompt_tokens')} tokens · "
                            f"query ≈ [{preview}, …]")

    def _failed(self, payload: dict[str, Any]) -> None:
        self.spinner.stop()
        self.run_btn.setEnabled(True)
        self.status.setText("Failed.")
        self.error.set_error(str(payload.get("message") or "Embedding failed."),
                             str(payload["hint"]) if payload.get("hint") else None)
