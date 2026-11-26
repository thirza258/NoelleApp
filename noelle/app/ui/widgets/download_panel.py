"""Per-model download manager panel on the workspace page (task T-301 UI)."""

from __future__ import annotations

import shutil
from typing import Any

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from noelle import paths
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import button, human_bytes, human_duration, label
from noelle.runtime.downloader import RepoFile, select_files


class DownloadPanel(QWidget):
    downloaded = Signal()

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.summary: Any = None
        self.repo_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.status = label("", "muted", wrap=True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.detail = label("", "faint", wrap=True)
        layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.download_btn = button("Download", "download", "primary")
        self.download_btn.clicked.connect(self._start)
        buttons.addWidget(self.download_btn)
        self.pause_btn = button("Pause", "pause")
        self.pause_btn.clicked.connect(lambda: self.repo_id and self.ctx.downloads.pause(self.repo_id))
        buttons.addWidget(self.pause_btn)
        self.resume_btn = button("Resume", "play")
        self.resume_btn.clicked.connect(lambda: self.repo_id and self.ctx.downloads.resume(self.repo_id))
        buttons.addWidget(self.resume_btn)
        self.open_page_btn = button("Open model page", "external")
        self.open_page_btn.setToolTip("Accept the model's licence on huggingface.co, then download again")
        self.open_page_btn.clicked.connect(self._open_model_page)
        self.open_page_btn.hide()
        buttons.addWidget(self.open_page_btn)
        self.cancel_btn = button("Cancel", "x", "ghost")
        self.cancel_btn.clicked.connect(lambda: self.repo_id and self.ctx.downloads.cancel(self.repo_id))
        buttons.addWidget(self.cancel_btn)
        self.reveal_btn = button("Show files", "folder", "ghost")
        self.reveal_btn.clicked.connect(self._reveal)
        buttons.addWidget(self.reveal_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.files = QTreeWidget()
        self.files.setHeaderLabels(["File", "Size", "Progress"])
        self.files.setRootIsDecorated(False)
        self.files.setColumnWidth(0, 170)
        self.files.setColumnWidth(1, 70)
        self.files.setMaximumHeight(170)
        self.files.hide()
        layout.addWidget(self.files)
        ctx.downloads.job_updated.connect(self._on_job)
        ctx.downloads.job_finished.connect(self._on_finished)

    def set_model(self, summary: Any) -> None:
        self.summary = summary
        self.repo_id = summary.id if summary is not None else None
        self.refresh()

    def planned_files(self) -> list[RepoFile]:
        siblings = getattr(self.summary, "siblings", None) or []
        files = [RepoFile(s["name"], s.get("size"), None, None) for s in siblings]
        return select_files(files, None, None)

    def refresh(self) -> None:
        self._refresh()
        self.detail.setVisible(bool(self.detail.text()))

    def _refresh(self) -> None:
        if self.repo_id is None:
            return
        job = self.ctx.downloads.job(self.repo_id)
        record = self.ctx.registry.get_model(self.repo_id, source="hub")
        downloaded = record is not None and record.exists_on_disk()
        free = shutil.disk_usage(paths.home()).free
        if job is not None and job.active:
            self._render_job(job)
            return
        self.progress.hide()
        self.files.hide()
        self.pause_btn.hide()
        self.cancel_btn.hide()
        gated = job is not None and job.error_code == "GATED"
        gated_model = bool(getattr(self.summary, "gated", False)) and not downloaded
        self.open_page_btn.setVisible(gated or gated_model)
        self.resume_btn.setVisible(job is not None and job.status in ("paused", "error", "cancelled")
                                   and not downloaded and not gated)
        if downloaded:
            size = human_bytes(record.size_bytes) if record.size_bytes else ""
            self.status.setText(f"✓ Downloaded {size}".strip())
            self.detail.setText(f"Revision {str(record.commit_hash or '')[:8]} · stored in the Hugging Face cache"
                                if record.commit_hash else "")
            self.download_btn.hide()
            self.reveal_btn.show()
            return
        self.reveal_btn.hide()
        self.download_btn.show()
        planned = self.planned_files()
        total = sum(f.size or 0 for f in planned)
        if job is not None and job.status == "error":
            self.status.setText(f"Download failed: {job.error or 'unknown error'}")
        elif job is not None and job.status == "paused":
            self.status.setText(f"Paused at {job.fraction * 100:.0f}% — resume to continue; finished files are kept.")
        else:
            self.status.setText(f"{len(planned)} files · {human_bytes(total)}" if planned else
                                "File list loads with the model details…")
        warn = " — not enough free space!" if total and total > free * 0.95 else ""
        self.detail.setText(f"{human_bytes(free)} free on this disk{warn}. Weights are verified against the "
                            "Hub's SHA-256 after download.")
        self.download_btn.setEnabled(bool(planned) and not warn)

    def _render_job(self, job: Any) -> None:
        self.download_btn.hide()
        self.reveal_btn.hide()
        self.open_page_btn.hide()
        self.progress.show()
        self.files.show()
        paused = job.status == "paused"
        self.pause_btn.setVisible(job.active)
        self.cancel_btn.setVisible(True)
        self.resume_btn.setVisible(paused)
        self.progress.setValue(int(job.fraction * 1000))
        if job.status == "planning":
            self.status.setText("Resolving files…")
        else:
            self.status.setText(f"Downloading · {job.fraction * 100:.0f}%")
        speed = f"{human_bytes(job.speed_bps)}/s" if job.speed_bps else "—"
        retries = f" · {job.retries} retr{'y' if job.retries == 1 else 'ies'}" if job.retries else ""
        self.detail.setText(f"{human_bytes(job.downloaded_bytes)} of {human_bytes(job.download_bytes)} · {speed} · "
                            f"ETA {human_duration(job.eta_s)}{retries}")
        self.files.clear()
        for entry in sorted(job.files.values(), key=lambda f: (f.done, -(f.size or 0))):
            if entry.cached:
                state = "cached"
            elif entry.done:
                state = "✓ verified" if entry.verified else ("✓ done" if entry.verified is None else "✗ mismatch")
            elif entry.size:
                state = f"{entry.downloaded / entry.size * 100:.0f}%"
            else:
                state = "…"
            QTreeWidgetItem(self.files, [entry.name, human_bytes(entry.size), state])

    def _on_job(self, repo_id: str) -> None:
        if repo_id == self.repo_id:
            self.refresh()

    def _on_finished(self, repo_id: str, ok: bool) -> None:
        if repo_id == self.repo_id:
            self.refresh()
            if ok:
                self.downloaded.emit()

    def _start(self) -> None:
        if self.repo_id is None:
            return
        if getattr(self.summary, "gated", False) and not self._has_token():
            self.ctx.toast("This model is gated: accept its license on huggingface.co and add your token in "
                           "Settings first.", "warning")
            return
        self.ctx.downloads.start(self.repo_id, summary=self.summary)
        self.refresh()

    @staticmethod
    def _has_token() -> bool:
        from noelle.app.services import hf_client

        return bool(hf_client.get_token())

    def _open_model_page(self) -> None:
        if not self.repo_id:
            return
        job = self.ctx.downloads.job(self.repo_id)
        url = (job.error_url if job is not None and job.error_url else
               f"https://huggingface.co/{self.repo_id}")
        from noelle.app.ui.widgets.common import open_external_url

        open_external_url(url)

    def _reveal(self) -> None:
        record = self.ctx.registry.get_model(self.repo_id or "", source="hub")
        if record and record.local_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(record.local_path))
