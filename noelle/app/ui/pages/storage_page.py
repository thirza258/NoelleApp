"""Storage page: downloads, Hugging Face cache governance, Noelle data and cache relocation."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle import paths
from noelle.app.context import AppContext
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import Banner, Panel, Spinner, StatTile, button, clear_layout, label, time_ago
from noelle.runtime.preflight import format_bytes

ROLE_KIND = Qt.ItemDataRole.UserRole
ROLE_ID = Qt.ItemDataRole.UserRole + 1
ROLE_PATH = Qt.ItemDataRole.UserRole + 2


class DownloadRow(QWidget):
    def __init__(self, ctx: AppContext, repo_id: str) -> None:
        super().__init__()
        self.ctx = ctx
        self.repo_id = repo_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)
        self.name = label(repo_id)
        self.name.setMinimumWidth(240)
        layout.addWidget(self.name)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        layout.addWidget(self.bar, 1)
        self.status = label("", "faint")
        self.status.setMinimumWidth(220)
        layout.addWidget(self.status)
        self.pause = button("Pause", "pause", "ghost")
        self.pause.clicked.connect(lambda: ctx.downloads.pause(repo_id))
        layout.addWidget(self.pause)
        self.resume = button("Resume", "play", "ghost")
        self.resume.clicked.connect(lambda: ctx.downloads.resume(repo_id))
        layout.addWidget(self.resume)
        self.cancel = button("Cancel", "x", "ghost")
        self.cancel.clicked.connect(lambda: ctx.downloads.cancel(repo_id))
        layout.addWidget(self.cancel)
        self.refresh()

    def refresh(self) -> None:
        job = self.ctx.downloads.job(self.repo_id)
        if job is None:
            return
        self.bar.setValue(int(job.fraction * 1000))
        speed = f" · {format_bytes(job.speed_bps)}/s" if job.speed_bps else ""
        text = {"done": "Done", "error": f"Failed: {job.error or ''}", "paused": "Paused",
                "cancelled": "Cancelled", "queued": "Queued", "planning": "Resolving files…"}.get(job.status)
        self.status.setText(text or f"{format_bytes(job.downloaded_bytes)} / {format_bytes(job.download_bytes)}{speed}")
        self.status.setToolTip(job.error or "")
        self.pause.setVisible(job.active)
        self.resume.setVisible(job.status in ("paused", "error", "cancelled"))
        self.cancel.setVisible(job.status not in ("done", "cancelled"))


class StoragePage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.report: Any = None
        self._rows: dict[str, DownloadRow] = {}
        self._scanning = False

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Storage", "h2"))
        self.cache_path = label("", "muted", selectable=True)
        titles.addWidget(self.cache_path)
        head.addLayout(titles, 1)
        self.spinner = Spinner(16)
        head.addWidget(self.spinner)
        rescan = button("Rescan", "refresh", "ghost")
        rescan.clicked.connect(self.rescan)
        head.addWidget(rescan)
        self.prune_incomplete_btn = button("Prune partial downloads", "trash")
        self.prune_incomplete_btn.clicked.connect(self._prune_incomplete)
        head.addWidget(self.prune_incomplete_btn)
        self.prune_detached_btn = button("Prune old revisions", "trash")
        self.prune_detached_btn.clicked.connect(self._prune_detached)
        head.addWidget(self.prune_detached_btn)
        add_cached = button("Add cached models", "box", "ghost",
                            tooltip="Register complete models already in the Hugging Face cache (offline)")
        add_cached.clicked.connect(self._import_cached)
        head.addWidget(add_cached)
        external = button("Find Ollama / LM Studio models", "box", "ghost",
                          tooltip="Import GGUF models already installed by another app (read-only, no copying)")
        external.clicked.connect(self._import_external)
        head.addWidget(external)
        relocate = button("Relocate cache…", "folder", "primary")
        relocate.clicked.connect(self._relocate)
        head.addWidget(relocate)
        root.addLayout(head)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_cache = StatTile("Hugging Face cache")
        self.tile_reclaim = StatTile("Reclaimable")
        self.tile_noelle = StatTile("Noelle data")
        self.tile_disk = StatTile("Free disk")
        for tile in (self.tile_cache, self.tile_reclaim, self.tile_noelle, self.tile_disk):
            tiles.addWidget(tile)
        root.addLayout(tiles)

        self.advisor = Panel()
        advisor_layout = QVBoxLayout(self.advisor)
        advisor_layout.setContentsMargins(14, 10, 14, 10)
        advisor_layout.setSpacing(6)
        advisor_head = QHBoxLayout()
        advisor_head.addWidget(label("STORAGE ADVISOR", "eyebrow"))
        advisor_head.addStretch(1)
        self.advisor_total = label("", "h3")
        advisor_head.addWidget(self.advisor_total)
        advisor_layout.addLayout(advisor_head)
        self.advisor_breakdown = label("", "muted", wrap=True)
        advisor_layout.addWidget(self.advisor_breakdown)
        self.advisor_rows = QVBoxLayout()
        self.advisor_rows.setSpacing(4)
        advisor_layout.addLayout(self.advisor_rows)
        self.advisor.hide()
        root.addWidget(self.advisor)

        self.warnings = Banner("warning", "")
        self.warnings.hide()
        root.addWidget(self.warnings)

        self.downloads_panel = Panel()
        dl_layout = QVBoxLayout(self.downloads_panel)
        dl_layout.setContentsMargins(14, 10, 14, 10)
        dl_layout.addWidget(label("DOWNLOADS", "eyebrow"))
        self.downloads_list = QVBoxLayout()
        dl_layout.addLayout(self.downloads_list)
        self.downloads_panel.hide()
        root.addWidget(self.downloads_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Repository", "Size", "Files", "Revisions", "Last used", "Status"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, width in ((1, 90), (2, 60), (3, 80), (4, 100), (5, 190)):
            self.tree.setColumnWidth(col, width)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemDoubleClicked.connect(self._open_item)
        splitter.addWidget(self.tree)
        side = Panel()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 12, 14, 12)
        side_layout.addWidget(label("NOELLE DATA", "eyebrow"))
        self.usage_rows = QVBoxLayout()
        side_layout.addLayout(self.usage_rows)
        side_layout.addStretch(1)
        open_home = button("Open data folder", "folder", "ghost")
        open_home.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.home()))))
        side_layout.addWidget(open_home)
        splitter.addWidget(side)
        splitter.setSizes([900, 300])
        root.addWidget(splitter, 1)

        ctx.downloads.job_added.connect(self._on_download)
        ctx.downloads.job_updated.connect(self._on_download)
        ctx.downloads.job_finished.connect(lambda *_: self.rescan())

    # ------------------------------------------------------------------ scan
    def on_activated(self) -> None:
        self.rescan()

    def cache_dir(self) -> Path:
        return paths.hf_hub_cache(self.ctx.settings.get("storage.hf_home"))

    def rescan(self) -> None:
        if self._scanning:
            return
        from noelle.app.services import storage_service

        self._scanning = True
        self.spinner.start()
        cache = self.cache_dir()
        self.cache_path.setText(f"Hugging Face cache: {cache}")

        def work() -> tuple[Any, dict[str, int], dict[str, int]]:
            report = storage_service.scan_hf_cache(cache)
            usage = storage_service.noelle_home_usage()
            disk = storage_service.disk_usage(cache if cache.exists() else paths.home())
            return report, usage, disk

        def done(result: tuple[Any, dict[str, int], dict[str, int]]) -> None:
            self._scanning = False
            self.spinner.stop()
            self.report, usage, disk = result
            self._render(usage, disk)

        def failed(message: str) -> None:
            self._scanning = False
            self.spinner.stop()
            self.warnings.set_text(f"Could not scan the cache: {message}")
            self.warnings.show()

        runner().submit(work, on_result=done, on_error=failed)

    def _render(self, usage: dict[str, int], disk: dict[str, int]) -> None:
        report = self.report
        registered = {r.repo_id for r in self.ctx.registry.list_models()}
        active = self.ctx.downloads.active_repos()
        self.tile_cache.set(format_bytes(report.total_size), f"{len(report.repos)} repositories · {report.scanner} scan")
        self.tile_reclaim.set(format_bytes(report.reclaimable_bytes),
                              f"{format_bytes(report.detached_bytes)} old revisions · "
                              f"{format_bytes(report.incomplete_bytes)} partial")
        self.tile_noelle.set(format_bytes(usage.get("total", 0)), str(paths.home()).replace(str(Path.home()), "~"))
        self.tile_disk.set(format_bytes(disk.get("free", 0)), f"of {format_bytes(disk.get('total', 0))}")
        self.prune_incomplete_btn.setEnabled(report.incomplete_bytes > 0)
        self.prune_detached_btn.setEnabled(report.detached_bytes > 0)
        self._render_advice(usage)
        if report.warnings:
            self.warnings.set_text("<br>".join(report.warnings[:5]))
            self.warnings.show()
        else:
            self.warnings.hide()
        self.tree.clear()
        for repo in sorted(report.repos, key=lambda r: r.size_bytes, reverse=True):
            status = []
            if repo.repo_id in active:
                status.append("downloading")
            if repo.repo_id in registered:
                status.append("in Noelle")
            detached = sum(1 for r in repo.revisions if r.detached)
            if detached:
                status.append(f"{detached} old revision{'s' if detached > 1 else ''}")
            if repo.incomplete:
                status.append(f"{len(repo.incomplete)} partial file{'s' if len(repo.incomplete) > 1 else ''}")
            if repo.corrupted:
                status.append("corrupted")
            item = QTreeWidgetItem([repo.repo_id, format_bytes(repo.size_bytes), str(repo.nb_files),
                                    str(len(repo.revisions)), time_ago(repo.last_accessed) if repo.last_accessed else "—",
                                    " · ".join(status)])
            item.setData(0, ROLE_KIND, "repo")
            item.setData(0, ROLE_ID, repo.repo_id)
            item.setData(0, ROLE_PATH, str(repo.path))
            item.setToolTip(0, f"{repo.repo_type} · {repo.path}")
            for revision in repo.revisions:
                refs = ", ".join(revision.refs) if revision.refs else "detached"
                child = QTreeWidgetItem([f"{revision.commit[:10]}  ({refs})", format_bytes(revision.size_bytes),
                                         str(revision.nb_files), "",
                                         time_ago(revision.last_modified) if revision.last_modified else "",
                                         f"frees {format_bytes(revision.exclusive_bytes)}" if revision.detached else ""])
                child.setData(0, ROLE_KIND, "revision")
                child.setData(0, ROLE_ID, revision.commit)
                child.setData(0, ROLE_PATH, str(revision.snapshot_path))
                child.setData(1, ROLE_ID, repo.repo_id)
                item.addChild(child)
            self.tree.addTopLevelItem(item)
        clear_layout(self.usage_rows)
        total = max(usage.get("total", 0), 1)
        for key in ("models", "environments", "vector_store", "projects", "logs", "other"):
            if key not in usage:
                continue
            row = QHBoxLayout()
            row.addWidget(label(key.replace("_", " "), "muted"))
            row.addStretch(1)
            row.addWidget(label(format_bytes(usage[key]), "mono"))
            self.usage_rows.addLayout(row)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(usage[key] / total * 1000))
            self.usage_rows.addWidget(bar)

    def _render_advice(self, usage: dict[str, int]) -> None:
        from noelle.app.services.storage_advisor import advise

        advice = advise(self.report, self.ctx.registry.list_models(), usage)
        self.advisor_total.setText(f"Potential cleanup: {format_bytes(advice.potential_bytes)}")
        self.advisor_breakdown.setText(
            f"Active models {format_bytes(advice.active_bytes)} · not used for {advice.unused_days}+ days "
            f"{format_bytes(advice.unused_bytes)} · partial downloads {format_bytes(advice.partial_bytes)} · "
            f"old revisions {format_bytes(advice.old_revision_bytes)} · local builds "
            f"{format_bytes(advice.derived_bytes)}")
        clear_layout(self.advisor_rows)
        for item in advice.unused[:5]:
            row = QHBoxLayout()
            row.addWidget(label(item.repo_id), 1)
            row.addWidget(label(f"{format_bytes(item.size_bytes)} · not used for {item.days} days", "faint"))
            delete = button("Delete…", "trash", "ghost")
            delete.clicked.connect(lambda _=False, repo=item.repo_id: self._delete_repo(repo))
            row.addWidget(delete)
            self.advisor_rows.addLayout(row)
        self.advisor.setVisible(bool(self.report.repos))

    # -------------------------------------------------------------- downloads
    def _on_download(self, repo_id: str) -> None:
        row = self._rows.get(repo_id)
        if row is None:
            row = DownloadRow(self.ctx, repo_id)
            self._rows[repo_id] = row
            self.downloads_list.addWidget(row)
        row.refresh()
        self.downloads_panel.setVisible(bool(self._rows))

    # ---------------------------------------------------------------- actions
    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def on_library_changed(self) -> None:
        self.rescan()

    def _run_storage_op(self, fn: Any, *args: Any, describe: Any, **kwargs: Any) -> None:
        """Run a (possibly slow) delete/prune on the thread pool, then refresh."""
        self.spinner.start()
        runner().submit(fn, *args, **kwargs, on_result=lambda result: (self.spinner.stop(),
                                                                       self._after_delete(describe(result))),
                        on_error=lambda message: (self.spinner.stop(),
                                                  QMessageBox.warning(self, "Storage", f"Operation failed: {message}")))

    def _import_cached(self) -> None:
        from noelle.app.services.library_import import import_cached_models

        def done(added: list[str]) -> None:
            self.ctx.toast(f"Added {len(added)} cached model(s) to the library." if added else
                           "No new complete models found in the cache.", "success" if added else "info")
            self.rescan()

        runner().submit(import_cached_models, self.ctx.registry, self.cache_dir(), on_result=done)

    def _import_external(self) -> None:
        from noelle.app.services import external_models

        def work() -> list[str]:
            if not self.ctx.settings.get("library.external_sources.ollama", True) and \
                    not self.ctx.settings.get("library.external_sources.lmstudio", True):
                return []
            found = external_models.discover_external_models(
                ollama=self.ctx.settings.get("library.ollama_path") or None,
                lmstudio=self.ctx.settings.get("library.lmstudio_path") or None)
            return external_models.register_external_models(self.ctx.registry, found)

        def done(added: list[str]) -> None:
            self.ctx.toast(f"Imported {len(added)} model(s) from Ollama / LM Studio." if added else
                           "No Ollama or LM Studio models were found.", "success" if added else "info")
            self.rescan()

        runner().submit(work, on_result=done,
                        on_error=lambda message: QMessageBox.warning(self, "External models", message))

    def _after_delete(self, message: str) -> None:
        removed = self.ctx.registry.prune_missing()
        self.ctx.toast(message + (f" · {len(removed)} model(s) removed from the library" if removed else ""), "success")
        self.rescan()

    def _prune_incomplete(self) -> None:
        from noelle.app.services import storage_service

        if self.report is None or not self._confirm(
                "Prune partial downloads",
                f"Delete {format_bytes(self.report.incomplete_bytes)} of unfinished download files (*.incomplete)? "
                "Files from downloads that are running right now are skipped."):
            return
        self._run_storage_op(storage_service.prune_incomplete, self.cache_dir(),
                             exclude_repos=self.ctx.downloads.active_repos(),
                             describe=lambda r: f"Freed {format_bytes(r.freed_bytes)} ({len(r.removed)} partial files)")

    def _prune_detached(self) -> None:
        from noelle.app.services import storage_service

        if self.report is None or not self._confirm(
                "Prune old revisions",
                f"Delete revisions that no branch or tag points to ({format_bytes(self.report.detached_bytes)})? "
                "Files shared with current revisions are kept."):
            return
        self._run_storage_op(storage_service.prune_detached, self.cache_dir(),
                             exclude_repos=self.ctx.downloads.active_repos(),
                             describe=lambda r: f"Freed {format_bytes(r.freed_bytes)} ({len(r.deleted)} revisions)")

    def _context_menu(self, pos: Any) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        kind = item.data(0, ROLE_KIND)
        menu.addAction("Show in folder", lambda: self._open_item(item))
        if kind == "repo":
            repo_id = item.data(0, ROLE_ID)
            if self.ctx.registry.get_model(repo_id) is not None:
                menu.addAction("Open in Workspace", lambda: self.ctx.open_model(repo_id))
            menu.addSeparator()
            menu.addAction("Delete repository…", lambda: self._delete_repo(repo_id))
        else:
            menu.addSeparator()
            menu.addAction("Delete this revision…", lambda: self._delete_revision(item.data(1, ROLE_ID),
                                                                               item.data(0, ROLE_ID)))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _open_item(self, item: QTreeWidgetItem, *_args: Any) -> None:
        path = item.data(0, ROLE_PATH)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _delete_repo(self, repo_id: str) -> None:
        from noelle.app.services import storage_service

        if repo_id in self.ctx.downloads.active_repos():
            self.ctx.toast("Pause or cancel its download first.", "warning")
            return
        repo = next((r for r in self.report.repos if r.repo_id == repo_id), None) if self.report else None
        size = format_bytes(repo.size_bytes) if repo else "?"
        if not self._confirm("Delete repository", f"Permanently delete {repo_id} ({size}) from the Hugging Face cache? "
                                                  "Other tools that use this cache will need to download it again."):
            return
        for loaded in self.ctx.runtime.loaded_models:
            if loaded.get("model_id") == repo_id:
                from noelle.runtime.protocol import Command

                self.ctx.runtime.request(Command.UNLOAD_MODEL, {"key": loaded["key"]})
        self._run_storage_op(storage_service.delete_repo, repo_id, self.cache_dir(),
                             describe=lambda freed: f"Deleted {repo_id} · freed {format_bytes(freed)}")

    def _delete_revision(self, repo_id: str, commit: str) -> None:
        from noelle.app.services import storage_service

        if not self._confirm("Delete revision", f"Delete revision {commit[:10]} of {repo_id}? Files shared with other "
                                                "revisions are kept."):
            return
        self._run_storage_op(storage_service.delete_revisions, [commit], self.cache_dir(), repo_id=repo_id,
                             describe=lambda r: f"Freed {format_bytes(r.freed_bytes)}")

    # ------------------------------------------------------------- relocation
    def _relocate(self) -> None:
        from noelle.app.services import storage_service

        if self.ctx.downloads.active_repos():
            self.ctx.toast("Pause running downloads before moving the cache.", "warning")
            return
        current_home = self.cache_dir().parent
        target_parent = QFileDialog.getExistingDirectory(self, "Choose the drive or folder for the model cache")
        if not target_parent:
            return
        new_home = Path(target_parent) / "huggingface"
        size = format_bytes(self.report.total_size) if self.report else "?"
        if not self._confirm(
                "Relocate cache",
                f"Move the whole Hugging Face cache ({size})\n\nfrom  {current_home}\nto    {new_home}\n\n"
                "Noelle will use the new location immediately. Other tools keep using the old path unless you set "
                "HF_HOME for them too. The model runtime restarts afterwards."):
            return
        stop = threading.Event()
        dialog = QProgressDialog("Moving the cache…", "Cancel", 0, 1000, self)
        dialog.setWindowTitle("Relocating cache")
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(stop.set)
        self.ctx.runtime.stop()

        def work(progress: Any) -> Any:
            return storage_service.relocate_cache(current_home, new_home, stop_event=stop,
                                                  progress=lambda done, total, name: progress((done, total, name)))

        def on_progress(data: tuple[int, int, str]) -> None:
            done, total, name = data
            dialog.setValue(int(done / total * 1000) if total else 0)
            dialog.setLabelText(f"Moving {name}…\n{format_bytes(done)} of {format_bytes(total)}")

        def on_done(result: Any) -> None:
            dialog.close()
            old_hub = str(current_home / "hub")
            new_hub = str(Path(result.new_hf_home) / "hub")
            for record in self.ctx.registry.list_models():
                if record.local_path and record.local_path.startswith(old_hub):
                    record.local_path = new_hub + record.local_path[len(old_hub):]
                    self.ctx.registry.upsert_model(record)
            self.ctx.settings.set("storage.hf_home", str(result.new_hf_home))
            os.environ["HF_HOME"] = str(result.new_hf_home)
            os.environ.pop("HF_HUB_CACHE", None)
            self.ctx.runtime.start()
            self.ctx.toast(f"Cache moved ({result.method}, {format_bytes(result.moved_bytes)}). For other tools add: "
                           f"{storage_service.shell_export_line(result.new_hf_home)}", "success")
            self.rescan()

        def on_error(message: str) -> None:
            dialog.close()
            self.ctx.runtime.start()
            QMessageBox.warning(self, "Relocation failed", f"The cache was not moved: {message}")

        runner().submit(work, on_progress=on_progress, on_result=on_done, on_error=on_error)
