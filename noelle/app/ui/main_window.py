"""Primary window: navigation bar, stacked pages, crash banner and telemetry bar."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from noelle import APP_NAME
from noelle.app import icons
from noelle.app.context import AppContext
from noelle.app.ui.navigation_bar import NavigationBar
from noelle.app.ui.widgets.common import Banner, Toast
from noelle.app.ui.widgets.vram_meter import TelemetryBar


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(f"{APP_NAME} — AI Studio")
        self.setWindowIcon(icons.app_icon())
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self._toasts: list[Toast] = []

        ctx.toast = self.toast
        ctx.open_model = self.open_model
        ctx.navigate = self.navigate

        root = QWidget()
        root.setObjectName("AppRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.nav = NavigationBar()
        self.nav.page_requested.connect(self.navigate)
        self.nav.settings_requested.connect(self.open_settings)
        self.nav.downloads_requested.connect(lambda: self.navigate("storage"))
        layout.addWidget(self.nav)

        self.crash_banner = Banner("danger", "")
        self.crash_banner.add_action("Restart runtime", self._restart_runtime, "primary")
        self.crash_banner.add_action("Show log", self._show_worker_log)
        self.crash_banner.hide()
        banner_host = QWidget()
        banner_layout = QVBoxLayout(banner_host)
        banner_layout.setContentsMargins(16, 10, 16, 0)
        banner_layout.addWidget(self.crash_banner)
        self._banner_host = banner_host
        banner_host.hide()
        layout.addWidget(banner_host)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.telemetry = TelemetryBar()
        self.telemetry.restart_requested.connect(self._restart_runtime)
        self.telemetry.server_clicked.connect(lambda: self.navigate("server"))
        layout.addWidget(self.telemetry)
        self.setCentralWidget(root)

        from noelle.app.ui.pages.comparative_page import ComparativePage
        from noelle.app.ui.pages.discovery_page import DiscoveryPage
        from noelle.app.ui.pages.model_detail_page import ModelDetailPage
        from noelle.app.ui.pages.server_page import ServerPage
        from noelle.app.ui.pages.storage_page import StoragePage

        self.pages: dict[str, QWidget] = {
            "discover": DiscoveryPage(ctx),
            "workspace": ModelDetailPage(ctx),
            "compare": ComparativePage(ctx),
            "storage": StoragePage(ctx),
            "server": ServerPage(ctx),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        self.pages["discover"].model_selected.connect(self.open_model)  # type: ignore[attr-defined]

        ctx.runtime.state_changed.connect(self._on_runtime_state)
        ctx.runtime.worker_crashed.connect(self._on_worker_crashed)
        ctx.hardware.hardware_telemetry_updated.connect(self.telemetry.update_metrics)
        ctx.server.state_changed.connect(lambda s: self.telemetry.set_server_state(s, ctx.server.port))
        ctx.downloads.job_updated.connect(lambda _r: self._update_download_indicator())
        ctx.downloads.job_finished.connect(self._on_download_finished)
        self.telemetry.set_runtime_state(ctx.runtime.state)

        for index, key in enumerate(("discover", "workspace", "compare", "storage", "server"), start=1):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{index}"), self)
            shortcut.activated.connect(lambda k=key: self.navigate(k))
        self.navigate("discover")

    # ----------------------------------------------------------------- API
    def start(self) -> None:
        for page in self.pages.values():
            start = getattr(page, "start", None)
            if callable(start):
                start()
        QTimer.singleShot(400, self._refresh_account)
        if not self.ctx.registry.list_models():
            QTimer.singleShot(600, self.import_cached_models)  # first run: show models already on disk

    def import_cached_models(self) -> None:
        from noelle.app.services.library_import import import_cached_models
        from noelle.app.services.task_runner import runner

        def done(added: list[str]) -> None:
            if added:
                self.toast(f"Found {len(added)} model(s) already on this machine — see Discover › Downloaded.", "success")
                for key in ("discover", "compare", "storage"):
                    refresh = getattr(self.pages[key], "on_library_changed", None)
                    if callable(refresh):
                        refresh()

        runner().submit(import_cached_models, self.ctx.registry, on_result=done)

    def navigate(self, key: str) -> None:
        page = self.pages.get(key)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        self.nav.set_current(key)
        activated = getattr(page, "on_activated", None)
        if callable(activated):
            activated()

    def open_model(self, model: Any) -> None:
        self.pages["workspace"].set_model(model)  # type: ignore[attr-defined]
        self.navigate("workspace")

    def toast(self, text: str, kind: str = "info") -> None:
        self._toasts = [t for t in self._toasts if t.isVisible()]
        toast = Toast(self.centralWidget(), text, kind)
        offset = sum(t.height() + 8 for t in self._toasts)
        toast.show_at_corner(offset)
        self._toasts.append(toast)

    def open_settings(self) -> None:
        from noelle.app.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self.ctx, self)
        if dialog.exec():
            self._refresh_account()
            if dialog.runtime_restart_needed:
                self.toast("Runtime settings changed — restarting the model runtime.", "info")
                self._restart_runtime()

    # ------------------------------------------------------------ handlers
    def _refresh_account(self) -> None:
        from noelle.app.services import hf_client
        from noelle.app.services.task_runner import runner

        if not hf_client.get_token():
            self.nav.set_account(None)
            return
        runner().submit(self.ctx.hf.whoami, on_result=lambda info: self.nav.set_account(info.get("name")),
                        on_error=lambda _m: self.nav.set_account(None))

    def _on_runtime_state(self, state: str) -> None:
        self.telemetry.set_runtime_state(state)
        if state in ("ready", "warming", "starting"):
            self._banner_host.hide()
        elif state == "unresponsive":
            self.crash_banner.set_text("<b>The model runtime stopped responding.</b> It may be stuck under heavy "
                                       "memory pressure. You can restart it; the app itself is unaffected.")
            self._banner_host.show()

    def _on_worker_crashed(self, exit_code: int, diagnosis: str) -> None:
        self.crash_banner.set_text(f"<b>The model runtime stopped unexpectedly.</b> {diagnosis} "
                                   "Your work in the app is safe — restart the runtime to continue.")
        self._banner_host.show()

    def _restart_runtime(self) -> None:
        self._banner_host.hide()
        self.ctx.runtime.restart()

    def _show_worker_log(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Runtime log")
        box.setText("Last lines written by the model runtime:")
        box.setDetailedText(self.ctx.runtime.stderr_tail() or "(empty)")
        box.exec()

    def _update_download_indicator(self) -> None:
        jobs = [j for j in self.ctx.downloads.jobs.values() if j.active]
        if not jobs:
            self.nav.set_active_downloads(0)
            return
        total = sum(j.download_bytes for j in jobs)
        done = sum(j.downloaded_bytes for j in jobs)
        self.nav.set_active_downloads(len(jobs), (done / total) if total else None)

    def _on_download_finished(self, repo_id: str, ok: bool) -> None:
        self._update_download_indicator()
        job = self.ctx.downloads.job(repo_id)
        if ok:
            self.toast(f"Downloaded {repo_id}", "success")
        elif job is not None and job.status == "error":
            self.toast(f"Download failed: {job.error or repo_id}", "error")

    # --------------------------------------------------------------- close
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        active = self.ctx.downloads.active_repos()
        if active:
            answer = QMessageBox.question(
                self, "Quit Noelle?",
                f"{len(active)} download(s) are in progress. They will pause and can be resumed next time.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        for page in self.pages.values():
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self.ctx.downloads.shutdown()
        self.ctx.server.stop()
        self.ctx.hardware.stop()
        self.ctx.runtime.stop()
        event.accept()

