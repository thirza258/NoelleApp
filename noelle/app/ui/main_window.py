"""Primary window: navigation bar, stacked pages, crash banner and telemetry bar."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from noelle import APP_NAME
from noelle.app import icons
from noelle.app.context import AppContext
from noelle.app.ui.navigation_bar import NavigationBar
from noelle.app.ui.widgets.common import Banner, Toast, open_external_url, widget_alive
from noelle.app.ui.widgets.vram_meter import TelemetryBar

log = logging.getLogger("noelle.ui")


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
        ctx.open_settings = self.open_settings

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
        self.update_banner = Banner("info", "")
        self.update_banner.hide()
        banner_layout.addWidget(self.update_banner)
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
        from noelle.app.ui.pages.evaluate_page import EvaluatePage
        from noelle.app.ui.pages.knowledge_page import KnowledgePage
        from noelle.app.ui.pages.mission_page import MissionPage
        from noelle.app.ui.pages.model_detail_page import ModelDetailPage
        from noelle.app.ui.pages.server_page import ServerPage
        from noelle.app.ui.pages.storage_page import StoragePage
        from noelle.app.ui.pages.training_page import TrainingPage
        from noelle.app.ui.pages.workflow_page import WorkflowPage

        self.pages: dict[str, QWidget] = {
            "discover": DiscoveryPage(ctx),
            "mission": MissionPage(ctx),
            "workspace": ModelDetailPage(ctx),
            "knowledge": KnowledgePage(ctx),
            "evaluate": EvaluatePage(ctx),
            "training": TrainingPage(ctx),
            "workflow": WorkflowPage(ctx),
            "compare": ComparativePage(ctx),
            "storage": StoragePage(ctx),
            "server": ServerPage(ctx),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        ctx.extras["evaluate_page"] = self.pages["evaluate"]
        self.pages["discover"].model_selected.connect(self.open_model)  # type: ignore[attr-defined]
        self.pages["knowledge"].collections_changed.connect(self._knowledge_changed)  # type: ignore[attr-defined]

        ctx.runtime.models_changed.connect(self._record_model_use)
        ctx.runtime.state_changed.connect(self._on_runtime_state)
        ctx.runtime.worker_crashed.connect(self._on_worker_crashed)
        ctx.hardware.hardware_telemetry_updated.connect(self.telemetry.update_metrics)
        ctx.server.state_changed.connect(lambda s: self.telemetry.set_server_state(s, ctx.server.port))
        ctx.downloads.job_updated.connect(lambda _r: self._update_download_indicator())
        ctx.downloads.job_finished.connect(self._on_download_finished)
        self.telemetry.set_runtime_state(ctx.runtime.state)

        from noelle.app.ui.navigation_bar import PAGES

        for index, key in enumerate([key for key, _text, _icon in PAGES][:9], start=1):
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
        QTimer.singleShot(2500, self._check_updates)
        if not self.ctx.registry.list_models():
            QTimer.singleShot(600, self.import_cached_models)  # first run: show models already on disk

    def import_cached_models(self) -> None:
        from noelle.app.services.library_import import import_cached_models
        from noelle.app.services.task_runner import runner

        def done(added: list[str]) -> None:
            if added:
                self.toast(f"Found {len(added)} model(s) already on this machine — see Discover › Downloaded.", "success")
                for key in ("discover", "evaluate", "compare", "storage"):
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
        # Qt deletes a faded-out toast; its Python wrapper then raises on any attribute access
        # (libshiboken: Internal C++ object already deleted). Check validity before isVisible().
        self._toasts = [t for t in self._toasts if widget_alive(t) and t.isVisible()]
        toast = Toast(self.centralWidget(), text, kind)
        toast.destroyed.connect(lambda _obj=None, t=toast: self._forget_toast(t))
        offset = sum(t.height() + 8 for t in self._toasts)
        toast.show_at_corner(offset)
        self._toasts.append(toast)

    def _forget_toast(self, toast: Toast) -> None:
        try:
            self._toasts.remove(toast)
        except ValueError:
            pass

    def open_settings(self) -> None:
        from noelle.app.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self.ctx, self)
        if dialog.exec():
            self._refresh_account()
            if dialog.theme_changed:
                answer = QMessageBox.question(self, "Restart Noelle?",
                                              "The new theme is applied when Noelle restarts. Restart now?",
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if answer == QMessageBox.StandardButton.Yes:
                    self.restart_app()
                    return
            if dialog.runtime_restart_needed:
                self.toast("Runtime settings changed — restarting the model runtime.", "info")
                self._restart_runtime()

    def restart_app(self) -> None:
        """Close this window (with the usual download check) and start a fresh instance."""
        self._relaunch = True
        if not self.close():
            self._relaunch = False

    # ------------------------------------------------------------ handlers
    def _check_updates(self) -> None:
        from noelle.app.services import update_checker
        from noelle.app.services.task_runner import runner

        if self.ctx.settings.get("privacy.local_only", False):
            return
        if self.ctx.settings.get("updates.check_app") is not True:
            return
        scheduler = update_checker.UpdateScheduler(self.ctx.settings)
        if not scheduler.should_check("app"):
            return

        def done(update: Any) -> None:
            scheduler.mark_checked("app")
            if update is None:
                return
            self.update_banner.set_text(f"<b>New Noelle version available:</b> {update.version}")
            self.update_banner.clear_actions()
            self.update_banner.add_action("View changes", lambda: open_external_url(update.url), "primary")
            self.update_banner.add_action("Skip this version", lambda: self._skip_update(update.version))
            self.update_banner.add_action("Later", self._hide_update_banner)
            self.update_banner.show()
            self._banner_host.show()

        runner().submit(update_checker.check_app,
                        skipped_version=self.ctx.settings.get("updates.skipped_version"),
                        include_prereleases=bool(self.ctx.settings.get("updates.beta", False)),
                        on_result=done, on_error=lambda _m: None)

    def _skip_update(self, version: str) -> None:
        self.ctx.settings.set("updates.skipped_version", version)
        self._hide_update_banner()

    def _hide_update_banner(self) -> None:
        self.update_banner.hide()
        if not self.crash_banner.isVisible():
            self._banner_host.hide()

    def _refresh_account(self) -> None:
        from noelle.app.services import hf_client
        from noelle.app.services.task_runner import runner

        if not hf_client.get_token():
            self.nav.set_account(None)
            return
        runner().submit(self.ctx.hf.whoami, on_result=lambda info: self.nav.set_account(info.get("name")),
                        on_error=lambda _m: self.nav.set_account(None))

    def _knowledge_changed(self) -> None:
        """Keep the Workspace chat's collection picker in step with the Knowledge page."""
        workspace = self.pages["workspace"]
        playground = getattr(workspace, "_playground", None)
        refresh = getattr(playground, "refresh_collections", None)
        if callable(refresh):
            refresh()
        knowledge_tab = getattr(workspace, "knowledge_tab", None)
        if hasattr(knowledge_tab, "refresh"):
            knowledge_tab.refresh()

    def _record_model_use(self, loaded: list[dict[str, Any]]) -> None:
        """Remember when each model was last loaded (Storage Advisor, recent models)."""
        for entry in loaded or []:
            model_id = entry.get("model_id") if isinstance(entry, dict) else None
            if model_id:
                self.ctx.registry.touch_model(model_id)

    def _on_runtime_state(self, state: str) -> None:
        self.telemetry.set_runtime_state(state)
        if state in ("ready", "warming", "starting"):
            self.crash_banner.hide()
            if not self.update_banner.isVisible():
                self._banner_host.hide()
        elif state == "unresponsive":
            self.crash_banner.set_text("<b>The model runtime stopped responding.</b> It may be stuck under heavy "
                                       "memory pressure. You can restart it; the app itself is unaffected.")
            self._banner_host.show()

    def _on_worker_crashed(self, exit_code: int, diagnosis: str) -> None:
        log.error("runtime crash reported to the user (code %s): %s", exit_code, diagnosis)
        self.crash_banner.set_text(f"<b>The model runtime stopped unexpectedly.</b> {diagnosis} "
                                   "Your work in the app is safe — restart the runtime to continue.")
        self._banner_host.show()

    def _restart_runtime(self) -> None:
        log.info("restarting the model runtime from the UI")
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
        training = self.ctx.training.active_runs() if self.ctx.training is not None else []
        if training:
            answer = QMessageBox.question(
                self, "Stop training and quit?",
                f"{len(training)} training run(s) are active. They will be stopped after the current step and can be "
                "resumed from their last checkpoint.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        log.info("shutting down: %d active download(s), %d training run(s), server=%s, runtime=%s",
                 len(active), len(training), self.ctx.server.state, self.ctx.runtime.state)
        for page in self.pages.values():
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self.ctx.downloads.shutdown()
        if self.ctx.training is not None:
            self.ctx.training.shutdown()
        self.ctx.server.stop()
        self.ctx.hardware.stop()
        self.ctx.runtime.stop()
        log.info("shutdown complete")
        event.accept()
        if getattr(self, "_relaunch", False):
            from PySide6.QtCore import QProcess

            from noelle.runtime.client import worker_command

            command = worker_command()  # the frozen executable, or ``python -m noelle``
            QProcess.startDetached(command[0], command[1:])

