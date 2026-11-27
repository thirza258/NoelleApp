"""Desktop entry point (task T-101)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from noelle import APP_ID, APP_NAME, __version__, logging_setup, paths
from noelle.config import get_settings


def _configure_logging() -> None:
    logging_setup.configure_desktop_logging()
    _install_excepthook()


def _install_excepthook() -> None:
    """Exceptions raised inside Qt slots would otherwise only reach a console the GUI build lacks."""
    log = logging.getLogger("noelle")

    def hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))  # type: ignore[arg-type]
        sys.__excepthook__(exc_type, exc, tb)  # type: ignore[arg-type]

    if sys.excepthook is not hook:
        sys.excepthook = hook


def _apply_environment() -> None:
    """Must run before huggingface_hub is imported: it reads HF_HOME at import time."""
    settings = get_settings()
    hf_home = settings.get("storage.hf_home")
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)
        os.environ.pop("HF_HUB_CACHE", None)
    endpoint = settings.get("hub.endpoint")
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def build_context():  # noqa: ANN201 - returns AppContext
    from noelle.app.context import AppContext
    from noelle.app.ipc.runtime_client import RuntimeClient
    from noelle.app.services.download_manager import DownloadManager
    from noelle.app.services.hardware_manager import HardwareManager
    from noelle.app.services.hf_client import HFClient
    from noelle.app.services.model_registry import get_registry
    from noelle.app.services.server_manager import ServerManager

    from noelle.app import i18n

    settings = get_settings()
    i18n.set_language(settings.get("ui.language"))
    registry = get_registry()
    runtime = RuntimeClient(settings)
    # Workflow nodes and the evaluation labs resolve providers through the global registry.
    # Register the real worker-backed provider here so they never fall back to an unconnected one.
    from noelle.providers.local_provider import LocalProvider
    from noelle.providers.registry import register_provider

    register_provider(LocalProvider(runtime_client=runtime))
    hardware = HardwareManager(runtime)
    downloads = DownloadManager(settings, registry)
    server = ServerManager(settings, runtime)
    return AppContext(settings=settings, registry=registry, hf=HFClient(settings.get("hub.endpoint")),
                      runtime=runtime, hardware=hardware, downloads=downloads, server=server)


def _clean_stale_token_handoffs() -> None:
    """Remove installer hand-over files left behind if Setup or the import process was killed."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return
    folder = Path(appdata) / "Noelle"
    try:
        entries = list(folder.glob("pending-hf-token-*.txt"))
    except OSError:
        return
    for entry in entries:
        try:
            entry.unlink()
            logging.getLogger("noelle").info("Removed a stale installer token hand-over file.")
        except OSError:
            pass


def _prepare_storage() -> OSError | None:
    """Create the data directories; return the error instead of crashing before any UI exists."""
    try:
        paths.ensure_dirs()
    except OSError as exc:
        return exc
    try:
        _configure_logging()
    except OSError as exc:  # e.g. a full disk: the app can still run, but say so on stderr
        print(f"Noelle could not open its log file: {exc}", file=sys.stderr)
    _clean_stale_token_handoffs()
    return None


def run(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import QCoreApplication, Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox

    from noelle.app import icons, theme
    from noelle.app.ui.main_window import MainWindow

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    QCoreApplication.setOrganizationName(APP_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(__version__)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication([sys.argv[0], *(argv or [])])
    app.setDesktopFileName(APP_ID)
    app.setWindowIcon(icons.app_icon())
    theme.apply(app)

    problem = _prepare_storage()
    if problem is not None:
        QMessageBox.critical(
            None, "Noelle cannot start",
            f"Noelle could not create its data folder at {paths.home()}.\n\n{problem}\n\n"
            "Check the folder permissions or set the NOELLE_HOME environment variable to a writable folder.")
        return 1
    _apply_environment()
    logging.getLogger("noelle").info(
        "Noelle %s starting (python %s, frozen=%s, home=%s, logs=%s)",
        __version__, sys.version.split()[0], getattr(sys, "frozen", False), paths.home(), paths.logs_dir())

    ctx = build_context()
    window = MainWindow(ctx)
    window.show()
    if ctx.settings.get("runtime.autostart", False):
        ctx.runtime.start()
    ctx.hardware.start()
    window.start()
    from noelle.app.ui.setup_assistant import run_assistant, should_run

    if "--setup" in (argv or []) or should_run(ctx.settings):
        QTimer.singleShot(350, lambda: run_assistant(ctx, window))
    logging.getLogger("noelle").info("Noelle %s started", __version__)
    return app.exec()


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
