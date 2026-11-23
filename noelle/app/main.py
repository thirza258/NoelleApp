"""Desktop entry point (task T-101)."""

from __future__ import annotations

import importlib
import logging
import logging.handlers
import os
import sys

from noelle import APP_ID, APP_NAME, __version__, paths
from noelle.config import get_settings


def _configure_logging() -> None:
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(paths.logs_dir() / "app.log", maxBytes=2_000_000, backupCount=3,
                                                   encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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

    settings = get_settings()
    registry = get_registry()
    runtime = RuntimeClient(settings)
    hardware = HardwareManager(runtime)
    downloads = DownloadManager(settings, registry)
    server = ServerManager(settings, runtime)
    return AppContext(settings=settings, registry=registry, hf=HFClient(settings.get("hub.endpoint")),
                      runtime=runtime, hardware=hardware, downloads=downloads, server=server)


def run(argv: list[str] | None = None) -> int:
    paths.ensure_dirs()
    _configure_logging()
    _apply_environment()

    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from noelle.app import icons, theme
    from noelle.app.ui.main_window import MainWindow

    # QtWebEngine (sandboxed model-card viewer) must be initialised before QApplication exists.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    try:
        importlib.import_module("PySide6.QtWebEngineWidgets")
    except ImportError:
        pass
    QCoreApplication.setOrganizationName(APP_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(__version__)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication([sys.argv[0], *(argv or [])])
    app.setDesktopFileName(APP_ID)
    app.setWindowIcon(icons.app_icon())
    theme.apply(app)

    ctx = build_context()
    window = MainWindow(ctx)
    window.show()
    ctx.runtime.start()
    ctx.hardware.start()
    window.start()
    logging.getLogger("noelle").info("Noelle %s started", __version__)
    return app.exec()


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
