"""Central logging configuration for the desktop shell, workers, server and CLI tools.

Every process writes a capped, rotating log file under ``~/.noelle/logs`` (override the data
root with ``NOELLE_HOME``). Dedicated files keep support requests and long sessions readable:

``app.log``            desktop overview (INFO+)
``errors.log``         every desktop ERROR/CRITICAL record
``ipc.log``            worker supervision, readiness, crashes, request refusals
``downloads.log``      desktop download manager decisions
``download-child.log`` per-file download child (checksums, retries)
``credentials.log``    OS credential-store operations (never the token itself)
``hub.log``            Hugging Face Hub metadata requests (never prompts)
``storage.log``        registry / cache / relocation operations
``models.log``         worker model load/unload/eviction
``worker.log``         inference worker process
``server.log``         OpenAI-compatible API server
``server-access.log``  API access log (written by the server itself)
``mcp.log``            MCP stdio server
``web.log`` / ``rag.log`` / ``quantize.log``  worker subsystems
``tasks.log``          UI background-task failures
``ui.log``             UI/notebook warnings
``settings.log``       settings reload/save problems
``debug.log``          opt-in verbose copy of the desktop log (``NOELLE_LOG_DEBUG=1``)

All file handlers use a formatter that redacts token-shaped strings before they reach disk.
Logging must never break the application: if a file cannot be opened, that handler is skipped.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

from noelle import paths
from noelle.runtime.errors import redact_secrets

FORMAT = "%(asctime)s %(levelname)-8s %(name)s [pid %(process)d %(threadName)s] %(message)s"
APP_MAX_BYTES = 2_000_000
APP_BACKUPS = 3
COMPONENT_MAX_BYTES = 1_000_000
COMPONENT_BACKUPS = 2

DESKTOP_COMPONENTS: Mapping[str, str] = {
    "noelle.ipc": "ipc.log",
    "noelle.runtime-client": "ipc.log",
    "noelle.server-manager": "server-manager.log",
    "noelle.downloads": "downloads.log",
    "noelle.credentials": "credentials.log",
    "noelle.setup": "setup.log",
    "noelle.hub": "hub.log",
    "noelle.registry": "storage.log",
    "noelle.storage": "storage.log",
    "noelle.models": "models.log",
    "noelle.tasks": "tasks.log",
    "noelle.ui": "ui.log",
    "noelle.notebook": "ui.log",
    "noelle.settings": "settings.log",
}

WORKER_COMPONENTS: Mapping[str, str] = {
    "noelle.models": "models.log",
    "noelle.web": "web.log",
    "noelle.rag": "rag.log",
    "noelle.quantize": "quantize.log",
}

NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "asyncio", "zmq", "PIL")

_handled_loggers: set[str] = set()
_logger_levels: dict[str, int] = {}
_root_level: int | None = None


class RedactingFormatter(logging.Formatter):
    """Mask token-shaped strings in the message, arguments and traceback text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def _level(value: str | int | None, default: int) -> int:
    if isinstance(value, int):
        return value
    if value:
        parsed = logging.getLevelName(str(value).upper())
        if isinstance(parsed, int):
            return parsed
    return default


def _env_level(default: int) -> int:
    return _level(os.environ.get("NOELLE_LOG_LEVEL"), default)


def _file_handler(filename: str, level: int, *, max_bytes: int = COMPONENT_MAX_BYTES,
                  backups: int = COMPONENT_BACKUPS) -> logging.Handler | None:
    try:
        paths.logs_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            paths.logs_dir() / filename, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    except OSError:
        return None
    handler.setLevel(level)
    handler.setFormatter(RedactingFormatter(FORMAT))
    handler._noelle_handler = True  # type: ignore[attr-defined]
    return handler


def _attach(logger_name: str, handler: logging.Handler | None) -> None:
    if handler is None:
        return
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    _handled_loggers.add(logger_name)


def _reset_our_handlers() -> None:
    global _root_level
    for name in list(_handled_loggers):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, "_noelle_handler", False):
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:  # noqa: BLE001 - closing a broken file handler must not raise
                    pass
    _handled_loggers.clear()
    for name, level in _logger_levels.items():
        logging.getLogger(name).setLevel(level)
    _logger_levels.clear()
    if _root_level is not None:
        logging.getLogger().setLevel(_root_level)
        _root_level = None


def _set_level(name: str, level: int) -> None:
    logger = logging.getLogger(name)
    _logger_levels.setdefault(name, logger.level)
    logger.setLevel(level)


def _set_root_level(level: int) -> None:
    global _root_level
    root = logging.getLogger()
    if _root_level is None:
        _root_level = root.level
    root.setLevel(level)


def _install_components(components: Iterable[tuple[str, str]], level: int) -> None:
    by_file: dict[str, list[str]] = {}
    for name, filename in components:
        by_file.setdefault(filename, []).append(name)
    for filename, names in by_file.items():
        handler = _file_handler(filename, level)
        for name in names:
            _set_level(name, level)
            _attach(name, handler)


def configure_desktop_logging() -> Path:
    """Install the desktop shell's files: app.log, errors.log, and one focused file per subsystem."""
    _reset_our_handlers()
    app_level = _env_level(logging.INFO)
    _set_root_level(app_level)
    _attach("", _file_handler("app.log", app_level, max_bytes=APP_MAX_BYTES, backups=APP_BACKUPS))
    _attach("", _file_handler("errors.log", logging.ERROR, max_bytes=COMPONENT_MAX_BYTES,
                              backups=COMPONENT_BACKUPS))
    if os.environ.get("NOELLE_LOG_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        _attach("", _file_handler("debug.log", logging.DEBUG, max_bytes=5_000_000, backups=1))
    _install_components(DESKTOP_COMPONENTS.items(), _env_level(logging.DEBUG))
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return paths.logs_dir()


def configure_process_logging(name: str, *, components: Mapping[str, str] | None = None,
                              console: bool = True, console_level: int = logging.INFO) -> Path:
    """Install file logging for a child process (worker, server, downloader, MCP)."""
    _reset_our_handlers()
    file_level = _env_level(logging.INFO)
    _set_root_level(file_level)
    _attach("", _file_handler(f"{name}.log", file_level, max_bytes=APP_MAX_BYTES, backups=APP_BACKUPS))
    if components:
        _install_components(components.items(), _env_level(logging.DEBUG))
    if console and sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(console_level)
        stream.setFormatter(RedactingFormatter("[%(name)s] %(levelname)s %(message)s"))
        stream._noelle_handler = True  # type: ignore[attr-defined]
        _attach("", stream)
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return paths.logs_dir()


def configure_worker_logging() -> Path:
    return configure_process_logging("worker", components=WORKER_COMPONENTS, console=True)


def configure_server_logging() -> Path:
    return configure_process_logging("server", console=True)


def configure_download_logging() -> Path:
    return configure_process_logging("download-child", console=True)


def configure_mcp_logging() -> Path:
    return configure_process_logging("mcp", console=True, console_level=logging.WARNING)


def configure_cli_logging(name: str = "cli") -> Path:
    return configure_process_logging(name, console=False)


def reset_logging() -> None:
    """Remove handlers installed by this module (tests, or switching data roots)."""
    _reset_our_handlers()