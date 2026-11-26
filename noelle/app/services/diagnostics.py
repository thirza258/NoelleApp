"""Local diagnostics bundle ("Report a problem").

Builds a zip the user can attach to an issue. Nothing is uploaded by Noelle. Every
text file passes through redaction: token-shaped strings, home paths, usernames,
e-mail addresses and credential-looking settings values never reach the archive.
"""

from __future__ import annotations

import getpass
import json
import platform
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from noelle import __version__, paths
from noelle.runtime.errors import redact_secrets

_LOG_NAMES = ("app.log", "errors.log", "worker.log", "models.log", "ipc.log", "downloads.log", "download-child.log",
              "hub.log", "storage.log", "ui.log", "tasks.log", "server.log", "rag.log", "quantize.log")
_SECRET_KEY = re.compile(r"(?i)token|key|secret|password|credential")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _redact_text(text: str) -> str:
    text = redact_secrets(text)
    home = str(paths.home())
    for candidate in {home, str(Path.home())}:
        if candidate and candidate != "/":
            text = text.replace(candidate, "~")
            text = text.replace(candidate.replace("\\", "/"), "~")
    try:
        username = getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry in some containers
        username = ""
    if username and len(username) > 2:
        text = re.sub(rf"\b{re.escape(username)}\b", "<user>", text)
    return _EMAIL.sub("<email>", text)


def _redact_value(key: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, str):
        return "<redacted>" if _SECRET_KEY.search(key) else _redact_text(value)
    return value


def _tail(path: Path, limit: int = 2_000_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > limit:
                fh.seek(size - limit)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {"noelle": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
                            "machine": platform.machine(), "processor": platform.processor()}
    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count()
        info["memory_bytes"] = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001
        pass
    try:
        import PySide6

        info["pyside6"] = PySide6.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def build_report(destination: Path | str | None = None, *, description: str = "", what_happened: str = "",
                 include_logs: bool = True, include_settings: bool = True, include_system: bool = True,
                 include_models: bool = True, settings: Any = None, loaded_models: Iterable[Any] = (),
                 runtime_caps: Mapping[str, Any] | None = None, log_dir: Path | None = None) -> Path:
    """Create the zip and return its path. The caller chooses whether to share it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(destination) if destination else Path(tempfile.gettempdir()) / f"noelle-report-{stamp}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Noelle diagnostics report — {stamp}", "", "## What happened", _redact_text(description or "(not provided)"),
             "", "## What I was doing", _redact_text(what_happened or "(not provided)")]
    if include_system:
        lines += ["", "## System", *[f"{key}: {value}" for key, value in _system_info().items()]]
    if runtime_caps:
        lines += ["", "## Runtime capabilities", _redact_text(json.dumps(runtime_caps, indent=2))]
    if include_models:
        models = [model if isinstance(model, Mapping) else getattr(model, "describe", lambda: {})()
                  for model in loaded_models]
        lines += ["", "## Loaded models", _redact_text(json.dumps(models, indent=2, default=str))]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.txt", _redact_text("\n".join(lines)))
        if include_logs:
            folder = Path(log_dir) if log_dir else paths.logs_dir()
            for name in _LOG_NAMES:
                text = _tail(folder / name)
                if text:
                    archive.writestr(f"logs/{name}", _redact_text(text))
        if include_settings and settings is not None:
            try:
                data = settings.as_dict() if hasattr(settings, "as_dict") else dict(settings)
            except Exception:  # noqa: BLE001
                data = {}
            archive.writestr("settings.json", json.dumps(_redact_value("", data), indent=2, default=str))
    return target