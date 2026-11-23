"""Where the llama.cpp tools live (installed by the Components feature, or found on PATH).

Pure Python; shared by the worker (GGUF inference), the quantization engine and the UI.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from noelle import paths

COMPONENT = "llama.cpp"
EXE = ".exe" if os.name == "nt" else ""
TOOL_NAMES = (f"llama-quantize{EXE}", f"llama-server{EXE}")


def install_root() -> Path:
    return paths.home() / "tools" / "llama.cpp"


def installed_info() -> dict[str, Any] | None:
    """Manifest of the component installed by Noelle, when its binaries are present."""
    try:
        data = json.loads((install_root() / "current.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    folder = Path(str(data.get("path", "")))
    if not all((folder / "bin" / name).is_file() for name in TOOL_NAMES):
        return None
    return data


def _configured_dirs() -> list[Path]:
    try:
        from noelle.config import get_settings

        value = get_settings().get("quantization.llama_cpp_dir")
    except Exception:
        return []
    if not value:
        return []
    root = Path(str(value)).expanduser()
    return [root / "bin", root, root / "build" / "bin"]


def find_llama_server() -> Path | None:
    info = installed_info()
    candidates = ([Path(info["path"]) / "bin"] if info else []) + _configured_dirs()
    for folder in candidates:
        binary = folder / f"llama-server{EXE}"
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
    found = shutil.which(f"llama-server{EXE}")
    return Path(found) if found else None
