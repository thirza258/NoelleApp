"""Filesystem layout of the Noelle data root.

Everything Noelle writes lives under ``~/.noelle`` (override with the
``NOELLE_HOME`` environment variable)::

    ~/.noelle/
    ├── models/        derived checkpoints (quantized, explicit local_dir downloads)
    ├── environments/  isolated runtime virtual environments
    ├── vector_store/  RAG indexes (SQLite + sqlite-vec)
    ├── projects/      playground sessions, benchmark exports, generated media
    ├── logs/          app, worker and API server logs
    ├── runtime/       live worker/server discovery files
    ├── registry.db    model / adapter / evaluation registry
    └── settings.json  user settings (defaults in noelle/config/default_settings.json)

Hub downloads go to the standard Hugging Face cache by default so they stay
deduplicated and visible to the storage page; see ``storage.download_mode``.

Paths are computed on each call (not frozen at import time) so tests can point
``NOELLE_HOME`` at a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    override = os.environ.get("NOELLE_HOME")
    return Path(override).expanduser() if override else Path.home() / ".noelle"


def models_dir() -> Path:
    return home() / "models"


def environments_dir() -> Path:
    return home() / "environments"


def vector_store_dir() -> Path:
    return home() / "vector_store"


def projects_dir() -> Path:
    return home() / "projects"


def outputs_dir() -> Path:
    return projects_dir() / "outputs"


def exports_dir() -> Path:
    return projects_dir() / "exports"


def logs_dir() -> Path:
    return home() / "logs"


def runtime_dir() -> Path:
    return home() / "runtime"


def registry_path() -> Path:
    return home() / "registry.db"


def settings_path() -> Path:
    return home() / "settings.json"


def ensure_dirs() -> Path:
    root = home()
    for path in (
        root,
        models_dir(),
        environments_dir(),
        vector_store_dir(),
        projects_dir(),
        outputs_dir(),
        exports_dir(),
        logs_dir(),
        runtime_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root


def hf_hub_cache(hf_home: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the Hugging Face hub cache without importing huggingface_hub.

    Precedence matches huggingface_hub: explicit argument, ``HF_HUB_CACHE``,
    ``HF_HOME``/hub, then ``~/.cache/huggingface/hub``.
    """
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"
