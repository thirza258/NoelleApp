"""JSON-backed settings with packaged defaults.

Values are addressed with dotted keys (``"rag.top_k"``). User overrides are
stored in ``~/.noelle/settings.json``; anything not overridden falls back to
``default_settings.json``. Secrets (the Hugging Face token) never go here:
they live in the OS credential store, see :mod:`noelle.app.services.hf_client`.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from noelle import paths

_DEFAULTS_PATH = Path(__file__).with_name("default_settings.json")
_MISSING = object()


def load_defaults() -> dict[str, Any]:
    with _DEFAULTS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or paths.settings_path()
        self._lock = threading.RLock()
        self._defaults = load_defaults()
        self._overrides: dict[str, Any] = {}
        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def reload(self) -> None:
        with self._lock:
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._overrides = data if isinstance(data, dict) else {}
            except FileNotFoundError:
                self._overrides = {}
            except (OSError, json.JSONDecodeError):
                # A corrupt settings file must never stop the app from starting.
                self._overrides = {}

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return _deep_merge(self._defaults, self._overrides)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            for source in (self._overrides, self._defaults):
                value = _lookup(source, key)
                if value is not _MISSING:
                    return copy.deepcopy(value)
            return default

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            node = self._overrides
            parts = key.split(".")
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[parts[-1]] = copy.deepcopy(value)
            self._save()

    def update(self, values: dict[str, Any]) -> None:
        with self._lock:
            for key, value in values.items():
                node = self._overrides
                parts = key.split(".")
                for part in parts[:-1]:
                    child = node.get(part)
                    if not isinstance(child, dict):
                        child = {}
                        node[part] = child
                    node = child
                node[parts[-1]] = copy.deepcopy(value)
            self._save()

    def reset(self, key: str) -> None:
        with self._lock:
            node = self._overrides
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.get(part)
                if not isinstance(node, dict):
                    return
            node.pop(parts[-1], None)
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".settings.", dir=self._path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._overrides, fh, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _lookup(source: dict[str, Any], key: str) -> Any:
    node: Any = source
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


_instance: Settings | None = None
_instance_lock = threading.Lock()


def get_settings() -> Settings:
    """Process-wide settings singleton (recreated if NOELLE_HOME changes)."""
    global _instance
    with _instance_lock:
        if _instance is None or _instance.path != paths.settings_path():
            _instance = Settings()
        return _instance
