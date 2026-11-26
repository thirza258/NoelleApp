"""Small JSON-catalog localization layer (English + Bahasa Indonesia).

Keys are namespaced (``chat.send``). ``tr`` falls back to English and then to the
caller-supplied default, so a missing translation never shows a raw key. Language
switching is immediate for widgets that call :func:`tr` from a retranslate hook;
existing pages are migrated incrementally.
"""

from __future__ import annotations

import json
import locale
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("noelle.i18n")

SUPPORTED = ("en", "id")
_LANGUAGE = "en"
_CATALOG: dict[str, str] = {}
_FALLBACK: dict[str, str] = {}
_CATALOG_DIR = Path(__file__).with_name("i18n")


def _load(language: str) -> dict[str, str]:
    path = _CATALOG_DIR / f"{language}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.debug("No translation catalog for %s (%s)", language, path)
        return {}
    return {str(key): str(value) for key, value in data.items() if isinstance(value, str)}


def resolve_language(language: str | None) -> str:
    if language in SUPPORTED:
        return str(language)
    if language and language not in ("system", "auto"):
        base = str(language).split("_")[0].split("-")[0].lower()
        if base in SUPPORTED:
            return base
    try:
        system = locale.getlocale()[0] or os.environ.get("LANG") or os.environ.get("LC_ALL") or "en"
    except Exception:  # noqa: BLE001
        system = "en"
    base = system.split("_")[0].lower()
    return base if base in SUPPORTED else "en"


def set_language(language: str | None) -> str:
    global _LANGUAGE, _CATALOG, _FALLBACK
    _LANGUAGE = resolve_language(language)
    _FALLBACK = _load("en")
    _CATALOG = _FALLBACK if _LANGUAGE == "en" else _load(_LANGUAGE)
    return _LANGUAGE


def current_language() -> str:
    return _LANGUAGE


def tr(key: str, default: str | None = None, **kwargs: Any) -> str:
    text = _CATALOG.get(key) or _FALLBACK.get(key) or default or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return text


def available_languages() -> tuple[str, ...]:
    return SUPPORTED


# Load English at import so `tr` is usable before the app installs a language.
set_language("en")