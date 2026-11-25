"""Secure credential store for AI providers (OpenAI, OpenRouter, Gemini, etc.).

Stores keys in the operating system credential manager (Keychain on macOS,
Windows Credential Manager on Windows, Secret Service/DBus on Linux) via ``keyring``.
Keys are never saved in workflow files, logs, git repositories, or exported artifacts.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("noelle.providers.credentials")

KEYRING_SERVICE = "noelle.providers"

# In-memory storage fallback for headless CI runs or when OS keyring is absent
_memory_keys: dict[str, str] = {}


def _env_var_name(provider_id: str) -> str:
    cleaned = provider_id.upper().replace("-", "_").replace(".", "_")
    return f"{cleaned}_API_KEY"


def get_api_key(provider_id: str) -> str | None:
    """Retrieve an API key for the specified provider.

    Checks:
    1. OS credential store via keyring
    2. Environment variable (e.g. ``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``)
    3. In-memory fallback
    """
    if provider_id in _memory_keys:
        return _memory_keys[provider_id]

    # Check environment variable first for rapid headless execution / CI
    env_name = _env_var_name(provider_id)
    env_val = os.environ.get(env_name)
    if env_val and env_val.strip():
        return env_val.strip()

    try:
        import keyring
        val = keyring.get_password(KEYRING_SERVICE, provider_id)
        if val and val.strip():
            return val.strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("Keyring access failed for %s: %s", provider_id, exc)

    return None


def set_api_key(provider_id: str, key: str) -> bool:
    """Save an API key in the OS credential vault."""
    clean = key.strip()
    _memory_keys[provider_id] = clean
    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, provider_id, clean)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not persist API key to OS credential vault for %s: %s", provider_id, exc)
        return False


def delete_api_key(provider_id: str) -> bool:
    """Remove an API key from the OS credential vault and memory."""
    _memory_keys.pop(provider_id, None)
    try:
        import keyring
        keyring.delete_password(KEYRING_SERVICE, provider_id)
        return True
    except Exception:  # noqa: BLE001
        return False


def has_api_key(provider_id: str) -> bool:
    """Check if an API key is available for the given provider."""
    return get_api_key(provider_id) is not None
