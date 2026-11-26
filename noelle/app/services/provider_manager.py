"""Cloud provider credential manager (Qt facade over :mod:`noelle.providers.credentials`).

Keys live in the OS credential vault (Keychain / Windows Credential Manager / Secret
Service). Workflows store only provider ids, never the key. ``keyring`` access is
synchronous and fast; callers may still run :meth:`test_connection` off the GUI thread.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from noelle.providers import credentials

log = logging.getLogger("noelle.providers.manager")

PROVIDERS = {
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1"},
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"},
    "local": {"name": "Noelle Local", "base_url": ""},
}


class ProviderManager(QObject):
    credentials_updated = Signal(str, bool)
    provider_status_changed = Signal(str, str)  # provider_id, status

    def set_api_key(self, provider_id: str, api_key: str) -> bool:
        ok = credentials.set_api_key(provider_id, api_key)
        self.credentials_updated.emit(provider_id, bool(api_key.strip()))
        self.provider_status_changed.emit(provider_id, "configured" if ok else "unavailable")
        return ok

    def get_api_key(self, provider_id: str) -> str | None:
        return credentials.get_api_key(provider_id)

    def has_valid_key(self, provider_id: str) -> bool:
        return credentials.has_api_key(provider_id)

    def delete_api_key(self, provider_id: str) -> bool:
        ok = credentials.delete_api_key(provider_id)
        self.credentials_updated.emit(provider_id, False)
        self.provider_status_changed.emit(provider_id, "not_configured")
        return ok

    def configured(self) -> dict[str, bool]:
        return {provider_id: self.has_valid_key(provider_id) for provider_id in PROVIDERS}

    def test_connection(self, provider_id: str, *, timeout: float = 10.0) -> tuple[bool, str]:
        """``GET /models`` (OpenAI-compatible) with the stored key. Never logs the key."""
        key = self.get_api_key(provider_id)
        if not key:
            return False, "No API key configured."
        base = PROVIDERS.get(provider_id, {}).get("base_url")
        if not base:
            return False, f"Provider '{provider_id}' has no HTTP endpoint."
        try:
            import httpx

            response = httpx.get(f"{base.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"},
                                 timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - network errors are reported, not raised
            log.warning("Provider %s connection test failed: %s", provider_id, exc)
            return False, str(exc)
        if response.status_code == 200:
            return True, "Authenticated"
        return False, f"HTTP {response.status_code}: {response.text[:200]}"

    def status(self, provider_id: str) -> dict[str, Any]:
        return {"provider_id": provider_id, **PROVIDERS.get(provider_id, {}),
                "configured": self.has_valid_key(provider_id)}