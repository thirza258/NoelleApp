"""Provider Registry and Dynamic Capability Matrix.

Coordinates registered cloud and local backends and validates node configurations.
"""

from __future__ import annotations

import logging

from noelle.providers.base import AIProvider, Modality
from noelle.providers.local_provider import LocalProvider
from noelle.providers.openai_provider import OpenAIProvider
from noelle.providers.openrouter_provider import OpenRouterProvider

log = logging.getLogger("noelle.providers.registry")

_registry: dict[str, AIProvider] = {}


def register_provider(provider: AIProvider) -> None:
    """Register a provider instance."""
    _registry[provider.id] = provider


def get_provider(provider_id: str) -> AIProvider | None:
    """Look up a provider by id."""
    ensure_default_providers()
    return _registry.get(provider_id)


def list_providers() -> list[AIProvider]:
    """Return all currently registered providers."""
    ensure_default_providers()
    return list(_registry.values())


def ensure_default_providers() -> None:
    """Initialize standard providers if not already registered."""
    if "local" not in _registry:
        register_provider(LocalProvider())
    if "openai" not in _registry:
        register_provider(OpenAIProvider())
    if "openrouter" not in _registry:
        register_provider(OpenRouterProvider())


def get_capability_matrix() -> dict[str, dict[str, bool]]:
    """Return an active Provider x Modality capability matrix."""
    ensure_default_providers()
    matrix: dict[str, dict[str, bool]] = {}
    modalities = [m.value for m in Modality]

    for pid, provider in _registry.items():
        caps = provider.get_capabilities()
        matrix[pid] = {mod: caps.supports_modality(mod) for mod in modalities}

    return matrix


def validate_node_target(provider_id: str, modality: str) -> tuple[bool, str]:
    """Validate if a provider can execute a given modality.

    Returns (is_valid, error_reason).
    """
    provider = get_provider(provider_id)
    if provider is None:
        return False, f"Provider '{provider_id}' is not registered."

    caps = provider.get_capabilities()
    if not caps.supports_modality(modality):
        return False, f"Provider '{provider.name}' does not support modality '{modality}'."

    return True, ""
