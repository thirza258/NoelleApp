"""Noelle AI Providers package.

Provides unified abstractions and adapters for local models (Hugging Face,
llama.cpp, Whisper) and cloud APIs (OpenAI, OpenRouter).
"""

from __future__ import annotations

from noelle.providers.base import (
    AIProvider,
    Modality,
    NodeRequest,
    NodeResponse,
    ProviderCapabilities,
    TelemetryRecord,
)
from noelle.providers.credentials import delete_api_key, get_api_key, has_api_key, set_api_key
from noelle.providers.local_provider import LocalProvider
from noelle.providers.openai_provider import OpenAIProvider
from noelle.providers.openrouter_provider import OpenRouterProvider
from noelle.providers.pricing import calculate_cost, lookup_price
from noelle.providers.registry import (
    get_capability_matrix,
    get_provider,
    list_providers,
    register_provider,
    validate_node_target,
)

__all__ = [
    "AIProvider",
    "Modality",
    "NodeRequest",
    "NodeResponse",
    "ProviderCapabilities",
    "TelemetryRecord",
    "get_api_key",
    "set_api_key",
    "delete_api_key",
    "has_api_key",
    "LocalProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "calculate_cost",
    "lookup_price",
    "get_provider",
    "list_providers",
    "register_provider",
    "get_capability_matrix",
    "validate_node_target",
]
