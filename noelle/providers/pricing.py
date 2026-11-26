"""Pricing schedules and token expenditure calculation for cloud providers.

Rates are specified in USD per 1,000,000 tokens (or per minute/character for audio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float   # USD per 1,000,000 input tokens
    output_per_m: float  # USD per 1,000,000 output tokens
    audio_per_minute: float = 0.0
    tts_per_1k_chars: float = 0.0


# Published baseline pricing schedule
MODEL_PRICING: Mapping[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(input_per_m=2.50, output_per_m=10.00),
    "gpt-4o-mini": ModelPrice(input_per_m=0.15, output_per_m=0.60),
    "gpt-4-turbo": ModelPrice(input_per_m=10.00, output_per_m=30.00),
    "o1": ModelPrice(input_per_m=15.00, output_per_m=60.00),
    "o1-mini": ModelPrice(input_per_m=3.00, output_per_m=12.00),
    "o3-mini": ModelPrice(input_per_m=1.10, output_per_m=4.40),
    "text-embedding-3-small": ModelPrice(input_per_m=0.02, output_per_m=0.0),
    "text-embedding-3-large": ModelPrice(input_per_m=0.13, output_per_m=0.0),
    "whisper-1": ModelPrice(input_per_m=0.0, output_per_m=0.0, audio_per_minute=0.006),
    "tts-1": ModelPrice(input_per_m=0.0, output_per_m=0.0, tts_per_1k_chars=0.015),
    "tts-1-hd": ModelPrice(input_per_m=0.0, output_per_m=0.0, tts_per_1k_chars=0.030),

    # OpenRouter common models
    "anthropic/claude-3.5-sonnet": ModelPrice(input_per_m=3.00, output_per_m=15.00),
    "anthropic/claude-3.5-haiku": ModelPrice(input_per_m=0.80, output_per_m=4.00),
    "google/gemini-2.0-flash": ModelPrice(input_per_m=0.10, output_per_m=0.40),
    "google/gemini-pro-1.5": ModelPrice(input_per_m=1.25, output_per_m=5.00),
    "meta-llama/llama-3.3-70b-instruct": ModelPrice(input_per_m=0.40, output_per_m=0.40),
    "qwen/qwen-2.5-72b-instruct": ModelPrice(input_per_m=0.35, output_per_m=0.40),
    "deepseek/deepseek-chat": ModelPrice(input_per_m=0.14, output_per_m=0.28),
    "deepseek/deepseek-r1": ModelPrice(input_per_m=0.55, output_per_m=2.19),
}


def lookup_price(model_id: str) -> ModelPrice:
    """Retrieve pricing for a model or fallback to reasonable tier."""
    clean = model_id.strip().lower()
    if clean in MODEL_PRICING:
        return MODEL_PRICING[clean]

    # Partial match
    for name, price in MODEL_PRICING.items():
        if name in clean or clean in name:
            return price

    # Heuristic fallback based on size
    if "mini" in clean or "flash" in clean or "haiku" in clean or "small" in clean:
        return ModelPrice(input_per_m=0.20, output_per_m=0.80)
    return ModelPrice(input_per_m=2.00, output_per_m=8.00)


def has_published_price(model_id: str) -> bool:
    """True when the model has an explicit schedule (not the size-based fallback)."""
    clean = model_id.strip().lower()
    return clean in MODEL_PRICING or any(name in clean or clean in name for name in MODEL_PRICING)


def calculate_cost(model_id: str, *, input_tokens: int = 0, output_tokens: int = 0,
                   audio_duration_seconds: float = 0.0, text_chars: int = 0) -> float:
    """Calculate exact USD cost for a given inference output."""
    price = lookup_price(model_id)
    cost = 0.0
    if input_tokens > 0:
        cost += (input_tokens / 1_000_000.0) * price.input_per_m
    if output_tokens > 0:
        cost += (output_tokens / 1_000_000.0) * price.output_per_m
    if audio_duration_seconds > 0.0 and price.audio_per_minute > 0.0:
        cost += (audio_duration_seconds / 60.0) * price.audio_per_minute
    if text_chars > 0 and price.tts_per_1k_chars > 0.0:
        cost += (text_chars / 1000.0) * price.tts_per_1k_chars
    return round(cost, 6)
