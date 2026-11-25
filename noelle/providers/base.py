"""Unified AI Provider Interface for Hybrid Cloud + Local execution.

This module defines standardized schemas for requests, responses, telemetry,
and capability reporting across local runtimes (Hugging Face, llama.cpp, Whisper)
and remote cloud providers (OpenAI, OpenRouter, Gemini, Anthropic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Modality(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    STT = "stt"
    TTS = "tts"
    VISION = "vision"
    RAG = "rag"


@dataclass
class TelemetryRecord:
    """Normalized 4-dimensional telemetry captured during execution."""
    latency_ms: float
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    vram_mb: float = 0.0
    real_time_factor: float = 0.0
    audio_duration_s: float = 0.0
    network_transfer_ms: float = 0.0
    compute_ms: float = 0.0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "vram_mb": round(self.vram_mb, 2),
            "real_time_factor": round(self.real_time_factor, 4),
            "audio_duration_s": round(self.audio_duration_s, 2),
            "network_transfer_ms": round(self.network_transfer_ms, 2),
            "compute_ms": round(self.compute_ms, 2),
        }


@dataclass
class ProviderCapabilities:
    """Declared capabilities and supported models of an AI provider."""
    provider_id: str
    display_name: str
    is_local: bool
    modalities: set[str] = field(default_factory=set)
    supported_models: list[str] = field(default_factory=list)
    supports_streaming: bool = True
    supports_json_schema: bool = True
    max_context_window: int = 128_000

    def supports_modality(self, modality: str | Modality) -> bool:
        mod = modality.value if isinstance(modality, Modality) else modality
        return mod in self.modalities


@dataclass
class NodeRequest:
    """Normalized input request dispatched to an AI provider node."""
    modality: str | Modality
    prompt: str = ""
    system_prompt: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    texts: list[str] = field(default_factory=list)
    audio_path: str = ""
    audio_bytes: bytes | None = None
    image_path: str = ""
    image_bytes: bytes | None = None
    voice: str = "alloy"
    language: str = "id"
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def modality_str(self) -> str:
        return self.modality.value if isinstance(self.modality, Modality) else self.modality


@dataclass
class NodeResponse:
    """Invariant output payload returned by every workflow node."""
    output: Any
    telemetry: TelemetryRecord
    provider_id: str
    model_id: str
    is_error: bool = False
    error_message: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return not self.is_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output if not isinstance(self.output, bytes) else f"<binary {len(self.output)} bytes>",
            "telemetry": self.telemetry.to_dict(),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "is_error": self.is_error,
            "error_message": self.error_message,
        }


class AIProvider(Protocol):
    """Protocol contract implemented by all cloud and local provider adapters."""
    id: str
    name: str
    is_local: bool

    def get_capabilities(self) -> ProviderCapabilities:
        """Query supported modalities and models."""
        ...

    def execute(self, request: NodeRequest) -> NodeResponse:
        """Synchronously execute a single node request and record telemetry."""
        ...

    def estimate_cost(self, request: NodeRequest) -> float:
        """Calculate predicted cost in USD for a proposed request."""
        ...
