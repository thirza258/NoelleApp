"""Multi-modal workflow nodes for the Noelle Hybrid Lab.

Encapsulates computation, parameterization, and provider dispatch across
LLM, Embedding, STT, TTS, Vision, RAG, and Comparison nodes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from noelle.providers.base import Modality, NodeRequest, NodeResponse, TelemetryRecord
from noelle.providers.registry import get_provider

log = logging.getLogger("noelle.workflow.nodes")


class ExecutionContext(Protocol):
    """Execution context passed through workflow execution."""
    privacy_mode: str
    node_outputs: dict[str, dict[str, Any]]
    telemetry_records: dict[str, TelemetryRecord]


@dataclass
class BaseNode:
    """Base class for all executable workflow nodes."""
    id: str
    title: str
    node_type: str
    provider_id: str = "local"
    model_id: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    ui_pos: tuple[float, float] = (0.0, 0.0)

    @property
    def is_cloud(self) -> bool:
        return self.provider_id != "local"

    def input_ports(self) -> list[str]:
        return ["input"]

    def output_ports(self) -> list[str]:
        return ["output"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """Execute node with resolved input port values."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "node_type": self.node_type,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "parameters": self.parameters,
            "ui_pos": list(self.ui_pos),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseNode:
        return cls(
            id=data["id"],
            title=data.get("title", data["id"]),
            node_type=data.get("node_type", "base"),
            provider_id=data.get("provider_id", "local"),
            model_id=data.get("model_id", ""),
            parameters=data.get("parameters", {}),
            ui_pos=tuple(data.get("ui_pos", [0.0, 0.0])),  # type: ignore[assignment]
        )


@dataclass
class InputNode(BaseNode):
    """Ingests raw text, audio, image, or structured data."""
    node_type: str = "input"

    def input_ports(self) -> list[str]:
        return []

    def output_ports(self) -> list[str]:
        return ["output", "text", "audio", "image"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        val = self.parameters.get("value", "")
        return {
            "output": val,
            "text": str(val),
            "audio": self.parameters.get("audio_bytes"),
            "image": self.parameters.get("image_bytes"),
        }


@dataclass
class LLMNode(BaseNode):
    """Causal language model node for chat, reasoning, and JSON emission."""
    node_type: str = "llm"

    def input_ports(self) -> list[str]:
        return ["prompt", "context", "system"]

    def output_ports(self) -> list[str]:
        return ["text", "response"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        prompt = inputs.get("prompt") or inputs.get("input") or self.parameters.get("prompt", "")
        context = inputs.get("context", "")
        system = inputs.get("system") or self.parameters.get("system_prompt", "")

        if context:
            prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"

        provider = get_provider(self.provider_id)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {self.provider_id}")

        req = NodeRequest(
            modality=Modality.LLM,
            prompt=str(prompt),
            system_prompt=str(system),
            model=self.model_id or self.parameters.get("model", ""),
            temperature=float(self.parameters.get("temperature", 0.7)),
            max_tokens=int(self.parameters.get("max_tokens", 1024)),
        )

        resp: NodeResponse = provider.execute(req)
        if resp.is_error:
            raise RuntimeError(f"Node '{self.id}' failed: {resp.error_message}")

        if hasattr(ctx, "record_telemetry"):
            ctx.record_telemetry(self.id, resp.telemetry)

        return {"text": resp.output, "response": resp}


@dataclass
class EmbeddingNode(BaseNode):
    """Converts texts into dense semantic vectors."""
    node_type: str = "embedding"

    def input_ports(self) -> list[str]:
        return ["text", "texts"]

    def output_ports(self) -> list[str]:
        return ["embedding", "embeddings"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        texts_input = inputs.get("texts")
        single_text = inputs.get("text") or inputs.get("input") or self.parameters.get("text", "")

        texts = texts_input if isinstance(texts_input, list) else [str(single_text)]
        provider = get_provider(self.provider_id)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {self.provider_id}")

        req = NodeRequest(
            modality=Modality.EMBEDDING,
            texts=texts,
            prompt=texts[0] if texts else "",
            model=self.model_id or self.parameters.get("model", ""),
        )
        resp = provider.execute(req)
        if resp.is_error:
            raise RuntimeError(f"Embedding node '{self.id}' failed: {resp.error_message}")

        if hasattr(ctx, "record_telemetry"):
            ctx.record_telemetry(self.id, resp.telemetry)

        return {
            "embedding": resp.output if not isinstance(resp.output, list) or not resp.output or not isinstance(resp.output[0], list) else resp.output[0],
            "embeddings": resp.output if isinstance(resp.output, list) and resp.output and isinstance(resp.output[0], list) else [resp.output],
        }


@dataclass
class STTNode(BaseNode):
    """Speech-to-text recognition node (Whisper local vs cloud)."""
    node_type: str = "stt"

    def input_ports(self) -> list[str]:
        return ["audio", "audio_path"]

    def output_ports(self) -> list[str]:
        return ["transcript", "text"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        audio_bytes = inputs.get("audio") or self.parameters.get("audio_bytes")
        audio_path = inputs.get("audio_path") or self.parameters.get("audio_path", "")

        provider = get_provider(self.provider_id)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {self.provider_id}")

        req = NodeRequest(
            modality=Modality.STT,
            audio_bytes=audio_bytes,
            audio_path=str(audio_path),
            model=self.model_id or self.parameters.get("model", "whisper-1"),
            language=self.parameters.get("language", "id"),
        )
        resp = provider.execute(req)
        if resp.is_error:
            raise RuntimeError(f"STT node '{self.id}' failed: {resp.error_message}")

        if hasattr(ctx, "record_telemetry"):
            ctx.record_telemetry(self.id, resp.telemetry)

        transcript = str(resp.output)
        return {"transcript": transcript, "text": transcript}


@dataclass
class TTSNode(BaseNode):
    """Text-to-speech acoustic synthesis node."""
    node_type: str = "tts"

    def input_ports(self) -> list[str]:
        return ["text"]

    def output_ports(self) -> list[str]:
        return ["audio", "audio_bytes"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        text = inputs.get("text") or inputs.get("input") or self.parameters.get("text", "")
        provider = get_provider(self.provider_id)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {self.provider_id}")

        req = NodeRequest(
            modality=Modality.TTS,
            prompt=str(text),
            model=self.model_id or self.parameters.get("model", "tts-1"),
            voice=self.parameters.get("voice", "alloy"),
        )
        resp = provider.execute(req)
        if resp.is_error:
            raise RuntimeError(f"TTS node '{self.id}' failed: {resp.error_message}")

        if hasattr(ctx, "record_telemetry"):
            ctx.record_telemetry(self.id, resp.telemetry)

        return {"audio": resp.output, "audio_bytes": resp.output}


@dataclass
class VisionNode(BaseNode):
    """Computer vision and document extraction node."""
    node_type: str = "vision"

    def input_ports(self) -> list[str]:
        return ["image", "image_path", "prompt"]

    def output_ports(self) -> list[str]:
        return ["text", "analysis"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        image_bytes = inputs.get("image") or self.parameters.get("image_bytes")
        image_path = inputs.get("image_path") or self.parameters.get("image_path", "")
        prompt = inputs.get("prompt") or self.parameters.get("prompt", "Analyze this image.")

        provider = get_provider(self.provider_id)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {self.provider_id}")

        req = NodeRequest(
            modality=Modality.VISION,
            prompt=str(prompt),
            image_bytes=image_bytes,
            image_path=str(image_path),
            model=self.model_id or self.parameters.get("model", "gpt-4o"),
        )
        resp = provider.execute(req)
        if resp.is_error:
            raise RuntimeError(f"Vision node '{self.id}' failed: {resp.error_message}")

        if hasattr(ctx, "record_telemetry"):
            ctx.record_telemetry(self.id, resp.telemetry)

        return {"text": resp.output, "analysis": resp.output}


@dataclass
class RAGNode(BaseNode):
    """Retrieval-Augmented Generation composite node (chunk -> embed -> query)."""
    node_type: str = "rag"

    def input_ports(self) -> list[str]:
        return ["query", "documents"]

    def output_ports(self) -> list[str]:
        return ["context", "passages"]

    def execute(self, inputs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        query = str(inputs.get("query") or inputs.get("input") or self.parameters.get("query", ""))
        docs = inputs.get("documents") or self.parameters.get("documents", [])
        top_k = int(self.parameters.get("top_k", 3))

        start_time = time.monotonic()
        passages: list[str] = []
        if isinstance(docs, list):
            # Simple keyword / token overlap ranking over documents
            query_words = set(query.lower().split())
            scored: list[tuple[float, str]] = []
            for doc in docs:
                text = str(doc)
                score = sum(1.0 for w in query_words if w in text.lower())
                scored.append((score, text))
            scored.sort(key=lambda x: x[0], reverse=True)
            passages = [text for _, text in scored[:top_k]]
        elif isinstance(docs, str) and docs.strip():
            passages = [docs.strip()]

        latency_ms = (time.monotonic() - start_time) * 1000.0
        context_str = "\n---\n".join(passages)

        if hasattr(ctx, "record_telemetry"):
            ctx.record_telemetry(self.id, TelemetryRecord(latency_ms=latency_ms, cost_usd=0.0))

        return {"context": context_str, "passages": passages}


NODE_REGISTRY: dict[str, type[BaseNode]] = {
    "input": InputNode,
    "llm": LLMNode,
    "embedding": EmbeddingNode,
    "stt": STTNode,
    "tts": TTSNode,
    "vision": VisionNode,
    "rag": RAGNode,
}


def create_node(node_type: str, node_id: str, **kwargs: Any) -> BaseNode:
    """Instantiate a node by its type identifier."""
    cls = NODE_REGISTRY.get(node_type, BaseNode)
    return cls(id=node_id, title=kwargs.pop("title", node_id), **kwargs)
