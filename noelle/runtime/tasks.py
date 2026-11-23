"""Hub pipeline tags -> Noelle modality families. Pure Python (shared with the UI)."""

from __future__ import annotations

from dataclasses import dataclass

# family -> playground used by the workspace page
TEXT = "text"
VISION = "vision"
DETECTION = "detection"
CLASSIFICATION = "classification"
ASR = "asr"
TTS = "tts"
DIFFUSION = "diffusion"
EMBEDDING = "embedding"

TASK_FAMILIES: dict[str, str] = {
    "text-generation": TEXT,
    "text2text-generation": TEXT,
    "conversational": TEXT,
    "summarization": TEXT,
    "translation": TEXT,
    "image-text-to-text": VISION,
    "image-to-text": VISION,
    "visual-question-answering": VISION,
    "document-question-answering": VISION,
    "video-text-to-text": VISION,
    "object-detection": DETECTION,
    "image-classification": CLASSIFICATION,
    "automatic-speech-recognition": ASR,
    "text-to-speech": TTS,
    "text-to-audio": TTS,
    "text-to-image": DIFFUSION,
    "image-to-image": DIFFUSION,
    "unconditional-image-generation": DIFFUSION,
    "sentence-similarity": EMBEDDING,
    "feature-extraction": EMBEDDING,
}


@dataclass(frozen=True)
class Modality:
    key: str
    label: str
    pipeline_tags: tuple[str, ...]
    description: str


# Discovery sidebar categories (PRD section 3).
MODALITIES: tuple[Modality, ...] = (
    Modality("text", "Text & Language", ("text-generation",),
             "Causal language models, instruction-following and chat."),
    Modality("vision", "Vision & Multimodal",
             ("image-text-to-text", "object-detection", "image-classification", "image-to-text"),
             "Vision-language models, OCR, detection and classification."),
    Modality("audio", "Audio & Speech", ("automatic-speech-recognition", "text-to-speech", "text-to-audio"),
             "Speech recognition and speech synthesis."),
    Modality("diffusion", "Diffusion & Synthesis", ("text-to-image", "image-to-image"),
             "Text-to-image and image-to-image pipelines."),
    Modality("embeddings", "Embeddings", ("sentence-similarity", "feature-extraction"),
             "Sentence embeddings for search and retrieval."),
)

MODALITY_BY_KEY = {m.key: m for m in MODALITIES}


def task_family(pipeline_tag: str | None) -> str | None:
    if not pipeline_tag:
        return None
    return TASK_FAMILIES.get(pipeline_tag)


def modality_for_tag(pipeline_tag: str | None) -> str | None:
    for modality in MODALITIES:
        if pipeline_tag in modality.pipeline_tags:
            return modality.key
    family = task_family(pipeline_tag)
    return {TEXT: "text", VISION: "vision", DETECTION: "vision", CLASSIFICATION: "vision", ASR: "audio",
            TTS: "audio", DIFFUSION: "diffusion", EMBEDDING: "embeddings"}.get(family or "")


def is_supported(pipeline_tag: str | None) -> bool:
    return task_family(pipeline_tag) is not None
