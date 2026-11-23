"""Every playground backend against a real worker with small cached models.

Covers the multi-modal acceptance criterion (text, vision, audio, diffusion and embeddings
all mount working backends) and document-grounding latency on a 50-page PDF.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from noelle.runtime.client import WorkerClient, spawn_worker
from noelle.runtime.hub_cache import cached_snapshot_path
from noelle.runtime.protocol import Command, Event
from tests.unit.test_rag_engine import write_pdf

pytestmark = pytest.mark.integration

TEXT = "HuggingFaceTB/SmolLM2-135M-Instruct"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
ASR = "openai/whisper-tiny"
TTS = "facebook/mms-tts-eng"
DETECT = "hustvl/yolos-tiny"
VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
DIFFUSION = "hf-internal-testing/tiny-stable-diffusion-torch"


def _need(*repos: str) -> None:
    missing = [r for r in repos if cached_snapshot_path(r) is None]
    if missing:
        pytest.skip(f"not cached: {', '.join(missing)}")


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Any:
    home = tmp_path_factory.mktemp("modalities-home")
    old = os.environ.get("NOELLE_HOME")
    os.environ["NOELLE_HOME"] = str(home)
    try:
        proc = spawn_worker(parent_pid=os.getpid())
        yield proc.client()
        proc.stop()
    finally:
        if old is None:
            os.environ.pop("NOELLE_HOME", None)
        else:
            os.environ["NOELLE_HOME"] = old


def _image(path: Path) -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([90, 60, 230, 180], fill=(220, 30, 30))
    image.save(path)
    return str(path)


def test_embeddings(client: WorkerClient) -> None:
    _need(EMBED)
    result = client.call(Command.EMBED, {"model_id": EMBED, "inputs": ["a cat", "a kitten", "tax law"]})
    assert result["dimension"] == 384 and len(result["embeddings"]) == 3
    import numpy as np

    a, b, c = (np.array(v) for v in result["embeddings"])
    assert float(a @ b) > float(a @ c)


def test_rag_on_a_50_page_pdf(client: WorkerClient, tmp_path: Path) -> None:
    _need(EMBED, TEXT)
    pages = []
    for i in range(50):
        topic = "The secret launch code for project Bluebird is 7-4-1-9." if i == 37 else \
            f"Page {i + 1} covers routine maintenance of conveyor belt number {i}, lubrication intervals and safety."
        pages.append(f"{topic} " + "Additional procedural notes about inspections and logs. " * 12)
    pdf = write_pdf(tmp_path / "manual.pdf", pages)
    started = time.perf_counter()
    ingest = client.call(Command.RAG_INGEST, {"collection": "manual", "files": [str(pdf)], "embedding_model": EMBED})
    assert ingest["documents_added"] == 1 and ingest["chunks_added"] >= 25  # chunks may span short pages
    assert time.perf_counter() - started < 60
    latencies = []
    for _ in range(5):
        result = client.call(Command.RAG_QUERY, {"collection": "manual", "query": "What is the Bluebird launch code?",
                                                 "top_k": 3})
        latencies.append(result["latency_ms"])
    assert result["hits"][0]["page"] in (37, 38) and "7-4-1-9" in result["hits"][0]["text"]
    assert statistics.median(latencies) < 150, latencies
    events = list(client.stream(Command.GENERATE_STREAM, {
        "model_id": TEXT, "messages": [{"role": "user", "content": "What is the Bluebird launch code?"}],
        "params": {"temperature": 0, "max_new_tokens": 32}, "grounding": {"rag": {"collection": "manual", "top_k": 3}}}))
    context = next(e["payload"] for e in events if e["event"] == Event.CONTEXT)
    assert context["sources"][0]["index"] == 1 and context["sources"][0]["page"] in (37, 38)
    final = events[-1]
    assert final["event"] == Event.GENERATION_COMPLETE and final["payload"]["sources"]


def test_tts_then_asr_round_trip(client: WorkerClient) -> None:
    _need(TTS, ASR)
    speech = client.call(Command.SYNTHESIZE_SPEECH, {"model_id": TTS, "text": "Hello world, this is a local test.",
                                                     "speed": 1.0})
    assert Path(speech["audio_path"]).is_file() and speech["duration_s"] > 0.5
    result = client.call(Command.TRANSCRIBE, {"model_id": ASR, "audio_path": speech["audio_path"], "language": "en"})
    assert result["text"].strip()
    assert "hello" in result["text"].lower() or "world" in result["text"].lower() or "test" in result["text"].lower()
    assert result["chunks"] and result["chunks"][0]["start"] is not None


def test_object_detection(client: WorkerClient, tmp_path: Path) -> None:
    _need(DETECT)
    result = client.call(Command.DETECT_OBJECTS, {"model_id": DETECT, "image_path": _image(tmp_path / "box.png"),
                                                  "threshold": 0.05})
    assert result["image_size"] == {"width": 320, "height": 240}
    assert isinstance(result["detections"], list)


def test_vision_language_model(client: WorkerClient, tmp_path: Path) -> None:
    _need(VLM)
    tokens: list[str] = []
    final: dict[str, Any] = {}
    for message in client.stream(Command.VISION_GENERATE, {
            "model_id": VLM, "task": "image-text-to-text", "image_path": _image(tmp_path / "red.png"),
            "prompt": "What color is the square?", "params": {"temperature": 0, "max_new_tokens": 20}}):
        if message["event"] == Event.TOKEN_STREAM:
            tokens.append(message["payload"]["text"])
        final = message
    assert final["event"] == Event.GENERATION_COMPLETE, final
    assert "".join(tokens).strip()


def test_diffusion_with_latent_previews(client: WorkerClient) -> None:
    _need(DIFFUSION)
    previews = 0
    final: dict[str, Any] = {}
    for message in client.stream(Command.GENERATE_IMAGE, {"model_id": DIFFUSION, "task": "text-to-image",
                                                          "prompt": "a red square", "steps": 3, "guidance": 5,
                                                          "width": 64, "height": 64, "seed": 42}):
        if message["event"] == Event.PROGRESS and message["payload"].get("preview_png"):
            previews += 1
        final = message
    assert final["event"] == Event.RESULT, final
    assert Path(final["payload"]["image_path"]).is_file() and final["payload"]["seed"] == 42
    assert previews >= 1
