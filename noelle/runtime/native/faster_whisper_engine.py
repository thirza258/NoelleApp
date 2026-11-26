"""faster-whisper / CTranslate2 speech recognition (optional native component)."""

from __future__ import annotations

import time
from typing import Any

from noelle.runtime import audio_dsp
from noelle.runtime.errors import DependencyMissing


class FasterWhisperEngine:
    engine = "faster-whisper"

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device
        self.model_type = "faster-whisper"

    @classmethod
    def load(cls, source: str, *, device: str = "auto", precision: str = "auto", **kwargs: Any) -> FasterWhisperEngine:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyMissing("faster-whisper is not installed.",
                                    hint='Install the "faster-whisper" component in Settings › Components.',
                                    component="faster-whisper") from exc
        target = "cuda" if device == "cuda" else "cpu"
        compute = "float16" if target == "cuda" and precision != "int8" else "int8"
        model = WhisperModel(source, device=target, compute_type=compute)
        return cls(model, target)

    def transcribe(self, audio_path: str, *, language: str | None = None, translate: bool = False,
                   timestamps: bool = True, **kwargs: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(audio_path, language=language or None,
                                               task="translate" if translate else "transcribe",
                                               vad_filter=True)
        text_parts: list[str] = []
        chunks: list[dict[str, Any]] = []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            text_parts.append(text)
            if timestamps:
                chunks.append({"start": float(getattr(segment, "start", 0.0)),
                               "end": float(getattr(segment, "end", 0.0)), "text": text})
        elapsed = time.perf_counter() - t0
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        audio, sr = audio_dsp.load_audio(audio_path)
        duration = duration or (len(audio) / sr if sr else 0.0)
        return {"text": " ".join(text_parts).strip(), "chunks": chunks, "duration_s": duration,
                "metrics": {"total_ms": elapsed * 1000.0,
                            "real_time_factor": elapsed / duration if duration else None,
                            "engine": self.engine},
                "envelope": audio_dsp.waveform_envelope(audio, 600)}

    def release(self) -> None:
        self.model = None