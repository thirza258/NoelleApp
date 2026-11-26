"""Kokoro TTS engine (optional native component)."""

from __future__ import annotations

import time
from typing import Any

from noelle import paths
from noelle.runtime import audio_dsp
from noelle.runtime.errors import BadRequest, DependencyMissing

_SAMPLE_RATE = 24_000
_LANG_CODES = {"en": "a", "ja": "j", "zh": "z", "es": "e", "fr": "f", "hi": "h", "it": "i", "pt": "p"}


class KokoroEngine:
    engine = "kokoro"
    sampling_rate = _SAMPLE_RATE

    def __init__(self, pipeline: Any, voice: str, language: str) -> None:
        self.pipeline = pipeline
        self.voice = voice
        self.language = language
        self.model_type = "kokoro"

    @classmethod
    def load(cls, source: str, *, device: str = "auto", language: str = "en", voice: str | None = None,
             **kwargs: Any) -> KokoroEngine:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise DependencyMissing("Kokoro TTS needs the 'kokoro' package.",
                                    hint='Install the "kokoro" component in Settings › Components.',
                                    component="kokoro") from exc
        code = _LANG_CODES.get(str(language).split("-")[0].lower())
        if not code:
            raise BadRequest(f"Kokoro has no voice for language '{language}'.",
                             hint="Use English or Japanese, or pick a different TTS model.")
        pipeline = KPipeline(lang_code=code, repo_id=source)
        return cls(pipeline, voice or "af_heart", language)

    def synthesize(self, text: str, *, voice: int | str = 0, speed: float = 1.0, pitch: float = 0.0,
                   **kwargs: Any) -> dict[str, Any]:
        import numpy as np

        if not str(text).strip():
            raise BadRequest("Nothing to synthesize: the text is empty.")
        voice_name = voice if isinstance(voice, str) else self.voice
        t0 = time.perf_counter()
        pieces = []
        for result in self.pipeline(text, voice=voice_name, speed=float(speed)):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not pieces:
            raise BadRequest("Kokoro produced no audio.")
        samples = np.concatenate(pieces)
        if pitch:
            samples = audio_dsp.pitch_shift(samples, _SAMPLE_RATE, float(pitch))
        samples = audio_dsp.peak_normalize(samples)
        elapsed = time.perf_counter() - t0
        path = paths.outputs_dir() / f"kokoro-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        sf.write(str(path), samples, _SAMPLE_RATE)
        duration = len(samples) / _SAMPLE_RATE
        return {"audio_path": str(path), "sampling_rate": _SAMPLE_RATE, "duration_s": duration,
                "metrics": {"total_ms": elapsed * 1000.0, "real_time_factor": elapsed / duration if duration else None,
                            "engine": self.engine},
                "envelope": audio_dsp.waveform_envelope(samples, 600)}

    def release(self) -> None:
        self.pipeline = None