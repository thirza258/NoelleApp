"""Small numpy DSP helpers for the audio playground (no librosa/ffmpeg needed)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode with libsndfile (WAV/FLAC/OGG/MP3), downmix to mono float32, optionally resample."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if target_sr and sr != target_sr:
        mono = resample(mono, sr, target_sr)
        sr = target_sr
    return mono, int(sr)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, int(target_sr) // g, int(orig_sr) // g).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float, frame: int = 1024, hop: int = 256) -> np.ndarray:
    """Overlap-add time stretch: ``rate`` > 1 plays faster (shorter), pitch preserved."""
    if abs(rate - 1.0) < 1e-3 or audio.size < frame:
        return audio.astype(np.float32)
    window = np.hanning(frame).astype(np.float32)
    hop_in = max(1, int(round(hop * rate)))
    n_frames = max(1, (len(audio) - frame) // hop_in + 1)
    out_len = (n_frames - 1) * hop + frame
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        src = audio[i * hop_in: i * hop_in + frame]
        if len(src) < frame:
            src = np.pad(src, (0, frame - len(src)))
        out[i * hop: i * hop + frame] += src * window
        norm[i * hop: i * hop + frame] += window
    norm[norm < 1e-6] = 1.0
    return out / norm


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Tempo change with pitch preserved."""
    return time_stretch(audio, speed)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch without changing duration: resample, then stretch back."""
    if abs(semitones) < 1e-3:
        return audio.astype(np.float32)
    factor = 2.0 ** (semitones / 12.0)
    shifted = resample(audio, int(round(sr * factor)), sr)  # shorter and higher when factor > 1
    return time_stretch(shifted, 1.0 / factor)[: len(audio)]


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    top = float(np.max(np.abs(audio))) if audio.size else 0.0
    if top < 1e-9:
        return audio.astype(np.float32)
    return (audio * (peak / top)).astype(np.float32)


def waveform_envelope(audio: np.ndarray, bins: int = 400) -> list[float]:
    """Peak envelope for drawing waveforms."""
    if audio.size == 0:
        return []
    bins = max(1, min(bins, audio.size))
    usable = audio[: (audio.size // bins) * bins]
    return np.abs(usable.reshape(bins, -1)).max(axis=1).round(4).tolist()
