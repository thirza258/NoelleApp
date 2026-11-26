"""Voice helpers: sentence chunking for progressive TTS and silence detection."""

from __future__ import annotations

import re
from typing import Iterable

_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "fig", "al",
    "jl", "dll", "dst", "dsb", "sdr", "ny", "tn", "thn", "hal", "hlm",
}
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def _is_abbreviation(prefix: str) -> bool:
    token = prefix.rstrip(".!?…").rsplit(None, 1)[-1].lower().strip("([{\"'")
    return token in _ABBREVIATIONS


def split_sentences(text: str, *, min_chars: int = 40, max_chars: int = 400) -> list[str]:
    """Split generated text into TTS chunks.

    Keeps decimals (``3.5``), common abbreviations (``Dr.``, ``dst.``) and ellipses inside
    a sentence; merges short fragments until ``min_chars``; hard-splits very long ones at
    the last comma/space before ``max_chars``.
    """
    text = " ".join(str(text).split())
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start() + 1
        if _is_abbreviation(text[start:end]):
            continue
        pieces.append(text[start:match.end()].strip())
        start = match.end()
    if start < len(text):
        pieces.append(text[start:].strip())

    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        if not piece:
            continue
        buffer = f"{buffer} {piece}".strip()
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)

    chunks: list[str] = []
    for chunk in merged:
        while len(chunk) > max_chars:
            cut = max(chunk.rfind(", ", 0, max_chars), chunk.rfind(" ", 0, max_chars))
            if cut <= 0:
                cut = max_chars
            chunks.append(chunk[:cut].strip())
            chunk = chunk[cut:].lstrip(" ,")
        if chunk:
            chunks.append(chunk)
    return chunks


def is_silence(levels: Iterable[float], *, threshold: float = 0.015, quiet_frames: int = 12) -> bool:
    """True when the most recent ``quiet_frames`` microphone levels stay below ``threshold``."""
    recent = list(levels)[-quiet_frames:]
    return len(recent) >= quiet_frames and all(level < threshold for level in recent)


def estimate_tokens(text: str) -> int:
    return max(len(str(text)) // 4, 0)