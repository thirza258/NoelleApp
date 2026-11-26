"""Local voice turn controller: ASR -> LLM -> sentence-level TTS, one worker job.

The worker runs the whole turn on its GPU lane (``VOICE_TURN``); the shell only
records, forwards events and queues audio playback. State is explicit so the UI can
show Listening / Transcribing / Generating / Speaking, and barge-in cancels both
generation and playback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal

from noelle.runtime.protocol import Command, Event

log = logging.getLogger("noelle.voice")


class VoiceSession(QObject):
    state_changed = Signal(str)
    transcript = Signal(str)
    token = Signal(str)
    audio_ready = Signal(str)
    finished = Signal(dict)
    failed = Signal(dict)

    STATES = ("idle", "recording", "transcribing", "generating", "speaking")

    def __init__(self, ctx: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.state = "idle"
        self._handle: Any = None
        self._pending_audio: list[str] = []
        self._player: Any = None
        self._audio_out: Any = None
        self._current_audio: str | None = None
        self._speaking_done = False
        self._text = ""

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def set_recording(self, recording: bool) -> None:
        if recording:
            self._set_state("recording")
        elif self.state == "recording":
            self._set_state("idle")

    def asr_model(self) -> str | None:
        return self._model_for("voice.asr_model", "asr")

    def tts_model(self) -> str | None:
        return self._model_for("voice.tts_model", "tts")

    def _model_for(self, setting: str, family: str) -> str | None:
        configured = self.ctx.settings.get(setting)
        if configured:
            return str(configured)
        try:
            records = [r for r in self.ctx.registry.list_models(family=family) if r.exists_on_disk()]
        except Exception:  # noqa: BLE001
            records = []
        return records[0].repo_id if records else None

    def start_turn(self, audio_path: str, messages: list[dict[str, Any]], *, llm_payload: dict[str, Any],
                   system_prompt: str | None = None, params: dict[str, Any] | None = None,
                   language: str | None = None, voice: int = 0, speed: float = 1.0) -> bool:
        asr = self.asr_model()
        tts = self.tts_model()
        if not asr or not tts:
            self.failed.emit({"code": "VOICE_MODELS_MISSING",
                              "message": "Voice needs a downloaded speech-recognition and speech-synthesis model.",
                              "hint": "Download an ASR model (e.g. whisper-small) and a TTS model (e.g. SpeechT5), "
                                      "then choose them in Settings › Voice.",
                              "suggested_actions": ["download_asr", "download_tts", "configure_voice"]})
            return False
        self._text = ""
        self._pending_audio = []
        self._speaking_done = False
        self._set_state("transcribing")
        payload = {"audio_path": audio_path, "messages": messages, "system_prompt": system_prompt,
                   "params": params or {}, "language": language or self.ctx.settings.get("voice.language") or None,
                   "voice": voice, "speed": speed,
                   "asr": {"model_id": asr, "device": "auto", "language": language},
                   "tts": {"model_id": tts, "device": "auto", "language": language},
                   "llm": dict(llm_payload)}
        handle = self.ctx.runtime.request(Command.VOICE_TURN, payload)
        handle.event.connect(self._on_event)
        handle.token.connect(self._on_token)
        handle.finished.connect(self._on_finished)
        handle.failed.connect(self._on_failed)
        self._handle = handle
        return True

    def _on_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == Event.TRANSCRIPT:
            self._set_state("generating")
            self.transcript.emit(str(payload.get("text") or ""))
        elif event == Event.AUDIO_CHUNK:
            path = payload.get("audio_path")
            if path and Path(path).is_file():
                self._pending_audio.append(str(path))
                self.audio_ready.emit(str(path))
                self._play_next()

    def _on_token(self, text: str) -> None:
        if self.state == "transcribing":
            self._set_state("generating")
        self._text += text
        self.token.emit(text)

    def _on_finished(self, payload: dict[str, Any]) -> None:
        self._handle = None
        self._text = str(payload.get("text") or self._text)
        self._speaking_done = True
        if not self._pending_audio and self._current_audio is None:
            self._set_state("idle")
        else:
            self._set_state("speaking")
        self.finished.emit(payload)

    def _on_failed(self, payload: dict[str, Any]) -> None:
        self._handle = None
        self._set_state("idle")
        self.failed.emit(payload)

    # ------------------------------------------------------------- playback
    def _ensure_player(self) -> None:
        if self._player is None:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._audio_out = QAudioOutput(self)
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.mediaStatusChanged.connect(self._on_media_status)
            self._player.playbackStateChanged.connect(self._on_playback_state)

    def _play_next(self) -> None:
        if self._current_audio is not None or not self._pending_audio:
            return
        self._ensure_player()
        self._current_audio = self._pending_audio.pop(0)
        self._player.setSource(QUrl.fromLocalFile(self._current_audio))
        self._player.play()
        self._set_state("speaking")

    def _on_media_status(self, status: Any) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        if status in (QMediaPlayer.MediaStatus.EndOfMedia, QMediaPlayer.MediaStatus.InvalidMedia):
            self._current_audio = None
            if self._pending_audio:
                self._play_next()
            elif self._speaking_done:
                self._set_state("idle")

    def _on_playback_state(self, state: Any) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        if state == QMediaPlayer.PlaybackState.StoppedState and self._current_audio is None:
            if self._pending_audio:
                self._play_next()
            elif self._speaking_done and self.state == "speaking":
                self._set_state("idle")

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._pending_audio = []
        self._speaking_done = False
        if self._player is not None:
            self._player.stop()
        self._current_audio = None
        self._set_state("idle")

    def shutdown(self) -> None:
        self.stop()
        if self._player is not None:
            self._player.deleteLater()
            self._player = None