"""Audio playground: speech recognition (file or microphone) and speech synthesis."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle.app.context import AppContext
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.audio_visualizer import MicRecorder, WaveformView, request_microphone
from noelle.app.ui.widgets.common import ErrorLine, Spinner, button, human_duration, label
from noelle.runtime import tasks
from noelle.runtime.protocol import Command

AUDIO_FILTER = "Audio (*.wav *.flac *.ogg *.mp3 *.m4a *.aiff *.aif);;All files (*)"
LANGUAGES = [("Auto-detect", ""), ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
             ("Italian", "it"), ("Portuguese", "pt"), ("Dutch", "nl"), ("Indonesian", "id"), ("Japanese", "ja"),
             ("Korean", "ko"), ("Chinese", "zh"), ("Hindi", "hi"), ("Arabic", "ar"), ("Russian", "ru")]


def _envelope_for(path: str) -> list[float]:
    from noelle.runtime.audio_dsp import load_audio, waveform_envelope

    audio, _sr = load_audio(path)
    return waveform_envelope(audio, 600)


def _srt_time(seconds: float | None) -> str:
    seconds = float(seconds or 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{int((secs % 1) * 1000):03d}"


class AudioPlayground(QWidget):
    def __init__(self, ctx: AppContext, payload: Callable[[], dict[str, Any]], family: str) -> None:
        super().__init__()
        self.ctx = ctx
        self.payload = payload
        self.family = family
        self.audio_path: str | None = None
        self.output_path: str | None = None
        self._chunks: list[dict[str, Any]] = []
        self._handle: Any = None
        self._player: Any = None
        self._audio_out: Any = None
        self.recorder = MicRecorder(self)
        self.recorder.envelope.connect(lambda env: self.wave.set_envelope(env, live=True))
        self.recorder.error.connect(lambda msg: self.ctx.toast(msg, "error"))

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)
        if family == tasks.ASR:
            self._build_asr(root)
        else:
            self._build_tts(root)

    # ================================================================== ASR
    def _build_asr(self, root: QVBoxLayout) -> None:
        root.addWidget(label("Speech recognition", "h2"))
        root.addWidget(label("Transcribe a file or a microphone recording. Audio never leaves this machine.", "muted"))
        source = QHBoxLayout()
        pick = button("Choose file…", "folder")
        pick.clicked.connect(self._pick_audio)
        source.addWidget(pick)
        self.record_btn = button("Record", "mic", checkable=True)
        self.record_btn.toggled.connect(self._toggle_record)
        source.addWidget(self.record_btn)
        self.file_label = label("Drop an audio file here", "faint")
        source.addWidget(self.file_label, 1)
        root.addLayout(source)
        self.setAcceptDrops(True)
        self.wave = WaveformView()
        root.addWidget(self.wave)
        options = QHBoxLayout()
        options.addWidget(label("Language", "muted"))
        self.language = QComboBox()
        for text, code in LANGUAGES:
            self.language.addItem(text, code)
        options.addWidget(self.language)
        self.timestamps = QCheckBox("Timestamps")
        self.timestamps.setChecked(True)
        options.addWidget(self.timestamps)
        self.translate = QCheckBox("Translate to English")
        options.addWidget(self.translate)
        options.addStretch(1)
        self.run_btn = button("Transcribe", "play", "primary")
        self.run_btn.clicked.connect(self._transcribe)
        options.addWidget(self.run_btn)
        self.spinner = Spinner(16)
        options.addWidget(self.spinner)
        root.addLayout(options)
        self.status = label("", "faint", wrap=True)
        root.addWidget(self.status)
        self.error = ErrorLine()
        root.addWidget(self.error)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("The transcript appears here.")
        self.transcript.setMaximumHeight(130)
        root.addWidget(self.transcript)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Text"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        root.addWidget(self.table, 1)
        export = QHBoxLayout()
        export.addStretch(1)
        srt = button("Export .srt", "save", "ghost")
        srt.clicked.connect(self._export_srt)
        export.addWidget(srt)
        txt = button("Export .txt", "save", "ghost")
        txt.clicked.connect(self._export_txt)
        export.addWidget(txt)
        root.addLayout(export)

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if self.family == tasks.ASR and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.set_audio(url.toLocalFile())
                return

    def _pick_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio", "", AUDIO_FILTER)
        if path:
            self.set_audio(path)

    def set_audio(self, path: str) -> None:
        self.audio_path = path
        self.file_label.setText(Path(path).name)
        runner().submit(_envelope_for, path, on_result=lambda env: self.wave.set_envelope(env),
                        on_error=lambda msg: self.file_label.setText(f"{Path(path).name} (preview unavailable: {msg})"))

    def _toggle_record(self, on: bool) -> None:
        if on:
            def granted(ok: bool) -> None:
                if not ok:
                    self.record_btn.setChecked(False)
                    self.ctx.toast("Microphone access is blocked. Allow it in System Settings › Privacy & Security › "
                                   "Microphone.", "error")
                    return
                self.recorder.start()
                if self.recorder.recording:
                    self.record_btn.setText("Stop")
                    self.file_label.setText("Recording…")
                else:
                    self.record_btn.setChecked(False)

            request_microphone(granted)
        else:
            self.record_btn.setText("Record")
            if self.recorder.recording:
                path = self.recorder.stop()
                if path:
                    self.set_audio(path)

    def _transcribe(self) -> None:
        if self._handle is not None:
            return
        if not self.audio_path:
            self.ctx.toast("Choose or record audio first.", "info")
            return
        payload = {**self.payload(), "audio_path": self.audio_path, "language": self.language.currentData() or None,
                   "translate": self.translate.isChecked(), "timestamps": self.timestamps.isChecked()}
        self._run(Command.TRANSCRIBE, payload, self._on_transcribed)

    def _on_transcribed(self, result: dict[str, Any]) -> None:
        self.transcript.setPlainText(result.get("text", ""))
        self._chunks = result.get("chunks") or []
        self.table.setRowCount(len(self._chunks))
        for i, chunk in enumerate(self._chunks):
            self.table.setItem(i, 0, QTableWidgetItem(f"{chunk.get('start') or 0:.2f}s"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{chunk.get('end') or 0:.2f}s" if chunk.get("end") else "—"))
            self.table.setItem(i, 2, QTableWidgetItem(chunk.get("text", "")))
        metrics = result.get("metrics") or {}
        rtf = metrics.get("real_time_factor")
        self.status.setText(f"{human_duration(result.get('duration_s'))} of audio in {metrics.get('total_ms', 0) / 1000:.1f} s"
                            + (f" · {1 / rtf:.1f}× real time" if rtf else ""))

    def _export_srt(self) -> None:
        if not self._chunks:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export subtitles", "transcript.srt", "SubRip (*.srt)")
        if path:
            blocks = [f"{i}\n{_srt_time(c.get('start'))} --> {_srt_time(c.get('end') or c.get('start'))}\n"
                      f"{c.get('text', '')}\n" for i, c in enumerate(self._chunks, start=1)]
            Path(path).write_text("\n".join(blocks), encoding="utf-8")

    def _export_txt(self) -> None:
        text = self.transcript.toPlainText()
        if not text:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export transcript", "transcript.txt", "Text (*.txt)")
        if path:
            Path(path).write_text(text, encoding="utf-8")

    # ================================================================== TTS
    def _build_tts(self, root: QVBoxLayout) -> None:
        root.addWidget(label("Speech synthesis", "h2"))
        self.text = QPlainTextEdit("Noelle runs open models right on your own machine.")
        self.text.setFixedHeight(110)
        root.addWidget(self.text)
        controls = QHBoxLayout()
        controls.setSpacing(14)
        controls.addWidget(label("Voice", "muted"))
        self.voice = QSpinBox()
        self.voice.setRange(0, 99)
        self.voice.setToolTip("Speaker index for multi-speaker models")
        controls.addWidget(self.voice)
        controls.addWidget(label("Speed", "muted"))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(50, 200)
        self.speed.setValue(100)
        self.speed.setFixedWidth(140)
        self.speed_value = label("1.00×", "mono")
        self.speed.valueChanged.connect(lambda v: self.speed_value.setText(f"{v / 100:.2f}×"))
        controls.addWidget(self.speed)
        controls.addWidget(self.speed_value)
        controls.addWidget(label("Pitch", "muted"))
        self.pitch = QSlider(Qt.Orientation.Horizontal)
        self.pitch.setRange(-12, 12)
        self.pitch.setFixedWidth(140)
        self.pitch_value = label("+0 st", "mono")
        self.pitch.valueChanged.connect(lambda v: self.pitch_value.setText(f"{v:+d} st"))
        controls.addWidget(self.pitch)
        controls.addWidget(self.pitch_value)
        controls.addStretch(1)
        self.run_btn = button("Synthesize", "volume", "primary")
        self.run_btn.clicked.connect(self._synthesize)
        controls.addWidget(self.run_btn)
        self.spinner = Spinner(16)
        controls.addWidget(self.spinner)
        root.addLayout(controls)
        self.status = label("", "faint", wrap=True)
        root.addWidget(self.status)
        self.error = ErrorLine()
        root.addWidget(self.error)
        self.wave = WaveformView(120)
        self.wave.seek_requested.connect(self._seek)
        root.addWidget(self.wave)
        playback = QHBoxLayout()
        self.play_btn = button("Play", "play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        playback.addWidget(self.play_btn)
        self.save_btn = button("Save .wav…", "save", "ghost")
        self.save_btn.clicked.connect(self._save_wav)
        self.save_btn.setEnabled(False)
        playback.addWidget(self.save_btn)
        playback.addStretch(1)
        root.addLayout(playback)
        root.addStretch(1)

    def _synthesize(self) -> None:
        if self._handle is not None:
            return
        text = self.text.toPlainText().strip()
        if not text:
            return
        payload = {**self.payload(), "text": text, "voice": self.voice.value(), "speed": self.speed.value() / 100,
                   "pitch": float(self.pitch.value())}
        self._run(Command.SYNTHESIZE_SPEECH, payload, self._on_synthesized)

    def _on_synthesized(self, result: dict[str, Any]) -> None:
        self.output_path = result.get("audio_path")
        self.wave.set_envelope(result.get("envelope") or [])
        metrics = result.get("metrics") or {}
        self.status.setText(f"{result.get('duration_s', 0):.1f} s at {result.get('sampling_rate')} Hz · generated in "
                            f"{metrics.get('total_ms', 0) / 1000:.1f} s")
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._ensure_player()
        self._player.setSource(QUrl.fromLocalFile(self.output_path))

    def _ensure_player(self) -> None:
        if self._player is None:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._audio_out = QAudioOutput(self)
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.positionChanged.connect(self._on_position)
            self._player.playbackStateChanged.connect(self._on_playback_state)

    def _toggle_play(self) -> None:
        if self._player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer

        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_position(self, position: int) -> None:
        duration = self._player.duration() if self._player else 0
        self.wave.set_playhead(position / duration if duration else None)

    def _on_playback_state(self, state: Any) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        self.play_btn.setText("Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play")

    def _seek(self, fraction: float) -> None:
        if self._player is not None and self._player.duration():
            self._player.setPosition(int(self._player.duration() * fraction))

    def _save_wav(self) -> None:
        if not self.output_path:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save audio", Path(self.output_path).name, "WAV (*.wav)")
        if path:
            shutil.copyfile(self.output_path, path)

    # =============================================================== shared
    def _run(self, cmd: Command, payload: dict[str, Any], on_done: Callable[[dict[str, Any]], None]) -> None:
        self.error.clear()
        self.spinner.start()
        self.run_btn.setEnabled(False)
        self.status.setText("Working…")
        handle = self.ctx.runtime.request(cmd, payload)
        handle.status.connect(lambda p: self.status.setText(p.get("message", "")))

        def finished(result: dict[str, Any]) -> None:
            self._handle = None
            self.spinner.stop()
            self.run_btn.setEnabled(True)
            on_done(result)

        def failed(error: dict[str, Any]) -> None:
            self._handle = None
            self.spinner.stop()
            self.run_btn.setEnabled(True)
            self.status.setText("Failed.")
            self.error.set_error(str(error.get("message") or "The request failed."),
                                 str(error["hint"]) if error.get("hint") else None)

        handle.finished.connect(finished)
        handle.failed.connect(failed)
        self._handle = handle

    def shutdown(self) -> None:
        if self.recorder.recording:
            self.recorder.stop()
        if self._player is not None:
            self._player.stop()
