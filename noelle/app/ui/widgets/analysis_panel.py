"""Noelle analysis: compatibility score, runtime/quantization advisor and capability facts.

``AnalysisSummary`` is the compact card in the Workspace sidebar; ``AnalysisPanel`` is
the full "Analysis" tab. Both render an :class:`noelle.runtime.advisor.Assessment`
and never compute anything themselves.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noelle.app import theme
from noelle.app.ui.widgets.common import FlowLayout, Panel, Pill, button, clear_layout, label
from noelle.runtime import advisor
from noelle.runtime.advisor import BUILD, INSTALL, READY, UNSUPPORTED, Assessment, RunOption
from noelle.runtime.preflight import format_bytes, format_params

STATUS_TEXT = {READY: "Ready", BUILD: "Build in Optimize", INSTALL: "Needs install", UNSUPPORTED: "Not on this device"}
STATUS_LEVEL = {READY: "green", BUILD: "info", INSTALL: "yellow", UNSUPPORTED: "unknown"}
ACTION_TEXT = {"load": "Load this build", "use": "Use local build", "quantize": "Build in Optimize",
               "install": "How to install…"}
RUNTIME_LABEL = {"transformers": "Transformers", "mlx": "MLX", "llama.cpp": "llama.cpp", "diffusers": "Diffusers",
                 "sentence-transformers": "Sentence Transformers"}
DEVICE_LABEL = {"mps": "Apple GPU (Metal)", "cuda": "NVIDIA GPU (CUDA)", "cpu": "CPU"}


def level_color(level: str) -> str:
    return theme.LEVEL_COLORS.get(level, theme.LEVEL_COLORS["unknown"])[0]


class ScoreDial(QWidget):
    """Circular 0-100 gauge."""

    def __init__(self, size: int = 72) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self._score: int | None = None
        self._level = "unknown"

    def set_score(self, score: int | None, level: str) -> None:
        self._score, self._level = score, level
        self.setToolTip("" if score is None else f"Compatibility {score} / 100")
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = max(4.0, self.width() * 0.085)
        rect = QRectF(width / 2 + 1, width / 2 + 1, self.width() - width - 2, self.height() - width - 2)
        track = QPen(QColor(theme.SURFACE_4), width)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track)
        painter.drawArc(rect, 225 * 16, -270 * 16)
        if self._score is not None:
            pen = QPen(QColor(level_color(self._level)), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, 225 * 16, int(-270 * 16 * max(0, min(self._score, 100)) / 100))
        w, h = float(self.width()), float(self.height())
        font = QFont(self.font())
        font.setPixelSize(int(h * 0.3))
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(theme.TEXT))
        text = "—" if self._score is None else str(self._score)
        painter.drawText(QRectF(0, h * 0.2, w, h * 0.42), Qt.AlignmentFlag.AlignCenter, text)
        small = QFont(self.font())
        small.setPixelSize(max(9, int(h * 0.13)))
        painter.setFont(small)
        painter.setPen(QColor(theme.TEXT_FAINT))
        painter.drawText(QRectF(0, h * 0.62, w, h * 0.18), Qt.AlignmentFlag.AlignCenter, "/ 100")


def _recommended_text(a: Assessment) -> str:
    option = a.recommended
    if option is None:
        return "No build of this model fits on this machine."
    runtime = RUNTIME_LABEL.get(option.runtime, option.runtime)
    memory = f" · ~{format_bytes(option.memory_bytes)}" if option.memory_bytes else ""
    return f"{option.label} with {runtime}{memory}"


class AnalysisSummary(QFrame):
    """Sidebar card: score, verdict, recommended build and speed, with a link to the full analysis."""

    details_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("card", True)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 12, 12, 12)
        row.setSpacing(12)
        self.dial = ScoreDial(64)
        row.addWidget(self.dial, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.grade = label("Analysing…", "h3")
        self.verdict = label("", "muted", wrap=True)
        self.recommended = label("", wrap=True)
        self.speed = label("", "faint", wrap=True)
        details = button("Full analysis", None, "link")
        details.clicked.connect(self.details_requested)
        for widget in (self.grade, self.verdict, self.recommended, self.speed, details):
            col.addWidget(widget)
        row.addLayout(col, 1)

    def set_assessment(self, a: Assessment | None) -> None:
        if a is None:
            self.dial.set_score(None, "unknown")
            self.grade.setText("Analysing…")
            for widget in (self.verdict, self.recommended, self.speed):
                widget.setText("")
            return
        self.dial.set_score(a.score, a.level)
        self.grade.setText(f"{a.grade} fit")
        self.verdict.setText(f"{a.compatibility.label} · {a.hardware.name or DEVICE_LABEL.get(a.hardware.device, '')}")
        self.recommended.setText(f"<b>Run as:</b> {_recommended_text(a)}")
        self.speed.setText(a.speed_text())


class AnalysisPanel(QWidget):
    """The Workspace "Analysis" tab."""

    profile_changed = Signal(str)
    action_requested = Signal(str, object)  # action, RunOption

    def __init__(self, profile: str = "balanced") -> None:
        super().__init__()
        self._assessment: Assessment | None = None
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(8, 12, 14, 16)
        root.setSpacing(16)

        # header
        head = QHBoxLayout()
        head.setSpacing(16)
        self.dial = ScoreDial(96)
        head.addWidget(self.dial, 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(label("NOELLE ANALYSIS", "eyebrow"))
        self.title = label("", "h2", wrap=True)
        self.subtitle = label("", "muted", wrap=True)
        self.hardware = label("", "faint", wrap=True)
        for widget in (self.title, self.subtitle, self.hardware):
            titles.addWidget(widget)
        head.addLayout(titles, 1)
        root.addLayout(head)

        # profiles
        root.addWidget(label("RUNTIME PROFILE", "eyebrow"))
        chips_host = QWidget()
        chips = FlowLayout(chips_host, spacing=6)
        self._profiles = QButtonGroup(self)
        self._profiles.setExclusive(True)
        self._profile_buttons: dict[str, Any] = {}
        for key, text in advisor.PROFILES.items():
            chip = button(text, None, "chip", checkable=True, tooltip=advisor.PROFILE_HINTS[key])
            chip.clicked.connect(lambda _=False, k=key: self._on_profile(k))
            self._profiles.addButton(chip)
            self._profile_buttons[key] = chip
            chips.addWidget(chip)
        root.addWidget(chips_host)
        self.profile_hint = label("", "faint", wrap=True)
        root.addWidget(self.profile_hint)

        # recommendation
        self.recommend_panel = Panel()
        rec = QVBoxLayout(self.recommend_panel)
        rec.setContentsMargins(16, 14, 16, 14)
        rec.setSpacing(6)
        rec.addWidget(label("RECOMMENDED", "eyebrow"))
        rec_row = QHBoxLayout()
        self.rec_title = label("", "h3", wrap=True)
        rec_row.addWidget(self.rec_title, 1)
        self.rec_status = Pill("", "unknown")
        rec_row.addWidget(self.rec_status, 0, Qt.AlignmentFlag.AlignTop)
        rec.addLayout(rec_row)
        self.rec_details = label("", "muted", wrap=True)
        rec.addWidget(self.rec_details)
        self.rec_reason = label("", "faint", wrap=True)
        rec.addWidget(self.rec_reason)
        self.rec_action = button("", "zap", "primary")
        self.rec_action.clicked.connect(self._on_recommended_action)
        action_row = QHBoxLayout()
        action_row.addWidget(self.rec_action)
        action_row.addStretch(1)
        rec.addLayout(action_row)
        root.addWidget(self.recommend_panel)

        # breakdown
        root.addWidget(label("SCORE BREAKDOWN", "eyebrow"))
        self.breakdown = QGridLayout()
        self.breakdown.setHorizontalSpacing(12)
        self.breakdown.setVerticalSpacing(8)
        self.breakdown.setColumnStretch(3, 1)
        root.addLayout(self.breakdown)
        self.cap_note = label("", "faint", wrap=True)
        root.addWidget(self.cap_note)

        # options
        root.addWidget(label("HOW SHOULD I RUN THIS?", "eyebrow"))
        self.options = QTableWidget(0, 7)
        self.options.setHorizontalHeaderLabels(["Build", "Runtime", "Memory", "Fits", "Quality", "Speed", "Status"])
        self.options.verticalHeader().setVisible(False)
        self.options.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.options.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.options.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.options.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.options.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.options.cellDoubleClicked.connect(self._on_option_activated)
        self.options.setToolTip("Double-click a row to use that build.")
        root.addWidget(self.options)
        self.options_note = label("Speeds marked “estimate” come from memory bandwidth; measured numbers come from "
                                  "your own Compare and Evaluation Lab runs.", "faint", wrap=True)
        root.addWidget(self.options_note)

        # capabilities
        root.addWidget(label("CAPABILITIES", "eyebrow"))
        self.facts = QFormLayout()
        self.facts.setHorizontalSpacing(16)
        self.facts.setVerticalSpacing(6)
        root.addLayout(self.facts)

        # notes
        self.notes_title = label("NOTES", "eyebrow")
        root.addWidget(self.notes_title)
        self.notes = QVBoxLayout()
        self.notes.setSpacing(4)
        root.addLayout(self.notes)
        root.addStretch(1)
        self.set_profile(profile)

    # ------------------------------------------------------------------ API
    @property
    def assessment(self) -> Assessment | None:
        return self._assessment

    def set_profile(self, profile: str) -> None:
        chip = self._profile_buttons.get(profile) or self._profile_buttons["balanced"]
        chip.setChecked(True)
        self.profile_hint.setText(advisor.PROFILE_HINTS.get(profile, ""))

    def set_assessment(self, a: Assessment | None) -> None:
        self._assessment = a
        if a is None:
            self.dial.set_score(None, "unknown")
            self.title.setText("Analysing…")
            return
        self.set_profile(a.profile)
        self.dial.set_score(a.score, a.level)
        self.title.setText(f"{a.score} / 100 · {a.grade}")
        self.subtitle.setText(f"{a.compatibility.label}. {a.compatibility.detail}")
        hw = a.hardware
        backends = [advisor_name for advisor_name, ok in sorted(hw.backends.items()) if ok]
        self.hardware.setText(f"On {hw.name or DEVICE_LABEL.get(hw.device, hw.device)} · "
                              f"{format_bytes(hw.usable_bytes)} usable · engines: {', '.join(backends) or 'none'}")
        self._render_recommendation(a)
        self._render_breakdown(a)
        self._render_options(a)
        self._render_facts(a)
        self._render_notes(a)

    # --------------------------------------------------------------- render
    def _render_recommendation(self, a: Assessment) -> None:
        option = a.recommended
        if option is None:
            self.rec_title.setText("Nothing fits")
            self.rec_status.set_level("red")
            self.rec_status.set_text("Too large")
            self.rec_details.setText(a.compatibility.detail)
            self.rec_reason.setText("")
            self.rec_action.hide()
            return
        self.rec_title.setText(_recommended_text(a))
        self.rec_status.set_level(STATUS_LEVEL.get(option.status, "unknown"))
        self.rec_status.set_text(STATUS_TEXT.get(option.status, option.status))
        parts = [f"Quality: {option.quality.lower()}", f"Speed: {a.speed_text()}"]
        if a.context_tokens:
            parts.append(f"Context: up to {a.context_tokens:,} tokens fit")
        parts.append(option.speed)
        self.rec_details.setText(" · ".join(parts))
        self.rec_reason.setText(option.reason or "")
        action = option.action
        self.rec_action.setVisible(action is not None)
        if action:
            self.rec_action.setText(ACTION_TEXT.get(action, action))

    def _render_breakdown(self, a: Assessment) -> None:
        clear_layout(self.breakdown)
        for row, item in enumerate(a.items):
            self.breakdown.addWidget(label(item.label, "muted"), row, 0)
            bar = QProgressBar()
            bar.setRange(0, item.max_points)
            bar.setValue(item.points)
            bar.setFixedWidth(120)
            ratio = item.points / max(item.max_points, 1)
            theme.set_prop(bar, "level", "green" if ratio >= 0.75 else "yellow" if ratio >= 0.4 else "red")
            self.breakdown.addWidget(bar, row, 1)
            self.breakdown.addWidget(label(f"{item.points}/{item.max_points}", "mono"), row, 2)
            self.breakdown.addWidget(label(item.note, "faint", wrap=True), row, 3)
        self.cap_note.setText(a.cap_note or "")
        self.cap_note.setVisible(bool(a.cap_note))

    def _render_options(self, a: Assessment) -> None:
        self.options.setRowCount(0)
        for option in a.options:
            row = self.options.rowCount()
            self.options.insertRow(row)
            measured = a.measured_for(option)
            speed = (f"{measured.tokens_per_second:.1f} tok/s measured" if measured and measured.tokens_per_second
                     else option.estimate_text() or option.speed)
            fits = "—" if option.fits is None else ("✓" if option.fits else "✗")
            star = "★ " if a.recommended is not None and option.key == a.recommended.key else ""
            values = [star + option.label, RUNTIME_LABEL.get(option.runtime, option.runtime),
                      f"~{format_bytes(option.memory_bytes)}" if option.memory_bytes else "?", fits,
                      option.quality, speed, STATUS_TEXT.get(option.status, option.status)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, option.key)
                if option.reason:
                    item.setToolTip(option.reason)
                if option.status == UNSUPPORTED or option.fits is False:
                    item.setForeground(QColor(theme.TEXT_FAINT))
                elif col == 3:
                    item.setForeground(QColor(theme.SUCCESS if option.fits else theme.TEXT_MUTED))
                self.options.setItem(row, col, item)
        height = self.options.horizontalHeader().height() + 4
        for row in range(self.options.rowCount()):
            height += self.options.rowHeight(row)
        self.options.setFixedHeight(min(height + 2, 420))

    def _render_facts(self, a: Assessment) -> None:
        while self.facts.rowCount():
            self.facts.removeRow(0)
        cap = a.capability

        def yes_no(value: bool | None, yes: str = "Yes", no: str = "No") -> str:
            return "Unknown" if value is None else (yes if value else no)

        io = f"{', '.join(cap.inputs) or '—'} → {', '.join(cap.outputs) or '—'}"
        quant = cap.quantization.upper() if cap.quantization else "None (original weights)"
        if cap.gguf_variants:
            quant += f" · {len(cap.gguf_variants)} GGUF files: " + ", ".join(v.quant for v in cap.gguf_variants[:8])
        remote = "Not required"
        if cap.remote_code.required:
            files = ", ".join(cap.remote_code.files[:6]) or ", ".join(cap.remote_code.auto_map[:3]) or "custom code"
            remote = f"Required — {files}"
        rows = [
            ("Task", cap.pipeline_tag or "—"),
            ("Inputs → outputs", io),
            ("Architecture", cap.architecture or "—"),
            ("Parameters", format_params(cap.param_count) if cap.param_count else "—"),
            ("Context window", f"{cap.context_length:,} tokens" if cap.context_length else "—"),
            ("Weights", ", ".join(cap.formats) or "Unknown until the file list loads"),
            ("Quantization", quant),
            ("Runs with", ", ".join(RUNTIME_LABEL.get(r, r) for r in cap.runtimes) or "No Noelle engine"),
            ("Local builds", ", ".join(cap.conversions) or "—"),
            ("Chat template", yes_no(cap.chat_template)),
            ("Tool calling", yes_no(cap.tool_calling, "Supported by the chat template", "Not in the chat template")),
            ("Languages", ", ".join(cap.languages[:12]) or "Not declared"),
            ("License", cap.license or "Not declared"),
            ("Access", "Gated — request access on huggingface.co" if cap.gated else "Open"),
            ("Remote code", remote),
        ]
        for key, value in rows:
            value_label = label(value, wrap=True, selectable=True)
            if key == "Remote code" and cap.remote_code.required:
                value_label.setStyleSheet(f"color: {theme.WARNING};")
            self.facts.addRow(label(key, "muted"), value_label)

    def _render_notes(self, a: Assessment) -> None:
        clear_layout(self.notes)
        for note in a.notes:
            self.notes.addWidget(label(f"• {note}", "muted", wrap=True))
        self.notes_title.setVisible(bool(a.notes))

    # -------------------------------------------------------------- actions
    def _on_profile(self, profile: str) -> None:
        self.profile_hint.setText(advisor.PROFILE_HINTS.get(profile, ""))
        self.profile_changed.emit(profile)

    def _on_recommended_action(self) -> None:
        a = self._assessment
        if a is not None and a.recommended is not None and a.recommended.action:
            self.action_requested.emit(a.recommended.action, a.recommended)

    def _on_option_activated(self, row: int, _column: int) -> None:
        a = self._assessment
        item = self.options.item(row, 0)
        if a is None or item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        option: RunOption | None = next((o for o in a.options if o.key == key), None)
        if option is not None and option.action:
            self.action_requested.emit(option.action, option)


class RemoteCodeNotice(QFrame):
    """Security notice for repositories that execute their own Python code when loading.

    States: ``blocked`` (default), ``once`` (allowed for this session), ``repo`` / ``publisher``
    (persistently trusted). Each choice is a separate signal so the page decides what to store.
    """

    allow_once = Signal()
    trust_repository = Signal()
    trust_publisher = Signal()
    revoke = Signal()
    review = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.state = "blocked"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        self.title = label("", wrap=True)
        self.title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.title)
        self.files = label("", "mono", wrap=True)
        layout.addWidget(self.files)
        self.risk = label("", "faint", wrap=True)
        layout.addWidget(self.risk)
        host = QWidget()
        self.buttons = FlowLayout(host, spacing=6)
        layout.addWidget(host)
        self.hide()

    def set_state(self, files: tuple[str, ...], auto_map: tuple[str, ...], publisher: str, state: str) -> None:
        self.state = state
        self.buttons.clear()
        listed = list(files[:6]) or list(auto_map[:4])
        more = len(files) - 6 if len(files) > 6 else 0
        self.files.setText(", ".join(listed) + (f" (+{more} more)" if more else "") if listed else "")
        self.files.setVisible(bool(listed))
        if state in ("repo", "publisher"):
            theme.set_prop(self, "banner", "info")
            whom = "this repository" if state == "repo" else f"every repository from {publisher}"
            self.title.setText(f"<b>Remote code allowed</b> — you trust {whom}.")
            self.risk.setText("Its Python files run inside the model runtime when the model loads.")
            self._add("Revoke trust", self.revoke)
        elif state == "once":
            theme.set_prop(self, "banner", "warning")
            self.title.setText("<b>Remote code allowed for this session only.</b>")
            self.risk.setText("Noelle will ask again next time.")
            self._add("Trust repository", self.trust_repository)
            self._add("Revoke", self.revoke)
        else:
            theme.set_prop(self, "banner", "warning")
            self.title.setText("<b>Security: this model ships executable Python code.</b>")
            self.risk.setText("The code runs in the model runtime as soon as the model loads. Review it before "
                              "allowing it, and only trust publishers you know.")
            self._add("Review files", self.review)
            self._add("Allow once", self.allow_once)
            self._add("Trust repository", self.trust_repository)
            if publisher:
                self._add(f"Trust {publisher}", self.trust_publisher)
        self.show()

    def _add(self, text: str, signal: Any) -> None:
        widget = button(text, None, "ghost" if text.startswith(("Revoke", "Review")) else None)
        widget.clicked.connect(signal)
        self.buttons.addWidget(widget)
