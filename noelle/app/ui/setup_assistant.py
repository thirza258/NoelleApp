"""First-run setup assistant (also: Settings › About › Run setup assistant, or `noelle setup`).

Steps: welcome -> Hugging Face token -> hardware check -> optional downloads -> install -> done.
Nothing is downloaded unless the user ticks it on the "Optional downloads" step.
The token goes to the OS credential store, never to a file.
"""

from __future__ import annotations

import shutil
import threading
from typing import Any

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from noelle import __version__, paths
from noelle.app import icons, theme
from noelle.app.context import AppContext
from noelle.app.services import components, hf_client
from noelle.app.services.task_runner import runner
from noelle.app.ui.widgets.common import Banner, Panel, button, label
from noelle.runtime.preflight import MemoryBudget, assess_compatibility, format_bytes, format_params

STARTER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
STARTER_SIZE = "≈ 724 MB"


class WelcomePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Welcome to Noelle")
        self.setSubTitle("A few quick steps and you're ready to run open models on this computer.")
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        row = QHBoxLayout()
        art = QLabel()
        art.setPixmap(icons.app_icon().pixmap(QSize(72, 72)))
        row.addWidget(art, 0, Qt.AlignmentFlag.AlignTop)
        text = label("Noelle discovers models on the Hugging Face Hub, checks whether they fit your hardware, "
                     "downloads them with checksum verification and runs them locally — chat, vision, speech, "
                     "image generation and embeddings. Your prompts, documents and outputs stay on this "
                     "computer.", wrap=True)
        row.addWidget(text, 1)
        layout.addLayout(row)
        steps = Panel()
        steps_layout = QVBoxLayout(steps)
        steps_layout.setContentsMargins(16, 12, 16, 12)
        for line in ("1.  Connect your Hugging Face account (optional)", "2.  Check what your hardware can run",
                     "3.  Choose optional downloads — nothing downloads unless you tick it",
                     "4.  Start exploring"):
            steps_layout.addWidget(label(line, "muted"))
        layout.addWidget(steps)
        layout.addStretch(1)


class AccountPage(QWizardPage):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setTitle("Hugging Face account (optional)")
        self.setSubTitle("A token lets Noelle download gated or private models. Public models work without one.")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        self.existing = hf_client.get_token() is not None
        layout.addWidget(label("Your token is stored in this computer's credential vault (macOS Keychain, Windows "
                               "Credential Manager or Secret Service). It is only sent to huggingface.co and never "
                               "to model processes.", "muted", wrap=True))
        row = QHBoxLayout()
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("hf_…  (a token is already saved — leave empty to keep it)" if self.existing
                                      else "hf_…  (optional — leave empty to skip)")
        self.token.textChanged.connect(lambda _t: self.status.setText(""))
        row.addWidget(self.token, 1)
        verify = button("Verify", "check")
        verify.clicked.connect(self._verify)
        row.addWidget(verify)
        layout.addLayout(row)
        self.status = label("A token is already saved." if self.existing else "", "faint", wrap=True)
        layout.addWidget(self.status)
        create = button("Create a read token on huggingface.co", "external", "link")
        create.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens")))
        layout.addWidget(create)
        layout.addStretch(1)

    def _verify(self) -> None:
        token = self.token.text().strip()
        if not token:
            self.status.setText("Enter a token to verify." if not self.existing else "Using the saved token.")
            return
        self.status.setText("Checking with huggingface.co…")
        runner().submit(self.ctx.hf.whoami, token,
                        on_result=lambda info: self.status.setText(f"✓ Valid — signed in as {info.get('name')}"),
                        on_error=lambda msg: self.status.setText(f"✗ This token was rejected: {msg}"))

    def validatePage(self) -> bool:  # noqa: N802
        token = self.token.text().strip()
        if not token:
            return True
        try:
            hf_client.set_token(token)
        except Exception as exc:
            self.status.setText(f"Could not save the token in the credential store: {exc}")
            return False
        return True


class HardwarePage(QWizardPage):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setTitle("Your hardware")
        self.setSubTitle("Noelle sizes every model against this machine before you download it.")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        self.summary = label("", wrap=True)
        layout.addWidget(self.summary)
        self.advice = Banner("info", "")
        layout.addWidget(self.advice)
        self.storage = label("", "muted", wrap=True)
        layout.addWidget(self.storage)
        layout.addStretch(1)

    def initializePage(self) -> None:  # noqa: N802
        hardware = self.ctx.hardware
        budget = hardware.memory_budget(float(self.ctx.settings.get("runtime.safety_headroom", 0.15)))
        gpu = hardware.gpu
        kind = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (Metal)", "cpu": "CPU only"}.get(gpu.kind, gpu.kind)
        self.summary.setText(f"<b>{gpu.name}</b> — {kind}<br>Memory available to models: "
                             f"{format_bytes(budget.usable_bytes)} (after a {int(budget.headroom * 100)}% safety "
                             f"margin)")
        native, four_bit = largest_models(budget)
        self.advice.set_text(f"Models up to about <b>{format_params(native)}</b> parameters run natively here; "
                             f"up to about <b>{format_params(four_bit)}</b> with a 4-bit build. Larger models are "
                             "marked red in Discover." if native else
                             "This machine can run small models; larger ones will be marked in Discover.")
        cache = paths.hf_hub_cache(self.ctx.settings.get("storage.hf_home"))
        target = cache if cache.exists() else paths.home()
        try:
            free = shutil.disk_usage(target).free
        except OSError:
            free = 0
        self.storage.setText(f"Models are stored in {cache} ({format_bytes(free)} free). You can move this later "
                             "from Storage › Relocate cache.")


def largest_models(budget: MemoryBudget) -> tuple[int, int]:
    """Largest parameter counts that are green (native) and at least yellow (4-bit) for this budget."""
    native = four_bit = 0
    count = int(0.1e9)
    while count <= int(400e9):
        level = assess_compatibility(count, budget).level
        if level == "green":
            native = count
        if level in ("green", "yellow"):
            four_bit = count
        elif level == "red":
            break
        count += int(0.1e9) if count < int(10e9) else int(1e9)
    return native, four_bit


class DownloadsPage(QWizardPage):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setTitle("Optional downloads")
        self.setSubTitle("Nothing is downloaded unless you tick it. You can add these later at any time.")
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        self.release: components.ComponentRelease | None = None
        installed = components.installed_llama_cpp()
        self.llama = QCheckBox("llama.cpp tools — create and run GGUF models")
        self.llama.setEnabled(installed is None)
        layout.addWidget(self.llama)
        self.llama_info = label("Already installed." if installed else "Checking the latest release…", "faint",
                                wrap=True)
        layout.addWidget(self.llama_info)
        self.starter = QCheckBox(f"Starter chat model — {STARTER_MODEL.split('/')[-1]} ({STARTER_SIZE})")
        already = self.ctx.registry.is_downloaded(STARTER_MODEL)
        self.starter.setEnabled(not already)
        layout.addWidget(self.starter)
        layout.addWidget(label("Already downloaded." if already else
                               "A small instruction-tuned model from huggingface.co so you can chat right away.",
                               "faint", wrap=True))
        layout.addStretch(1)
        if installed is None:
            runner().submit(components.resolve_llama_cpp, on_result=self._resolved,
                            on_error=lambda msg: self.llama_info.setText(f"Release lookup failed ({msg}); you can "
                                                                         "install it later from Settings."))

    def _resolved(self, release: components.ComponentRelease) -> None:
        self.release = release
        self.llama_info.setText(f"{release.tag} · {format_bytes(release.asset_size)} from {release.homepage} "
                                f"({release.license}) · stored in {components.install_root()}")

    def wants_llama(self) -> bool:
        return self.llama.isChecked() and self.llama.isEnabled()

    def wants_starter(self) -> bool:
        return self.starter.isChecked() and self.starter.isEnabled()

    def nextId(self) -> int:  # noqa: N802
        wizard = self.wizard()
        if self.wants_llama() or self.wants_starter():
            return wizard.INSTALL
        return wizard.FINISH


class InstallPage(QWizardPage):
    def __init__(self, ctx: AppContext, downloads: DownloadsPage) -> None:
        super().__init__()
        self.ctx = ctx
        self.downloads = downloads
        self.setTitle("Downloading")
        self.setSubTitle("You can keep this open or finish — model downloads continue in the background.")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        self.llama_label = label("llama.cpp tools", "h3")
        self.llama_bar = QProgressBar()
        self.llama_bar.setRange(0, 1000)
        self.llama_status = label("", "faint", wrap=True)
        for widget in (self.llama_label, self.llama_bar, self.llama_status):
            layout.addWidget(widget)
        self.starter_label = label(STARTER_MODEL, "h3")
        self.starter_bar = QProgressBar()
        self.starter_bar.setRange(0, 1000)
        self.starter_status = label("", "faint", wrap=True)
        for widget in (self.starter_label, self.starter_bar, self.starter_status):
            layout.addWidget(widget)
        layout.addStretch(1)
        self._stop = threading.Event()
        self._llama_done = True
        ctx.downloads.job_updated.connect(self._on_job)
        ctx.downloads.job_finished.connect(self._on_job)

    def initializePage(self) -> None:  # noqa: N802
        want_llama = self.downloads.wants_llama()
        want_starter = self.downloads.wants_starter()
        for widget in (self.llama_label, self.llama_bar, self.llama_status):
            widget.setVisible(want_llama)
        for widget in (self.starter_label, self.starter_bar, self.starter_status):
            widget.setVisible(want_starter)
        if want_starter:
            self.ctx.downloads.start(STARTER_MODEL, summary=hf_client.ModelSummary(
                id=STARTER_MODEL, pipeline_tag="text-generation", license="apache-2.0", param_count=361_821_120))
            self.starter_status.setText("Starting…")
        if want_llama:
            self._llama_done = False
            self.llama_status.setText("Starting…")

            def work(progress: Any) -> dict[str, Any]:
                release = self.downloads.release or components.resolve_llama_cpp()
                return components.install_llama_cpp(release, progress=progress, stop_event=self._stop)

            runner().submit(work, on_progress=self._llama_progress, on_result=self._llama_ok,
                            on_error=self._llama_failed)

    def _llama_progress(self, data: dict[str, Any]) -> None:
        if data.get("stage") == "download" and data.get("total"):
            self.llama_bar.setValue(int(data["current"] / data["total"] * 1000))
            self.llama_status.setText(f"{data.get('file')} — {format_bytes(data['current'])} of "
                                      f"{format_bytes(data['total'])}")
        elif data.get("stage") in ("verified", "extract"):
            self.llama_status.setText("Verified, installing…" if data["stage"] == "extract" else
                                      f"{data.get('file')} verified")

    def _llama_ok(self, info: dict[str, Any]) -> None:
        self._llama_done = True
        self.llama_bar.setValue(1000)
        self.llama_status.setText(f"✓ Installed llama.cpp {info.get('tag')}")
        self.completeChanged.emit()

    def _llama_failed(self, message: str) -> None:
        self._llama_done = True
        self.llama_status.setText(f"✗ {message} — you can retry from Settings › Components.")
        self.completeChanged.emit()

    def _on_job(self, repo_id: str, *_args: Any) -> None:
        if repo_id != STARTER_MODEL:
            return
        job = self.ctx.downloads.job(repo_id)
        if job is None:
            return
        self.starter_bar.setValue(int(job.fraction * 1000))
        text = {"done": "✓ Downloaded", "error": f"✗ {job.error}", "planning": "Resolving files…"}.get(job.status)
        self.starter_status.setText(text or f"{format_bytes(job.downloaded_bytes)} of "
                                            f"{format_bytes(job.download_bytes)}")

    def isComplete(self) -> bool:  # noqa: N802
        return self._llama_done  # model downloads may continue in the background

    def cleanupPage(self) -> None:  # noqa: N802
        self._stop.set()


class FinishPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("You're all set")
        self.setSubTitle("Noelle is ready.")
        layout = QVBoxLayout(self)
        layout.addWidget(label("Start in Discover to find a model — every card shows whether it fits this "
                               "computer. Downloaded models appear under Discover › Downloaded and in Compare.",
                               wrap=True))
        layout.addWidget(label("You can change your token, downloads and runtime options any time in Settings, "
                               "and run this assistant again from Settings › About.", "muted", wrap=True))
        layout.addStretch(1)


class SetupAssistant(QWizard):
    WELCOME, ACCOUNT, HARDWARE, DOWNLOADS, INSTALL, FINISH = range(6)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Noelle Setup")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(680, 520)
        self.setOption(QWizard.WizardOption.NoBackButtonOnLastPage, True)
        self.setPixmap(QWizard.WizardPixmap.LogoPixmap, icons.app_icon().pixmap(QSize(48, 48)))
        self.setStyleSheet(f"QWizard, QWizardPage {{ background: {theme.BG}; }}")
        self.downloads_page = DownloadsPage(ctx)
        self.setPage(self.WELCOME, WelcomePage())
        self.setPage(self.ACCOUNT, AccountPage(ctx))
        self.setPage(self.HARDWARE, HardwarePage(ctx))
        self.setPage(self.DOWNLOADS, self.downloads_page)
        self.setPage(self.INSTALL, InstallPage(ctx, self.downloads_page))
        self.setPage(self.FINISH, FinishPage())
        self.setStartId(self.WELCOME)
        self.finished.connect(self._on_finished)

    def _on_finished(self, result: int) -> None:
        if result == QWizard.DialogCode.Accepted:
            self.ctx.settings.update({"setup.completed": True, "setup.version": __version__})


def should_run(settings: Any) -> bool:
    return not settings.get("setup.completed", False)


def run_assistant(ctx: AppContext, parent: QWidget | None = None) -> int:
    wizard = SetupAssistant(ctx, parent)
    return wizard.exec()

