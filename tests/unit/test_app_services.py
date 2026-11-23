from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from noelle.app.services import hardware_manager, hf_client
from noelle.runtime import audio_dsp


# ----------------------------------------------------------------- token
def test_token_lives_in_keyring_only(memory_keyring: Any, tmp_path: Any) -> None:
    from noelle import paths

    assert hf_client.get_token() is None
    hf_client.set_token("  hf_secret  ")
    assert hf_client.get_token() == "hf_secret"
    assert memory_keyring.get_password(hf_client.KEYRING_SERVICE, hf_client.KEYRING_USER) == "hf_secret"
    settings_file = paths.settings_path()
    assert not settings_file.exists() or "hf_secret" not in settings_file.read_text()
    hf_client.clear_token()
    assert hf_client.get_token() is None


# ----------------------------------------------------------- model summary
def _fake_info(**overrides: Any) -> Any:
    base = dict(id="org/Cool-7B-Instruct", author="org", pipeline_tag="text-generation", library_name="transformers",
                downloads=1234, likes=56, trending_score=9, tags=["license:apache-2.0", "custom_code", "gguf"],
                safetensors=SimpleNamespace(parameters={"BF16": 7_000_000_000}), gguf=None, gated=False,
                created_at=None, last_modified=None, config={"architectures": ["CoolForCausalLM"]}, card_data=None,
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=14_000_000_000, lfs=None)])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_model_summary_from_info() -> None:
    summary = hf_client.ModelSummary.from_info(_fake_info())
    assert summary.param_count == 7_000_000_000
    assert summary.weight_bytes == 14_000_000_000
    assert summary.license == "apache-2.0"
    assert summary.custom_code and summary.prequantized and summary.is_gguf
    assert summary.architecture == "CoolForCausalLM"
    assert summary.family == "text" and summary.modality == "text"
    assert summary.siblings == [{"name": "model.safetensors", "size": 14_000_000_000}]
    fallback = hf_client.ModelSummary.from_info(_fake_info(safetensors=None, tags=[]))
    assert fallback.param_count == 7_000_000_000  # parsed from the name


class _Response:
    def __init__(self, items: list[dict[str, Any]], next_url: str | None) -> None:
        self._items = items
        self.links = {"next": {"url": next_url}} if next_url else {}
        self.status_code = 200

    def json(self) -> list[dict[str, Any]]:
        return self._items


def test_pager_follows_link_header(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    pages = {
        "first": _Response([{"id": "a/one", "pipeline_tag": "text-generation"},
                            {"id": "b/two", "pipeline_tag": "text-generation"}], "https://hub/page2"),
        "https://hub/page2": _Response([{"id": "c/three", "pipeline_tag": "text-generation"}], None),
    }

    class Session:
        def get(self, url: str, params: Any = None, headers: Any = None, timeout: Any = None) -> _Response:
            calls.append(url)
            return pages["first"] if params is not None else pages[url]

    import huggingface_hub.utils as hub_utils

    monkeypatch.setattr(hub_utils, "get_session", lambda: Session())
    monkeypatch.setattr(hub_utils, "hf_raise_for_status", lambda response: None)
    pager = hf_client.ModelPager(pipeline_tags=("text-generation",), page_size=2)
    first = pager.fetch_next()
    assert [m.id for m in first] == ["a/one", "b/two"] and not pager.exhausted
    second = pager.fetch_next()
    assert [m.id for m in second] == ["c/three"] and pager.exhausted
    assert pager.fetch_next() == []
    assert len(calls) == 2


# ------------------------------------------------------------------- audio
def test_audio_dsp_lengths() -> None:
    sr = 16000
    tone = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    assert len(audio_dsp.resample(tone, sr, 8000)) == 8000
    faster = audio_dsp.change_speed(tone, sr, 2.0)
    assert 0.4 * len(tone) < len(faster) < 0.6 * len(tone)
    shifted = audio_dsp.pitch_shift(tone, sr, 5)
    assert abs(len(shifted) - len(tone)) < 0.1 * len(tone)
    envelope = audio_dsp.waveform_envelope(tone, 100)
    assert len(envelope) == 100 and max(envelope) <= 1.0
    assert np.isclose(np.max(np.abs(audio_dsp.peak_normalize(tone * 0.1))), 0.95, atol=1e-3)


# ---------------------------------------------------------------- hardware
def test_gpu_detection_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware_manager.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware_manager, "_run", lambda args, timeout=2.0: "NVIDIA GeForce RTX 3090, 24576\n"
                        if "--query-gpu=name,memory.total" in args else "37, 2048\n")
    gpu = hardware_manager.detect_gpu()
    assert gpu.kind == "cuda" and gpu.name == "NVIDIA GeForce RTX 3090" and gpu.total_memory == 24576 * 1024 * 1024
    assert hardware_manager.sample_gpu(gpu) == (37.0, 2048 * 1024 * 1024)


def test_gpu_sampling_parses_ioreg(monkeypatch: pytest.MonkeyPatch) -> None:
    ioreg = '"PerformanceStatistics" = {"Device Utilization %"=42,"In use system memory"=173670400}'
    monkeypatch.setattr(hardware_manager, "_run", lambda args, timeout=2.0: ioreg)
    util, used = hardware_manager.sample_gpu(hardware_manager.GpuInfo("mps", "Apple M2 Pro GPU"))
    assert util == 42.0 and used == 173670400


# ---------------------------------------------------------- process isolation
def test_desktop_shell_never_imports_torch() -> None:
    """Build the whole main window in a fresh interpreter and check heavy ML modules stay unloaded."""
    script = textwrap.dedent("""
        import os, sys, tempfile
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["NOELLE_HOME"] = tempfile.mkdtemp()
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from noelle.app import main as appmain
        from noelle.app.ui.main_window import MainWindow
        ctx = appmain.build_context()
        window = MainWindow(ctx)
        from noelle.app.ui.navigation_bar import PAGES
        for key, _text, _icon in PAGES:
            window.navigate(key)
        window.pages["workspace"].set_model("org/some-model")
        window.pages["workspace"].tabs.setCurrentWidget(window.pages["workspace"].analysis_tab)
        from noelle.app.ui.settings_dialog import SettingsDialog
        SettingsDialog(ctx, window)
        heavy = sorted(m for m in ("torch", "transformers", "diffusers", "mlx", "sentence_transformers")
                       if m in sys.modules)
        print("HEAVY=" + ",".join(heavy))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith("HEAVY="))
    assert line == "HEAVY=", line


def test_external_links_only_open_http(monkeypatch: pytest.MonkeyPatch, qapp: Any) -> None:
    from PySide6.QtGui import QDesktopServices

    from noelle.app.ui.widgets.common import open_external_url

    opened: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    for url in ("file:///etc/passwd", "smb://host/share", "javascript:alert(1)", "x-apple.systempreferences:x",
                "https://", "vscode://open?file=/tmp/x"):
        assert open_external_url(url) is False
    assert open_external_url("https://huggingface.co/org/model") is True
    assert open_external_url("http://example.com/a") is True
    assert opened == ["https://huggingface.co/org/model", "http://example.com/a"]
