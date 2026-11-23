"""environment_manager: names, profiles, manifests and subprocess handling (no network, no pip installs)."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

from noelle import paths
from noelle.runtime import environment_manager as em
from noelle.runtime.errors import Cancelled

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# names and profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["default", "py3.12-cuda_1", "A", "a" * 64, ".hidden-ok", "env..2"])
def test_valid_names(name: str) -> None:
    assert em.validate_name(name) == name


@pytest.mark.parametrize("name", ["", ".", "..", "...", "a b", "../x", "a/b", "a\\b", "a" * 65, "name\n", "é"])
def test_invalid_names(name: str) -> None:
    with pytest.raises(em.EnvironmentManagerError):
        em.validate_name(name)


def test_profiles_always_include_base_in_canonical_order() -> None:
    assert em.resolve_profiles(None) == ["base"]
    assert em.resolve_profiles([]) == ["base"]
    assert em.resolve_profiles(["server", "rag", "server"]) == ["base", "rag", "server"]
    assert em.resolve_profiles(["peft,diffusion"]) == ["base", "diffusion", "peft"]


def test_unknown_profile() -> None:
    with pytest.raises(em.EnvironmentManagerError, match="Unknown profile 'gpu'"):
        em.resolve_profiles(["gpu"])


def test_mlx_profile_needs_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(em, "is_apple_silicon", lambda: False)
    with pytest.raises(em.EnvironmentManagerError, match="Apple Silicon"):
        em.resolve_profiles(["mlx"])
    monkeypatch.setattr(em, "is_apple_silicon", lambda: True)
    assert em.resolve_profiles(["mlx"]) == ["base", "mlx"]


def test_cuda_quant_profile_refused_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(em.sys, "platform", "darwin")
    with pytest.raises(em.EnvironmentManagerError, match="CUDA"):
        em.resolve_profiles(["cuda-quant"])
    monkeypatch.setattr(em.sys, "platform", "linux")
    assert em.resolve_profiles(["cuda-quant"]) == ["base", "cuda-quant"]


def test_requirements_are_deduplicated_and_start_with_base() -> None:
    reqs = em.requirements_for(["server", "base"])
    assert reqs[: len(em.PROFILES["base"].requirements)] == list(em.PROFILES["base"].requirements)
    assert reqs[-3:] == ["fastapi>=0.110", "uvicorn[standard]>=0.30", "tornado>=6.1"]
    assert len(reqs) == len(set(reqs))


def _normalise(req: str) -> str:
    return re.sub(r"\s+", "", req.split(";")[0]).lower()


def test_profiles_mirror_pyproject_optional_dependencies() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not available")
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    for name, profile in em.PROFILES.items():
        if name == "base":
            core = [d for d in project["dependencies"] if not d.lower().startswith("pyside6")]
            expected = extras["runtime"] + core
        else:
            expected = extras[name]
        assert sorted(map(_normalise, profile.requirements)) == sorted(map(_normalise, expected)), name


def test_pip_command_is_headless() -> None:
    cmd = em.pip_install_command("/env/bin/python", ["torch>=2.4", "fastapi>=0.110"])
    assert cmd == ["/env/bin/python", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                   "--progress-bar", "off", "torch>=2.4", "fastapi>=0.110"]
    assert em.venv_command("/usr/bin/python3", Path("/x/env"), with_pip=False)[-1] == "--without-pip"


# ---------------------------------------------------------------------------
# run_streaming (real subprocesses)
# ---------------------------------------------------------------------------


def test_run_streaming_streams_lines_including_carriage_returns() -> None:
    lines: list[str] = []
    code, tail = em.run_streaming(
        [sys.executable, "-c",
         "import sys\nprint('one', flush=True)\nsys.stdout.write('50%|##\\r100%|####\\n')\nsys.stdout.flush()\n"
         "sys.stderr.write('warn\\n')\nsys.exit(4)"],
        on_line=lines.append)
    assert code == 4
    assert lines == ["one", "50%|##", "100%|####", "warn"]
    assert tail[-1] == "warn"


def test_run_streaming_stop_event_terminates_process() -> None:
    stop = threading.Event()
    started = time.monotonic()

    def on_line(line: str) -> None:
        if line == "working":
            stop.set()

    with pytest.raises(Cancelled):
        em.run_streaming([sys.executable, "-c", "import time\nprint('working', flush=True)\ntime.sleep(60)"],
                         on_line=on_line, stop_event=stop)
    assert time.monotonic() - started < 15


# ---------------------------------------------------------------------------
# create_environment with a fake Popen
# ---------------------------------------------------------------------------


class FakePopen:
    calls: list[FakePopen] = []
    pip_lines: list[str] = ["Collecting torch>=2.4", "Successfully installed torch-2.15.0"]
    pip_code = 0

    def __init__(self, cmd: list[str], **kwargs: Any) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 0
        self.returncode: int | None = None
        FakePopen.calls.append(self)
        if cmd[1:3] == ["-m", "venv"]:
            python = em.python_path(Path(cmd[3]))
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n")
            (Path(cmd[3]) / "pyvenv.cfg").write_text("home = /usr/bin\n")
            lines, self._code = ["venv ok"], 0
        else:
            lines, self._code = FakePopen.pip_lines, FakePopen.pip_code
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._code
        return self._code

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> type[FakePopen]:
    FakePopen.calls = []
    FakePopen.pip_lines = ["Collecting torch>=2.4", "Successfully installed torch-2.15.0"]
    FakePopen.pip_code = 0
    monkeypatch.setattr(em.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(em, "_terminate", lambda proc, grace=5.0: pytest.fail("must not signal a fake pid"))
    return FakePopen


def test_create_environment_runs_venv_then_headless_pip(fake_popen: type[FakePopen]) -> None:
    progress: list[str] = []
    info = em.create_environment("dev", ["server"], progress=progress.append)

    target = paths.environments_dir() / "dev"
    venv_call, pip_call = fake_popen.calls
    assert venv_call.cmd == [sys.executable, "-m", "venv", str(target)]
    requirements = em.requirements_for(["server"])
    assert pip_call.cmd == em.pip_install_command(em.python_path(target), requirements)
    assert pip_call.kwargs["env"]["PIP_NO_INPUT"] == "1"
    assert "PYTHONPATH" not in pip_call.kwargs["env"]
    assert "Successfully installed torch-2.15.0" in progress

    assert info == {"name": "dev", "path": str(target), "python": str(em.python_path(target)),
                    "profiles": ["base", "server"], "created_at": info["created_at"], "verified": False,
                    "torch_version": None, "device": None, "status": "ready"}
    manifest = em.read_manifest(target)
    assert manifest["requirements"] == requirements
    assert manifest["base_python"] == sys.executable
    assert isinstance(manifest["created_at"], float)


def test_create_environment_pip_failure_removes_folder(fake_popen: type[FakePopen]) -> None:
    fake_popen.pip_lines = ["Collecting nothing", "ERROR: No matching distribution found for torch>=2.4"]
    fake_popen.pip_code = 1
    with pytest.raises(em.EnvironmentManagerError, match="No matching distribution"):
        em.create_environment("broken", None)
    assert not (paths.environments_dir() / "broken").exists()


def test_create_environment_refuses_existing_and_invalid(fake_popen: type[FakePopen]) -> None:
    em.create_environment("dup", ["base"])
    with pytest.raises(em.EnvironmentManagerError, match="already exists"):
        em.create_environment("dup", ["base"])
    with pytest.raises(em.EnvironmentManagerError):
        em.create_environment("..", ["base"])
    with pytest.raises(em.EnvironmentManagerError, match="Unknown profile"):
        em.create_environment("new", ["nope"])
    assert len(fake_popen.calls) == 2


def test_create_environment_cancel_terminates_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_venv(target: Path, **_kwargs: Any) -> Path:
        python = em.python_path(target)
        python.parent.mkdir(parents=True)
        python.write_text("")
        return python

    monkeypatch.setattr(em, "create_venv", fake_venv)
    monkeypatch.setattr(em, "pip_install_command", lambda python, reqs: [
        sys.executable, "-c", "import time\nprint('Collecting torch', flush=True)\ntime.sleep(60)"])
    stop = threading.Event()

    def progress(line: str) -> None:
        if line == "Collecting torch":
            stop.set()

    started = time.monotonic()
    with pytest.raises(Cancelled):
        em.create_environment("slow", ["base"], progress=progress, stop_event=stop)
    assert time.monotonic() - started < 15
    assert not (paths.environments_dir() / "slow").exists()


# ---------------------------------------------------------------------------
# list / remove / python_for on fake environment folders
# ---------------------------------------------------------------------------


def _fake_env(name: str, manifest: dict[str, Any] | None = None, *, pyvenv: bool = True) -> Path:
    root = paths.environments_dir() / name
    python = em.python_path(root)
    python.parent.mkdir(parents=True)
    python.write_text("")
    if pyvenv:
        (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
    if manifest is not None:
        em.write_manifest(root, manifest)
    return root


def test_list_environments() -> None:
    assert em.list_environments() == []
    _fake_env("b-env", {"profiles": ["base"], "created_at": 2.0})
    _fake_env("a-env", {"profiles": ["base", "rag"], "created_at": 1.0, "verified": True,
                        "torch_version": "2.15.0", "device": "mps"})
    _fake_env("half")
    (paths.environments_dir() / "not valid").mkdir()
    (paths.environments_dir() / "notes.txt").write_text("x")

    rows = em.list_environments()
    assert [r["name"] for r in rows] == ["a-env", "b-env", "half"]
    a, b, half = rows
    assert a["verified"] is True and a["torch_version"] == "2.15.0" and a["device"] == "mps"
    assert a["profiles"] == ["base", "rag"] and a["status"] == "ready"
    assert a["python"] == str(em.python_path(paths.environments_dir() / "a-env"))
    assert set(a) >= {"name", "path", "python", "profiles", "created_at", "verified", "torch_version", "device"}
    assert b["verified"] is False and b["created_at"] == 2.0
    assert half["status"] == "incomplete" and half["profiles"] == []


def test_remove_environment() -> None:
    root = _fake_env("gone", {"profiles": ["base"]})
    assert em.remove_environment("gone") is True
    assert not root.exists()
    assert em.remove_environment("gone") is False


def test_remove_refuses_unsafe_targets(tmp_path: Path) -> None:
    paths.environments_dir().mkdir(parents=True, exist_ok=True)
    sentinel = paths.home() / "registry.db"
    sentinel.write_text("precious")
    for bad in ("..", ".", "../registry.db", "a/b"):
        with pytest.raises(em.EnvironmentManagerError):
            em.remove_environment(bad)
    assert sentinel.read_text() == "precious"

    plain = paths.environments_dir() / "plain"
    plain.mkdir()
    (plain / "data.txt").write_text("not a venv")
    with pytest.raises(em.EnvironmentManagerError, match="not a Noelle environment"):
        em.remove_environment("plain")
    assert (plain / "data.txt").exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "pyvenv.cfg").write_text("")
    (paths.environments_dir() / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(em.EnvironmentManagerError, match="not inside"):
        em.remove_environment("link")
    assert (outside / "pyvenv.cfg").exists()


def test_python_for() -> None:
    root = _fake_env("py", {"profiles": ["base"]})
    assert em.python_for("py") == em.python_path(root)
    with pytest.raises(em.EnvironmentManagerError, match="does not exist"):
        em.python_for("missing")
    with pytest.raises(em.EnvironmentManagerError):
        em.python_for("../py")


def test_real_venv_without_pip(tmp_path: Path) -> None:
    target = tmp_path / "venv"
    python = em.create_venv(target, with_pip=False)
    assert python == em.python_path(target)
    prefix = subprocess.run([str(python), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True,
                            timeout=60, check=True).stdout.strip()
    assert Path(prefix).resolve() == target.resolve()
    assert (target / "pyvenv.cfg").is_file()


# ---------------------------------------------------------------------------
# verify_environment
# ---------------------------------------------------------------------------


def _wrapper_env(name: str, profiles: list[str]) -> Path:
    """An environment whose interpreter execs the test interpreter (which has torch installed)."""
    root = _fake_env(name, {"name": name, "profiles": profiles, "created_at": 1.0, "verified": False})
    python = em.python_path(root)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    return root


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell wrapper")
def test_verify_environment_real_subprocess() -> None:
    root = _wrapper_env("wrapped", ["base", "server"])
    result = em.verify_environment("wrapped")
    assert result["verified"] is True, result["errors"]
    assert result["tensor_ok"] is True and result["noelle_ok"] is True
    assert result["device"] in ("cuda", "mps", "cpu")
    assert result["torch_version"]
    assert result["missing_modules"] == []
    manifest = em.read_manifest(root)
    assert manifest["verified"] is True and manifest["device"] == result["device"]
    row = em.list_environments()[0]
    assert row["verified"] is True and row["torch_version"] == result["torch_version"]


def test_verify_environment_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fake_env("bad", {"profiles": ["base", "gguf"], "verified": True})
    report = {"python_version": "3.12.1", "errors": ["torch: ModuleNotFoundError: No module named 'torch'"],
              "missing_modules": ["torch", "llama_cpp"], "tensor_ok": False, "noelle_ok": True}
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["cmd"], seen["env"] = cmd, kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, "noise\nNOELLE_VERIFY " + json.dumps(report) + "\n", "")

    monkeypatch.setattr(em.subprocess, "run", fake_run)
    result = em.verify_environment("bad")
    assert result["verified"] is False
    assert "llama_cpp" in json.loads(seen["cmd"][-1])
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT)
    assert any("missing modules: torch, llama_cpp" in e for e in result["errors"])
    assert em.read_manifest(root)["verified"] is False


def test_verify_environment_handles_crashing_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_env("crash", {"profiles": ["base"]})
    monkeypatch.setattr(em.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "Traceback...\nSegmentation fault\n"))
    result = em.verify_environment("crash")
    assert result["verified"] is False
    assert any("Segmentation fault" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str]) -> argparse.Namespace:
    from noelle.__main__ import build_parser

    return build_parser().parse_args(argv)


def test_cli_list_remove_verify(capsys: pytest.CaptureFixture[str]) -> None:
    assert em.main(_cli(["env", "list"])) == 0
    assert "No environments yet" in capsys.readouterr().out

    _fake_env("cli-env", {"profiles": ["base", "rag"], "verified": True, "torch_version": "2.15.0", "device": "mps"})
    assert em.main(_cli(["env", "list"])) == 0
    out = capsys.readouterr().out
    assert "cli-env" in out and "base,rag" in out and "2.15.0" in out

    assert em.main(_cli(["env", "remove", "nope"])) == 1
    assert "No environment named 'nope'" in capsys.readouterr().err
    assert em.main(_cli(["env", "remove", "cli-env"])) == 0
    assert em.main(_cli(["env", "verify", "cli-env"])) == 1
    assert "does not exist" in capsys.readouterr().err


def test_cli_create_errors_and_usage(capsys: pytest.CaptureFixture[str], fake_popen: type[FakePopen]) -> None:
    assert em.main(_cli(["env", "create", "bad name"])) == 1
    assert "Invalid environment name" in capsys.readouterr().err
    assert em.main(_cli(["env", "create", "ok", "--profile", "server"])) == 0
    out = capsys.readouterr().out
    assert "Successfully installed" in out and "Created 'ok' (base, server)" in out
    assert em.main(_cli(["env"])) == 2
