"""Isolated runtime environments (``noelle env create|list|remove|verify``).

Each environment is a plain ``python -m venv`` under
``noelle.paths.environments_dir()/<name>/`` with a ``noelle-env.json`` manifest.
Packages come from *profiles* that mirror the optional-dependency groups in
``pyproject.toml``; ``base`` is always installed. The Noelle package itself is
not installed into the environment: workers started from it find the source
tree through ``PYTHONPATH`` (see :func:`noelle.runtime.client.build_worker_env`).

pip always runs headless (``python -m pip install --disable-pip-version-check
--no-input ...``); its output is streamed line by line to a progress callback
and the process is terminated when the caller's stop event is set.

Safe to import from the desktop shell: no torch, no Qt.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from noelle import __version__, paths
from noelle.runtime.errors import Cancelled

MANIFEST_NAME = "noelle-env.json"
VERIFY_MARKER = "NOELLE_VERIFY"
_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")

OnLine = Callable[[str], None]


class EnvironmentManagerError(RuntimeError):
    """A user-facing failure (bad name, unknown profile, pip error, ...)."""


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    requirements: tuple[str, ...]
    modules: tuple[str, ...]  # top-level import names checked by verify_environment
    requires: str | None = None  # None | "apple_silicon" | "cuda_platform"


# Mirrors [project.optional-dependencies] in pyproject.toml (a unit test keeps them in sync).
# Environment markers are resolved here instead: "mlx" is refused off Apple Silicon.
PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("base", "Inference runtime: PyTorch, transformers and the worker's IPC dependencies.",
                ("torch>=2.4", "transformers>=4.50", "accelerate>=1.0", "safetensors>=0.4", "numpy>=1.26",
                 "pillow>=10", "soundfile>=0.12", "pyzmq>=26", "psutil>=5.9", "huggingface_hub>=1.0",
                 "keyring>=25"),
                ("torch", "transformers", "accelerate", "safetensors", "numpy", "PIL", "soundfile", "zmq",
                 "psutil", "huggingface_hub", "keyring")),
        Profile("diffusion", "Text-to-image and image-to-image pipelines (diffusers).",
                ("diffusers>=0.30",), ("diffusers",)),
        Profile("rag", "Document chat: sentence embeddings, PDF/DOCX readers and the sqlite-vec index.",
                ("sentence-transformers>=3.0", "pypdf>=4.0", "python-docx>=1.1", "sqlite-vec>=0.1.6"),
                ("sentence_transformers", "pypdf", "docx", "sqlite_vec")),
        Profile("server", "OpenAI-compatible local REST server.",
                ("fastapi>=0.110", "uvicorn[standard]>=0.30", "tornado>=6.1"), ("fastapi", "uvicorn")),
        Profile("peft", "LoRA adapters.", ("peft>=0.12",), ("peft",)),
        Profile("mlx", "Apple Silicon 4-bit inference and quantization (MLX).",
                ("mlx>=0.20", "mlx-lm>=0.20"), ("mlx", "mlx_lm"), requires="apple_silicon"),
        Profile("cuda-quant", "CUDA-only quantization backends: bitsandbytes, GPTQModel, AutoAWQ.",
                ("bitsandbytes>=0.43", "gptqmodel>=1.0", "autoawq>=0.2"), ("bitsandbytes", "gptqmodel", "awq"),
                requires="cuda_platform"),
        Profile("gguf", "GGUF models through llama.cpp (llama-cpp-python, gguf).",
                ("llama-cpp-python>=0.3", "gguf>=0.10"), ("llama_cpp", "gguf")),
    )
}


# ---------------------------------------------------------------------------
# names, profiles, paths
# ---------------------------------------------------------------------------


def validate_name(name: str) -> str:
    """Return ``name`` if it is a safe folder name (``^[A-Za-z0-9_.-]{1,64}$``, not only dots)."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or set(name) == {"."}:
        raise EnvironmentManagerError(
            f"Invalid environment name {name!r}: use 1-64 letters, digits, '.', '_' or '-'.")
    return name


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _profile_supported(profile: Profile) -> str | None:
    """Return why ``profile`` cannot be installed on this machine, or None."""
    if profile.requires == "apple_silicon" and not is_apple_silicon():
        return f"The '{profile.name}' profile needs an Apple Silicon Mac."
    if profile.requires == "cuda_platform" and sys.platform == "darwin":
        return f"The '{profile.name}' profile needs an NVIDIA CUDA GPU (Linux or Windows); macOS has no CUDA."
    return None


def resolve_profiles(names: Iterable[str] | None) -> list[str]:
    """Validate profile names, always include ``base`` and return them in canonical order."""
    wanted = {"base"}
    for raw in names or ():
        for name in str(raw).split(","):
            name = name.strip()
            if not name:
                continue
            if name not in PROFILES:
                raise EnvironmentManagerError(
                    f"Unknown profile '{name}'. Choose from: {', '.join(PROFILES)}.")
            wanted.add(name)
    for name in wanted:
        reason = _profile_supported(PROFILES[name])
        if reason:
            raise EnvironmentManagerError(reason)
    return [name for name in PROFILES if name in wanted]


def requirements_for(profiles: Iterable[str]) -> list[str]:
    reqs: list[str] = []
    for name in resolve_profiles(profiles):
        for req in PROFILES[name].requirements:
            if req not in reqs:
                reqs.append(req)
    return reqs


def modules_for(profiles: Iterable[str]) -> list[str]:
    mods: list[str] = []
    for name in profiles:
        profile = PROFILES.get(name)
        for mod in profile.modules if profile else ():
            if mod not in mods:
                mods.append(mod)
    return mods


def env_dir(name: str) -> Path:
    return paths.environments_dir() / validate_name(name)


def python_path(env_root: Path) -> Path:
    """Interpreter inside a venv folder (``bin/python`` or ``Scripts/python.exe``)."""
    if os.name == "nt":
        return env_root / "Scripts" / "python.exe"
    return env_root / "bin" / "python"


def python_for(name: str) -> Path:
    """Interpreter of environment ``name``; raises if the environment does not exist."""
    root = env_dir(name)
    python = python_path(root)
    if not root.is_dir():
        raise EnvironmentManagerError(f"Environment '{name}' does not exist.")
    if not python.exists():
        raise EnvironmentManagerError(f"Environment '{name}' has no Python interpreter at {python}.")
    return python


def default_base_python() -> str:
    """Interpreter used for ``-m venv`` when none is given."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    for candidate in ("python3.13", "python3.12", "python3.11", "python3"):
        found = shutil.which(candidate)
        if found:
            return found
    raise EnvironmentManagerError("No Python 3.11+ interpreter found on PATH; pass --python.")


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def read_manifest(env_root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((env_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(env_root: Path, manifest: dict[str, Any]) -> None:
    target = env_root / MANIFEST_NAME
    tmp = target.with_name(f".{MANIFEST_NAME}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def _describe(env_root: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    data = manifest or {}
    return {
        "name": env_root.name,
        "path": str(env_root),
        "python": str(python_path(env_root)),
        "profiles": list(data.get("profiles") or []),
        "created_at": data.get("created_at"),
        "verified": bool(data.get("verified", False)),
        "torch_version": data.get("torch_version"),
        "device": data.get("device"),
        "status": "ready" if manifest else "incomplete",
    }


def list_environments() -> list[dict[str, Any]]:
    """Every environment folder, including incomplete ones (``status == "incomplete"``)."""
    root = paths.environments_dir()
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or not _NAME_RE.fullmatch(child.name):
            continue
        out.append(_describe(child, read_manifest(child)))
    return out


def remove_environment(name: str) -> bool:
    """Delete environment ``name``; False if it does not exist."""
    root = paths.environments_dir()
    target = env_dir(name)
    if not target.exists() and not target.is_symlink():
        return False
    if target.is_symlink() or target.resolve().parent != root.resolve():
        raise EnvironmentManagerError(f"Refusing to remove {target}: it is not inside {root}.")
    if not ((target / MANIFEST_NAME).is_file() or (target / "pyvenv.cfg").is_file()):
        raise EnvironmentManagerError(f"Refusing to remove {target}: it is not a Noelle environment.")
    shutil.rmtree(target)
    return True


# ---------------------------------------------------------------------------
# subprocesses
# ---------------------------------------------------------------------------


def _terminate(proc: subprocess.Popen[str], grace: float = 5.0) -> None:
    """Terminate ``proc`` and its process group (pip spawns build helpers)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
        proc.wait(timeout=5)


def run_streaming(cmd: list[str], *, on_line: OnLine | None = None, stop_event: threading.Event | None = None,
                  env: dict[str, str] | None = None, cwd: str | os.PathLike[str] | None = None,
                  tail_lines: int = 40) -> tuple[int, list[str]]:
    """Run ``cmd`` headless and stream its merged stdout/stderr lines to ``on_line``.

    ``\\r``-terminated progress-bar updates count as lines. When ``stop_event`` is
    set the process group is terminated and :class:`Cancelled` is raised; any
    other exception (including KeyboardInterrupt) also terminates the process.
    Returns ``(exit_code, last_lines)``.
    """
    extra: dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {}
    proc = subprocess.Popen([str(part) for part in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=env, cwd=cwd, **extra)
    tail: deque[str] = deque(maxlen=tail_lines)
    finished = threading.Event()
    stopped = threading.Event()

    def watch() -> None:
        assert stop_event is not None
        while not finished.is_set():
            if stop_event.wait(0.1):
                stopped.set()
                _terminate(proc)
                return

    watcher = threading.Thread(target=watch, name="noelle-proc-watch", daemon=True) if stop_event else None
    if watcher:
        watcher.start()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            tail.append(line)
            if on_line is not None:
                on_line(line)
        code = proc.wait()
    finally:
        finished.set()
        if proc.poll() is None:
            _terminate(proc)
        if proc.stdout is not None:
            proc.stdout.close()
        if watcher:
            watcher.join(timeout=10)
    if stopped.is_set():
        raise Cancelled("Stopped by the user.")
    return code, list(tail)


def pip_install_command(python: str | os.PathLike[str], requirements: Iterable[str]) -> list[str]:
    """Headless ``pip install`` invocation for the environment's interpreter."""
    return [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
            "--progress-bar", "off", *requirements]


def venv_command(python: str | os.PathLike[str], target: Path, *, with_pip: bool = True) -> list[str]:
    cmd = [str(python), "-m", "venv", str(target)]
    if not with_pip:
        cmd.append("--without-pip")
    return cmd


def _pip_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONUNBUFFERED="1")
    return env


def create_venv(target: Path, *, python: str | None = None, with_pip: bool = True, on_line: OnLine | None = None,
                stop_event: threading.Event | None = None) -> Path:
    """``python -m venv target``; returns the new interpreter path."""
    base = python or default_base_python()
    code, tail = run_streaming(venv_command(base, target, with_pip=with_pip), on_line=on_line,
                               stop_event=stop_event, env=_pip_env())
    if code != 0:
        raise EnvironmentManagerError(f"python -m venv failed (exit code {code}): {_last(tail)}")
    interpreter = python_path(target)
    if not interpreter.exists():
        raise EnvironmentManagerError(f"python -m venv did not create {interpreter}.")
    return interpreter


def _last(lines: list[str]) -> str:
    return next((line.strip() for line in reversed(lines) if line.strip()), "no output")


# ---------------------------------------------------------------------------
# create / verify
# ---------------------------------------------------------------------------


def create_environment(name: str, profiles: Iterable[str] | None, *, python: str | None = None,
                       progress: OnLine | None = None, stop_event: threading.Event | None = None) -> dict[str, Any]:
    """Create a venv named ``name`` and pip-install the requested profiles into it.

    A failed or cancelled creation removes the half-built folder.
    """
    name = validate_name(name)
    chosen = resolve_profiles(profiles)
    requirements = requirements_for(chosen)
    root = paths.environments_dir()
    target = root / name
    if target.exists() or target.is_symlink():
        raise EnvironmentManagerError(f"Environment '{name}' already exists at {target}.")
    base = python or default_base_python()
    if os.sep not in base and not Path(base).exists():
        base = shutil.which(base) or base
    if not Path(base).exists():
        raise EnvironmentManagerError(f"Python interpreter not found: {base}")
    say = progress or (lambda _line: None)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        say(f"Creating virtual environment '{name}' with {base}")
        interpreter = create_venv(target, python=base, on_line=say, stop_event=stop_event)
        say(f"Installing profiles {', '.join(chosen)}: {' '.join(requirements)}")
        code, tail = run_streaming(pip_install_command(interpreter, requirements), on_line=say,
                                   stop_event=stop_event, env=_pip_env())
        if code != 0:
            raise EnvironmentManagerError(f"pip install failed (exit code {code}): {_last(tail)}")
        manifest = {
            "name": name,
            "path": str(target),
            "python": str(interpreter),
            "base_python": base,
            "profiles": chosen,
            "requirements": requirements,
            "created_at": started,
            "install_seconds": round(time.time() - started, 1),
            "verified": False,
            "verified_at": None,
            "torch_version": None,
            "device": None,
            "noelle_version": __version__,
            "platform": f"{sys.platform}-{platform.machine()}",
        }
        write_manifest(target, manifest)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    say(f"Environment '{name}' is ready at {target}")
    return _describe(target, manifest)


_VERIFY_SCRIPT = r"""
import importlib.util, json, sys
report = {"python_version": sys.version.split()[0], "executable": sys.executable, "errors": []}
modules = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
report["missing_modules"] = [m for m in modules if importlib.util.find_spec(m) is None]
try:
    import torch
    report["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    x = torch.ones((64, 64), device=device)
    value = float((x @ x).sum().item())
    report["device"] = device
    report["tensor_ok"] = abs(value - 64.0 ** 3) < 1e-3
except Exception as exc:
    report["tensor_ok"] = False
    report["errors"].append("torch: %s: %s" % (type(exc).__name__, exc))
try:
    import noelle
    import noelle.runtime.protocol
    report["noelle_version"] = getattr(noelle, "__version__", None)
    report["noelle_ok"] = True
except Exception as exc:
    report["noelle_ok"] = False
    report["errors"].append("noelle: %s: %s" % (type(exc).__name__, exc))
print("NOELLE_VERIFY " + json.dumps(report), flush=True)
"""


def verify_environment(name: str, *, timeout: float = 600.0) -> dict[str, Any]:
    """Run the environment's interpreter: import torch, allocate a tensor, import noelle.

    Updates the manifest (``verified``, ``torch_version``, ``device``) and returns
    the report: ``{name, verified, python_version, torch_version, device,
    tensor_ok, noelle_ok, missing_modules, errors, verified_at}``.
    """
    from noelle.runtime.client import build_worker_env

    interpreter = python_for(name)
    root = env_dir(name)
    manifest = read_manifest(root)
    modules = modules_for((manifest or {}).get("profiles") or ["base"])
    env = build_worker_env(None, offline=True)
    try:
        proc = subprocess.run([str(interpreter), "-c", _VERIFY_SCRIPT, json.dumps(modules)], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
                              stdin=subprocess.DEVNULL)
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, code = "", f"verification timed out after {timeout:.0f}s", -1
    except OSError as exc:
        stdout, stderr, code = "", f"could not run {interpreter}: {exc}", -1
    report: dict[str, Any] = {}
    for line in stdout.splitlines():
        if line.startswith(VERIFY_MARKER + " "):
            try:
                report = json.loads(line[len(VERIFY_MARKER) + 1:])
            except ValueError:
                report = {}
    errors = list(report.get("errors") or [])
    if not report:
        errors.append(f"verification script failed (exit code {code}): {_last(stderr.splitlines())}")
    missing = list(report.get("missing_modules") or [])
    if missing:
        errors.append(f"missing modules: {', '.join(missing)}")
    verified = bool(report.get("tensor_ok")) and bool(report.get("noelle_ok")) and not missing
    result = {
        "name": name,
        "verified": verified,
        "python_version": report.get("python_version"),
        "torch_version": report.get("torch_version"),
        "device": report.get("device"),
        "tensor_ok": bool(report.get("tensor_ok")),
        "noelle_ok": bool(report.get("noelle_ok")),
        "missing_modules": missing,
        "errors": errors,
        "verified_at": time.time(),
    }
    if manifest is not None:
        manifest.update(verified=verified, verified_at=result["verified_at"], torch_version=result["torch_version"],
                        device=result["device"], python_version=result["python_version"])
        write_manifest(root, manifest)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ("NAME", "PROFILES", "STATUS", "VERIFIED", "TORCH", "DEVICE", "PATH")
    table = [headers] + [(
        row["name"], ",".join(row["profiles"]) or "-", row["status"], "yes" if row["verified"] else "no",
        row["torch_version"] or "-", row["device"] or "-", row["path"]) for row in rows]
    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]
    return "\n".join("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in table)


def main(args: argparse.Namespace) -> int:
    """Entry point for ``noelle env``; returns a process exit code."""
    command = getattr(args, "env_command", None)
    try:
        if command == "create":
            profiles = getattr(args, "profile", None) or ["base"]
            info = create_environment(args.name, profiles, python=getattr(args, "python", None),
                                      progress=lambda line: print(line, flush=True))
            print(f"Created '{info['name']}' ({', '.join(info['profiles'])}). "
                  f"Check it with: noelle env verify {info['name']}")
            return 0
        if command == "list":
            rows = list_environments()
            if not rows:
                print("No environments yet. Create one with: noelle env create <name> --profile base")
            else:
                print(_format_table(rows))
            return 0
        if command == "remove":
            if remove_environment(args.name):
                print(f"Removed environment '{args.name}'.")
                return 0
            print(f"No environment named '{args.name}'.", file=sys.stderr)
            return 1
        if command == "verify":
            result = verify_environment(args.name)
            state = "OK" if result["verified"] else "FAILED"
            print(f"{args.name}: {state}  python {result['python_version'] or '?'}  "
                  f"torch {result['torch_version'] or '-'}  device {result['device'] or '-'}  "
                  f"noelle import {'ok' if result['noelle_ok'] else 'failed'}")
            for error in result["errors"]:
                print(f"  - {error}")
            return 0 if result["verified"] else 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except (EnvironmentManagerError, Cancelled) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("usage: noelle env {create,list,remove,verify} ...", file=sys.stderr)
    return 2
