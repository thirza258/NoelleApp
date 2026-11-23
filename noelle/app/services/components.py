"""Optional components that Noelle downloads only after the user approves.

Currently one component: **llama.cpp tools** (``llama-quantize``, ``llama-server`` and the
``convert_hf_to_gguf.py`` converter with its ``gguf-py`` package). They power GGUF
quantization and GGUF inference. The prebuilt binaries for this platform and the converter
come from the same official release on github.com/ggml-org/llama.cpp (MIT licence).

Flow: :func:`resolve_llama_cpp` (metadata only) -> the UI shows version, size, source and
destination and asks for permission -> :func:`install_llama_cpp` downloads, verifies the
SHA-256 published by GitHub, extracts with path-traversal checks and points the
``quantization.llama_cpp_dir`` setting at the result.

Qt-free so it can be unit-tested and reused by the setup assistant.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tarfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = ["LLAMA_CPP", "ComponentCancelled", "ComponentError", "ComponentRelease", "describe", "install_llama_cpp",
           "install_root", "installed_llama_cpp", "llama_server_binary", "pick_release", "platform_asset_suffix",
           "remove_llama_cpp", "resolve_llama_cpp"]

LLAMA_CPP = "llama.cpp"
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
SOURCE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{tag}"
TREE_URL = "https://api.github.com/repos/ggml-org/llama.cpp/git/trees/{tag}?recursive=1"
MANIFEST = "noelle-component.json"
from noelle.runtime.llama_tools import TOOL_NAMES  # noqa: E402
MAX_ARCHIVE_BYTES = 1_500_000_000

Progress = Callable[[dict[str, Any]], None]


class ComponentError(RuntimeError):
    pass


class ComponentCancelled(ComponentError):
    pass


@dataclass
class ComponentRelease:
    tag: str
    asset_name: str
    asset_url: str
    asset_size: int
    asset_sha256: str | None
    source_url: str
    tree_url: str = ""
    homepage: str = "https://github.com/ggml-org/llama.cpp"
    license: str = "MIT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def platform_asset_suffix(system: str | None = None, machine: str | None = None) -> str | None:
    """Name ending of the prebuilt release archive for this OS/CPU (CPU or Metal builds)."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arm = machine in ("arm64", "aarch64")
    if system == "darwin":
        return "bin-macos-arm64.tar.gz" if arm else "bin-macos-x64.tar.gz"
    if system == "linux":
        return "bin-ubuntu-arm64.tar.gz" if arm else "bin-ubuntu-x64.tar.gz"
    if system == "windows":
        return "bin-win-cpu-arm64.zip" if arm else "bin-win-cpu-x64.zip"
    return None


def pick_release(releases: list[dict[str, Any]], suffix: str) -> ComponentRelease:
    """Newest release that ships a prebuilt archive for this platform."""
    for release in releases:
        if release.get("draft"):  # llama.cpp publishes every build as a "prerelease", so those are accepted
            continue
        for asset in release.get("assets") or []:
            name = str(asset.get("name", ""))
            if name.startswith("llama-") and name.endswith(suffix):
                digest = str(asset.get("digest") or "")
                tag = str(release["tag_name"])
                return ComponentRelease(
                    tag=tag, asset_name=name, asset_url=str(asset["browser_download_url"]),
                    asset_size=int(asset.get("size") or 0),
                    asset_sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else None,
                    source_url=SOURCE_URL.format(tag=tag), tree_url=TREE_URL.format(tag=tag))
    raise ComponentError(f"No recent llama.cpp release ships a '{suffix}' build.")


def _session() -> Any:
    from huggingface_hub.utils import get_session  # an httpx client the app already depends on

    return get_session()


def resolve_llama_cpp(session: Any = None) -> ComponentRelease:
    suffix = platform_asset_suffix()
    if suffix is None:
        raise ComponentError(f"No prebuilt llama.cpp is published for {platform.system()} {platform.machine()}.")
    session = session or _session()
    response = session.get(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "Noelle/1.0"}, timeout=20)
    if response.status_code != 200:
        raise ComponentError(f"GitHub answered HTTP {response.status_code} while listing llama.cpp releases.")
    return pick_release(response.json(), suffix)


# ------------------------------------------------------------------ state
from noelle.runtime.llama_tools import find_llama_server as llama_server_binary  # noqa: E402
from noelle.runtime.llama_tools import install_root  # noqa: E402
from noelle.runtime.llama_tools import installed_info as installed_llama_cpp  # noqa: E402


# --------------------------------------------------------------- download
def _download(session: Any, url: str, target: Path, *, expected_size: int | None, sha256: str | None,
              label: str, progress: Progress | None, stop_event: threading.Event | None) -> None:
    digest = hashlib.sha256()
    done = 0
    last = 0.0
    with session.stream("GET", url, headers={"User-Agent": "Noelle/1.0"}, timeout=60,
                        follow_redirects=True) as response:
        if response.status_code != 200:
            raise ComponentError(f"Download of {label} failed with HTTP {response.status_code}.")
        total = expected_size or int(response.headers.get("content-length") or 0) or None
        with target.open("wb") as fh:
            for chunk in response.iter_bytes():
                if stop_event is not None and stop_event.is_set():
                    raise ComponentCancelled("Download cancelled.")
                done += len(chunk)
                if done > MAX_ARCHIVE_BYTES:
                    raise ComponentError(f"{label} is larger than expected; download stopped.")
                fh.write(chunk)
                digest.update(chunk)
                now = time.monotonic()
                if progress and (now - last > 0.2):
                    last = now
                    progress({"stage": "download", "file": label, "current": done, "total": total})
    if expected_size and done != expected_size:
        raise ComponentError(f"{label} is {done} bytes, expected {expected_size}.")
    if sha256 and digest.hexdigest() != sha256.lower():
        raise ComponentError(f"{label} failed its SHA-256 check; nothing was installed.")
    if progress:
        progress({"stage": "verified" if sha256 else "downloaded", "file": label, "current": done, "total": done})


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if not str(target).startswith(str(root.resolve()) + os.sep) and target != root.resolve():
        raise ComponentError(f"Archive entry escapes the install folder: {member}")
    return target


def _extract(archive: Path, destination: Path, *, keep: Callable[[str], bool] | None = None) -> None:
    """Extract tar.gz/zip, refusing absolute paths, '..' escapes and links that point outside."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if keep and not keep(info.filename):
                    continue
                target = _safe_target(destination, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return
    with tarfile.open(archive, "r:gz") as tf:
        members = []
        for member in tf.getmembers():
            if keep and not keep(member.name):
                continue
            _safe_target(destination, member.name)
            if member.islnk() or member.issym():
                _safe_target(destination, os.path.join(os.path.dirname(member.name), member.linkname))
            if member.isdev() or member.isfifo():
                continue
            members.append(member)
        tf.extractall(destination, members=members, filter="data")


def _is_converter_path(path: str) -> bool:
    return path == "convert_hf_to_gguf.py" or path == "LICENSE" or path.startswith("gguf-py/")


def _converter_hashes(session: Any, release: ComponentRelease) -> dict[str, str]:
    """git blob SHA-1 of every converter file at the release tag, from GitHub's tree API.

    The source archive itself has no published checksum, so each extracted file is checked
    against the repository's own object hashes for that tag instead."""
    url = release.tree_url or TREE_URL.format(tag=release.tag)
    response = session.get(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "Noelle/1.0"},
                           timeout=30)
    if response.status_code != 200:
        raise ComponentError(f"Could not fetch the file hashes for llama.cpp {release.tag} "
                             f"(HTTP {response.status_code}); nothing was installed.")
    tree = response.json()
    if tree.get("truncated"):
        raise ComponentError("GitHub returned an incomplete file list, so the converter cannot be verified.")
    hashes = {entry["path"]: entry["sha"] for entry in tree.get("tree", [])
              if entry.get("type") == "blob" and _is_converter_path(str(entry.get("path", "")))}
    if "convert_hf_to_gguf.py" not in hashes or not any(p.startswith("gguf-py/") for p in hashes):
        raise ComponentError(f"llama.cpp {release.tag} has no GGUF converter.")
    return hashes


def _verify_converter(source_root: Path, expected: dict[str, str]) -> None:
    from noelle.runtime.downloader import git_blob_sha1

    roots = [p for p in source_root.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ComponentError("Unexpected layout in the llama.cpp source archive.")
    extracted = {p.relative_to(roots[0]).as_posix(): p for p in roots[0].rglob("*") if p.is_file() or p.is_symlink()}
    if set(extracted) != set(expected):
        missing = sorted(set(expected) - set(extracted))[:3]
        extra = sorted(set(extracted) - set(expected))[:3]
        raise ComponentError(f"The converter files do not match llama.cpp's file list (missing {missing}, "
                             f"unexpected {extra}); nothing was installed.")
    for relative, path in extracted.items():
        if path.is_symlink() or git_blob_sha1(path) != expected[relative]:
            raise ComponentError(f"{relative} does not match the llama.cpp release; nothing was installed.")


def _find_bin_dir(root: Path) -> Path:
    for candidate in sorted(root.rglob(TOOL_NAMES[0])):
        if all((candidate.parent / name).is_file() for name in TOOL_NAMES):
            return candidate.parent
    raise ComponentError("The llama.cpp archive does not contain llama-quantize and llama-server.")


def _want_source(name: str) -> bool:
    parts = name.split("/", 1)
    inner = parts[1] if len(parts) == 2 else ""
    return inner == "convert_hf_to_gguf.py" or inner.startswith("gguf-py/") or inner == "LICENSE"


def install_llama_cpp(release: ComponentRelease, *, progress: Progress | None = None,
                      stop_event: threading.Event | None = None, session: Any = None) -> dict[str, Any]:
    """Download, verify and install. Only call this after the user agreed to the download."""
    if not release.asset_sha256:
        raise ComponentError("GitHub did not publish a SHA-256 for this llama.cpp build, so Noelle will not "
                             "install it unverified.")
    session = session or _session()
    expected_sources = _converter_hashes(session, release)  # before any download: refuse early if unverifiable
    root = install_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{release.tag}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        binaries = staging / release.asset_name
        _download(session, release.asset_url, binaries, expected_size=release.asset_size or None,
                  sha256=release.asset_sha256, label=release.asset_name, progress=progress, stop_event=stop_event)
        source = staging / f"llama.cpp-{release.tag}-source.tar.gz"
        _download(session, release.source_url, source, expected_size=None, sha256=None,
                  label="converter (source archive)", progress=progress, stop_event=stop_event)
        if progress:
            progress({"stage": "extract", "file": release.asset_name})
        unpacked = staging / "unpacked"
        _extract(binaries, unpacked)
        _extract(source, staging / "source", keep=_want_source)
        _verify_converter(staging / "source", expected_sources)
        if progress:
            progress({"stage": "verified", "file": "converter (git blob hashes of the release tag)"})
        final = staging / "final"
        final.mkdir()
        shutil.move(str(_find_bin_dir(unpacked)), str(final / "bin"))
        src_roots = [p for p in (staging / "source").iterdir() if p.is_dir()]
        if not src_roots or not (src_roots[0] / "convert_hf_to_gguf.py").is_file():
            raise ComponentError("The llama.cpp source archive has no convert_hf_to_gguf.py.")
        for name in ("convert_hf_to_gguf.py", "gguf-py", "LICENSE"):
            if (src_roots[0] / name).exists():
                shutil.move(str(src_roots[0] / name), str(final / name))
        for tool in (final / "bin").iterdir():
            if tool.is_file() and os.name != "nt":
                tool.chmod(tool.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        target = root / release.tag
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(final), str(target))
        info = {"component": LLAMA_CPP, "tag": release.tag, "path": str(target), "asset": release.asset_name,
                "sha256": release.asset_sha256, "installed_at": time.time()}
        (target / MANIFEST).write_text(json.dumps(info, indent=2), encoding="utf-8")
        (root / "current.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    try:
        from noelle.config import get_settings

        get_settings().set("quantization.llama_cpp_dir", info["path"])
    except Exception:
        pass
    if progress:
        progress({"stage": "done", "file": release.tag})
    return info


def remove_llama_cpp() -> bool:
    root = install_root()
    if not root.exists():
        return False
    shutil.rmtree(root)
    try:
        from noelle.config import get_settings

        settings = get_settings()
        if str(settings.get("quantization.llama_cpp_dir") or "").startswith(str(root)):
            settings.reset("quantization.llama_cpp_dir")
    except Exception:
        pass
    return True


def describe(release: ComponentRelease) -> str:
    """Plain-language summary for the permission prompt."""
    size = f"{release.asset_size / 1e6:.1f} MB" if release.asset_size else "size unknown"
    return (f"llama.cpp {release.tag} for this computer ({release.asset_name}, {size}) and the GGUF converter "
            f"from the same release, downloaded from {release.homepage} ({release.license} licence). "
            f"The binaries are checked against GitHub's SHA-256 and every converter file against the release "
            f"tag's git hashes; they are stored in {install_root()}.")
