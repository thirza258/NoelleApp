"""Optional-component install (fake archives, no network) and GGUF inference via a fake llama-server."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import threading
from pathlib import Path
from typing import Any

import pytest

from noelle.app.services import components
from noelle.runtime import llama_tools

EXE = llama_tools.EXE


def _tar(entries: dict[str, bytes], *, mode: int = 0o755) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class _Stream:
    def __init__(self, data: bytes, status: int = 200) -> None:
        self.data, self.status_code, self.headers = data, status, {"content-length": str(len(data))}

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def iter_bytes(self) -> Any:
        for start in range(0, len(self.data), 1024):
            yield self.data[start:start + 1024]


class FakeSession:
    def __init__(self, files: dict[str, bytes], releases: list[dict[str, Any]] | None = None) -> None:
        self.files, self.releases = files, releases or []

    def stream(self, method: str, url: str, **_kw: Any) -> _Stream:
        return _Stream(self.files[url]) if url in self.files else _Stream(b"", 404)

    def get(self, url: str, **_kw: Any) -> Any:
        return type("R", (), {"status_code": 200, "json": lambda _self: self.releases})()


def _release(binary_archive: bytes, *, digest: str | None = None) -> components.ComponentRelease:
    return components.ComponentRelease(
        tag="b1", asset_name="llama-b1-bin-test.tar.gz", asset_url="https://x/bin.tar.gz",
        asset_size=len(binary_archive), asset_sha256=digest or hashlib.sha256(binary_archive).hexdigest(),
        source_url="https://x/src.tar.gz")


def _archives() -> tuple[bytes, bytes]:
    binaries = _tar({f"build/bin/llama-quantize{EXE}": b"#!/bin/sh\n", f"build/bin/llama-server{EXE}": b"#!/bin/sh\n",
                     "build/bin/libggml.dylib": b"lib"})
    source = _tar({"llama.cpp-b1/convert_hf_to_gguf.py": b"print('convert')\n",
                   "llama.cpp-b1/gguf-py/gguf/__init__.py": b"", "llama.cpp-b1/LICENSE": b"MIT",
                   "llama.cpp-b1/src/llama.cpp": b"// not needed"}, mode=0o644)
    return binaries, source


def test_platform_suffixes_and_release_pick() -> None:
    assert components.platform_asset_suffix("Darwin", "arm64") == "bin-macos-arm64.tar.gz"
    assert components.platform_asset_suffix("Linux", "x86_64") == "bin-ubuntu-x64.tar.gz"
    assert components.platform_asset_suffix("Windows", "AMD64") == "bin-win-cpu-x64.zip"
    assert components.platform_asset_suffix("Plan9", "mips") is None
    releases = [
        {"tag_name": "b3", "draft": True, "assets": [{"name": "llama-b3-bin-macos-arm64.tar.gz"}]},
        {"tag_name": "b2", "prerelease": True, "assets": [
            {"name": "llama-b2-bin-macos-arm64.tar.gz", "size": 10, "digest": "sha256:abc",
             "browser_download_url": "https://dl/b2.tgz"}]},
    ]
    picked = components.pick_release(releases, "bin-macos-arm64.tar.gz")
    assert (picked.tag, picked.asset_sha256, picked.asset_size) == ("b2", "abc", 10)
    assert picked.source_url.endswith("/refs/tags/b2")
    with pytest.raises(components.ComponentError):
        components.pick_release(releases, "bin-win-cpu-x64.zip")


@pytest.mark.skipif(os.name == "nt", reason="exec-bit checks are POSIX")
def test_install_verify_and_discovery(tmp_path: Path) -> None:
    binaries, source = _archives()
    session = FakeSession({"https://x/bin.tar.gz": binaries, "https://x/src.tar.gz": source})
    events: list[dict[str, Any]] = []
    info = components.install_llama_cpp(_release(binaries), progress=events.append, session=session)
    folder = Path(info["path"])
    assert (folder / "bin" / f"llama-server{EXE}").is_file() and (folder / "bin" / "libggml.dylib").is_file()
    assert os.access(folder / "bin" / f"llama-quantize{EXE}", os.X_OK)
    assert (folder / "convert_hf_to_gguf.py").is_file() and (folder / "gguf-py" / "gguf").is_dir()
    assert not (folder / "src").exists()  # only the converter is taken from the source archive
    assert {"download", "verified", "extract", "done"} <= {e["stage"] for e in events}
    assert components.installed_llama_cpp()["tag"] == "b1"
    assert llama_tools.find_llama_server() == folder / "bin" / f"llama-server{EXE}"
    from noelle.runtime.quantization_engine import find_llama_cpp

    tools = find_llama_cpp()
    assert tools.convert_script == folder / "convert_hf_to_gguf.py" and tools.gguf_package
    assert tools.quantize_binary == folder / "bin" / f"llama-quantize{EXE}"
    assert components.remove_llama_cpp() and components.installed_llama_cpp() is None


def test_checksum_mismatch_installs_nothing(tmp_path: Path) -> None:
    binaries, source = _archives()
    session = FakeSession({"https://x/bin.tar.gz": binaries, "https://x/src.tar.gz": source})
    with pytest.raises(components.ComponentError, match="SHA-256"):
        components.install_llama_cpp(_release(binaries, digest="0" * 64), session=session)
    assert components.installed_llama_cpp() is None
    assert not any(p.name.startswith(".staging") for p in components.install_root().iterdir())


def test_archive_path_traversal_is_refused(tmp_path: Path) -> None:
    evil = _tar({"../../escaped.txt": b"x", f"bin/llama-quantize{EXE}": b"", f"bin/llama-server{EXE}": b""})
    _, source = _archives()
    session = FakeSession({"https://x/bin.tar.gz": evil, "https://x/src.tar.gz": source})
    with pytest.raises(components.ComponentError, match="escapes"):
        components.install_llama_cpp(_release(evil), session=session)
    assert not (components.install_root().parent / "escaped.txt").exists()


def test_cancel_stops_the_download(tmp_path: Path) -> None:
    binaries, source = _archives()
    session = FakeSession({"https://x/bin.tar.gz": binaries, "https://x/src.tar.gz": source})
    stop = threading.Event()
    stop.set()
    with pytest.raises(components.ComponentCancelled):
        components.install_llama_cpp(_release(binaries), session=session, stop_event=stop)
    assert components.installed_llama_cpp() is None


# ------------------------------------------------------------------ GGUF engine
FAKE_SERVER = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
assert args[args.index("--host") + 1] == "127.0.0.1"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        body = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._json(200, {"status": "ok"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._json(200, {"tokens": list(range(len(body["content"].split())))})
        assert body["stream"] is True and body["max_tokens"] >= 1
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for word in ["Hello", " from", " GGUF"]:
            chunk = {"choices": [{"delta": {"content": word}, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n"); self.wfile.flush()
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.mark.skipif(os.name == "nt", reason="the fake llama-server is a POSIX shebang script")
def test_llama_server_engine_streams(tmp_path: Path) -> None:
    from noelle.runtime.generation import LlamaServerEngine
    from noelle.runtime.protocol import GenerationParams

    binary = tmp_path / "llama-server"
    binary.write_text(f"#!{sys.executable}\n{FAKE_SERVER}")
    binary.chmod(0o755)
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    engine = LlamaServerEngine.load(str(gguf), device="mps", binary=str(binary), startup_timeout=20)
    try:
        tokens: list[str] = []
        result = engine.stream_chat([{"role": "user", "content": "hi there"}], GenerationParams(max_new_tokens=8),
                                    stop_event=threading.Event(), on_text=tokens.append)
        assert "".join(tokens) == "Hello from GGUF" == result.text
        assert result.completion_tokens == 3 and result.prompt_tokens == 7 and result.finish_reason == "stop"
        assert engine.count_tokens("one two three") == 3
    finally:
        engine.release()
    assert engine.process is None


def test_gguf_without_tools_reports_the_missing_component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from noelle.runtime.errors import DependencyMissing
    from noelle.runtime.model_manager import ModelManager, ModelSpec

    monkeypatch.setattr(llama_tools.shutil, "which", lambda _name: None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "llama_cpp"
                        else real_find_spec(name, *a))
    spec = ModelSpec(model_id="org/m-GGUF", task="text-generation", gguf_file="m.gguf", local_path=str(tmp_path))
    with pytest.raises(DependencyMissing) as err:
        ModelManager._load_gguf(spec, str(tmp_path), "mps")
    assert err.value.extra == {"component": "llama.cpp"}


def test_import_hf_token_cli(tmp_path: Path, memory_keyring: Any) -> None:
    from noelle.__main__ import import_hf_token
    from noelle.app.services import hf_client

    handoff = tmp_path / "token.txt"
    handoff.write_text("  hf_fromInstaller  \n")
    assert import_hf_token(str(handoff)) == 0
    assert hf_client.get_token() == "hf_fromInstaller"
    assert not handoff.exists()  # the hand-over file is always deleted
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert import_hf_token(str(empty)) == 0 and not empty.exists()
    assert import_hf_token(str(tmp_path / "missing.txt")) == 1
