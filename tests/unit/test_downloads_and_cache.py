from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from noelle.runtime import downloader
from noelle.runtime.downloader import RepoFile, default_patterns, git_blob_sha1, select_files, verify_file
from noelle.runtime.hub_cache import cached_snapshot_path, is_downloaded


def _files(*names: str) -> list[RepoFile]:
    return [RepoFile(n, 10, None, None) for n in names]


def test_default_patterns_prefer_safetensors() -> None:
    chosen = {f.name for f in select_files(_files("config.json", "model.safetensors", "pytorch_model.bin",
                                                  "flax_model.msgpack", "onnx/model.onnx", "runs/x.tfevents.1",
                                                  "tokenizer.json"), None, None)}
    assert chosen == {"config.json", "model.safetensors", "tokenizer.json"}


def test_default_patterns_skip_fp16_variant_duplicates() -> None:
    chosen = {f.name for f in select_files(_files("unet/diffusion_pytorch_model.safetensors",
                                                  "unet/diffusion_pytorch_model.fp16.safetensors",
                                                  "model_index.json"), None, None)}
    assert chosen == {"unet/diffusion_pytorch_model.safetensors", "model_index.json"}


def test_default_patterns_pick_one_gguf() -> None:
    names = ["model-Q2_K.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf", "README.md"]
    _, ignore = default_patterns(names)
    chosen = {f.name for f in select_files(_files(*names), None, None)}
    assert chosen == {"model-Q4_K_M.gguf", "README.md"}
    assert "model-Q8_0.gguf" in ignore


def test_explicit_patterns_win() -> None:
    chosen = {f.name for f in select_files(_files("a.bin", "b.safetensors"), ["*.bin"], None)}
    assert chosen == {"a.bin"}


def test_verify_file_sha256_and_git_blob(tmp_path: Path) -> None:
    payload = b"hello noelle\n"
    path = tmp_path / "weights.safetensors"
    path.write_bytes(payload)
    assert verify_file(path, RepoFile(path.name, len(payload), hashlib.sha256(payload).hexdigest(), None)) is True
    assert verify_file(path, RepoFile(path.name, len(payload), "0" * 64, None)) is False
    blob = hashlib.sha1(b"blob %d\0" % len(payload) + payload).hexdigest()
    assert git_blob_sha1(path) == blob
    assert verify_file(path, RepoFile(path.name, len(payload), None, blob)) is True
    assert verify_file(path, RepoFile(path.name, len(payload), None, None)) is None


def test_clean_stale_incomplete(tmp_path: Path) -> None:
    blobs = tmp_path / "models--org--m" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.1234abcd.incomplete").write_bytes(b"x")
    (blobs / "abc").write_bytes(b"y")
    assert downloader.clean_stale_incomplete(tmp_path / "models--org--m") == 1
    assert sorted(p.name for p in blobs.iterdir()) == ["abc"]


def _fake_snapshot(cache: Path, repo: str, commit: str, files: dict[str, bytes], ref: str | None = "main") -> Path:
    folder = cache / f"models--{repo.replace('/', '--')}"
    snap = folder / "snapshots" / commit
    snap.mkdir(parents=True)
    for name, data in files.items():
        (snap / name).write_bytes(data)
    if ref:
        (folder / "refs").mkdir(exist_ok=True)
        (folder / "refs" / ref).write_text(commit)
    return snap


def test_cached_snapshot_resolution(tmp_path: Path) -> None:
    commit = "a" * 40
    snap = _fake_snapshot(tmp_path, "org/model", commit, {"config.json": b"{}"})
    assert cached_snapshot_path("org/model", cache_dir=tmp_path) == snap
    assert cached_snapshot_path("org/model", commit, cache_dir=tmp_path) == snap
    assert cached_snapshot_path("org/missing", cache_dir=tmp_path) is None
    assert is_downloaded("org/model", cache_dir=tmp_path)
    # a README-only snapshot is not a downloaded model
    _fake_snapshot(tmp_path, "org/readme-only", "b" * 40, {"README.md": b"# hi"})
    assert cached_snapshot_path("org/readme-only", cache_dir=tmp_path) is not None
    assert not is_downloaded("org/readme-only", cache_dir=tmp_path)


def test_cached_snapshot_falls_back_to_newest_without_refs(tmp_path: Path) -> None:
    old = _fake_snapshot(tmp_path, "org/x", "c" * 40, {"config.json": b"{}"}, ref=None)
    os.utime(old, (time.time() - 100, time.time() - 100))
    new = _fake_snapshot(tmp_path, "org/y", "d" * 40, {"config.json": b"{}"}, ref=None)
    assert cached_snapshot_path("org/x", cache_dir=tmp_path) == old
    assert cached_snapshot_path("org/y", cache_dir=tmp_path) == new


def test_download_file_retries_transient_errors(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    events: list[dict] = []
    monkeypatch.setattr(downloader, "emit", events.append)
    monkeypatch.setattr(downloader.time, "sleep", lambda _s: None)
    target = tmp_path / "model.safetensors"
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("network unreachable")
        target.write_bytes(b"weights")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", flaky)
    entry = RepoFile("model.safetensors", 7, hashlib.sha256(b"weights").hexdigest(), None)
    path = downloader.download_file({"repo_id": "org/m"}, entry, retries=5, verify=True, totals={})
    assert path == target and calls["n"] == 3
    kinds = [e["type"] for e in events]
    assert kinds.count("retry") == 2 and kinds[-2:] == ["verify", "file_done"]


def test_download_file_redownloads_on_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    monkeypatch.setattr(downloader, "emit", lambda _e: None)
    monkeypatch.setattr(downloader.time, "sleep", lambda _s: None)
    target = tmp_path / "model.safetensors"
    payloads = iter([b"corrupt", b"weights"])

    def fetch(**kwargs):
        target.write_bytes(next(payloads))
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    entry = RepoFile("model.safetensors", 7, hashlib.sha256(b"weights").hexdigest(), None)
    downloader.download_file({"repo_id": "org/m"}, entry, retries=2, verify=True, totals={})
    assert target.read_bytes() == b"weights"


def test_download_file_does_not_retry_missing_repo(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub
    import pytest

    monkeypatch.setattr(downloader, "emit", lambda _e: None)
    calls = {"n": 0}

    def missing(**kwargs):
        calls["n"] += 1
        raise FileNotFoundError("gone")  # permanent errors are not retried

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(FileNotFoundError):
        downloader.download_file({"repo_id": "org/m"}, RepoFile("x", 1, None, None), retries=5, verify=False, totals={})
    assert calls["n"] == 1
