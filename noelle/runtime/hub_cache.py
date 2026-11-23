"""Read-only helpers for the Hugging Face cache layout (no network, no hub import).

huggingface_hub >= 2 refuses ``snapshot_download(local_files_only=True)`` when a
snapshot is missing files that exist upstream. Noelle downloads curated subsets
on purpose (it skips ONNX/Flax/duplicate weights), so the worker resolves cached
snapshots directly from ``refs/`` and ``snapshots/`` instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from noelle.paths import hf_hub_cache

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def repo_folder(repo_id: str, cache_dir: str | Path | None = None, repo_type: str = "model") -> Path:
    cache = Path(cache_dir) if cache_dir else hf_hub_cache()
    return cache / f"{repo_type}s--{repo_id.replace('/', '--')}"


def cached_snapshot_path(repo_id: str, revision: str | None = None, cache_dir: str | Path | None = None,
                         repo_type: str = "model") -> Path | None:
    folder = repo_folder(repo_id, cache_dir, repo_type)
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return None
    rev = revision or "main"
    commit: str | None = None
    if _COMMIT_RE.match(rev):
        commit = rev
    else:
        ref = folder / "refs" / rev
        if ref.is_file():
            commit = ref.read_text(encoding="utf-8").strip() or None
    if commit:
        snap = snapshots / commit
        if snap.is_dir() and any(snap.iterdir()):
            return snap
    if revision not in (None, "main"):
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir() and any(p.iterdir())]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_downloaded(repo_id: str, revision: str | None = None, cache_dir: str | Path | None = None) -> bool:
    snap = cached_snapshot_path(repo_id, revision, cache_dir)
    if snap is None:
        return False
    return any((snap / name).exists() for name in ("config.json", "model_index.json", "modules.json",
                                                    "adapter_config.json")) or any(snap.glob("*.gguf"))
