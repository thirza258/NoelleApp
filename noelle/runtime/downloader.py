"""Download child process: ``noelle download <repo_id>``.

Runs in its own process so the desktop shell can pause/cancel by terminating
it. Progress is printed as JSON lines on stdout::

    {"type": "plan", "files": [...], "total_bytes": ..., "download_bytes": ...}
    {"type": "progress", "file": "model.safetensors", "downloaded": 123, "total": 456}
    {"type": "file_done", "file": "...", "size": ...}
    {"type": "retry", "file": "...", "attempt": 2, "wait_s": 4.0, "error": "..."}
    {"type": "verify", "file": "...", "ok": true}
    {"type": "done", "path": "...", "commit": "...", "total_bytes": ...}
    {"type": "error", "code": "...", "message": "..."}

Resilience: huggingface_hub retries transient network errors inside a file
download with byte-range resume; on top of that each file is retried with
exponential backoff, and finished files are never fetched again. Files are
written to a temporary name and atomically renamed, so an interrupted
download never leaves a truncated shard in place. After download, LFS blobs are
checked against the commit's SHA-256 and small files against their git blob id.

The Hugging Face token is read from the ``HF_TOKEN`` environment variable, never
from the command line.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm as _vanilla_tqdm

_print_lock = threading.Lock()
log = logging.getLogger("noelle.download-child")

# Weight formats that transformers/diffusers would not load when safetensors exist.
_REDUNDANT_WEIGHTS = ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack", "*.ot", "*.onnx", "*.onnx_data",
                      "*.tflite", "*.mlmodel", "flax_model*", "tf_model*", "rust_model*")
_ALWAYS_SKIP = ("onnx/*", "openvino/*", "coreml/*", "*.mlpackage/*", ".gitattributes", "*.md5", "runs/*",
                "*.tfevents.*", "optimizer.pt", "scheduler.pt", "rng_state*.pth", "training_args.bin")
_PREFERRED_GGUF = ("Q4_K_M", "q4_k_m", "Q4_K_S", "Q5_K_M", "Q4_0")


def emit(event: dict[str, Any]) -> None:
    from noelle.runtime.errors import redact_secrets

    clean = {key: redact_secrets(value) if isinstance(value, str) else value for key, value in event.items()}
    with _print_lock:
        sys.stdout.write(json.dumps(clean, separators=(",", ":")) + "\n")
        sys.stdout.flush()


@dataclass
class RepoFile:
    name: str
    size: int | None
    sha256: str | None  # LFS files
    blob_id: str | None  # git blob sha1 (non-LFS)


def default_patterns(names: list[str]) -> tuple[list[str] | None, list[str]]:
    """Pick sensible allow/ignore patterns for a repository file listing.

    * safetensors present -> skip duplicate legacy weight formats
    * diffusers fp16 variants -> keep the default (non-variant) weights only
    * GGUF-only repos -> pick a single Q4_K_M-style file instead of every quant
    """
    ignore = list(_ALWAYS_SKIP)
    has_safetensors = any(n.endswith(".safetensors") for n in names)
    if has_safetensors:
        ignore += list(_REDUNDANT_WEIGHTS)
        if any(".fp16." in n for n in names) and any(n.endswith(".safetensors") and ".fp16." not in n for n in names):
            ignore.append("*.fp16.*")
    ggufs = [n for n in names if n.endswith(".gguf")]
    if ggufs and not has_safetensors:
        choice = next((g for tag in _PREFERRED_GGUF for g in ggufs if tag in g), None) or min(ggufs)
        others = [g for g in ggufs if g != choice]
        return None, ignore + others
    return None, ignore


def select_files(files: list[RepoFile], include: list[str] | None, exclude: list[str] | None) -> list[RepoFile]:
    names = [f.name for f in files]
    if include is None and exclude is None:
        include, exclude = default_patterns(names)
    selected = []
    for f in files:
        if include and not any(fnmatch.fnmatch(f.name, pat) for pat in include):
            continue
        if exclude and any(fnmatch.fnmatch(f.name, pat) for pat in exclude):
            continue
        selected.append(f)
    return selected


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, on_progress: Any = None) -> str:
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            done += len(chunk)
            if on_progress:
                on_progress(done)
    return digest.hexdigest()


def verify_file(path: Path, entry: RepoFile, on_progress: Any = None) -> bool | None:
    """True/False when a checksum is known, None when nothing to compare against."""
    real = path.resolve()
    if entry.sha256:
        return sha256_file(real, on_progress) == entry.sha256
    if entry.blob_id:
        return git_blob_sha1(real) == entry.blob_id
    return None


def _reporter(file_name: str, totals: dict[str, int]) -> type:
    """tqdm subclass that turns byte updates into throttled JSON progress lines."""

    class Reporter(_vanilla_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("name", None)
            kwargs["file"] = open(os.devnull, "w")  # noqa: SIM115 - closed in close()
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            desc = str(kwargs.get("desc") or "")
            self._kind = "transfer" if desc.endswith("downloading bytes") else "file"
            self._last_emit = 0.0

        def update(self, n: float | None = 1) -> bool | None:
            result = super().update(n)
            if self._kind != "file":
                return result
            now = time.monotonic()
            if now - self._last_emit >= 0.25 or (self.total and self.n >= self.total):
                self._last_emit = now
                totals[file_name] = int(self.n)
                emit({"type": "progress", "file": file_name, "downloaded": int(self.n),
                      "total": int(self.total) if self.total else None})
            return result

        def close(self) -> None:
            try:
                super().close()
            finally:
                try:
                    self.fp.close()
                except Exception:
                    pass

    return Reporter


def _is_retryable(exc: BaseException) -> bool:
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, GatedRepoError)):
        return False
    if isinstance(exc, (PermissionError, FileNotFoundError)):
        return False
    return True


def download_file(api_kwargs: dict[str, Any], entry: RepoFile, *, retries: int, verify: bool,
                  totals: dict[str, int]) -> Path:
    from huggingface_hub import hf_hub_download

    attempt = 0
    while True:
        attempt += 1
        try:
            log.info("downloading %s (attempt %d)", entry.name, attempt)
            path = Path(hf_hub_download(filename=entry.name, tqdm_class=_reporter(entry.name, totals), **api_kwargs))
            if verify:
                ok = verify_file(path, entry)
                emit({"type": "verify", "file": entry.name, "ok": ok})
                if ok is False:
                    log.warning("checksum mismatch for %s; deleting the file so the retry downloads it again",
                                entry.name)
                    real = path.resolve()
                    real.unlink(missing_ok=True)
                    if path.is_symlink():
                        path.unlink(missing_ok=True)
                    raise IOError(f"checksum mismatch for {entry.name}")
            size = path.resolve().stat().st_size
            log.info("downloaded %s (%d bytes)", entry.name, size)
            emit({"type": "file_done", "file": entry.name, "size": size})
            return path
        except Exception as exc:  # noqa: BLE001
            if attempt > retries or not _is_retryable(exc):
                log.warning("giving up on %s after %d attempt(s): %s", entry.name, attempt, exc)
                raise
            wait = min(2 ** attempt, 60)
            log.warning("transient failure for %s (attempt %d): %s; retrying in %ss",
                        entry.name, attempt, exc, wait)
            emit({"type": "retry", "file": entry.name, "attempt": attempt, "wait_s": wait, "error": str(exc)[:300]})
            time.sleep(wait)


def repo_cache_folder(cache_dir: Path, repo_id: str, repo_type: str = "model") -> Path:
    return cache_dir / f"{repo_type}s--{repo_id.replace('/', '--')}"


def is_cached(entry: RepoFile, *, cache_dir: Path, repo_id: str, local_dir: Path | None = None) -> bool:
    """True only for a complete copy: an interrupted blob with the right name is not usable."""
    try:
        if local_dir is not None:
            local = local_dir / entry.name
            return local.is_file() and (entry.size is None or local.stat().st_size == entry.size)
        blob_name = entry.sha256 or entry.blob_id
        if not blob_name:
            return False
        blob = repo_cache_folder(cache_dir, repo_id) / "blobs" / blob_name
        return blob.is_file() and (entry.size is None or blob.stat().st_size == entry.size)
    except OSError:
        return False


def clean_stale_incomplete(folder: Path) -> int:
    """Remove ``*.incomplete`` temp files left by a killed download of this repo."""
    removed = 0
    blobs = folder / "blobs"
    if blobs.is_dir():
        for path in blobs.glob("*.incomplete"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def run(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi, constants, snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    log.info("resolving %s (revision %s, authenticated=%s)", args.repo_id, args.revision or "main", bool(token))
    try:
        info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    except GatedRepoError:
        emit({"type": "error", "code": "GATED", "message": "This repository is gated. Accept its license on the Hub "
                                                         "and add your Hugging Face token in Settings."})
        return 1
    except RepositoryNotFoundError:
        emit({"type": "error", "code": "NOT_FOUND", "message": f"Repository {args.repo_id} was not found."})
        return 1
    except Exception as exc:  # network
        emit({"type": "error", "code": "NETWORK", "message": f"Could not reach the Hub: {exc}"})
        return 1

    commit = info.sha
    files = [RepoFile(s.rfilename, s.size if s.size is not None else (s.lfs.size if s.lfs else None),
                      s.lfs.sha256 if s.lfs else None, s.blob_id) for s in (info.siblings or [])]
    selected = select_files(files, args.include, args.exclude)
    cache_dir = Path(constants.HF_HUB_CACHE)
    target_root = Path(args.local_dir).expanduser() if args.local_dir else cache_dir
    target_root.mkdir(parents=True, exist_ok=True)

    def cached(entry: RepoFile) -> bool:
        return is_cached(entry, cache_dir=cache_dir, repo_id=args.repo_id,
                         local_dir=target_root if args.local_dir else None)

    plan = [{"name": f.name, "size": f.size, "cached": cached(f)} for f in selected]
    total = sum(f.size or 0 for f in selected)
    needed = sum((f.size or 0) for f, p in zip(selected, plan) if not p["cached"])
    emit({"type": "plan", "repo_id": args.repo_id, "revision": args.revision or "main", "commit": commit,
          "files": plan, "total_bytes": total, "download_bytes": needed,
          "target": str(target_root), "mode": "local_dir" if args.local_dir else "hf_cache"})
    log.info("plan: %d file(s), %d selected, %d bytes to download, target %s",
             len(files), len(selected), needed, target_root)
    if args.dry_run:
        log.info("dry run: stopping before any download")
        return 0
    free = shutil.disk_usage(target_root).free
    if needed > free * 0.95:
        log.warning("not enough disk space: need %d bytes, free %d bytes on %s", needed, free, target_root)
        emit({"type": "error", "code": "DISK_FULL",
              "message": f"Needs {needed / 1e9:.2f} GB but only {free / 1e9:.2f} GB is free on {target_root}."})
        return 2
    if not args.local_dir:
        removed = clean_stale_incomplete(repo_cache_folder(cache_dir, args.repo_id))
        if removed:
            log.info("removed %d stale .incomplete file(s)", removed)
            emit({"type": "cleanup", "removed_incomplete": removed})

    api_kwargs: dict[str, Any] = {"repo_id": args.repo_id, "revision": commit, "token": token}
    if args.local_dir:
        api_kwargs["local_dir"] = str(target_root)
    totals: dict[str, int] = {}
    pending = [f for f, p in zip(selected, plan) if not p["cached"]]
    for f, p in zip(selected, plan):
        if p["cached"]:
            emit({"type": "file_done", "file": f.name, "size": f.size, "cached": True})
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {pool.submit(download_file, api_kwargs, f, retries=args.retries, verify=not args.no_verify,
                               totals=totals): f for f in pending}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(entry.name)
                emit({"type": "file_error", "file": entry.name, "message": str(exc)[:500]})
    if failures:
        log.warning("%d file(s) failed: %s", len(failures), ", ".join(failures[:10]))
        emit({"type": "error", "code": "INCOMPLETE",
              "message": f"{len(failures)} file(s) failed: {', '.join(failures[:5])}. Retry to resume."})
        return 1
    log.info("all files downloaded and verified")

    if args.local_dir:
        path = str(target_root)
    else:
        try:  # records refs/<revision> and returns the snapshot folder
            path = snapshot_download(args.repo_id, revision=args.revision or "main", token=token,
                                     allow_patterns=[f.name for f in selected])
        except Exception:
            folder = repo_cache_folder(cache_dir, args.repo_id)
            ref = folder / "refs" / (args.revision or "main")
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(commit or "", encoding="utf-8")
            path = str(folder / "snapshots" / (commit or ""))
    emit({"type": "done", "path": path, "commit": commit, "total_bytes": total,
          "files": [f.name for f in selected]})
    return 0


def main(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from noelle import logging_setup

    logging_setup.configure_download_logging()
    log.info("download child starting: %s", args.repo_id)
    try:
        code = run(args)
        log.info("download child finished with code %d", code)
        return code
    except KeyboardInterrupt:
        log.warning("download child interrupted by the user")
        emit({"type": "error", "code": "CANCELLED", "message": "Download cancelled."})
        return 130
    except Exception as exc:  # noqa: BLE001
        log.exception("download child failed")
        emit({"type": "error", "code": "INTERNAL", "message": f"{type(exc).__name__}: {exc}"})
        return 1
