"""Hugging Face cache inspection, cleanup and relocation (task T-302).

Qt-free: the storage page calls these functions from a background thread.
Every function that deletes something takes an explicit ``cache_dir``; when it
is omitted the live cache is used (``noelle.paths.hf_hub_cache()``, resolved at
call time so a relocated ``HF_HOME`` is honoured).

Layout handled (huggingface_hub 2.x)::

    <hub cache>/
    |-- models--org--name/
    |   |-- blobs/<etag>                      file, or relative symlink into the shared store
    |   |-- blobs/<etag>.<uuid8>.incomplete   temp file of a killed download
    |   |-- refs/main                         commit hash the ref points to
    |   `-- snapshots/<commit>/<files>        relative symlinks to ../../blobs/<etag>
    |-- blobs/<xx>/<xet hash>                 shared store, deduplicated across repos
    |-- .locks/<repo folder>/<etag>.lock      download locks
    `-- CACHEDIR.TAG

Only ``models--``, ``datasets--``, ``spaces--`` and ``kernels--`` folders are
repos. Every other top-level entry (the shared store, locks, tags, stray files)
is skipped: never reported as corrupted, never deleted.

Revision and repo deletion goes through huggingface_hub's ``DeleteCacheStrategy``
(blob exclusivity and shared-store garbage collection). Each strategy is scoped
to the repos being cleaned and every planned path is checked to sit inside
those repo folders before anything is removed.
"""

from __future__ import annotations

import collections
import dataclasses
import errno
import logging
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from noelle import paths

log = logging.getLogger("noelle.storage")

REPO_TYPES = {"models": "model", "datasets": "dataset", "spaces": "space", "kernels": "kernel"}
_REPO_DIR_RE = re.compile(r"^(models|datasets|spaces|kernels)--(.+)$")
_IGNORED_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
_SHARED_STORE = "blobs"
_SHARED_MARKER = ".huggingface-shared-blobs"
_HEX2_RE = re.compile(r"^[0-9a-f]{2}$")
_XET_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COPY_CHUNK = 8 * 1024 * 1024
_RELOCATION_MARGIN = 64 * 1024 * 1024

Progress = Callable[[int, int, str], None]


class StopEvent(Protocol):
    def is_set(self) -> bool: ...


class RelocationError(Exception):
    """The cache could not be relocated; the source is left untouched."""


class RelocationCancelled(RelocationError):
    """``stop_event`` was set during the copy; the partial destination was removed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class _Record:
    """Dataclass mixin: ``result["key"]`` item access and a JSON-friendly ``to_dict()``."""

    def _public(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if not f.name.startswith("_")]  # type: ignore[arg-type]

    def __getitem__(self, key: str) -> Any:
        if key in self._public():
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return self._public()

    def to_dict(self) -> dict[str, Any]:
        return {name: _plain(getattr(self, name)) for name in self._public()}


def _plain(value: Any) -> Any:
    if isinstance(value, _Record):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class CachedFile(_Record):
    name: str  # path inside the snapshot, POSIX separators
    size_bytes: int
    blob_path: Path  # repo blob entry the snapshot file points to


@dataclass
class CachedRevision(_Record):
    commit: str
    refs: list[str]
    size_bytes: int
    nb_files: int
    files: list[CachedFile]
    snapshot_path: Path
    detached: bool  # no ref points to this revision
    last_modified: float = 0.0
    exclusive_bytes: int = 0  # freed by deleting only this revision


@dataclass
class IncompleteFile(_Record):
    path: Path
    size_bytes: int
    modified: float
    repo_id: str


@dataclass
class CachedRepo(_Record):
    repo_id: str
    repo_type: str
    size_bytes: int
    nb_files: int
    last_accessed: float
    last_modified: float
    path: Path
    revisions: list[CachedRevision]
    incomplete: list[IncompleteFile] = field(default_factory=list)
    reclaimable_bytes: int = 0  # detached revisions + incomplete downloads of this repo
    corrupted: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class CacheReport(_Record):
    cache_dir: Path
    repos: list[CachedRepo]
    total_size: int  # physical bytes: shared and hard-linked blobs counted once
    reclaimable_bytes: int  # exclusive bytes of detached revisions + incomplete downloads
    warnings: list[str]
    detached_bytes: int = 0
    incomplete_bytes: int = 0
    scanner: str = "huggingface_hub"  # or "manual" when the hub scanner failed

    def repo(self, repo_id: str, repo_type: str = "model") -> CachedRepo | None:
        return next((r for r in self.repos if r.repo_id == repo_id and r.repo_type == repo_type), None)


@dataclass
class DeleteResult(_Record):
    freed_bytes: int
    deleted: list[str]  # commits removed (or that would be removed in a dry run)
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class PruneResult(_Record):
    removed: list[Path]
    freed_bytes: int
    skipped: list[Path] = field(default_factory=list)
    dry_run: bool = False


@dataclass
class RelocationResult(_Record):
    new_hf_home: Path
    moved_bytes: int
    method: str  # "rename" | "copy"
    files: int = 0
    relinked: int = 0  # absolute symlinks rewritten to relative ones
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass
class _RepoState:
    repo: CachedRepo
    hf: Any = None  # huggingface_hub CachedRepoInfo when the hub scanner accepted this folder


@dataclass
class _Scan:
    cache: Path
    states: list[_RepoState]
    warnings: list[str]
    hf_ok: bool

    def state(self, folder: Path) -> _RepoState | None:
        return next((s for s in self.states if s.repo.path == folder), None)


def scan_hf_cache(cache_dir: str | os.PathLike[str] | None = None) -> CacheReport:
    """Scan a hub cache (default: the live one). Never raises on corrupted content."""
    scan = _scan(cache_dir)
    prunable, _skipped = _prunable_detached(scan, (), 60.0)
    detached = _plan(scan, prunable)
    incomplete = sum(f.size_bytes for s in scan.states for f in s.repo.incomplete)
    report = CacheReport(
        cache_dir=scan.cache,
        repos=[s.repo for s in scan.states],
        total_size=_physical_bytes([s.repo.path for s in scan.states], scan.cache),
        reclaimable_bytes=detached.expected + incomplete,
        warnings=scan.warnings + detached.warnings,
        detached_bytes=detached.expected,
        incomplete_bytes=incomplete,
        scanner="huggingface_hub" if scan.hf_ok else "manual",
    )
    for state in scan.states:
        repo = state.repo
        for revision in repo.revisions:
            revision.exclusive_bytes = _plan(scan, {repo.path: {revision.commit}}).expected
        mine = prunable.get(repo.path)
        repo.reclaimable_bytes = (_plan(scan, {repo.path: mine}).expected if mine else 0) + sum(
            f.size_bytes for f in repo.incomplete)
    return report


def _scan(cache_dir: str | os.PathLike[str] | None) -> _Scan:
    cache = _cache_path(cache_dir)
    if not cache.is_dir():
        state = "is not a directory" if cache.exists() else "does not exist"
        return _Scan(cache, [], [f"Hugging Face cache {cache} {state}."], hf_ok=False)
    folders = _repo_folders(cache)
    info, warnings = _hub_scan(cache)
    hf_repos: dict[Path, Any] = {}
    hf_warnings: list[str] = []
    if info is not None:
        known = set(folders)
        hf_repos = {Path(r.repo_path): r for r in info.repos if Path(r.repo_path) in known}
        hf_warnings = [str(w) for w in info.warnings]
        warnings += [w for w in hf_warnings if not _mentions_any(w, folders) and not _mentions_skipped(w, cache, known)]
    states: list[_RepoState] = []
    for folder in folders:
        repo_type, repo_id = _parse_folder(folder.name)
        hf_repo = hf_repos.get(folder)
        try:
            state = _state_from_hub(folder, hf_repo) if hf_repo is not None else _state_from_walk(folder)
        except OSError as exc:
            state = _RepoState(_empty_repo(folder, repo_type, repo_id))
            state.repo.warnings.append(f"Could not read {folder}: {exc}")
        if hf_repo is None and info is not None:
            state.repo.corrupted = True
            state.repo.warnings[:0] = [w for w in hf_warnings if _mentions_any(w, [folder])]
        elif state.repo.warnings:
            state.repo.corrupted = True
        state.repo.incomplete = _incomplete_files(folder, repo_id)
        warnings += state.repo.warnings
        states.append(state)
    states.sort(key=lambda s: (s.repo.repo_type, s.repo.repo_id.lower()))
    return _Scan(cache, states, warnings, hf_ok=info is not None)


def _cache_path(cache_dir: str | os.PathLike[str] | None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else paths.hf_hub_cache()
    base = base.expanduser()
    try:
        return base.resolve()
    except OSError:
        return base.absolute()


def _repo_folders(cache: Path) -> list[Path]:
    """``<type>s--<id>`` folders; the shared store, locks and stray entries are skipped."""
    try:
        with os.scandir(cache) as entries:
            names = sorted(e.name for e in entries if _REPO_DIR_RE.match(e.name) and _is_dir(e))
    except OSError:
        return []
    return [cache / name for name in names]


def _is_dir(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def _parse_folder(name: str) -> tuple[str, str]:
    match = _REPO_DIR_RE.match(name)
    if not match:
        return "model", name
    return REPO_TYPES[match.group(1)], match.group(2).replace("--", "/")


def _mentions_any(text: str, folders: Iterable[Path]) -> bool:
    for folder in folders:
        if re.search(re.escape(str(folder)) + r"(?=$|[\\/)\s.,:;'\"])", text):
            return True
    return False


def _mentions_skipped(text: str, cache: Path, folders: set[Path]) -> bool:
    """True for hub-scanner warnings about non-repo entries (shared store, CACHEDIR.TAG, ...)."""
    try:
        with os.scandir(cache) as entries:
            skipped = [cache / e.name for e in entries if cache / e.name not in folders]
    except OSError:
        return False
    return _mentions_any(text, skipped)


def _hub_scan(cache: Path) -> tuple[Any, list[str]]:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as exc:  # pragma: no cover - huggingface_hub is a hard dependency
        return None, [f"huggingface_hub is unavailable ({exc}); used a manual scan."]
    try:
        return scan_cache_dir(cache), []
    except Exception as exc:  # e.g. a permission error inside one repo aborts the hub scan
        return None, [f"huggingface_hub could not scan {cache} ({type(exc).__name__}: {exc}); used a manual scan."]


def _empty_repo(folder: Path, repo_type: str, repo_id: str) -> CachedRepo:
    try:
        st = folder.stat()
        accessed, modified = st.st_atime, st.st_mtime
    except OSError:
        accessed = modified = 0.0
    return CachedRepo(repo_id=repo_id, repo_type=repo_type, size_bytes=0, nb_files=0, last_accessed=accessed,
                      last_modified=modified, path=folder, revisions=[])


def _state_from_hub(folder: Path, hf_repo: Any) -> _RepoState:
    revisions = []
    for rev in hf_repo.revisions:
        snapshot = Path(rev.snapshot_path)
        files = sorted((CachedFile(name=_snapshot_name(Path(f.file_path), snapshot), size_bytes=int(f.size_on_disk),
                                   blob_path=Path(f.blob_path)) for f in rev.files), key=lambda f: f.name)
        refs = sorted(rev.refs)
        revisions.append(CachedRevision(commit=rev.commit_hash, refs=refs, size_bytes=int(rev.size_on_disk),
                                        nb_files=len(files), files=files, snapshot_path=snapshot, detached=not refs,
                                        last_modified=float(rev.last_modified)))
    repo = CachedRepo(repo_id=hf_repo.repo_id, repo_type=str(hf_repo.repo_type), size_bytes=int(hf_repo.size_on_disk),
                      nb_files=int(hf_repo.nb_files), last_accessed=float(hf_repo.last_accessed),
                      last_modified=float(hf_repo.last_modified), path=folder, revisions=_sorted_revisions(revisions))
    return _RepoState(repo, hf_repo)


def _state_from_walk(folder: Path) -> _RepoState:
    """Tolerant scan: broken links and dangling refs become warnings, not failures."""
    repo_type, repo_id = _parse_folder(folder.name)
    repo = _empty_repo(folder, repo_type, repo_id)
    refs = _read_refs(folder / "refs", repo.warnings)
    stats: dict[Path, os.stat_result] = {}
    revisions: list[CachedRevision] = []
    snapshots = folder / "snapshots"
    if snapshots.is_dir():
        with os.scandir(snapshots) as entries:
            children = sorted(entries, key=lambda e: e.name)
        for child in children:
            if child.name in _IGNORED_NAMES:
                continue
            if not child.is_dir(follow_symlinks=False):
                repo.warnings.append(f"Unexpected entry in snapshots folder: {child.path}")
                continue
            revisions.append(_walk_revision(Path(child.path), refs.pop(child.name, []), stats, repo.warnings))
    else:
        repo.warnings.append(f"Snapshots folder missing in cached repo: {snapshots}")
    for commit, names in refs.items():
        repo.warnings.append(f"Ref(s) {', '.join(sorted(names))} point to missing revision {commit} ({folder})")
    used = {f.blob_path for r in revisions for f in r.files}
    repo.revisions = _sorted_revisions(revisions)
    repo.size_bytes = sum(stats[b].st_size for b in used)
    repo.nb_files = len(used)
    if used:
        repo.last_accessed = max(stats[b].st_atime for b in used)
        repo.last_modified = max(stats[b].st_mtime for b in used)
    return _RepoState(repo)


def _read_refs(refs_dir: Path, warnings: list[str]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    if not refs_dir.exists():
        return refs
    if not refs_dir.is_dir():
        warnings.append(f"Refs entry is not a directory: {refs_dir}")
        return refs
    for root, _dirs, names in os.walk(refs_dir):
        for name in names:
            if name in _IGNORED_NAMES:
                continue
            path = Path(root, name)
            try:
                commit = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                warnings.append(f"Unreadable ref {path}: {exc}")
                continue
            refs.setdefault(commit, []).append(path.relative_to(refs_dir).as_posix())
    return refs


def _walk_revision(snapshot: Path, refs: list[str], stats: dict[Path, os.stat_result],
                   warnings: list[str]) -> CachedRevision:
    files: list[CachedFile] = []
    for root, dirs, names in os.walk(snapshot):
        dirs.sort()
        for name in sorted(names):
            if name in _IGNORED_NAMES:
                continue
            path = Path(root, name)
            try:
                blob = Path(os.path.abspath(path.parent / os.readlink(path))) if path.is_symlink() else path
            except OSError as exc:
                warnings.append(f"Unreadable snapshot entry {path}: {exc}")
                continue
            st = stats.get(blob)
            if st is None:
                try:
                    st = blob.stat()  # follows a shared-store link to its payload
                except OSError:
                    warnings.append(f"Blob missing (broken symlink): {path} -> {blob}")
                    continue
                stats[blob] = st
            files.append(CachedFile(name=path.relative_to(snapshot).as_posix(), size_bytes=st.st_size, blob_path=blob))
    files.sort(key=lambda f: f.name)
    used = {f.blob_path for f in files}
    if used:
        modified = max(stats[b].st_mtime for b in used)
    else:
        try:
            modified = snapshot.stat().st_mtime
        except OSError:
            modified = 0.0
    return CachedRevision(commit=snapshot.name, refs=sorted(refs), size_bytes=sum(stats[b].st_size for b in used),
                          nb_files=len(files), files=files, snapshot_path=snapshot, detached=not refs,
                          last_modified=modified)


def _snapshot_name(path: Path, snapshot: Path) -> str:
    try:
        return path.relative_to(snapshot).as_posix()
    except ValueError:
        return path.name


def _sorted_revisions(revisions: list[CachedRevision]) -> list[CachedRevision]:
    return sorted(revisions, key=lambda r: (r.detached, -r.last_modified, r.commit))


def _incomplete_files(folder: Path, repo_id: str) -> list[IncompleteFile]:
    found: list[IncompleteFile] = []
    try:
        with os.scandir(folder / "blobs") as entries:
            candidates = [e for e in entries if e.name.endswith(".incomplete")]
    except OSError:
        return found
    for entry in candidates:
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            found.append(IncompleteFile(path=Path(entry.path), size_bytes=st.st_size, modified=st.st_mtime,
                                        repo_id=repo_id))
    return sorted(found, key=lambda f: f.path)


# ---------------------------------------------------------------------------
# Physical sizes
# ---------------------------------------------------------------------------


def _is_shared_store(path: Path) -> bool:
    marker = path / _SHARED_MARKER
    try:
        if not stat.S_ISDIR(path.lstat().st_mode) or not stat.S_ISREG(marker.lstat().st_mode):
            return False
        return marker.read_text(encoding="utf-8").strip() == "1"
    except (OSError, UnicodeError):
        return False


def _physical_bytes(folders: Iterable[Path], cache: Path) -> int:
    """Blob bytes on disk for the given repos plus the shared store; every inode counted once."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for folder in folders:
        if folder.is_symlink():
            continue
        total += _tree_bytes(folder / "blobs", seen) + _tree_bytes(folder / "snapshots", seen)
    store = cache / _SHARED_STORE
    if _is_shared_store(store):
        total += _store_payload_bytes(store, seen)
    return total


def _store_payload_bytes(store: Path, seen: set[tuple[int, int]]) -> int:
    total = 0
    try:
        prefixes = [p for p in store.iterdir() if _HEX2_RE.match(p.name)]
    except OSError:
        return 0
    for prefix in prefixes:
        try:
            entries = list(prefix.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not _XET_HASH_RE.match(entry.name):
                continue
            total += _count_file(entry, seen)
    return total


def _count_file(path: str | os.PathLike[str], seen: set[tuple[int, int]]) -> int:
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if not stat.S_ISREG(st.st_mode):
        return 0
    key = (st.st_dev, st.st_ino)
    if key in seen:
        return 0
    seen.add(key)
    return st.st_size


def _tree_bytes(root: Path, seen: set[tuple[int, int]], exclude: Iterable[Path] = ()) -> int:
    """Regular-file bytes under ``root``; symlinks are never followed, unreadable entries are skipped."""
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            return _count_file(root, seen) if root.exists() else 0
    except OSError:
        return 0
    skip = {os.path.normpath(str(p)) for p in exclude}
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _exc: None):
        if skip:
            dirnames[:] = [d for d in dirnames if os.path.normpath(os.path.join(dirpath, d)) not in skip]
        for name in filenames:
            total += _count_file(os.path.join(dirpath, name), seen)
    return total


# ---------------------------------------------------------------------------
# Deletion planning and execution
# ---------------------------------------------------------------------------


@dataclass
class _LocalStrategy:
    """Fallback executor with the hub strategy's shape (no shared-store garbage collection)."""

    expected_freed_size: int
    blobs: frozenset[Path]
    refs: frozenset[Path]
    repos: frozenset[Path]
    snapshots: frozenset[Path]

    def execute(self) -> None:
        for group in (self.refs, self.snapshots, self.blobs, self.repos):
            for path in group:
                _remove_path(path)


@dataclass
class _Plan:
    strategies: list[Any] = field(default_factory=list)
    expected: int = 0
    targets: list[tuple[Path, str]] = field(default_factory=list)  # (snapshot path, commit)
    folders: set[Path] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


def _plan(scan: _Scan, commits: dict[Path, set[str]], whole: Iterable[Path] = ()) -> _Plan:
    """Plan the removal of ``commits`` per repo folder and of ``whole`` repo folders."""
    plan = _Plan()
    whole = set(whole)
    hub_group: dict[Path, set[str]] = {}
    for folder in sorted(set(commits) | whole):
        state = scan.state(folder)
        if state is None:
            continue
        wanted = set() if folder in whole else commits.get(folder, set())
        revisions = [r for r in state.repo.revisions if folder in whole or r.commit in wanted]
        if not revisions and folder not in whole:
            continue
        plan.folders.add(folder)
        plan.targets += [(r.snapshot_path, r.commit) for r in revisions]
        if folder in whole:
            expected = _hub_expected(scan, {folder: {r.commit for r in state.repo.revisions}}) \
                if state.hf is not None and state.repo.revisions else _manual_whole_bytes(state)
            plan.strategies.append(_strategy(scan.cache, expected=expected, repos={folder}))
            plan.expected += expected
        elif state.hf is not None:
            hub_group[folder] = {r.commit for r in revisions}
        else:
            strategy = _manual_strategy(scan.cache, state, {r.commit for r in revisions})
            plan.strategies.append(strategy)
            plan.expected += strategy.expected_freed_size
    if hub_group:
        for strategy in _hub_strategies(scan, hub_group, plan.warnings):
            plan.strategies.append(strategy)
            plan.expected += strategy.expected_freed_size
    return plan


def _hub_info(scan: _Scan, states: list[_RepoState]) -> Any:
    from huggingface_hub.utils import HFCacheInfo

    names = {f.name for f in dataclasses.fields(HFCacheInfo)}
    kwargs: dict[str, Any] = {"size_on_disk": 0, "repos": frozenset(s.hf for s in states), "warnings": []}
    if "incomplete_files" in names:
        kwargs["incomplete_files"] = frozenset()
    if "cache_dir" in names:
        kwargs["cache_dir"] = scan.cache
    return HFCacheInfo(**kwargs)


def _hub_groups(scan: _Scan, wanted: dict[Path, set[str]]) -> list[list[_RepoState]]:
    """One group for all repos, or one per repo when a target commit hash exists in several repos.

    ``HFCacheInfo.delete_revisions`` matches bare hashes against every repo it holds, so a
    hash shared by two repos must never be planned in a combined scope.
    """
    states = [s for f in sorted(wanted) if (s := scan.state(f)) is not None and s.hf is not None]
    counts = collections.Counter(r.commit_hash for s in states for r in s.hf.revisions)
    if any(counts[c] > 1 for commits in wanted.values() for c in commits):
        return [[s] for s in states]
    return [states] if states else []


def _hub_strategies(scan: _Scan, wanted: dict[Path, set[str]], warnings: list[str]) -> list[Any]:
    strategies = []
    for group in _hub_groups(scan, wanted):
        commits = sorted({c for s in group for c in wanted[s.repo.path]})
        raw = _hub_info(scan, group).delete_revisions(*commits)
        strategies.append(_confine(scan.cache, raw, {s.repo.path for s in group}, group, warnings))
    return strategies


def _hub_expected(scan: _Scan, wanted: dict[Path, set[str]]) -> int:
    return sum(s.expected_freed_size for s in _hub_strategies(scan, wanted, []))


def _confine(cache: Path, strategy: Any, folders: set[Path], states: list[_RepoState], warnings: list[str]) -> Any:
    """Drop any planned path outside the repo folders being cleaned (e.g. absolute links elsewhere)."""
    sizes = {f.blob_path: f.size_bytes for s in states for r in s.repo.revisions for f in r.files}

    def blob_ok(path: Path) -> bool:
        return path.parent.name == "blobs" and path.parent.parent in folders

    def ref_ok(path: Path) -> bool:
        return any(folder / "refs" in path.parents for folder in folders)

    def snapshot_ok(path: Path) -> bool:
        return path.parent.name == "snapshots" and path.parent.parent in folders

    def repo_ok(path: Path) -> bool:
        return path in folders and path.parent == cache

    kept = {
        "blobs": frozenset(p for p in strategy.blobs if blob_ok(Path(p))),
        "refs": frozenset(p for p in strategy.refs if ref_ok(Path(p))),
        "snapshots": frozenset(p for p in strategy.snapshots if snapshot_ok(Path(p))),
        "repos": frozenset(p for p in strategy.repos if repo_ok(Path(p))),
    }
    dropped = [p for name in kept for p in getattr(strategy, name) if p not in kept[name]]
    if not dropped and not getattr(strategy, "files", None):
        return strategy
    for path in dropped:
        warnings.append(f"Skipped {path}: outside the repo folder being cleaned.")
    freed = max(0, int(strategy.expected_freed_size) - sum(sizes.get(Path(p), 0) for p in dropped))
    changes: dict[str, Any] = dict(kept, expected_freed_size=freed)
    if hasattr(strategy, "files"):
        changes["files"] = frozenset()
    return dataclasses.replace(strategy, **changes)


def _manual_strategy(cache: Path, state: _RepoState, commits: set[str]) -> Any:
    """Conservative plan for repos the hub scanner rejected: shared-store links never count as freed."""
    folder = state.repo.path
    targets = [r for r in state.repo.revisions if r.commit in commits]
    remaining = [r for r in state.repo.revisions if r.commit not in commits]
    if targets and not remaining:  # same rule as the hub: no revision left -> the folder goes
        return _strategy(cache, expected=_manual_whole_bytes(state), repos={folder})
    keep = {f.blob_path for r in remaining for f in r.files}
    blobs: set[Path] = set()
    refs: set[Path] = set()
    snapshots: set[Path] = set()
    sizes: dict[Path, int] = {}
    for revision in targets:
        snapshots.add(revision.snapshot_path)
        refs.update(folder / "refs" / ref for ref in revision.refs)
        for item in revision.files:
            if revision.snapshot_path in item.blob_path.parents:  # file stored inside the snapshot
                sizes[item.blob_path] = item.size_bytes
            elif item.blob_path not in keep and item.blob_path.parent == folder / "blobs":
                blobs.add(item.blob_path)
                if not os.path.islink(item.blob_path):
                    sizes[item.blob_path] = item.size_bytes
    return _strategy(cache, expected=sum(sizes.values()), blobs=blobs, refs=refs, snapshots=snapshots)


def _manual_whole_bytes(state: _RepoState) -> int:
    folder = state.repo.path
    sizes: dict[Path, int] = {}
    for revision in state.repo.revisions:
        for item in revision.files:
            inside = item.blob_path.parent == folder / "blobs" or revision.snapshot_path in item.blob_path.parents
            if inside and not os.path.islink(item.blob_path):
                sizes[item.blob_path] = item.size_bytes
    return sum(sizes.values())


def _strategy(cache: Path, *, expected: int, blobs: Iterable[Path] = (), refs: Iterable[Path] = (),
              repos: Iterable[Path] = (), snapshots: Iterable[Path] = ()) -> Any:
    fields = {"expected_freed_size": int(expected), "blobs": frozenset(blobs), "refs": frozenset(refs),
              "repos": frozenset(repos), "snapshots": frozenset(snapshots)}
    try:
        from huggingface_hub.utils import DeleteCacheStrategy
    except Exception:  # pragma: no cover - huggingface_hub is a hard dependency
        return _LocalStrategy(**fields)
    if "cache_dir" in {f.name for f in dataclasses.fields(DeleteCacheStrategy)}:
        fields["cache_dir"] = cache  # lets execute() garbage-collect shared-store payloads
    return DeleteCacheStrategy(**fields)


def _execute(scan: _Scan, plan: _Plan, *, dry_run: bool, warnings: list[str]) -> DeleteResult:
    warnings = warnings + plan.warnings
    commits = sorted({commit for _snapshot, commit in plan.targets})
    if dry_run or not plan.strategies:
        return DeleteResult(freed_bytes=plan.expected if dry_run else 0, deleted=commits if dry_run else [],
                            dry_run=dry_run, warnings=warnings)
    folders = [s.repo.path for s in scan.states]  # whole cache: hard links elsewhere keep their bytes
    before = _physical_bytes(folders, scan.cache)
    for strategy in plan.strategies:
        try:
            strategy.execute()
        except OSError as exc:
            warnings.append(f"Deletion incomplete: {exc}")
    after = _physical_bytes(folders, scan.cache)
    deleted = sorted({commit for snapshot, commit in plan.targets if not os.path.lexists(snapshot)})
    return DeleteResult(freed_bytes=max(0, before - after), deleted=deleted, dry_run=False, warnings=warnings)


def _excluded(state: _RepoState, exclude: Iterable[str]) -> bool:
    names = set(exclude)
    repo = state.repo
    return bool(names & {repo.repo_id, repo.path.name, f"{repo.repo_type}/{repo.repo_id}"})


def delete_revisions(commits: Iterable[str], cache_dir: str | os.PathLike[str] | None = None, *,
                     dry_run: bool = False, repo_id: str | None = None) -> DeleteResult:
    """Delete cached revisions by commit hash; blobs still used by another revision are kept.

    A hash is matched in every repo unless ``repo_id`` restricts it. When every revision
    of a repo is deleted its whole folder is removed (huggingface_hub semantics).
    """
    wanted = {commits} if isinstance(commits, str) else {c for c in commits if c}
    wanted = {c.strip() for c in wanted if c.strip()}
    scan = _scan(cache_dir)
    targets: dict[Path, set[str]] = {}
    for state in scan.states:
        if repo_id is not None and state.repo.repo_id != repo_id:
            continue
        hits = {r.commit for r in state.repo.revisions if r.commit in wanted}
        if hits:
            targets[state.repo.path] = hits
    found = set().union(*targets.values()) if targets else set()
    warnings = [f"Revision {commit} is not in the cache." for commit in sorted(wanted - found)]
    return _execute(scan, _plan(scan, targets), dry_run=dry_run, warnings=warnings)


def prune_detached(cache_dir: str | os.PathLike[str] | None = None, *, exclude_repos: Iterable[str] = frozenset(),
                   min_age_s: float = 60.0, dry_run: bool = False) -> DeleteResult:
    """Delete every revision no ref points to.

    Pass the ids of repos with active downloads in ``exclude_repos``: a first download
    in progress has no ``refs/main`` yet, so its snapshot looks detached. As a backstop,
    a repo without any ref is also skipped while it shows download activity (a partial
    file or snapshot folder touched within ``min_age_s`` seconds, or a held download lock).
    """
    scan = _scan(cache_dir)
    targets, skipped = _prunable_detached(scan, exclude_repos, min_age_s)
    warnings = [f"Skipped {repo_id}: a download looks active." for repo_id in skipped]
    return _execute(scan, _plan(scan, targets), dry_run=dry_run, warnings=warnings)


def _prunable_detached(scan: _Scan, exclude: Iterable[str], min_age_s: float) -> tuple[dict[Path, set[str]], list[str]]:
    targets: dict[Path, set[str]] = {}
    skipped: list[str] = []
    now = time.time()
    for state in scan.states:
        detached = {r.commit for r in state.repo.revisions if r.detached}
        if not detached or _excluded(state, exclude):
            continue
        if len(detached) == len(state.repo.revisions) and _download_activity(scan.cache, state, min_age_s, now):
            skipped.append(state.repo.repo_id)
            continue
        targets[state.repo.path] = detached
    return targets, skipped


def _download_activity(cache: Path, state: _RepoState, min_age_s: float, now: float) -> bool:
    """Fresh partial files, recently touched snapshot folders or a held lock: a download is running."""
    folder = state.repo.path
    for item in state.repo.incomplete:
        if now - item.modified < min_age_s or _download_active(cache, folder, item.path):
            return True
    for revision in state.repo.revisions:
        for dirpath, _dirs, _files in os.walk(revision.snapshot_path):
            try:
                if now - os.stat(dirpath).st_mtime < min_age_s:
                    return True
            except OSError:
                continue
    return False


def delete_repo(repo_id: str, cache_dir: str | os.PathLike[str] | None = None, *, repo_type: str = "model",
                dry_run: bool = False) -> int:
    """Remove a whole repo folder (all revisions, refs and partial downloads); returns freed bytes."""
    prefix = f"{repo_type}s"
    if prefix not in REPO_TYPES or not repo_id or repo_id != repo_id.strip() or "\\" in repo_id:
        raise ValueError(f"Invalid repo: {repo_type}/{repo_id}")
    name = f"{prefix}--{repo_id.replace('/', '--')}"
    if os.sep in name or (os.altsep and os.altsep in name) or name.endswith("--"):
        raise ValueError(f"Invalid repo id: {repo_id}")
    scan = _scan(cache_dir)
    folder = scan.cache / name
    if folder.parent != scan.cache or scan.state(folder) is None:
        return 0
    plan = _plan(scan, {}, whole={folder})
    incomplete = sum(f.size_bytes for f in scan.state(folder).repo.incomplete)  # type: ignore[union-attr]
    if dry_run:
        return plan.expected + incomplete
    freed = _execute(scan, plan, dry_run=False, warnings=[]).freed_bytes
    log.info("deleted %s/%s, freeing %d bytes", repo_type, repo_id, freed)
    return freed


def prune_incomplete(cache_dir: str | os.PathLike[str] | None = None, *, exclude_repos: Iterable[str] = frozenset(),
                     min_age_s: float = 60.0, dry_run: bool = False) -> PruneResult:
    """Delete ``blobs/*.incomplete`` left by killed downloads.

    Skips repos in ``exclude_repos`` (active downloads), files modified within
    ``min_age_s`` seconds and files whose download lock is currently held.
    """
    cache = _cache_path(cache_dir)
    exclude = set(exclude_repos)
    result = PruneResult(removed=[], freed_bytes=0, dry_run=dry_run)
    now = time.time()
    for folder in _repo_folders(cache) if cache.is_dir() else []:
        repo_type, repo_id = _parse_folder(folder.name)
        files = _incomplete_files(folder, repo_id)
        if exclude & {repo_id, folder.name, f"{repo_type}/{repo_id}"}:
            result.skipped += [f.path for f in files]
            continue
        for item in files:
            if now - item.modified < min_age_s or _download_active(cache, folder, item.path):
                result.skipped.append(item.path)
                continue
            if not dry_run:
                try:
                    item.path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    log.warning("Could not delete %s: %s", item.path, exc)
                    result.skipped.append(item.path)
                    continue
            result.removed.append(item.path)
            result.freed_bytes += item.size_bytes
    if result.removed:
        log.info("pruned %d incomplete file(s), freeing %d bytes", len(result.removed), result.freed_bytes)
    return result


def _download_active(cache: Path, folder: Path, incomplete: Path) -> bool:
    """True when a process holds huggingface_hub's lock for this blob (non-blocking flock probe)."""
    etag = incomplete.name.split(".", 1)[0]
    lock = cache / ".locks" / folder.name / f"{etag}.lock"
    if not lock.is_file():
        return False
    try:
        import fcntl
    except ImportError:  # Windows: rely on the age rule
        return False
    try:
        fd = os.open(lock, os.O_RDONLY)
    except OSError:
        return True  # cannot tell: assume busy
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            _rmtree(path)
    except FileNotFoundError:
        pass


def _rmtree(path: Path) -> None:
    def retry(func: Callable[..., Any], target: str, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            return
        if not isinstance(exc, PermissionError):
            raise exc
        parent = os.path.dirname(target)
        os.chmod(parent, os.stat(parent).st_mode | stat.S_IWUSR | stat.S_IXUSR)
        if not os.path.islink(target):
            os.chmod(target, os.lstat(target).st_mode | stat.S_IWUSR)
        func(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:  # pragma: no cover - Python 3.11
        shutil.rmtree(path, onerror=lambda func, target, info: retry(func, target, info[1]))


# ---------------------------------------------------------------------------
# Relocation
# ---------------------------------------------------------------------------


@dataclass
class _TreeStats:
    files: int = 0
    symlinks: int = 0
    bytes: int = 0
    dangling: int = 0
    escaping: int = 0  # symlinks resolving into a forbidden tree (the old location)


def relocate_cache(src_home: str | os.PathLike[str], dst_home: str | os.PathLike[str], *,
                   progress: Progress | None = None, stop_event: StopEvent | None = None,
                   force_copy: bool = False) -> RelocationResult:
    """Move an ``HF_HOME`` directory (hub cache with relative snapshot symlinks included).

    Same filesystem: ``os.rename``. Otherwise (or ``force_copy``): copy with byte
    progress, verify file count, bytes and links, then delete the source. Raises
    :class:`RelocationError` (source untouched) when the move is refused or fails.
    The caller updates settings (see :func:`shell_export_line`) and restarts workers.
    """
    src = Path(src_home).expanduser().absolute()
    dst = Path(dst_home).expanduser().absolute()
    if src.is_symlink():
        raise RelocationError(f"{src} is a symbolic link; relocate the directory it points to instead.")
    if not src.is_dir():
        raise RelocationError(f"Source {src} does not exist or is not a directory.")
    src_real = src.resolve()
    dst_real = dst.resolve()
    if dst_real == src_real or src_real in dst_real.parents:
        raise RelocationError(f"Destination {dst} is inside the source {src}.")
    if dst.is_symlink() or (dst.exists() and not dst.is_dir()):
        raise RelocationError(f"Destination {dst} exists and is not a directory.")
    dst_existed = dst.is_dir()
    if dst_existed and any(dst.iterdir()):
        raise RelocationError(f"Destination {dst} exists and is not empty.")
    stats = _tree_stats(src)
    anchor = _existing_ancestor(dst)
    method = "copy" if force_copy or _device(src) != _device(anchor) else "rename"
    _check_stop(stop_event)
    log.info("relocating the Hugging Face cache from %s to %s (%s, %d files, %d bytes)",
             src, dst, method, stats.files, stats.bytes)
    old_roots = [src, src_real]
    if method == "rename":
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst_existed:
                dst.rmdir()
            os.rename(src, dst)
        except OSError as exc:
            if dst_existed and not dst.exists():
                dst.mkdir()
            if exc.errno != errno.EXDEV:
                raise RelocationError(f"Could not move {src} to {dst}: {exc}") from exc
            method = "copy"  # bind mounts and overlay filesystems report EXDEV
        else:
            moved_warnings: list[str] = []
            try:
                relinked = _relink_absolute(dst, old_roots)
            except OSError as exc:  # the move itself succeeded; report instead of raising
                relinked = 0
                moved_warnings.append(f"Moved, but absolute symlinks could not be rewritten: {exc}")
            if progress is not None:
                progress(stats.bytes, stats.bytes, str(dst))
            log.info("cache relocation complete (rename) at %s", dst)
            return RelocationResult(dst, stats.bytes, "rename", stats.files, relinked, moved_warnings)
    need = stats.bytes + max(stats.bytes // 100, _RELOCATION_MARGIN)
    free = _free_space(anchor)
    if free < need:
        raise RelocationError(f"Not enough free space on {anchor}: {need} bytes needed, {free} available.")
    try:
        _copy_tree(src, dst, stats.bytes, progress, stop_event)
        relinked = _relink_absolute(dst, old_roots)
        copied = _tree_stats(dst, forbidden=[src, src_real])
        problems = _compare(stats, copied)
        if problems:
            raise RelocationError("Copy verification failed: " + "; ".join(problems))
    except RelocationError:
        _discard(dst, recreate=dst_existed)
        raise
    except OSError as exc:  # includes shutil.Error from copytree
        _discard(dst, recreate=dst_existed)
        raise RelocationError(f"Could not copy {src} to {dst}: {exc}") from exc
    except BaseException:
        _discard(dst, recreate=dst_existed)
        raise
    warnings: list[str] = []
    try:
        _rmtree(src)
    except OSError as exc:
        warnings.append(f"Copied and verified, but the old cache at {src} could not be fully removed: {exc}")
        log.warning("relocation left files behind at %s: %s", src, exc)
    log.info("cache relocation complete (copy) at %s", dst)
    return RelocationResult(dst, stats.bytes, "copy", stats.files, relinked, warnings)


def shell_export_line(new_home: str | os.PathLike[str]) -> str:
    """``export HF_HOME="..."`` for shell profiles, with double-quote escaping."""
    value = str(Path(new_home).expanduser())
    for char in ("\\", '"', "$", "`"):
        value = value.replace(char, "\\" + char)
    return f'export HF_HOME="{value}"'


def _tree_stats(root: Path, forbidden: Iterable[Path] = ()) -> _TreeStats:
    stats = _TreeStats()
    blocked = [str(p) for p in forbidden]

    def fail(exc: OSError) -> None:
        raise RelocationError(f"Cannot read {exc.filename}: {exc.strerror or exc}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=fail):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError as exc:
                raise RelocationError(f"Cannot read {path}: {exc}") from exc
            if stat.S_ISLNK(st.st_mode):
                stats.symlinks += 1
                if not os.path.exists(path):
                    stats.dangling += 1
                elif blocked and any(_inside(os.path.realpath(path), b) for b in blocked):
                    stats.escaping += 1
            elif stat.S_ISREG(st.st_mode):
                stats.files += 1
                stats.bytes += st.st_size
    return stats


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _compare(src: _TreeStats, dst: _TreeStats) -> list[str]:
    problems = []
    if (src.files, src.bytes) != (dst.files, dst.bytes):
        problems.append(f"{dst.files} files / {dst.bytes} bytes copied, expected {src.files} / {src.bytes}")
    if src.symlinks != dst.symlinks:
        problems.append(f"{dst.symlinks} symlinks copied, expected {src.symlinks}")
    if dst.dangling > src.dangling:
        problems.append(f"{dst.dangling - src.dangling} symlink(s) no longer resolve")
    if dst.escaping:
        problems.append(f"{dst.escaping} symlink(s) still point into the old location")
    return problems


def _copy_tree(src: Path, dst: Path, total: int, progress: Progress | None, stop_event: StopEvent | None) -> None:
    done = 0

    def copy_file(source: str, target: str) -> str:
        nonlocal done
        _check_stop(stop_event)
        try:
            if not stat.S_ISREG(os.lstat(source).st_mode):
                return target  # sockets and FIFOs are not cache content
            with open(source, "rb") as fin, open(target, "wb") as fout:
                while chunk := fin.read(_COPY_CHUNK):
                    fout.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total, os.path.relpath(source, src))
                    _check_stop(stop_event)
            shutil.copystat(source, target, follow_symlinks=False)
        except OSError as exc:
            raise RelocationError(f"Copy failed at {source}: {exc}") from exc
        return target

    shutil.copytree(src, dst, symlinks=True, copy_function=copy_file, dirs_exist_ok=True)
    if progress is not None:
        progress(done, total, "")


def _relink_absolute(root: Path, old_roots: list[Path]) -> int:
    """Rewrite absolute symlinks that point into the old location as relative links inside ``root``."""
    olds = sorted({str(p) for p in old_roots}, key=len, reverse=True)
    fixed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            link = os.path.join(dirpath, name)
            if not os.path.islink(link):
                continue
            target = os.readlink(link)
            if not os.path.isabs(target):
                continue
            old = next((o for o in olds if _inside(target, o)), None)
            if old is None:
                continue
            new_target = os.path.relpath(os.path.join(str(root), os.path.relpath(target, old)), start=dirpath)
            temp = os.path.join(dirpath, f".{name}.relink")
            os.symlink(new_target, temp)
            os.replace(temp, link)
            fixed += 1
    return fixed


def _discard(path: Path, *, recreate: bool) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            _rmtree(path)
        if recreate:
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Could not clean up %s: %s", path, exc)


def _check_stop(stop_event: StopEvent | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise RelocationCancelled("Relocation cancelled; the original cache is untouched.")


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def _free_space(path: Path) -> int:
    return shutil.disk_usage(path).free


# ---------------------------------------------------------------------------
# Noelle data usage
# ---------------------------------------------------------------------------


def noelle_home_usage() -> dict[str, int]:
    """Byte sizes of the Noelle data folders, ``other`` (registry, settings, runtime) and ``total``."""
    home = paths.home()
    parts = {
        "models": paths.models_dir(),
        "environments": paths.environments_dir(),
        "vector_store": paths.vector_store_dir(),
        "projects": paths.projects_dir(),
        "logs": paths.logs_dir(),
    }
    seen: set[tuple[int, int]] = set()
    usage = {name: 0 if path.is_symlink() else _tree_bytes(path, seen) for name, path in parts.items()}
    root = home.resolve() if home.is_symlink() else home  # NOELLE_HOME itself may be a link; its children may not
    usage["other"] = _tree_bytes(root, seen, exclude=[root / path.name for path in parts.values()])
    usage["total"] = sum(usage.values())
    return usage


def disk_usage(path: str | os.PathLike[str]) -> dict[str, int]:
    """``{total, used, free}`` bytes of the filesystem holding ``path`` (or its nearest existing parent)."""
    usage = shutil.disk_usage(_existing_ancestor(Path(path).expanduser().absolute()))
    return {"total": int(usage.total), "used": int(usage.used), "free": int(usage.free)}
