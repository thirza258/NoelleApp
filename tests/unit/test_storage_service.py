"""Storage service tests.

Every cache here is a fake built under ``tmp_path`` that mimics the huggingface_hub
layout (``blobs/<etag>``, ``refs/main`` and ``snapshots/<commit>/`` relative symlinks).
The real Hugging Face cache is never touched: an autouse fixture points every
default cache location at a throwaway directory and every call passes ``cache_dir``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from noelle import paths
from noelle.app.services import storage_service as ss

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake caches use POSIX symlinks")


@pytest.fixture(autouse=True)
def _guard_real_cache(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """An accidental default-path call can only reach a throwaway directory."""
    guard = tmp_path_factory.mktemp("hf-guard")
    monkeypatch.setenv("HF_HOME", str(guard / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(guard / "hub"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(guard / "xdg"))


def commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


def data(label: str, size: int) -> bytes:
    head = f"{label}:".encode()
    return (head + b"\0" * size)[:size]


class FakeRepo:
    def __init__(self, cache: Path, repo_id: str, repo_type: str = "model") -> None:
        self.folder = cache / f"{repo_type}s--{repo_id.replace('/', '--')}"
        (self.folder / "blobs").mkdir(parents=True, exist_ok=True)
        (self.folder / "snapshots").mkdir(exist_ok=True)

    def blob(self, content: bytes) -> Path:
        path = self.folder / "blobs" / hashlib.sha256(content).hexdigest()
        path.write_bytes(content)
        return path

    def snapshot(self, commit_hash: str, files: dict[str, Path]) -> Path:
        snap = self.folder / "snapshots" / commit_hash
        snap.mkdir(parents=True, exist_ok=True)
        for name, blob in files.items():
            link = snap / name
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(blob, link.parent), link)
        return snap

    def ref(self, name: str, commit_hash: str) -> None:
        path = self.folder / "refs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commit_hash)  # no newline: huggingface_hub reads refs unstripped

    def incomplete(self, size: int, age_s: float) -> Path:
        etag = hashlib.sha256(os.urandom(16)).hexdigest()
        path = self.folder / "blobs" / f"{etag}.{os.urandom(4).hex()}.incomplete"
        path.write_bytes(b"p" * size)
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
        return path


C_OLD, C_NEW = commit("alpha-old"), commit("alpha-new")
C_BETA, C_BROKEN = commit("beta-main"), commit("broken-main")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """alpha: detached + main revision sharing blobs; beta: one revision; broken/refsonly: corrupted."""
    hub = tmp_path / "hub"
    alpha = FakeRepo(hub, "org/alpha")
    config = alpha.blob(data("config", 100))
    tokenizer = alpha.blob(data("tokenizer", 300))  # shared by both revisions
    alpha.snapshot(C_OLD, {"config.json": config, "model.safetensors": alpha.blob(data("weights-v1", 1000)),
                           "tokenizer.json": tokenizer})
    alpha.snapshot(C_NEW, {"config.json": config, "model.safetensors": alpha.blob(data("weights-v2", 2000)),
                           "tokenizer.json": tokenizer, "sub/extra.txt": alpha.blob(data("extra", 50))})
    alpha.ref("main", C_NEW)
    alpha.incomplete(700, age_s=3600)
    alpha.incomplete(400, age_s=0)

    beta = FakeRepo(hub, "org/beta")
    beta.snapshot(C_BETA, {"model.safetensors": beta.blob(data("beta", 500))})
    beta.ref("main", C_BETA)
    beta.incomplete(250, age_s=7200)

    broken = FakeRepo(hub, "org/broken")
    snap = broken.snapshot(C_BROKEN, {"config.json": broken.blob(data("broken-config", 80))})
    os.symlink("../../blobs/" + "0" * 64, snap / "model.safetensors")  # blob deleted by hand
    broken.ref("main", C_BROKEN)

    refs_only = hub / "models--org--refsonly" / "refs"  # like an aborted download: refs but no snapshot
    refs_only.mkdir(parents=True)
    (refs_only / "main").write_text(commit("gone"))

    # Non-repo entries: must be skipped, never reported as corrupted, never deleted.
    (hub / ".locks" / "models--org--alpha").mkdir(parents=True)
    (hub / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55\n")
    (hub / "blobs" / "ab").mkdir(parents=True)
    (hub / "blobs" / "ab" / "keep.bin").write_bytes(b"k" * 123)
    (hub / "version.txt").write_text("1")
    return hub


def alpha_folder(hub: Path) -> Path:
    return hub / "models--org--alpha"


def blob_of(hub: Path, label: str, size: int, repo: str = "models--org--alpha") -> Path:
    return hub / repo / "blobs" / hashlib.sha256(data(label, size)).hexdigest()


def assert_non_repo_entries_intact(hub: Path) -> None:
    assert (hub / "blobs" / "ab" / "keep.bin").read_bytes() == b"k" * 123
    assert (hub / "CACHEDIR.TAG").is_file()
    assert (hub / "version.txt").is_file()
    assert (hub / ".locks").is_dir()


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_scan_sizes_revisions_and_reclaimable(cache: Path) -> None:
    report = ss.scan_hf_cache(cache)
    assert report.scanner == "huggingface_hub"
    assert report.cache_dir == cache.resolve()
    alpha = report.repo("org/alpha")
    assert alpha is not None and not alpha.corrupted
    assert (alpha.size_bytes, alpha.nb_files) == (3450, 5)
    new, old = alpha.revisions
    assert (new.commit, new.refs, new.detached) == (C_NEW, ["main"], False)
    assert (old.commit, old.refs, old.detached) == (C_OLD, [], True)
    assert (new.size_bytes, new.nb_files, new.exclusive_bytes) == (2450, 4, 2050)
    assert (old.size_bytes, old.nb_files, old.exclusive_bytes) == (1400, 3, 1000)
    assert [f.name for f in new.files] == ["config.json", "model.safetensors", "sub/extra.txt", "tokenizer.json"]
    assert new.snapshot_path == alpha.path / "snapshots" / C_NEW
    assert all(f.blob_path.parent == alpha.path / "blobs" for f in new.files)
    assert sorted(f.size_bytes for f in alpha.incomplete) == [400, 700]
    assert all(f.repo_id == "org/alpha" and f.modified > 0 for f in alpha.incomplete)
    assert alpha.reclaimable_bytes == 1000 + 1100

    beta = report.repo("org/beta")
    assert beta is not None
    assert (beta.size_bytes, beta.reclaimable_bytes) == (500, 250)
    assert beta.revisions[0].exclusive_bytes == 500

    assert report.detached_bytes == 1000  # only weights-v1: config and tokenizer are shared with main
    assert report.incomplete_bytes == 700 + 400 + 250
    assert report.reclaimable_bytes == 1000 + 1350
    assert report.total_size == 3450 + 1100 + 500 + 250 + 80  # the unmarked top-level blobs/ is not a cache entry


def test_scan_skips_non_repo_entries(cache: Path) -> None:
    report = ss.scan_hf_cache(cache)
    assert [r.repo_id for r in report.repos] == ["org/alpha", "org/beta", "org/broken", "org/refsonly"]
    hub = cache.resolve()
    for name in ("blobs", "version.txt", "CACHEDIR.TAG", ".locks"):
        path = str(hub / name)
        assert not any(path + os.sep in w or w.rstrip(".)").endswith(path) for w in report.warnings), name
    assert_non_repo_entries_intact(cache)


def test_scan_reports_corrupted_repos_without_failing(cache: Path) -> None:
    report = ss.scan_hf_cache(cache)
    broken = report.repo("org/broken")
    assert broken is not None and broken.corrupted
    assert any("broken symlink" in w.lower() for w in broken.warnings)
    (revision,) = broken.revisions
    assert (revision.commit, revision.refs, revision.nb_files, revision.size_bytes) == (C_BROKEN, ["main"], 1, 80)
    refs_only = report.repo("org/refsonly")
    assert refs_only is not None and refs_only.corrupted and refs_only.revisions == []
    assert any("snapshots" in w.lower() for w in refs_only.warnings)
    assert not report.repo("org/alpha").corrupted  # type: ignore[union-attr]


def _summary(report: ss.CacheReport) -> dict:
    return {
        "total": report.total_size,
        "reclaimable": report.reclaimable_bytes,
        "detached": report.detached_bytes,
        "incomplete": report.incomplete_bytes,
        "repos": [
            (r.repo_id, r.repo_type, r.size_bytes, r.nb_files, r.reclaimable_bytes, r.corrupted,
             [(v.commit, v.refs, v.size_bytes, v.nb_files, v.detached, v.exclusive_bytes,
               [(f.name, f.size_bytes, f.blob_path) for f in v.files]) for v in r.revisions],
             [(f.path, f.size_bytes) for f in r.incomplete])
            for r in report.repos
        ],
    }


def test_manual_fallback_matches_hub_scan(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    expected = _summary(ss.scan_hf_cache(cache))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("simulated unreadable repo")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", boom)
    report = ss.scan_hf_cache(cache)
    assert report.scanner == "manual"
    assert any("manual scan" in w for w in report.warnings)
    assert _summary(report) == expected


def test_scan_missing_cache_returns_empty_report(tmp_path: Path) -> None:
    report = ss.scan_hf_cache(tmp_path / "nope")
    assert report.repos == [] and report.total_size == 0 and report.reclaimable_bytes == 0
    assert report.warnings and "does not exist" in report.warnings[0]


def test_report_is_json_friendly_and_mapping_like(cache: Path) -> None:
    report = ss.scan_hf_cache(cache)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["reclaimable_bytes"] == report["reclaimable_bytes"] == 2350
    assert payload["repos"][0]["revisions"][0]["files"][0]["blob_path"].endswith(
        report.repos[0].revisions[0].files[0].blob_path.name)
    with pytest.raises(KeyError):
        report["_missing"]


# ---------------------------------------------------------------------------
# Incomplete downloads
# ---------------------------------------------------------------------------


def _incomplete(hub: Path, repo: str) -> list[Path]:
    return sorted((hub / repo / "blobs").glob("*.incomplete"))


def test_prune_incomplete_age_rule(cache: Path) -> None:
    result = ss.prune_incomplete(cache)
    assert result.freed_bytes == 700 + 250
    assert len(result.removed) == 2 and not result.dry_run
    assert [p.stat().st_size for p in _incomplete(cache, "models--org--alpha")] == [400]  # fresh file kept
    assert _incomplete(cache, "models--org--beta") == []
    assert any(p.stat().st_size == 400 for p in result.skipped)
    assert_non_repo_entries_intact(cache)


def test_prune_incomplete_excludes_active_repos(cache: Path) -> None:
    result = ss.prune_incomplete(cache, exclude_repos=frozenset({"org/beta"}))
    assert result.freed_bytes == 700
    assert [p.stat().st_size for p in _incomplete(cache, "models--org--beta")] == [250]
    assert len(_incomplete(cache, "models--org--alpha")) == 1


def test_prune_incomplete_min_age_zero_and_dry_run(cache: Path) -> None:
    dry = ss.prune_incomplete(cache, min_age_s=0, dry_run=True)
    assert dry.dry_run and dry.freed_bytes == 700 + 400 + 250
    assert len(_incomplete(cache, "models--org--alpha")) == 2  # nothing deleted
    real = ss.prune_incomplete(cache, min_age_s=0)
    assert real.freed_bytes == 1350 and _incomplete(cache, "models--org--alpha") == []


def test_prune_incomplete_skips_locked_download(cache: Path) -> None:
    fcntl = pytest.importorskip("fcntl")
    old = next(p for p in _incomplete(cache, "models--org--alpha") if p.stat().st_size == 700)
    lock = cache / ".locks" / "models--org--alpha" / f"{old.name.split('.')[0]}.lock"
    lock.touch()
    with lock.open("rb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        result = ss.prune_incomplete(cache)
    assert old.exists() and old in result.skipped
    assert result.freed_bytes == 250  # beta's stale file still goes


# ---------------------------------------------------------------------------
# Revision and repo deletion
# ---------------------------------------------------------------------------


def test_delete_detached_revision_keeps_shared_blobs(cache: Path) -> None:
    dry = ss.delete_revisions([C_OLD], cache, dry_run=True)
    assert (dry.freed_bytes, dry.deleted, dry.dry_run) == (1000, [C_OLD], True)
    assert (alpha_folder(cache) / "snapshots" / C_OLD).is_dir()

    result = ss.delete_revisions([C_OLD], cache)
    assert (result.freed_bytes, result.deleted) == (1000, [C_OLD])
    assert not (alpha_folder(cache) / "snapshots" / C_OLD).exists()
    assert not blob_of(cache, "weights-v1", 1000).exists()
    for label, size in (("config", 100), ("tokenizer", 300), ("weights-v2", 2000), ("extra", 50)):
        assert blob_of(cache, label, size).exists(), label
    snap = alpha_folder(cache) / "snapshots" / C_NEW
    assert (snap / "config.json").read_bytes() == data("config", 100)
    assert (snap / "tokenizer.json").read_bytes() == data("tokenizer", 300)
    assert (alpha_folder(cache) / "refs" / "main").read_text() == C_NEW
    assert_non_repo_entries_intact(cache)


def test_delete_referenced_revision_drops_its_ref(cache: Path) -> None:
    result = ss.delete_revisions(C_NEW, cache)  # a bare string is one commit
    assert (result.freed_bytes, result.deleted) == (2050, [C_NEW])
    assert not (alpha_folder(cache) / "refs" / "main").exists()
    old = alpha_folder(cache) / "snapshots" / C_OLD
    assert (old / "model.safetensors").read_bytes() == data("weights-v1", 1000)
    assert (old / "tokenizer.json").read_bytes() == data("tokenizer", 300)


def test_delete_every_revision_removes_the_repo_folder(cache: Path) -> None:
    result = ss.delete_revisions([C_OLD, C_NEW], cache)
    assert sorted(result.deleted) == sorted([C_OLD, C_NEW])
    assert result.freed_bytes == 3450 + 1100  # blobs plus the partial downloads inside the folder
    assert not alpha_folder(cache).exists()
    assert (cache / "models--org--beta").is_dir()
    assert_non_repo_entries_intact(cache)


def test_delete_unknown_commit_is_a_warning(cache: Path) -> None:
    result = ss.delete_revisions(["f" * 40], cache)
    assert (result.freed_bytes, result.deleted) == (0, [])
    assert result.warnings and "f" * 40 in result.warnings[0]


def test_same_commit_in_two_repos_is_scoped(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    shared, main_a, main_b = commit("fork-point"), commit("a-main"), commit("b-main")
    for name, main in (("org/a", main_a), ("org/b", main_b)):
        repo = FakeRepo(hub, name)
        repo.snapshot(shared, {"w.bin": repo.blob(data(f"{name}-old", 600))})
        repo.snapshot(main, {"w.bin": repo.blob(data(f"{name}-new", 900))})
        repo.ref("main", main)
    combined = ss.prune_detached(hub, dry_run=True)  # same hash in both repos: planned per repo
    assert (combined.freed_bytes, combined.deleted) == (1200, [shared])
    scoped = ss.delete_revisions([shared], hub, repo_id="org/a")
    assert (scoped.freed_bytes, scoped.deleted) == (600, [shared])
    assert not (hub / "models--org--a" / "snapshots" / shared).exists()
    assert (hub / "models--org--b" / "snapshots" / shared).is_dir()
    both = ss.prune_detached(hub)
    assert both.freed_bytes == 600 and not (hub / "models--org--b" / "snapshots" / shared).exists()
    assert (hub / "models--org--a" / "snapshots" / main_a / "w.bin").read_bytes() == data("org/a-new", 900)


def test_prune_detached(cache: Path) -> None:
    dry = ss.prune_detached(cache, dry_run=True)
    assert (dry.freed_bytes, dry.deleted) == (1000, [C_OLD])
    result = ss.prune_detached(cache)
    assert (result.freed_bytes, result.deleted) == (1000, [C_OLD])
    assert (cache / "models--org--beta" / "snapshots" / C_BETA).is_dir()
    assert (cache / "models--org--broken" / "snapshots" / C_BROKEN).is_dir()
    assert (cache / "models--org--refsonly" / "refs" / "main").is_file()
    assert_non_repo_entries_intact(cache)
    again = ss.prune_detached(cache)
    assert (again.freed_bytes, again.deleted) == (0, [])


def test_prune_detached_skips_excluded_repos(cache: Path) -> None:
    result = ss.prune_detached(cache, exclude_repos={"org/alpha"})
    assert (result.freed_bytes, result.deleted) == (0, [])
    assert (alpha_folder(cache) / "snapshots" / C_OLD).is_dir()


def test_prune_detached_never_empties_an_active_first_download(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    repo = FakeRepo(hub, "org/downloading")  # first download: snapshot exists, refs/main not written yet
    snap = repo.snapshot(commit("in-flight"), {"config.json": repo.blob(data("cfg", 10))})
    partial = repo.incomplete(4096, age_s=0)
    assert ss.prune_detached(hub, exclude_repos=["org/downloading"]).deleted == []
    bare = ss.prune_detached(hub)  # caller forgot exclude_repos: the activity guard still holds
    assert bare.deleted == [] and any("org/downloading" in w for w in bare.warnings)
    assert partial.exists() and snap.is_dir()
    assert ss.scan_hf_cache(hub).detached_bytes == 0

    stamp = time.time() - 3600  # the download died an hour ago: now it is prunable
    for path in (partial, snap):
        os.utime(path, (stamp, stamp))
    stale = ss.prune_detached(hub)
    assert stale.deleted == [commit("in-flight")] and not repo.folder.exists()


def test_prune_detached_respects_a_held_download_lock(tmp_path: Path) -> None:
    fcntl = pytest.importorskip("fcntl")
    hub = tmp_path / "hub"
    repo = FakeRepo(hub, "org/stalled")
    snap = repo.snapshot(commit("stalled"), {"config.json": repo.blob(data("cfg", 10))})
    partial = repo.incomplete(4096, age_s=3600)
    stamp = time.time() - 3600
    os.utime(snap, (stamp, stamp))
    lock = hub / ".locks" / repo.folder.name / f"{partial.name.split('.')[0]}.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    with lock.open("rb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        assert ss.prune_detached(hub).deleted == []
    assert partial.exists()


def test_delete_repo(cache: Path) -> None:
    assert ss.delete_repo("org/beta", cache, dry_run=True) == 750
    assert (cache / "models--org--beta").is_dir()
    assert ss.delete_repo("org/beta", cache) == 750
    assert not (cache / "models--org--beta").exists()
    assert alpha_folder(cache).is_dir()
    assert_non_repo_entries_intact(cache)


def test_delete_corrupted_and_missing_repos(cache: Path) -> None:
    assert ss.delete_repo("org/broken", cache) == 80
    assert not (cache / "models--org--broken").exists()
    assert ss.delete_repo("org/refsonly", cache) == 0
    assert not (cache / "models--org--refsonly").exists()
    assert ss.delete_repo("org/missing", cache) == 0
    assert ss.delete_repo("blobs", cache) == 0  # never maps onto the top-level store
    assert_non_repo_entries_intact(cache)


@pytest.mark.parametrize("repo_id,repo_type", [("", "model"), ("org\\x", "model"), ("org/", "model"),
                                               ("org/x", "weird")])
def test_delete_repo_rejects_bad_ids(cache: Path, repo_id: str, repo_type: str) -> None:
    with pytest.raises(ValueError):
        ss.delete_repo(repo_id, cache, repo_type=repo_type)


# ---------------------------------------------------------------------------
# Shared (Xet) blob store
# ---------------------------------------------------------------------------


C_ONE_OLD, C_ONE_NEW, C_TWO = commit("one-old"), commit("one-new"), commit("two-main")


@pytest.fixture
def shared_cache(tmp_path: Path) -> tuple[Path, Path]:
    return build_shared_hub(tmp_path / "hub")


def build_shared_hub(hub: Path) -> tuple[Path, Path]:
    """One payload in the shared store, linked from a detached revision of 'one' and from 'two' main."""
    store = hub / "blobs"
    store.mkdir(parents=True)
    (store / ".huggingface-shared-blobs").write_text("1\n")
    xet = hashlib.sha256(b"shared-weights").hexdigest()
    payload = store / xet[:2] / xet
    payload.parent.mkdir()
    payload.write_bytes(data("shared", 5000))
    payload.chmod(0o444)
    one, two = FakeRepo(hub, "org/one"), FakeRepo(hub, "org/two")
    link_one, link_two = one.folder / "blobs" / ("a" * 64), two.folder / "blobs" / ("b" * 64)
    for link in (link_one, link_two):
        os.symlink(os.path.relpath(payload, link.parent), link)
    (payload.parent / f"{xet}.refs").write_text(f"models--org--one/blobs/{'a' * 64}\nmodels--org--two/blobs/{'b' * 64}\n")
    config = one.blob(data("one-config", 100))
    one.snapshot(C_ONE_OLD, {"config.json": config, "model.safetensors": link_one})
    one.snapshot(C_ONE_NEW, {"config.json": config})
    one.ref("main", C_ONE_NEW)
    two.snapshot(C_TWO, {"model.safetensors": link_two})
    two.ref("main", C_TWO)
    return hub, payload


def test_shared_payload_counted_once_and_not_reclaimable(shared_cache: tuple[Path, Path]) -> None:
    from huggingface_hub import scan_cache_dir

    hub, _payload = shared_cache
    report = ss.scan_hf_cache(hub)
    assert [r.repo_id for r in report.repos] == ["org/one", "org/two"]  # the store is not a repo
    assert not any("blobs" in w and "models--" not in w for w in report.warnings)
    one, two = report.repo("org/one"), report.repo("org/two")
    assert one is not None and two is not None
    assert (one.size_bytes, two.size_bytes) == (5100, 5000)  # each repo sees the payload it uses
    assert report.total_size == 5100  # ...but the disk holds it once
    old = next(r for r in one.revisions if r.detached)
    assert old.exclusive_bytes == 0 and report.reclaimable_bytes == 0  # still used by org/two
    oracle = scan_cache_dir(hub).delete_revisions(C_ONE_OLD).expected_freed_size
    assert old.exclusive_bytes == oracle


def test_shared_payload_kept_while_referenced_then_collected(shared_cache: tuple[Path, Path]) -> None:
    hub, payload = shared_cache
    result = ss.prune_detached(hub)
    assert (result.freed_bytes, result.deleted) == (0, [C_ONE_OLD])
    assert payload.read_bytes() == data("shared", 5000)
    assert (hub / "models--org--two" / "snapshots" / C_TWO / "model.safetensors").read_bytes() == data("shared", 5000)
    assert ss.delete_repo("org/two", hub) == 5000  # last reference gone: the store payload is collected
    assert not payload.exists()
    assert (hub / "blobs" / ".huggingface-shared-blobs").is_file()
    assert (hub / "models--org--one" / "snapshots" / C_ONE_NEW / "config.json").read_bytes() == data("one-config", 100)


def test_hard_linked_blob_counted_once(shared_cache: tuple[Path, Path]) -> None:
    hub, payload = shared_cache
    three = FakeRepo(hub, "org/three")
    copy = three.folder / "blobs" / ("c" * 64)
    os.link(payload, copy)  # same inode as the store payload
    three.snapshot(commit("three-main"), {"model.safetensors": copy})
    three.ref("main", commit("three-main"))
    report = ss.scan_hf_cache(hub)
    assert report.repo("org/three").size_bytes == 5000  # type: ignore[union-attr]
    assert report.total_size == 5100
    assert ss.delete_repo("org/two", hub) == 0  # the store copy goes, but org/three still holds the bytes
    assert copy.read_bytes() == data("shared", 5000)


def test_relocate_cache_with_shared_store(tmp_path: Path) -> None:
    home = tmp_path / "hf_home"
    hub, payload = build_shared_hub(home / "hub")
    (home / "token").write_text("not-a-real-token")
    dst = tmp_path / "relocated"
    result = ss.relocate_cache(home, dst, force_copy=True)
    assert result.method == "copy" and not home.exists()
    report = ss.scan_hf_cache(dst / "hub")
    assert report.total_size == 5100 and [r.repo_id for r in report.repos] == ["org/one", "org/two"]
    moved = dst / "hub" / payload.relative_to(home / "hub")
    assert moved.read_bytes() == data("shared", 5000) and not os.access(moved, os.W_OK)  # read-only kept
    for repo, rev in (("models--org--one", C_ONE_OLD), ("models--org--two", C_TWO)):
        link = dst / "hub" / repo / "snapshots" / rev / "model.safetensors"
        assert link.read_bytes() == data("shared", 5000)


# ---------------------------------------------------------------------------
# Relocation
# ---------------------------------------------------------------------------

C_HOME = commit("home-main")


def make_hf_home(root: Path) -> Path:
    home = root / "hf_home"
    repo = FakeRepo(home / "hub", "org/model")
    config = repo.blob(data("cfg", 64))
    weights = repo.blob(data("weights", 3 * 1024 * 1024 + 7))
    snap = repo.snapshot(C_HOME, {"config.json": config, "nested/model.safetensors": weights})
    repo.ref("main", C_HOME)
    os.symlink(str(config), snap / "absolute.json")  # absolute link into the old location
    (home / "token").write_text("not-a-real-token")
    (home / "xet").mkdir()
    (home / "xet" / "chunk.bin").write_bytes(b"c" * 2048)
    return home


def tree(root: Path) -> dict[str, tuple[str, int | str]]:
    """Relative path -> ("file", size) | ("link", target); directories omitted."""
    out: dict[str, tuple[str, int | str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath, name)
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                out[rel] = ("link", os.readlink(path))
            elif path.is_file():
                out[rel] = ("file", path.stat().st_size)
    return out


def home_bytes(home: Path) -> int:
    return sum(size for kind, size in tree(home).values() if kind == "file")  # type: ignore[misc]


def assert_relocated(dst: Path) -> None:
    snap = dst / "hub" / "models--org--model" / "snapshots" / C_HOME
    for name in ("config.json", "nested/model.safetensors", "absolute.json"):
        link = snap / name
        assert link.is_symlink(), name
        assert not os.path.isabs(os.readlink(link)), name  # relative links survive any move
        assert link.resolve().is_relative_to(dst.resolve()), name
    assert (snap / "config.json").read_bytes() == data("cfg", 64)
    assert (snap / "absolute.json").read_bytes() == data("cfg", 64)
    assert (snap / "nested" / "model.safetensors").read_bytes() == data("weights", 3 * 1024 * 1024 + 7)
    assert (dst / "token").read_text() == "not-a-real-token"
    report = ss.scan_hf_cache(dst / "hub")
    assert report.repo("org/model") is not None and not report.repo("org/model").corrupted  # type: ignore[union-attr]


def test_relocate_rename_same_filesystem(tmp_path: Path) -> None:
    src = make_hf_home(tmp_path)
    total = home_bytes(src)
    calls: list[tuple[int, int, str]] = []
    dst = tmp_path / "moved" / "hf_home"
    result = ss.relocate_cache(src, dst, progress=lambda *a: calls.append(a))
    assert (result.method, result.moved_bytes, result.new_hf_home) == ("rename", total, dst)
    assert result["method"] == "rename" and result.relinked == 1
    assert not src.exists()
    assert calls[-1][:2] == (total, total)
    assert_relocated(dst)


def test_relocate_force_copy_preserves_symlinks(tmp_path: Path) -> None:
    src = make_hf_home(tmp_path)
    before = tree(src)
    total = home_bytes(src)
    calls: list[tuple[int, int, str]] = []
    dst = tmp_path / "copied"
    result = ss.relocate_cache(src, dst, progress=lambda *a: calls.append(a), force_copy=True)
    assert (result.method, result.moved_bytes, result.files) == ("copy", total, 5)
    assert not src.exists()
    done = [c[0] for c in calls]
    assert done == sorted(done) and done[-1] == total and all(c[1] == total for c in calls)
    after = tree(dst)
    assert {k: v for k, v in after.items() if not k.endswith("absolute.json")} == \
        {k: v for k, v in before.items() if not k.endswith("absolute.json")}
    assert_relocated(dst)


def test_relocate_falls_back_to_copy_across_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_hf_home(tmp_path)

    def cross_device(*_args: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(ss.os, "rename", cross_device)
    result = ss.relocate_cache(src, tmp_path / "other-disk")
    monkeypatch.undo()
    assert result.method == "copy" and not src.exists()
    assert_relocated(tmp_path / "other-disk")


def test_relocate_into_empty_existing_directory(tmp_path: Path) -> None:
    src = make_hf_home(tmp_path)
    dst = tmp_path / "empty"
    dst.mkdir()
    assert ss.relocate_cache(src, dst).method == "rename"
    assert_relocated(dst)


def test_relocate_refusals(tmp_path: Path) -> None:
    src = make_hf_home(tmp_path)
    snapshot = tree(src)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x")
    a_file = tmp_path / "a-file"
    a_file.write_text("x")
    link = tmp_path / "link-home"
    link.symlink_to(src)
    cases = [(src, src / "hub" / "inner"), (src, src), (src, busy), (src, a_file),
             (tmp_path / "missing", tmp_path / "x"), (link, tmp_path / "y")]
    for source, target in cases:
        with pytest.raises(ss.RelocationError):
            ss.relocate_cache(source, target)
    assert tree(src) == snapshot
    assert (busy / "file").read_text() == "x"


def test_relocate_refuses_without_free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_hf_home(tmp_path)
    monkeypatch.setattr(ss, "_free_space", lambda _path: 1024)
    with pytest.raises(ss.RelocationError, match="free space"):
        ss.relocate_cache(src, tmp_path / "small-disk", force_copy=True)
    assert src.is_dir() and not (tmp_path / "small-disk").exists()


def test_relocate_cancel_removes_partial_copy(tmp_path: Path) -> None:
    src = make_hf_home(tmp_path)
    snapshot = tree(src)
    stop = threading.Event()
    dst = tmp_path / "partial"
    with pytest.raises(ss.RelocationCancelled):
        ss.relocate_cache(src, dst, progress=lambda *_a: stop.set(), stop_event=stop, force_copy=True)
    assert tree(src) == snapshot
    assert not dst.exists()


def test_shell_export_line_round_trips_through_bash(tmp_path: Path) -> None:
    assert ss.shell_export_line("/Volumes/My Disk/hf") == 'export HF_HOME="/Volumes/My Disk/hf"'
    assert ss.shell_export_line("~/hf") == f'export HF_HOME="{Path.home()}/hf"'
    tricky = str(tmp_path / 'we"ird $HOME `id` back\\slash')
    line = ss.shell_export_line(tricky)
    out = subprocess.run(["bash", "-c", f'{line}\nprintf "%s" "$HF_HOME"'], capture_output=True, text=True, check=True)
    assert out.stdout == tricky


# ---------------------------------------------------------------------------
# Noelle data usage
# ---------------------------------------------------------------------------


def test_noelle_home_usage(tmp_path: Path) -> None:
    home = paths.ensure_dirs()
    (paths.models_dir() / "m.bin").write_bytes(b"m" * 1000)
    os.link(paths.models_dir() / "m.bin", paths.projects_dir() / "same-inode.bin")  # counted once
    venv = paths.environments_dir() / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib.so").write_bytes(b"e" * 300)
    (venv / "bin" / "python").symlink_to(sys.executable)  # never followed out of the tree
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "huge.bin").write_bytes(b"h" * 50_000)
    (paths.logs_dir() / "external").symlink_to(outside, target_is_directory=True)
    (paths.logs_dir() / "app.log").write_bytes(b"l" * 50)
    (paths.vector_store_dir() / "index.db").write_bytes(b"v" * 200)
    (home / "registry.db").write_bytes(b"r" * 70)
    usage = ss.noelle_home_usage()
    assert usage == {"models": 1000, "environments": 300, "vector_store": 200, "projects": 0, "logs": 50,
                     "other": 70, "total": 1620}


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_usage_tolerates_permission_errors() -> None:
    paths.ensure_dirs()
    locked = paths.logs_dir() / "locked"
    locked.mkdir()
    (locked / "secret.log").write_bytes(b"s" * 10)
    (paths.logs_dir() / "open.log").write_bytes(b"o" * 5)
    locked.chmod(0)
    try:
        usage = ss.noelle_home_usage()
    finally:
        locked.chmod(0o755)
    assert usage["logs"] == 5


def test_disk_usage(tmp_path: Path) -> None:
    usage = ss.disk_usage(tmp_path / "not" / "created" / "yet")
    assert set(usage) == {"total", "used", "free"}
    assert usage["total"] >= usage["used"] >= 0 and usage["free"] > 0
