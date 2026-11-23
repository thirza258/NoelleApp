"""Storage Advisor: active vs unused models, partial downloads, old revisions, potential cleanup."""

from __future__ import annotations

from types import SimpleNamespace

from noelle.app.services.storage_advisor import advise

DAY = 86400.0
NOW = 1_000 * DAY


def repo(repo_id: str, size: int, accessed_days_ago: float, repo_type: str = "model") -> SimpleNamespace:
    return SimpleNamespace(repo_id=repo_id, repo_type=repo_type, size_bytes=size,
                           last_accessed=NOW - accessed_days_ago * DAY, last_modified=NOW - 400 * DAY)


def record(repo_id: str, *, used_days_ago: float | None, downloaded_days_ago: float = 300,
           source: str = "hub") -> SimpleNamespace:
    return SimpleNamespace(repo_id=repo_id, source=source,
                           last_used_at=None if used_days_ago is None else NOW - used_days_ago * DAY,
                           downloaded_at=NOW - downloaded_days_ago * DAY)


def test_advice_splits_active_and_unused() -> None:
    report = SimpleNamespace(total_size=1000, incomplete_bytes=7, detached_bytes=21, repos=[
        repo("a/recent", 100, 1), repo("b/stale", 300, 200), repo("c/used-in-noelle", 200, 200),
        repo("d/never-used", 50, 1), repo("e/dataset", 999, 500, repo_type="dataset")])
    records = [record("c/used-in-noelle", used_days_ago=3), record("d/never-used", used_days_ago=None),
               record("local/build", used_days_ago=1, source="quantized")]
    advice = advise(report, records, {"models": 42}, now=NOW)
    assert advice.active_bytes == 300  # a/recent (cache access) + c (loaded 3 days ago)
    assert [u.repo_id for u in advice.unused] == ["b/stale", "d/never-used"]  # largest first
    assert advice.unused[1].in_noelle and advice.unused[1].days == 300  # downloaded long ago, never loaded
    assert advice.unused_bytes == 350 and advice.partial_bytes == 7 and advice.old_revision_bytes == 21
    assert advice.derived_bytes == 42 and advice.potential_bytes == 378 and advice.total_bytes == 1000


def test_empty_report() -> None:
    advice = advise(SimpleNamespace(repos=[], total_size=0, incomplete_bytes=0, detached_bytes=0), [], now=NOW)
    assert advice.potential_bytes == 0 and advice.unused == []
