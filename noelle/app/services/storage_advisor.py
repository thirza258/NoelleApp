"""Storage Advisor: where the disk space goes and what can safely be cleaned up (Qt-free)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

UNUSED_DAYS = 90


@dataclass
class UnusedModel:
    repo_id: str
    size_bytes: int
    days: int
    in_noelle: bool


@dataclass
class StorageAdvice:
    total_bytes: int
    active_bytes: int
    unused_bytes: int
    partial_bytes: int
    old_revision_bytes: int
    derived_bytes: int
    unused: list[UnusedModel] = field(default_factory=list)
    unused_days: int = UNUSED_DAYS

    @property
    def potential_bytes(self) -> int:
        return self.unused_bytes + self.partial_bytes + self.old_revision_bytes


def advise(report: Any, records: Iterable[Any], usage: dict[str, int] | None = None, *, now: float | None = None,
           unused_days: int = UNUSED_DAYS) -> StorageAdvice:
    """``report`` is a storage_service.CacheReport; ``records`` are registry ModelRecords.

    A model's last use is when Noelle last loaded it, else when it was downloaded, else
    the cache's own access time. Models used within ``unused_days`` count as active.
    """
    now = now or time.time()
    by_repo: dict[str, Any] = {}
    for record in records:
        if getattr(record, "source", "hub") == "hub":
            by_repo[record.repo_id] = record
    active = unused_bytes = 0
    unused: list[UnusedModel] = []
    for repo in getattr(report, "repos", []):
        if getattr(repo, "repo_type", "model") != "model":
            continue
        record = by_repo.get(repo.repo_id)
        last = None
        if record is not None:
            last = record.last_used_at or record.downloaded_at
        last = last or repo.last_accessed or repo.last_modified
        days = int((now - last) / 86400) if last else unused_days
        if days >= unused_days:
            unused_bytes += repo.size_bytes
            unused.append(UnusedModel(repo.repo_id, repo.size_bytes, days, record is not None))
        else:
            active += repo.size_bytes
    unused.sort(key=lambda u: u.size_bytes, reverse=True)
    return StorageAdvice(total_bytes=int(getattr(report, "total_size", 0)), active_bytes=active,
                         unused_bytes=unused_bytes, partial_bytes=int(getattr(report, "incomplete_bytes", 0)),
                         old_revision_bytes=int(getattr(report, "detached_bytes", 0)),
                         derived_bytes=int((usage or {}).get("models", 0)), unused=unused, unused_days=unused_days)
