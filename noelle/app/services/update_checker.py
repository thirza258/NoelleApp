"""Update notices for the app and downloaded Hub models.

No auto-install. Checks are skipped entirely when ``privacy.local_only`` is on, run at
most once per 24 h, and send only a generic User-Agent (plus repo ids the user already
downloaded, for the model check).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from noelle import __version__

log = logging.getLogger("noelle.updates")
CHECK_INTERVAL_S = 24 * 3600
RELEASES_URL = "https://api.github.com/repos/thirza258/NoelleApp/releases/latest"

_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    match = _SEMVER.match((text or "").strip())
    return tuple(int(part) for part in match.groups()) if match else None  # type: ignore[return-value]


def is_newer(candidate: str | None, current: str = __version__) -> bool:
    new, old = parse_version(candidate), parse_version(current)
    return bool(new and old and new > old)


@dataclass(frozen=True)
class AppUpdate:
    version: str
    url: str
    notes: str = ""
    prerelease: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "url": self.url, "notes": self.notes, "prerelease": self.prerelease}


@dataclass(frozen=True)
class ModelUpdate:
    repo_id: str
    current_sha: str | None
    latest_sha: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {"repo_id": self.repo_id, "current_sha": self.current_sha, "latest_sha": self.latest_sha,
                "url": self.url}


def check_app(*, current: str = __version__, get_json: Callable[[str], Any] | None = None,
              include_prereleases: bool = False, skipped_version: str | None = None) -> AppUpdate | None:
    """Return a newer stable release, or None. ``get_json`` is injectable for tests."""
    if get_json is None:
        def get_json(url: str) -> Any:
            import httpx

            response = httpx.get(url, headers={"User-Agent": f"Noelle/{current}", "Accept": "application/vnd.github+json"},
                                 timeout=15)
            response.raise_for_status()
            return response.json()

    try:
        data = get_json(RELEASES_URL)
    except Exception as exc:  # noqa: BLE001 - updates must never break startup
        log.debug("app update check failed: %s", exc)
        return None
    tag = str(data.get("tag_name") or "")
    prerelease = bool(data.get("prerelease") or data.get("draft"))
    if prerelease and not include_prereleases:
        return None
    if skipped_version and parse_version(skipped_version) == parse_version(tag):
        return None
    if not is_newer(tag, current):
        return None
    return AppUpdate(tag, str(data.get("html_url") or ""), str(data.get("body") or "")[:4000], prerelease)


def check_models(records: Iterable[Any], *, api: Any = None, limit: int = 200) -> list[ModelUpdate]:
    """Compare stored commit hashes with the Hub's current ``main`` sha. Read-only."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    updates: list[ModelUpdate] = []
    seen: set[str] = set()
    for record in list(records)[:limit]:
        if getattr(record, "repo_id", None) in seen:
            continue
        seen.add(getattr(record, "repo_id", ""))
        if getattr(record, "source", "hub") != "hub" or not getattr(record, "commit_hash", None):
            continue
        try:
            info = api.model_info(record.repo_id, revision=getattr(record, "revision", None) or "main")
        except Exception:  # noqa: BLE001 - one deleted/private repo must not stop the sweep
            continue
        latest = getattr(info, "sha", None)
        if latest and latest != record.commit_hash:
            updates.append(ModelUpdate(record.repo_id, record.commit_hash, latest,
                                       f"https://huggingface.co/{record.repo_id}"))
    return updates


class UpdateScheduler:
    """Throttle both checks to once per 24 h and honour ``privacy.local_only``."""

    def __init__(self, settings: Any, *, now: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.now = now

    @property
    def local_only(self) -> bool:
        return bool(self.settings.get("privacy.local_only", False))

    def should_check(self, kind: str) -> bool:
        if self.local_only:
            return False
        last = self.settings.get(f"updates.last_check.{kind}")
        return not isinstance(last, (int, float)) or self.now() - last >= CHECK_INTERVAL_S

    def mark_checked(self, kind: str) -> None:
        self.settings.set(f"updates.last_check.{kind}", self.now())

    def run(self, *, records: Iterable[Any] = (), app_check: Callable[..., AppUpdate | None] = check_app,
            model_api: Any = None) -> dict[str, Any]:
        result: dict[str, Any] = {"app": None, "models": [], "skipped": self.local_only}
        if self.local_only:
            return result
        # ``updates.check_app = None`` means "ask at first run"; we never check until the
        # user opts in (the setup assistant/Settings sets it to true).
        if self.should_check("app") and self.settings.get("updates.check_app", None) is True:
            result["app"] = app_check(current=__version__,
                                      include_prereleases=bool(self.settings.get("updates.beta", False)),
                                      skipped_version=self.settings.get("updates.skipped_version"))
            self.mark_checked("app")
        if self.should_check("models") and self.settings.get("updates.check_models", True):
            result["models"] = check_models(records, api=model_api)
            self.mark_checked("models")
        return result