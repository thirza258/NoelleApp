"""Shared fixtures.

Safety rules for every test:
* ``NOELLE_HOME`` always points at a temporary directory.
* The OS keychain is replaced by an in-memory keyring backend.
* Tests never prune or delete anything in the real Hugging Face cache; storage
  tests build fake caches under ``tmp_path``. Integration tests may only *read*
  models that are already cached and skip when they are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(autouse=True)
def isolated_noelle_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("noelle-home")
    monkeypatch.setenv("NOELLE_HOME", str(home))
    return home


class _MemoryKeyring:
    priority = 1

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


@pytest.fixture(autouse=True)
def memory_keyring(monkeypatch: pytest.MonkeyPatch) -> _MemoryKeyring:
    import keyring

    backend = _MemoryKeyring()
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    return backend


def cached_model_path(repo_id: str) -> Path | None:
    from noelle.runtime.hub_cache import cached_snapshot_path

    return cached_snapshot_path(repo_id)


@pytest.fixture(scope="session")
def text_model_path() -> Path:
    path = cached_model_path(TEXT_MODEL)
    if path is None:
        pytest.skip(f"{TEXT_MODEL} is not in the local Hugging Face cache")
    return path


@pytest.fixture(scope="session")
def worker_process(tmp_path_factory: pytest.TempPathFactory):
    """One real worker for the integration suite (reads models from the real HF cache)."""
    if cached_model_path(TEXT_MODEL) is None:
        pytest.skip(f"{TEXT_MODEL} is not in the local Hugging Face cache")
    from noelle.runtime.client import spawn_worker

    home = tmp_path_factory.mktemp("worker-home")
    old = os.environ.get("NOELLE_HOME")
    os.environ["NOELLE_HOME"] = str(home)
    try:
        proc = spawn_worker(parent_pid=os.getpid())
    finally:
        if old is None:
            os.environ.pop("NOELLE_HOME", None)
        else:
            os.environ["NOELLE_HOME"] = old
    yield proc
    proc.stop()
