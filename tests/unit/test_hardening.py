"""Regression tests for production-hardening fixes (corrupt state, IPC failures, child processes)."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QWidget

from noelle.app.ipc.runtime_client import RequestHandle, RuntimeClient
from noelle.app.services import server_manager as sm
from noelle.app.services.download_manager import DownloadJob, DownloadManager
from noelle.app.ui.widgets import jupyter_bridge as jb
from noelle.app.services.model_registry import ModelRecord, ModelRegistry
from noelle.config import settings as settings_module
from noelle.config.settings import Settings
from noelle.runtime import client as client_module
from noelle.runtime import devices
from noelle.runtime import tasks
from noelle.runtime.downloader import RepoFile, is_cached
from noelle.runtime.model_manager import LoadedModel, ModelManager, ModelSpec
from noelle.runtime.protocol import Command, GenerationParams, parse_ready_port


# =============================================================== settings
def test_settings_write_is_fsynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[int] = []
    monkeypatch.setattr(settings_module.os, "fsync", lambda fd: synced.append(fd))
    settings = Settings(tmp_path / "settings.json")
    settings.set("server.port", 9999)
    assert synced, "settings must be flushed to disk before the atomic rename"
    assert Settings(tmp_path / "settings.json").get("server.port") == 9999


def test_corrupt_settings_are_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="noelle.settings"):
        assert Settings(path).get("server.port") == 8765
    assert any("unreadable settings file" in record.message for record in caplog.records)


def test_unhandled_slot_exceptions_are_logged(monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
    from noelle.app import main as app_main

    original = app_main.sys.excepthook
    monkeypatch.setattr(app_main.sys, "__excepthook__", lambda *_args: None)
    try:
        app_main._install_excepthook()
        with caplog.at_level(logging.CRITICAL, logger="noelle"):
            app_main.sys.excepthook(ValueError, ValueError("slot blew up"), None)
    finally:
        app_main.sys.excepthook = original
    record = next(r for r in caplog.records if r.message == "Unhandled exception")
    assert record.exc_info is not None and "slot blew up" in str(record.exc_info[1])


def test_stale_installer_token_handoffs_are_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from noelle.app import main as app_main

    appdata = tmp_path / "appdata"
    folder = appdata / "Noelle"
    folder.mkdir(parents=True)
    stale = folder / "pending-hf-token-20260101120000.txt"
    stale.write_text("hf_secret", encoding="utf-8")
    keep = folder / "notes.txt"
    keep.write_text("keep me", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    app_main._clean_stale_token_handoffs()
    assert not stale.exists() and keep.exists()


def test_unwritable_data_folder_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    from noelle.app import main as app_main

    def deny() -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(app_main.paths, "ensure_dirs", deny)
    problem = app_main._prepare_storage()
    assert isinstance(problem, PermissionError)


def test_import_hf_token_accepts_utf16_handoff(tmp_path: Path) -> None:
    from noelle.__main__ import import_hf_token
    from noelle.app.services import hf_client

    handoff = tmp_path / "token.txt"
    handoff.write_bytes(b"\xff\xfe" + "hf_fromUtf16".encode("utf-16-le"))
    assert import_hf_token(str(handoff)) == 0
    assert hf_client.get_token() == "hf_fromUtf16"
    assert not handoff.exists()


def test_import_hf_token_rejects_embedded_whitespace(tmp_path: Path) -> None:
    from noelle.__main__ import import_hf_token
    from noelle.app.services import hf_client

    hf_client.clear_token()
    handoff = tmp_path / "token.txt"
    handoff.write_text("hf_bad token\n", encoding="utf-8")
    assert import_hf_token(str(handoff)) == 1
    assert hf_client.get_token() is None and not handoff.exists()


# =============================================================== logging
def test_desktop_logging_creates_component_files_and_redacts() -> None:
    from noelle import logging_setup, paths

    token = "hf_" + "b" * 34
    logging_setup.configure_desktop_logging()
    try:
        logging.getLogger("noelle.downloads").info("downloading with %s", token)
        logging.getLogger("noelle.credentials").warning("token stored")
        try:
            raise RuntimeError(f"request failed for {token}")
        except RuntimeError:
            logging.getLogger("noelle.downloads").exception("download failed")
        logs = paths.logs_dir()
        assert (logs / "downloads.log").exists()
        assert (logs / "credentials.log").exists()
        downloads = (logs / "downloads.log").read_text(encoding="utf-8")
        credentials = (logs / "credentials.log").read_text(encoding="utf-8")
        assert "hf_****" in downloads and token not in downloads  # message + traceback redacted
        assert "token stored" in credentials
    finally:
        logging_setup.reset_logging()


def test_process_logging_is_idempotent(tmp_path: Path) -> None:
    from noelle import logging_setup

    logging_setup.configure_process_logging("unit-test", console=False)
    try:
        logging_setup.configure_process_logging("unit-test", console=False)  # reconfigure must not duplicate
        logging.getLogger("noelle.worker").info("one line")
        content = (logging_setup.paths.logs_dir() / "unit-test.log").read_text(encoding="utf-8")
        assert content.count("one line") == 1
    finally:
        logging_setup.reset_logging()


def test_verbose_logging_adds_debug_file() -> None:
    from noelle import logging_setup, paths

    os.environ["NOELLE_LOG_DEBUG"] = "1"
    try:
        logging_setup.configure_desktop_logging()
        logging.getLogger("noelle.ui").debug("verbose detail")
        assert (paths.logs_dir() / "debug.log").read_text(encoding="utf-8").count("verbose detail") == 1
    finally:
        os.environ.pop("NOELLE_LOG_DEBUG", None)
        logging_setup.reset_logging()


# =============================================================== registry
def test_corrupt_registry_is_quarantined_and_recreated(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    path.write_bytes(b"this is not a sqlite database" * 40)
    registry = ModelRegistry(path)
    assert registry.recovered_path is not None
    assert registry.recovered_path.exists()
    assert registry.recovered_path.read_bytes().startswith(b"this is not")
    assert registry.list_models() == []
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    assert registry.upsert_model(ModelRecord("org/model", "text-generation", str(snapshot))).repo_id == "org/model"


# =============================================================== secrets
def test_tokens_are_redacted_from_download_events(capsys: pytest.CaptureFixture[str]) -> None:
    from noelle.runtime import downloader
    from noelle.runtime.errors import redact_secrets

    token = "hf_" + "a" * 34
    assert redact_secrets(f"failed with {token}") == "failed with hf_****"
    assert redact_secrets("hf_short") == "hf_short"
    assert "sk-****" in redact_secrets("key sk-" + "x" * 30)
    assert "Bearer ****" in redact_secrets("Authorization: Bearer " + "y" * 30)
    assert redact_secrets("a bearer of good news") == "a bearer of good news"
    downloader.emit({"type": "error", "message": f"auth failed for {token}"})
    out = capsys.readouterr().out
    assert token not in out and "hf_****" in out


def test_credential_store_falls_back_to_the_platform_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring
    import keyring.core
    from keyring.backends.fail import Keyring as FailKeyring

    from noelle.app.services import hf_client

    hf_client._reset_backend_cache()
    monkeypatch.setattr(keyring, "get_keyring", lambda: FailKeyring())
    loaded: list[str] = []
    installed: list[Any] = []
    fake = object()

    def load(name: str) -> Any:
        loaded.append(name)
        return fake

    monkeypatch.setattr(keyring.core, "load_keyring", load)
    monkeypatch.setattr(keyring.core, "set_keyring", installed.append)
    try:
        hf_client._ensure_backend()
    finally:
        hf_client._reset_backend_cache()
    assert installed == [fake] and loaded  # the explicit platform backend wins over "fail"


def test_set_token_fails_when_the_store_silently_drops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring

    from noelle.app.services import hf_client

    hf_client._reset_backend_cache()
    monkeypatch.setattr(hf_client, "_ensure_backend", lambda: None)
    monkeypatch.setattr(keyring, "set_password", lambda *_a: None)
    monkeypatch.setattr(keyring, "get_password", lambda *_a: None)
    with pytest.raises(RuntimeError, match="did not keep"):
        hf_client.set_token("hf_willVanish")
    with pytest.raises(ValueError, match="empty"):
        hf_client.set_token("   ")


# =============================================================== download integrity
def test_truncated_blob_is_not_treated_as_cached(tmp_path: Path) -> None:
    blobs = tmp_path / "models--org--m" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc123").write_bytes(b"partial")
    entry = RepoFile("model.safetensors", 10, "abc123", None)
    assert not is_cached(entry, cache_dir=tmp_path, repo_id="org/m")
    (blobs / "abc123").write_bytes(b"complete!!")  # exactly 10 bytes
    assert is_cached(entry, cache_dir=tmp_path, repo_id="org/m")


# =============================================================== model lifecycle
def test_unload_survives_a_failing_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed: list[bool] = []
    manager = ModelManager("cpu", on_change=lambda: changed.append(True))
    monkeypatch.setattr(devices, "empty_cache", lambda device=None: None)

    class BrokenEngine:
        def release(self) -> None:
            raise RuntimeError("native teardown failed")

    loaded = LoadedModel(key="k", spec=ModelSpec("org/model"), family=tasks.TEXT, device="cpu",
                         engine=BrokenEngine(), engine_name="fake", source=str(tmp_path), memory_bytes=1,
                         estimated_bytes=1, load_ms=0.0)
    manager._models["k"] = loaded
    assert manager.unload("k") is True
    assert manager.get("k") is None and loaded.engine is None and changed


# =============================================================== download child
class _ErrorSignal:
    def __init__(self) -> None:
        self.slot: Any = None

    def connect(self, slot: Any) -> None:
        self.slot = slot

    def emit(self, error: Any) -> None:
        if self.slot:
            self.slot(error)


class _Proc:
    class ProcessState:
        NotRunning = 0

    def __init__(self, *, running: bool = True) -> None:
        self._running = running
        self.terminated = False
        self.killed = False
        self.waited = 0
        self.errorOccurred = _ErrorSignal()

    def state(self) -> int:
        return 1 if self._running else 0

    def terminate(self) -> None:
        self.terminated = True
        self._running = False

    def kill(self) -> None:
        self.killed = True
        self._running = False

    def waitForFinished(self, _ms: int) -> bool:  # noqa: N802
        self.waited += 1
        return True

    def errorString(self) -> str:  # noqa: N802
        return "The executable was not found"

    def deleteLater(self) -> None:  # noqa: N802
        pass

    def readAllStandardOutput(self) -> bytes:  # noqa: N802
        return b""

    def readAllStandardError(self) -> bytes:  # noqa: N802
        return b""


@pytest.fixture()
def download_manager(qapp: Any, tmp_path: Path) -> DownloadManager:
    return DownloadManager(Settings(tmp_path / "s.json"), ModelRegistry(tmp_path / "r.db"))


def test_pause_does_not_block_the_gui_thread(download_manager: DownloadManager) -> None:
    job = DownloadJob("org/m", status="downloading")
    download_manager.jobs["org/m"] = job
    process = _Proc()
    download_manager._processes["org/m"] = process
    download_manager.pause("org/m")
    assert process.terminated and not process.waited  # escalation is a timer, not a wait loop
    assert job.status == "paused"


def test_failed_child_start_becomes_an_actionable_error(download_manager: DownloadManager) -> None:
    job = DownloadJob("org/m", status="planning")
    download_manager.jobs["org/m"] = job
    download_manager._processes["org/m"] = _Proc(running=False)
    finished: list[tuple[str, bool]] = []
    download_manager.job_finished.connect(lambda repo, ok: finished.append((repo, ok)))
    download_manager._on_process_error("org/m", QProcess.ProcessError.FailedToStart)
    assert job.status == "error" and job.error_code == "PROCESS_START"
    assert "could not start" in (job.error or "")
    assert finished == [("org/m", False)]
    assert "org/m" not in download_manager._processes
    # A late `finished` for the same child must not produce a second toast.
    download_manager._on_finished("org/m", 0)
    assert finished == [("org/m", False)]


def test_malformed_child_events_are_ignored(download_manager: DownloadManager) -> None:
    job = DownloadJob("org/m", status="downloading")
    download_manager.jobs["org/m"] = job
    download_manager._apply("org/m", ["not", "a", "dict"])  # type: ignore[arg-type]
    download_manager._apply("org/m", {"type": "plan", "total_bytes": "many",
                                      "files": [None, {"name": 3}, {"name": "a.bin", "size": "huge",
                                                                    "cached": "yes"}]})
    assert job.total_bytes == 0
    assert set(job.files) == {"a.bin"} and job.files["a.bin"].size is None
    job.status = "paused"
    download_manager._apply("org/m", {"type": "progress", "file": "a.bin", "downloaded": 999})
    assert job.status == "paused" and job.files["a.bin"].downloaded == 0  # stale progress is ignored
    download_manager._apply("org/m", {"type": "done", "path": "/tmp/x", "files": [1, 2]})
    assert job.status == "done"  # a genuinely completed child is still recorded


def test_output_from_a_replaced_child_does_not_touch_the_new_job(download_manager: DownloadManager,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    old = DownloadJob("org/m", status="downloading")
    new = DownloadJob("org/m", status="queued")
    download_manager.jobs["org/m"] = new
    process = _Proc()
    process.readAllStandardOutput = lambda: (  # type: ignore[method-assign]
        b'{"type":"done","path":"/tmp/old","files":["model.safetensors"]}\n')
    download_manager._processes["org/m"] = process
    download_manager._process_jobs["org/m"] = old
    launched: list[str] = []
    monkeypatch.setattr(download_manager, "_launch", lambda job: launched.append(job.repo_id))
    download_manager._on_stdout("org/m")
    assert new.status == "queued" and new.path is None
    download_manager._on_finished("org/m", 0)
    assert new.status == "queued" and download_manager.registry.get_model("org/m") is None
    assert launched == ["org/m"]  # the queued replacement is started after the old child exits


def test_done_event_with_hostile_file_list_still_registers(download_manager: DownloadManager, tmp_path: Path) -> None:
    job = DownloadJob("org/m", status="downloading")
    download_manager.jobs["org/m"] = job
    download_manager._apply("org/m", {"type": "done", "path": str(tmp_path), "total_bytes": "x",
                                      "files": [1, {"nested": True}, "model.safetensors"]})
    assert job.status == "done"
    record = download_manager.registry.get_model("org/m")
    assert record is not None and record.metadata["files"] == ["model.safetensors"]


# =============================================================== API server child
def test_api_server_failed_start_is_actionable(qapp: Any, tmp_path: Path) -> None:
    class Runtime:
        state_changed = _ErrorSignal()
        endpoint = "tcp://127.0.0.1:1"
        token = "t"
        state = "ready"

    manager = sm.ServerManager(Settings(tmp_path / "s.json"), Runtime())
    fake = _Proc(running=False)
    manager._process = fake  # type: ignore[assignment]
    manager._state = "starting"
    manager._on_process_error(QProcess.ProcessError.FailedToStart)
    assert manager.state == "error" and "could not start" in (manager.last_error or "")
    assert manager._process is None and fake.killed is False
    manager._on_finished(0, QProcess.ExitStatus.NormalExit)  # late signal must not overwrite the diagnosis
    assert manager.state == "error" and "could not start" in (manager.last_error or "")


# =============================================================== handshake parsing
@pytest.mark.parametrize(("line", "expected"), [
    ("NOELLE_WORKER_READY port=42000", 42000),
    ("NOELLE_WORKER_READY port=abc", None),
    ("NOELLE_WORKER_READY port=0", None),
    ("NOELLE_WORKER_READY port=70000", None),
    ("NOELLE_WORKER_READY", None),
    ("", None),
])
def test_parse_ready_port(line: str, expected: int | None) -> None:
    assert parse_ready_port(line) == expected


def test_generation_params_ignore_hostile_values() -> None:
    defaults = GenerationParams()
    params = GenerationParams.from_dict({
        "temperature": "hot", "top_p": float("nan"), "top_k": float("inf"),
        "repetition_penalty": None, "max_new_tokens": "many", "seed": "abc",
        "stop": [{"not": "text"}, "", "END"],
    })
    assert params.temperature == defaults.temperature
    assert params.top_p == defaults.top_p
    assert params.top_k == defaults.top_k
    assert params.max_new_tokens == defaults.max_new_tokens
    assert params.seed is None
    assert params.stop == ["END"]
    assert GenerationParams.from_dict({"seed": 7}).seed == 7


# =============================================================== runtime client
def _client(tmp_path: Path) -> RuntimeClient:
    return RuntimeClient(Settings(tmp_path / "s.json"))


def test_ipc_failure_fails_pending_requests(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)
    client._state = "ready"
    handle = RequestHandle("rid", Command.PING, client)
    failures: list[dict[str, Any]] = []
    crashes: list[tuple[int, str]] = []
    handle.failed.connect(failures.append)
    client.worker_crashed.connect(lambda code, why: crashes.append((code, why)))
    client._pending["rid"] = handle
    client._on_ipc_error("socket closed")
    assert not client._pending and client.state == "crashed"
    assert failures and failures[0]["code"] == "WORKER_CRASHED"
    assert len(crashes) == 1
    client._on_ipc_error("again")  # a second failure must not duplicate the banner
    assert len(crashes) == 1
    qapp.processEvents()  # run the handle's deferred delete while its client is still alive


def test_runtime_init_error_is_surfaced_once(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)
    handle = RequestHandle("rid", Command.PING, client)
    failures: list[dict[str, Any]] = []
    handle.failed.connect(failures.append)
    client._pending["rid"] = handle
    beat = {"id": None, "event": "HEARTBEAT",
            "payload": {"runtime_ready": True, "init_error": "ImportError: torch missing", "loaded": []}}
    client._on_message(beat)
    assert client.state == "crashed" and failures and "torch missing" in failures[0]["message"]
    crashes: list[tuple[int, str]] = []
    client.worker_crashed.connect(lambda code, why: crashes.append((code, why)))
    client._on_message(beat)
    assert not crashes  # already reported
    qapp.processEvents()  # run the handle's deferred delete while its client is still alive


def test_stop_detaches_the_old_process(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)

    class Process:
        class ProcessState:
            NotRunning = 0

        deleted = False

        def state(self) -> int:
            return 1

        def waitForFinished(self, _ms: int) -> bool:  # noqa: N802
            return True

        def deleteLater(self) -> None:  # noqa: N802
            self.deleted = True

    process = Process()
    client._process = process  # type: ignore[assignment]
    client.stop()
    assert client._process is None and process.deleted and client.state == "stopped"


def test_late_signals_from_an_old_worker_are_ignored(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)
    client._state = "ready"
    current = object()
    client._process = current  # type: ignore[assignment]
    old = object()
    client._on_process_error(old, QProcess.ProcessError.FailedToStart)  # type: ignore[arg-type]
    client._on_finished(old, 1, QProcess.ExitStatus.CrashExit)  # type: ignore[arg-type]
    client._on_stdout(old)  # type: ignore[arg-type]
    assert client.state == "ready"


def test_new_requests_are_refused_while_unresponsive(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)
    client._state = "unresponsive"
    client._process = object()  # type: ignore[assignment]
    handle = client.request(Command.PING)
    # The refusal is scheduled for the next event-loop turn (so the caller can connect first),
    # but it must never enter the pending set or the outbox.
    assert isinstance(handle, RequestHandle)
    assert not client._pending and not client._outbox


def test_unresponsive_runtime_fails_pending_requests(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)
    client._state = "ready"
    client._last_heartbeat = time.monotonic() - 3600
    handle = RequestHandle("rid", Command.GENERATE_STREAM, client)
    failures: list[dict[str, Any]] = []
    handle.failed.connect(failures.append)
    client._pending["rid"] = handle
    client._check_heartbeat()
    assert client.state == "unresponsive" and not client._pending
    assert failures and "stopped responding" in failures[0]["message"]


def test_malformed_messages_and_ready_lines_do_not_crash(qapp: Any, tmp_path: Path) -> None:
    client = _client(tmp_path)
    client._on_message({"id": None, "event": "HEARTBEAT", "payload": [1, 2]})  # type: ignore[dict-item]
    assert client.state == "warming"

    class Proc:
        killed = False

        def readAllStandardOutput(self) -> bytes:  # noqa: N802
            return b"NOELLE_WORKER_READY port=not-a-number\n"

        def kill(self) -> None:
            self.killed = True

    process = Proc()
    client._process = process  # type: ignore[assignment]
    client._state = "starting"
    client._on_stdout(process)  # type: ignore[arg-type]
    assert process.killed and client.state == "crashed"


# =============================================================== child startup
def _script(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    monkeypatch.setattr(client_module, "worker_command", lambda _python=None: [sys.executable, "-c", code])


def test_spawn_worker_times_out_even_when_the_child_never_writes(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, "import time; time.sleep(30)")
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client_module.spawn_worker(ready_timeout=0.4, log_path=tmp_path / "worker.log")
    assert time.monotonic() - started < 10.0  # readline() used to make this unbounded


def test_spawn_worker_rejects_a_malformed_handshake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, "print('NOELLE_WORKER_READY port=nope', flush=True); import time; time.sleep(30)")
    with pytest.raises(RuntimeError, match="invalid ready line"):
        client_module.spawn_worker(ready_timeout=5.0, log_path=tmp_path / "worker.log")


def test_spawn_worker_reads_a_valid_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, "print('NOELLE_WORKER_READY port=4567', flush=True); import time; time.sleep(30)")
    worker = client_module.spawn_worker(ready_timeout=5.0, log_path=tmp_path / "worker.log")
    try:
        assert worker.port == 4567
        assert worker.process.poll() is None
    finally:
        worker.process.kill()
        worker.process.wait(timeout=5)
        if worker.process.stdout is not None:
            worker.process.stdout.close()


# =============================================================== model load cancellation
def test_cancelled_model_load_is_unloaded_and_reported() -> None:
    from noelle.runtime.errors import Cancelled
    from noelle.runtime.worker_service import _cmd_load

    class Model:
        key = "org/m|cpu"

        def describe(self) -> dict[str, Any]:
            return {}

    class Models:
        def __init__(self) -> None:
            self.unloaded: list[str] = []

        def resolve_device(self, _spec: Any) -> str:
            return "cpu"

        def get(self, _key: str) -> None:
            return None

        def ensure(self, _spec: Any, on_status: Any = None) -> Model:
            return Model()

        def unload(self, key: str) -> bool:
            self.unloaded.append(key)
            return True

    class Ctx:
        payload = {"model_id": "org/m"}
        stop_event = SimpleNamespace(is_set=lambda: True)

        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def status(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def emit(self, event: Any, payload: dict[str, Any] | None = None) -> None:
            self.events.append((str(event), payload or {}))

    models = Models()
    worker = SimpleNamespace(models=models, spec=lambda payload, task: ModelSpec(payload["model_id"]))
    ctx = Ctx()
    with pytest.raises(Cancelled):
        _cmd_load(worker, ctx)  # type: ignore[arg-type]
    assert models.unloaded == ["org/m|cpu"] and not ctx.events


# =============================================================== API worker frames
def test_worker_events_with_malformed_payload_are_normalized() -> None:
    import anyio

    from noelle.runtime.server_service import WorkerCall

    class Client:
        async def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            yield {"event": "STATUS", "payload": [1, 2, 3]}
            yield {"event": "RESULT", "payload": {"ok": True}}

    call = WorkerCall(Client(), Command.PING, {})  # type: ignore[arg-type]

    async def run() -> list[dict[str, Any]]:
        return [await call.next(), await call.next()]

    first, second = anyio.run(run)
    assert first["payload"] == {} and second["payload"] == {"ok": True}
    assert call.finished is True


# =============================================================== llama-server child
def test_llama_server_release_reaps_after_kill() -> None:
    from noelle.runtime.generation import LlamaServerEngine

    class Process:
        def __init__(self) -> None:
            self.terminated = self.killed = False
            self.waits = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float = 0) -> int:
            self.waits += 1
            if self.waits == 1:
                raise TimeoutError("still running")
            return -9

        def kill(self) -> None:
            self.killed = True

    process = Process()
    engine = LlamaServerEngine(process, port=1, n_ctx=128, device="cpu")
    engine.release()
    assert process.terminated and process.killed and process.waits == 2
    assert engine.process is None


# =============================================================== torch 2.4 fallback
class _FakeTorch:
    def __init__(self, *, cuda: Any = None, mps: Any = None, accelerator: Any = None) -> None:
        self.cuda = cuda or SimpleNamespace()
        self.mps = mps or SimpleNamespace()
        self.accelerator = accelerator


def test_memory_info_uses_cuda_api_when_accelerator_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTorch(cuda=SimpleNamespace(mem_get_info=lambda: (300, 1000)), accelerator=None)
    monkeypatch.setattr(devices, "_torch", lambda: fake)
    assert devices._memory_info("cuda") == (300, 1000)
    assert devices.free_bytes("cuda") == 300
    assert devices.memory_budget("cuda", headroom=0.2).total_bytes == 1000


def test_memory_info_uses_mps_api_and_accelerator_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    mps = SimpleNamespace(recommended_max_memory=lambda: 1000, driver_allocated_memory=lambda: 400)
    monkeypatch.setattr(devices, "_torch", lambda: _FakeTorch(mps=mps, accelerator=None))
    assert devices._memory_info("mps") == (600, 1000)

    accelerator = SimpleNamespace(get_memory_info=lambda: (7, 8))
    monkeypatch.setattr(devices, "_torch", lambda: _FakeTorch(accelerator=accelerator))
    assert devices._memory_info("cuda") == (7, 8)  # cuda lacks mem_get_info on the fake


# =============================================================== notebook kernel
class _FakeRunner:
    def submit(self, fn: Any, *args: Any, on_result: Any = None, on_error: Any = None, **kwargs: Any) -> None:
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - mirrors TaskRunner
            if on_error:
                on_error(str(exc))
        else:
            if on_result:
                on_result(value)


class _FakeClient:
    def __init__(self) -> None:
        self.stopped = False

    def start_channels(self) -> None:
        pass

    def stop_channels(self) -> None:
        self.stopped = True


class _FakeManager:
    has_kernel = True

    def __init__(self, **_kwargs: Any) -> None:
        self.stopped = False
        self.client_instance = _FakeClient()

    def client(self) -> _FakeClient:
        return self.client_instance

    def shutdown_kernel(self, now: bool = False) -> None:
        self.stopped = True


class _FakeConsole(QWidget):
    def set_default_style(self, **_kwargs: Any) -> None:
        pass

    def _append_plain_text(self, *_args: Any, **_kwargs: Any) -> None:
        pass


@pytest.fixture()
def notebook_panel(qapp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    package = ModuleType("qtconsole")
    package.__path__ = []  # type: ignore[attr-defined]
    manager_module = ModuleType("qtconsole.manager")
    manager_module.QtKernelManager = _FakeManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qtconsole", package)
    monkeypatch.setitem(sys.modules, "qtconsole.manager", manager_module)
    monkeypatch.setattr(jb, "jupyter_available", lambda: True)
    monkeypatch.setattr(jb, "runner", lambda: _FakeRunner())
    monkeypatch.setattr(jb, "_console_class", lambda: _FakeConsole)
    ctx = SimpleNamespace(settings=Settings(tmp_path / "s.json"), toast=lambda *_a, **_k: None)
    return jb.NotebookPanel(ctx, SimpleNamespace(summary=None, worker_payload=lambda: {}))


def test_notebook_start_failure_is_visible_and_recoverable(notebook_panel: Any,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_manager: Any, _env: Any) -> None:
        raise RuntimeError("ipykernel is not installed")

    monkeypatch.setattr(jb, "_start_kernel_blocking", fail)
    notebook_panel.start_kernel()
    assert not notebook_panel.error_banner.isHidden()
    assert "ipykernel is not installed" in notebook_panel.error_banner.text.text()
    assert notebook_panel.start_btn.isEnabled()
    assert notebook_panel.kernel_manager is None and not notebook_panel._starting


def test_notebook_start_runs_off_the_gui_thread(notebook_panel: Any,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def start(manager: Any, _env: Any) -> None:
        calls.append("started")

    monkeypatch.setattr(jb, "_start_kernel_blocking", start)
    notebook_panel.start_kernel()
    assert calls == ["started"]  # the blocking call went through the task runner
    assert notebook_panel.console is not None and notebook_panel.console.kernel_client is not None
    assert notebook_panel.snippet_btn.isEnabled() and notebook_panel.stop_btn.isEnabled()
    manager = notebook_panel.kernel_manager
    client = notebook_panel.kernel_client
    notebook_panel.shutdown()
    assert manager.stopped and client.stopped and notebook_panel.console is None