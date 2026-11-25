"""Background model downloads, one ``noelle download`` child process per repository.

Pause and cancel terminate the child (files finished so far stay in the cache;
the hub writes each file to a temporary name and renames it atomically, so no
truncated shard is ever left in place). Resume restarts the child, which skips
completed files. Progress arrives as JSON lines on the child's stdout.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from noelle import paths
from noelle.config import Settings
from noelle.runtime.client import PACKAGE_PARENT, worker_command

from noelle.app.services import hf_client
from noelle.app.services.model_registry import ModelRecord, ModelRegistry

MAX_CONCURRENT = 2
TERMINATE_GRACE_MS = 3000
log = logging.getLogger("noelle.downloads")


def _text(value: Any, limit: int = 2000) -> str | None:
    """Coerce a JSON scalar to bounded text (never a list/dict repr)."""
    if isinstance(value, str):
        return value[:limit]
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _file_name(value: Any) -> str | None:
    """Repository paths come from the child as strings; anything else is malformed."""
    return value if isinstance(value, str) and value else None


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _parse_files(value: Any) -> dict[str, FileProgress]:
    files: dict[str, FileProgress] = {}
    if not isinstance(value, list):
        return files
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _file_name(item.get("name"))
        if not name:
            continue
        size = _positive_int(item.get("size"))
        cached = item.get("cached") is True
        files[name] = FileProgress(name, size, cached=cached, done=cached, downloaded=(size or 0) if cached else 0)
    return files


@dataclass
class FileProgress:
    name: str
    size: int | None
    downloaded: int = 0
    done: bool = False
    cached: bool = False
    verified: bool | None = None


@dataclass
class DownloadJob:
    repo_id: str
    status: str = "queued"  # queued | planning | downloading | paused | done | error | cancelled
    revision: str | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    files: dict[str, FileProgress] = field(default_factory=dict)
    total_bytes: int = 0
    download_bytes: int = 0
    speed_bps: float = 0.0
    error: str | None = None
    error_code: str | None = None
    path: str | None = None
    commit: str | None = None
    retries: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    summary: Any = None  # hf_client.ModelSummary used to enrich the registry entry
    log: list[str] = field(default_factory=list)
    _last_bytes: int = 0
    _last_time: float = 0.0
    _finished_notified: bool = False

    @property
    def downloaded_bytes(self) -> int:
        return sum(f.downloaded if not f.done else (f.size or f.downloaded) for f in self.files.values()
                   if not f.cached)

    @property
    def fraction(self) -> float:
        if self.status == "done":
            return 1.0
        if not self.download_bytes:
            return 0.0
        return min(self.downloaded_bytes / self.download_bytes, 1.0)

    @property
    def eta_s(self) -> float | None:
        if self.speed_bps <= 0 or not self.download_bytes:
            return None
        return max(self.download_bytes - self.downloaded_bytes, 0) / self.speed_bps

    @property
    def active(self) -> bool:
        return self.status in ("queued", "planning", "downloading")


class DownloadManager(QObject):
    job_added = Signal(str)
    job_updated = Signal(str)
    job_finished = Signal(str, bool)

    def __init__(self, settings: Settings, registry: ModelRegistry, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.registry = registry
        self.jobs: dict[str, DownloadJob] = {}
        self._processes: dict[str, QProcess] = {}
        self._process_jobs: dict[str, DownloadJob] = {}
        self._buffers: dict[str, str] = {}
        self._intentional_stop: set[str] = set()

    # ----------------------------------------------------------------- API
    def job(self, repo_id: str) -> DownloadJob | None:
        return self.jobs.get(repo_id)

    def active_repos(self) -> set[str]:
        return {r for r, j in self.jobs.items() if j.active}

    def start(self, repo_id: str, *, summary: Any = None, include: list[str] | None = None,
              exclude: list[str] | None = None, revision: str | None = None) -> DownloadJob:
        existing = self.jobs.get(repo_id)
        if existing is not None and existing.active:
            return existing
        job = DownloadJob(repo_id=repo_id, revision=revision, include=include, exclude=exclude,
                          summary=summary or (existing.summary if existing else None))
        self.jobs[repo_id] = job
        self.job_added.emit(repo_id)
        self._pump()
        return job

    def pause(self, repo_id: str) -> None:
        self._terminate(repo_id, "paused")

    def resume(self, repo_id: str) -> None:
        job = self.jobs.get(repo_id)
        if job is not None and job.status in ("paused", "error", "cancelled"):
            self.start(repo_id, summary=job.summary, include=job.include, exclude=job.exclude, revision=job.revision)

    def cancel(self, repo_id: str) -> None:
        self._terminate(repo_id, "cancelled")

    def shutdown(self) -> None:
        # Downloads are resumable and every completed file is already in the cache, so a
        # hard stop is safe and keeps quitting fast (terminate was given a short chance).
        for repo_id in list(self._processes):
            self._terminate(repo_id, "paused", force=True)
        for process in list(self._processes.values()):
            try:
                if process.state() != QProcess.ProcessState.NotRunning:
                    process.waitForFinished(500)
            except RuntimeError:
                pass

    # ----------------------------------------------------------- internals
    def _terminate(self, repo_id: str, status: str, *, force: bool = False) -> None:
        """Signal the child without blocking the GUI thread; escalation happens on a timer."""
        job = self.jobs.get(repo_id)
        if job is None:
            return
        process = self._processes.get(repo_id)
        running = False
        if process is not None:
            try:
                running = process.state() != QProcess.ProcessState.NotRunning
            except RuntimeError:  # already deleted by the QProcess own cleanup
                running = False
            if running:
                self._intentional_stop.add(repo_id)
                if force:
                    process.kill()
                else:
                    process.terminate()
                    QTimer.singleShot(TERMINATE_GRACE_MS, lambda p=process: self._kill_if_running(p))
        job.status = status
        job.speed_bps = 0.0
        job.finished_at = job.finished_at or time.time()
        self.job_updated.emit(repo_id)
        if process is None or not running:
            self._pump()

    @staticmethod
    def _kill_if_running(process: QProcess) -> None:
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
        except RuntimeError:
            pass  # the process finished and its wrapper was deleted before the timer fired

    def _pump(self) -> None:
        running = 0
        for process in self._processes.values():
            try:
                running += process.state() != QProcess.ProcessState.NotRunning
            except RuntimeError:  # the QProcess wrapper was deleted; it is not running any more
                continue
        for repo_id, job in self.jobs.items():
            if running >= MAX_CONCURRENT:
                break
            if job.status == "queued" and repo_id not in self._processes:
                self._launch(job)
                running += 1

    def _launch(self, job: DownloadJob) -> None:
        command = worker_command(None)
        args = command[1:] + ["download", job.repo_id,
                              "--max-workers", str(self.settings.get("storage.max_download_workers", 8)),
                              "--retries", str(self.settings.get("storage.download_retries", 5))]
        if job.revision:
            args += ["--revision", job.revision]
        if self.settings.get("storage.download_mode") == "local_dir":
            args += ["--local-dir", str(paths.models_dir() / job.repo_id.replace("/", "__"))]
        for pattern in job.include or []:
            args += ["--include", pattern]
        for pattern in job.exclude or []:
            args += ["--exclude", pattern]
        if not self.settings.get("storage.verify_checksums", True):
            args.append("--no-verify")
        env = QProcessEnvironment.systemEnvironment()
        token = hf_client.get_token()
        if token:
            env.insert("HF_TOKEN", token)  # environment only, never argv
        hf_home = self.settings.get("storage.hf_home")
        if hf_home:
            env.insert("HF_HOME", str(hf_home))
            env.remove("HF_HUB_CACHE")
        env.insert("HF_HUB_DISABLE_TELEMETRY", "1")
        env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        existing = env.value("PYTHONPATH")
        env.insert("PYTHONPATH", PACKAGE_PARENT + (os.pathsep + existing if existing else ""))
        process = QProcess(self)
        process.setProcessEnvironment(env)
        process.setProgram(command[0])
        process.setArguments(args)
        repo_id = job.repo_id
        process.readyReadStandardOutput.connect(lambda r=repo_id: self._on_stdout(r))
        process.readyReadStandardError.connect(lambda r=repo_id: self._on_stderr(r))
        process.finished.connect(lambda code, status, r=repo_id: self._on_finished(r, code))
        error_signal = getattr(process, "errorOccurred", None)  # absent on test doubles
        if error_signal is not None:
            error_signal.connect(lambda error, r=repo_id: self._on_process_error(r, error))
        self._processes[repo_id] = process
        self._process_jobs[repo_id] = job  # identifies this child's output even if the job was replaced
        self._buffers[repo_id] = ""
        job.status = "planning"
        job.error = None
        job.error_code = None
        job.started_at = time.time()
        job._last_time = time.monotonic()
        job._last_bytes = 0
        self.job_updated.emit(repo_id)
        process.start()

    def _job_for_process(self, repo_id: str) -> DownloadJob | None:
        """The job that owns the running child, or None when the child belongs to a replaced job."""
        owner = self._process_jobs.get(repo_id)
        current = self.jobs.get(repo_id)
        if owner is None:  # tests and any legacy caller seed only _processes
            return current
        return owner if current is owner else None

    def _on_stderr(self, repo_id: str) -> None:
        process = self._processes.get(repo_id)
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        job = self._job_for_process(repo_id)
        if job is None:
            return
        for line in text.splitlines()[-20:]:
            if line.strip():
                job.log.append(line.strip())
        del job.log[:-50]

    def _on_stdout(self, repo_id: str) -> None:
        process = self._processes.get(repo_id)
        if process is None:
            return
        buffer = self._buffers.get(repo_id, "") + bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = buffer.split("\n")
        self._buffers[repo_id] = lines.pop()
        job = self._job_for_process(repo_id)
        if job is None:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                self._apply(repo_id, event)
        self.job_updated.emit(repo_id)

    def _apply(self, repo_id: str, event: dict[str, Any]) -> None:
        """Apply one progress line. The child is external input: never trust its shape."""
        job = self.jobs.get(repo_id)
        if job is None or not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind not in ("plan", "progress", "file_done", "verify", "retry", "file_error", "done", "error"):
            return
        if kind not in ("done", "error") and job.status in ("paused", "cancelled"):
            return  # late output from a child that is being stopped must not resurrect the job
        try:
            if kind == "plan":
                job.commit = _text(event.get("commit")) or job.commit
                job.total_bytes = _positive_int(event.get("total_bytes")) or 0
                job.download_bytes = _positive_int(event.get("download_bytes")) or 0
                job.files = _parse_files(event.get("files"))
                job.status = "downloading"
            elif kind == "progress":
                entry = job.files.get(_file_name(event.get("file")) or "")
                if entry is not None:
                    downloaded = _positive_int(event.get("downloaded"))
                    if downloaded is not None:
                        entry.downloaded = downloaded
                    total = _positive_int(event.get("total"))
                    if total and not entry.size:
                        entry.size = total
                self._update_speed(job)
            elif kind == "file_done":
                entry = job.files.get(_file_name(event.get("file")) or "")
                if entry is not None:
                    entry.done = True
                    entry.downloaded = _positive_int(event.get("size")) or entry.size or entry.downloaded
                self._update_speed(job)
            elif kind == "verify":
                entry = job.files.get(_file_name(event.get("file")) or "")
                if entry is not None:
                    entry.verified = event.get("ok") if isinstance(event.get("ok"), bool) else None
            elif kind == "retry":
                job.retries += 1
                job.log.append(f"retry {_text(event.get('attempt'))} for {_text(event.get('file'))}: "
                               f"{_text(event.get('error'))}")
            elif kind == "file_error":
                job.log.append(f"{_text(event.get('file'))}: {_text(event.get('message'))}")
            elif kind == "done":
                job.path = _text(event.get("path")) or job.path
                job.commit = _text(event.get("commit")) or job.commit
                job.status = "done"
                job.finished_at = time.time()
                self._register(job, event)
            else:  # error
                job.error = _text(event.get("message")) or "The download failed."
                job.error_code = _text(event.get("code"))
                job.status = "error"
                job.finished_at = time.time()
        except (OSError, ValueError, TypeError) as exc:
            # A malformed/unexpected line is a diagnostic, not a crash in a Qt slot.
            log.warning("Ignoring malformed download event for %s: %s", repo_id, exc)
            job.log.append(f"ignored malformed download update: {exc}")
        del job.log[:-50]

    def _on_process_error(self, repo_id: str, error: QProcess.ProcessError) -> None:
        """Surface a child that never started; ``finished`` is not emitted for FailedToStart."""
        if error != QProcess.ProcessError.FailedToStart or repo_id in self._intentional_stop:
            return
        job = self._job_for_process(repo_id)
        process = self._processes.pop(repo_id, None)
        self._process_jobs.pop(repo_id, None)
        detail = ""
        if process is not None:
            try:
                detail = process.errorString().strip()
            except RuntimeError:
                pass
            process.deleteLater()
        self._buffers.pop(repo_id, None)
        if job is None:
            return
        job.status = "error"
        job.error_code = "PROCESS_START"
        job.error = ("The download helper could not start. The installed Python runtime may be missing or "
                     "inaccessible. Reinstall Noelle or select a valid runtime Python in Settings."
                     + (f" ({detail})" if detail else ""))
        job.finished_at = time.time()
        log.error("Download helper for %s failed to start: %s", repo_id, detail or "(no detail)")
        self.job_updated.emit(repo_id)
        self._notify_finished(repo_id, job)
        self._pump()

    def _notify_finished(self, repo_id: str, job: DownloadJob) -> None:
        if job._finished_notified:
            return
        job._finished_notified = True
        self.job_finished.emit(repo_id, job.status == "done")

    def _update_speed(self, job: DownloadJob) -> None:
        now = time.monotonic()
        elapsed = now - job._last_time
        if elapsed >= 0.5:
            current = job.downloaded_bytes
            instant = max(current - job._last_bytes, 0) / elapsed
            job.speed_bps = instant if job.speed_bps == 0 else 0.7 * job.speed_bps + 0.3 * instant
            job._last_bytes = current
            job._last_time = now

    def _register(self, job: DownloadJob, event: dict[str, Any]) -> None:
        summary = job.summary
        files = event.get("files")
        names = [name for name in files if isinstance(name, str)] if isinstance(files, list) else []
        record = ModelRecord(
            repo_id=job.repo_id,
            pipeline_tag=getattr(summary, "pipeline_tag", None),
            local_path=job.path,
            revision=job.revision or "main",
            commit_hash=job.commit,
            library_name=getattr(summary, "library_name", None),
            architecture=getattr(summary, "architecture", None),
            param_count=getattr(summary, "param_count", None),
            license=getattr(summary, "license", None),
            size_bytes=_positive_int(event.get("total_bytes")) or job.total_bytes or 0,
            metadata={"tags": list(getattr(summary, "tags", []) or [])[:40], "files": names[:200]},
        )
        gguf_files = [f for f in names if f.endswith(".gguf")]
        has_weights = any(f.endswith((".safetensors", ".bin")) for f in names)
        if gguf_files and not has_weights:
            record.gguf_file = gguf_files[0]
            record.precision = "gguf-q4_k_m" if "q4_k" in gguf_files[0].lower() else "auto"
        try:
            if getattr(summary, "is_adapter", False):
                from noelle.app.services.model_registry import AdapterRecord

                base = (getattr(summary, "base_models", None) or [None])[0]
                self.registry.add_adapter(AdapterRecord(job.repo_id, base, job.path, record.size_bytes))
            else:
                self.registry.upsert_model(record)
        except Exception:  # noqa: BLE001 - the files are downloaded; a registry failure must be reported, not crash
            log.exception("The download of %s finished but could not be added to the library", job.repo_id)
            job.log.append("Download finished, but saving it to the library failed. Run 'Rescan' in Storage.")
            job.error = "Downloaded, but could not be saved to the library. Run Rescan in Storage."
            job.error_code = "REGISTRY_WRITE"
            job.status = "error"

    def _on_finished(self, repo_id: str, exit_code: int) -> None:
        current = self.jobs.get(repo_id)
        owner = self._process_jobs.get(repo_id)
        job = owner if owner is not None else current
        if job is not current:  # a newer job replaced this child: its output must not touch the new one
            job = None
        process = self._processes.pop(repo_id, None)
        self._buffers.pop(repo_id, None)
        if process is not None:
            if job is not None:
                self._on_stdout_tail(repo_id, process)
            process.deleteLater()
        self._process_jobs.pop(repo_id, None)
        intentional = repo_id in self._intentional_stop
        self._intentional_stop.discard(repo_id)
        if job is not None and not intentional and job.status not in ("done", "error"):
            job.status = "error" if exit_code != 0 else "done"
            job.finished_at = job.finished_at or time.time()
            if exit_code != 0 and not job.error:
                job.error = "The download process stopped unexpectedly." + (
                    f" Last output: {job.log[-1]}" if job.log else "")
                job.error_code = "PROCESS_EXIT"
        if job is not None:
            job.speed_bps = 0.0
            self.job_updated.emit(repo_id)
            if not intentional:
                self._notify_finished(repo_id, job)
        self._pump()

    def _on_stdout_tail(self, repo_id: str, process: QProcess) -> None:
        remaining = self._buffers.pop(repo_id, "") + bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in remaining.splitlines():
            try:
                self._apply(repo_id, json.loads(line))
            except ValueError:
                continue
