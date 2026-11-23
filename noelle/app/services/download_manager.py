"""Background model downloads, one ``noelle download`` child process per repository.

Pause and cancel terminate the child (files finished so far stay in the cache;
the hub writes each file to a temporary name and renames it atomically, so no
truncated shard is ever left in place). Resume restarts the child, which skips
completed files. Progress arrives as JSON lines on the child's stdout.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from noelle import paths
from noelle.config import Settings
from noelle.runtime.client import PACKAGE_PARENT, worker_command

from noelle.app.services import hf_client
from noelle.app.services.model_registry import ModelRecord, ModelRegistry

MAX_CONCURRENT = 2


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
        for repo_id in list(self._processes):
            self._terminate(repo_id, "paused")

    # ----------------------------------------------------------- internals
    def _terminate(self, repo_id: str, status: str) -> None:
        job = self.jobs.get(repo_id)
        if job is None:
            return
        process = self._processes.get(repo_id)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            self._intentional_stop.add(repo_id)
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(2000)
        job.status = status
        job.speed_bps = 0.0
        self.job_updated.emit(repo_id)
        self._pump()

    def _pump(self) -> None:
        running = sum(1 for p in self._processes.values() if p.state() != QProcess.ProcessState.NotRunning)
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
        self._processes[repo_id] = process
        self._buffers[repo_id] = ""
        job.status = "planning"
        job.error = None
        job.error_code = None
        job.started_at = time.time()
        job._last_time = time.monotonic()
        job._last_bytes = 0
        self.job_updated.emit(repo_id)
        process.start()

    def _on_stderr(self, repo_id: str) -> None:
        process = self._processes.get(repo_id)
        job = self.jobs.get(repo_id)
        if process is None or job is None:
            return
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
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
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            self._apply(repo_id, event)
        self.job_updated.emit(repo_id)

    def _apply(self, repo_id: str, event: dict[str, Any]) -> None:
        job = self.jobs[repo_id]
        kind = event.get("type")
        if kind == "plan":
            job.commit = event.get("commit")
            job.total_bytes = int(event.get("total_bytes") or 0)
            job.download_bytes = int(event.get("download_bytes") or 0)
            job.files = {f["name"]: FileProgress(f["name"], f.get("size"), cached=bool(f.get("cached")),
                                                 done=bool(f.get("cached")),
                                                 downloaded=int(f.get("size") or 0) if f.get("cached") else 0)
                         for f in event.get("files", [])}
            job.status = "downloading"
        elif kind == "progress":
            entry = job.files.get(event.get("file", ""))
            if entry is not None:
                entry.downloaded = int(event.get("downloaded") or 0)
                if event.get("total") and not entry.size:
                    entry.size = int(event["total"])
            self._update_speed(job)
        elif kind == "file_done":
            entry = job.files.get(event.get("file", ""))
            if entry is not None:
                entry.done = True
                entry.downloaded = int(event.get("size") or entry.size or entry.downloaded)
            self._update_speed(job)
        elif kind == "verify":
            entry = job.files.get(event.get("file", ""))
            if entry is not None:
                entry.verified = event.get("ok")
        elif kind == "retry":
            job.retries += 1
            job.log.append(f"retry {event.get('attempt')} for {event.get('file')}: {event.get('error')}")
        elif kind == "file_error":
            job.log.append(f"{event.get('file')}: {event.get('message')}")
        elif kind == "done":
            job.path = event.get("path")
            job.commit = event.get("commit") or job.commit
            job.status = "done"
            job.finished_at = time.time()
            self._register(job, event)
        elif kind == "error":
            job.error = event.get("message")
            job.error_code = event.get("code")
            job.status = "error"

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
            size_bytes=int(event.get("total_bytes") or job.total_bytes or 0),
            metadata={"tags": list(getattr(summary, "tags", []) or [])[:40],
                      "files": event.get("files", [])[:200]},
        )
        gguf_files = [f for f in event.get("files", []) if str(f).endswith(".gguf")]
        has_weights = any(str(f).endswith((".safetensors", ".bin")) for f in event.get("files", []))
        if gguf_files and not has_weights:
            record.gguf_file = gguf_files[0]
            record.precision = "gguf-q4_k_m" if "q4_k" in gguf_files[0].lower() else "auto"
        if getattr(summary, "is_adapter", False):
            from noelle.app.services.model_registry import AdapterRecord

            base = (getattr(summary, "base_models", None) or [None])[0]
            self.registry.add_adapter(AdapterRecord(job.repo_id, base, job.path, record.size_bytes))
        else:
            self.registry.upsert_model(record)

    def _on_finished(self, repo_id: str, exit_code: int) -> None:
        job = self.jobs.get(repo_id)
        process = self._processes.pop(repo_id, None)
        if process is not None:
            self._on_stdout_tail(repo_id, process)
            process.deleteLater()
        intentional = repo_id in self._intentional_stop
        self._intentional_stop.discard(repo_id)
        if job is not None and not intentional and job.status not in ("done", "error"):
            job.status = "error" if exit_code != 0 else "done"
            if exit_code != 0 and not job.error:
                job.error = "The download process stopped unexpectedly." + (
                    f" Last output: {job.log[-1]}" if job.log else "")
        if job is not None:
            job.speed_bps = 0.0
            self.job_updated.emit(repo_id)
            if not intentional:
                self.job_finished.emit(repo_id, job.status == "done")
        self._pump()

    def _on_stdout_tail(self, repo_id: str, process: QProcess) -> None:
        remaining = self._buffers.pop(repo_id, "") + bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in remaining.splitlines():
            try:
                self._apply(repo_id, json.loads(line))
            except ValueError:
                continue
