"""Desktop-side training orchestrator: one child process per run, JSON progress lines."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from noelle.config import Settings
from noelle.runtime.client import PACKAGE_PARENT, worker_command
from noelle.training.config import ConfigError, TrainingConfig
from noelle.training.store import TrainingStore

log = logging.getLogger("noelle.training.manager")


def _alive(obj: Any) -> bool:
    try:
        import shiboken6

        return bool(shiboken6.isValid(obj))
    except Exception:  # noqa: BLE001 - shiboken is present with PySide6, but never assume
        return True

_DEPENDENCY_SCRIPT = """
import importlib.metadata as metadata, importlib.util as util, json
names = ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except Exception:
        versions[name] = None
missing = [name for name in names if not versions.get(name)]
bitsandbytes = util.find_spec("bitsandbytes") is not None
versions["bitsandbytes"] = "installed" if bitsandbytes else None
print(json.dumps({"versions": versions, "missing": missing, "bitsandbytes": bitsandbytes,
                  "compatible": not missing}))
"""


def _format_event(event: dict[str, Any]) -> str:
    kind = str(event.get("type") or "")
    if kind == "training.progress":
        parts = [f"step {event.get('step') or 0}/{event.get('total_steps') or '?'}"]
        loss = event.get("loss")
        if isinstance(loss, (int, float)):
            parts.append(f"loss {loss:.4f}")
        if isinstance(event.get("learning_rate"), (int, float)):
            parts.append(f"lr {event['learning_rate']:.2e}")
        if isinstance(event.get("gpu_memory_mb"), (int, float)):
            parts.append(f"VRAM {event['gpu_memory_mb']:.0f} MB")
        return " · ".join(parts)
    if kind == "training.plan":
        plan = event.get("plan") or {}
        return (f"plan: {plan.get('risk')} · {plan.get('mode')} · {plan.get('precision')} · "
                f"{plan.get('expected_steps')} steps")
    if kind == "training.dataset":
        report = event.get("report") or {}
        return (f"dataset: {report.get('train', 0)} train / {report.get('validation', 0)} validation rows · "
                f"sha256 {(report.get('sha256') or '')[:12]}")
    if kind == "training.preparing":
        return f"preparing {event.get('stage') or 'training'}"
    if kind == "training.started":
        return f"training started ({event.get('mode') or 'sft'})"
    if kind == "training.checkpoint":
        return f"checkpoint saved: {event.get('path')}"
    if kind == "training.completed":
        return f"completed: {event.get('output_path')}"
    if kind == "training.stopped":
        return f"stopped at a checkpoint: {event.get('output_path') or ''}".rstrip(": ")
    if kind in ("training.warning", "training.error"):
        return f"{kind.split('.')[-1]}: {event.get('message')}"
    return ""


def _dependency_report_for_python(python: str) -> dict[str, Any]:
    try:
        result = subprocess.run([python, "-c", _DEPENDENCY_SCRIPT], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60,
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"versions": {}, "missing": ["torch", "transformers", "datasets", "accelerate", "peft", "trl"],
                "bitsandbytes": False, "compatible": False, "error": str(exc)}
    if result.returncode != 0:
        return {"versions": {}, "missing": ["torch", "transformers", "datasets", "accelerate", "peft", "trl"],
                "bitsandbytes": False, "compatible": False,
                "error": (result.stderr or "The training interpreter could not run.").strip()[-400:]}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"versions": {}, "missing": [], "bitsandbytes": False, "compatible": False,
                "error": "The training interpreter returned an unreadable dependency report."}
    return payload if isinstance(payload, dict) else {"compatible": False, "missing": [], "versions": {}}


class TrainingManager(QObject):
    job_added = Signal(str)
    job_updated = Signal(str)
    job_finished = Signal(str, bool)
    log_line = Signal(str, str)

    def __init__(self, settings: Settings, registry: Any, store: TrainingStore | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.registry = registry
        self.store = store or TrainingStore()
        self._processes: dict[str, QProcess] = {}
        self._buffers: dict[str, str] = {}
        self._stderr: dict[str, list[str]] = {}
        self._python_usable: dict[str, bool] = {}
        self._reconcile()

    def _reconcile(self) -> None:
        """A run cannot outlive the shell's QProcess handles: mark leftovers resumable."""
        for run in self.store.list(500):
            if run.get("status") in ("queued", "preparing", "training", "stopping"):
                self.store.update(run["id"], status="interrupted",
                                  error="Noelle closed while this run was active. Resume from its last checkpoint.")

    # ------------------------------------------------------------------ API
    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list(limit)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.store.get(run_id)

    def active_runs(self) -> list[dict[str, Any]]:
        return [run for run in self.store.list(200) if run.get("id") in self._processes]

    def is_active(self, run_id: str) -> bool:
        return run_id in self._processes

    def start(self, config: TrainingConfig, *, run_id: str | None = None, resume: bool = False) -> str:
        config.validate()
        if self._processes:
            raise ConfigError(["Another training run is active. Stop it before starting a new one."])
        if resume and run_id:
            run = self.store.get(run_id)
            if run is None:
                raise ConfigError([f"Training run not found: {run_id}"])
            latest = self.store.latest_checkpoint(run_id)
            if latest is None:
                raise ConfigError(["No checkpoint is available to resume from."])
            config.resume_from = str(latest["path"])
        else:
            run = self.store.create(config)
            run_id = run["id"]
        path = self.store.config_path(run_id)
        path.write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
        self.store.update(run_id, status="queued", error=None, finished_at=None, pid=None)
        process = self._spawn(run_id, path)
        self.job_added.emit(run_id)
        self._append_log(run_id, f"$ {process.program()} {' '.join(process.arguments())}")
        process.start()
        return run_id

    def stop(self, run_id: str, *, force: bool = False) -> None:
        process = self._processes.get(run_id)
        if process is not None:
            self.store.update(run_id, status="stopping")
            try:
                if force:
                    process.kill()
                else:
                    process.terminate()
            except RuntimeError:
                pass
            self.job_updated.emit(run_id)
            return
        self.store.update(run_id, status="stopped", finished_at=time.time())
        self.job_finished.emit(run_id, False)

    def resume(self, run_id: str) -> str:
        if self._processes:
            raise ConfigError(["Another training run is active. Stop it before resuming."])
        run = self.store.get(run_id)
        if run is None:
            raise ConfigError([f"Training run not found: {run_id}"])
        config = TrainingConfig.from_dict(run.get("config") or {})
        latest = self.store.latest_checkpoint(run_id)
        if latest is None:
            raise ConfigError(["No checkpoint is available to resume from."])
        config.resume_from = str(latest["path"])
        self.store.update(run_id, status="queued", finished_at=None, error=None)
        config_path = self.store.config_path(run_id)
        config_path.write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
        process = self._spawn(run_id, config_path)
        process.start()
        self.job_updated.emit(run_id)
        return run_id

    def logs(self, run_id: str) -> str:
        path = self.store.log_path(run_id)
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.checkpoints(run_id)

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.metrics(run_id)

    def merge(self, base: str, adapter: str, output: str) -> str:
        """Merge in a child process so the desktop shell never imports torch."""
        python = self.training_python()
        command = ([python, "-m", "noelle"] if python else worker_command()) + [
            "train", "merge", "--base", base, "--adapter", adapter, "--output", output]
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = PACKAGE_PARENT + (os.pathsep + existing if existing else "")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                env=environment, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "The merge process failed.").strip()
            raise RuntimeError(detail[-800:])
        return output

    def dependencies(self) -> dict[str, Any]:
        python = self.training_python()
        if python and os.path.realpath(python) != os.path.realpath(sys.executable):
            return _dependency_report_for_python(python)
        from noelle.training.trainer import dependency_report

        return dependency_report()

    def training_python(self) -> str | None:
        environment = self.settings.get("training.environment")
        if not environment:
            return None
        from noelle.runtime import environment_manager

        try:
            python = str(environment_manager.python_for(str(environment)))
        except Exception:  # noqa: BLE001 - fall back to the app interpreter with a visible warning
            return None
        if getattr(sys, "frozen", False) and not self._environment_has_noelle(python):
            # A packaged build keeps Noelle inside the executable; a plain venv cannot
            # import it. Use the bundled training stack instead of failing at startup.
            log.warning("Training environment '%s' cannot import Noelle; using the bundled runtime.", environment)
            return None
        return python

    def _environment_has_noelle(self, python: str) -> bool:
        cached = self._python_usable.get(python)
        if cached is not None:
            return cached
        environment = dict(os.environ)
        environment["PYTHONPATH"] = PACKAGE_PARENT
        try:
            result = subprocess.run([python, "-c", "import noelle"], capture_output=True, timeout=20,
                                    env=environment, stdin=subprocess.DEVNULL)
            usable = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            usable = False
        self._python_usable[python] = usable
        return usable

    def shutdown(self) -> None:
        for process in list(self._processes.values()):
            try:
                if process.state() != QProcess.ProcessState.NotRunning:
                    process.terminate()
                    if not process.waitForFinished(5000):
                        process.kill()
                        process.waitForFinished(2000)
            except RuntimeError:
                pass

    # -------------------------------------------------------------- internal
    def _spawn(self, run_id: str, config_path: Path) -> QProcess:
        process = QProcess(self)
        process.setProcessEnvironment(self._environment())
        command, args = self._command(run_id, config_path)
        process.setProgram(command)
        process.setArguments(args)
        process.readyReadStandardOutput.connect(lambda r=run_id: self._on_stdout(r))
        process.readyReadStandardError.connect(lambda r=run_id: self._on_stderr(r))
        process.finished.connect(lambda code, status, r=run_id: self._on_finished(r, code))
        process.errorOccurred.connect(lambda error, r=run_id: self._on_process_error(r, error))
        self._processes[run_id] = process
        self._buffers[run_id] = ""
        self._stderr[run_id] = []
        return process

    def _command(self, run_id: str, config_path: Path) -> tuple[str, list[str]]:
        python = self.training_python()
        if python:
            # Use the environment's console interpreter directly (no pythonw swap).
            return python, ["-m", "noelle", "train-worker", "--run-id", run_id, "--config", str(config_path)]
        command = worker_command()
        return command[0], command[1:] + ["train-worker", "--run-id", run_id, "--config", str(config_path)]

    def _environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        existing = env.value("PYTHONPATH")
        env.insert("PYTHONPATH", PACKAGE_PARENT + (os.pathsep + existing if existing else ""))
        env.insert("NOELLE_TRAINING_WORKER", "1")
        return env

    def _on_stdout(self, run_id: str) -> None:
        process = self._processes.get(run_id)
        if process is None:
            return
        try:
            chunk = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        except RuntimeError:
            return
        buffer = self._buffers.get(run_id, "") + chunk
        lines = buffer.split("\n")
        self._buffers[run_id] = lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                self._append_log(run_id, line)
                continue
            if isinstance(event, dict):
                self._append_log(run_id, _format_event(event) or line)
                self._apply_event(run_id, event)
            else:
                self._append_log(run_id, line)

    def _on_stderr(self, run_id: str) -> None:
        process = self._processes.get(run_id)
        if process is None:
            return
        try:
            text = bytes(process.readAllStandardError()).decode("utf-8", "replace")
        except RuntimeError:
            return
        for line in text.splitlines():
            if line.strip():
                self._stderr.setdefault(run_id, []).append(line.strip())
                self._append_log(run_id, line.strip())
                self.log_line.emit(run_id, line.strip())

    def _apply_event(self, run_id: str, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "training.progress":
            # The worker owns metric persistence; adding it here would duplicate every row.
            self.job_updated.emit(run_id)
        elif kind == "training.dataset":
            report = event.get("report") or {}
            self.store.update(run_id, dataset_sha=report.get("sha256"))
            self.job_updated.emit(run_id)
        elif kind == "training.checkpoint":
            # Persisted by the worker's TrainerCallback.
            self.job_updated.emit(run_id)
        elif kind == "training.warning":
            self._append_log(run_id, "WARNING: " + str(event.get("message") or ""))
            self.job_updated.emit(run_id)
        elif kind == "training.error":
            self.store.update(run_id, status="oom" if event.get("code") == "OOM" else "failed",
                              error=str(event.get("message") or "Training failed."), finished_at=time.time())
            self.job_finished.emit(run_id, False)
        elif kind in ("training.stopped", "training.paused"):
            self.store.update(run_id, status="stopped", output_path=event.get("output_path"),
                              manifest_json=json.dumps(event.get("manifest") or {}, default=str),
                              finished_at=time.time())
            if event.get("output_path"):
                self._register_artifact(run_id)
            self.job_finished.emit(run_id, False)
        elif kind == "training.completed":
            self.store.update(run_id, status="completed", output_path=event.get("output_path"),
                              manifest_json=json.dumps(event.get("manifest") or {}, default=str),
                              finished_at=time.time())
            self._register_artifact(run_id)
            self.job_finished.emit(run_id, True)
        elif kind == "training.started":
            self.store.update(run_id, status="training", started_at=time.time())
            self.job_updated.emit(run_id)

    def _register_artifact(self, run_id: str) -> None:
        run = self.store.get(run_id)
        if run is None or not run.get("output_path"):
            return
        config = run.get("config") or {}
        model = config.get("model") if isinstance(config.get("model"), dict) else {}
        base = str(model.get("id") or "")
        output = Path(str(run["output_path"]))
        size = sum(item.stat().st_size for item in output.rglob("*") if item.is_file()) if output.exists() else 0
        try:
            if str(run.get("mode")) in ("lora", "qlora"):
                from noelle.app.services.model_registry import AdapterRecord

                adapter_id = f"{run.get('output_name') or 'adapter'}-{run_id[-6:]}"
                self.registry.add_adapter(AdapterRecord(adapter_id, base or None, str(output), size))
            else:
                from noelle.app.services.model_registry import ModelRecord

                repo_id = f"trained/{run.get('output_name') or run_id}"
                self.registry.upsert_model(ModelRecord(
                    repo_id=repo_id, pipeline_tag="text-generation", local_path=str(output), source="local",
                    base_model=base or None, size_bytes=size, engine="transformers",
                    metadata={"training_run": run_id, "mode": run.get("mode"), "base_model": base}))
        except Exception:  # noqa: BLE001 - training output must remain usable even if registration fails
            log.exception("Could not register training output for %s", run_id)
            self._append_log(run_id, "ERROR: training finished but could not be registered in the library.")

    def _on_finished(self, run_id: str, exit_code: int) -> None:
        process = self._processes.pop(run_id, None)
        self._buffers.pop(run_id, None)
        if process is not None and _alive(process):
            process.deleteLater()
        run = self.store.get(run_id)
        terminal = ("completed", "failed", "oom", "stopped")
        if run is not None and run.get("status") not in terminal:
            stopping = run.get("status") == "stopping"
            status = "stopped" if stopping or exit_code == 0 else "failed"
            self.store.update(run_id, status=status, finished_at=time.time(),
                              error=run.get("error") or (None if status == "stopped" else "Training worker exited."))
            self.job_finished.emit(run_id, status == "completed")
        self.job_updated.emit(run_id)

    def _on_process_error(self, run_id: str, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        process = self._processes.pop(run_id, None)
        if process is not None and _alive(process):
            process.deleteLater()
        message = "The training worker could not start. Check the training environment settings."
        self.store.update(run_id, status="failed", error=message, finished_at=time.time())
        self._append_log(run_id, "ERROR: " + message)
        self.job_finished.emit(run_id, False)
        self.job_updated.emit(run_id)

    def _append_log(self, run_id: str, line: str) -> None:
        path = self.store.log_path(run_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line.rstrip() + "\n")
        except OSError:
            pass
        self.log_line.emit(run_id, line)