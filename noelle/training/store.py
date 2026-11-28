"""SQLite tracking for training runs, metrics and checkpoints."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from noelle import paths

log = logging.getLogger("noelle.training.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    model_id TEXT,
    dataset_path TEXT,
    dataset_sha TEXT,
    output_name TEXT,
    base_model TEXT,
    config_json TEXT,
    manifest_json TEXT,
    output_path TEXT,
    epoch REAL,
    step INTEGER,
    total_steps INTEGER,
    loss REAL,
    learning_rate REAL,
    tokens_per_second REAL,
    vram_mb REAL,
    pid INTEGER,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    step INTEGER,
    total_steps INTEGER,
    epoch REAL,
    loss REAL,
    learning_rate REAL,
    tokens_per_second REAL,
    vram_mb REAL,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_metrics_run ON training_metrics(run_id, step);
CREATE TABLE IF NOT EXISTS training_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    step INTEGER,
    loss REAL,
    size_bytes INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_checkpoints_run ON training_checkpoints(run_id, step);
"""


def slugify(name: str, fallback: str = "training") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (name or "")).strip("-._")
    return (cleaned or fallback)[:80]


class TrainingStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else paths.home() / "training.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------- paths
    def root(self) -> Path:
        return paths.projects_dir() / "training"

    def run_dir(self, run_id: str) -> Path:
        return self.root() / run_id

    def dataset_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "datasets"

    def config_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "config.json"

    def log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "logs" / "training.log"

    def checkpoints_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "checkpoints"

    def final_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "final"

    def ensure_run_dirs(self, run_id: str) -> Path:
        for folder in (self.run_dir(run_id), self.dataset_dir(run_id), self.checkpoints_dir(run_id),
                       self.run_dir(run_id) / "logs", self.run_dir(run_id) / "exports"):
            folder.mkdir(parents=True, exist_ok=True)
        return self.run_dir(run_id)

    # ------------------------------------------------------------------ CRUD
    def create(self, config: Any, *, run_id: str | None = None, status: str = "preparing",
               dataset_sha: str | None = None) -> dict[str, Any]:
        run_id = run_id or f"{slugify(config.output_name)}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        now = time.time()
        data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        with self._lock:
            self._conn.execute(
                """INSERT INTO training_runs
                   (id, name, status, mode, model_id, dataset_path, dataset_sha, output_name, base_model,
                    config_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, getattr(config, "name", run_id), status, getattr(config, "mode", ""),
                 getattr(getattr(config, "model", None), "id", None),
                 getattr(getattr(config, "dataset", None), "path", None), dataset_sha,
                 getattr(config, "output_name", run_id), getattr(getattr(config, "model", None), "id", None),
                 json.dumps(data, default=str), now))
            self._conn.commit()
        self.ensure_run_dirs(run_id)
        (self.run_dir(run_id) / "config.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return self.get(run_id) or {}

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
        return _run(row) if row else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?",
                                      (limit,)).fetchall()
        return [_run(row) for row in rows]

    def update(self, run_id: str, **fields: Any) -> None:
        allowed = {"status", "manifest_json", "output_path", "epoch", "step", "total_steps", "loss",
                   "learning_rate", "tokens_per_second", "vram_mb", "pid", "error", "started_at", "finished_at",
                   "dataset_sha"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self._lock:
            self._conn.execute(f"UPDATE training_runs SET {assignments} WHERE id=?",
                               (*values.values(), run_id))
            self._conn.commit()

    def delete(self, run_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM training_runs WHERE id=?", (run_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # --------------------------------------------------------------- metrics
    def add_metric(self, run_id: str, payload: dict[str, Any]) -> None:
        now = time.time()
        step = _int(payload.get("step"))
        total = _int(payload.get("total_steps"))
        epoch = _float(payload.get("epoch"))
        loss = _float(payload.get("loss"))
        lr = _float(payload.get("learning_rate"))
        tps = _float(payload.get("tokens_per_second"))
        vram = _float(payload.get("gpu_memory_mb"))
        with self._lock:
            self._conn.execute(
                """INSERT INTO training_metrics
                   (run_id, created_at, step, total_steps, epoch, loss, learning_rate, tokens_per_second, vram_mb,
                    payload_json) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (run_id, now, step, total, epoch, loss, lr, tps, vram, json.dumps(payload, default=str)))
            self._conn.execute(
                """UPDATE training_runs SET step=COALESCE(?, step), total_steps=COALESCE(?, total_steps),
                       epoch=COALESCE(?, epoch), loss=COALESCE(?, loss), learning_rate=COALESCE(?, learning_rate),
                       tokens_per_second=COALESCE(?, tokens_per_second), vram_mb=COALESCE(?, vram_mb)
                   WHERE id=?""",
                (step, total, epoch, loss, lr, tps, vram, run_id))
            self._conn.commit()

    def metrics(self, run_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM training_metrics WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    # ----------------------------------------------------------- checkpoints
    def add_checkpoint(self, run_id: str, path: str | Path, *, step: int | None = None, loss: float | None = None,
                       size_bytes: int | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO training_checkpoints (run_id, path, step, loss, size_bytes, created_at) VALUES (?,?,?,?,?,?)",
                (run_id, str(path), step, loss, size_bytes, time.time()))
            self._conn.commit()

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM training_checkpoints WHERE run_id=? ORDER BY COALESCE(step, 0)",
                (run_id,)).fetchall()
        return [dict(row) for row in rows]

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        rows = self.checkpoints(run_id)
        return rows[-1] if rows else None


def _run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    try:
        data["config"] = json.loads(data.pop("config_json") or "{}")
    except ValueError:
        data["config"] = {}
    try:
        data["manifest"] = json.loads(data.pop("manifest_json") or "{}")
    except ValueError:
        data["manifest"] = {}
    return data


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _float(value: Any) -> float | None:
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number is None or (number == number and abs(number) != float("inf")) else None