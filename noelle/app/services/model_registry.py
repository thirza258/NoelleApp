"""Local model registry (SQLite, WAL mode).

Tables:

* ``models``      downloaded Hub snapshots and derived checkpoints (quantized, local)
* ``adapters``    LoRA adapters and the base model they target
* ``evaluations`` comparative-bench results (one row per view per run)

Qt-free: the API server process reads it to answer ``GET /v1/models``.
All methods are thread-safe (one connection guarded by a lock).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from noelle import paths
from noelle.runtime import tasks

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id TEXT NOT NULL,
    revision TEXT,
    commit_hash TEXT,
    pipeline_tag TEXT,
    library_name TEXT,
    architecture TEXT,
    param_count INTEGER,
    license TEXT,
    local_path TEXT,
    precision TEXT NOT NULL DEFAULT 'auto',
    engine TEXT,
    gguf_file TEXT,
    source TEXT NOT NULL DEFAULT 'hub',
    base_model TEXT,
    size_bytes INTEGER,
    trust_remote_code INTEGER NOT NULL DEFAULT 0,
    downloaded_at REAL,
    last_used_at REAL,
    metadata TEXT,
    UNIQUE(repo_id, precision, source)
);
CREATE TABLE IF NOT EXISTS adapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adapter_id TEXT NOT NULL UNIQUE,
    base_model_id TEXT,
    local_path TEXT,
    size_bytes INTEGER,
    added_at REAL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    view INTEGER,
    model_id TEXT,
    prompt TEXT,
    system_prompt TEXT,
    params TEXT,
    output TEXT,
    status TEXT,
    ttft_ms REAL,
    tokens_per_second REAL,
    peak_memory_bytes INTEGER,
    precision TEXT,
    metrics TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class ModelRecord:
    repo_id: str
    pipeline_tag: str | None = None
    local_path: str | None = None
    precision: str = "auto"
    source: str = "hub"  # hub | quantized | local
    revision: str | None = None
    commit_hash: str | None = None
    library_name: str | None = None
    architecture: str | None = None
    param_count: int | None = None
    license: str | None = None
    engine: str | None = None
    gguf_file: str | None = None
    base_model: str | None = None
    size_bytes: int | None = None
    trust_remote_code: bool = False
    downloaded_at: float | None = None
    last_used_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    @property
    def family(self) -> str | None:
        return tasks.task_family(self.pipeline_tag)

    @property
    def api_name(self) -> str:
        """Name exposed by the OpenAI-compatible server."""
        if self.source == "hub" and self.precision == "auto":
            return self.repo_id
        return f"{self.repo_id}:{self.precision}" if self.source == "hub" else self.repo_id

    @property
    def display_name(self) -> str:
        name = self.repo_id.split("/")[-1]
        if self.source == "hub" and self.precision != "auto":
            name += f" ({self.precision})"
        return name

    def exists_on_disk(self) -> bool:
        return bool(self.local_path) and Path(self.local_path).exists()

    def worker_payload(self) -> dict[str, Any]:
        """Fields for LOAD_MODEL / GENERATE_STREAM requests."""
        payload: dict[str, Any] = {
            "model_id": self.repo_id,
            "task": self.pipeline_tag or "text-generation",
            "precision": self.precision,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.local_path:
            payload["local_path"] = self.local_path
        if self.engine:
            payload["engine"] = self.engine
        if self.gguf_file:
            payload["gguf_file"] = self.gguf_file
        return payload

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["family"] = self.family
        data["api_name"] = self.api_name
        return data


@dataclass
class AdapterRecord:
    adapter_id: str
    base_model_id: str | None
    local_path: str | None
    size_bytes: int | None = None
    added_at: float | None = None
    id: int | None = None


class ModelRegistry:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else paths.registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                               (str(SCHEMA_VERSION),))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return int(cur.lastrowid or 0)

    # ------------------------------------------------------------- models
    def upsert_model(self, record: ModelRecord) -> ModelRecord:
        record.downloaded_at = record.downloaded_at or time.time()
        values = (
            record.repo_id, record.revision, record.commit_hash, record.pipeline_tag, record.library_name,
            record.architecture, record.param_count, record.license, record.local_path, record.precision,
            record.engine, record.gguf_file, record.source, record.base_model, record.size_bytes,
            int(record.trust_remote_code), record.downloaded_at, record.last_used_at,
            json.dumps(record.metadata or {}),
        )
        with self._lock:
            self._conn.execute(
                """INSERT INTO models (repo_id, revision, commit_hash, pipeline_tag, library_name, architecture,
                       param_count, license, local_path, precision, engine, gguf_file, source, base_model, size_bytes,
                       trust_remote_code, downloaded_at, last_used_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(repo_id, precision, source) DO UPDATE SET
                       revision=excluded.revision, commit_hash=excluded.commit_hash,
                       pipeline_tag=COALESCE(excluded.pipeline_tag, models.pipeline_tag),
                       library_name=COALESCE(excluded.library_name, models.library_name),
                       architecture=COALESCE(excluded.architecture, models.architecture),
                       param_count=COALESCE(excluded.param_count, models.param_count),
                       license=COALESCE(excluded.license, models.license),
                       local_path=excluded.local_path, engine=excluded.engine, gguf_file=excluded.gguf_file,
                       base_model=COALESCE(excluded.base_model, models.base_model),
                       size_bytes=excluded.size_bytes, trust_remote_code=excluded.trust_remote_code,
                       downloaded_at=excluded.downloaded_at, metadata=excluded.metadata""",
                values,
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM models WHERE repo_id=? AND precision=? AND source=?",
                                     (record.repo_id, record.precision, record.source)).fetchone()
        return _model_from_row(row)

    def get_model(self, repo_id: str, precision: str | None = None, source: str | None = None) -> ModelRecord | None:
        sql = "SELECT * FROM models WHERE repo_id=?"
        params: list[Any] = [repo_id]
        if precision:
            sql += " AND precision=?"
            params.append(precision)
        if source:
            sql += " AND source=?"
            params.append(source)
        sql += " ORDER BY CASE source WHEN 'hub' THEN 0 ELSE 1 END, downloaded_at DESC LIMIT 1"
        rows = self._rows(sql, tuple(params))
        return _model_from_row(rows[0]) if rows else None

    def get_model_by_id(self, model_pk: int) -> ModelRecord | None:
        rows = self._rows("SELECT * FROM models WHERE id=?", (model_pk,))
        return _model_from_row(rows[0]) if rows else None

    def find_by_api_name(self, name: str) -> ModelRecord | None:
        records = self.list_models()
        for record in records:  # exact API name first ("repo" must not resolve to "repo:fp16")
            if record.api_name == name:
                return record
        for record in records:
            if record.repo_id == name:
                return record
        if ":" in name:
            repo, precision = name.rsplit(":", 1)
            return self.get_model(repo, precision)
        return None

    def list_models(self, *, family: str | None = None, pipeline_tags: tuple[str, ...] | None = None,
                    only_existing: bool = False) -> list[ModelRecord]:
        records = [_model_from_row(r) for r in self._rows(
            "SELECT * FROM models ORDER BY COALESCE(last_used_at, downloaded_at) DESC")]
        if family:
            records = [r for r in records if r.family == family]
        if pipeline_tags:
            records = [r for r in records if r.pipeline_tag in pipeline_tags]
        if only_existing:
            records = [r for r in records if r.exists_on_disk()]
        return records

    def is_downloaded(self, repo_id: str) -> bool:
        record = self.get_model(repo_id)
        return record is not None and record.exists_on_disk()

    def touch_model(self, repo_id: str) -> None:
        self._write("UPDATE models SET last_used_at=? WHERE repo_id=?", (time.time(), repo_id))

    def set_trust_remote_code(self, repo_id: str, trusted: bool) -> None:
        self._write("UPDATE models SET trust_remote_code=? WHERE repo_id=?", (int(trusted), repo_id))

    def delete_model(self, repo_id: str, precision: str | None = None, source: str | None = None) -> int:
        sql = "DELETE FROM models WHERE repo_id=?"
        params: list[Any] = [repo_id]
        if precision:
            sql += " AND precision=?"
            params.append(precision)
        if source:
            sql += " AND source=?"
            params.append(source)
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur.rowcount

    def prune_missing(self) -> list[str]:
        """Drop records whose files were deleted outside Noelle (e.g. by the cache pruner)."""
        removed = []
        for record in self.list_models():
            if record.local_path and not Path(record.local_path).exists():
                self.delete_model(record.repo_id, record.precision, record.source)
                removed.append(record.repo_id)
        return removed

    # ----------------------------------------------------------- adapters
    def add_adapter(self, record: AdapterRecord) -> AdapterRecord:
        record.added_at = record.added_at or time.time()
        self._write("""INSERT INTO adapters (adapter_id, base_model_id, local_path, size_bytes, added_at)
                       VALUES (?,?,?,?,?) ON CONFLICT(adapter_id) DO UPDATE SET
                       base_model_id=excluded.base_model_id, local_path=excluded.local_path,
                       size_bytes=excluded.size_bytes""",
                    (record.adapter_id, record.base_model_id, record.local_path, record.size_bytes, record.added_at))
        return self.list_adapters(adapter_id=record.adapter_id)[0]

    def list_adapters(self, base_model_id: str | None = None, adapter_id: str | None = None) -> list[AdapterRecord]:
        sql = "SELECT * FROM adapters"
        params: tuple[Any, ...] = ()
        if adapter_id:
            sql += " WHERE adapter_id=?"
            params = (adapter_id,)
        elif base_model_id:
            sql += " WHERE base_model_id=? OR base_model_id IS NULL"
            params = (base_model_id,)
        sql += " ORDER BY added_at DESC"
        return [AdapterRecord(r["adapter_id"], r["base_model_id"], r["local_path"], r["size_bytes"], r["added_at"],
                              r["id"]) for r in self._rows(sql, params)]

    def delete_adapter(self, adapter_id: str) -> None:
        self._write("DELETE FROM adapters WHERE adapter_id=?", (adapter_id,))

    # -------------------------------------------------------- evaluations
    def add_evaluation(self, run_id: str, view: int, model_id: str, *, prompt: str, system_prompt: str | None,
                       params: dict[str, Any], output: str, status: str, metrics: dict[str, Any]) -> None:
        self._write(
            """INSERT INTO evaluations (run_id, view, model_id, prompt, system_prompt, params, output, status, ttft_ms,
                   tokens_per_second, peak_memory_bytes, precision, metrics, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, view, model_id, prompt, system_prompt, json.dumps(params), output, status,
             metrics.get("ttft_ms"), metrics.get("tokens_per_second"),
             metrics.get("model_peak_bytes") or metrics.get("peak_memory_bytes"), metrics.get("precision"),
             json.dumps(metrics), time.time()))

    def list_evaluations(self, run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if run_id:
            rows = self._rows("SELECT * FROM evaluations WHERE run_id=? ORDER BY view", (run_id,))
        else:
            rows = self._rows("SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (limit,))
        out = []
        for row in rows:
            item = dict(row)
            for key in ("params", "metrics"):
                try:
                    item[key] = json.loads(item[key]) if item[key] else {}
                except ValueError:
                    item[key] = {}
            out.append(item)
        return out

    def iter_models(self) -> Iterator[ModelRecord]:
        yield from self.list_models()


def _model_from_row(row: sqlite3.Row) -> ModelRecord:
    try:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    except ValueError:
        metadata = {}
    return ModelRecord(
        id=row["id"], repo_id=row["repo_id"], revision=row["revision"], commit_hash=row["commit_hash"],
        pipeline_tag=row["pipeline_tag"], library_name=row["library_name"], architecture=row["architecture"],
        param_count=row["param_count"], license=row["license"], local_path=row["local_path"],
        precision=row["precision"], engine=row["engine"], gguf_file=row["gguf_file"], source=row["source"],
        base_model=row["base_model"], size_bytes=row["size_bytes"], trust_remote_code=bool(row["trust_remote_code"]),
        downloaded_at=row["downloaded_at"], last_used_at=row["last_used_at"], metadata=metadata,
    )


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    with _registry_lock:
        if _registry is None or _registry.path != paths.registry_path():
            _registry = ModelRegistry()
        return _registry
