"""Local model registry (SQLite, WAL mode).

Tables:

* ``models``      downloaded Hub snapshots and derived checkpoints (quantized, local)
* ``adapters``    LoRA adapters and the base model they target
* ``evaluations`` comparative-bench results (one row per view per run)
* ``experiments`` / ``experiment_runs`` / ``experiment_cases``  Evaluation Lab runs with the
  dataset digest, generation config, hardware and per-model provenance (repo revision,
  commit, precision, engine), so a result can be traced to exactly what produced it

Qt-free: the API server process reads it to answer ``GET /v1/models``.
All methods are thread-safe (one connection guarded by a lock).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from noelle import paths
from noelle.runtime import tasks

SCHEMA_VERSION = 2
log = logging.getLogger("noelle.registry")

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
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT,
    status TEXT,
    created_at REAL,
    finished_at REAL,
    dataset TEXT,
    params TEXT,
    hardware TEXT,
    summary TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    model_id TEXT,
    api_name TEXT,
    precision TEXT,
    engine TEXT,
    device TEXT,
    status TEXT,
    accuracy REAL,
    mean_score REAL,
    ttft_ms REAL,
    tokens_per_second REAL,
    peak_memory_bytes INTEGER,
    load_ms REAL,
    summary TEXT,
    metrics TEXT,
    provenance TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS experiment_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id INTEGER,
    case_id TEXT,
    prompt TEXT,
    expected TEXT,
    output TEXT,
    score REAL,
    passed INTEGER,
    detail TEXT,
    ttft_ms REAL,
    tokens_per_second REAL,
    completion_tokens INTEGER,
    latency_ms REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiment_runs ON experiment_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_model ON experiment_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_experiment_cases ON experiment_cases(experiment_id, run_id);
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

    def worker_payload(self, settings: Any = None) -> dict[str, Any]:
        """Fields for LOAD_MODEL / GENERATE_STREAM requests.

        Pass the settings so the trust the user gave in the Workspace (a repository or a whole
        publisher) applies everywhere the model runs — Evaluate, Compare, Mission, CLI and API.
        """
        trusted = self.trust_remote_code or (settings is not None and remote_code_trusted(
            settings, self.repo_id, self.base_model))
        payload: dict[str, Any] = {
            "model_id": self.repo_id,
            "task": self.pipeline_tag or "text-generation",
            "precision": self.precision,
            "trust_remote_code": bool(trusted),
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


def remote_code_trusted(settings: Any, repo_id: str | None, base_model: str | None = None) -> bool:
    """True when the user trusts this repository (or its base model) or its publisher to run custom code."""
    repos = set(settings.get("runtime.trusted_remote_code") or [])
    publishers = set(settings.get("runtime.trusted_publishers") or [])
    for candidate in (repo_id, base_model):
        if candidate and (candidate in repos or ("/" in candidate and candidate.split("/")[0] in publishers)):
            return True
    return False


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
        self.recovered_path: Path | None = None
        try:
            self._open()
        except sqlite3.DatabaseError as exc:
            # A truncated or corrupt registry must not make Noelle unstartable. Keep the
            # unreadable file for diagnostics (never delete user data) and start fresh.
            self.recovered_path = self._quarantine()
            log.error("The model registry %s was unreadable (%s); it was moved to %s and a new one was created.",
                      self.path, exc, self.recovered_path)
            self._open()

    def _open(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                               (str(SCHEMA_VERSION),))
            self._conn.execute("UPDATE meta SET value=? WHERE key='schema_version' AND CAST(value AS INTEGER) < ?",
                               (str(SCHEMA_VERSION), SCHEMA_VERSION))
            self._conn.commit()

    def _quarantine(self) -> Path:
        """Close the broken database and rename it (and its WAL sidecars) out of the way."""
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{os.getpid()}")
        suffix = 1
        while target.exists() or any(target.with_name(target.name + tail).exists() for tail in ("-wal", "-shm")):
            suffix += 1
            target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{os.getpid()}-{suffix}")
        for source in (self.path, self.path.with_name(self.path.name + "-wal"),
                       self.path.with_name(self.path.name + "-shm")):
            if source.exists():
                try:
                    source.replace(target.with_name(target.name + source.name[len(self.path.name):]))
                except OSError as exc:
                    raise sqlite3.DatabaseError(f"could not quarantine {source}: {exc}") from exc
        return target

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
        log.info("registry: stored %s (%s, %s)", record.repo_id, record.precision, record.source)
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
            log.info("registry: removed %d record(s) for %s", cur.rowcount, repo_id)
            return cur.rowcount

    def prune_missing(self) -> list[str]:
        """Drop records whose files were deleted outside Noelle (e.g. by the cache pruner)."""
        removed = []
        for record in self.list_models():
            if record.local_path and not Path(record.local_path).exists():
                log.info("registry: pruning %s because %s no longer exists", record.repo_id, record.local_path)
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

    def measurements(self, model_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Speed measurements for one model on this machine (Compare + Evaluation Lab runs), newest first."""
        rows = self._rows("""SELECT precision, ttft_ms, tokens_per_second, status, metrics, created_at FROM evaluations
                             WHERE model_id=?
                             UNION ALL
                             SELECT r.precision, r.ttft_ms, r.tokens_per_second, r.status, r.metrics, e.created_at
                             FROM experiment_runs r JOIN experiments e ON e.id = r.experiment_id
                             WHERE r.model_id=? AND e.kind != 'compare'
                             ORDER BY created_at DESC LIMIT ?""", (model_id, model_id, limit))
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else {}
            except ValueError:
                item["metrics"] = {}
            out.append(item)
        return out

    # -------------------------------------------------------- experiments
    def create_experiment(self, *, kind: str, name: str, dataset: dict[str, Any], params: dict[str, Any],
                          hardware: dict[str, Any], notes: str | None = None) -> str:
        experiment_id = uuid.uuid4().hex[:12]
        self._write("""INSERT INTO experiments (id, kind, name, status, created_at, dataset, params, hardware, notes)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (experiment_id, kind, name, "running", time.time(), json.dumps(dataset), json.dumps(params),
                     json.dumps(hardware), notes))
        return experiment_id

    def add_experiment_run(self, experiment_id: str, run: dict[str, Any]) -> int:
        summary = run.get("summary") or {}
        metrics = run.get("metrics") or {}
        return self._write(
            """INSERT INTO experiment_runs (experiment_id, model_id, api_name, precision, engine, device, status,
                   accuracy, mean_score, ttft_ms, tokens_per_second, peak_memory_bytes, load_ms, summary, metrics,
                   provenance, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, run.get("model_id"), run.get("api_name") or run.get("model_id"), run.get("precision"),
             metrics.get("engine"), metrics.get("device"), run.get("status"), summary.get("accuracy"),
             summary.get("mean_score"), summary.get("ttft_ms_p50"), summary.get("tokens_per_second"),
             metrics.get("peak_bytes"), metrics.get("load_ms"), json.dumps(summary), json.dumps(metrics),
             json.dumps(run.get("provenance") or {}), json.dumps(run.get("error")) if run.get("error") else None))

    def add_experiment_cases(self, experiment_id: str, run_id: int, cases: list[dict[str, Any]]) -> None:
        rows = [(experiment_id, run_id, c.get("case_id"), c.get("prompt"),
                 json.dumps(c.get("expected")) if c.get("expected") is not None else None, c.get("output"),
                 c.get("score"), None if c.get("passed") is None else int(bool(c.get("passed"))), c.get("detail"),
                 c.get("ttft_ms"), c.get("tokens_per_second"), c.get("completion_tokens"), c.get("latency_ms"),
                 c.get("error")) for c in cases]
        with self._lock:
            self._conn.executemany(
                """INSERT INTO experiment_cases (experiment_id, run_id, case_id, prompt, expected, output, score,
                       passed, detail, ttft_ms, tokens_per_second, completion_tokens, latency_ms, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            self._conn.commit()

    def finish_experiment(self, experiment_id: str, status: str, summary: dict[str, Any] | None = None) -> None:
        self._write("UPDATE experiments SET status=?, finished_at=?, summary=? WHERE id=?",
                    (status, time.time(), json.dumps(summary or {}), experiment_id))

    def list_experiments(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM experiments"
        params: tuple[Any, ...] = ()
        if kind:
            sql += " WHERE kind=?"
            params = (kind,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        experiments = [_experiment(row) for row in self._rows(sql, (*params, limit))]
        for item in experiments:
            item["runs"] = [_run(row) for row in self._rows(
                "SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY id", (item["id"],))]
        return experiments

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM experiments WHERE id=?", (experiment_id,))
        if not rows:
            return None
        item = _experiment(rows[0])
        item["runs"] = []
        for row in self._rows("SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY id", (experiment_id,)):
            run = _run(row)
            run["cases"] = [_case(c) for c in self._rows(
                "SELECT * FROM experiment_cases WHERE run_id=? ORDER BY id", (run["id"],))]
            item["runs"].append(run)
        return item

    def delete_experiment(self, experiment_id: str) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM experiment_cases WHERE experiment_id=?", (experiment_id,))
            self._conn.execute("DELETE FROM experiment_runs WHERE experiment_id=?", (experiment_id,))
            cur = self._conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def performance_matrix(self) -> list[dict[str, Any]]:
        """Measured performance per model x precision x engine x device, from every finished run on this machine."""
        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in self._rows("""SELECT r.*, e.dataset, e.created_at FROM experiment_runs r
                                 JOIN experiments e ON e.id = r.experiment_id
                                 WHERE r.status='finished' ORDER BY e.created_at"""):
            key = (row["model_id"], row["precision"], row["engine"], row["device"])
            group = groups.setdefault(key, {"model_id": row["model_id"], "precision": row["precision"],
                                            "engine": row["engine"], "device": row["device"], "runs": 0, "tps": [],
                                            "ttft": [], "peak": [], "accuracy": {}, "last_run": None})
            group["runs"] += 1
            if row["tokens_per_second"]:
                group["tps"].append(row["tokens_per_second"])
            if row["ttft_ms"]:
                group["ttft"].append(row["ttft_ms"])
            if row["peak_memory_bytes"]:
                group["peak"].append(row["peak_memory_bytes"])
            if row["accuracy"] is not None:
                try:
                    dataset = json.loads(row["dataset"] or "{}").get("name") or "dataset"
                except ValueError:
                    dataset = "dataset"
                group["accuracy"][dataset] = row["accuracy"]  # latest wins (rows are oldest first)
            group["last_run"] = row["created_at"]
        matrix = []
        for group in groups.values():
            matrix.append({
                "model_id": group["model_id"], "precision": group["precision"], "engine": group["engine"],
                "device": group["device"], "runs": group["runs"], "last_run": group["last_run"],
                "tokens_per_second": statistics.median(group["tps"]) if group["tps"] else None,
                "ttft_ms": statistics.median(group["ttft"]) if group["ttft"] else None,
                "peak_memory_bytes": max(group["peak"]) if group["peak"] else None,
                "accuracy": group["accuracy"],
            })
        matrix.sort(key=lambda m: (m["model_id"] or "", m["precision"] or ""))
        return matrix

    def iter_models(self) -> Iterator[ModelRecord]:
        yield from self.list_models()


def _loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except ValueError:
        return default


def _experiment(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("dataset", "params", "hardware", "summary"):
        item[key] = _loads(item.get(key), {})
    return item


def _run(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("summary", "metrics", "provenance"):
        item[key] = _loads(item.get(key), {})
    item["error"] = _loads(item.get("error"), None)
    return item


def _case(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["expected"] = _loads(item.get("expected"), None)
    item["passed"] = None if item.get("passed") is None else bool(item["passed"])
    return item


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
