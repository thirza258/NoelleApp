"""SQLite token and cost ledger for cloud provider calls (WAL, thread-safe).

One row per completed provider request. Costs come from :mod:`noelle.providers.pricing`;
unknown models are recorded with cost 0 and ``pricing_missing = 1`` rather than guessing.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from noelle import paths
from noelle.providers.pricing import calculate_cost, has_published_price

log = logging.getLogger("noelle.providers.cost")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    pricing_missing INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_token_transactions_time ON token_transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_token_transactions_provider ON token_transactions(provider, model);
"""


class CostLedger:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else paths.home() / "cost_ledger.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record_transaction(self, provider: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0,
                           latency_ms: float = 0.0, *, cost_usd: float | None = None,
                           timestamp: float | None = None) -> dict[str, Any]:
        prompt_tokens = max(int(prompt_tokens or 0), 0)
        completion_tokens = max(int(completion_tokens or 0), 0)
        missing = not has_published_price(model)
        if cost_usd is None:
            cost_usd = calculate_cost(model, input_tokens=prompt_tokens, output_tokens=completion_tokens)
        row = (float(timestamp if timestamp is not None else time.time()), str(provider), str(model),
               prompt_tokens, completion_tokens, max(float(latency_ms or 0.0), 0.0), float(cost_usd or 0.0),
               int(missing))
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO token_transactions
                   (timestamp, provider, model, prompt_tokens, completion_tokens, latency_ms, cost_usd,
                    pricing_missing) VALUES (?,?,?,?,?,?,?,?)""", row)
            self._conn.commit()
            record_id = int(cur.lastrowid or 0)
        return {"id": record_id, "provider": provider, "model": model, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "latency_ms": float(latency_ms or 0.0),
                "cost_usd": float(cost_usd or 0.0), "pricing_missing": missing}

    def _summary(self, since: float | None = None) -> dict[str, Any]:
        where = "WHERE timestamp >= ?" if since is not None else ""
        params: tuple[Any, ...] = (since,) if since is not None else ()
        with self._lock:
            row = self._conn.execute(
                f"""SELECT COUNT(*) AS calls, COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                           COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                           COALESCE(SUM(cost_usd),0) AS cost_usd,
                           COALESCE(AVG(latency_ms),0) AS latency_ms,
                           COALESCE(SUM(pricing_missing),0) AS pricing_missing
                    FROM token_transactions {where}""", params).fetchone()
        total_tokens = int(row["prompt_tokens"]) + int(row["completion_tokens"])
        return {"calls": int(row["calls"]), "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]), "total_tokens": total_tokens,
                "cost_usd": round(float(row["cost_usd"]), 6), "avg_latency_ms": round(float(row["latency_ms"]), 2),
                "pricing_missing": int(row["pricing_missing"])}

    def get_session_summary(self) -> dict[str, Any]:
        return self._summary()

    def get_monthly_burn_rate(self) -> float:
        return float(self._summary(time.time() - 30 * 86_400)["cost_usd"])

    def by_provider(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT provider, model, COUNT(*) AS calls, SUM(prompt_tokens + completion_tokens) AS tokens,
                          SUM(cost_usd) AS cost_usd FROM token_transactions
                   GROUP BY provider, model ORDER BY cost_usd DESC""").fetchall()
        return [dict(row) for row in rows]

    def delete_all(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM token_transactions")
            self._conn.commit()
            return int(cur.rowcount or 0)


_ledger: CostLedger | None = None
_ledger_lock = threading.Lock()


def get_ledger() -> CostLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None or _ledger.path != paths.home() / "cost_ledger.db":
            _ledger = CostLedger()
        return _ledger