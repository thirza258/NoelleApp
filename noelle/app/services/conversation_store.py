"""Persistent chat history and prompt presets (``~/.noelle/chats.db``).

Separate from the model registry so "Delete all chats" removes one file and a chat
database problem can never quarantine model metadata. Writes are caller-controlled:
private mode simply never calls the store.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from noelle import paths

log = logging.getLogger("noelle.chats")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    model_id TEXT,
    adapter TEXT,
    system_prompt TEXT,
    prompt_id INTEGER,
    params_json TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    metrics_json TEXT,
    audio_path TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    tags TEXT,
    builtin INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
"""
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""
BUILTIN_PROMPTS = (
    ("Helpful assistant", "You are a helpful, accurate assistant. Say when you do not know."),
    ("Penerjemah Indonesia ↔ English", "Translate faithfully between Indonesian and English. Preserve meaning, "
                                       "tone and formatting. If the target language is ambiguous, ask."),
    ("Code reviewer", "Review the code for correctness, security, performance and clarity. Point to exact lines "
                      "and suggest minimal fixes."),
    ("Summarizer", "Summarize the text faithfully. Keep names, numbers and decisions. Use short paragraphs and "
                   "bullet points for the key facts."),
    ("Guru", "Explain the topic simply, in Indonesian, with a concrete example and a short check-your-understanding "
             "question at the end."),
)


class ConversationStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else paths.home() / "chats.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self.fts = True
            try:
                self._conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError:
                self.fts = False
                log.info("SQLite FTS5 is unavailable; chat search falls back to LIKE.")
            self._conn.commit()
        self._seed_prompts()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ conversations
    def create_conversation(self, title: str = "", *, model_id: str | None = None, adapter: str | None = None,
                            system_prompt: str | None = None, prompt_id: int | None = None,
                            params: dict[str, Any] | None = None) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO conversations (title, model_id, adapter, system_prompt, prompt_id, params_json,
                                              created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                (title.strip() or "New chat", model_id, adapter, system_prompt, prompt_id,
                 json.dumps(params or {}), now, now))
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def list_conversations(self, *, search: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if search and search.strip():
            if self.fts:
                where = ("WHERE c.id IN (SELECT m.conversation_id FROM messages m JOIN messages_fts f "
                         "ON f.rowid = m.id WHERE messages_fts MATCH ?)")
                params.append(_fts_query(search))
            else:
                where = "WHERE c.id IN (SELECT conversation_id FROM messages WHERE content LIKE ?)"
                params.append(f"%{search.strip()}%")
        params.append(limit)
        sql = (f"""SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                   FROM conversations c {where}
                   ORDER BY c.pinned DESC, c.updated_at DESC LIMIT ?""")
        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                # FTS query syntax varies with arbitrary user input; fall back to LIKE instead
                # of failing the sidebar.
                if not (search and search.strip()):
                    raise
                like = f"%{search.strip()}%"
                rows = self._conn.execute(
                    """SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                       FROM conversations c
                       WHERE c.id IN (SELECT conversation_id FROM messages WHERE content LIKE ?)
                       ORDER BY c.pinned DESC, c.updated_at DESC LIMIT ?""", (like, limit)).fetchall()
        return [_conversation(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if row is None:
                return None
            messages = self._conn.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
                                          (conversation_id,)).fetchall()
        data = _conversation(row)
        data["messages"] = [_message(item) for item in messages]
        return data

    def rename_conversation(self, conversation_id: int, title: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                                     (title.strip()[:200] or "New chat", time.time(), conversation_id))
            self._conn.commit()
            return cur.rowcount > 0

    def pin_conversation(self, conversation_id: int, pinned: bool = True) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE conversations SET pinned=? WHERE id=?",
                                     (int(bool(pinned)), conversation_id))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_conversation(self, conversation_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_all(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM conversations")
            self._conn.commit()
            return int(cur.rowcount or 0)

    # ---------------------------------------------------------------- messages
    def add_message(self, conversation_id: int, role: str, content: str, *, sources: Any = None,
                    metrics: Any = None, audio_path: str | None = None) -> int:
        if role not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"Unsupported message role: {role!r}")
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO messages (conversation_id, role, content, sources_json, metrics_json, audio_path,
                                         created_at) VALUES (?,?,?,?,?,?,?)""",
                (conversation_id, role, str(content), json.dumps(sources) if sources is not None else None,
                 json.dumps(metrics) if metrics is not None else None, audio_path, now))
            self._conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            if role == "user":
                # Auto-title from the first user message, trimmed to a word boundary.
                title = " ".join(str(content).split())[:60]
                if title:
                    self._conn.execute(
                        "UPDATE conversations SET title=? WHERE id=? AND (title='' OR title='New chat')",
                        (title.rsplit(" ", 1)[0] + "…" if len(title) == 60 and " " in title else title,
                         conversation_id))
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
                                      (conversation_id,)).fetchall()
        return [_message(row) for row in rows]

    # ----------------------------------------------------------------- prompts
    def _seed_prompts(self) -> None:
        now = time.time()
        with self._lock:
            for name, content in BUILTIN_PROMPTS:
                self._conn.execute(
                    """INSERT OR IGNORE INTO prompts (name, content, tags, builtin, updated_at)
                       VALUES (?,?,?,?,?)""", (name, content, "builtin", 1, now))
            self._conn.commit()

    def list_prompts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM prompts ORDER BY builtin DESC, name").fetchall()
        return [dict(row) for row in rows]

    def save_prompt(self, name: str, content: str, *, tags: str | None = None, prompt_id: int | None = None) -> int:
        now = time.time()
        with self._lock:
            if prompt_id is None:
                cur = self._conn.execute(
                    """INSERT INTO prompts (name, content, tags, builtin, updated_at) VALUES (?,?,?,?,?)
                       ON CONFLICT(name) DO UPDATE SET content=excluded.content, tags=excluded.tags,
                                                       updated_at=excluded.updated_at""",
                    (name.strip()[:120], content, tags, 0, now))
                prompt_id = int(cur.lastrowid or 0)
            else:
                self._conn.execute("UPDATE prompts SET name=?, content=?, tags=?, updated_at=? WHERE id=? AND builtin=0",
                                   (name.strip()[:120], content, tags, now, prompt_id))
            self._conn.commit()
        return prompt_id

    def delete_prompt(self, prompt_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM prompts WHERE id=? AND builtin=0", (prompt_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ----------------------------------------------------------------- exports
    def export(self, conversation_id: int, fmt: str = "markdown") -> str:
        data = self.get_conversation(conversation_id)
        if data is None:
            return ""
        if fmt == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        lines = [f"# {data['title']}", ""]
        if data.get("model_id"):
            lines.append(f"Model: `{data['model_id']}`")
        for message in data["messages"]:
            lines += ["", f"**{message['role']}**", "", message["content"]]
        return "\n".join(lines)


def _fts_query(text: str) -> str:
    """Quote each term so user input cannot inject FTS operators."""
    terms = [term for term in text.replace('"', " ").split() if term]
    return " AND ".join(f'"{term}"' for term in terms) or '""'


def _conversation(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["pinned"] = bool(data.get("pinned"))
    try:
        data["params"] = json.loads(data.pop("params_json") or "{}")
    except ValueError:
        data["params"] = {}
        data.pop("params_json", None)
    return data


def _message(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key, target in (("sources_json", "sources"), ("metrics_json", "metrics")):
        raw = data.pop(key, None)
        try:
            data[target] = json.loads(raw) if raw else None
        except ValueError:
            data[target] = None
    return data