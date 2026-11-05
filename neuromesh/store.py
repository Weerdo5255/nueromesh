"""Durable state: experience (labelled examples), capabilities, node lineage, tools,
orders, the mind (notes, inquiries, proposals), small key/value state (the temperament) and
the audit log, all in one SQLite file."""
from __future__ import annotations

import json
import sqlite3
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SOURCE_PRIORITY = {"human": 5, "tool-manifest": 4, "llm": 3, "heuristic": 1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS examples(
    id INTEGER PRIMARY KEY, capability TEXT, text TEXT, label TEXT, source TEXT,
    weight REAL, created_at REAL, UNIQUE(capability, text, source));
CREATE INDEX IF NOT EXISTS examples_cap ON examples(capability);
CREATE TABLE IF NOT EXISTS capabilities(name TEXT PRIMARY KEY, spec TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, capability TEXT, status TEXT, meta TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS tools(name TEXT PRIMARY KEY, state TEXT, digest TEXT, meta TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY, ts REAL, text TEXT, result TEXT, feedback TEXT);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, kind TEXT, subject TEXT, private INTEGER,
    data TEXT, updated_at REAL);
CREATE INDEX IF NOT EXISTS notes_subject ON notes(kind, subject);
CREATE TABLE IF NOT EXISTS inquiries(id TEXT PRIMARY KEY, status TEXT, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY, status TEXT, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts REAL, actor TEXT, kind TEXT, subject TEXT, detail TEXT);
"""


@dataclass
class Example:
    text: str
    label: str
    weight: float
    source: str


class Store:
    def __init__(self, path: Path, holdout_percent: int = 20, priorities: dict | None = None):
        """priorities: how much each source is trusted when two sources label the same text
        (the teacher weights from the directives); unknown sources rank lowest."""
        self.path = Path(path)
        self.holdout_percent = holdout_percent
        self.priorities = dict(priorities) if priorities else dict(SOURCE_PRIORITY)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ----------------------------------------------------------- experience
    def split_of(self, text: str) -> str:
        bucket = zlib.crc32(text.strip().lower().encode("utf-8")) % 100
        return "holdout" if bucket < self.holdout_percent else "train"

    def add_example(self, capability: str, text: str, label: str, source: str, weight: float) -> bool:
        cur = self.db.execute(
            "INSERT INTO examples(capability, text, label, source, weight, created_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(capability, text, source) DO UPDATE SET "
            "label=excluded.label, weight=excluded.weight",
            (capability, text, label, source, weight, time.time()))
        return cur.rowcount > 0

    def examples(self, capability: str, split: str | None = None) -> list[Example]:
        """All examples for a capability, one per text, the most trusted source winning."""
        rows = self.db.execute(
            "SELECT text, label, weight, source FROM examples WHERE capability=? ORDER BY id",
            (capability,)).fetchall()
        best: dict[str, Example] = {}
        rank = self.priorities
        for text, label, weight, source in rows:
            cur = best.get(text)
            if cur is None or rank.get(source, 0.5) >= rank.get(cur.source, 0.5):
                best[text] = Example(text, label, weight, source)
        out = list(best.values())
        if split:
            out = [e for e in out if self.split_of(e.text) == split]
        return out

    def count(self, capability: str) -> int:
        return self.db.execute("SELECT COUNT(DISTINCT text) FROM examples WHERE capability=?",
                               (capability,)).fetchone()[0]

    def label_counts(self, capability: str) -> Counter:
        return Counter(e.label for e in self.examples(capability))

    def source_counts(self, capability: str | None = None) -> Counter:
        if capability:
            rows = self.db.execute("SELECT source, COUNT(*) FROM examples WHERE capability=? "
                                   "GROUP BY source", (capability,)).fetchall()
        else:
            rows = self.db.execute("SELECT source, COUNT(*) FROM examples GROUP BY source").fetchall()
        return Counter(dict(rows))

    def relabel(self, capability: str, old: str, new: str) -> int:
        cur = self.db.execute("UPDATE examples SET label=? WHERE capability=? AND label=?",
                              (new, capability, old))
        return cur.rowcount

    # --------------------------------------------------------- capabilities
    def save_capability(self, name: str, spec: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO capabilities VALUES(?,?,?)",
                        (name, json.dumps(spec), time.time()))

    def load_capabilities(self) -> dict[str, dict]:
        return {n: json.loads(s) for n, s in self.db.execute("SELECT name, spec FROM capabilities")}

    # ----------------------------------------------------------------- nodes
    def save_node(self, meta: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?)",
                        (meta["id"], meta["capability"], meta["status"], json.dumps(meta), time.time()))

    def load_nodes(self) -> list[dict]:
        return [json.loads(m) for (m,) in self.db.execute("SELECT meta FROM nodes ORDER BY updated_at")]

    # ----------------------------------------------------------------- tools
    def save_tool(self, name: str, state: str, digest: str, meta: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO tools VALUES(?,?,?,?,?)",
                        (name, state, digest, json.dumps(meta), time.time()))

    def load_tools(self) -> dict[str, dict]:
        return {n: {"state": s, "digest": d, "meta": json.loads(m)}
                for n, s, d, m in self.db.execute("SELECT name, state, digest, meta FROM tools")}

    # ---------------------------------------------------------------- orders
    def save_order(self, order_id: str, text: str, result: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO orders(id, ts, text, result) VALUES(?,?,?,?)",
                        (order_id, time.time(), text, json.dumps(result, default=str)))

    def get_order(self, order_id: str) -> dict | None:
        row = self.db.execute("SELECT text, result, feedback FROM orders WHERE id=?",
                              (order_id,)).fetchone()
        if not row:
            return None
        return {"text": row[0], "result": json.loads(row[1]), "feedback": row[2]}

    def set_feedback(self, order_id: str, feedback: str) -> None:
        self.db.execute("UPDATE orders SET feedback=? WHERE id=?", (feedback, order_id))

    # ------------------------------------------------------------------ mind
    def save_note(self, note: dict) -> int:
        """Insert (no id) or update a note; returns its id."""
        args = (note["kind"], note["subject"], int(bool(note.get("private"))), json.dumps(note, default=str),
                time.time())
        if note.get("id"):
            self.db.execute("UPDATE notes SET kind=?, subject=?, private=?, data=?, updated_at=? WHERE id=?",
                            (*args, note["id"]))
            return int(note["id"])
        return int(self.db.execute("INSERT INTO notes(kind, subject, private, data, updated_at) VALUES(?,?,?,?,?)",
                                   args).lastrowid)

    def load_notes(self, limit: int = 5000) -> list[dict]:
        rows = self.db.execute("SELECT id, data FROM notes ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**json.loads(d), "id": i} for i, d in reversed(rows)]

    def delete_notes(self, ids: list[int]) -> None:
        self.db.executemany("DELETE FROM notes WHERE id=?", [(i,) for i in ids])

    def save_record(self, table: str, rid: str, status: str, data: dict) -> None:
        assert table in ("inquiries", "proposals")
        self.db.execute(f"INSERT OR REPLACE INTO {table}(id, status, data, created_at) VALUES(?,?,?,?)",
                        (rid, status, json.dumps(data, default=str), data.get("created_at", time.time())))

    def load_records(self, table: str) -> list[dict]:
        assert table in ("inquiries", "proposals")
        return [json.loads(d) for (d,) in self.db.execute(f"SELECT data FROM {table} ORDER BY created_at")]

    # ------------------------------------------------------------ small state
    def get_kv(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_kv(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES(?,?,?)", (key, value, time.time()))

    # ----------------------------------------------------------------- audit
    def audit(self, actor: str, kind: str, subject: str, detail: dict) -> None:
        self.db.execute("INSERT INTO audit(ts, actor, kind, subject, detail) VALUES(?,?,?,?,?)",
                        (time.time(), actor, kind, subject, json.dumps(detail, default=str)))

    def recent_audit(self, n: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT ts, actor, kind, subject, detail FROM audit "
                               "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"ts": r[0], "actor": r[1], "kind": r[2], "subject": r[3],
                 "detail": json.loads(r[4])} for r in reversed(rows)]

    def audit_counts(self) -> Counter:
        return Counter(dict(self.db.execute("SELECT kind, COUNT(*) FROM audit GROUP BY kind")))
