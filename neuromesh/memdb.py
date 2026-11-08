"""The memory's own database (state/memory.db), separate from the mesh's main database.

    tests   every memory node's tests: each question with the answer the node must give, the ways of
            asking it, and the questions it must not answer ([memory] tests: the contract a smaller or
            merged node must pass). People can change them - with /memory, the API, or any SQLite tool:
            triggers bump the node's revision on every change, and the mesh notices, relearns or forgets.
    nodes   one row a node: its revision and a summary for people (facts, size, passing, history)
    own     its own statements, kept in memory WITHOUT tests: its opinions, what its ego said, how its ego
            settled a disagreement between its nodes. Nothing checks them - they are its own views, never
            facts - so they are never served as an answer about the world (P8, P18).

Test files (memory_tests/<node>.toml, as earlier versions kept them) are imported once and moved aside.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, rev INTEGER NOT NULL DEFAULT 0, summary TEXT NOT NULL DEFAULT '{}',
                                 updated REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS tests(node TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL DEFAULT 'test',
                                 ask TEXT NOT NULL, answer TEXT NOT NULL DEFAULT '', variant_of TEXT NOT NULL DEFAULT '',
                                 source TEXT NOT NULL DEFAULT '', learned TEXT NOT NULL DEFAULT '', by TEXT NOT NULL DEFAULT '',
                                 reason TEXT NOT NULL DEFAULT '', PRIMARY KEY(node, seq));
CREATE INDEX IF NOT EXISTS tests_node ON tests(node);
CREATE TABLE IF NOT EXISTS own(id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, about TEXT NOT NULL DEFAULT '',
                               ask TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, mood TEXT NOT NULL DEFAULT '',
                               source TEXT NOT NULL DEFAULT '', node TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS own_about ON own(about);
CREATE INDEX IF NOT EXISTS own_ts ON own(ts);
CREATE TRIGGER IF NOT EXISTS tests_added AFTER INSERT ON tests BEGIN
  INSERT OR IGNORE INTO nodes(id, rev, updated) VALUES (NEW.node, 0, strftime('%s', 'now'));
  UPDATE nodes SET rev = rev + 1, updated = strftime('%s', 'now') WHERE id = NEW.node;
END;
CREATE TRIGGER IF NOT EXISTS tests_changed AFTER UPDATE ON tests BEGIN
  UPDATE nodes SET rev = rev + 1, updated = strftime('%s', 'now') WHERE id IN (OLD.node, NEW.node);
END;
CREATE TRIGGER IF NOT EXISTS tests_removed AFTER DELETE ON tests BEGIN
  UPDATE nodes SET rev = rev + 1, updated = strftime('%s', 'now') WHERE id = OLD.node;
END;
CREATE TRIGGER IF NOT EXISTS node_removed AFTER DELETE ON nodes BEGIN
  DELETE FROM tests WHERE node = OLD.id;
END;
"""
FIELDS = ("ask", "answer", "variant_of", "source", "learned", "by", "reason")


class MemoryDB:
    def __init__(self, path: Path, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly
        self.lock = threading.RLock()
        if readonly:
            if not self.path.exists():
                self.db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
                self.db.executescript(SCHEMA)
            else:
                self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False,
                                          isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=10)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------------ tests
    def revs(self) -> dict[str, int]:
        with self.lock:
            return {r[0]: int(r[1]) for r in self.db.execute("SELECT id, rev FROM nodes")}

    def rev(self, node_id: str) -> int | None:
        with self.lock:
            row = self.db.execute("SELECT rev FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return int(row[0]) if row else None

    def read(self, node_id: str) -> dict | None:
        """{"tests": [...], "unknowns": [...], "rev": n} or None when the database has no such node."""
        with self.lock:
            row = self.db.execute("SELECT rev, summary FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row is None:
                return None
            rows = self.db.execute(f"SELECT kind, {', '.join(FIELDS)} FROM tests WHERE node = ? ORDER BY seq",
                                   (node_id,)).fetchall()
        tests, unknowns = [], []
        for r in rows:
            item = dict(zip(FIELDS, r[1:]))
            (unknowns if r[0] == "unknown" else tests).append(item)
        return {"tests": tests, "unknowns": unknowns, "rev": int(row[0]), "summary": json.loads(row[1] or "{}")}

    def write(self, node_id: str, tests: list[dict], unknowns: list[dict], summary: dict) -> int:
        """Replace a node's tests (one transaction). Returns its new revision."""
        rows = [(node_id, i, "test", *[str(t.get(f, "") or "") for f in FIELDS]) for i, t in enumerate(tests)]
        rows += [(node_id, len(rows) + i, "unknown", *[str(u.get(f, "") or "") for f in FIELDS])
                 for i, u in enumerate(unknowns)]
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT OR IGNORE INTO nodes(id, rev, updated) VALUES (?, 0, ?)", (node_id, time.time()))
                self.db.execute("DELETE FROM tests WHERE node = ?", (node_id,))
                self.db.executemany(f"INSERT INTO tests(node, seq, kind, {', '.join(FIELDS)}) VALUES "
                                    f"(?, ?, ?, {', '.join('?' * len(FIELDS))})", rows)
                self.db.execute("UPDATE nodes SET summary = ?, updated = ? WHERE id = ?",
                                (json.dumps(summary, ensure_ascii=False), time.time(), node_id))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
            return int(self.db.execute("SELECT rev FROM nodes WHERE id = ?", (node_id,)).fetchone()[0])

    def delete(self, node_id: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

    def counts(self) -> dict:
        with self.lock:
            tests = self.db.execute("SELECT COUNT(*) FROM tests WHERE kind = 'test'").fetchone()[0]
            unknown = self.db.execute("SELECT COUNT(*) FROM tests WHERE kind = 'unknown'").fetchone()[0]
            own = self.db.execute("SELECT COUNT(*) FROM own").fetchone()[0]
        return {"tests": int(tests), "unknowns": int(unknown), "own": int(own)}

    # ------------------------------------------------------------------ its own statements (untested)
    def add_own(self, kind: str, text: str, about: str = "", ask: str = "", mood: str = "", source: str = "",
                node: str = "", ts: float | None = None) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO own(ts, kind, about, ask, text, mood, source, node) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                  (ts or time.time(), kind, about[:200], ask[:400], text[:2000], mood, source[:300], node))
            return int(cur.lastrowid)

    def own(self, kind: str | tuple | None = None, about: str | None = None, limit: int = 50, since: float = 0.0,
            until: float | None = None, node: str | None = None) -> list[dict]:
        where, args = ["ts >= ?"], [since]
        if until is not None:
            where.append("ts <= ?")
            args.append(until)
        if kind:
            kinds = (kind,) if isinstance(kind, str) else tuple(kind)
            where.append(f"kind IN ({', '.join('?' * len(kinds))})")
            args += list(kinds)
        if about is not None:
            where.append("lower(about) = lower(?)")
            args.append(about)
        if node is not None:
            where.append("node = ?")
            args.append(node)
        with self.lock:
            rows = self.db.execute(f"SELECT id, ts, kind, about, ask, text, mood, source, node FROM own WHERE {' AND '.join(where)} "
                                   f"ORDER BY ts DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        keys = ("id", "ts", "kind", "about", "ask", "text", "mood", "source", "node")
        return [dict(zip(keys, r)) for r in rows]

    def own_asks(self, node: str) -> list[dict]:
        """The newest statement for every question an own-memory node answers (what it learns from)."""
        with self.lock:
            rows = self.db.execute("SELECT o.ask, o.text, o.kind, o.about, o.ts FROM own o JOIN "
                                   "(SELECT lower(ask) AS k, MAX(id) AS last, MIN(id) AS first FROM own WHERE node = ? AND ask != '' "
                                   "GROUP BY lower(ask)) g ON o.id = g.last ORDER BY g.first", (node,)).fetchall()
        return [{"ask": r[0], "text": r[1], "kind": r[2], "about": r[3], "ts": r[4]} for r in rows]

    def own_nodes(self) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT DISTINCT node FROM own WHERE node != '' ORDER BY node")]

    def forget_own(self, about: str) -> int:
        with self.lock:
            return self.db.execute("DELETE FROM own WHERE lower(about) = lower(?)", (about,)).rowcount
