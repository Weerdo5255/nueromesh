"""The Mind's archive: everything that happens in its mind, kept by date and time.

Every event of its mind - questions it asked you and your replies, proposals, what its ego said and how it
settled disagreements, its thoughts, curiosity and Kiwix searches, objectives, directors, initiative, the
tools it writes, sleep, dreams and coin flips, changes of mood, opinions and notes - is written, as the line
the activity log shows, to the mind_log table of the main database, under the Mind tab it belongs to.

The web interface shows the last day in each Mind tab; anything older is archived behind a date and time
lookup (GET /mind/archive?tab=&at=&hours=, and /mind/archive/days for the days that have entries).
"""
from __future__ import annotations

import json
import time

from .report import describe_event

SCHEMA = """
CREATE TABLE IF NOT EXISTS mind_log(id INTEGER PRIMARY KEY, ts REAL NOT NULL, tab TEXT NOT NULL, topic TEXT NOT NULL,
                                    line TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS mind_log_tab_ts ON mind_log(tab, ts);
CREATE INDEX IF NOT EXISTS mind_log_ts ON mind_log(ts);
"""
# the Mind tab each kind of event belongs to (the first prefix that matches)
TABS = (("inquiry.", "questions"), ("mind.inquiry", "questions"), ("mind.proposal", "questions"),
        ("approval.", "questions"), ("ego.", "ego"), ("memory.own", "ego"), ("thinking.", "thinking"),
        ("kiwix.", "curiosity"), ("objective.", "objectives"), ("director.", "directors"), ("agency.", "initiative"), ("tools.decided", "initiative"), ("tools.switch", "initiative"),
        ("tool.drafted", "tools"), ("tool.experimented", "tools"), ("tool.unsafe", "tools"), ("dream.", "dreams"),
        ("temperament.", "temperament"), ("output.shaped", "temperament"), ("gitea.", "code"), ("code.", "code"),
        ("vision.", "sight"), ("tutor.", "lessons"), ("reward.", "rewards"), ("project.", "code"), ("mind.", "thoughts"),
        ("answer.right", "thoughts"))
NAMES = {"questions": "Questions", "ego": "Ego", "thinking": "Thinking", "curiosity": "Curiosity",
         "objectives": "Objectives", "directors": "Directors", "initiative": "Initiative", "tools": "Tools it writes",
         "dreams": "Dreams", "temperament": "Temperament", "thoughts": "Opinions and notes", "code": "Code",
         "sight": "Images", "lessons": "Lessons", "rewards": "Rewards"}
DAY = 86400.0


def tab_of(topic: str) -> str | None:
    return next((tab for prefix, tab in TABS if topic.startswith(prefix)), None)


class MindJournal:
    def __init__(self, system, keep: int = 200_000):
        self.sys = system
        self.keep = keep
        self.db = system.store.db
        self.db.executescript(SCHEMA)
        self.written = 0

    def on_event(self, ev) -> None:
        tab = tab_of(ev.topic)
        if tab is None:
            return
        try:
            line = describe_event(ev) or ev.topic
        except Exception:
            line = ev.topic
        data = json.dumps(ev.data, default=str)[:4000]
        try:
            self.db.execute("INSERT INTO mind_log(ts, tab, topic, line, data) VALUES (?, ?, ?, ?, ?)",
                            (getattr(ev, "ts", None) or time.time(), tab, ev.topic, " ".join(line.split())[:600], data))
        except Exception:
            return
        self.written += 1
        if self.written % 1000 == 0:
            self.db.execute("DELETE FROM mind_log WHERE id <= (SELECT MAX(id) FROM mind_log) - ?", (self.keep,))

    def query(self, tab: str | None = None, start: float = 0.0, end: float | None = None, limit: int = 200) -> list[dict]:
        where, args = ["ts >= ?"], [start]
        if end is not None:
            where.append("ts <= ?")
            args.append(end)
        if tab:
            where.append("tab = ?")
            args.append(tab)
        rows = self.db.execute(f"SELECT id, ts, tab, topic, line FROM mind_log WHERE {' AND '.join(where)} "
                               f"ORDER BY ts DESC, id DESC LIMIT ?", (*args, max(1, min(1000, limit)))).fetchall()
        return [{"id": r[0], "ts": r[1], "tab": r[2], "tab_name": NAMES.get(r[2], r[2]), "topic": r[3], "line": r[4]}
                for r in rows]

    def around(self, tab: str | None, at: float, hours: float, limit: int = 200) -> dict:
        """What happened in its mind within hours/2 either side of a moment (the archive lookup)."""
        half = max(0.25, float(hours)) * 3600 / 2
        rows = self.query(tab, at - half, at + half, limit)
        return {"tab": tab or "", "at": at, "hours": hours, "from": at - half, "to": at + half, "rows": rows,
                "more": len(rows) >= limit}

    def days(self, tab: str | None = None, tz_minutes: int = 0, limit: int = 60) -> list[dict]:
        """The days with entries, newest first: {"day": "2026-11-06", "count", "start"} (in the viewer's time zone:
        tz_minutes is JavaScript's getTimezoneOffset())."""
        shift = -int(tz_minutes) * 60
        where, args = "", []
        if tab:
            where, args = "WHERE tab = ?", [tab]
        rows = self.db.execute(f"SELECT CAST((ts + ?) / {DAY} AS INTEGER) AS d, COUNT(*) FROM mind_log {where} "
                               f"GROUP BY d ORDER BY d DESC LIMIT ?", (shift, *args, limit)).fetchall()
        out = []
        for d, n in rows:
            start = d * DAY - shift
            out.append({"day": time.strftime("%Y-%m-%d", time.gmtime(d * DAY)), "count": n, "start": start})
        return out

    def counts(self) -> dict:
        now = time.time()
        rows = self.db.execute("SELECT tab, COUNT(*), SUM(ts < ?) FROM mind_log GROUP BY tab", (now - DAY,)).fetchall()
        return {r[0]: {"total": r[1], "archived": int(r[2] or 0)} for r in rows}
