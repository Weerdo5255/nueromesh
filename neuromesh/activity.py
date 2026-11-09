"""Live activity: what the system is doing right now, and a running log of what it did.

ActivityTracker holds the work in flight (lessons from teachers, training slices,
evaluations, tool self-tests, orders) so a live view can show it as it happens.
ActivityLog turns bus events into one-line log entries, keeps the recent ones in memory
and appends every entry to state/activity.log.
"""
from __future__ import annotations

import contextlib
import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .bus import Event
from .report import describe_event


@dataclass
class Activity:
    id: int
    kind: str          # lesson | training | evaluation | self-test | order | synthesis
    actor: str         # who is doing it: a teacher name, "trainer", "evaluator", ...
    text: str
    started: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)
    ended: float | None = None

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started


class ActivityTracker:
    def __init__(self, keep: int = 40):
        self._active: dict[int, Activity] = {}
        self._done: deque[Activity] = deque(maxlen=keep)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def track(self, kind: str, actor: str, text: str, **data):
        act = Activity(next(self._ids), kind, actor, text, data=data)
        with self._lock:
            self._active[act.id] = act
        try:
            yield act
        finally:
            act.ended = time.time()
            with self._lock:
                self._active.pop(act.id, None)
                self._done.append(act)

    def current(self) -> list[Activity]:
        with self._lock:
            return sorted(self._active.values(), key=lambda a: a.started)

    def recent(self, seconds: float = 15.0) -> list[Activity]:
        """Finished work from the last few seconds, newest first."""
        cutoff = time.time() - seconds
        with self._lock:
            return [a for a in reversed(self._done) if (a.ended or 0) >= cutoff]


# --------------------------------------------------------------------- the log
ROUTINE = {"node.trained", "node.evaluated", "lesson.label", "memory.rejected", "memory.merge_refused",
           "dream.dreamt", "director.reviewed", "objective.step", "thinking.thought"}  # hidden by "/log important"


@dataclass
class LogEntry:
    ts: float
    tag: str
    text: str
    level: str   # info | good | warn | bad | lesson | order | tool | teacher | work
    topic: str
    data: dict = field(default_factory=dict)  # the event's data (so a line can be rewarded or punished: rewards.py)


def _snip(text: str, n: int = 48) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_event(ev: Event) -> LogEntry | None:
    """One log line per event worth logging (None = not shown)."""
    d, t = ev.data, ev.topic
    if t == "order.done":
        out = d.get("output") if d.get("ok") else f"not fulfilled: {d.get('note', '')}"
        return LogEntry(ev.ts, "order", f"{_snip(d.get('text', ''), 40)} -> {_snip(out, 30)} "
                        f"[{d.get('route')} | {d.get('served_by')} | {d.get('latency_ms', 0):.0f}ms]",
                        "order" if d.get("ok") else "warn", t)
    if t == "lesson.label":
        return LogEntry(ev.ts, "lesson", f"{d['teacher']}: {d['capability']}('{_snip(d['text'], 36)}') = "
                        f"{d['label']} ({d['seconds']:.1f}s)", "lesson", t)
    if t == "node.trained":
        verdict = "" if d.get("accepted", True) else " rejected (would regress)"
        loss = f" loss {d['loss']:.3f}" if d.get("loss") is not None else ""
        return LogEntry(ev.ts, "train", f"{d['node']} slice {d['steps']} on {d['examples']} examples{loss}{verdict}",
                        "work", t)
    if t == "node.evaluated":
        accs = ", ".join(f"{n}={a}" for n, a in d["results"].items())
        return LogEntry(ev.ts, "eval", f"{d['capability']} on {d['holdout']} held-out: {accs}", "work", t)
    if t == "node.disagreement":
        return LogEntry(ev.ts, "check", f"spot check: {d['node']} said {d['node_label']}, "
                        f"{d['teacher']} says {d['teacher_label']}", "warn", t)
    if t == "audit":
        kind = d.get("kind", "")
        if kind == "honesty.guard":
            return LogEntry(ev.ts, "guard", f"honesty guard: a '{d.get('detail', {}).get('action')}' shaping of "
                            f"{d.get('subject', '')} failed the check; the original answer was shown", "bad", t)
        if kind.startswith("deny.") or kind == "update.rejected":
            reason = d.get("detail", {}).get("reason", "")
            return LogEntry(ev.ts, "guard", f"{kind}: {d.get('subject', '')} - {reason}", "bad", t)
        return None
    line = describe_event(ev)
    if not line:
        return None
    tag, _, text = line.partition("]")
    tag = tag.strip("[ ").lower()
    level = {"spawn": "info", "promote": "good", "retire": "warn", "dispose": "bad", "tool": "tool",
             "approval": "warn", "label": "info", "skill": "info", "lesson": "lesson",
             "teacher": "teacher", "halt": "bad", "resume": "good", "learn": "good", "recall": "lesson",
             "forget": "warn", "research": "warn", "mood": "teacher", "temper": "lesson",
             "release": "good", "mind": "teacher", "ask": "warn", "propose": "good",
             "task": "good", "challenge": "good", "memory": "good", "director": "teacher", "sleep": "teacher",
             "dream": "lesson", "code": "good", "objective": "good", "ruminate": "teacher",
             "feedback": "lesson", "think": "teacher", "ego": "teacher", "idle": "work"}.get(tag, "info")
    if t == "output.shaped" and d.get("action") in ("confirm", "defer", "block"):
        level = "warn"
    if tag == "tool" and ("REJECTED" in text or "quarantined" in text):
        level = "bad"
    if tag == "teacher" and "unavailable" in text:
        level = "warn"
    if t in ("memory.rejected", "memory.fact", "memory.merge_refused"):
        level = "lesson" if t == "memory.fact" else "work"
    if t in ("memory.failing", "memory.unreadable", "memory.forgotten"):
        level = "warn"
    if t == "director.consulted" and d.get("verdict") == "disagrees" or t in ("director.restarted", "director.gave_up"):
        level = "warn"
    if t == "director.retried" and d.get("answered"):
        level = "good"
    if t in ("director.reviewed", "director.memory"):
        level = "work"
    if t in ("tool.unsafe", "tool.declined") or (t == "tool.experimented" and not d.get("proven")):
        level = "warn"
    if t == "tool.drafted":
        level = "teacher"
    if t == "objective.step":
        level = "work"
    if t == "objective.contradiction" or (t == "objective.ended" and d.get("status") == "unmet") or t == "objective.asked":
        level = "warn"
    return LogEntry(ev.ts, tag, text.strip(), level, t)


class ActivityLog:
    def __init__(self, path: Path | None = None, maxlen: int = 5000):
        self.entries: deque[LogEntry] = deque(maxlen=maxlen)
        self.path = Path(path) if path else None
        self.version = 0  # bumped on every entry, so a view knows when to redraw
        self._lock = threading.Lock()

    def on_event(self, ev: Event) -> None:
        entry = format_event(ev)
        if entry is not None:
            entry.data = ev.data
            self.add(entry)

    def add(self, entry: LogEntry) -> None:
        with self._lock:
            self.entries.append(entry)
            self.version += 1
        if self.path is not None:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.ts))} "
                             f"[{entry.tag}] {entry.text}\n")
            except OSError:
                pass

    def since(self, after: int = 0, n: int = 500) -> tuple[int, list[tuple[int, LogEntry]]]:
        """Entries numbered after `after` (each entry's number is its position in the whole log),
        so a viewer can fetch only what is new. Returns (latest number, [(number, entry), ...])."""
        with self._lock:
            items = list(self.entries)
            version = self.version
        first = version - len(items) + 1
        out = [(first + i, e) for i, e in enumerate(items) if first + i > after]
        return version, (out[-n:] if n > 0 else [])

    def tail(self, n: int, routine: bool = True) -> list[LogEntry]:
        with self._lock:
            items = list(self.entries)
        if not routine:
            items = [e for e in items if e.topic not in ROUTINE]
        return items[-n:] if n > 0 else []
