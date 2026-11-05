"""Background tasks: goals people set that the mesh keeps working on, continuously, in the
background.

    learn <topic>           keep studying a topic: research it, then the articles it links to (one per
                            run, depth-limited), read what it finds with its skill nodes, then keep
                            reviewing what it read so it stays fresh
    improve <skill> [to X]  keep training a skill until its serving node reaches X on held-out data:
                            the Trainer gives it the lessons and training slices first, the Architect
                            breeds challengers for it; once met it keeps watch and resumes if it slips
    every <interval>: <order>  or  run <order> every <interval>
                            repeat an order and say when its result changes (notify change|always|never)

A task runs only on background capacity: never while an order is in flight, within the
background duty cycle (P2), and within its own hourly budget (P4). It uses only approved tools;
if it needs one that is waiting for approval it asks (P3). /halt and /pause stop every task (P1).
Tasks persist across restarts until a person removes them.
"""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import asdict, dataclass, field, fields

from .tasks import Order

KINDS = ("learn", "improve", "run")
NOTIFY = ("change", "always", "never")
USAGE = ("background tasks: learn <topic> [every 2m] [max 30] [depth 2] | improve <skill> [to 0.97] | "
         "every 10m: <order> | run <order> every 10m [notify change|always|never]")

_UNITS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1, "m": 60, "min": 60, "mins": 60, "minute": 60,
          "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "d": 86400,
          "day": 86400, "days": 86400}
_DURATION = r"\d+(?:\.\d+)?\s*(?:" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\b"
_OPTION = re.compile(r"\s+(every\s+" + _DURATION + r"|max\s+\d+(?:\s+articles?)?|depth\s+\d+|"
                     r"(?:to|until)\s+\d+(?:\.\d+)?%?|notify\s+(?:always|changes?|never))\s*$", re.I)
_EVERY_FIRST = re.compile(r"^every\s+(?P<dur>" + _DURATION + r")\s*[:,]\s*(?P<order>.+)$", re.I | re.S)
_LEARN = re.compile(r"^(?:keep\s+(?:on\s+)?(?:learning|studying|reading(?:\s+up)?)\s+(?:about|on)\s+|"
                    r"learn(?:\s+about)?\s+|study\s+|read\s+up\s+on\s+)", re.I)
_IMPROVE = re.compile(r"^(?:keep\s+)?(?:improving|improve|training|train|practi[cs]ing|practi[cs]e)\s+", re.I)
_RUN = re.compile(r"^(?:keep\s+)?(?:run|running|watch|watching|check|checking|repeat|monitor|monitoring)\s+", re.I)
_ORDER_PHRASE = re.compile(r"^\s*(?:please\s+)?(?:in\s+the\s+background,?\s+)?(?P<spec>keep\s+(?:on\s+)?"
                           r"(?:learning|studying|reading(?:\s+up)?|improving|training|watching|checking|"
                           r"monitoring)\s+.+|every\s+" + _DURATION + r"\s*[:,]\s*.+)$", re.I | re.S)


def parse_duration(text: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]+)\s*", text.lower())
    if not m or m.group(2) not in _UNITS:
        raise ValueError(f"not a duration: '{text}' (e.g. 30s, 10m, 2h, 1d)")
    return float(m.group(1)) * _UNITS[m.group(2)]


def parse_spec(text: str) -> dict:
    """'learn quantum computing every 2m max 30' -> {"kind": "learn", "target": ..., options}."""
    t = " ".join(text.strip().split()) if "\n" not in text else text.strip()
    if t.lower().startswith("add "):
        t = t[4:].strip()
    m = _EVERY_FIRST.match(t)
    if m:
        return {"kind": "run", "target": m.group("order").strip(), "interval": parse_duration(m.group("dur"))}
    spec: dict = {}
    while True:  # options come at the end, in any order
        m = _OPTION.search(t)
        if not m:
            break
        opt = m.group(1).lower()
        t = t[: m.start()]
        word, _, value = opt.partition(" ")
        value = value.strip()
        if word == "every":
            spec["interval"] = parse_duration(value)
        elif word == "max":
            spec["max_articles"] = int(value.split()[0])
        elif word == "depth":
            spec["depth"] = int(value)
        elif word in ("to", "until"):
            number = float(value.rstrip("%"))
            spec["threshold"] = number / 100 if value.endswith("%") or number > 1 else number
        elif word == "notify":
            spec["notify"] = "change" if value.startswith("change") else value
    for kind, prefix in (("learn", _LEARN), ("improve", _IMPROVE), ("run", _RUN)):
        m = prefix.match(t)
        if m:
            target = t[m.end():].strip().strip("'\"")
            if not target:
                break
            spec.update(kind=kind, target=target)
            if kind == "run" and "interval" not in spec:
                raise ValueError("a run task needs an interval: run <order> every 10m")
            return spec
    raise ValueError(USAGE)


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 90 else f"{seconds // 60}m" if seconds < 5400 else f"{seconds // 3600}h"


def every(seconds: float) -> str:
    if seconds >= 86400 and seconds % 86400 == 0:
        return f"{seconds / 86400:g}d"
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds / 60:g}m"
    return f"{seconds:g}s"


@dataclass
class BackgroundTask:
    id: str
    kind: str                       # learn | improve | run
    target: str                     # topic, capability, or order text
    interval: float
    spec: str = ""                  # as the person wrote it
    threshold: float = 0.0          # improve: held-out accuracy to reach
    max_articles: int = 0           # learn
    depth: int = 0                  # learn
    notify: str = "change"          # run: change | always | never
    status: str = "active"          # active | paused
    note: str = ""                  # what it is doing / waiting for, in a few words
    created_by: str = "user"
    created_at: float = field(default_factory=time.time)
    last_run: float = 0.0
    next_run: float = 0.0
    runs: int = 0
    failures: int = 0               # consecutive
    met: bool = False               # improve: target reached; learn: first pass done
    progress: dict = field(default_factory=dict)
    history: list = field(default_factory=list)   # recent results: [ts, ok, summary]

    def describe(self) -> str:
        if self.kind == "learn":
            return f"learn '{self.target}'"
        if self.kind == "improve":
            return f"improve {self.target} to {self.threshold:.2f}"
        return f"every {every(self.interval)}: {self.target}"


class BackgroundTasks:
    def __init__(self, system, clock=time.time):
        self.sys = system
        self.pol = system.directives.tasks
        self.clock = clock
        names = {f.name for f in fields(BackgroundTask)}
        self.tasks: dict[str, BackgroundTask] = {}
        for d in system.store.load_records("bg_tasks"):
            self.tasks[d["id"]] = BackgroundTask(**{k: v for k, v in d.items() if k in names})
        nums = [int(i.split("-")[-1]) for i in self.tasks if i.split("-")[-1].isdigit()]
        self._ids = itertools.count(max(nums, default=0) + 1)

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def save(self, task: BackgroundTask) -> None:
        self.sys.store.save_record("bg_tasks", task.id, task.status, asdict(task))

    # ---------------------------------------------------------------- managing
    def add(self, text: str, by: str = "user") -> BackgroundTask:
        if not self.enabled:
            raise ValueError("background tasks are off ([tasks] enabled = false in the directives)")
        spec = parse_spec(text)
        s, pol = self.sys, self.pol
        if len(self.tasks) >= pol.max_tasks:
            raise ValueError(f"already {len(self.tasks)} background tasks (max {pol.max_tasks}); remove one first")
        kind = spec["kind"]
        default = {"learn": pol.learn_interval_seconds, "improve": pol.improve_interval_seconds}.get(kind, 0)
        interval = float(spec.get("interval", default))
        if interval < pol.min_interval_seconds:
            raise ValueError(f"the shortest interval allowed is {every(pol.min_interval_seconds)} "
                             f"([tasks] min_interval_seconds)")
        target = spec["target"]
        if kind == "improve":
            target = target.lower().removeprefix("my ").removesuffix(" skill").strip()
            if target in ("routing", "the router", "router"):
                target = "dispatch"
            if target not in s.capabilities:
                raise ValueError(f"no skill called '{target}' (skills: {', '.join(sorted(s.capabilities))})")
        same = next((t for t in self.tasks.values() if (t.kind, t.target.lower()) == (kind, target.lower())), None)
        if same is not None:
            raise ValueError(f"{same.id} already does that ({same.describe()})")
        threshold = float(spec.get("threshold", s.directives.nodes.target_accuracy)) if kind == "improve" else 0.0
        if kind == "improve" and not 0 < threshold <= 1:
            raise ValueError("an improve target is an accuracy between 0 and 1 (or a percentage)")
        task = BackgroundTask(
            f"task-{next(self._ids)}", kind, target, interval, spec=text.strip(), threshold=threshold,
            max_articles=int(spec.get("max_articles", pol.learn_max_articles)) if kind == "learn" else 0,
            depth=int(spec.get("depth", pol.learn_depth)) if kind == "learn" else 0,
            notify=spec.get("notify", "change"), created_by=by, created_at=self.clock(), next_run=self.clock(),
            note="starting")
        if kind == "learn":
            task.progress = {"frontier": [target], "depth": {target: 0}, "read": [], "failed": []}
        self.tasks[task.id] = task
        self.save(task)
        s.guardian.audit(by, "task.added", task.id, task_kind=kind, target=target, interval=interval)
        s.bus.publish("task.added", id=task.id, kind=kind, what=task.describe(), by=by, every=every(interval))
        return task

    def from_order(self, text: str, by: str) -> tuple[str, str, bool] | None:
        """Orders like "keep learning about jazz" or "every 10m: what is 2+2" set a task.
        Returns (answer, trace, ok), or None when the order is not about a background task."""
        m = _ORDER_PHRASE.match(text)
        if not m or not self.enabled:
            return None
        try:
            task = self.add(m.group("spec"), by=by)
        except ValueError as exc:
            return f"I couldn't set that background task: {exc}", "background task not set", False
        return (f"Background task {task.id} set: {task.describe()}, every {every(task.interval)}. "
                f"/tasks shows it; /task remove {task.id} stops it.", f"set background task {task.id}", True)

    def get(self, task_id: str) -> BackgroundTask | None:
        return self.tasks.get(task_id)

    def set_status(self, task_id: str, status: str, by: str) -> tuple[bool, str]:
        task = self.tasks.get(task_id)
        if task is None:
            return False, f"no background task {task_id} (/tasks lists them)"
        if status == "removed":
            del self.tasks[task_id]
            self.sys.store.delete_record("bg_tasks", task_id)
        else:
            if task.status == status:
                return False, f"{task_id} is already {status}"
            task.status = status
            if status == "active":
                task.failures, task.next_run, task.note = 0, self.clock(), "resumed"
            self.save(task)
        self.sys.guardian.audit(by, f"task.{status}", task_id, task_kind=task.kind, target=task.target)
        self.sys.bus.publish(f"task.{status}", id=task_id, what=task.describe(), by=by)
        return True, f"{task_id} ({task.describe()}) {status}"

    def focus(self) -> dict[str, float]:
        """Skills that active improve tasks want trained first, with their targets."""
        out: dict[str, float] = {}
        for t in self.tasks.values():
            if t.kind == "improve" and t.status == "active" and t.target in self.sys.capabilities:
                out[t.target] = max(out.get(t.target, 0.0), t.threshold)
        return out

    def next_due(self) -> BackgroundTask | None:
        now = self.clock()
        due = [t for t in self.tasks.values() if t.status == "active" and t.next_run <= now]
        return min(due, key=lambda t: t.next_run) if due else None

    # ----------------------------------------------------------------- running
    async def run(self, task: BackgroundTask) -> None:
        s = self.sys
        s.guardian.record_task_run()
        task.runs += 1
        task.last_run = self.clock()
        task.next_run = task.last_run + task.interval
        t0 = time.monotonic()
        try:
            with s.activity.track("task", task.id, task.describe()[:90]):
                if task.kind == "learn":
                    await self._learn(task)
                elif task.kind == "improve":
                    self._improve(task)
                else:
                    await self._run_order(task)
        except Exception as exc:  # a broken task never takes the runner down
            task.failures += 1
            task.note = f"error: {exc}"[:200]
        finally:
            s.guardian.note_background_busy(time.monotonic() - t0)
        if task.kind == "run" and task.failures >= self.pol.pause_after_failures and task.status == "active":
            task.status = "paused"
            task.note = f"paused after {task.failures} failures in a row: {task.note}"
            s.guardian.audit("tasks", "task.paused", task.id, reason=task.note)
            s.bus.publish("task.paused", id=task.id, what=task.describe(), by="tasks (repeated failures)")
        if task.id in self.tasks:
            self.save(task)

    def _tell(self, task: BackgroundTask, text: str) -> None:
        """A result worth telling the user: the activity log and the notes users can read."""
        if self.sys.mind.enabled:
            self.sys.mind.note("update", f"task:{task.id}", text, private=False, source=f"background task {task.id}")
        self.sys.bus.publish("task.update", id=task.id, what=task.describe(), text=text[:300])

    async def _learn(self, task: BackgroundTask) -> None:
        s = self.sys
        p = task.progress
        tool = s.knowledge_tool()
        if tool is None:
            pending = next((t for t in s.tools.tools.values()
                            if t.manifest.provides == "knowledge" and t.state == "pending_approval" and t.approval_id), None)
            if pending is not None:
                s.mind.ask("approval", f"Background task {task.id} wants to learn about '{task.target}', but the "
                           f"{pending.name} tool needs your approval (permissions: "
                           f"{', '.join(pending.manifest.permissions)}). Approve it? (yes/no)",
                           f"for {task.id}", approval_id=pending.approval_id)
                task.note = f"waiting for {pending.name} to be approved"
            else:
                task.note = "waiting for a research tool"
            return
        unlimited = s.unlimited(tool)  # your own server (kiwix): read several articles per run
        if not s.guardian.authorize_research(background=True, unlimited=unlimited).allowed:
            task.note = "waiting for research budget"
            return
        frontier, read, depth = p.setdefault("frontier", []), p.setdefault("read", []), p.setdefault("depth", {})
        if frontier and len(read) < task.max_articles:
            for _ in range(self.pol.learn_burst_unlimited if unlimited else 1):
                if not frontier or len(read) >= task.max_articles:
                    break
                await self._read_one(task, tool, frontier, read, depth)
            if not task.met and (not frontier or len(read) >= task.max_articles):
                task.met = True
                self._tell(task, f"I've read {len(read)} article{'s' if len(read) != 1 else ''} about '{task.target}': "
                                 f"{', '.join(read[:6])}{', ...' if len(read) > 6 else ''}. ({task.id} finished a "
                                 f"first pass and keeps them fresh from now on.)")
            return
        # first pass done: keep what was read fresh
        stale_after = s.directives.research.refresh_days * 86400
        docs = [d for d in (s.knowledge.get(t) for t in read) if d is not None]
        missing = [t for t in read if s.knowledge.get(t) is None]
        oldest = min(docs, key=lambda d: d.fetched_at, default=None)
        if missing:  # forgotten under the retention cap: read again
            frontier.append(missing[0])
            read.remove(missing[0])
            task.note = f"re-reading '{missing[0]}' (it was forgotten)"
        elif oldest is not None and oldest.age > stale_after:
            doc, _ = await s.research(oldest.title, reason=f"{task.id}: keeping '{task.target}' fresh",
                                      background=True, tool=tool)
            task.note = f"refreshed '{oldest.title}'" if doc else f"could not refresh '{oldest.title}'"
        else:
            task.note = f"reviewing {len(read)} articles; all fresh"

    async def _read_one(self, task: BackgroundTask, tool, frontier: list, read: list, depth: dict) -> None:
        s = self.sys
        p = task.progress
        title = frontier.pop(0)
        doc, why = await s.research(title, reason=f"{task.id}: learning about {task.target}", background=True, tool=tool)
        if doc is None:
            p.setdefault("failed", []).append(title)
            del p["failed"][:-20]
            task.note = f"could not read '{title}': {why}"[:200]
            return
        if doc.title not in read:
            read.append(doc.title)
        d = depth.get(title, 0)
        if d < task.depth:
            for link in doc.links[: self.pol.learn_links_per_article]:
                if link not in read and link not in frontier and len(read) + len(frontier) < task.max_articles:
                    frontier.append(link)
                    depth[link] = d + 1
        if len(read) == 1:  # the topic itself: read it with every skill node (forms an opinion)
            steps: list[str] = []
            await s.investigator._read_with_nodes(doc, steps)
        task.note = f"read '{doc.title}' from {doc.source} ({len(read)}/{task.max_articles})"

    def _improve(self, task: BackgroundTask) -> None:
        s = self.sys
        if task.target not in s.capabilities:
            task.note = f"the skill '{task.target}' no longer exists"
            return
        champ = s.registry.champion(task.target)
        acc = champ.metrics.get("holdout_acc") if champ else None
        held = champ.metrics.get("holdout_n", 0) if champ else 0
        p = task.progress
        p["examples"] = s.store.count(task.target)
        if acc is None:
            task.note = f"no node yet ({p['examples']} examples); asking teachers for lessons"
            return
        p["accuracy"] = round(acc, 3)
        p["best"] = round(max(acc, p.get("best", 0.0)), 3)
        enough = held >= s.directives.nodes.min_eval_samples
        if acc >= task.threshold and enough:
            task.note = f"at {acc:.3f} (target {task.threshold:.2f}); keeping watch"
            if not task.met:
                task.met = True
                self._tell(task, f"My {task.target} skill reached {acc:.3f} on {held} held-out examples, past the "
                                 f"target of {task.threshold:.2f}. ({task.id}; node {champ.id}; I'll keep watch.)")
        else:
            why = f"at {acc:.3f}, target {task.threshold:.2f}" if enough else f"{held} held-out examples so far"
            task.note = f"{why}; lessons and challengers go here first"
            if task.met and enough:
                task.met = False
                self._tell(task, f"My {task.target} skill slipped to {acc:.3f}, below the target of "
                                 f"{task.threshold:.2f}. ({task.id}; working on it again.)")

    async def _run_order(self, task: BackgroundTask) -> None:
        s = self.sys
        order = Order(text=task.target, source=f"task:{task.id}", background=True)
        res = await s.orchestrator.fulfil(order)
        summary = " ".join(str(res.output if res.ok else f"(not fulfilled) {res.note}").split())[:400]
        said = (res.answer or summary).split("\n", 1)[0]  # the result first, in the mesh's words
        last = task.history[-1][2] if task.history else None
        task.history.append([round(self.clock(), 1), res.ok, summary])
        del task.history[:-20]
        task.failures = 0 if res.ok else task.failures + 1
        task.note = f"{'ok' if res.ok else 'failed'}: {summary[:80]}"
        changed = last is not None and summary != last
        if task.notify == "always" or (task.notify == "change" and (last is None or changed)):
            where = f"{task.id}, every {every(task.interval)}: '{task.target[:60]}'"
            self._tell(task, f"{said} ({where}; changed, was {last[:120]})" if changed else f"{said} ({where})")

    # ------------------------------------------------------------------ status
    def lines(self) -> list[str]:
        now = self.clock()
        out = []
        for t in sorted(self.tasks.values(), key=lambda t: t.id):
            when = (f"next in {_ago(t.next_run - now)}" if t.status == "active" and t.next_run > now
                    else "due" if t.status == "active" else "paused")
            out.append(f"  {t.id} [{t.status}] {t.describe()} (every {every(t.interval)}, {t.runs} runs, {when}) - {t.note}")
        return out

    def status(self) -> dict:
        return {"enabled": self.enabled, "tasks": [asdict(t) for t in self.tasks.values()],
                "usage": self.sys.guardian.task_usage(), "focus": self.focus()}
