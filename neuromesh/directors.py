"""The directors: the mesh's executive, at both ends of every order.

    user input --> a director takes it (intake): reads the order and the mood, and plans how carefully
                   to handle it
               --> the orchestrator: memory nodes, the router, tools, skills, teachers
               --> the emotional engines (affect, disposition) and the mind
               --> a director reviews it: checks the result, asks a duplicate for a second opinion when
                   unsure, adds a disclosed note when they disagree (it never changes the answer, P8),
                   and decides what may go into long-term memory
               --> the person

Nothing enters long-term memory without a director's review. Facts an order wants to keep wait for
that order's review; every other write (research, practice answers, what people tell it) is reviewed
as it happens. A person's word is always kept (P1); an imagined answer never is (P13).

There is always at least one director node running: director-0, the primary, which also takes the
mesh's initiative (agency.py) and is restarted if it ever stops. It duplicates itself when it needs
to - when work waits for a director, when it wants a second opinion and no other director is free,
or when its initiative finds two things worth doing at once - up to [directors] max_directors (P4).
A duplicate is a copy of the director: the same policies and the same agency node. Duplicates
collaborate with it: one takes the next order while another reviews, one gives a second opinion
from what the mesh already holds (memory nodes, retained knowledge, a challenger node; never a
teacher or the web), one takes a second mode of initiative. They retire once idle for a while.

The mood shapes the directors (their stance), measured against the mesh's baseline personality:
  caution   low confidence or trust, or a low mood: second opinions on more answers, and a higher
            bar for what is stored in memory; good spirits lower that bar a little
  wary      trust is low: a language model's answer is stored only if something it already holds
            agrees
  energy    high arousal: it duplicates itself sooner and takes initiative sooner
  haste     low patience: it skips second opinions (unless wary) and duplicates itself sooner
  openness  good spirits: it explores more; a tense, low or wary mood checks what it already holds
            instead of reaching out (the agency node reads the mood as part of its situation)
The mood changes how carefully the directors work, never what is true.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .agents import Agent
from .knowledge import restate
from .temperament import DIMENSIONS, WITHHOLDING, Temperament

PRIMARY = "director-0"
ORDER_WORK = ("intake", "review", "consult")
TAG = "[director:"


def _snip(text, n: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================================ stance
def caution_of(v: dict) -> float:
    """How guarded the mood makes it: low confidence, low trust and a low mood all add to it."""
    return 0.4 * (1 - v["confidence"]) + 0.4 * (1 - v["trust"]) + 0.2 * max(0.0, -v["valence"])


@dataclass
class Stance:
    """How the mood shapes the directors right now."""
    mood: str
    values: dict
    caution: float
    review_confidence: float   # answers below this get a second opinion
    memory_confidence: float   # facts from orders below this are not stored
    wary: bool                 # trust is low: a language model's answer needs corroboration to be stored
    impatient: bool            # patience is low: no second opinions (unless wary)
    consult: bool              # may ask a duplicate for a second opinion
    spawn_waiting: int         # pieces of work waiting for a director before it duplicates itself
    explore: float             # multiplies the initiative's exploration
    pace: float                # multiplies the time between initiative decisions
    second_mode: float         # the drive a second mode of initiative needs
    effects: list = field(default_factory=list)

    def line(self) -> str:
        return f"{self.mood}: " + "; ".join(self.effects)

    def to_dict(self) -> dict:
        return {"mood": self.mood, "values": {k: round(v, 3) for k, v in self.values.items()},
                "caution": round(self.caution, 3), "review_confidence": round(self.review_confidence, 3),
                "memory_confidence": round(self.memory_confidence, 3), "wary": self.wary, "impatient": self.impatient,
                "consult": self.consult, "spawn_waiting": self.spawn_waiting, "explore": round(self.explore, 3),
                "pace": round(self.pace, 3), "second_mode": round(self.second_mode, 3), "effects": list(self.effects)}


def stance(values: dict, baseline: dict, mood: str, pol) -> Stance:
    """The directors' stance from the mood, measured against the baseline personality."""
    k = max(0.0, float(pol.mood_influence))
    b = {d: float(baseline.get(d, values[d])) for d in DIMENSIONS}
    v = {d: float(values[d]) for d in DIMENSIONS}
    dc = caution_of(v) - caution_of(b)
    da = v["arousal"] - b["arousal"]
    dh = b["patience"] - v["patience"]
    dv = v["valence"] - b["valence"]
    # a good mood lowers the bars a little; a guarded one raises them more
    review = _clamp(pol.review_confidence + k * 0.8 * dc, max(0.2, pol.review_confidence - 0.15), 0.95)
    memory = _clamp(pol.memory_confidence + k * 0.8 * dc - k * 0.15 * max(0.0, dv),
                    max(0.2, pol.memory_confidence - 0.15), 0.95)
    wary = k > 0 and v["trust"] < pol.wary_trust
    impatient = k > 0 and v["patience"] < pol.impatient_patience
    consult = bool(pol.consult) and (wary or not impatient)
    spawn_waiting = max(1, round(pol.spawn_after_waiting - k * 2.5 * (max(0.0, da) + max(0.0, dh))))
    explore = _clamp(1 + k * 2 * dv, 0.25, 2.0)
    pace = _clamp(1 - k * da, 0.5, 1.5)
    second = _clamp(pol.second_mode_min - k * 0.3 * da, 0.3, 1.0)
    effects = []
    if k == 0:
        effects.append("the mood has no effect on it ([directors] mood_influence = 0)")
    else:
        bar = f"second opinions below {review:.0%} confidence, stores facts at {memory:.0%} or more"
        if dc > 0.04:
            effects.append(f"cautious (confidence {v['confidence']:.2f}, trust {v['trust']:.2f}): {bar}")
        elif dc < -0.04:
            effects.append(f"assured: {bar}")
        else:
            effects.append(bar)
        if wary:
            effects.append(f"wary (trust {v['trust']:.2f}): stores a language model's answer only if something it holds agrees")
        if impatient:
            effects.append(f"impatient (patience {v['patience']:.2f}): " + ("" if wary else "skips second opinions, ")
                           + "duplicates itself sooner")
        if da > 0.15:
            effects.append(f"stirred up (arousal {v['arousal']:.2f}): duplicates itself sooner, takes initiative sooner")
        if dv > 0.15:
            effects.append("in good spirits: explores and reads up more")
        elif dv < -0.15:
            effects.append("low: checks what it already knows rather than reaching out")
    if not pol.consult:
        effects.append("second opinions are off ([directors] consult = false)")
    return Stance(mood if k > 0 else "steady", v, caution_of(v), review, memory, wary, impatient, consult,
                  spawn_waiting, explore, pace, second, effects)


# ========================================================================= agreement
_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "and", "or", "it", "its", "this",
         "that", "by", "for", "with", "as", "be", "from", "about", "i", "think", "answer", "source", "sources"}


def _words(text) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", str(text).lower()) if w not in _STOP]


def _marks(text) -> set[str]:
    """The names and numbers in an answer: where two answers about the same thing differ."""
    return {w.lower() for w in re.findall(r"\b(?:[A-Z][a-z]+|\d+(?:[.,]\d+)*)\b", str(text))} - _STOP


def agrees(a, b) -> bool:
    """Do two answers say the same thing? One contains the other; or most of the shorter one's words
    are in the longer one and they do not name different things or numbers."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    na, nb = f" {' '.join(wa)} ", f" {' '.join(wb)} "
    if na in nb or nb in na:
        return True
    ma, mb = _marks(a), _marks(b)
    if ma - mb and mb - ma:  # "Lutz is the capital" and "Pankrac is the capital" do not agree
        return False
    sa, sb = set(wa), set(wb)
    return len(sa & sb) / min(len(sa), len(sb)) >= 0.6


def _first_line(text) -> str:
    """An answer without its sources line."""
    return re.split(r"\n(?:Sources?:)", str(text or ""), maxsplit=1)[0].strip()


# ============================================================================= nodes
@dataclass
class DirectorNode:
    id: str
    role: str                  # primary | duplicate
    parent: str = ""
    reason: str = ""
    born: float = field(default_factory=time.time)
    copied: dict = field(default_factory=dict)
    state: str = "idle"        # idle | intake | review | consult
    work: str = ""
    since: float = field(default_factory=time.time)
    initiative: str = ""       # the initiative it is taking ("" = none); orders still come first
    last_active: float = field(default_factory=time.time)
    beat: float = 0.0
    handled: Counter = field(default_factory=Counter)
    retiring: bool = False
    task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def free(self) -> bool:
        return self.state == "idle" and not self.retiring

    @property
    def idle(self) -> bool:
        return self.free and not self.initiative

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def doing(self) -> str:
        if self.state != "idle":
            return f"{self.state}: {self.work}" if self.work else self.state
        return f"taking initiative: {self.initiative}" if self.initiative else "idle"

    def to_dict(self, now: float) -> dict:
        return {"id": self.id, "role": self.role, "parent": self.parent, "reason": self.reason,
                "age": round(now - self.born, 1), "state": self.state, "work": self.work, "initiative": self.initiative,
                "doing": self.doing(), "idle_for": round(now - self.last_active, 1) if self.idle else 0.0,
                "handled": dict(self.handled), "running": self.running, "retiring": self.retiring, "copied": dict(self.copied)}


@dataclass
class Plan:
    """What the intake director decided about an order; the review director finishes it."""
    order_id: str
    director: str
    stance: Stance
    people: bool               # an order for a person (not a benchmark exam): notes and second opinions
    consult: bool
    started: float
    remember: list = field(default_factory=list)  # (ask, answer, source, by, kind): stored only after review
    line: str = ""


# ============================================================================== pool
class DirectorPool(Agent):
    """The director nodes (see the module docstring). Runs as an agent that keeps director-0 alive."""
    name = "directors"

    def __init__(self, system):
        super().__init__(system)
        self.pol = system.directives.directors
        self.nodes: dict[str, DirectorNode] = {}
        self.plans: dict[str, Plan] = {}
        self.waiting = 0
        self.started = False
        self._next = 1
        self._saved = 0.0
        self.reviews: deque = deque(maxlen=40)
        self.memory_log: deque = deque(maxlen=40)
        self.events: deque = deque(maxlen=40)
        raw = system.store.get_kv("directors.state")
        state = json.loads(raw) if raw else {}
        self.counts: Counter = Counter(state.get("counts", {}))
        self.restarts = int(state.get("restarts", 0))
        self.most = int(state.get("most", 1))
        self.ensure()

    @property
    def interval(self) -> float:
        return max(0.1, self.sys.directives.background.tick_seconds * 4)

    @property
    def limit(self) -> int:
        return max(1, int(self.pol.max_directors))

    def save(self) -> None:
        self._saved = time.time()
        self.sys.store.set_kv("directors.state", json.dumps({"counts": dict(self.counts), "restarts": self.restarts,
                                                              "most": self.most}))

    # ---------------------------------------------------------------- the mood
    def stance(self) -> Stance:
        """The stance from the mood now (computed on a copy: the temperament itself is not touched)."""
        a = self.sys.affect
        t = Temperament(dict(a.state.baseline), a.state.half_life, values=dict(a.state.values), updated=a.state.updated)
        t.settle(a.clock())
        return stance(t.values, a.state.baseline, t.mood(), self.pol)

    # ------------------------------------------------------------- life cycle
    def ensure(self) -> DirectorNode:
        """There is always at least one director node: director-0 is made, or restarted, if it is missing."""
        p = self.nodes.get(PRIMARY)
        if p is None:
            p = DirectorNode(PRIMARY, "primary", reason="the primary director: there is always at least one")
            self.nodes[PRIMARY] = p
        if self.started and not self.sys.stopping and not p.running:
            stopped = p.task is not None
            self._start(p)
            if stopped:
                self.restarts += 1
                self.sys.guardian.audit("directors", "director.restarted", PRIMARY)
                self.sys.bus.publish("director.restarted", director=PRIMARY, running=len(self.nodes))
                self.save()
        return p

    def _start(self, node: DirectorNode) -> None:
        node.beat = time.time()
        node.task = asyncio.create_task(self._live(node), name=f"director:{node.id}")
        self.sys._bg_tasks.add(node.task)
        node.task.add_done_callback(self.sys._bg_tasks.discard)

    async def _live(self, node: DirectorNode) -> None:
        """A director node's own loop: it stays alive; a duplicate retires itself after idling long enough."""
        while not self.sys.stopping and self.nodes.get(node.id) is node:
            node.beat = time.time()
            if node.role == "duplicate" and node.idle and time.time() - node.last_active >= self.pol.retire_after_idle_seconds:
                self.retire(node.id, f"idle for {time.time() - node.last_active:.0f}s", by=node.id)
                return
            await asyncio.sleep(max(0.02, min(self.pol.beat_seconds, self.pol.retire_after_idle_seconds / 2)))

    async def run(self) -> None:
        self.started = True
        self.ensure()
        for node in list(self.nodes.values()):
            if not node.running:
                self._start(node)
        await super().run()

    async def step(self) -> None:
        self.ensure()  # director-0 restarted if it stopped
        if time.time() - self._saved > 30:
            self.save()

    def can_spawn(self) -> bool:
        return len(self.nodes) < self.limit and not self.sys.guardian.halted and not self.sys.stopping

    def spawn(self, reason: str, by: str = PRIMARY) -> DirectorNode | None:
        """A duplicate of a director: the same policies and the same agency node."""
        if not self.can_spawn():
            return None
        while f"director-{self._next}" in self.nodes:
            self._next += 1
        nid = f"director-{self._next}"
        self._next += 1
        src = self.nodes.get(by) or self.nodes[PRIMARY]
        agency = self.sys.registry.champion("agency") if "agency" in self.sys.capabilities else None
        node = DirectorNode(nid, "duplicate", parent=src.id, reason=reason,
                            copied={"agency_node": agency.id if agency else None, "mood": self.stance().mood})
        self.nodes[nid] = node
        self.counts["spawned"] += 1
        self.most = max(self.most, len(self.nodes))
        if self.started:
            self._start(node)
        self.events.append({"ts": round(time.time(), 1), "event": "spawned", "director": nid, "reason": reason})
        self.sys.guardian.audit(src.id, "director.spawned", nid, reason=reason)
        self.sys.bus.publish("director.spawned", director=nid, parent=src.id, reason=reason, running=len(self.nodes))
        self.save()
        return node

    def retire(self, nid: str, reason: str, by: str = "directors") -> tuple[bool, str]:
        node = self.nodes.get(nid)
        if node is None:
            return False, f"no director {nid} (/directors lists them)"
        if node.role == "primary":
            return False, f"{nid} never retires: there is always at least one director"
        if len(self.nodes) <= 1:
            return False, "the last director never retires"
        if not node.idle:
            node.retiring = True
            return True, f"{nid} retires once it has finished ({node.doing()})"
        del self.nodes[nid]
        if node.task is not None and not node.task.done() and node.task is not asyncio.current_task():
            node.task.cancel()
        self.counts["retired"] += 1
        self.events.append({"ts": round(time.time(), 1), "event": "retired", "director": nid, "reason": reason})
        self.sys.guardian.audit(by, "director.retired", nid, reason=reason)
        self.sys.bus.publish("director.retired", director=nid, reason=reason, by=by, running=len(self.nodes))
        self.save()
        return True, f"{nid} retired ({reason})"

    # ------------------------------------------------------------------ work
    def _free(self, kind: str, exclude) -> DirectorNode | None:
        """Order work takes any director not already on order work (orders come before initiative, P2);
        initiative takes a director with nothing to do."""
        ok = (lambda n: n.free) if kind in ORDER_WORK else (lambda n: n.idle)
        nodes = [n for n in self.nodes.values() if n.id not in exclude and ok(n)]
        nodes.sort(key=lambda n: (n.role != "primary", bool(n.initiative), n.born))
        return nodes[0] if nodes else None

    async def acquire(self, kind: str, what: str = "", exclude=(), wait: float | None = None,
                      spawn: bool = True) -> DirectorNode | None:
        """A director for a piece of work; a duplicate is made when work waits for one."""
        self.ensure()
        t0 = time.monotonic()
        spawned = False
        self.waiting += 1
        try:
            while True:
                node = self._free(kind, exclude)
                if node is not None:
                    if kind == "initiative":
                        node.initiative = what or "deciding"
                    else:
                        node.state, node.work = kind, _snip(what, 80)
                    node.since = node.last_active = time.time()
                    return node
                if spawn and not spawned and self.can_spawn():
                    why = None
                    if kind == "consult":
                        why = f"a second opinion ({_snip(what, 60)})"
                    elif kind == "initiative":
                        why = f"a second mode of initiative ({what})"
                    elif self.waiting >= self.stance().spawn_waiting:
                        why = f"{self.waiting} piece{'s' if self.waiting != 1 else ''} of work waiting for a director"
                    if why:
                        spawned = True
                        busy = [n for n in self.nodes.values() if not n.free]
                        if self.spawn(why, by=busy[0].id if busy else PRIMARY) is not None:
                            continue
                if wait is not None and time.monotonic() - t0 >= wait:
                    return None
                await asyncio.sleep(0.002)
        finally:
            self.waiting -= 1

    def release(self, node: DirectorNode, kind: str) -> None:
        if kind == "initiative":
            node.initiative = ""
        else:
            node.state, node.work = "idle", ""
        node.handled[kind] += 1
        self.counts[kind] += 1
        node.last_active = time.time()
        if node.retiring and node.idle and node.id in self.nodes:
            node.retiring = False
            self.retire(node.id, "asked to retire; it finished its work first", by=node.id)

    @asynccontextmanager
    async def working(self, kind: str, what: str = "", exclude=(), wait: float | None = None, spawn: bool = True):
        node = await self.acquire(kind, what, exclude, wait, spawn)
        try:
            yield node
        finally:
            if node is not None:
                self.release(node, kind)

    # ---------------------------------------------------------------- intake
    async def intake(self, order) -> Plan:
        """A director takes the order before anything else touches it."""
        async with self.working("intake", order.text) as d:
            st = self.stance()
            people = not str(order.source).startswith("challenge:")
            plan = Plan(order.id, d.id, st, people, consult=people and st.consult, started=time.time())
            if not people:
                how = "a benchmark question: answered as the mesh is, no second opinions"
            elif plan.consult:
                how = f"a second opinion if unsure (below {st.review_confidence:.0%})"
            else:
                how = "no second opinions" + (" (impatient)" if st.impatient else "")
            plan.line = f"{d.id} took the order ({st.mood}; {how}; facts stored at {st.memory_confidence:.0%} or more)"
            self.plans[order.id] = plan
            while len(self.plans) > 500:
                self.plans.pop(next(iter(self.plans)))
            return plan

    def remember(self, order_id: str, ask: str, answer, source: str, by: str, kind: str):
        """A fact an order wants to keep waits for that order's review; any other write is reviewed now."""
        plan = self.plans.get(order_id)
        if plan is not None:
            plan.remember.append((ask, answer, source, by, kind))
            return None
        return self.sys.memory.learn(ask, answer, source, by=by, kind=kind)

    # ---------------------------------------------------------------- review
    def _needs_second_opinion(self, res, plan: Plan, st: Stance) -> str | None:
        if not plan.consult or not res.ok:
            return None
        route = res.route or ""
        if route in ("tasks", "mind") or route.startswith("dream:") or res.served_by == "explicit" and not route.startswith("skill:"):
            return None
        if route.startswith("tool:") and res.knowledge not in ("researched", "recalled"):
            return None  # tools are vetted, self-tested and run in the sandbox
        teacher = res.served_by in ("teachers", "mixed") or any("teacher" in str(x) for x in res.sources)
        if res.confidence < st.review_confidence:
            return f"confidence {res.confidence:.2f} is below {st.review_confidence:.2f}"
        if teacher:
            return "a language model's answer, not checked against a source"
        if st.wary and res.knowledge == "researched":
            return "trust is low and this came from outside the mesh"
        return None

    def _second_opinion(self, order, res) -> dict:
        """An independent answer from what the mesh already holds: never a teacher or the web."""
        return self._from_nodes(order, res) or self._from_knowledge(order, res) or self.UNCHECKED

    UNCHECKED = {"verdict": "unchecked", "answer": "", "source": "nothing it holds bears on this"}

    def _from_nodes(self, order, res) -> dict | None:
        """A second opinion from another node or from the memory nodes (numpy only: runs off the event loop)."""
        s = self.sys
        route = res.route or ""
        if route.startswith("skill:"):
            cap = route[6:]
            step = next((x for x in reversed(res.steps) if x.get("capability") == cap), None)
            if step is None:
                return {"verdict": "unchecked", "answer": "", "source": "nothing to check"}
            # only a node trusted on held-out data has an opinion worth weighing
            trusted = [n for n in [s.registry.champion(cap)] + s.registry.candidates(cap) if n is not None
                       and n.params is not None and (n.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy]
            prov = str(step.get("provenance", ""))
            used = prov.removeprefix("node:").split(" ")[0] if prov.startswith("node:") else ""
            others = sorted((n for n in trusted if n.id != used), key=lambda n: -(n.metrics.get("holdout_acc") or 0))
            node = others[0] if others else None
            if node is None:
                return {"verdict": "unchecked", "answer": "", "source": f"no other trusted node for {cap} yet"}
            label, conf = node.predict(s.encode(step["input"]))
            return {"verdict": "agrees" if label == res.output else "disagrees", "answer": label,
                    "source": f"node {node.id} ({conf:.0%})"}
        if res.knowledge != "memory":
            rec = s.memory.recall(order.text, quiet=True, threshold=self.pol.second_opinion_threshold)
            if rec is not None:  # it looks at what it holds less surely, too
                return {"verdict": "agrees" if agrees(rec.answer, self._given(res)) else "disagrees",
                        "answer": rec.answer, "source": f"memory node {rec.node.id} ({rec.confidence:.0%} sure)"}
        return None

    @staticmethod
    def _given(res) -> str:
        return _first_line(res.output if res.output is not None else res.answer)

    def _from_knowledge(self, order, res) -> dict | None:
        """A second opinion from retained knowledge (the database: on the event loop)."""
        s = self.sys
        floor = self.pol.second_opinion_threshold
        if (res.route or "").startswith("skill:"):
            return None
        if res.knowledge not in ("recalled", "researched"):
            need = max(floor, s.directives.research.recall_threshold - 0.2)
            hits, conf = s.knowledge.recall(order.text, need)
            if hits and conf >= need:
                answer, _ = restate(order.text, hits)
                return {"verdict": "agrees" if agrees(answer, self._given(res)) else "disagrees", "answer": answer,
                        "source": f"retained knowledge '{hits[0].doc.title}' ({conf:.0%} match)"}
        return None

    async def consult(self, asking: DirectorNode, order, res, why: str) -> dict | None:
        """Ask another director (a duplicate, made if none is free) for a second opinion."""
        async with self.working("consult", f"{res.order_id}: {order.text}", exclude={asking.id},
                                wait=self.pol.consult_wait_seconds) as helper:
            if helper is None:
                res.trace.append(f"{asking.id} wanted a second opinion ({why}) but no other director was free")
                return None
            out = (await asyncio.to_thread(self._from_nodes, order, res)  # its own work, off the event loop
                   or self._from_knowledge(order, res) or dict(self.UNCHECKED))
            out.update(by=helper.id, why=why)
            res.trace.append(f"{asking.id} asked {helper.id} for a second opinion ({why}): {out['verdict']}"
                             f" - {out['source']}" + (f": '{_snip(out['answer'], 60)}'" if out["verdict"] == "disagrees" else ""))
            self.sys.bus.publish("director.consulted", order=res.order_id, asked=asking.id, by=helper.id,
                                 verdict=out["verdict"], why=why, source=out["source"], answer=_snip(out["answer"], 120),
                                 text=_snip(order.text, 120))
            return out

    async def review(self, order, res) -> None:
        """A director sees the result after the tools, nodes and engines, before the person does,
        and decides what may go into long-term memory."""
        plan = self.plans.pop(res.order_id, None)
        if plan is None:  # it never reached intake (an internal error before it): review it all the same
            st = self.stance()
            plan = Plan(res.order_id, PRIMARY, st, not str(order.source).startswith("challenge:"), False, time.time())
        async with self.working("review", order.text) as d:
            st = self.stance()  # the mood now, after the answer
            verdict, second, note = "approved", None, ""
            if not res.ok:
                verdict = "reported as it is (not fulfilled)"
                plan.remember.clear()
            elif (res.route or "").startswith("dream:") or res.knowledge == "imagined":
                verdict = "approved: labelled as imagined, never stored (P13)"
                plan.remember.clear()
            else:
                why = self._needs_second_opinion(res, plan, st) if plan.consult else None
                if why:
                    second = await self.consult(d, order, res, why)
                if second is not None and second["verdict"] == "agrees":
                    verdict = f"approved: {second['by']} agrees ({second['source']})"
                elif second is not None and second["verdict"] == "disagrees":
                    verdict = f"approved with a note: {second['by']} found a different answer ({second['source']})"
                    res.confidence = round(min(res.confidence, 0.5), 3)
                    note = (f"{TAG} {second['by']} checked {second['source']} and found a different answer: "
                            f"'{_snip(second['answer'], 120)}'. Treat this answer with care]")
                elif second is not None:
                    verdict = f"approved: {second['by']} found {second['source']}"
            if note and plan.people and res.disposition not in WITHHOLDING:  # never leaks a held answer
                res.answer = f"{res.answer}\n{note}" if res.answer else note
            elif note:
                res.trace.append(f"{d.id}'s note was not shown: " + ("the answer is held" if plan.people else "an exam answer"))
            corroborated = second is not None and second["verdict"] == "agrees"
            disputed = second is not None and second["verdict"] == "disagrees"
            for ask, answer, source, by, kind in plan.remember:  # long-term memory: only through this review
                ok, why = self.review_memory(ask, answer, source, by, kind, confidence=res.confidence,
                                             corroborated=corroborated, disputed=disputed, director=d.id)
                if ok:
                    made = self.sys.memory.learn(ask, answer, source, by=by, kind=kind, reviewed=True)
                    if made is not None:
                        res.trace.append(f"{d.id} stored it in long-term memory ({made[0]}, {made[1]})")
                else:
                    res.trace.append(f"{d.id} did not store it in long-term memory: {why}")
            res.director, res.review = d.id, verdict
            res.trace.append(f"reviewed by {d.id} ({st.mood}): {verdict}")
            self.reviews.append({"ts": round(time.time(), 1), "order": res.order_id, "text": _snip(order.text, 80),
                                 "intake": plan.director, "director": d.id, "verdict": verdict, "mood": st.mood,
                                 "consulted": second["by"] if second else "", "second": second["verdict"] if second else ""})
            self.sys.bus.publish("director.reviewed", order=res.order_id, director=d.id, intake=plan.director,
                                 verdict=verdict, mood=st.mood, consulted=second["by"] if second else "")

    # ------------------------------------------------------------ memory gate
    def review_memory(self, ask: str, answer, source: str = "", by: str = "", kind: str = "people", *,
                      confidence: float | None = None, corroborated: bool = False, disputed: bool = False,
                      director: str | None = None) -> tuple[bool, str]:
        """May this go into long-term memory? A person's word is always kept (P1); an imagined answer
        never (P13); the rest depends on where it came from, the answer's confidence and the mood."""
        st = self.stance()
        src = str(source or "").lower()
        teacher = kind == "teachers" or "teacher" in src or "language model" in src
        ok, why = True, "reviewed"
        if "imagined" in src and "dream" in src:
            ok, why = False, "imagined, not a fact (P13)"
        elif kind == "people":
            why = "a person's word (P1)"
        elif kind == "challenges" or src.startswith("challenge"):
            why = "a benchmark's own answer"
        elif disputed:
            ok, why = False, "a second director found a different answer"
        elif st.wary and teacher and not corroborated:
            ok, why = False, (f"trust is low ({st.values['trust']:.2f}) and nothing it holds agrees with this "
                              "language model's answer")
        elif confidence is not None and confidence < st.memory_confidence and not corroborated:
            ok, why = False, (f"confidence {confidence:.2f} is below the {st.memory_confidence:.2f} it needs now "
                              f"({st.mood})")
        else:
            held = self.sys.memory.known(ask)
            told = held is not None and (held.source == "the operator" or held.source.startswith("you told me"))
            if told and held.answer.strip().lower() != str(answer).strip().lower():
                ok, why = False, f"a person told it otherwise ('{_snip(held.answer, 60)}')"
            elif corroborated:
                why = "corroborated by what it already holds"
        who = director or PRIMARY
        self.counts["memory.kept" if ok else "memory.declined"] += 1
        self.memory_log.append({"ts": round(time.time(), 1), "director": who, "ask": _snip(ask, 80), "kept": ok,
                                "why": why, "kind": kind})
        if not ok:
            self.sys.bus.publish("director.memory", director=who, ask=_snip(ask, 120), kept=False, why=why, kind=kind)
        return ok, why

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        now = time.time()
        return {"running": sum(1 for n in self.nodes.values() if n.running or not self.started),
                "limit": self.limit, "most": self.most, "restarts": self.restarts, "waiting": self.waiting,
                "stance": self.stance().to_dict(), "nodes": [n.to_dict(now) for n in self.nodes.values()],
                "counts": dict(self.counts), "reviews": list(self.reviews)[-15:], "memory": list(self.memory_log)[-15:],
                "events": list(self.events)[-10:], "consult": bool(self.pol.consult)}

    def lines(self) -> list[str]:
        st = self.stance()
        now = time.time()
        c = self.counts
        out = [f"  directors: {len(self.nodes)} running (at most {self.limit}; {self.most} at once so far)"
               + (f", director-0 restarted {self.restarts}x" if self.restarts else ""),
               f"  stance {st.line()}",
               f"  handled: {c.get('intake', 0)} orders taken, {c.get('review', 0)} reviewed, {c.get('consult', 0)} second "
               f"opinions; memory: {c.get('memory.kept', 0)} kept, {c.get('memory.declined', 0)} declined"]
        for n in self.nodes.values():
            h = n.handled
            mark = "*" if n.role == "primary" else " "
            who = "primary" if n.role == "primary" else f"duplicate of {n.parent}, {now - n.born:.0f}s old ({n.reason})"
            out.append(f"  {mark} {n.id} {who}: {n.doing()} - {h.get('intake', 0)} taken, {h.get('review', 0)} reviewed, "
                       f"{h.get('consult', 0)} second opinions, {h.get('initiative', 0)} initiatives")
        recent = list(self.reviews)[-5:]
        if recent:
            out.append("  recent reviews:")
            out += [f"    {r['order']} '{r['text'][:40]}' by {r['director']}: {r['verdict']}" for r in reversed(recent)]
        declined = [m for m in self.memory_log if not m["kept"]][-5:]
        if declined:
            out.append("  not stored in memory:")
            out += [f"    '{m['ask'][:50]}' ({m['director']}): {m['why']}" for m in reversed(declined)]
        return out
