"""Independent agency: the mesh decides for itself what to learn and what to test next.

The Director is the mesh's own initiative. When no one has given it an order for a while and
background capacity is free, it looks at its situation - skills below target, benchmark exams
that are due, topics it could read up on, memory nodes that need checking, its own budgets - and
picks a mode:

    research     assign itself a learn task on a topic: a goal of its own, a question it could not
                 answer, an interest you told it about, or a link from something it read
    practise     assign itself a challenge task: practise on a suite (learning from every grade),
                 then re-sit its exam
    examine      sit a benchmark exam now, to measure itself
    improve      assign itself an improve task for its weakest skill
    consolidate  run every memory node against its test file, and relearn what fails
    rest         nothing is worth the budget right now

The Director is director-0, the primary director node (directors.py), and its mood shapes it: the
agency node reads the mood as part of its situation (a good mood reads up and practises more; a
tense, low or wary one checks what it already holds instead of reaching outside), good spirits
explore more, high arousal decides sooner. When two modes are both worth doing, a duplicate of the
director takes the second one alongside it.

A neural node sits in front of the Director: the agency node, an internal capability like the
emotional engines. It reads the situation and chooses the mode once it is validated and
confident; until then the drive policy (curiosity, competence, verification, variety) chooses and
its choices become the node's training data. Every choice is scored afterwards by what it achieved
- articles read and facts compressed, held-out accuracy gained, exam scores measured or improved,
memory faults found - and that outcome teaches the node too: a choice that paid off in a situation
is reinforced; one that did not is relabelled with the policy's next best. So the node comes to
prefer what has actually worked for this mesh, not just what the policy says.

People stay in charge (P1, P12): every decision is audited and logged, every task it assigns
itself shows up in /tasks where it can be paused or removed, a person can pin a mode or turn the
initiative off, and it never adds capabilities or approves tools (P10). It runs only when idle
(P2) and within its own hourly budget of decisions (P4).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import deque

from .agents import Agent, is_question
from .tasks import DIRECT
from .teachers import Lesson, Teacher

AGENCY = "agency"
MODES = ("research", "practise", "examine", "improve", "consolidate", "rest")
TASK_MODES = ("research", "practise", "improve")
FEATURES = ("skills", "gap", "topics", "research", "exams", "scores", "memory", "last", "streak", "budget",
            "selftasks", "mood")
MOODS = ("steady", "upbeat", "content", "alert", "tense", "subdued", "wary")  # the temperament's moods
VALUES = {"skills": ("weak", "ok", "none"), "gap": ("none", "small", "large"), "topics": ("none", "some", "many"),
          "research": ("ready", "off", "spent"), "exams": ("due", "fresh", "none"), "scores": ("low", "high", "none"),
          "memory": ("failing", "unverified", "ok", "none"), "last": MODES + ("none",),
          "streak": ("one", "two", "many"), "budget": ("ok", "low"), "selftasks": ("none", "some", "full"),
          "mood": MOODS}
WHY = {"research": "there are topics to read up on", "improve": "a skill is below its target",
       "examine": "an exam is due", "practise": "its exam scores are low",
       "consolidate": "memory nodes need checking", "rest": "nothing is worth the budget right now"}
# How the mood moves the drives (only of modes already worth something; rest always can).
MOOD_DRIVES = {"upbeat": {"research": 0.15, "practise": 0.05}, "content": {"research": 0.08},
               "alert": {"examine": 0.08, "consolidate": 0.05},
               "tense": {"consolidate": 0.15, "examine": 0.1, "research": -0.1},
               "subdued": {"rest": 0.3, "research": -0.1},
               "wary": {"consolidate": 0.2, "research": -0.3}}
MOOD_WHY = {"upbeat": "it is in good spirits, so it reads up and practises",
            "content": "it is content, so it follows its curiosity",
            "alert": "it is alert, so it measures and checks itself",
            "tense": "it is tense, so it checks what it already knows",
            "subdued": "its mood is low, so it rests",
            "wary": "it is wary, so it checks what it already holds instead of reaching outside"}


# ======================================================================== situation
def situation_text(flags: dict) -> str:
    """The situation as single words ("skillsweak topicsmany ...") so every feature is one token
    for the hashing encoder the agency node reads."""
    return " ".join(f"{k}{flags.get(k, VALUES[k][0])}" for k in FEATURES)


def parse_situation(text: str) -> dict:
    out = {}
    for word in text.split():
        for k in FEATURES:
            if word.startswith(k) and word[len(k):] in VALUES[k]:
                out[k] = word[len(k):]
                break
    return out


def drives(f: dict) -> dict[str, float]:
    """How much each mode is worth in a situation: the drive policy."""
    d = {m: 0.0 for m in MODES}
    d["rest"] = 0.2
    if f.get("budget") == "low":
        d["rest"] = 1.0
    tasks_free = f.get("selftasks") != "full"
    if tasks_free and f.get("research") == "ready" and f.get("topics") != "none":
        d["research"] = 0.5 + (0.15 if f.get("topics") == "many" else 0.0)                 # curiosity
    if tasks_free and f.get("skills") == "weak":
        d["improve"] = 0.45 + {"large": 0.3, "small": 0.1}.get(f.get("gap"), 0.0)          # competence
    if f.get("exams") == "due":
        d["examine"] = 0.6 + (0.1 if f.get("scores") == "none" else 0.0)                   # measure
    if tasks_free and f.get("exams") in ("due", "fresh") and f.get("scores") == "low":
        d["practise"] = 0.55
    if f.get("memory") == "failing":
        d["consolidate"] = 0.85                                                              # verification
    elif f.get("memory") == "unverified":
        d["consolidate"] = 0.4
    last = f.get("last")
    if last in d and last != "rest":                                                         # variety
        d[last] -= {"one": 0.0, "two": 0.15, "many": 0.35}.get(f.get("streak"), 0.0)
    for m, delta in MOOD_DRIVES.get(f.get("mood", "steady"), {}).items():                    # the mood
        if d[m] > 0 or m == "rest":
            d[m] += delta
    return d


def policy(f: dict, exclude: tuple = ()) -> tuple[str, str]:
    """The drive policy's choice and why. Ties go to the earlier mode."""
    d = drives(f)
    best = max((m for m in MODES if m not in exclude), key=lambda m: (d[m], -MODES.index(m)))
    mood = f.get("mood", "steady")
    if mood in MOOD_WHY and f.get("budget") != "low":
        calm = drives({**f, "mood": "steady"})
        if best != max((m for m in MODES if m not in exclude), key=lambda m: (calm[m], -MODES.index(m))):
            return best, MOOD_WHY[mood]
    if best == "rest" and f.get("budget") == "low":
        return best, "its own budget of decisions is nearly spent"
    if best == "consolidate" and f.get("memory") == "failing":
        return best, "memory nodes are failing tests"
    if best == "consolidate":
        return best, "its memory nodes have not been verified lately"
    return best, WHY[best]


def feasible(f: dict) -> set[str]:
    out = {"rest"}
    tasks_free = f.get("selftasks") != "full"
    if tasks_free and f.get("research") == "ready" and f.get("topics") != "none":
        out.add("research")
    if tasks_free and f.get("skills") == "weak":
        out.add("improve")
    if f.get("exams") == "due":
        out.add("examine")
    if tasks_free and f.get("exams") != "none" and f.get("scores") == "low":
        out.add("practise")
    if f.get("memory") != "none":
        out.add("consolidate")
    return out


class AgencyTeacher(Teacher):
    """The drive policy as a teacher: it labels situations for the agency node and writes practice
    situations for every mode. A specialist, asked only by the agency capability."""
    name = "agency"
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.enabled = lambda: True

    def status(self) -> dict:
        return {"detail": "specialist: teaches the agency node the drive policy"}

    async def label(self, spec, text):
        if spec.name != AGENCY or not self.enabled():
            return None
        f = parse_situation(text)
        if len(f) < len(FEATURES):
            return None
        mode, why = policy(f)
        return Lesson(mode, 0.9, self.name, why)

    async def generate(self, spec, label, n):
        if spec.name != AGENCY or not self.enabled():
            return []
        out: list[str] = []
        for _ in range(n * 300):
            f = {k: self.rng.choice(v) for k, v in VALUES.items()}
            if f["skills"] != "weak":
                f["gap"] = "none"
            if policy(f)[0] == label:
                text = situation_text(f)
                if text not in out:
                    out.append(text)
            if len(out) >= n:
                break
        return out


# ========================================================================= director
def _clean_topic(text: str) -> str:
    t = " ".join(str(text).split()).strip(" ?!.'\"")
    t = re.sub(r"\s+(every|max|depth|practice|exam|to|until|notify)\s+\S+\s*$", "", t, flags=re.I)
    return t[:120]


class Director(Agent):
    """The mesh's own initiative (see the module docstring). The agency node chooses the mode."""
    name = "director"

    def __init__(self, system):
        super().__init__(system)
        self.pol = system.directives.agency
        self.rng = random.Random(system.rng.randrange(1 << 30))
        raw = system.store.get_kv("agency.state")
        state = json.loads(raw) if raw else {}
        self.pinned: str = state.get("pinned", "auto")
        self.override: bool | None = state.get("enabled")      # a person turned it on or off
        self.self_tasks: dict[str, dict] = state.get("self_tasks", {})
        self.values: dict[str, float] = state.get("values", {})
        self.counts: dict[str, int] = state.get("counts", {})
        self.history: deque = deque(state.get("history", []), maxlen=60)
        self.verified_at: float = state.get("verified_at", 0.0)
        self.failed_topics: list = state.get("failed_topics", [])
        # targets that did not pay off rest for a while ("practise:gsm8k" -> until), so it does not loop
        self.cooldown: dict[str, float] = {k: v for k, v in state.get("cooldown", {}).items() if v > time.time()}
        self.recent: deque = deque(t for t in state.get("recent", []) if time.time() - t < 3600)
        self.next_at = time.time() + self.pol.idle_seconds
        self.busy = False
        teacher = system.teachers.teachers.get(AGENCY)
        if teacher is not None:
            teacher.enabled = lambda: self.enabled

    # ---------------------------------------------------------------- settings
    @property
    def enabled(self) -> bool:
        return self.pol.enabled if self.override is None else self.override

    @property
    def interval(self) -> float:
        return max(0.05, self.sys.directives.background.tick_seconds * 2)

    def save(self) -> None:
        state = {"pinned": self.pinned, "enabled": self.override, "self_tasks": self.self_tasks, "values": self.values,
                 "counts": self.counts, "history": list(self.history), "verified_at": self.verified_at,
                 "failed_topics": self.failed_topics[-50:], "recent": list(self.recent), "cooldown": self.cooldown}
        self.sys.store.set_kv("agency.state", json.dumps(state))

    def set_mode(self, mode: str, by: str) -> tuple[bool, str]:
        mode = mode.strip().lower()
        if mode not in ("auto",) + MODES:
            return False, f"modes: auto, {', '.join(MODES)}"
        if mode != "auto" and mode not in self.pol.modes:
            return False, f"the directives do not allow the {mode} mode ([agency] modes)"
        self.pinned = mode
        self.save()
        self.sys.guardian.audit(by, "agency.mode", mode)
        self.sys.bus.publish("agency.mode", mode=mode, by=by)
        return True, ("the agency node chooses the mode again" if mode == "auto"
                      else f"pinned to {mode}: it {self._describe(mode)} whenever it takes initiative")

    def set_enabled(self, on: bool, by: str) -> str:
        self.override = bool(on)
        if on:
            self.next_at = min(self.next_at, time.time() + self.pol.idle_seconds)
        self.save()
        self.sys.guardian.audit(by, "agency.on" if on else "agency.off", "director")
        self.sys.bus.publish("agency.toggled", on=bool(on), by=by)
        return ("initiative on: it decides what to learn and test when it is idle" if on
                else "initiative off: it only does what people ask (the tasks it set itself keep running until "
                     "you remove them)")

    @staticmethod
    def _describe(mode: str) -> str:
        return {"research": "sets itself a learn task", "practise": "practises on a benchmark suite",
                "examine": "sits a benchmark exam", "improve": "trains its weakest skill",
                "consolidate": "checks its memory nodes against their tests", "rest": "rests"}[mode]

    # ---------------------------------------------------------------- the loop
    async def step(self) -> None:
        s = self.sys
        if s.guardian.halted or s.guardian.paused:
            return
        if self.self_tasks:
            self.review()
        if not self.enabled or time.time() < self.next_at:
            return
        if s.inflight or not s.guardian.background_allowed():
            return
        if time.time() - getattr(s, "last_order_at", 0.0) < self.pol.idle_seconds:
            return
        await self.decide()

    def _budget_left(self) -> int:
        now = time.time()
        while self.recent and now - self.recent[0] > 3600:
            self.recent.popleft()
        return self.pol.max_decisions_per_hour - len(self.recent)

    def situation(self) -> dict:
        """What the Director knows when it decides: the flags the agency node reads, and the
        concrete candidates for each mode."""
        s, pol = self.sys, self.pol
        npol = s.directives.nodes
        now = time.time()
        cool = lambda mode, target: self.cooldown.get(f"{mode}:{target}", 0.0) > now  # noqa: E731
        focus = s.tasks.focus() if s.tasks.enabled else {}
        weak = []
        for cap in s.capabilities:
            if cap == AGENCY or cap in focus or cool("improve", cap):
                continue
            champ = s.registry.champion(cap)
            acc = champ.metrics.get("holdout_acc") if champ else None
            if acc is not None and champ.metrics.get("holdout_n", 0) >= npol.min_eval_samples and acc < npol.target_accuracy:
                weak.append((cap, round(acc, 4), npol.target_accuracy))
        weak.sort(key=lambda w: w[1] - w[2])
        gap = "none" if not weak else ("large" if weak[0][2] - weak[0][1] >= 0.05 else "small")
        topics = [t for t in self.topics() if not cool("research", t[0].lower())]
        tool = s.knowledge_tool()
        if tool is None:
            research = "off"
        elif not s.guardian.authorize_research(background=True, unlimited=s.unlimited(tool)).allowed:
            research = "spent"
        else:
            research = "ready"
        ch = s.challenges
        due, practice, scores = [], [], []
        if ch.enabled and ch.suites:
            busy = {t.target for t in s.tasks.tasks.values() if t.kind == "challenge"}
            for row in ch.summary():
                name = row["suite"]
                if name in busy or not row["exam"]:
                    continue
                exams = [r for r in ch.history if r["suite"] == name and r["mode"] == "exam"]
                practised = [r for r in ch.history if r["suite"] == name and r["mode"] == "practice"]
                last_exam = exams[-1]["created_at"] if exams else None
                if cool("examine", name):
                    pass
                elif last_exam is None:
                    due.append((name, "never examined"))
                elif practised and practised[-1]["created_at"] > last_exam:
                    due.append((name, "practised since its last exam"))
                elif now - last_exam > pol.reexam_after_seconds:
                    due.append((name, "its last exam is old"))
                if row["last_exam"] is not None:
                    scores.append(row["last_exam"])
                    if row["last_exam"] < pol.low_score and not cool("practise", name):
                        practice.append((name, row["last_exam"]))
            practice.sort(key=lambda p: p[1])
        any_suite = bool(ch.enabled and any(r["exam"] for r in ch.summary())) if ch.enabled else False
        mem = s.memory
        if not mem.enabled or not mem.nodes:
            memory = "none"
        elif any(n.failing or (n.total and n.passed < n.total) for n in mem.nodes.values() if not n.dirty):
            memory = "failing"
        elif now - self.verified_at > pol.verify_after_seconds:
            memory = "unverified"
        else:
            memory = "ok"
        modes = [h["mode"] for h in self.history]
        last = modes[-1] if modes else "none"
        streak = 0
        for m in reversed(modes):
            if m != last:
                break
            streak += 1
        active = [t for t in self.self_tasks if t in s.tasks.tasks]
        left = self._budget_left()
        flags = {
            "skills": "weak" if weak else ("ok" if any(s.registry.champion(c) for c in s.capabilities if c != AGENCY) else "none"),
            "gap": gap, "topics": "none" if not topics else ("many" if len(topics) >= 3 else "some"),
            "research": research, "exams": "due" if due else ("fresh" if any_suite else "none"),
            # low: a suite scored below low_score that is worth practising now (not resting after a try)
            "scores": "none" if not scores else ("low" if practice else "high"),
            "memory": memory, "last": last if last in MODES else "none",
            "streak": "one" if streak <= 1 else ("two" if streak == 2 else "many"),
            "budget": "low" if left <= max(1, pol.max_decisions_per_hour // 4) else "ok",
            "selftasks": "none" if not active else ("full" if len(active) >= pol.max_self_tasks else "some"),
            "mood": s.directors.stance().mood,  # the mood shapes what it chooses (directors.py)
        }
        allowed = set(pol.modes) | {"rest"}
        return {"flags": flags, "text": situation_text(flags), "weak": weak, "topics": topics, "due": due,
                "practice": practice, "feasible": feasible(flags) & allowed, "any_suite": any_suite}

    def topics(self) -> list[tuple[str, str]]:
        s = self.sys
        out, seen = [], set()
        taken = {t.target.lower() for t in s.tasks.tasks.values() if t.kind == "learn"}
        failed = {t.lower() for t in self.failed_topics}
        refresh = s.directives.research.refresh_days * 86400

        def add(topic: str, why: str) -> None:
            t = _clean_topic(topic)
            k = t.lower()
            if len(t) < 3 or k in seen or k in taken or k in failed:
                return
            doc = s.knowledge.get(t)
            if doc is not None and doc.age < refresh:
                return
            seen.add(k)
            out.append((t, why))
        if s.mind.enabled:
            for g in list(s.mind.goals)[-10:]:
                add(g.text, f"a goal of its own ({g.reason})")
            for u in list(s.mind.unmet.values())[-10:]:
                if u.route == DIRECT and is_question(u.text):
                    add(u.text, "a question it could not answer")
            for n in list(s.mind.notes.values()):
                if n.kind == "fact" and n.subject == "user" and n.text.startswith("interested in "):
                    add(n.text[len("interested in "):], "you said you are interested in it")
        for d in s.knowledge.documents(10):
            for link in d.links[:3]:
                add(link, f"linked from '{d.title}', which it read")
        return out[:10]

    async def decide(self, force: bool = False) -> dict | None:
        """Take initiative once: choose a mode (agency node, drive policy, or a person's pin) and act."""
        s, pol = self.sys, self.pol
        if force:  # a person asked: wait for a decision already under way, then take another
            for _ in range(600):
                if not self.busy:
                    break
                await asyncio.sleep(0.05)
        if self.busy:
            return None
        if self._budget_left() <= 0 and not force:
            self.next_at = time.time() + pol.decide_every_seconds
            return None
        self.busy = True
        pool = s.directors
        lead = await pool.acquire("initiative", "deciding", spawn=False)  # director-0 unless it is busy
        helper = None
        try:
            st = pool.stance()
            sit = self.situation()
            f = sit["flags"]
            if self.pinned != "auto":
                mode = self.pinned
                how = "pinned by a person"
                if mode not in sit["feasible"]:
                    how = f"pinned to {mode}, but there is nothing to {mode} right now"
                    mode = "rest"
                why = WHY[mode]
            elif AGENCY in s.capabilities:
                inf = await s.infer(AGENCY, sit["text"])  # the agency node, or the drive policy teaching it
                mode = inf.label if inf.label in MODES else policy(f)[0]
                champ = s.registry.champion(AGENCY)
                trusted = champ is not None and (champ.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy
                how = (f"agency node {inf.node_id} ({inf.confidence:.0%} sure)" if inf.kind == "node"
                       else "drive policy (the agency node was not sure here)" if trusted
                       else "drive policy (the agency node is still learning)")
                why = policy(f)[1] if inf.kind != "node" or mode == policy(f)[0] else f"it learned that {mode} pays off here"
                if mode not in sit["feasible"]:
                    wrong = mode
                    mode, why = policy(f, exclude=tuple(m for m in MODES if m not in sit["feasible"]))
                    s.learn(AGENCY, sit["text"], mode, AGENCY)
                    how += f"; {wrong} was not possible, so the drive policy chose"
                if self.rng.random() < pol.explore * st.explore:  # good spirits explore more
                    options = [m for m in sit["feasible"] if m not in (mode, "rest")]
                    if options:
                        mode = self.rng.choice(sorted(options))
                        why = f"exploring: what is {mode} worth here?"
                        how += "; exploring"
            else:
                mode, why = policy(f)
                how = "drive policy"
            record = {"ts": round(time.time(), 1), "mode": mode, "how": how, "why": why, "situation": sit["text"],
                      "target": "", "task": "", "summary": "", "value": None, "by": lead.id}
            lead.initiative = mode
            records = [record]
            second = self._second_mode(sit, mode, st) if self.pinned == "auto" and pool.pol.collaborate else None
            if second is not None:  # two things worth doing at once: a duplicate takes the second
                helper = await pool.acquire("initiative", second, exclude={lead.id}, wait=0.0)
            if helper is not None:
                records.append({"ts": round(time.time(), 1), "mode": second, "by": helper.id,
                                "how": f"a duplicate of {lead.id}, alongside its {mode}",
                                "why": WHY[second], "situation": sit["text"], "target": "", "task": "", "summary": "",
                                "value": None})
                await asyncio.gather(self.act(mode, sit, record), self.act(second, sit, records[1]))
            else:
                await self.act(mode, sit, record)
            for rec in records:
                self.history.append(rec)
                self.recent.append(time.time())
            # high arousal decides sooner, low arousal later
            self.next_at = time.time() + pol.decide_every_seconds * st.pace * (2 if record["mode"] == "rest" else 1)
            self.save()
            for rec in records:
                s.guardian.audit(rec["by"], "agency.decided", rec["mode"], how=rec["how"], why=rec["why"],
                                 target=rec["target"], task=rec["task"], situation=sit["text"])
                s.bus.publish("agency.decided", **{k: v for k, v in rec.items() if k != "value"})
            for rec in records:
                if rec["value"] is not None:
                    self.outcome(rec, rec["value"], rec["summary"], rec["target"])
            return record
        finally:
            pool.release(lead, "initiative")
            if helper is not None:
                pool.release(helper, "initiative")
            self.busy = False

    @staticmethod
    def _target(sit: dict, mode: str):
        pick = {"research": sit["topics"], "practise": sit["practice"], "improve": sit["weak"], "examine": sit["due"]}.get(mode)
        return pick[0][0] if pick else None

    def _second_mode(self, sit: dict, mode: str, st) -> str | None:
        """Another mode worth doing at the same time, on a different target, for a duplicate to take."""
        if mode == "rest" or self._budget_left() < 2:
            return None
        d = drives(sit["flags"])
        active = len([t for t in self.self_tasks if t in self.sys.tasks.tasks])
        options = []
        for m in sit["feasible"]:
            if m in (mode, "rest") or d[m] < st.second_mode:
                continue
            if m in TASK_MODES and active + (2 if mode in TASK_MODES else 1) > self.pol.max_self_tasks:
                continue
            if self._target(sit, m) is not None and self._target(sit, m) == self._target(sit, mode):
                continue
            options.append(m)
        return max(options, key=lambda m: (d[m], -MODES.index(m))) if options else None

    async def act(self, mode: str, sit: dict, record: dict) -> None:
        s, pol = self.sys, self.pol
        if mode == "research":
            topic, why = sit["topics"][0]
            record["why"] = f"{record['why']}: '{topic}' ({why})" if not record["why"].startswith("exploring") else record["why"]
            self._assign(f"learn {topic} max {pol.research_articles} depth 1", record, topic,
                         {"docs": s.knowledge.stats().documents, "facts": self._facts()})
        elif mode == "practise":
            suite, score = sit["practice"][0]
            self._assign(f"challenge {suite} practice {pol.practice_items}", record, suite,
                         {"exam": score if score is not None else self._last_exam(suite)})
        elif mode == "improve":
            cap, acc, target = sit["weak"][0]
            self._assign(f"improve {cap} to {target}", record, cap, {"acc": acc})
        elif mode == "examine":
            suite, why = sit["due"][0]
            prev = self._last_exam(suite)
            record["target"] = suite
            rep = await s.challenges.run(suite, "exam", by="director")
            change = "" if prev is None else f" (was {prev:.0%})"
            record["summary"] = f"{rep.line()}{change}"
            # what it learned about itself: a first measurement, or a change, is worth the most
            record["value"] = 0.8 if prev is None else round(min(1.0, 0.2 + abs(rep.score - prev) * 5), 3)
        elif mode == "consolidate":
            mem = s.memory
            mem.sync_files(force=True)
            failing = 0
            for node in list(mem.nodes.values()):
                r = await asyncio.to_thread(mem.run_tests, node)
                bad = r["total"] - r["passed"]
                if bad and node.params is not None:
                    failing += bad
                    node.checked_version = -1  # relearn until every test passes again
            self.verified_at = time.time()
            record["target"] = f"{len(mem.nodes)} memory nodes"
            record["summary"] = (f"{failing} failing tests found; relearning them" if failing
                                 else f"all {len(mem.nodes)} memory nodes pass their tests")
            record["value"] = 0.9 if failing else 0.25
        else:
            record["summary"] = "rests"

    def _facts(self) -> int:
        return sum(len(n.facts()) for n in self.sys.memory.nodes.values()) if self.sys.memory.enabled else 0

    def _last_exam(self, suite: str):
        exams = [r for r in self.sys.challenges.history if r["suite"] == suite and r["mode"] == "exam"]
        return exams[-1]["score"] if exams else None

    def _assign(self, spec: str, record: dict, target: str, baseline: dict) -> None:
        s = self.sys
        record["target"] = target
        try:
            task = s.tasks.add(spec, by="director")
        except ValueError as exc:
            record["summary"] = f"could not set itself the task: {exc}"
            record["mode"] = "rest"
            return
        record["task"] = task.id
        record["summary"] = f"assigned itself {task.id}: {task.describe()}"
        self.self_tasks[task.id] = {"mode": record["mode"], "situation": record["situation"], "started": time.time(),
                                    "baseline": baseline, "what": task.describe(), "target": target}
        s.guardian.audit("director", "agency.assigned", task.id, what=task.describe(), mode=record["mode"])

    def review(self) -> None:
        """Score the tasks it set itself once they have run their course, then retire them."""
        s, pol = self.sys, self.pol
        changed = False
        for tid, rec in list(self.self_tasks.items()):
            task = s.tasks.get(tid)
            if task is None:  # a person removed it
                del self.self_tasks[tid]
                s.bus.publish("agency.outcome", task=tid, mode=rec["mode"], what=rec["what"], value=None,
                              summary="removed by a person before it finished", lesson="")
                changed = True
                continue
            if s.tasks.running == tid:  # mid-run: score it once this run is over
                continue
            age = time.time() - rec["started"]
            done, value, summary = False, 0.0, ""
            if task.kind == "learn":
                read = len(task.progress.get("read", []))
                fails = len(task.progress.get("failed", []))
                facts = max(0, self._facts() - rec["baseline"].get("facts", 0))
                done = task.met or task.runs >= pol.max_task_runs or age > pol.task_max_age_seconds or (fails >= 2 and not read)
                value = min(1.0, 0.3 * read + 0.1 * facts)
                summary = f"read {read} article{'s' if read != 1 else ''}" + (f", {facts} facts compressed" if facts else "")
                if done and not read:
                    self.failed_topics.append(rec["target"])
                    summary = f"found nothing to read about '{rec['target']}' ({task.note})"
            elif task.kind == "challenge":
                done = task.runs >= 1 or age > pol.task_max_age_seconds
                exam, prev = task.progress.get("exam"), rec["baseline"].get("exam")
                learned = task.progress.get("learned", 0)
                if exam is None:
                    value, summary = 0.0, task.note or "did not run"
                else:  # practice is worth what it does to the exam score; lessons alone are not enough
                    value = 0.5 if prev is None else min(1.0, max(0.0, 0.2 + 4 * (exam - prev)))
                    summary = f"exam {exam:.0%}" + (f" (was {prev:.0%})" if prev is not None else "") + f", {learned} lessons"
            elif task.kind == "improve":
                champ = s.registry.champion(task.target)
                acc = champ.metrics.get("holdout_acc") if champ else None
                base = rec["baseline"].get("acc")
                done = task.met or age > pol.improve_window_seconds or age > pol.task_max_age_seconds
                value = 1.0 if task.met else (min(1.0, max(0.0, (acc - base) * 10)) if acc is not None and base is not None else 0.0)
                summary = (f"{task.target} reached {acc:.3f}" if task.met and acc is not None else
                           f"{task.target} {base:.3f} -> {acc:.3f}" if acc is not None and base is not None else task.note)
            if not done:
                continue
            s.tasks.set_status(tid, "removed", by="director")
            del self.self_tasks[tid]
            entry = next((h for h in reversed(self.history) if h.get("task") == tid), None) or \
                {"mode": rec["mode"], "situation": rec["situation"], "task": tid, "target": rec["target"]}
            self.outcome(entry, round(value, 3), f"{rec['what']} finished: {summary}", rec["target"])
            changed = True
        if changed:
            self.save()

    def outcome(self, entry: dict, value: float, summary: str, target: str = "") -> None:
        """What a choice achieved teaches the agency node: reinforce what paid off, relabel what did
        not (also a so-so result when it keeps choosing the same thing), and let a target that did
        not pay off rest for a while."""
        s, pol = self.sys, self.pol
        mode = entry["mode"]
        if target and mode in ("research", "practise", "improve", "examine") and value < pol.good_value:
            key = f"{mode}:{target.lower() if mode == 'research' else target}"
            self.cooldown[key] = time.time() + pol.retry_after_seconds
        old = self.values.get(mode)
        self.values[mode] = round(value if old is None else 0.7 * old + 0.3 * value, 3)
        self.counts[mode] = self.counts.get(mode, 0) + 1
        entry["value"], entry["summary"] = value, summary
        lesson = ""
        if mode != "rest" and AGENCY in s.capabilities and entry.get("situation"):
            if value >= pol.good_value:
                if s.learn(AGENCY, entry["situation"], mode, "outcome"):
                    lesson = f"the agency node learns that {mode} paid off here"
            elif value <= pol.poor_value or self._repeating(entry["situation"], mode):
                f = parse_situation(entry["situation"])
                alt, _ = policy(f, exclude=(mode,))
                if s.learn(AGENCY, entry["situation"], alt, "outcome"):
                    lesson = f"the agency node learns to {alt} here instead"
        self.save()
        s.guardian.audit("director", "agency.outcome", entry.get("task") or mode, mode=mode, value=value, summary=summary)
        s.bus.publish("agency.outcome", task=entry.get("task", ""), mode=mode, what=entry.get("target", ""), value=value,
                      summary=summary, lesson=lesson)

    @staticmethod
    def _repeating(situation: str, mode: str) -> bool:
        """It chose the same mode again after doing it twice or more in a row."""
        f = parse_situation(situation)
        return f.get("last") == mode and f.get("streak") in ("two", "many")

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        s = self.sys
        champ = s.registry.champion(AGENCY) if AGENCY in s.capabilities else None
        acc = champ.metrics.get("holdout_acc") if champ else None
        tasks = []
        for tid, rec in self.self_tasks.items():
            t = s.tasks.get(tid)
            if t is not None:
                tasks.append({"id": tid, "mode": rec["mode"], "what": rec["what"], "age": round(time.time() - rec["started"]),
                              "status": t.status, "note": t.note, "runs": t.runs})
        return {"enabled": self.enabled, "pinned": self.pinned, "modes": list(self.pol.modes),
                "next_in": max(0, round(self.next_at - time.time())) if self.enabled else None,
                "decisions_last_hour": self.pol.max_decisions_per_hour - self._budget_left(),
                "limit": self.pol.max_decisions_per_hour, "self_tasks": tasks, "values": dict(self.values),
                "counts": dict(self.counts), "history": list(self.history)[-15:],
                "node": ({"id": champ.id, "arch": champ.arch.describe(), "holdout": acc, "params": champ.param_count,
                          "serves": champ.metrics.get("serves", 0),
                          "trusted": acc is not None and acc >= s.directives.nodes.serve_min_accuracy} if champ else None),
                "examples": s.store.count(AGENCY) if AGENCY in s.capabilities else 0,
                "verified_at": self.verified_at}
