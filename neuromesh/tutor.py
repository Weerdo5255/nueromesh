"""Lessons: when it is idle, its Ollama teachers set it orders, judge its answers, and teach it (P22).

    no orders from people for [tutor] idle_seconds, an Ollama teacher available, background capacity (P2)
      --> a teacher sets an order of one kind ([tutor] kinds, in turn): a factual question, a skill order
          ("sentiment: ..."), arithmetic, a request one of its tools meets, or code to write - and says what it
          expects. It is told what the mesh can do (its skills, tools, topics it knows, what it got wrong lately)
      --> the order goes through the mesh like any other (source "teacher:<name>"): directors, nodes, tools, ego
      --> it is marked: a skill label or a number against what the teacher expected; anything else by the teacher,
          who reads the answer (and the code, never running it) and says right, partly or wrong, and why
      --> it learns, as a teacher's lesson and never as a person's (P1): right - the route and the skill's label
          are reinforced, and an answer from what it read or a teacher is kept in memory after a director's review;
          wrong - the teacher's label teaches the skill node, and the teacher's answer goes to memory the same way
          (if a person taught it otherwise, the person's answer stands, and the dispute is kept with the lesson)

Teacher-set orders do not count as people's orders: it stays idle (it may still sleep, and set itself objectives).
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, deque

from .agents import Agent
from .teachers import LLMTeacher, _extract_json
from .tasks import DIRECT, DISPATCH

KIND_SAY = {"question": "a factual question", "skill": "a skill order", "maths": "arithmetic", "tool": "a tool request",
            "code": "code to write"}


def _norm(x) -> str:
    return " ".join(re.sub(r"[^\w\s.\-]", " ", str(x).lower()).split()).strip(" .")


def _number(x) -> float | None:
    m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(x))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


async def set_order(teacher: LLMTeacher, kind: str, context: dict) -> dict | None:
    """Have a teacher set one order of a kind, with what it expects."""
    how = {
        "question": "a short factual question with one well-established answer (geography, science, history, words)"
                    + (f"; prefer a topic it knows something about: {', '.join(context['topics'][:12])}" if context.get("topics") else "")
                    + ". expect: the correct answer in a few words.",
        "skill": "an order of the form '<skill>: <text>' for one of these skills, with the one label you expect:\n"
                 + "\n".join(f"- {k}: {', '.join(v)}" for k, v in context.get("skills", {}).items())
                 + "\nWrite a realistic text; now and then make it a hard case. expect: exactly one label from that skill's list.",
        "maths": "a small arithmetic question like 'what is 17 * 23' or 'what is 144 / 12' (whole numbers). expect: the number only.",
        "tool": "a request one of its tools can meet:\n" + "\n".join(f"- {n}: {d}" for n, d in context.get("tools", {}).items())
                + "\nexpect: the exact output the tool should give.",
        "code": "a request to write one small Python function, starting with 'write a python function that'. "
                "expect: one sentence on what correct code must do.",
    }[kind]
    wrong = context.get("wrong") or []
    schema = {"type": "object", "required": ["order", "expect"],
              "properties": {"order": {"type": "string"}, "expect": {"type": "string"}, "why": {"type": "string"}}}
    system = ("You are a teacher setting a short test for a small learning system (the mesh) while it is idle. Reply "
              "with a single JSON object and nothing else.")
    user = (f"Set ONE order: {how}\n"
            + (f"It recently got these wrong; you may test something close to them: {'; '.join(w[:80] for w in wrong[:4])}\n" if wrong else "")
            + (f"Do not repeat these recent orders: {'; '.join(o[:60] for o in context.get('recent', [])[:8])}\n" if context.get("recent") else "")
            + 'JSON: {"order": "...", "expect": "...", "why": "<what it tests, a few words>"}')
    data = _extract_json(await teacher._ask(system, user, schema) or "")
    if not data or not str(data.get("order", "")).strip() or not str(data.get("expect", "")).strip():
        return None
    return {"order": " ".join(str(data["order"]).split())[:400], "expect": str(data["expect"]).strip()[:400],
            "why": str(data.get("why", ""))[:160]}


async def judge(teacher: LLMTeacher, order: str, expect: str, answer: str, code: str = "") -> dict | None:
    """Have a teacher mark the mesh's answer: right, partly or wrong, why, and the correct answer."""
    schema = {"type": "object", "required": ["verdict", "why"],
              "properties": {"verdict": {"type": "string", "enum": ["right", "partly", "wrong"]},
                             "why": {"type": "string"}, "correct": {"type": "string"}}}
    system = ("You are a fair teacher marking a small learning system's answer to an order you set. Reply with a single "
              "JSON object and nothing else.")
    user = (f"The order: {order}\nWhat you expected: {expect}\nThe mesh answered: {answer[:1500] or '(nothing)'}\n"
            + (f"The code it wrote (do not run it; read it):\n{code[:4000]}\n" if code else "")
            + "Is the mesh's answer correct? Other wording and extra correct detail are fine; a wrong fact, number or "
              "label is wrong; an answer that says it does not know is wrong. "
              'JSON: {"verdict": "right" | "partly" | "wrong", "why": "<one sentence>", "correct": "<the correct answer, short>"}')
    data = _extract_json(await teacher._ask(system, user, schema) or "")
    if not data or str(data.get("verdict", "")).lower() not in ("right", "partly", "wrong"):
        return None
    return {"verdict": str(data["verdict"]).lower(), "why": str(data.get("why", ""))[:300],
            "correct": str(data.get("correct", "") or "")[:400]}


class Tutor:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.tutor
        self.lessons: deque = deque(maxlen=self.pol.keep)
        for rec in system.store.load_records("lessons"):
            self.lessons.append(rec)
        raw = system.store.get_kv("tutor.state")
        st = json.loads(raw) if raw else {}
        self.stats: Counter = Counter(st.get("stats", {}))
        self.turn = int(st.get("turn", 0))
        self.last = 0.0
        self.hour: deque = deque()
        self.busy = False
        self._n = max((int(r["id"].split("-")[1]) for r in self.lessons if re.fullmatch(r"lesson-\d+", r.get("id", ""))), default=0)
        self.rng = random.Random()

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def save(self) -> None:
        self.sys.store.set_kv("tutor.state", json.dumps({"stats": dict(self.stats), "turn": self.turn}))

    def teachers(self) -> list[LLMTeacher]:
        s = self.sys
        out = []
        for name in s.teachers.order:
            t = s.teachers.teachers.get(name)
            if isinstance(t, LLMTeacher) and t.provider in self.pol.providers and t.available():
                out.append(t)
        return out

    def due(self) -> str | None:
        """Why it should not set a lesson now (None: it may)."""
        s = self.sys
        now = time.time()
        if not self.enabled:
            return "lessons are off"
        if not self.teachers():
            return f"no {' or '.join(self.pol.providers)} teacher is available"
        if now - (s.last_order_at or s.started_at) < self.pol.idle_seconds:
            return "people gave it orders lately"
        if now - self.last < self.pol.every_seconds:
            return "a lesson was set lately"
        while self.hour and now - self.hour[0] > 3600:
            self.hour.popleft()
        if len(self.hour) >= self.pol.max_per_hour:
            return f"{self.pol.max_per_hour} lessons this hour already"
        if getattr(s.dreams, "asleep", False):
            return "it is asleep"
        return None

    def context(self, kind: str) -> dict:
        s = self.sys
        skills = {}
        for name, spec in s.capabilities.items():
            if not spec.internal and name != DISPATCH and spec.labels and len(spec.labels) <= 12:
                skills[name] = list(spec.active_labels())
        tools = {t.name: t.manifest.description[:120] for t in s.tools.active() if t.manifest.provides != "knowledge"}
        topics = [d.title for d in s.knowledge.documents(30) if d.source != "image"][:20]
        if not topics and s.curiosity.enabled:
            topics = [a.name for a in s.curiosity.ranked(12)]
        wrong = [s.corrections.marks.get(o, {}).get("text", "") for o in s.corrections.wrong[-6:]]
        wrong += [r["order"] for r in list(self.lessons)[-20:] if r.get("verdict") == "wrong"][-4:]
        return {"skills": skills, "tools": tools, "topics": topics, "wrong": [w for w in wrong if w],
                "recent": [r["order"] for r in list(self.lessons)[-10:]]}

    def kinds(self) -> list[str]:
        ctx_ok = {"skill": bool(self.context("skill")["skills"]), "tool": bool(self.context("tool")["tools"]),
                  "code": self.sys.scratch.enabled}
        return [k for k in self.pol.kinds if k in KIND_SAY and ctx_ok.get(k, True)]

    async def lesson(self, by: str = "idle", kind: str | None = None) -> dict | None:
        """One lesson: a teacher sets an order, the mesh answers it, the teacher marks it, and it learns."""
        s = self.sys
        teachers = self.teachers()
        if not teachers or self.busy:
            return None
        self.busy = True
        try:
            kinds = self.kinds()
            if not kinds:
                return None
            kind = kind if kind in kinds else kinds[self.turn % len(kinds)]
            self.turn += 1
            t = teachers[self.turn % len(teachers)]
            self.last = time.time()
            self.hour.append(self.last)
            with s.activity.track("lesson", t.name, f"setting {KIND_SAY[kind]}"):
                made = await set_order(t, kind, self.context(kind))
            if made is None:
                self.stats["not set"] += 1
                self.save()
                return None
            res = await s.submit(made["order"], source=f"teacher:{t.name}")
            answer = (res.output if res.ok else (res.note or "")) or ""
            answer = str(answer) if not isinstance(answer, str) else answer
            code = ""
            if res.route == "scratch":
                item = next((i for i in s.scratch.items.values() if i["id"] in answer), None)
                code = item["code"] if item else ""
            mark = self.mark(kind, made, res, answer)
            if mark is None:
                with s.activity.track("lesson", t.name, f"marking: {made['order'][:60]}"):
                    mark = await judge(t, made["order"], made["expect"], answer, code)
                if mark is not None:
                    mark["by"] = f"the {t.name} teacher"
            if mark is None:
                self.stats["not marked"] += 1
                self.save()
                return None
            learned = self.learn(kind, made, res, mark, t.name) if self.pol.learn else []
            self._n += 1
            rec = {"id": f"lesson-{self._n}", "ts": round(time.time(), 1), "teacher": t.name, "model": t.model, "kind": kind,
                   "order": made["order"], "expect": made["expect"], "why": made["why"], "order_id": res.order_id,
                   "answer": answer[:600], "route": res.route, "verdict": mark["verdict"], "mark_why": mark["why"],
                   "correct": mark.get("correct", ""), "marked_by": mark.get("by", ""), "learned": learned, "by": by}
            self.lessons.append(rec)
            s.store.save_record("lessons", rec["id"], rec["verdict"], rec)
            self.stats["lessons"] += 1
            self.stats[mark["verdict"]] += 1
            self.stats[f"{kind} {mark['verdict']}"] += 1
            self.save()
            s.bus.publish("tutor.lesson", id=rec["id"], teacher=t.name, kind=kind, order=made["order"][:160],
                          verdict=mark["verdict"], why=mark["why"][:200], learned="; ".join(learned)[:300])
            return rec
        finally:
            self.busy = False

    def mark(self, kind: str, made: dict, res, answer: str) -> dict | None:
        """Marks it can give without the teacher: a skill's label, or a number."""
        if kind == "skill" and res.route and res.route.startswith("skill:") and res.steps:
            got = str(res.output or "")
            want = _norm(made["expect"])
            right = _norm(got) == want
            return {"verdict": "right" if right else "wrong", "by": "the teacher's expected label",
                    "why": f"it said {got or 'nothing'}; the teacher expected {made['expect']}", "correct": made["expect"]}
        if kind == "maths":
            want, got = _number(made["expect"]), _number(answer)
            if want is not None and got is not None:
                right = abs(want - got) <= 1e-6 * max(1.0, abs(want))
                return {"verdict": "right" if right else "wrong", "by": "the teacher's expected number",
                        "why": f"it said {got:g}; the teacher expected {want:g}", "correct": f"{want:g}"}
        if kind == "tool" and res.ok and _norm(answer) and _norm(answer) == _norm(made["expect"]):
            return {"verdict": "right", "by": "the teacher's expected output", "why": "exactly what the teacher expected",
                    "correct": made["expect"]}
        return None

    def learn(self, kind: str, made: dict, res, mark: dict, teacher: str) -> list[str]:
        """What the mark teaches - as a teacher's lesson, never as a person's (P1)."""
        s = self.sys
        out: list[str] = []
        steps = res.steps or []
        dstep = next((x for x in steps if x.get("capability") == DISPATCH), None)
        sstep = next((x for x in steps if x.get("capability") not in (DISPATCH,)), None)
        if mark["verdict"] == "right":
            if dstep is not None and res.route and s.learn(DISPATCH, dstep.get("input") or made["order"], res.route, "tutor"):
                out.append(f"the router: orders like this go to {res.route}")
            if sstep is not None and sstep.get("label") and s.learn(sstep["capability"], sstep.get("input") or made["order"],
                                                                    sstep["label"], "tutor"):
                out.append(f"the {sstep['capability']} skill: '{sstep['label']}'")
            if kind == "question" and res.route == DIRECT and answer_worth_keeping(res):
                s.directors.remember(res.order_id, made["order"], res.answer or str(res.output),
                                     f"the {teacher} teacher marked it right (a lesson)", by=teacher, kind="teachers")
                out.append("memory keeps the answer, after a director's review")
            return out
        correct = (mark.get("correct") or made["expect"]).strip()
        if sstep is not None and kind == "skill":
            spec = s.capabilities.get(sstep["capability"])
            lab = _norm(correct).replace(" ", "_")
            if spec is not None and lab in spec.labels and s.learn(spec.name, sstep.get("input") or made["order"], lab, "tutor"):
                out.append(f"the {spec.name} skill learns '{lab}' for it")
        elif kind in ("question", "maths") and correct and s.memory.allows("teachers"):
            known = s.memory.known(made["order"]) if s.memory.enabled else None
            people = known is not None and (str(known.source).startswith(("you told me", "you confirmed"))
                                            or known.by in ("user", "web", "console", "api", "tester"))
            if people:
                out.append(f"a person taught it otherwise ('{known.answer[:60]}'): that stands (P1); the dispute is kept here")
            else:
                s.directors.remember(res.order_id, made["order"], correct, f"the {teacher} teacher's answer (a lesson)",
                                     by=teacher, kind="teachers")
                out.append(f"memory learns the teacher's answer ('{correct[:60]}'), after a director's review")
        return out

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        recs = list(self.lessons)[::-1]
        n = self.stats.get("lessons", 0)
        by_kind = {}
        for k in KIND_SAY:
            r, w, p = (self.stats.get(f"{k} {v}", 0) for v in ("right", "wrong", "partly"))
            if r + w + p:
                by_kind[k] = {"right": r, "wrong": w, "partly": p}
        return {"enabled": self.enabled, "teachers": [t.name for t in self.teachers()], "waiting": self.due(),
                "lessons": recs[:40], "count": n, "right": (self.stats.get("right", 0) / n) if n else None,
                "by_kind": by_kind, "every": self.pol.every_seconds, "idle": self.pol.idle_seconds, "busy": self.busy}

    def lines(self) -> list[str]:
        st = self.status()
        out = [f"Lessons: {st['count']} set by its teachers when it was idle"
               + (f"; it got {st['right']:.0%} right" if st["right"] is not None else "")
               + (f". Next: {st['waiting']}" if st["waiting"] else ". It may set one now") + "."]
        for k, v in st["by_kind"].items():
            out.append(f"  {KIND_SAY[k]}: {v['right']} right, {v['partly']} partly, {v['wrong']} wrong")
        for r in st["lessons"][:8]:
            out.append(f"  {r['id']:<10} {r['verdict']:<6} {r['teacher']}: \"{r['order'][:60]}\" -> \"{r['answer'][:50]}\" ({r['mark_why'][:60]})")
        return out


def answer_worth_keeping(res) -> bool:
    """An answer from what it read or a teacher (not one its memory already gave) is worth keeping."""
    return any("teacher" in str(x) or "read" in str(x) or "Wikipedia" in str(x) or "Kiwix" in str(x) for x in res.sources) \
        and not any("memory node" in t for t in res.trace)


class Schoolroom(Agent):
    """When it is idle, a teacher sets it a lesson ([tutor] every_seconds)."""
    name = "schoolroom"

    @property
    def interval(self) -> float:
        return max(1.0, min(10.0, self.sys.directives.tutor.every_seconds / 6))

    async def step(self) -> None:
        s = self.sys
        if s.tutor.due() is not None or s.inflight or not s.guardian.background_allowed():
            return
        rec = await s.tutor.lesson()
        self.stats["lessons" if rec else "no lesson"] += 1
