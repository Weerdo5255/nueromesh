"""Questions nobody answered in time, searched on the web through your own SearXNG (plugins/searxng).

    a question it asked     in the Mind's Questions, open for [websearch] answer_after_minutes: what an order meant or
                            what its answer is ("clarify"), or what an objective couldn't find out ("objective").
                            Never approvals, proposals or questions about you (P3, P23)
      --> what to search    the order itself, or the objective's question. Questions about you or your things ("my",
                            "I", "our"), or with anything like an address, a long number or a key in them, never
                            leave the house ([websearch] skip_personal)
      --> SearXNG           your own instance asks its engines (Google, DuckDuckGo, ... as you set it up) at once
      --> an answer         an LLM teacher (your Ollama) reads the results and answers from them only, citing them by
                            number, or says they don't answer it; with no teacher, SearXNG's own answers and infoboxes,
                            or a result whose snippet covers the question and gives what was asked (less sure; it
                            says so)
      --> answered          the question is answered by "the web": its answer, who read it and its sources. It is a
                            teacher's answer, not a person's (P1): kept only if a director's review lets it (P14) - for
                            an order, a memory node keeps it and the order is tried again; for an objective, it carries
                            on with it - and a person's word still overrides it. What it could not answer stays open.

Each question is searched once on its own (again when a person presses Search the web now); [websearch] max_per_hour
searches an hour at most. The Searcher agent looks for due questions every [websearch] check_every_seconds.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, deque

from .agents import Agent
from .knowledge import content_words

PERSONAL = re.compile(r"\b(i|i'm|i've|i'd|me|my|mine|myself|our|ours|we|we're|us)\b", re.I)
SECRET = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{7,}\b|\b(?:sk|pk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{8,}|\b[A-Fa-f0-9]{32,}\b"
                    r"|\b(?:password|passwd|token|api[_ ]?key)\b", re.I)
_LEAD = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:tell\s+me\s+)?", re.I)


def personal(text: str) -> str:
    """Why this question stays at home ('' if it may be searched)."""
    if SECRET.search(text):
        return "it has something like an address, a long number or a key in it"
    if PERSONAL.search(text):
        return "it is about you or your things"
    return ""


def covers(question: str, text: str) -> float:
    """How much of the question's content words a snippet has."""
    q = set(content_words(question))
    if not q:
        return 0.0
    words = set(content_words(text))
    return len(q & words) / len(q)


def fits(question: str, text: str) -> bool:
    """Does the snippet give what was asked (a number for how many, a time for when)?"""
    low = question.lower()
    if re.match(r"\s*(how many|how much|how old|how tall|how long|how far|what year)", low):
        return bool(re.search(r"\d", text))
    if re.match(r"\s*when\b", low):
        return bool(re.search(r"\b(1\d{3}|20\d{2}|january|february|march|april|may|june|july|august|september|"
                              r"october|november|december|century)\b", text.lower()))
    return True


class WebSearch:
    """See the module docstring."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.websearch
        raw = system.store.get_kv("websearch.state")
        st = json.loads(raw) if raw else {}
        self.history: deque = deque(st.get("history", []), maxlen=max(10, self.pol.keep))
        self.tried: dict[str, float] = st.get("tried", {})
        self.counts: Counter = Counter(st.get("counts", {}))
        self.times: deque = deque(st.get("times", []), maxlen=500)
        self._busy = False

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def save(self) -> None:
        tried = dict(sorted(self.tried.items(), key=lambda kv: kv[1])[-500:])
        self.sys.store.set_kv("websearch.state", json.dumps({"history": list(self.history), "tried": tried,
                                                             "counts": dict(self.counts), "times": list(self.times)}))

    def tool(self):
        from .tools import ACTIVE
        t = self.sys.tools.tools.get("searxng")
        if t is None or t.state != ACTIVE or self.sys.tool_switch.refuses(t):
            return None
        return t

    def budget(self) -> int:
        now = time.time()
        while self.times and now - self.times[0] > 3600:
            self.times.popleft()
        return max(0, self.pol.max_per_hour - len(self.times))

    # ------------------------------------------------------------------ which questions
    def question_of(self, q) -> str | None:
        """What to search for an open question: the order it couldn't answer, or the objective's question."""
        s = self.sys
        if q.kind == "clarify" and q.order_id:
            rec = s.store.get_order(q.order_id)
            return " ".join(str(rec["text"]).split()) if rec else None
        if q.kind == "objective" and q.about:
            oid, _, qid = q.about.partition(":")
            obj = s.objectives.items.get(oid)
            item = next((x for x in obj.questions if x["id"] == qid), None) if obj else None
            return " ".join(str(item["ask"]).split()) if item else None
        return None

    def eligible(self, q) -> tuple[bool, str]:
        if q.status != "open":
            return False, "it is not open"
        if q.kind not in self.pol.kinds or q.approval_id or q.proposal_id:
            return False, "only people answer that kind of question"
        text = self.question_of(q)
        if not text:
            return False, "it has nothing to search for"
        if self.pol.skip_personal and personal(text):
            return False, f"it stays at home: {personal(text)}"
        return True, text

    def due(self) -> list:
        if not self.enabled or not getattr(self.sys.mind, "enabled", False):
            return []
        wait = self.pol.answer_after_minutes * 60
        now = time.time()
        out = []
        for q in self.sys.mind.open_inquiries():
            if q.id in self.tried or now - q.created_at < wait:
                continue
            if self.eligible(q)[0]:
                out.append(q)
        return sorted(out, key=lambda q: q.created_at)

    # ------------------------------------------------------------------ searching and answering
    async def search(self, text: str) -> tuple[dict | None, str]:
        s, t = self.sys, self.tool()
        if t is None:
            return None, "the searxng tool isn't active (approve it, and switch it on)"
        if not s.guardian.authorize_tool_run(t).allowed:
            return None, "the guardian won't let it run the searxng tool now"
        payload = json.dumps({"op": "search", "q": text, "engines": self.pol.engines, "categories": self.pol.categories,
                              "language": self.pol.language, "results": self.pol.results})
        timeout = t.manifest.timeout_s or s.directives.tools.default_timeout_s
        self.times.append(time.time())
        run = await s.sandbox.run(t, payload, timeout, s.tool_config(t))
        s.record_tool_run(t, run)
        if not run.ok:
            return None, (run.error or "SearXNG found nothing")[:240]
        return dict(run.extra), ""

    async def compose(self, question: str, found: dict) -> tuple[dict | None, str]:
        """An answer from what SearXNG found: a teacher's reading of the results, or its own."""
        s = self.sys
        results = found.get("results") or []
        if self.pol.teacher and results and s.teachers.has_llm():
            made = await s.teachers.answer_from_web(question, results)
            if made is not None:
                answer, teacher = made
                used = [results[i - 1] for i in answer["sources"] if 0 < i <= len(results)]
                if len(used) >= max(1, self.pol.min_sources):
                    return {"answer": answer["answer"], "sources": [{"title": r["title"], "url": r["url"]} for r in used],
                            "read_by": teacher, "confidence": 0.7, "how": "teacher"}, ""
                return None, f"the {teacher} teacher's answer cited none of the results"
            if not self.pol.own_reading:
                return None, "the results don't answer it (its teacher read them)"
        if not self.pol.own_reading:
            return None, "no teacher to read the results ([websearch] own_reading is off)"
        for a in found.get("answers") or []:  # SearXNG's own answers (calculators, definitions, ...)
            if covers(question, a["answer"]) >= 0.3 or len(a["answer"]) < 200:
                return {"answer": a["answer"], "sources": [{"title": "SearXNG", "url": a.get("url", "")}] if a.get("url") else [],
                        "read_by": "its own reading (SearXNG's answer)", "confidence": 0.55, "how": "answer"}, ""
        for b in found.get("infoboxes") or []:
            if b["content"] and covers(question, f"{b['infobox']} {b['content']}") >= 0.5 and fits(question, b["content"]):
                first = re.split(r"(?<=[.!?])\s+", b["content"])[0]
                return {"answer": first[:600], "sources": [{"title": b.get("infobox", ""), "url": b.get("url") or ((b.get("urls") or [{}])[0] or {}).get("url", "")}],
                        "read_by": "its own reading (an infobox)", "confidence": 0.5, "how": "infobox"}, ""
        agree = [r for r in results if r["content"] and covers(question, f"{r['title']} {r['content']}") >= 0.6
                 and fits(question, r["content"])]
        if len(agree) >= max(1, self.pol.min_sources):
            best = agree[0]
            sentence = max(re.split(r"(?<=[.!?])\s+", best["content"]), key=lambda x: covers(question, x))
            return {"answer": sentence[:600], "sources": [{"title": r["title"], "url": r["url"]} for r in agree[:3]],
                    "read_by": "its own reading (a result's snippet)", "confidence": 0.4, "how": "snippet"}, ""
        return None, "nothing it found answers it"

    async def answer(self, q, by: str = "the searcher") -> dict:
        """Search the web for an open question and, if it finds an answer, answer the question with it."""
        s = self.sys
        ok, text = self.eligible(q)
        rec = {"ts": round(time.time(), 1), "inquiry": q.id, "kind": q.kind, "question": (text if ok else "")[:300],
               "by": by, "ok": False, "answer": "", "sources": [], "read_by": "", "why": "", "engines": [], "results": 0}
        if not ok:
            rec["why"] = text
            return rec
        if self.budget() <= 0:
            rec["why"] = f"its budget of {self.pol.max_per_hour} searches this hour is spent"
            return rec
        self.tried[q.id] = time.time()
        with s.activity.track("websearch", "searcher", f"searching the web: {text[:70]}"):
            found, why = await self.search(text)
            made = None
            if found is not None:
                rec["results"] = len(found.get("results") or [])
                rec["engines"] = sorted({e for r in found.get("results") or [] for e in r.get("engines", [])})[:8]
                made, why = await self.compose(text, found)
        self.counts["searched"] += 1
        if made is None:
            rec["why"] = why
            self.counts["not answered"] += 1
            self._record(rec)
            s.bus.publish("web.unanswered", inquiry=q.id, question=text[:200], why=why[:200], by=by)
            return rec
        rec.update(ok=True, answer=made["answer"], sources=made["sources"], read_by=made["read_by"], how=made["how"],
                   confidence=made["confidence"])
        self.apply(q, text, made)
        self.counts["answered"] += 1
        self._record(rec)
        s.guardian.audit("the web", "inquiry.answered_by_web", q.id, question=text[:200], answer=made["answer"][:200],
                         read_by=made["read_by"], sources=[x["url"] for x in made["sources"]][:5])
        s.bus.publish("web.answered", inquiry=q.id, question=text[:200], answer=made["answer"][:300],
                      read_by=made["read_by"], sources=len(made["sources"]), by=by)
        return rec

    def apply(self, q, text: str, made: dict) -> None:
        """The web's answer, as a teacher's (P1): kept only if a director lets it (P14); a person can still override it."""
        s = self.sys
        where = "; ".join(x["url"] or x["title"] for x in made["sources"][:3]) or "SearXNG"
        source = f"the web (SearXNG, read by {made['read_by']}): {where}"
        kind = "teachers" if made["how"] == "teacher" else "knowledge"
        by = f"the web ({made['read_by']})"
        if q.kind == "clarify":
            if s.memory.allows(kind):
                s.directors.remember("", text, made["answer"], source, by=by, kind=kind)
            unmet = s.mind.unmet.get(q.order_id)
            if unmet is not None:
                unmet.next_try = time.time()  # try the order again: its memory may answer it now
        elif q.kind == "objective":
            oid, _, qid = q.about.partition(":")
            obj = s.objectives.items.get(oid)
            item = next((x for x in obj.questions if x["id"] == qid), None) if obj else None
            if obj is not None and item is not None:
                s.objectives._answer(obj, item, made["answer"], "web", made["confidence"],
                                     [x["url"] or x["title"] for x in made["sources"]])
                obj.next_at = time.time()
                s.objectives.save(obj)
            if s.memory.allows(kind):
                s.directors.remember("", text, made["answer"], source, by=by, kind=kind)
        q.by, q.sources = by, made["sources"][:6]
        s.mind._close(q, "answered", made["answer"], by)

    def _record(self, rec: dict) -> None:
        self.history.append(rec)
        self.save()

    async def tick(self) -> dict | None:
        if self._busy or not self.enabled or self.tool() is None or self.budget() <= 0:
            return None
        due = self.due()
        if not due:
            return None
        self._busy = True
        try:
            return await self.answer(due[0])
        finally:
            self._busy = False

    async def now(self, inquiry_id: str, by: str) -> dict:
        """A person asked it to search the web for a question now (before its time is up, or again)."""
        q = self.sys.mind.inquiries.get(inquiry_id)
        if q is None:
            return {"ok": False, "why": f"no question {inquiry_id}"}
        self.tried.pop(q.id, None)
        return await self.answer(q, by=by)

    def status(self) -> dict:
        t = self.sys.tools.tools.get("searxng")
        return {"enabled": self.enabled, "tool": t.state if t else "missing", "after_minutes": self.pol.answer_after_minutes,
                "kinds": list(self.pol.kinds), "budget": self.budget(), "max_per_hour": self.pol.max_per_hour,
                "counts": dict(self.counts), "history": list(self.history)[-30:][::-1],
                "teacher": self.pol.teacher and self.sys.teachers.has_llm(),
                "due": [q.id for q in self.due()],
                "waiting": [{"id": q.id, "minutes": round((time.time() - q.created_at) / 60, 1)}
                            for q in self.sys.mind.open_inquiries() if self.eligible(q)[0] and q.id not in self.tried]}

    def lines(self) -> list[str]:
        st = self.status()
        c = st["counts"]
        out = [f"Web search (SearXNG, tool {st['tool']}): questions open for {st['after_minutes']:g} minutes are searched "
               f"({', '.join(st['kinds'])}); {c.get('searched', 0)} searched, {c.get('answered', 0)} answered; "
               f"{st['budget']} searches left this hour. /web <inquiry id> searches one now."]
        for r in st["history"][:8]:
            out.append(f"  {r['inquiry']:<8} {'answered' if r['ok'] else 'no answer':<10} {r['question'][:60]}"
                       + (f" -> {r['answer'][:80]} ({r['read_by']})" if r["ok"] else f" ({r['why'][:80]})"))
        return out


class Searcher(Agent):
    """Now and then: questions nobody answered in time are searched on the web."""
    name = "searcher"

    @property
    def interval(self) -> float:
        return max(1.0, float(self.sys.directives.websearch.check_every_seconds))

    async def step(self) -> None:
        s = self.sys
        if not s.websearch.enabled or not s.guardian.background_allowed():
            return
        rec = await s.websearch.tick()
        if rec is not None:
            self.stats["answered" if rec.get("ok") else "not answered"] += 1
