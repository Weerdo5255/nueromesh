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
searches an hour at most. The Searcher agent looks for due questions every [websearch] check_every_seconds - it is light
work, so it runs even while the mesh is busy training (not while it is paused, halted or asleep).

Patience with SearXNG: one search at a time; a timeout, a 429 or a 5xx rests it (plugins/searxng, the sandbox) and the
Searcher sends nothing while it rests. A question whose search failed that way is tried again later ([websearch]
retry_minutes, then double, up to [websearch] retries times), not given up on. A clarify question superseded by a
newer one (the mind keeps only the newest open, so it doesn't nag) is still searched when its time is up.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter, deque

from .agents import Agent
from .knowledge import content_words

PERSONAL = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|our|ours|ourselves)\b", re.I)
_GENERIC = re.compile(r"\b(?:tell|show|give|teach|remind|let)\s+me\b|\bhow\s+(?:do|can|should|would|could)\s+(?:i|we)\b|"
                      r"\b(?:can|should|could)\s+(?:i|we)\b|\bfor\s+me\b|\bwhat\s+(?:do|should|can)\s+i\b", re.I)
SECRET = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{7,}\b|\b(?:sk|pk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{8,}|\b[A-Fa-f0-9]{32,}\b"
                    r"|\b(?:password|passwd|token|api[_ ]?key)\b", re.I)
_LEAD = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:tell\s+me\s+)?", re.I)


def personal(text: str) -> str:
    """Why this question stays at home ('' if it may be searched). "tell me ...", "how do I ..." ask about the
    world, not about you; "my", "our", "I am" are about you."""
    if SECRET.search(text):
        return "it has something like an address, a long number or a key in it"
    rest = _GENERIC.sub(" ", _LEAD.sub("", text))
    if PERSONAL.search(rest):
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
        # question -> {"at", "attempts", "next_at", "done", "why"} (older states kept a time: searched, and done)
        self.tried: dict[str, dict] = {k: (v if isinstance(v, dict) else {"at": v, "attempts": 1, "next_at": 0.0,
                                                                           "done": True, "why": ""})
                                       for k, v in st.get("tried", {}).items()}
        self.counts: Counter = Counter(st.get("counts", {}))
        self.times: deque = deque(st.get("times", []), maxlen=500)
        self._busy = False
        self.settings: dict = st.get("settings", {})  # what a person set on the page: the window, on/off, SearXNG's address
        self._apply()

    def _apply(self) -> None:
        st = self.settings
        if st.get("after_minutes") is not None:
            self.pol.answer_after_minutes = float(st["after_minutes"])
        if st.get("enabled") is not None:
            self.pol.enabled = bool(st["enabled"])
        given = self.sys.directives.tools.config.get("searxng", {}).get("url")
        if st.get("url") and (not given or given == st.get("url") or st.get("url_over")):
            self.sys.directives.tools.config.setdefault("searxng", {})["url"] = st["url"]

    def url(self) -> tuple[str, str]:
        """Where its SearXNG is, and who said so."""
        t = self.sys.tools.tools.get("searxng")
        given = self.sys.directives.tools.config.get("searxng", {}).get("url")
        if self.settings.get("url") and given == self.settings["url"]:
            return given, "set on this page" + ("" if self.settings.get("url_by") in (None, "", "web") else f" by {self.settings['url_by']}")
        if given:
            return given, "--searxng or [tools.config.searxng] url"
        return (t.manifest.config.get("url", "") if t else "http://localhost:8888"), "the plugin's default"

    def configure(self, by: str, after_minutes=None, enabled=None, url=None) -> tuple[bool, str]:
        """A person sets how long a question waits before it is searched, web search on or off, or SearXNG's address."""
        said = []
        if after_minutes is not None:
            try:
                m = float(after_minutes)
            except (TypeError, ValueError):
                return False, "the wait is a number of minutes"
            if not 1 <= m <= 7 * 24 * 60:
                return False, "the wait is between 1 minute and a week"
            self.settings["after_minutes"] = self.pol.answer_after_minutes = m
            said.append(f"questions nobody answers within {m:g} minute{'s' if m != 1 else ''} are searched on the web")
        if enabled is not None:
            on = enabled if isinstance(enabled, bool) else str(enabled).lower() in ("1", "true", "on", "yes")
            self.settings["enabled"] = self.pol.enabled = on
            said.append("web search is " + ("on" if on else "off: questions wait for you"))
        if url is not None:
            u = str(url).strip().rstrip("/")
            if not re.match(r"^https?://[^\s/]+", u):
                return False, "SearXNG's address starts with http:// or https://"
            self.settings.update(url=u, url_by=by, url_over=True)
            self.sys.directives.tools.config.setdefault("searxng", {})["url"] = u
            said.append(f"SearXNG is at {u}")
            t = self.sys.tools.tools.get("searxng")
            if t is not None and t.state in ("quarantined", "active"):
                self.sys.retest_tool("searxng")
                said.append("its self-test runs again")
        self.save()
        self.sys.guardian.audit(by, "websearch.settings", "websearch", **{k: v for k, v in self.settings.items() if k != "url_over"})
        self.sys.bus.publish("web.settings", by=by, after_minutes=self.pol.answer_after_minutes, enabled=self.pol.enabled,
                             url=self.url()[0])
        return True, "; ".join(said) or "nothing changed"

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def save(self) -> None:
        tried = dict(sorted(self.tried.items(), key=lambda kv: kv[1].get("at", 0))[-500:])
        self.sys.store.set_kv("websearch.state", json.dumps({"history": list(self.history), "tried": tried,
                                                             "counts": dict(self.counts), "times": list(self.times),
                                                             "settings": self.settings}))

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
        superseded = (q.kind == "clarify" and q.status == "dropped" and q.answer.startswith("superseded")
                      and time.time() - q.created_at < 86400)  # the mind asks only the newest; the web still looks
        if q.status != "open" and not superseded:
            return False, "it is not open"
        if q.kind not in self.pol.kinds or q.approval_id or q.proposal_id:
            return False, "only people answer that kind of question"
        text = self.question_of(q)
        if not text:
            return False, "it has nothing to search for"
        if self.pol.skip_personal and personal(text):
            return False, f"it stays at home: {personal(text)}"
        return True, text

    def candidates(self) -> list:
        """Open questions, and clarify questions the mind superseded with a newer one."""
        now = time.time()
        return [q for q in self.sys.mind.inquiries.values()
                if q.status == "open" or (q.kind == "clarify" and q.status == "dropped" and now - q.created_at < 86400)]

    def finished(self, q) -> bool:
        t = self.tried.get(q.id)
        return bool(t) and (t.get("done") or time.time() < t.get("next_at", 0))

    def due(self) -> list:
        if not self.enabled or not getattr(self.sys.mind, "enabled", False):
            return []
        wait = self.pol.answer_after_minutes * 60
        now = time.time()
        out = []
        for q in self.candidates():
            if self.finished(q) or now - q.created_at < wait:
                continue
            if self.eligible(q)[0]:
                out.append(q)
        return sorted(out, key=lambda q: q.created_at)

    def waiting_why(self) -> str:
        """Why the Searcher isn't searching right now ('' when it may)."""
        g = self.sys.guardian
        if not self.enabled:
            return "web search is off"
        if self.tool() is None:
            return "the searxng tool isn't active"
        if g.halted or g.paused:
            return "the mesh is paused"
        if g.sleeping:
            return "the mesh is asleep (it searches when it wakes)"
        if self.budget() <= 0:
            return f"its {self.pol.max_per_hour} searches this hour are spent"
        left = self.sys.sandbox.rest_left("searxng")
        if left > 0:
            return f"SearXNG is resting after '{self.sys.sandbox.resting['searxng']['why'][:80]}' ({left:.0f}s left)"
        return ""

    # ------------------------------------------------------------------ searching and answering
    async def search(self, text: str) -> tuple[dict | None, str]:
        found, why, _ = await self._search(text)
        return found, why

    async def _search(self, text: str) -> tuple[dict | None, str, float]:
        """SearXNG's results, or why not and - when it may answer later (slow, busy, resting) - how long to wait."""
        s, t = self.sys, self.tool()
        if t is None:
            return None, "the searxng tool isn't active (approve it, and switch it on)", 0.0
        if not s.guardian.authorize_tool_run(t).allowed:
            return None, "the guardian won't let it run the searxng tool now", 60.0
        payload = json.dumps({"op": "search", "q": text, "engines": self.pol.engines, "categories": self.pol.categories,
                              "language": self.pol.language, "results": self.pol.results})
        timeout = t.manifest.timeout_s or s.directives.tools.default_timeout_s
        self.times.append(time.time())
        run = await s.sandbox.run(t, payload, timeout, s.tool_config(t))
        s.record_tool_run(t, run, text=text)
        if not run.ok:
            later = (run.retry_after or 60.0) if run.transient else 0.0
            return None, (run.error or "SearXNG found nothing")[:240], later
        return dict(run.extra), "", 0.0

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
        t = self.tried.setdefault(q.id, {"at": 0.0, "attempts": 0, "next_at": 0.0, "done": False, "why": ""})
        t["at"], t["attempts"] = time.time(), t["attempts"] + 1
        with s.activity.track("websearch", "searcher", f"searching the web: {text[:70]}"):
            found, why, later = await self._search(text)
            made = None
            if found is not None:
                rec["results"] = len(found.get("results") or [])
                rec["engines"] = sorted({e for r in found.get("results") or [] for e in r.get("engines", [])})[:8]
                made, why = await self.compose(text, found)
        self.counts["searched"] += 1
        if made is None:
            rec["why"] = why
            if later and t["attempts"] < max(1, self.pol.retries):  # slow or busy: tried again, not given up on
                wait = max(later, self.pol.retry_minutes * 60 * 2 ** (t["attempts"] - 1))
                t.update(next_at=time.time() + wait, done=False, why=why)
                rec["retry_at"] = round(t["next_at"], 1)
                rec["why"] = f"{why} - it tries again in {wait / 60:.0f} min (attempt {t['attempts']} of {self.pol.retries})"
                self.counts["will retry"] += 1
            else:
                t.update(done=True, why=why)
                self.counts["not answered"] += 1
            self._record(rec)
            s.bus.publish("web.unanswered", inquiry=q.id, question=text[:200], why=rec["why"][:200], by=by)
            return rec
        t.update(done=True, why="answered")
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
        if self.sys.sandbox.rest_left("searxng") > 0:  # SearXNG is resting after a timeout: nothing more until it has
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
        left = self.sys.sandbox.rest_left("searxng")
        if 0 < left <= 60:  # resting after a timeout: a person can wait a minute; longer, they are told
            await asyncio.sleep(left)
        return await self.answer(q, by=by)

    def status(self) -> dict:
        t = self.sys.tools.tools.get("searxng")
        url, url_from = self.url()
        approval = self.sys.guardian.approval_for(t.approval_id, "tool", "searxng") if t is not None else None
        return {"enabled": self.enabled, "tool": t.state if t else "missing", "after_minutes": self.pol.answer_after_minutes,
                "url": url, "url_from": url_from, "tool_note": t.note if t else "the searxng plugin isn't installed",
                "tool_error": t.last_error[:400] if t is not None and t.state == "quarantined" else "",
                "approval": approval.id if approval is not None and approval.status == "pending" else "",
                "settings": {k: v for k, v in self.settings.items() if k != "url_over"},
                "kinds": list(self.pol.kinds), "budget": self.budget(), "max_per_hour": self.pol.max_per_hour,
                "counts": dict(self.counts), "history": list(self.history)[-30:][::-1],
                "teacher": self.pol.teacher and self.sys.teachers.has_llm(),
                "due": [q.id for q in self.due()], "paused": self.waiting_why(),
                "resting": round(self.sys.sandbox.rest_left("searxng"), 1),
                "waiting": [{"id": q.id, "minutes": round((time.time() - q.created_at) / 60, 1),
                             "search_in": round(max(0.0, q.created_at + self.pol.answer_after_minutes * 60 - time.time(),
                                                    (self.tried.get(q.id) or {}).get("next_at", 0) - time.time()) / 60, 1),
                             "attempts": (self.tried.get(q.id) or {}).get("attempts", 0),
                             "superseded": q.status == "dropped"}
                            for q in self.candidates() if self.eligible(q)[0] and not (self.tried.get(q.id) or {}).get("done")]}

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
        # light work for a person's question: it doesn't wait for training to quieten down (the duty cycle), only
        # for the mesh to be awake and running (P2)
        if not s.websearch.enabled or not s.guardian.background_allowed(light=True):
            return
        rec = await s.websearch.tick()
        if rec is not None:
            self.stats["answered" if rec.get("ok") else "not answered"] += 1
