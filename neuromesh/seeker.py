"""The Kiwix search: to answer a question, the mesh searches the Kiwix sources it can reach (every book
on every kiwix-serve configured for the kiwix tool: Wikipedia, Wiktionary, Wikivoyage, Stack
Exchange, Project Gutenberg, ...) again and again - different books, different wordings, title and
full-text search - and reads what it finds, until an article actually answers the question, or
[research] kiwix_attempts searches have failed. Then it gives up, and the question goes to you.

    plan        the wordings worth searching for (the question's subject, the names in it, what it asks
                about, its key words) x the books most likely to hold the answer (Wiktionary first for
                "what does X mean", Wikivoyage for travel, Wikiquote for "who said", Stack Exchange for
                code, Gutenberg for books' text; Wikipedia first otherwise), title search before full text
    an attempt  one search in one book; the article it finds is retained like any research (it is read,
                after all) and must *answer*: it covers the question's names and most of its words, and
                the answer it gives fits what was asked (a time for "when", a number for "how many", a
                name for "who"). An article already read is never read twice for the same question
    giving up   after kiwix_attempts searches; the order's result says what was searched (Result.searched),
                and when nothing else answers it either, the mind asks you: what it searched, and whether
                you know (reply "answer: ..." and a memory node keeps it)

Fast: [research] kiwix_parallel searches run at once (the first in the plan that answers wins), and the
kiwix tool stays running between them (tools.py Worker: kept-alive connections, a cache, every book of a
search read at the same time).

Browsing: now and then ([research] browse_every_seconds) the Reader picks a Kiwix book it has never
opened, or not for [research] browse_stale_days, reads a few random articles from it and learns something
new (retained, and kept by memory nodes).

Research budgets apply to every attempt (your own Kiwix server is exempt: [research] unlimited).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from .knowledge import Hit, content_words, tokens

# which books suit which questions: (cue in the question, words in a book's name or category), best first
KINDS = [
    (re.compile(r"\b(?:mean(?:s|ing)?|definition|define|synonyms?|antonyms?|etymology|pronounce|spell(?:ed|ing)?|"
                r"plural of|what is the word)\b", re.I), ("wiktionary",)),
    (re.compile(r"\b(?:travel|visit|visiting|trip|hotel|hostel|stay in|tourist|itinerary|get to|sights?|"
                r"things to do)\b", re.I), ("wikivoyage",)),
    (re.compile(r"\b(?:quote|quotes|quotation|who said|saying)\b", re.I), ("wikiquote",)),
    (re.compile(r"\b(?:code|coding|program(?:ming)?|python|javascript|java|rust|sql|linux|bash|compile|error|"
                r"exception|function|api|library|regex)\b", re.I), ("stackoverflow", "stackexchange", "devdocs", "mdn")),
    (re.compile(r"\b(?:how (?:do|can) i|how to|fix|repair|replace|install|recipe|cook)\b", re.I),
     ("wikihow", "ifixit", "stackexchange")),
    (re.compile(r"\b(?:novel|poem|chapter|full text|book by|wrote the (?:book|novel|poem))\b", re.I),
     ("gutenberg", "wikisource")),
]
_ASK = re.compile(r"^\s*(?:(?:please|so|hey)\s+)?(?:(?:can|could)\s+you\s+(?:tell\s+me\s+)?)?(?:who|what|when|where|why|how|"
                  r"which|whose|whom)(?:'s|\s+(?:is|was|are|were|did|does|do|many|much|old|tall|long|far))?\s+", re.I)


_MEANS = re.compile(r"\bwhat (?:does|do|did) (?:the (?:word|term) )?(.+?) mean\b|\b(?:meaning|definition) of (?:the (?:word|term) )?(.+?)[?.!]*$", re.I)


@dataclass
class Attempt:
    n: int
    query: str
    book: str
    server: str
    mode: str                  # title | fulltext
    title: str = ""            # the article it read ("" for none)
    outcome: str = ""          # answered | not it | nothing found | unreachable | refused
    why: str = ""
    ms: float = 0.0


@dataclass
class Search:
    question: str
    order: str
    started: float = field(default_factory=time.time)
    attempts: list = field(default_factory=list)
    ok: bool = False
    answer: str = ""
    source: str = ""
    how: str = ""
    gave_up: str = ""
    books: int = 0

    def summary(self) -> dict:
        tried = [a for a in self.attempts]
        books = list(dict.fromkeys(a.book for a in tried if a.book))
        read = list(dict.fromkeys(a.title for a in tried if a.title))
        return {"question": self.question, "ok": self.ok, "attempts": len(tried), "books": books, "read": read,
                "queries": list(dict.fromkeys(a.query for a in tried)), "gave_up": self.gave_up}

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class Seeker:
    """The Kiwix search (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.research
        self._catalog: tuple[float, list[dict], list[str]] | None = None
        raw = system.store.get_kv("seeker.state")
        st = json.loads(raw) if raw else {}
        self.history: deque = deque(st.get("history", []), maxlen=30)
        self.counts = {"searches": 0, "answered": 0, "gave_up": 0, "attempts": 0, "browsed": 0, **st.get("counts", {})}
        self.books_seen: dict[str, float] = st.get("books_seen", {})   # book -> when it last read an article from it
        self.browsed: deque = deque(st.get("browsed", []), maxlen=30)
        self.rng = random.Random(system.rng.randrange(1 << 30))

    def save(self) -> None:
        self.sys.store.set_kv("seeker.state", json.dumps({"history": list(self.history), "counts": self.counts,
                                                          "books_seen": self.books_seen, "browsed": list(self.browsed)}))

    def note_book(self, name: str) -> None:
        """It read an article from this book (a search, a look-up, browsing)."""
        if name:
            self.books_seen[str(name)] = round(time.time(), 1)

    def tool(self):
        t = self.sys.tools.tools.get("kiwix")
        return t if t is not None and t in self.sys.knowledge_tools() else None

    def wanted(self, question: str) -> bool:
        """Is this a question for the Kiwix search (rather than a single look-up)?"""
        from .agents import is_question
        return int(self.pol.kiwix_attempts) > 1 and self.tool() is not None and is_question(question)

    # ---------------------------------------------------------------- the books
    async def catalog(self, refresh: bool = False) -> tuple[list[dict], list[str]]:
        """Every book on every Kiwix server the kiwix tool knows (cached for [research] catalog_seconds)."""
        now = time.time()
        if self._catalog and not refresh and now - self._catalog[0] < self.pol.catalog_seconds:
            return self._catalog[1], self._catalog[2]
        t = self.tool()
        if t is None:
            return [], ["the kiwix tool is not active"]
        s = self.sys
        run = await s.sandbox.run(t, "__catalog__", t.manifest.timeout_s or s.directives.tools.default_timeout_s, s.tool_config(t))
        if not run.ok:
            self._catalog = (now, [], [run.error or "the catalog could not be read"])
            return [], [run.error or "the catalog could not be read"]
        books = [b for b in run.extra.get("books", []) if isinstance(b, dict) and b.get("name")]
        down = [str(x) for x in run.extra.get("down", [])]
        self._catalog = (now, books, down)
        return books, down

    @staticmethod
    def rank(question: str, books: list[dict]) -> list[dict]:
        """The books most likely to hold the answer first."""
        wanted: list[str] = []
        for cue, kinds in KINDS:
            if cue.search(question):
                wanted += [k for k in kinds if k not in wanted]

        def score(b: dict) -> tuple:
            blob = f"{b.get('name', '')} {b.get('category', '')} {b.get('title', '')}".lower()
            fit = next((i for i, k in enumerate(wanted) if k in blob), None)
            wiki = "wikipedia" in blob
            return (fit is None, fit or 0, not wiki, 0)
        return sorted(books, key=score)

    @staticmethod
    def wordings(question: str) -> list[str]:
        """What to search for, likeliest first: the word a "what does X mean" asks about, the names in the
        question, what it asks about, the question as it is (the tool finds its subject), and its key words."""
        from .curiosity import topics_in
        from .objectives import _object_of
        q = " ".join(question.split()).strip()
        out = []
        m = _MEANS.search(q)
        if m:
            out.append((m.group(1) or m.group(2) or "").strip(" '\""))
        out += topics_in(q)
        obj = _object_of(q)
        if obj:
            out.append(obj)
        out.append(q)
        rest = _ASK.sub("", q).rstrip("?!. ")
        if rest and rest != q:
            out.append(rest)
        words = content_words(q)
        if len(words) >= 2:
            out.append(" ".join(words))
        seen, uniq = set(), []
        for w in out:
            k = " ".join(content_words(w))
            if k and k not in seen:
                seen.add(k)
                uniq.append(w)
        return uniq[:5]

    def plan(self, question: str, books: list[dict]) -> list[tuple[str, dict, str]]:
        """(wording, book, title | fulltext), best first: the likeliest book with each wording, title search
        before full text, then the next books."""
        steps = []
        for bi, book in enumerate(self.rank(question, books)):
            for wi, w in enumerate(self.wordings(question)):
                steps.append((wi + 1.5 * bi, w, book, "title"))
                steps.append((wi + 1.5 * bi + 0.6, w, book, "fulltext"))
        steps.sort(key=lambda x: x[0])
        return [(w, b, m) for _, w, b, m in steps]

    # ------------------------------------------------------------ does it answer
    @staticmethod
    def covers(question: str, hits: list[Hit]) -> bool:
        """Do these passages mention every name in the question? (An article about murals does not answer
        who painted the Zorblax mural.)"""
        from .curiosity import topics_in
        if not hits:
            return False
        words = set(tokens(" ".join([hits[0].doc.title] + [h.text for h in hits[:6]])))
        return all(all(t in words for t in tokens(n)) for n in topics_in(question))

    def answers(self, question: str, hits: list[Hit]) -> tuple[bool, str]:
        """Does what it read answer the question? The question's names are in it, most of its words are, and
        (checked later) the answer fits what was asked."""
        from .curiosity import topics_in
        if not hits:
            return False, "nothing in it matches the question"
        best = hits[0]
        text = " ".join(tokens(" ".join([best.doc.title] + [h.text for h in hits[:4]])))
        words = set(text.split())
        missing = [n for n in topics_in(question) if not all(t in words for t in tokens(n))]
        if missing:
            return False, f"it does not mention {', '.join(missing[:3])}"
        if best.passage_coverage < self.pol.kiwix_coverage:
            return False, f"it covers only {best.passage_coverage:.0%} of what was asked"
        return True, ""

    # --------------------------------------------------------------- searching
    async def seek(self, question: str, order_id: str = "", background: bool = False, learn: bool = True) -> Search:
        """Search Kiwix until an article answers the question, up to [research] kiwix_attempts times."""
        from .directors import REFUSAL
        from .objectives import fits
        s = self.sys
        search = Search(question, order_id)
        t = self.tool()
        if t is None:
            search.gave_up = "the kiwix tool is not active"
            return search
        books, down = await self.catalog()
        search.books = len(books)
        if not books:
            search.gave_up = "no Kiwix books could be listed" + (f" ({'; '.join(down)})" if down else "")
            return search
        limit = max(1, int(self.pol.kiwix_attempts))
        read: list[str] = []
        unlimited = s.unlimited(t)
        actor = "researcher" if background else "orchestrator"
        self.counts["searches"] += 1
        plan = self.plan(question, books)
        par = max(1, int(self.pol.kiwix_parallel))
        timeout = t.manifest.timeout_s or s.directives.tools.default_timeout_s
        pi, refused = 0, None
        while pi < len(plan) and len(search.attempts) < limit and not search.ok and refused is None:
            wave = []  # several searches at once; then the first in the plan that answers wins
            while pi < len(plan) and len(wave) < par and len(search.attempts) + len(wave) < limit:
                wording, book, mode = plan[pi]
                pi += 1
                a = Attempt(len(search.attempts) + len(wave) + 1, wording, book["name"], book.get("server", ""), mode)
                decision = s.guardian.authorize_research(background, unlimited=unlimited)
                if not decision.allowed:
                    a.outcome, a.why = "refused", decision.reason
                    refused = a
                    break
                s.guardian.record_research(background, unlimited=unlimited)
                wave.append(a)
            if wave:
                cfgs = [{**s.tool_config(t), "only": [{"server": a.server, "book": a.book}], "exclude": list(read),
                         "search": a.mode} for a in wave]
                started = time.time()
                label = (f"kiwix search {wave[0].n}/{limit}: '{wave[0].query[:60]}' in {wave[0].book}" if len(wave) == 1 else
                         f"kiwix searches {wave[0].n}-{wave[-1].n}/{limit} at once for '{question[:50]}'")
                with s.activity.track("research", actor, label):
                    runs = await asyncio.gather(*(s.sandbox.run(t, a.query, timeout, c) for a, c in zip(wave, cfgs)))
                took = round((time.time() - started) * 1000, 1)
                for a, run in zip(wave, runs):  # in the plan's order: the first that answers wins
                    a.ms = took
                    search.attempts.append(a)
                    self.counts["attempts"] += 1
                    s.record_tool_run(t, run)
                    got = self._attempt(a, run, question, read, limit, order_id, background, actor)
                    if got is None:
                        continue
                    hits, k, doc = got
                    answer, source, how = await s.answer_from_knowledge(question, hits)
                    if REFUSAL.search(answer) or not fits(question, answer):
                        a.outcome, a.why = "not it", "what it says does not give what was asked"
                        s.knowledge.touch(doc)
                        continue
                    a.outcome = "answered"
                    search.ok, search.answer, search.source = True, answer, source
                    search.how = (f"{how}; found by Kiwix search {a.n} of {limit} ({a.mode} search for '{a.query}' "
                                  f"in {k.get('book') or a.book})")
                    break
            if refused is not None:
                search.attempts.append(refused)
                search.gave_up = f"the research budget ran out: {refused.why}"
        if not search.ok and not search.gave_up:
            search.gave_up = (f"searched {len(search.attempts)} time{'s' if len(search.attempts) != 1 else ''} "
                              f"in {len({x.book for x in search.attempts})} of {len(books)} Kiwix book"
                              f"{'s' if len(books) != 1 else ''} without finding it")
        self.counts["answered" if search.ok else "gave_up"] += 1
        self.history.append(search.to_dict())
        self.save()
        s.bus.publish("kiwix.searched", order=order_id, question=question[:160], ok=search.ok,
                      attempts=len(search.attempts), limit=limit, title=next((x.title for x in search.attempts
                                                                              if x.outcome == "answered"), ""),
                      why=search.gave_up[:160])
        return search

    def _attempt(self, a: Attempt, run, question: str, read: list, limit: int, order_id: str,
                 background: bool, actor: str):
        """What one search found: retained, and checked against the question. (hits, knowledge, document) when it
        may answer it (the caller composes the answer and checks it gives what was asked), else None."""
        s = self.sys
        k = run.extra.get("knowledge") if run.ok else None
        if not isinstance(k, dict) or not k.get("title"):
            a.outcome = "unreachable" if run.transient else "nothing found"
            a.why = (run.error or "no article")[:200]
            return None
        if str(k.get("url", "")) in read:  # another search of the same wave read it already
            a.outcome, a.why = "not it", "the same article another search found"
            return None
        read.append(str(k.get("url", "")))
        a.title = k["title"]
        self.note_book(str(k.get("revision") or a.book))
        doc = s.knowledge.add(k, f"kiwix search for order {order_id}" if order_id else "kiwix search")
        s.guardian.audit(actor, "knowledge.learned", doc.title, source=doc.source, url=doc.url, passages=doc.passages,
                         reason=f"kiwix search, attempt {a.n}")
        s.memory.learn_article(doc)
        s.bus.publish("knowledge.learned", title=doc.title, url=doc.url, source=doc.source, passages=doc.passages,
                      reason=f"kiwix search {a.n}/{limit}", background=background, topic=a.query[:200])
        hits = s.knowledge.search(question, doc_id=doc.id) or [Hit(doc, "Introduction", doc.summary, 0.0, 0.0)]
        ok, why = self.answers(question, hits)
        if ok:
            return hits, k, doc
        a.outcome, a.why = "not it", why
        s.knowledge.touch(doc)
        return None

    # ------------------------------------------------------------------ browsing
    async def browse(self, by: str = "reader") -> dict:
        """Pick a Kiwix book it has never opened, or not for a long time, and learn something new from it."""
        s = self.sys
        t = self.tool()
        if t is None:
            return {"ok": False, "why": "the kiwix tool is not active"}
        books, down = await self.catalog()
        if not books:
            return {"ok": False, "why": "no Kiwix books could be listed" + (f" ({'; '.join(down)})" if down else "")}
        now = time.time()
        stale = float(self.pol.browse_stale_days) * 86400
        never = [b for b in books if b["name"] not in self.books_seen]
        old = sorted((b for b in books if now - self.books_seen.get(b["name"], 0) >= stale),
                     key=lambda b: self.books_seen.get(b["name"], 0))
        if never:
            book, why = self.rng.choice(never), "a book it had never opened"
        elif old:
            book = old[0]
            days = (now - self.books_seen[book["name"]]) / 86400
            why = f"a book it had not opened for {days:.0f} day{'s' if round(days) != 1 else ''}"
        else:
            return {"ok": False, "why": f"it has opened every book in the last {self.pol.browse_stale_days:g} days"}
        unlimited = s.unlimited(t)
        timeout = t.manifest.timeout_s or s.directives.tools.default_timeout_s
        learned, read, note = [], [], ""
        for _ in range(max(1, int(self.pol.browse_articles))):
            decision = s.guardian.authorize_research(True, unlimited=unlimited)
            if not decision.allowed:
                note = decision.reason
                break
            s.guardian.record_research(True, unlimited=unlimited)
            cfg = {**s.tool_config(t), "only": [{"server": book.get("server", ""), "book": book["name"]}], "exclude": read}
            with s.activity.track("research", by, f"browsing {book['name']}: {why}"):
                run = await s.sandbox.run(t, "__random__", timeout, cfg)
            s.record_tool_run(t, run)
            k = run.extra.get("knowledge") if run.ok else None
            if not isinstance(k, dict) or not k.get("title"):
                note = run.error or "no article"
                continue
            read.append(str(k.get("url", "")))
            doc = s.knowledge.add(k, f"browsing {book['name']}")
            s.guardian.audit(by, "knowledge.learned", doc.title, source=doc.source, url=doc.url, passages=doc.passages,
                             reason=f"browsing {book['name']} ({why})")
            s.memory.learn_article(doc)
            s.bus.publish("knowledge.learned", title=doc.title, url=doc.url, source=doc.source, passages=doc.passages,
                          reason=f"browsing {book['name']}", background=True, topic=doc.title[:200])
            learned.append(doc.title)
        self.note_book(book["name"])
        rec = {"ts": round(now, 1), "book": book["name"], "why": why, "learned": learned, "note": note[:200]}
        self.browsed.append(rec)
        self.counts["browsed"] += 1
        self.save()
        s.bus.publish("kiwix.browsed", book=book["name"], why=why, learned=learned[:5], note=note[:160])
        return {"ok": bool(learned), **rec}

    # ------------------------------------------------------------------ views
    def status(self) -> dict:
        books = self._catalog[1] if self._catalog else []
        now = time.time()
        return {"enabled": self.tool() is not None, "attempts": int(self.pol.kiwix_attempts), "counts": dict(self.counts),
                "parallel": int(self.pol.kiwix_parallel),
                "books": [{**b, "seen": self.books_seen.get(b["name"]),
                           "stale": now - self.books_seen.get(b["name"], 0) >= self.pol.browse_stale_days * 86400}
                          for b in books],
                "down": self._catalog[2] if self._catalog else [],
                "browse": {"on": bool(self.pol.browse), "every": self.pol.browse_every_seconds,
                           "stale_days": self.pol.browse_stale_days, "recent": list(self.browsed)[-8:][::-1]},
                "worker": self.sys.sandbox.worker_status().get("kiwix"),
                "history": list(self.history)[-12:][::-1]}

    async def lines(self) -> list[str]:
        if self.tool() is None:
            return ["the Kiwix search needs an active kiwix tool (start with --kiwix URL and approve it: /approvals)"]
        books, down = await self.catalog(refresh=True)
        c = self.counts
        out = [f"Kiwix search: up to {int(self.pol.kiwix_attempts)} searches a question before it asks you "
               f"([research] kiwix_attempts), {int(self.pol.kiwix_parallel)} at once; {c['searches']} searches, "
               f"{c['answered']} answered, {c['gave_up']} given up"]
        w = self.sys.sandbox.worker_status().get("kiwix")
        if w:
            out.append(f"  the kiwix tool stays running (pid {w['pid']}): {w['served']} requests served by this worker")
        servers = list(dict.fromkeys(b.get("server", "") for b in books))
        out.append(f"  {len(books)} book{'s' if len(books) != 1 else ''} on {len(servers)} server(s):")
        for b in books[:20]:
            seen = self.books_seen.get(b["name"])
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(seen)) if seen else "never opened"
            out.append(f"    {b['name']}  ({b.get('title', '')}; {b.get('server', '')}; last read: {when})")
        if self.pol.browse:
            out.append(f"  browsing: every {self.pol.browse_every_seconds:g}s it may open a book it has never read, or not "
                       f"for {self.pol.browse_stale_days:g} days ({c.get('browsed', 0)} so far)")
            for r in list(self.browsed)[-3:][::-1]:
                out.append(f"    {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts']))} {r['book']} ({r['why']}): "
                           + (", ".join(r["learned"]) or f"nothing ({r['note']})"))
        out += [f"  down: {d}" for d in down]
        for h in list(self.history)[-5:][::-1]:
            out.append(f"  {'answered' if h['ok'] else 'gave up'}: '{h['question'][:70]}' after {len(h['attempts'])} "
                       f"search{'es' if len(h['attempts']) != 1 else ''}")
            for a in h["attempts"][:8]:
                out.append(f"    {a['n']}. {a['mode']} '{a['query'][:40]}' in {a['book']}: {a['outcome']}"
                           + (f" ({a['title']})" if a["title"] else "") + (f" - {a['why'][:80]}" if a["why"] else ""))
        return out
