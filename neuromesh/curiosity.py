"""Curiosity: what the mesh finds interesting, and where what it knows falls short.

    areas    the things it hears and knows about: names and subjects in what people ask, in its
             memory nodes' facts, in what it has read and in its objectives
    interest  people ask about it (more often, more interest), a person said they are interested
              in it, it has read about it, its thinking nodes favour it, what it read links to it
    lack      questions about it that it could not answer, answers people marked wrong, objective
              questions left open or disputed (its sources disagree), names it knows only by name
              (they turn up in its facts, but it holds nothing about them), and areas people ask
              about more than it knows
    aims      what the director may set itself as an objective (agency mode pursue, P12, P16):
                find out <a question it could not answer>        where information is lacking
                understand <an area>                             an area of interest, or one it knows little of

Nothing here acts on its own: the director chooses (agency.py), within [agency] max_self_objectives
and its budgets, and every objective it sets itself says why (the area, its interest and what it lacks).
Thinking nodes hold the same picture as desires (thinking.py): "I want to find out ...".
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .agents import is_question
from .tasks import DIRECT

# capitalised words that are not names: question words, articles, pronouns, linking words
STOP_CAPS = {
    "what", "who", "whom", "whose", "when", "where", "why", "how", "which", "is", "are", "was", "were", "do", "does",
    "did", "the", "a", "an", "in", "on", "at", "tell", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "i", "it", "its", "this", "that", "these", "those", "there", "his", "her", "hers", "their",
    "our", "my", "your", "he", "she", "they", "we", "you", "and", "but", "or", "if", "as", "for", "from", "of", "to",
    "with", "by", "after", "before", "during", "about", "please", "yes", "no", "not", "so", "then", "also", "some",
    "any", "all", "each", "every", "many", "much", "more", "most", "one", "two", "three", "here", "now", "today",
    "explain", "describe", "define", "list", "name", "give", "show", "find", "compare", "hello", "hi", "hey", "thanks",
    "ok", "okay", "sentiment", "urgency", "count", "convert", "objective", "remind", "imagine", "keep", "learn",
    "understand", "study", "research", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "an", "since", "while", "although", "though", "because", "however", "yet", "until", "unless", "according",
    "around", "over", "under", "between", "into", "onto", "upon", "within", "without", "via", "per", "me", "mine",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "ad", "bc", "bce", "ce",
}
CONNECT = {"of", "the", "da", "de", "del", "della", "di", "du", "von", "van", "der", "den", "la", "le", "y", "bin", "al"}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*|\d[\d,.]*|[^\sA-Za-z\d]")
_SENT_END = {".", "!", "?", ";", ":", "\n", "(", ")", "[", "]", '"', "“", "”"}
# what an identity question asks about ("What is the speed of light?" -> "speed of light")
_IDENTITY = re.compile(r"^\s*(?:please\s+)?(?:tell me about|what(?:'s| is| are| was| were)|who(?:'s| is| are| was| were)|"
                       r"explain|describe|define|understand|learn about|find out about|read up on)\s+(?:the\s+|a\s+|an\s+)?"
                       r"(?P<s>[^?!.]{3,60}?)\s*[?!.]*$", re.I)


def norm(name: str) -> str:
    """The key of an area: lower case, without a leading article or a possessive."""
    s = " ".join(re.sub(r"['’]s\b", "", str(name)).split()).strip(" .,;:!?\"'()").lower()
    return re.sub(r"^(?:the|a|an)\s+", "", s)


@lru_cache(maxsize=20000)
def _topics(text: str) -> tuple:
    words = _WORD.findall(str(text))
    out: list[str] = []
    run: list[str] = []
    start = True  # at the start of a sentence

    def flush():
        while run and run[-1].lower() in CONNECT | {"and"}:  # "Bank of" -> "Bank"
            run.pop()
        if run:
            name = " ".join(run)
            if not (len(run) == 1 and (name.lower() in STOP_CAPS or len(name) < 2)):
                out.append(re.sub(r"['’]s$", "", name))
        run.clear()

    def cap(i: int) -> bool:
        return 0 <= i < len(words) and words[i][:1].isupper() and words[i].lower() not in STOP_CAPS
    for i, w in enumerate(words):
        if w in _SENT_END or (len(w) == 1 and not w.isalnum()):
            flush()
            start = w in _SENT_END or start
            continue
        capital = w[:1].isupper() and w.lower() not in STOP_CAPS
        if capital:
            run.append(w)
        elif run and w.lower() in CONNECT:
            run.append(w)
        elif run and w.lower() == "and" and len(run) == 1 and cap(i + 1) and not cap(i + 2):
            run.append(w)  # "Pride and Prejudice", "Romeo and Juliet": one name
        else:
            flush()
        start = False
    flush()
    m = _IDENTITY.match(" ".join(str(text).split()))
    if m and not any(norm(o) in norm(m.group("s")) for o in out):
        subject = re.sub(r"^(?:the|a|an)\s+", "", m.group("s").strip(), flags=re.I)
        if 2 < len(subject) <= 60 and len(subject.split()) <= 6 and re.fullmatch(r"[A-Za-z][\w\s'’\-]*", subject):
            out.append(subject)
    seen, uniq = set(), []
    for o in out:
        k = norm(o)
        if k and k not in seen and len(k) > 1 and not k.isdigit():
            seen.add(k)
            uniq.append(o)
    return tuple(uniq)


def topics_in(text: str, limit: int = 6) -> list[str]:
    """The named things and subjects in a text: runs of capitalised words joined by of/de/von/...
    ("Leonardo da Vinci", "Bank of England"), and an identity question's subject ("speed of light")."""
    return list(_topics(" ".join(str(text or "").split())[:600]))[:limit]


def aim_key(text: str) -> str:
    """Two aims are the same aim whatever their articles and case ("understand the Krebs cycle")."""
    return " ".join(w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in ("the", "a", "an"))


def question_core(question: str) -> str:
    """'Who painted the Zorblax mural?' -> 'who painted the Zorblax mural' (for 'find out ...')."""
    q = " ".join(str(question).split()).rstrip(" ?!.")
    return q[:1].lower() + q[1:] if q[:2] != q[:2].upper() else q


@dataclass
class Area:
    name: str
    key: str
    asked: int = 0
    facts: int = 0
    read: int = 0
    named: int = 0                   # facts that name it without being about it
    linked: int = 0                  # articles it read that link to it
    people: bool = False             # a person said they are interested in it
    stance: str = ""                 # what its thinking nodes (or mind) make of it
    ego: int = 0                     # times its ego said it is curious about it (ego.py)
    gaps: list = field(default_factory=list)   # [{"question", "why", "order"}]
    interest: float = 0.0
    lack: float = 0.0
    score: float = 0.0
    why_interest: list = field(default_factory=list)
    why_lack: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Curiosity:
    """Areas of interest and gaps in what it knows (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.curiosity
        raw = system.store.get_kv("curiosity.state")
        st = json.loads(raw) if raw else {}
        self.asked: Counter = Counter(st.get("asked", {}))           # area key -> people's orders about it
        self.names: dict[str, str] = st.get("names", {})             # area key -> how people write it
        self.unanswered: deque = deque(st.get("unanswered", []), maxlen=100)  # [ts, order id, question]
        self.set_aims: list = st.get("set", [])[-100:]               # objectives it set itself from here
        self.lower: Counter = Counter(st.get("lower", {}))           # words people write in lower case (not names)
        self._cache: tuple[float, dict] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def save(self) -> None:
        top = dict(self.asked.most_common(500))
        self.sys.store.set_kv("curiosity.state", json.dumps({
            "asked": top, "names": {k: v for k, v in self.names.items() if k in top},
            "unanswered": list(self.unanswered), "set": self.set_aims[-100:],
            "lower": dict(self.lower.most_common(3000))}))

    # ------------------------------------------------------------- observing
    def observe(self, order, res) -> None:
        """What people ask about (interest), and what it could not answer (lack). Only questions and
        orders for what it knows count: a review to read for sentiment is not about its names."""
        if not self.enabled or order.background or str(order.source).startswith(("challenge:", "objective", "task:")):
            return
        for w in re.findall(r"(?<![.!?]\s)(?<!^)\b([a-z][a-z'\-]{3,})\b", order.text):  # "meeting", not "Meeting"
            self.lower[w] += 1
        route = res.route or ""
        if route in ("tasks", "mind", "toolsmith", "objectives", "thinking", "corrections", "ego") or route.startswith("skill:"):
            return
        tool = self.sys.tools.tools.get(route[5:]) if route.startswith("tool:") else None
        knowledge = tool is not None and tool.manifest.provides == "knowledge"
        if not (is_question(order.text) or route == DIRECT or knowledge):
            return
        for t in topics_in(order.text):
            k = norm(t)
            self.asked[k] += 1
            self.names.setdefault(k, t)
        if not res.ok and is_question(order.text):
            self.unanswered.append([round(time.time(), 1), order.id, " ".join(order.text.split())[:240]])
        self._cache = None
        self.save()

    # ------------------------------------------------------------- the picture
    def _gap_questions(self) -> list[dict]:
        """Questions it could not answer, answers marked wrong, objective questions left open or disputed."""
        s = self.sys
        out, seen = [], set()
        from .memory import key as fact_key
        known = {fact_key(t.ask) for n in s.memory.fact_nodes() for t in n.facts()} if s.memory.enabled else set()

        def add(question: str, why: str, order: str = "") -> None:
            q = " ".join(str(question or "").split())
            k = q.lower().rstrip("?!. ")
            if len(q) < 6 or k in seen or not is_question(q):
                return
            if fact_key(q) in known:  # it has learned it since
                return
            seen.add(k)
            out.append({"question": q, "why": why, "order": order})
        for ts, oid, q in list(self.unanswered)[::-1]:
            add(q, "a question it could not answer", oid)
        if s.mind.enabled:
            for u in list(s.mind.unmet.values())[::-1]:
                if u.route == DIRECT:
                    add(u.text, "a question it could not answer", u.order_id)
        corr = getattr(s, "corrections", None)
        for oid in (corr.wrong if corr is not None else [])[::-1]:
            o = s.store.get_order(oid)
            if o:
                add(o["text"], "you said its answer was wrong", oid)
        book = getattr(s, "objectives", None)
        if book is not None:
            for obj in book.items.values():
                if obj.status == "removed":
                    continue
                for q in obj.questions:
                    if q["kind"] != "ask":
                        continue
                    if q["status"] in ("unknown", "skipped", "failed") and obj.status in ("met", "unmet"):
                        add(q["ask"], f"{obj.id} could not find it out")
                    elif q["status"] == "answered" and q.get("confirmed") is False:
                        add(q["ask"], f"its sources disagree ({obj.id})")
        return out

    def areas(self) -> dict[str, Area]:
        """Every area it hears or knows about, with its interest and what it lacks. Cached briefly."""
        now = time.time()
        if self._cache is not None and now - self._cache[0] < self.pol.cache_seconds:
            return self._cache[1]
        s = self.sys
        areas: dict[str, Area] = {}

        def area(name: str) -> Area:
            k = norm(name)
            a = areas.get(k)
            if a is None:
                a = areas[k] = Area(self.names.get(k, name), k)
            return a
        for k, n in self.asked.items():
            if n > 0:
                area(self.names.get(k, k)).asked = n
        if s.memory.enabled:  # what it knows: facts about an area, and names it only knows by name
            for node in s.memory.fact_nodes():
                for t in node.facts():
                    about = {norm(x) for x in topics_in(t.ask)}
                    for x in topics_in(t.ask):
                        area(x).facts += 1
                    for x in topics_in(t.answer):
                        if norm(x) not in about:
                            area(x).named += 1
        for d in s.knowledge.documents(self.pol.documents):
            area(d.title).read += 1
            for link in d.links[: self.pol.links_per_document]:
                if len(link) > 2:
                    area(link).linked += 1
        if s.mind.enabled:
            for n in list(s.mind.notes.values()):
                if n.kind == "fact" and n.subject == "user" and n.text.startswith("interested in "):
                    area(n.text[len("interested in "):]).people = True
                elif n.kind == "opinion" and n.subject.startswith("topic:"):
                    k = norm(n.subject[6:])
                    if k in areas or len(areas) < 5000:
                        area(n.subject[6:]).stance = area(n.subject[6:]).stance or n.stance
        thinking = getattr(s, "thinking", None)
        if thinking is not None and thinking.enabled:
            for th in thinking.thoughts.values():
                if th.opinion:
                    area(th.topic).stance = th.opinion.get("stance", "")
        for g in self._gap_questions():
            names = topics_in(g["question"]) or [question_core(g["question"])[:60]]
            area(names[0]).gaps.append(g)
        ego = getattr(s, "ego", None)
        fascinated = ego.fascinations() if ego is not None else {}  # what its ego said it is curious about (ego.py)
        for k, rec in fascinated.items():
            area(rec.get("name") or k).ego = int(rec.get("times", 1))
        common = self.common_words()
        for k in [k for k, a in areas.items() if " " not in k and k in common and not a.people and not a.gaps]:
            del areas[k]  # a word people write in lower case is not a name ("Meeting", "Overall")
        p = self.pol
        for a in areas.values():
            i, why_i = 0.0, []
            if a.people:
                i += 0.6
                why_i.append("a person said they are interested in it")
            if a.asked:
                i += min(0.6, 0.2 * math.log2(1 + a.asked))
                why_i.append(f"people asked about it {a.asked} time{'s' if a.asked != 1 else ''}")
            if a.read:
                i += 0.15
                why_i.append("it has read about it")
            if a.stance in ("favourable", "fascinated"):
                i += 0.2
                why_i.append(f"its thinking nodes find it {a.stance}")
            if a.linked:
                i += min(0.15, 0.05 * a.linked)
                why_i.append(f"{a.linked} article{'s' if a.linked != 1 else ''} it read link to it")
            if a.ego:  # its ego is curious about it: it wants to understand it, so it sets objectives for it sooner
                i += ego.pol.fascination_weight
                why_i.insert(0, "its ego is curious about it")
            lack, why_l = 0.0, []
            if a.gaps:
                lack += min(0.8, 0.4 * len(a.gaps))
                kinds = Counter(g["why"] if not g["why"].startswith(("obj-", "its sources")) else
                                ("its sources disagree" if g["why"].startswith("its sources") else "an objective left it open")
                                for g in a.gaps)
                why_l += [f"{v} x {k}" if v > 1 else k for k, v in kinds.items()]
            if a.asked >= 2 and a.facts < max(2, a.asked // 2):
                lack += 0.3
                why_l.append("it holds no facts about it yet" if not a.facts else
                             f"it holds only {a.facts} fact{'s' if a.facts != 1 else ''} about it")
            if a.facts == 0 and a.named == 1 and not a.read:
                lack += 0.25
                why_l.append("it knows it only from a mention in another fact")
            a.interest, a.lack = round(min(1.0, i), 3), round(min(1.0, lack), 3)
            a.why_interest, a.why_lack = why_i, why_l
            a.score = round(p.interest_weight * a.interest + p.lack_weight * a.lack + p.both_weight * a.interest * a.lack, 3)
            rw = getattr(self.sys, "rewards", None)
            if rw is not None and rw.enabled and a.score > 0:  # people rewarded (or punished) curiosity about it
                pull = max(rw.direction("curiosity", a.key), rw.direction("reading", a.key), rw.direction("research", a.key),
                           key=abs) + rw.pol.node_weight * rw.predict(f"curiosity {a.name}") * (1 if rw.pol.use_node else 0)
                if abs(pull) > 0.02:
                    a.score = round(max(0.0, a.score + rw.pol.curiosity_weight * pull), 3)
                    (a.why_interest if pull > 0 else a.why_lack).append(
                        f"people {'encouraged' if pull > 0 else 'discouraged'} curiosity about it ({pull:+.2f})")
        self._cache = (now, areas)
        return areas

    def common_words(self) -> set:
        """Words seen in lower case, often enough: in what people write and in the facts it holds."""
        cached = getattr(self, "_common", None)
        if cached and time.time() - cached[0] < 30:
            return cached[1]
        seen = Counter({w: n for w, n in self.lower.items() if n >= 2})
        if self.sys.memory.enabled:
            for node in self.sys.memory.fact_nodes():
                for t in node.facts():
                    for w in re.findall(r"\b([a-z][a-z'\-]{3,})\b", f"{t.ask[1:]} {t.answer[1:]}"):
                        seen[w] += 2
        words = {w for w, n in seen.items() if n >= 2}
        self._common = (time.time(), words)
        return words

    def ranked(self, limit: int = 20) -> list[Area]:
        return sorted((a for a in self.areas().values() if a.score > 0), key=lambda a: (-a.score, a.key))[:limit]

    def area(self, name: str) -> Area | None:
        return self.areas().get(norm(name))

    def aims(self, limit: int = 10) -> list[dict]:
        """Objectives the director could set itself: {"text", "why", "kind" (gap | interest), "area", "score"}."""
        if not self.enabled:
            return []
        s = self.sys
        taken = {aim_key(o.text) for o in s.objectives.items.values() if o.status != "removed"}
        out: list[dict] = []
        p = self.pol
        for a in self.ranked(60):
            for g in a.gaps[:2]:
                text = f"find out {question_core(g['question'])}"
                if aim_key(text) in taken:
                    continue
                # a question a person asked (or said was answered wrong) weighs more than one an objective left open
                score = a.score + (0.35 if g["why"] in ("a question it could not answer", "you said its answer was wrong") else 0.15)
                why = g["why"] + (f"; {a.why_interest[0]}" if a.why_interest else "")
                out.append({"text": text, "why": why, "kind": "gap", "area": a.name, "score": round(score, 3),
                            "question": g["question"]})
            if a.interest >= p.min_interest and (a.lack > 0 or a.facts < p.enough_facts) and len(a.key) > 2:
                text = f"understand {a.name}"
                if aim_key(text) in taken or is_question(a.name):
                    continue
                why = "; ".join(a.why_interest[:2] + a.why_lack[:1])
                out.append({"text": text, "why": why, "kind": "interest", "area": a.name, "score": a.score})
            elif a.lack >= p.min_lack and not a.gaps and a.facts == 0 and len(a.key) > 2:
                text = f"understand {a.name}"
                if aim_key(text) not in taken:
                    out.append({"text": text, "why": "; ".join(a.why_lack[:2]), "kind": "gap", "area": a.name,
                                "score": round(a.score * 0.8, 3)})
        seen, uniq = set(), []
        for x in sorted(out, key=lambda x: -x["score"]):
            if aim_key(x["text"]) not in seen:
                seen.add(aim_key(x["text"]))
                uniq.append(x)
        return uniq[:limit]

    def note_set(self, obj_id: str, aim: dict) -> None:
        """The director set itself an objective from here (agency.py): remembered with why."""
        self.set_aims.append({"ts": round(time.time(), 1), "objective": obj_id, **{k: aim.get(k) for k in
                                                                                    ("text", "why", "kind", "area", "score")}})
        self.save()

    # ------------------------------------------------------------------ views
    def status(self) -> dict:
        ranked = self.ranked(self.pol.show)
        book = getattr(self.sys, "objectives", None)
        mine = []
        for rec in self.set_aims[-12:][::-1]:
            obj = book.get(rec["objective"]) if book is not None else None
            mine.append({**rec, "status": obj.status if obj else "removed",
                         "progress": list(obj.progress()) if obj else [0, 0]})
        return {"enabled": self.enabled, "areas": [a.to_dict() for a in ranked],
                "aims": self.aims(8), "set": mine, "asked": sum(self.asked.values()),
                "gaps": sum(len(a.gaps) for a in self.areas().values())}

    def lines(self) -> list[str]:
        if not self.enabled:
            return ["curiosity is off ([curiosity] enabled = false in the directives)"]
        st = self.status()
        out = [f"curiosity: {len(st['areas'])} areas worth its attention; {st['gaps']} gaps in what it knows"]
        for a in st["areas"][:12]:
            why = "; ".join(a["why_interest"][:2] + a["why_lack"][:2])
            out.append(f"  {a['name'][:40]:<40} interest {a['interest']:.2f}  lacks {a['lack']:.2f}  ({why})")
        if st["aims"]:
            out.append("objectives it could set itself (agency mode pursue):")
            out += [f"  {x['text'][:70]}  [{x['kind']}, {x['score']:.2f}]: {x['why']}" for x in st["aims"][:6]]
        if st["set"]:
            out.append("objectives it set itself from here:")
            out += [f"  {r['objective']} [{r['status']} {r['progress'][0]}/{r['progress'][1]}] {r['text']}: {r['why']}"
                    for r in st["set"][:6]]
        return out
