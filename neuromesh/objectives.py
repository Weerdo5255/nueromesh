"""Objectives: long-term aims the mesh plans for, works on by asking itself, ruminates over, and
writes up (P16).

    "objective: understand the Berlin Wall"            (also /objective ..., the web, or its own initiative)
      --> [planner node]   which strategy? explain | answer | compare | survey | watch | improve | build
      --> plan             the questions it must answer (an LLM teacher may draft them), and the steps
                           around them: research a subject, hand a watch to a background task, ...
      --> pursue           one step at a time, only when idle (P2) and within budget (P4): each question
                           is a query against itself - an order that passes a director on the way in and
                           out (P14), so the router, the tools, memory nodes, retained knowledge and the
                           teachers all answer it, and a director tries other ways when they do not
      --> ruminate         when the steps run out it thinks it over: checks each answer against a second
                           source (agrees: confirmed; disagrees: a contradiction it must settle), tries
                           what failed another way (reworded, researched, or, at last, asked of a person),
                           follows up what the answers raise (terms it does not know; the items of a list)
      --> report           state/objectives/<id>.md, rewritten as it goes: what it found, each question with
                           its answer, source, confidence and whether a second source confirmed it, what it
                           could not find out, and how it got there
      --> met              every core question answered, nothing contradicted; or it ends and says what is missing

Rumination also turns on its own experience: "second thoughts" re-ask recent questions it answered
unsurely or not at all, and note what it would say now (agency mode ruminate).

The planner node is an ordinary internal capability: grown, trained, evolved and promoted like any
node, taught by the PlannerTeacher (cues and practice objectives), people, and LLM teachers.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import random
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .agents import Agent, is_question
from .directors import agrees
from .tasks import DIRECT, Order
from .teachers import Lesson, Teacher

PLANNER = "planner"
STRATEGIES = ("explain", "answer", "compare", "survey", "watch", "improve", "build")
DONE = ("met", "unmet", "removed")

# A person sets an objective: "objective: ...", "long-term goal: ...", "make it your objective to ...".
OBJECTIVE_ORDER = re.compile(
    r"^\s*(?:please\s+)?(?:(?:set\s+(?:yourself\s+)?(?:an?\s+|the\s+)?)?(?:long[- ]term\s+)?(?:objective|goal|aim)"
    r"(?:\s+(?:of|to))?\s*[:\-]\s*|(?:make\s+it\s+your|your)\s+(?:long[- ]term\s+)?(?:objective|goal|aim)\s+(?:is\s+)?to\s+"
    r"|work\s+towards?\s+|over\s+time,?\s+)(?P<what>.+?)\s*$", re.I | re.S)


def objective_request(text: str) -> str | None:
    """What a person set as an objective, if the order sets one."""
    m = OBJECTIVE_ORDER.match(str(text))
    return " ".join(m.group("what").split()).rstrip(".") if m else None


# ======================================================================= the planner teacher
CUES = {
    "explain": (r"\bunderstand\b", r"\blearn (?:all |everything )?about\b", r"^explain\b", r"\bexpert (?:on|in)\b",
                r"\breport (?:on|about)\b", r"^stud(?:y|ying)\b", r"\bin depth\b", r"\beverything about\b",
                r"^find out (?:all |more )?about\b", r"\bget to know\b", r"^research\b", r"\bread up on\b"),
    "answer": (r"^(?:find out|work out|figure out|determine|discover|establish) (?:who|what|when|where|why|how|which|whether)\b",
               r"^(?:who|what|when|where|why|how|which)\b", r"\?\s*$", r"^answer\b"),
    "compare": (r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifferences? between\b", r"\bwhich is (?:better|bigger|larger|older)\b",
                r"\bdiffer from\b"),
    "survey": (r"^(?:list|name|catalogue|catalog|enumerate)\b", r"\b(?:all|every|each) (?:of )?the\b", r"^find (?:all|every)\b",
               r"\bmake a list of\b", r"^what are all\b"),
    "watch": (r"\bkeep an eye on\b", r"\b(?:track|monitor|watch|follow)\b", r"\bkeep tabs on\b", r"\bnotify me (?:when|if)\b",
              r"\blet me know (?:when|if)\b"),
    "improve": (r"\bget (?:my |your |its )?\w+ (?:skill )?(?:to|up to) \d", r"\bimprove (?:my |your |its )?\w+", r"\bget better at\b",
                r"\breach \d+(?:\.\d+)?%"),
    "build": (r"\bbe able to\b", r"\bmake (?:a |yourself a )?tool\b", r"\bwrite (?:a |yourself a )?tool\b", r"\blearn (?:how )?to (?!know)\w+",
              r"\bteach yourself to\b"),
}
TOPICS = ["the Berlin Wall", "photosynthesis", "the Krebs cycle", "jazz", "black holes", "the Roman Empire", "volcanoes",
          "machine learning", "the printing press", "Ada Lovelace", "the Apollo program", "plate tectonics"]
PAIRS = [("cats", "dogs"), ("Python", "JavaScript"), ("Mars", "Venus"), ("jazz", "blues"), ("tea", "coffee"),
         ("the Moon", "the Sun"), ("rivers", "lakes")]
SETS = ["the planets of the solar system", "the noble gases", "the continents", "the Great Lakes", "the primary colours",
        "the moons of Mars", "the members of the Beatles"]
PRACTICE = {
    "explain": ["understand {t}", "learn all about {t}", "explain {t}", "become an expert on {t}", "write a report on {t}",
                "study {t} in depth", "find out about {t}", "get to know {t}", "everything about {t}", "read up on {t}"],
    "answer": ["find out who invented the telephone", "work out how tall Mount Everest is", "find out when {t} began",
               "what is the capital of Australia", "figure out why the sky is blue", "determine where {t} came from",
               "find out who wrote Pride and Prejudice and when it was published", "which planet is the largest?"],
    "compare": ["compare {a} and {b}", "{a} versus {b}", "{a} vs {b}", "the difference between {a} and {b}",
                "which is better, {a} or {b}", "how do {a} differ from {b}"],
    "survey": ["list {s}", "name all {s}", "find every one of {s}", "make a list of {s}", "list all of {s}",
               "catalogue {s}", "what are all {s}"],
    "watch": ["keep an eye on the price of gold", "track {t}", "monitor the server status", "keep tabs on {t}",
              "let me know when {t} changes", "watch the weather in Lisbon", "follow the news about {t}"],
    "improve": ["get your sentiment skill to 95%", "improve your urgency skill", "get better at routing",
                "reach 97% on sentiment", "improve your topic skill to 0.9", "get urgency up to 98%"],
    "build": ["be able to reverse text", "learn to convert roman numerals", "make a tool that counts vowels",
              "be able to write numbers in words", "teach yourself to decode base64", "write a tool for morse code"],
}


def cue_scores(text: str) -> dict[str, float]:
    t = " ".join(str(text).lower().split())
    out: dict[str, float] = {}
    for strategy, cues in CUES.items():
        score = sum(1.0 + (0.5 if c.startswith("^") else 0.0) for c in cues if re.search(c, t))
        if score:
            out[strategy] = score
    return out


class PlannerTeacher(Teacher):
    """Teaches the planner node which strategy an objective needs, from the strategies' cues, and
    writes practice objectives for every strategy. A specialist."""
    name = "planner"
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def status(self) -> dict:
        return {"detail": "specialist: teaches the planner node which strategy an objective needs"}

    def label_sync(self, text: str) -> Lesson:
        scores = cue_scores(text)
        if not scores:
            return Lesson("explain", 0.55, self.name, "no cue: understanding it is the default")
        best = max(scores.items(), key=lambda kv: (kv[1], -STRATEGIES.index(kv[0])))
        return Lesson(best[0], round(0.55 + 0.4 * best[1] / sum(scores.values()), 3), self.name, f"cues={scores}")

    async def label(self, spec, text):
        return self.label_sync(text) if spec.name == PLANNER else None

    async def generate(self, spec, label, n):
        if spec.name != PLANNER or label not in PRACTICE:
            return []
        out: list[str] = []
        for _ in range(n * 8):
            a, b = self.rng.choice(PAIRS)
            text = self.rng.choice(PRACTICE[label]).format(t=self.rng.choice(TOPICS), a=a, b=b, s=self.rng.choice(SETS))
            if text not in out and self.label_sync(text).label == label:
                out.append(text)
            if len(out) >= n:
                break
        return out


# ============================================================================ planning
_LEAD = re.compile(r"^(?:please\s+)?(?:try\s+to\s+|aim\s+to\s+|i\s+want\s+you\s+to\s+)?(?:understand|learn\s+(?:all\s+|everything\s+)?about|"
                   r"explain|become\s+an\s+expert\s+(?:on|in)|write\s+(?:me\s+)?(?:a\s+)?report\s+(?:on|about)|study|research|"
                   r"find\s+out\s+(?:all\s+|more\s+)?about|get\s+to\s+know|read\s+up\s+on|know\s+(?:all\s+|everything\s+)?about|"
                   r"everything\s+about|keep\s+an\s+eye\s+on|keep\s+tabs\s+on|track|monitor|watch|follow(?:\s+the\s+news\s+about)?|"
                   r"list(?:\s+all)?(?:\s+of)?|name\s+(?:all|every)(?:\s+of)?|find\s+(?:all|every\s+one)(?:\s+of)?|"
                   r"make\s+a\s+list\s+of|catalogue|catalog|enumerate|what\s+are\s+all)\s+", re.I)
_TAIL = re.compile(r"\s+(?:in\s+depth|thoroughly|well|properly|in\s+detail|over\s+time|for\s+me)\s*$", re.I)
_SPLIT = re.compile(r"\s*(?:[?;]|\band\s+(?=(?:who|what|when|where|why|how|which)\b)|,\s*(?=(?:who|what|when|where|why|how|which)\b))\s*", re.I)
_EMBEDDED = re.compile(r"^(?P<wh>how(?:\s+(?:many|much|tall|old|far|long|big|high|deep|large))?|what|who|where|when|why|which(?:\s+\w+)?)"
                       r"\s+(?P<rest>.+?)\s+(?P<aux>is|are|was|were)$", re.I)
_EMBEDDED_PRONOUN = re.compile(r"^(?P<wh>when|where|why|how|whether)\s+(?P<subj>it|he|she|they|this|that)\s+"
                               r"(?P<aux>was|is|were|are|did|does|do|will|has|had)\s+(?P<rest>.+)$", re.I)
_LEADING_VERB = re.compile(r"^(?:find\s+out|work\s+out|figure\s+out|determine|discover|establish|answer)\s+", re.I)


def subject_of(text: str, strategy: str) -> str:
    t = " ".join(str(text).split()).strip(" .?!")
    t = _LEAD.sub("", t, count=1)
    t = _TAIL.sub("", t)
    if strategy == "watch":
        t = re.sub(r"^(?:let me know|notify me)\s+(?:when|if)\s+", "", t, flags=re.I)
    if strategy == "build":
        t = re.sub(r"^(?:be\s+able\s+to|learn\s+(?:how\s+)?to|teach\s+yourself\s+(?:how\s+)?to|(?:make|write|build)\s+"
                   r"(?:yourself\s+)?(?:a|an)\s+tool\s+(?:that\s+(?:can\s+)?|to\s+|for\s+))\s*", "", t, flags=re.I)
    return t.strip(" .?!") or str(text).strip()


def compare_pair(text: str) -> tuple[str, str] | None:
    t = " ".join(str(text).split()).strip(" .?!")
    for pat in (r"(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+)", r"differences?\s+between\s+(.+?)\s+and\s+(.+)",
                r"which\s+is\s+\w+,?\s+(.+?)\s+or\s+(.+)", r"how\s+(?:do|does)\s+(.+?)\s+differ\s+from\s+(.+)",
                r"^(.+?)\s+(?:vs\.?|versus)\s+(.+)"):
        m = re.search(pat, t, re.I)
        if m:
            return m.group(1).strip(" ,"), m.group(2).strip(" ,")
    return None


def _object_of(question: str) -> str:
    """What a question is about: "who wrote Pride and Prejudice" -> "Pride and Prejudice"."""
    m = re.match(r"^(?:who|what|when|where|which|why|how(?:\s+many|\s+much|\s+tall|\s+old|\s+far|\s+long)?)\s+\w+\s+(.+)$",
                 question.strip(" ?"), re.I)
    return m.group(1).strip() if m else ""


def split_questions(text: str) -> list[str]:
    """The questions in an "answer" objective, each a question of its own:
    "find out who wrote Pride and Prejudice and when it was published" ->
    ["Who wrote Pride and Prejudice?", "When was Pride and Prejudice published?"]."""
    t = _LEADING_VERB.sub("", " ".join(str(text).split()).strip(), count=1)
    parts = [p.strip(" ,.") for p in _SPLIT.split(t) if p and p.strip(" ,.")]
    out: list[str] = []
    for p in parts:
        m = _EMBEDDED_PRONOUN.match(p) or _EMBEDDED.match(p)  # "when it was published" -> "when was it published"
        if m:
            d = m.groupdict()
            p = f"{d['wh']} {d['aux']} {d.get('subj') or d['rest']}" + (f" {d['rest']}" if d.get("subj") else "")
        if out:  # "it" is what the first question was about
            subject = _object_of(out[0])
            if subject:
                p = re.sub(r"\b(?:it|they|he|she)\b", subject, p, count=1, flags=re.I)
        out.append(p[0].upper() + p[1:] + ("" if p.endswith("?") else "?"))
    return out


def _be(subject: str) -> str:
    """'is' or 'are': "volcanoes" are, "the Berlin Wall" is."""
    last = subject.split()[-1].lower() if subject.split() else ""
    plural = last.endswith("s") and not last.endswith(("ss", "us", "is", "ics")) and " of " not in subject
    return "are" if plural else "is"


def looks_like_a_person(name: str) -> bool:
    words = name.split()
    return 2 <= len(words) <= 3 and all(w[:1].isupper() for w in words) and words[0].lower() not in ("the", "a", "an")


def plan_questions(strategy: str, text: str, subject: str) -> list[dict]:
    """The plan's steps for a strategy (offline: templates; an LLM teacher may draft them instead)."""
    def ask(q, core=True, why=""):
        return {"kind": "ask", "ask": q, "core": core, "why": why}

    def research(topic):
        return {"kind": "research", "ask": topic, "core": False, "why": f"read up on {topic} first"}
    s = subject
    if strategy == "explain":
        be = _be(s)
        first = f"Who was {s}?" if looks_like_a_person(s) else f"What {be} {s}?"
        return [research(s), ask(first, why="what it is"), ask(f"What {be} {s} known for?", why="why it matters"),
                ask(f"What is the history of {s}?", core=False, why="where it comes from")]
    if strategy == "answer":
        return [ask(q, why="asked for") for q in split_questions(text)] or [ask(text if text.endswith("?") else text + "?")]
    if strategy == "compare":
        pair = compare_pair(text) or (s, "")
        a, b = pair
        if not b:
            return plan_questions("explain", text, s)
        return [research(a), research(b), ask(f"What {_be(a)} {a}?", why="one side"), ask(f"What {_be(b)} {b}?", why="the other side"),
                ask(f"What is the difference between {a} and {b}?", core=False, why="how they differ")]
    if strategy == "survey":
        return [research(s), ask(f"What are {s}?", why="the list itself")]
    if strategy == "watch":
        order = s if is_question(s) else f"what is {s}"
        return [{"kind": "delegate", "ask": order, "core": True, "why": "keep watching it", "delegate": "watch"}]
    if strategy == "improve":
        return [{"kind": "delegate", "ask": text, "core": True, "why": "train the skill", "delegate": "improve"}]
    if strategy == "build":
        return [{"kind": "delegate", "ask": s, "core": True, "why": "a tool would do it", "delegate": "build"}]
    return [ask(text)]


# ======================================================================== working it out
_STOP_TERMS = {"The", "A", "An", "It", "Its", "This", "That", "These", "Those", "In", "On", "At", "By", "Of", "For", "And",
               "But", "Or", "As", "He", "She", "They", "We", "I", "You", "There", "Here", "When", "Where", "What", "Who",
               "Why", "How", "Which", "Source", "Sources", "Today", "Yes", "No", "January", "February", "March", "April",
               "May", "June", "July", "August", "September", "October", "November", "December", "Monday", "Tuesday",
               "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
_NUMBER_WORDS = r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|hundred|thousand|million|billion|dozen"


def specific(question: str) -> str:
    """What kind of thing a question asks for, when an answer can be checked for it: a time, a number,
    a name, a place ("" for questions any answer might fit)."""
    q = question.lower().strip()
    if re.match(r"^(?:when|in what year|what year|what date|on what date)\b", q):
        return "a time"
    if re.match(r"^how (?:many|much|tall|old|far|long|big|high|deep|large|fast|heavy)\b", q):
        return "a number"
    if re.match(r"^who\b", q):
        return "a name"
    if re.match(r"^where\b", q):
        return "a place"
    return ""


def fits(question: str, answer) -> bool:
    """Does the answer give what the question asks for? (A time for "when", a number for "how many", ...)"""
    need, a = specific(question), str(answer)
    if need == "a time":
        return bool(re.search(rf"\b\d{{3,4}}\b|\b(?:{_MONTHS})\b|\bcentury\b|\bago\b", a, re.I))
    if need == "a number":
        return bool(re.search(rf"\d|\b(?:{_NUMBER_WORDS})\b", a, re.I))
    if need in ("a name", "a place"):
        asked = set(re.findall(r"[A-Za-z]+", question.lower()))
        return any(w.lower() not in asked for w in re.findall(r"\b[A-Z][a-z]+\b", a))
    return True


def new_terms(answer: str, known: str, limit: int) -> list[str]:
    """Names an answer mentions that the objective does not cover yet ("... built by the GDR ..." -> "GDR")."""
    out: list[str] = []
    low = known.lower()
    for m in re.finditer(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:of\s+|the\s+|and\s+)?(?:[A-Z][a-z]+|[A-Z]{2,}))*\b", str(answer)):
        term = m.group(0).strip()
        start = m.start()
        sentence_start = start == 0 or str(answer)[max(0, start - 2):start].strip() in (".", "!", "?", "")
        if term in _STOP_TERMS or term.lower() in low or len(term) < 3:
            continue
        if sentence_start and " " not in term:
            continue  # a capital that only starts a sentence
        if term not in out:
            out.append(term)
        if len(out) >= limit:
            break
    return out


def list_items(answer: str, limit: int = 12) -> list[str]:
    """The items of a list answer: "Mercury, Venus, Earth and Mars" -> four items."""
    t = str(answer).split("\n")[0]
    t = re.sub(r"^.*?\b(?:are|is|include|includes|were)\s*:?\s+", "", t, count=1)
    t = re.sub(r"\.\s.*$", "", t).strip(" .")
    parts = [p.strip(" .;:") for p in re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+|;\s*", t) if p.strip(" .;:")]
    items = [re.sub(r"^(?:the|a|an)\s+", "", p, flags=re.I) for p in parts if 0 < len(p.split()) <= 5]
    return items[:limit] if len(items) >= 2 else []


def reword(question: str) -> str | None:
    """The same question in other words, for a second try."""
    q = question.strip(" ?")
    for pat, rep in ((r"^what is (.+?) known for$", r"why is \1 famous"), (r"^what is the history of (.+)$", r"how did \1 begin"),
                     (r"^what is the difference between (.+?) and (.+)$", r"how does \1 differ from \2"),
                     (r"^what (?:is|are) (.+)$", r"tell me about \1"), (r"^who (?:is|was) (.+)$", r"tell me about \1"),
                     (r"^when (was|were|did) (.+)$", r"in what year \1 \2"), (r"^how does (.+?) work$", r"explain how \1 works"),
                     (r"^why (.+)$", r"explain why \1")):
        if re.match(pat, q, re.I):
            return re.sub(pat, rep, q, flags=re.I) + ("?" if not rep.startswith(("tell", "explain")) else "")
    return None


def source_of(res, system=None) -> str:
    """Where an answer came from, as a kind a second source can be told apart from: memory nodes,
    retained knowledge (a research tool's answer is reading, too), a teacher, or a tool or skill."""
    route = res.route or ""
    if res.knowledge == "memory":
        return "memory"
    if res.knowledge in ("recalled", "researched"):
        return "knowledge"
    tool = system.tools.tools.get(route[5:]) if system is not None and route.startswith("tool:") else None
    if tool is not None and tool.manifest.provides == "knowledge":
        return "knowledge"
    if route.startswith(("tool:", "skill:")):
        return route
    if res.served_by == "teachers":
        return "teacher"
    return route or "unknown"


def _answer_words(text) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in
            ("the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "and", "what", "who", "how", "why", "it")]


def first_sentence(text: str, n: int = 260) -> str:
    t = " ".join(str(text).split())
    m = re.match(r"(.+?[.!?])(?:\s|$)", t)
    t = m.group(1) if m else t
    return t if len(t) <= n else t[: n - 1] + "…"


@dataclass
class Objective:
    id: str
    text: str
    strategy: str = ""
    subject: str = ""
    status: str = "active"          # active | paused | met | unmet | removed
    created_by: str = "user"
    created_at: float = field(default_factory=time.time)
    planned_by: str = ""
    self_set: bool = False          # set by its own initiative (agency mode pursue)
    questions: list = field(default_factory=list)
    thoughts: list = field(default_factory=list)   # [ts, kind, text]
    steps: int = 0
    rounds: int = 0
    next_at: float = 0.0
    ended_at: float = 0.0
    note: str = ""
    summary: str = ""
    summary_by: str = ""

    def core(self) -> list[dict]:
        return [q for q in self.questions if q.get("core") and q["kind"] != "research"]

    def answered(self, qs=None) -> list[dict]:
        return [q for q in (self.core() if qs is None else qs) if q["status"] == "answered"]

    def progress(self) -> tuple[int, int]:
        core = self.core()
        return len(self.answered(core)), len(core)

    def describe(self) -> str:
        done, total = self.progress()
        return f"{self.id} [{self.status}] {self.text} ({self.strategy}; {done}/{total} core questions answered)"


class Objectives:
    """The objectives people set, and the ones it sets itself (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.objectives
        self.dir = system.state_dir / "objectives"
        names = {f.name for f in fields(Objective)}
        self.items: dict[str, Objective] = {}
        for d in system.store.load_records("objectives"):
            self.items[d["id"]] = Objective(**{k: v for k, v in d.items() if k in names})
        nums = [int(i.split("-")[-1]) for i in self.items if i.split("-")[-1].isdigit()]
        self._ids = itertools.count(max(nums, default=0) + 1)
        raw = system.store.get_kv("objectives.state")
        st = json.loads(raw) if raw else {}
        self.recent: deque = deque(t for t in st.get("recent", []) if time.time() - t < 3600)  # steps this hour (P4)
        self.second: deque = deque(st.get("second", []), maxlen=40)       # second thoughts it had
        self.reconsidered: list[str] = st.get("reconsidered", [])[-300:]   # orders it has thought over again
        self.counts: dict[str, int] = st.get("counts", {})
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def save(self, obj: Objective | None = None) -> None:
        if obj is not None:
            self.sys.store.save_record("objectives", obj.id, obj.status, asdict(obj))
        self.sys.store.set_kv("objectives.state", json.dumps({"recent": list(self.recent), "second": list(self.second),
                                                               "reconsidered": self.reconsidered[-300:],
                                                               "counts": self.counts}))

    def budget_left(self) -> int:
        now = time.time()
        while self.recent and now - self.recent[0] > 3600:
            self.recent.popleft()
        return self.pol.steps_per_hour - len(self.recent)

    def _count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def get(self, oid: str) -> Objective | None:
        return self.items.get(oid)

    def active(self) -> list[Objective]:
        return [o for o in self.items.values() if o.status == "active"]

    # ---------------------------------------------------------------- setting
    async def add(self, text: str, by: str = "user", self_set: bool = False) -> Objective:
        if not self.enabled:
            raise ValueError("objectives are off ([objectives] enabled = false in the directives)")
        text = " ".join(str(text).split()).strip(" .")
        if len(text) < 4:
            raise ValueError("say what the objective is, e.g. objective: understand the Berlin Wall")
        live = [o for o in self.items.values() if o.status in ("active", "paused")]
        if len(live) >= self.pol.max_objectives:
            raise ValueError(f"already {len(live)} objectives (max {self.pol.max_objectives}); finish or remove one first")
        same = next((o for o in live if o.text.lower() == text.lower()), None)
        if same is not None:
            raise ValueError(f"{same.id} already works on that")
        s = self.sys
        obj = Objective(f"obj-{next(self._ids)}", text, created_by=by, self_set=self_set, next_at=time.time())
        self.items[obj.id] = obj
        await self.plan(obj)
        s.guardian.audit(by, "objective.set", obj.id, text=text, strategy=obj.strategy, planned_by=obj.planned_by,
                         questions=len(obj.questions), self_set=self_set)
        s.bus.publish("objective.set", id=obj.id, text=text[:200], strategy=obj.strategy, by=by,
                      planned_by=obj.planned_by, steps=len(obj.questions), self_set=self_set)
        self.save(obj)
        self.write_report(obj)
        return obj

    async def plan(self, obj: Objective) -> None:
        """The planner node (or its teacher) picks a strategy; an LLM teacher may draft the questions."""
        s = self.sys
        if PLANNER in s.capabilities:
            inf = await s.infer(PLANNER, obj.text)
            strategy, how = inf.label, (f"planner node {inf.node_id} ({inf.confidence:.0%} sure)" if inf.kind == "node"
                                        else f"the planner's cues ({inf.provenance})")
        else:
            lesson = PlannerTeacher().label_sync(obj.text)
            strategy, how = lesson.label, "the planner's cues"
        if strategy not in STRATEGIES:
            strategy, how = PlannerTeacher().label_sync(obj.text).label, "the planner's cues"
        obj.strategy = strategy
        obj.subject = subject_of(obj.text, strategy)
        steps = plan_questions(strategy, obj.text, obj.subject)
        if self.pol.llm_plan and strategy in ("explain", "answer", "compare", "survey"):
            drafted = await s.teachers.plan(obj.text, strategy, sorted(s.capabilities), sorted(s.tools.tools))
            if drafted and drafted[0].get("questions"):
                qs = [{"kind": "ask", "ask": str(q.get("ask", "")).strip(), "core": bool(q.get("core", True)),
                       "why": str(q.get("why", ""))[:120]} for q in drafted[0]["questions"] if str(q.get("ask", "")).strip()]
                if qs:
                    steps = [x for x in steps if x["kind"] == "research"] + qs[: self.pol.max_questions - 2]
                    how += f"; questions drafted by the {drafted[1]} teacher"
        obj.planned_by = how
        obj.questions = []
        for st in steps:
            self._add_question(obj, st["ask"], st["kind"], st.get("why", ""), "plan", st.get("core", True),
                               delegate=st.get("delegate", ""))
        self._think(obj, "plan", f"Planned with the {strategy} strategy ({how}): "
                    + "; ".join(q["ask"] for q in obj.questions if q["kind"] != "research")[:400])

    def _add_question(self, obj: Objective, ask: str, kind: str = "ask", why: str = "", origin: str = "rumination",
                      core: bool = False, parent: str = "", delegate: str = "", role: str = "") -> dict | None:
        ask = " ".join(str(ask).split())
        if not ask or len(obj.questions) >= self.pol.max_questions:
            return None
        if any(q["ask"].lower() == ask.lower() and q["kind"] == kind for q in obj.questions):
            return None
        q = {"id": f"q{len(obj.questions) + 1}", "ask": ask, "kind": kind, "why": why, "from": origin, "core": core,
             "status": "todo", "tries": 0, "tried": [], "answer": None, "source": "", "confidence": 0.0, "order": "",
             "confirmed": None, "checked_by": "", "conflict": "", "parent": parent, "role": role, "inquiry": "",
             "delegate": delegate, "ts": round(time.time(), 1)}
        obj.questions.append(q)
        return q

    def ask(self, oid: str, question: str, by: str) -> tuple[bool, str]:
        """A person adds a question to an objective."""
        obj = self.items.get(oid)
        if obj is None or obj.status in DONE:
            return False, f"no open objective {oid} (/objectives lists them)"
        q = self._add_question(obj, question if question.rstrip().endswith("?") else question.rstrip() + "?", "ask",
                               f"{by} asked", "person", core=True)
        if q is None:
            return False, "it already asks that, or the objective has its full share of questions"
        obj.next_at = min(obj.next_at, time.time())
        self._think(obj, "plan", f"{by} added a question: {q['ask']}")
        self.save(obj)
        self.write_report(obj)
        return True, f"added {q['id']} to {oid}: {q['ask']}"

    def set_status(self, oid: str, status: str, by: str) -> tuple[bool, str]:
        obj = self.items.get(oid)
        if obj is None:
            return False, f"no objective {oid} (/objectives lists them)"
        if status == "removed":
            obj.status = "removed"
            obj.ended_at = time.time()
            self.sys.store.delete_record("objectives", oid)
            del self.items[oid]
        elif obj.status in DONE:
            return False, f"{oid} has ended ({obj.status})"
        elif obj.status == status:
            return False, f"{oid} is already {status}"
        else:
            obj.status = status
            if status == "active":
                obj.next_at = time.time()
            self.save(obj)
        self.sys.guardian.audit(by, f"objective.{status}", oid, text=obj.text)
        self.sys.bus.publish(f"objective.{status}", id=oid, text=obj.text[:200], by=by)
        return True, f"{oid} ({obj.text}) {status}"

    # ---------------------------------------------------------------- working
    def due(self) -> Objective | None:
        now = time.time()
        ready = [o for o in self.items.values() if o.status == "active" and o.next_at <= now]
        return min(ready, key=lambda o: o.next_at) if ready else None

    async def work(self, obj: Objective, by: str = "ruminator") -> str | None:
        """One unit of work on an objective: the next step, or a round of rumination."""
        lock = self._locks.setdefault(obj.id, asyncio.Lock())
        async with lock:
            if obj.status != "active":
                return None
            if self.budget_left() <= 0:
                obj.note = f"waiting for its budget of steps ({self.pol.steps_per_hour} an hour)"
                obj.next_at = time.time() + 60
                return None
            now = time.time()
            q = next((x for x in obj.questions if x["status"] == "todo"), None)
            if q is not None:
                line = await self._step(obj, q, by)
                obj.next_at = time.time() + self.pol.step_seconds
            else:
                waiting = [x for x in obj.questions if x["status"] == "waiting"]
                expired = [x for x in waiting if now - x["ts"] > self.pol.person_wait_seconds]
                for x in expired:
                    x["status"] = "unknown"
                    self._think(obj, "gap", f"No one answered '{x['ask']}' in time; it stays an open question.")
                if waiting and len(expired) < len(waiting):
                    obj.note = f"waiting for you: {', '.join(x['inquiry'] for x in waiting if x['inquiry'])} (/inquiries)"
                    obj.next_at = now + min(60.0, self.pol.person_wait_seconds)
                    self.save(obj)
                    return None
                line = await self.ruminate(obj, by)
                obj.next_at = time.time() + self.pol.step_seconds
            self.save(obj)
            self.write_report(obj)
            return line

    async def _step(self, obj: Objective, q: dict, by: str) -> str:
        s = self.sys
        self.recent.append(time.time())
        obj.steps += 1
        q["tries"] += 1
        self._count("steps")
        t0 = time.monotonic()
        try:
            with s.activity.track("objective", obj.id, f"{q['ask'][:80]}"):
                if q["kind"] == "research":
                    line = await self._research(obj, q)
                elif q["kind"] == "person":
                    line = self._ask_person(obj, q)
                elif q["kind"] == "delegate":
                    line = await self._delegate(obj, q, by)
                else:
                    line = await self._query(obj, q)
        finally:
            s.guardian.note_background_busy(time.monotonic() - t0)
        obj.note = line[:200]
        s.bus.publish("objective.step", id=obj.id, question=q["id"], ask=q["ask"][:160], kind=q["kind"],
                      status=q["status"], line=line[:240])
        return line

    async def _query(self, obj: Objective, q: dict) -> str:
        """A query against itself: an order through a director, the router, tools, memory and teachers."""
        s = self.sys
        order = Order(text=q["ask"], source=f"objective:{obj.id}", background=True)
        res = await s.orchestrator.fulfil(order)
        q["order"] = order.id
        ok, why = s.directors.answered(order, res)
        src = source_of(res, s) if ok else (res.route or "nothing")
        if ok and not fits(q["ask"], res.output):
            ok, why = False, f"its answer ('{first_sentence(str(res.output), 80)}') does not give {specific(q['ask'])}"
        q["tried"].append(src if ok else f"{src} ({why.split(';')[0].strip()})"[:160])
        if ok:
            self._answer(obj, q, str(res.output), src, res.confidence, res.sources[:2])
            return f"{q['ask']} -> {first_sentence(q['answer'], 120)} ({src}, {res.confidence:.0%})"
        q["status"] = "failed"
        q["why_failed"] = why[:160]
        return f"{q['ask']} -> not answered: {why[:120]}"

    def _answer(self, obj, q, answer: str, source: str, confidence: float, sources=()) -> None:
        q.update(status="answered", answer=answer[:2000], source=source, confidence=round(float(confidence), 3),
                 sources=list(sources), confirmed=None, checked_by="")
        if source.startswith("tool:") or source == "person":  # computed, or a person's word (P1): nothing to check
            q["confirmed"], q["checked_by"] = True, ("a person" if source == "person" else f"{source} (computed)")
        parent = next((x for x in obj.questions if x["id"] == q.get("parent")), None)
        if parent is not None and q.get("role") in ("alt", "retry", "settle") and parent["kind"] == "ask" and \
                (parent["status"] != "answered" or parent["confirmed"] is False):
            settled = parent["confirmed"] is False
            self._answer(obj, parent, answer, source, confidence, sources)
            parent["via"] = q["id"]
            if settled:
                parent["settled"] = f"{q['id']} ({source})"

    async def _research(self, obj: Objective, q: dict) -> str:
        s = self.sys
        tool = s.knowledge_tool()
        if tool is None:
            q.update(status="skipped", note="no research tool is active")
            return f"research '{q['ask']}' skipped: no research tool is active"
        if not s.guardian.authorize_research(background=True, unlimited=s.unlimited(tool)).allowed:
            q["status"] = "todo" if q["tries"] < 3 else "skipped"
            q["note"] = "its research budget is spent"
            obj.next_at = time.time() + 60
            return f"research '{q['ask']}' waits for research budget"
        doc, why = await s.research(q["ask"], reason=f"{obj.id}: {obj.text[:80]}", background=True, tool=tool)
        if doc is None:
            q.update(status="unknown", note=str(why)[:160])
            q["tried"].append(f"research ({why})"[:120])
            return f"research '{q['ask']}': {why}"
        q.update(status="answered", answer=f"read '{doc.title}' ({doc.source})", source="research", confidence=1.0,
                 confirmed=True, checked_by="retained")
        for other in obj.questions:  # what failed, or is disputed, gets another try now it has read this
            if other["kind"] == "ask" and other["status"] in ("failed", "retried") and not other.get("after_research"):
                other.update(status="todo", after_research=True)
        return f"read '{doc.title}' from {doc.source}"

    def _ask_person(self, obj: Objective, q: dict) -> str:
        s = self.sys
        inq = s.mind.ask("objective", f"I'm working on my objective '{obj.text}' ({obj.id}) and couldn't find out: "
                                      f"{q['ask']} Do you know? (or 'skip')",
                         f"I tried {', '.join(t.split(' (')[0] for t in q['tried'][:4]) or 'everything I have'}",
                         about=f"{obj.id}:{q['id']}") if s.mind.enabled else None
        if inq is None:
            q["status"] = "unknown"
            return f"could not ask anyone: {q['ask']}"
        q.update(status="waiting", inquiry=inq.id, ts=round(time.time(), 1))
        self._count("asked")
        s.bus.publish("objective.asked", id=obj.id, question=q["id"], ask=q["ask"][:160], inquiry=inq.id)
        return f"asked you ({inq.id}): {q['ask']}"

    async def _delegate(self, obj: Objective, q: dict, by: str) -> str:
        """Watch, improve and build objectives are handed to what already does them."""
        s = self.sys
        kind = q.get("delegate")
        try:
            if kind == "watch":
                task = s.tasks.add(f"every {_every(self.pol.watch_every_seconds)}: {q['ask']}", by=f"objective:{obj.id}")
                q.update(status="answered", answer=f"handed to background task {task.id}, which asks it "
                                                   f"every {_every(task.interval)} and tells you when it changes",
                         source=f"task:{task.id}", confidence=1.0, confirmed=True, checked_by="delegated")
                return f"handed to {task.id}: {task.describe()}"
            if kind == "improve":
                skill = next((c for c in s.capabilities if re.search(rf"\b{re.escape(c)}\b", q["ask"], re.I)), None)
                if skill is None and re.search(r"\brout(?:er|ing)\b", q["ask"], re.I):
                    skill = "dispatch"
                if skill is None:
                    q["status"] = "unknown"
                    return f"no skill named in '{q['ask']}' (skills: {', '.join(sorted(s.capabilities))})"
                num = re.search(r"(\d+(?:\.\d+)?)\s*(%?)", q["ask"])
                to = ""
                if num:
                    v = float(num.group(1))
                    to = f" to {v / 100 if num.group(2) or v > 1 else v:g}"
                task = s.tasks.add(f"improve {skill}{to}", by=f"objective:{obj.id}")
                q.update(status="waiting-task", answer=f"handed to background task {task.id}: {task.describe()}",
                         source=f"task:{task.id}", task=task.id, confidence=0.0)
                return f"handed to {task.id}: {task.describe()}"
            if kind == "build":
                made = await s.toolsmith.build([], f"objective {obj.id}: {obj.text}", by, asked=q["ask"])
                if made is None:
                    q["status"] = "unknown"
                    last = s.toolsmith.history[-1]["line"] if s.toolsmith.history else "no recipe fits"
                    return f"the toolsmith could not write it: {last}"
                q.update(status="waiting-tool", answer=f"its toolsmith wrote {made['name']} v{made['version']}; "
                                                       f"waiting for your approval ({made['approval']})",
                         source=f"toolsmith:{made['name']}", tool=made["name"])
                return f"the toolsmith wrote {made['name']}; {made['approval']} asks you to approve it"
        except ValueError as exc:
            q["status"] = "unknown"
            return f"could not hand it on: {exc}"
        q["status"] = "unknown"
        return f"nothing to hand '{q['ask']}' to"

    def _poll_delegated(self, obj: Objective) -> None:
        s = self.sys
        for q in obj.questions:
            if q["status"] == "waiting-task":
                task = s.tasks.get(q.get("task", ""))
                if task is None:
                    q["status"] = "unknown"
                elif task.met:
                    q.update(status="answered", answer=f"{task.describe()}: {task.note}", confidence=1.0, confirmed=True,
                             checked_by=f"{task.id} met its target")
            elif q["status"] == "waiting-tool":
                d = s.toolsmith.drafts.get(q.get("tool", ""))
                if d is None or d["status"] == "declined":
                    q["status"] = "unknown"
                elif d["status"] in ("active", "proven"):
                    q.update(status="answered", answer=f"{d['name']} v{d['version']} is installed and {d['status']}",
                             confidence=1.0, confirmed=True, checked_by="the toolsmith's experiments" if d["status"] == "proven"
                             else "its self-tests")

    # ---------------------------------------------------------------- ruminating
    async def ruminate(self, obj: Objective, by: str = "ruminator") -> str:
        """Think an objective over: check answers against a second source, try what failed another way,
        follow up what the answers raise, then end it, or plan the next round."""
        s, pol = self.sys, self.pol
        obj.rounds += 1
        self._count("rounds")
        added: list[dict] = []
        notes: list[str] = []
        self._poll_delegated(obj)
        # 1. the same answer to different questions: a source that is not really answering them
        notes += self._echoes(obj)
        # 2. answers not yet checked: does a second source agree?
        checks = 0
        for q in obj.questions:
            if checks >= pol.checks_per_round or self.budget_left() <= 0:
                break
            if q.get("role") in ("alt", "retry", "settle"):
                continue  # it stands in for another question, which carries its answer
            if q["kind"] == "ask" and q["status"] == "answered" and q["confirmed"] is None and pol.confirm:
                checks += 1
                notes.append(await self._check(obj, q))
        # 3. contradictions to settle
        for q in obj.questions:
            pending = any(x.get("parent") == q["id"] and x["status"] in ("todo", "waiting") for x in obj.questions)
            if q["kind"] == "ask" and q["status"] == "answered" and q["confirmed"] is False and not pending:
                step = self._settle(obj, q)
                if step is not None:
                    added.append(step)
        # 4. what failed: other words, research, or a person. A way that did not work hands the question back.
        by_id = {q["id"]: q for q in obj.questions}
        for q in obj.questions:
            parent = by_id.get(q.get("parent", ""))
            if q.get("role") in ("alt", "retry", "settle") and parent is not None and parent["status"] == "retried":
                if q["status"] in ("failed", "unknown", "skipped"):
                    if q["status"] == "failed":
                        q["status"] = "unknown"
                    parent["status"] = "failed" if parent["confirmed"] is not False else "answered"
        for q in obj.questions:
            if q["kind"] == "ask" and q["status"] == "failed":
                if q.get("role") in ("alt", "retry", "settle"):
                    q["status"] = "unknown"  # the question it stood in for carries on
                    continue
                step = self._retry(obj, q)
                if step is not None:
                    added.append(step)
        # 5. what the answers raise
        if obj.rounds <= pol.max_rounds:
            added += self._follow_ups(obj)
        for a in added:
            verb = {"ask": "ask itself", "research": "read up on", "person": "ask you"}.get(a["kind"], a["kind"])
            self._think(obj, "next", f"Next: {verb} '{a['ask']}' ({a['why']})")
        if added:
            line = f"round {obj.rounds}: {len(added)} more step{'s' if len(added) != 1 else ''}" + (f"; {'; '.join(notes)}" if notes else "")
        elif any(q["status"] in ("waiting", "waiting-task", "waiting-tool") for q in obj.questions):
            line = f"round {obj.rounds}: waiting" + (f"; {'; '.join(notes)}" if notes else "")
            obj.note = "waiting for " + ", ".join(sorted({q["status"].replace("waiting-", "its ").replace("waiting", "you")
                                                         for q in obj.questions if q["status"].startswith("waiting")}))
            obj.next_at = time.time() + 30
        elif pol.confirm and any(q["kind"] == "ask" and q["status"] == "answered" and q["confirmed"] is None
                                 and q.get("role") not in ("alt", "retry", "settle") for q in obj.questions) \
                and obj.rounds <= pol.max_rounds + 2:
            line = f"round {obj.rounds}: more answers to check" + (f"; {'; '.join(notes)}" if notes else "")
        else:
            line = await self.finish(obj, by)
        s.bus.publish("objective.ruminated", id=obj.id, round=obj.rounds, added=len(added), line=line[:300])
        return line

    def _echoes(self, obj: Objective) -> list[str]:
        """Different questions with the same answer: it keeps the answer where it gives what the question
        asks for (a time for "when", a name for "who"), or else only for the question it fits best."""
        groups: dict[str, list[dict]] = {}
        for q in obj.questions:
            if q["kind"] == "ask" and q["status"] == "answered" and q.get("role") not in ("alt", "retry", "settle") \
                    and q["source"] != "person" and not q["source"].startswith("tool:"):
                groups.setdefault(" ".join(_answer_words(q["answer"])), []).append(q)
        notes = []
        for qs in groups.values():
            if len(qs) < 2:
                continue
            keep = [q for q in qs if specific(q["ask"]) and fits(q["ask"], q["answer"])]
            if not keep:
                words = set(_answer_words(qs[0]["answer"]))
                keep = [max(qs, key=lambda q: len(set(_answer_words(q["ask"])) & words))]
            for q in qs:
                if q in keep:
                    continue
                q["status"], q["echo"] = "failed", keep[0]["id"]
                q["tried"].append(f"{q['source']} (the same answer as to '{keep[0]['ask']}')"[:160])
                self._think(obj, "doubt", f"'{q['ask']}' got the same answer as '{keep[0]['ask']}' "
                                          f"('{first_sentence(q['answer'], 80)}'); that does not answer it, so it looks again.")
                notes.append(f"{q['id']} echoed {keep[0]['id']}")
        return notes

    async def _check(self, obj: Objective, q: dict) -> str:
        """Ask a second source (memory nodes, retained knowledge, a teacher), not the one that answered."""
        s = self.sys
        tried = []
        others = [m for m in ("memory", "knowledge", "teacher") if m != q["source"]]
        if q["source"] == "teacher" and s.knowledge_tool() is not None:
            others.append("research")
        for method in others:
            if method == "teacher" and not s.teachers.has_llm():
                continue
            self.recent.append(time.time())
            obj.steps += 1
            order = Order(text=q["ask"], source=f"objective:{obj.id}", background=True)
            res = await s.orchestrator.attempt(order, method)
            ok, _ = s.directors.answered(order, res)
            tried.append(method)
            if not ok:
                continue
            other = str(res.output)
            if agrees(q["answer"], other):
                q["confirmed"], q["checked_by"] = True, f"{method} agrees"
                self._think(obj, "confirm", f"'{q['ask']}': {method} agrees with {q['source']}.")
                return f"{q['id']} confirmed by {method}"
            q["confirmed"], q["checked_by"] = False, method
            q["conflict"] = first_sentence(other, 300)
            self._think(obj, "contradiction", f"'{q['ask']}': {q['source']} says '{first_sentence(q['answer'], 140)}' "
                                              f"but {method} says '{first_sentence(other, 140)}'.")
            self._count("contradictions")
            s.bus.publish("objective.contradiction", id=obj.id, question=q["id"], ask=q["ask"][:160],
                          one=first_sentence(q["answer"], 120), other=first_sentence(other, 120), by=method)
            return f"{q['id']} contradicted by {method}"
        q["confirmed"], q["checked_by"] = "single", "no second source"
        self._think(obj, "gap", f"'{q['ask']}': only {q['source']} knows; no second source to check it against"
                                + (f" (tried {', '.join(tried)})" if tried else "") + ".")
        return f"{q['id']} has one source"

    def _settle(self, obj: Objective, q: dict) -> dict | None:
        """Two sources disagree: read up on it and ask again if it can, else ask a person."""
        s = self.sys
        topic = _object_of(q["ask"]) or obj.subject
        if s.knowledge_tool() is not None and not q.get("researched"):
            q["researched"] = True
            step = self._add_question(obj, topic, "research", f"to settle '{q['ask']}'", parent=q["id"], role="settle")
            if step is not None:
                q.update(status="retried", after_research=False)
                return step
        if self.pol.ask_people and not q.get("asked"):
            q["asked"] = True
            return self._add_question(obj, q["ask"], "person", "two sources disagree", parent=q["id"], role="settle")
        return None

    def _retry(self, obj: Objective, q: dict) -> dict | None:
        """Something failed: other words first, then reading up on it, then (for what the objective needs)
        asking a person; then it is an open question."""
        s = self.sys
        if not q.get("reworded"):
            q["reworded"] = True
            alt = reword(q["ask"])
            if alt:
                step = self._add_question(obj, alt, "ask", f"'{q['ask']}' in other words", parent=q["id"], role="alt")
                if step is not None:
                    q["status"] = "retried"
                    return step
        if q.get("role") == "follow":  # a follow-up is worth one more try, not a person's time
            q["status"] = "unknown"
            return None
        topic = _object_of(q["ask"]) or obj.subject
        if not q.get("researched") and s.knowledge_tool() is not None:
            q["researched"] = True
            step = self._add_question(obj, topic, "research", f"to find out '{q['ask']}'", parent=q["id"], role="retry")
            if step is not None:
                q["status"] = "retried"
                return step
        if not q.get("asked") and self.pol.ask_people and q.get("core"):
            if any(x["kind"] == "person" and x["status"] in ("todo", "waiting") for x in obj.questions):
                return None  # one question for a person at a time: this one waits its turn
            q["asked"] = True
            step = self._add_question(obj, q["ask"], "person", "nothing it has could answer it", parent=q["id"], role="retry")
            if step is not None:
                q["status"] = "retried"
                return step
        q["status"] = "unknown"
        self._think(obj, "gap", f"Could not find out '{q['ask']}' (tried: {', '.join(t.split(' (')[0] for t in q['tried']) or 'nothing'}).")
        return None

    def _follow_ups(self, obj: Objective) -> list[dict]:
        pol, out = self.pol, []
        if obj.strategy == "survey":
            for q in obj.questions:
                if q["from"] == "plan" and q["kind"] == "ask" and q["status"] == "answered" and not q.get("followed"):
                    q["followed"] = True
                    for item in list_items(q["answer"], pol.max_questions):
                        step = self._add_question(obj, f"What is {item}?", "ask", f"one of {obj.subject}", core=True,
                                                  parent=q["id"], role="item")
                        if step is not None:
                            out.append(step)
        elif obj.strategy == "explain" and pol.follow_ups:
            known = " ".join([obj.text] + [q["ask"] for q in obj.questions])
            for q in obj.questions:
                if len(out) >= pol.follow_ups:
                    break
                if q["from"] == "plan" and q["kind"] == "ask" and q["status"] == "answered" and not q.get("followed"):
                    q["followed"] = True
                    for term in new_terms(q["answer"], known, pol.follow_ups - len(out)):
                        ask = f"Who was {term}?" if looks_like_a_person(term) else f"What {_be(term)} {term}?"
                        step = self._add_question(obj, ask, "ask", f"'{q['ask']}' mentions {term}", parent=q["id"], role="follow")
                        if step is not None:
                            out.append(step)
                            known += " " + term
        return out

    async def finish(self, obj: Objective, by: str) -> str:
        s = self.sys
        done, total = obj.progress()
        open_conflicts = [q for q in obj.questions if q["kind"] == "ask" and q["confirmed"] is False]
        met = total > 0 and done / total >= self.pol.met_share and not open_conflicts
        obj.summary, obj.summary_by = await self._summary(obj)
        obj.status = "met" if met else "unmet"
        obj.ended_at = time.time()
        missing = [q["ask"] for q in obj.core() if q["status"] != "answered"]
        if met:
            obj.note = f"met: {done}/{total} core questions answered after {obj.steps} steps and {obj.rounds} rounds of thought"
        else:
            obj.note = (f"ended: found {done} of {total}" + (f"; could not find out: {'; '.join(missing[:4])}" if missing else "")
                        + (f"; {len(open_conflicts)} answers its sources disagree on" if open_conflicts else ""))
        self._think(obj, "conclusion", obj.note)
        self._count("met" if met else "unmet")
        report = self.report_path(obj)
        if s.mind.enabled:
            s.mind.note("finding", f"objective:{obj.id}", f"{obj.text}: {obj.note}. {obj.summary[:300]} (report: {report})",
                        private=False, source=f"objective {obj.id}")
        s.guardian.audit(by, f"objective.{obj.status}", obj.id, text=obj.text, steps=obj.steps, rounds=obj.rounds,
                         answered=done, core=total)
        s.bus.publish("objective.ended", id=obj.id, text=obj.text[:200], status=obj.status, note=obj.note[:300],
                      steps=obj.steps, rounds=obj.rounds, report=str(report))
        return obj.note

    async def _summary(self, obj: Objective) -> tuple[str, str]:
        """What it found, in a paragraph: its own answers, first sentences (an LLM teacher may write it)."""
        s = self.sys
        found = [q for q in obj.questions if q["kind"] == "ask" and q["status"] == "answered"]
        if not found:
            return "It found no answers.", "the mesh"
        if obj.strategy in ("watch", "improve", "build"):
            return " ".join(first_sentence(q["answer"]) for q in found), "the mesh"
        if self.pol.llm_summary and s.teachers.has_llm():
            facts = "\n".join(f"- {q['ask']} {first_sentence(q['answer'], 300)}" for q in found[:12])
            reply = await s.teachers.answer(f"In one short paragraph, using only these findings, sum up what is known "
                                            f"about: {obj.text}\n{facts}")
            if reply:
                return " ".join(str(reply[0]).split())[:1500], f"the {reply[1]} teacher, from the findings below"
        parts = []
        for q in found:
            sent = first_sentence(q["answer"])
            if sent and sent not in parts:
                parts.append(sent if sent.endswith((".", "!", "?", "…")) else sent + ".")
        return " ".join(parts)[:1500], "the mesh, from its own answers"

    # ---------------------------------------------------------------- people answer
    def on_event(self, ev) -> None:
        if ev.topic == "inquiry.answered" and ev.data.get("kind") == "objective" and ev.data.get("status") == "dropped":
            for obj in self.items.values():
                for q in obj.questions:
                    if q.get("inquiry") == ev.data.get("id") and q["status"] == "waiting":
                        q["status"] = "unknown"
                        self._think(obj, "gap", f"You did not know '{q['ask']}' either; it stays an open question.")
                        obj.next_at = time.time()
                        self.save(obj)
                        self.write_report(obj)

    def reply(self, about: str, answer: str, inquiry_id: str, by: str) -> tuple[bool, str]:
        """A person answered a question an objective asked."""
        oid, _, qid = about.partition(":")
        obj = self.items.get(oid)
        q = next((x for x in obj.questions if x["id"] == qid), None) if obj else None
        if obj is None or q is None:
            return False, "that objective is gone"
        self._answer(obj, q, answer, "person", 1.0, [f"you ({inquiry_id})"])
        self._think(obj, "answer", f"{by} told it: {q['ask']} -> {first_sentence(answer, 160)}")
        s = self.sys
        if s.memory.allows("people"):
            s.memory.learn(q["ask"], answer, f"you told me ({inquiry_id}, for {obj.id})", by=by, kind="people")
        obj.next_at = time.time()
        self.save(obj)
        self.write_report(obj)
        return True, f"thanks - {obj.id} carries on with that"

    # ---------------------------------------------------------------- second thoughts
    def doubts(self, limit: int = 20) -> list[dict]:
        """Recent questions it answered unsurely, or could not answer: worth thinking over again."""
        s, pol = self.sys, self.pol
        if not self.enabled:
            return []
        out = []
        now = time.time()
        done = set(self.reconsidered)
        for oid, ts, text, res in s.store.recent_orders(80):
            if oid in done or now - ts < pol.second_thought_after_seconds or not is_question(text):
                continue
            route = res.get("route") or ""
            if route not in (DIRECT, "") and not route.startswith("tool:"):
                continue
            unsure = res.get("ok") and float(res.get("confidence") or 0) < pol.doubt_confidence
            disputed = "disagree" in str(res.get("review", ""))
            failed = not res.get("ok") and route == DIRECT
            if unsure or disputed or failed:
                out.append({"order": oid, "ts": ts, "text": text, "ok": bool(res.get("ok")), "was": str(res.get("output") or res.get("note") or ""),
                            "confidence": float(res.get("confidence") or 0), "why": "unsure" if unsure else "disputed" if disputed else "unanswered"})
            if len(out) >= limit:
                break
        return out

    async def second_thoughts(self, by: str = "ruminator", n: int | None = None) -> list[dict]:
        """Think again about recent answers it was unsure of or could not give: what would it say now?"""
        s = self.sys
        n = n or self.pol.second_thoughts
        results = []
        for d in self.doubts(n):
            if self.budget_left() <= 0:
                break
            self.recent.append(time.time())
            self.reconsidered.append(d["order"])
            order = Order(text=d["text"], source="ruminator", background=True)
            res = await s.orchestrator.fulfil(order)
            ok, _ = s.directors.answered(order, res)
            now_said = first_sentence(str(res.output), 200) if ok else ""
            if ok and not d["ok"]:
                verdict, text = "now answers", f"I couldn't answer '{d['text']}' before; now I'd say: {now_said}"
            elif ok and agrees(d["was"], res.output):
                verdict, text = "still thinks", f"I thought again about '{d['text']}' and still say: {now_said}"
            elif ok:
                verdict, text = "changed its mind", (f"On second thought, '{d['text']}': {now_said} "
                                                     f"(I said '{first_sentence(d['was'], 120)}' before)")
            else:
                verdict, text = "still cannot", f"I thought again about '{d['text']}' and still can't answer it"
            rec = {"ts": round(time.time(), 1), "order": d["order"], "text": d["text"][:200], "why": d["why"],
                   "verdict": verdict, "line": text[:400], "source": source_of(res, s) if ok else "", "by": by}
            self.second.append(rec)
            results.append(rec)
            self._count("second_thoughts")
            if s.mind.enabled and verdict in ("now answers", "changed its mind"):
                s.mind.note("update", f"order:{d['order']}", text, private=False, source="second thoughts")
            s.bus.publish("mind.second_thought", order=d["order"], text=d["text"][:160], verdict=verdict, line=text[:300], by=by)
        self.save()
        return results

    # ---------------------------------------------------------------- the report
    def report_path(self, obj: Objective) -> Path:
        return self.dir / f"{obj.id}.md"

    def write_report(self, obj: Objective) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.report_path(obj)
        path.write_text(self.report(obj), encoding="utf-8")
        return path

    def report(self, obj: Objective) -> str:
        """The long-form write-up: what it found, every question with its answer and where it came from,
        what it could not find out, and how it got there."""
        done, total = obj.progress()
        made = time.strftime("%Y-%m-%d %H:%M", time.localtime(obj.created_at))
        who = f"{obj.created_by} (its own initiative)" if obj.self_set else obj.created_by
        state = {"active": "in progress", "paused": "paused", "met": "met", "unmet": "ended without meeting it"}.get(obj.status, obj.status)
        lines = [f"# {obj.text[0].upper()}{obj.text[1:]}", "",
                 f"Objective {obj.id}, set by {who} on {made}. Strategy: {obj.strategy}, planned by {obj.planned_by}.",
                 f"Status: {state}. {done} of {total} core questions answered after {obj.steps} steps and "
                 f"{obj.rounds} round{'s' if obj.rounds != 1 else ''} of rumination.", ""]
        if obj.summary:
            lines += ["## What it found", "", obj.summary, "", f"_Written by {obj.summary_by}._", ""]
        by_id = {q["id"]: q for q in obj.questions}
        asks = [q for q in obj.questions if q["kind"] in ("ask", "person", "delegate") and q.get("role") not in ("alt", "retry", "settle")]
        if asks:
            lines += ["## Questions", ""]
        for q in asks:
            mark = "" if q.get("core") else " (follow-up)" if q["from"] == "rumination" else " (extra)"
            lines += [f"### {q['ask']}{mark}", ""]
            if q["status"] == "answered":
                lines += [str(q["answer"]).strip(), ""]
                check = ("confirmed: " + q["checked_by"] if q["confirmed"] is True else
                         f"**disputed**: {q['checked_by']} says \"{q['conflict']}\"" if q["confirmed"] is False else
                         "only one source; nothing else to check it against" if q["confirmed"] == "single" else "not checked yet")
                src = q["source"] + (f" ({'; '.join(q.get('sources') or [])})" if q.get("sources") else "")
                via = by_id.get(q.get("via", ""))
                how = (f" Answered when it asked {'you' if via['kind'] == 'person' else 'itself'} "
                       f"{'again' if via['kind'] == 'person' else repr(via['ask'])}." if via else "")
                if q.get("settled"):
                    how += f" Its sources disagreed; {q['settled']} settled it."
                lines += [f"_Source: {src}; confidence {q['confidence']:.0%}; {check}.{how}_", ""]
            elif q["status"] in ("waiting", "waiting-task", "waiting-tool"):
                lines += [f"Waiting: {q.get('answer') or ('asked you, ' + q['inquiry'])}.", ""]
            elif q["status"] == "todo":
                lines += ["Not asked yet.", ""]
            elif q["status"] == "retried":
                lines += [f"Not answered yet; trying another way. So far: {'; '.join(q['tried']) or '-'}.", ""]
            else:
                lines += [f"Not found. Tried: {'; '.join(q['tried']) or 'nothing it has could answer it'}.", ""]
        research = [q for q in obj.questions if q["kind"] == "research"]
        if research:
            lines += ["## Reading", ""] + [f"- {q['ask']}: {q.get('answer') or q.get('note') or q['status']}" for q in research] + [""]
        missing = [q["ask"] for q in asks if q["kind"] in ("ask", "person") and q["status"] in ("unknown", "failed")]
        if missing:
            lines += ["## Still open", ""] + [f"- {m}" for m in missing] + [""]
        if obj.thoughts:
            lines += ["## How it got here", ""]
            lines += [f"- {time.strftime('%H:%M:%S', time.localtime(t[0]))} {t[1]}: {t[2]}" for t in obj.thoughts[-60:]]
            lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------------------- status
    def _think(self, obj: Objective, kind: str, text: str) -> None:
        obj.thoughts.append([round(time.time(), 1), kind, text[:500]])
        del obj.thoughts[:-200]
        if kind in ("contradiction", "conclusion", "gap", "confirm"):
            self.sys.bus.publish("objective.thought", id=obj.id, kind=kind, text=text[:300])

    def status(self) -> dict:
        return {"enabled": self.enabled, "steps_left": self.budget_left(), "limit": self.pol.steps_per_hour,
                "counts": dict(self.counts), "doubts": len(self.doubts(10)),
                "objectives": [{**asdict(o), "progress": o.progress(), "report": str(self.report_path(o))}
                               for o in sorted(self.items.values(), key=lambda o: o.created_at)],
                "second_thoughts": list(self.second)[-12:]}

    def lines(self) -> list[str]:
        if not self.enabled:
            return ["  objectives are off ([objectives] enabled = false in the directives)"]
        out = [f"  {len(self.active())} active objectives; {self.budget_left()} of {self.pol.steps_per_hour} steps left this hour"]
        if not self.items:
            out.append("  none yet: /objective understand the Berlin Wall   (or as an order: objective: ...)")
        for o in sorted(self.items.values(), key=lambda o: o.created_at):
            done, total = o.progress()
            out.append(f"  {o.id} [{o.status}] {o.text} - {o.strategy}, {done}/{total} answered, {o.steps} steps, "
                       f"{o.rounds} rounds" + (" (its own)" if o.self_set else ""))
            if o.note:
                out.append(f"      {o.note[:160]}")
        recent = list(self.second)[-3:]
        if recent:
            out.append("  second thoughts:")
            out += [f"    {r['verdict']}: {r['line'][:150]}" for r in reversed(recent)]
        return out

    def show(self, oid: str) -> list[str]:
        o = self.items.get(oid)
        if o is None:
            return [f"  no objective {oid} (/objectives lists them)"]
        done, total = o.progress()
        out = [f"  {o.id} [{o.status}] {o.text}", f"  strategy {o.strategy} ({o.planned_by}); {done}/{total} core answered; "
                                                  f"{o.steps} steps, {o.rounds} rounds; report: {self.report_path(o)}"]
        if o.note:
            out.append(f"  {o.note}")
        for q in o.questions:
            tag = {"answered": "ok ", "unknown": "?? ", "failed": "no ", "todo": "...", "skipped": "-- "}.get(q["status"], q["status"][:3])
            conf = ("" if q["status"] != "answered" else " confirmed" if q["confirmed"] is True else " DISPUTED" if q["confirmed"] is False
                    else " one source" if q["confirmed"] == "single" else "")
            out.append(f"    {tag} {q['id']} [{q['kind']}] {q['ask'][:70]}" + (f" -> {first_sentence(q['answer'] or '', 70)}{conf}"
                                                                               if q["status"] == "answered" else ""))
        for t in o.thoughts[-5:]:
            out.append(f"    thought ({t[1]}): {t[2][:140]}")
        return out


def _every(seconds: float) -> str:
    from .background import every
    return every(seconds)


# ======================================================================== the agent
class Ruminator(Agent):
    """Works on objectives one step at a time, only on background capacity (P2) and within the
    objectives' budget (P4); never while asleep, halted or paused (P1)."""
    name = "ruminator"

    @property
    def interval(self) -> float:
        return max(0.05, self.sys.directives.background.tick_seconds * 2)

    async def step(self) -> None:
        s = self.sys
        book = s.objectives
        if not book.enabled or s.guardian.halted or s.guardian.paused or s.dreams.asleep:
            return
        if s.inflight or not s.guardian.background_allowed():
            return
        obj = book.due()
        if obj is None:
            return
        line = await book.work(obj)
        if line:
            self.stats["steps"] += 1
