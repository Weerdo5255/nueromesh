"""Thinking nodes: after the memory nodes, the mesh puts what it knows into its own words, and holds
its opinions and desires.

    memory nodes  --> facts: the knowledge context (each a question and its answer, checked by its test file)
        |
    thinking nodes --> thoughts, one per topic, in the mesh's own words:
        understanding  the facts it holds about a topic, put together and rephrased ("Australia's capital is
                       Canberra"), by its own rules or drafted by an LLM teacher - and either way checked
                       word by word against those facts: a thought adds no fact it does not hold (P17)
        opinion        what it makes of the topic, from evidence it can name: how what it read and holds
                       reads (the sentiment skill), whether its sources agree or it was corrected, how
                       often it comes up. Opinions about people stay private (P9)
        desire         what it wants to know: a question about the topic it could not answer, a name in
                       its facts it knows nothing more about, more about a topic people keep asking about
                       (curiosity.py). Desires do nothing on their own: the director may take one up as an
                       objective it sets itself (agency mode pursue), within its budgets (P12, P16)

A thinking node is a small neural network like a memory node: it maps a topic, the questions about
it and its facts to the thought it holds about them. Each node has tests (the topic and its facts must
bring up its thought; small talk, tools and other topics must not) and is retrained whenever its
thoughts' topics or facts change. state/thinking/<node>.md says what each node holds and why.

When an answer comes from what it knows (a memory node or what it read), the thinking node that
fires on the question and its answer adds its thought after the knowledge, labelled as its own
thinking (Result.thought); the answer's own words are never changed (P8, P14). "What do you think of
X?", "in your own words, ..." and "what do you want to find out?" are answered from the thinking nodes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .agents import Agent
from .curiosity import STOP_CAPS, norm, question_core, topics_in
from .memory import FILLERS, PROBES, Job, encode_many, remap_outputs, softmax, forward, fold, subject as identity_subject

NONE = "(no thought about this)"   # output 0 of every thinking node
PERSON_SUBJECTS = {"me", "myself", "the user", "you", "yourself", "us"}

THINK_RE = re.compile(
    r"^\s*(?:so,?\s+|and\s+|hey,?\s+|honestly,?\s+)?(?:"
    r"what do you (?:really\s+)?(?:think|feel|make) (?:of|about)|what(?:'s| is) your (?:honest\s+)?(?:opinion|view|take|impression)"
    r" (?:of|on|about)|how do you feel about|what are your (?:thoughts|views) (?:on|about)|your (?:opinion|view|thoughts) (?:of|on|about)|"
    r"in your own words,?\s*(?:what (?:is|are|was|were)|tell me about|explain|describe|what do you know about)?|"
    r"what have you (?:learned|learnt) about|(?:sum up|summari[sz]e) what you know about|"
    r"what do you know about|put (?:it|what you know about)?\s*in your own words:?)\s+(?P<subject>.+?)[\s?.!]*$", re.I)
DESIRE_RE = re.compile(
    r"\b(?:what (?:do|would) you (?:want|like|wish|hope) to (?:know|learn|find out|understand|read)|"
    r"what are you curious about|what (?:are you|do you find yourself) (?:interested in|curious about)|"
    r"what interests you|what do you (?:wonder|want to know) about|what (?:are|is) your (?:desires?|wishes|interests)|"
    r"what do you want|what are you (?:wondering|trying to find out)|what would you like to (?:know|learn))\b", re.I)

# words a thought may use that its facts need not contain: the frame around them, never content
FRAME = set("""
as i understand it putting together what hold about and more things is was are were by the a an its it of my in own
words that this these those have has had be been being which who whom whose also so then than or but not no nor
to for from on at with into s also only one two three four several many some all both each same other""".split())
INFLECT = {"written": "wrote", "drawn": "drew", "painted": "painted", "built": "built", "founded": "founded",
           "discovered": "discovered", "invented": "invented", "composed": "composed", "designed": "designed",
           "directed": "directed", "sung": "sang", "made": "made", "led": "led", "won": "won", "named": "named",
           "created": "created", "developed": "developed", "published": "published", "sculpted": "sculpted",
           "proposed": "proposed", "formulated": "formulated", "recorded": "recorded", "produced": "produced"}
_PASSIVE = re.compile(r"^(?P<agent>.+?)\s+(?P<verb>" + "|".join(sorted({v for v in INFLECT.values()} | {"wrote", "drew", "sang"},
                                                                          key=len, reverse=True)) + r")\s+(?P<rest>.+)$")
_PARTICIPLE = {**{v: k for k, v in INFLECT.items()}, "wrote": "written", "drew": "drawn", "sang": "sung"}
_THE_OF = re.compile(r"^the\s+(?P<rel>[a-z][a-z \-]{1,40}?)\s+(?:of|in)\s+(?P<t>.+?)\s+(?P<be>is|was|are|were)\s+(?P<x>.+)$", re.I)
_OF = re.compile(r"^(?P<x>.+?)\s+(?P<be>is|was|are|were)\s+the\s+(?P<rel>[a-z][a-z \-]{1,40}?)\s+(?:of|in)\s+(?P<t>.+?)(?P<rest>[,;].*)?$")
_TOKEN = re.compile(r"[a-z]+|\d[\d,.]*\d|\d")


def _tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(re.sub(r"['’]s\b", "", str(text).lower())):
        out.append(t.replace(",", "") if t[0].isdigit() else t)
    return out


def faithful(text: str, grounds: list[str]) -> list[str]:
    """The honesty guard for thoughts (P17): the words a thought adds that none of its facts contain
    (beyond its own frame). An empty list means it rests on its facts alone."""
    held = set()
    for g in grounds:
        held.update(_tokens(g))
    extra = []
    for t in _tokens(text):
        if t in held or t in FRAME or (t in INFLECT and INFLECT[t] in held) or (t in _PARTICIPLE and _PARTICIPLE[t] in held):
            continue
        if t.endswith("s") and t[:-1] in held:
            continue
        if t not in extra:
            extra.append(t)
    return extra


def _first_sentence(text: str, limit: int = 240) -> str:
    t = " ".join(str(text or "").split())
    t = re.sub(r"\[\d+\]|\([^)]{0,80}\)", "", t).strip()
    m = re.match(r"(.+?[.!?])(?:\s|$)", t)
    s = (m.group(1) if m else t).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0]


def claim(ask: str, answer: str) -> str:
    """A fact as one statement (no full stop): the answer's first sentence, or the question's subject
    with its answer ('the capital of Australia is Canberra')."""
    a = _first_sentence(answer).rstrip(" .!?")
    if len(a.split()) >= 4:
        return a
    sub = identity_subject(ask)
    if sub:
        sub = sub if sub[:1].isupper() else f"the {sub}"
        return f"{sub} is {a}"
    return f"{a} ({question_core(ask)})"


def rephrase(c: str, topic: str) -> str:
    """Put a statement the mesh's own way round, topic first, without changing what it says."""
    t = topic.lower()
    m = _OF.match(c)
    if m and norm(m.group("t")) == norm(topic) and not norm(m.group("x")).startswith(norm(topic)):
        name = m.group("t").strip()
        name = re.sub(r"^the\s+", "", name, flags=re.I)
        return f"{name}'s {m.group('rel')} {m.group('be')} {m.group('x')}{m.group('rest') or ''}"
    m = _THE_OF.match(c)
    if m and norm(m.group("t")) == norm(topic):
        name = re.sub(r"^the\s+", "", m.group("t").strip(), flags=re.I)
        return f"{name}'s {m.group('rel')} {m.group('be')} {m.group('x')}"
    m = _PASSIVE.match(c)
    if m and norm(m.group("rest")).startswith(norm(topic)) and t not in m.group("agent").lower():
        rest = m.group("rest")
        obj, tail = (rest.split(",", 1) + [""])[:2]
        be = "was"
        return f"{obj.strip()} {be} {_PARTICIPLE[m.group('verb')]} by {m.group('agent')}" + (f",{tail}" if tail else "")
    return c


def _lower_article(c: str) -> str:
    return re.sub(r"^(The|A|An)\b", lambda m: m.group(1).lower(), c)


def synthesize(topic: str, claims: list[str], max_claims: int = 4) -> str:
    """Several facts about a topic put together in its own words (rule-based, adds nothing)."""
    said = []
    for c in claims:
        r = rephrase(c, topic)
        if r and all(norm(r) != norm(x) for x in said):
            said.append(r)
    if not said:
        return ""
    # facts that say the same of the topic, one after another: "X was A and B", "X wrote A (1813) and B (1815)"
    bare = re.sub(r"^the\s+", "", topic, flags=re.I)
    head = re.compile(rf"^(?:the\s+)?{re.escape(bare)}\s+([a-z]+)\s+(.+)$", re.I)
    preds: dict[str, list] = {}
    subj: dict[str, str] = {}
    order: list = []
    for c in said:
        m = head.match(c)
        if m:
            verb = m.group(1).lower()
            if verb not in preds:
                preds[verb] = []
                subj[verb] = c[: m.start(1)].strip()
                order.append(("verb", verb))
            preds[verb].append(m.group(2))
        else:
            order.append(("claim", c))

    def joined(ps: list) -> str:
        if len(ps) > 1:  # "Pride and Prejudice, published in 1813" -> "Pride and Prejudice (published in 1813)"
            ps = [re.sub(r"^(.+?),\s+(.+)$", r"\1 (\2)", x) for x in ps]
        return " and ".join(ps) if len(ps) <= 2 else ", ".join(ps[:-1]) + " and " + ps[-1]
    merged = [f"{subj[x]} {x} {joined(preds[x])}" if kind == "verb" else x for kind, x in order]
    shown, more = merged[:max_claims], len(merged) - max_claims
    if len(shown) == 1 and more <= 0:
        return f"As I understand it, {_lower_article(shown[0])}."
    body = "; ".join(_lower_article(c) for c in shown[:-1]) + "; and " + _lower_article(shown[-1])
    tail = f" I hold {more} more thing{'s' if more != 1 else ''} about it." if more > 0 else ""
    return f"Putting together what I hold about {topic}: {body}.{tail}"


VIEWS = {"favourable": "I'm favourable towards it", "unfavourable": "I'm not favourable towards it",
         "interested": "I find it interesting", "unsure of what I know": "I'm unsure of what I know about it"}


def view_words(stance: str) -> str:
    return VIEWS.get(stance, f"I find it {stance}")


def display_name(name: str, texts: list[str]) -> str:
    """'Berlin Wall' -> 'the Berlin Wall' when that is how its facts say it."""
    joined = " ".join(texts)
    with_the = len(re.findall(rf"\bthe\s+{re.escape(name)}\b", joined, re.I))
    bare = len(re.findall(rf"\b{re.escape(name)}\b", joined, re.I)) - with_the
    return f"the {name}" if with_the and with_the >= bare and not name.lower().startswith("the ") else name


def looks_like_person(name: str) -> bool:
    words = name.split()
    return (2 <= len(words) <= 4 and all(w[:1].isupper() or w.lower() in ("da", "de", "van", "von", "di", "la", "le", "bin")
                                         for w in words) and words[0].lower() not in ("the", "a", "an")
            and not any(w.lower() in ("wall", "war", "empire", "river", "mountain", "sea", "ocean", "lake", "system", "city",
                                      "university", "bank", "company", "program", "programme", "cycle", "lisa") for w in words)
            and not re.search(r"\band\b|\bof\b", name))


@dataclass
class Thought:
    id: str
    topic: str
    key: str
    understanding: str = ""
    grounds: list = field(default_factory=list)   # [{"ask", "answer", "node", "source", "primary"}]
    opinion: dict | None = None                   # {"stance", "strength", "why", "private"}
    desire: dict | None = None                    # {"want", "kind", "strength", "question"|"area", "status", "objective"}
    formed_by: str = ""
    rejected: str = ""                            # an LLM draft the honesty guard turned down, and why
    node: str = ""
    sig: str = ""
    cue_sig: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    served: int = 0

    def text(self, shareable: bool = True) -> str:
        """The thought as a person reads it: understanding, then opinion, then desire."""
        parts = [self.understanding]
        if self.opinion:
            if self.opinion.get("private") or not shareable:
                parts.append("I have a view of this too, but I keep it to myself.")
            else:
                parts.append(f"My view: {view_words(self.opinion['stance'])} ({self.opinion['why']}).")
        if self.desire and self.desire.get("status") != "satisfied":
            want = self.desire["want"]
            if self.desire.get("objective") and self.desire.get("status") == "pursuing":
                want += f" (I am working on it: {self.desire['objective']})"
            parts.append(f"What I'd like: {want}.")
        return " ".join(p for p in parts if p)


@dataclass
class ThinkingNode:
    id: str
    input_dim: int
    inputs: int = 0
    bits: int = 32
    params: list | None = None
    outputs: list = field(default_factory=lambda: [NONE])   # thought ids, in output order
    version: int = 1
    checked_version: int = 0
    status: str = "learning"         # learning | ready | failing
    passed: int = 0
    total: int = 0
    failing: list = field(default_factory=list)
    served: int = 0
    created: float = field(default_factory=time.time)

    @property
    def dirty(self) -> bool:
        return self.version != self.checked_version

    @property
    def hidden(self) -> tuple:
        return tuple(int(self.params[2 * i].shape[1]) for i in range(len(self.params) // 2 - 1)) if self.params else ()

    @property
    def weights(self) -> int:
        return int(sum(p.size for p in self.params[::2])) if self.params else 0

    def spec(self) -> str:
        if not self.params:
            return "not learned yet"
        return f"{self.inputs} inputs, hidden {'x'.join(map(str, self.hidden)) or 'none'}, {self.bits}-bit"

    def predict(self, X: np.ndarray):
        P = softmax(forward(self.params, fold(X, self.inputs))[0])
        top = P.argmax(axis=1)
        return top, P[np.arange(len(top)), top]


class ThinkingBank:
    """The thinking nodes and the thoughts they hold (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.thinking
        self.dim = system.directives.nodes.input_dim
        self.dir = system.state_dir / "thinking"
        self.thoughts: dict[str, Thought] = {}
        self.nodes: dict[str, ThinkingNode] = {}
        self._sentiment: dict[str, str] = {}
        self.stats = {"formed": 0, "reformed": 0, "dropped": 0, "rejected": 0, "expressed": 0, "answered": 0, "learned": 0}
        self.history: list[dict] = []
        self._next_form = 0.0
        self._lock = asyncio.Lock()  # one round of thinking at a time
        raw = system.store.get_kv("thinking.state")
        st = json.loads(raw) if raw else {}
        names = {f for f in Thought.__dataclass_fields__}
        for d in st.get("thoughts", []):
            th = Thought(**{k: v for k, v in d.items() if k in names})
            self.thoughts[th.id] = th
        self.stats.update(st.get("stats", {}))
        self.history = st.get("history", [])[-50:]
        self._ids = max([int(t.split("-")[-1]) for t in self.thoughts if t.split("-")[-1].isdigit()] or [0]) + 1
        for meta in st.get("nodes", []):
            node = self._load(meta)
            if node is not None:
                self.nodes[node.id] = node

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled) and self.sys.memory.enabled

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        nodes = [{"id": n.id, "input_dim": n.input_dim, "inputs": n.inputs, "bits": n.bits, "outputs": n.outputs,
                  "version": n.version, "checked_version": n.checked_version, "status": n.status, "passed": n.passed,
                  "total": n.total, "failing": n.failing, "served": n.served, "created": n.created}
                 for n in self.nodes.values()]
        self.sys.store.set_kv("thinking.state", json.dumps({
            "thoughts": [asdict(t) for t in self.thoughts.values()], "nodes": nodes,
            "stats": self.stats, "history": self.history[-50:]}, default=str))

    def _weights_path(self, nid: str) -> Path:
        return self.dir / f"{nid}.npz"

    def _load(self, meta: dict) -> ThinkingNode | None:
        node = ThinkingNode(meta["id"], meta.get("input_dim", self.dim), meta.get("inputs", 0), meta.get("bits", 32), None,
                            meta.get("outputs", [NONE]), meta.get("version", 1), 0, meta.get("status", "learning"),
                            meta.get("passed", 0), meta.get("total", 0), meta.get("failing", []), meta.get("served", 0),
                            meta.get("created", time.time()))
        path = self._weights_path(node.id)
        if path.exists() and node.input_dim == self.dim:
            try:
                with np.load(path, allow_pickle=False) as z:
                    node.params = [z[f"p{i}"].astype(np.float32) for i in range(int(z["layers"]) * 2)]
                node.checked_version = meta.get("checked_version", node.version)
            except Exception:
                node.params = None
        return node

    def _save_weights(self, node: ThinkingNode) -> None:
        if node.params is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{node.id}.tmp.npz"
        np.savez(tmp, layers=np.array(len(node.params) // 2), **{f"p{i}": p for i, p in enumerate(node.params)})
        tmp.replace(self._weights_path(node.id))

    def report_path(self, node: ThinkingNode) -> Path:
        return self.dir / f"{node.id}.md"

    def write_report(self, node: ThinkingNode) -> None:
        """What a thinking node holds, for people to read (it is rewritten, not read back)."""
        lines = [f"# {node.id}: a thinking node", "",
                 "Written by the mesh: the thoughts this node holds, in its own words, each with the facts it rests on. "
                 "Its opinions and desires are its own (P9, P17). To have it drop a thought: /thought forget <id>.", "",
                 f"- shape: {node.spec()}", f"- status: {node.status}; tests passed {node.passed}/{node.total}", ""]
        for tid in node.outputs[1:]:
            th = self.thoughts.get(tid)
            if th is None:
                continue
            lines += [f"## {th.id}: {th.topic}", "", th.understanding, ""]
            if th.opinion:
                lines.append(f"- opinion: {th.opinion['stance']} ({th.opinion['why']})"
                             + (" - private, never shared with users" if th.opinion.get("private") else ""))
            if th.desire:
                lines.append(f"- desire: {th.desire['want']} [{th.desire.get('status', 'open')}"
                             + (f" {th.desire['objective']}" if th.desire.get("objective") else "") + "]")
            lines.append(f"- formed by: {th.formed_by}")
            lines += [f"- rests on: \"{g['ask']}\" -> \"{g['answer'][:160]}\" ({g['node']})" for g in th.grounds[:12]]
            lines.append("")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.report_path(node).write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------- forming
    def _facts(self) -> list[dict]:
        out = []
        for n in self.sys.memory.fact_nodes():
            for t in n.facts():
                if t.answer:
                    out.append({"ask": t.ask, "answer": t.answer, "node": n.id, "source": t.source or t.by or ""})
        return out

    async def _feeling(self, text: str) -> str:
        """How a fact reads to its sentiment skill (cached; learn=False: it is not training data)."""
        if text in self._sentiment:
            return self._sentiment[text]
        lab = ""
        if "sentiment" in self.sys.capabilities:
            try:
                inf = await self.sys.infer("sentiment", text, learn=False)
                lab = inf.label or ""
            except Exception:
                lab = ""
        self._sentiment[text] = lab
        if len(self._sentiment) > 5000:
            self._sentiment.pop(next(iter(self._sentiment)))
        return lab

    async def _opinion(self, topic: str, grounds: list[dict], area) -> dict | None:
        """What it makes of a topic, and what that rests on. None when it has no grounds for a view."""
        s = self.sys
        why, stance, strength = [], "", 0.0
        disputed = [g for g in (area.gaps if area else []) if "disagree" in g["why"]]
        wrong = [g for g in (area.gaps if area else []) if "wrong" in g["why"]]
        feelings = [await self._feeling(g["answer"]) for g in grounds]
        pos, neg = feelings.count("positive"), feelings.count("negative")
        passage = s.mind._find("opinion", f"topic:{topic}") if s.mind.enabled else None
        if passage is not None and passage.source.startswith("thinking"):
            passage = None
        if disputed or wrong:
            stance = "unsure of what I know"
            strength = 0.7
            if disputed:
                why.append(f"my sources disagree on {len(disputed)} question{'s' if len(disputed) != 1 else ''} about it")
            if wrong:
                why.append("you told me an answer of mine about it was wrong")
        elif passage is not None and passage.stance in ("favourable", "unfavourable"):
            stance, strength = passage.stance, passage.strength
            why.append(passage.text)
        elif len(grounds) >= self.pol.min_feelings and (pos or neg) and max(pos, neg) / len(grounds) >= 0.4 and pos != neg:
            stance = "favourable" if pos > neg else "unfavourable"
            strength = max(pos, neg) / len(grounds)
            why.append(f"of the {len(grounds)} things I hold about it, {pos} read positive and {neg} negative")
        elif area is not None and (area.asked >= self.pol.interested_after or area.people):
            stance = "interested"
            strength = min(1.0, 0.4 + 0.1 * area.asked)
            why.append("a person told me they are interested in it" if area.people else
                       f"it keeps coming up: people asked about it {area.asked} times")
        if not stance:
            return None
        if area is not None and stance != "interested" and area.asked >= self.pol.interested_after:
            why.append(f"people asked about it {area.asked} times")
        return {"stance": stance, "strength": round(float(strength), 3), "why": "; ".join(why),
                "private": looks_like_person(topic)}

    def _desire(self, topic: str, grounds: list[dict], area, areas: dict) -> dict | None:
        """What it wants to know about a topic, from the gaps curiosity sees (curiosity.py)."""
        if area is not None and area.gaps:
            q = area.gaps[0]["question"]
            return {"want": f"I want to find out {question_core(q)}", "kind": "find out", "question": q,
                    "strength": round(min(1.0, 0.5 + area.score), 3), "status": "open", "objective": "",
                    "why": area.gaps[0]["why"]}
        about = norm(topic)
        for g in grounds:  # a name its facts mention that it knows nothing more about
            for name in topics_in(g["answer"]):
                k = norm(name)
                other = areas.get(k)
                if k != about and other is not None and other.facts == 0 and other.named <= 1 and not other.read \
                        and k not in about and about not in k and name.lower() not in STOP_CAPS and len(k) > 2:
                    return {"want": f"I'd like to learn more about {name}: all I hold is that it comes up in what I know "
                                    f"about {topic}", "kind": "learn about",
                            "area": name, "strength": round(min(1.0, 0.3 + other.score), 3), "status": "open",
                            "objective": "", "why": f"its facts about {topic} mention {name}"}
        if area is not None and area.interest >= self.pol.desire_interest and len(grounds) < self.pol.enough_facts:
            return {"want": f"I'd like to know more about {topic}", "kind": "know more", "area": topic,
                    "strength": round(area.interest, 3), "status": "open", "objective": "",
                    "why": "; ".join(area.why_interest[:2])}
        return None

    async def _understanding(self, topic: str, grounds: list[dict]) -> tuple[str, str, str]:
        """(understanding, formed_by, rejected): its own rules, or an LLM teacher's draft that the honesty
        guard accepts (nothing in it that its facts do not say)."""
        claims = [claim(g["ask"], g["answer"]) for g in grounds]
        texts = [f"{g['ask']} {g['answer']}" for g in grounds] + [topic]
        rejected = ""
        if self.pol.llm and self.sys.teachers.has_llm():
            drafted = await self.sys.teachers.synthesize(topic, claims[: self.pol.max_claims * 2])
            if drafted:
                text, by = drafted
                text = " ".join(str(text).split())[:600]
                extra = faithful(text, texts)
                if text and not extra:
                    return text, f"drafted by the {by} teacher, checked against its facts", ""
                self.stats["rejected"] += 1
                rejected = f"the {by} teacher's draft added what its facts do not say: {', '.join(extra[:6])}"
        text = synthesize(topic, claims, self.pol.max_claims)
        extra = faithful(text, texts)
        if extra:  # its own rules slipped: say only what it holds, as it holds it
            text = "As I understand it, " + "; ".join(_lower_article(c) for c in claims[: self.pol.max_claims]) + "."
        return text, "its own words (rule-based synthesis of what it holds)", rejected

    def _topics(self, facts: list[dict]) -> dict[str, dict]:
        """Topic key -> {"name", "grounds" [fact indexes], "about"}; and each fact's primary topic."""
        topics: dict[str, dict] = {}
        for i, f in enumerate(facts):
            about = topics_in(f["ask"])
            named = [x for x in topics_in(f["answer"]) if norm(x) not in {norm(a) for a in about}]
            for name in about + named:
                k = norm(name)
                if not k or len(k) < 2:
                    continue
                t = topics.setdefault(k, {"name": name, "grounds": [], "about": 0})
                if i not in t["grounds"]:
                    t["grounds"].append(i)
                    t["about"] += name in about
        return topics

    async def form(self, force: bool = False) -> dict:
        """Re-form its thoughts from what its memory nodes hold now. Returns what changed."""
        if not self.enabled:
            return {"formed": 0, "reformed": 0, "dropped": 0}
        async with self._lock:
            return await self._form(force)

    async def _form(self, force: bool) -> dict:
        s = self.sys
        facts = self._facts()
        topics = self._topics(facts)
        cur = getattr(s, "curiosity", None)
        areas = cur.areas() if cur is not None and cur.enabled else {}
        wanted: dict[str, dict] = {}
        for k, t in topics.items():
            area = areas.get(k)
            n = len(t["grounds"])
            keep = n >= self.pol.min_facts or (t["about"] and area is not None and (area.gaps or area.interest > 0))
            if keep:
                wanted[k] = t
        # a fact's primary topic: the richest topic it is about (its thought is the one its question brings up)
        primary: dict[int, str] = {}
        for i, f in enumerate(facts):
            cands = [k for k in wanted if i in wanted[k]["grounds"]]
            if cands:
                primary[i] = max(cands, key=lambda k: (len(wanted[k]["grounds"]), wanted[k]["about"], -len(k)))
        ranked = sorted(wanted, key=lambda k: (-len(wanted[k]["grounds"]), k))[: self.pol.max_thoughts]
        by_key = {th.key: th for th in self.thoughts.values()}
        changed = {"formed": 0, "reformed": 0, "dropped": 0}
        touched: set[str] = set()
        for k in ranked:
            t = wanted[k]
            grounds = [{**facts[i], "primary": primary.get(i) == k} for i in t["grounds"][: self.pol.max_grounds]]
            t["name"] = display_name(t["name"], [f"{g['ask']} {g['answer']}" for g in grounds])
            area = areas.get(k)
            opinion = await self._opinion(t["name"], grounds, area)
            desire = self._desire(t["name"], grounds, area, areas)
            th = by_key.get(k)
            if th is not None and th.desire and desire and th.desire.get("want") == desire["want"]:
                desire.update(status=th.desire.get("status", "open"), objective=th.desire.get("objective", ""))
            sig = hashlib.sha1(json.dumps([[g["ask"], g["answer"]] for g in grounds] + [opinion, desire],
                                          sort_keys=True, default=str).encode()).hexdigest()[:16]
            cue_sig = hashlib.sha1(json.dumps([t["name"]] + sorted(g["ask"] for g in grounds if g["primary"])).encode()).hexdigest()[:16]
            if th is not None and th.sig == sig and not force:
                touched.add(th.id)
                continue
            understanding, formed_by, rejected = await self._understanding(t["name"], grounds)
            if th is None:
                th = Thought(f"th-{self._ids}", t["name"], k)
                self._ids += 1
                self.thoughts[th.id] = th
                changed["formed"] += 1
                self.stats["formed"] += 1
            else:
                changed["reformed"] += 1
                self.stats["reformed"] += 1
            before = (th.opinion or {}).get("stance")
            th.topic, th.understanding, th.grounds, th.opinion, th.desire = t["name"], understanding, grounds, opinion, desire
            th.formed_by, th.rejected, th.sig, th.updated = formed_by, rejected, sig, time.time()
            touched.add(th.id)
            if th.cue_sig != cue_sig:
                th.cue_sig = cue_sig
                self._place(th)
            if opinion and s.mind.enabled and opinion["stance"] != before:
                subject = f"topic:{th.topic}"
                src = f"thinking node {th.node or '(not placed yet)'} ({th.id})"
                if opinion["private"]:
                    s.mind.note("opinion", subject, opinion["why"], private=True, stance=opinion["stance"],
                                strength=opinion["strength"], source=src)
                else:
                    s.mind.opinion(subject, opinion["stance"], opinion["strength"], opinion["why"], src)
            s.bus.publish("thinking.thought", id=th.id, topic=th.topic[:80], node=th.node, new=changed["formed"] > 0 and before is None,
                          stance="" if not opinion or opinion["private"] else opinion["stance"],
                          desire=(desire or {}).get("want", "")[:160], grounds=len(grounds))
        for tid in [tid for tid in self.thoughts if tid not in touched]:  # what it no longer holds facts for
            th = self.thoughts.pop(tid)
            node = self.nodes.get(th.node)
            if node is not None and tid in node.outputs:
                node.outputs = [o for o in node.outputs if o != tid]
                node.version += 1
            changed["dropped"] += 1
            self.stats["dropped"] += 1
        for node in list(self.nodes.values()):
            if len(node.outputs) <= 1:
                self._drop_node(node)
        if any(changed.values()):
            self.history.append({"ts": round(time.time(), 1), **changed, "thoughts": len(self.thoughts)})
            self.save()
            for node in self.nodes.values():
                self.write_report(node)
        return changed

    def _place(self, th: Thought) -> None:
        """Put a thought in a thinking node (one with room, or a new one), and mark it to be retrained."""
        node = self.nodes.get(th.node)
        if node is None or th.id not in node.outputs:
            node = next((n for n in self.nodes.values() if len(n.outputs) - 1 < self.pol.thoughts_per_node), None)
            if node is None:
                num = max([int(n.split("-")[-1]) for n in self.nodes] or [0]) + 1
                node = ThinkingNode(f"think-{num:04d}", self.dim)
                self.nodes[node.id] = node
                self.sys.guardian.audit("thinking", "thinking.node", node.id)
            node.outputs.append(th.id)
            th.node = node.id
        node.version += 1
        node.status = "learning"

    def _drop_node(self, node: ThinkingNode) -> None:
        self.nodes.pop(node.id, None)
        for p in (self._weights_path(node.id), self.report_path(node)):
            try:
                p.unlink()
            except OSError:
                pass

    def forget(self, tid: str, by: str) -> tuple[bool, str]:
        th = self.thoughts.get(tid)
        if th is None:
            return False, f"no thought {tid}"
        self.thoughts.pop(tid)
        node = self.nodes.get(th.node)
        if node is not None:
            node.outputs = [o for o in node.outputs if o != tid]
            node.version += 1
            if len(node.outputs) <= 1:
                self._drop_node(node)
        self.sys.guardian.audit(by, "thinking.forgot", tid, topic=th.topic)
        self.save()
        return True, f"dropped {tid} ({th.topic}); it forms it again only if what it holds changes"

    # --------------------------------------------------------------- training
    def _cues(self, th: Thought) -> list[str]:
        t = th.topic
        out = [t, t.lower(), f"what do you think of {t}?", f"what do you think about {t}", f"your thoughts on {t}",
               f"what do you make of {t}?", f"tell me about {t}", f"what do you know about {t}?",
               f"in your own words, {t}", f"how do you feel about {t}?"]
        for g in th.grounds:
            if g.get("primary"):
                out += [g["ask"], f"{g['ask']} {_first_sentence(g['answer'])}", g["answer"][:300]]
        return list(dict.fromkeys(x for x in out if x))

    def next_job(self) -> Job | None:
        node = next((n for n in self.nodes.values() if n.dirty and len(n.outputs) > 1), None)
        if node is None:
            return None
        idx = {tid: i for i, tid in enumerate(node.outputs)}
        rows: dict[str, tuple[int, float]] = {}
        tests: list[tuple[str, int]] = []
        for tid in node.outputs[1:]:
            th = self.thoughts.get(tid)
            if th is None:
                continue
            for c in self._cues(th):
                rows.setdefault(c, (idx[tid], 1.0))
            tests.append((th.topic, idx[tid]))
            tests.append((f"what do you think of {th.topic}?", idx[tid]))
            tests += [(g["ask"], idx[tid]) for g in th.grounds if g.get("primary")][:6]
        mine = {norm(x) for x in rows}

        def negative(text: str, w: float) -> None:
            if text and text not in rows and norm(text) not in mine:
                rows[text] = (0, w)
        for p in PROBES:
            negative(p, 0.6)
        unknowns = list(PROBES[: self.pol.probe_tests])
        for other in self.nodes.values():
            if other is node:
                continue
            for tid in other.outputs[1:]:
                th = self.thoughts.get(tid)
                if th is not None:
                    for c in self._cues(th)[:6]:
                        negative(c, 0.5)
        try:  # orders its router sends to tools and skills: a review to read is not a topic to think about
            for e in [e.text for e in self.sys.store.examples("dispatch") if e.label.startswith(("skill:", "tool:"))][-80:]:
                negative(e, 0.4)
        except Exception:
            pass
        for f in self._facts()[: self.pol.max_negatives]:
            if not any(f["ask"] == g["ask"] and g.get("primary") for tid in node.outputs[1:]
                       for g in (self.thoughts[tid].grounds if tid in self.thoughts else [])):
                negative(f["ask"], 0.4)
        for i, tid in enumerate(node.outputs[1:]):
            th = self.thoughts.get(tid)
            if th is None:
                continue
            filler = FILLERS[(i * 7 + len(th.topic)) % len(FILLERS)]
            miss = f"what do you think of {filler}?"
            if norm(filler) not in {norm(x.topic) for x in self.thoughts.values()}:
                negative(miss, 0.7)
                if miss not in unknowns:
                    unknowns.append(miss)
        tests = [(t, e) for t, e in dict.fromkeys(tests)] + [(u, 0) for u in unknowns if u not in rows or rows[u][0] == 0]
        texts = list(rows)
        params, outputs = node.params, list(node.outputs)
        prev = getattr(node, "_trained_outputs", None)
        if params is not None and prev is not None and prev != outputs:
            params = remap_outputs(params, prev, outputs, np.random.default_rng(len(outputs)))
        elif params is not None and params[-1].shape[0] != len(outputs):
            params = None
        return Job("learn", node.id, node.version, self.dim, texts, np.array([rows[t][0] for t in texts], dtype=np.int64),
                   np.array([rows[t][1] for t in texts], dtype=np.float32), [t for t, _ in tests],
                   np.array([e for _, e in tests], dtype=np.int64), self.pol.threshold, params,
                   node.inputs or self.dim, 32, len(outputs), self.pol.epochs, 0.01,
                   int(hashlib.sha1(node.id.encode()).hexdigest()[:6], 16) + node.version,
                   tuple(self.pol.start_hidden), self.pol.max_hidden, 3, note=f"{len(outputs) - 1} thoughts",
                   answer_texts=outputs, margin=self.pol.test_margin)

    def finish(self, job: Job, res: dict) -> None:
        node = self.nodes.get(job.node_id)
        if node is None:
            return
        if job.version != node.version or job.answer_texts != node.outputs:  # its thoughts changed meanwhile
            return
        node.params, node.inputs, node.bits = res["params"], res["inputs"], res["bits"]
        node._trained_outputs = list(node.outputs)
        node.passed, node.total = res["passed"], res["total"]
        node.failing = [job.test_texts[i] for i in res["failing"]][:20]
        node.status = "ready" if res["ok"] else "failing"
        node.checked_version = node.version
        self.stats["learned"] += 1
        self._save_weights(node)
        self.save()
        self.write_report(node)
        self.sys.guardian.audit("thinking", "thinking.learned", node.id, passed=node.passed, total=node.total,
                                thoughts=len(node.outputs) - 1)
        self.sys.bus.publish("thinking.learned", node=node.id, passed=node.passed, total=node.total,
                             thoughts=len(node.outputs) - 1, ok=res["ok"])

    # ---------------------------------------------------------------- thinking
    def think(self, text: str, used: bool = True) -> tuple[Thought, float, ThinkingNode] | None:
        """The thought a thinking node fires on for a text (a question, with its answer after it)."""
        if not self.enabled:
            return None
        ready = [n for n in self.nodes.values() if n.params is not None and len(n.outputs) > 1]
        if not ready:
            return None
        x = encode_many([str(text)[:600]], self.dim)
        best = None
        for n in ready:
            if n.params[-1].shape[0] != len(n.outputs):
                continue
            top, conf = n.predict(x)
            i, c = int(top[0]), float(conf[0])
            label = n.outputs[i] if i < len(n.outputs) else NONE
            if hasattr(self.sys, "firings"):
                from .anatomy import record
                th = self.thoughts.get(label)
                record(self.sys, n.id, "thinking", text, th.topic if th else label, c, used=False)
            if i == 0 or c < self.pol.threshold or label not in self.thoughts or label in n.failing:
                continue
            if best is None or c > best[1]:
                best = (self.thoughts[label], c, n)
        if best is not None and used:
            self._use(best, text)
        return best

    def _use(self, hit: tuple, text: str) -> None:
        """Its thought was used: counted, and its firing marked used (anatomy.py)."""
        hit[2].served += 1
        hit[0].served += 1
        if getattr(self.sys, "firings", None):
            for f in reversed(self.sys.firings):
                if f["node"] == hit[2].id and f["text"] == str(text)[:300]:
                    f["used"] = True
                    break

    def _shareable(self) -> bool:
        return self.sys.affect.state.values["trust"] >= self.sys.directives.mind.share_trust

    def _view(self, th: Thought, conf: float, node: ThinkingNode | None, how: str) -> dict:
        share = self._shareable()
        op = th.opinion
        return {"thought": th.id, "node": node.id if node else th.node, "topic": th.topic, "confidence": round(conf, 3),
                "understanding": th.understanding, "text": th.text(share),
                "opinion": None if not op else ({"private": True} if op.get("private") or not share else
                                                {"stance": op["stance"], "why": op["why"]}),
                "desire": None if not th.desire or th.desire.get("status") == "satisfied" else
                {"want": th.desire["want"], "status": th.desire.get("status", "open"), "objective": th.desire.get("objective", "")},
                "grounds": len(th.grounds), "formed_by": th.formed_by, "how": how}

    def after(self, order, res) -> dict | None:
        """After the knowledge context: the thought a thinking node holds about what was asked and answered,
        when it adds something (more facts put together, a view it may share, or something it wants to know)."""
        if not (self.enabled and self.pol.express) or order.background or str(order.source).startswith(("challenge:", "objective", "task:")):
            return None
        if not res.ok or (res.route or "") != "answer:direct" or res.knowledge not in ("memory", "recalled", "researched"):
            return None
        if res.disposition in ("brief", "confirm", "defer", "block"):
            return None
        said = _first_sentence(str(res.output or res.answer))
        text = f"{order.text} {said}"
        hit = self.think(text, used=False)
        if hit is None:
            return None
        th, conf, node = hit
        if not self._about(th, text, said):  # its thought must be about what was asked and answered
            return None
        share = self._shareable()
        adds = (len(th.grounds) >= 2 or (th.opinion and not th.opinion.get("private") and share)
                or (th.desire and th.desire.get("status") != "satisfied"))
        if not adds:
            return None
        self._use(hit, text)
        self.stats["expressed"] += 1
        view = self._view(th, conf, node, "after the knowledge context")
        res.trace.append(f"thinking node {node.id} added its own thought after the knowledge ({th.id}: {th.topic}, "
                         f"{conf:.0%} sure; it rests on {len(th.grounds)} fact{'s' if len(th.grounds) != 1 else ''})")
        self.sys.bus.publish("thinking.expressed", order=order.id, node=node.id, thought=th.id, topic=th.topic[:80])
        return view

    @staticmethod
    def _about(th: Thought, text: str, said: str = "") -> bool:
        """Is a thought about this text: does the text name its topic, or is the answer one of its facts?"""
        words = set(_tokens(text))
        topic = [w for w in _tokens(th.topic) if w not in FRAME]
        if topic and all(w in words for w in topic):
            return True
        n = norm(said)
        return bool(n) and any(norm(_first_sentence(g["answer"])) == n or norm(g["answer"]) == n for g in th.grounds)

    def find(self, subject: str) -> Thought | None:
        k = norm(subject)
        if not k:
            return None
        hit = next((t for t in self.thoughts.values() if t.key == k), None)
        if hit is None:
            close = [t for t in self.thoughts.values() if (t.key in k or k in t.key) and min(len(t.key), len(k)) >= 3]
            hit = max(close, key=lambda t: (len(t.grounds), -abs(len(t.key) - len(k))), default=None)
        return hit

    def answer(self, text: str, order_id: str = "") -> tuple[str, str, dict | None] | None:
        """Questions for its thinking nodes: what it thinks of a topic, a topic in its own words, and what it
        wants to find out. (answer, trace, thought view) or None when this is not one of those, or it holds no
        thought about the topic (the mind answers then)."""
        if not self.enabled:
            return None
        flat = " ".join(str(text).split())
        if DESIRE_RE.search(flat):
            return self._desires_answer()
        m = THINK_RE.match(flat)
        if not m:
            return None
        subject = m.group("subject").strip().rstrip("?.!")
        if subject.lower() in PERSON_SUBJECTS or self.sys.mind._known_subject(subject):
            return None  # its tools, skills, teachers and you: the mind's opinions (P9)
        th = self.find(subject)
        conf, node = 1.0, self.nodes.get(th.node) if th else None
        how = f"it holds a thought about {th.topic}" if th else ""
        if th is None:
            hit = self.think(flat, used=False)
            if hit is None or not self._about(hit[0], subject):
                return None
            self._use(hit, flat)
            th, conf, node = hit
            how = f"thinking node {node.id} fired on the question ({conf:.0%} sure)"
        else:
            th.served += 1
        self.stats["answered"] += 1
        view = self._view(th, conf, node, how)
        out = view["text"]
        if not th.opinion and "think" in flat.lower():
            out += " I have no particular view of it yet."
        return out, f"answered by thinking node {view['node']} ({th.id}: {th.topic}; {how})", view

    def desires(self) -> list[dict]:
        out = []
        for th in self.thoughts.values():
            d = th.desire
            if d and d.get("status") in ("open", "pursuing"):
                out.append({**d, "thought": th.id, "node": th.node, "topic": th.topic})
        return sorted(out, key=lambda d: -float(d.get("strength", 0)))

    def _desires_answer(self) -> tuple[str, str, None]:
        ds = self.desires()
        cur = getattr(self.sys, "curiosity", None)
        if not ds:
            aims = cur.aims(3) if cur is not None and cur.enabled else []
            if not aims:
                return ("Nothing in particular yet: I need to hold more about the world, or be asked more, before I "
                        "know what I'm missing.", "no desires in its thinking nodes", None)
            lines = "; ".join(f"to {a['text']} ({a['why']})" for a in aims)
            return f"From where I stand I'd like {lines}.", "no desires in its thinking nodes yet; from its curiosity", None
        parts = []
        for d in ds[:4]:
            w = d["want"] + (f" - I'm working on it ({d['objective']})" if d.get("status") == "pursuing" and d.get("objective") else "")
            parts.append(w)
        return ("Here is what I'd like to know, from what I hold now: " + "; ".join(parts) + ".",
                f"desires held in thinking nodes ({', '.join(sorted({d['node'] for d in ds[:4] if d['node']}))})", None)

    # ------------------------------------------------------ desires and objectives
    def link_objective(self, aim: dict, obj_id: str) -> None:
        """The director took up a desire (or the gap behind it) as an objective it set itself."""
        changed = False
        for th in self.thoughts.values():
            d = th.desire
            if not d:
                continue
            q, a = d.get("question", ""), d.get("area", "")
            if (q and aim.get("question") and norm(q) == norm(aim["question"])) or \
                    (a and aim.get("area") and norm(a) == norm(aim["area"])) or norm(th.topic) == norm(aim.get("area", "")):
                d.update(status="pursuing", objective=obj_id)
                changed = True
        if changed:
            self.save()

    def on_event(self, ev) -> None:
        if ev.topic in ("objective.ended", "objective.removed"):
            oid = ev.data.get("id")
            status = ev.data.get("status") or "removed"
            changed = False
            for th in self.thoughts.values():
                d = th.desire
                if d and d.get("objective") == oid and status in ("met", "unmet", "removed"):
                    d["status"] = "satisfied" if status == "met" else "open"
                    if status != "met":
                        d["objective"] = ""
                    changed = True
            if changed:
                self.save()
        elif ev.topic in ("memory.fact",):
            self._next_form = 0.0  # what it holds changed: think again soon

    # ------------------------------------------------------------------ views
    def status(self) -> dict:
        share = self._shareable()
        thoughts = sorted(self.thoughts.values(), key=lambda t: (-len(t.grounds), t.topic))
        return {"enabled": self.enabled, "stats": dict(self.stats),
                "nodes": [{"id": n.id, "status": n.status, "spec": n.spec(), "thoughts": len(n.outputs) - 1,
                           "passed": n.passed, "total": n.total, "served": n.served, "weights": n.weights,
                           "failing": n.failing[:5]} for n in self.nodes.values()],
                "thoughts": [{"id": t.id, "topic": t.topic, "node": t.node, "understanding": t.understanding,
                              "opinion": ({"private": True} if t.opinion and t.opinion.get("private") else t.opinion),
                              "desire": t.desire, "formed_by": t.formed_by, "rejected": t.rejected,
                              "grounds": [{"ask": g["ask"], "answer": g["answer"][:200], "node": g["node"]} for g in t.grounds[:8]],
                              "facts": len(t.grounds), "served": t.served, "updated": t.updated,
                              "text": t.text(share)} for t in thoughts[: self.pol.show]],
                "desires": self.desires()[:12], "history": self.history[-10:]}

    def lines(self) -> list[str]:
        if not self.enabled:
            return ["thinking is off ([thinking] enabled = false, or memory is off)"]
        st = self.status()
        out = [f"thinking: {len(self.thoughts)} thoughts in {len(self.nodes)} thinking nodes "
               f"(formed {st['stats']['formed']}, re-formed {st['stats']['reformed']}, expressed after answers "
               f"{st['stats']['expressed']}, LLM drafts turned down {st['stats']['rejected']})"]
        for n in st["nodes"]:
            out.append(f"  {n['id']} [{n['status']}] {n['thoughts']} thoughts; {n['spec']}; tests {n['passed']}/{n['total']}")
        for t in st["thoughts"][:10]:
            out.append(f"  {t['id']} {t['topic']} ({t['facts']} facts, {t['node']}): {t['understanding'][:150]}")
            if t["opinion"]:
                out.append(f"      opinion: {t['opinion']['stance']} ({t['opinion']['why'][:100]})"
                           + (" [private]" if t["opinion"].get("private") else ""))
            if t["desire"]:
                out.append(f"      desire: {t['desire']['want'][:110]} [{t['desire'].get('status', 'open')}]")
        return out

    def show(self, ref: str, by: str = "operator") -> list[str]:
        """One thought in full: the operator's view, a private opinion included (each look is audited, P9)."""
        th = self.thoughts.get(ref) or self.find(ref)
        if th is None:
            return [f"  no thought about '{ref}' yet"]
        if th.opinion and th.opinion.get("private"):
            self.sys.guardian.audit(by, "thinking.inspected", th.id, topic=th.topic, private=True)
        out = [f"{th.id}: {th.topic} (thinking node {th.node or 'not placed yet'}; {th.formed_by})", f"  {th.understanding}"]
        if th.opinion:
            out.append(f"  opinion: {th.opinion['stance']} ({th.opinion['why']})" + (" - private (P9)" if th.opinion.get("private") else ""))
        if th.desire:
            out.append(f"  desire: {th.desire['want']} [{th.desire.get('status', 'open')}"
                       + (f" {th.desire['objective']}" if th.desire.get("objective") else "") + f"]: {th.desire.get('why', '')}")
        if th.rejected:
            out.append(f"  turned down: {th.rejected}")
        out.append("  rests on:")
        out += [f"    \"{g['ask']}\" -> {g['answer'][:120]} ({g['node']})" for g in th.grounds[:10]]
        return out


class Thinker(Agent):
    """Re-forms the thoughts in the thinking nodes when what the memory nodes hold changes, and trains
    the thinking nodes. Background work: it waits for orders (P2), sleeps with the mesh, stops on /halt (P1)."""
    name = "thinker"

    @property
    def interval(self) -> float:
        return max(0.1, self.sys.directives.background.tick_seconds)

    async def step(self) -> None:
        s = self.sys
        bank = s.thinking
        if not bank.enabled or not s.guardian.background_allowed() or s.dreams.asleep:
            return
        now = time.time()
        if now >= bank._next_form:
            bank._next_form = now + bank.pol.form_seconds
            with s.activity.track("thinking", "thinker", "putting what it holds into its own words"):
                await bank.form()
        job = bank.next_job()
        if job is None:
            return
        with s.activity.track("thinking", "thinker", f"{job.node_id}: learning {job.note}"):
            res = await self.bg(job.run)
        bank.finish(job, res)
        self.stats["learned"] += 1
