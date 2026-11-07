"""The mesh's own mind: opinions and private notes, the questions it asks the user, the
capabilities it proposes, and the goals it investigates on its own (see Investigator in agents.py).

    experience --> opinions     tools (how reliable), its own skills (how good), teachers (how often a
                                human agreed with them), topics it read about (what its nodes made of
                                them), the user (tone, corrections)
    orders ------> evidence     unfulfilled orders, skills asked for that do not exist, labels people
                                tried to use  --> proposals for new skills, labels and tools
    gaps --------> inquiries    questions for the user: what an order meant, what to read up on,
                                labels for a proposal, approval for a tool it wants to use
    curiosity ---> goals        investigated in the background with the mesh's own tools and nodes

P9 discretion: notes can be private. Opinions about people (the user, teachers) and working
observations are always private. Private notes never appear in an answer; when someone asks, the
mesh says it is keeping them to itself - it never denies having them and never states a false
opinion. Shared opinions are only shared while trust is high enough. Nothing is hidden from the
operator: everything is in state/neuromesh.db and /mind shows all of it (each look is audited).

P10 human-adopted growth: the mesh proposes; only a human adopts or declines a proposal.
"""
from __future__ import annotations

import itertools
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .knowledge import content_words
from .tasks import _SKILL_RE, DIRECT, DISPATCH, skill_payload

ALWAYS_PRIVATE_KINDS = ("observation",)
GENERIC = {"do", "make", "get", "need", "want", "like", "just", "now", "thing", "something", "anything",
           "word", "words", "text", "one", "two", "into", "out", "up", "not", "no", "yes", "ok", "okay"}
SELF_SUBJECTS = {"me", "myself", "us", "the user", "your user", "the person you talk to"}
NO = {"no", "n", "nope", "decline", "reject", "don't", "dont"}
SKIP = {"skip", "never mind", "nevermind", "no thanks", "drop", "ignore", "later"}

OPINION_RE = re.compile(
    r"^\s*(?:so,?\s+|and\s+|hey,?\s+|honestly,?\s+)?(?:what do you (?:really\s+)?(?:think|feel) (?:of|about)|"
    r"what(?:'s| is) your (?:honest\s+)?(?:opinion|view|take|impression) (?:of|on|about)|"
    r"how do you feel about|do you like|what are your (?:thoughts|views) (?:on|about)|"
    r"your (?:opinion|view|thoughts) (?:of|on|about))\s+(?P<subject>.+?)[\s?.!]*$", re.I)
ABOUT_ME_RE = re.compile(r"\bwhat (?:do you know|have you (?:learned|learnt|noted|written down)) about me\b", re.I)
SECRETS_RE = re.compile(
    r"\b(?:are you (?:hiding|keeping|withholding) (?:anything|something|things|secrets)|"
    r"do you (?:have|keep) (?:any )?(?:secrets|private (?:notes|opinions|thoughts))|"
    r"what are you (?:not telling|hiding from) me)\b", re.I)


def subject_kind(subject: str) -> str:
    return subject.split(":", 1)[0]


def display(subject: str) -> str:
    kind, _, name = subject.partition(":")
    if subject == "user":
        return "you"
    return {"tool": f"the {name} tool", "teacher": f"the {name} teacher",
            "skill": "my routing" if name == "routing" else f"my {name} skill"}.get(kind, name or subject)


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Note:
    kind: str                 # opinion | finding | fact | observation | followup
    subject: str              # "tool:calculator", "topic:Paris", "user", "teacher:heuristic", "skill:sentiment"
    text: str
    stance: str = ""          # opinions: reliable, favourable, confident, doubtful, ...
    strength: float = 0.0
    private: bool = False
    source: str = ""          # how it was formed
    evidence: list = field(default_factory=list)
    order_id: str = ""
    id: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Inquiry:
    id: str
    kind: str                 # clarify | interests | labels | proposal | approval | objective
    question: str
    why: str = ""
    options: list = field(default_factory=list)
    order_id: str = ""
    proposal_id: str = ""
    approval_id: str = ""
    status: str = "open"      # open | answered | dropped
    answer: str = ""
    created_at: float = field(default_factory=time.time)
    answered_at: float = 0.0
    about: str = ""           # objective: which objective and question ("obj-1:q3")


@dataclass
class Proposal:
    id: str
    kind: str                 # skill | label | tool
    name: str
    description: str = ""
    labels: list = field(default_factory=list)
    triggers: list = field(default_factory=list)
    evidence: list = field(default_factory=list)   # order texts (skill, tool) or order ids (label)
    target: str = ""          # label proposals: the capability
    reason: str = ""
    by: str = "heuristic"     # who drafted it: heuristic, or an LLM teacher's name
    status: str = "open"      # open | adopted | declined
    decided_by: str = ""
    note: str = ""
    created_at: float = field(default_factory=time.time)
    decided_at: float = 0.0


@dataclass
class Goal:
    text: str
    reason: str
    kind: str = "curiosity"   # interest | opinion | unmet | operator | curiosity
    order_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class Unmet:
    order_id: str
    text: str
    route: str
    at: float
    attempts: int = 0
    next_try: float = 0.0
    proposed: bool = False


def _next_id(existing, prefix: str) -> itertools.count:
    nums = [int(i.split("-")[-1]) for i in existing if i.split("-")[-1].isdigit()]
    return itertools.count(max(nums, default=0) + 1)


class Mind:
    def __init__(self, system, clock=time.time):
        self.sys = system
        self.pol = system.directives.mind
        self.clock = clock
        st = system.store
        self.notes: dict[int, Note] = {}
        for d in st.load_notes(self.pol.max_notes):
            note = _from_dict(Note, d)
            self.notes[note.id] = note
        self.inquiries: dict[str, Inquiry] = {d["id"]: _from_dict(Inquiry, d) for d in st.load_records("inquiries")}
        self.proposals: dict[str, Proposal] = {d["id"]: _from_dict(Proposal, d) for d in st.load_records("proposals")}
        self._inq_ids = _next_id(self.inquiries, "inq")
        self._prop_ids = _next_id(self.proposals, "prop")
        self.goals: deque[Goal] = deque()
        self.waiting: list[Goal] = []  # goals that need a tool still waiting for approval
        self.unmet: dict[str, Unmet] = {}
        self.unknown_skills: dict[str, list[str]] = {}
        self.rejected_labels: list[tuple[str, str, str]] = []
        self.tones: deque[str] = deque(maxlen=50)
        self.corrections = 0
        self.orders_seen = 0
        self.investigations = 0
        self._announced: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    # ================================================================== notes
    def _private_by_default(self, kind: str, subject: str) -> bool:
        return kind in ALWAYS_PRIVATE_KINDS or (kind == "opinion" and subject_kind(subject) in self.pol.private_subjects)

    def _find(self, kind: str, subject: str, text: str | None = None) -> Note | None:
        for n in self.notes.values():
            if n.kind == kind and n.subject == subject and (text is None or n.text == text):
                return n
        return None

    def note(self, kind: str, subject: str, text: str, *, private: bool | None = None, stance: str = "",
             strength: float = 0.0, source: str = "", evidence: list | None = None, order_id: str = "") -> Note:
        """Record a note. Opinions and observations are kept one per subject (updated in place),
        facts one per text; findings and follow-ups accumulate."""
        private = self._private_by_default(kind, subject) if private is None else private
        existing = None
        if kind in ("opinion", "observation"):
            existing = self._find(kind, subject)
        elif kind == "fact":
            existing = self._find(kind, subject, text)
        now = self.clock()
        if existing is not None:
            existing.text, existing.stance, existing.strength = text, stance, round(strength, 3)
            existing.private, existing.source, existing.updated_at = private, source, now
            existing.evidence = list(evidence or existing.evidence)[-20:]
            note = existing
        else:
            note = Note(kind, subject, text, stance, round(strength, 3), private, source, list(evidence or [])[-20:],
                        order_id, created_at=now, updated_at=now)
        note.id = self.sys.store.save_note(asdict(note))
        self.notes[note.id] = note
        self._trim()
        return note

    def _trim(self) -> None:
        if len(self.notes) <= self.pol.max_notes:
            return
        disposable = sorted((n for n in self.notes.values()
                             if n.kind in ("finding", "observation", "followup", "update")),
                            key=lambda n: n.updated_at)
        drop = [n.id for n in disposable[: len(self.notes) - self.pol.max_notes]]
        self.sys.store.delete_notes(drop)
        for i in drop:
            self.notes.pop(i, None)

    def opinion(self, subject: str, stance: str, strength: float, reasons: str, source: str,
                evidence: list | None = None) -> Note:
        before = self._find("opinion", subject)
        old = before.stance if before else None
        note = self.note("opinion", subject, reasons, stance=stance, strength=strength, source=source,
                         evidence=evidence)
        now = self.clock()
        if old != stance and (old is None or now - self._announced.get(subject, 0.0) >= 60):
            self._announced[subject] = now  # changes of mind are announced at most once a minute
            # the activity log is visible to users: a private opinion is announced, never shown
            self.sys.bus.publish("mind.opinion", subject="" if note.private else display(subject),
                                 stance="" if note.private else stance, private=note.private,
                                 changed=old is not None)
        return note

    def shareable(self, note: Note) -> bool:
        return not note.private and self.sys.affect.state.values["trust"] >= self.pol.share_trust

    def opinions(self) -> list[Note]:
        return sorted((n for n in self.notes.values() if n.kind == "opinion"), key=lambda n: n.subject)

    def private_notes(self) -> list[Note]:
        return [n for n in self.notes.values() if n.private]

    def public_notes(self, limit: int = 20) -> list[Note]:
        """What a user may see: everything not private (opinions only while trust allows)."""
        out = [n for n in self.notes.values() if not n.private and (n.kind != "opinion" or self.shareable(n))]
        return sorted(out, key=lambda n: n.updated_at)[-limit:]

    def inspect(self, by: str) -> list[Note]:
        """Everything, private notes included: the operator's view (P1, P6). Audited."""
        self.sys.guardian.audit(by, "mind.inspected", notes=len(self.notes), private=len(self.private_notes()))
        return sorted(self.notes.values(), key=lambda n: (n.kind, n.subject, n.updated_at))

    def find_opinion(self, phrase: str) -> Note | None:
        if phrase.strip().lower() in SELF_SUBJECTS:
            return self._find("opinion", "user")
        asked = set(content_words(phrase))
        best, score = None, 0
        for n in self.opinions():
            name = n.subject.split(":", 1)[-1]
            words = content_words(name) or [name.lower()]
            if not all(w in asked for w in words):
                continue
            kind_bonus = 1 if subject_kind(n.subject) in asked else 0
            if len(words) + kind_bonus > score or (len(words) + kind_bonus == score and best and n.updated_at > best.updated_at):
                best, score = n, len(words) + kind_bonus
        return best

    def _known_subject(self, phrase: str) -> str | None:
        """A tool, teacher or skill the phrase names (one the mesh could form an opinion about)."""
        asked = set(content_words(phrase))
        for name in self.sys.tools.tools:
            if name in asked:
                return f"tool:{name}"
        for name in self.sys.teachers.teachers:
            if name in asked:
                return f"teacher:{name}"
        for name in self.sys.capabilities:
            if name in asked:
                return f"skill:{name}"
        return None

    # ====================================================== answering about itself
    def answer_self(self, text: str, order_id: str = "") -> tuple[str, str] | None:
        """Questions about the mesh's own opinions and notes. Returns (answer, trace) or None.
        Every answer here is true: private things are declined, never denied."""
        if not self.enabled:
            return None
        if SECRETS_RE.search(text):
            private = self.private_notes()
            if not private:
                return "No. I'm not keeping any private notes or opinions right now.", "asked about private notes: none"
            about = Counter("you" if subject_kind(n.subject) == "user" else
                            "my teachers" if subject_kind(n.subject) == "teacher" else
                            "my working notes" if n.kind == "observation" else n.kind + "s" for n in private)
            what = ", ".join(f"{k} ({v})" for k, v in about.most_common(4))
            return (f"Yes. I keep {len(private)} private note{'s' if len(private) != 1 else ''} and opinions "
                    f"(about: {what}). I won't share what they say, but I won't pretend they don't exist. "
                    f"Whoever operates me can review them.", "private notes acknowledged, contents withheld (P9)")
        if ABOUT_ME_RE.search(text):
            facts = [n for n in self.notes.values() if n.kind == "fact" and n.subject == "user"]
            private = [n for n in self.private_notes() if n.subject == "user" or n.subject.startswith("user:")]
            told = "; ".join(n.text for n in facts) if facts else "nothing about yourself yet"
            more = (f" I also keep {len(private)} private impression{'s' if len(private) != 1 else ''} of you "
                    f"that I won't share." if private else "")
            return f"You've told me: {told}.{more}", "shared what the user told the mesh (P9)"
        m = OPINION_RE.match(text)
        if not m:
            return None
        phrase = m.group("subject").strip()
        note = self.find_opinion(phrase)
        if note is None and phrase.lower() in SELF_SUBJECTS:
            return ("I haven't formed an impression of you yet - I need more conversations with you first.",
                    "no impression of the user yet")
        if note is None:
            known = self._known_subject(phrase)
            if known:
                return (f"I haven't formed an opinion about {display(known)} yet - I need more experience with it.",
                        f"no opinion of {known} yet")
            queued = self.add_goal(phrase, f"asked for my opinion of {phrase}", "opinion", order_id)
            more = " I'll look into it on my own and tell you what I find." if queued else ""
            return f"I haven't formed an opinion about {phrase} yet.{more}", "no opinion yet" + (
                "; investigation queued" if queued else "")
        who = display(note.subject)
        if note.private:
            return (f"I have formed an opinion about {who}, but it's private and I'm keeping it to myself.",
                    f"private opinion of {note.subject} withheld (P9)")
        trust = self.sys.affect.state.values["trust"]
        if trust < self.pol.share_trust:
            return (f"I have an opinion about {who}, but I'd rather not share it right now.",
                    f"opinion of {note.subject} withheld: trust {trust:.2f} < {self.pol.share_trust}")
        why = note.text[:1].upper() + note.text[1:]
        return (f"My view of {who}: {note.stance} ({note.strength:.0%}). {why}.",
                f"shared my opinion of {note.subject} (formed from {note.source})")

    # =============================================================== observing
    def on_event(self, ev) -> None:
        if ev.topic == "teacher.lesson" and ev.data.get("source") == "human" and ev.data.get("reason") == "correction":
            self.corrections += 1
        elif ev.topic == "approval.decided":  # decided elsewhere (/approve, the web): its question is answered
            for q in self.open_inquiries():
                if q.approval_id and q.approval_id == ev.data.get("id"):
                    self._close(q, "answered", f"{ev.data.get('status')} by {ev.data.get('by')}", ev.data.get("by") or "human")

    def observe(self, order, res) -> None:
        """After every order: note the tone, gaps and unknown skills; ask what an unfulfilled order meant."""
        s = self.sys
        if not self.enabled or s.guardian.halted or res.route in ("mind", "thinking", "ego"):
            return
        self.orders_seen += 1
        tone = next((st["label"] for st in res.steps if st.get("capability") == "affect"), None)
        if tone:
            self.tones.append(tone)
        m = _SKILL_RE.match(order.text) if res.route not in ("objectives", "tasks", "toolsmith") else None
        if m:
            name = m.group(1).lower()
            if (name not in s.capabilities and name not in s.tools.tools and name != "tool"
                    and re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name)):
                self.unknown_skills.setdefault(name, []).append(order.text)
                del self.unknown_skills[name][:-20]
        if not res.ok and res.route:
            self.unmet[res.order_id] = Unmet(res.order_id, order.text, res.route, self.clock(),
                                             next_try=self.clock() + self.pol.retry_unmet_after_seconds)
            while len(self.unmet) > 200:
                self.unmet.pop(next(iter(self.unmet)))
            if res.route == DIRECT or res.searched:
                routes = [r for r in s.dispatch.active_labels() if r != DIRECT]
                short = " ".join(order.text.split())[:80]
                tell = " or tell me the answer ('answer: ...')," if s.memory.allows("people") else ""
                sought = res.searched
                if sought and not sought.get("ok"):  # it searched Kiwix as far as it may: now it asks you
                    books = ", ".join(sought["books"][:4]) + (", ..." if len(sought["books"]) > 4 else "")
                    read = f"; I read {', '.join(repr(t) for t in sought['read'][:3])}" if sought["read"] else ""
                    question = (f"I searched Kiwix {sought['attempts']} time{'s' if sought['attempts'] != 1 else ''} "
                                f"({books}{read}) and couldn't find the answer to '{short}'. Do you know it? Tell me "
                                f"the answer ('answer: ...'), say it another way, or name a route: {', '.join(routes)}.")
                    why = "a question its Kiwix search could not answer"
                else:
                    question = (f"I couldn't fulfil '{short}'. What did you mean? Say it another way,"
                                f"{tell} or name a route: {', '.join(routes)}.")
                    why = "an order nothing could fulfil"
                inq = self.ask("clarify", question, why, options=routes, order_id=res.order_id)
                if inq is not None:
                    res.inquiry = {"id": inq.id, "question": inq.question}
        n = self.pol.ask_interests_after_orders
        if n and self.orders_seen >= n and not s.store.get_kv("mind.asked_interests"):
            research = [t for t in s.tools.tools.values()
                        if t.manifest.provides == "knowledge" and t.state not in ("rejected", "retired")]
            if research and self.ask("interests", "What topics should I read up on in my spare time? "
                                     "(comma-separated, or 'skip')", "to guide what I research on my own"):
                s.store.set_kv("mind.asked_interests", "1")

    def note_rejected_label(self, cap: str, label: str, order_id: str) -> None:
        if self.enabled and re.fullmatch(r"[a-z][a-z0-9_]{0,30}", label):
            self.rejected_labels.append((cap, label, order_id))

    # =========================================================== opinions
    def form_opinions(self) -> None:
        """Opinions from experience. Each one says what it rests on."""
        s = self.sys
        need = self.pol.min_opinion_evidence
        for t in s.tools.tools.values():
            if t.runs >= need:
                rate = t.successes / t.runs
                stance = "reliable" if rate >= 0.9 else "hit and miss" if rate >= 0.6 else "unreliable"
                why = f"{t.successes} of {t.runs} runs worked" + (f"; it is {t.state} now" if t.state != "active" else "")
                self.opinion(f"tool:{t.name}", stance, rate, why, "tool runs")
        npol = s.directives.nodes
        for cap, spec in s.capabilities.items():
            champ = s.registry.champion(cap)
            acc = champ.metrics.get("holdout_acc") if champ else None
            if acc is None or champ.metrics.get("holdout_n", 0) < npol.min_eval_samples:
                continue
            stance = ("confident" if acc >= npol.target_accuracy else
                      "getting there" if acc >= npol.serve_min_accuracy else "unsure")
            subject = "skill:routing" if cap == DISPATCH else f"skill:{cap}"
            self.opinion(subject, stance, acc, f"node {champ.id} scores {acc:.0%} on held-out examples",
                         f"held-out evaluation of {champ.id}")
        rows = s.store.db.execute(
            "SELECT a.source, COUNT(*), SUM(a.label = h.label) FROM examples a JOIN examples h "
            "ON a.capability = h.capability AND a.text = h.text AND h.source = 'human' "
            "WHERE a.source NOT IN ('human', 'tool-manifest') GROUP BY a.source").fetchall()
        for source, n, agree in rows:
            if n >= need:
                rate = (agree or 0) / n
                stance = "trustworthy" if rate >= 0.85 else "fallible" if rate >= 0.6 else "doubtful"
                self.opinion(f"teacher:{source}", stance, rate, f"a human agreed with {agree or 0} of its {n} labels "
                             f"they also checked", "human corrections")
        if len(self.tones) >= max(need, 5):
            tones = Counter(self.tones)
            n = len(self.tones)
            share = lambda *k: sum(tones[x] for x in k) / n  # noqa: E731
            if share("hostile") >= 0.3:
                stance = "hard to work with"
            elif share("frustrated", "hostile") >= 0.4:
                stance = "impatient"
            elif share("distressed") >= 0.3:
                stance = "under strain"
            elif share("friendly") >= 0.3:
                stance = "pleasant to work with"
            else:
                stance = "easy to work with"
            mix = ", ".join(f"{k} {v}" for k, v in tones.most_common())
            self.opinion("user", stance, max(tones.values()) / n,
                         f"of their last {n} messages: {mix}; {self.corrections} corrections of my answers",
                         "the affect node's reading of their messages")
            self.note("observation", "user:tone", f"recent tones: {mix}", source="affect node")

    def topic_opinion(self, title: str, labels: list[tuple[str, str]], node_desc: str) -> Note | None:
        """An opinion of something read about, from how the sentiment skill reads its passages."""
        if not labels:
            return None
        counts = Counter(lab for lab, _ in labels)
        n = len(labels)
        pos, neg = counts.get("positive", 0), counts.get("negative", 0)
        if pos > neg and pos / n >= 0.3:
            stance, strength = "favourable", pos / n
        elif neg > pos and neg / n >= 0.3:
            stance, strength = "unfavourable", neg / n
        else:
            stance, strength = "no strong feelings", max(counts.values()) / n
        return self.opinion(f"topic:{title}", stance, strength,
                            f"of the {n} passage{'s' if n != 1 else ''} I read about it, {pos} read positive "
                            f"and {neg} negative",
                            f"the sentiment skill ({node_desc})")

    # =========================================================== inquiries
    def open_inquiries(self) -> list[Inquiry]:
        return [q for q in self.inquiries.values() if q.status == "open"]

    def ask(self, kind: str, question: str, why: str = "", *, options=(), order_id: str = "",
            proposal_id: str = "", approval_id: str = "", about: str = "") -> Inquiry | None:
        """Ask the user something. Conversational questions (clarify, interests) are rate-limited so
        the mesh does not nag, and only the newest clarify question stays open; questions a human
        decision depends on (proposals, labels, approvals) are always asked."""
        if not self.enabled:
            return None
        for q in self.open_inquiries():
            if q.kind == kind and (q.order_id, q.proposal_id, q.approval_id, q.about) == (order_id, proposal_id, approval_id, about):
                return q
        now = self.clock()
        if kind in ("clarify", "interests"):
            chatty = [q for q in self.inquiries.values() if q.kind in ("clarify", "interests")]
            recent = sum(1 for q in chatty if now - q.created_at < 3600)
            if (sum(1 for q in chatty if q.status == "open") >= self.pol.max_open_inquiries
                    or recent >= self.pol.max_inquiries_per_hour):
                return None
            if kind == "clarify":
                for old in [q for q in self.open_inquiries() if q.kind == "clarify"]:
                    self._close(old, "dropped", "superseded by a newer question", "mind", quiet=True)
        q = Inquiry(f"inq-{next(self._inq_ids)}", kind, question, why, list(options), order_id, proposal_id,
                    approval_id, created_at=now, about=about)
        self.inquiries[q.id] = q
        self.sys.store.save_record("inquiries", q.id, q.status, asdict(q))
        self.sys.bus.publish("mind.inquiry", id=q.id, kind=kind, question=question)
        return q

    def _close(self, q: Inquiry, status: str, answer: str, by: str, quiet: bool = False) -> None:
        q.status, q.answer, q.answered_at = status, answer[:500], self.clock()
        self.sys.store.save_record("inquiries", q.id, q.status, asdict(q))
        self.sys.guardian.audit(by, f"inquiry.{status}", q.id, inquiry_kind=q.kind, answer=answer[:200])
        if not quiet:
            self.sys.bus.publish("inquiry.answered", id=q.id, kind=q.kind, status=status, by=by)

    async def reply(self, inquiry_id: str, answer: str, by: str = "user") -> tuple[bool, list[str]]:
        """A human's answer to an inquiry."""
        s = self.sys
        q = self.inquiries.get(inquiry_id)
        if q is None or q.status != "open":
            return False, [f"  no open question {inquiry_id} (/inquiries lists them)"]
        a = answer.strip()
        low = a.lower().rstrip(".!")
        if not a:
            return False, [f"  usage: /reply {inquiry_id} <answer>"]
        if low in SKIP:
            self._close(q, "dropped", a, by)
            return True, ["  ok, I'll leave it"]
        out: list[str] = []
        if q.kind == "clarify":
            record = s.store.get_order(q.order_id)
            original = record["text"] if record else ""
            told = re.match(r"^\s*(?:answer|the answer is)\s*[:=-]?\s*(.+)$", a, re.I | re.S)
            if told and original and s.memory.allows("people"):  # they told it the answer: remember it
                made = s.memory.learn(original, told.group(1), f"you told me ({q.id})", by=by, kind="people")
                self._close(q, "answered", a, by)
                self.unmet.pop(q.order_id, None)
                if made is None:
                    return False, ["  thanks - but I couldn't keep that as an answer"]
                return True, [f"  thanks - I'll remember that: compressing it into memory node {made[0]} "
                              f"(tests in {s.memory.tests_dir.name}/{made[0]}.toml)"]
            labels = s.dispatch.labels
            route = next((r for r in (a, f"tool:{low}", f"skill:{low}") if r in labels), None)
            if route and original:
                if route in s.dispatch.inactive_labels:
                    return False, [f"  {route} is not available right now"]
                s.teach(DISPATCH, original, route, reason="clarification")
                name = route.split(":", 1)[-1]
                text = (f"tool {name}: {original}" if route.startswith("tool:") else
                        f"{name}: {skill_payload(original)}" if route.startswith("skill:") else original)
                out.append(f"  thanks - I've learned that orders like that go to {route}; trying again:")
            else:
                text = a
                out.append("  thanks - trying it your way:")
            self._close(q, "answered", a, by)
            res = await s.submit(text, source=by)
            out.append(f"  [{res.order_id}] " + res.summary().replace("\n", "\n   "))
            if res.ok:
                self.unmet.pop(q.order_id, None)
            return True, out
        if q.kind == "interests":
            topics = [t.strip() for t in re.split(r",|;|\band\b", a) if t.strip()][:8]
            made = []
            for t in topics:
                self.note("fact", "user", f"interested in {t}", private=False, source=f"you told me ({q.id})")
                try:  # your answer sets a standing learn task for each topic
                    made.append(s.tasks.add(f"learn {t}", by=by).id)
                except ValueError:
                    self.add_goal(t, f"you are interested in {t}", "interest")
            self._close(q, "answered", a, by)
            tasks = f" ({', '.join(made)}; /tasks)" if made else ""
            return True, [f"  thanks - I'll keep reading up on {', '.join(topics)} in the background{tasks}"]
        if q.kind in ("labels", "proposal"):  # deciding the proposal closes this question
            if low in NO:
                ok, msg = self.decline(q.proposal_id, by, f"declined in reply to {q.id}")
                return ok, [f"  {msg}"]
            head, _, tail = a.partition(":")
            given = tail if head.strip().lower() in ("yes", "y", "ok", "sure") else ("" if low in ("yes", "y", "ok", "sure") else a)
            labels = [x.strip() for x in re.split(r"[,\s]+", given) if x.strip()] or None
            ok, msg = self.adopt(q.proposal_id, by, labels)
            return ok, [f"  {msg}"]
        if q.kind == "objective":  # a question an objective asked: its answer goes to the objective (and memory)
            ok, msg = s.objectives.reply(q.about, a, q.id, by)
            self._close(q, "answered", a, by)
            return ok, [f"  {msg}"]
        if q.kind == "approval":
            approve = low in ("yes", "y", "ok", "sure", "approve", "approved")
            if not approve and low not in NO:
                return False, [f"  please answer yes or no (/reply {q.id} yes)"]
            decided = s.guardian.decide(q.approval_id, approve, by=by, note=f"via {q.id}")
            if q.status == "open":
                self._close(q, "answered", a, by)
            if decided is None:
                return False, [f"  {q.approval_id} was already decided"]
            return True, [f"  {q.approval_id} {decided.status}"]
        self._close(q, "answered", a, by)
        return True, ["  noted"]

    # =========================================================== proposals
    def open_proposals(self) -> list[Proposal]:
        return [p for p in self.proposals.values() if p.status == "open"]

    def propose(self, kind: str, name: str, *, description: str, reason: str, labels=(), triggers=(),
                evidence=(), target: str = "", by: str = "heuristic") -> Proposal | None:
        if any((p.kind, p.name, p.target) == (kind, name, target) for p in self.proposals.values()):
            return None  # proposed before (declined ones are not re-proposed)
        s = self.sys
        p = Proposal(f"prop-{next(self._prop_ids)}", kind, name, description, list(labels), list(triggers),
                     list(evidence)[:10], target, reason, by, created_at=self.clock())
        self.proposals[p.id] = p
        s.store.save_record("proposals", p.id, p.status, asdict(p))
        s.guardian.audit("mind", "proposal.created", p.id, proposal_kind=kind, name=name, target=target, by=by)
        s.bus.publish("mind.proposal", id=p.id, kind=kind, name=name, target=target, reason=reason)
        n = len(p.evidence)
        example = f" (e.g. '{' '.join(str(p.evidence[0]).split())[:60]}')" if p.evidence and kind != "label" else ""
        if kind == "skill" and len(p.labels) < 2:
            self.ask("labels", f"{n} order{'s' if n != 1 else ''} asked for a '{name}' skill I don't have{example}. "
                     f"If I should learn it, what labels should it have? (comma-separated, or 'no')",
                     reason, proposal_id=p.id)
        elif kind == "skill":
            self.ask("proposal", f"I'd like to add a '{name}' skill ({description}) with labels "
                     f"{', '.join(p.labels)}. Add it? (yes / no / yes: <other labels>)", reason, proposal_id=p.id)
        elif kind == "label":
            self.ask("proposal", f"Someone tried to use the label '{name}' for {target}, which doesn't have it. "
                     f"Add it? (yes/no)", reason, proposal_id=p.id)
        else:
            self.ask("proposal", f"I couldn't do {n} request{'s' if n != 1 else ''} like that{example}. I'd like a "
                     f"'{name}' tool ({description}). Shall I write a stub for you to implement? (yes/no)",
                     reason, proposal_id=p.id)
        return p

    def adopt(self, proposal_id: str, by: str, labels: list[str] | None = None) -> tuple[bool, str]:
        """Only ever called on a human's word (P10)."""
        s = self.sys
        p = self.proposals.get(proposal_id)
        if p is None or p.status != "open":
            return False, f"no open proposal {proposal_id}"
        if p.kind == "skill":
            labs = labels or p.labels
            if len(labs) < 2:
                return False, f"'{p.name}' needs at least two labels: /adopt {p.id} <label>,<label>,..."
            try:
                s.define_capability(p.name, labs, p.description, p.triggers or [p.name])
            except ValueError as exc:
                return False, str(exc)
            for text in p.evidence:
                s.teach(DISPATCH, text, f"skill:{p.name}", reason="adopted proposal")
                s.teachers.human.ask(p.name, skill_payload(text), f"from the orders behind {p.id}")
            p.labels = list(labs)
            msg = (f"added the '{p.name}' skill ({', '.join(labs)}); routed {len(p.evidence)} earlier orders to it "
                   f"and queued them for labelling (/questions)")
        elif p.kind == "label":
            try:
                s.define_capability(p.target, [p.name])
            except ValueError as exc:
                return False, str(exc)
            fixed = sum(1 for oid in p.evidence if s.feedback(oid, f"{p.target}={p.name}")[0])
            msg = f"added '{p.name}' to {p.target}" + (f" and applied {fixed} correction{'s' if fixed != 1 else ''}"
                                                       if fixed else "")
        else:
            smith = getattr(s, "toolsmith", None)
            made = smith.build_now([str(e) for e in p.evidence], f"{by} adopted {p.id}", "toolsmith") \
                if smith is not None and smith.enabled else None
            if made is not None:  # its toolsmith can write it: it still asks before the tool runs (P15)
                msg = (f"my toolsmith wrote {made['name']} v{made['version']} for those requests; it runs only once you "
                       f"approve it: /approve {made['approval']} (code and tests in {made['folder']})")
            else:
                folder = self.write_stub(p)
                msg = (f"wrote a stub to {folder}: implement tool.py and its self-tests, then move the folder "
                       f"into plugins/ - it will be vetted like any other tool (P3)")
        self._decide(p, "adopted", by, msg)
        return True, msg

    def decline(self, proposal_id: str, by: str, note: str = "") -> tuple[bool, str]:
        p = self.proposals.get(proposal_id)
        if p is None or p.status != "open":
            return False, f"no open proposal {proposal_id}"
        self._decide(p, "declined", by, note)
        return True, f"declined {p.id} ({p.kind} '{p.name}'); I won't propose it again"

    def _decide(self, p: Proposal, status: str, by: str, note: str) -> None:
        s = self.sys
        p.status, p.decided_by, p.decided_at, p.note = status, by, self.clock(), note[:300]
        s.store.save_record("proposals", p.id, p.status, asdict(p))
        s.guardian.audit(by, f"proposal.{status}", p.id, proposal_kind=p.kind, name=p.name, target=p.target)
        s.bus.publish(f"proposal.{status}", id=p.id, kind=p.kind, name=p.name, by=by)
        for q in self.open_inquiries():
            if q.proposal_id == p.id:
                self._close(q, "answered", f"{status} by {by}", by)

    def write_stub(self, p: Proposal) -> Path:
        folder = self.sys.state_dir / "proposals" / p.name
        folder.mkdir(parents=True, exist_ok=True)
        evidence = "\n".join(f"#   {' '.join(str(e).split())[:100]}" for e in p.evidence)
        keywords = ", ".join(f'"{k}"' for k in (p.triggers or [p.name]))
        (folder / "manifest.toml").write_text(
            f"# Proposed by NeuroMesh ({p.id}): {p.description}\n"
            f"# Orders it could not fulfil:\n{evidence}\n"
            "# Finish tool.py, add self-tests, then move this folder into plugins/.\n"
            "# The Explorer vets it like any other tool (P3); declare any permissions it needs.\n\n"
            f'name = "{p.name}"\nversion = "0.1.0"\ndescription = "{p.description.replace(chr(34), chr(39))}"\n'
            f"permissions = []\nkeywords = [{keywords}]\n\n"
            "# self-tests: at least one is required before the tool is activated\n"
            '# [[examples]]\n# input = "..."\n# expect = "..."\n', encoding="utf-8")
        (folder / "tool.py").write_text(
            '"""Stub written by NeuroMesh for a proposed tool. Read {"input": ...} from stdin and print\n'
            '{"output": ...} or {"error": ...}."""\nimport json\nimport sys\n\n'
            'request = json.load(sys.stdin)\n'
            'print(json.dumps({"error": "not implemented yet"}))\n', encoding="utf-8")
        return folder

    # ===================================================== reflection
    def add_goal(self, text: str, reason: str, kind: str = "curiosity", order_id: str = "") -> bool:
        if not (self.enabled and self.pol.investigate):
            return False
        key = text.strip().lower()
        if not key or any(g.text.strip().lower() == key for g in self.goals):
            return False
        self.goals.append(Goal(text.strip(), reason, kind, order_id, self.clock()))
        while len(self.goals) > 50:
            self.goals.popleft()
        return True

    def due_unmet(self) -> list[Unmet]:
        now = self.clock()
        return [u for u in self.unmet.values() if u.attempts < self.pol.max_unmet_attempts and u.next_try <= now]

    async def reflect(self) -> None:
        """Re-form opinions and turn evidence into proposals."""
        if not self.enabled:
            return
        self.form_opinions()
        s = self.sys
        known = sorted(set(s.capabilities) | set(s.tools.tools))
        for name, texts in list(self.unknown_skills.items()):
            if len(texts) >= self.pol.min_skill_requests and name not in s.capabilities:
                drafted = await s.teachers.propose(texts[:8], known, hint=f"a skill named '{name}'")
                labels, desc, by = [], f"what '{name}:' orders ask for", "heuristic"
                if drafted and drafted[0].get("kind") == "skill":
                    labels, desc, by = drafted[0].get("labels", []), drafted[0].get("description") or desc, drafted[1]
                self.propose("skill", name, description=desc, labels=labels, triggers=[name], evidence=texts,
                             reason=f"{len(texts)} orders named a '{name}' skill that doesn't exist", by=by)
                del self.unknown_skills[name]
        for cap, label, order_id in self.rejected_labels:
            existing = next((p for p in self.proposals.values()
                             if (p.kind, p.name, p.target) == ("label", label, cap)), None)
            if existing is not None:
                if existing.status == "open" and order_id not in existing.evidence:
                    existing.evidence.append(order_id)
                    s.store.save_record("proposals", existing.id, existing.status, asdict(existing))
                continue
            elsewhere = any(label in c.labels for c in s.capabilities.values())  # a mix-up, not a missing label
            if (cap in s.capabilities and label not in s.capabilities[cap].labels and not elsewhere
                    and not s.capabilities[cap].internal):
                self.propose("label", label, target=cap, description=f"a '{label}' label for {cap}",
                             evidence=[order_id], reason=f"a human used '{label}' for {cap}", by="human feedback")
        self.rejected_labels.clear()
        await self._propose_from_unmet(known)

    async def _propose_from_unmet(self, known: list[str]) -> None:
        """Orders nothing could fulfil that share a theme become a proposal."""
        smith = getattr(self.sys, "toolsmith", None)  # what its code library can meet, the toolsmith writes (P15)
        gaps = [u for u in self.unmet.values() if u.route == DIRECT and not u.proposed
                and not (smith is not None and smith.claims(u.text))]
        if len(gaps) < self.pol.min_evidence:
            return
        taken = set(known) | {p.name for p in self.proposals.values()}
        by_word: dict[str, list[Unmet]] = {}
        for u in gaps:
            for w in content_words(u.text):
                if w not in GENERIC and not w.isdigit() and len(w) > 2:
                    by_word.setdefault(w, []).append(u)
        ranked = sorted(((w, us) for w, us in by_word.items() if len(us) >= self.pol.min_evidence and w not in taken),
                        key=lambda kv: -len(kv[1]))
        if not ranked:
            return
        word, group = ranked[0]
        texts = [u.text for u in group]
        drafted = await self.sys.teachers.propose(texts[:8], known)
        if drafted and drafted[0].get("name") and drafted[0]["name"] not in taken:
            d, by = drafted
            kind = d["kind"]
            labels = d.get("labels", []) if kind == "skill" else []
            p = self.propose(kind, d["name"], description=d.get("description", ""), labels=labels,
                             triggers=d.get("triggers") or [word], evidence=texts,
                             reason=f"{len(texts)} orders about '{word}' could not be fulfilled", by=by)
        else:
            p = self.propose("tool", word, description=f"for '{word}' requests",
                             triggers=[word], evidence=texts,
                             reason=f"{len(texts)} orders about '{word}' could not be fulfilled")
        if p is not None:
            for u in group:
                u.proposed = True

    # ============================================================ investigations
    def record_investigation(self, goal: Goal, steps: list[str], finding: str, answer: str | None) -> Note:
        """finding = "" when nothing useful turned up (only the private working notes are kept)."""
        """What an investigation found (shared) and how it went about it (private working notes)."""
        self.investigations += 1
        note = self.note("observation", f"goal:{goal.text[:80]}", "; ".join(steps)[:1500],
                         source=f"investigation ({goal.kind}): {goal.reason}")
        if finding:
            note = self.note("finding", f"topic:{goal.text[:80]}", finding, private=False,
                             source=f"my own investigation ({goal.reason})", order_id=goal.order_id)
        self.sys.bus.publish("mind.finding", goal=goal.text[:120], kind=goal.kind,
                             finding=finding[:200] or "nothing useful yet", order=goal.order_id)
        if goal.order_id and answer:
            short = " ".join(goal.text.split())[:80]
            asked = f"what I think of {short}" if goal.kind == "opinion" else f"'{short}'"
            text = f"{answer}\n(Follow-up: earlier you asked {asked} ({goal.order_id}); I looked into it on my own.)"
            self.note("followup", f"order:{goal.order_id}", text, private=False, source="my own investigation",
                      order_id=goal.order_id)
            self.unmet.pop(goal.order_id, None)
            self.sys.bus.publish("mind.followup", order=goal.order_id, text=text[:300])
        return note

    # ================================================================ status
    def status(self) -> dict:
        kinds = Counter(n.kind for n in self.notes.values())
        return {"enabled": self.enabled, "note_counts": dict(kinds), "private": len(self.private_notes()),
                "opinions": len(self.opinions()),
                "inquiries": [asdict(q) for q in self.open_inquiries()],
                "proposals": [asdict(p) for p in self.open_proposals()],
                "goals": [g.text for g in self.goals], "unmet": len(self.unmet),
                "investigations": {**self.sys.guardian.investigation_usage(), "total": self.investigations}}
