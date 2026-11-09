"""The conversation it is having with you, and its short-term memory (P25).

A conversation is the run of turns between a person and the mesh: one per conversation id (the web page sends
its own; otherwise the order's source - "web", "console", "api"). After [conversation] idle_hours of quiet it
ends, and the next message starts a new one.

Each turn of a conversation:
    follow-ups   read against the last turn: "what about Spain?" after "what is the capital of France?" is
                 "what is the capital of Spain?"; "how tall is it?" after a question about the Eiffel Tower is
                 "how tall is the Eiffel Tower?"; "when was he born?" asks about the person the last answer named
    told         what you tell it - "remember that ...", "note: ...", "my dog is called Rex", "I live in Leeds",
                 "in my story the dragon Vex lives on the moon", "the meeting is on Friday" - goes into a short-term
                 memory node: one small node per thing, kept exactly as you said it (short-term memory is never
                 compressed), marked fact or fiction (realm.py), personal or not, passing or lasting, a secret or not.
                 Something that says the same thing again replaces the older node ("superseded").
    asked        a question is answered from short-term memory first: "what is my dog called?" -> "Rex - you
                 told me at 14:02."

When a conversation has been quiet for idle_hours, its ego looks at every short-term node and decides
(Ego.keep: its keep node, taught by the policy below and by what people choose):
    keep         re-designated as long-term memory: learned into a memory node (compressed, tested, reviewed by a
                 director, P14) as fact or as fiction - "my dog is called Rex" becomes "what is my dog called?" ->
                 "Rex"; your story stays your story
    let go       forgotten: a secret is never kept (P25), nor what you corrected, nor what was only about that
                 moment ("I'm tired", "today ..."), nor what it already knows
You can keep or forget any of them yourself, now (the web page's Memory, /stm keep|forget <id>), and end a
conversation early (/conversation end).
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from .agents import Agent
from .teachers import Lesson, Teacher

KEEP = "keep"
KEEP_LABELS = ("keep", "let_go")
KEEP_FEATURES = ("kkind", "ksecret", "kstate", "klasting", "kasked", "kknown", "kliked")
KEEP_VALUES = {"kkind": ("personal", "fact", "fiction", "chatter"), "ksecret": ("no", "yes"),
               "kstate": ("current", "corrected"), "klasting": ("lasting", "passing"), "kasked": ("no", "yes"),
               "kknown": ("no", "yes"), "kliked": ("neutral", "liked", "disliked")}
INTERNAL = ("challenge:", "objective", "task:", "teacher:", "tutor", "director", "dream", "agency", "investigator",
            "researcher", "seeker", "toolsmith", "explorer", "evaluation", "probe")

_EXPLICIT = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would|will) you\s+)?(?:remember|note|keep in mind|bear in mind|"
                       r"don't forget|do not forget|make a note|take note|for the record|fyi|f\.y\.i\.|just so you know)"
                       r"(?:\s+(?:that|this))?\s*[:,-]?\s+(.+)$", re.I | re.S)
_SECRET = re.compile(r"\b(?:password|passcode|pass code|pin(?: code| number)?|passphrase|api[ -]?key|secret key|access token|"
                     r"token|security code|cvv|credit card|card number|bank account|account number|sort code|routing number|"
                     r"social security|ssn|national insurance|passport number|private key|seed phrase|recovery phrase|"
                     r"combination(?: to| for)?)\b", re.I)
_PASSING = re.compile(r"\b(?:today|tonight|this (?:morning|afternoon|evening|week)|right now|at the moment|just now|"
                      r"currently|for now|in a minute|in a bit|later today|tomorrow|yesterday)\b|"
                      r"^\s*(?:i'?m|i am|i feel|feeling)\s+(?:so\s+|very\s+|a bit\s+|really\s+|quite\s+)?(?:tired|hungry|"
                      r"bored|sleepy|sad|happy|angry|annoyed|stressed|busy|cold|hot|sick|ill|fine|ok|okay|good|great|"
                      r"back|here|off|done|ready|on my way|leaving|going)\b", re.I)
_PATTERNS = [  # (regex, subject, value, the question it answers) - first match wins
    (re.compile(r"^my ([\w' -]{2,40}?)'?s name is (.+)$", re.I), "my {0}", "{1}", "what is my {0} called"),
    (re.compile(r"^my ([\w' -]{2,40}?) (?:is|are) (?:called|named) (.+)$", re.I), "my {0}", "{1}", "what is my {0} called"),
    (re.compile(r"^my name is (.+)$", re.I), "my name", "{0}", "what is my name"),
    (re.compile(r"^(?:i'?m|i am) called (.+)$", re.I), "my name", "{0}", "what is my name"),
    (re.compile(r"^my ([\w' -]{2,40}?) (?:is|are|was|were) (.+)$", re.I), "my {0}", "{1}", "what is my {0}"),
    (re.compile(r"^i live (?:in|at|on|near) (.+)$", re.I), "where I live", "{0}", "where do I live"),
    (re.compile(r"^i (?:work|am working) (?:at|for) (.+)$", re.I), "where I work", "{0}", "where do I work"),
    (re.compile(r"^i (?:work|am working) as (?:an? )?(.+)$", re.I), "my job", "{0}", "what is my job"),
    (re.compile(r"^i(?:'m| am) (?:an? )?(.+?) by (?:trade|profession)$", re.I), "my job", "{0}", "what is my job"),
    (re.compile(r"^i was born (?:in|on) (.+)$", re.I), "where or when I was born", "{0}", "where was I born"),
    (re.compile(r"^i (?:like|love|enjoy|prefer) (.+)$", re.I), "what I like", "{0}", "what do I like"),
    (re.compile(r"^i (?:dislike|hate|can't stand|cannot stand|don't like|do not like) (.+)$", re.I), "what I dislike", "{0}",
     "what do I dislike"),
    (re.compile(r"^i (?:have|own|have got|'ve got) (.+)$", re.I), "what I have", "{0}", "what do I have"),
    (re.compile(r"^(?:in|from) (?:my|our) (?:own )?(?:story|novel|book|game|campaign|world|screenplay|script|tale)[, ]+"
                r"(?:the )?([\w' -]{2,40}?) (is|are|lives|live|was|were|has|have) (.+)$", re.I), "{0}", "{2}", "story:{1}"),
    (re.compile(r"^(?:i'?m|i am|i feel|i'm feeling|i am feeling) (?!called\b)(.+)$", re.I), "how I am", "{0}", "how am I"),
    (re.compile(r"^(?:the |our |this |that )?([\w' -]{2,40}?) (?:is|are|was|were|will be) (?:on|at|in) (.+)$", re.I),
     "the {0}", "{1}", "when or where is the {0}"),
    (re.compile(r"^(?:the |our )?([\w' -]{2,40}?) (?:is|are|was|were) (?:called|named) (.+)$", re.I), "the {0}", "{1}",
     "what is the {0} called"),
    (re.compile(r"^(?:the |our |a |an )?([\w' -]{2,40}?) (?:is|are|was|were) (.+)$", re.I), "{0}", "{1}", "what is {0}"),
]
_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "what", "whats", "who", "whos", "where", "when", "which", "how",
         "do", "does", "did", "i", "you", "it", "of", "to", "in", "on", "at", "for", "called", "named", "name", "s", "me",
         "tell", "remind", "again", "please", "can", "could", "your", "and", "or", "that", "this", "there"}
_PRONOUN_IT = re.compile(r"\b(it|its|it's|that one|this one|there)\b", re.I)
_PRONOUN_PERSON = re.compile(r"\b(he|she|him|her|his|hers)\b", re.I)
_PRONOUN_THEY = re.compile(r"\b(they|them|their|theirs)\b", re.I)
_WHAT_ABOUT = re.compile(r"^\s*(?:and|but|so|ok|okay|then)?[\s,]*(?:what|how) about\s+(.+?)[\s?.!]*$", re.I)
_ENTITY = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+(?:of|the|de|du|von|van|la|le|and)?\s*[A-Z][\w'’.-]*)*")
_QWORDS = {"What", "Who", "When", "Where", "Why", "How", "Which", "Is", "Are", "Was", "Were", "Do", "Does", "Did", "Can",
           "Could", "Tell", "Explain", "Describe", "And", "But", "So", "Ok", "Okay", "Please", "I", "In", "On", "The", "A", "An"}


def you_said(text: str) -> str:
    """Your words said back: my -> your, I -> you."""
    swaps = {"my": "your", "mine": "yours", "i": "you", "i'm": "you're", "im": "you're", "me": "you", "myself": "yourself",
             "i've": "you've", "i'll": "you'll", "i'd": "you'd", "our": "your", "we": "you", "us": "you", "am": "are",
             "was": "were"}
    words = str(text).split()
    out = []
    for i, w in enumerate(words):
        core = w.strip(".,!?;:").lower()
        if core in swaps and (core not in ("was", "am") or (i and words[i - 1].strip(".,!?;:").lower() in ("i", "i'm"))):
            rep = swaps[core]
            out.append(w.lower().replace(core, rep, 1) if w[:1].islower() or i else rep.capitalize())
        else:
            out.append(w)
    return " ".join(out)


def parse_statement(text: str) -> tuple[str, str, str] | None:
    """(subject, value, the question it answers) for something a person told it, or None."""
    t = " ".join(str(text).split()).strip().rstrip(".!")
    if not t or len(t) > 300 or "?" in t:
        return None
    for rx, subj, val, ask in _PATTERNS:
        m = rx.match(t)
        if m:
            g = [x.strip(" ,.") for x in m.groups()]
            if not all(g) or len(g[-1]) > 200:
                continue
            if ask.startswith("story:"):  # in your story: the question follows the verb
                verb = g[1].lower()
                q = {"lives": "where does {0} live", "live": "where do {0} live", "has": "what does {0} have",
                     "have": "what do {0} have", "was": "what was {0}", "were": "what were {0}",
                     "are": "what are {0}"}.get(verb, "what is {0}")
                return subj.format(*g), val.format(*g), q.format(f"the {g[0]}" if not g[0][:1].isupper() else g[0])
            return subj.format(*g), val.format(*g), ask.format(*g)
    return None


_TOLD_NOW = re.compile(r"^\s*(?:my name is|i'?m called|i am called|my [\w' -]{2,40}?(?:'s name is| is called| is named| are called|"
                       r" are named)|i live (?:in|at|on|near)|i (?:work|am working) (?:at|for|as)|i was born (?:in|on)|"
                       r"(?:in|from) (?:my|our) (?:own )?(?:story|novel|book|game|campaign|world|screenplay|script|tale))\b", re.I)


def statement_like(text: str) -> bool:
    """A sentence that tells rather than asks or orders."""
    t = " ".join(str(text).split())
    if not t or "?" in t or len(t) > 300 or ":" in t[:30]:
        return False
    if re.match(r"^\s*(?:please|can|could|would|will|let's|let us|show|give|make|write|find|list|tell|explain|describe|"
                r"calculate|compute|count|translate|summari[sz]e|imagine|draw|play|run|open|search|look|go|stop|start|"
                r"help|thanks|thank|hi|hello|hey|ok|okay|yes|no|sure|great|cool)\b", t, re.I):
        return False
    return parse_statement(t) is not None


def entities(text: str) -> list[str]:
    out = []
    for m in _ENTITY.finditer(str(text)):
        e = m.group(0).strip(" .")
        words = e.split()
        while words and words[0] in _QWORDS:
            words = words[1:]
        e = " ".join(words)
        if e and len(e) > 1 and e not in _QWORDS:
            out.append(e)
    return out


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower().replace("'s", "")) if w not in _STOP and len(w) > 1}


# ------------------------------------------------------------------ the keep node's policy and teacher
def keep_text(f: dict) -> str:
    return "keep " + " ".join(f"{k}{f.get(k, KEEP_VALUES[k][0])}" for k in KEEP_FEATURES)


def parse_keep(text: str) -> dict:
    ws = str(text).split()
    if not ws or ws[0] != "keep":
        return {}
    out = {}
    for w in ws[1:]:
        for k in KEEP_FEATURES:
            if w.startswith(k) and w[len(k):] in KEEP_VALUES[k]:
                out[k] = w[len(k):]
                break
    return out


def keep_policy(f: dict) -> tuple[str, str]:
    """Whether to keep something it was told, once the conversation has gone quiet."""
    if f.get("ksecret") == "yes":
        return "let_go", "it was a secret, and secrets are never kept"
    if f.get("kstate") == "corrected":
        return "let_go", "you corrected it later in the conversation"
    if f.get("kliked") == "disliked":
        return "let_go", "people discouraged it keeping things like this"
    if f.get("kasked") == "yes":
        return "keep", "you asked about it again, so it matters to you"
    if f.get("klasting") == "passing":
        return "let_go", "it was about that moment, not something lasting"
    if f.get("kknown") == "yes":
        return "let_go", "its long-term memory already holds it"
    if f.get("kliked") == "liked":
        return "keep", "people encouraged it keeping things like this"
    if f.get("kkind") == "personal":
        return "keep", "it is about you, and lasting"
    if f.get("kkind") == "fiction":
        return "keep", "it is from your story: kept as fiction"
    if f.get("kkind") == "fact":
        return "keep", "you told it something lasting it didn't know"
    return "let_go", "it was conversation, not something to remember"


class KeepTeacher(Teacher):
    """The keep policy as a teacher, for the keep node only."""
    name = KEEP
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def status(self) -> dict:
        return {"detail": "specialist: teaches the keep node what to keep from a conversation"}

    async def label(self, spec, text):
        if spec.name != KEEP:
            return None
        f = parse_keep(text)
        if len(f) < len(KEEP_FEATURES):
            return None
        label, why = keep_policy(f)
        return Lesson(label, 0.9, self.name, why)

    async def generate(self, spec, label, n):
        if spec.name != KEEP or label not in KEEP_LABELS:
            return []
        out: list[str] = []
        for _ in range(n * 300):
            f = {k: self.rng.choice(v) for k, v in KEEP_VALUES.items()}
            if keep_policy(f)[0] == label:
                t = keep_text(f)
                if t not in out:
                    out.append(t)
            if len(out) >= n:
                break
        return out


# ------------------------------------------------------------------ conversations
@dataclass
class Conversation:
    id: str
    key: str
    started: float
    last: float
    turns: list = field(default_factory=list)     # [{"ts", "said", "read_as", "answer", "order", "entities"}]
    items: list = field(default_factory=list)     # short-term memory node ids
    status: str = "open"                          # open | ended
    ended_at: float = 0.0
    outcome: dict = field(default_factory=dict)   # what its ego kept and let go when it ended
    created_at: float = 0.0


class Conversations:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.conversation
        self.convs: dict[str, Conversation] = {}
        for d in system.store.load_records("conversations")[-200:]:
            try:
                c = Conversation(**d)
            except TypeError:
                continue
            self.convs[c.id] = c
        self.short: dict[str, dict] = {d["id"]: d for d in system.store.load_records("short_memory") if d.get("id")}
        st = json.loads(system.store.get_kv("conversation.state") or "{}")
        self.seq = int(st.get("seq", 0))
        self.stm_seq = int(st.get("stm_seq", 0))
        self.log: deque = deque(st.get("log", []), maxlen=60)

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def _save_state(self) -> None:
        self.sys.store.set_kv("conversation.state", json.dumps({"seq": self.seq, "stm_seq": self.stm_seq,
                                                                "log": list(self.log)}))

    def _save(self, c: Conversation) -> None:
        self.sys.store.save_record("conversations", c.id, c.status, asdict(c))

    def _save_item(self, item: dict) -> None:
        self.sys.store.save_record("short_memory", item["id"], item["status"], item)

    @staticmethod
    def conversational(order) -> bool:
        return not order.background and not str(order.source).startswith(INTERNAL)

    def key_of(self, order) -> str:
        return (getattr(order, "conversation", "") or str(order.source) or "user")[:80]

    def current(self, key: str, now: float | None = None) -> Conversation | None:
        now = now or time.time()
        c = next((c for c in sorted(self.convs.values(), key=lambda c: -c.last) if c.key == key and c.status == "open"), None)
        return c

    def _open(self, key: str, now: float) -> Conversation:
        c = self.current(key, now)
        if c is not None and now - c.last >= self.pol.idle_hours * 3600:
            self.end(c, "it went quiet")  # a new conversation; the old one's short-term memory is settled first
            c = None
        if c is None:
            self.seq += 1
            c = Conversation(f"conv-{self.seq}", key, now, now, created_at=now)
            self.convs[c.id] = c
            self._save_state()
            self.sys.bus.publish("conversation.started", conversation=c.id, key=key)
        return c

    # ------------------------------------------------------------------ on the way in
    def before(self, order) -> dict | None:
        """A turn begins: which conversation, and what a follow-up means. Rewrites order.text for a follow-up."""
        if not self.enabled or not self.conversational(order):
            return None
        now = time.time()
        c = self._open(self.key_of(order), now)
        said = order.text
        read_as, why = self.resolve(c, said) if self.pol.follow_ups else (None, "")
        if read_as and read_as != said:
            order.text = read_as
        c.last = now
        ctx = {"id": c.id, "turn": len(c.turns) + 1, "said": said, "read_as": read_as or "", "why": why}
        order.conversation_ctx = ctx
        return ctx

    def resolve(self, c: Conversation, text: str) -> tuple[str | None, str]:
        """A follow-up, read against the last turn: (the question in full, why), or (None, "")."""
        last = next((t for t in reversed(c.turns) if t.get("question")), None)
        if last is None or time.time() - last["ts"] > 3600:
            return None, ""
        t = " ".join(str(text).split())
        from .agents import is_question
        if not is_question(t) and not _WHAT_ABOUT.match(t):
            return None, ""
        prev = last["read_as"] or last["said"]
        prev_ents = last.get("entities") or entities(prev)
        m = _WHAT_ABOUT.match(t)
        if m and prev_ents:
            new = m.group(1).strip()
            old = prev_ents[-1]
            if old.lower() in prev.lower():
                full = re.sub(re.escape(old), new, prev, count=1, flags=re.I)
                full = re.sub(r"\bthe the\b", "the", full, flags=re.I)
                return full, f"'{t}' follows on from '{prev}': {old} -> {new}"
        if entities(t) or len(t.split()) > 12:
            return None, ""
        if _PRONOUN_PERSON.search(t):
            names = entities(last.get("answer", ""))[:3] or prev_ents
            person = next((n for n in names if len(n.split()) >= 1), None)
            if person:
                full = _PRONOUN_PERSON.sub(lambda mm: person + ("'s" if mm.group(1).lower() in ("his", "hers") else ""), t, count=1)
                return full, f"'{mm_word(t, _PRONOUN_PERSON)}' is {person}, from the last answer"
        for rx in (_PRONOUN_IT, _PRONOUN_THEY):
            if rx.search(t) and prev_ents:
                ent = prev_ents[-1]
                if re.search(r"\bthe\s+" + re.escape(ent), prev, re.I):  # "the Eiffel Tower", as it was asked
                    ent = "the " + ent
                word = mm_word(t, rx)
                if word.lower() == "there" and not re.search(r"\b(?:in|at|to|from)\s+" + re.escape(ent), prev, re.I):
                    continue
                full = rx.sub(lambda mm: (ent + "'s") if mm.group(1).lower() in ("its", "their", "theirs") else ent, t, count=1)
                return full, f"'{word}' is {ent}, from '{prev}'"
        return None, ""

    def after(self, order, res) -> None:
        """A turn ends: what was said and answered, kept as context."""
        ctx = getattr(order, "conversation_ctx", None)
        if ctx is None:
            return
        c = self.convs.get(ctx["id"])
        if c is None:
            return
        answer = (res.answer or str(res.output or res.note or "")).split("\nSource", 1)[0]
        from .agents import is_question
        q = is_question(order.text)
        c.turns.append({"ts": round(time.time(), 1), "said": ctx["said"][:400], "read_as": ctx["read_as"][:400],
                        "answer": answer[:400], "order": order.id, "question": q,
                        "entities": entities(order.text)[:6]})
        c.turns = c.turns[-max(2, self.pol.turns):]
        c.last = time.time()
        res.conversation = {**(res.conversation or {}), "id": c.id, "turn": len(c.turns),
                            **({"read_as": ctx["read_as"], "why": ctx["why"]} if ctx["read_as"] else {})}
        if ctx["read_as"]:
            res.trace.insert(0, f"conversation {c.id}: read as '{ctx['read_as']}' ({ctx['why']})")
        self._save(c)

    # ------------------------------------------------------------------ what you tell it
    def handle(self, order, res) -> bool:
        """Something you told it ("remember that ...", "my dog is called Rex"), or a question its short-term memory
        answers. True when this answered the turn."""
        ctx = getattr(order, "conversation_ctx", None)
        if ctx is None or not self.enabled:
            return False
        text = " ".join(order.text.split())
        m = _EXPLICIT.match(text)
        if m and self.pol.capture:
            return self.tell(order, res, m.group(1).strip(), explicit=True)
        from .agents import is_question
        if is_question(text):
            return self.answer(order, res)
        if self.pol.capture and _TOLD_NOW.match(text) and parse_statement(text) is not None:
            return self.tell(order, res, text)  # plainly something about you, or your story
        return False

    def told(self, order, res) -> bool:
        """After the dispatch node chose to answer it directly: a plain statement ("the meeting is on Friday") is
        something you told it, not a question for its teachers."""
        if not self.enabled or not self.pol.capture or getattr(order, "conversation_ctx", None) is None:
            return False
        if not statement_like(order.text):
            return False
        return self.tell(order, res, " ".join(order.text.split()))

    def tell(self, order, res, text: str, explicit: bool = False) -> bool:
        ctx = order.conversation_ctx
        c = self.convs.get(ctx["id"])
        if c is None:
            return False
        item = self.remember(c, text, by=str(order.source), order_id=order.id)
        realm = item["realm"]
        echo = you_said(re.sub(r"^\s*(?:in|from) (?:my|our) (?:own )?[\w&' ]{2,30}?(?:story|novel|book|game|campaign|world|"
                               r"screenplay|script|tale)[, ]+", "", text, flags=re.I) if realm == "fiction" else text)
        if item["secret"]:
            say = ("I'll keep that for this conversation only - it looks like a secret, so it is never written into my "
                   "long-term memory, and I'll let it go when we finish.")
        elif realm == "fiction":
            say = f"Noted, from {you_said(item['work']) if item['work'] else 'your story'}: {echo}. I'll keep it as fiction, not fact."
        else:
            say = f"Noted: {echo}."
        if item.get("replaces"):
            say += f" (That replaces what you told me before: \"{you_said(item['replaces_text'])}\".)"
        say += " It's in my short-term memory; if it still matters once we've been quiet a while, I'll decide whether to keep it."
        res.route, res.output, res.answer, res.confidence, res.served_by = "conversation", say, say, 1.0, "explicit"
        res.realm = {"realm": realm, "work": item["work"], "confidence": item["realm_confidence"], "how": item["realm_how"]}
        res.conversation = {**(res.conversation or {}), "id": c.id, "told": item["id"]}
        res.trace.append(f"kept in short-term memory node {item['id']} (verbatim, uncompressed; {realm}"
                         + (f", {item['work']}" if item["work"] else "") + (", a secret" if item["secret"] else "")
                         + (", passing" if item["transient"] else "") + ")")
        return True

    def remember(self, c: Conversation, text: str, by: str = "user", order_id: str = "") -> dict:
        """A short-term memory node: what you said, exactly; nothing is compressed."""
        parsed = parse_statement(text)
        subject, value, ask = parsed if parsed else ("", "", "")
        j = self.sys.realm.judge(text) if getattr(self.sys, "realm", None) is not None else None
        self.stm_seq += 1
        now = time.time()
        item = {"id": f"stm-{self.stm_seq}", "conversation": c.id, "text": text[:600], "subject": subject[:120],
                "value": value[:300], "ask": ask[:200], "realm": j.realm if j else "fact", "work": (j.work if j else "")[:120],
                "realm_confidence": j.confidence if j else 0.5, "realm_how": j.how if j else "",
                "personal": bool(re.match(r"^\s*(?:my|i|i'm|i am|i've|our|we)\b", text, re.I)),
                "secret": bool(_SECRET.search(text)), "transient": bool(_PASSING.search(text)),
                "by": by, "order": order_id, "created_at": round(now, 1), "used": 0, "last_used": 0.0,
                "status": "short", "why": "", "node": "", "decided_by": "", "replaces": "", "replaces_text": ""}
        for old in self.items(c.id, ("short",)):
            if subject and old["subject"].lower() == subject.lower():
                old["status"], old["why"] = "superseded", f"replaced by {item['id']}"
                item["replaces"], item["replaces_text"] = old["id"], old["text"]
                self._save_item(old)
        self.short[item["id"]] = item
        c.items.append(item["id"])
        self._save_item(item)
        self._save(c)
        self._trim()
        self.sys.bus.publish("memory.short", id=item["id"], conversation=c.id, realm=item["realm"],
                             text=("(a secret)" if item["secret"] else text[:160]))
        return item

    def _trim(self) -> None:
        live = sorted((i for i in self.short.values() if i["status"] == "short"), key=lambda i: i["created_at"])
        for item in live[:max(0, len(live) - self.pol.max_short)]:
            self._settle(item, "let go", "short-term memory was full; the oldest goes first", "its short-term memory")

    def items(self, conv_id: str | None = None, status: tuple = ("short",)) -> list[dict]:
        return [i for i in self.short.values() if (conv_id is None or i["conversation"] == conv_id) and i["status"] in status]

    # ------------------------------------------------------------------ what it was told, asked about
    def answer(self, order, res) -> bool:
        """A question its short-term memory answers ("what is my dog called?" -> "Rex - you told me at 14:02")."""
        hit = self.recall(order.text, order.conversation_ctx["id"])
        if hit is None:
            return False
        item, score = hit
        item["used"] += 1
        item["last_used"] = round(time.time(), 1)
        self._save_item(item)
        when = time.strftime("%H:%M", time.localtime(item["created_at"]))
        if item["secret"]:
            say = f"You told me at {when}, but it's a secret, so I won't repeat it here: look at what you said then."
        elif item["value"]:
            say = f"{_cap(you_said(item['value']))} - you told me at {when}"
            say += f" (in {you_said(item['work']) if item['work'] else 'your story'}: fiction, not fact)." if item["realm"] == "fiction" else "."
        else:
            say = f"You told me at {when}: \"{item['text']}\"" + (" (fiction, not fact)." if item["realm"] == "fiction" else ".")
        res.route, res.output, res.answer, res.confidence, res.served_by = "conversation", say, say, round(score, 3), "nodes"
        res.realm = {"realm": item["realm"], "work": item["work"], "confidence": item["realm_confidence"],
                     "how": f"you told it, as {item['realm']}"}
        res.conversation = {**(res.conversation or {}), "id": order.conversation_ctx["id"], "recalled": when, "item": item["id"]}
        res.sources.append(f"short-term memory node {item['id']} (what you said at {when}, kept verbatim)")
        res.trace.append(f"answered from short-term memory node {item['id']} ({score:.2f} match; uncompressed)")
        return True

    def recall(self, question: str, conv_id: str | None = None) -> tuple[dict, float] | None:
        """The short-term memory node that answers a question best, this conversation's first."""
        q = words(question)
        if not q:
            return None
        best, best_score = None, 0.0
        for item in self.short.values():
            if item["status"] not in ("short", "long") or (item["status"] == "long" and item.get("node")):
                continue  # long-term ones with a memory node are recalled by that node
            subj = words(item["subject"]) or words(item["text"])
            if not subj:
                continue
            if item["personal"] and not re.search(r"\b(?:i|my|me|mine|i'm|myself)\b", question, re.I):
                continue  # about you: only when you ask about yourself
            if item["subject"]:
                if not subj <= q:
                    continue
                score = 0.75 + 0.2 * len(subj & q) / max(1, len(q))
            else:
                overlap = len(subj & q) / max(1, len(subj | q))
                if overlap < 0.5:
                    continue
                score = overlap
            if item["conversation"] == conv_id:
                score += 0.05
            score += item["created_at"] / 1e12  # the newest of equals
            if score > best_score:
                best, best_score = item, score
        return (best, min(best_score, 0.99)) if best is not None else None

    # ------------------------------------------------------------------ the end of a conversation
    def due(self, now: float | None = None) -> list[Conversation]:
        now = now or time.time()
        return [c for c in self.convs.values() if c.status == "open" and now - c.last >= self.pol.idle_hours * 3600]

    def tick(self) -> int:
        n = 0
        for c in self.due():
            self.end(c, f"quiet for {self.pol.idle_hours:g} hours")
            n += 1
        return n

    def end(self, c: Conversation, why: str, by: str = "") -> dict:
        """The conversation is over: its ego decides, for every short-term node, whether to keep it."""
        kept, let_go = [], []
        for item in self.items(c.id, ("short",)):
            decision, reason, who = self.sys.ego.keep(item, self)
            if decision == "keep":
                self.keep(item, who, reason)
            else:
                self._settle(item, "let go", reason, who)
            (kept if item["status"] == "long" else let_go).append(item["id"])
        c.status, c.ended_at = "ended", round(time.time(), 1)
        c.outcome = {"kept": kept, "let_go": let_go, "why": why, "by": by or "its ego"}
        self._save(c)
        line = f"{c.id} ended ({why}): kept {len(kept)}, let go {len(let_go)}"
        self.log.append({"at": round(time.time(), 1), "line": line})
        self._save_state()
        if kept or let_go:
            said = [you_said(self.short[i]["text"])[:80] for i in kept[:4] if not self.short[i]["secret"]]
            self.sys.ego.remember("ego:kept", f"From conversation {c.id} I kept {len(kept)} thing(s)"
                                  + (f" ({'; '.join(said)})" if said else "") + f" and let {len(let_go)} go.",
                                  about="conversation", source=c.id)
        self.sys.guardian.audit(by or "ego", "conversation.ended", c.id, kept=kept, let_go=let_go, why=why)
        self.sys.bus.publish("conversation.ended", conversation=c.id, kept=len(kept), let_go=len(let_go), why=why)
        return c.outcome

    def keep(self, item: dict, by: str, why: str) -> bool:
        """Re-designated as long-term memory: learned into a memory node (compressed, tested), as fact or fiction."""
        if item["secret"]:
            self._settle(item, "let go", "it was a secret, and secrets are never kept", by)
            return False
        mem = self.sys.memory
        ask = item["ask"] or item["text"]
        answer = item["value"] or item["text"]
        node, how = "", ""
        if mem.enabled and mem.allows("people"):
            ask_mem = you_said_question(ask)
            made = mem.learn(ask_mem, you_said(answer), f"you, in conversation {item['conversation']} "
                             f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(item['created_at']))})",
                             by=item["by"], kind="people", realm=item["realm"], work=item["work"])
            if made is not None:
                node, how = made
        item.update(status="long", why=why, node=node, decided_by=by, kept_at=round(time.time(), 1),
                    long_ask=you_said_question(ask) if node else "")
        self._save_item(item)
        self.sys.bus.publish("memory.short_kept", id=item["id"], node=node, by=by, why=why[:120])
        return True

    def _settle(self, item: dict, status: str, why: str, by: str) -> None:
        item.update(status=status, why=why, decided_by=by, settled_at=round(time.time(), 1))
        if item["secret"] and status != "short":
            item["text"], item["value"] = "(a secret: forgotten)", ""
        self._save_item(item)
        self.sys.bus.publish("memory.short_settled", id=item["id"], status=status, by=by, why=why[:120])

    def decide(self, item_id: str, action: str, by: str) -> tuple[bool, str]:
        """A person keeps or forgets a short-term memory node now (and teaches the keep node)."""
        item = self.short.get(item_id)
        if item is None:
            return False, f"no short-term memory node {item_id}"
        if item["status"] != "short":
            return False, f"{item_id} is already {item['status']}"
        f = self.sys.ego.keep_situation(item, self)
        if action == "keep":
            if item["secret"]:
                return False, f"{item_id} looks like a secret: secrets are never kept in long-term memory"
            self.keep(item, by, f"{by} kept it")
            if KEEP in self.sys.capabilities:
                self.sys.learn(KEEP, keep_text(f), "keep", "human")
            return True, f"kept {item_id} in long-term memory" + (f" (memory node {item['node']})" if item["node"] else "")
        if action in ("forget", "let go", "let_go"):
            self._settle(item, "let go", f"{by} let it go", by)
            if KEEP in self.sys.capabilities:
                self.sys.learn(KEEP, keep_text(f), "let_go", "human")
            return True, f"forgot {item_id}"
        return False, "action is keep or forget"

    def end_now(self, conv_id: str, by: str) -> tuple[bool, str]:
        c = self.convs.get(conv_id) or next((c for c in self.convs.values() if c.key == conv_id and c.status == "open"), None)
        if c is None or c.status != "open":
            return False, f"no open conversation {conv_id}"
        out = self.end(c, f"{by} ended it", by)
        return True, f"ended {c.id}: kept {len(out['kept'])}, let go {len(out['let_go'])}"

    # ------------------------------------------------------------------ status
    def public_item(self, i: dict) -> dict:
        d = {k: v for k, v in i.items() if k not in ("realm_confidence",)}
        if i["secret"]:
            d["text"], d["value"] = "(a secret: hidden)", ""
        d["expires_at"] = 0.0
        c = self.convs.get(i["conversation"])
        if c is not None and c.status == "open" and i["status"] == "short":
            d["expires_at"] = round(c.last + self.pol.idle_hours * 3600, 1)
        return d

    def status(self) -> dict:
        convs = sorted(self.convs.values(), key=lambda c: -c.last)[:20]
        return {"enabled": self.enabled, "idle_hours": self.pol.idle_hours,
                "conversations": [{"id": c.id, "key": c.key, "started": c.started, "last": c.last, "turns": len(c.turns),
                                   "status": c.status, "items": len(c.items), "outcome": c.outcome,
                                   "recent": c.turns[-6:]} for c in convs],
                "short": [self.public_item(i) for i in sorted(self.short.values(), key=lambda i: -i["created_at"])[:200]],
                "counts": {s: sum(1 for i in self.short.values() if i["status"] == s)
                           for s in ("short", "long", "let go", "superseded")},
                "log": list(self.log)[-20:]}

    def lines(self) -> list[str]:
        st = self.status()
        out = []
        for c in st["conversations"][:5]:
            out.append(f"  {c['id']} ({c['key']}): {c['turns']} turns, {c['items']} things told, {c['status']}"
                       + (f"; kept {len(c['outcome'].get('kept', []))}, let go {len(c['outcome'].get('let_go', []))}"
                          if c["outcome"] else ""))
        for i in [i for i in st["short"] if i["status"] == "short"][:12]:
            left = max(0.0, i["expires_at"] - time.time()) / 3600 if i["expires_at"] else 0
            out.append(f"  {i['id']} [{i['realm']}{', ' + i['work'] if i['work'] else ''}] \"{i['text'][:70]}\""
                       + (f" - its ego decides in {left:.1f}h" if left else ""))
        return out or ["  no conversation yet"]


def you_said_question(ask: str) -> str:
    """'what is my dog called' stays as you would ask it (memory keeps your own words)."""
    return " ".join(str(ask).split())


def mm_word(text: str, rx) -> str:
    m = rx.search(text)
    return m.group(1) if m else ""


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


class Keeper(Agent):
    """Now and then ([conversation] check_every_seconds): conversations quiet for idle_hours end, and its ego
    decides what of each to keep (see the module docstring)."""
    name = "keeper"

    def __init__(self, system):
        super().__init__(system)
        self.last = 0.0

    @property
    def interval(self) -> float:
        return max(0.05, min(5.0, self.sys.directives.conversation.check_every_seconds / 4))

    async def step(self) -> None:
        s = self.sys
        if not s.conversations.enabled or s.guardian.halted:
            return
        if time.time() - self.last < s.directives.conversation.check_every_seconds:
            return
        self.last = time.time()
        ended = s.conversations.tick()
        if ended:
            self.stats["ended"] += ended
