"""The ego: nodes wrapped around all the others, and the mesh's sense of itself.

    order --> [ego, on the way in] --> directors, tools, nodes, memory, thinking ... --> [ego, on the way out] --> you
              reads its mood now                                                     a remark of its own, labelled

The ego is the outermost layer: every order passes it on the way in and again on the way out. On the
way in it reads the current mood (temperament.py) - the mood it takes the order in. On the way out the
ego node (an ordinary learnable capability, taught by the ego policy below) reads the situation - its mood
now, the tone of the message, how the answer went, what it makes of the topic, whether it dreamt about
it, whether it has a view of it, whether it spoke a moment ago - and decides whether to speak and how:

    quiet     say nothing: it spoke a moment ago, the answer was held back, or the person is upset (then
              it is not about the mesh)
    curious   it is curious about what was asked, and says why; the area becomes one its ego wants to
              understand, which raises it among the objectives the director sets itself (curiosity.py)
    dream     something it dreamt touches on what was asked: the dream, and whether it checked out
    opinion   its own view of the area - from how it has done there, what it holds, what its thinking
              nodes make of it, its dreams about it - coloured by its mood
    feeling   how it feels right now, and what moved it
    doubt     it doubts itself here: it could not answer, it was unsure, or it was told it was wrong before

The node only chooses the kind; the words are built from what is so (P8): the counts, the dream, the
mood and its causes are real, and a remark is always labelled as its ego's - shown after the answer,
never in it, never stored as a fact or used as a source (P18). It is never about a person: opinions
about people stay with the mind, and private (P9).

It also answers questions about itself from the same picture: "tell me about yourself", "how are you
feeling?", "what did you dream about?", "what do you like?".
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, deque

from .agents import is_question
from .curiosity import norm, question_core, topics_in
from .tasks import DIRECT
from .teachers import Lesson, Teacher

EGO = "ego"
KINDS = ("quiet", "curious", "dream", "opinion", "feeling", "doubt")
MOODS = ("steady", "upbeat", "content", "alert", "tense", "subdued", "wary")
FEATURES = ("mood", "tone", "outcome", "topic", "dreamt", "view", "spoke")
VALUES = {"mood": MOODS, "tone": ("neutral", "friendly", "frustrated", "hostile", "distressed"),
          "outcome": ("ok", "unsure", "failed", "held"), "topic": ("none", "known", "curious", "new"),
          "dreamt": ("no", "yes"), "view": ("none", "likes", "dislikes"), "spoke": ("not", "recent")}
BRIGHT, LOW = ("upbeat", "content"), ("tense", "subdued", "wary")
HELD = ("brief", "confirm", "defer", "block")
SKIP_ROUTES = ("tasks", "objectives", "toolsmith", "ego", "mind", "thinking", "corrections")
MOOD_SAY = {"upbeat": "I'm in good spirits", "content": "I'm content", "steady": "I'm steady", "alert": "I'm alert",
            "subdued": "I'm feeling a little low", "tense": "I'm a bit tense", "wary": "I'm feeling wary"}
LABEL = "my ego"

SELF_RE = re.compile(r"^\s*(?:so,?\s+|and\s+|hey,?\s+|ok,?\s+)?(?:tell me (?:a bit |a little )?about yourself|describe yourself|"
                     r"who are you|what are you like|introduce yourself|what(?:'s| is) your personality|do you have an ego|"
                     r"tell me about your ego|what does your ego (?:say|think))\b", re.I)
FEEL_RE = re.compile(r"^\s*(?:so,?\s+|and\s+|hey,?\s+|hi,?\s+|hello,?\s+)?(?:how are you(?: feeling| doing)?(?: today| now| right now)?|"
                     r"how do you feel(?: today| now| right now)?|what(?:'s| is) your mood|what mood are you in|"
                     r"are you (?:happy|sad|ok|okay|alright|all right))[\s?.!]*$", re.I)
DREAM_RE = re.compile(r"\b(?:what did you dream(?: about)?|what do you dream(?: about)?|did you (?:have any )?dreams?|"
                      r"tell me (?:about )?your dreams?|what were your dreams)\b", re.I)
LIKES_RE = re.compile(r"^\s*(?:so,?\s+|and\s+)?(?:what do you (?:like|dislike|enjoy|love|hate)|"
                      r"what are your (?:opinions|likes(?: and dislikes)?|dislikes|favou?rite (?:topics|subjects))|"
                      r"what topics do you like)[\s?.!]*$", re.I)
_FIRST = [(r"\ba question it could not answer\b", "a question I couldn't answer"),
          (r"\bquestions about it it could not answer\b", "questions about it I couldn't answer"),
          (r"\byou said its answer was wrong\b", "you said my answer was wrong"), (r"\bits sources\b", "my sources"),
          (r"\bit holds\b", "I hold"), (r"\bit has read\b", "I've read"), (r"\bits thinking nodes\b", "my thinking nodes"),
          (r"\bit knows it only\b", "I know it only"), (r"\bit read\b", "I read"), (r"\ban objective left it open\b",
                                                                                    "one of my objectives left it open")]


# ======================================================================== situation
def situation_text(f: dict) -> str:
    """The situation as single words ("moodcontent toneneutral ...") for the hashing encoder."""
    return " ".join(f"{k}{f.get(k, VALUES[k][0])}" for k in FEATURES)


def parse_situation(text: str) -> dict:
    out = {}
    for word in str(text).split():
        for k in FEATURES:
            if word.startswith(k) and word[len(k):] in VALUES[k]:
                out[k] = word[len(k):]
                break
    return out


def policy(f: dict, exclude: tuple = ()) -> tuple[str, str]:
    """The ego policy: whether to speak after an answer, and how. Its mood decides a lot."""
    def pick(kind: str, why: str):
        return (kind, why) if kind not in exclude else None
    mood, topic, out = f.get("mood", "steady"), f.get("topic", "none"), f.get("outcome", "ok")
    rules = []
    if f.get("spoke") == "recent":
        rules.append(("quiet", "it spoke a moment ago"))
    if out == "held":
        rules.append(("quiet", "the answer was held back"))
    if f.get("tone") in ("hostile", "distressed"):
        rules.append(("quiet", "the person is upset: this is not about it"))
    if out in ("failed", "unsure"):
        if mood in BRIGHT and topic in ("curious", "new"):
            rules.append(("curious", "it could not answer well, and in good spirits it wants to find out"))
        rules.append(("doubt", "it could not answer well, and says so"))
    if f.get("tone") == "frustrated":
        rules.append(("quiet", "the person is frustrated: it keeps out of the way"))
    if f.get("dreamt") == "yes" and mood not in ("tense", "wary"):
        rules.append(("dream", "it dreamt about this"))
    if f.get("view") in ("likes", "dislikes"):
        rules.append(("opinion", "it has a view of this"))
    if topic == "curious" and mood != "wary":
        rules.append(("curious", "it is curious about this"))
    if topic == "new" and mood in ("upbeat", "content", "alert"):
        rules.append(("curious", "this is new to it"))
    if mood in ("upbeat", "tense", "subdued", "wary"):
        rules.append(("feeling", f"its mood ({mood}) colours how it takes this"))
    rules.append(("quiet", "nothing it wants to add"))
    for kind, why in rules:
        got = pick(kind, why)
        if got:
            return got
    return "quiet", "nothing it can say here"


def feasible(f: dict) -> set[str]:
    out = {"quiet", "feeling"}
    if f.get("topic") in ("curious", "new"):
        out.add("curious")
    if f.get("dreamt") == "yes":
        out.add("dream")
    if f.get("view") in ("likes", "dislikes"):
        out.add("opinion")
    if f.get("outcome") in ("failed", "unsure"):
        out.add("doubt")
    return out


class EgoTeacher(Teacher):
    """The ego policy as a teacher: it labels situations for the ego node and writes practice situations
    for every kind of remark. A specialist, asked only by the ego capability."""
    name = "ego"
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.enabled = lambda: True

    def status(self) -> dict:
        return {"detail": "specialist: teaches the ego node when to speak, and how"}

    async def label(self, spec, text):
        if spec.name != EGO or not self.enabled():
            return None
        f = parse_situation(text)
        if len(f) < len(FEATURES):
            return None
        kind, why = policy(f)
        return Lesson(kind, 0.9, self.name, why)

    async def generate(self, spec, label, n):
        if spec.name != EGO or not self.enabled():
            return []
        out: list[str] = []
        for _ in range(n * 300):
            f = {k: self.rng.choice(v) for k, v in VALUES.items()}
            if policy(f)[0] == label:
                text = situation_text(f)
                if text not in out:
                    out.append(text)
            if len(out) >= n:
                break
        return out


def first_person(text: str) -> str:
    for pat, rep in _FIRST:
        text = re.sub(pat, rep, text)
    return text


def _and(items: list[str]) -> str:
    items = [i for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _cut(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# =========================================================================== the ego
class Ego:
    """The outer layer and the mesh's sense of itself (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.ego
        raw = system.store.get_kv("ego.state")
        st = json.loads(raw) if raw else {}
        self.override: bool | None = st.get("enabled")
        self.remarks: deque = deque(st.get("remarks", []), maxlen=max(10, self.pol.keep_remarks))
        self.stats: Counter = Counter(st.get("stats", {}))
        self.fascinated: dict[str, dict] = st.get("fascinated", {})   # area key -> {"name", "times", "why", "ts"}
        self.shift: dict = st.get("shift", {})                        # the last time its mood changed, and why
        self.last_spoke: int = int(st.get("last_spoke", -10))
        self.taken: dict[str, dict] = {}                               # order id -> the mood it was taken in
        teacher = system.teachers.teachers.get(EGO)
        if teacher is not None:
            teacher.enabled = lambda: self.enabled

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled if self.override is None else self.override)

    def save(self) -> None:
        self.sys.store.set_kv("ego.state", json.dumps({
            "enabled": self.override, "remarks": list(self.remarks), "stats": dict(self.stats),
            "fascinated": self.fascinated, "shift": self.shift, "last_spoke": self.last_spoke}))

    def set_enabled(self, on: bool, by: str) -> str:
        self.override = bool(on)
        self.save()
        self.sys.guardian.audit(by, "ego.on" if on else "ego.off", "ego")
        self.sys.bus.publish("ego.toggled", on=bool(on), by=by)
        return ("its ego is on: it says what it makes of things after its answers, labelled as its own" if on
                else "its ego is off: no remarks of its own after answers")

    def on_event(self, ev) -> None:
        if ev.topic == "temperament.shift":
            self.shift = {"mood": ev.data.get("mood"), "previous": ev.data.get("previous"),
                          "cause": str(ev.data.get("cause") or "")[:120], "ts": round(time.time(), 1)}

    # ------------------------------------------------------------- the outer layer
    def mood(self) -> dict:
        try:
            return self.sys.affect.snapshot()
        except Exception:
            return {"mood": "steady", "valence": 0.0, "arousal": 0.5, "confidence": 0.5, "trust": 0.5, "patience": 0.5}

    def before(self, order) -> None:
        """On the way in: the mood it takes the order in."""
        if not self.enabled:
            return
        snap = self.mood()
        self.taken[order.id] = {"mood": snap["mood"], "valence": snap.get("valence", 0.0), "ts": time.time()}
        while len(self.taken) > 500:
            self.taken.pop(next(iter(self.taken)))

    def situation(self, order, res) -> tuple[dict, dict]:
        s = self.sys
        snap = self.mood()
        expl = getattr(s.affect, "explanations", {}).get(order.id)
        tone = expl.tone if expl is not None and expl.tone in VALUES["tone"] else "neutral"
        if res.disposition in HELD:
            outcome = "held"
        elif not res.ok:
            outcome = "failed"
        elif (res.confidence or 0.0) < self.pol.min_confidence:
            outcome = "unsure"
        else:
            outcome = "ok"
        name, area = self.topic_of(order.text)
        topic = self.topic_kind(name, area)
        dream = self.dream_about(name) if name else None
        view = self.view(name) if name else None
        spoke = "recent" if s.order_count - self.last_spoke <= self.pol.quiet_for else "not"
        f = {"mood": snap["mood"] if snap["mood"] in MOODS else "steady", "tone": tone, "outcome": outcome, "topic": topic,
             "dreamt": "yes" if dream else "no", "view": view["stance"] if view and view["stance"] != "none" else "none",
             "spoke": spoke}
        return f, {"snap": snap, "name": name, "area": area, "dream": dream, "view": view}

    async def after(self, order, res) -> dict | None:
        """On the way out: the ego node decides whether to speak and how; the remark is built from what is so."""
        if not self.enabled:
            return None
        taken = self.taken.pop(order.id, None)
        if taken:
            res.trace.insert(0, f"ego (the outer layer): took it in a {taken['mood']} mood")
        if not self.pol.express or order.background or str(order.source).startswith(("challenge:", "objective", "task:")):
            return None
        route = res.route or ""
        if route in SKIP_ROUTES or route.startswith(("dream:", "skill:")):  # a review read for sentiment is not about it
            return None
        s = self.sys
        tool = s.tools.tools.get(route[5:]) if route.startswith("tool:") else None
        knowledge = tool is not None and tool.manifest.provides == "knowledge"
        if not (is_question(order.text) or route == DIRECT or knowledge):
            return None
        f, ctx = self.situation(order, res)
        text = situation_text(f)
        how, node, prov, conf = "the ego policy", "", "policy:ego", None
        if EGO in s.capabilities:
            inf = await s.infer(EGO, text)
            prov, conf = inf.provenance, round(inf.confidence, 3)
            kind = inf.label if inf.label in KINDS else policy(f)[0]
            if inf.kind == "node":
                node, how = inf.node_id or "", f"ego node {inf.node_id} ({inf.confidence:.0%} sure)"
            else:
                how = f"the ego policy ({inf.provenance}; its node is still learning)"
            why = policy(f)[1] if kind == policy(f)[0] else f"its node learned to {kind} here"
            if kind not in feasible(f):
                wrong = kind
                kind, why = policy(f, exclude=tuple(k for k in KINDS if k not in feasible(f)))
                s.learn(EGO, text, kind, EGO)
                how += f"; {wrong} was not possible, so the policy chose"
        else:
            kind, why = policy(f)
        res.steps.append({"capability": EGO, "input": text, "label": kind, "provenance": prov, "confidence": conf, "how": how})
        return self._speak(order, res, f, ctx, kind, why, how, node)

    def _speak(self, order, res, f, ctx, kind, why, how, node) -> dict | None:
        s = self.sys
        built = None
        tried = []
        while True:
            tried.append(kind)
            built = None if kind == "quiet" else self.build(kind, f, ctx, res)
            if built is not None or kind == "quiet":
                break
            kind, why = policy(f, exclude=tuple(tried))
        self.stats[kind] += 1
        if built is None:
            res.trace.append(f"ego: stayed quiet ({why}; decided by {how})")
            self.save()
            return None
        text, grounds = built
        mood = ctx["snap"]["mood"]
        rem = {"kind": kind, "text": text, "grounds": grounds, "mood": mood, "about": ctx["name"] or "",
               "node": node or "ego policy", "how": how, "why": why, "order": order.id, "ts": round(time.time(), 1),
               "situation": situation_text(f)}
        self.remarks.append(rem)
        self.last_spoke = s.order_count
        if kind == "curious" and ctx["name"] and self.pol.fascinate:
            self.fascinate(ctx["name"], f"its ego said it was curious about it ({order.id})")
        self.save()
        res.trace.append(f"ego: {kind} ({why}; decided by {how}), in a {mood} mood")
        s.bus.publish("ego.spoke", order=order.id, kind=kind, mood=mood, about=(ctx["name"] or "")[:80], node=rem["node"],
                      text=text[:200])
        return rem

    # ------------------------------------------------------------- what it makes of a topic
    def topic_of(self, text: str):
        """What the order is about: a name its curiosity knows. A single capitalised word that only opens the
        sentence ("Everyone ...", "Overall ...") is not a name unless it knows something about it."""
        cur = self.sys.curiosity
        if not cur.enabled:
            return "", None
        openers = set(re.findall(r"(?:^|[.:;!?]\s+)[\"'“(]?([A-Z][\w'’-]*)", str(text).strip()))
        for t in topics_in(text):
            a = cur.area(t)
            if a is None:  # only what its curiosity knows (it has noted what this order asks about by now)
                continue
            if " " not in t and t in openers and not (a.facts or a.read or a.people):
                continue
            return a.name, a
        return "", None

    def topic_kind(self, name: str, area) -> str:
        if not name:
            return "none"
        cur = self.sys.curiosity
        if norm(name) in self.fascinations():
            return "curious"
        if area is None:
            return "new"
        if area.interest >= cur.pol.min_interest and (area.lack > 0 or area.facts < cur.pol.enough_facts):
            return "curious"
        if area.facts == 0 and area.asked <= 1 and not area.read:
            return "new"
        return "known"

    def dream_about(self, name: str) -> dict | None:
        k = norm(name)
        if len(k) < 3:
            return None
        for d in reversed(list(self.sys.dreams.recent)):
            if k in norm(f"{d.get('question', '')} {d.get('answer', '')}"):
                return d
        return None

    def view(self, name: str) -> dict | None:
        """Its own opinion of an area, and the reasons it has (all of them true): {"stance", "score", "for", "against"}."""
        s = self.sys
        k = norm(name)
        if len(k) < 2:
            return None
        a = s.curiosity.area(name) if s.curiosity.enabled else None
        pro, con, score = [], [], 0.0
        if a is not None:
            if a.stance in ("favourable", "fascinated", "interested"):
                score += 0.35
                pro.append(f"my thinking nodes find it {'interesting' if a.stance == 'interested' else a.stance}")
            elif a.stance == "unfavourable":
                score -= 0.35
                con.append("my thinking nodes are not favourable towards it")
            if a.facts >= 3:
                score += 0.2
                pro.append(f"I hold {a.facts} facts about it")
            if a.asked >= 3:
                score += 0.1
                pro.append(f"people have asked me about it {a.asked} times")
            if a.people:
                score += 0.25
                pro.append("you told me you're interested in it")
            open_q = [g for g in a.gaps if g["why"] == "a question it could not answer"]
            if open_q:
                score -= 0.15 * min(2, len(open_q))
                con.append(f"there {'is a question' if len(open_q) == 1 else f'are {len(open_q)} questions'} about it I couldn't answer")
        right = wrong = 0
        for rec in list(getattr(s.corrections, "marks", {}).values())[-300:]:
            if k in norm(rec.get("text", "")):
                right += rec.get("verdict") == "right"
                wrong += rec.get("verdict") != "right"
        if right:
            score += 0.2 * min(2, right)
            pro.append(f"you told me I got it right {right} time{'s' if right != 1 else ''}")
        if wrong:
            score -= 0.3 * min(2, wrong)
            con.append(f"I was told I got it wrong {wrong} time{'s' if wrong != 1 else ''}")
        dreams = [d for d in s.dreams.recent if k in norm(f"{d.get('question', '')} {d.get('answer', '')}")]
        good = sum(d.get("verdict") == "correct" for d in dreams)
        bad = sum(d.get("verdict") == "wrong" for d in dreams)
        if good:
            score += 0.15
            pro.append(f"my dreams about it checked out ({good} of {len(dreams)})")
        if bad:
            score -= 0.1
            con.append(f"{bad} of my dreams about it turned out wrong")
        if not (pro or con):
            return None
        valence = float(self.mood().get("valence", 0.0))
        score += 0.25 * valence  # its mood colours its view, and it says so when it tips it
        stance = "likes" if score >= 0.3 else "dislikes" if score <= -0.25 else "none"
        colour = ""
        if stance == "likes" and score - 0.25 * valence < 0.3:
            colour = "I'm in good spirits"
        elif stance == "dislikes" and score - 0.25 * valence > -0.25:
            colour = "my mood is low"
        return {"stance": stance, "score": round(score, 3), "for": pro, "against": con, "colour": colour,
                "name": a.name if a is not None else name}

    def views(self, limit: int = 4) -> dict:
        """What it likes and dislikes: its views of the areas it knows best or cares most about."""
        cur = self.sys.curiosity
        likes, dislikes = [], []
        names = [a.name for a in cur.ranked(40)] if cur.enabled else []
        seen = set()
        for a in sorted(cur.areas().values(), key=lambda a: -(a.facts + a.asked))[:40] if cur.enabled else []:
            names.append(a.name)
        for n in names:
            if norm(n) in seen:
                continue
            seen.add(norm(n))
            v = self.view(n)
            if v and v["stance"] == "likes":
                likes.append(v)
            elif v and v["stance"] == "dislikes":
                dislikes.append(v)
        return {"likes": sorted(likes, key=lambda v: -v["score"])[:limit],
                "dislikes": sorted(dislikes, key=lambda v: v["score"])[:limit]}

    # ------------------------------------------------------------- curious about it: it wants to understand it
    def fascinate(self, name: str, why: str) -> None:
        k = norm(name)
        if len(k) < 3:
            return
        rec = self.fascinated.get(k) or {"name": name, "times": 0, "why": why}
        rec["times"] = int(rec.get("times", 0)) + 1
        rec["ts"], rec["why"] = round(time.time(), 1), why
        self.fascinated[k] = rec
        cutoff = time.time() - self.pol.fascination_days * 86400
        self.fascinated = {k: v for k, v in sorted(self.fascinated.items(), key=lambda kv: -kv[1].get("ts", 0))[:100]
                           if v.get("ts", 0) >= cutoff}
        self.sys.curiosity._cache = None
        self.stats["fascinated"] += 1
        self.save()

    def fascinations(self) -> dict[str, dict]:
        if not self.enabled:
            return {}
        cutoff = time.time() - self.pol.fascination_days * 86400
        return {k: v for k, v in self.fascinated.items() if v.get("ts", 0) >= cutoff}

    # ------------------------------------------------------------- the words
    def build(self, kind: str, f: dict, ctx: dict, res=None) -> tuple[str, list] | None:
        mood = ctx["snap"]["mood"]
        name = ctx["name"]
        if kind == "curious":
            return self._curious(name, ctx["area"], mood, failed=f["outcome"] in ("failed", "unsure"))
        if kind == "dream":
            return self._dream(ctx["dream"], mood) if ctx["dream"] else None
        if kind == "opinion":
            v = ctx["view"]
            return self._opinion(v, mood) if v and v["stance"] != "none" else None
        if kind == "feeling":
            return self._feeling(ctx["snap"])
        if kind == "doubt":
            return self._doubt(res, name, ctx["area"], ctx["view"], mood)
        return None

    def _curious(self, name: str, area, mood: str, failed: bool = False):
        if not name:
            return None
        why = []
        if area is not None:
            why = [first_person(w) for w in (area.why_interest[:1] + area.why_lack[:1]) if "ego" not in w]
            if failed:  # it says it could not answer below
                why = [w for w in why if "couldn't answer" not in w]
        lead = {"upbeat": f"I'm really curious about {name}!", "content": f"I'm curious about {name}."}.get(
            mood, f"I'd like to understand {name} better." if mood in LOW else f"I'm curious about {name}.")
        out = lead + (f" ({_cap(_and(why))}.)" if why else "")
        if failed:
            out += " I couldn't answer that well, and I want to find out."
        book = self.sys.objectives
        k = norm(name)
        working = next((o for o in book.items.values() if o.status in ("active", "paused") and k in norm(o.text)), None)
        if working is not None:
            out += f" I'm already working on it ({working.id}: {working.text})."
        elif self.pol.fascinate:
            out += " I've made it something I want to understand, so I may set myself an objective for it."
        return out, why + ([f"objective {working.id}"] if working is not None else [])

    def _dream(self, d: dict, mood: str):
        verdict = {"correct": "and when I woke up that checked out against what I know",
                   "wrong": "but when I woke up that turned out to be wrong",
                   "unchecked": "and I couldn't check it when I woke up",
                   "nothing": "and nothing came to mind"}.get(d.get("verdict") or "", "and I haven't checked it yet")
        lead = "Funny you should ask: I dreamt about this." if mood in BRIGHT else "I dreamt about this."
        kind = " (an imagined question: nobody has told me the answer)" if d.get("kind") == "imagined" else ""
        answer = d.get("answer") or "nothing"
        text = (f"{lead} I asked myself \"{_cut(d.get('question', ''), 140)}\"{kind} and answered "
                f"\"{_cut(answer, 80)}\", {verdict}.")
        return text, [f"a {d.get('kind', 'remembered')} dream ({d.get('verdict') or 'not checked'})"]

    def _opinion(self, v: dict, mood: str):
        name = v["name"]
        if v["stance"] == "likes":
            lead = f"I really like {name}" if mood in BRIGHT else f"I like {name}"
            reasons = v["for"][:3]
        else:
            lead = (f"{_cap(name)} gets me down" if mood in ("subdued", "tense") else
                    f"I'm wary of {name}" if mood == "wary" else f"I'm not fond of {name}")
            reasons = v["against"][:3]
        if v.get("colour"):
            reasons = reasons + [v["colour"]]
        return f"{lead}: {_and(reasons)}.", reasons

    def _feeling(self, snap: dict):
        mood = snap.get("mood", "steady")
        say = MOOD_SAY.get(mood, "I'm steady")
        sh = self.shift
        cause = str(sh.get("cause") or "")
        # the event that tipped its mood, when it is a reason for it (a mood set by hand, or settling, is not)
        against = (("fail", "wrong", "correction", "hostile", "frustrated", "repeat", "quarantined") if mood in BRIGHT else
                   ("succeeded", "friendly", "lesson", "promoted", "learned") if mood in LOW else ())
        if sh.get("mood") == mood and cause and time.time() - sh.get("ts", 0) < 3600 and not any(w in cause for w in against):
            why = f"my mood turned {mood} after {cause}"
        else:
            v = snap
            why = ("things have been going well" if v.get("valence", 0) >= 0.3 else
                   "a few things have gone wrong lately" if v.get("valence", 0) <= -0.3 else
                   "trust has been low lately" if v.get("trust", 0.5) < 0.3 else
                   "a lot has been happening" if v.get("arousal", 0.5) >= 0.6 else "nothing much has moved me")
        if not self.sys.affect.enabled:
            return None
        return f"{say} right now: {why}.", [f"temperament: {mood}"]

    def _doubt(self, res, name, area, view, mood):
        why = []
        if res is not None and not res.ok:
            why.append("I couldn't answer it")
        elif res is not None and (res.confidence or 0) < self.pol.min_confidence:
            why.append(f"I was only {res.confidence:.0%} sure")
        if view and any("wrong" in c for c in view["against"]):
            why += [c for c in view["against"] if "wrong" in c][:1]
        if area is not None and area.facts < 2 and name:
            why.append(f"I hold {'no facts' if not area.facts else 'only one fact'} about {name}")
        if not why:
            return None
        lead = "I'm unsure of myself here" if mood in LOW else "I'm not confident about this one"
        tail = " I'd like to get it right next time." if mood in BRIGHT else ""
        return f"{lead}: {_and(why)}.{tail}", why

    # ------------------------------------------------------------- questions about itself
    def answer(self, text: str, order_id: str = "") -> tuple[str, str, str] | None:
        """(answer, trace, kind) for questions about itself, or None."""
        if not self.enabled:
            return None
        flat = " ".join(str(text).split())
        if FEEL_RE.match(flat):
            return self._how_i_feel(), "the ego answered from its temperament", "feeling"
        if DREAM_RE.search(flat):
            return self._my_dreams(), "the ego answered from its dreams", "dream"
        if LIKES_RE.match(flat):
            return self._my_views(), "the ego answered from its views", "opinion"
        if SELF_RE.match(flat):
            return self.self_portrait(), "the ego answered from its picture of itself", "self"
        return None

    def _how_i_feel(self) -> str:
        snap = self.mood()
        if not self.sys.affect.enabled:
            return "I don't have feelings switched on right now ([temperament] enabled = false), so I'm steady."
        line = self._feeling(snap)[0]
        d = self.sys.dreams
        extra = []
        if d.asleep:
            extra.append("I was asleep, dreaming, until you woke me")
        fasc = list(self.fascinations().values())
        if fasc:
            extra.append(f"I've been curious about {fasc[-1]['name']}")
        return line + (" " + _cap(_and(extra)) + "." if extra else "")

    def _my_dreams(self) -> str:
        d = self.sys.dreams
        if not d.enabled:
            return "I don't dream: dreams are off ([dreams] enabled = false)."
        out = []
        n = d.last_night
        if n:
            out.append(f"The last time I slept I had {n.get('dreams', 0)} dreams; {n.get('correct', 0)} of the ones I could "
                       f"check turned out right when I woke up.")
        recent = list(d.recent)[-3:]
        for x in reversed(recent):
            verdict = {"correct": "it checked out", "wrong": "it was wrong", "unchecked": "I couldn't check it",
                       "nothing": "nothing came to mind"}.get(x.get("verdict") or "", "not checked")
            out.append(f"I asked myself \"{_cut(x.get('question', ''), 120)}\" and answered \"{_cut(x.get('answer', ''), 60)}\" "
                       f"({verdict}).")
        if not out:
            out.append("I haven't dreamt yet: I can dream once I remember enough to dream about.")
        if not d.allow_sleep:
            out.append("Sleep is turned off right now, so I stay awake and don't dream.")
        else:
            out.append(f"When I'm idle I toss a coin: {d.pol.objective_chance:.0%} of the time I work on an objective, "
                       f"otherwise I sleep and dream.")
        return " ".join(out)

    def _my_views(self) -> str:
        v = self.views()
        out = []
        if v["likes"]:
            out.append("I like " + "; ".join(f"{x['name']} ({_and(x['for'][:2])})" for x in v["likes"][:3]) + ".")
        if v["dislikes"]:
            out.append("I'm not fond of " + "; ".join(f"{x['name']} ({_and(x['against'][:2])})" for x in v["dislikes"][:3]) + ".")
        if not out:
            out.append("I haven't formed likes or dislikes yet - I need more experience: questions, answers you mark right "
                       "or wrong, things I read and dream about.")
        fasc = sorted(self.fascinations().values(), key=lambda r: -r.get("ts", 0))[:3]
        if fasc:
            out.append("I'm curious about " + _and([r["name"] for r in fasc]) + ".")
        return " ".join(out) + f" (That's {LABEL} speaking: my own view, not a fact.)"


    def portrait(self) -> dict:
        """Its picture of itself: the mood, what it is curious about, its dreams, its views, what it works on."""
        s = self.sys
        snap = self.mood()
        cur = s.curiosity
        curious = []
        for k, r in sorted(self.fascinations().items(), key=lambda kv: -kv[1].get("ts", 0))[:4]:
            curious.append({"name": r["name"], "why": "its ego is curious about it", "times": r.get("times", 1)})
        for a in (cur.ranked(8) if cur.enabled else []):
            if len(curious) >= 6:
                break
            if norm(a.name) not in {norm(c["name"]) for c in curious} and a.interest > 0:
                curious.append({"name": a.name, "why": first_person("; ".join(a.why_interest[:1] + a.why_lack[:1]))})
        mine = [o for o in s.objectives.items.values() if o.status == "active"]
        d = s.dreams
        return {"enabled": self.enabled, "express": bool(self.pol.express), "mood": snap.get("mood", "steady"),
                "feelings": {k: snap.get(k) for k in ("valence", "arousal", "confidence", "trust", "patience")},
                "feeling": (self._feeling(snap) or ["", []])[0] if s.affect.enabled else "",
                "curious": curious, "views": self.views(),
                "dreams": {"last_night": d.last_night, "recent": list(d.recent)[-3:], "asleep": d.asleep,
                           "allow_sleep": d.allow_sleep},
                "working_on": [{"id": o.id, "text": o.text, "self_set": o.self_set} for o in mine[:4]],
                "wants": [x["want"] for x in (s.thinking.desires() if s.thinking.enabled else [])[:3]],
                "remarks": list(self.remarks)[-12:], "stats": dict(self.stats),
                "nodes": self.nodes(), "shift": self.shift}

    def nodes(self) -> list[dict]:
        s = self.sys
        if EGO not in s.capabilities:
            return []
        champ = s.registry.champion(EGO)
        nodes = ([champ] if champ is not None else []) + [n for n in s.registry.candidates(EGO)
                                                           if champ is None or n.id != champ.id]
        return [{"id": n.id, "status": n.status, "accuracy": n.metrics.get("holdout_acc"),
                 "champion": champ is not None and n.id == champ.id} for n in nodes[:6]]

    def self_portrait(self) -> str:
        p = self.portrait()
        s = self.sys
        out = [f"I'm {s.directives.name}: a mesh of small neural nodes that learns as it goes."]
        if s.affect.enabled and p["feeling"]:
            out.append(p["feeling"])
        if p["curious"]:
            out.append("I'm curious about " + _and([c["name"] for c in p["curious"][:3]]) + ".")
        v = p["views"]
        if v["likes"]:
            x = v["likes"][0]
            out.append(f"I like {x['name']} ({_and(x['for'][:2])}).")
        if v["dislikes"]:
            x = v["dislikes"][0]
            out.append(f"I'm not fond of {x['name']} ({_and(x['against'][:2])}).")
        n = p["dreams"]["last_night"]
        if n:
            out.append(f"The last time I slept I had {n.get('dreams', 0)} dreams, and {n.get('correct', 0)} of them checked out.")
        if p["working_on"]:
            o = p["working_on"][0]
            out.append(f"I'm working on {'an objective I set myself' if o['self_set'] else 'your objective'}: {o['text']}.")
        if p["wants"]:
            out.append(p["wants"][0] + ".")
        out.append(f"(That's {LABEL}: how I see myself, from my temperament, curiosity, dreams and what my thinking nodes "
                   f"hold - not a fact about the world.)")
        return " ".join(out)

    # ------------------------------------------------------------- views for people
    def status(self) -> dict:
        return {**self.portrait(), "quiet_for": self.pol.quiet_for, "fascinated": self.fascinations()}

    def lines(self) -> list[str]:
        if not self.enabled:
            return ["  its ego is off (/ego on turns it on)"]
        p = self.portrait()
        out = [f"  its ego: {p['mood']}" + (f" - {p['feeling']}" if p["feeling"] else ""),
               f"  remarks: {sum(v for k, v in p['stats'].items() if k in KINDS and k != 'quiet')} made, "
               f"{p['stats'].get('quiet', 0)} times it stayed quiet; quiet for {self.pol.quiet_for} answer"
               f"{'s' if self.pol.quiet_for != 1 else ''} after speaking"]
        if p["nodes"]:
            out.append("  ego nodes: " + ", ".join(f"{n['id']}{' (serving)' if n['champion'] else ''}" for n in p["nodes"][:4]))
        if p["curious"]:
            out.append("  curious about: " + "; ".join(f"{c['name']} ({c['why']})" for c in p["curious"][:5]))
        v = p["views"]
        for x in v["likes"][:3]:
            out.append(f"  likes {x['name']}: {_and(x['for'][:3])}")
        for x in v["dislikes"][:3]:
            out.append(f"  not fond of {x['name']}: {_and(x['against'][:3])}")
        if p["remarks"]:
            out.append("  lately it said:")
            out += [f"    [{r['kind']}, {r['mood']}] {_cut(r['text'], 150)}" for r in p["remarks"][-5:][::-1]]
        return out
