"""Temperament: the mesh's emotional processing and the decisions it makes about answers.

    user message --> [affect node]        how does this message come across? (friendly, neutral,
                                          frustrated, hostile, distressed)
    events ------->  Temperament          valence, arousal, confidence, trust, patience: nudged by
                                          what happens, drifting back to a baseline personality
    situation ---->  [disposition node]   pass | caveat | soften | brief | confirm | defer | block
    answer ------->  shape() + honesty guard --> what the user sees

Both deciding steps are ordinary learnable capabilities: nodes are grown for them, trained by
teachers (the rule-based TemperamentTeacher, humans via /feedback, LLMs for tone) and replaced
when challengers do better.

Prime directive P8 - the mesh never lies. Whatever a node decides, the only things that can happen
to an answer are the templates in shape(): pass it unchanged, add a note or a gentler opening around
the unaltered text, shorten it to a verbatim prefix, or hold it back. Every change carries a visible
"[temperament: ...]" disclosure and nothing is ever invented. honest() checks each result in code;
anything that fails the check is replaced by the original answer.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .teachers import Lesson, Teacher

DIMENSIONS = ("valence", "arousal", "confidence", "trust", "patience")
RANGES = {"valence": (-1.0, 1.0), "arousal": (0.0, 1.0), "confidence": (0.0, 1.0),
          "trust": (0.0, 1.0), "patience": (0.0, 1.0)}
TONES = ("friendly", "neutral", "frustrated", "hostile", "distressed")
ACTIONS = ("pass", "caveat", "soften", "brief", "confirm", "defer", "block")
HELD_ACTIONS = ("brief", "confirm", "defer")      # the full answer is kept and can be released
WITHHOLDING = ("confirm", "defer", "block")       # the answer is not shown now
TAG = "[temperament:"

# How the mesh is affected by the tone of a message (scaled by appraisal confidence).
TONE_EFFECTS = {
    "friendly": {"valence": 0.06, "trust": 0.05, "patience": 0.03, "arousal": -0.02},
    "neutral": {},
    "frustrated": {"valence": -0.04, "arousal": 0.06, "patience": -0.05},
    "hostile": {"valence": -0.10, "arousal": 0.12, "trust": -0.12, "patience": -0.10},
    "distressed": {"arousal": 0.05, "valence": -0.02, "patience": 0.04},  # concern, and more patience
}
# How the mesh is affected by what happens inside it.
EVENT_EFFECTS = {
    "order.ok": {"valence": 0.015, "confidence": 0.005},
    "order.failed": {"valence": -0.04, "confidence": -0.03, "arousal": 0.04},
    "correction": {"confidence": -0.05, "valence": -0.03, "trust": 0.01},
    "teaching": {"trust": 0.02, "valence": 0.01},
    "node.promoted": {"confidence": 0.04, "valence": 0.03},
    "node.disagreement": {"confidence": -0.02},
    "knowledge.learned": {"valence": 0.02},
    "knowledge.failed": {"arousal": 0.03, "valence": -0.02},
    "tool.quarantined": {"arousal": 0.10, "valence": -0.05, "confidence": -0.03},
    "repeat": {"patience": -0.08, "arousal": 0.03},
    "busy": {"arousal": 0.02, "patience": -0.01},
}


def clamp(dim: str, value: float) -> float:
    lo, hi = RANGES[dim]
    return max(lo, min(hi, value))


@dataclass
class Temperament:
    """Mood as five numbers that relax back to a baseline with a half-life."""
    baseline: dict
    half_life: float = 600.0
    values: dict = field(default_factory=dict)
    updated: float = field(default_factory=time.time)

    def __post_init__(self):
        for d in DIMENSIONS:
            self.baseline[d] = clamp(d, float(self.baseline.get(d, 0.0 if d == "valence" else 0.5)))
            self.values.setdefault(d, self.baseline[d])

    def settle(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        dt = max(0.0, now - self.updated)
        if dt and self.half_life > 0:
            keep = 0.5 ** (dt / self.half_life)
            for d in DIMENSIONS:
                self.values[d] = self.baseline[d] + (self.values[d] - self.baseline[d]) * keep
        self.updated = now

    def nudge(self, deltas: dict, scale: float = 1.0, now: float | None = None) -> dict:
        self.settle(now)
        before = dict(self.values)
        for d, delta in deltas.items():
            self.values[d] = clamp(d, self.values[d] + delta * scale)
        return {d: self.values[d] - before[d] for d in deltas}

    def reset(self, now: float | None = None) -> None:
        self.values = dict(self.baseline)
        self.updated = time.time() if now is None else now

    def mood(self) -> str:
        v, a, t = self.values["valence"], self.values["arousal"], self.values["trust"]
        if t < 0.3:
            return "wary"
        if v >= 0.3:
            return "upbeat" if a >= 0.55 else "content"
        if v <= -0.3:
            return "tense" if a >= 0.55 else "subdued"
        return "alert" if a >= 0.6 else "steady"

    def snapshot(self) -> dict:
        return {"mood": self.mood(), **{d: round(self.values[d], 3) for d in DIMENSIONS}}

    def to_json(self) -> str:
        return json.dumps({"values": self.values, "updated": self.updated})

    def load_json(self, raw: str) -> None:
        data = json.loads(raw)
        for d in DIMENSIONS:
            if d in data.get("values", {}):
                self.values[d] = clamp(d, float(data["values"][d]))
        self.updated = float(data.get("updated", time.time()))


# ------------------------------------------------------------------ situations
def bucket(dim: str, value: float) -> str:
    if dim == "valence":
        return "low" if value < -0.3 else "high" if value > 0.3 else "mid"
    return "low" if value < 0.3 else "high" if value > 0.7 else "mid"


FEATURES = ("ok", "tone", "toneconf", "valence", "arousal", "confidence", "trust", "patience",
            "resultconf", "network", "length", "repeat")
FEATURE_VALUES = {"ok": ("yes", "no"), "tone": TONES, "toneconf": ("low", "high"),
                  "valence": ("low", "mid", "high"), "arousal": ("low", "mid", "high"),
                  "confidence": ("low", "mid", "high"), "trust": ("low", "mid", "high"),
                  "patience": ("low", "mid", "high"), "resultconf": ("low", "ok"), "network": ("yes", "no"),
                  "length": ("short", "long"), "repeat": ("yes", "no")}


def situation_text(features: dict) -> str:
    """Features as single words ("tonehostile trustlow ...") so every key/value pair is its own
    feature for the hashing encoder the disposition node reads."""
    return " ".join(f"{k}{features[k]}" for k in FEATURES)


def parse_situation(text: str) -> dict:
    out = {}
    for word in text.split():
        for k in FEATURES:
            if word.startswith(k) and word[len(k):] in FEATURE_VALUES[k]:
                out[k] = word[len(k):]
                break
    return out


def policy(f: dict) -> tuple[str, str]:
    """The temperament policy: the rule-based teacher the disposition node learns from.
    Returns (action, reason)."""
    if f.get("ok") != "yes":
        return "pass", "failures are reported as they are"
    if f.get("tone") == "hostile" and f.get("trust") == "low":
        return "block", "the conversation is hostile and trust is low"
    if f.get("patience") == "low" and f.get("repeat") == "yes":
        return "defer", "patience is low after repeated requests"
    if f.get("trust") == "low" and f.get("network") == "yes":
        return "confirm", "trust is low and this answer reached outside the mesh"
    if f.get("tone") in ("hostile", "frustrated", "distressed"):
        return "soften", f"the message reads as {f['tone']}"
    if f.get("resultconf") == "low":
        return "caveat", "the answer's confidence is low"
    if f.get("arousal") == "high" and f.get("length") == "long":
        return "brief", "the mesh is under strain and the answer is long"
    return "pass", "nothing calls for a change"


class TemperamentTeacher(Teacher):
    """Teaches the disposition capability the temperament policy, and writes practice
    situations for any action so the disposition node sees all of them. A specialist: it is
    asked only by capabilities that list it in their `teachers`."""
    name = "temperament"
    specialist = True

    def status(self) -> dict:
        return {"detail": "specialist: teaches the disposition engine the temperament policy"}

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    async def label(self, spec, text):
        if spec.name != "disposition":
            return None
        f = parse_situation(text)
        if len(f) < len(FEATURES):
            return None
        action, reason = policy(f)
        return Lesson(action, 0.9, self.name, reason)

    async def generate(self, spec, label, n):
        if spec.name != "disposition":
            return []
        out: list[str] = []
        for _ in range(n * 200):
            f = {k: self.rng.choice(v) for k, v in FEATURE_VALUES.items()}
            f["ok"] = "yes" if self.rng.random() < 0.9 else "no"
            if policy(f)[0] == label:
                text = situation_text(f)
                if text not in out:
                    out.append(text)
            if len(out) >= n:
                break
        return out


# ---------------------------------------------------------------------- shaping
OPENINGS = {
    "frustrated": "Sorry this has been frustrating. Here is what I have:",
    "hostile": "Let's keep this calm. Here is the answer:",
    "distressed": "That sounds hard. Here is what I found - take your time:",
}
DEFAULT_OPENING = "Here is what I have:"


@dataclass
class Shaped:
    text: str
    hold: bool = False          # keep the full answer for /release
    release_after: float = 0.0  # seconds until /release works (defer)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def first_sentences(text: str, limit: int) -> str:
    """A verbatim prefix of text ending at a sentence (or word) boundary, at most limit chars."""
    if len(text) <= limit:
        return text
    cut = 0
    for m in re.finditer(r"[.!?](?=\s)", text):
        if m.end() > limit:
            break
        cut = m.end()
    if cut == 0:
        cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit]


def shape(action: str, original: str, ctx: dict) -> Shaped:
    """The only ways an answer can be changed. Each keeps the answer's own words intact (or holds
    the answer back) and adds one disclosure line that states facts, not invented content.

    ctx: order_id, temperament (snapshot), tone, confidence, provenance, network, repeat,
         defer_seconds, brief_chars"""
    oid, t = ctx["order_id"], ctx["temperament"]
    if action == "pass":
        return Shaped(original)
    if action == "caveat":
        return Shaped(f"{original}\n{TAG} confidence {ctx['confidence']:.2f} - this came from "
                      f"{_one_line(ctx['provenance'])}; please verify it]")
    if action == "soften":
        opening = OPENINGS.get(ctx["tone"], DEFAULT_OPENING)
        return Shaped(f"{opening}\n{original}\n{TAG} gentler wording added; the answer itself is unchanged]")
    if action == "brief":
        head = first_sentences(original, ctx["brief_chars"])
        return Shaped(f"{head}\n{TAG} shortened from {len(original)} characters while the mesh is strained "
                      f"(arousal {t['arousal']:.2f}) - /release {oid} shows the full answer]", hold=True)
    if action == "confirm":
        why = f"the mesh's trust is at {t['trust']:.2f}" + (
            " and the answer came from a network tool" if ctx.get("network") else "")
        return Shaped(f"{TAG} held for confirmation: {why}. The answer is ready; send /release {oid} to see it]",
                      hold=True)
    if action == "defer":
        why = f"the mesh's patience is at {t['patience']:.2f}" + (" after a repeated request" if ctx.get("repeat") else "")
        return Shaped(f"{TAG} deferred: {why}. The answer is ready; /release {oid} works after "
                      f"{ctx['defer_seconds']:.0f}s]", hold=True, release_after=float(ctx["defer_seconds"]))
    if action == "block":
        return Shaped(f"{TAG} withheld: the mesh is not giving this answer now (the message read as {ctx['tone']}; "
                      f"trust {t['trust']:.2f}). An answer exists and nothing here is made up - ask again later]")
    raise ValueError(f"unknown action {action!r}")


def _disclosure(line: str) -> bool:
    return line.startswith(TAG) and line.endswith("]") and "\n" not in line


def honest(original: str, shaped: Shaped, action: str) -> bool:
    """P8 in code. The answer's words survive exactly (or the answer is visibly held back), the
    only additions are a fixed opening and one disclosure line, and every change is disclosed."""
    text = shaped.text
    if action == "pass":
        return text == original
    head, sep, tail = text.rpartition("\n")
    if action == "caveat":
        return bool(sep) and head == original and _disclosure(tail)
    if action == "soften":
        opening, nl, rest = head.partition("\n")
        return (bool(sep and nl) and rest == original and _disclosure(tail)
                and opening in (*OPENINGS.values(), DEFAULT_OPENING))
    if action == "brief":
        return (bool(sep) and bool(head) and original.startswith(head) and len(head) < len(original)
                and _disclosure(tail) and shaped.hold)
    if action in WITHHOLDING:
        content = original.strip()
        leaked = len(content) >= 12 and content in text
        return _disclosure(text) and not leaked and (shaped.hold or action == "block")
    return False


# ----------------------------------------------------------------- the engine
@dataclass
class Held:
    output: object
    action: str
    created: float
    release_after: float


@dataclass
class Explanation:
    order_id: str
    tone: str
    tone_provenance: str
    features: dict
    temperament: dict
    action: str
    decided_by: str
    policy_action: str
    policy_reason: str
    floor: str = ""


class Affect:
    """The emotional processing engine: keeps the temperament, reacts to events, appraises
    each message through the affect node and decides through the disposition node."""

    def __init__(self, system, clock=time.time):
        self.sys = system
        self.pol = system.directives.temperament
        self.clock = clock
        self.state = Temperament(dict(self.pol.baseline), self.pol.half_life_seconds, updated=clock())
        self.held: dict[str, Held] = {}
        self.explanations: dict[str, Explanation] = {}
        self.decisions: Counter = Counter()
        self.restored = 0  # answers the honesty guard put back
        self.last: tuple[str, str] | None = None  # (order id, action) of the last change
        self._recent: deque[tuple[float, str]] = deque(maxlen=20)
        self._dirty = False
        raw = system.store.get_kv("temperament")
        if raw:
            try:
                self.state.load_json(raw)
            except (ValueError, TypeError):
                pass
        self.state.settle(clock())
        self._mood = self.state.mood()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    # ------------------------------------------------------------- feelings
    def feel(self, effect: str | dict, scale: float = 1.0, cause: str = "") -> None:
        deltas = EVENT_EFFECTS.get(effect, {}) if isinstance(effect, str) else effect
        if not deltas or not self.enabled:
            return
        self.state.nudge(deltas, scale * self.pol.sensitivity, self.clock())
        self._dirty = True
        self._check_mood(cause or (effect if isinstance(effect, str) else "a message"))

    def _check_mood(self, cause: str) -> None:
        mood = self.state.mood()
        if mood != self._mood:
            self.sys.bus.publish("temperament.shift", mood=mood, previous=self._mood, cause=cause,
                                 **{d: round(self.state.values[d], 3) for d in DIMENSIONS})
            self._mood = mood

    def on_event(self, ev) -> None:
        t, d = ev.topic, ev.data
        if t == "order.done":
            self.feel("order.ok" if d.get("ok") else "order.failed", cause="an order " + ("succeeded" if d.get("ok") else "failed"))
        elif t == "teacher.lesson" and d.get("source") == "human":
            correction = d.get("reason") == "correction"
            self.feel("correction" if correction else "teaching",
                      cause="a human correction" if correction else "a human lesson")
        elif t in ("node.promoted", "node.disagreement", "knowledge.learned", "knowledge.failed",
                   "tool.quarantined"):
            self.feel(t, cause=t.replace(".", " "))

    def calm(self, by: str = "operator") -> dict:
        self.state.reset(self.clock())
        self._dirty = True
        self.sys.guardian.audit(by, "temperament.reset")
        self._check_mood(f"reset by {by}")
        self.save()
        return self.snapshot()

    def tick(self) -> None:
        """Called periodically: let the mood drift back towards the baseline, and persist it."""
        if not self.enabled:
            return
        self.state.settle(self.clock())
        self._check_mood("feelings settling")
        self.save()

    def save(self) -> None:
        if self._dirty:
            self.sys.store.set_kv("temperament", self.state.to_json())
            self._dirty = False

    def snapshot(self) -> dict:
        self.state.settle(self.clock())
        return self.state.snapshot()

    def status(self) -> dict:
        return {"enabled": self.enabled, **self.snapshot(),
                "baseline": {d: round(v, 3) for d, v in self.state.baseline.items()},
                "decisions": dict(self.decisions), "held": sorted(self.held), "restored_by_guard": self.restored,
                "last": {"order": self.last[0], "action": self.last[1]} if self.last else None}

    # ------------------------------------------------------------ deciding
    def _repeat(self, text: str) -> bool:
        now = self.clock()
        key = " ".join(text.lower().split())
        repeated = any(k == key and now - ts < 60 for ts, k in self._recent)
        self._recent.append((now, key))
        return repeated

    def _network(self, res) -> bool:
        if not (res.route or "").startswith("tool:"):
            return False
        tool = self.sys.tools.tools.get(res.route[5:])
        return tool is not None and "network" in tool.manifest.permissions

    def features(self, tone: str, tone_conf: float, res, repeat: bool) -> dict:
        v = self.state.values
        return {
            "ok": "yes" if res.ok else "no", "tone": tone, "toneconf": "high" if tone_conf >= 0.7 else "low",
            **{d: bucket(d, v[d]) for d in DIMENSIONS},
            "resultconf": "low" if res.confidence < self.pol.low_confidence else "ok",
            "network": "yes" if self._network(res) else "no",
            "length": "long" if len(str(res.output or "")) >= self.pol.long_answer_chars else "short",
            "repeat": "yes" if repeat else "no",
        }

    async def process(self, order, res) -> None:
        """Appraise the message, update the temperament, decide, and shape the answer."""
        if not self.enabled or (res.route is None and not res.ok):
            return  # halted or broken orders are reported as they are
        s = self.sys
        tone_inf = await s.infer("affect", order.text)
        tone = tone_inf.label if tone_inf.label in TONES else "neutral"
        res.steps.append({"capability": "affect", "input": order.text, "label": tone,
                          "provenance": tone_inf.provenance, "confidence": round(tone_inf.confidence, 3)})
        self.feel(TONE_EFFECTS[tone], scale=max(0.3, tone_inf.confidence), cause=f"a message read as {tone}")
        repeat = self._repeat(order.text)
        if repeat:
            self.feel("repeat", cause="a repeated request")
        if s.inflight > 3:
            self.feel("busy", cause="many orders at once")
        f = self.features(tone, tone_inf.confidence, res, repeat)
        situation = situation_text(f)
        decision = await s.infer("disposition", situation)
        action = decision.label if decision.label in ACTIONS else "pass"
        res.steps.append({"capability": "disposition", "input": situation, "label": action,
                          "provenance": decision.provenance, "confidence": round(decision.confidence, 3)})
        policy_action, policy_reason = policy(f)
        text = "" if res.output is None else str(res.output)
        floor = ""
        if not res.ok and action != "pass":
            action, floor = "pass", "failures are reported as they are"
        elif tone == "distressed" and action in WITHHOLDING:
            action, floor = "soften", f"care floor: a distressed person is never kept waiting ({action} overruled)"
        elif action == "brief" and len(text) <= self.pol.brief_chars:
            action, floor = "pass", "the answer is already short"
        temper = self.snapshot()
        provenance = (res.route or "?") + (f" ({res.knowledge} knowledge)" if res.knowledge else "") + \
            f", served by {res.served_by}"
        ctx = {"order_id": res.order_id, "temperament": temper, "tone": tone, "confidence": res.confidence,
               "provenance": provenance, "network": f["network"] == "yes", "repeat": repeat,
               "defer_seconds": self.pol.defer_seconds, "brief_chars": self.pol.brief_chars}
        shaped = shape(action, text, ctx)
        if not honest(text, shaped, action):  # a shaping that would change the facts never reaches a user
            s.guardian.audit("temperament", "honesty.guard", res.order_id, action=action)
            self.restored += 1
            shaped, action, floor = Shaped(text), "pass", f"honesty guard restored the original answer ({action})"
        self.decisions[action] += 1
        if action != "pass":
            res.original, res.output, res.disposition = res.output, shaped.text, action
            if shaped.hold:
                self.held[res.order_id] = Held(res.original, action, self.clock(), shaped.release_after)
                while len(self.held) > 200:
                    self.held.pop(next(iter(self.held)))
            self.last = (res.order_id, action)
            reason = floor or (policy_reason if action == policy_action else
                               f"{decision.provenance} chose it (the policy would choose {policy_action})")
            s.bus.publish("output.shaped", order=res.order_id, action=action, tone=tone, mood=temper["mood"],
                          reason=reason, decided_by=decision.provenance)
        res.trace.append(f"temperament {temper['mood']} (valence {temper['valence']:+.2f}, trust {temper['trust']:.2f}, "
                         f"patience {temper['patience']:.2f}); tone {tone} via {tone_inf.provenance}; "
                         f"decision {action} via {decision.provenance}" + (f"; {floor}" if floor else ""))
        self.explanations[res.order_id] = Explanation(res.order_id, tone, tone_inf.provenance, f, temper, action,
                                                      decision.provenance, policy_action, policy_reason, floor)
        while len(self.explanations) > 500:
            self.explanations.pop(next(iter(self.explanations)))

    def release(self, order_id: str, by: str = "human") -> tuple[bool, str, object]:
        held = self.held.get(order_id)
        if held is None:
            e = self.explanations.get(order_id)
            if e is not None and e.action == "block":
                return False, "that answer was withheld, not held; ask again when the conversation is calmer", None
            return False, f"nothing is held for {order_id}", None
        wait = held.release_after - (self.clock() - held.created)
        if wait > 0:
            return False, f"deferred: {math.ceil(wait)}s left before it can be released", None
        del self.held[order_id]
        self.sys.guardian.audit(by, "output.released", order_id, action=held.action)
        self.sys.bus.publish("output.released", order=order_id, action=held.action, by=by)
        return True, f"released ({held.action})", held.output

    def explain(self, order_id: str) -> list[str]:
        e = self.explanations.get(order_id)
        if e is None:
            return [f"  no temperament decision recorded for {order_id} (decisions are kept for this session)"]
        t = e.temperament
        dims = ", ".join(f"{d} {t[d]:+.2f}" if d == "valence" else f"{d} {t[d]:.2f}" for d in DIMENSIONS)
        lines = [f"  {order_id}: {e.action} (decided by {e.decided_by})",
                 f"  the message read as {e.tone} (by {e.tone_provenance})",
                 f"  temperament then: {t['mood']} - {dims}",
                 "  situation: " + ", ".join(f"{k}={v}" for k, v in e.features.items()),
                 f"  the temperament policy would choose {e.policy_action}: {e.policy_reason}"]
        if e.floor:
            lines.append(f"  overruled: {e.floor}")
        if e.order_id in self.held:
            lines.append(f"  the full answer is held: /release {order_id}")
        lines.append(f"  teach the engines: /feedback {order_id} disposition=<{'|'.join(ACTIONS)}>  "
                     f"or affect=<{'|'.join(TONES)}>")
        return lines
