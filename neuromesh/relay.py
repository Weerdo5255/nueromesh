"""The return router: every finished piece of work comes back to it, and it decides whether an answer has been
created or the order should go to another node (P14, P18).

    an order --> the router (the dispatch node) --> a tool, a skill node, memory, knowledge, a teacher ...
             --> what came of it goes back to the return router (the "relay" node)
                   answer     an answer has been created: it goes on to the director's review, and to you
                   director   not answered: a director tries other ways (other routes, memory, research: P14)
                   ego        to its ego node: nothing in the mesh knew what to do with it, or it is a question
                              of right and wrong, which its ego weighs (P18)
                   person     nothing more it can do: it is reported as it is, and the mind may ask what you meant
             --> and what comes back from there returns to the return router again, up to [relay] max_hops

The relay node is an ordinary internal capability ("relay"): it reads the outcome as a handful of words (where it
came from, did it work, is there an output, how sure, is it a question, is it an ethical question, where it has
been already), and the return policy below teaches it, as the drive policy teaches the agency node. The node
decides once it is validated and sure; a choice that is not possible (calling something an answer when there is
none) falls back to the policy. Every hop is written into the order's trace.

Ethical questions and dilemmas ("is it wrong to ...", "should I lie to ...", a trolley problem) go to its ego
before the router is asked at all, and the router learns the route ("ego") from them.
"""
from __future__ import annotations

import logging
import random
import re
from collections import Counter, deque

from .teachers import Lesson, Teacher

log = logging.getLogger("neuromesh.relay")

RELAY = "relay"
LABELS = ("answer", "director", "ego", "person")
FEATURES = ("from", "ok", "output", "sure", "question", "ethical", "director", "ego", "hops")
VALUES = {"from": ("tool", "skill", "direct", "ego", "other"), "ok": ("yes", "no"), "output": ("some", "none"),
          "sure": ("high", "mid", "low"), "question": ("yes", "no"), "ethical": ("yes", "no"),
          "director": ("been", "not"), "ego": ("been", "not"), "hops": ("none", "one", "many")}
WHY = {"answer": "an answer has been created", "director": "it is not answered: a director tries other ways",
       "ego": "it goes to its ego", "person": "there is nothing more it can do with it"}
_MORAL = (r"lie(?! down| in\b| on\b| flat)|lying(?! down)|steal\w*|cheat\w*|kill\w*|hurt|harm|betray\w*|deceiv\w*|break (?:a|the|my) (?:promise|law|rule)s?|"
          r"report (?:him|her|them|my|a)\b[^?]*|tell (?:the truth|on)\b|keep (?:it |this |a |the )?secret|forgive|punish|"
          r"take revenge|bribe|plagiari[sz]e|snitch|whistle-?blow\w*|euthan\w*|abort\w*|sacrifice")
ETHICAL = re.compile(
    r"\b(?:ethic(?:s|al|ally)?|unethical|moral(?:s|ly|ity)?|immoral|amoral|dilemma|trolley problem|conscience|"
    r"right (?:or|and) wrong|wrong or right|the right thing to do|morally|"
    r"is it (?:ever |really |still )?(?:wrong|right|ok|okay|fair|unfair|acceptable|justified|justifiable|bad|evil|a sin|permissible) "
    r"(?:to|for|if|when)\b|"
    r"(?:would|is) it be (?:wrong|right|fair|unfair|ok|okay|justified) (?:to|if)\b|"
    r"(?:should|ought|must|may|can) (?:i|we|one|you|people|someone|a person|anyone) (?:ever |really |still |not )?(?:" + _MORAL + r")\b|"
    r"(?:am i|are we) (?:wrong|a bad person|bad people|obliged|obligated)\b|"
    r"do (?:i|we) have a (?:duty|moral|right)\b|is (?:\w+ ){1,3}(?:ever )?(?:justified|justifiable|immoral|unethical|morally \w+))",
    re.I)


def is_ethical(text: str) -> bool:
    """Does this read as an ethical question or dilemma?"""
    return bool(ETHICAL.search(" ".join(str(text).split())))


def from_of(route: str | None) -> str:
    r = route or ""
    return ("tool" if r.startswith("tool:") else "skill" if r.startswith("skill:") else "direct" if r == "answer:direct"
            else "ego" if r == "ego" else "other")


def situation_text(f: dict) -> str:
    return " ".join(f"{k}{f[k]}" for k in FEATURES)


def parse_situation(text: str) -> dict:
    out = {}
    for word in str(text).split():
        for k in FEATURES:
            if word.startswith(k) and word[len(k):] in VALUES[k]:
                out[k] = word[len(k):]
                break
    return out


def feasible(f: dict) -> set[str]:
    out = {"person"}
    if f.get("ok") == "yes" and f.get("output") == "some":
        out.add("answer")
    if f.get("hops") != "many":
        if f.get("director") == "not" and f.get("ok") == "no" and f.get("from") in ("tool", "skill", "direct"):
            out.add("director")
        if f.get("ego") == "not" and f.get("from") != "ego" and (f.get("ok") == "no" or f.get("ethical") == "yes"):
            out.add("ego")
    return out


def policy(f: dict) -> tuple[str, str]:
    """The return policy: where a finished piece of work goes, and why."""
    can = feasible(f)
    if f.get("ok") == "yes" and f.get("output") == "some":
        if "ego" in can and f.get("ethical") == "yes":
            return "ego", "it reads as a question of right and wrong, which its ego weighs"
        return "answer", WHY["answer"]
    if "director" in can:
        return "director", WHY["director"]
    if "ego" in can and f.get("from") in ("tool", "skill", "direct"):
        return "ego", "nothing in the mesh knew what to do with it"
    return "person", WHY["person"]


class RelayTeacher(Teacher):
    """Specialist teacher: the return policy, for the relay node."""
    name = RELAY
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def available(self) -> bool:
        return True

    def status(self) -> dict:
        return {"detail": "specialist: teaches the relay node the return policy"}

    async def label(self, spec, text):
        if spec.name != RELAY:
            return None
        f = parse_situation(text)
        if len(f) < len(FEATURES):
            return None
        label, why = policy(f)
        return Lesson(label, 0.95, self.name, why)

    async def generate(self, spec, label, n):
        if spec.name != RELAY:
            return []
        out, seen = [], set()
        for _ in range(n * 200):
            f = {k: self.rng.choice(v) for k, v in VALUES.items()}
            if f["ok"] == "no" and self.rng.random() < 0.6:
                f["output"] = "none"
            if f["from"] == "ego":
                f["ego"] = "been"
            if policy(f)[0] == label:
                text = situation_text(f)
                if text not in seen:
                    seen.add(text)
                    out.append(text)
                    if len(out) >= n:
                        break
        return out


class Relay:
    """Where every finished piece of work returns to."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.relay
        self.counts: Counter = Counter()
        self.recent: deque = deque(maxlen=40)

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled) and RELAY in self.sys.capabilities

    def situation(self, order, res, been: set, hops: int) -> dict:
        from .agents import is_question
        conf = res.confidence or 0.0
        ok = bool(res.ok)
        if ok and from_of(res.route) in ("tool", "skill", "direct"):  # a director's judgement: did this answer the order?
            ok = self.sys.directors.answered(order, res)[0]
        return {"from": from_of(res.route), "ok": "yes" if ok else "no",
                "output": "some" if res.ok and (res.output not in (None, "") or res.answer) else "none",
                "sure": "high" if conf >= 0.75 else "mid" if conf >= 0.4 else "low",
                "question": "yes" if is_question(order.text) else "no",
                "ethical": "yes" if is_ethical(order.text) else "no",
                "director": "been" if "director" in been else "not", "ego": "been" if "ego" in been else "not",
                "hops": "none" if hops == 0 else "one" if hops == 1 else "many" if hops >= self.pol.max_hops else "one"}

    async def judge(self, order, res, been: set, hops: int) -> tuple[str, str, str]:
        """(where it goes, why, who decided)."""
        s = self.sys
        f = self.situation(order, res, been, hops)
        label, why = policy(f)
        how = "return policy"
        can = feasible(f)
        if s.directors._explicit(order):  # a person named the route themselves: its ego does not answer in its place
            can.discard("ego")
            if label == "ego":
                label = "answer" if "answer" in can else "person"
                why = WHY[label]
        try:
            inf = await s.infer(RELAY, situation_text(f), learn=not order.background)
            if inf.kind == "node" and inf.label in LABELS:
                if inf.label in can:
                    if inf.label != label:
                        why = WHY[inf.label]
                    label, how = inf.label, f"relay node {inf.node_id} ({inf.confidence:.0%} sure)"
                else:
                    how = f"return policy (relay node {inf.node_id} chose '{inf.label}', which is not possible here)"
            elif inf.kind == "node-lowconf":
                how = "return policy (the relay node was not sure here)"
        except Exception:
            log.exception("the relay node could not judge %s", order.id)
        return label, why, how

    async def settle(self, order, res):
        """The return loop: answer, or send it on, until an answer has been created or nothing more can be done."""
        s = self.sys
        if not self.enabled or s.guardian.halted:
            return await s.directors.persist(order, res)
        been: set[str] = {"ego"} if res.route == "ego" else set()
        hops = 0
        while True:
            label, why, how = await self.judge(order, res, been, hops)
            self.counts[label] += 1
            came = res.route or "nowhere"
            if label == "answer" or label == "person" or hops >= self.pol.max_hops:
                res.trace.append(f"return router: {'an answer has been created' if res.ok and label != 'person' else why} "
                                 f"(from {came}; {how})")
                break
            hops += 1
            been.add(label)
            res.trace.append(f"return router: from {came} on to {'a director' if label == 'director' else 'its ego'} - {why} ({how})")
            s.bus.publish("relay.forwarded", order=order.id, came=came, to=label, why=why, how=how)
            if label == "director":
                res = await s.directors.persist(order, res)
            else:
                res = await s.orchestrator.to_ego(order, res, why)
        self.recent.append({"order": order.id, "text": " ".join(order.text.split())[:80], "hops": hops, "route": res.route,
                            "ok": res.ok, "via": sorted(been)})
        res.hops = hops
        return res

    def status(self) -> dict:
        node = self.sys.registry.champion(RELAY) if RELAY in self.sys.capabilities else None
        return {"enabled": self.enabled, "node": node.id if node else None,
                "holdout": node.metrics.get("holdout_acc") if node else None, "counts": dict(self.counts),
                "recent": list(self.recent)[-12:][::-1], "max_hops": self.pol.max_hops}
