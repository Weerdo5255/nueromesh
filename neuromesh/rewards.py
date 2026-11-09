"""Rewards and punishments: people encourage it in some directions and away from others (P23).

Anything it does can be rewarded or punished, by +1 to +3 or -1 to -3, with a reason: an answer, any line of
its activity (a remark of its ego, an initiative it took, a book it read, a tool it chose, a coin it tossed,
code it wrote, an image it read, a lesson, a change to a project), a node, a tool, or a direction in words
("reading about astronomy", "long answers"). Every reward or punishment:

  - moves a direction in its ledger: (what kind of thing, which one) -> a score that fades over [rewards]
    half_life_days, so old encouragement wears off unless it is repeated
  - teaches its reward node (capability "reward": encouraged / discouraged): from the words of what was
    rewarded, it learns to predict what people would encourage, so the direction reaches things nobody
    rewarded yet
  - is felt: rewards lift its mood, punishments lower it (temperament.py), and its ego keeps it in memory
  - changes what it does next, where it chooses:
        its initiative     modes it was rewarded for weigh more in the drive policy, punished ones less
        its ego            a kind of remark that was punished, it keeps to itself
        curiosity          rewarded subjects become more interesting, punished ones less
        the coin it tosses when idle (an objective or sleep)
        the tool switch    "decide" weighs whether people liked it using the tool
        lessons            kinds of lesson people discouraged come round less often
        nodes              a rewarded node is fitter (evolution's parents and its instances), a punished one less
        answers            a rewarded answer's route and labels are reinforced, as a teacher's lesson
        its projects       objectives and approaches people rewarded come first (projects.py)

Rewards steer preferences; they never override a prime directive, a guardian decision or a person's taught
answer, and it cannot reward itself: only people (and the API on their behalf) give them.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import deque

REWARD = "reward"
LABELS = ("encouraged", "discouraged")

# what each kind of event is about: topic prefix -> (domain, the data field naming which one)
DOMAINS = (
    ("order.done", "answer", "route"), ("ego.arbitrated", "arbitration", "strategy"), ("ego.", "ego", "kind"),
    ("memory.own", "ego", "kind"), ("agency.", "initiative", "mode"), ("kiwix.browsed", "reading", "book"),
    ("kiwix.", "research", "query"), ("research.", "research", "query"), ("curiosity.", "curiosity", "topic"),
    ("thinking.", "thinking", "kind"), ("mind.opinion", "opinion", "subject"), ("mind.", "mind", "kind"),
    ("dream.flip", "dreams", "outcome"), ("dream.", "dreams", "kind"), ("tools.decided", "tool", "tool"),
    ("tool.", "tool", "tool"), ("code.written", "code", "how"), ("code.practised", "practice", "how"),
    ("code.", "code", "id"), ("gitea.", "study", "repo"), ("project.", "project", "objective"),
    ("vision.", "sight", "label"), ("tutor.", "lessons", "kind"), ("objective.", "objective", "strategy"),
    ("director.", "director", "kind"), ("node.", "node", "node"), ("temperament.", "mood", "mood"),
    ("approval.", "approval", "kind"),
)
STRENGTH_SAY = {3: "rewarded a lot", 2: "rewarded", 1: "rewarded a little", -1: "punished a little", -2: "punished",
                -3: "punished hard"}


def domain_of(topic: str, data: dict) -> tuple[str, str]:
    for prefix, domain, field in DOMAINS:
        if topic.startswith(prefix):
            key = data.get(field)
            if key is None:
                key = data.get("node") or data.get("tool") or data.get("kind") or topic.split(".", 1)[-1]
            return domain, " ".join(str(key).split())[:80].lower()
    return topic.split(".")[0], topic


class Rewards:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.rewards
        raw = system.store.get_kv("rewards.state")
        st = json.loads(raw) if raw else {}
        self.ledger: dict[str, dict] = st.get("ledger", {})     # "domain|key" -> {"score", "n", "ts", "text"}
        self.given: deque = deque(st.get("given", []), maxlen=self.pol.keep)
        self._n = int(st.get("n", 0))

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def save(self) -> None:
        self.sys.store.set_kv("rewards.state", json.dumps({"ledger": self.ledger, "given": list(self.given), "n": self._n}))

    # ------------------------------------------------------------------ the ledger
    def _decayed(self, entry: dict, now: float | None = None) -> float:
        now = now or time.time()
        hl = max(0.01, self.pol.half_life_days) * 86400
        return entry["score"] * 0.5 ** (max(0.0, now - entry["ts"]) / hl)

    def direction(self, domain: str, key: str) -> float:
        """How people have pushed one direction: tanh of its fading score, in (-1, 1)."""
        e = self.ledger.get(f"{domain}|{str(key).lower()}")
        return math.tanh(self._decayed(e) / 3.0) if e else 0.0

    def predict(self, text: str) -> float:
        """What its reward node expects people to think of something, in (-1, 1) (0 until it is validated)."""
        s = self.sys
        node = s.registry.champion(REWARD) if REWARD in s.capabilities else None
        if node is None or node.params is None or set(LABELS) - set(node.labels):
            return 0.0
        if (node.metrics.get("holdout_acc") or 0.0) < s.directives.nodes.serve_min_accuracy:
            return 0.0
        p = node.predict_proba(s.encode(str(text))[None, :])[0]
        lab = list(node.labels)
        return float(p[lab.index("encouraged")] - p[lab.index("discouraged")])

    def score(self, domain: str, key: str, text: str = "") -> float:
        """The pull on a choice: its direction in the ledger, and (more weakly) what its reward node expects."""
        if not self.enabled:
            return 0.0
        d = self.direction(domain, key)
        p = self.predict(text or f"{domain} {key}") if self.pol.use_node else 0.0
        return max(-1.0, min(1.0, d + self.pol.node_weight * p))

    def top(self, n: int = 12) -> list[dict]:
        now = time.time()
        rows = []
        for k, e in self.ledger.items():
            v = self._decayed(e, now)
            if abs(v) < 0.05:
                continue
            domain, key = k.split("|", 1)
            rows.append({"domain": domain, "key": key, "score": round(v, 2), "pull": round(math.tanh(v / 3.0), 3),
                         "n": e["n"], "ts": e["ts"], "text": e.get("text", "")})
        return sorted(rows, key=lambda r: -abs(r["score"]))[:n]

    # ------------------------------------------------------------------ giving one
    def resolve(self, target: str, kind: str = "") -> dict | None:
        """What a target names: an order ("ord-12", "last"), a line of the activity log ("log:123"), a node, a tool,
        a scratch pad item, an image, a lesson, a project change, or (anything else) a direction in words."""
        s = self.sys
        t = str(target).strip()
        if not t:
            return None
        if kind in ("", "order") and (t == "last" or re.fullmatch(r"ord-[\w-]+", t)):
            ev = next((e for e in s.bus.recent("order.done", 400)[::-1] if t == "last" or e.data.get("order") == t), None)
            if ev is not None:
                d = ev.data
                return {"kind": "order", "ref": d.get("order"), "domain": "answer", "key": str(d.get("route") or ""),
                        "text": f"answer {d.get('route')}: {d.get('text', '')} -> {str(d.get('output', ''))[:160]}",
                        "event": {"topic": ev.topic, "data": d}}
            if kind == "order":
                return None
        if kind in ("", "log") and (t.startswith("log:") or kind == "log"):
            seq = int(re.sub(r"\D", "", t) or 0)
            _, items = s.log.since(seq - 1, 100000)
            entry = next((e for i, e in items if i == seq), None)
            if entry is None:
                return None
            if entry.topic.startswith("reward.") or (getattr(entry, "data", {}) or {}).get("kind") in ("reward", "punishment"):
                # a line about a reward someone gave (or its memory of one) is not something it did
                return {"refuse": "that line is a reward or punishment a person gave, not something it did"}
            domain, key = domain_of(entry.topic, getattr(entry, "data", {}) or {})
            return {"kind": "log", "ref": f"log:{seq}", "domain": domain, "key": key, "text": f"{domain} {key}: {entry.text}",
                    "event": {"topic": entry.topic, "data": getattr(entry, "data", {}) or {}}}
        node = s.registry.get(t)
        if node is not None and kind in ("", "node"):
            return {"kind": "node", "ref": t, "domain": "node", "key": t, "text": f"node {t} for {node.capability}"}
        if t in s.tools.tools and kind in ("", "tool"):
            return {"kind": "tool", "ref": t, "domain": "tool", "key": t, "text": f"using the {t} tool ({s.tools.tools[t].manifest.description.split(':')[0].split('.')[0][:60]})"}
        if t in getattr(s.scratch, "items", {}):
            item = s.scratch.items[t]
            how = item["origin"].get("how", "")
            return {"kind": "code", "ref": t, "domain": "code", "key": how, "text": f"code {how}: {item['title']}"}
        if t in getattr(s.vision, "records", {}):
            rec = s.vision.records[t]
            return {"kind": "image", "ref": t, "domain": "sight", "key": str((rec.get("sees") or {}).get("label") or ""),
                    "text": f"image {rec['name']}: it saw {(rec.get('sees') or {}).get('label')}; it read {((rec.get('text') or {}).get('text') or '')[:80]}"}
        lesson = next((r for r in getattr(s.tutor, "lessons", []) if r.get("id") == t), None)
        if lesson is not None:
            return {"kind": "lesson", "ref": t, "domain": "lessons", "key": lesson["kind"],
                    "text": f"lesson {lesson['kind']}: {lesson['order']} -> {lesson['answer'][:80]}"}
        projects = getattr(s, "projects", None)
        if projects is not None:
            ch = projects.changes.get(t)
            if ch is not None:
                return {"kind": "change", "ref": t, "domain": "project", "key": ch.get("objective_kind", ""),
                        "text": f"project change for '{ch.get('objective', '')}': {ch.get('summary', '')}"}
        if kind in ("", "direction"):
            words = " ".join(t.split())[:160]
            return {"kind": "direction", "ref": words, "domain": "direction", "key": words.lower(), "text": words}
        return None

    def give(self, target: str, value: int, by: str, why: str = "", kind: str = "") -> tuple[bool, str, dict | None]:
        """Reward (value > 0) or punish (value < 0) something it did, or a direction."""
        s = self.sys
        if not self.enabled:
            return False, "rewards are off ([rewards] enabled)", None
        if str(by).startswith(("teacher:", "self", "node", "agency", "director")):
            return False, "only people reward or punish it (P23)", None
        try:
            v = int(value)
        except (TypeError, ValueError):
            return False, "a reward is +1 to +3, a punishment -1 to -3", None
        v = max(-3, min(3, v))
        if v == 0:
            return False, "a reward is +1 to +3, a punishment -1 to -3", None
        what = self.resolve(target, kind)
        if what is None:
            return False, f"nothing called {target} to reward or punish", None
        if what.get("refuse"):
            return False, what["refuse"], None
        now = time.time()
        self._n += 1
        rec = {"id": f"rw-{self._n}", "ts": round(now, 1), "value": v, "by": by, "why": " ".join(str(why).split())[:300],
               "target": {"kind": what["kind"], "ref": what["ref"]}, "domain": what["domain"], "key": what["key"],
               "text": what["text"][:300], "effects": []}
        k = f"{what['domain']}|{what['key']}"
        e = self.ledger.get(k)
        score = (self._decayed(e, now) if e else 0.0) + v
        self.ledger[k] = {"score": round(score, 4), "n": (e["n"] if e else 0) + 1, "ts": now, "text": what["text"][:160]}
        if len(self.ledger) > self.pol.max_directions:
            weakest = min(self.ledger, key=lambda x: abs(self._decayed(self.ledger[x], now)))
            self.ledger.pop(weakest, None)
        rec["effects"] = self.effects(what, v, rec, by)
        self.given.append(rec)
        self.save()
        s.guardian.audit(by, "reward" if v > 0 else "punishment", what["ref"], value=v, domain=what["domain"],
                         key=what["key"], why=rec["why"])
        s.bus.publish("reward.given", id=rec["id"], value=v, by=by, domain=what["domain"], key=what["key"],
                      kind=what["kind"], ref=what["ref"], text=what["text"][:160], why=rec["why"])
        say = STRENGTH_SAY[v]
        return True, f"{say}: {what['text'][:120]}" + (f" - {'; '.join(rec['effects'][:4])}" if rec["effects"] else ""), rec

    def effects(self, what: dict, v: int, rec: dict, by: str) -> list[str]:
        """What a reward or punishment does, beyond the ledger."""
        s = self.sys
        out: list[str] = []
        label = "encouraged" if v > 0 else "discouraged"
        if REWARD in s.capabilities:
            s.teach(REWARD, what["text"][:300], label, reason=f"{by} {'rewarded' if v > 0 else 'punished'} it")
            out.append(f"its reward node learns that this is {label}")
        if s.affect.enabled:
            s.affect.feel({"valence": 0.12 * v, "trust": 0.04 * v, "arousal": 0.03 * abs(v)}, cause=f"{'a reward' if v > 0 else 'a punishment'} from {by}")
            out.append("it feels it")
        if s.memory.enabled:
            try:
                s.memory.remember_own("reward" if v > 0 else "punishment",
                                      f"{by} {STRENGTH_SAY[v]} me for {what['text'][:140]}" + (f" ({rec['why']})" if rec["why"] else ""),
                                      about=what["domain"], mood=s.affect.status().get("mood", "") if s.affect.enabled else "",
                                      source=f"{by} ({rec['id']})")
            except Exception:
                pass
        kind = what["kind"]
        if kind == "order" and v > 0:  # its route and labels are reinforced, as a lesson (a person's right/wrong is /feedback)
            ev = what.get("event") or {}
            order_id = what["ref"]
            stored = s.store.get_order(order_id) or {}
            res_steps = (stored.get("result") or {}).get("steps") or []
            route = (ev.get("data") or {}).get("route")
            from .tasks import DISPATCH
            text = (ev.get("data") or {}).get("text", "")
            if route and text and s.learn(DISPATCH, text, route, "reward", weight=min(3.0, 1.0 + 0.5 * v)):
                out.append(f"the router learns that orders like this go to {route}")
            for st in res_steps or []:
                if st.get("capability") not in (DISPATCH, None) and st.get("label") and st.get("input"):
                    s.learn(st["capability"], st["input"], st["label"], "reward", weight=min(3.0, 1.0 + 0.5 * v))
        if kind == "node":
            node = s.registry.get(what["ref"])
            if node is not None:
                node.metrics["reward"] = round(float(node.metrics.get("reward", 0.0)) + v, 2)
                s.registry.persist(node)
                out.append(f"node {node.id} is {'fitter' if v > 0 else 'less fit'} (reward {node.metrics['reward']:+g})")
        if what["domain"] in ("curiosity", "reading", "research", "direction"):
            s.curiosity._cache = None  # what interests it is worked out again, with the new pull
        return out

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        s = self.sys
        node = s.registry.champion(REWARD) if REWARD in s.capabilities else None
        given = list(self.given)[::-1]
        pos = sum(1 for r in given if r["value"] > 0)
        return {"enabled": self.enabled, "given": given[:40], "count": len(given), "rewards": pos, "punishments": len(given) - pos,
                "directions": self.top(16), "half_life_days": self.pol.half_life_days,
                "node": {"id": node.id, "accuracy": node.metrics.get("holdout_acc"),
                         "validated": (node.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy}
                if node is not None else None, "examples": s.store.count(REWARD) if REWARD in s.capabilities else 0}

    def lines(self) -> list[str]:
        st = self.status()
        out = [f"Rewards: {st['rewards']} reward(s) and {st['punishments']} punishment(s); they fade with a half-life of "
               f"{st['half_life_days']:g} days. /reward <what> [+1..+3] [why], /punish <what> [-1..-3] [why]"]
        if st["node"]:
            acc = st["node"]["accuracy"]
            out.append(f"  reward node {st['node']['id']}" + (f": {acc:.0%} on held-out rewards" if acc is not None else "")
                       + ("" if st["node"]["validated"] else " (still learning: only the ledger steers it)"))
        for d in st["directions"]:
            out.append(f"  {'+' if d['score'] > 0 else '-'} {d['domain']}: {d['key'][:50]}  (pull {d['pull']:+.2f}, {d['n']} time(s))")
        for r in st["given"][:6]:
            out.append(f"  {r['id']:<7} {r['value']:+d} by {r['by']}: {r['text'][:70]}" + (f" ({r['why'][:40]})" if r["why"] else ""))
        return out
