"""Node budgets: how many nodes it may keep, and how large they may grow, set from the web page (P4).

The directives (directives.toml) set every budget at start. The ones about nodes can also be changed while it
runs - the Nodes view's "Node budgets", /budgets, or POST /budgets - by a person only: the mesh never raises its
own budgets. A change takes effect at once (the agents read the directives each time they act), is audited, and
is kept in the store, so it holds across restarts until it is set back ("as the directives say"). Each budget
shows what is in use now, so you can see what a lower number would squeeze.
"""
from __future__ import annotations

import json
import time


def _live(s) -> int:
    return len([n for n in s.registry.nodes.values() if n.status != "retired"])


def _largest(s) -> int:
    return max((n.param_count for n in s.registry.nodes.values() if n.status != "retired"), default=0)


def _widest(s) -> int:
    return max((max(n.arch.hidden) for n in s.registry.nodes.values() if n.status != "retired" and n.arch.hidden), default=0)


def _deepest(s) -> int:
    return max((len(n.arch.hidden) for n in s.registry.nodes.values() if n.status != "retired"), default=0)


def _challengers(s) -> int:
    return max((len(s.registry.candidates(c)) for c in s.capabilities), default=0)


def _mem_facts(s) -> int:
    return max((len(n.facts()) for n in s.memory.fact_nodes()), default=0) if s.memory.enabled else 0


def _mem_hidden(s) -> int:
    return max((max(n.hidden) for n in s.memory.nodes.values() if n.hidden), default=0)


def _thoughts(s) -> int:
    return max((len(n.outputs) for n in s.thinking.nodes.values()), default=0)


def _think_hidden(s) -> int:
    return max((max(n.hidden) for n in s.thinking.nodes.values() if n.hidden), default=0)


def _dreamers(s) -> int:
    return len(getattr(s.dreams, "nodes", {}) or {})


def _short(s) -> int:
    conv = getattr(s, "conversations", None)
    return len([i for i in conv.short.values() if i.get("status") == "short"]) if conv else 0


def _directors(s) -> int:
    return len(getattr(s.directors, "nodes", {}) or {})


# (path, group, label, what it limits, lowest, highest, step, what is in use now, the word for it)
BUDGETS = [
    ("nodes.max_nodes", "Skill and inner nodes", "Nodes alive at once",
     "Every champion, instance and challenger of every capability counts. At the limit nothing new is grown; "
     "lower it and the weakest challengers are retired first.", 4, 2000, 1, _live, "alive now"),
    ("nodes.max_candidates_per_capability", "Skill and inner nodes", "Challengers per capability",
     "How many challengers may train against one champion at a time.", 1, 16, 1, _challengers, "most for one capability now"),
    ("nodes.instances", "Skill and inner nodes", "Nodes that answer together",
     "The champion and its instances: nodes of one kind that answer an order together.", 1, 9, 1, None, ""),
    ("nodes.max_running", "Skill and inner nodes", "Node instances running at once",
     "Across the whole mesh, while it answers.", 1, 64, 1, None, ""),
    ("nodes.max_hidden", "How large a node may grow", "Neurons in a layer",
     "The widest hidden layer its genetic algorithm may breed.", 16, 8192, 16, _widest, "widest now"),
    ("nodes.max_depth", "How large a node may grow", "Hidden layers",
     "The most hidden layers a node may have.", 1, 12, 1, _deepest, "deepest now"),
    ("evolution.max_params", "How large a node may grow", "Weights in a node",
     "The largest node it may breed, counted in weights.", 10_000, 200_000_000, 10_000, _largest, "largest now"),
    ("memory.facts_per_node", "Memory nodes", "Facts a memory node holds",
     "A node that is full is closed and a new one opened; more facts a node means fewer, larger nodes.", 2, 2000, 1,
     _mem_facts, "fullest now"),
    ("memory.max_hidden", "Memory nodes", "Neurons a memory node may grow to",
     "How far a memory node may grow when it cannot hold its facts.", 16, 4096, 16, _mem_hidden, "widest now"),
    ("conversation.max_short", "Memory nodes", "Short-term memory nodes",
     "What you tell it, kept word for word until its ego decides; the oldest are let go first.", 10, 20000, 10,
     _short, "kept now"),
    ("thinking.thoughts_per_node", "Thinking and dream nodes", "Thoughts a thinking node holds", "", 2, 500, 1,
     _thoughts, "fullest now"),
    ("thinking.max_hidden", "Thinking and dream nodes", "Neurons a thinking node may grow to", "", 16, 4096, 16,
     _think_hidden, "widest now"),
    ("dreams.population", "Thinking and dream nodes", "Dream nodes kept", "The population its genetic algorithm keeps.",
     1, 64, 1, _dreamers, "kept now"),
    ("dreams.max_hidden", "Thinking and dream nodes", "Neurons a dream node may grow to", "", 16, 4096, 16, None, ""),
    ("directors.max_directors", "Directors", "Director nodes",
     "director-0 and the duplicates it makes of itself; never fewer than one.", 1, 32, 1, _directors, "running now"),
]
BY_PATH = {b[0]: b for b in BUDGETS}


class Budgets:
    def __init__(self, system):
        self.sys = system
        self.defaults = {b[0]: self._get(b[0]) for b in BUDGETS}  # what the directives say
        raw = system.store.get_kv("budgets.set")
        self.set_by: dict[str, dict] = json.loads(raw) if raw else {}
        for path, rec in list(self.set_by.items()):  # a person's choices hold across restarts
            if path in BY_PATH:
                self._put(path, rec["value"])
            else:
                del self.set_by[path]

    def _policy(self, path: str):
        section, field = path.split(".")
        return getattr(self.sys.directives, section), field

    def _get(self, path: str):
        pol, field = self._policy(path)
        return getattr(pol, field)

    def _put(self, path: str, value) -> None:
        pol, field = self._policy(path)
        setattr(pol, field, type(self.defaults[path])(value))

    def set(self, path: str, value, by: str) -> tuple[bool, str]:
        """A person sets a budget. Returns (changed, what happened)."""
        b = BY_PATH.get(path)
        if b is None:
            return False, f"no budget '{path}' (budgets: {', '.join(BY_PATH)})"
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return False, f"{b[2].lower()} is a whole number"
        lo, hi = b[4], b[5]
        if not lo <= v <= hi:
            return False, f"{b[2].lower()} must be between {lo:,} and {hi:,}"
        before = self._get(path)
        if v == before:
            return True, f"{b[2].lower()} is {v:,} already"
        self._put(path, v)
        if v == self.defaults[path]:
            self.set_by.pop(path, None)
        else:
            self.set_by[path] = {"value": v, "by": by, "ts": round(time.time(), 1)}
        self._saved(path, before, v, by)
        used = b[7](self.sys) if b[7] else None
        over = f"; {used:,} {b[8]}, so it will shed some" if used is not None and used > v and path in (
            "nodes.max_nodes", "nodes.max_candidates_per_capability", "conversation.max_short", "dreams.population") else ""
        return True, f"{b[2].lower()}: {before:,} -> {v:,}{over}"

    def reset(self, path: str, by: str) -> tuple[bool, str]:
        if path == "all":
            for p in list(self.set_by):
                self.reset(p, by)
            return True, "every node budget is as the directives say again"
        if path not in BY_PATH:
            return False, f"no budget '{path}'"
        before = self._get(path)
        self._put(path, self.defaults[path])
        self.set_by.pop(path, None)
        self._saved(path, before, self.defaults[path], by)
        return True, f"{BY_PATH[path][2].lower()} is as the directives say again ({self.defaults[path]:,})"

    def _saved(self, path, before, after, by) -> None:
        s = self.sys
        s.store.set_kv("budgets.set", json.dumps(self.set_by))
        s.guardian.audit(by, "budget.set", path, before=before, after=after)
        s.bus.publish("budget.set", budget=path, before=before, after=after, by=by)

    def status(self) -> dict:
        out = []
        for path, group, label, what, lo, hi, step, used, word in BUDGETS:
            try:
                now = used(self.sys) if used else None
            except Exception:
                now = None
            rec = self.set_by.get(path)
            out.append({"path": path, "group": group, "label": label, "what": what, "min": lo, "max": hi, "step": step,
                        "value": self._get(path), "default": self.defaults[path], "used": now, "used_word": word,
                        "set_by": rec["by"] if rec else "", "set_at": rec["ts"] if rec else None})
        return {"budgets": out, "changed": len(self.set_by)}

    def lines(self) -> list[str]:
        out = ["Node budgets (set: /budget <name> <number>, /budget reset <name>|all):"]
        for b in self.status()["budgets"]:
            used = f"  ({b['used']:,} {b['used_word']})" if b["used"] is not None else ""
            mark = f"  [set by {b['set_by']}; the directives say {b['default']:,}]" if b["set_by"] else ""
            out.append(f"  {b['path']:<38} {b['value']:>12,}{used}{mark}")
        return out
