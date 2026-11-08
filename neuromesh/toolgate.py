"""The tool switch: tools on, off, or the NeuroMesh decides.

A person sets the switch for all tools ([tools] mode, /tools on|off|decide, the web page), and may set one tool
apart from the rest (/tool <name> on|off|decide|default):

    on       tools are used whenever an order is routed to one (as before)
    off      no tool runs at all - not for orders, research, experiments or self-tests. Orders it would have
             sent to a tool are answered another way (its memory, what it has read, a teacher) or not at all,
             and it says so. Nothing counts against a tool while it is off.
    decide   it decides, each time, whether to use the tool: its tool-gate node weighs how the order asked
             for it, the tool's record, its own mood, whether something else could answer, and how slow the
             tool is. A tool you name yourself ("tool calculator: ...") is always used, and so is its own
             vetting of a tool (self-tests, experiments). When it decides not to, the order is answered
             another way and the trace says why.

The tool-gate node is an ordinary internal capability (capabilities.toml [toolgate]): taught by the gate
policy below (ToolGateTeacher) and people, grown, trained, evolved and promoted like any node. Until it is
validated, the policy decides; the policy keeps teaching it either way.
"""
from __future__ import annotations

import json
import random
import time

from .teachers import Lesson, Teacher

GATE = "toolgate"
MODES = ("on", "off", "decide")
GATE_LABELS = ("use", "skip")
MOODS = ("steady", "upbeat", "content", "alert", "tense", "subdued", "wary")
GATE_FEATURES = ("tasked", "trecord", "tmood", "tother", "tspeed")
GATE_VALUES = {"tasked": ("routed", "explicit", "self"), "trecord": ("new", "good", "mixed", "poor"),
               "tmood": MOODS, "tother": ("none", "memory", "knowledge", "teacher"), "tspeed": ("fast", "slow")}
OTHER_SAY = {"memory": "its memory holds an answer", "knowledge": "what it has read covers it",
             "teacher": "a teacher can answer"}


def gate_text(f: dict) -> str:
    """A situation as single words ("toolgate taskedrouted trecordgood tmoodsteady ...")."""
    return "toolgate " + " ".join(f"{k}{f.get(k, GATE_VALUES[k][0])}" for k in GATE_FEATURES)


def parse_gate(text: str) -> dict:
    words = str(text).split()
    if not words or words[0] != "toolgate":
        return {}
    out = {}
    for word in words[1:]:
        for k in GATE_FEATURES:
            if word.startswith(k) and word[len(k):] in GATE_VALUES[k]:
                out[k] = word[len(k):]
                break
    return out


def gate_policy(f: dict) -> tuple[str, str]:
    """Whether to use a tool. What a person asked for, and its own vetting, always run; a tool that rarely
    works, or one it does not know yet while it feels wary, gives way to something else that can answer."""
    other = f.get("tother", "none")
    if f.get("tasked") == "explicit":
        return "use", "you asked for that tool"
    if f.get("tasked") == "self":
        return "use", "it is vetting the tool itself"
    if f.get("trecord") == "poor" and other != "none":
        return "skip", f"the tool rarely works, and {OTHER_SAY[other]}"
    if f.get("tmood") in ("wary", "tense") and f.get("trecord") == "new" and other != "none":
        return "skip", f"feeling {f.get('tmood')}, it does not try a tool it does not know yet when {OTHER_SAY[other]}"
    if f.get("tspeed") == "slow" and other in ("memory", "knowledge") and f.get("trecord") != "good":
        return "skip", f"the tool is slow, and {OTHER_SAY[other]}"
    if f.get("trecord") == "poor":
        return "use", "the tool rarely works, but nothing else could answer"
    return "use", {"good": "the tool usually works", "new": "it tries a new tool", "mixed": "the tool works often enough"}.get(
        f.get("trecord"), "it uses the tool")


class ToolGateTeacher(Teacher):
    """The gate policy as a teacher, for the tool-gate node only."""
    name = "toolgate"
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def status(self) -> dict:
        return {"detail": "specialist: teaches the tool-gate node when to use a tool"}

    async def label(self, spec, text):
        if spec.name != GATE:
            return None
        f = parse_gate(text)
        if len(f) < len(GATE_FEATURES):
            return None
        kind, why = gate_policy(f)
        return Lesson(kind, 0.9, self.name, why)

    async def generate(self, spec, label, n):
        if spec.name != GATE or label not in GATE_LABELS:
            return []
        out: list[str] = []
        for _ in range(n * 300):
            f = {k: self.rng.choice(v) for k, v in GATE_VALUES.items()}
            if gate_policy(f)[0] == label:
                t = gate_text(f)
                if t not in out:
                    out.append(t)
            if len(out) >= n:
                break
        return out


class ToolSwitch:
    """The switch, per tool overrides, and the decisions it made."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.tools
        raw = system.store.get_kv("tools.switch")
        st = json.loads(raw) if raw else {}
        mode = str(st.get("mode") or self.pol.mode or "on").lower()
        self.mode = mode if mode in MODES else "on"
        self.set_by: str = st.get("set_by", "directives")
        self.per_tool: dict[str, str] = {k: v for k, v in (st.get("tools") or {}).items() if v in MODES}
        self.record: dict[str, dict] = st.get("record", {})  # tool -> {"used", "skipped", "refused", "last"}
        self.recent: list[dict] = st.get("recent", [])[-60:]

    def save(self) -> None:
        self.sys.store.set_kv("tools.switch", json.dumps({
            "mode": self.mode, "set_by": self.set_by, "tools": self.per_tool, "record": self.record,
            "recent": self.recent[-60:]}))

    # ------------------------------------------------------------------ settings
    def mode_of(self, name: str) -> str:
        return self.per_tool.get(name, self.mode)

    def set_mode(self, mode: str, by: str) -> tuple[bool, str]:
        mode = str(mode).lower().strip()
        mode = {"auto": "decide", "let it decide": "decide", "enable": "on", "disable": "off"}.get(mode, mode)
        if mode not in MODES:
            return False, f"the switch is on, off or decide, not '{mode}'"
        before, self.mode, self.set_by = self.mode, mode, by
        self.save()
        s = self.sys
        s.guardian.audit(by, "tools.switch", "all tools", mode=mode, before=before)
        s.bus.publish("tools.switch", mode=mode, tool="", by=by, before=before)
        return True, f"tools: {self.say(mode)}" + (f" ({len(self.per_tool)} tool(s) set apart: "
                                                    f"{', '.join(f'{k} {v}' for k, v in sorted(self.per_tool.items()))})"
                                                    if self.per_tool else "")

    def set_tool(self, name: str, mode: str, by: str) -> tuple[bool, str]:
        s = self.sys
        if name not in s.tools.tools:
            return False, f"no tool called {name} (/tools lists them)"
        mode = str(mode).lower().strip()
        mode = {"auto": "decide", "enable": "on", "disable": "off", "reset": "default", "inherit": "default"}.get(mode, mode)
        if mode not in MODES + ("default",):
            return False, "a tool is on, off, decide, or default (the switch for all tools)"
        before = self.mode_of(name)
        if mode == "default":
            self.per_tool.pop(name, None)
        else:
            self.per_tool[name] = mode
        self.save()
        s.guardian.audit(by, "tools.switch", name, mode=mode, before=before)
        s.bus.publish("tools.switch", mode=mode, tool=name, by=by, before=before)
        now = self.mode_of(name)
        return True, f"{name}: {self.say(now)}" + (" (as the switch for all tools)" if mode == "default" else "")

    @staticmethod
    def say(mode: str) -> str:
        return {"on": "on", "off": "off - it will not run", "decide": "it decides each time"}.get(mode, mode)

    # ------------------------------------------------------------------ the gate
    def refuses(self, tool) -> str | None:
        """Why a tool may not run at all (the switch is off for it), else None. The sandbox asks this
        before every run, whatever it is for."""
        if tool is None:
            return None
        if self.mode_of(tool.name) == "off":
            return f"tools are switched off ({tool.name})" if tool.name not in self.per_tool else f"{tool.name} is switched off"
        return None

    def _rec(self, name: str) -> dict:
        return self.record.setdefault(name, {"used": 0, "skipped": 0, "refused": 0, "last": ""})

    def note_refused(self, tool) -> None:
        self._rec(tool.name)["refused"] += 1

    def situation(self, tool, tasked: str, text: str) -> dict:
        s = self.sys
        runs, ok = int(tool.runs or 0), int(tool.successes or 0)
        record = "new" if runs < 3 else "good" if ok / runs >= 0.8 else "mixed" if ok / runs >= 0.5 else "poor"
        mood = "steady"
        if getattr(s, "ego", None) is not None:
            try:
                mood = s.ego.mood()["mood"]
            except Exception:
                mood = "steady"
        other = "none"
        if tasked == "routed":
            other = self.other_way(text)
        stats = s.store.get_kv(f"tools.ms.{tool.name}")
        avg = float(stats) if stats else 0.0
        slow = avg > max(1500.0, 0.5 * 1000 * float(tool.manifest.timeout_s or self.pol.default_timeout_s))
        return {"tasked": tasked, "trecord": record, "tmood": mood if mood in MOODS else "steady", "tother": other,
                "tspeed": "slow" if slow else "fast"}

    def other_way(self, text: str) -> str:
        """Something else that could answer this order without the tool."""
        s = self.sys
        from .agents import is_question
        if is_question(text) and s.memory.enabled:
            try:
                if s.memory.recall(text, quiet=True) is not None:
                    return "memory"
            except Exception:
                pass
        if is_question(text):
            hits, conf = s.knowledge.recall(text, s.directives.research.recall_threshold)
            if hits and conf >= s.directives.research.recall_threshold:
                return "knowledge"
        if s.teachers.has_llm():
            return "teacher"
        return "none"

    def note_speed(self, tool, ms: float) -> None:
        key = f"tools.ms.{tool.name}"
        prev = self.sys.store.get_kv(key)
        avg = float(prev) * 0.8 + ms * 0.2 if prev else ms
        self.sys.store.set_kv(key, f"{avg:.1f}")

    def decide(self, tool, text: str, tasked: str = "routed") -> tuple[bool, str, str]:
        """Use the tool for this order? (use, why, who decided). Only asked while the switch says 'decide'."""
        s = self.sys
        f = self.situation(tool, tasked, text)
        sit = gate_text(f)
        label, why = gate_policy(f)
        by = "its gate policy"
        node = s.registry.champion(GATE) if GATE in s.capabilities else None
        if (tasked == "routed" and node is not None and set(GATE_LABELS) <= set(node.labels)
                and (node.metrics.get("holdout_acc") or 0.0) >= s.directives.nodes.serve_min_accuracy):
            got, conf = node.predict(s.encode(sit))
            if got in GATE_LABELS:
                if got != label:
                    why = f"its tool-gate node chose to {got} ({conf:.0%} sure)"
                label, by = got, f"tool-gate node {node.id}"
        if GATE in s.capabilities:
            s.learn(GATE, sit, gate_policy(f)[0], GATE)  # the policy keeps teaching the tool-gate node
        use = label == "use"
        rec = self._rec(tool.name)
        rec["used" if use else "skipped"] += 1
        rec["last"] = f"{'used' if use else 'skipped'}: {why}"
        self.recent.append({"ts": round(time.time(), 1), "tool": tool.name, "use": use, "why": why, "by": by,
                            "text": " ".join(str(text).split())[:120], "situation": f})
        self.recent = self.recent[-60:]
        if len(self.recent) % 5 == 0:
            self.save()
        s.bus.publish("tools.decided", tool=tool.name, use=use, why=why, by=by, text=" ".join(str(text).split())[:120])
        return use, why, by

    def note_used(self, tool) -> None:
        self._rec(tool.name)["used"] += 1

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        s = self.sys
        tools = {}
        for name in sorted(s.tools.tools):
            r = self.record.get(name, {})
            tools[name] = {"mode": self.mode_of(name), "own": self.per_tool.get(name), "used": r.get("used", 0),
                           "skipped": r.get("skipped", 0), "refused": r.get("refused", 0), "last": r.get("last", "")}
        node = s.registry.champion(GATE) if GATE in s.capabilities else None
        return {"mode": self.mode, "set_by": self.set_by, "modes": list(MODES), "tools": tools,
                "recent": self.recent[-12:][::-1],
                "node": {"id": node.id, "accuracy": node.metrics.get("holdout_acc")} if node is not None else None}

    def lines(self) -> list[str]:
        st = self.status()
        out = [f"Tools: {self.say(st['mode'])} (set by {st['set_by']}). /tools on|off|decide sets it for all; "
               f"/tool <name> on|off|decide|default sets one apart."]
        for name, t in st["tools"].items():
            own = " (set apart)" if t["own"] else ""
            out.append(f"  {name:<14} {t['mode']}{own}; used {t['used']}, skipped {t['skipped']}, refused {t['refused']}"
                       + (f"; last {t['last']}" if t["last"] else ""))
        if st["recent"]:
            out.append("  lately it decided:")
            out += [f"    {'used' if r['use'] else 'skipped'} {r['tool']} for \"{r['text'][:60]}\": {r['why']} ({r['by']})"
                    for r in st["recent"][:5]]
        return out
