"""Corrections: a person says whether an answer was right, and the mesh learns from it.

    right            the route the router chose, the label a skill node gave, and (for a question
                     answered from what it read or by a teacher) the answer itself are confirmed: human
                     examples for the nodes, and a fact for the memory nodes
    wrong            it shows the next most likely answers, best first:
                       the other labels its skill node gave some weight
                       the other routes the router gave some weight, each worked out now (the tool
                       run, the skill node asked, memory, what it has read, a teacher)
                       the other answers its memory nodes hold for the question
                       what it has read about it, and a teacher's answer (with an LLM teacher)
                     the person picks one, or none of them (and may say what is right):
                       a route or label     a human example for the router or the skill node
                       an answer            a fact for the memory nodes, with its test
                       none                 the wrong answer is one a memory node must not give again;
                                            what the person says is right is learned; if they do not
                                            know either, it stays an open question it keeps working on
                                            (the Investigator, the toolsmith, second thoughts) and asks

What a person teaches takes effect at once for that wording (System.infer: a person's label stands,
P1), and the nodes train on it in the background so that wordings like it come out right too.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from .agents import is_question
from .directors import agrees
from .tasks import DIRECT, DISPATCH, Order, skill_payload

NOT_CORRECTABLE = ("tasks", "objectives", "toolsmith", "mind", "thinking", "ego")


def _same(a, b) -> bool:
    x, y = " ".join(str(a).lower().split()).strip(" ."), " ".join(str(b).lower().split()).strip(" .")
    return bool(x) and (x == y or (len(x) > 12 and len(y) > 12 and agrees(x, y) and (x in y or y in x)))


def _pct(p: float) -> str:
    return "under 1%" if p < 0.005 else f"{p:.0%}"


def _first_line(text) -> str:
    return " ".join(str(text or "").split("\n")[0].split())


class Corrections:
    def __init__(self, system):
        self.sys = system
        raw = system.store.get_kv("corrections.state")
        st = json.loads(raw) if raw else {}
        self.marks: dict[str, dict] = st.get("marks", {})
        self.wrong: list[str] = st.get("wrong", [])[-100:]   # marked wrong, with nothing right given yet
        self.counts: dict[str, int] = st.get("counts", {})
        self.pending: dict[str, dict] = {}                   # alternatives offered, by order

    def save(self) -> None:
        marks = dict(list(self.marks.items())[-300:])
        self.sys.store.set_kv("corrections.state", json.dumps({"marks": marks, "wrong": self.wrong[-100:],
                                                                "counts": self.counts}))

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    # ---------------------------------------------------------------- the order
    def _order(self, order_id: str):
        record = self.sys.store.get_order(order_id)
        if record is None:
            return None
        res = record["result"]
        steps = [st for st in res.get("steps") or [] if st.get("capability") in self.sys.capabilities]
        dstep = next((st for st in steps if st["capability"] == DISPATCH), None)
        sstep = next((st for st in reversed(steps) if st["capability"] != DISPATCH
                      and not self.sys.capabilities[st["capability"]].internal), None)
        sources = " ".join(map(str, res.get("sources") or []))
        source = ("memory" if res.get("knowledge") == "memory" else
                  "knowledge" if res.get("knowledge") in ("recalled", "researched") else
                  "teacher" if res.get("served_by") == "teachers" and (res.get("route") or "") == DIRECT else
                  res.get("route") or "")
        mem = re.search(r"memory node (mem-[\w-]+)", sources)
        return {"text": record["text"], "res": res, "route": res.get("route") or "", "dstep": dstep, "sstep": sstep,
                "answer": _first_line(res.get("output") if res.get("ok") else ""), "source": source,
                "memory_node": mem.group(1) if mem else None, "feedback": record.get("feedback")}

    # ---------------------------------------------------------------- the alternatives
    async def alternatives(self, order_id: str, n: int = 5) -> dict | None:
        """The next most likely answers to an order, best first, each worked out now."""
        s = self.sys
        o = self._order(order_id)
        if o is None:
            return None
        text, given = o["text"], o["answer"]
        cands: list[dict] = []

        def add(answer, source: str, how: str, likelihood: float, kind: str, teach: dict) -> None:
            a = _first_line(answer)
            if not a or _same(a, given):
                return
            for c in cands:
                if _same(a, c["answer"]):
                    if likelihood > c["likelihood"]:
                        c.update(source=source, how=how, likelihood=round(likelihood, 4), kind=kind, teach=teach)
                    return
            cands.append({"answer": a[:400], "source": source, "how": how, "likelihood": round(float(likelihood), 4),
                          "kind": kind, "teach": teach})

        # 1. the skill node's other labels
        st = o["sstep"]
        if st is not None:
            cap = st["capability"]
            for label, p in self._ranked(cap, st.get("input") or text):
                if label != st.get("label"):
                    add(label, f"the {cap} skill", self._weight(cap, st.get("input") or text, label, p), p, "label",
                        {"cap": cap, "text": st.get("input") or text, "label": label})
        # 2. the router's other routes, each worked out now (not when the person named the route themselves)
        dinput = (o["dstep"] or {}).get("input") or text
        named = any(str(t).startswith("explicit route") for t in o["res"].get("trace") or [])
        tried = found = 0
        for route, p in ([] if named else self._ranked(DISPATCH, dinput)):
            if route == o["route"] or tried >= 5 or found >= 3:  # however unlikely: with few nodes trained, long shots matter
                continue
            tried += 1
            got = await self._work_out(route, text)
            if got is not None:
                found += 1
                answer, conf, source, extra = got
                add(answer, source, f"the router gave {route} {_pct(p)}", p * (0.5 + 0.5 * conf), "route",
                    {"route": route, "answer": answer, **extra})
        # 3. the other answers its memory nodes hold
        if s.memory.enabled and (is_question(text) or o["route"] == DIRECT):
            for node_id, answer, p in self._memory_answers(text):
                add(answer, f"memory node {node_id}", f"{_pct(p)} sure", p, "fact", {"answer": answer, "node": node_id})
        # 4. what it has read, and a teacher
        for method, label, weight in (("knowledge", "what it has read", 0.8), ("teacher", "a teacher", 0.5)):
            if o["source"] == method or (method == "teacher" and not s.teachers.has_llm()):
                continue
            order = Order(text=text, source="corrections", background=True, learn=False)
            r = await s.orchestrator.attempt(order, method)
            if s.directors.answered(order, r)[0]:
                add(r.output, label, f"{_pct(float(r.confidence or 0.5))} sure", weight * float(r.confidence or 0.5), "fact",
                    {"answer": _first_line(r.output)})
        cands.sort(key=lambda c: -c["likelihood"])
        out = []
        for i, c in enumerate(cands[:n], 1):
            out.append({"id": f"a{i}", **c})
        self.pending[order_id] = {"ts": time.time(), "alternatives": out}
        return {"order": order_id, "text": text, "given": {"answer": given, "route": o["route"], "source": o["source"]},
                "alternatives": out, "question": is_question(text), "marked": self.marks.get(order_id)}

    def _weight(self, cap: str, text: str, label: str, p: float) -> str:
        """How much weight a label got, said plainly: from its node, or (with no node yet) from its cues."""
        node = self.sys.registry.champion(cap)
        if node is not None:
            return f"node {node.id} gave it " + (_pct(p) if p >= 0.005 else "almost no weight")
        heur = self.sys.teachers.teachers.get("heuristic")
        scores = heur.scores(self.sys.capabilities[cap], text) if heur is not None and hasattr(heur, "scores") else {}
        return "a cue in the words points to it" if scores.get(label) else "no node yet, and no cue for it either way"

    def _ranked(self, cap: str, text: str) -> list[tuple[str, float]]:
        """A capability's labels by how likely its serving node (or, without one, its cues) finds them."""
        s = self.sys
        spec = s.capabilities[cap]
        node = s.registry.champion(cap)
        active = [lab for lab in spec.labels if lab not in spec.inactive_labels]
        if node is not None:
            p = node.predict_proba(s.encode(text)[None, :])[0]
            pairs = [(lab, float(p[i])) for i, lab in enumerate(node.labels) if lab in active]
        else:
            heur = s.teachers.teachers.get("heuristic")
            scores = heur.scores(spec, text) if heur is not None and hasattr(heur, "scores") else {}
            floor = 0.02 * max(1.0, sum(scores.values()))  # cues say nothing of the rest: a little weight each, not none
            total = sum(scores.values()) + floor * len(active)
            pairs = [(lab, (scores.get(lab, 0.0) + floor) / total) for lab in active]
        return sorted(pairs, key=lambda kv: -kv[1])

    async def _work_out(self, route: str, text: str):
        """What a route would answer, worked out now (a tool run exploratory, so a miss counts for nothing)."""
        s = self.sys
        if route.startswith("tool:"):
            tool = s.tools.tools.get(route[5:])
            if tool is None or tool.state != "active" or not s.guardian.authorize_tool_run(tool).allowed:
                return None
            run = await s.sandbox.run(tool, text, tool.manifest.timeout_s or s.directives.tools.default_timeout_s,
                                      s.tool_config(tool))
            s.record_tool_run(tool, run, exploratory=True)
            return (str(run.output), 0.9, f"the {tool.name} tool", {}) if run.ok and str(run.output).strip() else None
        if route.startswith("skill:"):
            cap = route[6:]
            if cap not in s.capabilities:
                return None
            payload = skill_payload(text)
            inf = await s.infer(cap, payload, learn=False)
            return (inf.label, float(inf.confidence), f"the {cap} skill", {"label": inf.label, "cap": cap, "text": payload}) \
                if inf.label else None
        if route == DIRECT:
            for method in ("memory", "knowledge", "teacher"):
                if method == "teacher" and not s.teachers.has_llm():
                    continue
                order = Order(text=text, source="corrections", background=True, learn=False)
                r = await s.orchestrator.attempt(order, method)
                if s.directors.answered(order, r)[0]:
                    return (_first_line(r.output), float(r.confidence or 0.5),
                            {"memory": "its memory nodes", "knowledge": "what it has read", "teacher": "a teacher"}[method], {})
        return None

    def _memory_answers(self, text: str) -> list[tuple[str, str, float]]:
        from . import memory as mm
        mem = self.sys.memory
        x = mm.encode_many([text], mem.dim)
        out = []
        for node in mem.nodes.values():
            if node.params is None:
                continue
            p = mm.softmax(mm.forward(node.params, mm.fold(x, node.inputs))[0])[0]
            for i in np.argsort(-p)[:3]:
                if i == 0 or i >= len(node.answers) or p[i] < 0.01:
                    continue
                out.append((node.id, node.answers[i], float(p[i])))
        return sorted(out, key=lambda t: -t[2])[:4]

    # ---------------------------------------------------------------- learning
    async def right(self, order_id: str, by: str = "user") -> tuple[bool, str, list[str]]:
        """A person says the answer was right: the nodes that gave it learn so (human examples)."""
        s = self.sys
        o = self._order(order_id)
        if o is None:
            return False, f"unknown order {order_id}", []
        if o["route"] in NOT_CORRECTABLE:
            return False, "that was not an answer to mark", []
        learned = []
        d = o["dstep"]
        if d is not None and o["route"] in s.dispatch.labels:
            s.teach(DISPATCH, d.get("input") or o["text"], o["route"], reason="confirmed")
            learned.append(f"the router learns that orders like this go to {o['route']}")
        st = o["sstep"]
        if st is not None and st.get("label") in s.capabilities[st["capability"]].labels:
            s.teach(st["capability"], st.get("input") or o["text"], st["label"], reason="confirmed")
            learned.append(f"the {st['capability']} skill learns '{st['label']}' for this")
        if is_question(o["text"]) and o["source"] in ("knowledge", "teacher") and o["answer"] and s.memory.allows("people"):
            made = s.memory.learn(o["text"], o["answer"], f"you confirmed it ({order_id})", by=by, kind="people")
            if made is not None:
                learned.append(f"memory node {made[0]} keeps the answer, with a test")
        self._mark(order_id, "right", by, learned, o)
        return True, "thanks - " + ("; ".join(learned) if learned else "noted"), learned

    async def choose(self, order_id: str, choice: str, answer: str = "", by: str = "user") -> tuple[bool, str, list[str]]:
        """A person says the answer was wrong and picks one of the alternatives, or none of them
        (saying what is right, if they know)."""
        s = self.sys
        o = self._order(order_id)
        if o is None:
            return False, f"unknown order {order_id}", []
        if o["route"] in NOT_CORRECTABLE:
            return False, "that was not an answer to correct", []
        offered = (self.pending.get(order_id) or {}).get("alternatives")
        if offered is None and choice != "none":
            offered = (await self.alternatives(order_id) or {}).get("alternatives", [])
        learned: list[str] = []
        text = o["text"]
        if choice == "none":
            typed = " ".join(str(answer or "").split())
            if o["memory_node"] and not typed and s.memory.enabled:
                if s.memory.must_not_answer(o["memory_node"], text, f"{by} said '{o['answer'][:60]}' was wrong ({order_id})", by):
                    learned.append(f"memory node {o['memory_node']} learns not to answer this")
            if typed:
                learned += self._teach_answer(order_id, o, typed, by)
                if order_id in self.wrong:
                    self.wrong.remove(order_id)
            else:
                from .mind import Unmet
                if s.mind.enabled:
                    now = s.mind.clock()
                    s.mind.unmet[order_id] = Unmet(order_id, text, DIRECT, now, next_try=now + s.directives.mind.retry_unmet_after_seconds)
                    q = s.mind.ask("clarify", f"You said my answer to '{text[:80]}' was wrong, and none of the others I "
                                              f"found were right either. Do you know the right answer? ('answer: ...', or 'skip')",
                                   "a person marked the answer wrong", order_id=order_id)
                    if q is not None:
                        learned.append(f"it asks you ({q.id}) and keeps working on it")
                if order_id not in self.wrong:
                    self.wrong.append(order_id)
                learned.append("it keeps it as an open question: it tries again later, with what it learns meanwhile")
            verdict = "none"
        else:
            alt = next((a for a in offered or [] if a["id"] == choice), None)
            if alt is None:
                return False, f"no alternative {choice} for {order_id} (/correct {order_id} lists them)", []
            t = alt["teach"]
            if alt["kind"] == "label":
                ok, msg = s.feedback(order_id, f"{t['cap']}={t['label']}")
                if ok:
                    learned.append(f"the {t['cap']} skill learns '{t['label']}' for this")
            elif alt["kind"] == "route":
                route = t["route"]
                ok, msg = s.feedback(order_id, route)
                if ok:
                    learned.append(f"the router learns that orders like this go to {route}")
                if route.startswith("skill:") and t.get("label") and t.get("cap") in s.capabilities:
                    s.teach(t["cap"], t["text"], t["label"], reason="correction")
                    learned.append(f"the {t['cap']} skill learns '{t['label']}' for it")
                if route == DIRECT and is_question(text):
                    learned += self._remember(order_id, text, alt["answer"], by)
            else:
                learned += self._remember(order_id, text, alt["answer"], by)
            verdict = choice
        self._mark(order_id, verdict, by, learned, o, answer=answer)
        return True, "thanks - " + ("; ".join(learned) if learned else "noted"), learned

    def _teach_answer(self, order_id: str, o: dict, typed: str, by: str) -> list[str]:
        """What a person says is right: a route or a label teaches the node; anything else is a fact."""
        s = self.sys
        low = typed.lower()
        if low in s.dispatch.labels or low.startswith(("tool:", "skill:")):
            ok, msg = s.feedback(order_id, low)
            return [f"the router learns that orders like this go to {low}"] if ok else [msg]
        st = o["sstep"]
        if st is not None and low in s.capabilities[st["capability"]].labels:
            ok, msg = s.feedback(order_id, f"{st['capability']}={low}")
            return [f"the {st['capability']} skill learns '{low}' for this"] if ok else [msg]
        out = []
        if is_question(o["text"]) and o["route"] != DIRECT and not o["res"].get("ok") and DIRECT in s.dispatch.labels:
            ok, _ = s.feedback(order_id, DIRECT)  # the route it took gave nothing: a question like this is answered directly
            if ok:
                out.append(f"the router learns that questions like this are answered directly, not by {o['route'] or 'nothing'}")
        out += self._remember(order_id, o["text"], typed, by)
        if not is_question(o["text"]) and s.mind.enabled:  # a request no tool met: the toolsmith may take it on
            from .mind import Unmet
            now = s.mind.clock()
            s.mind.unmet[order_id] = Unmet(order_id, o["text"], DIRECT, now, next_try=now + s.directives.mind.retry_unmet_after_seconds)
            out.append("no tool gives that yet, so it keeps the request among those it could write a tool for")
        return out

    def _remember(self, order_id: str, text: str, answer: str, by: str) -> list[str]:
        s = self.sys
        if not s.memory.allows("people"):
            return ["(memory nodes are off, so it cannot keep the answer itself)"]
        made = s.memory.learn(text, answer, f"you told me ({order_id})", by=by, kind="people")
        if made is None:
            return ["it could not keep that as a fact"]
        return [f"memory node {made[0]} learns: '{text[:60]}' -> '{answer[:60]}' (with a test); it answers that way "
                f"once it has learned it"]

    def _mark(self, order_id: str, verdict: str, by: str, learned: list[str], o: dict, answer: str = "") -> None:
        s = self.sys
        rec = {"ts": round(time.time(), 1), "verdict": verdict, "by": by, "text": o["text"][:200], "was": o["answer"][:200],
               "answer": answer[:300], "learned": learned}
        self.marks[order_id] = rec
        self._count("right" if verdict == "right" else "wrong")
        s.store.set_feedback(order_id, json.dumps({"verdict": verdict, "by": by, "answer": answer[:300]}))
        s.guardian.audit(by, "answer.right" if verdict == "right" else "answer.corrected", order_id, verdict=verdict,
                         text=o["text"][:200], learned=learned[:4])
        s.bus.publish("answer.right" if verdict == "right" else "answer.corrected", order=order_id, text=o["text"][:160],
                      verdict=verdict, by=by, learned="; ".join(learned)[:300])
        self.pending.pop(order_id, None)
        temperament = getattr(s, "affect", None)
        if temperament is not None and hasattr(temperament, "expect_again"):
            temperament.expect_again(o["text"])  # asking again to see the change is not a repeated request
        self.save()

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        recent = sorted(self.marks.items(), key=lambda kv: -kv[1]["ts"])[:12]
        return {"counts": dict(self.counts), "open": list(self.wrong[-12:]), "recent": [{"order": k, **v} for k, v in recent]}

    def lines(self, order_id: str, alts: dict) -> list[str]:
        g = alts["given"]
        out = [f"  {order_id}: '{alts['text'][:80]}' was answered '{g['answer'][:80] or '(nothing)'}' ({g['source'] or g['route']})"]
        if not alts["alternatives"]:
            out.append("  it has no other answer to offer")
        for a in alts["alternatives"]:
            out.append(f"    {a['id']}  {a['answer'][:90]}  - {a['source']}; {a['how']}")
        out.append(f"  /correct {order_id} yes | a1 | none [:: the right answer]")
        return out
