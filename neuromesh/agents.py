"""The agents. Each runs as its own concurrent loop and coordinates through the bus,
the registry and the store. Only the Orchestrator serves users; the rest work in the
background, yielding whenever an order is in flight (prime directive P2).

  Orchestrator - takes user orders, routes them, runs tools / nodes, returns results; a director
                 (directors.py) takes every order first and reviews every result last
  Architect    - grows the network: seeds nodes for capability gaps, expands nodes when
                 new labels or tools appear, breeds challengers when a champion is weak
  Trainer      - continually trains nodes on teacher-labelled experience and asks
                 teachers for new lessons where the network is weakest
  Evaluator    - benchmarks champions vs challengers on held-out data, promotes winners
  Curator      - disposes superseded and failed nodes, enforces the node budget
  Explorer     - discovers new tools, routes them through approval, self-tests them,
                 activates them and runs health checks
  Researcher   - reads up in the background (within its budget) on articles linked from
                 what users asked about, and refreshes retained knowledge that has aged
  Sentinel     - watches the kill switch, re-checks local teacher servers (Ollama) and
                 publishes heartbeats with the scorecard
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter, defaultdict, deque

import numpy as np

from .guardian import Verdict
from .knowledge import Hit
from .nodes import Architecture, NeuralNode
from .tasks import DIRECT, DISPATCH, Result, imagine_prompt, parse_order, skill_payload
from .evolution import judge as evolution_judge
from .tools import ACTIVE, APPROVED, PENDING, QUARANTINED, REJECTED, Tool, outputs_match
from . import voice


class Agent:
    name = "agent"

    def __init__(self, system):
        self.sys = system
        self.log = logging.getLogger(f"neuromesh.{self.name}")
        self.stats: Counter = Counter()
        self.last_step = 0.0

    @property
    def interval(self) -> float:
        return self.sys.directives.background.tick_seconds

    async def run(self) -> None:
        while not self.sys.stopping:
            try:
                await self.step()
                self.last_step = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["errors"] += 1
                self.log.exception("%s step failed", self.name)
                self.sys.guardian.audit(self.name, "agent.error", error=repr(exc))
            await asyncio.sleep(self.interval)

    async def step(self) -> None:  # pragma: no cover - overridden
        pass

    async def bg(self, fn, *args):
        """Run CPU-heavy work in a thread, only when no order is in flight and the
        background duty-cycle budget allows it."""
        await self.sys.background_slot()
        t0 = time.monotonic()
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            self.sys.guardian.note_background_busy(time.monotonic() - t0)


# =========================================================================== orders
class Orchestrator(Agent):
    name = "orchestrator"

    def __init__(self, system):
        super().__init__(system)
        self._tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        sem = asyncio.Semaphore(self.sys.directives.orders.max_concurrent_orders)
        while not self.sys.stopping:
            order = await self.sys.orders.get()
            if self.sys.dreams.asleep:  # orders first: an order always wakes it (P2)
                self.sys.dreams.wake("an order arrived")
            await sem.acquire()
            self.sys.inflight += 1
            task = asyncio.create_task(self._handle(order, sem))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _handle(self, order, sem) -> None:
        t0 = time.perf_counter()
        try:
            with self.sys.activity.track("order", order.source, " ".join(order.text.split())[:100]):
                try:
                    result = await self.fulfil(order, review=False)  # a director takes it first (P14)
                    try:  # the emotional engines see every answer before the user does (P8 limits them)
                        await self.sys.affect.process(order, result)
                    except Exception:
                        self.log.exception("temperament could not process %s; the answer passes unchanged", order.id)
                    try:  # the mind notes gaps and may ask the user what they meant
                        self.sys.mind.observe(order, result)
                    except Exception:
                        self.log.exception("the mind could not observe %s", order.id)
                except Exception as exc:
                    self.log.exception("order %s failed", order.id)
                    result = Result(order.id, order.text, None, None, 0.0, ok=False, note=f"internal error: {exc}")
                    voice.finish(result)
                try:  # and a director reviews it last, before the person sees it or memory keeps it (P14)
                    await self.sys.directors.review(order, result)
                except Exception:
                    self.log.exception("the director could not review %s; it is reported as it is", order.id)
        finally:
            self.sys.inflight -= 1
            sem.release()
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.sys.record_order(result)
        self.stats["orders"] += 1
        if order.future is not None and not order.future.done():
            order.future.set_result(result)

    async def fulfil(self, order, review: bool = True) -> Result:
        """Do the order between two directors (P14): one takes it, the tools and nodes do it, one
        reviews it. The reply leads with the result in the mesh's own words (voice.py)."""
        plan = await self.sys.directors.intake(order)
        res = await self._fulfil(order)
        res.trace.insert(0, plan.line)
        voice.finish(res)
        if review:
            await self.sys.directors.review(order, res)
        return res

    async def _fulfil(self, order) -> Result:
        s = self.sys
        res = Result(order.id, order.text, None, None, 0.0)
        if s.guardian.halted:
            res.ok, res.note = False, f"system halted by operator: {s.guardian.halt_reason}"
            res.trace.append("refused (P1-human-authority)")
            return res
        if not order.background:  # "keep learning about X", "every 10m: ..." set a background task
            made = s.tasks.from_order(order.text, by=order.source)
            if made is not None:
                res.route, res.confidence, res.served_by = "tasks", 1.0, "explicit"
                if made[2]:
                    res.output = made[0]
                else:
                    res.ok, res.note = False, made[0]
                res.trace.append(made[1])
                return res
        prompt = imagine_prompt(order.text) if s.dreams.enabled else None
        if prompt is not None:  # "imagine ...": its dream nodes, labelled as imagination (P13)
            self.imagine(res, prompt)
            return res
        own = s.mind.answer_self(order.text, order.id)  # questions about the mesh's own opinions and notes (P9)
        if own is not None:
            res.route, res.output, res.confidence, res.served_by = "mind", own[0], 1.0, "explicit"
            res.trace.append(f"answered by the mind: {own[1]}")
            return res
        route, payload = parse_order(order.text, s.capabilities)
        kinds: list[str] = []
        conf = 1.0
        if route is None and is_question(order.text):  # memory nodes first: what it already learned
            remembered = s.memory.recall(order.text)
            if remembered is not None:
                res.route = DIRECT
                res.trace.append("memory nodes are asked before routing")
                self.answer_from_memory_node(res, remembered)
                res.confidence, res.served_by = round(remembered.confidence, 3), "nodes"
                return res
        if route is None:
            inf = await s.infer(DISPATCH, order.text, learn=order.learn)
            res.steps.append({"capability": DISPATCH, "input": order.text, "label": inf.label,
                              "provenance": inf.provenance, "confidence": round(inf.confidence, 3)})
            res.trace.append(f"dispatch -> {inf.label} via {inf.provenance} ({inf.confidence:.2f})")
            kinds.append(inf.kind)
            conf = inf.confidence
            route = inf.label or DIRECT
            if route.startswith("skill:"):
                payload = skill_payload(order.text)
        else:
            res.trace.append(f"explicit route {route}")
        res.route = route

        if route.startswith("tool:"):
            tool = s.tools.tools.get(route[5:])
            decision = s.guardian.authorize_tool_run(tool)
            if not decision.allowed:
                res.ok, res.note = False, f"tool refused: {decision.reason}"
            elif tool.manifest.provides == "knowledge":
                await self.knowledge_order(res, payload, tool, background=order.background, learn=order.learn)
            else:
                timeout = tool.manifest.timeout_s or s.directives.tools.default_timeout_s
                run = await s.sandbox.run(tool, payload, timeout, s.tool_config(tool))
                s.record_tool_run(tool, run)
                res.trace.append(f"{route} {'ok' if run.ok else 'failed'} in {run.ms:.0f} ms")
                if run.ok:
                    res.output = run.output
                    res.answer = voice.tool_phrase(tool, run)
                else:
                    res.ok, res.note = False, f"tool error: {run.error}"
        elif route.startswith("skill:"):
            cap = route[6:]
            if cap not in s.capabilities:
                res.ok, res.note = False, f"unknown capability '{cap}'"
            else:
                inf = await s.infer(cap, payload, learn=order.learn)
                res.steps.append({"capability": cap, "input": payload, "label": inf.label,
                                  "provenance": inf.provenance, "confidence": round(inf.confidence, 3)})
                res.trace.append(f"{cap} -> {inf.label} via {inf.provenance} ({inf.confidence:.2f})")
                kinds.append(inf.kind)
                conf *= inf.confidence
                if inf.label is None:
                    res.ok, res.note = False, "nothing could label this yet; queued for a human teacher"
                else:
                    res.output = inf.label
                    res.answer = voice.skill_phrase(s.capabilities[cap], inf.label)
        else:
            pol = s.directives.research
            remembered = s.memory.recall(order.text)  # memory nodes first: what it learned, compressed
            hits, confidence, tool = [], 0.0, None
            if remembered is None:
                hits, confidence = s.knowledge.recall(order.text, pol.recall_threshold)
                tool = s.knowledge_tool()
            if remembered is not None:
                self.answer_from_memory_node(res, remembered)
                kinds.append("node")
                conf *= remembered.confidence
            elif hits and confidence >= pol.recall_threshold:
                await self.answer_from_retained(res, order.text, hits, confidence, learn=order.learn)
            elif tool is not None and pol.research_questions and is_question(order.text):
                await self.knowledge_order(res, order.text, tool, background=order.background, learn=order.learn)
            else:
                reply = await s.teachers.answer(order.text)
                if reply:
                    res.output = reply[0]
                    res.answer = voice.lead_first(reply[0])
                    res.sources.append(f"the {reply[1]} teacher (a language model; not checked against a source)")
                    res.trace.append(f"answered directly by the {reply[1]} teacher")
                    kinds.append("teacher")
                    if order.learn and is_question(order.text):  # into a memory node, if the director's review agrees
                        s.directors.remember(order.id, order.text, res.answer, f"the {reply[1]} teacher (a language model)",
                                             by=reply[1], kind="teachers")
                else:
                    res.ok = False
                    res.note = "no tool, skill or teacher can fulfil this yet; queued for a human teacher"
                    s.teachers.human.ask(DISPATCH, order.text, "unroutable order")

        res.confidence = round(conf, 3)
        node_kinds = {"node", "node-lowconf"}
        if not kinds:
            res.served_by = "explicit"
        elif "none" in kinds:
            res.served_by = "none"
        elif all(k in node_kinds for k in kinds):
            res.served_by = "nodes"
        elif all(k == "teacher" for k in kinds):
            res.served_by = "teachers"
        else:
            res.served_by = "mixed"
        return res


    # ------------------------------------------------------------- imagining
    def imagine(self, res: Result, prompt: str) -> None:
        s = self.sys
        res.route, res.knowledge = "dream:imagine", "imagined"
        got = s.dreams.imagine(prompt) if prompt else None
        if got is None:
            res.ok = False
            res.note = ("nothing comes to mind yet: it needs to sleep and dream first (/sleep)" if prompt
                        else "imagine what? (imagine <something>)")
            res.trace.append("no dream node could imagine anything")
            return
        res.output, res.answer, res.confidence, res.served_by = got["text"], got["text"], round(got["ideas"][0][1], 3), "nodes"
        res.sources.append(got["source"])
        res.trace.append(f"imagined by {got['node']}: ideas {', '.join(f'{i} ({c:.0%})' for i, c in got['ideas'])}"
                         + (f"; blended into '{got['blended_from'][:80]}'" if got.get("blended_from") else ""))
        s.bus.publish("dream.imagined", node=got["node"], prompt=prompt[:160], text=got["text"].split("\n")[0][:200])

    # ------------------------------------------------------------- knowledge
    def answer_from_memory_node(self, res: Result, rec) -> None:
        """An answer a memory node holds: knowledge it learned earlier, compressed into its weights."""
        node = rec.node
        res.output, res.answer, res.knowledge = rec.answer, rec.answer, "memory"
        was = f", from {node.first_bytes:,}" if node.first_bytes and node.first_bytes != node.bytes else ""
        source = (f"memory node {node.id} ({node.bytes:,} bytes{was}; {node.passed}/{node.total} tests pass)"
                  + (f", learned from {rec.source}" if rec.source else ""))
        res.sources.append(source)
        res.trace.append(f"answered by memory node {node.id} (confidence {rec.confidence:.2f}; {node.spec()})")
        self.sys.bus.publish("memory.recalled", node=node.id, question=res.text[:200], confidence=rec.confidence)

    def _remember(self, res: Result, question: str, answer: str, source: str) -> None:
        if answer:  # into a memory node, if the director's review agrees (P14)
            self.sys.directors.remember(res.order_id, question, answer, source.split(' ("', 1)[0], by="research",
                                        kind="knowledge")

    async def answer_from_retained(self, res: Result, question: str, hits, confidence: float,
                                   learn: bool = True) -> None:
        s = self.sys
        doc = hits[0].doc
        s.knowledge.touch(doc)
        answer, source, how = await s.answer_from_knowledge(question, hits)
        res.output, res.answer, res.knowledge = answer, answer, "recalled"
        res.sources.append(source)
        res.trace.append(f"recalled from retained knowledge: '{doc.title}' (confidence {confidence:.2f}, "
                         f"retained {_ago(doc.age)} ago; {how})")
        s.bus.publish("knowledge.recalled", question=question[:200], title=doc.title,
                      confidence=confidence, how=how)
        if learn:  # exam questions are never learned from
            self._remember(res, question, answer, source)

    async def knowledge_order(self, res: Result, question: str, tool: Tool, background: bool = False,
                              learn: bool = True) -> None:
        """Memory first: answer from retained knowledge when it covers the question,
        otherwise research it (and retain what was learned), then answer."""
        s = self.sys
        pol = s.directives.research
        remembered = s.memory.recall(question)
        if remembered is not None:
            self.answer_from_memory_node(res, remembered)
            return
        hits, confidence = s.knowledge.recall(question, pol.recall_threshold)
        fresh = bool(hits) and hits[0].doc.age < pol.refresh_days * 86400
        if hits and confidence >= pol.recall_threshold and fresh:
            await self.answer_from_retained(res, question, hits, confidence, learn=learn)
            return
        doc, note = await s.research(question, reason=f"order {res.order_id}", tool=tool, background=background)
        if doc is not None:
            found = s.knowledge.search(question, doc_id=doc.id) or [Hit(doc, "Introduction", doc.summary, 1.0, 1.0)]
            s.knowledge.touch(doc)
            answer, source, how = await s.answer_from_knowledge(question, found)
            res.output, res.answer, res.knowledge = answer, answer, "researched"
            res.sources.append(source)
            n = doc.passages
            res.trace.append(f"researched with {doc.source}: '{doc.title}' ({n} passage{'s' if n != 1 else ''} "
                             f"retained; {how})")
            if learn:
                self._remember(res, question, answer, source)
        elif hits:
            answer, source, how = await s.answer_from_knowledge(question, hits)
            res.output, res.answer, res.knowledge = answer, answer, "recalled"
            res.sources.append(source)
            res.trace.append(f"could not research ({note}); answered from the closest retained knowledge "
                             f"'{hits[0].doc.title}' (confidence {confidence:.2f})")
        else:
            res.ok, res.note = False, f"could not research this: {note}"


_QUESTION = re.compile(r"^\s*(who|what|when|where|why|how|which|tell me about|explain|describe|"
                       r"what's|whats|who's)\b|\?\s*$", re.I)


def is_question(text: str) -> bool:
    return bool(_QUESTION.search(text))


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


# ======================================================================= architect
MUTATIONS = ("widen", "retune", "deepen", "fresh")


class Architect(Agent):
    name = "architect"

    def __init__(self, system):
        super().__init__(system)
        self.last_spawn: dict[str, float] = {}
        self.mutation_index: dict[str, int] = defaultdict(int)

    async def step(self) -> None:
        s = self.sys
        npol = s.directives.nodes
        if not s.guardian.background_allowed():
            return
        for cap, spec in list(s.capabilities.items()):
            champ = s.registry.champion(cap)
            cands = s.registry.candidates(cap)
            counts = s.store.label_counts(cap)
            have = [lab for lab in spec.active_labels() if counts.get(lab, 0) > 0]
            if champ is None:
                if not cands and sum(counts.values()) >= npol.min_examples_to_spawn and len(have) >= 2:
                    node = NeuralNode(cap, have, npol.input_dim,
                                      Architecture(hidden=tuple(npol.default_hidden), lr=npol.default_lr,
                                                   batch_size=npol.batch_size),
                                      origin="seed", seed=s.rng.randrange(1 << 30), backend=s.backend)
                    self._spawn(node, f"capability gap: {sum(counts.values())} examples and no node")
                continue
            missing = [lab for lab in have if lab not in champ.labels]
            if missing:
                if not any(set(have) <= set(c.labels) for c in cands):
                    self._spawn(champ.spawn_expanded(missing),
                                f"new labels {missing}: growing {champ.id}'s output layer")
                continue
            focus = s.tasks.focus()  # improve tasks: their skills get challengers first, to their own target
            evo = s.evolution
            acc = champ.metrics.get("holdout_acc")
            target = max(npol.target_accuracy, focus.get(cap, 0.0))
            below = acc is not None and acc < target
            # below target: breed at the normal pace; at target, evolution keeps going more slowly
            idle = evo.enabled and evo.pol.keep_evolving and acc is not None and not below
            cooldown = npol.challenger_cooldown_seconds / (2 if cap in focus else 1)
            if idle:
                cooldown *= evo.pol.idle_cooldown_factor
            if time.time() - self.last_spawn.get(cap, 0.0) < cooldown:
                continue
            if (below or idle) and len(cands) < npol.max_candidates_per_capability:
                if evo.enabled:
                    child, kind = evo.breed(champ, spec)
                else:
                    child, kind = self._mutate(champ, spec)
                why = (f"champion at {acc:.3f} < target {target}" + (" (a background task)" if cap in focus else "")
                       if below else f"champion at {acc:.3f} meets its target; evolution continues")
                self._spawn(child, f"{why}: {kind}")

    def _mutate(self, champ: NeuralNode, spec) -> tuple[NeuralNode, str]:
        kind = "fresh"
        for _ in range(len(MUTATIONS)):
            kind = MUTATIONS[self.mutation_index[champ.capability] % len(MUTATIONS)]
            self.mutation_index[champ.capability] += 1
            npol = self.sys.directives.nodes
            if kind == "widen" and max(champ.arch.hidden or (0,)) * 1.5 > npol.max_hidden:
                continue
            if kind == "deepen" and len(champ.arch.hidden) >= npol.max_depth:
                continue
            break
        if kind == "widen":
            return champ.spawn_wider(1.5), kind
        if kind == "deepen":
            return champ.spawn_deeper(), kind
        if kind == "retune":
            return champ.spawn_retuned(self.sys.rng.choice([0.5, 2.0])), kind
        npol = self.sys.directives.nodes
        base = int(npol.default_hidden[0]) if npol.default_hidden else 64
        options = [(max(8, base // 2),), (base,), (min(base * 2, npol.max_hidden),),
                   (min(base * 3 // 2, npol.max_hidden), max(8, base // 2))]
        hidden = self.sys.rng.choice([o for o in options if len(o) <= max(1, npol.max_depth)])
        active = [lab for lab in champ.labels if lab not in spec.inactive_labels]
        return champ.spawn_fresh(hidden, labels=active or None), "fresh"

    def _spawn(self, node: NeuralNode, reason: str) -> NeuralNode | None:
        s = self.sys
        decision = s.guardian.authorize_spawn(len(s.registry.live()),
                                              len(s.registry.candidates(node.capability)), node.capability)
        if not decision.allowed:
            self.stats["spawn_denied"] += 1
            return None
        s.registry.add(node, reason)
        self.last_spawn[node.capability] = time.time()
        self.stats["spawned"] += 1
        return node


# ========================================================================= trainer
class Trainer(Agent):
    name = "trainer"

    def __init__(self, system):
        super().__init__(system)
        self.last_synthesis = 0.0
        self.barren: dict[tuple[str, str], float] = {}

    def _pick(self) -> NeuralNode | None:
        s = self.sys
        npol = s.directives.nodes
        focus = s.tasks.focus()
        cands = [n for n in s.registry.candidates() if n.train_steps < npol.max_candidate_slices]
        if cands:
            return min(cands, key=lambda n: (n.capability not in focus, n.train_steps, n.created_at))
        best, gap = None, 0
        for cap in sorted(s.capabilities, key=lambda c: c not in focus):
            champ = s.registry.champion(cap)
            if champ is not None:
                g = s.store.count(cap) - champ.examples_seen
                if g > gap:
                    best, gap = champ, g
        return best

    async def step(self) -> None:
        if not self.sys.guardian.background_allowed():
            return
        await self.synthesize()  # rate-limited by its own cooldown
        node = self._pick()
        if node is not None:
            await self.train(node)

    async def train(self, node: NeuralNode) -> None:
        s = self.sys
        cap = node.capability
        if cap not in s.capabilities:
            return
        count = s.store.count(cap)
        train = [e for e in s.store.examples(cap, "train") if e.label in node.labels]
        if len(train) < 2:
            node.examples_seen = count
            node.train_steps += 1
            return
        with s.activity.track("training", "trainer", f"{node.id} [{node.status}, {node.arch.describe()}] "
                              f"slice {node.train_steps + 1} on {len(train)} examples", node=node.id):
            await self._train_slice(node, train, count)

    async def _train_slice(self, node: NeuralNode, train: list, count: int) -> None:
        s = self.sys
        cap = node.capability
        X = await self.bg(s.encode_many, [e.text for e in train])
        y = node.label_indices([e.label for e in train])
        w = np.array([e.weight for e in train], dtype=np.float32)
        params, opt, loss = await self.bg(node.train_copy, X, y, w, s.directives.background.train_epochs_per_slice)
        if node.status == "champion":
            hold = s.store.examples(cap, "holdout")
            if hold:
                Xh = await self.bg(s.encode_many, [e.text for e in hold])
                labels = [e.label for e in hold]
                old = float((await self.bg(node.correct, Xh, labels)).mean())
                new = float((await self.bg(node.correct, Xh, labels, params)).mean())
                decision = s.guardian.authorize_champion_update(cap, old, new)
                if not decision.allowed:
                    self.stats["updates_rejected"] += 1
                    s.guardian.audit("trainer", "update.rejected", node.id, reason=decision.reason)
                    node.examples_seen = count
                    s.registry.persist(node)
                    return
                node.metrics["holdout_acc"] = round(new, 4)
                node.metrics["holdout_n"] = len(labels)
        node.commit(params, opt, loss)
        node.examples_seen = count
        s.registry.persist(node, weights=True)
        self.stats["slices"] += 1
        s.bus.publish("node.trained", node=node.id, capability=cap, status=node.status,
                      loss=node.metrics.get("last_loss"), steps=node.train_steps, examples=len(train))

    def _weakest(self) -> tuple[str, str, str] | None:
        s = self.sys
        npol, bp = s.directives.nodes, s.directives.background
        now = time.time()
        best = None
        focus = s.tasks.focus()
        for cap, spec in s.capabilities.items():
            counts = s.store.label_counts(cap)
            if not counts and cap not in focus:
                continue  # never used yet and nobody asked: no reason to invest
            options = []
            held_out = Counter(e.label for e in s.store.examples(cap, "holdout"))
            for lab in spec.active_labels():
                c = counts.get(lab, 0)
                if c < bp.min_examples_per_label:
                    options.append((float(c), lab, f"only {c} examples of '{lab}'"))
                elif held_out.get(lab, 0) < 2:
                    options.append((bp.min_examples_per_label - 0.5 + held_out.get(lab, 0), lab,
                                    f"too little held-out evidence for '{lab}'"))
            if cap in focus and sum(held_out.values()) < npol.min_eval_samples and spec.active_labels():
                # an improve task needs enough held-out evidence to tell whether the target is met
                lab = min(spec.active_labels(), key=lambda lb: held_out.get(lb, 0))
                options.append((float(held_out.get(lab, 0)), lab, f"too little held-out evidence for {cap}"))
            champ = s.registry.champion(cap)
            acc = champ.metrics.get("holdout_acc") if champ else None
            if champ and acc is not None and acc < max(npol.target_accuracy, focus.get(cap, 0.0)):
                recall = {k: v for k, v in (champ.metrics.get("per_label_recall") or {}).items()
                          if k in spec.active_labels()}
                if recall:
                    lab, r = min(recall.items(), key=lambda kv: kv[1])
                    options.append((bp.min_examples_per_label + r, lab, f"lowest recall {r:.2f} on '{lab}'"))
            for score, lab, why in options:
                if self.barren.get((cap, lab), 0.0) > now:
                    continue
                if cap in focus:  # a background task asked for this skill to improve
                    score, why = score - 1000, f"{why} (a background task)"
                if best is None or score < best[0]:
                    best = (score, cap, lab, why)
        return best[1:] if best else None

    async def synthesize(self) -> None:
        s = self.sys
        bp = s.directives.background
        if time.time() - self.last_synthesis < bp.synthesis_cooldown_seconds:
            return
        self.last_synthesis = time.time()
        target = self._weakest()
        if target is None:
            return
        cap, label, why = target
        texts, source = await s.teachers.generate(s.capabilities[cap], label, bp.synthesis_batch)
        added = sum(1 for t in texts if s.learn(cap, t, label, source))
        if not added:
            self.barren[(cap, label)] = time.time() + 10 * bp.synthesis_cooldown_seconds
            return
        self.stats["lessons"] += 1
        s.bus.publish("teacher.lesson", capability=cap, label=label, source=source, examples=added, reason=why)


# ======================================================================= evaluator
def _per_label_recall(ok: np.ndarray, labels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    arr = np.array(labels)
    for lab in sorted(set(labels)):
        mask = arr == lab
        out[lab] = round(float(ok[mask].mean()), 3)
    return out


class Evaluator(Agent):
    name = "evaluator"

    def __init__(self, system):
        super().__init__(system)
        self._seen: dict[str, tuple] = {}

    async def step(self) -> None:
        s = self.sys
        npol = s.directives.nodes
        if not s.guardian.background_allowed():
            return
        for cap in list(s.capabilities):
            champ = s.registry.champion(cap)
            cands = [c for c in s.registry.candidates(cap) if c.train_steps >= npol.min_train_steps_before_eval]
            if champ is None and not cands:
                continue
            nodes = ([champ] if champ else []) + cands
            key = (s.store.count(cap), tuple((n.id, n.train_steps) for n in nodes))
            if self._seen.get(cap) == key:
                continue
            hold = s.store.examples(cap, "holdout")
            labels = [e.label for e in hold]
            oks: dict[str, np.ndarray] = {}
            if hold:
                with s.activity.track("evaluation", "evaluator",
                                      f"{cap}: {len(nodes)} node(s) on {len(hold)} held-out examples"):
                    X = await self.bg(s.encode_many, [e.text for e in hold])
                    for n in nodes:
                        ok = await self.bg(n.correct, X, labels)
                        oks[n.id] = ok
                        n.metrics["holdout_acc"] = round(float(ok.mean()), 4)
                        n.metrics["holdout_n"] = len(labels)
                        n.metrics["per_label_recall"] = _per_label_recall(ok, labels)
                s.bus.publish("node.evaluated", capability=cap, holdout=len(hold),
                              results={n.id: n.metrics["holdout_acc"] for n in nodes})
            for n in nodes:
                n.metrics["evaluated_steps"] = n.train_steps
                s.registry.persist(n)
            self._seen[cap] = key
            self.stats["evaluations"] += 1

            if champ is None:
                best = max(cands, key=lambda n: n.metrics.get("holdout_acc") or 0.0)
                s.registry.promote(best.id, "first node for this capability",
                                   {"holdout_acc": best.metrics.get("holdout_acc"), "holdout_n": len(labels)})
                self.stats["promotions"] += 1
                continue
            if len(labels) < npol.min_eval_samples or not cands:
                continue
            c_ok = oks[champ.id]
            c_acc = float(c_ok.mean())
            counts = s.store.label_counts(cap)
            # labels with data that the champion structurally cannot output (e.g. a new tool)
            uncovered = {lab for lab in s.capabilities[cap].active_labels()
                         if counts.get(lab, 0) > 0 and lab not in champ.labels}
            evo = s.evolution
            for n in nodes:
                evo.record(n)
            # fittest first: accuracy, then (with the growth trend) size
            for cand in sorted(cands, key=lambda n: (n.metrics.get("holdout_acc") or 0.0, n.param_count), reverse=True):
                ok = oks[cand.id]
                wins, losses = int((ok & ~c_ok).sum()), int((~ok & c_ok).sum())
                z = (wins - losses) / math.sqrt(wins + losses) if wins + losses else 0.0
                margin = float(ok.mean()) - c_acc
                gain = sorted(uncovered & set(cand.labels))
                promote, why = evolution_judge(c_ok, ok, champ.param_count, cand.param_count,
                                               margin=npol.promotion_margin, z_needed=npol.promotion_z,
                                               larger_on_tie=evo.enabled and evo.pol.promote_larger_on_tie)
                better = why == "better"
                grown = promote and not better and cand.train_steps >= npol.min_train_steps_before_eval * 2
                if better or grown or (gain and margin >= 0):
                    decision = s.guardian.authorize_promotion(cap, c_acc, float(ok.mean()))
                    if decision.allowed:
                        evidence = {"candidate_acc": round(float(ok.mean()), 4), "champion_acc": round(c_acc, 4),
                                    "wins": wins, "losses": losses, "z": round(z, 2), "holdout_n": len(labels),
                                    "labels_gained": gain}
                        reason = (f"beat the champion on held-out data ({ok.mean():.3f} vs {c_acc:.3f}, z={z:.2f})"
                                  if better else
                                  f"at least as accurate on held-out data ({ok.mean():.3f} vs {c_acc:.3f}) and larger "
                                  f"({cand.arch.describe()}, {cand.param_count:,} weights vs {champ.param_count:,}): "
                                  f"evolution favours more capacity" if grown else
                                  f"adds {gain} with no held-out regression ({ok.mean():.3f} vs {c_acc:.3f})")
                        evidence["evolution"] = cand.metrics.get("evolution")
                        s.registry.promote(cand.id, reason, evidence)
                        self.stats["promotions"] += 1
                        break


# ========================================================================= curator
class Curator(Agent):
    name = "curator"

    async def step(self) -> None:
        s = self.sys
        if s.guardian.halted or s.guardian.paused:
            return
        npol = s.directives.nodes
        now = time.time()
        for n in s.registry.retired():
            if now - (n.retired_at or now) >= npol.retired_grace_seconds:
                s.registry.dispose(n.id, f"superseded; {npol.retired_grace_seconds:g}s rollback window elapsed")
                self.stats["disposed"] += 1
        for n in s.registry.candidates():
            evaluated = n.metrics.get("evaluated_steps") == n.train_steps
            too_old = now - n.created_at > npol.candidate_max_age_seconds
            champ = s.registry.champion(n.capability)
            ca = champ.metrics.get("holdout_acc") if champ else None
            # a challenger that is ahead but still waiting for enough held-out evidence may stay
            ahead = (n.metrics.get("holdout_acc") or 0.0) > (ca or 0.0)
            if (n.train_steps >= npol.max_candidate_slices and evaluated and not ahead) or too_old:
                s.registry.dispose(n.id, f"failed challenger (held-out {n.metrics.get('holdout_acc')} "
                                         f"vs champion {ca})")
                self.stats["disposed"] += 1
        for n in s.registry.live():
            if n.capability not in s.capabilities:
                s.registry.dispose(n.id, "capability no longer exists")
        live = s.registry.live()
        ceiling = max(1, npol.max_nodes - 2)
        if len(live) > ceiling:
            for n in sorted(s.registry.retired(), key=lambda n: n.retired_at or 0)[: len(live) - ceiling]:
                s.registry.dispose(n.id, "node budget pressure (P4)")
        for title in s.knowledge.enforce_cap(s.directives.research.max_documents):
            s.bus.publish("knowledge.forgotten", title=title, reason="retention cap (least recently used)")


# ======================================================================== explorer
class Explorer(Agent):
    name = "explorer"

    def __init__(self, system):
        super().__init__(system)
        self._reported_invalid: set[str] = set()

    @property
    def interval(self) -> float:
        return self.sys.directives.tools.scan_seconds

    async def step(self) -> None:
        s = self.sys
        if s.guardian.halted or s.guardian.paused:
            return
        for manifest, change, name in s.tools.scan():
            await self._on_change(manifest, change, name)
        for tool in list(s.tools.tools.values()):
            if tool.state == PENDING:
                self._check_approval(tool)
        for tool in [t for t in s.tools.tools.values() if t.state == APPROVED]:
            await self._self_test(tool)
        interval = s.directives.background.health_check_seconds
        for tool in s.tools.active():
            if "network" in tool.manifest.permissions and not s.unlimited(tool):
                continue  # no background traffic to outside services just to check health (your own server is fine)
            if time.time() - tool.last_health > interval and s.inflight == 0 and s.guardian.background_allowed():
                await self._health_check(tool)

    async def _on_change(self, manifest, change: str, name: str) -> None:
        s = self.sys
        if change.startswith("invalid"):
            if name not in self._reported_invalid:
                self._reported_invalid.add(name)
                s.guardian.audit("explorer", "tool.invalid", name, error=change)
                s.bus.publish("tool.invalid", tool=name, error=change)
            return
        if change == "restored":
            s.bus.publish("tool.restored", tool=name, state=s.tools.tools[name].state)
            return
        if change == "removed":
            s.retire_tool(s.tools.tools[name], "plugin folder removed")
            return
        if change == "changed":
            old = s.tools.tools[name]
            if old.state == ACTIVE:
                s.dispatch.deactivate(old.manifest.route)
                s.save_capability(s.dispatch)
            s.guardian.audit("explorer", "tool.changed", name, reason="code changed after vetting; re-vetting")
        tool = Tool(manifest)
        s.tools.tools[manifest.name] = tool
        s.bus.publish("tool.discovered", tool=manifest.name, version=manifest.version,
                      permissions=list(manifest.permissions), change=change)
        decision = s.guardian.authorize_tool(manifest)
        if decision.verdict == Verdict.DENY:
            tool.state, tool.note = REJECTED, decision.reason
            s.bus.publish("tool.rejected", tool=manifest.name, reason=decision.reason)
        elif decision.verdict == Verdict.NEEDS_APPROVAL:
            tool.state, tool.note = PENDING, decision.reason
            approval = s.guardian.request_approval("tool", manifest.name, {
                "permissions": list(manifest.permissions), "version": manifest.version,
                "description": manifest.description})
            tool.approval_id = approval.id
        else:
            tool.state, tool.note = APPROVED, "auto-approved: no elevated permissions"
            s.bus.publish("tool.approved", tool=manifest.name, by="policy")
        s.tools.persist(tool)

    def _check_approval(self, tool: Tool) -> None:
        s = self.sys
        approval = s.guardian.approvals.get(tool.approval_id or "")
        if approval is None:  # e.g. after a restart: ask again
            approval = s.guardian.request_approval("tool", tool.name, {
                "permissions": list(tool.manifest.permissions), "version": tool.manifest.version,
                "description": tool.manifest.description})
            tool.approval_id = approval.id
        elif approval.status == "approved":
            tool.state, tool.note = APPROVED, f"approved by {approval.decided_by}"
            s.bus.publish("tool.approved", tool=tool.name, by=approval.decided_by)
            s.tools.persist(tool)
        elif approval.status == "denied":
            tool.state, tool.note = REJECTED, f"denied by {approval.decided_by}"
            s.bus.publish("tool.rejected", tool=tool.name, reason=tool.note)
            s.tools.persist(tool)

    async def _self_test(self, tool: Tool) -> None:
        s = self.sys
        examples = tool.manifest.examples
        if not examples:
            tool.state, tool.note = QUARANTINED, "manifest declares no self-tests"
            s.tools.persist(tool)
            s.bus.publish("tool.quarantined", tool=tool.name, reason=tool.note)
            return
        timeout = tool.manifest.timeout_s or s.directives.tools.default_timeout_s
        passed, failures = 0, []
        with s.activity.track("self-test", "explorer", f"{tool.name} v{tool.manifest.version}: "
                              f"{len(examples)} sandboxed self-tests"):
            for ex in examples:
                run = await s.sandbox.run(tool, ex["input"], timeout, s.tool_config(tool))
                if run.ok and outputs_match(run.output, ex.get("expect")):
                    passed += 1
                else:
                    failures.append({"input": ex["input"], "expected": ex.get("expect"),
                                     "got": run.output, "error": run.error})
        rate = passed / len(examples)
        if rate >= s.directives.tools.self_test_pass_rate:
            s.activate_tool(tool, f"self-tests passed {passed}/{len(examples)}")
        else:
            tool.state = QUARANTINED
            tool.note = f"self-tests passed {passed}/{len(examples)}"
            tool.last_error = str(failures[:2])
            s.tools.persist(tool)
            s.guardian.audit("explorer", "tool.quarantined", tool.name, failures=failures[:3])
            s.bus.publish("tool.quarantined", tool=tool.name, reason=tool.note)

    async def _health_check(self, tool: Tool) -> None:
        s = self.sys
        ex = tool.manifest.examples[0]
        run = await s.sandbox.run(tool, ex["input"], tool.manifest.timeout_s or s.directives.tools.default_timeout_s,
                                  s.tool_config(tool))
        if run.ok and not outputs_match(run.output, ex.get("expect")):
            run.ok, run.error = False, f"health check expected {ex.get('expect')!r}, got {run.output!r}"
        tool.last_health = time.time()
        s.record_tool_run(tool, run)


# ====================================================================== researcher
class Researcher(Agent):
    """Background research within its own budget: reads up on articles linked from what
    users asked about (curiosity, depth-limited) and re-reads retained articles once they
    are older than refresh_days. Never runs while an order is in flight."""
    name = "researcher"

    def __init__(self, system):
        super().__init__(system)
        self.queue: deque[tuple[str, str, int]] = deque()
        self._queued: set[str] = set()
        self._last_refresh_scan = 0.0

    def enqueue(self, topic: str, reason: str, depth: int) -> bool:
        key = topic.strip().lower()
        if not key or key in self._queued or self.sys.knowledge.get(topic) is not None:
            return False
        self._queued.add(key)
        self.queue.append((topic, reason, depth))
        return True

    async def step(self) -> None:
        s = self.sys
        pol = s.directives.research
        if not pol.enabled or s.inflight or not s.guardian.background_allowed():
            return
        tool = s.knowledge_tool()
        if tool is None:
            return
        if time.time() - self._last_refresh_scan > 60:
            self._last_refresh_scan = time.time()
            for doc in s.knowledge.stale(pol.refresh_days, limit=3):
                key = doc.title.lower()
                if key not in self._queued:
                    self._queued.add(key)
                    self.queue.append((doc.title, f"refresh (retained {doc.age / 86400:.0f} days ago)", pol.max_depth))
        if not self.queue or not s.guardian.authorize_research(background=True, unlimited=s.unlimited(tool)).allowed:
            return
        topic, reason, depth = self.queue.popleft()
        self._queued.discard(topic.strip().lower())
        doc, _ = await s.research(topic, reason=reason, background=True, tool=tool)
        self.stats["researched" if doc else "failed"] += 1
        if doc is not None and depth < pol.max_depth:
            for link in doc.links[: pol.follow_links]:
                self.enqueue(link, f"linked from '{doc.title}'", depth + 1)


# ==================================================================== investigator
class Investigator(Agent):
    """The mesh's own initiative.

    Reflection: re-forms its opinions and turns evidence into proposals (the mind).
    Investigation: works through goals with whatever it has. The router node picks a tool or
    skill; approved tools are run (exploratory: a failure never counts against a tool); research
    tools look things up and the mesh retains them; then every skill node reads what was found,
    which is how the mesh forms opinions of what it reads about. Goals come from the user's
    interests, opinions it was asked for, orders it could not fulfil (retried later, with a
    follow-up when it now can) and /investigate. Background only, within its own hourly budget
    (P4). It can ask a human to approve a tool it wants; it never approves one itself (P3)."""
    name = "investigator"

    def __init__(self, system):
        super().__init__(system)
        self.last_reflect = 0.0
        self.quarantine_seen: dict[str, float] = {}
        self.history: deque[dict] = deque(maxlen=50)

    @property
    def interval(self) -> float:
        return max(0.25, self.sys.directives.background.tick_seconds * 4)

    async def step(self) -> None:
        s = self.sys
        mind, pol = s.mind, s.directives.mind
        if not mind.enabled or s.guardian.halted:
            return
        if time.time() - self.last_reflect >= pol.reflect_seconds:
            self.last_reflect = time.time()
            await mind.reflect()
        if not pol.investigate or s.inflight or not s.guardian.background_allowed():
            return
        self._retest_quarantined()
        if mind.waiting and s.knowledge_tool() is not None:
            for g in mind.waiting:
                mind.goals.append(g)
            mind.waiting.clear()
        for u in mind.due_unmet():
            u.attempts += 1
            u.next_try = mind.clock() + pol.retry_unmet_after_seconds * (u.attempts + 1)
            mind.add_goal(u.text, f"an order I couldn't fulfil ({u.order_id})", "unmet", u.order_id)
        if not mind.goals or not s.guardian.authorize_investigation().allowed:
            return
        await self.investigate(mind.goals.popleft())

    def _retest_quarantined(self) -> None:
        """Give a quarantined tool another go at its self-tests after a cool-down (it stays
        quarantined unless they pass)."""
        s = self.sys
        cool = s.directives.mind.retest_quarantined_after_seconds
        if cool <= 0:
            return
        now = time.time()
        for tool in s.tools.tools.values():
            if tool.state != QUARANTINED:
                self.quarantine_seen.pop(tool.name, None)
                continue
            if not tool.manifest.examples:
                continue
            since = self.quarantine_seen.setdefault(tool.name, now)
            if now - since >= cool and s.retest_tool(tool.name)[0]:
                self.quarantine_seen.pop(tool.name, None)
                s.guardian.audit("investigator", "tool.retest", tool.name,
                                 reason=f"quarantined for {cool:.0f}s; running its self-tests again")
                s.bus.publish("mind.retest", tool=tool.name, after=cool)

    async def investigate(self, goal) -> dict:
        s = self.sys
        mind = s.mind
        rpol = s.directives.research
        text = goal.text
        short = " ".join(text.split())[:80]
        steps: list[str] = []
        found: list[str] = []
        answer: str | None = None
        docs = []
        s.guardian.record_investigation()
        with s.activity.track("investigation", "investigator", f"{goal.kind}: {short}"):
            hits, conf = s.knowledge.recall(text, rpol.recall_threshold)
            if hits and conf >= rpol.recall_threshold:
                docs.append(hits[0].doc)
                steps.append(f"recalled '{hits[0].doc.title}' from memory ({conf:.2f})")
            if goal.kind in ("opinion", "interest", "curiosity"):
                route = DIRECT  # a topic to read up on, not a task for a tool or skill
            else:  # a task: let the router node pick the tool or skill, as for an order
                inf = await s.infer(DISPATCH, text, learn=False)
                route = inf.label or DIRECT
                steps.append(f"router -> {route} via {inf.provenance} ({inf.confidence:.2f})")
            if route.startswith("tool:"):
                tool = s.tools.tools.get(route[5:])
                if tool is not None and tool.manifest.provides == "knowledge":
                    pass  # researched below
                elif tool is not None and s.guardian.authorize_tool_run(tool).allowed:
                    run = await s.sandbox.run(tool, text, tool.manifest.timeout_s or s.directives.tools.default_timeout_s,
                                              s.tool_config(tool))
                    s.record_tool_run(tool, run, exploratory=True)
                    steps.append(f"ran {tool.name}: " + ("ok" if run.ok else f"failed ({run.error[:80]})"))
                    if run.ok:
                        answer = voice.tool_phrase(tool, run)
                        found.append(f"the {tool.name} tool gives {str(run.output)[:200]}")
                else:
                    steps.append(f"{route} is not usable now")
            elif route.startswith("skill:") and route[6:] in s.capabilities:
                cap = route[6:]
                read = await s.infer(cap, skill_payload(text), learn=False)
                steps.append(f"{cap} -> {read.label} via {read.provenance}")
                if read.label:
                    answer = voice.skill_phrase(s.capabilities[cap], read.label)
                    found.append(f"my {cap} skill reads it as {read.label}")
            if answer is None and not docs:
                tool = s.knowledge_tool()
                if tool is not None:
                    doc, note = await s.research(text, reason=f"investigation: {goal.reason}", background=True,
                                                 tool=tool)
                    steps.append(f"researched with {doc.source}: '{doc.title}'" if doc else f"could not research: {note}")
                    if doc is not None:
                        docs.append(doc)
                else:
                    pending = next((t for t in s.tools.tools.values()
                                    if t.manifest.provides == "knowledge" and t.state == PENDING and t.approval_id), None)
                    if pending is not None:
                        q = mind.ask("approval", f"I'd like to look into '{short}' on my own, but the {pending.name} "
                                     f"tool needs your approval (permissions: {', '.join(pending.manifest.permissions)}). "
                                     f"Approve it? (yes/no)", "to research on my own", approval_id=pending.approval_id)
                        steps.append(f"asked the user to approve {pending.name} ({q.id})" if q else
                                     f"{pending.name} is waiting for approval")
                        if goal.kind != "unmet" and all(g.text != goal.text for g in mind.waiting):
                            mind.waiting.append(goal)  # picked up again once the tool is approved
                    else:
                        steps.append("no research tool available")
            for doc in docs[:1]:
                passages = s.knowledge.search(text, doc_id=doc.id) or [Hit(doc, "Introduction", doc.summary, 1.0, 1.0)]
                s.knowledge.touch(doc)
                if answer is None:
                    body, source, how = await s.answer_from_knowledge(text, passages)
                    answer = f"{body}\nSource: {source}"
                    steps.append(f"answer {how}")
                found.append(f"'{doc.title}': {' '.join(doc.summary.split())[:200]}")
                found.extend(await self._read_with_nodes(doc, steps))
        if goal.kind == "opinion" and mind.find_opinion(text) is not None:
            answer = (mind.answer_self(f"what do you think of {text}") or (None,))[0]
        elif goal.kind == "opinion":
            answer = None
        finding = "; ".join(found)
        mind.record_investigation(goal, steps, finding, answer)
        record = {"goal": text, "kind": goal.kind, "reason": goal.reason, "steps": steps, "finding": finding,
                  "answer": answer}
        self.history.append(record)
        self.stats["investigations"] += 1
        return record

    async def _read_with_nodes(self, doc, steps: list[str]) -> list[str]:
        """Read what was found with every skill node; the sentiment reading becomes an opinion."""
        s = self.sys
        hits = s.knowledge.search(doc.title, k=5, doc_id=doc.id)
        passages = [" ".join(h.text.split())[:600] for h in hits] or [doc.summary[:600]]
        readings = []
        for cap, spec in s.capabilities.items():
            if cap == DISPATCH or spec.internal or len(spec.active_labels()) < 2:
                continue
            labels = []
            for p in passages:
                inf = await s.infer(cap, p, learn=False)
                if inf.label:
                    labels.append((inf.label, inf.provenance.split(" ")[0]))
            if not labels:
                continue
            counts = Counter(lab for lab, _ in labels)
            readings.append(f"my {cap} skill reads it as " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
            if cap == "sentiment":
                by = Counter(p for _, p in labels).most_common(1)[0][0]
                s.mind.topic_opinion(doc.title, labels, by)
        steps.append(f"read {len(passages)} passages of '{doc.title}' with my skill nodes")
        return readings


# ====================================================================== task runner
class TaskRunner(Agent):
    """Runs the background tasks people set (learn / improve / run), one at a time, whenever
    there is background capacity: never while an order is in flight, within the duty cycle (P2)
    and the hourly task budget (P4). /halt and /pause stop it (P1)."""
    name = "tasks"

    @property
    def interval(self) -> float:
        return max(0.1, self.sys.directives.background.tick_seconds * 2)

    async def step(self) -> None:
        s = self.sys
        if not s.tasks.enabled or s.inflight or not s.guardian.background_allowed():
            return
        task = s.tasks.next_due()
        if task is None or not s.guardian.authorize_task_run().allowed:
            return
        await s.tasks.run(task)
        self.stats["runs"] += 1


# ====================================================================== compressor
class Compressor(Agent):
    """Compresses what the mesh learns into memory nodes, then keeps trying smaller ones: a smaller
    node is accepted only if every answer still matches the node's external test file (P11). It
    notices people's edits to the test files and retrains to match them. Background work: it waits
    for orders (P2) and stops on /halt or /pause (P1)."""
    name = "compressor"

    def __init__(self, system):
        super().__init__(system)
        self._next = 0.0

    @property
    def interval(self) -> float:
        return max(0.05, self.sys.directives.background.tick_seconds)

    async def step(self) -> None:
        s = self.sys
        mem = s.memory
        if not mem.enabled or not s.guardian.background_allowed():
            return
        now = time.time()
        if now < self._next:
            return
        self._next = now + mem.pol.interval_seconds
        mem.sync_files()
        job = mem.next_job()
        if job is None:
            return
        with s.activity.track("memory", "compressor", job.describe()):
            result = await self.bg(job.run)
        mem.finish(job, result)
        self.stats[job.kind] += 1


# ======================================================================== sentinel
class Sentinel(Agent):
    name = "sentinel"

    def __init__(self, system):
        super().__init__(system)
        self.last_beat = 0.0

    @property
    def interval(self) -> float:
        return 0.25

    async def step(self) -> None:
        s = self.sys
        s.guardian.check_kill_switch()
        if not s.guardian.halted:
            await s.teachers.refresh()  # rate-limited per teacher (probe_seconds)
        if time.time() - self.last_beat >= 30:
            self.last_beat = time.time()
            s.affect.tick()  # feelings settle towards the baseline; the mood is persisted
            s.evolution.save()
            s.bus.publish("system.heartbeat", scorecard=s.scorecard(), live_nodes=len(s.registry.live()),
                          duty_cycle=round(s.guardian.duty_cycle(), 3))
