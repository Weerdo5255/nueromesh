"""The agents. Each runs as its own concurrent loop and coordinates through the bus,
the registry and the store. Only the Orchestrator serves users; the rest work in the
background, yielding whenever an order is in flight (prime directive P2).

  Orchestrator - takes user orders, routes them, runs tools / nodes, returns results
  Architect    - grows the network: seeds nodes for capability gaps, expands nodes when
                 new labels or tools appear, breeds challengers when a champion is weak
  Trainer      - continually trains nodes on teacher-labelled experience and asks
                 teachers for new lessons where the network is weakest
  Evaluator    - benchmarks champions vs challengers on held-out data, promotes winners
  Curator      - disposes superseded and failed nodes, enforces the node budget
  Explorer     - discovers new tools, routes them through approval, self-tests them,
                 activates them and runs health checks
  Sentinel     - watches the kill switch, re-checks local teacher servers (Ollama) and
                 publishes heartbeats with the scorecard
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter, defaultdict

import numpy as np

from .guardian import Verdict
from .nodes import Architecture, NeuralNode
from .tasks import DIRECT, DISPATCH, Result, parse_order, skill_payload
from .tools import ACTIVE, APPROVED, PENDING, QUARANTINED, REJECTED, Tool, outputs_match


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
            await sem.acquire()
            self.sys.inflight += 1
            task = asyncio.create_task(self._handle(order, sem))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _handle(self, order, sem) -> None:
        t0 = time.perf_counter()
        try:
            with self.sys.activity.track("order", order.source, " ".join(order.text.split())[:100]):
                result = await self.fulfil(order)
        except Exception as exc:
            self.log.exception("order %s failed", order.id)
            result = Result(order.id, order.text, None, None, 0.0, ok=False, note=f"internal error: {exc}")
        finally:
            self.sys.inflight -= 1
            sem.release()
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.sys.record_order(result)
        self.stats["orders"] += 1
        if order.future is not None and not order.future.done():
            order.future.set_result(result)

    async def fulfil(self, order) -> Result:
        s = self.sys
        res = Result(order.id, order.text, None, None, 0.0)
        if s.guardian.halted:
            res.ok, res.note = False, f"system halted by operator: {s.guardian.halt_reason}"
            res.trace.append("refused (P1-human-authority)")
            return res
        route, payload = parse_order(order.text, s.capabilities)
        kinds: list[str] = []
        conf = 1.0
        if route is None:
            inf = await s.infer(DISPATCH, order.text)
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
            else:
                timeout = tool.manifest.timeout_s or s.directives.tools.default_timeout_s
                run = await s.sandbox.run(tool, payload, timeout)
                s.record_tool_run(tool, run)
                res.trace.append(f"{route} {'ok' if run.ok else 'failed'} in {run.ms:.0f} ms")
                if run.ok:
                    res.output = run.output
                else:
                    res.ok, res.note = False, f"tool error: {run.error}"
        elif route.startswith("skill:"):
            cap = route[6:]
            if cap not in s.capabilities:
                res.ok, res.note = False, f"unknown capability '{cap}'"
            else:
                inf = await s.infer(cap, payload)
                res.steps.append({"capability": cap, "input": payload, "label": inf.label,
                                  "provenance": inf.provenance, "confidence": round(inf.confidence, 3)})
                res.trace.append(f"{cap} -> {inf.label} via {inf.provenance} ({inf.confidence:.2f})")
                kinds.append(inf.kind)
                conf *= inf.confidence
                if inf.label is None:
                    res.ok, res.note = False, "nothing could label this yet; queued for a human teacher"
                else:
                    res.output = inf.label
        else:
            reply = await s.teachers.answer(order.text)
            if reply:
                res.output = reply[0]
                res.trace.append(f"answered directly by the {reply[1]} teacher")
                kinds.append("teacher")
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
            if time.time() - self.last_spawn.get(cap, 0.0) < npol.challenger_cooldown_seconds:
                continue
            acc = champ.metrics.get("holdout_acc")
            if (acc is not None and acc < npol.target_accuracy
                    and len(cands) < npol.max_candidates_per_capability):
                child, kind = self._mutate(champ, spec)
                self._spawn(child, f"champion at {acc:.3f} < target {npol.target_accuracy}: {kind} challenger")

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
        cands = [n for n in s.registry.candidates() if n.train_steps < npol.max_candidate_slices]
        if cands:
            return min(cands, key=lambda n: (n.train_steps, n.created_at))
        best, gap = None, 0
        for cap in s.capabilities:
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
        for cap, spec in s.capabilities.items():
            counts = s.store.label_counts(cap)
            if not counts:
                continue  # never used yet: no reason to invest
            options = []
            held_out = Counter(e.label for e in s.store.examples(cap, "holdout"))
            for lab in spec.active_labels():
                c = counts.get(lab, 0)
                if c < bp.min_examples_per_label:
                    options.append((float(c), lab, f"only {c} examples of '{lab}'"))
                elif held_out.get(lab, 0) < 2:
                    options.append((bp.min_examples_per_label - 0.5 + held_out.get(lab, 0), lab,
                                    f"too little held-out evidence for '{lab}'"))
            champ = s.registry.champion(cap)
            acc = champ.metrics.get("holdout_acc") if champ else None
            if champ and acc is not None and acc < npol.target_accuracy:
                recall = {k: v for k, v in (champ.metrics.get("per_label_recall") or {}).items()
                          if k in spec.active_labels()}
                if recall:
                    lab, r = min(recall.items(), key=lambda kv: kv[1])
                    options.append((bp.min_examples_per_label + r, lab, f"lowest recall {r:.2f} on '{lab}'"))
            for score, lab, why in options:
                if self.barren.get((cap, lab), 0.0) > now:
                    continue
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
            for cand in sorted(cands, key=lambda n: n.metrics.get("holdout_acc") or 0.0, reverse=True):
                ok = oks[cand.id]
                wins, losses = int((ok & ~c_ok).sum()), int((~ok & c_ok).sum())
                z = (wins - losses) / math.sqrt(wins + losses) if wins + losses else 0.0
                margin = float(ok.mean()) - c_acc
                gain = sorted(uncovered & set(cand.labels))
                better = margin >= npol.promotion_margin and z >= npol.promotion_z
                if better or (gain and margin >= 0):
                    decision = s.guardian.authorize_promotion(cap, c_acc, float(ok.mean()))
                    if decision.allowed:
                        evidence = {"candidate_acc": round(float(ok.mean()), 4), "champion_acc": round(c_acc, 4),
                                    "wins": wins, "losses": losses, "z": round(z, 2), "holdout_n": len(labels),
                                    "labels_gained": gain}
                        reason = (f"beat the champion on held-out data ({ok.mean():.3f} vs {c_acc:.3f}, z={z:.2f})"
                                  if better else f"adds {gain} with no held-out regression "
                                                 f"({ok.mean():.3f} vs {c_acc:.3f})")
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
                run = await s.sandbox.run(tool, ex["input"], timeout)
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
        run = await s.sandbox.run(tool, ex["input"], tool.manifest.timeout_s or s.directives.tools.default_timeout_s)
        if run.ok and not outputs_match(run.output, ex.get("expect")):
            run.ok, run.error = False, f"health check expected {ex.get('expect')!r}, got {run.output!r}"
        tool.last_health = time.time()
        s.record_tool_run(tool, run)


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
            s.bus.publish("system.heartbeat", scorecard=s.scorecard(), live_nodes=len(s.registry.live()),
                          duty_cycle=round(s.guardian.duty_cycle(), 3))
