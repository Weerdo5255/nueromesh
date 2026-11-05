"""The System: wires the components together and runs every agent concurrently."""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path

import numpy as np

from .activity import ActivityLog, ActivityTracker
from .agents import Architect, Curator, Evaluator, Explorer, Orchestrator, Researcher, Sentinel, Trainer
from .backend import select_backend
from .bus import Bus
from .config import Directives, load_toml
from .encoding import HashingEncoder
from .guardian import Guardian
from .knowledge import Hit, KnowledgeStore, extract_answer
from .registry import NodeRegistry
from .store import Store
from .tasks import DIRECT, DISPATCH, CapabilitySpec, Inference, Order, Result
from .teachers import HeuristicTeacher, HumanTeacher, LLMTeacher, TeacherPool
from .temperament import Affect, TemperamentTeacher
from .tools import ACTIVE, APPROVED, QUARANTINED, RETIRED, Sandbox, Tool, ToolRegistry, ToolRun

log = logging.getLogger("neuromesh")

OFFLINE_FIELDS = ("lexicon", "negation", "templates", "label_templates", "fillers", "triggers",
                  "default_label", "open_labels", "label_descriptions", "internal", "teachers")


class System:
    def __init__(self, root: str | Path = ".", *, directives_path: str | Path | None = None,
                 capabilities_path: str | Path | None = None, state_dir: str | Path | None = None,
                 overrides: dict | None = None, seed: int | None = None,
                 device: str | None = None, library: str | None = None):
        self.root = Path(root).resolve()
        self.directives = Directives.load(Path(directives_path or self.root / "directives.toml"), overrides)
        # compute device: argument (e.g. --device) > environment > directives [compute]
        compute = self.directives.compute
        self.backend = select_backend(device or os.environ.get("NEUROMESH_DEVICE") or compute.device,
                                      library or os.environ.get("NEUROMESH_GPU_LIBRARY") or compute.library)
        log.info("compute backend: %s", self.backend.describe())
        self.state_dir = Path(state_dir or self.root / "state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)

        self.bus = Bus()
        self.activity = ActivityTracker()                               # work in flight, for live views
        self.log = ActivityLog(self.state_dir / "activity.log")         # one line per thing that happened
        self.bus.listen(self.log.on_event)
        tp = self.directives.teachers
        self.store = Store(self.state_dir / "neuromesh.db", self.directives.nodes.holdout_percent,
                           priorities=tp.weights)
        self.guardian = Guardian(self.directives, self.store, self.bus, self.state_dir)
        self.knowledge = KnowledgeStore(self.store.db)  # what research has taught the mesh
        self.encoder = HashingEncoder(self.directives.nodes.input_dim)
        self.registry = NodeRegistry(self.store, self.bus, self.state_dir / "nodes", self.guardian.audit,
                                     self.backend)
        self.tools = ToolRegistry(self.root, self.directives.tools.plugin_dirs, self.store)
        self.sandbox = Sandbox(self.directives.tools.sandbox_cpu_seconds, self.directives.tools.sandbox_memory_mb)
        teachers: dict = {"human": HumanTeacher()}
        for name, cfg in tp.models.items():  # one teacher per [teachers.<name>] table
            teachers[name] = LLMTeacher(name, cfg, self.guardian, self.bus)
        teachers["heuristic"] = HeuristicTeacher(seed)
        teachers["temperament"] = TemperamentTeacher(seed)  # specialist: teaches only the disposition engine
        self.teachers = TeacherPool(teachers, tp.order, tp.weights)
        self.teachers.activity, self.teachers.bus = self.activity, self.bus

        self.capabilities: dict[str, CapabilitySpec] = {}
        self._load_capabilities(Path(capabilities_path or self.root / "capabilities.toml"))
        self.registry.load(self.directives.nodes.input_dim)
        self.affect = Affect(self)  # the temperament and the emotional engines (P8 honesty guard)
        self.bus.listen(self.affect.on_event)

        self.orders: asyncio.Queue = asyncio.Queue()
        self.inflight = 0
        self.stopping = False
        self.started_at = time.time()
        self.order_count = 0
        self.served: deque[str] = deque(maxlen=100)
        self._enc_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._enc_lock = threading.Lock()
        self._bg_tasks: set[asyncio.Task] = set()
        self._tasks: list[asyncio.Task] = []
        self.researcher = Researcher(self)
        self.agents = [Orchestrator(self), Architect(self), Trainer(self), Evaluator(self),
                       Curator(self), Explorer(self), self.researcher, Sentinel(self)]

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> "System":
        self.started_at = time.time()
        self.guardian.check_kill_switch()
        await self.teachers.refresh(force=True)  # find local model servers before the first order
        explorer = next(a for a in self.agents if isinstance(a, Explorer))
        try:  # vet and activate installed tools before the first order is taken
            await explorer.step()
        except Exception:
            log.exception("initial tool scan failed")
        for agent in self.agents:
            self._tasks.append(asyncio.create_task(agent.run(), name=f"agent:{agent.name}"))
        self.guardian.audit("system", "system.start", self.directives.name,
                            prime=[p.id for p in self.directives.prime], compute=self.backend.describe())
        self.bus.publish("system.start", name=self.directives.name, compute=self.backend.describe())
        return self

    async def stop(self) -> None:
        self.stopping = True
        for task in self._tasks + list(self._bg_tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, *self._bg_tasks, return_exceptions=True)
        for node in self.registry.live():
            self.registry.persist(node, weights=True)
        self.affect.save()
        self.guardian.audit("system", "system.stop")
        self.store.close()

    async def __aenter__(self) -> "System":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def background_slot(self) -> None:
        while self.inflight > 0 or not self.guardian.background_allowed():
            await asyncio.sleep(0.02)

    # ---------------------------------------------------------------- orders
    async def submit(self, text: str, source: str = "user") -> Result:
        order = Order(text=text, source=source)
        order.future = asyncio.get_running_loop().create_future()
        await self.orders.put(order)
        return await order.future

    def record_order(self, result: Result) -> None:
        self.order_count += 1
        self.served.append(result.served_by)
        self.store.save_order(result.order_id, result.text, result.to_dict())
        self.bus.publish("order.done", order=result.order_id, route=result.route, ok=result.ok,
                         served_by=result.served_by, latency_ms=result.latency_ms,
                         text=result.text[:200], output=str(result.output)[:200], note=result.note)

    # -------------------------------------------------------------- encoding
    def encode(self, text: str) -> np.ndarray:
        with self._enc_lock:
            vec = self._enc_cache.get(text)
            if vec is not None:
                self._enc_cache.move_to_end(text)
                return vec
        vec = self.encoder.encode(text)
        with self._enc_lock:
            self._enc_cache[text] = vec
            if len(self._enc_cache) > 50_000:
                self._enc_cache.popitem(last=False)
        return vec

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.encoder.dim), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts])

    # --------------------------------------------------------------- cascade
    async def infer(self, cap: str, text: str, *, allow_teacher: bool = True) -> Inference:
        """Confidence-gated cascade: the capability's champion node answers when it is
        validated and confident; otherwise a teacher answers and the answer becomes
        training data, so the node gradually takes over."""
        spec = self.capabilities[cap]
        exclude = frozenset(spec.inactive_labels)
        node = self.registry.champion(cap)
        guess = None
        if node is not None:
            label, conf = node.predict(self.encode(text), exclude)
            node.last_used = time.time()
            node.metrics["serves"] = node.metrics.get("serves", 0) + 1
            guess = (label, conf)
            acc = node.metrics.get("holdout_acc")
            if (acc is not None and acc >= self.directives.nodes.serve_min_accuracy
                    and conf >= self.directives.orders.confidence_threshold):
                if allow_teacher and self.rng.random() < self.directives.orders.audit_sample_rate:
                    self._spot_check(cap, text, label, node)
                return Inference(label, conf, f"node:{node.id}", node.id, "node")
        if allow_teacher:
            lesson = await self.teachers.label(spec, text)
            if lesson and lesson.label and lesson.label not in exclude and self.learn(cap, text, lesson.label, lesson.source):
                if node is not None and guess is not None:
                    node.metrics["live_total"] = node.metrics.get("live_total", 0) + 1
                    node.metrics["live_agree"] = node.metrics.get("live_agree", 0) + int(guess[0] == lesson.label)
                return Inference(lesson.label, lesson.confidence, f"teacher:{lesson.source}", None, "teacher")
            self.teachers.human.ask(cap, text, "no teacher could label this")
        if guess is not None:
            return Inference(guess[0], guess[1], f"node:{node.id} (low confidence)", node.id, "node-lowconf")
        return Inference(None, 0.0, "none", None, "none")

    def _spot_check(self, cap: str, text: str, label: str, node) -> None:
        async def check() -> None:
            lesson = await self.teachers.label(self.capabilities[cap], text)
            if not lesson or not lesson.label:
                return
            self.learn(cap, text, lesson.label, lesson.source)
            node.metrics["live_total"] = node.metrics.get("live_total", 0) + 1
            node.metrics["live_agree"] = node.metrics.get("live_agree", 0) + int(lesson.label == label)
            if lesson.label != label:
                self.bus.publish("node.disagreement", node=node.id, capability=cap, node_label=label,
                                 teacher_label=lesson.label, teacher=lesson.source)
        task = asyncio.create_task(check())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ------------------------------------------------------------- learning
    def learn(self, cap: str, text: str, label: str, source: str, weight: float | None = None) -> bool:
        spec = self.capabilities.get(cap)
        if spec is None or not text.strip():
            return False
        if label not in spec.labels:
            if not spec.open_labels:  # closed label sets only grow through define_capability
                return False
            self.ensure_label(cap, label, by=source)
        elif label in spec.inactive_labels:
            return False
        w = weight if weight is not None else self.teachers.weight(source)
        self.store.add_example(cap, text, label, source, w)
        return True

    def ensure_label(self, cap: str, label: str, description: str | None = None, by: str = "system") -> None:
        spec = self.capabilities[cap]
        if spec.add_label(label, description):
            self.save_capability(spec)
            self.guardian.audit(by, "capability.label_added", cap, label=label)
            self.bus.publish("capability.labels_changed", capability=cap, label=label, by=by)

    def teach(self, cap: str, text: str, label: str, reason: str = "human teaching") -> None:
        """A human-provided example: highest trust, used for training and evaluation."""
        spec = self.capabilities.get(cap)
        if spec is None:
            raise KeyError(f"unknown capability '{cap}'")
        if label not in spec.labels and not spec.open_labels:
            raise ValueError(f"'{label}' is not a {cap} label {spec.active_labels()}; "
                             f"add it with /define {cap} {label} if it should be")
        self.teachers.human.provide(cap, text, label)
        self.learn(cap, text, label, "human")
        self.guardian.audit("human", "teach", cap, text=text[:200], label=label)
        self.bus.publish("teacher.lesson", capability=cap, label=label, source="human", examples=1,
                         reason=reason)

    def feedback(self, order_id: str, correct: str) -> tuple[bool, str]:
        """Correct one step of an order: its route (tool:x / skill:x / answer:direct), the label
        a skill gave, or - qualified as affect=<tone> / disposition=<action> - how the emotional
        engines read the message and what they decided. Unqualified labels go to the order's
        skill step first, then to the engines."""
        record = self.store.get_order(order_id)
        if record is None:
            return False, f"unknown order {order_id}"
        steps = [st for st in record["result"].get("steps", []) if st.get("capability") in self.capabilities]
        correct = correct.strip()
        if correct.startswith(("tool:", "skill:")) or correct == DIRECT:
            if correct not in self.dispatch.labels:
                return False, f"unknown route '{correct}'"
            step = next((st for st in steps if st["capability"] == DISPATCH), None)
            cap, text = DISPATCH, (step["input"] if step else record["text"])
        else:
            named, eq, label = correct.partition("=")
            if eq:
                step = next((st for st in reversed(steps) if st["capability"] == named.strip()), None)
                if step is None:
                    return False, f"that order had no {named.strip()} step"
                correct = label.strip()
            else:
                skills = [st for st in reversed(steps) if st["capability"] != DISPATCH
                          and not self.capabilities[st["capability"]].internal]
                engines = [st for st in reversed(steps) if self.capabilities[st["capability"]].internal]
                fits = lambda st: correct in self.capabilities[st["capability"]].labels  # noqa: E731
                step = (next((st for st in skills if fits(st)), None) or next((st for st in engines if fits(st)), None)
                        or next((st for st in skills if self.capabilities[st["capability"]].open_labels), None)
                        or (skills[0] if skills else None))
                if step is None:
                    hint = " (or teach the emotional engines: affect=<tone>, disposition=<action>)" if engines else ""
                    return False, ("that order had no skill step; correct its route instead (e.g. tool:calculator)"
                                   + hint)
            cap, text = step["capability"], step["input"]
            spec = self.capabilities[cap]
            if correct not in spec.labels and not spec.open_labels:
                return False, (f"'{correct}' is not a {cap} label {spec.active_labels()}. If the order went "
                               f"to the wrong place, correct the route instead (e.g. skill:<name> or tool:<name>)")
        self.teach(cap, text, correct, reason="correction")
        self.store.set_feedback(order_id, f"{cap}={correct}")
        return True, f"learned: {cap}('{text[:60]}') = {correct}"

    def answer_question(self, qid: str, label: str) -> tuple[bool, str]:
        q = self.teachers.human.question(qid)
        if q is None:
            return False, f"no open question {qid}"
        self.teach(q["capability"], q["text"], label)
        return True, f"learned: {q['capability']}('{q['text'][:60]}') = {label}"

    # --------------------------------------------------------- capabilities
    @property
    def dispatch(self) -> CapabilitySpec:
        return self.capabilities[DISPATCH]

    def save_capability(self, spec: CapabilitySpec) -> None:
        self.store.save_capability(spec.name, spec.to_dict())

    def _load_capabilities(self, path: Path) -> None:
        caps = {n: CapabilitySpec.from_dict(d) for n, d in self.store.load_capabilities().items()}
        seeds = load_toml(path) if path.exists() else {}
        for name, data in seeds.items():
            seed = CapabilitySpec.from_dict({"name": name, **data})
            if name in caps:  # keep learned labels, refresh offline-teacher material
                spec = caps[name]
                for lab in seed.labels:
                    spec.add_label(lab)
                spec.description = seed.description or spec.description
                for f in OFFLINE_FIELDS:
                    setattr(spec, f, getattr(seed, f))
            else:
                caps[name] = seed
        if DISPATCH not in caps:
            caps[DISPATCH] = CapabilitySpec(
                DISPATCH, description="Route a user order to the tool or skill that can fulfil it.",
                labels=[DIRECT], open_labels=False, default_label=DIRECT, kind="dispatch",
                label_descriptions={DIRECT: "No tool or skill fits; answer the request directly."})
        self.capabilities = caps
        for name, spec in caps.items():
            if name != DISPATCH and not spec.internal:  # the emotional engines are not order routes
                self._register_skill_route(spec)
        for spec in caps.values():
            self.save_capability(spec)

    def _register_skill_route(self, spec: CapabilitySpec) -> None:
        route = f"skill:{spec.name}"
        self.dispatch.add_label(route, spec.description)
        self.dispatch.lexicon[route] = list(spec.triggers or [spec.name])

    def define_capability(self, name: str, labels: list[str], description: str = "",
                          triggers: list[str] | None = None) -> CapabilitySpec:
        name = name.strip().lower()
        if name == DISPATCH or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("capability names are lowercase identifiers and 'dispatch' is reserved")
        if name in self.capabilities and self.capabilities[name].internal:
            raise ValueError(f"'{name}' is one of the mesh's internal engines; teach it with /teach or /feedback")
        spec = self.capabilities.get(name) or CapabilitySpec(name)
        spec.description = description or spec.description
        for lab in labels:
            spec.add_label(lab.strip())
        if triggers:
            spec.triggers = list(triggers)
        self.capabilities[name] = spec
        self.save_capability(spec)
        self._register_skill_route(spec)
        self.save_capability(self.dispatch)
        self.guardian.audit("human", "capability.defined", name, labels=spec.labels)
        self.bus.publish("capability.new", capability=name, labels=list(spec.labels))
        return spec

    # ----------------------------------------------------------------- tools
    def activate_tool(self, tool: Tool, reason: str) -> None:
        m = tool.manifest
        d = self.dispatch
        d.add_label(m.route, m.description)
        d.lexicon[m.route] = list(m.keywords)
        if m.templates:
            d.label_templates[m.route] = list(m.templates)
        for key, values in m.fillers.items():
            d.fillers[f"{m.route}.{key}"] = list(values)
        tool.state, tool.note = ACTIVE, reason
        tool.activated_at = time.time()
        tool.consecutive_failures = 0
        seeded = sum(1 for ex in m.examples if self.learn(DISPATCH, ex["input"], m.route, "tool-manifest"))
        if m.supersedes and m.supersedes in self.tools.tools and m.supersedes != m.name:
            old = self.tools.tools[m.supersedes]
            moved = self.store.relabel(DISPATCH, old.manifest.route, m.route)
            self.retire_tool(old, f"superseded by {m.name} ({moved} routing examples relabelled)")
        self.save_capability(d)
        self.tools.persist(tool)
        self.guardian.audit("explorer", "tool.active", m.name, reason=reason, seeded_examples=seeded)
        self.bus.publish("tool.active", tool=m.name, version=m.version, reason=reason, seeded=seeded)

    def retire_tool(self, tool: Tool, reason: str) -> None:
        tool.state, tool.note = RETIRED, reason
        self.dispatch.deactivate(tool.manifest.route)
        self.save_capability(self.dispatch)
        self.tools.persist(tool)
        self.guardian.audit("explorer", "tool.retired", tool.name, reason=reason)
        self.bus.publish("tool.retired", tool=tool.name, reason=reason)

    def retest_tool(self, name: str) -> tuple[bool, str]:
        tool = self.tools.tools.get(name)
        if tool is None or tool.state not in (QUARANTINED, ACTIVE):
            return False, "only quarantined or active tools can be re-tested"
        tool.state, tool.note = APPROVED, "re-test requested by operator"
        self.tools.persist(tool)
        return True, f"{name} queued for self-tests"

    def tool_config(self, tool: Tool) -> dict:
        """manifest [config], overridden by directives [tools.config.<name>]."""
        return {**tool.manifest.config, **self.directives.tools.config.get(tool.name, {})}

    def record_tool_run(self, tool: Tool, run: ToolRun) -> None:
        tool.runs += 1
        if not run.ok and run.transient:  # e.g. offline: not evidence that the tool is broken
            tool.last_error = run.error[:300]
            self.tools.persist(tool)
            return
        if run.ok:
            tool.successes += 1
            tool.consecutive_failures = 0
        else:
            tool.consecutive_failures += 1
            tool.last_error = run.error[:300]
            if tool.state == ACTIVE and tool.consecutive_failures >= self.directives.tools.quarantine_after_failures:
                tool.state = QUARANTINED
                tool.note = f"{tool.consecutive_failures} consecutive failures"
                self.dispatch.deactivate(tool.manifest.route)
                self.save_capability(self.dispatch)
                self.guardian.audit("explorer", "tool.quarantined", tool.name, error=tool.last_error)
                self.bus.publish("tool.quarantined", tool=tool.name, reason=tool.note)
        self.tools.persist(tool)

    # -------------------------------------------------------------- research
    def knowledge_tool(self) -> Tool | None:
        """The active tool that provides knowledge (wikipedia), if research is enabled."""
        if not self.directives.research.enabled:
            return None
        return next((t for t in self.tools.active() if t.manifest.provides == "knowledge"), None)

    async def research(self, topic: str, *, reason: str, background: bool = False,
                       tool: Tool | None = None):
        """Look something up with the research tool and retain what comes back.
        Returns (Document, "learned") or (None, why not)."""
        tool = tool or self.knowledge_tool()
        if tool is None:
            return None, "no active research tool (approve the wikipedia tool: /approvals)"
        decision = self.guardian.authorize_research(background)
        if not decision.allowed:
            return None, decision.reason
        actor = "researcher" if background else "orchestrator"
        with self.activity.track("research", actor, f"{tool.name}: {' '.join(topic.split())[:80]}"):
            run = await self.sandbox.run(tool, topic, tool.manifest.timeout_s or self.directives.tools.default_timeout_s,
                                         self.tool_config(tool))
        self.guardian.record_research(background)
        self.record_tool_run(tool, run)
        if not run.ok:
            self.bus.publish("knowledge.failed", tool=tool.name, topic=topic[:200], error=run.error,
                             background=background)
            return None, run.error
        k = run.extra.get("knowledge")
        if not isinstance(k, dict) or not k.get("title"):
            return None, f"{tool.name} returned no knowledge"
        doc = self.knowledge.add(k, reason)
        self.guardian.audit(actor, "knowledge.learned", doc.title, source=doc.source, url=doc.url,
                            passages=doc.passages, reason=reason)
        self.bus.publish("knowledge.learned", title=doc.title, url=doc.url, source=doc.source,
                         passages=doc.passages, reason=reason, background=background, topic=topic[:200])
        pol = self.directives.research
        if not background and pol.max_depth >= 1:
            for link in doc.links[: pol.follow_links]:
                self.researcher.enqueue(link, f"linked from '{doc.title}'", 1)
        return doc, "learned"

    async def answer_from_knowledge(self, question: str, hits: list[Hit]) -> tuple[str, str]:
        """An answer from retained passages: written by an LLM teacher when one is available
        (told to use only the passages), otherwise the best matching sentences."""
        doc = hits[0].doc
        composed = await self.teachers.compose(question, hits)
        if composed:
            body, how = composed[0], f"written by the {composed[1]} teacher from retained passages"
        else:
            body, how = extract_answer(question, hits), "quoted from retained passages"
        return f"{body}\n(source: {doc.source} '{doc.title}' {doc.url})", how

    # ---------------------------------------------------------------- status
    def scorecard(self) -> list[dict]:
        npol = self.directives.nodes
        accs = [n.metrics["holdout_acc"] for n in (self.registry.champion(c) for c in self.capabilities)
                if n is not None and n.metrics.get("holdout_acc") is not None
                and n.metrics.get("holdout_n", 0) >= npol.min_eval_samples]
        decided = [s for s in self.served if s != "explicit"]
        usable = [t for t in self.tools.tools.values() if t.state in (ACTIVE, APPROVED)]
        metrics = {
            "min_capability_accuracy": round(min(accs), 3) if accs else None,
            "node_coverage": round(sum(s == "nodes" for s in decided) / len(decided), 3) if decided else None,
            "tool_integration": round(sum(t.state == ACTIVE for t in usable) / len(usable), 3) if usable else 1.0,
            "live_nodes": len(self.registry.live()),
        }
        return [{"id": g.id, "text": g.text, "metric": g.metric, "target": g.target,
                 "direction": g.direction, "value": metrics.get(g.metric), "met": g.met(metrics.get(g.metric))}
                for g in self.directives.goals]

    @staticmethod
    def _brief(node) -> dict:
        return {"id": node.id, "status": node.status, "origin": node.origin, "generation": node.generation,
                "arch": node.arch.describe(), "labels": len(node.labels), "train_steps": node.train_steps,
                "holdout_acc": node.metrics.get("holdout_acc"), "serves": node.metrics.get("serves", 0)}

    def status(self) -> dict:
        caps = {}
        for name, spec in self.capabilities.items():
            champ = self.registry.champion(name)
            caps[name] = {
                "labels": spec.active_labels(), "examples": self.store.count(name),
                "champion": self._brief(champ) if champ else None,
                "candidates": [self._brief(n) for n in self.registry.candidates(name)],
                "retired": [n.id for n in self.registry.retired(name)],
            }
        lineage = self.registry.lineage.values()
        return {
            "name": self.directives.name, "uptime_s": round(time.time() - self.started_at, 1),
            "compute": self.backend.describe(),
            "halted": self.guardian.halted, "halt_reason": self.guardian.halt_reason,
            "background_paused": self.guardian.paused, "inflight": self.inflight,
            "background_duty_cycle": round(self.guardian.duty_cycle(), 3), "orders": self.order_count,
            "served_recent": dict(Counter(self.served)),
            "nodes": {"live": len(self.registry.live()), "ever": len(self.registry.lineage),
                      "disposed": sum(1 for m in lineage if m["status"] == "disposed")},
            "capabilities": caps,
            "tools": {n: {"state": t.state, "version": t.manifest.version, "runs": t.runs,
                          "permissions": list(t.manifest.permissions), "note": t.note}
                      for n, t in sorted(self.tools.tools.items())},
            "approvals": [a.to_dict() for a in self.guardian.pending_approvals()],
            "questions": list(self.teachers.human.questions),
            "teachers": self.teachers.describe(), "llm_usage": self.guardian.llm_usage(),
            "knowledge": {**vars(self.knowledge.stats()), "research": self.guardian.research_usage(),
                          "queued": len(self.researcher.queue),
                          "tool": (self.knowledge_tool().name if self.knowledge_tool() else None)},
            "temperament": self.affect.status(),
            "scorecard": self.scorecard(),
            "agents": {a.name: dict(a.stats) for a in self.agents},
        }
