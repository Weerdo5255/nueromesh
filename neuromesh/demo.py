"""Offline, scripted demonstration of the full life cycle.

Runs in a throw-away copy of the project with time constants shortened so a few minutes
of behaviour fit into well under a minute:

 1. cold start      - teachers answer orders; the architect seeds nodes for each capability
 2. warm-up         - nodes take over orders; weak champions face challengers
 3. new tools       - plugins appear; one is refused, one waits for human approval,
                      two are auto-approved, self-tested and learned; calculator2 supersedes
                      calculator; the routing node is expanded, the old one disposed
 4. steady state    - human feedback, background improvement while idle
 5. kill switch     - the operator halts everything with a STOP file
"""
from __future__ import annotations

import asyncio
import random
import shutil
import tempfile
import time
from pathlib import Path

from .config import load_toml
from .report import describe_event, render_lineage, render_status
from .runtime import System
from .tasks import CapabilitySpec
from .teachers import HeuristicTeacher, fill_template
from .tools import ToolManifest

PROJECT = Path(__file__).resolve().parent.parent

DEMO_OVERRIDES = {
    "orders": {"audit_sample_rate": 0.15},
    "nodes": {"min_examples_to_spawn": 10, "challenger_cooldown_seconds": 2.0, "retired_grace_seconds": 3.0,
              "candidate_max_age_seconds": 20.0, "max_candidate_slices": 8, "min_eval_samples": 8,
              "promotion_z": 1.0, "target_accuracy": 0.97},
    "background": {"tick_seconds": 0.1, "max_duty_cycle": 0.9, "synthesis_cooldown_seconds": 0.6,
                   "synthesis_batch": 8, "health_check_seconds": 3600},
    "tools": {"scan_seconds": 0.5},
    "temperament": {"enabled": False},  # scored on raw answers; the temperament has its own tests
}

IMPLICIT = {"sentiment": ["how does this review sound: ", "what's the tone of this review: ", "review check: "],
            "urgency": ["how urgent is this: ", "triage this: ", "what priority is this: "]}


class OrderFactory:
    """Generates orders with known ground truth so the demo can score the system."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.heuristic = HeuristicTeacher(seed + 1)
        caps = load_toml(PROJECT / "capabilities.toml")
        self.caps = {n: CapabilitySpec.from_dict({"name": n, **d}) for n, d in caps.items()}
        folders = list((PROJECT / "plugins").iterdir()) + list((PROJECT / "examples" / "incoming_plugins").iterdir())
        self.tools = {m.name: m for m in (ToolManifest.load(f) for f in folders if (f / "manifest.toml").exists())}

    def tool(self, name: str) -> tuple[str, str, str | None]:
        m = self.tools[name]
        return fill_template(self.rng.choice(m.templates), m.fillers, self.rng), f"tool:{name}", None

    def skill(self, cap: str, implicit: bool) -> tuple[str, str, str | None]:
        spec = self.caps[cap]
        label = self.rng.choice(spec.labels)
        text = self.heuristic.generate_sync(spec, label, 1)[0]
        prefix = self.rng.choice(IMPLICIT[cap]) if implicit else f"{cap}: "
        return prefix + text, f"skill:{cap}", label

    def batch(self, n: int, mix: dict[str, float]) -> list[tuple[str, str, str | None]]:
        kinds, weights = zip(*mix.items())
        out = []
        for _ in range(n):
            kind = self.rng.choices(kinds, weights)[0]
            if kind.startswith("tool:"):
                out.append(self.tool(kind[5:]))
            else:
                cap, _, mode = kind.partition("/")
                out.append(self.skill(cap, mode == "implicit"))
        return out


class Demo:
    def __init__(self, system: System, factory: OrderFactory):
        self.s = system
        self.f = factory
        self.t0 = time.time()
        self.rows: list[dict] = []
        self.lifecycle = {"node.spawned": 0, "node.promoted": 0, "node.retired": 0, "node.disposed": 0}
        self.lessons = 0

    def on_event(self, ev) -> None:
        if ev.topic in self.lifecycle:
            self.lifecycle[ev.topic] += 1
        if ev.topic == "teacher.lesson" and ev.data.get("source") != "human":
            self.lessons += 1
            return  # summarised per phase instead of printed one by one
        line = describe_event(ev)
        if line:
            print(f"  t+{ev.ts - self.t0:5.1f}s {line}", flush=True)

    def say(self, text: str) -> None:
        print(f"\n== {text} ==", flush=True)

    async def run_orders(self, phase: str, orders: list, show: int = 3) -> list:
        results = []
        lessons_before = self.lessons
        for i, (text, route, label) in enumerate(orders):
            res = await self.s.submit(text)
            correct = res.ok and res.route == route and (label is None or res.output == label)
            results.append((res, route, label, correct))
            if i < show:
                print(f"  > {text}\n      {res.summary()}" + ("" if correct else f"   (expected {label or route})"))
            await asyncio.sleep(0.03)
        decided = [r for r, *_ in results]
        self.rows.append({
            "phase": phase, "orders": len(results),
            "fulfilled": sum(r.ok for r in decided),
            "correct": sum(c for *_, c in results),
            "nodes": sum(r.served_by == "nodes" for r in decided),
            "teachers": sum(r.served_by == "teachers" for r in decided),
            "mixed": sum(r.served_by == "mixed" for r in decided),
            "latency": sum(r.latency_ms for r in decided) / max(len(decided), 1),
        })
        row = self.rows[-1]
        print(f"  -> {row['orders']} orders: {row['correct']} correct, served by nodes {row['nodes']}, "
              f"teachers {row['teachers']}, mixed {row['mixed']}; "
              f"{self.lessons - lessons_before} background lessons during the phase")
        return results

    async def idle(self, seconds: float, why: str) -> None:
        lessons_before = self.lessons
        print(f"  ... {seconds:.0f}s with no orders ({why})", flush=True)
        await asyncio.sleep(seconds)
        print(f"  ... idle period done: {self.lessons - lessons_before} lessons from teachers, "
              f"{len(self.s.registry.live())} live nodes", flush=True)

    async def wait_for(self, predicate, timeout: float, what: str) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.1)
        print(f"  (timed out waiting for {what})")
        return False


BASE_MIX = {"tool:calculator": 0.22, "tool:word_stats": 0.18, "sentiment/explicit": 0.2,
            "sentiment/implicit": 0.12, "urgency/explicit": 0.16, "urgency/implicit": 0.12}
NEW_MIX = {"tool:calculator2": 0.2, "tool:unit_convert": 0.15, "tool:glossary": 0.1, "tool:word_stats": 0.1,
           "sentiment/explicit": 0.15, "sentiment/implicit": 0.08, "urgency/explicit": 0.12, "urgency/implicit": 0.1}


async def run_demo(use_llm: bool = False, keep: bool = False, seed: int = 7,
                   device: str | None = None, library: str | None = None,
                   ollama: str | None = None, ollama_url: str | None = None) -> None:
    work = Path(tempfile.mkdtemp(prefix="neuromesh-demo-"))
    for name in ("directives.toml", "capabilities.toml"):
        shutil.copy(PROJECT / name, work / name)
    shutil.copytree(PROJECT / "plugins", work / "plugins", ignore=shutil.ignore_patterns("wikipedia", "kiwix"))  # offline
    overrides = dict(DEMO_OVERRIDES)
    if ollama or ollama_url:  # a local model teaches; the heuristic teacher covers any gaps
        local = {"provider": "ollama", "enabled": True, "model": ollama or "auto"}
        if ollama_url:
            local["base_url"] = ollama_url
        overrides["teachers"] = {"order": ["human", "ollama", "heuristic"], "ollama": local}
    elif not use_llm:
        overrides["teachers"] = {"order": ["human", "heuristic"]}
    try:
        system = System(work, overrides=overrides, seed=seed, device=device, library=library)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    demo = Demo(system, OrderFactory(seed))
    system.bus.listen(demo.on_event)
    print(f"NeuroMesh demo - working folder {work}\ncompute: {system.backend.describe()}")

    async with system:
        teachers = [t.name for t in system.teachers.ranked()]
        hint = "" if (use_llm or ollama or ollama_url) else " (offline; --ollama MODEL or --llm to add an LLM teacher)"
        print(f"teachers in use: {teachers}{hint}")
        for name, info in system.teachers.describe().items():
            if info.get("provider") and info.get("in_order") and not info["available"]:
                print(f"  note: {name} is unavailable - {info.get('detail')}; the demo continues without it")
        demo.t0 = time.time()
        demo.say("Phase 1 - cold start: no nodes exist; teachers answer and their answers become training data")
        await demo.run_orders("1 cold start", demo.f.batch(70, BASE_MIX))
        await demo.idle(4, "background agents keep training, evaluating and asking teachers for lessons")

        demo.say("Phase 2 - warm: nodes answer what they are confident about; weak champions face challengers")
        await demo.run_orders("2 warm", demo.f.batch(70, BASE_MIX))
        await demo.idle(3, "background improvement continues")

        demo.say("Phase 3 - four new plugins are dropped into plugins/")
        for name in ("unit_convert", "glossary", "calculator2", "shell_exec"):
            shutil.copytree(PROJECT / "examples" / "incoming_plugins" / name, work / "plugins" / name)
        tools = system.tools.tools
        await demo.wait_for(lambda: all(n in tools for n in ("unit_convert", "glossary", "calculator2", "shell_exec"))
                            and tools["unit_convert"].state == "active" and tools["calculator2"].state == "active",
                            15, "auto-approved tools to activate")
        pending = system.guardian.pending_approvals()
        await asyncio.sleep(1.0)
        for approval in pending:
            print(f"  operator reviews {approval.id} ({approval.subject}, permissions "
                  f"{approval.detail.get('permissions')}) and approves it")
            system.guardian.decide(approval.id, True, by="demo-operator", note="read-only access to its own folder")
        await demo.wait_for(lambda: tools.get("glossary") and tools["glossary"].state == "active", 10,
                            "glossary to activate")
        routes = [f"tool:{n}" for n in ("unit_convert", "glossary", "calculator2")]

        def router_updated():
            champ = system.registry.champion("dispatch")
            return champ is not None and all(r in champ.labels for r in routes)
        print("  ... no orders; the architect grows the routing node and the trainer drills it on the new tools")
        await demo.wait_for(router_updated, 30, "an expanded routing node to win promotion")
        await demo.idle(4, "rollback window for the superseded router, then disposal")

        demo.say("Phase 4 - orders that need the new tools")
        results = await demo.run_orders("4 new tools", demo.f.batch(80, NEW_MIX))

        demo.say("Human feedback - an operator reviews answers")
        wrong = [(r, route, label) for r, route, label, ok in results if not ok and r.route is not None][:3]
        verb = "corrects it to"
        if not wrong:
            print("  no mistakes in phase 4; the operator confirms one answer instead (confirmations teach too)")
            r, route, label, _ = next(x for x in results if x[2])
            wrong, verb = [(r, route, label)], "confirms"
        for r, route, label in wrong:
            correction = route if r.route != route else label  # wrong place -> fix the route
            ok, msg = system.feedback(r.order_id, correction)
            print(f"  {r.order_id} {r.text[:58]!r}\n      system said {r.output!r} via {r.route}; "
                  f"operator {verb} {correction}: {msg}")
        await demo.idle(3, "human examples are weighted 3x and folded into training")

        demo.say("Phase 5 - steady state")
        await demo.run_orders("5 steady", demo.f.batch(60, NEW_MIX))
        await demo.idle(2, "")

        demo.say("Results")
        print(f"{'phase':<14} {'orders':>6} {'correct':>8} {'by nodes':>9} {'by teachers':>12} {'mixed':>6} {'avg ms':>7}")
        for row in demo.rows:
            print(f"{row['phase']:<14} {row['orders']:>6} {row['correct'] / row['orders']:>8.0%} "
                  f"{row['nodes'] / row['orders']:>9.0%} {row['teachers'] / row['orders']:>12.0%} "
                  f"{row['mixed'] / row['orders']:>6.0%} {row['latency']:>7.1f}")
        lc = demo.lifecycle
        print(f"\nnode life cycle: {lc['node.spawned']} built, {lc['node.promoted']} promoted, "
              f"{lc['node.retired']} retired, {lc['node.disposed']} disposed; "
              f"{len(system.registry.live())} live now. {demo.lessons} background lessons taken.")
        print("\n" + render_status(system.status()))
        print("\nnode lineage (every node ever built):")
        print(render_lineage(system.registry.lineage))

        demo.say("Kill switch - the operator creates state/STOP")
        (system.state_dir / "STOP").touch()
        await asyncio.sleep(0.6)
        res = await system.submit("what is 2 + 2")
        print(f"  > what is 2 + 2\n      {res.summary()}")
        counts = system.store.audit_counts()
        print(f"\naudit log: {sum(counts.values())} entries, e.g. "
              + ", ".join(f"{k}={v}" for k, v in counts.most_common(8)))

    if keep:
        print(f"\nkept {work} (state/neuromesh.db holds experience, lineage and the audit log)")
    else:
        shutil.rmtree(work, ignore_errors=True)
