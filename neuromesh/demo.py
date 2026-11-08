"""Offline, scripted demonstration of the full life cycle.

Runs in a throw-away copy of the project with time constants shortened so a few minutes
of behaviour fit into well under a minute:

 1. cold start      - teachers answer orders; the architect seeds nodes for each capability
 2. warm-up         - nodes take over orders; weak champions face challengers
 3. new tools       - plugins appear; one is refused, one waits for human approval,
                      two are auto-approved, self-tested and learned; calculator2 supersedes
                      calculator; the routing node is expanded, the old one disposed
 4. steady state    - human feedback, background improvement while idle
 5. memory          - facts are compressed into memory nodes with human-readable test files, then
                      into ever smaller nodes that still pass every test; compressed nodes are merged
                      into one; an operator edits a test
 6. initiative      - with no orders, the director decides on its own what to test and learn:
                      it sits benchmark exams, practises, and sets itself tasks to improve skills
 7. sleep           - it falls asleep and dreams up questions and answers from what it knows;
                      on waking it checks them and evolves its dream nodes; then it imagines
 8. directors       - a director takes every order and reviews every result before the person sees
                      it or memory keeps it; it duplicates itself for second opinions and when work
                      waits; its mood shapes how carefully it works. The calculator solves
                      equations and calculus, and asks a director for what it does not know; an
                      order a director deems unanswered is tried other ways, up to max_attempts
 9. kill switch     - the operator halts everything with a STOP file
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
    "memory": {"interval_seconds": 0.05, "sync_seconds": 0.25, "facts_per_node": 4, "merge_every_seconds": 0.5},
    "agency": {"enabled": False},  # switched on for the initiative phase, so the scored phases compare
    "dreams": {"enabled": False},  # switched on for the sleep phase
}

FACTS = [("What is the capital of Australia?", "Canberra is the capital of Australia."),
         ("How tall is Mount Everest?", "Mount Everest is 8,849 metres tall."),
         ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice, published in 1813."),
         ("What is the speed of light?", "Light travels at about 299,792 kilometres per second in a vacuum."),
         ("When did the Berlin Wall fall?", "The Berlin Wall fell on 9 November 1989."),
         ("What is the largest planet?", "Jupiter is the largest planet in the Solar System."),
         ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
         ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
         ("How many bones are in the human body?", "An adult human body has 206 bones."),
         ("What is the chemical symbol for gold?", "Au is the chemical symbol for gold.")]

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
        self.quiet = {"director.reviewed": 0, "director.consulted": 0}  # every order has them: summarised per phase

    def on_event(self, ev) -> None:
        if ev.topic in self.lifecycle:
            self.lifecycle[ev.topic] += 1
        if ev.topic in self.quiet:
            self.quiet[ev.topic] += 1
            return
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
        quiet_before = dict(self.quiet)
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
              f"{self.lessons - lessons_before} background lessons during the phase; directors reviewed "
              f"{self.quiet['director.reviewed'] - quiet_before['director.reviewed']}, with "
              f"{self.quiet['director.consulted'] - quiet_before['director.consulted']} second opinions")
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
    shutil.copytree(PROJECT / "challenges", work / "challenges")  # sample benchmark suites, for the initiative phase
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

        demo.say("Phase 6 - memory: what it learns is compressed into memory nodes, each with a test file")
        mem = system.memory
        for q, a in FACTS:
            mem.learn(q, a, "the demo operator", by="demo-operator", kind="people")
        settled = lambda: (mem.fact_nodes() and all(not n.dirty and n.status == "compressed" for n in mem.fact_nodes())  # noqa: E731
                           and not mem.merge_candidates())
        await demo.wait_for(settled, 40, "the memory nodes to compress and merge")
        node = max(mem.fact_nodes(), key=lambda n: len(n.facts()))
        made = f" (merged from {len(node.parts)} nodes)" if len(node.parts) > 1 else ""
        print(f"  {node.id}{made}: {len(node.facts())} facts, first learned in {node.first_bytes:,} bytes, now "
              f"{node.bytes:,} bytes ({node.spec()}); {node.passed}/{node.total} tests pass")
        for q in ("what's the capital of Australia?", "how tall is mount everest", "What is the capital of Austria?"):
            res = await system.submit(q)
            print(f"  > {q}\n      " + res.summary().replace("\n", "\n      "))
        print(f"  its tests, in the memory database {mem.db_path.relative_to(work)} (/memory tests {node.id}; excerpt):")
        text = mem.render_tests(node)
        for line in text.splitlines()[1:6] + text.split("\n[[test]]\n")[1].strip().splitlines()[:2]:
            print(f"    {line}")
        print("  the operator edits the tests with an SQLite tool: Mount Everest's height changes to 8,848.86 metres")
        import sqlite3
        with sqlite3.connect(mem.db_path) as db:
            db.execute("UPDATE tests SET answer = ? WHERE answer = ?",
                       ("Mount Everest is 8,848.86 metres tall.", "Mount Everest is 8,849 metres tall."))
        edited = "Mount Everest is 8,848.86 metres tall."
        await demo.wait_for(lambda: edited in node.answers and settled(), 30, "the memory node to relearn the edit")
        res = await system.submit("How tall is Mount Everest?")
        print("  > How tall is Mount Everest?\n      " + res.summary().replace("\n", "\n      "))

        demo.say("Phase 7 - initiative: no orders; the director decides for itself what to test and learn")
        a, t = system.directives.agency, system.directives.tasks
        a.decide_every_seconds, a.idle_seconds, a.improve_window_seconds, a.explore, a.practice_items = 1.5, 0.5, 6, 0.0, 10
        a.max_decisions_per_hour = 100
        t.min_interval_seconds, t.improve_interval_seconds, t.challenge_interval_seconds = 1, 2, 3
        system.challenges.pol.practice_batch = 10
        print(f"  the operator turns its initiative on: {system.director.set_enabled(True, 'demo-operator')}")
        await asyncio.sleep(24)
        system.director.set_enabled(False, "demo-operator")
        from .commands import director_lines
        for line in director_lines(system)[1:3]:
            print(f"  {line.strip()}")

        demo.say("Phase 8 - sleep: it dreams about what it knows, checks its dreams on waking, then imagines")
        dp = system.directives.dreams
        dp.enabled, dp.sleep_after_idle_seconds, dp.check_every_seconds, dp.sleep_chance = True, 0.5, 0.3, 1.0
        dp.objective_chance = 0.0  # the idle coin always lands on sleep here
        dp.sleep_max_seconds, dp.dreams_per_night, dp.ga_epochs = 5, 400, 40
        evolved = lambda: any(e.topic == "dream.evolved" for e in system.bus.recent("dream.*", 300))  # noqa: E731
        await demo.wait_for(evolved, 40, "a night's dreams to be checked")
        dp.sleep_chance = 0.0
        dp.objective_chance = 0.75
        system.dreams.wake("the demo moves on")
        for prompt in ("imagine the capital of Australia", "what if Jane Austen painted the Mona Lisa",
                       "imagine a mountain on Jupiter"):
            res = await system.submit(prompt)
            print(f"  > {prompt}\n      " + res.summary().replace("\n", "\n      "))

        demo.say("Phase 9 - directors: one takes every order, one reviews every result; the mood shapes them")
        pool = system.directors
        for line in pool.lines()[:3]:
            print(f"  {line.strip()}")
        made = pool.counts.get("spawned", 0)
        print(f"  up to {pool.most} directors ran at once; {made} duplicate{'' if made == 1 else 's'} made so far")
        print("  the demo stirs up its mood: trust 0.20, patience 0.15, arousal 0.85 (as a run of hostile messages would)")
        system.affect.state.values.update(trust=0.2, patience=0.15, arousal=0.85)
        system.affect.state.updated = time.time()
        st = pool.stance()
        print(f"  stance {st.line()}")
        ok, why = pool.review_memory("Who first climbed Mount Everest?", "Edmund Hillary and Tenzing Norgay, in 1953.",
                                     "the ollama teacher (a language model)", "ollama", "teachers", confidence=0.9)
        print(f"  a language model's answer offered to long-term memory now: {'kept' if ok else 'not kept'} - {why}")
        ok, why = pool.review_memory("Who first climbed Mount Everest?", "Edmund Hillary and Tenzing Norgay, in 1953.",
                                     "the demo operator", "demo-operator", "people")
        print(f"  the same fact from a person: {'kept' if ok else 'not kept'} - {why}")
        before = pool.counts.get("consult", 0)
        burst = demo.f.batch(24, NEW_MIX)
        results = await asyncio.gather(*(system.submit(text) for text, _, _ in burst))
        print(f"  24 orders at once, each taken and reviewed by a director ({', '.join(sorted({r.director for r in results}))}); "
              f"{pool.counts.get('consult', 0) - before} needed a second opinion at the wary bar; "
              f"{len(pool.nodes)} directors running, up to {pool.most} at once")
        system.affect.calm(by="demo-operator")
        print(f"  the operator calms it: stance {pool.stance().line()}")
        print(f"  maths, tools that ask a director, and orders tried other ways (up to {pool.pol.max_attempts}):")
        for q in ("solve x^2 - 5x + 6 = 0", "what is the derivative of x^3 * sin(x)", "integrate x^2 from 0 to 3",
                  "what is the speed of light times 3", "what is the airspeed velocity of an unladen swallow?"):
            res = await system.submit(q)
            print(f"  > {q}\n      " + res.summary().replace("\n", "\n      "))
            for t in res.trace:
                if " asked director" in t or "attempt " in t or "gave up" in t or "router learns" in t:
                    print(f"      . {t[:200]}")

        demo.say("Phase 10 - code: it writes tools for requests nothing met, asks first, then tries them out")
        smith, director = system.toolsmith, system.director
        asks = ("reverse the word lantern", "what is the acronym for portable network graphics",
                "abbreviate frequently asked questions")
        for q in asks:
            res = await system.submit(q)
            print(f"  > {q}\n      " + res.summary().replace("\n", "\n      "))
        needs = smith.needs()
        print("  requests its coder node says a tool could meet: "
              + ("; ".join(f"{g['recipe']} x{len(g['texts'])} ({g['how']})" for g in needs) or "none"))
        director.set_mode("build", "demo-operator")
        for _ in needs:
            rec = await director.decide(force=True)
            print(f"  initiative ({rec['by']}): {rec['mode']} - {rec['summary']}")
        made = [a for a in system.guardian.pending_approvals() if a.kind == "tool.made"]
        for a in made:
            d = smith.drafts[a.subject]
            code = (Path(d["folder"]) / "tool.py").read_text().splitlines()
            print(f"  {a.id}: {a.subject} v{d['version']} from the '{d['recipe']}' recipe, {len(code)} lines, "
                  f"{len(d['tests'])} self-tests; nothing it wrote has run yet")
        print("  the operator reads the code and approves: " + ", ".join(a.id for a in made))
        for a in made:
            system.guardian.decide(a.id, True, by="demo-operator")
        await demo.wait_for(lambda: all(system.tools.tools.get(a.subject) is not None
                                        and system.tools.tools[a.subject].state == "active" for a in made), 30,
                            "the new tools to pass their self-tests")
        director.set_mode("experiment", "demo-operator")
        for _ in made:
            rec = await director.decide(force=True)
            print(f"  initiative ({rec['by']}): {rec['mode']} - {rec['summary']}")
        director.set_mode("auto", "demo-operator")
        for q in asks + ("abbreviate very important person", "write stressed backwards"):
            res = await system.submit(q)
            print(f"  > {q}  ->  {res.output if res.ok else res.note}  [{res.route}]")

        demo.say("Phase 11 - objectives: long-term aims it plans, pursues by asking itself, thinks over and writes up")
        book = system.objectives
        book.pol.step_seconds, book.pol.second_thought_after_seconds = 0.2, 0
        for text in ("objective: find out who wrote Pride and Prejudice and when it was published",
                     "objective: understand the Berlin Wall"):
            res = await system.submit(text)
            print(f"  > {text}\n      {res.output}")
        first, second = book.get("obj-1"), book.get("obj-2")
        replies = {"published": "Pride and Prejudice was first published in 1813.",
                   "berlin wall": "It was the guarded concrete barrier that divided Berlin from 1961 to 1989."}
        deadline = time.time() + 60
        while first is not None and second is not None and time.time() < deadline \
                and (first.status == "active" or second.status == "active"):
            for obj in (first, second):  # the operator answers what it asks (P1: a person's word stands)
                for q in [q for q in obj.questions if q["status"] == "waiting"]:
                    answer = next((a for k, a in replies.items() if k in q["ask"].lower()), "skip")
                    print(f"  {obj.id} asks the operator ({q['inquiry']}): {q['ask']}\n  the operator replies: {answer}")
                    await system.mind.reply(q["inquiry"], answer, by="demo-operator")
            await asyncio.sleep(0.2)
        for obj in (first, second):
            if obj is None:
                continue
            shown = [t for t in obj.thoughts if t[1] in ("doubt", "confirm", "contradiction")][:4]
            for t in shown:
                print(f"  {obj.id} thought ({t[1]}): {t[2][:170]}")
            print(f"  {obj.id}: {obj.note}")
        if second is not None:
            print(f"  the write-up of {second.id}, {book.report_path(second).relative_to(work)} (excerpt):")
            for line in book.report(second).splitlines()[:18]:
                if line.strip():
                    print(f"    {line[:150]}")
        res = await system.submit("What is the capital of Freedonia?")
        print(f"  > What is the capital of Freedonia?\n      {res.summary()}")
        system.memory.learn("What is the capital of Freedonia?", "Fredville is the capital of Freedonia.", "the demo operator",
                            by="demo-operator", kind="people")
        await demo.wait_for(lambda: all(not n.dirty for n in system.memory.fact_nodes()), 30, "memory to learn it")
        for t in await book.second_thoughts(by="demo-operator"):
            print(f"  second thoughts ({t['verdict']}): {t['line'][:180]}")

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
