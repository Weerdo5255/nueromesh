"""Operator commands and orders, shared by the plain console and the live view."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .background import every

from .report import render_lineage, render_status

HELP = """Type an order (e.g. "what is 12 * 7", "sentiment: the hotel was lovely") or a command:
  /status                         overview + directive scorecard
  /lineage [capability]           family tree of every node ever built
  /tools                          tools and their states
  /teachers                       teacher status (Ollama, Claude, ...)
  /research <topic>               look it up on Wikipedia now and retain it
  /knowledge [question]           what the mesh has retained, or what it would recall for a question
  /forget <article title>         drop a retained article
  /approvals                      pending approvals      /approve <id> [note]   /deny <id> [note]
  /questions                      questions waiting for a human teacher   /answer <qid> <label>
  /correct <order-id>             was it right? lists the next most likely answers, worked out now
  /correct <order-id> yes | a2 | none [:: the right answer]   it learns from what you say
  /feedback <order-id> <label>    correct an answer: a label, a route like tool:calculator, or the
                                  emotional engines with affect=<tone> / disposition=<action>
  /task learn <topic> [every 2m] [max 30]     keep studying a topic in the background
  /task improve <skill> [to 0.97]             keep training a skill until it reaches the target
  /task every 10m: <order>                    repeat an order; you are told when its result changes
  /tasks    /task show|pause|resume|remove <task-id>     (also: "keep learning about X" as an order)
  /evolution [skill]              how each skill's nodes have evolved: generations, shapes, sizes, fitness
  /memory [node-id]               memory nodes: what it learned, compressed; sizes, tests, compression history
  /memory test                    run every memory node against its test file now
  /memorize <question> :: <answer>   teach it a fact (compressed into a memory node with tests)
  /memory forget <question>       forget a fact (its tests go too)
  /memory merge [node node]       try merging two compressed memory nodes now (random pair if none given)
  /director                       its own initiative: mode, who chose it, the tasks it set itself, what paid off
  /director mode <auto|research|...|build|experiment|pursue|ruminate|rest>   pin a mode
  /director now | on | off        take initiative now, or turn its initiative on or off
  /directors                      the director nodes at both ends of every order: who is doing what, the
                                  stance its mood gives them, recent reviews, what memory was not given
  /directors spawn | retire <id>  add a duplicate director now, or retire one (never director-0)
  /directors attempts <n>         how many ways a director tries an unanswered order before it gives up
  /anatomy <node> [:: text]       how a node is built (layers, synapses), and what fires in it for a text;
                                  <node> is a node id, a memory node id, or a skill (its serving node)
  /traces                         the paths recent orders took through the mesh, node by node
  /objectives                     long-term objectives: their plans, progress, what it found, second thoughts
  /objective <what>               set one (also as an order: "objective: understand the Berlin Wall")
  /objective show|report|pause|resume|remove|think <id>    a plan and its answers; the write-up; think it over now
  /objective <id> ask <question>  add a question to an objective     /ruminate   think again about recent answers
  /toolsmith                      tools it wrote itself: the coder node, drafts waiting for approval, experiments
  /toolsmith build <what it should do>   have it write a tool now (it asks before the tool runs; P15)
  /toolsmith show <tool>          a made tool's code, tests and experiments    /toolsmith recipes   its code library
  /toolsmith try <tool>           have it experiment with a tool it made now
  /dreams                         sleep and dreams: the dream nodes, last night's dreams, what checked out
  /sleep    /wake                 send it to sleep (dreaming) now, or wake it
  /imagine <prompt>               its dream nodes imagine something (also: "imagine ...", "what if ...")
  /challenges                     benchmark suites, exam scores and trends
  /challenge run <suite> [n]      sit the exam now (graded, never learned from)
  /challenge practice <suite> [n] practise now: every graded answer teaches the mesh
  /challenge show <suite>         the last exam, question by question
  /challenge fetch <benchmark> [rows] [split]   download gsm8k, sst2, ag_news, boolq, arc_easy, arc_challenge,
                                  commonsense_qa or squad from Hugging Face   (/task challenge <suite> every 30m)
  /inquiries                      questions the mesh is asking you      /reply <inq-id> <answer>
  /proposals                      capabilities the mesh would like      /adopt <prop-id> [labels]  /decline <prop-id>
  /opinions   /notes              the opinions and notes it shares (private ones are only counted)
  /investigate <goal>             have the mesh look into something now with its own tools and nodes
  /mind                           operator: everything it keeps, private notes included (audited)
  /temperament                    the mesh's mood, its emotional engines and what they decided
  /why <order-id>                 why an answer was passed, changed, held or withheld
  /release <order-id>             show an answer the temperament held back     /calm   reset the mood
  /teach <capability> <label> :: <text>
  /define <name> <label,label,...> [:: description]
  /rollback <capability>          reinstate the previous champion
  /retest <tool>                  re-run a tool's self-tests
  /pause  /unpause  /halt [reason]  /resume
  /events [n]   /audit [n]   /help   /quit"""

LIVE_HELP = """Live view:
  Enter        send the order or command        Up/Down      history
  Tab          switch the left panel: network / output
  PgUp/PgDn    scroll the activity log (End jumps back to the newest)
  /log all     show every activity (default)    /log important   hide training, evaluation, single lessons
  /clear       clear the output panel           Ctrl-C or /quit  stop NeuroMesh"""


@dataclass
class Output:
    lines: list[str] = field(default_factory=list)
    quit: bool = False
    is_order: bool = False


def teacher_lines(system) -> list[str]:
    lines = [f"teaching order: {' -> '.join(system.teachers.order)}"]
    usage = system.guardian.llm_usage()
    for name, info in system.teachers.describe().items():
        model = f"{info['provider']}:{info['model']}" if info.get("provider") else ""
        state = "ready" if info["available"] else "off"
        used = usage.get(name)
        extra = f"  {used['calls_last_hour']} calls this hour" if used else ""
        lines.append(f"  {name:<10} {state:<5} {model:<32} {info['lessons']} lessons{extra}  {info.get('detail') or ''}")
    return lines


def _bar(value: float, lo: float, hi: float, width: int = 12) -> str:
    filled = round((value - lo) / (hi - lo) * width)
    return "#" * filled + "." * (width - filled)


def temperament_lines(system) -> list[str]:
    from .temperament import DIMENSIONS, RANGES
    a = system.affect
    st = a.status()
    if not st["enabled"]:
        return ["temperament: off ([temperament] enabled = false in the directives); answers pass unchanged"]
    lines = [f"temperament: {st['mood']}  (feelings fade to the baseline with a "
             f"{a.pol.half_life_seconds:.0f}s half-life; sensitivity {a.pol.sensitivity:g})"]
    for d in DIMENSIONS:
        lo, hi = RANGES[d]
        v, b = st[d], st["baseline"][d]
        shown = f"{v:+.2f}" if d == "valence" else f"{v:.2f}"
        lines.append(f"  {d:<11} {_bar(v, lo, hi)} {shown:>6}   baseline {b:+.2f}" if d == "valence"
                     else f"  {d:<11} {_bar(v, lo, hi)} {shown:>6}   baseline {b:.2f}")
    decisions = ", ".join(f"{k} {v}" for k, v in sorted(st["decisions"].items(), key=lambda kv: -kv[1]))
    lines.append(f"decisions this run: {decisions or 'none yet'}"
                 + (f"; honesty guard restored {st['restored_by_guard']}" if st["restored_by_guard"] else ""))
    for cap in ("affect", "disposition"):
        champ = system.registry.champion(cap)
        if champ is None:
            n = system.store.count(cap)
            who = "the temperament policy" if cap == "disposition" else "the teachers"
            lines.append(f"  {cap:<11} no node yet ({n}/{system.directives.nodes.min_examples_to_spawn} examples); "
                         f"{who} decide")
        else:
            acc = champ.metrics.get("holdout_acc")
            trusted = acc is not None and acc >= system.directives.nodes.serve_min_accuracy
            who = "" if trusted else (" - not trusted yet, so " + ("the temperament policy decides" if cap == "disposition"
                                                                   else "the teachers read the tone"))
            lines.append(f"  {cap:<11} node {champ.id} {champ.arch.describe()} held-out "
                         f"{'-' if acc is None else f'{acc:.3f}'}{who}")
    if st["held"]:
        lines.append("held answers (/release <id>): " + ", ".join(st["held"]))
    return lines


def _note_line(n, private_shown: bool = False) -> str:
    from .mind import display
    lock = "[private] " if n.private and private_shown else ""
    stance = f"{n.stance} ({n.strength:.0%}) - " if n.kind == "opinion" else ""
    return f"  {lock}{n.kind:<11} {display(n.subject)[:28]:<28} {stance}{' '.join(n.text.split())[:110]}"


def evolution_lines(system, capability: str = "") -> list[str]:
    evo = system.evolution
    if not evo.enabled:
        return ["  evolution is off ([evolution] enabled = false); challengers are bred by fixed mutations"]
    p = evo.pol
    out = [f"evolution: mutations grow {p.growth_bias:.0%} of the time; a larger challenger that is at least as "
           f"accurate wins{'' if p.promote_larger_on_tie else ' (off)'}; bounds {system.directives.nodes.max_depth} layers "
           f"x {system.directives.nodes.max_hidden} neurons, {p.max_params:,} weights"]
    caps = [capability] if capability else [c for c in system.capabilities if evo.archive.get(c)]
    if not caps:
        return out + ["  nothing evolved yet"]
    for cap in caps:
        sm = evo.summary(cap)
        ch = sm["champion"]
        if ch is None:
            out.append(f"  {cap}: no node yet")
            continue
        trend = " -> ".join(f"{n / 1000:.0f}k" for n in sm["sizes"]) or "-"
        out.append(f"  {cap}: champion {ch['id']} gen {ch['generation']} {'x'.join(map(str, ch['hidden']))} "
                   f"({ch['params']:,} weights, fitness {ch['fitness'] if ch['fitness'] is None else round(ch['fitness'], 3)}); "
                   f"{sm['tried']} genomes tried, {sm['promoted']} promoted; champion sizes {trend}")
        if capability:
            for e in sorted(evo.archive.get(cap, []), key=lambda e: e["created"])[-15:]:
                acc = "-" if e.get("acc") is None else f"{e['acc']:.3f}"
                change = "; ".join(e.get("changes") or []) or e.get("origin", "")
                out.append(f"    gen {e['generation']:<3} {'x'.join(map(str, e['hidden'])):<14} {e['params']:>9,} weights "
                           f"held-out {acc:<6} {e.get('outcome', 'candidate'):<9} {change[:70]}")
    return out


def _kb(n: int) -> str:
    return f"{n:,} B" if n < 10_000 else f"{n / 1024:,.1f} KB"


def memory_lines(system, rest: str = "") -> list[str]:
    mem = system.memory
    if not mem.enabled:
        return ["  memory nodes are off ([memory] enabled = false in the directives)"]
    p = mem.pol
    st = mem.status()
    head = (f"memory: {len(st['nodes'])} node(s), {st['facts']} facts in {_kb(st['bytes'])}"
            + (f" (first learned in {_kb(st['first_bytes'])})" if st["first_bytes"] > st["bytes"] else "")
            + f"; answers at confidence >= {p.answer_threshold:g}; tests in {mem.tests_dir}/")
    if rest and rest in mem.nodes:
        n = mem.nodes[rest]
        out = [f"  {n.id} [{n.status}] {len(n.facts())} facts, {len(n.expected_answers())} answers, {_kb(n.bytes)} "
               f"({n.spec()}), {n.passed}/{n.total} tests pass, served {n.served}x",
               f"  test file: {mem.tests_path(n)}"]
        if n.merged_from:
            out.append(f"  merged from {' and '.join(n.merged_from)}; made of {', '.join(n.parts)}")
        out += [f"    {mem._history_line(h)}" for h in n.history[-15:]]
        out.append("  facts:")
        out += [f"    {t.ask[:60]!r} -> {t.answer[:70]!r}" for t in n.facts()[:20]]
        if n.failing:
            out.append("  failing: " + ", ".join(repr(a[:50]) for a in n.failing[:8]))
        return out
    out = [head]
    if not st["nodes"]:
        return out + ["  nothing learned yet - it learns from research, teachers, challenges and /memorize"]
    for x in st["nodes"]:
        was = f" (was {_kb(x['first_bytes'])})" if x["first_bytes"] > x["bytes"] else ""
        tests = f"{x['passed']}/{x['total']} tests" if x["total"] else f"{x['tests']} tests, not checked yet"
        merged = f" - merged from {len(x['parts'])} nodes" if len(x["parts"]) > 1 else ""
        out.append(f"  {x['id']} [{x['status']}] {x['facts']} facts {_kb(x['bytes'])}{was} - {x['spec']} - {tests}"
                   + (f" - served {x['served']}x" if x["served"] else "") + merged)
    return out


async def memory_command(system, rest: str) -> list[str]:
    mem = system.memory
    sub, _, arg = rest.partition(" ")
    if sub == "test":
        if not mem.enabled:
            return ["  memory nodes are off"]
        mem.sync_files(force=True)
        out = []
        for node in list(mem.nodes.values()):
            r = await asyncio.to_thread(mem.run_tests, node)
            mark = "pass" if r["passed"] == r["total"] else "FAIL"
            out.append(f"  {node.id}: {r['passed']}/{r['total']} {mark}  ({mem.tests_path(node).name})")
            for x in [x for x in r["results"] if not x["pass"]][:5]:
                out.append(f"    x {x['ask'][:60]!r}: wanted {x['want'][:50]!r}, got {x['got'][:50]!r} ({x['confidence']:.2f})")
        return out or ["  no memory nodes yet"]
    if sub == "merge":
        if not mem.enabled:
            return ["  memory nodes are off"]
        names = arg.split()
        if names:
            if len(names) != 2 or any(x not in mem.nodes for x in names) or names[0] == names[1]:
                return ["  usage: /memory merge <node> <node> (two different memory nodes; /memory lists them)"]
            a, b = mem.nodes[names[0]], mem.nodes[names[1]]
            if any(n.params is None or n.dirty for n in (a, b)):
                return ["  both nodes must have learned their tests first"]
        else:
            pairs = mem.merge_candidates()
            if not pairs:
                return ["  no pair to try: merges need two compressed nodes that pass all their tests (not tried as they are)"]
            a, b = pairs[int(mem.rng.integers(len(pairs)))]
        job = mem.merge_job(a, b)
        res = await asyncio.to_thread(job.run)
        before = set(mem.nodes)
        mem.finish(job, res)
        made = sorted(set(mem.nodes) - before)
        rate = f"{res['pos']}/{res['pos_total']} answers and {res['neg']}/{res['neg_total']} unknowns pass"
        if made:
            n = mem.nodes[made[0]]
            return [f"  merged {a.id} and {b.id} into {n.id}: {len(n.facts())} facts, {_kb(n.bytes)} ({n.spec()}); {rate}"
                    + (f"; relearning the {len(n.failing)} it fails" if n.failing else "")]
        return [f"  {a.id} and {b.id} would not merge: {rate} (needs {mem.pol.merge_pass_rate:.0%}); both kept"]
    if sub == "forget":
        nid = mem.forget(arg.strip())
        return [f"  forgot '{arg.strip()}' (from {nid})" if nid else f"  no memory node knows '{arg.strip()}'"]
    return memory_lines(system, rest)


def director_lines(system) -> list[str]:
    d = system.director
    st = d.status()
    if not st["enabled"]:
        head = "director: initiative off - it only does what people ask (/director on)"
    else:
        nxt = f"next look in {st['next_in']}s" if st["next_in"] is not None else ""
        head = (f"director: initiative on, mode {st['pinned']}"
                + (" (the agency node chooses)" if st["pinned"] == "auto" else " (pinned by a person)")
                + f"; {st['decisions_last_hour']}/{st['limit']} decisions this hour; {nxt}")
    out = [head]
    node = st["node"]
    if node:
        held = "-" if node["holdout"] is None else f"{node['holdout']:.3f}"
        out.append(f"  agency node {node['id']} {node['arch']}: held-out {held}"
                   + (" - it chooses the mode" if node["trusted"] else " - still learning; the drive policy chooses")
                   + f" ({st['examples']} situations learned)")
    else:
        out.append(f"  agency node: not built yet ({st['examples']} situations learned); the drive policy chooses")
    if st["values"]:
        out.append("  what has paid off: " + ", ".join(f"{m} {v:.2f} ({st['counts'].get(m, 0)}x)"
                                                       for m, v in sorted(st["values"].items(), key=lambda kv: -kv[1])))
    if st["self_tasks"]:
        out.append("  tasks it set itself (/task remove <id> stops one):")
        out += [f"    {t['id']} [{t['status']}] {t['mode']}: {t['what']} ({t['runs']} runs, {t['age']}s) - {t['note']}"
                for t in st["self_tasks"]]
    if st["history"]:
        out.append("  recent decisions:")
        for h in st["history"][-8:]:
            when = time.strftime("%H:%M:%S", time.localtime(h["ts"]))
            worth = "" if h.get("value") is None else f" [worth {h['value']:.2f}]"
            out.append(f"    {when} {h['mode']}: {h['why']} ({h['how']}){(' - ' + h['summary']) if h.get('summary') else ''}{worth}")
    return out


async def director_command(system, rest: str) -> list[str]:
    d = system.director
    sub, _, arg = rest.partition(" ")
    by = os.environ.get("USER", "operator")
    if sub in ("on", "off"):
        return [f"  {d.set_enabled(sub == 'on', by)}"]
    if sub == "mode":
        ok, msg = d.set_mode(arg or "auto", by)
        return [f"  {msg}"]
    if sub in ("now", "decide"):
        if not d.enabled:
            return ["  its initiative is off (/director on)"]
        if system.guardian.halted:
            return ["  halted (/resume first)"]
        rec = await d.decide(force=True)
        if rec is None:
            return ["  it is already deciding; try again in a moment"]
        return [f"  {rec['mode']}: {rec['why']} ({rec['how']})" + (f" - {rec['summary']}" if rec.get("summary") else "")]
    return director_lines(system)


def directors_command(system, rest: str) -> list[str]:
    pool = system.directors
    sub, _, arg = rest.strip().partition(" ")
    by = os.environ.get("USER", "operator")
    if sub == "spawn":
        node = pool.spawn(f"asked for by {by}", by="director-0")
        return [f"  {node.id} is running: a duplicate of director-0" if node else
                f"  no more directors: {len(pool.nodes)} of at most {pool.limit} are running ([directors] max_directors)"]
    if sub == "retire":
        ok, msg = pool.retire(arg.strip(), f"retired by {by}", by=by)
        return [f"  {msg}"]
    if sub == "attempts":
        if not arg.strip().isdigit():
            return [f"  usage: /directors attempts <1-10> (now {int(pool.pol.max_attempts)})"]
        return [f"  {pool.set_attempts(int(arg), by)}"]
    return pool.lines()


async def objective_command(system, cmd: str, rest: str) -> list[str]:
    book = system.objectives
    by = os.environ.get("USER", "operator")
    rest = rest.strip()
    if cmd == "ruminate":
        if rest:
            obj = book.get(rest)
            if obj is None:
                return [f"  no objective {rest} (/objectives lists them)"]
            return [f"  {await book.ruminate(obj, by)}"] + book.show(rest)[-6:]
        got = await book.second_thoughts(by=by)
        return [f"  {g['verdict']}: {g['line']}" for g in got] or ["  nothing to think over again right now: no recent "
                                                                    "answers it was unsure of, or could not give"]
    if cmd == "objectives" or not rest:
        return book.lines()
    sub, _, arg = rest.partition(" ")
    arg = arg.strip()
    if sub in ("show", "report", "pause", "resume", "remove", "think") and arg:
        if sub == "show":
            return book.show(arg)
        obj = book.get(arg)
        if obj is None:
            return [f"  no objective {arg} (/objectives lists them)"]
        if sub == "report":
            return [f"  {book.write_report(obj)}", ""] + ["  " + ln for ln in book.report(obj).splitlines()]
        if sub == "think":
            return [f"  {await book.ruminate(obj, by)}"]
        ok, msg = book.set_status(arg, {"pause": "paused", "resume": "active", "remove": "removed"}[sub], by)
        return [f"  {msg}"]
    if re.fullmatch(r"obj-\d+", sub) and arg.lower().startswith("ask "):
        return [f"  {book.ask(sub, arg[4:].strip(), by)[1]}"]
    try:
        obj = await book.add(rest, by=by)
    except ValueError as exc:
        return [f"  {exc}"]
    return [f"  set {obj.id}: {obj.text} ({obj.strategy} strategy, planned by {obj.planned_by})"] + \
        [f"    {q['id']} [{q['kind']}] {q['ask']}" for q in obj.questions] + \
        [f"  it works on it when idle; /objective show {obj.id}, or read {book.report_path(obj)}"]


async def toolsmith_command(system, rest: str) -> list[str]:
    smith = system.toolsmith
    sub, _, arg = rest.strip().partition(" ")
    by = os.environ.get("USER", "operator")
    arg = arg.strip()
    if not smith.enabled and sub:
        return ["  the toolsmith is off ([toolsmith] enabled = false in the directives)"]
    if sub == "build":
        if not arg:
            return ["  usage: /toolsmith build <what the tool should do>   e.g. /toolsmith build reverse a word"]
        from .toolsmith import tool_request
        asked = tool_request(arg) or arg
        recipe = smith.classify(asked)[0]
        like = [u.text for u in system.mind.unmet.values() if not tool_request(u.text)
                and smith.classify(u.text)[0] == recipe] if recipe != "none" else []
        made = await smith.build(like, f"{by} asked for a tool that {asked}", "toolsmith", asked=asked)
        if made is None:
            last = smith.history[-1]["line"] if smith.history else "it could not write one"
            return [f"  it couldn't write that tool: {last}"]
        return [f"  wrote {made['name']} v{made['version']} ({made['recipe']}): {made['title']}",
                f"  its code and {len(made['tests'])} self-tests: {made['folder']}",
                f"  nothing it wrote runs until you approve it: /approve {made['approval']}   (/toolsmith show {made['name']})"]
    if sub in ("try", "experiment"):
        if arg not in smith.drafts:
            return [f"  no tool it made called '{arg}' (/toolsmith lists them)"]
        rep = await smith.experiment(arg, by)
        if rep is None:
            return [f"  {arg} is not active yet (it runs only once a person approves it, and it passes its self-tests)"]
        out = [f"  {rep['line']}"]
        for r in rep.get("results", [])[:12]:
            want = f" (want {r['want']})" if r["want"] is not None and not r["ok"] else ""
            out.append(f"    {'ok ' if r['ok'] else 'NO '} [{r['kind']}] {r['input'][:60]} -> {r['got'][:50]}{want}")
        return out
    if sub == "show":
        d = smith.drafts.get(arg)
        if d is None:
            return [f"  no tool it made called '{arg}' (/toolsmith lists them)"]
        out = [f"  {d['name']} v{d['version']} [{d['status']}]: {d['title']} (recipe {d['recipe']}, by {d['by']})",
               f"  why: {d['why']}", f"  files: {d['folder']}/tool.py, manifest.toml, TESTS.md"]
        if d.get("live") and d["live"] != d["version"]:
            out.append(f"  installed: v{d['live']} (v{d['version']} waits for approval)")
        if d["status"] == "awaiting approval":
            out.append(f"  waiting for you: /approve {d['approval']}  or  /deny {d['approval']}")
        out.append("  made for: " + ("; ".join(t[:60] for t in d["needs"][:4]) or f"a person asked for a tool that {d.get('asked', '')}"))
        out.append(f"  self-tests ({len(d['tests'])}):")
        out += [f"    {t['input'][:70]!r} -> {str(t['expect'])[:40]!r}" for t in d["tests"][:8]]
        for e in d.get("experiments", [])[-3:]:
            out.append(f"  experiment ({time.strftime('%H:%M', time.localtime(e['ts']))}, v{e.get('version', '?')}): "
                       f"{e['score']:.0%} of {e['runs']}" + (f"; failed: {', '.join(repr(f['input'][:30]) for f in e['failed'][:3])}"
                                                             if e["failed"] else ""))
        try:
            code = (Path(d["folder"]) / "tool.py").read_text().splitlines()
            body = [ln for ln in code if ln.startswith("def ")]
            out.append(f"  tool.py: {len(code)} lines; functions: {', '.join(ln[4:ln.index('(')] for ln in body)}")
        except (OSError, ValueError):
            pass
        return out
    if sub == "recipes":
        from .codelib import RECIPES
        return [f"  {r.id:<15} -> {r.name:<20} {r.title}" for r in RECIPES.values()]
    return smith.lines()


def dream_lines(system) -> list[str]:
    d = system.dreams
    if not d.enabled:
        return ["dreams are off ([dreams] enabled = false in the directives)"]
    st = d.status()
    if st["asleep"]:
        head = f"dreams: asleep for {st['slept_for']}s, dreaming ({st['dreams_tonight']} dreams so far; everything else waits)"
    else:
        head = (f"dreams: awake for {st['awake_for']}s; it may fall asleep after {d.pol.sleep_after_idle_seconds:g}s with no "
                f"orders (chance {d.pol.sleep_chance:.0%} each {d.pol.check_every_seconds:g}s)")
    out = [head, f"  {st['verified']} checked dreams kept; {st.get('sleeps', 0)} sleeps, {st.get('dreams', 0)} dreams, "
                 f"{st.get('evolutions', 0)} evolutions, {st.get('imagined', 0)} things imagined"]
    for n in st["nodes"]:
        fit = "-" if n["fitness"] is None else f"{n['fitness']:.0%}"
        out.append(f"  {'*' if n['champion'] else ' '} {n['id']} gen {n['generation']} {n['shape']} ({n['weights']:,} weights): "
                   f"{fit} of checked dreams right" + (f" - from {n['parent']}" if n["parent"] else ""))
    ln = st["last_night"]
    if ln:
        out.append(f"  last night: {ln['dreams']} dreams ({ln['rehearsed']} rehearsed); {ln['remembered_correct']}/{ln['remembered']} "
                   f"remembered answers right, {ln['imagined_confirmed']}/{ln['imagined']} imagined ones confirmed by its own memory"
                   + (f"; evolved: {ln['champion']} gen {ln['generation']} at {ln['fitness']:.0%}" if ln.get("evolved") else
                      "; nothing checked out, so no evolution"))
    for r in st["recent"][-6:]:
        out.append(f"    {r['kind']:<10} {r['question'][:60]!r} -> {r['answer'][:40]!r} [{r['verdict']}]")
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def challenge_lines(system) -> list[str]:
    ch = system.challenges
    if not ch.enabled:
        return ["  challenges are off ([challenges] enabled = false in the directives)"]
    rows = ch.summary()
    if not rows:
        return ["  no challenge suites; /challenge fetch gsm8k 200 downloads one"]
    out = []
    for r in rows:
        trend = (f"exam {_pct(r['last_exam'])} (first {_pct(r['first_exam'])}, best {_pct(r['best_exam'])}, "
                 f"{r['exams']} runs)" if r["exams"] else "not examined yet")
        what = f"{r['kind']}" + (f" -> {r['capability']}" if r["capability"] else "")
        out.append(f"  {r['suite']:<22} {what:<20} {r['practice']:>4} practice {r['exam']:>4} exam  {trend}")
    for name, err in ch.errors.items():
        out.append(f"  {name}: could not load ({err})")
    out.append(f"  routes learned from challenges: {ch.status()['routes_learned']}")
    return out


async def challenge_command(system, rest: str) -> list[str]:
    from .challenges import ADAPTERS
    ch = system.challenges
    verb, _, args = rest.partition(" ")
    parts = args.split()
    by = os.environ.get("USER", "operator")
    if verb in ("run", "exam", "practice"):
        if not parts:
            return [f"  usage: /challenge {verb} <suite> [n]"]
        mode = "practice" if verb == "practice" else "exam"
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        prev = ch.last(parts[0], mode)
        rep = await ch.run(parts[0], mode, n, by=by)
        now, before = round(rep.score, 4), prev["score"] if prev else None
        change = ("" if before is None else f", same as last time" if now == before else
                  f", {'up' if now > before else 'down'} from {before:.0%}")
        out = [f"  {rep.line()}{change}"]  # the result first
        if mode == "practice":
            out.append(f"  learned: {rep.learned} lessons; better routes found for {rep.rerouted} questions")
        for r in [r for r in rep.results if r["verdict"] != "pass"][:8]:
            out.append(f"   {r['verdict']:<5} want {str(r['want'])[:30]!r}, got {str(r['got'])[:40]!r} via {r['route']}: "
                       f"{r['input'][:60]}")
        return out
    if verb == "show":
        last = ch.last(parts[0] if parts else "", "exam")
        if last is None:
            return ["  that suite has not been examined yet (/challenge run <suite>)"]
        out = [f"  {last['suite']} exam: {last['score']:.0%} ({last['pass']} pass, {last['close']} close, {last['fail']} fail "
               f"of {last['n']})"]
        for r in last["results"]:
            out.append(f"   {r['verdict']:<5} {r['input'][:60]:<60} want {str(r['want'])[:24]!r} got {str(r['got'])[:30]!r}")
        return out
    if verb == "fetch":
        if not parts or parts[0] not in ADAPTERS:
            return ["  usage: /challenge fetch <benchmark> [rows] [split]; benchmarks: " + ", ".join(ADAPTERS)]
        rows = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        split = parts[2] if len(parts) > 2 else ""
        suite = await asyncio.to_thread(ch.fetch, parts[0], rows, split, "", by)
        return [f"  fetched {suite.name}: {len(suite.items)} items ({len(suite.practice())} practice, "
                f"{len(suite.exam())} exam) from {suite.source}",
                f"  /challenge run {suite.name} sits the exam; /challenge practice {suite.name} teaches"]
    return ["  usage: /challenge run|practice|show|fetch ...  (/challenges lists the suites)"]


def mind_lines(system, view: str) -> list[str]:
    """view: opinions | notes (what users may see) or mind (the operator's full view)."""
    m = system.mind
    if not m.enabled:
        return ["  the mind is off ([mind] enabled = false in the directives)"]
    if view == "mind":
        notes = m.inspect(os.environ.get("USER", "operator"))
        st = m.status()
        lines = [f"everything the mesh keeps ({len(notes)} notes, {st['private']} private; this look is audited):"]
        lines += [_note_line(n, True) for n in notes] or ["  nothing yet"]
        if st["goals"]:
            lines.append("investigation goals queued: " + "; ".join(st["goals"][:6]))
        lines.append(f"investigations: {st['investigations']['last_hour']}/{st['investigations']['limit']} this hour, "
                     f"{st['investigations']['total']} this run; {st['unmet']} unfulfilled orders being followed up")
        return lines
    if view == "opinions":
        ops = m.opinions()
        shared = [n for n in ops if m.shareable(n)]
        withheld = len(ops) - len(shared)
        lines = [_note_line(n) for n in shared] or ["  no opinions to share yet"]
        if withheld:
            lines.append(f"  ... and {withheld} opinion{'s' if withheld != 1 else ''} I'm keeping to myself")
        return lines
    notes = [n for n in m.public_notes(25) if n.kind != "opinion"]
    private = len(m.private_notes())
    lines = [_note_line(n) for n in notes] or ["  no notes to share yet"]
    if private:
        lines.append(f"  ... and {private} private note{'s' if private != 1 else ''} I'm not sharing")
    return lines


def knowledge_lines(system, question: str = "") -> list[str]:
    k = system.knowledge
    st = k.stats()
    usage = system.guardian.research_usage()
    tool = system.knowledge_tool()
    chain = " > ".join(t.name for t in system.knowledge_tools())
    local = f", {usage['unlimited_last_hour']} on your own server (unlimited)" if usage["unlimited_last_hour"] else ""
    lines = [f"knowledge: {st.documents} articles, {st.passages} passages retained; {st.recalled} answers "
             f"recalled, {st.researched} articles researched this run; research {usage['last_hour']}/"
             f"{usage['limit']} this hour ({usage['background_last_hour']}/{usage['background_limit']} background{local}); "
             f"research tools: {chain or 'none active (approve kiwix or wikipedia)'}"]
    if question:
        threshold = system.directives.research.recall_threshold
        hits = k.search(question)
        if not hits:
            return lines + [f"  nothing retained matches '{question}'"]
        for h in hits[:6]:
            verdict = "would answer from memory" if h.confidence >= threshold else "would research instead"
            lines.append(f"  {h.confidence:.2f} {h.doc.title} / {h.section}: {' '.join(h.text.split())[:90]}  ({verdict})")
        return lines
    for d in k.documents(15):
        lines.append(f"  {d.title:<34} {d.source:<9} {d.passages:>3} passages  used {d.hits:>3}x  {d.reason[:40]}")
    return lines


async def execute(system, line: str) -> Output:
    """Run one order or /command and return what to show."""
    line = line.strip()
    if not line:
        return Output()
    if not line.startswith("/"):
        res = await system.submit(line)
        first, *more = f"[{res.order_id}] {res.summary()}".splitlines() or [""]
        ask = ([f"   ? [{res.inquiry['id']}] {res.inquiry['question']}  (/reply {res.inquiry['id']} <answer>)"]
               if res.inquiry else [])
        return Output([first] + [f"   {m}" for m in more] + ask + [f"   . {step}" for step in res.trace],
                      is_order=True)
    cmd, _, rest = line[1:].partition(" ")
    rest = rest.strip()
    g = system.guardian
    out: list[str] = []
    try:
        if cmd in ("quit", "exit"):
            return Output(quit=True)
        elif cmd == "help":
            out = HELP.splitlines()
        elif cmd == "status":
            out = render_status(system.status()).splitlines()
        elif cmd == "lineage":
            out = render_lineage(system.registry.lineage, rest or None).splitlines() or ["no nodes built yet"]
        elif cmd == "tools":
            for name, t in system.status()["tools"].items():
                out.append(f"  {name:<14} {t['state']:<17} v{t['version']:<7} runs={t['runs']:<4} "
                           f"perms={t['permissions']}  {t['note']}")
        elif cmd == "teachers":
            out = teacher_lines(system)
        elif cmd == "research":
            if not rest:
                return Output(["  usage: /research <topic>"])
            doc, note = await system.research(rest, reason="operator request")
            if doc is None:
                out = [f"  could not research '{rest}': {note}"]
            else:
                out = [f"  learned '{doc.title}' ({doc.passages} passages) {doc.url}", f"  {doc.summary}"]
                if doc.links:
                    out.append(f"  linked articles queued for background reading: {', '.join(doc.links[:system.directives.research.follow_links])}")
        elif cmd in ("tasks", "goals") or (cmd in ("task", "goal") and not rest):
            out = system.tasks.lines() or ["  no background tasks; e.g. /task learn the Krebs cycle, "
                                           "/task improve sentiment to 0.97, /task every 10m: <order>"]
            usage = system.guardian.task_usage()
            out.append(f"  task runs this hour: {usage['last_hour']}/{usage['limit']}")
        elif cmd in ("task", "goal"):
            verb, _, arg = rest.partition(" ")
            verb = verb.lower()
            by = os.environ.get("USER", "user")
            if verb in ("pause", "resume", "remove", "stop", "delete"):
                status = {"pause": "paused", "resume": "active"}.get(verb, "removed")
                out = ["  " + system.tasks.set_status(arg.strip(), status, by)[1]]
            elif verb == "show":
                t = system.tasks.get(arg.strip())
                if t is None:
                    out = [f"  no background task {arg.strip()}"]
                else:
                    out = [f"  {t.id} [{t.status}] {t.describe()}, every {every(t.interval)}, set by {t.created_by}",
                           f"  {t.runs} runs; {t.note}"]
                    if t.kind == "learn":
                        p = t.progress
                        out.append(f"  read ({len(p.get('read', []))}/{t.max_articles}): {', '.join(p.get('read', [])) or '-'}")
                        if p.get("frontier"):
                            out.append(f"  next: {', '.join(p['frontier'][:8])}")
                    elif t.kind == "improve":
                        out.append(f"  accuracy {t.progress.get('accuracy', '-')} (best {t.progress.get('best', '-')}), "
                                   f"{t.progress.get('examples', 0)} examples; target {'met' if t.met else 'not met yet'}")
                    for ts, ok, summary in t.history[-8:]:
                        out.append(f"  {time.strftime('%H:%M:%S', time.localtime(ts))} {'ok ' if ok else 'err'} {summary[:120]}")
            else:
                t = system.tasks.add(rest, by=by)
                out = [f"  {t.id} set: {t.describe()}, every {every(t.interval)} (in the background; "
                       f"/task remove {t.id} stops it)"]
        elif cmd == "evolution":
            out = evolution_lines(system, rest)
        elif cmd == "memory":
            out = await memory_command(system, rest)
        elif cmd in ("dreams", "dream"):
            out = dream_lines(system)
        elif cmd == "sleep":
            ok, msg = system.dreams.fall_asleep("asked to", by=os.environ.get("USER", "operator"))
            out = [f"  {msg}"]
        elif cmd == "wake":
            out = ["  awake" if system.dreams.wake("woken by a person", by=os.environ.get("USER", "operator"))
                   else "  it is not asleep"]
        elif cmd == "imagine":
            res = await system.submit(f"imagine {rest}")
            out = [f"[{res.order_id}] {res.summary()}".replace("\n", "\n   ")] + [f"   . {t}" for t in res.trace]
        elif cmd in ("director", "agency", "initiative"):
            out = await director_command(system, rest)
        elif cmd == "directors":
            out = directors_command(system, rest)
        elif cmd in ("toolsmith", "coder"):
            out = await toolsmith_command(system, rest)
        elif cmd == "correct":
            book = system.corrections
            head, _, typed = rest.partition("::")
            parts = head.split()
            by = os.environ.get("USER", "user")
            if not parts:
                out = ["  usage: /correct <order-id> [yes | a1 | none [:: the right answer]]"]
            elif len(parts) == 1:
                alts = await book.alternatives(parts[0])
                out = book.lines(parts[0], alts) if alts else [f"  unknown order {parts[0]}"]
            elif parts[1].lower() in ("yes", "right", "correct", "y"):
                out = ["  " + (await book.right(parts[0], by))[1]]
            else:
                out = ["  " + (await book.choose(parts[0], parts[1].lower(), typed.strip(), by))[1]]
        elif cmd == "anatomy":
            from . import anatomy
            ref, _, text = rest.partition("::")
            out = anatomy.lines(system, ref.strip() or "dispatch", text.strip() or None)
        elif cmd == "traces":
            from . import anatomy
            out = []
            for t in anatomy.traces(system, 8):
                hops = " -> ".join((h.get("node") or h.get("tool") or h["stage"]) + (f" [{h['label']}]" if h.get("label") and h["stage"] not in ("director",) else "")
                                   for h in t["hops"])
                out.append(f"  {t['order']} '{t['text'][:50]}': {hops}")
            out = out or ["  no orders yet"]
        elif cmd in ("objective", "objectives", "ruminate"):
            out = await objective_command(system, cmd, rest)
        elif cmd in ("memorize", "memorise", "remember"):
            ask, sep, answer = rest.partition("::")
            if not sep or not ask.strip() or not answer.strip():
                out = ["  usage: /memorize <question> :: <answer>"]
            elif not system.memory.allows("people"):
                out = ["  memory nodes are off, or not learning from people ([memory] learn_from)"]
            else:
                made = system.memory.learn(ask.strip(), answer.strip(), "the operator",
                                           by=os.environ.get("USER", "operator"), kind="people")
                if made is None:
                    out = ["  that can't be kept as a fact (too short, empty, or a tool/skill question)"]
                else:
                    verb = {"new": "learning", "updated": "updating", "known": "already knows"}[made[1]]
                    out = [f"  {verb} it in memory node {made[0]}; tests in {system.memory.tests_dir.name}/{made[0]}.toml "
                           f"(/memory {made[0]})"]
        elif cmd == "challenges" or (cmd == "challenge" and not rest):
            out = challenge_lines(system)
        elif cmd == "challenge":
            out = await challenge_command(system, rest)
        elif cmd in ("opinions", "notes", "mind"):
            out = mind_lines(system, cmd)
        elif cmd == "inquiries":
            qs = system.mind.open_inquiries()
            out = [f"  {q.id} [{q.kind}] {q.question}" for q in qs] or ["  no questions for you right now"]
        elif cmd == "reply":
            iid, _, answer = rest.partition(" ")
            ok, out = await system.mind.reply(iid, answer, by=os.environ.get("USER", "user"))
        elif cmd == "proposals":
            ps = sorted(system.mind.proposals.values(), key=lambda p: (p.status != "open", -p.created_at))[:12]
            out = [f"  {p.id} [{p.status}] {p.kind} '{p.name}'" + (f" for {p.target}" if p.target else "")
                   + (f" labels={','.join(p.labels)}" if p.labels else "") + f" - {p.reason} (drafted by {p.by})"
                   for p in ps] or ["  no proposals yet"]
        elif cmd == "adopt":
            pid, _, labels = rest.partition(" ")
            labs = [x for x in labels.replace(",", " ").split() if x] or None
            out = ["  " + system.mind.adopt(pid, os.environ.get("USER", "human"), labs)[1]]
        elif cmd == "decline":
            pid, _, note = rest.partition(" ")
            out = ["  " + system.mind.decline(pid, os.environ.get("USER", "human"), note)[1]]
        elif cmd == "investigate":
            if not rest:
                return Output(["  usage: /investigate <goal>"])
            from .mind import Goal
            rec = await system.investigator.investigate(Goal(rest, "operator request", "operator"))
            out = [f"  {ln}" for ln in str(rec["answer"] or "I found nothing useful yet.").splitlines()]
            if rec["finding"]:
                out.append(f"  found: {rec['finding']}")
            out += [f"   . {st}" for st in rec["steps"]]  # how it went about it, last
        elif cmd == "temperament":
            out = temperament_lines(system)
        elif cmd == "why":
            out = system.affect.explain(rest) if rest else ["  usage: /why <order-id>"]
        elif cmd == "release":
            ok, msg, output = system.affect.release(rest, by=os.environ.get("USER", "human"))
            out = [f"  [{rest}] {msg}"] + ([f"   {ln}" for ln in str(output).splitlines()] if ok else [])
        elif cmd == "calm":
            st = system.affect.calm(by=os.environ.get("USER", "operator"))
            out = [f"  temperament reset to its baseline: {st['mood']}"]
        elif cmd == "knowledge":
            out = knowledge_lines(system, rest)
        elif cmd == "forget":
            out = [f"  forgot '{rest}'" if system.knowledge.forget(rest) else f"  nothing retained under '{rest}'"]
        elif cmd == "approvals":
            out = [(f"  {a.id}: a tool its toolsmith wrote, {a.subject} v{a.detail.get('version')} - "
                    f"{a.detail.get('purpose')} (code and tests: {a.detail.get('folder')})" if a.kind == "tool.made" else
                    f"  {a.id}: {a.kind} '{a.subject}' {a.detail}") for a in g.pending_approvals()] or ["  none pending"]
        elif cmd in ("approve", "deny"):
            aid, _, note = rest.partition(" ")
            a = g.decide(aid, cmd == "approve", by=os.environ.get("USER", "operator"), note=note)
            out = [f"  {a.id} {a.status}" if a else "  no pending approval with that id"]
            if a is not None and a.kind == "tool.made" and a.status == "approved":
                out.append(f"  installed {a.subject} in plugins/: it is vetted and self-tested next, then it tries it out "
                           f"on its own (/toolsmith show {a.subject})")
        elif cmd == "questions":
            out = [f"  {q['id']}: [{q['capability']}] {q['text'][:90]!r}  ({q['reason']})"
                   for q in system.teachers.human.questions] or ["  no open questions"]
        elif cmd == "answer":
            qid, _, label = rest.partition(" ")
            out = ["  " + system.answer_question(qid, label.strip())[1]]
        elif cmd == "feedback":
            oid, _, label = rest.partition(" ")
            out = ["  " + system.feedback(oid, label.strip())[1]]
        elif cmd == "teach":
            head, _, text = rest.partition("::")
            cap, label = head.split()
            system.teach(cap, text.strip(), label)
            out = ["  thanks - added as a human-labelled example"]
        elif cmd == "define":
            head, _, desc = rest.partition("::")
            name, labels = head.split(None, 1)
            spec = system.define_capability(name, [x for x in labels.replace(" ", "").split(",") if x], desc.strip())
            out = [f"  capability '{spec.name}' labels={spec.labels}. Orders like '{spec.name}: <text>' "
                   f"are taught by the teachers until a node takes over."]
        elif cmd == "rollback":
            node = system.registry.rollback(rest)
            out = [f"  {node.id} reinstated" if node else "  nothing to roll back to"]
        elif cmd == "retest":
            out = ["  " + system.retest_tool(rest)[1]]
        elif cmd == "pause":
            g.pause_background(True)
            out = ["  background work paused (orders are still served)"]
        elif cmd == "unpause":
            g.pause_background(False)
            out = ["  background work resumed"]
        elif cmd == "halt":
            g.halt(rest or "operator command")
            out = ["  halted: no orders or background work until /resume"]
        elif cmd == "resume":
            out = ["  " + g.resume()[1]]
        elif cmd == "events":
            out = [f"  {e.topic:<24} {json.dumps(e.data, default=str)[:140]}"
                   for e in system.bus.recent("*", int(rest or 20))
                   if e.topic not in ("audit", "order.done", "node.trained")]
        elif cmd == "audit":
            out = [f"  {a['actor']:<10} {a['kind']:<26} {a['subject']:<22} {json.dumps(a['detail'])[:90]}"
                   for a in system.store.recent_audit(int(rest or 20))]
        else:
            out = [f"  unknown command /{cmd}; /help lists them"]
    except Exception as exc:
        out = [f"  error: {exc}"]
    return Output(out)
