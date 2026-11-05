"""Operator commands and orders, shared by the plain console and the live view."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

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
  /feedback <order-id> <label>    correct an answer: a label, a route like tool:calculator, or the
                                  emotional engines with affect=<tone> / disposition=<action>
  /task learn <topic> [every 2m] [max 30]     keep studying a topic in the background
  /task improve <skill> [to 0.97]             keep training a skill until it reaches the target
  /task every 10m: <order>                    repeat an order; you are told when its result changes
  /tasks    /task show|pause|resume|remove <task-id>     (also: "keep learning about X" as an order)
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
            out = [f"  investigated '{rest}':"] + [f"   . {st}" for st in rec["steps"]]
            out.append(f"  found: {rec['finding'] or 'nothing useful yet'}")
            if rec["answer"]:
                out += [f"  answer: {ln}" for ln in str(rec["answer"]).splitlines()]
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
            out = [f"  {a.id}: {a.kind} '{a.subject}' {a.detail}" for a in g.pending_approvals()] or ["  none pending"]
        elif cmd in ("approve", "deny"):
            aid, _, note = rest.partition(" ")
            a = g.decide(aid, cmd == "approve", by=os.environ.get("USER", "operator"), note=note)
            out = [f"  {a.id} {a.status}" if a else "  no pending approval with that id"]
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
