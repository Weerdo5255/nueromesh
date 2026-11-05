"""Operator commands and orders, shared by the plain console and the live view."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .report import render_lineage, render_status

HELP = """Type an order (e.g. "what is 12 * 7", "sentiment: the hotel was lovely") or a command:
  /status                         overview + directive scorecard
  /lineage [capability]           family tree of every node ever built
  /tools                          tools and their states
  /teachers                       teacher status (Ollama, Claude, ...)
  /approvals                      pending approvals      /approve <id> [note]   /deny <id> [note]
  /questions                      questions waiting for a human teacher   /answer <qid> <label>
  /feedback <order-id> <label>    correct an answer (a label, or a route like tool:calculator)
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


async def execute(system, line: str) -> Output:
    """Run one order or /command and return what to show."""
    line = line.strip()
    if not line:
        return Output()
    if not line.startswith("/"):
        res = await system.submit(line)
        return Output([f"[{res.order_id}] {res.summary()}"] + [f"   . {step}" for step in res.trace],
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
