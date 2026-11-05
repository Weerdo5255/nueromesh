"""Command line entry point.

  python -m neuromesh run      interactive console (orders + operator commands)
  python -m neuromesh serve    daemon with an HTTP API on 127.0.0.1:8765
  python -m neuromesh order    send one order to a running daemon
  python -m neuromesh demo     offline, scripted demonstration of the whole life cycle
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

HELP = """Type an order (e.g. "what is 12 * 7", "sentiment: the hotel was lovely") or a command:
  /status                         overview + directive scorecard
  /lineage [capability]           family tree of every node ever built
  /tools                          tools and their states
  /approvals                      pending approvals      /approve <id> [note]   /deny <id> [note]
  /questions                      questions waiting for a human teacher   /answer <qid> <label>
  /feedback <order-id> <label>    correct an answer (a label, or a route like tool:calculator)
  /teach <capability> <label> :: <text>
  /define <name> <label,label,...> [:: description]
  /rollback <capability>          reinstate the previous champion
  /retest <tool>                  re-run a tool's self-tests
  /pause  /unpause  /halt [reason]  /resume
  /events [n]   /audit [n]   /help   /quit"""


def _setup_logging(root: Path, verbose: bool) -> None:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(state / "neuromesh.log")])
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        logging.getLogger().addHandler(console)


async def _console(args) -> None:
    from .report import describe_event, render_lineage, render_status
    from .runtime import System

    system = System(args.root)
    loop = asyncio.get_running_loop()

    def show(ev):
        line = describe_event(ev)
        if line:
            print(f"   {line}", flush=True)

    system.bus.listen(show)
    async with system:
        print(f"{system.directives.name} running - {len(system.agents)} agents, "
              f"{len(system.registry.live())} live nodes. /help for commands.\n")
        while True:
            try:
                line = (await loop.run_in_executor(None, input, "neuromesh> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if not line.startswith("/"):
                res = await system.submit(line)
                print(f"[{res.order_id}] {res.summary()}")
                for step in res.trace:
                    print(f"      . {step}")
                continue
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            g = system.guardian
            try:
                if cmd in ("quit", "exit"):
                    break
                elif cmd == "help":
                    print(HELP)
                elif cmd == "status":
                    print(render_status(system.status()))
                elif cmd == "lineage":
                    print(render_lineage(system.registry.lineage, rest or None))
                elif cmd == "tools":
                    for name, t in system.status()["tools"].items():
                        print(f"  {name:<14} {t['state']:<17} v{t['version']:<7} runs={t['runs']:<4} "
                              f"perms={t['permissions']}  {t['note']}")
                elif cmd == "approvals":
                    for a in g.pending_approvals():
                        print(f"  {a.id}: {a.kind} '{a.subject}' {a.detail}")
                    if not g.pending_approvals():
                        print("  none pending")
                elif cmd in ("approve", "deny"):
                    aid, _, note = rest.partition(" ")
                    a = g.decide(aid, cmd == "approve", by=os.environ.get("USER", "operator"), note=note)
                    print(f"  {a.id} {a.status}" if a else "  no pending approval with that id")
                elif cmd == "questions":
                    for q in system.teachers.human.questions:
                        print(f"  {q['id']}: [{q['capability']}] {q['text'][:90]!r}  ({q['reason']})")
                    if not system.teachers.human.questions:
                        print("  no open questions")
                elif cmd == "answer":
                    qid, _, label = rest.partition(" ")
                    print("  " + system.answer_question(qid, label.strip())[1])
                elif cmd == "feedback":
                    oid, _, label = rest.partition(" ")
                    print("  " + system.feedback(oid, label.strip())[1])
                elif cmd == "teach":
                    head, _, text = rest.partition("::")
                    cap, label = head.split()
                    system.teach(cap, text.strip(), label)
                    print("  thanks - added as a human-labelled example")
                elif cmd == "define":
                    head, _, desc = rest.partition("::")
                    name, labels = head.split(None, 1)
                    spec = system.define_capability(name, [x for x in labels.replace(" ", "").split(",") if x],
                                                    desc.strip())
                    print(f"  capability '{spec.name}' labels={spec.labels}. Orders like '{spec.name}: <text>' "
                          f"are taught by the teachers until a node takes over.")
                elif cmd == "rollback":
                    node = system.registry.rollback(rest)
                    print(f"  {node.id} reinstated" if node else "  nothing to roll back to")
                elif cmd == "retest":
                    print("  " + system.retest_tool(rest)[1])
                elif cmd == "pause":
                    g.pause_background(True)
                elif cmd == "unpause":
                    g.pause_background(False)
                elif cmd == "halt":
                    g.halt(rest or "operator command")
                elif cmd == "resume":
                    print("  " + g.resume()[1])
                elif cmd == "events":
                    for e in system.bus.recent("*", int(rest or 20)):
                        if e.topic not in ("audit", "order.done", "node.trained"):
                            print(f"  {e.topic:<24} {json.dumps(e.data, default=str)[:140]}")
                elif cmd == "audit":
                    for a in system.store.recent_audit(int(rest or 20)):
                        print(f"  {a['actor']:<10} {a['kind']:<26} {a['subject']:<22} {json.dumps(a['detail'])[:90]}")
                else:
                    print("  unknown command; /help")
            except Exception as exc:
                print(f"  error: {exc}")


async def _serve(args) -> None:
    from .report import describe_event
    from .runtime import System
    from .server import serve_http

    system = System(args.root)
    system.bus.listen(lambda ev: (line := describe_event(ev)) and print(line, flush=True))
    async with system:
        server = await serve_http(system, args.host, args.port)
        print(f"{system.directives.name} serving on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
        async with server:
            await server.serve_forever()


def _order(args) -> None:
    req = urllib.request.Request(f"http://{args.host}:{args.port}/order",
                                 data=json.dumps({"text": args.text}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    if os.environ.get("NEUROMESH_TOKEN"):
        req.add_header("authorization", f"Bearer {os.environ['NEUROMESH_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read())
    print(json.dumps(res, indent=2) if args.json else
          f"{res['output'] if res['ok'] else '(not fulfilled) ' + res['note']}  "
          f"[{res['route']} | conf {res['confidence']:.2f} | {res['served_by']}]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="neuromesh", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="interactive console")
    p_serve = sub.add_parser("serve", help="daemon with HTTP API")
    for p in (p_run, p_serve):
        p.add_argument("--root", type=Path, default=Path("."), help="folder with directives.toml, "
                       "capabilities.toml and plugins/ (default: current folder)")
        p.add_argument("-v", "--verbose", action="store_true")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_order = sub.add_parser("order", help="send an order to a running daemon")
    p_order.add_argument("text")
    p_order.add_argument("--host", default="127.0.0.1")
    p_order.add_argument("--port", type=int, default=8765)
    p_order.add_argument("--json", action="store_true")
    p_demo = sub.add_parser("demo", help="offline scripted demonstration")
    p_demo.add_argument("--llm", action="store_true", help="also use the LLM teacher if configured")
    p_demo.add_argument("--keep", action="store_true", help="keep the demo's working folder")
    p_demo.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.cmd in ("run", "serve"):
        args.root = args.root.resolve()
        if not (args.root / "directives.toml").exists():
            sys.exit(f"no directives.toml in {args.root}; run from the project folder or pass --root")
        _setup_logging(args.root, args.verbose)
    try:
        if args.cmd == "run":
            asyncio.run(_console(args))
        elif args.cmd == "serve":
            asyncio.run(_serve(args))
        elif args.cmd == "order":
            _order(args)
        elif args.cmd == "demo":
            from .demo import run_demo
            asyncio.run(run_demo(use_llm=args.llm, keep=args.keep, seed=args.seed))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
