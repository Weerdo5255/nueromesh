"""Command line entry point.

  python -m neuromesh run      live console: network, lessons in progress, activity log, prompt
  python -m neuromesh serve    daemon with an HTTP API on 127.0.0.1:8765
  python -m neuromesh order    send one order to a running daemon
  python -m neuromesh demo     offline, scripted demonstration of the whole life cycle
  python -m neuromesh bench    compare CPU and GPU speed for node training and inference
  python -m neuromesh teachers show which teachers are available; --test runs a live lesson

run, serve, demo and bench take --device cpu|auto|cuda|cuda:N|mps (GPU needs PyTorch or CuPy).
run, serve, demo and teachers take --ollama MODEL to teach with a local Ollama model.
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

from .commands import HELP  # noqa: E402,F401  (kept importable from here)


def _setup_logging(root: Path, verbose: bool, console: bool = True) -> None:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(state / "neuromesh.log")])
    if verbose and console:
        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(handler)


def _live_possible(announce: bool = True) -> bool:
    """The live view needs a real terminal on both ends and the curses module."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        import curses  # noqa: F401,PLC0415
    except ImportError:
        if announce:
            print("live view needs curses (on Windows: pip install windows-curses); using the plain console")
        return False
    return True


async def _live(args) -> None:
    from .live import LiveView
    from .runtime import System

    system = System(args.root, device=args.device, library=args.gpu_library,
                    overrides=_teacher_overrides(args, args.root))
    if args.web:
        from .server import serve_http
        server = await serve_http(system, args.host, args.web)
        async with server:
            await LiveView(system, subtitle=f"web http://{args.host}:{args.web}/", ascii=args.ascii).run()
        return
    await LiveView(system, ascii=args.ascii).run()


def _teacher_overrides(args, root: Path) -> dict | None:
    """--ollama MODEL / --ollama-url URL: enable the local Ollama teacher and make sure it is
    in the teaching order (right after the human). --kiwix URL: where the local Kiwix server is."""
    model, url = getattr(args, "ollama", None), getattr(args, "ollama_url", None)
    kiwix = getattr(args, "kiwix", None)
    if not model and not url:
        return {"tools": {"config": {"kiwix": {"url": kiwix}}}} if kiwix else None
    from .config import Directives
    ollama = {"provider": "ollama", "enabled": True}
    if model:
        ollama["model"] = model
    if url:
        ollama["base_url"] = url
    order = list(Directives.load(root / "directives.toml").teachers.order)
    if "ollama" not in order:
        order.insert(order.index("human") + 1 if "human" in order else 0, "ollama")
    out: dict = {"teachers": {"order": order, "ollama": ollama}}
    if kiwix:
        out["tools"] = {"config": {"kiwix": {"url": kiwix}}}
    return out


async def _console(args) -> None:
    """Plain line-by-line console (used with --plain, or when not attached to a terminal)."""
    from .commands import execute
    from .report import describe_event
    from .runtime import System

    system = System(args.root, device=args.device, library=args.gpu_library,
                    overrides=_teacher_overrides(args, args.root))
    loop = asyncio.get_running_loop()

    def show(ev):
        line = describe_event(ev)
        if line:
            print(f"   {line}", flush=True)

    system.bus.listen(show)
    async with system:
        server = None
        if args.web:
            from .server import serve_http
            server = await serve_http(system, args.host, args.web)
        print(f"{system.directives.name} running - {len(system.agents)} agents, "
              f"{len(system.registry.live())} live nodes, compute: {system.backend.describe()}. "
              f"/help for commands." + (f" Web interface: http://{args.host}:{args.web}/" if server else "") + "\n")
        while True:
            try:
                line = (await loop.run_in_executor(None, input, "neuromesh> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            result = await execute(system, line)
            if result.quit:
                break
            for out in result.lines:
                print(out)
        if server is not None:
            server.close()


async def _serve(args) -> None:
    from .report import describe_event
    from .runtime import System
    from .server import serve_http

    system = System(args.root, device=args.device, library=args.gpu_library,
                    overrides=_teacher_overrides(args, args.root))
    if args.live:
        from .live import LiveView
        server = await serve_http(system, args.host, args.port)
        async with server:
            await LiveView(system, subtitle=f"web http://{args.host}:{args.port}/", ascii=args.ascii).run()
        return
    system.bus.listen(lambda ev: (line := describe_event(ev)) and print(line, flush=True))
    async with system:
        server = await serve_http(system, args.host, args.port)
        print(f"{system.directives.name} serving on http://{args.host}:{args.port}  "
              f"(compute: {system.backend.describe()}; Ctrl-C to stop)\n"
              f"web interface: http://{args.host}:{args.port}/", flush=True)
        async with server:
            await server.serve_forever()


async def _teachers(args) -> None:
    """Show every teacher's status and, with --test, give each available LLM teacher a live lesson."""
    import tempfile
    import time

    from .runtime import System
    from .teachers import LLMTeacher

    with tempfile.TemporaryDirectory() as tmp:  # a throw-away state folder: this is only a check
        system = System(args.root, state_dir=tmp, device="cpu", overrides=_teacher_overrides(args, args.root))
        await system.teachers.refresh(force=True)
        pool = system.teachers
        print(f"teaching order: {' -> '.join(pool.order)}\n")
        for name, info in pool.describe().items():
            model = f"{info['provider']}:{info['model']}" if info.get("provider") else "-"
            state = "ready" if info["available"] else "off"
            print(f"  {name:<10} {state:<5} {model:<34} {info.get('detail') or ''}")
            t = pool.teachers[name]
            if isinstance(t, LLMTeacher) and t.provider == "ollama" and t.installed:
                print(f"  {'':<10}       installed on {t.base_url}: {', '.join(t.installed)}")
        if args.test:
            spec = system.capabilities.get("sentiment") or next(
                s for n, s in system.capabilities.items() if n != "dispatch")
            print(f"\nlive check (capability '{spec.name}'):")
            tested = False
            for name in pool.order:
                t = pool.teachers.get(name)
                if not isinstance(t, LLMTeacher) or not t.available():
                    continue
                tested = True
                t0 = time.perf_counter()
                lesson = await t.label(spec, "The staff were friendly and the room was spotless.")
                got = f"{lesson.label} (confidence {lesson.confidence:.2f})" if lesson else f"no answer: {t.detail}"
                print(f"  {name}: label -> {got} in {time.perf_counter() - t0:.1f}s")
                t0 = time.perf_counter()
                examples = await t.generate(spec, spec.active_labels()[-1], 3)
                print(f"  {name}: wrote {len(examples)} examples of '{spec.active_labels()[-1]}' in "
                      f"{time.perf_counter() - t0:.1f}s" + (f", e.g. {examples[0]!r}" if examples else ""))
            if not tested:
                print("  no LLM teacher is available. For a local model: install Ollama, run "
                      "`ollama pull llama3.1`, then try again (or pass --ollama MODEL).")
        await system.stop()


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


def _challenge(args) -> None:
    """Download a benchmark into state/challenges/ without starting the mesh."""
    from .challenges import ADAPTERS, convert, fetch_rows, load_suite, save_suite
    from .config import Directives
    root = args.root.resolve()
    pol = Directives.load(root / "directives.toml").challenges
    if args.action == "list":
        print("benchmarks you can fetch:")
        for name, a in ADAPTERS.items():
            print(f"  {name:<15} {a['kind']:<9} {a['dataset']} ({a['config']}/{a['split']}): {a['description']}")
        folders = [root / f for f in pol.folders] + [root / "state" / "challenges"]
        suites = [d for base in folders if base.is_dir() for d in sorted(base.iterdir()) if (d / "manifest.toml").exists()]
        print("suites:" if suites else "no suites yet")
        for d in suites:
            s = load_suite(d)
            print(f"  {s.name:<22} {s.kind:<9} {len(s.items):>5} items ({len(s.exam())} exam)  {s.source}")
        return
    if args.benchmark not in ADAPTERS:
        sys.exit(f"choose a benchmark: {', '.join(ADAPTERS)}")
    a = ADAPTERS[args.benchmark]
    split = args.split or a["split"]
    import os
    rows = fetch_rows(a["dataset"], a["config"], split, args.rows or pol.fetch_limit, api=pol.api_url,
                      token=os.environ.get(pol.token_env) if pol.token_env else None)
    name = args.name or (args.benchmark if split == a["split"] else f"{args.benchmark}-{split}")
    suite = convert(args.benchmark, rows, name, f"huggingface:{a['dataset']} {a['config']}/{split} (first {len(rows)} rows)")
    suite.exam_percent = pol.exam_percent
    folder = save_suite(root / "state" / "challenges" / name, suite)
    print(f"fetched {name}: {len(suite.items)} items ({len(suite.exam())} held out as the exam) -> {folder}")


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
    p_run.add_argument("--plain", action="store_true", help="line-by-line console instead of the live view")
    p_run.add_argument("--web", nargs="?", type=int, const=8765, default=None, metavar="PORT",
                       help="also serve the web interface and HTTP API (default port 8765)")
    p_run.add_argument("--host", default="127.0.0.1", help="address for --web (default 127.0.0.1)")
    p_serve.add_argument("--live", action="store_true", help="show the live view (with a prompt) while serving")
    p_serve.add_argument("--web", action="store_true", help=argparse.SUPPRESS)  # the web interface is always on
    for p in (p_run, p_serve):
        p.add_argument("--ascii", action="store_true", help="live view without Unicode line drawing")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_order = sub.add_parser("order", help="send an order to a running daemon")
    p_order.add_argument("text")
    p_order.add_argument("--host", default="127.0.0.1")
    p_order.add_argument("--port", type=int, default=8765)
    p_order.add_argument("--json", action="store_true")
    p_teachers = sub.add_parser("teachers", help="show teacher status (Ollama, Claude, ...); --test runs a live lesson")
    p_teachers.add_argument("--root", type=Path, default=Path("."))
    p_teachers.add_argument("--test", action="store_true", help="send each available LLM teacher a real lesson")
    p_demo = sub.add_parser("demo", help="offline scripted demonstration")
    p_demo.add_argument("--llm", action="store_true", help="also use the configured LLM teachers (Claude, Ollama, ...)")
    p_demo.add_argument("--keep", action="store_true", help="keep the demo's working folder")
    p_demo.add_argument("--seed", type=int, default=7)
    p_chal = sub.add_parser("challenge", help="download benchmark suites (GSM8K, SST-2, ...) to test the mesh against")
    p_chal.add_argument("action", choices=["fetch", "list"], help="fetch a benchmark, or list adapters and suites")
    p_chal.add_argument("benchmark", nargs="?", help="gsm8k, sst2, ag_news, boolq, arc_easy, arc_challenge, "
                                                     "commonsense_qa or squad")
    p_chal.add_argument("--rows", type=int, default=0, help="rows to download (default: [challenges] fetch_limit)")
    p_chal.add_argument("--split", default="", help="dataset split (default: the benchmark's usual evaluation split)")
    p_chal.add_argument("--name", default="", help="suite name (default: the benchmark's name)")
    p_chal.add_argument("--root", type=Path, default=Path("."))
    p_mem = sub.add_parser("memory", help="run the memory nodes against their tests (without starting the mesh)")
    p_mem.add_argument("action", choices=["test"], help="test: every memory node against its tests in state/memory.db")
    p_mem.add_argument("--root", type=Path, default=Path("."))
    p_bench = sub.add_parser("bench", help="compare CPU and GPU speed for node training and inference")
    p_bench.add_argument("--input-dim", type=int, default=1024, help="encoder width (directives: input_dim)")
    p_bench.add_argument("--hidden", default="64", help="hidden layer sizes, e.g. 64 or 512,256")
    p_bench.add_argument("--batch-size", type=int, default=32)
    p_bench.add_argument("--examples", type=int, default=4000)
    p_bench.add_argument("--labels", type=int, default=8)
    p_bench.add_argument("--epochs", type=int, default=3)
    for p in (p_run, p_serve, p_demo, p_teachers):
        p.add_argument("--ollama", metavar="MODEL", default=None,
                       help="teach with a local Ollama model, e.g. llama3.1:8b ('auto' = first installed)")
        p.add_argument("--ollama-url", metavar="URL", default=None,
                       help="Ollama server (default: $OLLAMA_HOST or http://localhost:11434)")
        p.add_argument("--kiwix", metavar="URL", default=None,
                       help="local Kiwix server (kiwix-serve) to research with, e.g. http://localhost:8080")
    for p in (p_run, p_serve, p_demo, p_bench):
        p.add_argument("--device", default=None,
                       help="cpu | auto | cuda | cuda:N | mps (default: [compute] device in directives.toml, "
                            "or NEUROMESH_DEVICE; bench defaults to auto)")
        p.add_argument("--gpu-library", default=None, choices=["auto", "torch", "cupy", "numpy"],
                       help="which library drives the GPU (default: auto = PyTorch, then CuPy)")
    args = parser.parse_args(argv)

    if args.cmd in ("run", "serve", "teachers"):
        args.root = args.root.resolve()
        if not (args.root / "directives.toml").exists():
            sys.exit(f"no directives.toml in {args.root}; run from the project folder or pass --root")
    live = False
    if args.cmd == "run":
        live = not args.plain and _live_possible()
    elif args.cmd == "serve" and args.live:
        live = _live_possible()
        if not live:
            sys.exit("serve --live needs an interactive terminal with curses")
    if args.cmd in ("run", "serve"):
        _setup_logging(args.root, args.verbose, console=not live)
    from .backend import BackendUnavailable
    try:
        if args.cmd == "run":
            asyncio.run(_live(args) if live else _console(args))
        elif args.cmd == "serve":
            asyncio.run(_serve(args))
        elif args.cmd == "order":
            _order(args)
        elif args.cmd == "teachers":
            asyncio.run(_teachers(args))
        elif args.cmd == "demo":
            from .demo import run_demo
            asyncio.run(run_demo(use_llm=args.llm, keep=args.keep, seed=args.seed,
                                 device=args.device, library=args.gpu_library,
                                 ollama=args.ollama, ollama_url=args.ollama_url))
        elif args.cmd == "challenge":
            _challenge(args)
        elif args.cmd == "memory":
            from .memory import check_files
            sys.exit(1 if check_files(args.root) else 0)
        elif args.cmd == "bench":
            from .bench import run_bench
            hidden = tuple(int(h) for h in args.hidden.replace(" ", "").split(",") if h)
            run_bench(device=args.device or "auto", library=args.gpu_library or "auto",
                      input_dim=args.input_dim, hidden=hidden, batch_size=args.batch_size,
                      examples=args.examples, labels=args.labels, epochs=args.epochs)
    except BackendUnavailable as exc:
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
