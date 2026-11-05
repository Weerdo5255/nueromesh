"""Live terminal view for the console (curses, standard library only).

    +-- header: name, compute, uptime, orders, nodes, background load, run state ------+
    | NETWORK: orders -> router node -> routes -> skill nodes / tools,  | ACTIVITY LOG  |
    |   with challengers and retired nodes under each champion          | (every event, |
    | WORK IN PROGRESS: lessons from teachers, training, evaluations... |  newest last, |
    | ANSWERS: affect node -> disposition node -> honesty guard -> user |  scrollable)  |
    | TEACHERS / KNOWLEDGE / GOALS / MOOD / MIND (questions for you)    |               |
    | last result                                                       |               |
    | neuromesh> prompt                                                 |               |
    +-------------------------------------------------------------------+---------------+

The panel contents are built by plain functions (network_lines, work_lines, ...) so they
can be tested without a terminal; LiveView only lays them out and handles keys.
"""
from __future__ import annotations

import asyncio
import locale
import os
import re
import textwrap
import time

from .commands import HELP, LIVE_HELP, execute
from .tasks import DIRECT, DISPATCH

UNICODE = dict(h="─", v="│", tee="├", end="└", pipe="│", arrow="─▶",
               dashed="╌▶", to="⇒", star="★", sub="↳", ok="✓", no="✗",
               spin="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", dot="·", up="↑")
ASCII = dict(h="-", v="|", tee="+", end="`", pipe="|", arrow="->", dashed="~>", to="=>", star="*", sub=">",
             ok="ok", no="x", spin="|/-\\", dot="-", up="^")

Line = tuple[str, str]  # (text, style)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def glyphs_for_terminal(force_ascii: bool = False) -> dict:
    if force_ascii or os.environ.get("NEUROMESH_ASCII"):
        return ASCII
    return UNICODE if "utf" in (locale.getpreferredencoding(False) or "").lower() else ASCII


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _acc(node) -> str:
    acc = node.metrics.get("holdout_acc")
    return "held-out -" if acc is None else f"held-out {acc:.3f}"


# ---------------------------------------------------------------- panel content
def header_text(system, g: dict, extra: str = "") -> tuple[str, str]:
    s = system
    reg = s.registry
    built = len(reg.lineage)
    if s.guardian.halted:
        state, style = f"HALTED: {s.guardian.halt_reason}", "bad"
    elif s.guardian.paused:
        state, style = "BACKGROUND PAUSED", "warn"
    else:
        state, style = "RUNNING", "good"
    parts = [s.directives.name, s.backend.describe(), f"up {_ago(time.time() - s.started_at)}",
             f"orders {s.order_count}" + (f" ({s.inflight} in flight)" if s.inflight else ""),
             f"nodes {len(reg.live())} live / {built} built",
             f"background {s.guardian.duty_cycle():.0%}"]
    if extra:
        parts.append(extra)
    parts.append(state)
    return f" {g['v']} ".join(parts), style


def _node_detail(system, node, g: dict | None = None) -> str:
    dot = (g or UNICODE)["dot"]
    return f"{node.id} {node.arch.describe()} {dot} {_acc(node)} {dot} {node.metrics.get('serves', 0)} served"


def _route_target(system, route: str, g: dict) -> tuple[str, str]:
    s = system
    if route.startswith("tool:"):
        tool = s.tools.tools.get(route[5:])
        if tool is None:
            return f"{route[5:]} (missing)", "bad"
        style = {"active": "tool", "retired": "dim", "quarantined": "bad"}.get(tool.state, "warn")
        extra = ""
        if tool.manifest.provides == "knowledge":
            st = s.knowledge.stats()
            extra = f" {g['dot']} {st.documents} articles retained"
        return f"{tool.name} v{tool.manifest.version} [{tool.state}] {tool.runs} runs{extra}", style
    if route.startswith("skill:"):
        cap = route[6:]
        champ = s.registry.champion(cap)
        if champ is None:
            n = s.store.count(cap)
            return f"no node yet ({n}/{s.directives.nodes.min_examples_to_spawn} examples); teachers answer", "warn"
        return f"{g['star']} {_node_detail(s, champ, g)}", "champ"
    if route == DIRECT:
        llm = [t.name for t in s.teachers.ranked() if getattr(t, "provider", None)]
        return "teachers: " + (", ".join(llm) if llm else "none available (queued for a human)"), "teacher"
    return route, "info"


def _family_lines(system, cap: str, indent: str, g: dict) -> list[Line]:
    """Challengers and retired nodes of one capability, shown under its champion."""
    s = system
    npol = s.directives.nodes
    out: list[Line] = []
    for c in s.registry.candidates(cap):
        out.append((f"{indent}{g['sub']} challenger {c.id} {c.origin} {c.arch.describe()} {g['dot']} "
                    f"slice {c.train_steps}/{npol.max_candidate_slices} {g['dot']} {_acc(c)}", "chall"))
    for r in s.registry.retired(cap):
        left = npol.retired_grace_seconds - (time.time() - (r.retired_at or time.time()))
        out.append((f"{indent}{g['sub']} retired {r.id} {g['dot']} {_acc(r)} {g['dot']} disposal in {_ago(left)}", "dim"))
    return out


def network_lines(system, g: dict) -> list[Line]:
    """How the nodes are interconnected: orders flow into the router node, which routes them
    to tools or to skill nodes; challengers and retired nodes hang under their champion."""
    s = system
    lines: list[Line] = [("orders", "title")]
    router = s.registry.champion(DISPATCH)
    if router is not None:
        lines.append((f" {g['end']}{g['arrow']} router {g['star']} {_node_detail(s, router, g)}", "champ"))
    else:
        lines.append((f" {g['end']}{g['arrow']} router: no node yet ({s.store.count(DISPATCH)} examples) "
                      f"- teachers route orders", "warn"))
    lines += _family_lines(s, DISPATCH, "      ", g)
    spec = s.capabilities[DISPATCH]
    rank = {"skill": 0, "tool": 1}
    routes = sorted(spec.labels, key=lambda r: (rank.get(r.split(":")[0], 2), r))
    width = max((len(r) for r in routes), default=10)
    for i, route in enumerate(routes):
        last = i == len(routes) - 1
        branch = g["end"] if last else g["tee"]
        inactive = route in spec.inactive_labels
        arrow = g["dashed"] if inactive else g["arrow"]
        target, style = _route_target(s, route, g)
        if inactive:
            style = "dim"
        lines.append((f"    {branch}{arrow} {route:<{width}} {g['to']} {target}", style))
        if route.startswith("skill:"):
            cont = "    " if last else f"    {g['pipe']}"
            lines += _family_lines(s, route[6:], cont + " " * (width + 6), g)
    waiting = [t for t in s.tools.tools.values() if t.manifest.route not in spec.labels]
    for t in waiting:
        style = "bad" if t.state in ("rejected", "quarantined") else "warn"
        note = f" ({t.note})" if t.note else ""
        lines.append((f"    {g['dot']} tool {t.name} [{t.state}]{note}", style))
    lines += engine_lines(s, g)
    lineage = s.registry.lineage.values()
    disposed = sum(1 for m in lineage if m["status"] == "disposed")
    lines.append((f"nodes: {len(s.registry.live())} live {g['dot']} {len(s.registry.lineage)} built "
                  f"{g['dot']} {disposed} disposed", "dim"))
    return lines


def engine_lines(system, g: dict) -> list[Line]:
    """The emotional engines every answer passes through on its way to the user."""
    s = system
    a = s.affect
    if not a.enabled:
        return [("answers -> user directly (temperament off)", "dim")]
    lines: list[Line] = [("answers", "title")]
    engines = [c for c, spec in s.capabilities.items() if spec.internal]
    what = {"affect": "reads the tone", "disposition": "decides"}
    for cap in engines:
        champ = s.registry.champion(cap)
        if champ is None:
            n = s.store.count(cap)
            who = "temperament policy" if cap == "disposition" else "teachers"
            target, style = f"no node yet ({n}/{s.directives.nodes.min_examples_to_spawn} examples); {who} decide", "warn"
        else:
            target, style = f"{g['star']} {_node_detail(s, champ, g)}", "champ"
        lines.append((f" {g['tee']}{g['arrow']} {cap:<11} {what.get(cap, ''):<14} {g['to']} {target}", style))
        lines += _family_lines(s, cap, f" {g['pipe']}" + " " * 18, g)
    d = a.decisions
    changed = sum(v for k, v in d.items() if k != "pass")
    held = len(a.held)
    guard = (f"honesty guard (P8) {g['to']} {sum(d.values())} checked {g['dot']} {changed} changed "
             f"{g['dot']} {held} held {g['dot']} {d.get('block', 0)} withheld"
             + (f" {g['dot']} {a.restored} restored" if a.restored else ""))
    lines.append((f" {g['end']}{g['arrow']} {guard} {g['arrow']} user", "bad" if a.restored else "good"))
    return lines


def temperament_line(system, g: dict) -> Line:
    a = system.affect
    if not a.enabled:
        return ("mood:     temperament off", "dim")
    t = a.snapshot()
    last = f" (last change: {a.last[1]} {a.last[0]})" if a.last else ""
    style = {"wary": "bad", "tense": "bad", "alert": "warn", "subdued": "warn"}.get(t["mood"], "good")
    # most telling first, so a narrow panel cuts the least important
    return (f"mood:     {t['mood']}{last} {g['dot']} valence {t['valence']:+.2f} {g['dot']} trust {t['trust']:.2f} "
            f"{g['dot']} patience {t['patience']:.2f} {g['dot']} arousal {t['arousal']:.2f} {g['dot']} "
            f"confidence {t['confidence']:.2f}", style)


def _took(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def work_lines(system, g: dict, limit: int = 6) -> list[Line]:
    """Lessons being given and other work in flight right now (spinner), then the work
    that finished in the last few seconds (tick), newest first."""
    s = system
    frame = g["spin"][int(time.time() * 8) % len(g["spin"])]
    style = {"lesson": "lesson", "training": "work", "evaluation": "work", "self-test": "tool",
             "order": "order", "research": "lesson", "investigation": "lesson"}
    active = s.activity.current()
    lines: list[Line] = []
    for a in active[-limit:]:
        lines.append((f"{frame} {a.kind:<10} {a.actor:<9} {a.text}  {_took(a.elapsed)}...", style.get(a.kind, "info")))
    seen: dict[tuple, list] = {}  # collapse repeats of the same piece of work
    for a in s.activity.recent(15.0):
        key = (a.kind, a.actor, a.text)
        if key in seen:
            seen[key][1] += 1
        else:
            seen[key] = [a, 1]
    for a, n in list(seen.values())[: max(0, limit - len(lines))]:
        times = f" x{n}" if n > 1 else ""
        lines.append((f"{g['ok']} {a.kind:<10} {a.actor:<9} {a.text}{times}  {_took(a.elapsed)}, "
                      f"{_ago(time.time() - (a.ended or time.time()))} ago", "dim"))
    if not lines:
        lines.append(("  idle - nothing in flight or finished in the last 15s", "dim"))
    counts = ", ".join(f"{k} {v}" for k, v in sorted(s.teachers.lessons.items(), key=lambda kv: -kv[1]))
    lines.append((f"lessons given so far: {counts or 'none'}", "dim"))
    return lines


def teacher_goal_lines(system, g: dict) -> list[Line]:
    s = system
    usage = s.guardian.llm_usage()
    parts = []
    for name, info in s.teachers.describe().items():
        if not info.get("in_order", True):
            continue
        model = f":{info['model']}" if info.get("provider") else ""
        if info["available"]:
            calls = usage.get(name, {}).get("calls_last_hour")
            parts.append(f"{name}{model} {g['ok']}" + (f" {calls} calls/h" if calls else ""))
        else:
            parts.append(f"{name}{model} {g['no']}")
    goals = []
    for goal in s.scorecard():
        v = goal["value"]
        shown = "-" if v is None else (f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))
        op = ">=" if goal["direction"] == "at_least" else "<="
        goals.append(f"{goal['id'].split('-')[0]}:{shown} {op}{goal['target']:g} {g['ok'] if goal['met'] else g['no']}")
    k = s.knowledge.stats()
    r = s.guardian.research_usage()
    tool = s.knowledge_tool()
    queued = len(s.researcher.queue)
    knowledge = (f"knowledge: {k.documents} articles ({k.passages} passages) {g['dot']} {r['last_hour']}/{r['limit']} "
                 f"lookups/h {g['dot']} recalled {k.recalled} {g['dot']} researched {k.researched}"
                 + (f" {g['dot']} {queued} queued" if queued else "")
                 + ("" if tool else f" {g['dot']} no research tool active"))
    return [("teachers: " + f" {g['dot']} ".join(parts), "teacher"),
            (knowledge, "good" if tool else "dim"),
            ("goals:    " + f" {g['dot']} ".join(goals), "info"),
            temperament_line(s, g),
            mind_line(s, g)]


def mind_line(system, g: dict) -> Line:
    m = system.mind
    if not m.enabled:
        return ("mind:     off", "dim")
    st = m.status()
    asks = m.open_inquiries()
    inv = st["investigations"]
    parts = []
    if asks:  # the newest question first: it is what the user should see
        q = max(asks, key=lambda q: q.created_at)
        parts.append(f"asks [{q.id}] {q.question}" + (f" (+{len(asks) - 1} more: /inquiries)" if len(asks) > 1 else ""))
    if st["proposals"]:
        parts.append(f"{len(st['proposals'])} proposal{'s' if len(st['proposals']) != 1 else ''} (/proposals)")
    parts.append(f"{st['opinions']} opinions, {st['private']} private notes")
    parts.append(f"investigations {inv['last_hour']}/{inv['limit']}/h" + (f", {len(st['goals'])} queued" if st["goals"] else ""))
    return ("mind:     " + f" {g['dot']} ".join(parts), "warn" if asks else "info")


def wrap_log(entries, width: int, g: dict) -> list[list[tuple[str, str]]]:
    """Log entries as display rows; each row is a list of (text, style) segments."""
    rows = []
    indent = 18  # "HH:MM:SS " + 8-wide tag + space
    body_w = max(10, width - indent)
    for e in entries:
        stamp = time.strftime("%H:%M:%S", time.localtime(e.ts))
        chunks = textwrap.wrap(e.text, body_w, break_long_words=True, break_on_hyphens=False) or [""]
        rows.append([(f"{stamp} ", "dim"), (f"{e.tag[:8]:<8} ", e.level), (chunks[0], "info")])
        for chunk in chunks[1:]:
            rows.append([(" " * indent, "dim"), (chunk, "info")])
    return rows


# ----------------------------------------------------------------- terminal view
ESCAPES = {"[A": "up", "OA": "up", "[B": "down", "OB": "down", "[C": "right", "OC": "right",
           "[D": "left", "OD": "left", "[H": "home", "OH": "home", "[1~": "home", "[7~": "home",
           "[F": "end", "OF": "end", "[4~": "end", "[8~": "end", "[5~": "pgup", "[6~": "pgdn", "[3~": "del"}


class LiveView:
    REFRESH = 0.12          # seconds between redraws
    MIN_W, MIN_H = 70, 16

    def __init__(self, system, *, subtitle: str = "", manage_system: bool = True, ascii: bool = False):
        self.sys = system
        self.subtitle = subtitle
        self.manage = manage_system
        self.g = glyphs_for_terminal(ascii)
        self.text = ""
        self.cursor = 0
        self.history: list[str] = []
        self.hist_pos: int | None = None
        self.result: list[str] = ["Type an order or /help. Tab switches the left panel; PgUp/PgDn scroll the log."]
        self.output: list[str] = LIVE_HELP.splitlines() + [""] + HELP.splitlines()
        self.view = "network"
        self.log_scroll = 0
        self.routine = True
        self.pending: set[asyncio.Task] = set()
        self.quit = False
        self.status = "starting..."
        self._snapshot = os.environ.get("NEUROMESH_LIVE_SNAPSHOT")
        self._last_snap = 0.0
        self._log_rows = 20
        self._W = self._H = 0
        self._shadow: list[list[str]] = []  # what was drawn, for snapshots

    # ------------------------------------------------------------------ main
    async def run(self) -> None:
        import curses  # noqa: PLC0415 - optional at import time (Windows needs windows-curses)
        self.curses = curses
        locale.setlocale(locale.LC_ALL, "")
        os.environ.setdefault("ESCDELAY", "25")  # a lone Esc should not wait a second
        scr = curses.initscr()
        try:
            curses.noecho()
            curses.cbreak()
            scr.keypad(True)
            scr.nodelay(True)
            try:
                curses.curs_set(1)
            except curses.error:
                pass
            self._init_colors()
            self.scr = scr
            starter = asyncio.create_task(self.sys.start()) if self.manage else None
            if starter is None:
                self.status = ""
            last_draw = 0.0
            while not self.quit:
                if starter is not None and starter.done():
                    exc = starter.exception()
                    self.status = ""
                    if exc:
                        self.result = [f"start-up failed: {exc}"]
                    starter = None
                changed = self._read_keys()
                now = time.monotonic()
                if changed or now - last_draw >= self.REFRESH:
                    self._draw()
                    last_draw = now
                await asyncio.sleep(0.03)
            self.status = "stopping..."
            self._draw()
            for task in self.pending:
                task.cancel()
            if self.manage:
                await self.sys.stop()
        finally:
            try:
                scr.keypad(False)
                curses.nocbreak()
                curses.echo()
            finally:
                curses.endwin()

    def _init_colors(self) -> None:
        c = self.curses
        self.styles: dict[str, int] = {}
        base = {"info": 0, "dim": c.A_DIM, "title": c.A_BOLD}
        colored = {"good": (c.COLOR_GREEN, 0), "champ": (c.COLOR_GREEN, c.A_BOLD), "work": (c.COLOR_YELLOW, 0),
                   "chall": (c.COLOR_YELLOW, 0), "lesson": (c.COLOR_YELLOW, c.A_BOLD), "warn": (c.COLOR_YELLOW, 0),
                   "bad": (c.COLOR_RED, c.A_BOLD), "tool": (c.COLOR_CYAN, 0), "teacher": (c.COLOR_MAGENTA, 0),
                   "order": (c.COLOR_BLUE, c.A_BOLD), "frame": (c.COLOR_CYAN, 0)}
        self.styles.update(base)
        if c.has_colors():
            c.start_color()
            try:
                c.use_default_colors()
                bg = -1
            except c.error:
                bg = c.COLOR_BLACK
            for i, (name, (fg, extra)) in enumerate(colored.items(), start=1):
                c.init_pair(i, fg, bg)
                self.styles[name] = c.color_pair(i) | extra
        else:
            for name, (_, extra) in colored.items():
                self.styles[name] = extra or c.A_BOLD

    # ----------------------------------------------------------------- input
    def _escape(self) -> str | None:
        """Read the rest of an escape sequence curses did not translate itself."""
        seq = ""
        for _ in range(6):
            try:
                ch = self.scr.get_wch()
            except self.curses.error:
                break
            if not isinstance(ch, str):
                break
            seq += ch
            if seq in ESCAPES:
                return ESCAPES[seq]
            if not (seq in ("[", "O") or seq[-1].isdigit() or seq[-1] == ";"):
                break
        return "esc" if not seq else None

    def _read_keys(self) -> bool:
        c = self.curses
        changed = False
        while True:
            try:
                key = self.scr.get_wch()
            except c.error:
                return changed
            changed = True
            if key == "\x1b":
                named = self._escape()
                key = {"up": c.KEY_UP, "down": c.KEY_DOWN, "left": c.KEY_LEFT, "right": c.KEY_RIGHT,
                       "home": c.KEY_HOME, "end": c.KEY_END, "pgup": c.KEY_PPAGE, "pgdn": c.KEY_NPAGE,
                       "del": c.KEY_DC, "esc": "\x1b"}.get(named or "", None)
                if key is None:
                    continue
            if key == c.KEY_RESIZE:
                c.update_lines_cols()
                self.scr.clear()
            elif key in ("\n", "\r", c.KEY_ENTER):
                self._submit()
            elif key in (c.KEY_BACKSPACE, "\x7f", "\b"):
                if self.cursor > 0:
                    self.text = self.text[: self.cursor - 1] + self.text[self.cursor:]
                    self.cursor -= 1
            elif key == c.KEY_DC:
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1:]
            elif key == c.KEY_LEFT:
                self.cursor = max(0, self.cursor - 1)
            elif key == c.KEY_RIGHT:
                self.cursor = min(len(self.text), self.cursor + 1)
            elif key in (c.KEY_HOME, "\x01"):
                self.cursor = 0
            elif key in (c.KEY_END, "\x05"):
                self.cursor = len(self.text)
            elif key == "\x15":  # Ctrl-U: clear the line
                self.text, self.cursor = "", 0
            elif key == "\x17":  # Ctrl-W: delete the previous word
                head = self.text[: self.cursor].rstrip()
                cut = head.rfind(" ") + 1
                self.text, self.cursor = self.text[:cut] + self.text[self.cursor:], cut
            elif key == c.KEY_UP:
                self._history(-1)
            elif key == c.KEY_DOWN:
                self._history(1)
            elif key == "\t":
                self.view = "output" if self.view == "network" else "network"
            elif key == "\x1b":
                self.view = "network"
            elif key == c.KEY_PPAGE:
                self.log_scroll += max(1, self._log_rows - 2)
            elif key == c.KEY_NPAGE:
                self.log_scroll = max(0, self.log_scroll - max(1, self._log_rows - 2))
            elif key in ("\x03", "\x04") and not self.text:
                self.quit = True
            elif isinstance(key, str) and key.isprintable():
                self.text = self.text[: self.cursor] + key + self.text[self.cursor:]
                self.cursor += len(key)

    def _history(self, step: int) -> None:
        if not self.history:
            return
        pos = len(self.history) if self.hist_pos is None else self.hist_pos
        pos = min(len(self.history), max(0, pos + step))
        self.hist_pos = pos
        self.text = self.history[pos] if pos < len(self.history) else ""
        self.cursor = len(self.text)

    def _submit(self) -> None:
        line = self.text.strip()
        self.text, self.cursor, self.hist_pos = "", 0, None
        if not line:
            return
        if not self.history or self.history[-1] != line:
            self.history.append(line)
        cmd, _, rest = line.partition(" ")
        if cmd in ("/quit", "/exit"):
            self.quit = True
        elif cmd == "/log":
            self.routine = rest.strip() != "important"
            self.result = [f"activity log: {'everything' if self.routine else 'important events only'}"]
        elif cmd == "/clear":
            self.output, self.result = [], []
        elif cmd == "/help":
            self.output = LIVE_HELP.splitlines() + [""] + HELP.splitlines()
            self.view = "output"
        else:
            self.result = [f"{self.g['spin'][0]} {line}"]
            task = asyncio.create_task(self._execute(line))
            self.pending.add(task)
            task.add_done_callback(self.pending.discard)

    async def _execute(self, line: str) -> None:
        try:
            out = await execute(self.sys, line)
        except Exception as exc:  # never let a command take the view down
            self.result = [f"error: {exc}"]
            return
        if out.quit:
            self.quit = True
        elif out.is_order:
            self.result = out.lines
            self.output = [f"> {line}"] + out.lines  # Tab shows the whole result and its trace
        else:
            self.output = [f"> {line}"] + out.lines
            self.result = out.lines[:1] if len(out.lines) == 1 else [f"> {line}: {len(out.lines)} lines (Tab shows them)"]
            if len(out.lines) > 1:
                self.view = "output"

    # ------------------------------------------------------------------ draw
    def _put(self, y: int, x: int, text: str, style: str = "info", width: int | None = None) -> None:
        room = self._W - x  # never let curses wrap a line onto the next row
        width = room if width is None else min(width, room)
        if width <= 0 or not 0 <= y < self._H:
            return
        text = _CONTROL.sub(" ", text)[:width]  # a newline would move the cursor and wreck the layout
        try:
            self.scr.addstr(y, x, text, self.styles.get(style, 0))
        except self.curses.error:
            pass  # writing the bottom-right cell raises; harmless
        row = self._shadow[y] if y < len(self._shadow) else None
        if row is not None:
            for i, ch in enumerate(text):
                if x + i < len(row):
                    row[x + i] = ch

    def _rule(self, y: int, x: int, width: int, title: str, style: str = "frame") -> None:
        g = self.g
        label = f"{g['h']} {title} " if title else ""
        self._put(y, x, (label + g["h"] * width)[:width], style)

    def _draw(self) -> None:
        scr = self.scr
        H, W = scr.getmaxyx()
        self._H, self._W = H, W
        self._shadow = [[" "] * W for _ in range(H)]
        scr.erase()
        g = self.g
        if W < self.MIN_W or H < self.MIN_H:
            self._put(0, 0, f"NeuroMesh: enlarge the terminal to at least {self.MIN_W}x{self.MIN_H}", "warn", W - 1)
            self._draw_prompt(H - 1, 0, W)
            scr.refresh()
            self._maybe_snapshot(H, W)
            return
        text, style = header_text(self.sys, g, self.subtitle)
        if self.status:
            text, style = f"{text} {g['v']} {self.status}", "warn"
        self._put(0, 0, f" {text} ".ljust(W)[: W - 1], style if style == "bad" else "title")

        log_w = max(38, int(W * 0.40))
        left_w = W - log_w - 1
        for y in range(1, H):
            self._put(y, left_w, g["v"], "frame")
        self._draw_log(1, left_w + 1, log_w, H - 1)

        prompt_y = H - 1
        result = self.result  # the answer first; the trace as far as it fits
        if len(result) > 4:
            result = result[:3] + [f"   ... {len(result) - 3} more lines (Tab shows them)"]
        result_top = prompt_y - len(result)
        self._rule(result_top - 1, 0, left_w, "result", "frame")
        for i, line in enumerate(result):
            self._put(result_top + i, 1, line, "order" if i == 0 else "info", left_w - 2)
        self._draw_prompt(prompt_y, 0, left_w)

        top, bottom = 1, result_top - 1  # rows [top, bottom) are for the panels
        if self.view == "output":
            self._rule(top, 0, left_w, "output  (Tab: back to the network)")
            rows = bottom - top - 1
            lines = []
            for line in self.output:  # wrap long lines, keeping their indent
                indent = " " * (len(line) - len(line.lstrip()) + 2)
                lines += textwrap.wrap(line, max(20, left_w - 2), subsequent_indent=indent,
                                       break_long_words=True, drop_whitespace=True) or [""]
            shown = lines[:rows] if len(lines) <= rows else lines[: rows - 1] + [f"... {len(lines) - rows + 1} more lines"]
            for i, line in enumerate(shown):
                self._put(top + 1 + i, 1, line, "info", left_w - 2)
        else:
            tg = teacher_goal_lines(self.sys, g)
            work = work_lines(self.sys, g, limit=max(1, min(6, (bottom - top) // 4)))
            tg_top = bottom - len(tg)
            work_top = tg_top - len(work) - 1
            net_rows = work_top - top - 1
            self._rule(top, 0, left_w, "network: how orders reach nodes and tools")
            net = network_lines(self.sys, g)
            if len(net) > net_rows:
                net = net[: max(0, net_rows - 1)] + [(f"... {len(net) - net_rows + 1} more (resize, or /status)", "dim")]
            for i, (line, st) in enumerate(net):
                self._put(top + 1 + i, 1, line, st, left_w - 2)
            self._rule(work_top, 0, left_w, "lessons and work in progress")
            for i, (line, st) in enumerate(work):
                self._put(work_top + 1 + i, 1, line, st, left_w - 2)
            for i, (line, st) in enumerate(tg):
                self._put(tg_top + i, 1, line, st, left_w - 2)
        # leave the terminal cursor in the input line
        prefix = len("neuromesh> ")
        visible = max(1, left_w - prefix - 1)
        start = max(0, self.cursor - visible + 1)
        try:
            scr.move(prompt_y, min(left_w - 1, prefix + self.cursor - start))
        except self.curses.error:
            pass
        scr.refresh()
        self._maybe_snapshot(H, W)

    def _draw_prompt(self, y: int, x: int, width: int) -> None:
        prefix = "neuromesh> "
        visible = max(1, width - len(prefix) - 1)
        start = max(0, self.cursor - visible + 1)
        self._put(y, x, prefix, "title")
        self._put(y, x + len(prefix), self.text[start:start + visible], "info")

    def _draw_log(self, top: int, x: int, width: int, height: int) -> None:
        g = self.g
        self._log_rows = height - 1
        entries = self.sys.log.tail(600, routine=self.routine)
        rows = wrap_log(entries, width - 1, g)
        self.log_scroll = min(self.log_scroll, max(0, len(rows) - self._log_rows))
        end = len(rows) - self.log_scroll
        view = rows[max(0, end - self._log_rows): end]
        mode = "all" if self.routine else "important"
        scrolled = f" {g['up']}{self.log_scroll} (PgDn)" if self.log_scroll else ""
        self._rule(top, x, width, f"activity log [{mode}]{scrolled}")
        for i, row in enumerate(view):
            col = x + 1
            for text, style in row:
                self._put(top + 1 + i, col, text, style, x + width - col)
                col += len(text)

    def _maybe_snapshot(self, H: int, W: int) -> None:
        """Debug/test hook: write the screen as text to $NEUROMESH_LIVE_SNAPSHOT."""
        if not self._snapshot or time.monotonic() - self._last_snap < 0.5:
            return
        self._last_snap = time.monotonic()
        rows = ["".join(row).rstrip() for row in self._shadow]
        try:
            with open(self._snapshot, "w", encoding="utf-8") as fh:
                fh.write("\n".join(rows) + "\n")
        except OSError:
            pass
