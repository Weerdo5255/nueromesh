"""The web interface: what the dashboard at http://127.0.0.1:8765/ shows, and its static files.

`neuromesh serve` (and `neuromesh run --web`) serve the page from neuromesh/web/. The page is
static; everything it shows comes from the JSON API (server.py), which these helpers feed:
the network of nodes as a structure to draw, the work in flight, and the activity log by
sequence number so the browser only ever fetches what is new.
"""
from __future__ import annotations

import time
from pathlib import Path

from .activity import ROUTINE
from .tasks import DIRECT, DISPATCH

WEB_DIR = Path(__file__).resolve().parent / "web"
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".ico": "image/x-icon"}


def static_file(path: str) -> tuple[bytes, str] | None:
    """The page and its files: / and /ui/<file>. Nothing outside neuromesh/web/ is ever served."""
    if path in ("/", "/ui", "/ui/", "/index.html"):
        name = "index.html"
    elif path.startswith("/ui/"):
        name = path[4:]
    else:
        return None
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    f = WEB_DIR / name
    if f.suffix not in TYPES or not f.is_file():
        return None
    return f.read_bytes(), TYPES[f.suffix]


def node_brief(node) -> dict:
    m = node.metrics
    return {"id": node.id, "status": node.status, "origin": node.origin, "generation": node.generation,
            "arch": node.arch.describe(), "hidden": list(node.arch.hidden), "params": node.param_count,
            "holdout": m.get("holdout_acc"), "serves": m.get("serves", 0), "steps": node.train_steps,
            "labels": len(node.labels), "retired_at": node.retired_at}


def _family(system, cap: str) -> dict:
    champ = system.registry.champion(cap)
    return {"capability": cap, "examples": system.store.count(cap),
            "champion": node_brief(champ) if champ else None,
            "candidates": [node_brief(n) for n in system.registry.candidates(cap)],
            "retired": [node_brief(n) for n in system.registry.retired(cap)]}


def mesh(system) -> dict:
    """How orders reach the nodes: the router, each route and what serves it, the memory nodes
    asked first, and the emotional engines every answer passes on its way out."""
    s = system
    spec = s.capabilities[DISPATCH]
    rank = {"skill": 0, "tool": 1}
    routes = []
    for route in sorted(spec.labels, key=lambda r: (rank.get(r.split(":")[0], 2), r)):
        kind = route.split(":")[0]
        item = {"route": route, "kind": kind, "name": route.split(":", 1)[-1],
                "inactive": route in spec.inactive_labels}
        if kind == "skill" and route[6:] in s.capabilities:
            item["skill"] = _family(s, route[6:])
        elif kind == "tool":
            t = s.tools.tools.get(route[5:])
            item["tool"] = ({"name": t.name, "state": t.state, "version": t.manifest.version, "runs": t.runs,
                             "provides": t.manifest.provides, "note": t.note} if t else None)
        elif route == DIRECT:
            item["teachers"] = [t.name for t in s.teachers.ranked() if getattr(t, "provider", None)]
        routes.append(item)
    waiting = [{"name": t.name, "state": t.state, "note": t.note, "route": t.manifest.route}
               for t in s.tools.tools.values() if t.manifest.route not in spec.labels]
    engines = ([_family(s, c) for c, sp in s.capabilities.items() if sp.internal and sp.kind != "agency"]
               if s.affect.enabled else [])
    mem = s.memory
    memory = [{"id": n.id, "facts": len(n.facts()), "bytes": n.bytes, "first_bytes": n.first_bytes,
               "spec": n.spec(), "passed": n.passed, "total": n.total, "served": n.served,
               "status": "learning" if n.dirty else n.status}
              for n in sorted(mem.nodes.values(), key=lambda n: n.created)] if mem.enabled else []
    a = s.affect
    d = s.director
    agency = s.registry.champion("agency") if "agency" in s.capabilities else None
    last = d.history[-1] if d.history else None
    director = {"enabled": d.enabled, "pinned": d.pinned, "node": node_brief(agency) if agency else None,
                "trusted": bool(agency and (agency.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy),
                "last": {k: last.get(k) for k in ("mode", "target", "summary", "value", "ts")} if last else None,
                "tasks": [{"id": t, "mode": r["mode"], "what": r["what"]} for t, r in d.self_tasks.items()
                          if t in s.tasks.tasks]}
    return {"director": director, "router": _family(s, DISPATCH), "routes": routes, "waiting": waiting, "engines": engines,
            "memory": memory, "memory_enabled": mem.enabled,
            "guard": {"checked": sum(a.decisions.values()), "changed": sum(v for k, v in a.decisions.items() if k != "pass"),
                      "held": len(a.held), "withheld": a.decisions.get("block", 0)} if a.enabled else None}


def work(system) -> dict:
    now = time.time()
    return {"active": [{"kind": x.kind, "actor": x.actor, "text": x.text, "elapsed": round(x.elapsed, 2)}
                       for x in system.activity.current()],
            "recent": [{"kind": x.kind, "actor": x.actor, "text": x.text, "took": round(x.elapsed, 2),
                        "ago": round(now - (x.ended or now), 1)} for x in system.activity.recent(20)]}


def log_since(system, after: int = 0, n: int = 500, routine: bool = True) -> dict:
    seq, items = system.log.since(after, n)
    return {"seq": seq, "entries": [{"seq": i, "ts": e.ts, "tag": e.tag, "text": e.text, "level": e.level,
                                     "topic": e.topic, "routine": e.topic in ROUTINE}
                                    for i, e in items if routine or e.topic not in ROUTINE]}


def orders(system, n: int = 30) -> list[dict]:
    return [{"ts": e.ts, **{k: e.data.get(k) for k in ("order", "route", "ok", "served_by", "text", "output",
                                                         "latency_ms")}}
            for e in system.bus.recent("order.done", n)]


def dashboard(system, after: int = 0) -> dict:
    """Everything the live view needs in one request."""
    return {"now": time.time(), "status": system.status(), "mesh": mesh(system), "work": work(system),
            "log": log_since(system, after, 400), "orders": orders(system)}


def memory_node(system, nid: str) -> dict | None:
    mem = system.memory
    n = mem.nodes.get(nid)
    if n is None:
        return None
    return {"id": n.id, "status": "learning" if n.dirty else n.status, "spec": n.spec(), "bytes": n.bytes,
            "first_bytes": n.first_bytes, "passed": n.passed, "total": n.total, "served": n.served,
            "file": str(mem.tests_path(n)), "failing": n.failing,
            "facts": [{"ask": t.ask, "answer": t.answer, "source": t.source, "learned": t.learned, "by": t.by,
                       "variants": [v.ask for v in n.tests if v.variant_of == t.ask]} for t in n.facts()],
            "unknowns": [{"ask": u.ask, "reason": u.reason} for u in n.unknowns],
            "history": [mem._history_line(h) for h in n.history[-30:]]}
