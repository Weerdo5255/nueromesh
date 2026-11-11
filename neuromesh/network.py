"""The whole network, as the web page draws it (GET /network; ?detail=1 for every node's neurons and synapses).

    families   every capability's nodes - the router, each skill, and the inner capabilities (its ego, agency, tool
               gate, realm, keep, reward, sight, glyphs, code, disposition, affect) - and its memory, thinking and dream
               nodes: each node's layers, weights, status, scores, and whether it is being trained, evaluated or used now
    stations   where each family sits in the way an order travels: orders come in, a director takes them, the router
               sends them down a route (a skill, a tool, or an answer from memory and teachers), and answers go out
               through the emotional engines and its ego; the inner capabilities attach where they work
    links      router -> each route; each family -> its station; the trainer, evaluator, teachers and the other
               background agents -> the nodes they are working on right now (from the activity tracker)
    detail     per node and per pair of layers: the neurons with the strongest weights (inputs, hidden, outputs) and
               the strongest synapses between them, signed - the network as it is, not a sketch of it

Big meshes are big: the page sizes each node by its weights and each family by all of them together.
"""
from __future__ import annotations

import re
import time

import numpy as np

STATION = {"agency": "directors", "relay": "answers", "ego": "ego", "toolgate": "tools", "reward": "answers", "vision": "images",
           "codekind": "code", "planner": "code", "coder": "code", "realm": "memory", "keep": "memory"}
STATION_BY_NAME = {"affect": "answers", "disposition": "answers", "sight": "images", "glyphs": "images"}
WHAT = {"orders": "orders come in", "directors": "a director takes each order, and reviews its answer",
        "router": "the router node chooses a route", "routes": "the skills it routes to", "tools": "tools in the sandbox", "memory": "memory nodes, asked first",
        "teachers": "teachers answer what nothing else can", "answers": "answers go out", "images": "what it sees",
        "code": "code it writes", "ego": "its ego, around everything"}
_ID = re.compile(r"\b([a-z][a-z0-9_]*-[0-9a-f]{6})\b")
_MEM = re.compile(r"\b(mem-\d{4,})\b")
_THINK = re.compile(r"\b(think-\d{4,})\b")
_CACHE: dict[tuple, dict] = {}


def _status_rank(status: str) -> int:
    return {"champion": 0, "instance": 1, "candidate": 2, "retired": 3}.get(status, 4)


def _sizes(node) -> list[int]:
    try:
        return [int(node.params[0].shape[0])] + [int(node.params[2 * i].shape[1]) for i in range(len(node.params) // 2)]
    except Exception:
        return [int(getattr(node, "input_dim", 0))] + [int(x) for x in getattr(node.arch, "hidden", ())] + [len(node.labels)]


def _busy(system) -> dict:
    """What the background is doing to which node now (and in the last few seconds): node -> [{kind, actor, text, now}]."""
    out: dict[str, list] = {}
    fams: dict[str, list] = {}
    caps = set(system.capabilities)
    items = [(a, True) for a in system.activity.current()] + [(a, False) for a in system.activity.recent(6.0)]
    for a, now in items:
        rec = {"kind": a.kind, "actor": a.actor, "text": a.text[:140], "now": now}
        ids = set(_ID.findall(a.text)) | set(_MEM.findall(a.text)) | set(_THINK.findall(a.text))
        for nid in ids:
            out.setdefault(nid, []).append(rec)
        head = a.text.split(":", 1)[0].strip()
        if head in caps and not ids:  # "ego: 4 node(s) on 33 held-out examples": the whole family
            fams.setdefault(head, []).append(rec)
    return {"nodes": out, "families": fams}


def _layers(params: list, labels: list[str], top_neurons: int, top_synapses: int) -> list[dict]:
    """Per pair of layers: the neurons it shows (the strongest), and the strongest synapses between them."""
    n = len(params) // 2
    Ws = [np.asarray(params[2 * i], dtype=np.float32) for i in range(n)]
    sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
    # how much each neuron matters: the weight going out of it and coming into it
    strength = []
    for k, size in enumerate(sizes):
        s = np.zeros(size, dtype=np.float32)
        if k < n:
            s += np.abs(Ws[k]).sum(axis=1)
        if k > 0:
            s += np.abs(Ws[k - 1]).sum(axis=0)
        strength.append(s)
    shown = []
    for k, size in enumerate(sizes):
        if size <= top_neurons or k == len(sizes) - 1 and size <= 2 * top_neurons:
            idx = np.arange(size)
        else:
            idx = np.sort(np.argsort(-strength[k])[:top_neurons])
        shown.append(idx)
    out = []
    for k in range(len(sizes)):
        layer = {"size": int(sizes[k]), "kind": "input" if k == 0 else "output" if k == len(sizes) - 1 else "hidden",
                 "shown": [int(i) for i in shown[k]],
                 "strength": [round(float(strength[k][i]), 4) for i in shown[k]]}
        if k == len(sizes) - 1:
            layer["labels"] = [str(labels[i])[:40] if i < len(labels) else str(i) for i in shown[k]]
        if k < n:
            sub = Ws[k][np.ix_(shown[k], shown[k + 1])]
            flat = np.argsort(-np.abs(sub), axis=None)[:top_synapses]
            rows, cols = np.unravel_index(flat, sub.shape)
            layer["synapses"] = [[int(r), int(c), round(float(sub[r, c]), 4)] for r, c in zip(rows, cols)]
            a = np.abs(Ws[k])
            layer["weights"] = {"count": int(Ws[k].size), "mean_abs": round(float(a.mean()), 5), "max_abs": round(float(a.max()), 4),
                                "excitatory": round(float((Ws[k] > 0).mean()), 3)}
        out.append(layer)
    return out


def _detail(kind: str, node, system, top_neurons: int, top_synapses: int) -> list[dict] | None:
    key = (node.id, getattr(node, "train_steps", None), getattr(node, "rev", None), getattr(node, "bits", None),
           len(getattr(node, "tests", []) or []), top_neurons, top_synapses)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit["layers"]
    try:
        from .anatomy import _io
        params, labels, _, _ = _io(system, kind, node)
    except Exception:
        return None
    layers = _layers(params, labels, top_neurons, top_synapses)
    _CACHE[key] = {"layers": layers, "at": time.time()}
    while len(_CACHE) > 400:
        _CACHE.pop(next(iter(_CACHE)))
    return layers


def fill(system, todo: list, top_neurons: int = 16, top_synapses: int = 48) -> None:
    """Every node's neurons and strongest synapses (the heavy part: the server runs it off the event loop)."""
    for kind, node, item in todo:
        item["layers"] = _detail(kind, node, system, top_neurons, top_synapses)


def network(system, detail: bool = False, todo: list | None = None) -> dict:
    """The network (see the module docstring). With detail, the nodes whose neurons are to be filled in are added to
    todo: fill(system, todo) then adds them (network() itself only reads what is cheap)."""
    s = system
    todo = todo if todo is not None else []
    busy = _busy(s)
    nb, fb = busy["nodes"], busy["families"]
    routed = {r[6:] for r in s.capabilities["dispatch"].labels if r.startswith("skill:")} if "dispatch" in s.capabilities else set()
    families = []
    by_cap: dict[str, list] = {}
    for n in s.registry.live():
        by_cap.setdefault(n.capability, []).append(n)
    for cap, spec in s.capabilities.items():
        nodes = sorted(by_cap.get(cap, []), key=lambda n: (_status_rank(n.status), -n.param_count))
        station = ("router" if cap == "dispatch" else "routes" if cap in routed else
                   STATION_BY_NAME.get(cap) or STATION.get(spec.kind or "", "answers" if spec.internal else "routes"))
        fam = {"id": cap, "name": "the router" if cap == "dispatch" else cap, "kind": "router" if cap == "dispatch" else
               "skill" if cap in routed else "inner" if spec.internal else "skill",
               "station": station, "routed": cap in routed, "labels": list(spec.labels)[:40],
               "description": (spec.description or "")[:240], "examples": s.store.count(cap),
               "busy": fb.get(cap, []), "nodes": []}
        for n in nodes:
            m = n.metrics
            item = {"id": n.id, "status": n.status, "params": n.param_count, "sizes": _sizes(n), "holdout": m.get("holdout_acc"),
                    "serves": m.get("serves", 0), "steps": n.train_steps, "generation": n.generation, "origin": n.origin,
                    "parent": n.parent_id, "lr": n.arch.lr, "busy": nb.get(n.id, [])}
            if detail and n.status != "retired":
                todo.append(("capability", n, item))
            fam["nodes"].append(item)
        fam["params"] = sum(x["params"] for x in fam["nodes"] if x["status"] != "retired")
        families.append(fam)
    mem = s.memory
    if mem.enabled:
        fam = {"id": "memory", "name": "memory", "kind": "memory", "station": "memory", "routed": False, "labels": [],
               "description": "facts compressed into small tested nodes, asked before anything else", "examples": 0,
               "busy": [], "nodes": []}
        for n in sorted(mem.fact_nodes(), key=lambda n: n.created):
            params = int(sum(np.asarray(p).size for p in n.params)) if n.params is not None else 0
            item = {"id": n.id, "status": "learning" if n.dirty else n.status, "params": params,
                    "sizes": [int(np.asarray(n.params[0]).shape[0])] + [int(np.asarray(n.params[2 * i]).shape[1])
                                                                      for i in range(len(n.params) // 2)] if n.params is not None else [],
                    "facts": len(n.facts()), "bytes": n.bytes, "tests": f"{n.passed}/{n.total}", "serves": n.served,
                    "busy": nb.get(n.id, [])}
            if detail and n.params is not None:
                todo.append(("memory", n, item))
            fam["nodes"].append(item)
        fam["params"] = sum(x["params"] for x in fam["nodes"])
        families.append(fam)
    th = getattr(s, "thinking", None)
    if th is not None and th.enabled:
        fam = {"id": "thinking", "name": "thinking", "kind": "thinking", "station": "answers", "routed": False, "labels": [],
               "description": "its own thoughts, beside an answer and never the answer", "examples": 0, "busy": [], "nodes": []}
        for n in th.nodes.values():
            params = int(sum(np.asarray(p).size for p in n.params)) if n.params is not None else 0
            item = {"id": n.id, "status": n.status, "params": params, "thoughts": len(n.outputs) - 1, "serves": n.served,
                    "sizes": [int(np.asarray(n.params[0]).shape[0])] + [int(np.asarray(n.params[2 * i]).shape[1])
                                                                      for i in range(len(n.params) // 2)] if n.params is not None else [],
                    "busy": nb.get(n.id, [])}
            if detail and n.params is not None:
                todo.append(("thinking", n, item))
            fam["nodes"].append(item)
        fam["params"] = sum(x["params"] for x in fam["nodes"])
        families.append(fam)
    dr = s.dreams
    if dr.enabled and dr.nodes:
        champ = dr.champion()
        families.append({"id": "dreams", "name": "dreams", "kind": "dreams", "station": "ego", "routed": False, "labels": [],
                         "description": "what it dreams with while it sleeps: never facts", "examples": 0,
                         "busy": [{"kind": "dream", "actor": "dreamer", "text": "dreaming", "now": True}] if dr.asleep else [],
                         "nodes": [{"id": n.id, "status": "champion" if champ is not None and n.id == champ.id else "dream",
                                    "params": int(n.weights or 0), "shape": n.shape(), "fitness": n.fitness,
                                    "generation": n.generation, "busy": nb.get(n.id, [])}
                                   for n in sorted(dr.nodes.values(), key=lambda n: -(n.fitness or 0))],
                         "params": sum(int(n.weights or 0) for n in dr.nodes.values())})
    tools = [{"id": t.name, "name": t.name, "state": t.state, "runs": t.runs, "version": t.manifest.version,
              "route": t.manifest.route, "provides": t.manifest.provides,
              "busy": [{"kind": a.kind, "actor": a.actor, "text": a.text[:140], "now": True}
                       for a in s.activity.current() if t.name in a.text.split()[:4] or a.text.startswith(t.name)]}
             for t in s.tools.tools.values() if t.state != "removed"]
    directors = [{"id": n.id, "role": n.role, "busy": n.state != "idle" or bool(n.initiative), "doing": n.doing()}
                 for n in s.directors.nodes.values()]
    teachers = [{"id": t.name, "llm": bool(getattr(t, "provider", None)), "specialist": bool(getattr(t, "specialist", False))}
                for t in s.teachers.ranked()]
    # who is working on what, right now: background agents to the nodes and families they touch
    work = []
    for a, now in [(a, True) for a in s.activity.current()] + [(a, False) for a in s.activity.recent(4.0)][:12]:
        ids = sorted(set(_ID.findall(a.text)) | set(_MEM.findall(a.text)) | set(_THINK.findall(a.text)))
        head = a.text.split(":", 1)[0].strip()
        fam = head if head in s.capabilities else ""
        work.append({"kind": a.kind, "actor": a.actor, "text": a.text[:160], "elapsed": round(time.time() - a.started, 1),
                     "nodes": ids, "family": fam, "now": now})
    tasks = [{"id": t.id, "what": t.describe()[:120], "state": getattr(t, "state", ""), "next": getattr(t, "next_run", 0)}
             for t in list(s.tasks.tasks.values())[:20]] if hasattr(s, "tasks") else []
    total = sum(f.get("params", 0) for f in families)
    return {"at": time.time(), "families": families, "tools": tools, "directors": directors, "teachers": teachers,
            "work": work, "tasks": tasks, "stations": WHAT, "params": total,
            "nodes": sum(len(f["nodes"]) for f in families), "ego": bool(s.ego.enabled), "asleep": bool(s.guardian.sleeping),
            "detail": detail}


def full(system) -> dict:
    """network() with every node's neurons, in one go (for tests and the console)."""
    todo: list = []
    out = network(system, True, todo)
    fill(system, todo)
    return out
