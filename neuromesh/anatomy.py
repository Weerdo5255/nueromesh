"""Anatomy: a closer look at the mesh - how its nodes are wired together, how one neural node is
built, and which of its synapses fire for an input.

    connections  the paths recent orders took: the director that took each one, the router node and
                 the route it chose, the skill node, tool or memory node that served it, the emotional
                 engines on the way out, and the director that reviewed it - each hop with the node,
                 what it decided, how sure it was, and the input it saw
    a node       its layers (input features, hidden ReLU layers, output labels), its weights per layer
                 (how many, how large, how many are near zero), and, for an input:
                   the input features that are active (and the words, word pairs and letter trigrams
                   that hashed onto each), the neurons each hidden layer fired, the output it chose,
                   the synapses that carried the most signal (activation x weight, excitatory or
                   inhibitory), the strongest path from an input to the output it chose, and which
                   words pushed it towards that output or away from it (gradient x input)
    firings      the recent inputs each node fired on (orders, objectives, its own initiative)

Capability nodes (the router, skills, the emotional engines, the agency, coder and planner nodes),
memory nodes and thinking nodes (thinking.py) can be looked at. The weights are the node's as they are now: a node that has trained
since an order may fire a little differently on it today.
"""
from __future__ import annotations

import re
import time
import zlib

import numpy as np

from .encoding import HashingEncoder

_NODE = re.compile(r"node:([\w-]+)")
_MEM = re.compile(r"memory node (mem-[\w-]+)")


# ============================================================================ the node
def find(system, ref: str):
    """("capability", node), ("memory", node), ("thinking", node), or None. A capability name means its serving node."""
    ref = str(ref or "").strip()
    node = system.registry.get(ref)
    if node is not None:
        return "capability", node
    mem = system.memory.nodes.get(ref) if system.memory.enabled else None
    if mem is not None and mem.params is not None:
        return "memory", mem
    bank = getattr(system, "thinking", None)
    th = bank.nodes.get(ref) if bank is not None else None
    if th is not None and th.params is not None:
        return "thinking", th
    if ref in system.capabilities:
        champ = system.registry.champion(ref)
        if champ is not None:
            return "capability", champ
    return None


def _io(system, kind: str, node):
    """The node's weights (numpy), output labels, and how it reads a text: (x, features with buckets)."""
    if kind == "capability":
        params = node.numpy_params()
        labels = list(node.labels)
        dim = node.input_dim

        def read(text: str):
            feats = HashingEncoder.features(text)
            x = HashingEncoder(dim).encode(text)
            return x, [(f, *_bucket(f, dim, dim)) for f in feats]
        return params, labels, dim, read
    from . import memory as mm
    params = [np.asarray(p, dtype=np.float32) for p in node.params]
    if kind == "thinking":  # its outputs are thoughts: shown by their topics
        bank = system.thinking
        labels = [o if i == 0 else f"{o}: {bank.thoughts[o].topic}" if o in bank.thoughts else o
                  for i, o in enumerate(node.outputs)]
        full = bank.dim
    else:
        labels = [str(a) for a in node.answers]
        full = system.memory.dim
    inputs = node.inputs or full

    def read(text: str):
        feats = mm.features(text)
        x = mm.fold(mm.encode(text, full)[None, :], inputs)[0]
        return x, [(f, *_bucket(f, full, inputs)) for f in feats]
    return params, labels, inputs, read


def _bucket(feature: str, dim: int, fold: int) -> tuple[int, int]:
    h = zlib.crc32(feature.encode("utf-8"))
    return (h % dim) % fold, (1 if (h >> 31) & 1 else -1)


def _word_of(feature: str) -> list[str]:
    """The words a hashed feature came from: "w:great" -> great, "n:good" -> not good, "b:a_b" -> a, b."""
    kind, _, body = feature.partition(":")
    if kind == "w":
        return [body]
    if kind == "n":
        return [f"not {body}"]
    if kind == "b":
        return [w for w in body.split("_") if w]
    if kind == "d":
        return [body]
    return []


def _trigram_words(feats: list[str]) -> dict[str, list[str]]:
    words = {f.partition(":")[2] for f in feats if f.startswith(("w:", "n:")) and f.partition(":")[2].isalpha()}
    out: dict[str, list[str]] = {}
    for f in feats:
        kind, _, tri = f.partition(":")
        if kind in ("c", "nc"):
            out[f] = [w for w in words if tri in f"<{w}>"] or [tri]
    return out


def describe_feature(feature: str) -> str:
    kind, _, body = feature.partition(":")
    return {"w": f"'{body}'", "n": f"'{body}' (negated)", "b": f"'{body.replace('_', ' ')}' (pair)",
            "c": f"'{body}' (letters)", "nc": f"'{body}' (letters, negated)", "d": f"'{body}' (number)"}.get(kind, feature)


def _layer_stats(params: list) -> list[dict]:
    out = []
    for i in range(len(params) // 2):
        W, b = params[2 * i], params[2 * i + 1]
        a = np.abs(W)
        out.append({"from": int(W.shape[0]), "to": int(W.shape[1]), "weights": int(W.size), "biases": int(b.size),
                    "mean_abs": round(float(a.mean()), 5), "max_abs": round(float(a.max()), 4),
                    "near_zero": round(float((a < 1e-3).mean()), 4),
                    "excitatory": round(float((W > 0).mean()), 4)})
    return out


def anatomy(system, ref: str, text: str | None = None, *, latest: bool = False, inputs: int = 16, neurons: int = 14,
            outputs: int = 5, synapses: int = 40) -> dict | None:
    """The construction of a node and, for a text, what fires in it (see the module docstring)."""
    found = find(system, ref)
    if found is None:
        return None
    kind, node = found
    params, labels, dim, read = _io(system, kind, node)
    sizes = [int(params[0].shape[0])] + [int(params[2 * i].shape[1]) for i in range(len(params) // 2)]
    if kind == "capability":
        m = node.metrics
        head = {"id": node.id, "kind": "capability", "capability": node.capability, "status": node.status,
                "origin": node.origin, "generation": node.generation, "parent": node.parent_id,
                "arch": node.arch.describe(), "holdout": m.get("holdout_acc"), "serves": m.get("serves", 0),
                "train_steps": node.train_steps, "lr": node.arch.lr}
    elif kind == "thinking":
        head = {"id": node.id, "kind": "thinking", "capability": "thinking", "status": node.status, "origin": "thinking",
                "generation": 0, "parent": None, "arch": node.spec(), "holdout": None, "serves": node.served,
                "train_steps": None, "bits": node.bits, "thoughts": len(node.outputs) - 1, "tests": f"{node.passed}/{node.total}"}
    else:
        head = {"id": node.id, "kind": "memory", "capability": "memory", "status": node.status, "origin": "memory",
                "generation": 0, "parent": ", ".join(node.merged_from) or None, "arch": node.spec(), "holdout": None,
                "serves": node.served, "train_steps": None, "bits": node.bits, "facts": len(node.facts()),
                "tests": f"{node.passed}/{node.total}"}
    out = {**head, "sizes": sizes, "params": int(sum(p.size for p in params)),
           "labels": [_short(lab) for lab in labels], "layers": _layer_stats(params),
           "activation": "ReLU in every hidden layer, softmax over the outputs",
           "encoder": f"signed feature hashing into {dim} inputs: words, word pairs and letter trigrams",
           "firings": [f for f in reversed(getattr(system, "firings", [])) if f["node"] == node.id][:12]}
    if not text and latest and out["firings"]:  # what it fired on last
        text = out["firings"][0]["text"]
        out["text_from"] = "its latest firing"
    if not text:
        out["fire"] = None
        return out
    out["fire"] = fire(params, labels, read, text, inputs=inputs, neurons=neurons, outputs=outputs, synapses=synapses)
    return out


def _short(label: str, n: int = 70) -> str:
    t = " ".join(str(label).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def fire(params: list, labels: list[str], read, text: str, *, inputs: int, neurons: int, outputs: int,
         synapses: int) -> dict:
    """One forward pass, recorded: what each layer did, the synapses that carried the signal, the
    strongest path, and the words that pushed it towards its choice (gradient x input)."""
    x, feats = read(text)
    n_layers = len(params) // 2
    acts = [x.astype(np.float32)]
    h = acts[0]
    for i in range(n_layers):
        z = h @ params[2 * i] + params[2 * i + 1]
        h = np.maximum(z, 0.0) if i < n_layers - 1 else z
        acts.append(h)
    logits = acts[-1]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    chosen = int(probs.argmax())
    # gradient x input: how much each input moved the chosen output's logit (exact for this ReLU pattern)
    g = np.zeros_like(logits)
    g[chosen] = 1.0
    grads = [None] * (n_layers + 1)
    grads[n_layers] = g
    for i in reversed(range(n_layers)):
        g = params[2 * i] @ g
        if i > 0:
            g = g * (acts[i] > 0)
        grads[i] = g
    attr = acts[0] * grads[0]
    # which features landed on which input
    by_input: dict[int, list[tuple[str, int]]] = {}
    for f, b, sign in feats:
        by_input.setdefault(b, []).append((f, sign))
    active = [int(i) for i in np.flatnonzero(acts[0])]
    # the words: each feature takes its share of its input's attribution
    tri = _trigram_words([f for f, _, _ in feats])
    words: dict[str, float] = {}
    for b, fs in by_input.items():
        if acts[0][b] == 0:
            continue
        share = float(attr[b]) / len(fs)
        for f, sign in fs:
            ws = _word_of(f) or tri.get(f, [])
            for w in ws:
                words[w] = words.get(w, 0.0) + share * (1 if sign * np.sign(acts[0][b]) >= 0 else -1) / max(1, len(ws))
    # what to show: the inputs that mattered most, the neurons that fired most, the likeliest outputs
    path = _strongest_path(params, acts, chosen)
    shown: list[list[int]] = []
    shown.append(_top([i for i in active], key=lambda i: abs(attr[i]), n=inputs, must=[path[0]] if path else []))
    for k in range(1, n_layers):
        fired = [int(i) for i in np.flatnonzero(acts[k] > 0)]
        shown.append(_top(fired, key=lambda i, k=k: acts[k][i], n=neurons, must=[path[k]] if path else []))
    shown.append(_top(list(range(len(probs))), key=lambda i: probs[i], n=outputs, must=[chosen]))
    layers = []
    for k, idx in enumerate(shown):
        a = acts[k]
        if k == 0:
            items = [{"i": i, "value": round(float(a[i]), 4), "attr": round(float(attr[i]), 4),
                      "features": [describe_feature(f) for f, _ in by_input.get(i, [])][:4]} for i in idx]
            layers.append({"kind": "input", "size": len(a), "active": len(active), "shown": items})
        elif k < n_layers:
            items = [{"i": i, "value": round(float(a[i]), 4), "grad": round(float(grads[k][i]), 4)} for i in idx]
            layers.append({"kind": "hidden", "size": len(a), "active": int((a > 0).sum()), "shown": items})
        else:
            items = [{"i": i, "label": _short(labels[i]) if i < len(labels) else str(i), "prob": round(float(probs[i]), 4),
                      "logit": round(float(a[i]), 3)} for i in idx]
            layers.append({"kind": "output", "size": len(a), "active": len(a), "shown": items})
    syn = []
    for k in range(n_layers):
        W = params[2 * k]
        a = acts[k]
        cand = []
        for u in shown[k]:
            if a[u] == 0:
                continue
            for v in shown[k + 1]:
                c = float(a[u] * W[u, v])
                cand.append((abs(c), u, v, c, float(W[u, v])))
        cand.sort(reverse=True)
        on_path = {(path[k], path[k + 1])} if path else set()
        keep = [c for c in cand[:synapses]] + [c for c in cand[synapses:] if (c[1], c[2]) in on_path]
        for _, u, v, c, w in keep:
            syn.append({"layer": k, "from": u, "to": v, "weight": round(w, 4), "signal": round(c, 5),
                        "path": (u, v) in on_path})
    total = float(np.abs(attr).sum()) or 1.0
    pushes = sorted(({"word": w, "push": round(v / total, 4)} for w, v in words.items() if abs(v) > 1e-6),
                    key=lambda d: -abs(d["push"]))
    return {"text": text, "chosen": {"i": chosen, "label": _short(labels[chosen]) if chosen < len(labels) else str(chosen),
                                     "prob": round(float(probs[chosen]), 4)},
            "layers": layers, "synapses": syn, "path": path,
            "path_signal": [round(float(acts[k][path[k]] * params[2 * k][path[k], path[k + 1]]), 5) for k in range(n_layers)] if path else [],
            "words": pushes[:14], "features": len(feats)}


def _top(items: list[int], key, n: int, must: list[int]) -> list[int]:
    ranked = sorted(items, key=lambda i: -abs(float(key(i))))[:n]
    for m in must:
        if m not in ranked:
            ranked = ranked[: max(0, n - 1)] + [m]
    return sorted(ranked, key=lambda i: -float(key(i)))


def _strongest_path(params: list, acts: list, chosen: int) -> list[int]:
    """From the output it chose back to an input: at each layer, the neuron whose signal into the next
    one on the path was largest (activation x weight)."""
    n_layers = len(params) // 2
    path = [chosen]
    for k in reversed(range(n_layers)):
        a = acts[k]
        if not np.any(a):
            return []
        contrib = a * params[2 * k][:, path[0]]
        path.insert(0, int(np.argmax(contrib)))
    return path


# ======================================================================== the wiring
def traces(system, n: int = 15) -> list[dict]:
    """The paths recent orders took through the mesh, hop by hop."""
    out = []
    for oid, ts, text, res in system.store.recent_orders(n):
        hops = []
        trace = res.get("trace") or []
        took = next((m.group(1) for t in trace for m in [re.match(r"^(director-\d+) took", str(t))] if m), None)
        hops.append({"stage": "director", "node": took or "director-0", "label": "took it", "kind": "director"})
        steps = res.get("steps") or []
        route = res.get("route") or ""
        served = False
        for st in steps:
            cap = st.get("capability", "")
            m = _NODE.search(str(st.get("provenance", "")))
            hop = {"stage": {"dispatch": "router"}.get(cap, cap), "capability": cap, "node": m.group(1) if m else None,
                   "label": st.get("label"), "confidence": st.get("confidence"), "how": st.get("provenance", ""),
                   "input": st.get("input") or text, "kind": "node" if m else "teacher"}
            if cap in ("affect", "disposition") and not served:
                hops.extend(_served(res, route, text))
                served = True
            hops.append(hop)
        if not served:
            hops.extend(_served(res, route, text))
        if res.get("director"):
            hops.append({"stage": "director", "node": res["director"], "label": res.get("review", "")[:80] or "reviewed it",
                         "kind": "director"})
        out.append({"order": oid, "ts": ts, "text": text[:200], "route": route, "ok": bool(res.get("ok")),
                    "answer": _short(str(res.get("output") or res.get("note") or "").split("\n")[0], 200),
                    "confidence": res.get("confidence"), "served_by": res.get("served_by"),
                    "latency_ms": res.get("latency_ms"), "hops": hops})
    return out


def _served(res: dict, route: str, text: str) -> list[dict]:
    """What served the order after the router (a tool, a memory node, retained knowledge, a teacher), and the
    thinking node that added its thought after the knowledge."""
    hops = _served_by(res, route, text)
    th = res.get("thought")
    if th and th.get("node"):
        hops.append({"stage": "thinking", "node": th["node"], "label": _short(th.get("topic", ""), 60), "kind": "thinking",
                     "confidence": th.get("confidence"), "input": text})
    return hops


def _served_by(res: dict, route: str, text: str) -> list[dict]:
    sources = " ".join(map(str, res.get("sources") or []))
    m = _MEM.search(sources)
    if m:
        return [{"stage": "memory", "node": m.group(1), "label": _short(res.get("output") or "", 60), "kind": "memory",
                 "input": text}]
    if route.startswith("tool:"):
        return [{"stage": "tool", "node": None, "tool": route[5:], "label": "ok" if res.get("ok") else "failed", "kind": "tool"}]
    if route.startswith("skill:"):
        return []  # the skill node's own step is in the steps
    if res.get("knowledge") in ("recalled", "researched"):
        return [{"stage": "knowledge", "node": None, "label": res.get("knowledge"), "kind": "knowledge"}]
    if res.get("served_by") == "teachers" and route == "answer:direct":
        return [{"stage": "teacher", "node": None, "label": "answered" if res.get("ok") else "could not", "kind": "teacher"}]
    return []


def record(system, node_id: str, capability: str, text: str, label, confidence: float, used: bool | None = None) -> dict:
    """Note that a node fired (System.infer and memory recall call this)."""
    f = {"ts": round(time.time(), 2), "node": node_id, "capability": capability, "text": str(text)[:300],
         "label": _short(label or "", 60), "confidence": round(float(confidence), 4), "used": used}
    system.firings.append(f)
    return f


# ======================================================================= for consoles
def lines(system, ref: str, text: str | None = None) -> list[str]:
    a = anatomy(system, ref, text)
    if a is None:
        return [f"  no node '{ref}' (a node id like dispatch-ab12cd, a memory node like mem-0001, or a skill name)"]
    shape = " -> ".join(str(s) for s in a["sizes"])
    out = [f"  {a['id']} ({a['capability']}, {a['status']}): {shape} neurons, {a['params']:,} weights; {a['arch']}",
           f"  {a['encoder']}; {a['activation']}"]
    for k, L in enumerate(a["layers"]):
        out.append(f"    layer {k + 1}: {L['from']} -> {L['to']}, {L['weights']:,} synapses, mean |w| {L['mean_abs']:.4f}, "
                   f"max {L['max_abs']:.3f}, {L['near_zero']:.0%} near zero, {L['excitatory']:.0%} excitatory")
    f = a["fire"]
    if f is None:
        if a["firings"]:
            out.append("  recent firings: " + "; ".join(f"'{x['text'][:40]}' -> {x['label']}" for x in a["firings"][:4]))
        out.append("  /anatomy <node> :: <text> shows what fires for an input")
        return out
    out.append(f"  fired on '{_short(f['text'], 80)}': {f['features']} features -> chose {f['chosen']['label']} "
               f"({f['chosen']['prob']:.0%})")
    for L in f["layers"]:
        if L["kind"] == "input":
            out.append(f"    input: {L['active']} of {L['size']} active; most telling: "
                       + ", ".join(f"#{x['i']} {'/'.join(x['features'][:2]) or '?'} ({x['attr']:+.3f})" for x in L["shown"][:6]))
        elif L["kind"] == "hidden":
            out.append(f"    hidden: {L['active']} of {L['size']} fired; strongest " + ", ".join(f"h{x['i']}={x['value']:.2f}"
                                                                                       for x in L["shown"][:6]))
        else:
            out.append("    output: " + ", ".join(f"{x['label']} {x['prob']:.0%}" for x in L["shown"]))
    if f["path"]:
        names = [f"#{f['path'][0]}"] + [f"h{i}" for i in f["path"][1:-1]] + [f["chosen"]["label"]]
        out.append("    strongest path: " + " -> ".join(names) + "  (signal " + ", ".join(f"{s:+.3f}" for s in f["path_signal"]) + ")")
    if f["words"]:
        out.append("    words that pushed it: " + ", ".join(f"{w['word']} {w['push']:+.0%}" for w in f["words"][:8]))
    return out
