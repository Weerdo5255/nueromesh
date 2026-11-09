"""Evolution: a genetic algorithm over each capability's nodes.

    genome      the architecture: hidden layers (how many, and how many neurons in each) and the
                learning rate
    population  a capability's live nodes (champion, challengers, recently retired) plus an archive
                of genomes it has tried and how they scored
    selection   a tournament on fitness = held-out accuracy + a small bonus per doubling of size, so
                between equally accurate genomes the larger one is preferred
    variation   mutations that grow or shrink - neurons in a layer, or whole layers - biased towards
                growth (growth_bias), sometimes crossed over with another good genome
    inheritance a child starts from its parent's weights, morphed to the new shape:
                  grow neurons    Net2Wider: duplicate neurons and split their outgoing weights (exact)
                  add a layer     Net2Deeper: an identity layer after a ReLU layer (exact)
                  shrink neurons  keep the most important neurons (|in| x |out| weight norms)
                  remove a layer  fold it into the next layer (W_i @ W_i+1; exact where the ReLU is idle)
                then it is trained like any challenger
    replacement the Evaluator promotes a challenger that beats the champion; a larger one that is at
                least as accurate also wins (the trend towards larger, more complex nodes); a smaller
                one has to be clearly better. P5 still holds: nothing that scores lower is promoted.

Bounds: [nodes] max_hidden neurons per layer, max_depth layers, [evolution] min_neurons, max_params.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np


# =========================================================================== operators
def hidden_of(params: list[np.ndarray]) -> tuple[int, ...]:
    """Hidden layer widths of a [W0, b0, W1, b1, ..., Wout, bout] parameter list."""
    return tuple(int(params[2 * i].shape[1]) for i in range(len(params) // 2 - 1))


def widen_layer(params: list, i: int, k: int, rng: np.random.Generator, noise: float = 1e-3) -> list:
    """Add k neurons to hidden layer i by duplicating existing ones and splitting their outgoing
    weights (Net2Wider), so the network computes exactly the same function (up to `noise`)."""
    p = [x.copy() for x in params]
    W_in, b, W_out = p[2 * i], p[2 * i + 1], p[2 * i + 2]
    h = W_in.shape[1]
    idx = rng.integers(0, h, size=k)
    counts = np.bincount(idx, minlength=h).astype(np.float32) + 1.0
    jitter = rng.normal(0.0, noise, (W_in.shape[0], k)).astype(np.float32) if noise else 0.0
    p[2 * i] = np.concatenate([W_in, W_in[:, idx] + jitter], axis=1).astype(np.float32)
    p[2 * i + 1] = np.concatenate([b, b[idx]]).astype(np.float32)
    scaled = W_out / counts[:, None]
    p[2 * i + 2] = np.concatenate([scaled, scaled[idx]], axis=0).astype(np.float32)
    return p


def importance(params: list, i: int) -> np.ndarray:
    """How much each neuron of hidden layer i matters: |incoming| x |outgoing| weight norms."""
    return np.linalg.norm(params[2 * i], axis=0) * np.linalg.norm(params[2 * i + 2], axis=1) + 1e-9


def prune_layer(params: list, i: int, k: int) -> list:
    """Remove the k least important neurons of hidden layer i."""
    p = [x.copy() for x in params]
    h = p[2 * i].shape[1]
    keep = np.sort(np.argsort(importance(p, i))[k:]) if k < h else np.array([int(np.argmax(importance(p, i)))])
    p[2 * i] = p[2 * i][:, keep]
    p[2 * i + 1] = p[2 * i + 1][keep]
    p[2 * i + 2] = p[2 * i + 2][keep, :]
    return p


def insert_layer(params: list, i: int, rng: np.random.Generator, noise: float = 1e-3) -> list:
    """Insert a new hidden layer at position i (1 <= i <= depth): an identity map of the width of
    the layer before it. After a ReLU layer this is exact (Net2Deeper)."""
    depth = len(params) // 2 - 1
    if not 1 <= i <= depth:
        raise ValueError("a layer can be inserted after an existing hidden layer only")
    h = params[2 * (i - 1)].shape[1]
    eye = np.eye(h, dtype=np.float32)
    if noise:
        eye += rng.normal(0.0, noise, (h, h)).astype(np.float32)
    return [x.copy() for x in params[: 2 * i]] + [eye, np.zeros(h, dtype=np.float32)] + \
        [x.copy() for x in params[2 * i:]]


def remove_layer(params: list, i: int) -> list:
    """Remove hidden layer i by folding it into the next layer: W = W_i @ W_i+1,
    b = b_i @ W_i+1 + b_i+1 (exact wherever layer i's ReLU was not cutting anything off)."""
    depth = len(params) // 2 - 1
    if not 0 <= i < depth:
        raise ValueError("no such hidden layer")
    W_i, b_i, W_n, b_n = params[2 * i], params[2 * i + 1], params[2 * i + 2], params[2 * i + 3]
    W = (W_i @ W_n).astype(np.float32)
    b = (b_i @ W_n + b_n).astype(np.float32)
    return [x.copy() for x in params[: 2 * i]] + [W, b] + [x.copy() for x in params[2 * i + 4:]]


def morph(params: list, target: tuple[int, ...], rng: np.random.Generator, noise: float = 1e-3) -> list:
    """Reshape a parent's weights to any genome: fold away or insert layers at the end, then widen
    or prune each layer to its target width."""
    p = [x.copy() for x in params]
    while len(hidden_of(p)) > max(1, len(target)):
        p = remove_layer(p, len(hidden_of(p)) - 1)
    while len(hidden_of(p)) < len(target):
        p = insert_layer(p, len(hidden_of(p)), rng, noise)
    for i, want in enumerate(target):
        have = hidden_of(p)[i]
        if want > have:
            p = widen_layer(p, i, want - have, rng, noise)
        elif want < have:
            p = prune_layer(p, i, have - want)
    return p


def param_count(input_dim: int, hidden: tuple[int, ...], outputs: int) -> int:
    sizes = [input_dim, *hidden, outputs]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


# ============================================================================ genomes
@dataclass
class Genome:
    hidden: tuple
    lr: float
    changes: list = field(default_factory=list)   # what was done to get here, in words
    parents: list = field(default_factory=list)


def mutate(hidden: tuple, lr: float, rng: np.random.Generator, pol, *, max_hidden: int, max_depth: int,
           input_dim: int, outputs: int) -> Genome:
    """One or two mutations, growing with probability pol.growth_bias, within the bounds."""
    h = list(hidden) or [pol.min_neurons]
    changes: list[str] = []
    steps = 1 + int(rng.random() < pol.double_mutation)
    for _ in range(steps):
        grow = rng.random() < pol.growth_bias
        layer = rng.random() < pol.layer_rate
        for attempt in range(2):  # if the chosen change is out of bounds, try the opposite direction
            done = _apply(h, grow, layer, rng, pol, max_hidden, max_depth, input_dim, outputs, changes)
            if done:
                break
            grow = not grow if attempt == 0 else grow
    if rng.random() < pol.lr_rate or not changes:
        factor = float(rng.choice([0.5, 0.7, 1.4, 2.0]))
        new_lr = float(np.clip(lr * factor, 1e-4, 0.1))
        if new_lr != lr:
            changes.append(f"learning rate {lr:g} -> {new_lr:g}")
            lr = new_lr
    return Genome(tuple(h), lr, changes)


def _apply(h: list, grow: bool, layer: bool, rng, pol, max_hidden, max_depth, input_dim, outputs,
           changes: list) -> bool:
    if layer and grow:
        if len(h) >= max_depth:
            return False
        at = int(rng.integers(1, len(h) + 1))           # after hidden layer at-1
        width = h[at - 1]
        trial = h[:at] + [width] + h[at:]
        if param_count(input_dim, tuple(trial), outputs) > pol.max_params:
            return False
        h[:] = trial
        changes.append(f"add layer {at + 1} ({width} neurons)")
        return True
    if layer and not grow:
        if len(h) <= 1:
            return False
        at = int(rng.integers(0, len(h)))
        removed = h.pop(at)
        changes.append(f"remove layer {at + 1} ({removed} neurons)")
        return True
    i = int(rng.integers(0, len(h)))
    if grow:
        if h[i] >= max_hidden:
            i = min(range(len(h)), key=lambda j: h[j])   # grow the narrowest layer instead
            if h[i] >= max_hidden:
                return False
        k = max(1, int(round(h[i] * rng.uniform(0.25, 0.75))))
        new = min(max_hidden, h[i] + k)
        trial = h[:i] + [new] + h[i + 1:]
        if param_count(input_dim, tuple(trial), outputs) > pol.max_params:
            return False
        changes.append(f"layer {i + 1}: {h[i]} -> {new} neurons")
        h[i] = new
        return True
    if h[i] <= pol.min_neurons:
        return False
    new = max(pol.min_neurons, h[i] - max(1, int(round(h[i] * rng.uniform(0.2, 0.5)))))
    changes.append(f"layer {i + 1}: {h[i]} -> {new} neurons")
    h[i] = new
    return True


def crossover(a: tuple, b: tuple, rng: np.random.Generator, growth_bias: float) -> tuple:
    """Layer-by-layer recombination of two genomes; the child is as deep as one of them (the
    deeper one with probability growth_bias)."""
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    depth = len(longer) if rng.random() < growth_bias else len(shorter)
    out = []
    for i in range(depth):
        options = [g[i] for g in (a, b) if i < len(g)]
        out.append(int(options[int(rng.integers(0, len(options)))]))
    return tuple(out)


def fitness(acc: float | None, params: int, base_params: int, weight: float) -> float | None:
    """Held-out accuracy plus `weight` per doubling of size over the default node."""
    if acc is None:
        return None
    return acc + weight * math.log2(max(params, 1) / max(base_params, 1))


def judge(champ_ok: np.ndarray, cand_ok: np.ndarray, champ_params: int, cand_params: int, *, margin: float,
          z_needed: float, larger_on_tie: bool) -> tuple[bool, str]:
    """Should a challenger replace the champion? Better wins; larger and at least as accurate wins
    (the trend towards complexity); smaller has to be clearly better."""
    wins, losses = int((cand_ok & ~champ_ok).sum()), int((~cand_ok & champ_ok).sum())
    z = (wins - losses) / math.sqrt(wins + losses) if wins + losses else 0.0
    diff = float(cand_ok.mean()) - float(champ_ok.mean())
    if diff >= margin and z >= z_needed:
        return True, "better"
    if larger_on_tie and cand_params > champ_params and diff >= 0 and wins >= losses:
        return True, "as accurate and larger"
    return False, "smaller and not clearly better" if cand_params < champ_params else "not better"


# ============================================================================== engine
class Evolution:
    """Breeds challengers for the Architect and keeps each capability's genome archive."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.evolution
        self.rng = np.random.default_rng(system.rng.randrange(1 << 30))
        self.archive: dict[str, list[dict]] = {}
        raw = system.store.get_kv("evolution.archive")
        if raw:
            try:
                self.archive = json.loads(raw)
            except ValueError:
                self.archive = {}
        self._dirty = False

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def base_params(self, outputs: int) -> int:
        npol = self.sys.directives.nodes
        return param_count(npol.input_dim, tuple(npol.default_hidden), outputs)

    def node_fitness(self, node) -> float | None:
        bonus = 0.02 * math.tanh(float(node.metrics.get("reward", 0.0) or 0.0) / 3.0)  # people's rewards (rewards.py)
        f = self._fitness(node)
        return None if f is None else f + bonus

    def _fitness(self, node) -> float | None:
        return fitness(node.metrics.get("holdout_acc"), node.param_count, self.base_params(len(node.labels)),
                       self.pol.complexity_weight)

    # ---------------------------------------------------------------- archive
    def record(self, node, outcome: str = "") -> None:
        if not self.enabled:
            return
        entries = self.archive.setdefault(node.capability, [])
        entry = next((e for e in entries if e["node"] == node.id), None)
        if entry is None:
            entry = {"node": node.id, "hidden": list(node.arch.hidden), "lr": node.arch.lr,
                     "generation": node.generation, "origin": node.origin, "created": round(node.created_at, 1),
                     "changes": node.metrics.get("evolution", {}).get("changes", [])}
            entries.append(entry)
        entry.update(params=node.param_count, acc=node.metrics.get("holdout_acc"), fitness=self.node_fitness(node))
        if outcome:
            entry["outcome"] = outcome
        keep = self.pol.archive_size
        if len(entries) > keep:  # keep the fittest half and the most recent half
            by_fit = sorted(entries, key=lambda e: e.get("fitness") or 0.0, reverse=True)[: keep // 2]
            recent = entries[-(keep - len(by_fit)):]
            self.archive[node.capability] = [e for e in entries if e in by_fit or e in recent]
        self._dirty = True

    def on_event(self, ev) -> None:
        t, d = ev.topic, ev.data
        if t in ("node.promoted", "node.disposed", "node.retired"):
            node = self.sys.registry.get(d["node"])
            outcome = {"node.promoted": "champion", "node.disposed": "disposed", "node.retired": "retired"}[t]
            if node is not None:
                self.record(node, outcome)
            else:
                for entry in self.archive.get(d.get("capability", ""), []):
                    if entry["node"] == d["node"]:
                        entry["outcome"] = outcome
                        self._dirty = True
            if t == "node.promoted":
                self.save()

    def save(self) -> None:
        if self._dirty:
            self.sys.store.set_kv("evolution.archive", json.dumps(self.archive))
            self._dirty = False

    # ----------------------------------------------------------------- breeding
    def parents(self, capability: str) -> list:
        """Live nodes with weights, fittest first (the champion is always eligible)."""
        reg = self.sys.registry
        champ = reg.champion(capability)
        need = set(champ.labels) if champ else set()  # a parent must cover every label the champion has
        pool = [n for n in [champ, *reg.candidates(capability), *reg.retired(capability)]
                if n is not None and set(n.labels) >= need]
        return sorted(pool, key=lambda n: self.node_fitness(n) or -1.0, reverse=True)

    def select(self, capability: str, champ):
        """Tournament selection among the live population (the champion when it is alone)."""
        pool = [n for n in self.parents(capability) if self.node_fitness(n) is not None] or [champ]
        k = min(len(pool), max(1, self.pol.tournament))
        picks = [pool[int(i)] for i in self.rng.choice(len(pool), size=k, replace=False)]
        return max(picks, key=lambda n: self.node_fitness(n) or -1.0)

    def breed(self, champ, spec):
        """A challenger for `champ`'s capability: select a parent, maybe cross it with another good
        genome from the archive, mutate (biased towards growth) and morph the parent's weights."""
        from .nodes import NeuralNode  # noqa: F401  (type of the child)
        npol = self.sys.directives.nodes
        parent = self.select(champ.capability, champ)
        hidden, lr, parents = tuple(parent.arch.hidden), parent.arch.lr, [parent.id]
        changes: list[str] = []
        partners = [e for e in self.archive.get(champ.capability, [])
                    if e["node"] != parent.id and e.get("fitness") is not None and tuple(e["hidden"]) != hidden]
        if partners and self.rng.random() < self.pol.crossover_rate:
            mate = max(self.rng.choice(len(partners), size=min(2, len(partners)), replace=False),
                       key=lambda i: partners[int(i)]["fitness"])
            other = partners[int(mate)]
            mixed = crossover(hidden, tuple(other["hidden"]), self.rng, self.pol.growth_bias)
            mixed = tuple(min(npol.max_hidden, max(self.pol.min_neurons, w)) for w in mixed[: npol.max_depth])
            if mixed != hidden:
                changes.append(f"crossover {'x'.join(map(str, hidden))} with "
                               f"{'x'.join(map(str, other['hidden']))} ({other['node']})")
                hidden = mixed
                parents.append(other["node"])
        g = mutate(hidden, lr, self.rng, self.pol, max_hidden=npol.max_hidden, max_depth=npol.max_depth,
                   input_dim=npol.input_dim, outputs=len(parent.labels))
        params = morph(parent.numpy_params(), g.hidden, self.rng)
        arch = type(parent.arch)(hidden=g.hidden, lr=g.lr, weight_decay=parent.arch.weight_decay,
                                 batch_size=parent.arch.batch_size)
        child = parent._child("evolve", params=params, arch=arch)
        all_changes = changes + g.changes
        before, after = parent.param_count, child.param_count
        direction = "grow" if after > before else "shrink" if after < before else "same size"
        child.metrics["evolution"] = {"parents": parents, "changes": all_changes, "direction": direction,
                                      "from": list(parent.arch.hidden), "to": list(g.hidden)}
        desc = (f"{direction} {parent.arch.describe()} -> {arch.describe()} from {parent.id}: "
                + "; ".join(all_changes))
        return child, desc

    # ------------------------------------------------------------------- status
    def summary(self, capability: str) -> dict:
        entries = self.archive.get(capability, [])
        champs = [e for e in entries if e.get("outcome") in ("champion", "retired")]
        grown = sum(1 for e in entries if e.get("origin") == "evolve" and e.get("outcome") in ("champion", "retired")
                    and e.get("changes"))
        champ = self.sys.registry.champion(capability)
        return {"capability": capability, "tried": len(entries), "promoted": len(champs), "evolved_promotions": grown,
                "champion": {"id": champ.id, "hidden": list(champ.arch.hidden), "params": champ.param_count,
                             "generation": champ.generation, "fitness": self.node_fitness(champ)} if champ else None,
                "sizes": [e["params"] for e in sorted(champs, key=lambda e: e["created"])][-12:]}

    def status(self) -> dict:
        return {"enabled": self.enabled, "growth_bias": self.pol.growth_bias,
                "capabilities": {c: self.summary(c) for c in self.sys.capabilities if self.archive.get(c)}}
