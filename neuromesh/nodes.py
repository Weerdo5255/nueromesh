"""Neural nodes: small specialist networks the system grows, trains, replaces and disposes.

Each node is an MLP (ReLU hidden layers, softmax output, Adam) bound to one capability.
Its math runs on a compute backend (numpy on the CPU by default, or PyTorch / CuPy on a
GPU; see backend.py). Everything outside the node sees numpy arrays only.

New nodes are usually *derived* from an existing one so knowledge carries over:

* expand  - add output units for new labels, initialised "quiet" so old behaviour is kept
* widen   - Net2Wider: function-preserving widening of every hidden layer
* deepen  - Net2Deeper: insert an identity layer (exact for post-ReLU inputs)
* retune  - same weights, different learning rate
* fresh   - new random initialisation with a different architecture

Derivations are rare, so they are done on CPU copies of the weights and the child moves
its weights to the device.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backend import Backend, NumpyBackend

_DEFAULT_BACKEND = NumpyBackend()


@dataclass
class Architecture:
    hidden: tuple = (64,)
    lr: float = 0.01
    weight_decay: float = 1e-4
    batch_size: int = 32

    def describe(self) -> str:
        shape = "x".join(str(h) for h in self.hidden) or "linear"
        return f"{shape} lr={self.lr:g}"

    def to_dict(self) -> dict:
        return {"hidden": list(self.hidden), "lr": self.lr,
                "weight_decay": self.weight_decay, "batch_size": self.batch_size}

    @classmethod
    def from_dict(cls, d: dict) -> "Architecture":
        return cls(hidden=tuple(d.get("hidden", (64,))), lr=d.get("lr", 0.01),
                   weight_decay=d.get("weight_decay", 1e-4), batch_size=d.get("batch_size", 32))


# ------------------------------------------------------------------ backend math
def _forward(B: Backend, params: list, X, keep: bool = False):
    acts = [X]
    h = X
    n_layers = len(params) // 2
    for i in range(n_layers):
        z = h @ params[2 * i] + params[2 * i + 1]
        h = B.relu(z) if i < n_layers - 1 else z
        acts.append(h)
    e = B.exp(h - B.max_rows(h))
    probs = e / B.sum_rows(e)
    return (probs, acts) if keep else probs


def _grads(B: Backend, params: list, X, y, w, weight_decay: float):
    probs, acts = _forward(B, params, X, keep=True)
    rows = B.arange(X.shape[0])
    wn = w / w.sum()
    d = B.copy(probs)
    d[rows, y] -= 1.0
    d = d * wn[:, None]
    grads: list = [None] * len(params)
    for i in reversed(range(len(params) // 2)):
        grads[2 * i] = acts[i].T @ d + weight_decay * params[2 * i]
        grads[2 * i + 1] = B.sum0(d)
        if i > 0:
            d = (d @ params[2 * i].T) * B.gt0(acts[i])
    loss = -(wn * B.log(probs[rows, y] + 1e-9)).sum()
    return grads, loss


class NeuralNode:
    STATUSES = ("candidate", "champion", "instance", "retired", "disposed")

    def __init__(self, capability: str, labels: list[str], input_dim: int,
                 arch: Architecture | None = None, *, node_id: str | None = None,
                 parent_id: str | None = None, generation: int = 0, origin: str = "seed",
                 seed: int | None = None, params: list[np.ndarray] | None = None,
                 backend: Backend | None = None):
        self.id = node_id or f"{capability}-{uuid.uuid4().hex[:6]}"
        self.capability = capability
        self.labels = list(labels)
        self.input_dim = input_dim
        self.arch = arch or Architecture()
        self.backend = backend or _DEFAULT_BACKEND
        self.parent_id = parent_id
        self.generation = generation
        self.origin = origin
        self.status = "candidate"
        now = time.time()
        self.created_at = now
        self.updated_at = now
        self.promoted_at: float | None = None
        self.retired_at: float | None = None
        self.disposed_at: float | None = None
        self.last_used = 0.0
        self.train_steps = 0
        self.examples_seen = 0
        self.metrics: dict = {"holdout_acc": None, "holdout_n": 0, "per_label_recall": {},
                              "live_agree": 0, "live_total": 0, "serves": 0, "last_loss": None}
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(seed)
        host = params if params is not None else self._init_params()
        B = self.backend
        with B.scope():
            self.params = [B.asarray(p) for p in host]
            self.opt = {"m": [B.zeros_like(p) for p in self.params],
                        "v": [B.zeros_like(p) for p in self.params], "t": 0}

    # ------------------------------------------------------------------ setup
    def _init_params(self) -> list[np.ndarray]:
        """Initialised on the CPU from the node's seed, so every backend starts identical."""
        sizes = [self.input_dim, *self.arch.hidden, len(self.labels)]
        params: list[np.ndarray] = []
        for a, b in zip(sizes, sizes[1:]):
            params.append(self.rng.normal(0.0, np.sqrt(2.0 / a), (a, b)).astype(np.float32))
            params.append(np.zeros(b, dtype=np.float32))
        return params

    def numpy_params(self) -> list[np.ndarray]:
        """CPU copies of the current weights."""
        with self.lock:
            params = list(self.params)
        with self.backend.scope():
            return [np.array(self.backend.to_numpy(p), dtype=np.float32) for p in params]

    # ------------------------------------------------------------- inference
    def predict_proba(self, X: np.ndarray, params: list | None = None) -> np.ndarray:
        B = self.backend
        with B.scope():
            probs = _forward(B, params if params is not None else self.params, B.asarray(X))
            return np.asarray(B.to_numpy(probs))

    def predict(self, x: np.ndarray, exclude: set[str] | frozenset = frozenset()) -> tuple[str, float]:
        p = self.predict_proba(x[None, :])[0]
        if exclude:
            mask = np.array([lab not in exclude for lab in self.labels], dtype=np.float32)
            if mask.sum() > 0:
                p = p * mask
                p = p / max(p.sum(), 1e-9)
        i = int(p.argmax())
        return self.labels[i], float(p[i])

    def correct(self, X: np.ndarray, labels: list[str], params: list | None = None) -> np.ndarray:
        if len(labels) == 0:
            return np.zeros(0, dtype=bool)
        B = self.backend
        with B.scope():
            pred = B.argmax_rows(_forward(B, params if params is not None else self.params, B.asarray(X)))
        return np.array([self.labels[i] == lab for i, lab in zip(pred, labels)], dtype=bool)

    def label_indices(self, labels: list[str]) -> np.ndarray:
        index = {lab: i for i, lab in enumerate(self.labels)}
        return np.array([index.get(lab, -1) for lab in labels], dtype=np.int64)

    # -------------------------------------------------------------- training
    def train_copy(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, epochs: int):
        """Train a copy of the weights (on the node's device); the caller decides whether
        to commit it. Inputs are numpy arrays; they are moved to the device once."""
        B = self.backend
        keep = y >= 0
        X, y, w = X[keep], y[keep], w[keep]
        rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        lr, bs, wd = self.arch.lr, self.arch.batch_size, self.arch.weight_decay
        b1, b2, eps = 0.9, 0.999, 1e-8
        with self.lock:
            src_params, src_opt = list(self.params), self.opt
        with B.scope():
            params = [B.copy(p) for p in src_params]
            opt = {"m": [B.copy(m) for m in src_opt["m"]], "v": [B.copy(v) for v in src_opt["v"]],
                   "t": src_opt["t"]}
            if len(y) == 0:
                return params, opt, None
            Xd, yd, wd_ = B.asarray(X), B.asarray(y, integer=True), B.asarray(w)
            total, batches = 0.0, 0
            for _ in range(epochs):
                order = B.asarray(rng.permutation(len(y)), integer=True)
                for start in range(0, len(y), bs):
                    idx = order[start:start + bs]
                    grads, loss = _grads(B, params, Xd[idx], yd[idx], wd_[idx], wd)
                    opt["t"] += 1
                    t = opt["t"]
                    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
                    for k, g in enumerate(grads):
                        opt["m"][k] = b1 * opt["m"][k] + (1 - b1) * g
                        opt["v"][k] = b2 * opt["v"][k] + (1 - b2) * g * g
                        params[k] = params[k] - lr * (opt["m"][k] / c1) / (B.sqrt(opt["v"][k] / c2) + eps)
                    total = total + loss
                    batches += 1
            return params, opt, float(total) / max(batches, 1)

    def commit(self, params, opt, loss=None) -> None:
        with self.lock:
            self.params = params
            self.opt = opt
            self.train_steps += 1
            self.updated_at = time.time()
            if loss is not None:
                self.metrics["last_loss"] = round(float(loss), 4)

    # ------------------------------------------------------------ derivation
    def _child(self, origin: str, *, params=None, labels=None, arch=None) -> "NeuralNode":
        return NeuralNode(
            self.capability, labels or list(self.labels), self.input_dim,
            arch or copy.deepcopy(self.arch), parent_id=self.id,
            generation=self.generation + 1, origin=origin,
            seed=int(self.rng.integers(1 << 31)),
            params=params if params is not None else self.numpy_params(), backend=self.backend)

    def spawn_expanded(self, new_labels: list[str]) -> "NeuralNode":
        add = [lab for lab in new_labels if lab not in self.labels]
        params = self.numpy_params()
        W, b = params[-2], params[-1]
        params[-2] = np.concatenate(
            [W, self.rng.normal(0.0, 0.01, (W.shape[0], len(add))).astype(np.float32)], axis=1)
        params[-1] = np.concatenate([b, np.full(len(add), float(b.min()) - 3.0, dtype=np.float32)])
        return self._child("expand", params=params, labels=self.labels + add)

    def spawn_wider(self, factor: float = 1.5, noise: float = 1e-3) -> "NeuralNode":
        params = self.numpy_params()
        n_hidden = len(params) // 2 - 1
        if n_hidden == 0:
            return self.spawn_fresh((64,))
        hidden = []
        for i in range(n_hidden):
            W_in, b, W_out = params[2 * i], params[2 * i + 1], params[2 * i + 2]
            h = W_in.shape[1]
            k = max(1, int(round(h * (factor - 1.0))))
            idx = self.rng.integers(0, h, size=k)
            counts = np.bincount(idx, minlength=h).astype(np.float32) + 1.0
            jitter = self.rng.normal(0.0, noise, (W_in.shape[0], k)).astype(np.float32) if noise else 0.0
            params[2 * i] = np.concatenate([W_in, W_in[:, idx] + jitter], axis=1).astype(np.float32)
            params[2 * i + 1] = np.concatenate([b, b[idx]]).astype(np.float32)
            scaled = W_out / counts[:, None]
            params[2 * i + 2] = np.concatenate([scaled, scaled[idx]], axis=0).astype(np.float32)
            hidden.append(h + k)
        arch = copy.deepcopy(self.arch)
        arch.hidden = tuple(hidden)
        return self._child("widen", params=params, arch=arch)

    def spawn_deeper(self, noise: float = 1e-3) -> "NeuralNode":
        if len(self.params) // 2 - 1 == 0:
            return self.spawn_fresh((64,))
        params = self.numpy_params()
        h = params[-2].shape[0]
        eye = np.eye(h, dtype=np.float32)
        if noise:
            eye += self.rng.normal(0.0, noise, (h, h)).astype(np.float32)
        params = params[:-2] + [eye, np.zeros(h, dtype=np.float32)] + params[-2:]
        arch = copy.deepcopy(self.arch)
        arch.hidden = tuple(list(self.arch.hidden) + [h])
        return self._child("deepen", params=params, arch=arch)

    def spawn_retuned(self, lr_factor: float) -> "NeuralNode":
        arch = copy.deepcopy(self.arch)
        arch.lr = float(np.clip(self.arch.lr * lr_factor, 1e-4, 0.1))
        return self._child("retune", arch=arch)

    def spawn_fresh(self, hidden: tuple, labels: list[str] | None = None) -> "NeuralNode":
        arch = copy.deepcopy(self.arch)
        arch.hidden = tuple(hidden)
        return NeuralNode(self.capability, labels or list(self.labels), self.input_dim, arch,
                          parent_id=self.id, generation=self.generation + 1, origin="fresh",
                          seed=int(self.rng.integers(1 << 31)), backend=self.backend)

    # ----------------------------------------------------------- persistence
    @property
    def param_count(self) -> int:
        return int(sum(int(np.prod(tuple(p.shape))) for p in self.params))

    def meta(self) -> dict:
        return {
            "id": self.id, "capability": self.capability, "labels": self.labels,
            "input_dim": self.input_dim, "arch": self.arch.to_dict(), "parent_id": self.parent_id,
            "generation": self.generation, "origin": self.origin, "status": self.status,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "promoted_at": self.promoted_at, "retired_at": self.retired_at,
            "disposed_at": self.disposed_at, "last_used": self.last_used,
            "train_steps": self.train_steps, "examples_seen": self.examples_seen,
            "metrics": self.metrics, "params": self.param_count,
        }

    def weights_path(self, directory: Path) -> Path:
        return Path(directory) / f"{self.id}.npz"

    def save(self, directory: Path) -> None:
        np.savez(self.weights_path(directory), *self.numpy_params())

    @classmethod
    def load(cls, meta: dict, directory: Path, backend: Backend | None = None) -> "NeuralNode":
        with np.load(Path(directory) / f"{meta['id']}.npz") as data:
            params = [data[f"arr_{i}"] for i in range(len(data.files))]
        node = cls(meta["capability"], meta["labels"], meta["input_dim"],
                   Architecture.from_dict(meta["arch"]), node_id=meta["id"],
                   parent_id=meta.get("parent_id"), generation=meta.get("generation", 0),
                   origin=meta.get("origin", "seed"), params=params, backend=backend)
        for key in ("status", "created_at", "updated_at", "promoted_at", "retired_at",
                    "last_used", "train_steps", "examples_seen"):
            if key in meta:
                setattr(node, key, meta[key])
        node.metrics.update(meta.get("metrics", {}))
        return node
