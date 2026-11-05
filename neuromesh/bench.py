"""Benchmark node training and inference on the CPU and on a GPU.

    python -m neuromesh bench                       # default node size, best GPU found
    python -m neuromesh bench --device cuda --input-dim 16384 --hidden 512,256 --batch-size 256

Uses synthetic data shaped like the hashed text features nodes normally see. Both
backends start from identical weights, so their training accuracy should match closely;
a large difference points at a backend problem rather than a speed difference.
"""
from __future__ import annotations

import time

import numpy as np

from .backend import Backend, NumpyBackend, select_backend
from .nodes import Architecture, NeuralNode


def _synthetic(examples: int, input_dim: int, labels: int, seed: int):
    rng = np.random.default_rng(seed)
    active = min(48, input_dim)
    X = np.zeros((examples, input_dim), dtype=np.float32)
    rows = np.repeat(np.arange(examples), active)
    cols = rng.integers(0, input_dim, examples * active)
    np.add.at(X, (rows, cols), rng.choice(np.array([-1.0, 1.0], dtype=np.float32), examples * active))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    teacher = rng.normal(size=(input_dim, labels)).astype(np.float32)
    y = (X @ teacher).argmax(axis=1).astype(np.int64)
    return X, y


def _measure(B: Backend, X, y, names, hidden, batch_size, epochs, predictions) -> dict:
    n = len(y)
    w = np.ones(n, dtype=np.float32)
    labels = [names[i] for i in y]
    node = NeuralNode("bench", names, X.shape[1], Architecture(tuple(hidden), batch_size=batch_size),
                      seed=1, backend=B)
    warm = min(n, batch_size * 2)  # first calls pay for device init / kernel compilation
    node.train_copy(X[:warm], y[:warm], w[:warm], 1)
    node.predict(X[0])
    node.correct(X[:64], labels[:64])
    B.synchronize()

    t0 = time.perf_counter()
    params, opt, loss = node.train_copy(X, y, w, epochs)
    B.synchronize()
    train_s = time.perf_counter() - t0
    node.commit(params, opt, loss)

    t0 = time.perf_counter()
    for i in range(predictions):
        node.predict(X[i % n])
    predict_ms = (time.perf_counter() - t0) / predictions * 1000

    t0 = time.perf_counter()
    acc = float(node.correct(X, labels).mean())
    eval_ms = (time.perf_counter() - t0) * 1000
    return {"backend": B.describe(), "train_s": train_s, "rate": n * epochs / train_s,
            "predict_ms": predict_ms, "eval_ms": eval_ms, "acc": acc, "loss": loss,
            "params": node.param_count}


def run_bench(device: str = "auto", library: str = "auto", input_dim: int = 1024, hidden: tuple = (64,),
              batch_size: int = 32, examples: int = 4000, labels: int = 8, epochs: int = 3,
              predictions: int = 200, seed: int = 0) -> list[dict]:
    target = select_backend(device, library)
    backends: list[Backend] = [NumpyBackend()]
    if target.name != "numpy":
        backends.append(target)
    X, y = _synthetic(examples, input_dim, labels, seed)
    names = [f"label{i}" for i in range(labels)]
    print(f"node: input_dim={input_dim} hidden={list(hidden)} labels={labels} batch_size={batch_size}; "
          f"{examples} examples x {epochs} epochs\n")
    if len(backends) == 1:
        print(f"(no GPU backend for device '{device}' on this machine; showing the CPU only)\n")
    rows = [_measure(B, X, y, names, hidden, batch_size, epochs, predictions) for B in backends]

    print(f"{'backend':<44} {'train s':>8} {'examples/s':>11} {'predict ms':>11} "
          f"{'eval ' + str(examples) + ' ms':>13} {'train acc':>9}")
    for r in rows:
        print(f"{r['backend'][:44]:<44} {r['train_s']:>8.2f} {r['rate']:>11,.0f} {r['predict_ms']:>11.3f} "
              f"{r['eval_ms']:>13.1f} {r['acc']:>9.3f}")
    print(f"\nparameters per node: {rows[0]['params']:,}")
    if len(rows) == 2:
        cpu, gpu = rows
        speed = cpu["train_s"] / gpu["train_s"]
        print(f"GPU training speed-up: {speed:.1f}x   batch evaluation: {cpu['eval_ms'] / gpu['eval_ms']:.1f}x   "
              f"single prediction: {cpu['predict_ms'] / gpu['predict_ms']:.1f}x")
        if abs(cpu["acc"] - gpu["acc"]) > 0.02:
            print("warning: CPU and GPU accuracy differ by more than 2 points; run tests/test_backend.py")
        if speed >= 1.2:
            print(f"-> worth it: set [compute] device = \"{target.device}\" (or run with --device {target.device}), "
                  "and use these node sizes in [nodes].")
        else:
            print("-> at this node size the CPU is as fast or faster (GPU launch and transfer overhead "
                  "dominates). GPUs pay off for bigger nodes, e.g.:\n   python -m neuromesh bench "
                  f"--device {device} --input-dim 16384 --hidden 512,256 --batch-size 256")
    return rows
