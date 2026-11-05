"""Compute backends. GPU parity tests run for every backend available on this machine
(PyTorch CPU/CUDA/ROCm/MPS, CuPy) and are skipped where none is installed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuromesh.backend import BackendUnavailable, NumpyBackend, TorchBackend, select_backend
from neuromesh.nodes import Architecture, NeuralNode

ROOT = Path(__file__).resolve().parent.parent


def available_backends():
    """Every non-numpy backend usable here, as (label, backend)."""
    found = []
    try:
        import torch
        found.append(("torch-cpu", TorchBackend("cpu")))
        if torch.cuda.is_available():
            found.append(("torch-cuda", TorchBackend("cuda:0")))
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            found.append(("torch-mps", TorchBackend("mps")))
    except ImportError:
        pass
    try:
        from neuromesh.backend import CupyBackend
        found.append(("cupy", CupyBackend(0)))
    except Exception:
        pass
    return found


def gpu_present() -> bool:
    return any(b.is_gpu for _, b in available_backends())


def data(n=256, dim=64, labels=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = (X[:, :labels] * 2 + rng.normal(scale=0.3, size=(n, labels))).argmax(1).astype(np.int64)
    names = [f"l{i}" for i in range(labels)]
    return X, y, names, [names[i] for i in y]


class SelectionTests(unittest.TestCase):
    def test_cpu_is_numpy(self):
        self.assertIsInstance(select_backend("cpu"), NumpyBackend)

    def test_auto_always_returns_something_usable(self):
        B = select_backend("auto")
        X, y, names, _ = data()
        node = NeuralNode("t", names, X.shape[1], Architecture((16,)), seed=0, backend=B)
        self.assertEqual(node.predict_proba(X).shape, (len(X), len(names)))
        if not gpu_present():
            self.assertIsInstance(B, NumpyBackend)

    @unittest.skipIf(gpu_present(), "a GPU is present")
    def test_explicit_gpu_without_gpu_is_an_error_not_a_fallback(self):
        with self.assertRaises(BackendUnavailable):
            select_backend("cuda")

    def test_bad_arguments(self):
        for device, library in (("tpu", "auto"), ("cuda:x", "auto"), ("cpu", "jax"), ("cuda", "numpy")):
            with self.subTest(device=device, library=library), self.assertRaises(BackendUnavailable):
                select_backend(device, library)


class ParityTests(unittest.TestCase):
    """Each backend must compute what numpy computes, from identical starting weights."""

    def setUp(self):
        self.backends = available_backends()
        if not self.backends:
            self.skipTest("no PyTorch or CuPy installed")
        self.X, self.y, self.names, self.labels = data()

    def pair(self, B, hidden=(32, 16)):
        cpu = NeuralNode("t", self.names, self.X.shape[1], Architecture(hidden, batch_size=32), seed=3)
        dev = NeuralNode("t", self.names, self.X.shape[1], Architecture(hidden, batch_size=32), seed=3, backend=B)
        return cpu, dev

    def test_forward_matches(self):
        for label, B in self.backends:
            with self.subTest(backend=label):
                cpu, dev = self.pair(B)
                np.testing.assert_allclose(dev.predict_proba(self.X), cpu.predict_proba(self.X), atol=1e-5)

    def test_training_matches(self):
        w = np.ones(len(self.y), dtype=np.float32)
        for label, B in self.backends:
            with self.subTest(backend=label):
                cpu, dev = self.pair(B)
                for node in (cpu, dev):
                    for _ in range(3):
                        params, opt, loss = node.train_copy(self.X, self.y, w, 2)
                        node.commit(params, opt, loss)
                np.testing.assert_allclose(dev.predict_proba(self.X), cpu.predict_proba(self.X), atol=2e-3)
                self.assertAlmostEqual(dev.metrics["last_loss"], cpu.metrics["last_loss"], places=2)
                agree = (dev.predict_proba(self.X).argmax(1) == cpu.predict_proba(self.X).argmax(1)).mean()
                self.assertGreaterEqual(agree, 0.99)
                self.assertGreater(dev.correct(self.X, self.labels).mean(), 0.8)

    def test_derivations_preserve_function_on_device(self):
        for label, B in self.backends:
            with self.subTest(backend=label):
                _, dev = self.pair(B)
                base = dev.predict_proba(self.X)
                for child in (dev.spawn_wider(1.5, noise=0), dev.spawn_deeper(noise=0)):
                    self.assertIs(child.backend, B)
                    np.testing.assert_allclose(child.predict_proba(self.X), base, atol=1e-5)
                expanded = dev.spawn_expanded(["new"])
                self.assertTrue((expanded.predict_proba(self.X).argmax(1) == base.argmax(1)).all())

    def test_weights_move_between_devices(self):
        for label, B in self.backends:
            with self.subTest(backend=label), tempfile.TemporaryDirectory() as d:
                _, dev = self.pair(B)
                dev.save(Path(d))
                on_cpu = NeuralNode.load(dev.meta(), Path(d))
                back = NeuralNode.load(dev.meta(), Path(d), B)
                np.testing.assert_allclose(on_cpu.predict_proba(self.X), dev.predict_proba(self.X), atol=1e-5)
                np.testing.assert_allclose(back.predict_proba(self.X), dev.predict_proba(self.X), atol=1e-6)


class InputDimChangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_nodes_for_another_input_dim_are_retired_and_rebuilt(self):
        from neuromesh.runtime import System
        from test_system import FAST, make_root, orders, wait_for
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(Path(tmp))
            system = System(root, overrides=FAST, seed=1, device="cpu")
            await system.start()
            for text in orders(50):
                await system.submit(text)
            self.assertTrue(await wait_for(lambda: system.registry.champion("dispatch") is not None))
            old = system.registry.champion("dispatch").id
            await system.stop()

            smaller = {**FAST, "nodes": {**FAST["nodes"], "input_dim": 512}}
            system = System(root, overrides=smaller, seed=2, device="cpu")
            self.assertEqual(system.registry.lineage[old]["status"], "disposed")
            self.assertFalse((system.state_dir / "nodes" / f"{old}.npz").exists())
            await system.start()
            try:
                res = await system.submit("what is 6 * 7")
                self.assertEqual(res.output, 42)
                self.assertTrue(await wait_for(lambda: system.registry.champion("dispatch") is not None))
                self.assertEqual(system.registry.champion("dispatch").input_dim, 512)
            finally:
                await system.stop()


class BenchTests(unittest.TestCase):
    def test_bench_runs(self):
        from neuromesh.bench import run_bench
        rows = run_bench(device="cpu", input_dim=128, hidden=(16,), examples=200, epochs=1, predictions=5)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["acc"], 0.3)


if __name__ == "__main__":
    unittest.main()
