import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuromesh.encoding import HashingEncoder
from neuromesh.nodes import Architecture, NeuralNode


def toy_data(n=300, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)
    labels = ["a", "b", "c"]
    return X, [labels[i] for i in y], labels


def train(node, X, labs, rounds=10):
    for _ in range(rounds):
        params, opt, loss = node.train_copy(X, node.label_indices(labs), np.ones(len(labs), np.float32), 5)
        node.commit(params, opt, loss)


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.X, self.labs, labels = toy_data()
        self.node = NeuralNode("toy", labels, 32, Architecture((16,), lr=0.01), seed=1)
        train(self.node, self.X, self.labs)

    def test_learns(self):
        self.assertGreater(self.node.correct(self.X, self.labs).mean(), 0.95)

    def test_widen_preserves_function(self):
        child = self.node.spawn_wider(1.5, noise=0)
        diff = np.abs(child.predict_proba(self.X) - self.node.predict_proba(self.X)).max()
        self.assertLess(diff, 1e-5)
        self.assertEqual(child.arch.hidden, (24,))
        self.assertEqual(child.parent_id, self.node.id)

    def test_deepen_preserves_function(self):
        child = self.node.spawn_deeper(noise=0)
        diff = np.abs(child.predict_proba(self.X) - self.node.predict_proba(self.X)).max()
        self.assertLess(diff, 1e-5)
        self.assertEqual(child.arch.hidden, (16, 16))

    def test_expand_keeps_old_behaviour_and_learns_new_label(self):
        child = self.node.spawn_expanded(["d"])
        self.assertEqual(child.labels, ["a", "b", "c", "d"])
        same = (child.predict_proba(self.X).argmax(1) == self.node.predict_proba(self.X).argmax(1)).mean()
        self.assertEqual(same, 1.0)
        rng = np.random.default_rng(5)
        Xd = (rng.normal(size=(100, 32)) + np.eye(32)[5] * 6).astype(np.float32)
        X2 = np.concatenate([self.X, Xd])
        labs2 = self.labs + ["d"] * 100
        train(child, X2, labs2)
        self.assertGreater(child.correct(Xd, ["d"] * 100).mean(), 0.9)

    def test_exclude_masks_labels(self):
        label, _ = self.node.predict(self.X[0], exclude=frozenset({"a", "b"}))
        self.assertEqual(label, "c")

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.node.status = "champion"
            self.node.save(Path(d))
            clone = NeuralNode.load(self.node.meta(), Path(d))
            self.assertTrue(np.allclose(clone.predict_proba(self.X), self.node.predict_proba(self.X)))
            self.assertEqual(clone.status, "champion")


class EncoderTests(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        enc = HashingEncoder(256)
        a, b = enc.encode("What is 12 * 7?"), enc.encode("What is 12 * 7?")
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertFalse(np.array_equal(a, enc.encode("convert 5 km to miles")))


if __name__ == "__main__":
    unittest.main()
