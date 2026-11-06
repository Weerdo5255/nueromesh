"""The genetic algorithm over node architectures: operators, growth-biased mutation, selection
and the promotion rule that trends evolution towards larger nodes."""
import asyncio
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from neuromesh.commands import execute
from neuromesh.config import EvolutionPolicy
from neuromesh.evolution import (crossover, fitness, hidden_of, insert_layer, judge, morph, mutate, param_count,
                                 prune_layer, remove_layer, widen_layer)
from neuromesh.nodes import Architecture, NeuralNode
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for


def net(hidden=(16, 12), seed=0):
    node = NeuralNode("cap", ["a", "b", "c"], 20, Architecture(hidden=hidden), seed=seed)
    return node, node.numpy_params()


def _forward(params, X):
    h = X
    n = len(params) // 2
    for i in range(n):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < n - 1:
            h = np.maximum(h, 0)
    return h


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.X = self.rng.normal(size=(40, 20)).astype(np.float32)

    def test_growth_is_function_preserving(self):
        _, p = net()
        base = _forward(p, self.X)
        wide = widen_layer(p, 1, 9, self.rng, noise=0.0)
        self.assertEqual(hidden_of(wide), (16, 21))
        np.testing.assert_allclose(_forward(wide, self.X), base, rtol=1e-4, atol=1e-4)
        deep = insert_layer(p, 1, self.rng, noise=0.0)
        self.assertEqual(hidden_of(deep), (16, 16, 12))
        np.testing.assert_allclose(_forward(deep, self.X), base, rtol=1e-4, atol=1e-4)
        with self.assertRaises(ValueError):
            insert_layer(p, 0, self.rng)  # an identity at the input would be the input's width

    def test_contraction_keeps_what_matters(self):
        _, p = net()
        pruned = prune_layer(p, 0, 6)
        self.assertEqual(hidden_of(pruned), (10, 12))
        p2 = [x.copy() for x in p]
        p2[0][:, 3] *= 0.0  # a dead neuron is the first to go, and losing it changes nothing
        np.testing.assert_allclose(_forward(prune_layer(p2, 0, 1), self.X), _forward(p2, self.X), rtol=1e-4, atol=1e-4)
        folded = remove_layer(p, 1)
        self.assertEqual(hidden_of(folded), (16,))
        pos = [np.abs(x) for x in p]  # with no ReLU cutting anything off, folding a layer is exact
        Xp = np.abs(self.X)
        np.testing.assert_allclose(_forward(remove_layer(pos, 1), Xp), _forward(pos, Xp), rtol=1e-3, atol=1e-3)

    def test_morph_reaches_any_shape(self):
        _, p = net()
        for target in [(16, 12), (40,), (8, 8, 8, 8), (64, 32, 16), (9,)]:
            m = morph(p, target, self.rng)
            self.assertEqual(hidden_of(m), target)
            self.assertEqual(m[-2].shape[1], 3)
            self.assertEqual(m[0].shape[0], 20)

    def test_mutation_trends_larger_within_bounds(self):
        pol = EvolutionPolicy(growth_bias=0.7, min_neurons=8, max_params=10 ** 7)
        rng = np.random.default_rng(3)
        dirs = Counter()
        for _ in range(2000):
            g = mutate((64, 32), 0.01, rng, pol, max_hidden=128, max_depth=4, input_dim=100, outputs=3)
            self.assertTrue(1 <= len(g.hidden) <= 4 and all(8 <= h <= 128 for h in g.hidden), g)
            self.assertTrue(g.changes)
            before, after = param_count(100, (64, 32), 3), param_count(100, g.hidden, 3)
            dirs["grow" if after > before else "shrink" if after < before else "same"] += 1
        self.assertGreater(dirs["grow"], 1.8 * dirs["shrink"], dirs)
        pol.growth_bias = 0.5
        neutral = Counter()
        for _ in range(2000):
            g = mutate((64, 32), 0.01, rng, pol, max_hidden=128, max_depth=4, input_dim=100, outputs=3)
            after = param_count(100, g.hidden, 3)
            neutral["grow" if after > param_count(100, (64, 32), 3) else "shrink"] += 1
        self.assertLess(dirs["grow"] / 2000, 0.9)
        self.assertLess(neutral["grow"], dirs["grow"])
        at_max = mutate((128, 128, 128, 128), 0.01, rng, EvolutionPolicy(growth_bias=1.0, max_params=10 ** 7),
                        max_hidden=128, max_depth=4, input_dim=100, outputs=3)
        self.assertTrue(all(h <= 128 for h in at_max.hidden) and len(at_max.hidden) <= 4)

    def test_crossover_selection_and_judging(self):
        rng = np.random.default_rng(0)
        kids = [crossover((64, 32), (100, 50, 25), rng, growth_bias=0.7) for _ in range(500)]
        self.assertTrue(all(k[0] in (64, 100) for k in kids))
        self.assertGreater(sum(len(k) == 3 for k in kids), sum(len(k) == 2 for k in kids))
        self.assertGreater(fitness(0.9, 200_000, 100_000, 0.004), fitness(0.9, 100_000, 100_000, 0.004))
        self.assertGreater(fitness(0.95, 50_000, 100_000, 0.004), fitness(0.9, 400_000, 100_000, 0.004))
        champ = np.array([True] * 18 + [False] * 2)
        same = champ.copy()
        kw = dict(margin=0.01, z_needed=1.0, larger_on_tie=True)
        self.assertEqual(judge(champ, same, 1000, 2000, **kw), (True, "as accurate and larger"))
        self.assertFalse(judge(champ, same, 2000, 1000, **kw)[0], "smaller must be clearly better")
        better = np.array([True] * 20)
        self.assertEqual(judge(np.array([True] * 14 + [False] * 6), better, 2000, 1000, **kw), (True, "better"))
        self.assertFalse(judge(champ, np.array([True] * 17 + [False] * 3), 1000, 4000, **kw)[0], "no regression")
        self.assertFalse(judge(champ, same, 1000, 2000, **{**kw, "larger_on_tie": False})[0])


class EvolutionSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        overrides = {**FAST, "evolution": {"growth_bias": 0.9, "idle_cooldown_factor": 1.0}}
        self.system = System(self.root, overrides=overrides, seed=5)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_nodes_evolve_and_grow(self):
        s = self.system
        for i in range(24):
            await s.submit(["sentiment: the hotel was lovely", "sentiment: the food was awful",
                            "sentiment: it was fine I guess"][i % 3] + f" {i}")
        first = await wait_for(lambda: s.registry.champion("sentiment") is not None, timeout=30)
        self.assertTrue(first)
        start = s.registry.champion("sentiment").param_count

        def grown():
            ch = s.registry.champion("sentiment")
            return ch is not None and ch.origin == "evolve" and ch.param_count > start
        self.assertTrue(await wait_for(grown, timeout=90), s.evolution.summary("sentiment"))
        ch = s.registry.champion("sentiment")
        self.assertIn("evolution", ch.metrics)
        self.assertEqual(ch.metrics["evolution"]["parents"][0], ch.parent_id)
        spawned = [e.data for e in s.bus.recent("node.spawned", 200) if e.data["origin"] == "evolve"]
        self.assertTrue(spawned and all(" from sentiment-" in d["reason"] for d in spawned))
        lines = "\n".join((await execute(s, "/evolution sentiment")).lines)
        self.assertIn("champion sentiment-", lines)
        self.assertIn("genomes tried", lines)
        code, body = await _route(s, "GET", "/evolution", {}, {})
        self.assertEqual(code, 200)
        self.assertTrue(body["enabled"] and body["archive"]["sentiment"])
        archive = dict(s.evolution.archive)
        await s.stop()
        again = System(self.root, overrides=FAST, seed=6)
        self.assertEqual(len(again.evolution.archive["sentiment"]), len(archive["sentiment"]))
        again.store.close()

    async def test_disabled_falls_back_to_fixed_mutations(self):
        s = self.system
        s.directives.evolution.enabled = False
        node = NeuralNode("sentiment", ["positive", "negative"], s.directives.nodes.input_dim, seed=1)
        architect = next(a for a in s.agents if type(a).__name__ == "Architect")
        child, kind = architect._mutate(node, s.capabilities["sentiment"])
        self.assertIn(kind, ("widen", "retune", "deepen", "fresh"))
        self.assertIn("evolution is off", (await execute(s, "/evolution")).lines[0])


if __name__ == "__main__":
    unittest.main()
