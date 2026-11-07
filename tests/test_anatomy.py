"""A closer look: how nodes are wired together (the paths orders take), how one node is built, and
which of its synapses fire for an input (anatomy.py, the web Live view's closer look)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuromesh import anatomy, web
from neuromesh.commands import execute
from neuromesh.demo import FACTS
from neuromesh.encoding import HashingEncoder
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root, orders, wait_for


class FiringTests(unittest.TestCase):
    def test_a_forward_pass_is_recorded_faithfully(self):
        rng = np.random.default_rng(3)
        dim = 64
        params = [rng.normal(0, 0.4, (dim, 8)).astype(np.float32), np.zeros(8, np.float32),
                  rng.normal(0, 0.4, (8, 6)).astype(np.float32), np.zeros(6, np.float32),
                  rng.normal(0, 0.4, (6, 3)).astype(np.float32), np.zeros(3, np.float32)]
        labels = ["red", "green", "blue"]

        def read(text):
            feats = HashingEncoder.features(text)
            return HashingEncoder(dim).encode(text), [(f, *anatomy._bucket(f, dim, dim)) for f in feats]
        text = "the quick brown fox was not lazy"
        f = anatomy.fire(params, labels, read, text, inputs=10, neurons=5, outputs=3, synapses=12)
        x = read(text)[0]
        h1 = np.maximum(x @ params[0], 0)
        h2 = np.maximum(h1 @ params[2], 0)
        logits = h2 @ params[4]
        self.assertEqual(f["chosen"]["label"], labels[int(np.argmax(logits))])
        self.assertEqual([L["kind"] for L in f["layers"]], ["input", "hidden", "hidden", "output"])
        self.assertEqual(f["layers"][1]["active"], int((h1 > 0).sum()))
        self.assertEqual(len(f["path"]), 4)
        self.assertEqual(f["path"][-1], f["chosen"]["i"])
        # each step of the strongest path is the largest signal into the next neuron on it
        acts = [x, h1, h2]
        for k in range(3):
            contrib = acts[k] * params[2 * k][:, f["path"][k + 1]]
            self.assertEqual(f["path"][k], int(np.argmax(contrib)))
        # gradient x input: the attributions add up to the chosen logit (biases are zero, ReLU is piecewise linear)
        attr_total = sum(u["attr"] for u in f["layers"][0]["shown"])
        everything = anatomy.fire(params, labels, read, text, inputs=dim, neurons=8, outputs=3, synapses=12)
        self.assertAlmostEqual(sum(u["attr"] for u in everything["layers"][0]["shown"]), float(logits.max()), places=3)
        self.assertLessEqual(abs(attr_total), sum(abs(u["attr"]) for u in everything["layers"][0]["shown"]) + 1e-6)
        for s in f["synapses"]:
            self.assertAlmostEqual(s["signal"], round(float(acts[s["layer"]][s["from"]] * params[2 * s["layer"]][s["from"], s["to"]]), 5), places=4)
        self.assertTrue(any(s["path"] for s in f["synapses"]))
        self.assertIn("lazy", " ".join(w["word"] for w in f["words"]) + " not lazy")
        self.assertEqual(anatomy.describe_feature("b:was_not"), "'was not' (pair)")
        self.assertEqual(anatomy._word_of("n:lazy"), ["not lazy"])


class AnatomyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides={**FAST, "memory": {"enabled": True}}, seed=21)
        await self.system.start()
        s = self.system
        for text in orders(70, seed=5):
            await s.submit(text)
        self.assertTrue(await wait_for(lambda: s.registry.champion("dispatch") is not None
                                       and (s.registry.champion("dispatch").metrics.get("holdout_acc") or 0) >= 0.9, 90))

    async def asyncTearDown(self):
        await self.system.stop()
        self.tmp.cleanup()

    async def test_inside_a_node(self):
        s = self.system
        router = s.registry.champion("dispatch")
        res = await s.submit("what is 12 * 7")
        a = anatomy.anatomy(s, "dispatch", "what is 12 * 7")  # a skill name means its serving node
        self.assertEqual((a["id"], a["kind"], a["capability"]), (router.id, "capability", "dispatch"))
        self.assertEqual(a["sizes"], [router.input_dim, *router.arch.hidden, len(router.labels)])
        self.assertEqual(a["params"], router.param_count)
        self.assertEqual(len(a["layers"]), len(router.arch.hidden) + 1)
        label, conf = router.predict(s.encode("what is 12 * 7"))
        self.assertEqual(a["fire"]["chosen"]["label"], label)
        self.assertAlmostEqual(a["fire"]["chosen"]["prob"], conf, places=3)
        self.assertEqual(res.route, "tool:calculator")
        # it fired on that order, and its answer was used
        fired = [f for f in a["firings"] if f["text"] == "what is 12 * 7"]
        self.assertTrue(fired and fired[0]["used"] is True, a["firings"][:3])
        self.assertIsNone(anatomy.anatomy(s, router.id)["fire"])
        latest = anatomy.anatomy(s, router.id, latest=True)
        self.assertEqual(latest["text_from"], "its latest firing")
        self.assertIsNone(anatomy.anatomy(s, "no-such-node"))

    async def test_a_memory_node(self):
        s = self.system
        for q, ans in FACTS[:4]:
            s.memory.learn(q, ans, "the operator", by="tester", kind="people")
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 60))
        res = await s.submit("What is the capital of Australia?")
        node = next(iter(s.memory.nodes.values()))
        a = anatomy.anatomy(s, node.id, "What is the capital of Australia?")
        self.assertEqual(a["kind"], "memory")
        self.assertIn("Canberra", a["fire"]["chosen"]["label"])
        self.assertIn("australia", [w["word"] for w in a["fire"]["words"]][:3])
        self.assertEqual(a["sizes"][0], node.inputs or s.memory.dim)
        hops = anatomy.traces(s, 1)[0]["hops"]
        self.assertIn(("memory", node.id), [(h["stage"], h.get("node")) for h in hops])
        self.assertIn("Canberra", res.output)

    async def test_the_paths_orders_take(self):
        s = self.system
        router = s.registry.champion("dispatch")
        await s.submit("what is 9 * 9")
        t = anatomy.traces(s, 3)[0]
        self.assertEqual(t["text"], "what is 9 * 9")
        stages = [h["stage"] for h in t["hops"]]
        self.assertEqual(stages[0], "director")
        self.assertEqual(stages[-1], "director")
        rhop = next(h for h in t["hops"] if h["stage"] == "router")
        self.assertEqual((rhop["node"], rhop["label"], rhop["input"]), (router.id, "tool:calculator", "what is 9 * 9"))
        self.assertIn(("tool", "calculator"), [(h["stage"], h.get("tool")) for h in t["hops"]])
        lines = "\n".join((await execute(s, "/traces")).lines)
        self.assertIn(f"{router.id} [tool:calculator] -> calculator [ok]", lines)
        lines = "\n".join((await execute(s, "/anatomy dispatch :: what is 9 * 9")).lines)
        for part in (router.id, "layer 1:", "fired on 'what is 9 * 9'", "strongest path:", "words that pushed it:"):
            self.assertIn(part, lines)
        status, body = await _route(s, "GET", "/anatomy", {"node": router.id, "text": "what is 9 * 9"}, {})
        self.assertEqual((status, body["fire"]["chosen"]["label"]), (200, "tool:calculator"))
        status, body = await _route(s, "GET", "/traces", {"n": "2"}, {})
        self.assertEqual((status, len(body["traces"])), (200, 2))
        # a node evolution has moved on from: its weights are gone, and the API says so
        s.registry.lineage["dispatch-gone00"] = {"id": "dispatch-gone00", "capability": "dispatch", "status": "disposed"}
        status, body = await _route(s, "GET", "/anatomy", {"node": "dispatch-gone00"}, {})
        self.assertEqual((status, body["capability"]), (410, "dispatch"))
        self.assertIn("has been disposed", body["error"])
        status, _ = await _route(s, "GET", "/anatomy", {"node": "nothing-here"}, {})
        self.assertEqual(status, 404)

    def test_the_web_page_has_a_closer_look(self):
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        for part in ('id="closer"', 'aria-pressed="false"', 'id="microscope"', 'id="mic-traces"', 'id="mic-anatomy"'):
            self.assertIn(part, page)
        for part in ("/anatomy?node=", "/traces?n=", "function anatomySvg", "data-node"):
            self.assertIn(part, script)


if __name__ == "__main__":
    unittest.main()
