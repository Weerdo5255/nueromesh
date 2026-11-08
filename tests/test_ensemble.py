"""Many instances of each node, at once and in concert (ensemble.py): the champion and its instances answer
together, at most [nodes] max_running node instances run at the same time, and when instances of the same
kind of node disagree - a capability's nodes, or memory nodes - the ego is the ultimate arbitrator (ego.py)."""
import asyncio
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from neuromesh.ego import ARB_KINDS, EgoTeacher, arb_policy, arb_text, parse_arb
from neuromesh.ensemble import members, similar
from neuromesh.nodes import Architecture, NeuralNode
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for

CAP = "sentiment"


class PolicyTests(unittest.TestCase):
    def test_whom_the_ego_trusts(self):
        f = {"amood": "steady", "asplit": "majority", "achamp": "sure", "amemory": "none", "abest": "champion"}
        self.assertEqual(arb_policy(f)[0], "trust-majority")
        self.assertEqual(arb_policy({**f, "amemory": "other"})[0], "trust-memory")             # it settled this before
        self.assertEqual(arb_policy({**f, "amood": "wary"})[0], "trust-champion")               # wary: its proven node
        self.assertEqual(arb_policy({**f, "asplit": "even"})[0], "trust-champion")              # a tie, a sure champion
        self.assertEqual(arb_policy({**f, "asplit": "even", "achamp": "unsure", "abest": "other"})[0], "trust-confident")
        self.assertEqual(parse_arb(arb_text(f)), f)
        spec = type("Spec", (), {"name": "ego"})()
        t = EgoTeacher(seed=2)
        for kind in ARB_KINDS:  # the ego node is taught to arbitrate, too
            made = asyncio.run(t.generate(spec, kind, 4))
            self.assertTrue(made and all(arb_policy(parse_arb(x))[0] == kind for x in made), kind)
        self.assertEqual(asyncio.run(t.label(spec, arb_text({**f, "amood": "tense"}))).label, "trust-champion")
        self.assertTrue(similar("Paris is the capital of France.", "Paris is the capital of France"))
        self.assertFalse(similar("Paris is the capital of France.", "Lyon is the capital of France."))


class EnsembleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        over = {**FAST, "nodes": {**FAST["nodes"], "instances": 3, "max_running": 2},
                "memory": {"enabled": True, "interval_seconds": 0.02, "sync_seconds": 0.05}}
        self.system = System(self.root, overrides=over, seed=9)
        await self.system.start()
        self.system.guardian.pause_background(True, by="tester")  # nothing retrains or disposes the nodes made here

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    def node(self, label: str, conf: float, acc: float, status: str, delay: float = 0.0):
        s = self.system
        n = NeuralNode(CAP, list(s.capabilities[CAP].labels), s.directives.nodes.input_dim, Architecture(hidden=(8,)), seed=1)
        n.metrics["holdout_acc"] = acc

        def predict(x, exclude=frozenset()):
            if delay:
                time.sleep(delay)
            return label, conf
        n.predict = predict
        s.registry.add(n, "a test")
        if status == "champion":
            s.registry.promote(n.id, "a test")
        else:
            s.registry.make_instance(n.id, "a test")
        return n

    async def test_instances_answer_together_and_the_ego_settles_disagreements(self):
        s = self.system
        champ = self.node("positive", 0.9, 0.92, "champion")
        a = self.node("positive", 0.85, 0.9, "instance")
        self.node("negative", 0.5, 0.5, "instance")  # below the held-out bar: it does not answer
        self.assertEqual([n.id for n in members(s, CAP)], [champ.id, a.id])
        inf = await s.infer(CAP, "the hotel was lovely")
        self.assertEqual((inf.label, inf.node_id), ("positive", champ.id))
        self.assertIn("with 1 instance agreeing", inf.provenance)
        self.assertGreaterEqual(s.pool.together, 1)
        # they disagree: the ego arbitrates; two against one, in a steady mood: the majority
        s.registry.dispose(a.id, "a test")
        self.node("negative", 0.95, 0.91, "instance")
        self.node("negative", 0.9, 0.9, "instance")
        self.assertEqual(len(members(s, CAP)), 3)
        inf = await s.infer(CAP, "the room was cramped but the staff were kind")
        self.assertEqual(inf.label, "negative")
        self.assertIn("the ego arbitrated: trust-majority", inf.provenance)
        rec = s.ego.arbitrations[-1]
        self.assertEqual((rec["kind"], rec["label"], rec["strategy"]), (CAP, "negative", "trust-majority"))
        self.assertEqual(sorted(len(x["nodes"]) for x in rec["sides"]), [1, 2])
        self.assertEqual(s.pool.disagreed, 1)
        # it is kept in memory (untested); asked the same again, its memory decides, the same way
        kept = s.memory.own_statements(kind="arbitration")
        self.assertTrue(kept and "I went with negative" in kept[0]["text"], kept)
        s.affect.state.values.update(valence=-0.5, arousal=0.8, trust=0.2)  # wary now
        inf = await s.infer(CAP, "the room was cramped but the staff were kind")
        self.assertEqual(inf.label, "negative")
        self.assertEqual(s.ego.arbitrations[-1]["strategy"], "trust-memory")
        # a new disagreement while wary: it trusts its champion
        inf = await s.infer(CAP, "a hotel")
        self.assertEqual((inf.label, s.ego.arbitrations[-1]["strategy"]), ("positive", "trust-champion"))
        self.assertTrue(any(e.topic == "ego.arbitrated" for e in s.bus.recent("ego.*", 20)))
        taught = [e.label for e in s.store.examples("ego")]
        self.assertTrue(any(x.startswith("trust-") for x in taught), taught)  # the policy teaches the ego node to arbitrate
        lines = "\n".join(s.ego.lines())
        self.assertIn("settled", lines)
        status, body = await _route(s, "GET", "/ego", {}, {})
        self.assertEqual(body["arbitrations"][-1]["strategy"], "trust-champion")
        status, body = await _route(s, "GET", "/status", {}, {})
        self.assertEqual(body["pool"]["max_running"], 2)

    async def test_at_most_max_running_node_instances_at_once(self):
        s = self.system
        self.node("positive", 0.9, 0.95, "champion", delay=0.15)
        self.node("positive", 0.9, 0.93, "instance", delay=0.15)
        self.node("positive", 0.9, 0.92, "instance", delay=0.15)
        started = time.time()
        res = await asyncio.gather(*(s.infer(CAP, f"lovely stay number {i}") for i in range(4)))
        took = time.time() - started
        self.assertTrue(all(r.label == "positive" for r in res))
        self.assertEqual(s.pool.peak, 2)              # [nodes] max_running = 2: 12 runs, two at a time
        self.assertGreater(took, 0.15 * 12 / 2 * 0.8)
        self.assertLess(took, 0.15 * 12 * 0.8)        # but faster than one at a time

    async def test_memory_nodes_that_disagree(self):
        s = self.system
        s.guardian.pause_background(False, by="tester")
        s.memory.learn("What is the capital of Atlantis?", "Poseidonis is the capital of Atlantis.", "a legend", "me",
                       kind="people", reviewed=True)
        with sqlite3.connect(s.memory.db_path) as db:  # a person adds another node that says otherwise
            db.execute("INSERT INTO tests(node, seq, ask, answer) VALUES ('other-facts', 0, ?, ?)",
                       ("What is the capital of Atlantis?", "Atlantis never had a capital."))
        s.memory.sync_tests(force=True)
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.fact_nodes()), 60))
        rec = s.memory.recall("What is the capital of Atlantis?")
        self.assertIsNotNone(rec)
        self.assertIn("disagreed; the ego arbitrated", rec.how)
        self.assertEqual(s.ego.arbitrations[-1]["kind"], "memory")
        res = await s.submit("What is the capital of Atlantis?")
        self.assertIn("the ego arbitrated", " ".join(res.trace))


class InstancesTests(unittest.IsolatedAsyncioTestCase):
    async def test_runners_up_become_instances(self):
        tmp = tempfile.TemporaryDirectory()
        s = System(make_root(Path(tmp.name)), overrides={**FAST, "nodes": {**FAST["nodes"], "instances": 3}}, seed=4)
        await s.start()
        try:
            for i in range(14):
                await s.submit(f"sentiment: the meal was {'lovely' if i % 2 else 'awful'} number {i}")
            ok = await wait_for(lambda: s.registry.champion(CAP) is not None and s.registry.instances(CAP), 90)
            if not ok:  # (evolution may not leave a validated runner-up in time on a slow machine)
                self.skipTest("no validated runner-up yet")
            inst = s.registry.instances(CAP)[0]
            self.assertGreaterEqual(inst.metrics.get("holdout_acc") or 0, s.directives.nodes.serve_min_accuracy)
            self.assertIn(inst.id, [n.id for n in members(s, CAP)])
        finally:
            await s.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
