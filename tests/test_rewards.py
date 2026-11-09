"""Rewards and punishments (rewards.py): people reward or punish anything it does, or a direction in words; it
feels them, remembers them, its reward node learns from them, and they pull on what it chooses next."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.agency import MODES
from neuromesh.rewards import domain_of
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for


class RewardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        over = {**FAST, "memory": {"enabled": True, "interval_seconds": 0.05}, "temperament": {"enabled": True}}
        self.system = System(make_root(Path(self.tmp.name)), overrides=over, seed=11)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_anything_can_be_rewarded_or_punished(self):
        s = self.system
        self.assertEqual(domain_of("ego.spoke", {"kind": "curious"}), ("ego", "curious"))
        self.assertEqual(domain_of("agency.decided", {"mode": "research"}), ("initiative", "research"))
        res = await s.submit("sentiment: the staff were wonderful and kind")
        before = s.affect.state.values["valence"]
        ok, msg, rec = s.rewards.give("last", 2, "tester", "a good reading")
        self.assertTrue(ok, msg)
        self.assertEqual((rec["target"]["kind"], rec["domain"]), ("order", "answer"))
        self.assertGreater(s.affect.state.values["valence"], before)                # it feels it
        self.assertTrue(any(e.source == "reward" for e in s.store.examples("dispatch")))  # its route is reinforced
        self.assertIn(("reward", rec["text"][:300]), [(c, t) for (c, t) in s.teachers.human.answers])  # its reward node learns
        own = s.memory.own_statements(kind="reward")
        self.assertTrue(own and "tester rewarded me" in own[0]["text"], own)       # its ego remembers it
        self.assertGreater(s.rewards.direction("answer", res.route), 0)
        # a line of its activity log
        seq = max(i for i, e in s.log.since(0, 100000)[1] if not e.topic.startswith("reward.") and e.data.get("kind") not in ("reward", "punishment"))
        ok, msg, rec = s.rewards.give(f"log:{seq}", -1, "tester", "not that")
        self.assertTrue(ok, msg)
        self.assertEqual(rec["target"]["kind"], "log")
        # a node, a tool, and a direction in words
        champ = next((n for n in s.registry.live() if n.status in ("champion", "candidate")), None)
        if champ is not None:
            ok, msg, rec = s.rewards.give(champ.id, 3, "tester")
            self.assertTrue(ok)
            self.assertEqual(champ.metrics["reward"], 3)
        ok, msg, rec = s.rewards.give("calculator", -2, "tester", "work it out yourself")
        self.assertTrue(ok)
        self.assertLess(s.rewards.direction("tool", "calculator"), -0.4)
        ok, msg, rec = s.rewards.give("reading about astronomy", 3, "tester")
        self.assertEqual(rec["target"]["kind"], "direction")
        # who may give them, and how much
        self.assertFalse(s.rewards.give("last", 1, "teacher:ollama")[0])           # only people (P23)
        self.assertFalse(s.rewards.give("last", 0, "tester")[0])
        ok, msg, rec = s.rewards.give("last", 9, "tester")
        self.assertEqual(rec["value"], 3)                                           # at most 3
        self.assertFalse(s.rewards.give("ord-nope", 1, "tester", kind="order")[0])
        seq = max(i for i, e in s.log.since(0, 100000)[1] if e.topic == "reward.given")
        self.assertIn("a person gave", s.rewards.give(f"log:{seq}", 1, "tester")[1])  # not a reward for a reward
        # fading
        k = "tool|calculator"
        s.rewards.ledger[k]["ts"] -= s.rewards.pol.half_life_days * 86400
        self.assertAlmostEqual(s.rewards._decayed(s.rewards.ledger[k]), -1.0, places=2)
        status, body = await _route(s, "POST", "/reward", {}, {"target": "last", "value": -1, "why": "too long", "by": "tester"})
        self.assertEqual(status, 200)
        status, body = await _route(s, "GET", "/rewards", {}, {})
        self.assertGreaterEqual(body["count"], 6)
        self.assertTrue(any(d["domain"] == "direction" for d in body["directions"]))
        self.assertTrue(any(e.topic == "reward.given" for e in s.bus.recent("reward.*", 20)))
        self.assertIn("Rewards:", "\n".join(s.rewards.lines()))

    async def test_they_pull_on_what_it_chooses(self):
        s = self.system
        # its initiative: a punished mode loses out to a rewarded one
        from neuromesh.agency import reward_pull
        f = s.director.situation()["flags"]
        s.rewards.ledger["initiative|rest"] = {"score": -9.0, "n": 1, "ts": time.time(), "text": "rest"}
        s.rewards.ledger["initiative|examine"] = {"score": 9.0, "n": 1, "ts": time.time(), "text": "examine"}
        mode, why = reward_pull(f, {"rest", "examine", "research"}, "rest", s.rewards)
        self.assertEqual(mode, "examine")
        self.assertIn("people encouraged examine", why)
        self.assertIsNone(reward_pull(f, {"rest"}, "rest", s.rewards))           # nothing else it could do
        self.assertLess(s.rewards.score("initiative", "rest"), -0.9)
        # the tool switch, deciding: people discouraged the calculator
        self.assertTrue(await wait_for(lambda: s.tools.tools.get("calculator") is not None
                                       and s.tools.tools["calculator"].state == "active", 20))
        calc = s.tools.tools["calculator"]
        s.rewards.give("calculator", -3, "tester")
        s.tool_switch.other_way = lambda text: "teacher"
        use, why, _ = s.tool_switch.decide(calc, "what is 6 * 7")
        self.assertFalse(use)
        self.assertIn("people discouraged", why)
        s.rewards.give("calculator", 3, "tester")
        s.rewards.give("calculator", 3, "tester")
        use, why, _ = s.tool_switch.decide(calc, "what is 6 * 7")
        self.assertTrue(use)
        # the idle coin leans towards sleep when people reward sleeping
        s.rewards.ledger["dreams|sleep"] = {"score": 9.0, "n": 1, "ts": time.time(), "text": "sleep"}
        flips = []
        s.dreams.pol.objective_chance = 0.5
        s.directives.rewards.coin_weight = 0.45
        for _ in range(4):
            s.dreamer.rng.seed(3)
            flips.append((await s.dreamer.flip())["outcome"])
            s.dreams.wake("a test") if getattr(s.dreams, "asleep", False) else None
        self.assertNotIn("objective", flips)
        # the ego keeps to itself a kind of remark people punished
        s.rewards.ledger["ego|curious"] = {"score": -9.0, "n": 1, "ts": time.time(), "text": "curious"}
        self.assertLess(s.rewards.score("ego", "curious"), -0.4)


if __name__ == "__main__":
    unittest.main()
