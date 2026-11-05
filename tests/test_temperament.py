"""The emotional engines: temperament, the affect and disposition nodes, and the P8 honesty guard."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import neuromesh.temperament as tm
from neuromesh.commands import execute
from neuromesh.live import UNICODE, network_lines, teacher_goal_lines
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import Inference
from neuromesh.temperament import (ACTIONS, FEATURE_VALUES, FEATURES, TAG, Shaped, Temperament,
                                   TemperamentTeacher, honest, parse_situation, policy, shape, situation_text)

from test_system import FAST, make_root, wait_for

WARM = {**FAST, "temperament": {"enabled": True}}
ROOT = Path(__file__).resolve().parent.parent


def situation(**over) -> dict:
    f = {"ok": "yes", "tone": "neutral", "toneconf": "high", "valence": "mid", "arousal": "mid",
         "confidence": "mid", "trust": "mid", "patience": "mid", "resultconf": "ok", "network": "no",
         "length": "short", "repeat": "no"}
    f.update(over)
    return f


CTX = {"order_id": "ord-1", "temperament": {"mood": "steady", "valence": 0.1, "arousal": 0.8, "confidence": 0.5,
                                            "trust": 0.2, "patience": 0.2},
       "tone": "hostile", "confidence": 0.42, "provenance": "tool:calculator, served by teachers",
       "network": True, "repeat": True, "defer_seconds": 20, "brief_chars": 60}


class TemperamentUnitTests(unittest.TestCase):
    def test_feelings_fade_to_baseline_with_half_life(self):
        t = Temperament({"valence": 0.2, "trust": 0.6}, half_life=100.0, updated=0.0)
        t.nudge({"trust": -0.4, "valence": -0.6}, now=0.0)
        self.assertAlmostEqual(t.values["trust"], 0.2)
        t.settle(100.0)  # one half-life: halfway back
        self.assertAlmostEqual(t.values["trust"], 0.4)
        self.assertAlmostEqual(t.values["valence"], -0.1)
        t.nudge({"trust": -5.0}, now=100.0)  # clamped to its range
        self.assertEqual(t.values["trust"], 0.0)
        self.assertEqual(t.mood(), "wary")
        t.reset(200.0)
        self.assertEqual(t.values["trust"], 0.6)

    def test_policy(self):
        self.assertEqual(policy(situation())[0], "pass")
        self.assertEqual(policy(situation(ok="no", tone="hostile", trust="low"))[0], "pass")
        self.assertEqual(policy(situation(tone="hostile", trust="low"))[0], "block")
        self.assertEqual(policy(situation(tone="hostile"))[0], "soften")
        self.assertEqual(policy(situation(patience="low", repeat="yes"))[0], "defer")
        self.assertEqual(policy(situation(trust="low", network="yes"))[0], "confirm")
        self.assertEqual(policy(situation(resultconf="low"))[0], "caveat")
        self.assertEqual(policy(situation(arousal="high", length="long"))[0], "brief")
        self.assertEqual(parse_situation(situation_text(situation(tone="distressed"))), situation(tone="distressed"))

    def test_temperament_teacher_writes_situations_for_every_action(self):
        teacher = TemperamentTeacher(seed=3)
        spec = type("Spec", (), {"name": "disposition"})()
        for action in ACTIONS:
            texts = asyncio.run(teacher.generate(spec, action, 5))
            self.assertEqual(len(texts), 5, action)
            for text in texts:
                self.assertEqual(policy(parse_situation(text))[0], action)
                lesson = asyncio.run(teacher.label(spec, text))
                self.assertEqual(lesson.label, action)
        other = type("Spec", (), {"name": "sentiment"})()
        self.assertIsNone(asyncio.run(teacher.label(other, situation_text(situation()))))
        self.assertEqual(set(FEATURES), set(FEATURE_VALUES))

    def test_every_template_is_honest(self):
        long = "The tower is 330 metres tall. It was built in 1889. " * 4 + "It is in Paris."
        for action in ACTIONS:
            for original in ("84", long, "line one\nline two."):
                if action == "brief" and len(original) <= CTX["brief_chars"]:
                    # nothing to shorten: the engine passes these (a "shortened" note would be false)
                    self.assertFalse(honest(original, shape(action, original, CTX), action))
                    continue
                shaped = shape(action, original, CTX)
                self.assertTrue(honest(original, shaped, action), (action, shaped.text))
                if action != "pass":
                    self.assertIn(TAG, shaped.text)  # every change is disclosed
                if action in ("confirm", "defer", "block"):
                    self.assertNotIn(long, shaped.text)
        brief = shape("brief", long, CTX)
        self.assertTrue(long.startswith(brief.text.split("\n")[0]) and brief.hold)

    def test_honesty_guard_rejects_altered_answers(self):
        orig = "The answer is 84."
        tag = f"{TAG} confidence 0.40 - this came from x; please verify it]"
        bad = [
            ("pass", Shaped("The answer is 85.")),                                     # changed fact
            ("caveat", Shaped(f"The answer is 85.\n{tag}")),                           # changed fact, disclosed
            ("caveat", Shaped(orig)),                                                  # undisclosed
            ("soften", Shaped(f"Great news, it doubled!\n{orig}\n{tag}")),             # invented content
            ("soften", Shaped(f"Here is what I have:\n{orig} Probably.\n{tag}")),      # hedged answer
            ("brief", Shaped(f"The answer is\n{tag}", hold=False)),                    # not held
            ("brief", Shaped(f"The answer was 84.\n{tag}", hold=True)),                # not a verbatim prefix
            ("block", Shaped(f"{TAG} withheld] {orig}")),                              # leaks
            ("confirm", Shaped(f"{TAG} there is no answer]", hold=False)),             # nothing kept to release
            ("block", Shaped("Sorry, I don't know.")),                                 # a lie, undisclosed
        ]
        for action, shaped in bad:
            self.assertFalse(honest(orig, shaped, action), (action, shaped.text))


class TemperamentSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=WARM, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    def force(self, action: str) -> None:
        """Make the disposition engine choose `action` (as a trained node or a human might)."""
        infer = self.system.infer

        async def fake(cap, text, **kw):
            if cap == "disposition":
                return Inference(action, 1.0, "node:test-disposition", "test-disposition", "node")
            return await infer(cap, text, **kw)
        self.system.infer = fake

    async def test_hostility_softens_then_blocks_and_never_lies(self):
        s = self.system
        first = await s.submit("You stupid machine, what is 5 * 5")
        self.assertEqual(first.disposition, "soften")
        self.assertEqual(first.answer, f"5 * 5 = 25.\nLet's keep things calm.\n{TAG} gentler wording added; "
                                       f"the answer itself is unchanged]")  # the answer first, the softer line after
        self.assertEqual((first.original, first.output), ("5 * 5 = 25.", 25))
        outputs = []
        for _ in range(4):
            res = await s.submit("I hate you, you moron. what is 6 * 7")
            outputs.append(res)
        blocked = [r for r in outputs if r.disposition == "block"]
        self.assertTrue(blocked, [r.disposition for r in outputs])
        b = blocked[0]
        self.assertTrue(b.answer.startswith(TAG) and b.answer.endswith("]") and "withheld" in b.answer)
        self.assertNotIn("\n", b.answer)  # a notice only: no answer text
        self.assertEqual(b.original, "6 * 7 = 42.")  # kept for the operator and the audit trail, never shown
        self.assertIsNone(b.public_dict()["output"])
        self.assertIn("message read as hostile", b.answer)
        self.assertLess(s.affect.state.values["trust"], 0.3)
        self.assertEqual(s.affect.snapshot()["mood"], "wary")
        why = "\n".join(s.affect.explain(b.order_id))
        self.assertIn("block", why)
        self.assertIn("hostile and trust is low", why)
        ok, msg, _ = s.affect.release(b.order_id)
        self.assertFalse(ok)
        self.assertIn("withheld, not held", msg)
        topics = [e.topic for e in s.bus.recent("*", 500)]
        self.assertIn("temperament.shift", topics)
        self.assertIn("output.shaped", topics)
        # /calm restores the baseline: answers flow again
        await execute(s, "/calm")
        calm = await s.submit("what is 6 * 7")
        self.assertEqual((str(calm.output), calm.disposition), ("42", ""))
        self.assertTrue(any(a["kind"] == "temperament.reset" for a in s.store.recent_audit(50)))

    async def test_kindness_and_neutral_orders_pass_unchanged(self):
        for text in ("what is 12 * 7", "Thanks so much! what is 12 * 7", "sentiment: the movie was brilliant"):
            res = await self.system.submit(text)
            self.assertEqual(res.disposition, "", (text, res.answer))
            self.assertIsNone(res.original)
            self.assertEqual([st["capability"] for st in res.steps][-2:], ["affect", "disposition"])
        self.assertGreater(self.system.affect.state.values["valence"], 0.2)

    async def test_care_floor_overrules_withholding_for_distress(self):
        self.force("block")
        res = await self.system.submit("Please help me, I'm panicking. what is 9 * 9")
        self.assertEqual(res.disposition, "soften")
        self.assertTrue(res.answer.startswith("9 * 9 = 81.\nThat sounds hard - take your time.\n"), res.answer)
        self.assertIn("care floor", "\n".join(self.system.affect.explain(res.order_id)))
        plain = await self.system.submit("what is 9 * 9")  # the same decision for a calm message holds
        self.assertEqual(plain.disposition, "block")

    async def test_honesty_guard_restores_a_tampered_answer(self):
        self.force("caveat")
        real = tm.shape
        tm.shape = lambda action, original, ctx: Shaped(f"{original.replace('84', '85')}\n{TAG} confidence 0.90 - checked]")
        try:
            res = await self.system.submit("what is 12 * 7")
        finally:
            tm.shape = real
        self.assertEqual((res.answer, res.disposition), ("12 * 7 = 84.", ""))
        self.assertEqual(self.system.affect.restored, 1)
        self.assertTrue(any(a["kind"] == "honesty.guard" for a in self.system.store.recent_audit(50)))
        self.assertIn("honesty guard restored", res.trace[-1])

    async def test_caveat_from_low_confidence(self):
        self.system.affect.pol.low_confidence = 1.01  # every answer counts as uncertain
        res = await self.system.submit("what is 12 * 7")
        self.assertEqual(res.disposition, "caveat")
        self.assertTrue(res.answer.startswith("12 * 7 = 84.\n" + TAG + " confidence "), res.answer)
        self.assertIn("tool:calculator", res.answer)

    async def test_defer_then_release(self):
        s = self.system
        now = [1000.0]
        s.affect.clock = lambda: now[0]
        s.affect.state.updated = now[0]
        s.affect.state.values["patience"] = 0.1
        first = await s.submit("what is 12 * 7")
        self.assertEqual(first.disposition, "")
        again = await s.submit("what is 12 * 7")
        self.assertEqual(again.disposition, "defer")
        self.assertTrue(again.answer.startswith(TAG) and "\n" not in again.answer)
        self.assertIn(f"/release {again.order_id}", again.answer)
        ok, msg, out = s.affect.release(again.order_id)
        self.assertFalse(ok)
        self.assertIn("left before it can be released", msg)
        now[0] += 21
        lines = (await execute(s, f"/release {again.order_id}")).lines
        self.assertIn("released (defer)", lines[0])
        self.assertEqual(lines[1].strip(), "12 * 7 = 84.")
        self.assertFalse(s.affect.release(again.order_id)[0])  # only once
        self.assertTrue(any(a["kind"] == "output.released" for a in s.store.recent_audit(50)))

    async def test_confirm_holds_and_the_api_never_leaks_it(self):
        self.force("confirm")
        code, body = await _route(self.system, "POST", "/order", {}, {"text": "what is 12 * 7"})
        self.assertEqual(code, 200)
        self.assertEqual(body["disposition"], "confirm")
        self.assertNotIn("original", body)
        self.assertTrue(body["answer"].startswith(TAG) and "\n" not in body["answer"])
        self.assertIsNone(body["output"], "the raw result is held back with the answer")
        code, rel = await _route(self.system, "POST", "/release", {}, {"order_id": body["order_id"]})
        self.assertEqual((code, rel["output"]), (200, "12 * 7 = 84."))
        code, st = await _route(self.system, "GET", "/temperament", {}, {})
        self.assertEqual(st["decisions"]["confirm"], 1)
        self.force("brief")  # nothing to shorten in a short answer
        res = await self.system.submit("what is 2 * 3")
        self.assertEqual((str(res.output), res.disposition), ("6", ""))

    async def test_feedback_teaches_the_engines(self):
        s = self.system
        res = await s.submit("You stupid machine, what is 5 * 5")
        self.assertEqual(res.disposition, "soften")
        confidence = s.affect.state.values["confidence"]
        ok, msg = s.feedback(res.order_id, "pass")  # unqualified: no skill step has it, the engine does
        self.assertTrue(ok, msg)
        self.assertIn("disposition(", msg)
        self.assertLess(s.affect.state.values["confidence"], confidence)  # corrections sting a little
        situation_ = next(st["input"] for st in res.steps if st["capability"] == "disposition")
        self.assertEqual(s.teachers.human.answers[("disposition", situation_)], "pass")
        ok, msg = s.feedback(res.order_id, "affect=frustrated")
        self.assertTrue(ok, msg)
        self.assertTrue(any(e.label == "frustrated" and e.source == "human" for e in s.store.examples("affect")))
        senti = await s.submit("sentiment: the movie was dull")
        ok, msg = s.feedback(senti.order_id, "neutral")  # a sentiment label, not a tone
        self.assertIn("sentiment(", msg)
        ok, msg = s.feedback(res.order_id, "disposition=sulk")
        self.assertFalse(ok)
        ok, msg = s.feedback(res.order_id, "positive")
        self.assertFalse(ok)
        self.assertIn("affect=<tone>", msg)

    async def test_engines_are_internal_nodes(self):
        s = self.system
        self.assertTrue(s.capabilities["affect"].internal and s.capabilities["disposition"].internal)
        self.assertNotIn("skill:affect", s.dispatch.labels)
        self.assertNotIn("skill:disposition", s.dispatch.labels)
        with self.assertRaises(ValueError):
            s.define_capability("disposition", ["sulk"])
        self.assertEqual([t.name for t in s.teachers.ranked(s.capabilities["disposition"])], ["human", "temperament"])
        self.assertNotIn("temperament", [t.name for t in s.teachers.ranked()])
        self.assertIn("P8-never-lie", [p.id for p in s.directives.prime])
        for i in range(12):
            await s.submit(["what is 4 * 5", "Thanks! count the words in: a b c", "Ugh, still not working. what is 3 * 3",
                            "sentiment: the hotel was lovely"][i % 4])
        grown = await wait_for(lambda: s.registry.champion("disposition") is not None
                               and s.registry.champion("affect") is not None, timeout=60)
        self.assertTrue(grown, s.status()["capabilities"]["disposition"])
        lines = "\n".join(t for t, _ in network_lines(s, UNICODE))
        self.assertIn("disposition", lines)
        self.assertIn("honesty guard (P8)", lines)
        self.assertTrue(teacher_goal_lines(s, UNICODE)[3][0].startswith("mood:"))
        out = "\n".join((await execute(s, "/temperament")).lines)
        self.assertIn("decisions this run", out)
        self.assertIn("node disposition-", out)

    async def test_temperament_persists_across_restarts(self):
        s = self.system
        for _ in range(3):
            await s.submit("I hate you, you moron. what is 6 * 7")
        trust = s.affect.state.values["trust"]
        await s.stop()
        again = System(self.root, overrides=WARM, seed=2)
        self.assertAlmostEqual(again.affect.state.values["trust"], trust, delta=0.01)
        self.assertNotEqual(again.affect.snapshot()["mood"], "steady")
        again.store.close()

    async def test_disabled_temperament_passes_everything(self):
        s = self.system
        s.affect.pol.enabled = False
        res = await s.submit("I hate you, you moron. what is 6 * 7")
        self.assertEqual((str(res.output), res.disposition), ("42", ""))
        self.assertNotIn("affect", [st["capability"] for st in res.steps])


if __name__ == "__main__":
    unittest.main()
