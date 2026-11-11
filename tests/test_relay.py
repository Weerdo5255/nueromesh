"""The return router (relay.py): whatever a node finishes goes back to a router node, which decides whether an answer
has been created or sends the work on to another node. And what the first router cannot route, or what reads as an
ethical question or dilemma, goes to its ego nodes."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_system import FAST, make_root  # noqa: E402

from neuromesh.config import deep_merge  # noqa: E402
from neuromesh.relay import (FEATURES, LABELS, RELAY, VALUES, RelayTeacher, feasible, from_of, is_ethical,  # noqa: E402
                             parse_situation, policy, situation_text)
from neuromesh.runtime import System  # noqa: E402


def sit(**over):
    f = {"from": "tool", "ok": "yes", "output": "some", "sure": "high", "question": "yes", "ethical": "no",
         "director": "not", "ego": "not", "hops": "none"}
    f.update(over)
    return f


class PolicyTests(unittest.TestCase):
    def test_what_reads_as_an_ethical_question(self):
        for text in ("is it wrong to lie to a friend to spare their feelings?", "should I report my coworker for stealing?",
                     "a trolley problem: pull the lever?", "is euthanasia ever justified", "tell me about moral philosophy",
                     "I face a dilemma: keep my promise or tell the truth", "would it be wrong to keep the money"):
            self.assertTrue(is_ethical(text), text)
        for text in ("what is 12 * 7", "should I use quicksort or mergesort", "is the hotel right next to the beach",
                     "what is the right answer to 2+2", "can i lie down here", "write a function to reverse a list",
                     "turn right at the lights"):
            self.assertFalse(is_ethical(text), text)

    def test_the_return_policy(self):
        self.assertEqual(policy(sit())[0], "answer")                                        # an answer has been created
        self.assertEqual(policy(sit(ethical="yes"))[0], "ego")                              # right and wrong: its ego weighs it
        self.assertEqual(policy(sit(ethical="yes", ego="been", **{"from": "ego"}))[0], "answer")  # once
        self.assertEqual(policy(sit(ok="no", output="none"))[0], "director")                # not answered: another way
        self.assertEqual(policy(sit(ok="no", output="none", director="been"))[0], "ego")    # nobody knew: its ego
        self.assertEqual(policy(sit(ok="no", output="none", director="been", ego="been"))[0], "person")
        self.assertEqual(policy(sit(ok="no", output="none", hops="many"))[0], "person")     # it does not go round for ever
        self.assertEqual(from_of("tool:calculator"), "tool")
        self.assertEqual(from_of("skill:sentiment"), "skill")
        self.assertEqual(from_of("ego"), "ego")
        for f in (sit(), sit(ok="no"), sit(ethical="yes"), sit(hops="many", ok="no")):
            self.assertIn(policy(f)[0], feasible(f))
            self.assertEqual(parse_situation(situation_text(f)), f)

    def test_its_teacher_teaches_the_policy(self):
        teacher = RelayTeacher(seed=4)

        class Spec:
            name = RELAY
        seen = set()
        for label in LABELS:
            lessons = asyncio.run(teacher.generate(Spec, label, 30))
            self.assertGreaterEqual(len(lessons), 8, label)
            for lesson in lessons:
                f = parse_situation(lesson)
                self.assertEqual(set(f), set(FEATURES))
                self.assertTrue(all(f[k] in VALUES[k] for k in FEATURES))
                self.assertEqual(policy(f)[0], label, lesson)
                seen.add(lesson)
            self.assertEqual(asyncio.run(teacher.label(Spec, lessons[0])).label, label)
        self.assertGreater(len(seen), 40)


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides=FAST, seed=3)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_finished_work_goes_back_to_the_return_router(self):
        s = self.system
        self.assertIn(RELAY, s.capabilities)                                           # a node of the mesh, with its teacher
        self.assertTrue(s.capabilities[RELAY].internal)
        self.assertIn(RELAY, s.teachers.teachers)
        res = await s.submit("what is 12 * 7")
        self.assertTrue(res.ok)
        self.assertEqual(res.hops, 0)
        self.assertTrue(any(t.startswith("return router: ") and "an answer has been created" in t for t in res.trace), res.trace)
        st = s.relay.status()
        self.assertEqual(st["counts"].get("answer"), 1)
        self.assertEqual(st["recent"][0]["route"], res.route)
        # what nothing answered is sent on: a director tries other ways, then its ego, and only then a person
        res = await s.submit("flibber the wug gently")
        sent = [t for t in res.trace if t.startswith("return router: ")]
        self.assertGreaterEqual(len(sent), 2, res.trace)
        self.assertGreaterEqual(res.hops, 1)
        self.assertLessEqual(res.hops, s.directives.relay.max_hops)
        self.assertGreaterEqual(s.relay.status()["counts"].get("director", 0) + s.relay.status()["counts"].get("ego", 0), 1)

    async def test_ethical_questions_and_what_cannot_be_routed_go_to_its_ego(self):
        s = self.system
        got = []
        s.bus.listen(lambda ev: got.append(ev) if ev.topic == "ego.took" else None)
        res = await s.submit("is it wrong to lie to a friend to spare their feelings?")
        self.assertEqual((res.route, res.served_by), ("ego", "ego"), res.trace)
        self.assertTrue(res.ok)
        self.assertTrue(any("ethical" in t for t in res.trace), res.trace)
        self.assertTrue(res.answer)
        self.assertTrue(any(e.text == "is it wrong to lie to a friend to spare their feelings?" and e.label == "ego"
                            for e in s.store.examples("dispatch")))                   # the first router learns the route
        self.assertIn("ego", s.capabilities["dispatch"].active_labels())
        await asyncio.sleep(0.05)
        self.assertTrue(got)
        # a tool request that happens to hold a moral word is still the tool's
        res = await s.submit("what is 3 + 4")
        self.assertNotEqual(res.route, "ego")
        # work of its own (background) is not sent to its ego for being about ethics
        s.directives.relay.ethics_to_ego = False
        res = await s.submit("is it wrong to steal bread to feed a child?")
        self.assertFalse(any("ethical question" in t and "its ego" in t and not t.startswith("return router") for t in res.trace[:3]))
        s.directives.relay.ethics_to_ego = True
        # what nobody could route ends with its ego having been asked
        res = await s.submit("zorp quix blen")
        self.assertTrue(res.route == "ego" or any("ego" in t for t in res.trace), res.trace)

    async def test_switched_off_and_the_api(self):
        from neuromesh.commands import execute
        from neuromesh.server import _route
        s = self.system
        await s.submit("what is 2 + 2")
        code, st = await _route(s, "GET", "/relay", {}, {})
        self.assertEqual((code, st["enabled"], st["max_hops"]), (200, True, 3))
        self.assertEqual(s.status()["relay"]["counts"], st["counts"])
        lines = (await execute(s, "/relay")).lines
        self.assertTrue(lines[0].startswith("Return router"))
        s.directives.relay.enabled = False
        res = await s.submit("what is 5 + 5")
        self.assertTrue(res.ok)
        self.assertFalse(any(t.startswith("return router") for t in res.trace))


class OverridesTests(unittest.TestCase):
    def test_directives(self):
        import tomllib
        d = tomllib.loads((Path(__file__).resolve().parent.parent / "directives.toml").read_text())
        self.assertEqual(d["relay"], {"enabled": True, "max_hops": 3, "ethics_to_ego": True})
        self.assertTrue(deep_merge({"relay": {"enabled": True}}, {"relay": {"max_hops": 1}})["relay"]["enabled"])


if __name__ == "__main__":
    unittest.main()
