"""The ego (ego.py, P18): nodes wrapped around all the others. It reads the mood an order is taken in and, on
the way out, says something of its own - curious, a dream, its view, how it feels, a doubt - built from what is
so and labelled. What it is curious about becomes an objective sooner. And idle, it tosses a coin: an objective
75% of the time, sleep otherwise - unless sleep is turned off."""
import asyncio
import random
import tempfile
import unittest
from pathlib import Path

from neuromesh import anatomy, web
from neuromesh.commands import execute
from neuromesh.dreams import DreamBank
from neuromesh.ego import EGO, KINDS, EgoTeacher, feasible, first_person, parse_situation, policy, situation_text
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for


def sit(**kw):
    f = {"mood": "steady", "tone": "neutral", "outcome": "ok", "topic": "none", "dreamt": "no", "view": "none", "spoke": "not"}
    f.update(kw)
    return f


class PolicyTests(unittest.TestCase):
    def test_when_it_speaks_and_how(self):
        self.assertEqual(policy(sit())[0], "quiet")
        self.assertEqual(policy(sit(mood="upbeat"))[0], "feeling")                        # good spirits show
        self.assertEqual(policy(sit(topic="curious"))[0], "curious")
        self.assertEqual(policy(sit(topic="curious", mood="wary"))[0], "feeling")          # wary: not reaching out
        self.assertEqual(policy(sit(dreamt="yes", topic="curious"))[0], "dream")
        self.assertEqual(policy(sit(dreamt="yes", mood="tense"))[0], "feeling")
        self.assertEqual(policy(sit(view="likes"))[0], "opinion")
        self.assertEqual(policy(sit(view="dislikes", topic="curious"))[0], "opinion")      # a view of its own first
        self.assertEqual(policy(sit(outcome="failed"))[0], "doubt")
        self.assertEqual(policy(sit(outcome="failed", mood="upbeat", topic="curious"))[0], "curious")  # wants to find out
        self.assertEqual(policy(sit(outcome="unsure", mood="subdued", topic="curious"))[0], "doubt")
        # it keeps out of the way: it spoke a moment ago, the answer was held back, the person is upset
        for quiet in (sit(spoke="recent", topic="curious"), sit(outcome="held", view="likes"),
                      sit(tone="hostile", dreamt="yes"), sit(tone="distressed", outcome="failed"),
                      sit(tone="frustrated", view="likes")):
            self.assertEqual(policy(quiet)[0], "quiet", quiet)
        self.assertEqual(policy(sit(tone="frustrated", outcome="failed"))[0], "doubt")   # owning a failure is fine
        self.assertEqual(policy(sit(dreamt="yes", view="likes"), exclude=("dream",))[0], "opinion")
        self.assertEqual(feasible(sit()), {"quiet", "feeling"})
        self.assertEqual(feasible(sit(topic="new", dreamt="yes", view="dislikes", outcome="unsure")), set(KINDS))
        self.assertEqual(parse_situation(situation_text(sit(mood="content", view="likes"))), sit(mood="content", view="likes"))
        self.assertEqual(first_person("it holds no facts about it yet; a question it could not answer"),
                         "I hold no facts about it yet; a question I couldn't answer")

    def test_the_policy_teaches_the_ego_node(self):
        spec = type("Spec", (), {"name": EGO})()
        t = EgoTeacher(seed=1)
        lesson = asyncio.run(t.label(spec, situation_text(sit(dreamt="yes"))))
        self.assertEqual((lesson.label, lesson.source), ("dream", "ego"))
        for kind in KINDS:
            made = asyncio.run(t.generate(spec, kind, 5))
            self.assertTrue(made, kind)
            self.assertTrue(all(policy(parse_situation(x))[0] == kind for x in made))
        self.assertIsNone(asyncio.run(t.label(type("Spec", (), {"name": "agency"})(), situation_text(sit()))))


class EgoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from neuromesh.runtime import System
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.over = {**FAST, "memory": {"enabled": True}, "temperament": {"enabled": True}, "ego": {"quiet_for": 0},
                     "dreams": {"enabled": True, "check_every_seconds": 9999}}  # it dreams, but no idle coin here
        self.system = System(self.root, overrides=self.over, seed=11)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    def mood(self, **values):
        s = self.system
        s.affect.state.values.update(values)
        s.affect.state.updated = s.affect.clock()

    async def learned(self):
        s = self.system
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 60))

    async def test_it_wraps_every_order_and_says_what_it_is_curious_about(self):
        s = self.system
        self.mood(valence=0.6, arousal=0.75, trust=0.7)  # upbeat
        res = await s.submit("Who painted the Zorblax mural?")
        self.assertFalse(res.ok)
        self.assertTrue(res.trace[0].startswith("ego (the outer layer): took it in an upbeat mood") or
                        res.trace[0].startswith("ego (the outer layer): took it in a upbeat mood"), res.trace[0])
        self.assertEqual(res.steps[-1]["capability"], EGO)
        self.assertTrue(res.trace[-1].startswith("ego: "), res.trace[-1])
        self.assertIsNotNone(res.ego, res.trace)
        self.assertEqual(res.ego["kind"], "curious")  # it could not answer, and in good spirits it wants to find out
        self.assertIn("curious about Zorblax", res.ego["text"])
        self.assertIn("I couldn't answer that well, and I want to find out", res.ego["text"])
        self.assertNotIn(res.ego["text"], res.answer)  # beside the answer, never in it
        # what its ego is curious about becomes an area it wants to understand: an objective, sooner
        self.assertIn("zorblax", s.ego.fascinations())
        area = s.curiosity.area("Zorblax")
        self.assertEqual(area.why_interest[0], "its ego is curious about it")
        self.assertGreaterEqual(area.interest, s.directives.ego.fascination_weight)
        aims = s.director.aims()
        self.assertTrue(any("Zorblax" in a["text"] for a in aims), aims)
        self.assertEqual(s.directives.agency.max_self_objectives, 4)
        # the trace through the mesh: the ego first in and last out
        hops = anatomy.traces(s, 1)[0]["hops"]
        self.assertEqual((hops[0]["stage"], hops[-1]["stage"]), ("ego", "ego"))
        self.assertEqual(hops[1]["stage"], "director")
        self.assertTrue(any(e.topic == "ego.spoke" for e in s.bus.recent("ego.*", 20)))

    async def test_dreams_views_feelings_and_doubts(self):
        s = self.system
        s.memory.learn("What is the capital of France?", "Paris is the capital of France.", "a test", "me")
        await self.learned()
        # a dream about France that checked out on waking
        s.dreams.recent.append({"question": "Paris is the capital of what?", "answer": "France", "kind": "remembered",
                                "verdict": "correct", "truth": "France", "confidence": 0.8})
        res = await s.submit("What is the capital of France?")
        self.assertTrue(res.ok, res.note)
        self.assertEqual(res.ego["kind"], "dream", res.trace)
        self.assertIn('I asked myself "Paris is the capital of what?"', res.ego["text"])
        self.assertIn("checked out", res.ego["text"])
        # its view: you told it it got France right, twice
        s.dreams.recent.clear()
        for oid in ("o-a", "o-b"):
            s.corrections.marks[oid] = {"verdict": "right", "text": "What is the capital of France?"}
        v = s.ego.view("France")
        self.assertEqual(v["stance"], "likes")
        self.assertIn("you told me I got it right 2 times", v["for"])
        res = await s.submit("What is the capital of France?")
        self.assertEqual(res.ego["kind"], "opinion", res.trace)
        self.assertTrue(res.ego["text"].startswith("I like France"), res.ego["text"])
        # how it feels: in good spirits, about a sum (no topic)
        self.mood(valence=0.7, arousal=0.8, trust=0.7)
        res = await s.submit("what is 2 + 2?")
        self.assertEqual((res.output, res.ego["kind"]), (4, "feeling"))
        self.assertTrue(res.ego["text"].startswith("I'm in good spirits right now"), res.ego["text"])
        # a doubt, in a low mood
        self.mood(valence=-0.6, arousal=0.3, trust=0.6)
        res = await s.submit("Who invented the flumbrous engine?")
        self.assertEqual(res.ego["kind"], "doubt", res.trace)
        self.assertIn("I couldn't answer it", res.ego["text"])
        # it stays quiet for a while after speaking ([ego] quiet_for)
        s.directives.ego.quiet_for = 1
        res = await s.submit("Who invented the flumbrous engine?")
        self.assertIsNone(res.ego)
        self.assertIn("ego: stayed quiet (it spoke a moment ago", " ".join(res.trace))
        # remarks are its own: never stored as facts
        self.assertFalse(any("ego" in t.answer.lower() for n in s.memory.nodes.values() for t in n.facts()))

    async def test_questions_about_itself_the_console_and_the_api(self):
        s = self.system
        s.memory.learn("What is the capital of France?", "Paris is the capital of France.", "a test", "me")
        for oid in ("o-a", "o-b"):
            s.corrections.marks[oid] = {"verdict": "right", "text": "What is the capital of France?"}
        s.dreams.recent.append({"question": "Paris is the capital of what?", "answer": "France", "kind": "remembered",
                                "verdict": "correct", "truth": "France", "confidence": 0.8})
        s.ego.fascinate("Zorblax", "a test")
        self.mood(valence=0.5, arousal=0.3, trust=0.7)  # content
        res = await s.submit("Tell me about yourself")
        self.assertEqual((res.route, res.ok), ("ego", True))
        self.assertIsNone(res.ego)  # no remark on top of its own answer about itself
        for part in ("I'm NeuroMesh", "I'm content", "I'm curious about Zorblax", "I like France", "That's my ego"):
            self.assertIn(part, res.output)
        res = await s.submit("How are you feeling?")
        self.assertEqual(res.route, "ego")
        self.assertTrue(res.output.startswith("I'm content right now"), res.output)
        res = await s.submit("What did you dream about?")
        self.assertIn('I asked myself "Paris is the capital of what?"', res.output)
        self.assertIn("75% of the time I work on an objective", res.output)
        res = await s.submit("What do you like?")
        self.assertIn("I like France (you told me I got it right 2 times", res.output)
        lines = "\n".join((await execute(s, "/ego")).lines)
        self.assertIn("its ego: content", lines)
        self.assertIn("likes France", lines)
        out = await execute(s, "What is the capital of Atlantis?")
        self.assertTrue(any(line.startswith("   @ its ego (") for line in out.lines), out.lines)
        status, body = await _route(s, "GET", "/ego", {}, {})
        self.assertEqual((status, body["enabled"], body["mood"]), (200, True, "content"))
        self.assertEqual(body["views"]["likes"][0]["name"], "France")
        status, body = await _route(s, "POST", "/ego", {}, {"enabled": False, "by": "tester"})
        self.assertEqual((status, body["ok"]), (200, True))
        res = await s.submit("Who painted the Zorblax mural?")
        self.assertIsNone(res.ego)
        self.assertFalse(any(t.startswith("ego") for t in res.trace))
        res = await s.submit("How are you feeling?")
        self.assertNotEqual(res.route, "ego")  # off: someone else answers
        self.assertEqual((await _route(s, "POST", "/ego", {}, {}))[0], 400)
        self.assertIn("ego is on", (await execute(s, "/ego on")).lines[0])
        self.assertTrue(s.ego.enabled)
        self.assertTrue(any(a["kind"] == "ego.off" for a in s.store.recent_audit(50)))
        mesh = web.mesh(s)
        self.assertEqual(mesh["ego"]["mood"], "content")


class CoinFlipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from neuromesh.runtime import System
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.over = {**FAST, "memory": {"enabled": True},
                     "agency": {"enabled": True, "idle_seconds": 9999, "decide_every_seconds": 9999},
                     "dreams": {"enabled": True, "sleep_after_idle_seconds": 0.3, "check_every_seconds": 9999,
                                "min_material": 1, "sleep_max_seconds": 1}}  # the coin is tossed by hand, until a test lets it
        self.system = System(self.root, overrides=self.over, seed=13)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_three_times_in_four_it_works_on_an_objective(self):
        s, d = self.system, self.system.dreams
        self.assertEqual((d.pol.objective_chance, d.pol.sleep_chance, d.allow_sleep), (0.75, 1.0, True))
        calls = []

        async def worked():
            calls.append(1)
            return "objective", "stub"
        s.dreamer.attempt_objective = worked
        d.fall_asleep = lambda reason="idle", by="dreamer": (False, "stub")
        s.dreamer.rng = random.Random(4)
        outcomes = [(await s.dreamer.flip())["outcome"] for _ in range(400)]
        self.assertTrue(260 <= outcomes.count("objective") <= 340, outcomes.count("objective"))
        self.assertEqual(outcomes.count("objective"), len(calls))
        self.assertEqual(outcomes.count("objective") + outcomes.count("awake"), 400)
        self.assertEqual(d.stats["coin flips"], 400)

    async def test_an_objective_it_has_or_one_it_sets_itself_from_curiosity(self):
        s, d = self.system, self.system.dreams
        d.pol.objective_chance = 1.0
        obj = await s.objectives.add("understand volcanoes", by="tester")
        rec = await s.dreamer.flip()
        self.assertEqual(rec["outcome"], "objective", rec)
        self.assertIn(f"worked on {obj.id} (understand volcanoes", rec["detail"])
        self.assertGreaterEqual(obj.steps, 1)
        s.objectives.set_status(obj.id, "removed", by="tester")
        # none left: its initiative sets itself one from what its ego is curious about
        s.ego.fascinate("Krakatoa", "a test")
        rec = await s.dreamer.flip()
        self.assertEqual(rec["outcome"], "objective", rec)
        self.assertIn("set itself obj-", rec["detail"])
        mine = [o for o in s.objectives.items.values() if o.self_set]
        self.assertTrue(mine and "Krakatoa" in mine[0].text, [o.text for o in s.objectives.items.values()])
        self.assertIn("its ego is curious about it", mine[0].why)
        # initiative off and nothing to do: it says so
        for o in list(s.objectives.items.values()):
            s.objectives.set_status(o.id, "removed", by="tester")
        s.director.set_enabled(False, "tester")
        rec = await s.dreamer.flip()
        self.assertEqual(rec["outcome"], "nothing")
        self.assertIn("initiative is off", rec["detail"])
        self.assertTrue(any(e.topic == "dream.flip" for e in s.bus.recent("dream.*", 50)))
        lines = "\n".join((await execute(s, "/dreams")).lines)
        self.assertIn("an objective 100% of the time", lines)
        self.assertIn("flip 0.", lines)

    async def test_the_idle_loop_tosses_the_coin(self):
        s, d = self.system, self.system.dreams
        d.pol.objective_chance = 1.0
        await s.objectives.add("understand volcanoes", by="tester")
        d.pol.check_every_seconds = 0.2
        s.dreamer.next_check = 0
        self.assertTrue(await wait_for(lambda: d.stats["flip objective"] >= 1, 20), dict(d.stats))
        self.assertFalse(d.asleep)

    async def test_sleep_can_be_turned_off(self):
        s, d = self.system, self.system.dreams
        s.memory.learn("What is the capital of France?", "Paris is the capital of France.", "a test", "me")
        s.memory.learn("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.", "a test", "me")
        d.pol.objective_chance = 0.0  # the coin always lands on sleep
        rec = await s.dreamer.flip()
        self.assertEqual(rec["outcome"], "sleep", rec)
        self.assertTrue(d.asleep)
        msg = (await execute(s, "/sleep off")).lines[0]
        self.assertIn("sleep turned off", msg)
        self.assertIn("it woke up", msg)
        self.assertFalse(d.asleep)
        rec = await s.dreamer.flip()
        self.assertEqual(rec["outcome"], "awake")
        self.assertIn("sleep is turned off", rec["detail"])
        ok, why = d.fall_asleep("asked to", by="tester")
        self.assertFalse(ok)
        self.assertIn("sleep is turned off", why)
        self.assertIn("sleep is turned off", (await execute(s, "/sleep")).lines[0])
        # it stays off across a restart
        self.assertFalse(DreamBank(s).allow_sleep)
        status, body = await _route(s, "GET", "/dreams", {}, {})
        self.assertEqual((status, body["allow_sleep"], body["objective_chance"]), (200, False, 0.0))
        status, body = await _route(s, "POST", "/dreams", {}, {"allow_sleep": True, "by": "tester"})
        self.assertEqual((status, body["status"]["allow_sleep"]), (200, True))
        status, body = await _route(s, "POST", "/dreams", {}, {"action": "flip", "by": "tester"})
        self.assertEqual((status, body["flip"]["outcome"]), (200, "sleep"))
        self.assertIn("sleep allowed", (await execute(s, "/sleep on")).lines[0])
        audit = {a["kind"] for a in s.store.recent_audit(80)}
        self.assertTrue({"dream.prevent", "dream.allow"} <= audit)

    def test_the_web_page_has_the_ego_and_the_coin(self):
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        css = (web.WEB_DIR / "app.css").read_text()
        self.assertIn('id="mind-tab-ego"', page)
        for part in ("function egoAside", "function egoMembrane", "function egoPanel", '"/ego"', "allow_sleep", "Toss the coin now",
                     "Let it sleep", '"ego"'):
            self.assertIn(part, script)
        for part in (".m-ego-outer", ".ego-by", ".switch input"):
            self.assertIn(part, css)


if __name__ == "__main__":
    unittest.main()
