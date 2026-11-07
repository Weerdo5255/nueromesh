"""Independent agency: the Director, its modes, the agency node in front of it, the tasks it sets
itself, how outcomes teach the node, and the people who stay in charge (P12)."""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from neuromesh.agency import (AGENCY, MODES, VALUES, AgencyTeacher, _clean_topic, feasible, parse_situation, policy,
                              situation_text)
from neuromesh.commands import execute
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.web import mesh

from test_system import FAST, ROOT, make_root, wait_for

AGENCY_ON = {**FAST,
             "agency": {"enabled": True, "decide_every_seconds": 0.3, "idle_seconds": 0.1, "max_decisions_per_hour": 500,
                        "improve_window_seconds": 3, "explore": 0.0},
             "tasks": {"min_interval_seconds": 1, "challenge_interval_seconds": 2, "improve_interval_seconds": 1},
             "challenges": {"practice_batch": 6}}


def flags(**kw):
    f = {"skills": "ok", "gap": "none", "topics": "none", "research": "off", "exams": "none", "scores": "none",
         "memory": "none", "last": "none", "streak": "one", "budget": "ok", "selftasks": "none", "mood": "steady",
         "needs": "none", "newtools": "none", "aims": "none", "doubts": "none"}
    f.update(kw)
    return f


class PolicyTests(unittest.TestCase):
    def test_situations_round_trip(self):
        f = flags(skills="weak", gap="large", last="practise", streak="many")
        self.assertEqual(parse_situation(situation_text(f)), f)
        self.assertEqual(situation_text(f).split()[0], "skillsweak")  # one token per feature for the node

    def test_the_drive_policy(self):
        self.assertEqual(policy(flags())[0], "rest")
        self.assertEqual(policy(flags(memory="failing", skills="weak", gap="large"))[0], "consolidate")
        self.assertEqual(policy(flags(skills="weak", gap="large"))[0], "improve")
        self.assertEqual(policy(flags(exams="due"))[0], "examine")
        self.assertEqual(policy(flags(exams="fresh", scores="low"))[0], "practise")
        self.assertEqual(policy(flags(topics="many", research="ready"))[0], "research")
        self.assertEqual(policy(flags(topics="many", research="off"))[0], "rest")  # no research tool
        self.assertEqual(policy(flags(skills="weak", budget="low"))[0], "rest")
        self.assertEqual(policy(flags(skills="weak", selftasks="full"))[0], "rest")  # no room for another task
        self.assertEqual(policy(flags(skills="weak", selftasks="full", exams="due"))[0], "examine")  # no task needed
        # variety: after many improves in a row, another worthwhile mode wins
        self.assertEqual(policy(flags(skills="weak", gap="small", exams="due", last="improve", streak="one"))[0], "examine")
        self.assertEqual(policy(flags(skills="weak", gap="large", topics="some", research="ready", last="improve",
                                      streak="many"))[0], "research")
        self.assertEqual(policy(flags(skills="weak", gap="large"), exclude=("improve",))[0], "rest")
        # the toolsmith (P15): requests nothing met, and tools it made but has not tried out yet
        self.assertEqual(policy(flags(needs="some"))[0], "build")
        self.assertEqual(policy(flags(newtools="fresh", needs="many"))[0], "experiment")
        self.assertEqual(policy(flags(needs="some", memory="failing"))[0], "consolidate")
        self.assertEqual(policy(flags(needs="some", mood="wary"))[0], "rest")   # wary: it makes nothing new
        self.assertEqual(policy(flags(needs="some", mood="wary"))[1], "it is wary, so it checks what it already holds instead of reaching outside")
        # objectives (P16): something to understand properly, and answers worth a second thought
        self.assertEqual(policy(flags(aims="some"))[0], "pursue")
        self.assertEqual(policy(flags(doubts="some", aims="some"))[0], "pursue")      # curiosity comes first
        self.assertEqual(policy(flags(doubts="some", aims="some", mood="tense"))[0], "ruminate")
        self.assertEqual(policy(flags(aims="gaps", topics="many", research="ready"))[0], "pursue")
        self.assertEqual(policy(flags(aims="some", mood="wary"))[0], "rest")      # wary: no new aims
        self.assertEqual(policy(flags(doubts="some", mood="tense"))[0], "ruminate")

    def test_what_is_possible(self):
        self.assertEqual(feasible(flags()), {"rest"})
        self.assertIn("practise", feasible(flags(exams="fresh", scores="low")))
        self.assertNotIn("practise", feasible(flags(exams="fresh", scores="high")))
        self.assertNotIn("examine", feasible(flags(exams="fresh")))
        self.assertNotIn("improve", feasible(flags(skills="weak", selftasks="full")))
        self.assertEqual(_clean_topic("  the Krebs cycle every 2m?"), "the Krebs cycle")
        self.assertEqual(feasible(flags(needs="some", newtools="fresh")), {"rest", "build", "experiment"})
        self.assertEqual(feasible(flags(aims="some", doubts="some")), {"rest", "pursue", "ruminate"})

    def test_the_policy_teaches_the_agency_node(self):
        spec = type("Spec", (), {"name": AGENCY})()
        t = AgencyTeacher(seed=1)
        lesson = asyncio.run(t.label(spec, situation_text(flags(memory="failing"))))
        self.assertEqual((lesson.label, lesson.source), ("consolidate", "agency"))
        for mode in MODES:
            texts = asyncio.run(t.generate(spec, mode, 5))
            self.assertTrue(texts, mode)
            self.assertTrue(all(policy(parse_situation(x))[0] == mode for x in texts))
            self.assertTrue(all(len(parse_situation(x)) == len(VALUES) for x in texts))
        t.enabled = lambda: False
        self.assertIsNone(asyncio.run(t.label(spec, situation_text(flags()))))
        self.assertEqual(asyncio.run(t.generate(spec, "rest", 3)), [])
        self.assertIsNone(asyncio.run(t.label(type("S", (), {"name": "sentiment"})(), situation_text(flags()))))


class DirectorTests(unittest.IsolatedAsyncioTestCase):
    overrides = AGENCY_ON

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        shutil.copytree(ROOT / "challenges", self.root / "challenges")
        self.system = System(self.root, overrides=self.overrides, seed=4)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def warm_up(self):
        for i in range(16):
            await self.system.submit(["sentiment: the hotel was lovely", "sentiment: the food was awful",
                                      "urgency: the server is down, fix it asap", "what is 3 * 4"][i % 4] + f" {i}")

    async def test_it_examines_practises_and_improves_on_its_own(self):
        s, d = self.system, self.system.director
        await self.warm_up()
        did = lambda mode: any(h["mode"] == mode for h in d.history)  # noqa: E731
        self.assertTrue(await wait_for(lambda: did("examine"), timeout=40), list(d.history))
        exam = next(h for h in d.history if h["mode"] == "examine")
        self.assertTrue(exam["target"].startswith("sample-") and exam["value"] == 0.8 and "exam" in exam["summary"])
        self.assertTrue(await wait_for(lambda: any(h["mode"] in ("practise", "improve") and h["task"] for h in d.history),
                                       timeout=40), list(d.history))
        mine = next(h for h in d.history if h["task"])
        # the task it set itself is an ordinary background task people can see, pause and remove
        self.assertIn(mine["task"], {t["id"] for t in s.tasks.status()["tasks"]} | {h["task"] for h in d.history})
        audit = {a["kind"] for a in s.store.recent_audit(500)}
        self.assertTrue({"agency.decided", "agency.assigned", "task.added"} <= audit, audit)
        added = [a for a in s.store.recent_audit(500) if a["kind"] == "task.added"]
        self.assertTrue(added and all(a["actor"] == "director" for a in added))
        # once it has run its course it is scored, retired, and the outcome teaches the agency node
        self.assertTrue(await wait_for(lambda: any(h.get("task") and h.get("value") is not None for h in d.history),
                                       timeout=40), list(d.history))
        self.assertTrue(any(e.source == "outcome" for e in s.store.examples(AGENCY)) or
                        all(0.15 < h["value"] < 0.5 for h in d.history if h.get("task") and h.get("value") is not None))
        self.assertTrue(any(e.source == "agency" for e in s.store.examples(AGENCY)))  # the drive policy's lessons
        lines = "\n".join((await execute(s, "/director")).lines)
        self.assertIn("director: initiative on, mode auto", lines)
        self.assertIn("recent decisions:", lines)
        code, body = await _route(s, "GET", "/director", {}, {})
        self.assertEqual((code, body["enabled"]), (200, True))
        self.assertIsNotNone(mesh(s)["director"])
        self.assertTrue(all(e["capability"] != AGENCY for e in mesh(s)["engines"]))  # not an emotional engine

    async def test_people_stay_in_charge(self):
        s, d = self.system, self.system.director
        out = (await execute(s, "/director mode examine")).lines[0]
        self.assertIn("pinned to examine", out)
        out = (await execute(s, "/director now")).lines[0]
        self.assertTrue(out.startswith("  examine:"), out)
        self.assertIn("pinned by a person", d.history[-1]["how"])
        self.assertFalse((await execute(s, "/director mode dance")).lines[0].startswith("  pinned"))
        code, body = await _route(s, "POST", "/director", {}, {"mode": "auto"})
        self.assertEqual((code, d.pinned), (200, "auto"))
        # a task it set itself can be removed by a person; it notices and lets go
        d.pinned = "auto"
        await self.warm_up()
        self.assertTrue(await wait_for(lambda: bool(d.self_tasks), timeout=40), list(d.history))
        tid = next(iter(d.self_tasks))
        ok, _ = s.tasks.set_status(tid, "removed", by="a person")
        self.assertTrue(ok)
        self.assertTrue(await wait_for(lambda: tid not in d.self_tasks, timeout=10))
        self.assertTrue(any(e.data.get("task") == tid and "removed by a person" in e.data["summary"]
                            for e in s.bus.recent("agency.outcome", 50)))
        # off means off
        self.assertIn("initiative off", (await execute(s, "/director off")).lines[0])
        n = len(d.history)
        await asyncio.sleep(1.2)
        self.assertEqual(len(d.history), n)
        self.assertIn("is off", (await execute(s, "/director now")).lines[0])
        # and it stays off across restarts, until a person turns it on
        await s.stop()
        again = System(self.root, overrides=self.overrides, seed=5)
        self.assertFalse(again.director.enabled)
        self.assertIn("initiative on", again.director.set_enabled(True, "a person"))
        again.store.close()

    async def test_outcomes_teach_the_agency_node(self):
        s, d = self.system, self.system.director
        text = situation_text(flags(skills="weak", gap="large", exams="due"))
        d.outcome({"mode": "examine", "situation": text, "task": ""}, 0.9, "a first measurement", "sample-yesno")
        ex = {e.text: e for e in s.store.examples(AGENCY)}
        self.assertEqual((ex[text].label, ex[text].source), ("examine", "outcome"))  # reinforced, over the policy
        poor = situation_text(flags(skills="weak", gap="large", topics="some", research="ready"))
        d.outcome({"mode": "research", "situation": poor, "task": "task-9"}, 0.0, "found nothing", "Atlantis")
        ex = {e.text: e for e in s.store.examples(AGENCY)}
        self.assertEqual(ex[poor].label, "improve")  # relabelled with the policy's next best
        self.assertIn("research:atlantis", d.cooldown)  # and that topic rests for a while
        self.assertEqual(d.values["examine"], 0.9)
        # a goal of its own becomes a topic it could read up on (research waits for a research tool)
        s.mind.add_goal("the Krebs cycle", "curious", "curiosity")
        sit = d.situation()
        self.assertIn("the Krebs cycle", [t for t, _ in sit["topics"]])
        self.assertEqual(sit["flags"]["research"], "off")
        self.assertNotIn("research", sit["feasible"])


class AgencyNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_agency_node_takes_over_from_the_policy(self):
        tmp = tempfile.TemporaryDirectory()
        s = System(make_root(Path(tmp.name)), overrides={**AGENCY_ON, "agency": {**AGENCY_ON["agency"], "enabled": True,
                                                                                    "decide_every_seconds": 3600}}, seed=6)
        await s.start()
        try:
            teacher = s.teachers.teachers[AGENCY]
            spec = s.capabilities[AGENCY]
            for mode in MODES:
                for text in await teacher.generate(spec, mode, 30):
                    s.learn(AGENCY, text, mode, "agency")

            def trusted():
                ch = s.registry.champion(AGENCY)
                return ch is not None and (ch.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy
            self.assertTrue(await wait_for(trusted, timeout=60), s.director.status())
            rec = None
            for _ in range(5):
                rec = await s.director.decide(force=True)
                if rec["how"].startswith("agency node"):
                    break
            self.assertTrue(rec["how"].startswith("agency node"), rec)
            self.assertIn("it chooses the mode", "\n".join((await execute(s, "/director")).lines))
        finally:
            await s.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
