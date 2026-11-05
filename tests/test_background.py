"""Background tasks people set: learn a topic, improve a skill, repeat an order."""
import tempfile
import unittest
from pathlib import Path

from neuromesh.background import parse_duration, parse_spec
from neuromesh.commands import execute
from neuromesh.live import UNICODE, teacher_goal_lines
from neuromesh.runtime import System
from neuromesh.server import _route

from test_research import MockWikipedia, start_wiki
from test_system import FAST, make_root, wait_for

QUICK = {**FAST, "tasks": {"min_interval_seconds": 0.1, "improve_interval_seconds": 0.3}}


class SpecTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_spec("learn quantum computing every 2m max 30"),
                         {"kind": "learn", "target": "quantum computing", "interval": 120.0, "max_articles": 30})
        self.assertEqual(parse_spec("keep learning about jazz")["target"], "jazz")
        self.assertEqual(parse_spec("improve sentiment to 97%"), {"kind": "improve", "target": "sentiment",
                                                                  "threshold": 0.97})
        self.assertEqual(parse_spec("every 10m: count the words in: every word here"),
                         {"kind": "run", "target": "count the words in: every word here", "interval": 600.0})
        self.assertEqual(parse_spec("run what is 2+2 every 30 seconds notify always"),
                         {"kind": "run", "target": "what is 2+2", "interval": 30.0, "notify": "always"})
        self.assertEqual(parse_duration("1.5h"), 5400)
        for bad in ("run what is 2+2", "dance wildly", "learn", "every soon: x"):
            with self.assertRaises(ValueError, msg=bad):
                parse_spec(bad)


class BackgroundTaskTests(unittest.IsolatedAsyncioTestCase):
    overrides = QUICK

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.overrides, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_repeating_order_tells_you_when_the_result_changes(self):
        s = self.system
        out = (await execute(s, "/task every 0.2s: sentiment: the soup was cold")).lines[0]
        self.assertIn("task-1 set", out)
        task = s.tasks.get("task-1")
        self.assertTrue(await wait_for(lambda: task.runs >= 3))
        first = task.history[0][2]
        updates = [n for n in s.mind.notes.values() if n.kind == "update"]
        self.assertEqual(len(updates), 1, "told once about the first result, not on every run")
        other = "negative" if first != "negative" else "positive"
        s.teach("sentiment", "the soup was cold", other)  # a human changes the answer
        self.assertTrue(await wait_for(lambda: any("changed" in n.text for n in s.mind.notes.values()
                                                   if n.kind == "update")))
        self.assertEqual(s.order_count, 0, "task runs are not user orders")
        self.assertTrue(any(e.topic == "task.update" for e in s.bus.recent("task.*", 50)))
        notes = "\n".join((await execute(s, "/notes")).lines)
        self.assertIn("task-1", notes)

    async def test_pause_resume_remove_halt_and_failures(self):
        s = self.system
        t = s.tasks.add("every 0.2s: what is 6 * 7")
        self.assertTrue(await wait_for(lambda: len(t.history) >= 1))
        self.assertEqual(t.history[-1][2], "42")
        self.assertTrue(s.tasks.set_status(t.id, "paused", "tester")[0])
        runs = t.runs
        await self._idle(1.0)
        self.assertEqual(t.runs, runs, "paused tasks don't run")
        s.tasks.set_status(t.id, "active", "tester")
        self.assertTrue(await wait_for(lambda: t.runs > runs))
        s.guardian.halt("test")
        runs = t.runs
        await self._idle(1.0)
        self.assertEqual(t.runs, runs, "a halt stops background tasks (P1)")
        s.guardian.resume()
        bad = s.tasks.add("every 0.2s: frobnicate the gizmo")
        self.assertTrue(await wait_for(lambda: bad.status == "paused", timeout=20), bad.note)
        self.assertIn("failures in a row", bad.note)
        self.assertTrue(s.tasks.set_status(t.id, "removed", "tester")[0])
        self.assertIsNone(s.tasks.get(t.id))
        with self.assertRaises(ValueError):
            s.tasks.add("every 0.01s: what is 1 + 1")  # faster than min_interval_seconds
        with self.assertRaises(ValueError):
            s.tasks.add("improve astrology")  # no such skill

    async def _idle(self, seconds):
        import asyncio
        await asyncio.sleep(seconds)

    async def test_improve_trains_a_skill_to_its_target_without_any_orders(self):
        s = self.system
        self.assertEqual(s.store.count("urgency"), 0)
        t = s.tasks.add("improve urgency to 0.9")
        self.assertEqual(s.tasks.focus(), {"urgency": 0.9})
        self.assertTrue(await wait_for(lambda: t.met, timeout=60), t.note)
        champ = s.registry.champion("urgency")
        self.assertGreaterEqual(champ.metrics["holdout_acc"], 0.9)
        self.assertIn("keeping watch", t.note)
        self.assertTrue(any("reached" in n.text for n in s.mind.notes.values() if n.kind == "update"))

    async def test_orders_api_live_view_and_persistence(self):
        s = self.system
        res = await s.submit("keep learning about jazz")
        self.assertEqual(res.route, "tasks")
        self.assertIn("Background task task-1 set: learn 'jazz'", res.output)
        res = await s.submit("every 0.01s: what is 2 + 2")
        self.assertFalse(res.ok)
        self.assertIn("shortest interval", res.note)
        code, body = await _route(s, "POST", "/tasks", {}, {"spec": "every 0.5s: what is 3 * 3"})
        self.assertEqual((code, body["kind"]), (200, "run"))
        code, body = await _route(s, "POST", "/task", {}, {"id": body["id"], "action": "pause"})
        self.assertEqual(code, 200)
        code, body = await _route(s, "GET", "/tasks", {}, {})
        self.assertEqual({t["status"] for t in body["tasks"]}, {"active", "paused"})
        line = teacher_goal_lines(s, UNICODE)[5][0]
        self.assertTrue(line.startswith("tasks:") and "task-1 learn jazz" in line and "1 paused" in line, line)
        self.assertIn("waiting for a research tool", s.tasks.get("task-1").note if s.tasks.get("task-1").runs
                      else "waiting for a research tool")
        await s.stop()
        again = System(self.root, overrides=self.overrides, seed=2)
        self.assertEqual(sorted(again.tasks.tasks), ["task-1", "task-2"])
        self.assertEqual(again.tasks.get("task-2").status, "paused")
        self.assertEqual(again.tasks.add("every 1s: what is 1 + 1").id, "task-3")
        again.store.close()


class LearnTaskTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MockWikipedia.requests.clear()
        MockWikipedia.outage = False
        self.server, api = start_wiki()
        self.tmp = tempfile.TemporaryDirectory()
        overrides = {**QUICK, "tools": {**FAST["tools"], "config": {"wikipedia": {"api_url": api}}},
                     "research": {"follow_links": 0}}
        self.system = System(make_root(Path(self.tmp.name), wikipedia=True), overrides=overrides, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        await self.system.stop()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    async def test_learn_asks_for_its_tool_then_studies_the_topic_and_its_links(self):
        s = self.system
        self.assertTrue(await wait_for(lambda: "wikipedia" in s.tools.tools))
        t = s.tasks.add("learn Ada Lovelace every 0.2s max 3 depth 1")
        self.assertTrue(await wait_for(lambda: any(q.kind == "approval" for q in s.mind.open_inquiries())))
        self.assertIn("waiting for wikipedia to be approved", t.note)
        ask = next(q for q in s.mind.open_inquiries() if q.kind == "approval")
        self.assertTrue((await s.mind.reply(ask.id, "yes"))[0])
        self.assertTrue(await wait_for(lambda: t.met, timeout=30), t.note)
        self.assertEqual(t.progress["read"][0], "Ada Lovelace")
        self.assertEqual(set(t.progress["read"]), {"Ada Lovelace", "Analytical Engine", "Charles Babbage"})
        for title in t.progress["read"]:
            self.assertIsNotNone(s.knowledge.get(title))
        self.assertIsNotNone(s.mind.find_opinion("Ada Lovelace"), "read with the skill nodes")
        self.assertTrue(any(n.text.startswith("I've read 3 articles about 'Ada Lovelace'")
                            for n in s.mind.notes.values() if n.kind == "update"))
        self.assertTrue(await wait_for(lambda: "all fresh" in t.note), t.note)
        show = "\n".join((await execute(s, f"/task show {t.id}")).lines)
        self.assertIn("read (3/3): Ada Lovelace, ", show)


if __name__ == "__main__":
    unittest.main()
