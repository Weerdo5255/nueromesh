"""The Mind's archive (journal.py): everything in its mind is kept by date and time; the web interface shows
the last day in each Mind tab and looks older entries up by date and time."""
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh import web
from neuromesh.bus import Event
from neuromesh.journal import tab_of
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root


class JournalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides=FAST, seed=2)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_everything_in_its_mind_by_date_and_time(self):
        s = self.system
        self.assertEqual((tab_of("ego.spoke"), tab_of("dream.flip"), tab_of("agency.decided"), tab_of("mind.opinion"),
                          tab_of("inquiry.answered"), tab_of("node.trained")),
                         ("ego", "dreams", "initiative", "thoughts", "questions", None))
        three_days_ago = time.time() - 3 * 86400
        s.journal.on_event(Event(three_days_ago, "ego.spoke", {"order": "ord-1", "mood": "content", "kind": "curious",
                                                               "node": "ego policy", "text": "I'm curious about Krakatoa."}))
        s.journal.on_event(Event(three_days_ago + 600, "dream.flip", {"roll": 0.4, "outcome": "objective", "chance": 0.75,
                                                                      "detail": "worked on obj-1"}))
        res = await s.submit("What is the capital of Atlantis?")  # today: the ego takes it in (and may speak)
        s.bus.publish("temperament.shift", mood="content", previous="steady", cause="a test")
        found = s.journal.around("ego", three_days_ago, 1)
        self.assertEqual(len(found["rows"]), 1)
        self.assertIn("curious about Krakatoa", found["rows"][0]["line"])
        self.assertEqual(found["rows"][0]["tab_name"], "Ego")
        everything = s.journal.around(None, three_days_ago, 2)
        self.assertEqual({r["tab"] for r in everything["rows"]}, {"ego", "dreams"})
        days = s.journal.days(None, tz_minutes=0)
        self.assertGreaterEqual(len(days), 2)  # today, and three days ago
        self.assertEqual(sum(d["count"] for d in days if d["start"] <= three_days_ago < d["start"] + 86400), 2)
        counts = s.journal.counts()
        self.assertEqual(counts["ego"]["archived"], 1)
        self.assertGreaterEqual(counts["temperament"]["total"], 1)
        status, body = await _route(s, "GET", "/mind/archive", {"tab": "dreams", "at": str(three_days_ago), "hours": "6"}, {})
        self.assertEqual((status, len(body["rows"])), (200, 1))
        self.assertIn("tossed the coin", body["rows"][0]["line"])
        status, body = await _route(s, "GET", "/mind/archive/days", {"tab": "ego", "tz": "-60"}, {})
        self.assertEqual(status, 200)
        self.assertTrue(body["days"])
        status, _ = await _route(s, "GET", "/mind/archive", {"at": "yesterday"}, {})
        self.assertEqual(status, 400)
        self.assertIsNotNone(res)

    def test_the_web_page_archives_the_mind(self):
        script = (web.WEB_DIR / "app.js").read_text()
        css = (web.WEB_DIR / "app.css").read_text()
        for part in ("function lastDay", "function archiveBar", "/mind/archive?${q}", "/mind/archive/days?${q}", 'type: "datetime-local"',
                     "older than a day"):
            self.assertIn(part, script)
        self.assertIn(".archive-rows", css)


if __name__ == "__main__":
    unittest.main()
