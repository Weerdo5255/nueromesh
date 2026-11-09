"""Questions nobody answered in time, searched on the web through your own SearXNG (websearch.py, plugins/searxng)."""
import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.websearch import covers, fits, personal

from mock_servers import MockOllama, MockSearXNG
from test_system import FAST, ROOT, make_root, wait_for

EIFFEL = {"results": [
    {"title": "Eiffel Tower - Wikipedia", "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
     "content": "The Eiffel Tower is a wrought-iron lattice tower in Paris. It is 330 metres (1,083 ft) tall.",
     "engines": ["google", "wikipedia"]},
    {"title": "Eiffel Tower facts", "url": "https://example.org/eiffel",
     "content": "Standing 330 metres tall, the Eiffel Tower was the tallest structure in the world until 1930.",
     "engines": ["duckduckgo"]}],
    "infoboxes": [{"infobox": "Eiffel Tower", "content": "The Eiffel Tower is 330 metres tall, in Paris.",
                   "urls": [{"url": "https://www.wikidata.org/wiki/Q243"}], "engine": "wikidata"}]}


def plugin():
    spec = importlib.util.spec_from_file_location("searxng_tool", ROOT / "plugins" / "searxng" / "tool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HelperTests(unittest.TestCase):
    def test_what_stays_at_home(self):
        self.assertTrue(personal("what is my wifi password"))
        self.assertTrue(personal("when is our anniversary"))
        self.assertTrue(personal("who owns bob@example.com"))
        self.assertTrue(personal("whose number is 5551234567"))
        self.assertEqual(personal("how tall is the Eiffel Tower"), "")
        self.assertGreater(covers("how tall is the Eiffel Tower", EIFFEL["results"][0]["content"]), 0.6)
        self.assertTrue(fits("how tall is the Eiffel Tower", "330 metres"))
        self.assertFalse(fits("how many moons does Mars have", "Mars has moons"))
        self.assertFalse(fits("when was it built", "it was built of iron"))

    def test_the_plugin(self):
        sx = MockSearXNG()
        try:
            sx.answers["eiffel"] = EIFFEL
            tool = plugin()
            cfg = {"url": sx.url, "engines": "google,duckduckgo"}
            out = tool.handle({"input": "__selftest__", "config": cfg})["output"]
            self.assertTrue(out.startswith("searxng ok: Home SearXNG 2026.10.1, 3 engines"), out)
            got = tool.handle({"input": json.dumps({"op": "search", "q": "how tall is the Eiffel Tower"}), "config": cfg})
            self.assertEqual([r["url"] for r in got["results"]], [r["url"] for r in EIFFEL["results"]])
            self.assertEqual(got["results"][0]["engines"], ["google", "wikipedia"])
            self.assertEqual(got["infoboxes"][0]["url"], "https://www.wikidata.org/wiki/Q243")
            last = sx.requests[-1]
            self.assertEqual((last["query"]["format"], last["query"]["engines"]), ("json", "google,duckduckgo"))
            self.assertEqual(last["agent"], "neuromesh")                      # a plain client name, no one's address
            said = tool.handle({"input": "search the web for the Eiffel Tower height", "config": cfg})
            self.assertIn("330 metres", said["output"])
            self.assertEqual(sx.requests[-1]["query"]["q"], "the Eiffel Tower height")
            sx.json = False
            refused = tool.reply({"input": json.dumps({"op": "search", "q": "eiffel"}), "config": cfg})
            self.assertIn("settings.yml", refused["error"])
            down = tool.reply({"input": json.dumps({"op": "search", "q": "x"}), "config": {"url": "http://127.0.0.1:9"}})
            self.assertTrue(down.get("transient"), down)
        finally:
            sx.stop()


def reader(call):
    if "numbered web search results" in call["system"]:
        if "Eiffel" in call["user"]:
            return {"found": True, "answer": "The Eiffel Tower is 330 metres tall.", "sources": [1, 2]}
        return {"found": False, "answer": "", "sources": []}
    if "fallback answerer" in call["system"]:
        return "I am not sure."
    return {"label": "positive", "confidence": 0.9}


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sx = MockSearXNG()
        self.sx.answers["eiffel"] = EIFFEL
        self.llm = MockOllama(reader)
        self.tmp = tempfile.TemporaryDirectory()
        over = {**FAST, "tools": {**FAST["tools"], "config": {"searxng": {"url": self.sx.url}}},
                "teachers": {"order": ["human", "ollama", "heuristic"],
                             "ollama": {"provider": "ollama", "base_url": self.llm.url, "model": "llama3.1:8b"}},
                "memory": {"enabled": True, "interval_seconds": 0.05},
                "mind": {"max_inquiries_per_hour": 50},
                "websearch": {"answer_after_minutes": 0, "check_every_seconds": 0.2}}
        self.system = s = System(make_root(Path(self.tmp.name), searxng=True), overrides=over, seed=4)
        await s.start()
        self.assertTrue(await wait_for(lambda: "searxng" in s.tools.tools and s.tools.tools["searxng"].state == "pending_approval", 20))
        apv = next(a for a in s.guardian.pending_approvals() if a.subject == "searxng")
        self.assertIn("network", apv.detail["permissions"])                # it asks first, like Kiwix (P3)
        s.guardian.decide(apv.id, True, by="tester")
        self.assertTrue(await wait_for(lambda: s.tools.tools["searxng"].state == "active", 30), s.tools.tools["searxng"].note)

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.sx.stop()
        self.llm.stop()
        self.tmp.cleanup()

    async def test_a_question_nobody_answered_is_searched(self):
        s = self.system
        res = await s.submit("how tall is the Eiffel Tower")
        q = s.mind.ask("clarify", "I couldn't fulfil 'how tall is the Eiffel Tower'. Do you know it?", "a test",
                       order_id=res.order_id)
        self.assertTrue(await wait_for(lambda: q.status == "answered", 30), s.websearch.status()["history"])
        self.assertEqual(q.answer, "The Eiffel Tower is 330 metres tall.")
        self.assertEqual(q.by, "the web (ollama)")                          # a teacher's answer, not a person's
        self.assertEqual([x["url"] for x in q.sources], [r["url"] for r in EIFFEL["results"]])
        self.assertTrue(any(r["query"].get("q") == "how tall is the Eiffel Tower" for r in self.sx.requests))
        rec = s.websearch.history[-1]
        self.assertEqual((rec["ok"], rec["read_by"], rec["how"]), (True, "ollama", "teacher"))
        self.assertTrue(any(e.topic == "web.answered" for e in s.bus.recent("web.*", 20)))
        self.assertTrue(await wait_for(lambda: (s.memory.recall("how tall is the Eiffel Tower", quiet=True) or
                                                 type("R", (), {"answer": ""})()).answer.count("330"), 10))  # kept (P14)
        status, body = await _route(s, "GET", "/mind", {}, {})
        self.assertTrue(any(a["id"] == q.id for a in body["web_answers"]))
        # a person's word still overrides it (P1)
        ok, out = await s.mind.reply(q.id, "answer: 330 metres, 1,083 feet", by="tester")
        self.assertTrue(ok, out)
        self.assertEqual(q.by, "")
        # questions about you never leave the house
        mine = await s.submit("what is my wifi password")
        q2 = s.mind.ask("clarify", "I couldn't fulfil 'what is my wifi password'.", "a test", order_id=mine.order_id)
        self.assertFalse(s.websearch.eligible(q2)[0])
        await asyncio.sleep(0.8)
        self.assertEqual(q2.status, "open")
        self.assertFalse(any("wifi" in r["query"].get("q", "") for r in self.sx.requests))
        self.assertIn("stays at home", s.mind._web(q2)["why"])
        # approvals and proposals are only ever people's
        self.assertFalse(s.websearch.eligible(type("Q", (), {"status": "open", "kind": "approval", "approval_id": "apv-1",
                                                             "proposal_id": "", "order_id": "", "about": ""})())[0])

    async def test_without_a_teacher_it_reads_the_results_itself(self):
        s = self.system
        s.directives.websearch.teacher = False
        made, why = await s.websearch.compose("how tall is the Eiffel Tower", {**EIFFEL, "answers": []})
        self.assertIsNotNone(made, why)
        self.assertEqual(made["how"], "infobox")
        self.assertIn("330 metres", made["answer"])
        made, why = await s.websearch.compose("how tall is the Eiffel Tower", {"results": EIFFEL["results"]})
        self.assertEqual(made["how"], "snippet")
        self.assertLess(made["confidence"], 0.5)                             # less sure, and it says so
        made, why = await s.websearch.compose("how many moons does Mars have", {"results": EIFFEL["results"]})
        self.assertIsNone(made)
        # a person asks it to search now: an objective's question
        res = await s.submit("how tall is the Eiffel Tower")
        q = s.mind.ask("clarify", "Do you know?", "a test", order_id=res.order_id)
        s.directives.websearch.answer_after_minutes = 600                    # not yet due on its own
        status, body = await _route(s, "POST", "/websearch", {}, {"inquiry": q.id, "by": "tester"})
        self.assertEqual(status, 200, body)
        self.assertEqual(q.status, "answered")
        self.assertIn("its own reading", q.by)
        s.directives.websearch.max_per_hour = 0                             # its budget of searches
        again = await s.submit("how tall is the Eiffel Tower, really")
        q3 = s.mind.ask("clarify", "Do you know?", "a test", order_id=again.order_id)
        self.assertIn("budget", (await s.websearch.now(q3.id, "tester"))["why"])
        status, body = await _route(s, "GET", "/websearch", {}, {})
        self.assertGreaterEqual(body["counts"]["answered"], 1)
        self.assertIn("Web search (SearXNG", "\n".join(s.websearch.lines()))


if __name__ == "__main__":
    unittest.main()
