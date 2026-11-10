"""Questions nobody answered in time, searched on the web through your own SearXNG (websearch.py, plugins/searxng)."""
import asyncio
import importlib.util
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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


SLOW_TOOL = """import json, sys, time
req = json.load(sys.stdin)
if req["input"] == "fail":
    print(json.dumps({"error": "the server timed out", "transient": True}))
elif req["input"] == "later":
    print(json.dumps({"error": "rate limited", "transient": True, "retry_after": 7}))
else:
    time.sleep(0.4)
    print(json.dumps({"output": "done " + req["input"]}))
"""


class SandboxPatienceTests(unittest.IsolatedAsyncioTestCase):
    """A tool whose manifest says one_at_a_time and backoff_s: requests wait their turn, and after a timeout or a
    transient failure no request goes out until it has rested (doubling), so a slow server is not swamped."""

    def setUp(self):
        from neuromesh.tools import Sandbox, Tool, ToolManifest
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name) / "slowpoke"
        d.mkdir()
        (d / "tool.py").write_text(SLOW_TOOL)
        (d / "manifest.toml").write_text('name = "slowpoke"\nversion = "1.0.0"\ndescription = "slow"\nentrypoint = "tool.py"\n'
                                         'permissions = []\nkeywords = ["slowpoke"]\none_at_a_time = true\nbackoff_s = 0.5\n')
        self.tool = Tool(ToolManifest.load(d), "active")
        self.sb = Sandbox(workers=False)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_one_at_a_time_then_rest_after_a_failure(self):
        m, sb = self.tool.manifest, self.sb
        self.assertTrue(m.one_at_a_time)
        self.assertEqual(m.backoff_s, 0.5)
        t0 = time.time()
        runs = await asyncio.gather(*(sb.run(self.tool, f"q{i}", 10) for i in range(3)))
        self.assertTrue(all(r.ok for r in runs), [r.error for r in runs])
        self.assertGreaterEqual(time.time() - t0, 1.1, "three slow requests, one after another")
        # a caller that can't wait its turn hears so, as a passing failure with when to come back
        first = asyncio.ensure_future(sb.run(self.tool, "long", 10))
        await asyncio.sleep(0.05)
        busy = await sb.run(self.tool, "impatient", 0.1)
        self.assertFalse(busy.ok)
        self.assertTrue(busy.transient)
        self.assertIn("still busy", busy.error)
        self.assertTrue((await first).ok)
        # a transient failure rests it: the next request waits it out (it fits the caller's time) ...
        fail = await sb.run(self.tool, "fail", 10)
        self.assertTrue(fail.transient)
        self.assertGreater(sb.rest_left("slowpoke"), 0.3)
        t0 = time.time()
        self.assertTrue((await sb.run(self.tool, "after", 10)).ok)
        self.assertGreaterEqual(time.time() - t0, 0.3)
        self.assertEqual(sb.rest_left("slowpoke"), 0)  # a success ends the rest
        # ... the rest doubles with each failure in a row, and a server's own retry_after is kept
        await sb.run(self.tool, "fail", 10)
        await sb.run(self.tool, "fail", 10)
        self.assertEqual(sb.resting["slowpoke"]["fails"], 2)
        self.assertGreater(sb.rest_left("slowpoke"), 0.6)
        sb.resting.clear()
        await sb.run(self.tool, "later", 10)
        self.assertGreater(sb.rest_left("slowpoke"), 6)
        # a caller with less time than the rest is told when to come back, and nothing is sent
        told = await sb.run(self.tool, "now please", 2)
        self.assertFalse(told.ok)
        self.assertTrue(told.transient)
        self.assertIn("resting", told.error)
        self.assertGreater(told.retry_after, 5)


class HelperTests(unittest.TestCase):
    def test_what_stays_at_home(self):
        self.assertTrue(personal("what is my wifi password"))
        self.assertTrue(personal("when is our anniversary"))
        self.assertTrue(personal("who owns bob@example.com"))
        self.assertTrue(personal("whose number is 5551234567"))
        self.assertEqual(personal("how tall is the Eiffel Tower"), "")
        # asking it to tell you something is not about you
        self.assertEqual(personal("can you tell me how tall the Eiffel Tower is"), "")
        self.assertEqual(personal("how do I convert Celsius to Fahrenheit"), "")
        self.assertTrue(personal("when is my dentist appointment"))
        self.assertTrue(personal("I am allergic to what"))
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
            h = {k.lower(): v for k, v in last["headers"].items()}             # what SearXNG's bot detection looks for
            self.assertIn("text/html", h["accept"])
            self.assertTrue(h["accept-language"] and "gzip" in h["accept-encoding"])  # (and its answer in gzip is read)
            sx.json = False
            refused = tool.reply({"input": json.dumps({"op": "search", "q": "eiffel"}), "config": cfg})
            self.assertIn("settings.yml", refused["error"])
            tool._JSON_OK.clear()
            failed = tool.reply({"input": "__selftest__", "config": cfg})            # the self-test says what to change
            self.assertIn("formats: [html, json]", failed["error"])
            down = tool.reply({"input": json.dumps({"op": "search", "q": "x"}), "config": {"url": "http://127.0.0.1:9"}})
            self.assertTrue(down.get("transient"), down)
        finally:
            sx.stop()

    def test_the_plugin_waits_out_a_slow_searxng(self):
        sx = MockSearXNG()
        try:
            sx.answers["eiffel"] = EIFFEL
            tool = plugin()
            cfg = {"url": sx.url, "timeout": 0.5, "backoff": 2}
            sx.slow, sx.delay = 1, 1.5
            slow = tool.reply({"input": json.dumps({"op": "search", "q": "eiffel"}), "config": cfg})
            self.assertTrue(slow.get("transient"), slow)
            self.assertIn("timed out after 0.5s", slow["error"])
            self.assertEqual(slow["retry_after"], 2.0)
            searched = sum(1 for r in sx.requests if r["path"] == "/search")
            self.assertEqual(searched, 1, "a timed-out search is not sent again straight away")
            resting = tool.reply({"input": json.dumps({"op": "search", "q": "eiffel"}), "config": cfg})
            self.assertIn("resting", resting["error"])                     # nothing more until it has rested
            self.assertEqual(sum(1 for r in sx.requests if r["path"] == "/search"), searched)
            time.sleep(2.1)
            ok = tool.reply({"input": json.dumps({"op": "search", "q": "eiffel"}), "config": cfg})
            self.assertEqual(len(ok["results"]), 2, ok)
            # many at once (its --serve worker): SearXNG gets them one at a time
            sx.slow, sx.delay = 4, 0.2
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(lambda _: tool.reply({"input": json.dumps({"op": "search", "q": "eiffel"}),
                                                     "config": {**cfg, "timeout": 5}}), range(4)))
            self.assertEqual(sx.most_at_once, 1)
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

    async def test_a_slow_searxng_is_waited_out_and_tried_again(self):
        s = self.system
        s.directives.websearch.retry_minutes = 0.01  # (seconds in a test)
        s.tools.tools["searxng"].manifest.backoff_s = 0.5
        self.sx.slow, self.sx.delay = 1, 3.0
        cfg = s.directives.tools.config["searxng"]
        cfg["timeout"] = 1  # it waits a second for SearXNG
        res = await s.submit("how tall is the Eiffel Tower")
        q = s.mind.ask("clarify", "I couldn't fulfil 'how tall is the Eiffel Tower'. Do you know it?", "a test",
                       order_id=res.order_id)
        self.assertTrue(await wait_for(lambda: q.status == "answered", 40), s.websearch.status()["history"])
        first = s.websearch.history[0]
        self.assertFalse(first["ok"])
        self.assertIn("timed out", first["why"])
        self.assertIn("tries again", first["why"])
        self.assertEqual(s.websearch.tried[q.id]["attempts"], 2)
        self.assertEqual(s.tools.tools["searxng"].state, "active", "a slow server is not the tool's fault")
        self.assertEqual(self.sx.most_at_once, 1)

    async def test_superseded_questions_are_still_searched_and_any_slow_work_waits(self):
        s = self.system
        res = await s.submit("how tall is the Eiffel Tower")
        q1 = s.mind.ask("clarify", "I couldn't fulfil 'how tall is the Eiffel Tower'.", "a test", order_id=res.order_id)
        s.websearch.pol.enabled = False
        res2 = await s.submit("frobnicate the gizmo")
        s.mind.ask("clarify", "I couldn't fulfil 'frobnicate the gizmo'.", "a test", order_id=res2.order_id)
        self.assertEqual(q1.status, "dropped", "the mind keeps only the newest clarify question open")
        s.websearch.pol.enabled = True
        self.assertTrue(await wait_for(lambda: q1.status == "answered", 30), s.websearch.status())
        self.assertEqual(q1.answer, "The Eiffel Tower is 330 metres tall.")
        # busy training doesn't hold it up (light work), but a paused mesh does
        self.assertTrue(s.guardian.background_allowed(light=True) or s.guardian.sleeping)
        s.guardian.paused = True
        try:
            self.assertEqual(s.websearch.waiting_why(), "the mesh is paused")
        finally:
            s.guardian.paused = False

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
        # a person sets the wait, turns it off and on, and moves SearXNG, on the page
        status, body = await _route(s, "POST", "/websearch", {}, {"after_minutes": 15, "by": "tester"})
        self.assertEqual((status, body["after_minutes"]), (200, 15))
        self.assertEqual(s.directives.websearch.answer_after_minutes, 15)
        self.assertEqual((await _route(s, "POST", "/websearch", {}, {"after_minutes": 0, "by": "tester"}))[0], 400)
        status, body = await _route(s, "POST", "/websearch", {}, {"url": "http://127.0.0.1:9", "by": "tester"})
        self.assertEqual((status, body["url"]), (200, "http://127.0.0.1:9"))
        self.assertIn("set on this page", body["url_from"])
        self.assertTrue(await wait_for(lambda: s.tools.tools["searxng"].state == "quarantined", 30))
        status, body = await _route(s, "GET", "/websearch", {}, {})
        self.assertIn("no SearXNG at http://127.0.0.1:9", body["tool_error"])       # the reason, not just "0/1"
        self.assertIn("no SearXNG", s.status()["tools"]["searxng"]["error"])
        await _route(s, "POST", "/websearch", {}, {"url": self.sx.url, "by": "tester"})
        self.assertTrue(await wait_for(lambda: s.tools.tools["searxng"].state == "active", 30), s.tools.tools["searxng"].last_error)
        status, body = await _route(s, "POST", "/websearch", {}, {"enabled": False, "by": "tester"})
        self.assertFalse(body["enabled"])
        self.assertEqual(s.websearch.due(), [])
        await _route(s, "POST", "/websearch", {}, {"enabled": True, "by": "tester"})
        s.directives.websearch.max_per_hour = 0                             # its budget of searches
        again = await s.submit("how tall is the Eiffel Tower, really")
        q3 = s.mind.ask("clarify", "Do you know?", "a test", order_id=again.order_id)
        self.assertIn("budget", (await s.websearch.now(q3.id, "tester"))["why"])
        status, body = await _route(s, "GET", "/websearch", {}, {})
        self.assertGreaterEqual(body["counts"]["answered"], 1)
        self.assertIn("Web search (SearXNG", "\n".join(s.websearch.lines()))


if __name__ == "__main__":
    unittest.main()
