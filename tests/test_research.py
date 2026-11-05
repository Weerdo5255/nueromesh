"""Wikipedia research and retained knowledge, against a mock MediaWiki API (no network needed)."""
import asyncio
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from neuromesh.tools import Sandbox, Tool, ToolManifest

ROOT = Path(__file__).resolve().parent.parent

ARTICLES = {
    "Ada Lovelace": {
        "extract": ("Augusta Ada King, Countess of Lovelace, was an English mathematician and writer, chiefly known "
                    "for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical "
                    "Engine. She was the only legitimate child of the poet Lord Byron.\n\n"
                    "== Early life ==\nAda was born in London on 10 December 1815.\n\n"
                    "== Death ==\nLovelace died on 27 November 1852 at the age of 36.\n\n"
                    "== References ==\nThis section is reference noise and must be dropped."),
        "links": ["Analytical Engine", "Charles Babbage", "Lord Byron", "London", "Zebra"],
    },
    "Charles Babbage": {"extract": "Charles Babbage was an English polymath who originated the concept of a "
                                   "digital programmable computer.", "links": []},
    "Analytical Engine": {"extract": "The Analytical Engine was a proposed mechanical general-purpose computer "
                                     "designed by Charles Babbage.", "links": []},
    "Paris": {"extract": "Paris is the capital and largest city of France. Its population is about 2.1 million.",
              "links": ["France"]},
    "Eiffel Tower": {"extract": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
                                "France. It is 330 metres tall.", "links": ["Paris"]},
}
SEARCH_ALIASES = {"capital of france": "Paris"}


class MockWikipedia(BaseHTTPRequestHandler):
    requests: list = []
    outage = False

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        MockWikipedia.requests.append({"params": q, "agent": self.headers.get("User-Agent", "")})
        if MockWikipedia.outage:
            self._send(503, {"error": "maintenance"})
            return
        if q.get("list") == "search":
            term = q["srsearch"].lower()
            title = SEARCH_ALIASES.get(term) or next(
                (t for t in ARTICLES if t.lower() in term or term in t.lower()), None)
            self._send(200, {"query": {"search": [{"ns": 0, "title": title}] if title else []}})
            return
        title = q.get("titles", "")
        art = ARTICLES.get(title)
        if art is None:
            self._send(200, {"query": {"pages": [{"ns": 0, "title": title, "missing": True}]}})
            return
        self._send(200, {"query": {"pages": [{
            "pageid": 1, "ns": 0, "title": title, "extract": art["extract"], "lastrevid": 4242,
            "fullurl": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "links": [{"ns": 0, "title": t} for t in art["links"]]}]}})

    def log_message(self, *args):
        pass


def start_wiki():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockWikipedia)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/w/api.php"


def fetches(kind="extracts"):
    key = "prop" if kind == "extracts" else "list"
    return [r for r in MockWikipedia.requests if kind in r["params"].get(key, "")]


class WikipediaToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.api = start_wiki()
        cls.tool = Tool(ToolManifest.load(ROOT / "plugins" / "wikipedia"), "active")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        MockWikipedia.requests.clear()
        MockWikipedia.outage = False

    def run_tool(self, text):
        config = {**self.tool.manifest.config, "api_url": self.api}
        return asyncio.run(Sandbox().run(self.tool, text, 20, config))

    def test_article_becomes_knowledge(self):
        run = self.run_tool("Who was Ada Lovelace?")
        self.assertTrue(run.ok, run.error)
        self.assertEqual(MockWikipedia.requests[0]["params"]["srsearch"], "Ada Lovelace")
        self.assertIn("NeuroMeshResearchBot", MockWikipedia.requests[0]["agent"])
        k = run.extra["knowledge"]
        self.assertEqual((k["title"], k["source"], k["revision"]), ("Ada Lovelace", "wikipedia", "4242"))
        self.assertIn("English mathematician", k["summary"])
        sections = [p["section"] for p in k["passages"]]
        self.assertEqual(sections, ["Introduction", "Early life", "Death"])  # References dropped
        self.assertEqual(k["links"][:3], ["Charles Babbage", "Analytical Engine", "Lord Byron"])
        self.assertNotIn("Zebra", k["links"])  # only links mentioned in the introduction
        self.assertIn("(Wikipedia: Ada Lovelace)", run.output)

    def test_errors_are_classified(self):
        run = self.run_tool("tell me about quantum chromodynamics")
        self.assertFalse(run.ok)
        self.assertFalse(run.transient)
        self.assertIn("no article", run.error)
        MockWikipedia.outage = True
        run = self.run_tool("tell me about Paris")
        self.assertFalse(run.ok)
        self.assertTrue(run.transient, "an outage is transient, not a broken tool")


class ResearchSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from neuromesh.runtime import System
        from test_system import FAST, make_root
        MockWikipedia.requests.clear()
        MockWikipedia.outage = False
        self.server, self.api = start_wiki()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name), wikipedia=True)
        self.overrides = {**FAST, "tools": {**FAST["tools"], "config": {"wikipedia": {"api_url": self.api}}},
                          "research": {"follow_links": 2}}
        self.System = System
        self.system = System(self.root, overrides=self.overrides, seed=1, device="cpu")
        await self.system.start()
        await self.approve_wikipedia()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    async def approve_wikipedia(self):
        from test_system import wait_for
        tools = self.system.tools.tools
        self.assertTrue(await wait_for(lambda: "wikipedia" in tools))
        if tools["wikipedia"].state == "pending_approval":
            approval = next(a for a in self.system.guardian.pending_approvals() if a.subject == "wikipedia")
            self.system.guardian.decide(approval.id, True, by="test")
        self.assertTrue(await wait_for(lambda: tools["wikipedia"].state == "active"), tools["wikipedia"].note)

    async def test_network_tool_needs_approval_first(self):
        from test_system import FAST, make_root
        with tempfile.TemporaryDirectory() as tmp:
            system = self.System(make_root(Path(tmp), wikipedia=True), overrides=FAST, seed=1, device="cpu")
            await system.start()
            try:
                self.assertEqual(system.tools.tools["wikipedia"].state, "pending_approval")
                self.assertIsNone(system.knowledge_tool())
                doc, note = await system.research("Paris", reason="test")
                self.assertIsNone(doc)
                self.assertIn("approve", note)
            finally:
                await system.stop()

    async def test_research_then_recall_then_remember_across_restart(self):
        from test_system import wait_for
        s = self.system
        res = await s.submit("who was Ada Lovelace?")
        self.assertTrue(res.ok, res.note)
        self.assertEqual((res.route, res.knowledge), ("tool:wikipedia", "researched"))
        self.assertIn("English mathematician", res.output)
        self.assertIn("source: wikipedia 'Ada Lovelace'", res.output)
        fetched = len(fetches())

        res = await s.submit("when did Ada Lovelace die?")
        self.assertEqual(res.knowledge, "recalled", res.trace)
        self.assertIn("1852", res.output)
        self.assertEqual(len(fetches()), fetched, "recall must not touch the network")

        # curiosity: linked articles are read in the background
        self.assertTrue(await wait_for(lambda: s.knowledge.get("Charles Babbage") is not None
                                       and s.knowledge.get("Analytical Engine") is not None, timeout=20))
        self.assertIsNone(s.knowledge.get("Lord Byron"), "follow_links = 2")

        await s.stop()
        self.system = self.System(self.root, overrides=self.overrides, seed=2, device="cpu")
        await self.system.start()
        before = len(fetches())
        res = await self.system.submit("tell me about Ada Lovelace")
        self.assertEqual(res.knowledge, "recalled")
        self.assertEqual(len(fetches()), before)

    async def test_question_aliases_and_unroutable_questions(self):
        s = self.system
        res = await s.submit("what is the capital of France?")
        self.assertEqual(res.knowledge, "researched", res.trace)
        self.assertIn("capital", res.output)
        n = len(fetches())
        res = await s.submit("what is the capital of France?")
        self.assertEqual(res.knowledge, "recalled")
        self.assertEqual(len(fetches()), n)
        res = await s.submit("tool wikipedia: Eiffel Tower")
        self.assertIn("330 metres", res.output + " ".join(res.trace) + s.knowledge.get("Eiffel Tower").summary)

    async def test_budget_outage_and_commands(self):
        from neuromesh.commands import execute
        s = self.system
        s.directives.research.max_fetches_per_hour = s.guardian.research_usage()["last_hour"] + 1
        res = await s.submit("who was Ada Lovelace?")
        self.assertEqual(res.knowledge, "researched")
        res = await s.submit("tell me about the Eiffel Tower")  # not linked from Ada, so not pre-read
        self.assertFalse(res.ok)
        self.assertIn("budget", res.note)
        s.directives.research.max_fetches_per_hour = 100

        MockWikipedia.outage = True
        for _ in range(4):
            res = await s.submit("tell me about the Eiffel Tower")
            self.assertFalse(res.ok)
        self.assertEqual(s.tools.tools["wikipedia"].state, "active", "outages must not quarantine the tool")
        MockWikipedia.outage = False

        out = await execute(s, "/research Paris")
        self.assertIn("learned 'Paris'", out.lines[0])
        out = await execute(s, "/knowledge capital of France")
        self.assertTrue(any("Paris" in line for line in out.lines))
        out = await execute(s, "/forget Paris")
        self.assertIn("forgot", out.lines[0])
        self.assertIsNone(s.knowledge.get("Paris"))

    async def test_answers_written_by_an_llm_teacher_from_retained_passages(self):
        from test_ollama import EMBED, LLAMA, MockOllama, start_server
        MockOllama.models = [EMBED, LLAMA]
        MockOllama.requests.clear()
        ollama, url = start_server()
        try:
            await self.system.stop()
            overrides = {**self.overrides, "teachers": {"order": ["human", "ollama", "heuristic"],
                                                         "ollama": {"base_url": url}}}
            self.system = self.System(self.root, overrides=overrides, seed=3, device="cpu")
            await self.system.start()
            await self.approve_wikipedia()
            res = await self.system.submit("tool wikipedia: Paris")
            self.assertIn("written by the ollama teacher", " ".join(res.trace))
            compose = [r for r in MockOllama.requests if "Passages from the wikipedia article" in
                       r["body"]["messages"][-1]["content"]]
            self.assertTrue(compose, "the teacher was given the retained passages")
            self.assertIn("only the passages", compose[-1]["body"]["messages"][0]["content"])
        finally:
            ollama.shutdown()
            ollama.server_close()


if __name__ == "__main__":
    unittest.main()
