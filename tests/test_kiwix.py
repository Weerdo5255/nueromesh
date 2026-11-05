"""The Kiwix research tool against a mock kiwix-serve, and research preferring it over Wikipedia."""
import asyncio
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from neuromesh.tools import Sandbox, Tool, ToolManifest

from test_research import MockWikipedia, fetches, start_wiki
from test_system import FAST, make_root, wait_for

ROOT = Path(__file__).resolve().parent.parent
BOOK = "wikipedia_en_test_2026-01"

ARTICLES = {
    "Ada Lovelace": """
<h1 id="titleHeading">Ada Lovelace</h1>
<table class="infobox"><tr><td>Born 1815 (infobox, skipped)</td></tr></table>
<div class="hatnote">Not to be confused with Lovelace (film).</div>
<p><b>Augusta Ada King, Countess of Lovelace</b> was an English mathematician and writer, chiefly known for
her work on <a href="Charles_Babbage">Charles Babbage</a>'s proposed mechanical general-purpose computer, the
<a href="Analytical_Engine">Analytical Engine</a>.<sup class="reference">[1]</sup> She was the only legitimate
child of the poet <a href="Lord_Byron">Lord Byron</a>. <a href="File:Ada.jpg">portrait</a></p>
<details><summary class="section-heading"><h2 id="Early_life">Early life</h2></summary>
<p>Ada was born in London on 10 December 1815.</p></details>
<h2>Death</h2><p>Lovelace died on 27 November 1852 at the age of 36.</p>
<h2>References</h2><ol class="references"><li>Reference noise that must be dropped.</li></ol>
<div class="navbox">Navbox noise that must be dropped.</div>
<script>var tracking = "script noise";</script>""",
    "Charles Babbage": "<p>Charles Babbage was an English polymath who originated the concept of a digital "
                       "programmable computer. He designed the <a href=\"Analytical_Engine\">Analytical Engine</a>.</p>",
    "Analytical Engine": "<p>The Analytical Engine was a proposed mechanical general-purpose computer designed by "
                         "<a href=\"Charles_Babbage\">Charles Babbage</a>.</p>",
    "Kiwix": "<p>Kiwix is free software that lets people read Wikipedia and other content offline.</p>",
}


class MockKiwix(BaseHTTPRequestHandler):
    requests: list = []
    suggest_broken = False
    outage = False

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        MockKiwix.requests.append({"path": url.path, "params": q, "proxied": url.scheme != ""})
        if MockKiwix.outage:
            self._send(503, "busy", "text/plain")
            return
        prefix = "/kiwix" if url.path.startswith("/kiwix/") else ""
        path = url.path[len(prefix):]
        if path == "/catalog/v2/entries":
            self._send(200, f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opds="https://specs.opds.io/opds-1.2">
  <title>All zims</title>
  <entry>
    <id>urn:uuid:1234</id><title>Wikipedia (test)</title><name>{BOOK}</name><language>eng</language>
    <category>wikipedia</category>
    <link rel="http://opds-spec.org/image/thumbnail" href="{prefix}/catalog/v2/illustration/1234?size=48&kind=x" />
    <link type="text/html" href="{prefix}/content/{BOOK}" />
  </entry>
</feed>""", "application/atom+xml")
            return
        if path == "/suggest":
            if MockKiwix.suggest_broken or q.get("content") != BOOK:
                self._send(404, "not found", "text/plain")
                return
            term = q.get("term", "").lower()
            items = [{"value": t, "label": t, "kind": "path", "path": "A/" + t.replace(" ", "_")}
                     for t in ARTICLES if term and term in t.lower()]
            items.append({"value": term + " ", "label": f"containing '{term}'...", "kind": "pattern"})
            self._send(200, json.dumps(items), "application/json")
            return
        if path == "/search":
            term = q.get("pattern", "").lower()
            if q.get("format") != "xml":
                self._send(404, "not found", "text/plain")
                return
            items = "".join(f"<item><title>{t}</title><link>{prefix}/content/{BOOK}/A/{t.replace(' ', '_')}</link>"
                            f"<description>...</description></item>" for t in ARTICLES if term and term in t.lower())
            self._send(200, f'<?xml version="1.0"?><rss version="2.0"><channel><title>Search</title>{items}'
                            f"</channel></rss>", "application/rss+xml")
            return
        head = f"/content/{BOOK}/A/"
        if path.startswith(head):
            title = unquote(path[len(head):]).replace("_", " ")
            body = ARTICLES.get(title)
            if body is None:
                self._send(404, "not found", "text/html")
                return
            self._send(200, f"<!DOCTYPE html><html><head><title>{title}</title>"
                            f"<style>.x{{color:red}}</style></head><body>{body}</body></html>", "text/html; charset=utf-8")
            return
        self._send(404, "not found", "text/plain")


def start_kiwix():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockKiwix)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def articles_read():
    return [r for r in MockKiwix.requests if "/content/" in r["path"]]


class KiwixToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.url = start_kiwix()
        cls.tool = Tool(ToolManifest.load(ROOT / "plugins" / "kiwix"), state="active")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        MockKiwix.requests.clear()
        MockKiwix.suggest_broken = MockKiwix.outage = False

    def run_tool(self, text, **config):
        cfg = {**self.tool.manifest.config, "url": self.url, **config}
        return asyncio.run(Sandbox(10, 512).run(self.tool, text, 20, cfg))

    def test_self_test_and_article(self):
        run = self.run_tool("__selftest__")
        self.assertTrue(run.ok, run.error)
        self.assertEqual(run.output, f"kiwix ok: 1 book ({BOOK})")
        run = self.run_tool("who was Ada Lovelace?")
        self.assertTrue(run.ok, run.error)
        k = run.extra["knowledge"]
        self.assertEqual((k["source"], k["title"], k["revision"], k["language"]), ("kiwix", "Ada Lovelace", BOOK, "eng"))
        self.assertTrue(k["url"].endswith(f"/content/{BOOK}/A/Ada_Lovelace"))
        self.assertIn("English mathematician", k["summary"])
        sections = [p["section"] for p in k["passages"]]
        self.assertEqual(sections, ["Introduction", "Early life", "Death"])
        text = " ".join(p["text"] for p in k["passages"])
        for noise in ("Reference noise", "Navbox noise", "script noise", "infobox", "[1]", "Not to be confused"):
            self.assertNotIn(noise, text)
        self.assertEqual(k["links"], ["Charles Babbage", "Analytical Engine", "Lord Byron"])
        self.assertIn("(Kiwix: Ada Lovelace, Wikipedia (test))", run.output)

    def test_fallbacks_misses_outages_proxies_and_prefixes(self):
        MockKiwix.suggest_broken = True  # older or limited servers: the XML search finds it
        run = self.run_tool("tell me about the Analytical Engine")
        self.assertTrue(run.ok, run.error)
        self.assertEqual(run.extra["knowledge"]["title"], "Analytical Engine")
        MockKiwix.suggest_broken = False
        run = self.run_tool("who was Napoleon")
        self.assertFalse(run.ok)
        self.assertTrue(run.not_found and not run.transient, run.error)
        MockKiwix.outage = True
        run = self.run_tool("who was Ada Lovelace")
        self.assertTrue(run.transient and not run.not_found, run.error)
        MockKiwix.outage = False
        run = self.run_tool("who was Ada Lovelace", url="http://127.0.0.1:9")  # nothing listening
        self.assertTrue(run.transient, run.error)
        saved = {k: os.environ.get(k) for k in ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy")}
        try:  # a proxy in the environment is not used for the local server
            os.environ.update(HTTP_PROXY="http://127.0.0.1:9", http_proxy="http://127.0.0.1:9", NO_PROXY="", no_proxy="")
            run = self.run_tool("Kiwix")
            self.assertTrue(run.ok, run.error)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        run = self.run_tool("who was Charles Babbage", url=self.url + "/kiwix")  # served under a URL prefix
        self.assertTrue(run.ok, run.error)
        self.assertIn("/kiwix/content/", run.extra["knowledge"]["url"])


class KiwixSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from neuromesh.runtime import System
        MockKiwix.requests.clear()
        MockKiwix.suggest_broken = MockKiwix.outage = False
        MockWikipedia.requests.clear()
        MockWikipedia.outage = False
        self.kiwix, kiwix_url = start_kiwix()
        self.wiki, wiki_api = start_wiki()
        self.tmp = tempfile.TemporaryDirectory()
        self.overrides = {**FAST, "tools": {**FAST["tools"], "config": {"wikipedia": {"api_url": wiki_api},
                                                                       "kiwix": {"url": kiwix_url}}},
                          "research": {"follow_links": 0, "max_fetches_per_hour": 3},
                          "tasks": {"min_interval_seconds": 0.1}}
        self.System = System
        self.root = make_root(Path(self.tmp.name), wikipedia=True, kiwix=True)
        self.system = System(self.root, overrides=self.overrides, seed=1)
        await self.system.start()
        tools = self.system.tools.tools
        self.assertTrue(await wait_for(lambda: {"kiwix", "wikipedia"} <= set(tools)))
        for a in self.system.guardian.pending_approvals():  # the operator approves both (P3)
            self.system.guardian.decide(a.id, True, by="test")
        self.assertTrue(await wait_for(lambda: tools["kiwix"].state == "active" and tools["wikipedia"].state == "active"),
                        (tools["kiwix"].note, tools["wikipedia"].note))
        MockKiwix.requests.clear()  # forget the self-tests
        MockWikipedia.requests.clear()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        for server in (self.kiwix, self.wiki):
            server.shutdown()
            server.server_close()
        self.tmp.cleanup()

    async def test_prefers_kiwix_and_falls_back_to_wikipedia(self):
        s = self.system
        s.directives.research.max_fetches_per_hour = 60  # misses fall through to Wikipedia, which costs budget
        self.assertEqual([t.name for t in s.knowledge_tools()], ["kiwix", "wikipedia"])
        self.assertFalse(any(e.text == "__selftest__" for e in s.store.examples("dispatch")))
        res = await s.submit("who was Ada Lovelace?")
        self.assertTrue(res.ok, res.note)
        self.assertTrue(res.answer.startswith("Ada Lovelace was an English mathematician"), res.answer)
        self.assertIn("\nSource: kiwix 'Ada Lovelace'", res.answer)  # the source comes after the answer
        self.assertIn("researched with kiwix", " ".join(res.trace))
        self.assertEqual(fetches(), [], "Wikipedia is not asked when Kiwix has the article")
        res = await s.submit("tell me about the Eiffel Tower")  # not in this ZIM: Wikipedia answers
        self.assertTrue(res.ok, res.note)
        self.assertIn("\nSource: wikipedia 'Eiffel Tower'", res.answer)
        for topic in ("Napoleon", "the Moon", "Mars", "Venus"):  # misses never count against Kiwix
            await s.research(topic, reason="test")
        self.assertEqual(s.tools.tools["kiwix"].state, "active")
        MockKiwix.outage = True  # Kiwix down: Wikipedia covers, and Kiwix is not quarantined
        doc, _ = await s.research("Paris", reason="test")
        self.assertEqual(doc.source, "wikipedia")
        self.assertEqual((s.tools.tools["kiwix"].state, s.tools.tools["kiwix"].consecutive_failures), ("active", 0))

    async def test_no_limits_on_kiwix(self):
        s = self.system
        for _ in range(8):  # far past max_fetches_per_hour = 3
            doc, why = await s.research("Charles Babbage", reason="test")
            self.assertIsNotNone(doc, why)
            self.assertEqual(doc.source, "kiwix")
        usage = s.guardian.research_usage()
        self.assertEqual((usage["last_hour"], usage["unlimited_last_hour"]), (0, 8))
        MockKiwix.outage = True
        sources = [(await s.research("Paris", reason="test"))[0] for _ in range(5)]
        self.assertEqual(sum(d is not None for d in sources), 3, "Wikipedia keeps its budget")

    async def test_learn_task_reads_in_bursts_from_kiwix(self):
        s = self.system
        t = s.tasks.add("learn Ada Lovelace every 30s max 3 depth 1")
        self.assertTrue(await wait_for(lambda: t.met, timeout=20), t.note)
        self.assertEqual(t.runs, 1, "one run read all three articles")
        self.assertEqual(set(t.progress["read"]), {"Ada Lovelace", "Charles Babbage", "Analytical Engine"})
        self.assertTrue(all(s.knowledge.get(title, "kiwix") for title in t.progress["read"]))

    async def test_preference_is_configurable(self):
        await self.system.stop()
        self.system = self.System(self.root, overrides={**self.overrides, "research": {"prefer": ["wikipedia", "kiwix"]}},
                                  seed=2)
        await self.system.start()
        self.assertTrue(await wait_for(lambda: len(self.system.knowledge_tools()) == 2), self.system.status()["tools"])
        self.assertEqual([t.name for t in self.system.knowledge_tools()], ["wikipedia", "kiwix"])


if __name__ == "__main__":
    unittest.main()
