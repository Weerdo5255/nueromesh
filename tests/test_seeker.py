"""The Kiwix search (seeker.py): every book on every Kiwix server, searched again and again - other
books, other wordings, title and full text - until an article answers the question, or
[research] kiwix_attempts searches fail; then the question goes to the user."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from neuromesh.commands import execute
from neuromesh.seeker import Seeker
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for

WIKI, DICT, VOYAGE = "wikipedia_en_test_2026-01", "wiktionary_en_test_2026-01", "wikivoyage_en_test_2026-01"
SHELVES = {
    "a": {WIKI: {"Analytical Engine": "<p>The Analytical Engine was a proposed mechanical general-purpose computer "
                                      "designed by Charles Babbage in 1837.</p>",
                 "Charles Babbage": "<p>Charles Babbage was an English polymath and mechanical engineer.</p>",
                 "Mural": "<p>A mural is any piece of artwork painted or applied directly on a wall or ceiling.</p>"},
          DICT: {"serendipity": "<p>Serendipity means finding something good without looking for it; a happy accident.</p>"}},
    "b": {VOYAGE: {"Kyoto": "<p>Kyoto is a city in Japan with many temples. Stay in a ryokan near Gion.</p>"}},
}


def handler(shelf: str):
    books = SHELVES[shelf]

    class Mock(BaseHTTPRequestHandler):
        requests: list = []

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="text/html"):
            data = body.encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            Mock.requests.append((url.path, q))
            if url.path == "/catalog/v2/entries":
                entries = "".join(f"<entry><title>{b.split('_')[0].title()} (test)</title><name>{b}</name><language>eng"
                                  f"</language><category>{b.split('_')[0]}</category><link type=\"text/html\" "
                                  f"href=\"/content/{b}\" /></entry>" for b in books)
                self._send(200, f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>',
                           "application/atom+xml")
                return
            if url.path == "/suggest":
                arts = books.get(q.get("content", ""))
                if arts is None:
                    self._send(404, "no")
                    return
                term = q.get("term", "").lower()
                items = [{"value": t, "label": t, "kind": "path", "path": "A/" + t.replace(" ", "_")}
                         for t in arts if term and term in t.lower()]
                self._send(200, json.dumps(items), "application/json")
                return
            if url.path == "/search":
                arts = books.get(q.get("books.name", ""))
                if arts is None or q.get("format") != "xml":
                    self._send(404, "no")
                    return
                words = [w for w in q.get("pattern", "").lower().replace("?", "").split() if len(w) > 3]
                hits = [t for t, body in arts.items()
                        if words and sum(w in f"{t} {body}".lower() for w in words) / len(words) >= 0.5]
                items = "".join(f"<item><title>{t}</title><link>/content/{q['books.name']}/A/{t.replace(' ', '_')}</link></item>"
                                for t in hits)
                self._send(200, f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>', "application/rss+xml")
                return
            for b, arts in books.items():
                head = f"/content/{b}/A/"
                if url.path.startswith(head):
                    body = arts.get(unquote(url.path[len(head):]).replace("_", " "))
                    if body is None:
                        break
                    title = unquote(url.path[len(head):]).replace("_", " ")
                    self._send(200, f"<html><head><title>{title}</title></head><body>{body}</body></html>")
                    return
            self._send(404, "not found")
    return Mock


def serve(shelf: str):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(shelf))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


class PlanTests(unittest.TestCase):
    def test_books_and_wordings(self):
        books = [{"name": WIKI, "category": "wikipedia"}, {"name": DICT, "category": "wiktionary"},
                 {"name": VOYAGE, "category": "wikivoyage"}]
        self.assertEqual(Seeker.rank("What does serendipity mean?", books)[0]["name"], DICT)
        self.assertEqual(Seeker.rank("Where should I stay when I visit Kyoto?", books)[0]["name"], VOYAGE)
        self.assertEqual(Seeker.rank("Who designed the Analytical Engine?", books)[0]["name"], WIKI)
        words = Seeker.wordings("Who designed the Analytical Engine?")
        self.assertEqual(words[:2], ["Analytical Engine", "Who designed the Analytical Engine?"])
        self.assertEqual(Seeker.wordings("What does serendipity mean?")[0], "serendipity")


class SeekerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from neuromesh.runtime import System
        self.a, url_a = serve("a")
        self.b, url_b = serve("b")
        self.tmp = tempfile.TemporaryDirectory()
        self.overrides = {**FAST, "memory": {"enabled": True},
                          "tools": {**FAST["tools"], "config": {"kiwix": {"url": url_a, "servers": [url_b]}}},
                          "research": {"follow_links": 0, "kiwix_attempts": 4}}
        self.system = System(make_root(Path(self.tmp.name), kiwix=True), overrides=self.overrides, seed=5)
        await self.system.start()
        tools = self.system.tools.tools
        self.assertTrue(await wait_for(lambda: "kiwix" in tools))
        for ap in self.system.guardian.pending_approvals():
            self.system.guardian.decide(ap.id, True, by="test")
        self.assertTrue(await wait_for(lambda: tools["kiwix"].state == "active"), tools["kiwix"].note)

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        for srv in (self.a, self.b):
            srv.shutdown()
            srv.server_close()
        self.tmp.cleanup()

    async def test_every_book_on_every_server(self):
        s = self.system
        books, down = await s.seeker.catalog(refresh=True)
        self.assertEqual(sorted(b["name"] for b in books), sorted([WIKI, DICT, VOYAGE]))
        self.assertEqual(down, [])
        res = await s.submit("What does serendipity mean?")
        self.assertTrue(res.ok, res.note)
        self.assertIn("happy accident", res.answer)
        self.assertEqual((res.searched["attempts"], res.searched["books"]), (1, [DICT]))  # the dictionary first
        self.assertIn("researched with kiwix: ", " ".join(res.trace))
        self.assertIn("found by Kiwix search 1 of 4", " ".join(res.trace))
        res = await s.submit("Where should I stay when I visit Kyoto?")  # a book on the second server
        self.assertTrue(res.ok, res.note)
        self.assertIn("ryokan", res.answer)
        self.assertEqual(res.searched["books"][0], VOYAGE)
        self.b.shutdown()  # a server that is down is skipped
        self.b.server_close()
        books, down = await s.seeker.catalog(refresh=True)
        self.assertEqual(sorted(b["name"] for b in books), sorted([WIKI, DICT]))
        self.assertEqual(len(down), 1)
        self.b, _ = serve("b")  # (so tearDown has something to shut down)

    async def test_it_tries_again_until_an_article_answers(self):
        s = self.system
        res = await s.submit("Who designed the Analytical Engine?")  # its subject's article answers at once
        self.assertTrue(res.ok, res.note)
        self.assertIn("Charles Babbage", res.answer)
        self.assertEqual((res.searched["attempts"], res.searched["read"]), (1, ["Analytical Engine"]))
        # the article named after him does not say it; the full-text search finds the one that does
        res = await s.submit("Which machine did Charles Babbage design in 1837?")
        self.assertTrue(res.ok, res.note)
        self.assertIn("Analytical Engine", res.answer)
        self.assertEqual(res.searched["read"], ["Charles Babbage", "Analytical Engine"])
        h = s.seeker.history[-1]
        self.assertEqual([(a["outcome"], a["mode"]) for a in h["attempts"]], [("not it", "title"), ("answered", "fulltext")])
        self.assertIn("covers only", h["attempts"][0]["why"])

    async def test_it_gives_up_and_asks_you(self):
        s = self.system
        res = await s.submit("Who painted the Zorblax mural?")
        self.assertFalse(res.ok)
        sought = res.searched
        self.assertEqual(sought["attempts"], 4)  # [research] kiwix_attempts
        self.assertIn("Mural", sought["read"])   # it read the article about murals: not it
        h = s.seeker.history[-1]
        self.assertTrue(any(a["outcome"] == "not it" and "Zorblax" in a["why"] for a in h["attempts"]), h["attempts"])
        urls = [a["title"] for a in h["attempts"] if a["title"]]
        self.assertEqual(len(urls), len(set(urls)), "no article is read twice for the same question")
        self.assertIn("searched 4 times", sought["gave_up"])
        asked = [q for q in s.mind.open_inquiries() if q.order_id == res.order_id]
        self.assertTrue(asked, "it asks you")
        self.assertIn("I searched Kiwix 4 times", asked[0].question)
        self.assertIn("Do you know it?", asked[0].question)
        self.assertIn(res.inquiry["id"], asked[0].id)
        ok, lines = await s.mind.reply(asked[0].id, "answer: Ada Lindqvist painted it.", by="tester")
        self.assertTrue(ok, lines)
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 60))
        again = await s.submit("Who painted the Zorblax mural?")
        self.assertIn("Ada Lindqvist", again.output)
        self.assertIsNone(again.searched)  # it knows now: no search

    async def test_budgets_one_look_up_and_the_console(self):
        s = self.system
        s.directives.research.unlimited = ()  # Kiwix counted like any research: two searches left this hour
        s.directives.research.max_fetches_per_hour = 2
        found = await s.seeker.seek("Who painted the Zorblax mural?")
        self.assertEqual([a.outcome for a in found.attempts][-1], "refused")
        self.assertIn("research budget", found.gave_up)
        s.directives.research.unlimited = ("kiwix",)
        s.directives.research.kiwix_attempts = 1  # a single look-up, as before
        self.assertFalse(s.seeker.wanted("Who designed the Analytical Engine?"))
        s.directives.research.kiwix_attempts = 4
        lines = "\n".join((await execute(s, "/kiwix")).lines)
        self.assertIn("up to 4 searches a question", lines)
        self.assertIn(VOYAGE, lines)
        lines = "\n".join((await execute(s, "/kiwix What does serendipity mean?")).lines)
        self.assertIn("answered after 1 search", lines)
        status, body = await _route(s, "GET", "/kiwix", {}, {})
        self.assertEqual((status, body["attempts"]), (200, 4))
        self.assertEqual(len(body["books"]), 3)
        status, body = await _route(s, "POST", "/kiwix", {}, {"question": "Who designed the Analytical Engine?"})
        self.assertEqual((status, body["ok"]), (200, True))
        status, _ = await _route(s, "POST", "/kiwix", {}, {})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
