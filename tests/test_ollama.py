"""Ollama teacher, tested against a mock Ollama server that speaks the native API
(/api/tags, /api/chat with JSON-schema `format`). No real model needed."""
import asyncio
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from neuromesh.bus import Bus
from neuromesh.config import Directives
from neuromesh.guardian import Guardian
from neuromesh.store import Store
from neuromesh.tasks import CapabilitySpec
from neuromesh.teachers import LLMTeacher

ROOT = Path(__file__).resolve().parent.parent
LLAMA = {"name": "llama3.1:8b", "model": "llama3.1:8b", "details": {"family": "llama", "families": ["llama"]}}
EMBED = {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest",
         "details": {"family": "nomic-bert", "families": ["nomic-bert"]}}
RULES = [("tool:calculator", lambda t: any(c.isdigit() for c in t)),
         ("positive", lambda t: "love" in t or "great" in t),
         ("negative", lambda t: "hate" in t or "awful" in t)]


class MockOllama(BaseHTTPRequestHandler):
    """A toy model that obeys the requested JSON schema."""
    models: list = [EMBED, LLAMA]
    requests: list = []

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": MockOllama.models})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        MockOllama.requests.append({"path": self.path, "body": body,
                                    "headers": {k.lower(): v for k, v in self.headers.items()}})
        if body["model"] not in [m["name"] for m in MockOllama.models]:
            self._send(404, {"error": f"model \"{body['model']}\" not found, try pulling it first"})
            return
        fmt = body.get("format")
        user = body["messages"][-1]["content"]
        if isinstance(fmt, dict) and "examples" in fmt["properties"]:
            content = json.dumps({"examples": ["what a lovely day", "the food was great"]})
        elif isinstance(fmt, dict):
            enum = fmt["properties"]["label"].get("enum") or ["positive"]
            text = user.split("<<<", 1)[1].split(">>>", 1)[0].lower() if "<<<" in user else user.lower()
            pick = next((lab for lab, rule in RULES if lab in enum and rule(text)), enum[0])
            # reasoning models may emit a think block (with braces) before the answer
            content = "<think>{scratch: maybe}</think>" + json.dumps({"label": pick, "confidence": 0.9, "why": "mock"})
        else:
            content = "Paris is the capital of France."
        self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": content},
                         "done": True, "prompt_eval_count": 30, "eval_count": 12})

    def log_message(self, *args):
        pass


def start_server():
    server = HTTPServer(("127.0.0.1", 0), MockOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def free_port_url():
    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    return f"http://127.0.0.1:{port}"


SENTIMENT = CapabilitySpec("sentiment", "Sentiment of text", ["positive", "negative", "neutral"], open_labels=False)


class OllamaTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.url = start_server()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        MockOllama.models = [EMBED, LLAMA]
        MockOllama.requests.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "db.sqlite")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def teacher(self, base_url=None, **cfg):
        d = Directives.load(ROOT / "directives.toml", {"teachers": {"ollama": {
            "base_url": base_url or self.url, **cfg}}})
        self.guardian = Guardian(d, self.store, Bus(), Path(self.tmp.name))
        t = LLMTeacher("ollama", d.teachers.models["ollama"], self.guardian, Bus())
        asyncio.run(t.refresh(force=True))
        return t

    def test_auto_picks_a_chat_model_not_an_embedding_model(self):
        t = self.teacher()
        self.assertTrue(t.available())
        self.assertEqual(t.model, "llama3.1:8b")
        self.assertIn("auto-selected", t.status()["detail"])

    def test_untagged_name_matches_latest(self):
        MockOllama.models = [{"name": "llama3.2:latest", "model": "llama3.2:latest", "details": {}}]
        self.assertTrue(self.teacher(model="llama3.2").available())

    def test_missing_model_says_how_to_pull_it(self):
        t = self.teacher(model="qwen2.5:7b")
        self.assertFalse(t.available())
        self.assertEqual(t.state, "no_model")
        self.assertIn("ollama pull qwen2.5:7b", t.detail)
        self.assertIn("llama3.1:8b", t.detail)  # lists what is installed

    def test_server_down(self):
        t = self.teacher(base_url=free_port_url())
        self.assertFalse(t.available())
        self.assertEqual(t.state, "unreachable")
        self.assertIn("ollama serve", t.detail)

    def test_label_uses_native_chat_with_schema_enum(self):
        t = self.teacher(model="llama3.1:8b")
        lesson = asyncio.run(t.label(SENTIMENT, "I love this place"))
        self.assertEqual((lesson.label, lesson.source), ("positive", "ollama"))
        req = MockOllama.requests[-1]
        body = req["body"]
        self.assertEqual(req["path"], "/api/chat")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["format"]["properties"]["label"]["enum"], ["positive", "negative", "neutral"])
        self.assertEqual(body["options"], {"temperature": 0.1, "num_predict": 400})
        self.assertEqual(body["keep_alive"], "10m")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertNotIn("think", body)
        self.assertNotIn("authorization", req["headers"])
        self.assertEqual(self.guardian.llm_usage()["ollama"]["tokens_last_hour"], 42)

    def test_open_label_set_has_no_enum_and_think_is_passed_through(self):
        spec = CapabilitySpec("topic", "Topic of text", ["sports"], open_labels=True)
        t = self.teacher(think=False)
        asyncio.run(t.label(spec, "the match was great"))
        body = MockOllama.requests[-1]["body"]
        self.assertNotIn("enum", body["format"]["properties"]["label"])
        self.assertIs(body["think"], False)

    def test_generate_and_answer(self):
        t = self.teacher()
        texts = asyncio.run(t.generate(SENTIMENT, "positive", 5))
        self.assertEqual(texts, ["what a lovely day", "the food was great"])
        self.assertEqual(MockOllama.requests[-1]["body"]["format"]["properties"]["examples"]["type"], "array")
        self.assertIn("Paris", asyncio.run(t.answer("capital of France?")))
        self.assertNotIn("format", MockOllama.requests[-1]["body"])

    def test_model_removed_while_running_takes_teacher_offline(self):
        t = self.teacher(model="llama3.1:8b")
        MockOllama.models = [EMBED]
        self.assertIsNone(asyncio.run(t.label(SENTIMENT, "I love it")))
        self.assertEqual(t.state, "no_model")
        self.assertFalse(t.available())

    def test_proxy_settings_are_ignored_for_a_local_server(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                                          "NO_PROXY": "", "no_proxy": ""}):
            t = self.teacher()
            self.assertTrue(t.available())
            self.assertEqual(asyncio.run(t.label(SENTIMENT, "great")).label, "positive")


class OllamaSystemTests(unittest.IsolatedAsyncioTestCase):
    """The whole system with a local Ollama teacher, including losing the server mid-run."""

    async def test_orders_are_taught_by_ollama_and_fall_back_when_it_goes_away(self):
        from neuromesh.runtime import System
        from test_system import FAST, make_root
        MockOllama.models = [EMBED, LLAMA]
        server, url = start_server()
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(Path(tmp))
            overrides = {**FAST, "teachers": {"order": ["human", "ollama", "heuristic"],
                                              "ollama": {"base_url": url, "probe_seconds": 0.2}}}
            system = System(root, overrides=overrides, seed=1, device="cpu")
            system.bus.listen(lambda ev: events.append(ev), "teacher.status")
            await system.start()
            try:
                self.assertTrue(system.teachers.describe()["ollama"]["available"])
                res = await system.submit("sentiment: I love this hotel")
                self.assertEqual(res.output, "positive")
                self.assertIn("via teacher:ollama", res.trace[-1])
                res = await system.submit("what is 6 * 7")
                self.assertEqual((res.route, res.output), ("tool:calculator", 42))
                self.assertIn("teacher:ollama", res.trace[0])
                sources = {e.source for e in system.store.examples("sentiment")}
                self.assertIn("ollama", sources)

                server.shutdown()
                server.server_close()
                await system.teachers.refresh(force=True)
                self.assertFalse(system.teachers.describe()["ollama"]["available"])
                self.assertEqual(events[-1].data["state"], "unreachable")
                res = await system.submit("sentiment: the room was awful")
                self.assertTrue(res.ok)
                self.assertIn("teacher:heuristic", res.trace[-1])
            finally:
                await system.stop()

    async def test_a_teacher_left_out_of_order_is_never_used(self):
        from neuromesh.runtime import System
        from test_system import FAST, make_root
        MockOllama.models = [EMBED, LLAMA]
        MockOllama.requests.clear()
        server, url = start_server()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                overrides = {**FAST, "teachers": {"order": ["human", "heuristic"], "ollama": {"base_url": url}}}
                system = System(make_root(Path(tmp)), overrides=overrides, seed=1, device="cpu")
                await system.start()
                try:
                    res = await system.submit("sentiment: I love this hotel")
                    self.assertIn("teacher:heuristic", res.trace[-1])
                    info = system.teachers.describe()["ollama"]
                    self.assertFalse(info["available"])
                    self.assertEqual(info["detail"], "not listed in teachers.order")
                    self.assertEqual(MockOllama.requests, [])
                finally:
                    await system.stop()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
