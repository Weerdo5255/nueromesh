"""LLM teacher wire format, tested against a local mock server (no network needed)."""
import asyncio
import json
import os
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
from neuromesh.teachers import HeuristicTeacher, LLMTeacher

ROOT = Path(__file__).resolve().parent.parent


class MockLLM(BaseHTTPRequestHandler):
    requests: list = []
    reply_text = '{"label": "positive", "confidence": 0.9, "why": "upbeat"}'

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        MockLLM.requests.append({"path": self.path, "body": body,
                                 "headers": {k.lower(): v for k, v in self.headers.items()}})
        if self.path == "/v1/messages":
            payload = {"content": [{"type": "text", "text": MockLLM.reply_text}],
                       "usage": {"input_tokens": 40, "output_tokens": 10}}
        else:
            payload = {"choices": [{"message": {"content": MockLLM.reply_text}}], "usage": {"total_tokens": 50}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class LLMTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), MockLLM)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        MockLLM.requests.clear()
        MockLLM.reply_text = '{"label": "positive", "confidence": 0.9, "why": "upbeat"}'
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"TEST_LLM_KEY": "sk-test", "NO_PROXY": "127.0.0.1",
                                                "no_proxy": "127.0.0.1"})
        self.env.start()
        self.spec = CapabilitySpec("sentiment", "Sentiment of text", ["positive", "negative", "neutral"],
                                   open_labels=False)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def teacher(self, provider, **llm):
        d = Directives.load(ROOT / "directives.toml", {"teachers": {"llm": {
            "provider": provider, "base_url": self.url, "api_key_env": "TEST_LLM_KEY", **llm}}})
        self.store = Store(Path(self.tmp.name) / "db.sqlite")
        self.guardian = Guardian(d, self.store, Bus(), Path(self.tmp.name))
        return LLMTeacher(d.teachers.llm, self.guardian)

    def test_anthropic_messages_format(self):
        t = self.teacher("anthropic")
        self.assertTrue(t.available())
        lesson = asyncio.run(t.label(self.spec, "what a lovely day"))
        self.assertEqual((lesson.label, lesson.source), ("positive", "llm"))
        req = MockLLM.requests[-1]
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], "sk-test")
        self.assertEqual(req["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(req["body"]["model"], "claude-haiku-5-5")
        self.assertEqual(req["body"]["messages"][0]["role"], "user")
        self.assertIn("what a lovely day", req["body"]["messages"][0]["content"])
        self.assertEqual(self.guardian.llm_usage()["tokens_last_hour"], 50)

    def test_openai_compatible_format(self):
        t = self.teacher("openai", model="llama3")
        lesson = asyncio.run(t.label(self.spec, "fine"))
        self.assertEqual(lesson.label, "positive")
        req = MockLLM.requests[-1]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual([m["role"] for m in req["body"]["messages"]], ["system", "user"])

    def test_closed_label_set_rejects_invented_labels(self):
        MockLLM.reply_text = '{"label": "ecstatic", "confidence": 0.9}'
        self.assertIsNone(asyncio.run(self.teacher("anthropic").label(self.spec, "wow")))

    def test_generate_synthetic_examples(self):
        MockLLM.reply_text = '{"examples": ["loved it", "best meal ever", "  "]}'
        texts = asyncio.run(self.teacher("anthropic").generate(self.spec, "positive", 5))
        self.assertEqual(texts, ["loved it", "best meal ever"])

    def test_budget_blocks_calls(self):
        t = self.teacher("anthropic", max_calls_per_hour=1)
        asyncio.run(t.label(self.spec, "a"))
        self.assertIsNone(asyncio.run(t.label(self.spec, "b")))
        self.assertEqual(len(MockLLM.requests), 1)  # the second call never left the process

    def test_unavailable_without_key(self):
        with mock.patch.dict(os.environ, {"TEST_LLM_KEY": ""}):
            self.assertFalse(self.teacher("anthropic").available())


class HeuristicTeacherTests(unittest.TestCase):
    def test_negation_flips_label(self):
        spec = CapabilitySpec("s", labels=["positive", "negative", "neutral"], default_label="neutral",
                              lexicon={"positive": ["great"], "negative": ["awful"]},
                              negation={"positive": "negative", "negative": "positive"})
        h = HeuristicTeacher(0)
        self.assertEqual(h.label_sync(spec, "the film was great").label, "positive")
        self.assertEqual(h.label_sync(spec, "the film was not great").label, "negative")
        self.assertEqual(h.label_sync(spec, "the film was long").label, "neutral")


if __name__ == "__main__":
    unittest.main()
