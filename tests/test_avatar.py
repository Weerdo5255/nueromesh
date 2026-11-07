"""The avatar (avatar.py): its mind at a glance, the face its mood makes, and a voice from a ComfyUI
text-to-speech workflow (against a mock ComfyUI)."""
import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from neuromesh import web
from neuromesh.avatar import SpeechError, expression, glance, spoken_line
from neuromesh.runtime import System
from neuromesh.server import _route, serve_http

from test_system import FAST, make_root
from test_web import http

AUDIO = b"fLaC\x00\x00\x00\x22" + bytes(range(64))


class MockComfy(BaseHTTPRequestHandler):
    prompts: list = []
    polls: dict = {}
    mode = "ok"            # ok | fail | silent

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        if self.path != "/prompt":
            self._send(404, {"error": "no"})
            return
        MockComfy.prompts.append(body)
        self._send(200, {"prompt_id": body.get("prompt_id", "p1"), "number": len(MockComfy.prompts), "node_errors": {}})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/history/"):
            pid = url.path.rsplit("/", 1)[-1]
            MockComfy.polls[pid] = MockComfy.polls.get(pid, 0) + 1
            if MockComfy.polls[pid] < 2:  # still running
                self._send(200, {})
                return
            if MockComfy.mode == "fail":
                self._send(200, {pid: {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
                    ["execution_error", {"exception_message": "voice model not found"}]]}}})
                return
            outputs = {} if MockComfy.mode == "silent" else {
                "9": {"audio": [{"filename": "neuromesh_00001_.flac", "subfolder": "audio", "type": "output"}]}}
            self._send(200, {pid: {"outputs": outputs, "status": {"status_str": "success", "completed": True, "messages": []}}})
            return
        if url.path == "/view":
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if q == {"filename": "neuromesh_00001_.flac", "subfolder": "audio", "type": "output"}:
                self._send(200, AUDIO, "audio/flac")
                return
        self._send(404, {"error": "no"})


WORKFLOW = {
    "3": {"class_type": "KokoroTTS", "inputs": {"text": "{{text}}", "voice": "af_heart", "speed": 1.0}},
    "9": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "audio/neuromesh"}},
}


class AvatarTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MockComfy.prompts.clear()
        MockComfy.polls.clear()
        MockComfy.mode = "ok"
        self.comfy = ThreadingHTTPServer(("127.0.0.1", 0), MockComfy)
        threading.Thread(target=self.comfy.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        root = make_root(Path(self.tmp.name))
        (root / "tts.json").write_text(json.dumps(WORKFLOW))
        self.root = root
        self.system = System(root, overrides={**FAST, "memory": {"enabled": True}, "avatar": {
            "comfyui_url": f"http://127.0.0.1:{self.comfy.server_port}", "tts_workflow": "tts.json", "timeout_seconds": 10}},
            seed=3)
        await self.system.start()
        self.server = None

    async def asyncTearDown(self):
        if self.server:
            self.server.close()
        if not self.system.stopping:
            await self.system.stop()
        self.comfy.shutdown()
        self.comfy.server_close()
        self.tmp.cleanup()

    async def test_its_mind_at_a_glance(self):
        s = self.system
        s.memory.learn("What is the capital of France?", "Paris is the capital of France.", "a test", "me")
        await s.objectives.add("understand volcanoes", by="tester")
        g = glance(s)
        for key in ("mood", "feelings", "face", "state", "doing", "focus", "thought", "want", "lacks", "waiting",
                    "initiative", "holds", "line"):
            self.assertIn(key, g)
        self.assertEqual(set(g["feelings"]), {"valence", "arousal", "confidence", "trust", "patience"})
        self.assertEqual(g["focus"]["text"], "understand volcanoes")
        self.assertEqual(g["holds"]["facts"], 1)
        self.assertIn("I'm working on your objective: understand volcanoes, 0 of", g["line"])
        # its face follows its mood and state, and says only what is so
        s.affect.state.values.update(valence=0.6, arousal=0.8, trust=0.7)
        self.assertEqual(expression(s, [])[0], "happy")
        s.affect.state.values.update(valence=-0.6, arousal=0.8)
        self.assertEqual(expression(s, [])[0], "frustrated")
        self.assertEqual(expression(s, [{"kind": "research"}])[0], "frustrated")  # a bad mood shows even while busy
        s.affect.state.values.update(valence=0.0, arousal=0.4, patience=0.6, confidence=0.6)
        self.assertEqual(expression(s, [{"kind": "thinking"}])[0], "thinking")
        s.dreams.asleep = True
        self.assertEqual(glance(s)["face"], "asleep")
        self.assertTrue(glance(s)["line"].startswith("I'm asleep"))
        s.dreams.asleep = False
        s.guardian.halt("test", by="tester")
        self.assertEqual((glance(s)["face"], glance(s)["state"]), ("still", "halted"))
        self.assertEqual(spoken_line(glance(s)), "I'm halted. I won't do anything until you resume me.")
        s.guardian.resume(by="tester")
        status, body = await _route(s, "GET", "/avatar", {}, {})
        self.assertEqual((status, body["enabled"], body["voice"]["ready"]), (200, True, True))

    async def test_it_speaks_with_comfyui(self):
        s = self.system
        said = await s.voice_tts.speak("Hello! I'm *content*. See https://example.org for more.")
        self.assertEqual((said["type"], said["cached"]), ("audio/flac", False))
        self.assertEqual(said["text"], "Hello! I'm content. See for more.")  # no markdown or links aloud
        sent = MockComfy.prompts[-1]
        self.assertEqual(sent["client_id"], "neuromesh")
        self.assertEqual(sent["prompt"]["3"]["inputs"]["text"], said["text"])  # where the workflow said {{text}}
        self.assertEqual(sent["prompt"]["3"]["inputs"]["voice"], "af_heart")  # the rest of the workflow untouched
        self.assertEqual(said["where"], "KokoroTTS.text")
        self.assertEqual((s.voice_tts.dir / said["file"]).read_bytes(), AUDIO)
        again = await s.voice_tts.speak("Hello! I'm *content*. See https://example.org for more.")
        self.assertTrue(again["cached"])
        self.assertEqual(len(MockComfy.prompts), 1, "the same words are not spoken twice")
        both = await asyncio.gather(s.voice_tts.speak("Two at once."), s.voice_tts.speak("Two at once."))
        self.assertEqual(both[0]["file"], both[1]["file"])
        self.assertEqual(len(MockComfy.prompts), 2, "the same words asked twice at once are spoken once")
        self.assertTrue(any(a["kind"] == "avatar.spoke" for a in s.store.recent_audit(20)))
        # over HTTP: the API speaks how it is, and serves the audio it made (nothing else)
        self.server = await serve_http(s, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        code, _, raw = await http(port, "POST", "/avatar/speak", {})
        body = json.loads(raw)
        self.assertEqual(code, 200, body)
        self.assertTrue(body["text"].startswith("I'm "), body["text"])  # how it is, in the first person
        code, headers, data = await http(port, "GET", body["url"])
        self.assertEqual((code, headers["content-type"], data), (200, "audio/flac", AUDIO))
        code, _, _ = await http(port, "GET", "/avatar/audio/..%2Fneuromesh.db")
        self.assertEqual(code, 404)
        self.assertIsNone(s.voice_tts.audio("../../etc/passwd"))

    async def test_workflows_without_a_placeholder_and_failures(self):
        s = self.system
        plain = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "not this one"}},
                 "3": {"class_type": "ChatterboxTTS", "inputs": {"text": "Hello world", "exaggeration": 0.5}},
                 "9": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0]}}}
        (self.root / "tts.json").write_text(json.dumps(plain))
        said = await s.voice_tts.speak("The text-to-speech node gets the words.")
        sent = MockComfy.prompts[-1]["prompt"]
        self.assertEqual(sent["3"]["inputs"]["text"], "The text-to-speech node gets the words.")
        self.assertEqual(sent["1"]["inputs"]["text"], "not this one")
        self.assertIn("ChatterboxTTS.text", said["where"])
        MockComfy.mode = "fail"
        with self.assertRaisesRegex(SpeechError, "the workflow failed in ComfyUI: voice model not found"):
            await s.voice_tts.speak("This one fails.")
        MockComfy.mode = "silent"
        with self.assertRaisesRegex(SpeechError, "made no audio"):
            await s.voice_tts.speak("This one makes no sound.")
        self.assertIn("made no audio", s.voice_tts.status()["last_error"])
        (self.root / "tts.json").write_text(json.dumps({"nodes": [], "links": []}))  # saved with Save, not Save (API Format)
        with self.assertRaisesRegex(SpeechError, "API format"):
            await s.voice_tts.speak("Wrong format.")
        s.directives.avatar.comfyui_url = ""
        status, body = await _route(s, "POST", "/avatar/speak", {}, {"text": "Hello"})
        self.assertEqual((status, body["ok"]), (503, False))
        self.assertIn("no ComfyUI server set", body["error"])
        self.comfy.shutdown()  # unreachable
        s.directives.avatar.comfyui_url = "http://127.0.0.1:9"
        (self.root / "tts.json").write_text(json.dumps(WORKFLOW))
        with self.assertRaisesRegex(SpeechError, "unreachable"):
            await s.voice_tts.speak("Anyone there?")
        self.comfy = ThreadingHTTPServer(("127.0.0.1", 0), MockComfy)  # (for tearDown)
        threading.Thread(target=self.comfy.serve_forever, daemon=True).start()

    def test_the_web_page_has_an_avatar(self):
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        css = (web.WEB_DIR / "app.css").read_text()
        for part in ('id="avatar-toggle"', 'aria-pressed="false"', '<aside id="avatar"'):
            self.assertIn(part, page)
        for part in ("function avatarSvg", "function setFace", '"/avatar/speak"', "createAnalyser", "Read answers aloud",
                     "speechSynthesis", "asleep:", "frustrated:"):
            self.assertIn(part, script)
        self.assertIn("@keyframes av-blink", css)
        self.assertIn("prefers-reduced-motion", css)


if __name__ == "__main__":
    unittest.main()
