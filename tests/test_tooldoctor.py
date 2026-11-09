"""When a tool fails: is it the tool's code, or the network, its server, its settings? A fix for its code that a
person approves (backed up, version bumped, self-tested, undone if it still fails), and a person's last word:
removing a tool from the mesh, and restoring it."""
import asyncio
import socket
import tempfile
import unittest
from pathlib import Path

from neuromesh.commands import execute
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tooldoctor import bump, check_fix, classify

from mock_servers import MockOllama
from test_system import FAST, make_root, wait_for

BROKEN = '''import json
import sys
import traceback


def rot13(text):
    out = []
    for ch in text:
        if ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            out.append(chr((ord(ch) - base + 13) % 26 + base))
        else:
            out.append(ch)
    return "".join(out).uper()


def main():
    req = json.load(sys.stdin)
    text = req["input"].split(":", 1)[-1].strip()
    print(json.dumps({"output": rot13(text)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}))
'''
WRONG_FIX = BROKEN.replace('"".join(out).uper()', '"".join(out)').replace("+ 13) % 26", "+ 12) % 26")  # runs, wrong answers
RIGHT_FIX = BROKEN.replace('"".join(out).uper()', '"".join(out)')
UNSAFE_FIX = BROKEN.replace("import json\n", "import json\nimport subprocess\n").replace('"".join(out).uper()', '"".join(out)')
MANIFEST = '''name = "rot13"
version = "1.0.0"
description = "ROT13-encode text"
keywords = ["rot13"]
examples = [{input = "rot13: abc", expect = "nop"}, {input = "rot13: Hello", expect = "Uryyb"}]
'''
FETCH = '''import json
import sys
import urllib.request


def main():
    req = json.load(sys.stdin)
    url = req["config"]["url"]
    try:
        with urllib.request.urlopen(url + "/status", timeout=2) as resp:
            print(json.dumps({"output": resp.read().decode()}))
    except OSError as exc:
        print(json.dumps({"error": f"could not reach {url}: {exc}", "transient": True}))


if __name__ == "__main__":
    main()
'''
FETCH_MANIFEST = '''name = "status_check"
version = "1.0.0"
description = "Ask a server for its status"
permissions = ["network"]
keywords = ["status"]
examples = [{input = "status of the server", expect = "ok"}]
'''


def closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class ClassifyTests(unittest.TestCase):
    def test_causes(self):
        trace = 'Traceback (most recent call last):\n  File "/x/plugins/rot13/tool.py", line 14, in rot13\n' \
                "AttributeError: 'str' object has no attribute 'uper'"
        cause, why, _ = classify(["AttributeError: 'str' object has no attribute 'uper'"], [trace], ["tool.py"])
        self.assertEqual(cause, "code")
        self.assertIn("tool.py line 14, in rot13", why)
        self.assertEqual(classify(["could not reach http://nas:8080: [Errno 111] Connection refused"], [], ["tool.py"])[0], "network")
        self.assertEqual(classify(["the server returned HTTP 503"], [], ["tool.py"])[0], "service")
        self.assertEqual(classify(["SearXNG answered 403: format not allowed"], [], ["tool.py"])[0], "config")
        self.assertEqual(classify(["ModuleNotFoundError: No module named 'lxml'"], [], ["tool.py"])[0], "environment")
        self.assertEqual(classify(["timeout after 10s"], [], ["tool.py"])[0], "limits")
        self.assertEqual(classify(["expected '500', got '501'"], [], ["tool.py"])[0], "code", "numbers in an answer are not HTTP codes")
        self.assertEqual(classify([""], [], ["tool.py"])[0], "unknown")

    def test_a_fix_may_not_do_more_than_the_code_it_replaces(self):
        self.assertEqual(check_fix(BROKEN, RIGHT_FIX, "tool.py"), [])
        self.assertIn("tool.py now imports subprocess", check_fix(BROKEN, UNSAFE_FIX, "tool.py"))
        self.assertIn("doesn't compile", check_fix(BROKEN, "def broken(:\n", "tool.py")[0])
        self.assertEqual(check_fix(FETCH, FETCH.replace("timeout=2", "timeout=5"), "tool.py"), [], "what it already did is fine")
        self.assertEqual(bump("1.0.9"), "1.0.10")


class DoctorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixes = []      # what the teacher answers, in turn
        self.reviews = []

        def teacher(call):
            if "maintain the tools" in call["system"]:
                self.reviews.append(call["user"])
                return self.fixes.pop(0) if self.fixes else {"cause": "code", "explanation": "no idea", "files": {}}
            if "fallback answerer" in call["system"]:
                return "I am not sure."
            return {"label": "positive", "confidence": 0.9}
        self.llm = MockOllama(teacher)
        self.tmp = tempfile.TemporaryDirectory()
        root = make_root(Path(self.tmp.name))
        (root / "plugins" / "rot13").mkdir()
        (root / "plugins" / "rot13" / "tool.py").write_text(BROKEN)
        (root / "plugins" / "rot13" / "manifest.toml").write_text(MANIFEST)
        self.root = root
        self.over = {**FAST, "teachers": {"order": ["human", "ollama", "heuristic"],
                                          "ollama": {"provider": "ollama", "base_url": self.llm.url, "model": "llama3.1:8b"}},
                     "mind": {"max_inquiries_per_hour": 50, "retest_quarantined_after_seconds": 0}}
        if self._testMethodName == "test_an_unsafe_fix_is_never_offered":
            self.fixes.append({"cause": "code", "explanation": "fixed it", "files": {"tool.py": UNSAFE_FIX}})
        self.fixes.append({"cause": "code", "explanation": "rot13() calls .uper(), which strings don't have.", "files": {"tool.py": WRONG_FIX}})
        self.fixes.append({"cause": "code", "explanation": "rot13() calls .uper(), which strings don't have; it "
                                    "should return the text as it is.", "files": {"tool.py": RIGHT_FIX}})
        self.s = System(root, overrides=self.over, seed=2)
        await self.s.start()

    async def asyncTearDown(self):
        if not self.s.stopping:
            await self.s.stop()
        self.llm.stop()
        self.tmp.cleanup()

    def fix_approval(self):
        return next((a for a in self.s.guardian.pending_approvals() if a.kind == "tool.fix"), None)

    async def test_a_fault_in_its_code_is_found_fixed_and_tested(self):
        s, doc = self.s, self.s.tooldoctor
        self.assertTrue(await wait_for(lambda: s.tools.tools["rot13"].state == "quarantined", 20))
        # it looks into why, reviews the code with its teacher, and asks to install a fix
        self.assertTrue(await wait_for(lambda: self.fix_approval() is not None, 20), doc.history)
        d = doc.diagnoses["rot13"]
        self.assertEqual((d["cause"], d["where"]), ("code", "code"))
        self.assertIn("'str' object has no attribute 'uper'", self.reviews[0], "the teacher sees the traceback")
        self.assertIn("def rot13(text):", self.reviews[0], "and the code")
        apv = self.fix_approval()
        self.assertEqual((apv.subject, apv.detail["version"], apv.detail["to_version"]), ("rot13", "1.0.0", "1.0.1"))
        self.assertIn('-    return "".join(out).uper()', apv.detail["diff"])
        self.assertTrue(any(q.approval_id == apv.id for q in s.mind.open_inquiries()), "it asks in conversation too")
        code, st = await _route(s, "GET", "/status", {}, {})
        self.assertEqual(st["tools"]["rot13"]["diagnosis"]["fix"]["status"], "awaiting approval")
        self.assertIn("uper", (Path(self.root) / "plugins" / "rot13" / "tool.py").read_text(), "nothing changes before approval")

        # approved: installed, version bumped, old files backed up; the first fix still fails, so it is undone ...
        s.guardian.decide(apv.id, True, by="Chris")
        fix1 = next(f for f in doc.fixes.values() if f["approval"] == apv.id)
        self.assertTrue(await wait_for(lambda: fix1["status"] == "rolled back", 30), fix1)
        self.assertTrue(Path(fix1["backup"], "tool.py").read_text() == BROKEN)
        # ... and it tries again, told why the first one didn't work
        self.assertTrue(await wait_for(lambda: self.fix_approval() is not None, 30), doc.history)
        self.assertIn("Your previous fix did not work", self.reviews[-1])
        self.assertEqual((Path(self.root) / "plugins" / "rot13" / "tool.py").read_text(), BROKEN, "the old files are back")
        apv2 = self.fix_approval()
        s.guardian.decide(apv2.id, True, by="Chris")
        self.assertTrue(await wait_for(lambda: s.tools.tools["rot13"].state == "active", 30), s.tools.tools["rot13"].note)
        fix2 = next(f for f in doc.fixes.values() if f["approval"] == apv2.id)
        self.assertTrue(await wait_for(lambda: fix2["status"] == "worked", 10), fix2)
        self.assertEqual(s.tools.tools["rot13"].manifest.version, "1.0.1")
        self.assertEqual(len(s.guardian.pending_approvals()), 0, "no second approval for the files it already approved")
        res = await s.submit("tool rot13: Hello")
        self.assertEqual(res.output, "Uryyb")
        kinds = {a["kind"] for a in s.store.recent_audit(200)}
        self.assertTrue({"tool.diagnosed", "tool.fix_drafted", "tool.fix_applied", "tool.fix_rolled_back", "tool.fix_worked"} <= kinds)
        self.assertIn("rot13", "\n".join((await execute(s, "/doctor")).lines))

    async def test_an_unsafe_fix_is_never_offered(self):
        s, doc = self.s, self.s.tooldoctor
        self.assertTrue(await wait_for(lambda: any(f["status"] == "failed" for f in doc.fixes.values()), 20), doc.history)
        fix = next(iter(doc.fixes.values()))
        self.assertIn("now imports subprocess", fix["result"])
        self.assertIsNone(self.fix_approval())

    async def test_remove_and_restore(self):
        s = self.s
        self.assertTrue(await wait_for(lambda: s.tools.tools.get("calculator") and s.tools.tools["calculator"].state == "active", 20))
        code, r = await _route(s, "POST", "/tools/remove", {}, {"tool": "calculator", "by": "Chris", "why": "testing"})
        self.assertEqual((code, r["ok"]), (200, True), r)
        self.assertEqual(s.tools.tools["calculator"].state, "removed")
        self.assertNotIn("tool:calculator", s.dispatch.active_labels())
        res = await s.submit("tool calculator: 2 + 2")
        self.assertFalse(res.ok and res.route == "tool:calculator", res.note)
        self.assertIn("tool is removed", res.note)
        await asyncio.sleep(0.8)  # a few scans
        self.assertEqual(s.tools.tools["calculator"].state, "removed", "scans pass it by")
        self.assertTrue((Path(self.root) / "plugins" / "calculator" / "tool.py").exists(), "its files stay")
        code, st = await _route(s, "GET", "/status", {}, {})
        self.assertEqual(st["tools"]["calculator"]["removed"]["by"], "Chris")
        code, r = await _route(s, "POST", "/tools/remove", {}, {"tool": "calculator"})
        self.assertEqual(code, 400)
        await s.stop()
        # a restart keeps it removed
        s2 = System(self.root, overrides=self.over, seed=2)
        await s2.start()
        try:
            self.assertEqual(s2.tools.tools["calculator"].state, "removed")
            lines = (await execute(s2, "/tool calculator restore")).lines
            self.assertIn("restored calculator", lines[0])
            self.assertTrue(await wait_for(lambda: getattr(s2.tools.tools.get("calculator"), "state", "") == "active", 20),
                            getattr(s2.tools.tools.get("calculator"), "note", "not scanned yet"))
            self.assertIn("tool:calculator", s2.dispatch.active_labels())
        finally:
            await s2.stop()


class OutsideTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_server_that_isnt_there_is_not_a_fault_in_the_code(self):
        reviews = []

        def teacher(call):
            if "maintain the tools" in call["system"]:
                reviews.append(call)
            return {"label": "positive", "confidence": 0.9}
        llm = MockOllama(teacher)
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(Path(tmp))
            (root / "plugins" / "status_check").mkdir()
            (root / "plugins" / "status_check" / "tool.py").write_text(FETCH)
            (root / "plugins" / "status_check" / "manifest.toml").write_text(FETCH_MANIFEST)
            url = f"http://127.0.0.1:{closed_port()}"
            s = System(root, overrides={**FAST, "tools": {**FAST["tools"], "auto_approve_permissions": ["network"],
                                                          "config": {"status_check": {"url": url}}},
                                        "teachers": {"order": ["human", "ollama", "heuristic"],
                                                     "ollama": {"provider": "ollama", "base_url": llm.url, "model": "llama3.1:8b"}}},
                       seed=2)
            await s.start()
            try:
                doc = s.tooldoctor
                self.assertTrue(await wait_for(lambda: "status_check" in doc.diagnoses, 20), doc.history)
                d = doc.diagnoses["status_check"]
                self.assertEqual((d["cause"], d["where"]), ("network", "outside"))
                self.assertIn("refused the connection", d["probe"])
                self.assertIn("server is running", d["advice"])
                self.assertEqual(reviews, [], "nothing to fix in the code: its teacher isn't asked to")
                self.assertFalse(any(a.kind == "tool.fix" for a in s.guardian.pending_approvals()))
                self.assertTrue(any("status_check" in n.subject and "the network" in n.text for n in s.mind.notes.values()))
                code, r = await _route(s, "POST", "/tools/diagnose", {}, {"tool": "status_check"})
                self.assertEqual(code, 200)
                self.assertIn("the network", r["message"])
            finally:
                await s.stop()
                llm.stop()


if __name__ == "__main__":
    unittest.main()
