"""Its coder nodes learn to program from a local Gitea (gitea.py), and write code in a scratch pad where a validator
checks it and nothing runs until a person approves exactly that code (scratchpad.py, P20)."""
import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from neuromesh.gitea import brace_functions, kind_of, purpose_of, python_functions, words_of
from neuromesh.runtime import System
from neuromesh.scratchpad import Validator, code_request, shared
from neuromesh.server import _route

from mock_servers import TEXTKIT, WEBTOOLS, MockGitea, MockOllama
from test_system import FAST, make_root, wait_for


class ReadingCodeTests(unittest.TestCase):
    def test_it_takes_functions_apart(self):
        fns = {f["name"]: f for f in python_functions(TEXTKIT["textkit/words.py"] + TEXTKIT["textkit/stats.py"]
                                                      + TEXTKIT["textkit/config.py"])}
        self.assertEqual(set(fns), {"slugify", "reverse_words", "shout", "mean", "hypotenuse", "parse_settings", "load_settings"})
        self.assertEqual(kind_of(fns["shout"]), "text")
        self.assertEqual(kind_of(fns["mean"]), "maths")
        self.assertEqual(kind_of(fns["hypotenuse"]), "maths")
        self.assertEqual(kind_of(fns["parse_settings"]), "parse")
        self.assertEqual(kind_of(fns["load_settings"]), "files")
        self.assertEqual(kind_of(python_functions(TEXTKIT["tests/test_words.py"])[0]), "test")
        self.assertEqual(purpose_of("reverse_words", fns["reverse_words"]["doc"], ["sentence"]),
                         "reverse words. Reverse the words in a sentence. given sentence")
        js = {f["name"]: f for f in brace_functions(WEBTOOLS["web/client.js"], "javascript")}
        self.assertEqual(set(js), {"fetchUser", "capitalize"})
        self.assertEqual(js["capitalize"]["doc"], "Capitalize the first letter of a word.")
        self.assertEqual(kind_of(js["fetchUser"]), "network")
        go = brace_functions(WEBTOOLS["cmd/days.go"], "go")
        self.assertEqual((go[0]["name"], kind_of(go[0])), ("DaysUntil", "dates"))
        self.assertEqual(words_of("parseHTTPDate"), "parse http date")

    def test_the_validator(self):
        v = Validator()
        self.assertTrue(v.check("def f(x):\n    return x * 2\n", "python")["ok"])
        bad = v.check("def f(x:\n    return x\n", "python")
        self.assertFalse(bad["ok"])
        self.assertIn("line 1", bad["problems"][0])
        reach = v.check("import os, subprocess, urllib.request\nopen('x').read()\n", "python")["touches"]
        self.assertTrue(any("network" in r for r in reach) and any("files" in r for r in reach)
                        and any("processes" in r for r in reach), reach)
        self.assertTrue(v.check('{"a": 1}', "json")["ok"])
        self.assertFalse(v.check('{"a": }', "json")["ok"])
        self.assertTrue(v.check('a = 1\n', "toml")["ok"])
        if shutil.which("node"):
            self.assertTrue(v.check("function f(a) { return a + 1; }\n", "javascript")["ok"])
            self.assertFalse(v.check("function f(a { return a; }\n", "javascript")["ok"])
        if shutil.which("bash"):
            self.assertTrue(v.check("for i in 1 2; do echo $i; done\n", "shell")["ok"])
            self.assertFalse(v.check("if true; then echo x\n", "shell")["ok"])
        self.assertIsNone(v.check("fn main() { let x = 1; }", "rust")["ok"])      # (no checker: brackets only)
        self.assertFalse(v.check("fn main() { let x = (1; }", "rust")["ok"])
        self.assertEqual(code_request("write a python function that reverses each word"), ("reverses each word", "python"))
        self.assertEqual(code_request("Could you write me a JavaScript snippet to capitalize a word?"), ("capitalize a word", "javascript"))
        self.assertIsNone(code_request("write a tool that reverses text"))   # (the toolsmith's)
        self.assertIsNone(code_request("what is code golf?"))
        self.assertGreater(shared("def mean(xs): return sum(xs) / len(xs)", "def average(numbers): return sum(numbers) / len(numbers)"), 0.2)


class ApprovalTests(unittest.IsolatedAsyncioTestCase):
    """Approving code runs exactly that code, also after a restart (approvals live in memory; what waits on one is
    kept, so approval numbers carry on and never name two things)."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=FAST, seed=5)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_approving_runs_that_code_after_a_restart(self):
        s = self.system
        old = s.scratch.add('print("old")\n', "python", "old", "tester")
        self.assertTrue(s.scratch.ask(old["id"], "tester")[0])                # waits for approval when it stops
        ran = s.scratch.add('print("ran")\n', "python", "ran", "tester")
        self.assertTrue(s.scratch.approve(ran["id"], "tester")[0])
        self.assertTrue(await wait_for(lambda: ran["status"] == "ran", 30))
        edited = s.scratch.add('print("first")\n', "python", "edited", "tester")
        s.scratch.ask(edited["id"], "tester")
        first = edited["approval"]
        s.scratch.edit(edited["id"], 'print("second")\n', "tester")             # changed while it waited:
        self.assertEqual(s.guardian.approvals[first].status, "withdrawn")       # the question is withdrawn
        self.assertNotIn(first, [a.id for a in s.guardian.pending_approvals()])
        self.assertTrue(s.scratch.approve(edited["id"], "tester")[0])            # approving runs what is there now
        self.assertTrue(await wait_for(lambda: edited["status"] == "ran", 30))
        self.assertEqual(edited["runs"][-1]["stdout"].strip(), "second")
        last = max(int(a.split("-")[1]) for a in s.guardian.approvals)
        await s.stop()

        self.system = s = System(self.root, overrides=FAST, seed=5)
        await s.start()
        self.assertTrue(all(int(a.split("-")[1]) > last for a in s.guardian.approvals))  # numbers carry on
        waiting = s.scratch.get(old["id"])
        self.assertEqual(waiting["status"], "awaiting approval")                # it asks again for what was waiting
        self.assertIsNotNone(s.guardian.approval_for(waiting["approval"], "code", old["id"]))
        new = s.scratch.add('print("new")\n', "python", "new", "tester")
        ok, msg = s.scratch.approve(new["id"], "tester")
        self.assertTrue(ok, msg)
        self.assertTrue(await wait_for(lambda: new["status"] == "ran", 30))
        self.assertEqual(new["runs"][-1]["stdout"].strip(), "new")              # the code that was approved ran
        self.assertEqual((waiting["status"], waiting.get("runs", [])), ("awaiting approval", []))  # and nothing else
        # an approval for one thing never acts on another that holds the same number
        s.projects.changes["ch-99"] = {"id": "ch-99", "status": "awaiting approval", "approval": waiting["approval"]}
        s.guardian.decide(waiting["approval"], True, by="tester")
        self.assertTrue(await wait_for(lambda: waiting["status"] == "ran", 30))
        self.assertEqual(s.projects.changes["ch-99"]["status"], "awaiting approval")
        self.assertIsNone(s.guardian.approval_for(waiting["approval"], "project-change", "ch-99"))


class AutoRunAndToolTests(unittest.IsolatedAsyncioTestCase):
    """The auto-run switch for code it writes, and its own code kept as tools (codetools.py)."""

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides=FAST, seed=5)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_auto_run(self):
        s, sp = self.system, self.system.scratch
        self.assertEqual(sp.mode("auto_run"), "ask")
        self.assertFalse(sp.set_mode("auto_run", "always", "neuromesh")[0])       # only a person switches it
        self.assertFalse(sp.set_mode("auto_run", "always", "teacher:ollama")[0])
        self.assertFalse(sp.set_mode("auto_run", "sometimes", "tester")[0])
        item = await sp.write("write a python function that reverses the word hello", by="tester")
        self.assertEqual((item["status"], item["runs"]), ("valid", []))         # ask: it waits
        ok, msg = sp.set_mode("auto_run", "harmless", "tester")
        self.assertTrue(ok, msg)
        res = await s.submit("write a python function that reverses the word stressed")
        self.assertIn("It ran by itself (auto-run: harmless, switched on by tester)", res.output)
        made = sp.get(res.output.split(" as ")[1].split(" ")[0])
        self.assertIn(f"It printed:\n{made['runs'][-1]['stdout'].strip()}", res.output)
        self.assertEqual(made["status"], "ran")
        self.assertTrue(made["runs"][-1]["by"].startswith("auto-run"))
        self.assertTrue(any(e.topic == "code.mode" for e in s.bus.recent("code.*", 40)))
        reach = sp._new("fetch a page", "import urllib.request\nprint(urllib.request.urlopen('http://x').read())\n",
                        "python", "tester", origin={"how": "teacher", "teacher": "ollama"})
        ok, why = sp.auto_runs(reach)
        self.assertFalse(ok)                                                    # harmless only: it asks
        self.assertIn("the network", why)
        sp.set_mode("auto_run", "always", "tester")
        self.assertTrue(sp.auto_runs(reach)[0])                                 # always: it would run
        mine = sp.add('print("hi")\n', "python", "mine", "tester")
        self.assertFalse(sp.auto_runs(mine)[0])                                 # people's code always asks
        self.assertEqual(mine["status"], "valid")
        status, body = await _route(s, "POST", "/code", {}, {"auto_run": "ask", "by": "tester"})
        self.assertEqual((status, body["auto_run"]), (200, "ask"))
        self.assertFalse(sp.auto_runs(reach)[0])

    async def test_its_code_becomes_a_tool(self):
        s, sp, ct = self.system, self.system.scratch, self.system.codetools
        mine = sp.add('def add(a, b):\n    """Add two numbers."""\n    return a + b\n\n\nprint(add(2, 3))\nprint(add(10, -4))\n',
                      "python", "add two numbers", "tester")
        ok, msg = ct.make(mine["id"], "tester")
        self.assertFalse(ok)
        self.assertIn("hasn't run", msg)                                       # it shows what it does first
        sp.approve(mine["id"], "tester")
        self.assertTrue(await wait_for(lambda: mine["status"] == "ran", 30))
        p, why = ct.candidate(mine)
        self.assertEqual(p["examples"], [{"input": "2, 3", "expect": "5"}, {"input": "10, -4", "expect": "6"}])
        ok, msg = ct.make(mine["id"], "tester")                                 # a person makes it a tool: installed
        self.assertTrue(ok, msg)
        self.assertTrue(await wait_for(lambda: s.tools.tools.get("add") is not None and s.tools.tools["add"].state == "active", 30),
                        s.tools.tools.get("add") and s.tools.tools["add"].note)
        self.assertEqual(s.tools.tools["add"].manifest.permissions, ())        # made tools get no permissions
        res = await s.submit("tool add: 40, 2")
        self.assertEqual(str(res.output), "42")
        self.assertEqual(mine["tool"]["name"], "add")
        # it offers code it wrote itself ([code] make_tools = "ask"): a person approves
        sp.set_mode("auto_run", "harmless", "tester")
        item = await sp.write("write a python function that reverses the word hello", by="tester")
        self.assertEqual(item["status"], "ran")
        self.assertTrue(await wait_for(lambda: item["id"] in ct.made, 10))
        name = ct.made[item["id"]]["tool"]
        draft = s.toolsmith.drafts[name]
        self.assertEqual(draft["status"], "awaiting approval")
        s.guardian.decide(draft["approval"], True, by="tester")
        self.assertTrue(await wait_for(lambda: s.tools.tools.get(name) is not None and s.tools.tools[name].state == "active", 30),
                        s.tools.tools.get(name) and s.tools.tools[name].note)
        res = await s.submit(f"tool {name}: stressed")
        self.assertEqual(str(res.output), "desserts")
        # code that can't be a tool says why
        bare = sp.add("print(sum(range(10)))\n", "python", "a sum", "tester")
        sp.approve(bare["id"], "tester")
        self.assertTrue(await wait_for(lambda: bare["status"] == "ran", 30))
        self.assertIn("no function", ct.make(bare["id"], "tester")[1])
        net = sp.add("import socket\n\n\ndef host(name):\n    return socket.gethostbyname(name)\n\n\nprint(host('localhost'))\n",
                     "python", "host", "tester")
        sp.approve(net["id"], "tester")
        self.assertTrue(await wait_for(lambda: net["status"] in ("ran", "failed"), 30))
        if net["status"] == "ran":
            self.assertIn("a tool may not", ct.make(net["id"], "tester")[1])  # reaching outside: never a tool
        self.assertFalse(sp.set_mode("make_tools", "auto", "neuromesh")[0])
        status, body = await _route(s, "POST", "/code", {}, {"make_tools": "auto", "by": "tester"})
        self.assertEqual((status, body["make_tools"]), (200, "auto"))


class CodeSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gitea = MockGitea()
        self.tmp = tempfile.TemporaryDirectory()
        self.token = os.environ.get("GITEA_TOKEN")
        os.environ["GITEA_TOKEN"] = "s3cret"
        os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "sk-test-not-for-tools")
        over = {**FAST, "tools": {**FAST["tools"], "config": {"gitea": {"url": self.gitea.url}}},
                "gitea": {"study_every_seconds": 3600, "practice": False}}
        self.system = System(make_root(Path(self.tmp.name), gitea=True), overrides=over, seed=5)
        await self.system.start()
        s = self.system
        self.assertTrue(await wait_for(lambda: "gitea" in s.tools.tools and s.tools.tools["gitea"].state == "pending_approval", 20))
        apv = next(a for a in s.guardian.pending_approvals() if a.subject == "gitea")
        self.assertEqual(apv.detail["secrets"], ["GITEA_TOKEN", "GITEA_WRITE_TOKEN"])  # the approval says what it is given
        s.guardian.decide(apv.id, True, by="tester")                        # a person approves the plugin (P3)
        self.assertTrue(await wait_for(lambda: s.tools.tools["gitea"].state == "active", 30), s.tools.tools["gitea"].note)

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.gitea.stop()
        if self.token is None:
            os.environ.pop("GITEA_TOKEN", None)
        else:
            os.environ["GITEA_TOKEN"] = self.token
        if os.environ.get("ANTHROPIC_API_KEY") == "sk-test-not-for-tools":
            os.environ.pop("ANTHROPIC_API_KEY", None)
        self.tmp.cleanup()

    async def test_it_studies_gitea_and_learns_from_it(self):
        s = self.system
        res = await s.code_study.visit()
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["repo"], "ann/textkit")
        self.assertEqual((res["files"], res["functions"]), (4, 8))  # the README is not code
        self.assertIn("token s3cret", self.gitea.auth)                # the plugin is given $GITEA_TOKEN, for its Gitea
        kinds = {e["name"]: e["kind"] for e in s.code_study.examples(50)}
        self.assertEqual((kinds["mean"], kinds["shout"], kinds["load_settings"], kinds["test_slugify"]),
                         ("maths", "text", "files", "test"))
        taught = {e.text: e.label for e in s.store.examples("codekind")}
        self.assertEqual(taught.get("mean. Calculate the average of a list of numbers. given numbers"), "maths")
        self.assertTrue(any(e.source == "gitea" for e in s.store.examples("coder")))  # "reverse the words ..." -> reverse
        res = await s.code_study.visit()
        self.assertEqual(res["repo"], "ann/webtools")
        self.assertEqual(res["functions"], 3)
        res = await s.code_study.visit()                               # it studied both lately
        self.assertFalse(res["ok"])
        self.gitea.repos["ann/textkit"]["textkit/stats.py"] += '''

def median(numbers):
    """The middle value of a list of numbers."""
    ordered = sorted(numbers)
    return ordered[len(ordered) // 2]
'''
        res = await s.code_study.visit(repo="ann/textkit")             # only the file that changed is read again
        self.assertEqual((res["files"], res["functions"]), (1, 3))
        best = s.code_study.similar("work out the average of some numbers", k=1)
        self.assertEqual(best[0][0]["name"], "mean")
        lines = "\n".join(s.code_study.lines())
        self.assertIn("ann/textkit", lines)
        self.assertTrue(any(e.topic == "gitea.studied" for e in s.bus.recent("gitea.*", 10)))
        status, body = await _route(s, "GET", "/code", {}, {})
        self.assertEqual(body["gitea"]["state"], "ready")
        self.assertGreaterEqual(s.tools.tools["gitea"].runs, 6)           # it went through the tool, in its sandbox
        res = await s.submit("tool gitea: textkit")                        # an order to the tool: the repositories that match
        self.assertIn("ann/textkit", str(res.output))
        s.tool_switch.set_tool("gitea", "off", "tester")                    # the tool switch applies to it
        self.assertFalse(s.code_study.enabled)
        res = await s.code_study.visit()
        self.assertFalse(res["ok"])
        self.assertIn("switched off", res["why"])
        s.tool_switch.set_tool("gitea", "default", "tester")
        state = s.code_study.set_url("http://127.0.0.1:9", "tester")         # nobody there: it says so
        self.assertEqual(state, "active")
        res = await s.code_study.visit()
        self.assertFalse(res["ok"])
        self.assertEqual(s.code_study.state, "unreachable")
        self.assertEqual(s.tools.tools["gitea"].consecutive_failures, 0)  # Gitea being down is not a broken tool

    async def test_a_tool_never_gets_a_teachers_key(self):
        from neuromesh.guardian import Verdict
        from neuromesh.tools import ToolManifest
        s = self.system
        m = ToolManifest.load(Path(self.tmp.name) / "plugins" / "gitea")
        self.assertEqual(m.secrets, ("GITEA_TOKEN", "GITEA_WRITE_TOKEN"))
        env = s.sandbox._env(m)
        self.assertEqual(env.get("GITEA_TOKEN"), "s3cret")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        greedy = ToolManifest(name="greedy", path=m.path, digest="x", permissions=("network",), secrets=("ANTHROPIC_API_KEY",))
        self.assertEqual(s.guardian.authorize_tool(greedy).verdict, Verdict.DENY)
        self.assertNotIn("ANTHROPIC_API_KEY", s.sandbox._env(greedy))      # and the sandbox withholds it anyway
        quiet = ToolManifest(name="quiet", path=m.path, digest="y", permissions=(), secrets=("GITEA_TOKEN",))
        self.assertEqual(s.guardian.authorize_tool(quiet).verdict, Verdict.NEEDS_APPROVAL)  # a secret always asks

    async def test_the_scratch_pad_runs_only_what_a_person_approved(self):
        s = self.system
        res = await s.submit("write a python function that reverses the word hello")
        self.assertEqual(res.route, "scratch")
        item = s.scratch.get(res.output.split(" as ")[1].split(" ")[0])
        self.assertEqual((item["origin"]["how"], item["status"]), ("recipe", "valid"))
        self.assertIn("/code approve", res.output)
        self.assertIn("def reverse_text", item["code"])
        out = await s.scratch.run(item["id"], "nobody")                 # not approved: it does not run
        self.assertFalse(out["ok"])
        self.assertIn("approved", out["error"])
        ok, msg = s.scratch.ask(item["id"], "tester")
        self.assertTrue(ok)
        self.assertEqual(item["status"], "awaiting approval")
        apv = item["approval"]
        self.assertIn(apv, [a.id for a in s.guardian.pending_approvals()])
        s.guardian.decide(apv, True, by="tester")                       # a person approves: it runs
        self.assertTrue(await wait_for(lambda: item["status"] in ("ran", "failed"), 30))
        self.assertEqual(item["status"], "ran", item["runs"])
        self.assertTrue(item["runs"][-1]["stdout"].strip())
        ok, _ = s.scratch.edit(item["id"], item["code"] + "\nprint('more')\n", "tester")
        self.assertTrue(ok)
        self.assertIsNone(item["approved_digest"])                      # changed: the approval was for other code
        out = await s.scratch.run(item["id"], "tester")
        self.assertFalse(out["ok"])
        bad = s.scratch.add("def broken(:\n    pass\n", "python", "broken", "tester")
        self.assertEqual(bad["status"], "invalid")
        ok, msg = s.scratch.ask(bad["id"], "tester")
        self.assertFalse(ok)
        self.assertIn("does not check out", msg)
        mine = s.scratch.add("print(sum(range(10)))\n", "python", "a sum", "tester")
        ok, _ = s.scratch.approve(mine["id"], "tester")
        self.assertTrue(ok)
        self.assertTrue(await wait_for(lambda: mine["status"] == "ran", 30))
        self.assertEqual(mine["runs"][-1]["stdout"].strip(), "45")
        status, body = await _route(s, "POST", "/code", {}, {"id": mine["id"], "action": "remove", "by": "tester"})
        self.assertEqual(status, 200)
        self.assertIsNone(s.scratch.get(mine["id"]))
        self.assertTrue(any(e.topic == "code.ran" for e in s.bus.recent("code.*", 20)))

    async def test_it_starts_from_the_closest_example_and_practises(self):
        s = self.system
        await s.code_study.visit()
        code, origin = await s.scratch.draft("calculate the average of a list of numbers", "python", exclude=None)
        if origin["how"] == "example":                                   # (no recipe fits; no teacher)
            self.assertIn("Adapted from ann/textkit/textkit/stats.py (mean)", code)
            self.assertEqual(origin["examples"][0]["name"], "mean")
        rec = await s.scratch.practice()
        self.assertIsNotNone(rec)
        self.assertIn(rec["how"], ("example", "recipe", "nothing", "teacher"))
        self.assertEqual(s.scratch.stats["practised"], 1 if rec["how"] != "nothing" else 0)
        self.assertTrue(any(e.topic == "code.practised" for e in s.bus.recent("code.*", 10)))

    async def test_a_teacher_writes_code_following_the_examples(self):
        s = self.system
        await s.code_study.visit()
        replies = iter([{"code": "def average(xs:\n    return 0\n", "explanation": "first try"},
                        {"code": "def average(xs):\n    return sum(xs) / len(xs)\n\nprint(average([1, 2, 3]))\n",
                         "explanation": "the mean"}])
        llm = MockOllama(lambda call: next(replies))
        try:
            from neuromesh.config import LLMConfig
            from neuromesh.teachers import LLMTeacher
            t = LLMTeacher("ollama", LLMConfig.from_dict({"provider": "ollama", "base_url": llm.url, "model": "llama3.1:8b"}),
                           s.guardian, s.bus)
            await t.refresh(force=True)
            s.teachers.teachers["ollama"] = t
            s.teachers.order.insert(1, "ollama")
            item = await s.scratch.write("work out the average of numbers in a list", "tester", "python")
            self.assertEqual(item["origin"]["how"], "teacher")
            self.assertEqual(item["status"], "valid")                    # the first try did not check out: fixed
            self.assertIn("return sum(xs) / len(xs)", item["code"])
            self.assertIn("Example 1 - mean from ann/textkit", llm.calls[0]["user"])
            self.assertIn("did not check out", llm.calls[1]["user"])
        finally:
            llm.stop()


if __name__ == "__main__":
    unittest.main()
