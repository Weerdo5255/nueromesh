"""Projects (projects.py): it focuses on a coding project in Gitea and works on the objectives a person sets; a
change goes to a new branch and a pull request only once a person approves exactly that change (P20)."""
import os
import tempfile
import unittest
from pathlib import Path

from neuromesh.projects import add_docstrings, kind_of_objective, tidy
from neuromesh.runtime import System
from neuromesh.server import _route

from mock_servers import MockGitea, MockOllama
from test_system import FAST, make_root, wait_for

CONVERT = '''def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32


def kph_to_mph(kph):
    return kph * 0.621371
'''
MESSY = "def add(a, b):   \n    return a + b\t\n"


class HelperTests(unittest.TestCase):
    def test_what_needs_no_judgement(self):
        out, n = add_docstrings(CONVERT)
        self.assertEqual(n, 2)
        self.assertIn('    """Celsius to fahrenheit."""\n    return c * 9 / 5 + 32', out)
        compile(out, "x", "exec")
        self.assertEqual(add_docstrings(out)[1], 0)
        self.assertEqual(tidy(MESSY)[0], "def add(a, b):\n    return a + b\n")
        self.assertEqual(kind_of_objective("write tests for stats.py"), "tests")
        self.assertEqual(kind_of_objective("add docstrings to every function"), "docs")
        self.assertEqual(kind_of_objective("fix the crash on empty lines"), "fix")


class ProjectTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gitea = MockGitea()
        self.gitea.repos["ann/textkit"]["textkit/convert.py"] = CONVERT
        self.gitea.repos["ann/textkit"]["textkit/messy.py"] = MESSY
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {k: os.environ.get(k) for k in ("GITEA_TOKEN", "GITEA_WRITE_TOKEN")}
        os.environ["GITEA_TOKEN"] = "s3cret"
        os.environ.pop("GITEA_WRITE_TOKEN", None)
        over = {**FAST, "tools": {**FAST["tools"], "config": {"gitea": {"url": self.gitea.url}}},
                "gitea": {"study_every_seconds": 3600, "practice": False}, "projects": {"work_every_seconds": 3600}}
        self.system = System(make_root(Path(self.tmp.name), gitea=True), overrides=over, seed=7)
        await self.system.start()
        s = self.system
        self.assertTrue(await wait_for(lambda: "gitea" in s.tools.tools and s.tools.tools["gitea"].state == "pending_approval", 20))
        apv = next(a for a in s.guardian.pending_approvals() if a.subject == "gitea")
        self.assertEqual(apv.detail["secrets"], ["GITEA_TOKEN", "GITEA_WRITE_TOKEN"])
        s.guardian.decide(apv.id, True, by="tester")
        self.assertTrue(await wait_for(lambda: s.tools.tools["gitea"].state == "active", 30), s.tools.tools["gitea"].note)

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.gitea.stop()
        for k, v in self.env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.tmp.cleanup()

    async def test_it_works_on_a_project_and_opens_pull_requests_once_approved(self):
        s = self.system
        p = s.projects
        self.assertFalse(p.add_objective("add docstrings", "tester")[0])            # a project first
        ok, msg = p.set_focus("ann/textkit", "", "tester")
        self.assertTrue(ok, msg)
        ok, msg, obj = p.add_objective("add docstrings to every function in textkit/convert.py", "tester")
        self.assertEqual(obj["kind"], "docs")
        res = await p.work(by="tester")
        self.assertTrue(res["ok"], res)
        ch = res["change"]
        self.assertEqual((ch["status"], ch["how"], [f["path"] for f in ch["files"]]),
                         ("awaiting approval", ["mechanical"], ["textkit/convert.py"]))
        self.assertIn('+    """Celsius to fahrenheit."""', ch["files"][0]["diff"])
        self.assertTrue(ch["files"][0]["check"]["ok"])
        self.assertEqual(self.gitea.writes, [])                                     # nothing written before approval
        self.assertIsNone(p.next_objective())                                       # (its change is waiting)
        # approved, but no write token: it cannot push, and says so
        ok, msg = p.approve(ch["id"], "tester")
        self.assertTrue(ok)
        self.assertTrue(await wait_for(lambda: p.changes[ch["id"]]["status"] in ("failed", "pushed"), 20))
        self.assertEqual(p.changes[ch["id"]]["status"], "failed")
        self.assertIn("GITEA_WRITE_TOKEN", p.changes[ch["id"]]["error"])
        # with a write token
        os.environ["GITEA_WRITE_TOKEN"] = "w-secret"
        res = await p.work(by="tester")
        ch = res["change"]
        ok, msg = p.approve(ch["id"], "tester")
        self.assertTrue(await wait_for(lambda: p.changes[ch["id"]]["status"] in ("failed", "pushed"), 20))
        got = p.changes[ch["id"]]
        self.assertEqual(got["status"], "pushed", got.get("error"))
        self.assertEqual(obj["status"], "in review")                                # until a person merges or closes it
        branch = got["branch"]
        self.assertTrue(branch.startswith("neuromesh/add-docstrings"), branch)
        self.assertIn('"""Kph to mph."""', self.gitea.branches[("ann/textkit", branch)]["textkit/convert.py"])
        self.assertNotIn('"""', self.gitea.repos["ann/textkit"]["textkit/convert.py"])  # main is untouched
        pr = self.gitea.pulls[("ann/textkit", 1)]
        self.assertEqual((pr["head"], pr["base"]), (branch, "main"))
        self.assertIn("add docstrings to every function", pr["body"])
        self.assertTrue(all(w[3] == "token w-secret" for w in self.gitea.writes if w[1].endswith(("/contents", "/pulls"))))
        # it never writes to an existing branch
        bad, why = await s.code_study.call({"op": "change", "repo": "ann/textkit", "base": "main", "branch": "main",
                                            "message": "x", "files": [{"path": "a.py", "content": "x = 1\n"}]})
        self.assertIsNone(bad)
        self.assertIn("only writes to new branches", why)
        # merged on Gitea: the objective is done, and that counts as a reward
        pr["merged"], pr["state"] = True, "closed"
        self.assertEqual(await p.poll(), 1)
        self.assertEqual((p.changes[ch["id"]]["status"], obj["status"]), ("merged", "done"))
        self.assertTrue(any(r["value"] > 0 and r["target"]["ref"] == ch["id"] for r in s.rewards.given))
        # declined: the objective stays open, and that counts as a punishment
        ok, msg, tidy_obj = p.add_objective("tidy the whitespace in textkit/messy.py", "tester")
        res = await p.work(by="tester")
        self.assertEqual(res["change"]["objective_kind"], "cleanup")
        ok, _ = p.decline(res["change"]["id"], "tester")
        self.assertTrue(ok)
        self.assertEqual(tidy_obj["status"], "open")
        self.assertTrue(any(r["value"] < 0 and r["target"]["ref"] == res["change"]["id"] for r in s.rewards.given))
        # an objective it needs a teacher for
        p.add_objective("refactor textkit/stats.py to use statistics.fmean", "tester")
        p.set_objective(tidy_obj["id"], "drop", "tester")
        res = await p.work(by="tester")
        self.assertFalse(res["ok"])
        self.assertIn("needs a teacher", res["why"])
        status, body = await _route(s, "GET", "/projects", {}, {})
        self.assertEqual(body["focus"]["repo"], "ann/textkit")
        self.assertTrue(body["can_write"])
        self.assertTrue(any(e.topic == "project.change" for e in s.bus.recent("project.*", 20)))
        self.assertIn("ann/textkit", "\n".join(p.lines()))

    async def test_a_teacher_changes_files_for_an_objective(self):
        s = self.system
        replies = iter([{"content": "def parse_settings(text:\n    pass\n", "summary": "first try"},
                        {"content": TEXT_FIXED, "summary": "comment lines are skipped"}])
        llm = MockOllama(lambda call: next(replies))
        try:
            from neuromesh.config import LLMConfig
            from neuromesh.teachers import LLMTeacher
            t = LLMTeacher("ollama", LLMConfig.from_dict({"provider": "ollama", "base_url": llm.url, "model": "llama3.1:8b"}),
                           s.guardian, s.bus)
            await t.refresh(force=True)
            s.teachers.teachers["ollama"] = t
            s.teachers.order.insert(1, "ollama")
            s.projects.set_focus("ann/textkit", "main", "tester")
            s.projects.add_objective("fix textkit/config.py so parse_settings skips comment lines", "tester")
            res = await s.projects.work(by="tester")
            self.assertTrue(res["ok"], res)
            ch = res["change"]
            self.assertEqual((ch["how"], ch["status"]), (["the ollama teacher"], "awaiting approval"))
            self.assertIn('startswith("#")', ch["files"][0]["diff"])
            self.assertIn("did not check out", llm.calls[1]["user"])               # fixed once, with the problems
            self.assertIn("textkit/config.py", llm.calls[0]["user"])
        finally:
            llm.stop()


TEXT_FIXED = '''import json


def parse_settings(text):
    """Parse key=value lines into a dictionary."""
    out = {}
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_settings(path):
    """Read settings from a JSON file."""
    with open(path) as f:
        return json.load(f)
'''


if __name__ == "__main__":
    unittest.main()
