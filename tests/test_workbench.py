"""The workbench (workbench.py): whole programs it builds by itself, with no teacher, from the functions it studied in
Gitea - helpers, imports and constants found in the repository, doc examples and people's tests as checks, inputs from
argument names - checked, and run only as P20 allows. And Gitea read more often: a repository that changed comes first."""
import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_servers import NUMKIT, MockGitea  # noqa: E402
from test_system import FAST, make_root, wait_for  # noqa: E402

from neuromesh.gitea import python_header  # noqa: E402
from neuromesh.runtime import System  # noqa: E402
from neuromesh.workbench import free_names, kind_of_value, outcome, sample, standalone  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_reading_code(self):
        tree = ast.parse("def f(text, n=2):\n    out = [w for w in WORDS if w]\n    return re.sub('a', 'b', text) * n + len(out)\n")
        self.assertEqual(free_names(tree), {"WORDS", "re"})
        method = 'def area(self):\n        """Its area."""\n        return self.w * self.h'  # a method, as gitea.py keeps it
        code, tree = standalone(method)
        self.assertIsInstance(tree.body[0], ast.FunctionDef)
        self.assertIsNone(standalone("def broken(:\n    pass"))
        head = python_header('import re\nimport os.path as osp\nfrom collections import Counter\nfrom .util import helper\n'
                             'VOWELS = "aeiou"\nLIMIT: int = 3\ndef f():\n    pass\n')
        self.assertEqual([i["binds"] for i in head["imports"]], ["re", "osp", "Counter", "helper"])
        self.assertTrue(head["imports"][-1]["local"])
        self.assertEqual(set(head["constants"]), {"VOWELS", "LIMIT"})

    def test_inputs_from_names_and_what_comes_out(self):
        args = ast.parse("def f(text, n, numbers, words, path, callback, flag: bool, settings, radius): pass").body[0].args.args
        got = [sample(a, 0) for a in args]
        self.assertEqual(got, ['"Hello World, this is a test"', "3", "[3, 1, 4, 1, 5]", '["pear", "apple", "fig"]', None, None,
                               "True", '{"name": "ada", "year": 1815}', "3"])
        self.assertEqual([kind_of_value(x) for x in ("'abc'", "3.5", "[1]", "{}", "True", "<object>")],
                         ["text", "number", "list", "dict", "truth", "other"])
        out = outcome("PASS a: got 1\nFAIL b: got 2, expected 3\nDEMO f('x') -> 'X'\nnoise\n")
        self.assertEqual((out["passed"], out["failed"], out["demos"], out["returns"]), (1, 1, 1, "text"))


class WorkbenchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gitea = MockGitea()
        self.gitea.repos["ann/numkit"] = dict(NUMKIT)
        self.tmp = tempfile.TemporaryDirectory()
        self.token = os.environ.get("GITEA_TOKEN")
        os.environ["GITEA_TOKEN"] = "s3cret"
        over = {**FAST, "tools": {**FAST["tools"], "config": {"gitea": {"url": self.gitea.url}}},
                "gitea": {"study_every_seconds": 3600, "practice": False}, "workbench": {"every_seconds": 3600, "max_waiting": 3}}
        self.system = System(make_root(Path(self.tmp.name), gitea=True), overrides=over, seed=5)
        await self.system.start()
        s = self.system
        self.assertTrue(await wait_for(lambda: "gitea" in s.tools.tools and s.tools.tools["gitea"].state == "pending_approval", 20))
        apv = next(a for a in s.guardian.pending_approvals() if a.subject == "gitea")
        s.guardian.decide(apv.id, True, by="tester")
        self.assertTrue(await wait_for(lambda: s.tools.tools["gitea"].state == "active", 30), s.tools.tools["gitea"].note)
        for repo in ("ann/numkit", "ann/textkit"):
            self.assertTrue((await s.code_study.visit(repo=repo))["ok"])

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.gitea.stop()
        if self.token is None:
            os.environ.pop("GITEA_TOKEN", None)
        else:
            os.environ["GITEA_TOKEN"] = self.token
        self.tmp.cleanup()

    def example(self, name):
        return next(e for e in self.system.code_study.examples(200) if e["name"] == name)

    async def test_it_builds_whole_programs_from_what_it_studied(self):
        s = self.system
        w = s.workbench
        head = s.code_study.header("ann/numkit", "numkit/core.py")  # what the file sets up at the top was kept
        self.assertEqual({i["binds"] for i in head["imports"]}, {"math", "Counter"})
        self.assertEqual(head["constants"], {"VOWELS": 'VOWELS = "aeiou"'})
        rec = await w.build(self.example("count_vowels"))
        code = rec["code"]
        self.assertEqual(rec["status"], "built")
        self.assertIn('VOWELS = "aeiou"', code)                                    # its constant, from its file
        self.assertIn("def test_count_vowels():", code)                             # the test people wrote for it
        self.assertEqual(code.count("def test_count_vowels():"), 1)
        self.assertIn("_report(\"CHECK\", 'count_vowels(\"banana\")', lambda: count_vowels(\"banana\"), '3')", code)  # its doc example
        self.assertIn('_report("DEMO"', code)
        self.assertEqual(rec["checks"], 2)
        compile(code, "program.py", "exec")
        rec = await w.build(self.example("vowel_share"))
        self.assertEqual(rec["helpers"], ["count_vowels"])                          # a function it calls, from the same file
        self.assertIn("def count_vowels(text):", rec["code"])
        rec = await w.build(self.example("most_common_letter"))
        self.assertIn("from collections import Counter", rec["code"])               # as its file imported it
        rec = await w.build(self.example("circle_area"))
        self.assertIn("import math", rec["code"])
        self.assertIn("circle_area(3)", rec["code"])                                # a radius is a number
        rec = await w.build(self.example("broken"))
        self.assertEqual((rec["status"], rec["unresolved"]), ("incomplete", ["missing_helper"]))
        rec = await w.build(self.example("load_settings"))
        self.assertNotIn('_report("DEMO"', rec["code"])                            # it can't make up a file

    async def test_its_programs_run_only_as_p20_allows_and_it_learns_from_them(self):
        s = self.system
        w = s.workbench
        self.assertEqual(s.scratch.mode("auto_run"), "ask")
        rec = await w.experiment(example=self.example("count_vowels")["id"])
        self.assertEqual(rec["status"], "checks out")
        item = s.scratch.get(rec["item"])
        self.assertEqual((item["kind"], item["origin"]["how"], item["runs"]), ("experiment", "workbench", []))  # it waits
        self.assertIn("auto-run is off", rec["line"])
        ok, _ = s.scratch.approve(item["id"], "tester")                              # a person runs it
        self.assertTrue(ok)
        self.assertTrue(await wait_for(lambda: w.programs(5)[0]["status"] == "passed", 20), w.programs(5)[0])
        p = w.programs(5)[0]
        self.assertEqual((p["passed"], p["failed"], p["returns"]), (2, 0, "number"))
        self.assertTrue(any(e.source == "experiment" for e in s.store.examples("codekind")))  # what ran teaches the code-kind node
        # a person switches auto-run to harmless: what reaches for nothing runs by itself
        s.scratch.set_mode("auto_run", "harmless", "tester")
        rec = await w.experiment(example=self.example("slugify")["id"])
        self.assertEqual(rec["status"], "passed", rec)
        self.assertIn("1 check(s): it ran and passed 1 check(s)", rec["line"])
        rec = await w.experiment(example=self.example("load_settings")["id"])
        self.assertEqual(rec["status"], "checks out")                                # it opens files: it still asks
        self.assertIn("reaches for files", rec["line"])
        for name in ("shout", "shout_vowels"):
            self.assertIn((await w.experiment(example=self.example(name)["id"]))["status"], ("ran", "passed"))
        # chains: what one returns is what the other takes
        rec = await w.experiment(chain=True)
        self.assertEqual(rec["kind"], "chain")
        self.assertIn(rec["status"], ("ran", "passed"))
        self.assertIn(" -> ", rec["name"])
        # only the newest few wait in the scratch pad to be run
        waiting = [i for i in s.scratch.items.values() if i.get("kind") == "experiment" and not i.get("runs")]
        self.assertLessEqual(len(waiting), 3)
        st = w.status()
        self.assertGreaterEqual(st["counts"].get("passed", 0), 2)

    async def test_with_no_teacher_a_program_it_built_comes_before_a_copied_example(self):
        s = self.system
        w = s.workbench
        s.scratch.set_mode("auto_run", "harmless", "tester")
        await w.experiment(example=self.example("circle_area")["id"])
        item = await s.scratch.write("write a python function for the area of a circle with a given radius", "tester")
        self.assertEqual(item["origin"]["how"], "workbench", item["origin"])
        self.assertIn("Built by its workbench", item["code"])
        self.assertIn("import math", item["code"])
        # practice is on its own, without the teacher ([gitea] practice_teacher)
        called = []

        async def write_code(*a, **k):
            called.append(a)
            return None
        s.teachers.has_llm = lambda: True
        s.teachers.write_code = write_code
        rec = await s.scratch.practice()
        self.assertIsNotNone(rec)
        self.assertEqual(called, [])

    async def test_gitea_is_read_more_often_and_what_changed_comes_first(self):
        s = self.system
        import tomllib
        from neuromesh.config import GiteaPolicy
        d = tomllib.loads((Path(__file__).resolve().parent.parent / "directives.toml").read_text())["gitea"]
        self.assertEqual((d["study_every_seconds"], d["restudy_days"], d["practice_teacher"]), (30, 0.25, False))
        self.assertEqual((GiteaPolicy().study_every_seconds, GiteaPolicy().restudy_days), (30.0, 0.25))
        self.assertTrue((await s.code_study.visit())["ok"])                          # the one it had not read: ann/webtools
        res = await s.code_study.visit()
        self.assertFalse(res["ok"])                                                  # nothing new, nothing changed
        self.gitea.repos["ann/numkit"]["numkit/extra.py"] = 'def double(n):\n    """Twice n."""\n    return n * 2\n'
        self.gitea.updated["ann/numkit"] = "2099-01-01T00:00:00Z"                    # Gitea says it changed
        res = await s.code_study.visit()
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["repo"], res["files"]), ("ann/numkit", 1))            # only what changed is read again
        self.assertTrue(any(e["name"] == "double" for e in s.code_study.examples(200)))

    async def test_a_big_repository_is_finished_over_visits_that_follow_at_once(self):
        s = self.system
        from neuromesh.gitea import Apprentice
        s.directives.gitea.files_per_visit, s.directives.gitea.files_per_call = 2, 1
        self.gitea.repos["ann/big"] = {f"big/m{i}.py": f'def part_{i}(n):\n    """Add {i} to n."""\n    return n + {i}\n' for i in range(5)}
        res = await s.code_study.visit(repo="ann/big")
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["files"], res["left"]), (2, 3))
        self.assertEqual(s.code_study.status()["backlog"], {"ann/big": 3})
        res = await s.code_study.visit()                                             # the unfinished one comes first
        self.assertEqual((res["repo"], res["files"], res["left"]), ("ann/big", 2, 1))
        res = await s.code_study.visit()
        self.assertEqual((res["repo"], res["files"], res["left"]), ("ann/big", 1, 0))
        self.assertEqual(s.code_study.status()["backlog"], {})
        self.assertEqual(sum(e["name"].startswith("part_") for e in s.code_study.examples(200)), 5)
        self.assertLess(s.directives.gitea.catch_up_seconds, s.directives.gitea.study_every_seconds)
        self.assertTrue(any(isinstance(a, Apprentice) for a in s.agents))

    async def test_it_never_practises_the_same_challenge_twice(self):
        s = self.system
        n = s.store.db.execute("SELECT COUNT(DISTINCT lang || purpose) FROM code_examples").fetchone()[0]
        seen = []
        for _ in range(n + 5):
            rec = await s.scratch.practice()
            if rec is None:
                break
            seen.append(rec["example"])
        self.assertEqual(len(seen), len(set(seen)))                                  # each function once
        self.assertGreaterEqual(len(seen), 5)
        self.assertLessEqual(len(seen), n)
        self.assertIsNone(await s.scratch.practice())                                # nothing new: it does not start over
        self.assertEqual(s.scratch.practised_count(), len(seen))
        for repo in ("ann/numkit", "ann/textkit"):                                   # studied again: still done
            s.store.db.execute("UPDATE code_examples SET practised = 0")
        self.assertIsNone(await s.scratch.practice())
        self.gitea.repos["ann/numkit"]["numkit/extra.py"] = 'def triple(n):\n    """Three times n."""\n    return n * 3\n'
        self.gitea.updated["ann/numkit"] = "2099-01-01T00:00:00Z"
        self.assertTrue((await s.code_study.visit(repo="ann/numkit"))["ok"])
        rec = await s.scratch.practice()                                             # something new to practise on
        self.assertEqual(rec["name"], "triple")
        self.assertIsNone(await s.scratch.practice())

    async def test_the_web_page_and_the_api(self):
        from neuromesh.commands import execute
        from neuromesh.server import _route
        s = self.system
        code, out = await _route(s, "POST", "/workbench", {}, {"experiment": True, "example": self.example("shout")["id"]})
        self.assertEqual(code, 200, out)
        code, st = await _route(s, "GET", "/code", {}, {})
        self.assertEqual(st["workbench"]["programs"][0]["name"], "shout")
        code, got = await _route(s, "GET", "/workbench/program", {"id": out["id"]}, {})
        self.assertIn("def shout(text):", got["code"])
        lines = (await execute(s, "/workbench")).lines
        self.assertTrue(any("shout" in ln for ln in lines))


if __name__ == "__main__":
    unittest.main()
