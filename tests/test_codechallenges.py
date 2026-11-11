"""Code challenges from the Hugging Face Hub (codechallenges.py): downloaded through the dataset viewer, solved by
itself in the scratch pad with the challenge's own tests deciding, never attempted twice; practice solutions studied,
exam ones never. And it is always on at least one long programming project (works.py): a library of its solutions."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_servers import DATASETS, MockDatasets  # noqa: E402
from test_system import FAST, make_root, wait_for  # noqa: E402

from neuromesh.codechallenges import SUITES, Solver, from_row  # noqa: E402
from neuromesh.config import CodeChallengePolicy, WorksPolicy, deep_merge  # noqa: E402
from neuromesh.runtime import System  # noqa: E402
from neuromesh.works import Journeyman  # noqa: E402


class RowTests(unittest.TestCase):
    def test_dataset_rows_become_challenges(self):
        row = DATASETS["google-research-datasets/mbpp"]["sanitized"]["train"]["rows"][0]
        ch = from_row("mbpp", row, "practice")
        self.assertEqual((ch["task"], ch["entry"], ch["split"]), ("1", "count_vowels", "practice"))
        self.assertEqual(len(ch["tests"]), 2)
        self.assertIn("count the vowels", ch["prompt"])
        rows = DATASETS["openai/openai_humaneval"]["openai_humaneval"]["test"]["rows"]
        got = [from_row("humaneval", r, "") for r in rows]
        self.assertEqual(got[0]["entry"], "double_it")
        self.assertEqual(got[0]["prompt"], "Write a python function double_it: Return twice the number n.")
        self.assertIn("def check(candidate):", got[0]["tests"][0])
        self.assertEqual({g["split"] for g in got}, {"practice", "exam"})         # split once, by its name
        self.assertEqual([g["split"] for g in got], [from_row("humaneval", r, "")["split"] for r in rows])
        self.assertIsNone(from_row("mbpp", {"task_id": 5, "prompt": "", "code": "", "test_list": []}, "practice"))
        self.assertEqual(set(SUITES), {"mbpp", "humaneval"})

    def test_directives(self):
        import tomllib
        d = tomllib.loads((Path(__file__).resolve().parent.parent / "directives.toml").read_text())
        self.assertEqual((d["codechallenges"]["start_with"], d["codechallenges"]["teacher"]), (["mbpp", "humaneval"], False))
        self.assertEqual(tuple(CodeChallengePolicy().start_with), ("mbpp", "humaneval"))
        self.assertEqual((d["works"]["always_one"], WorksPolicy().always_one), (True, True))


class ChallengeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.hub = MockDatasets()
        self.tmp = tempfile.TemporaryDirectory()
        over = deep_merge(FAST, {"challenges": {"api_url": self.hub.url, "hub_url": self.hub.url}})
        self.system = System(make_root(Path(self.tmp.name)), overrides=over, seed=3)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.hub.stop()
        self.tmp.cleanup()

    async def get_all(self):
        c = self.system.codechallenges
        self.assertEqual((await c.fetch("mbpp"))["added"], 6)
        self.assertEqual((await c.fetch("humaneval"))["added"], 4)
        return c

    async def test_it_downloads_them_and_solves_them_by_itself_never_twice(self):
        s = self.system
        c = await self.get_all()
        self.assertEqual((await c.fetch("mbpp"))["have"], 6)                         # the same ones are not kept twice
        with self.assertRaises(ValueError):
            await c.fetch("nosuch")
        s.scratch.set_mode("auto_run", "harmless", "tester")                        # P20: a person lets harmless code run
        called = []

        async def write_code(*a, **k):
            called.append(a)
            return None
        s.teachers.has_llm = lambda: True
        s.teachers.write_code = write_code
        done = []
        for _ in range(14):
            rec = await c.attempt()
            if rec is None:
                break
            done.append(rec)
        self.assertEqual(called, [])                                                # on its own, without a teacher
        self.assertEqual(len(done), 10)
        self.assertEqual(len({r["id"] for r in done}), 10)                          # each challenge once
        self.assertIsNone(await c.attempt())                                        # none left: it does not start over
        self.assertIsNone(await c.attempt(done[0]["id"]))                           # nor when one is named
        self.assertEqual(c.count("status = 'new'"), 0)
        solved = [r for r in done if r["status"] == "solved"]
        self.assertGreaterEqual(len(solved), 1, [r["line"] for r in done])
        self.assertTrue(all(r["status"] in ("solved", "failed", "no attempt") for r in done))
        one = c.get(solved[0]["id"])
        self.assertIn(one["entry"], one["code"])                                    # named what the challenge calls it
        self.assertEqual(s.scratch.get(one["item"])["kind"], "challenge")
        # practice challenges teach it afterwards; exam challenges are never studied
        studied = {r[0] for r in s.store.db.execute("SELECT path FROM code_examples WHERE repo LIKE 'hub:%'").fetchall()}
        practice = {c.get(r["id"])["task"] for r in done if r["split"] == "practice"}
        self.assertEqual(studied, practice)
        self.assertTrue(all(c.get(r["id"])["studied"] == (1 if r["split"] == "practice" else 0) for r in done))
        st = c.status()
        mbpp = next(x for x in st["suites"] if x["name"] == "mbpp")
        self.assertEqual((mbpp["have"], mbpp["new"], mbpp["exam"]["attempted"]), (6, 0, 3))
        self.assertIsNotNone(mbpp["exam"]["score"])
        self.assertEqual(st["stats"]["attempts"], 10)
        # what it failed is worth going back to once it has studied more - and only then
        failed = [r["id"] for r in done if r["status"] == "failed" and r["split"] == "practice"]
        if failed:
            keys = [x["key"] for x in s.works.candidates()]
            self.assertIn(f"challenge:{failed[0]}", keys)
            rec = await s.works.revisit(f"challenge:{failed[0]}", by="tester")
            self.assertIsNotNone(rec)
            self.assertNotIn(f"challenge:{failed[0]}", [x["key"] for x in s.works.candidates()])

    async def test_on_ask_its_solution_waits_for_a_person_and_is_graded_when_run(self):
        s = self.system
        c = await self.get_all()
        self.assertEqual(s.scratch.mode("auto_run"), "ask")
        rec = await c.attempt("mbpp:11")
        self.assertEqual(rec["status"], "waiting", rec)
        item = s.scratch.get(rec["item"])
        self.assertEqual(item["runs"], [])                                          # nothing ran
        self.assertIn('vowel_count("banana") == 3', item["code"])                 # the challenge's own tests
        self.assertIsNone(await c.attempt("mbpp:11"))
        ok, _ = s.scratch.approve(item["id"], "tester")
        self.assertTrue(ok)
        self.assertTrue(await wait_for(lambda: c.get("mbpp:11")["status"] in ("solved", "failed"), 20), c.get("mbpp:11"))
        self.assertEqual(c.get("mbpp:11")["passed"] + c.get("mbpp:11")["failed"], 2)
        rec = await c.attempt("mbpp:13")                                            # nothing it knows fits: it says so
        self.assertEqual(rec["status"], "no attempt", rec)

    async def test_it_is_always_on_a_long_programming_project(self):
        s = self.system
        w = s.works
        c = await self.get_all()
        s.scratch.set_mode("auto_run", "harmless", "tester")
        w.tend()                                                                    # nothing studied in Gitea: its solutions
        self.assertEqual([p["kind"] for p in w.projects.values()], ["solutions"])
        on = w.working_on()
        self.assertIsNotNone(on)
        self.assertEqual(w.status()["working_on"], on["id"])
        lines = []
        for _ in range(16):                                                         # a session at a time: solve, assemble
            rec = await w.session(by="tester")
            if rec is None:
                break
            lines.append(rec["line"])
        proj = next(iter(w.projects.values()))
        self.assertGreaterEqual(proj["version"], 1, lines)
        self.assertGreaterEqual(len(proj["parts"]), 2)
        code = s.scratch.get(proj["item"])["code"]
        for part in proj["parts"]:
            self.assertIn(part, code)
        self.assertEqual(proj["result"]["failed"], 0, proj["result"])
        self.assertEqual(c.count("status = 'new'"), 0)                              # its project attempted them, each once
        self.assertEqual(c.stats["attempts"] - sum(1 for ln in lines if "went back to" in ln), 10)
        w.tend()
        self.assertEqual(len(w.projects), 1)                                        # one such project, not one a look

    async def test_its_agents_the_api_and_the_console(self):
        from neuromesh.commands import execute
        from neuromesh.server import _route
        s = self.system
        self.assertTrue(any(isinstance(a, Solver) for a in s.agents))
        self.assertTrue(any(isinstance(a, Journeyman) for a in s.agents))
        s.directives.codechallenges.start_with = ["mbpp"]
        s.directives.codechallenges.idle_seconds = 0
        solver = next(a for a in s.agents if isinstance(a, Solver))
        await solver.step()                                                         # it gets what it starts with by itself
        self.assertEqual(s.codechallenges.count("suite = 'mbpp'"), 6)
        code, out = await _route(s, "POST", "/codechallenges", {}, {"get": "humaneval"})
        self.assertEqual((code, out["added"]), (200, 4), out)
        code, out = await _route(s, "POST", "/codechallenges", {}, {"get": "nosuch"})
        self.assertEqual(code, 400)
        code, out = await _route(s, "POST", "/codechallenges", {}, {"attempt": "mbpp:11"})
        self.assertEqual((code, out["status"]), (200, "waiting"), out)
        code, out = await _route(s, "POST", "/codechallenges", {}, {"attempt": "mbpp:11"})
        self.assertEqual(code, 400)                                                 # never the same one twice
        code, st = await _route(s, "GET", "/code", {}, {})
        self.assertEqual(st["challenges"]["recent"][0]["id"], "mbpp:11")
        code, st = await _route(s, "GET", "/codechallenges", {}, {})
        self.assertEqual(sum(x["have"] for x in st["suites"]), 10)
        code, one = await _route(s, "GET", "/codechallenges/one", {"id": "mbpp:11"}, {})
        self.assertEqual((one["split"], one["solution"]), ("exam", ""))            # exam answers are not shown
        lines = (await execute(s, "/codechallenges")).lines
        self.assertTrue(lines[0].startswith("Code challenges from the Hugging Face Hub"))
        self.assertTrue(any("nothing to attempt" in ln for ln in (await execute(s, "/codechallenges try mbpp:11")).lines))


if __name__ == "__main__":
    unittest.main()
