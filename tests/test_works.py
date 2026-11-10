"""Node budgets set from the web page (budgets.py); the director's optical, coder, thinking and ego nodes
(agency.py: look, code, reflect, revisit, and its wiring); and its works (works.py): earlier work it goes back to
and improves on its own initiative, and long-term projects of its own."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_system import FAST, make_root  # noqa: E402
import test_workbench as wb_tests  # noqa: E402

from neuromesh.agency import MODES, drives, feasible, policy  # noqa: E402
from neuromesh.runtime import System  # noqa: E402
from neuromesh.server import _route  # noqa: E402

QUIET = {"skills": "ok", "gap": "none", "topics": "none", "research": "off", "exams": "none", "scores": "none",
         "memory": "ok", "last": "none", "streak": "one", "budget": "ok", "selftasks": "none", "mood": "steady",
         "needs": "none", "newtools": "none", "aims": "none", "doubts": "none", "sight": "none", "coding": "none",
         "thoughts": "none", "works": "none"}


class PolicyTests(unittest.TestCase):
    def test_the_new_modes_have_drives_of_their_own(self):
        self.assertEqual(policy(QUIET)[0], "rest")
        for flag, value, mode in [("sight", "lessons", "look"), ("sight", "practice", "look"), ("coding", "ready", "code"),
                                  ("thoughts", "due", "reflect"), ("works", "due", "revisit")]:
            f = {**QUIET, flag: value}
            self.assertEqual(policy(f)[0], mode, f)
            self.assertIn(mode, feasible(f))
            self.assertNotIn(mode, feasible(QUIET))
        self.assertTrue(all(m in MODES for m in ("look", "code", "reflect", "revisit")))
        f = {**QUIET, "works": "due", "last": "revisit", "streak": "many", "coding": "ready"}  # variety: not the same again
        self.assertEqual(policy(f)[0], "code")
        self.assertGreater(drives({**QUIET, "memory": "failing", "works": "due"})["consolidate"], 0.6)


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_node_budgets_are_set_by_a_person_and_hold(self):
        tmp = tempfile.TemporaryDirectory()
        root = make_root(Path(tmp.name))
        s = System(root, overrides=FAST, seed=2)
        await s.start()
        try:
            was = s.directives.nodes.max_nodes
            code, st = await _route(s, "GET", "/budgets", {}, {})
            rows = {b["path"]: b for b in st["budgets"]}
            self.assertEqual((rows["nodes.max_nodes"]["value"], rows["nodes.max_nodes"]["default"], st["changed"]), (was, was, 0))
            self.assertIsNotNone(rows["nodes.max_nodes"]["used"])
            code, out = await _route(s, "POST", "/budgets", {}, {"path": "nodes.max_nodes", "value": was + 20, "by": "ann"})
            self.assertEqual(code, 200, out)
            self.assertEqual(s.directives.nodes.max_nodes, was + 20)                       # at once
            self.assertTrue(s.guardian.authorize_spawn(was + 5, 0, "x").allowed)          # the guardian reads the same budget
            self.assertEqual(out["changed"], 1)
            self.assertEqual({b["path"]: b for b in out["budgets"]}["nodes.max_nodes"]["set_by"], "ann")
            self.assertTrue(any(e["kind"] == "budget.set" for e in s.store.recent_audit(20)))
            for bad in ({"path": "nodes.max_nodes", "value": 1}, {"path": "nodes.max_nodes", "value": "many"},
                        {"path": "agency.max_decisions_per_hour", "value": 5}):
                code, out = await _route(s, "POST", "/budgets", {}, bad)
                self.assertEqual(code, 400, out)
            s.budgets.set("memory.facts_per_node", 7, "ann")
        finally:
            await s.stop()
        s2 = System(root, overrides=FAST, seed=2)                                          # across a restart
        self.assertEqual((s2.directives.nodes.max_nodes, s2.directives.memory.facts_per_node), (was + 20, 7))
        ok, msg = s2.budgets.reset("all", "ann")
        self.assertTrue(ok)
        self.assertEqual(s2.directives.nodes.max_nodes, was)
        self.assertEqual(s2.budgets.status()["changed"], 0)
        s2.store.close()
        tmp.cleanup()


class WorksTests(wb_tests.WorkbenchTests):
    """(the workbench tests' mesh: a mock Gitea with two repositories studied)"""
    test_it_builds_whole_programs_from_what_it_studied = None
    test_its_programs_run_only_as_p20_allows_and_it_learns_from_them = None
    test_with_no_teacher_a_program_it_built_comes_before_a_copied_example = None
    test_gitea_is_read_more_often_and_what_changed_comes_first = None
    test_the_web_page_and_the_api = None

    async def test_the_director_can_call_on_its_optical_coder_thinking_and_ego_nodes(self):
        s = self.system
        d = s.director
        self.assertFalse({w["family"]: w for w in d.wiring()}["thinker"]["wired"])   # memory is off in these tests: it says so
        s.directives.memory.enabled = True
        wiring = {w["family"]: w for w in d.wiring()}
        self.assertEqual(set(wiring), {"vision", "coder", "thinker", "ego"})
        self.assertTrue(all(w["wired"] for w in wiring.values()), wiring)
        self.assertEqual(wiring["vision"]["modes"], ["look"])
        self.assertIn("code", wiring["coder"]["modes"])
        self.assertIn("reflect", wiring["thinker"]["modes"])
        self.assertTrue(wiring["coder"]["available"])                                # it has studied code
        self.assertEqual({n["capability"] for n in wiring["coder"]["nodes"]}, {"codekind", "coder", "planner"})
        self.assertEqual(wiring["ego"]["nodes"][0]["capability"], "ego")
        sit = d.situation()
        self.assertEqual(sit["flags"]["coding"], "ready")
        self.assertIn("code", sit["feasible"])
        # pinned to each, the director has that family of nodes work
        ok, _ = d.set_mode("code", "tester")
        self.assertTrue(ok)
        rec = await d.decide(force=True)
        self.assertEqual((rec["mode"], rec["target"]), ("code", "workbench"))
        self.assertIn("into a whole program", rec["summary"])
        self.assertGreater(rec["value"], 0.2)
        s.memory.learn("what is the capital of Peru?", "Lima", "test", by="tester", kind="people", reviewed=True)
        s.memory.learn("what is the capital of Chile?", "Santiago", "test", by="tester", kind="people", reviewed=True)
        self.assertEqual(d.situation()["flags"]["thoughts"], "due")
        d.set_mode("reflect", "tester")
        rec = await d.decide(force=True)
        self.assertEqual(rec["mode"], "reflect")
        self.assertIn("thinking nodes reflected", rec["summary"])
        self.assertEqual(d.situation()["flags"]["thoughts"], "none")                 # until memory holds something new
        if s.vision.can_draw and s.vision.fonts:
            self.assertEqual(d.situation()["flags"]["sight"], "practice")
            d.set_mode("look", "tester")
            rec = await d.decide(force=True)
            self.assertEqual(rec["mode"], "look")
            self.assertIn("practised reading", rec["summary"])
        # its ego's say: what it is fascinated by comes first among its topics
        s.ego.fascinate("glaciers", "it came up twice")
        self.assertIn("glaciers", d.situation()["fascinated"])
        self.assertEqual(len(s.director.status()["wiring"]), 4)

    async def test_it_goes_back_to_earlier_work_and_improves_it(self):
        s = self.system
        w = s.works
        s.scratch.set_mode("auto_run", "harmless", "tester")
        rec = await s.workbench.experiment(example=self.example("broken")["id"])
        self.assertEqual(rec["status"], "incomplete")
        cands = w.candidates()
        self.assertEqual([(c["kind"], c["why"]) for c in cands], [("program", "it has not gone back to it yet")])
        self.assertIn("missing missing_helper", cands[0]["state"])
        self.assertEqual(s.director.situation()["flags"]["works"], "due")
        visit = await w.session(by="tester")                                           # nothing has changed: no better
        self.assertFalse(visit["improved"])
        self.assertIn("still incomplete", visit["line"])
        self.assertEqual(w.candidates(), [])                                           # it rests, and does not loop
        self.assertEqual(s.director.situation()["flags"]["works"], "none")
        # someone writes the missing helper; Gitea is studied again; the work is due because what it was made from changed
        self.gitea.repos["ann/numkit"]["numkit/helpers.py"] = 'def missing_helper(text):\n    """How long a text is."""\n    return len(text)\n'
        self.gitea.updated["ann/numkit"] = "2099-01-01T00:00:00Z"
        self.assertTrue((await s.code_study.visit())["ok"])
        cands = w.candidates()
        self.assertEqual(cands[0]["why"], "what it was made from has changed since")
        s.director.set_mode("revisit", "tester")                                       # on its own initiative
        rec = await s.director.decide(force=True)
        self.assertEqual((rec["mode"], rec["target"]), ("revisit", "its works"))
        self.assertIn("incomplete -> ", rec["summary"])
        self.assertGreater(rec["value"], 0.8)
        last = w.journal[-1]
        self.assertTrue(last["improved"])
        self.assertEqual((last["before"], last["after"]), ("incomplete", "ran"))
        # code it wrote that does not check out: written again, the earlier version kept; a person's code is left alone
        item = s.scratch._new("area of a circle", "def area(r:\n    return", "python", "neuromesh",
                              request="write a python function for the area of a circle with a given radius", origin={"how": "example"})
        mine = s.scratch.add("def f(:\n  pass", "python", "mine", "tester")
        await s.workbench.experiment(example=self.example("circle_area")["id"])
        self.assertEqual([c["key"] for c in w.candidates() if c["kind"] == "code"], [f"code:{item['id']}"])
        visit = await w.revisit(f"code:{item['id']}", by="tester")
        self.assertTrue(visit["improved"], visit)
        self.assertEqual(s.scratch.items[item["id"]]["status"], "valid")
        self.assertEqual(len(s.scratch.items[item["id"]]["versions"]), 1)
        self.assertIn("def area(r:", s.scratch.items[item["id"]]["versions"][0]["code"])
        self.assertEqual(s.scratch.items[mine["id"]]["code"], "def f(:\n  pass")

    async def test_long_term_projects_of_its_own(self):
        s = self.system
        w = s.works
        s.scratch.set_mode("auto_run", "harmless", "tester")
        self.assertEqual(w.due()["projects"], [])
        for name in ("count_vowels", "circle_area"):
            await s.workbench.experiment(example=self.example(name)["id"])
        self.assertEqual([p["repo"] for p in w.due()["projects"]], ["ann/numkit"])      # it starts one of its own
        proj = next(iter(w.projects.values()))
        self.assertEqual((proj["by"], proj["version"], proj["status"]), ("its own initiative", 0, "active"))
        ok, msg = w.start("ann/numkit", "tester")
        self.assertFalse(ok)
        steps = []
        for _ in range(12):                                                            # session after session
            rec = await w.session(by="tester")
            if rec is None:
                break
            steps.append(rec.get("step") or rec["kind"])
        self.assertIn("build", steps)
        self.assertIn("assemble", steps)
        self.assertGreaterEqual(proj["version"], 1)
        self.assertGreaterEqual(len(proj["parts"]), 4)
        self.assertIn("count_vowels", proj["parts"])
        self.assertNotIn("broken", proj["parts"])
        module = s.state_dir / "works" / proj["id"] / f"v{proj['version']}" / f"{proj['module']}.py"
        code = module.read_text()
        compile(code, str(module), "exec")
        self.assertEqual(code.count("def count_vowels(text):"), 1)
        self.assertIn("def circle_area(radius):", code)
        self.assertEqual(s.scratch.items[proj["item"]]["kind"], "project")
        self.assertEqual((proj["result"]["ok"], proj["result"]["failed"]), (True, 0))  # it ran (harmless), every check passed
        self.assertGreaterEqual(proj["result"]["passed"], 3)
        self.assertIn("wait until the repository changes", proj["next"])
        v = proj["version"]
        # the repository grows: the project has more to do, and a new version follows
        self.gitea.repos["ann/numkit"]["numkit/more.py"] = ('def double(n):\n    """Twice n.\n\n    >>> double(4)\n    8\n    """\n    return n * 2\n')
        self.gitea.updated["ann/numkit"] = "2099-01-01T00:00:00Z"
        self.assertTrue((await s.code_study.visit())["ok"])
        self.assertEqual(w._plan(proj)[0], "build")
        for _ in range(6):
            if await w.work(proj["id"], by="tester") is None or proj["version"] > v:
                break
        self.assertEqual(proj["version"], v + 1)
        self.assertIn("double", proj["parts"])
        self.assertTrue((s.state_dir / "works" / proj["id"] / f"v{v}").is_dir())        # earlier versions are kept
        # people stay in charge: pause, the page, the console
        code_, out = await _route(s, "POST", "/works", {}, {"project": proj["id"], "status": "paused"})
        self.assertEqual(code_, 200)
        self.assertEqual([p for p in w.due()["projects"] if p["id"] == proj["id"]], [])
        code_, st = await _route(s, "GET", "/code", {}, {})
        self.assertEqual(st["works"]["projects"][0]["plan"], "paused")
        from neuromesh.commands import execute
        self.assertTrue(any(proj["id"] in ln for ln in (await execute(s, "/works")).lines))
        self.assertTrue(any("nodes.max_nodes" in ln for ln in (await execute(s, "/budgets")).lines))


if __name__ == "__main__":
    unittest.main()
