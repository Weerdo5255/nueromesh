"""The toolsmith (P15): coder nodes that write tools from the code library, ask permission before any
of it runs, install what a person approves, and experiment with it on their own initiative."""
import asyncio
import itertools
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.codelib import RECIPES
from neuromesh.commands import execute
from neuromesh.mind import Unmet
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tools import outputs_match
from neuromesh.toolsmith import (CODER, NONE, NOT_TOOLS, CoderTeacher, _lead_in, check_code, cue_scores, tool_request,
                                 tool_source)
from neuromesh.web import mesh

from test_system import FAST, make_root, wait_for

SMITH = {**FAST, "toolsmith": {"builds_per_hour": 50, "experiments_per_hour": 50}}


class CodeLibraryTests(unittest.TestCase):
    def test_every_recipe_writes_a_tool_that_passes_its_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for r in RECIPES.values():
                source = tool_source(r, {"triggers": list(r.triggers), **r.params}, f"{r.name}: a test")
                self.assertEqual(check_code(source), [], r.id)
                path = Path(tmp) / f"{r.id}.py"
                path.write_text(source)
                for tmpl, slots, expect in r.probes:
                    keys = list(slots)
                    for combo in list(itertools.product(*(slots[k] for k in keys)))[:2]:
                        vals = dict(zip(keys, combo))
                        text = tmpl.format(**vals)
                        out = subprocess.run([sys.executable, "-I", str(path)], input=json.dumps({"input": text}),
                                             capture_output=True, text=True, timeout=20)
                        data = json.loads(out.stdout.strip().splitlines()[-1])
                        self.assertNotIn("error", data, (r.id, text, data))
                        self.assertTrue(outputs_match(data["output"], expect(vals)), (r.id, text, data, expect(vals)))

    def test_the_code_check_keeps_out_what_reaches_outside(self):
        ok = 'import json\nimport sys\n\ndef main():\n    print(json.dumps({"output": 1}))\n'
        self.assertEqual(check_code(ok), [])
        for bad, why in (("import os\n", "imports os"), ("import subprocess\n", "imports subprocess"),
                         ("from socket import socket\n", "imports from socket"), ("open('x')\n", "uses open"),
                         ("eval('1')\n", "uses eval"), ("exec('1')\n", "uses exec"), ("__import__('os')\n", "uses __import__"),
                         ("import json\njson.loads.__globals__\n", "uses .__globals__"), ("def main(:\n", "does not parse")):
            problems = check_code(bad + ok)
            self.assertTrue(any(why in p for p in problems), (bad, problems))

    def test_the_coder_teacher_knows_which_recipe_a_request_needs(self):
        t = CoderTeacher(seed=1)
        spec = type("Spec", (), {"name": CODER})()
        for rid in RECIPES:
            texts = asyncio.run(t.generate(spec, rid, 5))
            self.assertTrue(texts, rid)
            for text in texts:
                self.assertEqual(t.label_sync(text).label, rid, text)
        for text in NOT_TOOLS:
            self.assertEqual(t.label_sync(text).label, NONE, text)
        self.assertEqual(asyncio.run(t.generate(spec, NONE, 3))[0] in NOT_TOOLS, True)
        self.assertIsNone(asyncio.run(t.label(type("S", (), {"name": "sentiment"})(), "reverse it")))
        self.assertEqual(max(cue_scores("translate sos to morse code").items(), key=lambda kv: kv[1])[0], "morse")

    def test_a_person_can_ask_for_a_tool(self):
        self.assertEqual(tool_request("make a tool that reverses text"), "reverses text")
        self.assertEqual(tool_request("Could you please write me a tool to count the vowels in a word?"),
                         "count the vowels in a word")
        self.assertEqual(tool_request("build a new tool for roman numerals"), "roman numerals")
        self.assertIsNone(tool_request("make a cake"))
        self.assertIsNone(tool_request("what tool would reverse text"))

    def test_a_refinement_learns_only_what_it_can_check(self):
        r = RECIPES["count_letters"]
        params = {"triggers": list(r.triggers)}
        odd = "tally up the vowels found within banana"
        self.assertNotEqual(r.block(odd, params)["output"], 3)
        trig = _lead_in(r, params, odd, "3")
        self.assertEqual(trig, "tally up the vowels found within")
        self.assertEqual(r.block("tally up the vowels found within orbit", {"triggers": params["triggers"] + [trig]})["output"], 2)
        self.assertIsNone(_lead_in(r, params, odd, None))  # no known answer: nothing to check a fix against


class ToolsmithTests(unittest.IsolatedAsyncioTestCase):
    overrides = SMITH

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.overrides, seed=11)
        await self.system.start()

    async def asyncTearDown(self):
        await self.system.stop()
        self.tmp.cleanup()

    async def active(self, name, version=None):
        s = self.system
        return await wait_for(lambda: name in s.tools.tools and s.tools.tools[name].state == "active"
                              and (version is None or s.tools.tools[name].manifest.version == version), 20)

    async def test_it_writes_a_tool_asks_permission_and_tries_it_out(self):
        s, smith = self.system, self.system.toolsmith
        res = await s.submit("reverse the word lantern")
        self.assertFalse(res.ok)
        self.assertTrue(any("toolsmith could write one from the 'reverse' recipe" in t for t in res.trace), res.trace)
        needs = smith.needs()
        self.assertEqual([g["recipe"] for g in needs], ["reverse"])
        made = await smith.build(needs[0]["texts"], "a test", "tester")
        self.assertEqual((made["name"], made["version"], made["status"]), ("reverser", "1.0.0", "awaiting approval"))
        folder = Path(made["folder"])
        for f in ("tool.py", "manifest.toml", "TESTS.md"):
            self.assertTrue((folder / f).is_file(), f)
        self.assertIn('author = "neuromesh"', (folder / "manifest.toml").read_text())
        self.assertIn("def reverse_text(text, params):", (folder / "tool.py").read_text())  # the recipe's own code
        # nothing it wrote is installed or runs until a person approves it (P15)
        self.assertFalse((self.root / "plugins" / "reverser").exists())
        self.assertNotIn("reverser", s.tools.tools)
        approval = s.guardian.approvals[made["approval"]]
        self.assertEqual((approval.kind, approval.detail["permissions"]), ("tool.made", []))
        self.assertTrue(any(q.approval_id == made["approval"] for q in s.mind.open_inquiries()))
        self.assertEqual(smith.needs(), [])  # it does not write the same tool twice
        lines = (await execute(s, "/approvals")).lines
        self.assertIn("a tool its toolsmith wrote, reverser v1.0.0", "\n".join(lines))
        out = (await execute(s, f"/approve {made['approval']}")).lines
        self.assertIn("installed reverser in plugins/", "\n".join(out))
        self.assertTrue(await self.active("reverser"))
        self.assertEqual(smith.drafts["reverser"]["status"], "active")
        self.assertEqual(smith.fresh(), ["reverser"])
        rep = await smith.experiment("reverser", "tester")
        self.assertEqual(rep["score"], 1.0, rep["results"])
        self.assertEqual(smith.drafts["reverser"]["status"], "proven")
        self.assertEqual(smith.fresh(), [])
        self.assertNotIn("reverse the word lantern", [u.text for u in s.mind.unmet.values()])
        res = await s.submit("reverse the word lantern")
        self.assertEqual((res.ok, res.route, res.output), (True, "tool:reverser", "nretnal"))
        kinds = [e.topic for e in s.bus.recent("tool.*", 100)]
        for topic in ("tool.drafted", "tool.installed", "tool.active", "tool.experimented"):
            self.assertIn(topic, kinds)
        audit = {k for k in s.store.audit_counts()}
        self.assertTrue({"tool.drafted", "tool.installed", "tool.experimented"} <= audit, audit)
        self.assertIn("reverser v1.0.0 [proven]", "\n".join((await execute(s, "/toolsmith")).lines))
        show = "\n".join((await execute(s, "/toolsmith show reverser")).lines)
        self.assertIn("functions: target, reverse_text, main", show)
        self.assertIn("experiment", show)
        m = mesh(s)
        self.assertEqual((m["toolsmith"]["tools"][0]["name"], m["toolsmith"]["tools"][0]["state"]), ("reverser", "active"))

    async def test_a_person_can_say_no(self):
        s, smith = self.system, self.system.toolsmith
        made = await smith.build(["what is the acronym for portable network graphics"], "a test", "tester")
        self.assertEqual(made["recipe"], "acronym")
        out = (await execute(s, f"/deny {made['approval']}")).lines
        self.assertIn("denied", "\n".join(out))
        self.assertEqual(smith.drafts[made["name"]]["status"], "declined")
        self.assertFalse((self.root / "plugins" / made["name"]).exists())
        s.mind.unmet["ord-a"] = Unmet("ord-a", "abbreviate frequently asked questions", "answer:direct", time.time())
        self.assertEqual(smith.needs(), [])  # it does not offer the recipe again for a while
        self.assertIn("acronym", smith.cooldown)

    async def test_its_own_code_runs_only_as_approved(self):
        s, smith = self.system, self.system.toolsmith
        made = await smith.build(["count the vowels in banana"], "a test", "tester")
        # copying the files into plugins/ by hand does not get around the approval (P15)
        import shutil
        shutil.copytree(made["folder"], self.root / "plugins" / "sneaky")
        manifest = self.root / "plugins" / "sneaky" / "manifest.toml"
        manifest.write_text(manifest.read_text().replace('name = "letter_counter"', 'name = "sneaky"'))
        self.assertTrue(await wait_for(lambda: "sneaky" in s.tools.tools, 10))
        self.assertEqual(s.tools.tools["sneaky"].state, "pending_approval")
        self.assertIn("P15", s.tools.tools["sneaky"].note)
        s.guardian.decide(made["approval"], True, by="tester")
        self.assertTrue(await self.active("letter_counter"))
        # files changed after approval need approving again
        tool = self.root / "plugins" / "letter_counter" / "tool.py"
        tool.write_text(tool.read_text() + "\n# changed by hand\n")
        self.assertTrue(await wait_for(lambda: s.tools.tools["letter_counter"].state == "pending_approval", 10))
        pending = [a for a in s.guardian.pending_approvals() if a.subject == "letter_counter"]
        self.assertEqual(pending[0].detail.get("author"), "neuromesh")
        s.guardian.decide(pending[0].id, True, by="tester")
        self.assertTrue(await self.active("letter_counter"))
        self.assertTrue(smith.approved("letter_counter", s.tools.tools["letter_counter"].manifest.digest))

    async def test_a_tool_can_be_asked_for_by_name(self):
        s, smith = self.system, self.system.toolsmith
        res = await s.submit("make a tool that converts numbers to roman numerals")
        self.assertTrue(res.ok, res.note)
        self.assertEqual(res.route, "toolsmith")
        self.assertIn("/approve apv-", res.output)
        self.assertFalse(any("attempt" in t for t in res.trace))  # the directors do not retry it
        d = smith.drafts["roman_numerals"]
        self.assertEqual((d["recipe"], d["asked"], d["needs"]), ("roman", "converts numbers to roman numerals", []))
        res = await s.submit("make a tool that books flights to Lisbon")
        self.assertFalse(res.ok)
        self.assertIn("couldn't write a tool", res.note)
        lines = (await execute(s, "/toolsmith build sort words alphabetically")).lines
        self.assertIn("wrote word_sorter v1.0.0", "\n".join(lines))
        status, body = await _route(s, "POST", "/toolsmith", {}, {"build": "spell numbers out in words"})
        self.assertEqual(status, 200, body)
        status, body = await _route(s, "GET", "/toolsmith", {}, {})
        self.assertEqual(status, 200)
        self.assertEqual(len([d for d in body["drafts"] if d["status"] == "awaiting approval"]), 3)
        made = await smith.build(["reverse the word hello"], "a test", "tester")
        self.assertIsNone(made)  # three already wait for a person ([toolsmith] max_waiting)
        self.assertIn("already waiting for approval", smith.history[-1]["line"])

    async def test_experiments_refine_what_falls_short(self):
        s, smith = self.system, self.system.toolsmith
        made = await smith.build(["count the vowels in banana"], "a test", "tester")
        s.guardian.decide(made["approval"], True, by="tester")
        self.assertTrue(await self.active("letter_counter"))
        odd = "tally up the vowels found within banana"
        s.mind.unmet["ord-x"] = Unmet("ord-x", odd, "answer:direct", time.time())
        orig = smith._examples
        smith._examples = lambda texts: [(t, "3" if t == odd else None) for t in texts]  # a person told it the answer
        try:
            rep = await smith.experiment("letter_counter", "tester")
        finally:
            smith._examples = orig
        self.assertLess(rep["score"], 1.0)
        self.assertIsNotNone(rep["refined"], rep["line"])
        d = smith.drafts["letter_counter"]
        self.assertEqual((d["version"], d["status"], d["live"]), ("1.1.0", "awaiting approval", "1.0.0"))
        self.assertIn("tally up the vowels found within", d["params"]["triggers"])
        self.assertEqual(d["previous"]["status"], "needs work")
        self.assertEqual(s.tools.tools["letter_counter"].manifest.version, "1.0.0")  # the old one runs until approved
        # declined: the installed version stays, and that version is not refined again
        s.guardian.decide(d["approval"], False, by="tester")
        d = smith.drafts["letter_counter"]
        self.assertEqual((d["version"], d["status"], d.get("refine_declined")), ("1.0.0", "needs work", "1.0.0"))
        smith._examples = lambda texts: [(t, "3" if t == odd else None) for t in texts]
        try:
            rep = await smith.experiment("letter_counter", "tester")
        finally:
            smith._examples = orig
        self.assertIsNone(rep["refined"])

    async def test_initiative_builds_and_experiments(self):
        s, smith, d = self.system, self.system.toolsmith, self.system.director
        for text in ("what is the acronym for portable network graphics", "abbreviate frequently asked questions"):
            await s.submit(text)
        sit = d.situation()
        self.assertEqual(sit["flags"]["needs"], "some")
        self.assertIn("build", sit["feasible"])
        d.set_mode("build", "tester")
        rec = await d.decide(force=True)
        self.assertEqual(rec["mode"], "build", rec)
        self.assertIn("asked permission", rec["summary"])
        made = next(iter(smith.drafts.values()))
        self.assertEqual(made["by"], rec["by"])
        self.assertEqual(d.situation()["flags"]["newtools"], "none")  # nothing runs before a person approves it
        s.guardian.decide(made["approval"], True, by="tester")
        self.assertTrue(await self.active(made["name"]))
        self.assertEqual(d.situation()["flags"]["newtools"], "fresh")
        d.set_mode("experiment", "tester")
        rec = await d.decide(force=True)
        self.assertEqual(rec["mode"], "experiment", rec)
        self.assertGreaterEqual(rec["value"], 0.9)
        self.assertEqual(smith.drafts[made["name"]]["status"], "proven")
        res = await s.submit("abbreviate very important person")
        self.assertEqual((res.route, res.output), (f"tool:{made['name']}", "VIP"))


class CoderNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_coder_node_learns_to_pick_recipes(self):
        tmp = tempfile.TemporaryDirectory()
        s = System(make_root(Path(tmp.name)), overrides=SMITH, seed=12)
        await s.start()
        try:
            teacher, spec = s.teachers.teachers[CODER], s.capabilities[CODER]
            self.assertTrue(spec.internal)
            for label in spec.labels:
                for text in await teacher.generate(spec, label, 12):
                    s.learn(CODER, text, label, "coder")

            def trusted():
                ch = s.registry.champion(CODER)
                return ch is not None and (ch.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy
            self.assertTrue(await wait_for(trusted, timeout=90), s.toolsmith.status()["node"])
            smith = s.toolsmith
            hits = [smith.classify(t) for t in ("reverse the word python", "translate orbit to morse", "rot13 banana")]
            self.assertTrue(any(h[2].startswith("coder node") for h in hits), hits)
            self.assertEqual([h[0] for h in hits], ["reverse", "morse", "caesar"])
            self.assertIn("coder node", "\n".join(smith.lines()))
        finally:
            await s.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
