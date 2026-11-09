"""The tool switch (toolgate.py): tools on, off, or the NeuroMesh decides each time, for all tools or one."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.toolgate import GATE_LABELS, ToolGateTeacher, gate_policy, gate_text, parse_gate

from test_system import FAST, make_root, wait_for


class PolicyTests(unittest.TestCase):
    def test_when_it_uses_a_tool(self):
        f = {"tasked": "routed", "trecord": "good", "tmood": "steady", "tother": "teacher", "tspeed": "fast", "tliked": "neutral"}
        self.assertEqual(gate_policy(f)[0], "use")
        self.assertEqual(gate_policy({**f, "trecord": "poor"})[0], "skip")                      # rarely works; a teacher can
        self.assertEqual(gate_policy({**f, "trecord": "poor", "tother": "none"})[0], "use")      # nothing else could
        self.assertEqual(gate_policy({**f, "trecord": "poor", "tasked": "explicit"})[0], "use")  # you named it
        self.assertEqual(gate_policy({**f, "trecord": "new", "tmood": "wary"})[0], "skip")
        self.assertEqual(gate_policy({**f, "trecord": "mixed", "tspeed": "slow", "tother": "memory"})[0], "skip")
        self.assertEqual(gate_policy({**f, "tliked": "disliked"})[0], "skip")                   # people punished using it
        self.assertEqual(gate_policy({**f, "trecord": "poor", "tliked": "liked"})[0], "use")     # people rewarded it
        self.assertEqual(parse_gate(gate_text(f)), f)
        spec = type("Spec", (), {"name": "toolgate"})()
        t = ToolGateTeacher(seed=1)
        for label in GATE_LABELS:
            made = asyncio.run(t.generate(spec, label, 5))
            self.assertTrue(made and all(gate_policy(parse_gate(x))[0] == label for x in made), label)
        self.assertEqual(asyncio.run(t.label(spec, gate_text({**f, "trecord": "poor"}))).label, "skip")


class SwitchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides={**FAST, "tools": {**FAST["tools"], "mode": "on"}}, seed=3)
        await self.system.start()
        self.assertTrue(await wait_for(lambda: self.system.tools.tools.get("calculator") is not None
                                       and self.system.tools.tools["calculator"].state == "active", 20))

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_off_on_and_one_tool_set_apart(self):
        s = self.system
        calc = s.tools.tools["calculator"]
        res = await s.submit("tool calculator: 3 + 4")
        self.assertEqual(str(res.output), "7")
        ok, msg = s.tool_switch.set_mode("off", "tester")
        self.assertTrue(ok and "off" in msg)
        runs = calc.runs
        res = await s.submit("tool calculator: 3 + 4")  # you named it, but tools are off: it says so
        self.assertFalse(res.ok)
        self.assertIn("switched off", res.note)
        res = await s.submit("what is 12 * 7")  # routed to it: it answers another way, or says it can't
        self.assertIn("the tool switch is off for calculator", " ".join(res.trace))
        self.assertNotEqual(str(res.output), "84")
        self.assertEqual(calc.runs, runs)                     # nothing ran, and nothing counts against it
        self.assertEqual(calc.state, "active")
        run = await s.sandbox.run(calc, "1 + 1", 5)           # not for anything: the sandbox refuses it too
        self.assertTrue(run.refused and not run.ok)
        s.record_tool_run(calc, run)
        self.assertEqual((calc.runs, calc.consecutive_failures), (runs, 0))
        self.assertGreaterEqual(s.tool_switch.status()["tools"]["calculator"]["refused"], 2)
        ok, msg = s.tool_switch.set_tool("calculator", "on", "tester")  # one tool set apart from the switch
        self.assertTrue(ok)
        res = await s.submit("tool calculator: 6 * 7")
        self.assertEqual(str(res.output), "42")
        self.assertEqual(s.status()["tools"]["calculator"]["switch"], "on")
        self.assertTrue(s.status()["tools"]["calculator"]["set_apart"])
        ok, _ = s.tool_switch.set_tool("calculator", "default", "tester")
        self.assertTrue(ok and s.tool_switch.mode_of("calculator") == "off")
        self.assertFalse(s.tool_switch.set_mode("sometimes", "tester")[0])
        self.assertFalse(s.tool_switch.set_tool("nope", "on", "tester")[0])
        status, body = await _route(s, "POST", "/tools/switch", {}, {"mode": "decide", "by": "tester"})
        self.assertEqual((status, body["mode"]), (200, "decide"))
        status, body = await _route(s, "GET", "/tools/switch", {}, {})
        self.assertEqual(body["mode"], "decide")
        self.assertTrue(any(e.topic == "tools.switch" for e in s.bus.recent("tools.*", 20)))

    async def test_it_decides(self):
        s = self.system
        calc = s.tools.tools["calculator"]
        s.tool_switch.set_mode("decide", "tester")
        res = await s.submit("tool calculator: 2 + 2")  # named: always used
        self.assertEqual(str(res.output), "4")
        self.assertIn("you asked for that tool", " ".join(res.trace))
        calc.runs, calc.successes = 10, 2                # it rarely works, and (say) a teacher could answer
        s.tool_switch.other_way = lambda text: "teacher"
        res = await s.submit("what is 5 * 6")
        trace = " ".join(res.trace)
        if "dispatch -> tool:calculator" in trace:       # (the router sent it to the calculator)
            self.assertIn("decided not to use calculator", trace)
            self.assertIn("rarely works", trace)
            # no teacher could answer after all: it uses the tool
            self.assertIn("so it uses calculator after all", trace)
            self.assertEqual(str(res.output), "30")
        use, why, by = s.tool_switch.decide(calc, "what is 5 * 6")
        self.assertFalse(use)
        self.assertIn("rarely works", why)
        calc.runs, calc.successes = 10, 10
        use, why, _ = s.tool_switch.decide(calc, "what is 5 * 6")
        self.assertTrue(use)
        taught = [e.label for e in s.store.examples("toolgate")]
        self.assertIn("skip", taught)                    # the policy teaches the tool-gate node
        self.assertIn("use", taught)
        self.assertTrue(s.tool_switch.recent)
        lines = "\n".join(s.tool_switch.lines())
        self.assertIn("it decides each time", lines)


if __name__ == "__main__":
    unittest.main()
