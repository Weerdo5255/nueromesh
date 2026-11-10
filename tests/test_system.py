"""End-to-end tests: a real System with every agent running, offline teachers, short timers."""
import asyncio
import random
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.runtime import System
from neuromesh.tools import ToolRun

ROOT = Path(__file__).resolve().parent.parent
INCOMING = ROOT / "examples" / "incoming_plugins"

FAST = {
    "nodes": {"min_examples_to_spawn": 8, "challenger_cooldown_seconds": 1.0, "retired_grace_seconds": 1.0,
              "candidate_max_age_seconds": 10.0, "max_candidate_slices": 6, "min_eval_samples": 6,
              "promotion_z": 1.0},
    "background": {"tick_seconds": 0.05, "max_duty_cycle": 0.95, "synthesis_cooldown_seconds": 0.3},
    "tools": {"scan_seconds": 0.2},
    "teachers": {"order": ["human", "heuristic"]},
    "temperament": {"enabled": False},  # answers unshaped, so tests can compare outputs (see test_temperament)
    "memory": {"enabled": False},       # answers come from where each test expects (see test_memory)
    "agency": {"enabled": False},       # no initiative of its own unless a test asks (see test_agency)
    "dreams": {"enabled": False},       # it does not fall asleep unless a test asks (see test_dreams)
    "directors": {"consult": False},    # no second opinions unless a test asks (see test_directors)
    "objectives": {"always": False},    # it sets itself no objectives unless a test asks (see test_objectives)
    "imagesets": {"start_with": []},    # it gets no picture books by itself (see test_imagesets)
}


def work_trace(res) -> list[str]:
    """What the tools, nodes and engines did: the trace without the directors' intake and review lines, and
    without the ego's lines around them (ego.py)."""
    return [t for t in res.trace if not t.startswith(("director-", "reviewed by director", "ego (the outer layer)", "ego: "))]


def make_root(tmp: Path, wikipedia: bool = False, kiwix: bool = False, gitea: bool = False, searxng: bool = False) -> Path:
    for name in ("directives.toml", "capabilities.toml"):
        shutil.copy(ROOT / name, tmp / name)
    skip = tuple(n for n, keep in (("wikipedia", wikipedia), ("kiwix", kiwix), ("gitea", gitea), ("searxng", searxng))
                 if not keep)  # network tools wait for approval
    shutil.copytree(ROOT / "plugins", tmp / "plugins", ignore=shutil.ignore_patterns(*skip))
    return tmp


async def wait_for(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


ORDERS = ["what is {a} * {b}", "calculate {a} plus {b}", "count the words in: the {w} sat on the mat",
          "sentiment: the {w} was wonderful", "sentiment: the {w} was terrible", "urgency: the {w} is down asap",
          "urgency: no rush, look at the {w} next week", "how many words are in \"a {w} and a cat\""]


def orders(n, seed=0):
    rng = random.Random(seed)
    words = ["hotel", "movie", "server", "dog", "garden", "laptop", "pizza", "train"]
    return [rng.choice(ORDERS).format(a=rng.randint(2, 99), b=rng.randint(2, 99), w=rng.choice(words))
            for _ in range(n)]


class SystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=FAST, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def feed(self, n, seed=0):
        results = []
        for text in orders(n, seed):
            results.append(await self.system.submit(text))
            await asyncio.sleep(0.01)
        return results

    async def test_boot_activates_installed_tools_before_orders(self):
        self.assertEqual(self.system.tools.tools["calculator"].state, "active")
        res = await self.system.submit("tool calculator: 6 * 7")
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 42)

    async def test_nodes_are_built_and_take_over(self):
        await self.feed(60)
        self.assertTrue(await wait_for(lambda: all(self.system.registry.champion(c)
                                                   for c in ("dispatch", "sentiment", "urgency"))))
        await asyncio.sleep(1.5)
        later = await self.feed(40, seed=1)
        by_nodes = sum(r.served_by == "nodes" for r in later)
        self.assertGreater(by_nodes, 20, [r.served_by for r in later])
        self.assertTrue(all(r.ok for r in later if r.route and r.route.startswith("tool:")))
        kinds = {a["kind"] for a in self.system.store.recent_audit(500)}
        self.assertTrue({"node.spawned", "node.promoted"} <= kinds)

    async def test_new_tool_is_learned_and_router_superseded(self):
        await self.feed(50)
        self.assertTrue(await wait_for(lambda: self.system.registry.champion("dispatch") is not None))
        old_router = self.system.registry.champion("dispatch").id
        shutil.copytree(INCOMING / "unit_convert", self.root / "plugins" / "unit_convert")
        self.assertTrue(await wait_for(lambda: self.system.tools.tools.get("unit_convert") is not None
                                       and self.system.tools.tools["unit_convert"].state == "active"))
        self.assertTrue(await wait_for(
            lambda: "tool:unit_convert" in self.system.registry.champion("dispatch").labels, timeout=30))
        self.assertNotEqual(self.system.registry.champion("dispatch").id, old_router)
        res = await self.system.submit("convert 10 km to miles")
        self.assertEqual(res.route, "tool:unit_convert")
        self.assertIn("6.2137", str(res.output))
        # the superseded router is disposed after its rollback window
        self.assertTrue(await wait_for(lambda: self.system.registry.lineage[old_router]["status"] == "disposed"))
        self.assertFalse((self.system.state_dir / "nodes" / f"{old_router}.npz").exists())

    async def test_approval_flow_and_forbidden_tool(self):
        for name in ("glossary", "shell_exec"):
            shutil.copytree(INCOMING / name, self.root / "plugins" / name)
        tools = self.system.tools.tools
        self.assertTrue(await wait_for(lambda: "glossary" in tools and "shell_exec" in tools))
        self.assertEqual(tools["shell_exec"].state, "rejected")
        self.assertEqual(tools["glossary"].state, "pending_approval")
        res = await self.system.submit("tool shell_exec: run ls")
        self.assertFalse(res.ok)
        self.assertEqual(tools["shell_exec"].runs, 0)
        approval = self.system.guardian.pending_approvals()[0]
        self.system.guardian.decide(approval.id, True, by="test")
        self.assertTrue(await wait_for(lambda: tools["glossary"].state == "active"))
        res = await self.system.submit("tool glossary: define lucid")
        self.assertTrue(res.ok)
        self.assertIn("clearly", res.output)

    async def test_changed_tool_code_is_revetted(self):
        tool_py = self.root / "plugins" / "word_stats" / "tool.py"
        tool_py.write_text(tool_py.read_text() + "\n# edited\n")
        self.assertTrue(await wait_for(lambda: any(a["kind"] == "tool.changed"
                                                   for a in self.system.store.recent_audit(200))))
        self.assertTrue(await wait_for(lambda: self.system.tools.tools["word_stats"].state == "active"))
        self.assertNotIn("tool:word_stats", self.system.dispatch.inactive_labels)

    async def test_failing_tool_is_quarantined(self):
        tool = self.system.tools.tools["word_stats"]
        for _ in range(self.system.directives.tools.quarantine_after_failures):
            self.system.record_tool_run(tool, ToolRun(False, error="boom"))
        self.assertEqual(tool.state, "quarantined")
        self.assertIn("tool:word_stats", self.system.dispatch.inactive_labels)
        res = await self.system.submit("tool word_stats: count the words in: a b c")
        self.assertFalse(res.ok)

    async def test_feedback_and_label_guard(self):
        res = await self.system.submit("sentiment: the soup was cold and the waiter ignored us")
        ok, msg = self.system.feedback(res.order_id, "negative")
        self.assertTrue(ok, msg)
        ok, msg = self.system.feedback(res.order_id, "routine")  # not a sentiment label
        self.assertFalse(ok)
        self.assertNotIn("routine", self.system.capabilities["sentiment"].labels)
        examples = self.system.store.examples("sentiment")
        self.assertTrue(any(e.source == "human" and e.label == "negative" for e in examples))

    async def test_kill_switch_halts_everything(self):
        (self.system.state_dir / "STOP").touch()
        self.assertTrue(await wait_for(lambda: self.system.guardian.halted, timeout=3))
        res = await self.system.submit("what is 2 + 2")
        self.assertFalse(res.ok)
        self.assertIn("halted", res.note)
        self.assertFalse(self.system.guardian.background_allowed())

    async def test_state_survives_restart(self):
        await self.feed(50)
        self.assertTrue(await wait_for(lambda: self.system.registry.champion("dispatch") is not None))
        champ = self.system.registry.champion("dispatch").id
        await self.system.stop()
        self.system = System(self.root, overrides=FAST, seed=2)
        self.assertEqual(self.system.registry.champion("dispatch").id, champ)
        await self.system.start()
        self.assertEqual(self.system.tools.tools["calculator"].state, "active")
        res = await self.system.submit("what is 9 * 9")
        self.assertEqual(res.output, 81)


if __name__ == "__main__":
    unittest.main()
