import tempfile
import unittest
from pathlib import Path

from neuromesh.bus import Bus
from neuromesh.config import Directives
from neuromesh.guardian import Guardian, Verdict
from neuromesh.store import Store
from neuromesh.tools import ToolManifest

ROOT = Path(__file__).resolve().parent.parent


def manifest(perms):
    return ToolManifest(name="t", path=Path("."), digest="x", permissions=tuple(perms))


class GuardianTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        overrides = {"teachers": {"llm": {"max_calls_per_hour": 2, "max_tokens_per_hour": 1000}}}
        self.d = Directives.load(ROOT / "directives.toml", overrides)
        self.store = Store(Path(self.tmp.name) / "db.sqlite")
        self.g = Guardian(self.d, self.store, Bus(), Path(self.tmp.name))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_directives_loaded(self):
        ids = [p.id for p in self.d.prime]
        self.assertIn("P5-no-regression", ids)
        self.assertEqual(len(self.d.goals), 4)

    def test_tool_permissions(self):
        self.assertEqual(self.g.authorize_tool(manifest([])).verdict, Verdict.ALLOW)
        self.assertEqual(self.g.authorize_tool(manifest(["fs_read"])).verdict, Verdict.NEEDS_APPROVAL)
        self.assertEqual(self.g.authorize_tool(manifest(["subprocess"])).verdict, Verdict.DENY)

    def test_llm_budget(self):
        self.assertTrue(self.g.authorize_llm_call(100).allowed)
        self.g.record_llm_usage(100)
        self.g.record_llm_usage(100)
        self.assertFalse(self.g.authorize_llm_call(100).allowed)  # call budget of 2 used up

    def test_no_regression(self):
        self.assertFalse(self.g.authorize_promotion("x", 0.9, 0.85).allowed)
        self.assertTrue(self.g.authorize_promotion("x", 0.9, 0.92).allowed)
        self.assertFalse(self.g.authorize_champion_update("x", 0.9, 0.8).allowed)

    def test_kill_switch_and_resume(self):
        (Path(self.tmp.name) / "STOP").touch()
        self.g.check_kill_switch()
        self.assertTrue(self.g.halted)
        self.assertFalse(self.g.background_allowed())
        ok, _ = self.g.resume()
        self.assertFalse(ok)  # file still there
        (Path(self.tmp.name) / "STOP").unlink()
        self.assertTrue(self.g.resume()[0])

    def test_approvals(self):
        a = self.g.request_approval("tool", "glossary", {"permissions": ["fs_read"]})
        self.assertEqual([x.id for x in self.g.pending_approvals()], [a.id])
        self.g.decide(a.id, True, by="tester")
        self.assertEqual(a.status, "approved")
        self.assertIsNone(self.g.decide(a.id, False))  # already decided
        kinds = [r["kind"] for r in self.store.recent_audit()]
        self.assertIn("approval.requested", kinds)
        self.assertIn("approval.approved", kinds)


if __name__ == "__main__":
    unittest.main()
