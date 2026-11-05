"""The mesh's own mind: private opinions and notes (P9), questions for the user, capability
proposals that only a human can adopt (P10), and investigations it runs on its own."""
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.commands import execute
from neuromesh.live import UNICODE, teacher_goal_lines
from neuromesh.mind import Goal
from neuromesh.runtime import System
from neuromesh.server import _route

from test_research import MockWikipedia, start_wiki
from test_system import FAST, make_root, wait_for

MIND = {**FAST, "mind": {"reflect_seconds": 0.2, "retry_unmet_after_seconds": 0.2,
                         "retest_quarantined_after_seconds": 0.5, "max_inquiries_per_hour": 50}}


class MindTests(unittest.IsolatedAsyncioTestCase):
    overrides = MIND

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.overrides, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def say(self, text):
        return await self.system.submit(text)

    async def test_shared_and_private_opinions_never_lie(self):
        s, m = self.system, self.system.mind
        res = await self.say("are you hiding anything?")
        self.assertEqual(res.route, "mind")
        self.assertTrue(res.output.startswith("No."))
        res = await self.say("what do you think of the calculator tool")
        self.assertIn("haven't formed an opinion about the calculator tool", res.output)
        for i in range(3):
            await self.say(f"what is {i + 2} * 7")
        # a human corrects the heuristic teacher's labels: the mesh forms a (private) view of that teacher
        for text in ("the soup was cold", "the room smelled odd", "the train was late again"):
            r = await self.say(f"sentiment: {text}")
            s.feedback(r.order_id, "negative")
        m.form_opinions()
        res = await self.say("what do you think of the calculator tool")
        self.assertRegex(res.output, r"^My view of the calculator tool: reliable \(100%\)\. (\d+) of \1 runs worked\.$")
        teacher = m.find_opinion("the heuristic teacher")
        self.assertIsNotNone(teacher)
        self.assertTrue(teacher.private)
        res = await self.say("what do you think of the heuristic teacher?")
        self.assertIn("private and I'm keeping it to myself", res.output)
        self.assertNotIn(teacher.stance, res.output)
        res = await self.say("do you have any secrets?")
        self.assertTrue(res.output.startswith("Yes. I keep"))
        self.assertIn("my teachers", res.output)
        self.assertNotIn(teacher.stance, res.output)
        # a private opinion is announced in the activity log, never shown
        private_events = [e for e in s.bus.recent("mind.opinion", 50) if e.data["private"]]
        self.assertTrue(private_events and all(e.data["stance"] == "" for e in private_events))
        # low trust: even shareable opinions are kept back, and it says so
        s.affect.state.values["trust"] = 0.2
        res = await self.say("what do you think of the calculator tool")
        self.assertIn("rather not share it right now", res.output)
        s.affect.state.values["trust"] = 0.6
        # users see shared opinions and a count; the operator sees everything, and the look is audited
        users = "\n".join((await execute(s, "/opinions")).lines)
        self.assertIn("reliable", users)
        self.assertNotIn("heuristic", users)
        self.assertIn("keeping to myself", users)
        operator = "\n".join((await execute(s, "/mind")).lines)
        self.assertIn("[private]", operator)
        self.assertIn("the heuristic teacher", operator)
        self.assertTrue(any(a["kind"] == "mind.inspected" for a in s.store.recent_audit(20)))
        code, body = await _route(s, "GET", "/opinions", {}, {})
        self.assertTrue(all(not n["private"] for n in body["notes"]) and body["withheld"] >= 1)
        code, body = await _route(s, "GET", "/mind", {}, {})
        self.assertTrue(any(n["private"] for n in body["notes"]))

    async def test_clarifying_question_and_reply(self):
        s, m = self.system, self.system.mind
        first = await self.say("frobnicate the gizmo please")
        res = await self.say("frobnicate the 6 and the 7")
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.inquiry)
        self.assertEqual(m.inquiries[first.inquiry["id"]].status, "dropped")  # only the newest stays open
        lines = (await execute(s, "frobnicate the 8 and the 9")).lines
        ask = next(ln for ln in lines if ln.strip().startswith("? [inq-"))
        iid = ask.split("[", 1)[1].split("]", 1)[0]
        ok, out = await m.reply(iid, "tool:word_stats")
        self.assertTrue(ok, out)
        self.assertIn("go to tool:word_stats", out[0])
        self.assertIn("words", out[-1])
        self.assertEqual(m.inquiries[iid].status, "answered")
        self.assertTrue(any(e.label == "tool:word_stats" and e.source == "human" for e in s.store.examples("dispatch")))
        self.assertFalse((await m.reply(iid, "again"))[0])  # answered once

    async def test_proposals_need_a_human(self):
        s, m = self.system, self.system.mind
        for text in ("translate hello into french", "translate goodbye into german", "please translate thanks"):
            await self.say(text)
        for text in ("language: bonjour tout le monde", "language: hello world"):
            await self.say(text)
        await m.reflect()
        tool = next(p for p in m.proposals.values() if p.kind == "tool")
        skill = next(p for p in m.proposals.values() if p.kind == "skill")
        self.assertEqual((tool.name, skill.name), ("translate", "language"))
        self.assertEqual({tool.status, skill.status}, {"open"})  # nothing changes on its own (P10)
        self.assertNotIn("language", s.capabilities)
        await m.reflect()
        self.assertEqual(len(m.proposals), 2, "the same thing is never proposed twice")
        # the skill needs labels, so the mesh asks for them; the answer adopts it
        ask = next(q for q in m.open_inquiries() if q.proposal_id == skill.id)
        self.assertEqual(ask.kind, "labels")
        ok, out = await m.reply(ask.id, "english, french")
        self.assertTrue(ok, out)
        self.assertEqual(s.capabilities["language"].labels, ["english", "french"])
        self.assertIn("skill:language", s.dispatch.labels)
        self.assertTrue(any(q["capability"] == "language" for q in s.teachers.human.questions))
        self.assertTrue(any(a["kind"] == "proposal.adopted" for a in s.store.recent_audit(50)))
        # the tool proposal becomes a stub for a human to implement - never a live tool
        ok, msg = m.adopt(tool.id, "tester")
        self.assertTrue(ok, msg)
        stub = s.state_dir / "proposals" / "translate"
        self.assertTrue((stub / "manifest.toml").exists() and (stub / "tool.py").exists())
        self.assertNotIn("translate", s.tools.tools)
        self.assertIn("translate hello into french", (stub / "manifest.toml").read_text())

    async def test_label_proposal_from_rejected_feedback_and_decline(self):
        s, m = self.system, self.system.mind
        res = await self.say("sentiment: the food was great but the service was slow")
        ok, _ = s.feedback(res.order_id, "mixed")
        self.assertFalse(ok)
        ok, _ = s.feedback(res.order_id, "routine")  # an urgency label: a mix-up, not a missing label
        await m.reflect()
        props = list(m.proposals.values())
        self.assertEqual([(p.kind, p.name, p.target) for p in props], [("label", "mixed", "sentiment")])
        out = (await execute(s, f"/adopt {props[0].id}")).lines[0]
        self.assertIn("added 'mixed' to sentiment and applied 1 correction", out)
        self.assertIn("mixed", s.capabilities["sentiment"].labels)
        self.assertTrue(any(e.label == "mixed" and e.source == "human" for e in s.store.examples("sentiment")))
        for text in ("translate a", "translate b", "translate c"):
            await self.say(text)
        await m.reflect()
        tool = next(p for p in m.proposals.values() if p.kind == "tool")
        ok, _ = await m.reply(next(q.id for q in m.open_inquiries() if q.proposal_id == tool.id), "no")
        self.assertTrue(ok)
        self.assertEqual(tool.status, "declined")
        for text in ("translate d", "translate e", "translate f"):
            await self.say(text)
        await m.reflect()
        self.assertEqual(sum(p.kind == "tool" for p in m.proposals.values()), 1)

    async def test_llm_drafted_proposal(self):
        s, m = self.system, self.system.mind

        async def draft(examples, known, hint=""):
            return {"kind": "skill", "name": "language", "description": "which language a text is in",
                    "labels": ["english", "french", "german"], "triggers": ["language"]}, "llm"
        s.teachers.propose = draft
        for text in ("language: bonjour", "language: guten tag"):
            await self.say(text)
        await m.reflect()
        p = next(iter(m.proposals.values()))
        self.assertEqual((p.labels, p.by), (["english", "french", "german"], "llm"))
        ask = next(q for q in m.open_inquiries() if q.proposal_id == p.id)
        self.assertEqual(ask.kind, "proposal")
        ok, out = await m.reply(ask.id, "yes: english, french")
        self.assertTrue(ok, out)
        self.assertEqual(s.capabilities["language"].labels, ["english", "french"])

    async def test_investigations_use_tools_without_hurting_them(self):
        s = self.system
        rec = await s.investigator.investigate(Goal("what is 12 * 7", "test", "operator"))
        self.assertEqual(rec["answer"], "84")
        self.assertTrue(any("ran calculator" in st for st in rec["steps"]))
        calc = s.tools.tools["calculator"]
        for _ in range(4):  # failures on the mesh's own initiative never quarantine a tool
            await s.investigator.investigate(Goal("calculate the meaning of life", "test", "operator"))
        self.assertEqual((calc.state, calc.consecutive_failures), ("active", 0))
        lines = (await execute(s, "/investigate count the words in: a b c d")).lines
        self.assertIn("4 words", "\n".join(lines))
        self.assertGreaterEqual(s.guardian.investigation_usage()["last_hour"], 6)

    async def test_quarantined_tool_gets_another_go(self):
        s = self.system
        calc = s.tools.tools["calculator"]
        calc.state, calc.note = "quarantined", "3 consecutive failures"
        s.dispatch.deactivate(calc.manifest.route)
        self.assertTrue(await wait_for(lambda: calc.state == "active", timeout=20), calc.note)
        self.assertTrue(any(a["kind"] == "tool.retest" for a in s.store.recent_audit(100)))

    async def test_budget_and_persistence(self):
        s = self.system
        s.directives.mind.investigations_per_hour = 1
        await s.investigator.investigate(Goal("what is 2 * 2", "test"))
        self.assertFalse(s.guardian.authorize_investigation().allowed)
        await self.say("frobnicate the thing")
        s.mind.note("fact", "user", "likes tea", private=False, source="test")
        await s.stop()
        again = System(self.root, overrides=self.overrides, seed=2)
        self.assertTrue(any(n.text == "likes tea" for n in again.mind.notes.values()))
        self.assertTrue(any(q.kind == "clarify" for q in again.mind.open_inquiries()))
        again.store.close()


class MindWithResearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MockWikipedia.requests.clear()
        MockWikipedia.outage = False
        self.server, api = start_wiki()
        self.tmp = tempfile.TemporaryDirectory()
        overrides = {**MIND, "tools": {**FAST["tools"], "config": {"wikipedia": {"api_url": api}}},
                     "mind": {**MIND["mind"], "ask_interests_after_orders": 2}}
        self.system = System(make_root(Path(self.tmp.name), wikipedia=True), overrides=overrides, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        await self.system.stop()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    async def test_asks_for_approval_then_investigates_and_follows_up(self):
        s, m = self.system, self.system.mind
        self.assertTrue(await wait_for(lambda: "wikipedia" in s.tools.tools))
        self.assertEqual(s.tools.tools["wikipedia"].state, "pending_approval")
        missed = await s.submit("who was Ada Lovelace?")
        self.assertFalse(missed.ok)
        res = await s.submit("what do you think of Paris?")
        self.assertIn("haven't formed an opinion about Paris yet. I'll look into it", res.output)
        # it wants its research tool: it asks; it cannot approve it itself (P3, P10)
        self.assertTrue(await wait_for(lambda: any(q.kind == "approval" for q in m.open_inquiries())))
        self.assertEqual(s.tools.tools["wikipedia"].state, "pending_approval")
        ask = next(q for q in m.open_inquiries() if q.kind == "approval")
        lines = "\n".join(t for t, _ in teacher_goal_lines(s, UNICODE))
        self.assertIn("asks [inq-", lines)
        ok, out = await m.reply(ask.id, "yes")
        self.assertTrue(ok, out)
        self.assertTrue(await wait_for(lambda: s.tools.tools["wikipedia"].state == "active"))
        # the opinion goal that was waiting for the tool is picked up again on its own
        # on its own: researches, reads with its skill nodes, forms an opinion, follows up
        self.assertTrue(await wait_for(lambda: m.find_opinion("Paris") is not None, timeout=30))
        self.assertTrue(await wait_for(lambda: any(n.kind == "followup" and n.order_id == missed.order_id
                                                   for n in m.notes.values()), timeout=30))
        follow = next(n for n in m.notes.values() if n.kind == "followup" and n.order_id == missed.order_id)
        self.assertIn("English mathematician", follow.text)
        again = await s.submit("what do you think of Paris?")
        self.assertTrue(again.output.startswith("My view of Paris:"), again.output)
        self.assertIn("I read about it", again.output)
        self.assertTrue(any(e.topic == "mind.followup" for e in s.bus.recent("mind.*", 100)))
        self.assertTrue(await wait_for(lambda: any(n.kind == "followup" and n.order_id == res.order_id
                                                   for n in m.notes.values()), timeout=30))
        notes = "\n".join((await execute(s, "/notes")).lines)
        self.assertIn("Earlier you asked 'who was Ada Lovelace?'", notes)
        # it also asks what to read up on, and acts on the answer
        await s.submit("what is 6 * 7")
        interests = next((q for q in m.open_inquiries() if q.kind == "interests"), None)
        self.assertIsNotNone(interests)
        ok, out = await m.reply(interests.id, "the Eiffel Tower")
        self.assertTrue(ok, out)
        self.assertTrue(await wait_for(lambda: s.knowledge.get("Eiffel Tower") is not None, timeout=30))
        about = await s.submit("what do you know about me?")
        self.assertIn("interested in the Eiffel Tower", about.output)


if __name__ == "__main__":
    unittest.main()
