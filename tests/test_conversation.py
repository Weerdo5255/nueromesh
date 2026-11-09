"""The conversation it is having: follow-ups read against the last turn, what you tell it kept verbatim in short-term
memory nodes (fact or fiction, secrets never kept), questions answered from them, and - once the conversation has been
quiet a while - its ego keeping what matters as long-term memory and letting the rest go."""
import tempfile
import unittest
from pathlib import Path

from neuromesh.commands import execute
from neuromesh.conversation import keep_policy, parse_statement, statement_like, you_said
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for


class ReadingTests(unittest.TestCase):
    def test_statements(self):
        self.assertEqual(parse_statement("my dog is called Rex"), ("my dog", "Rex", "what is my dog called"))
        self.assertEqual(parse_statement("My name is Chris."), ("my name", "Chris", "what is my name"))
        self.assertEqual(parse_statement("I live in Leeds"), ("where I live", "Leeds", "where do I live"))
        self.assertEqual(parse_statement("in my story the dragon Vex lives on the moon"),
                         ("dragon Vex", "on the moon", "where does the dragon Vex live"))
        self.assertIsNone(parse_statement("what is my dog called?"))
        self.assertTrue(statement_like("the meeting is on Friday"))
        self.assertFalse(statement_like("please count the words in: the meeting is on Friday"))
        self.assertFalse(statement_like("sentiment: the hotel was lovely"))
        self.assertEqual(you_said("my dog is called Rex and I live in Leeds"), "your dog is called Rex and you live in Leeds")

    def test_keep_policy(self):
        base = {"kkind": "personal", "ksecret": "no", "kstate": "current", "klasting": "lasting", "kasked": "no",
                "kknown": "no", "kliked": "neutral"}
        self.assertEqual(keep_policy(base)[0], "keep")
        self.assertEqual(keep_policy({**base, "ksecret": "yes", "kasked": "yes"})[0], "let_go", "a secret is never kept")
        self.assertEqual(keep_policy({**base, "kstate": "corrected"})[0], "let_go")
        self.assertEqual(keep_policy({**base, "klasting": "passing"})[0], "let_go")
        self.assertEqual(keep_policy({**base, "kkind": "fiction"})[0], "keep")
        self.assertEqual(keep_policy({**base, "kkind": "chatter"})[0], "let_go")


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = System(make_root(Path(self.tmp.name)), overrides={
            **FAST, "memory": {"enabled": True, "interval_seconds": 0.05},
            "conversation": {"idle_hours": 1.0, "check_every_seconds": 0.2}}, seed=1)
        await self.s.start()
        s = self.s
        for ask, answer in (("what is the capital of France", "Paris"), ("what is the capital of Spain", "Madrid"),
                            ("how tall is the Eiffel Tower", "330 metres")):
            s.memory.learn(ask, answer, "a test", by="tester", reviewed=True)
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.fact_nodes()), 60))

    async def asyncTearDown(self):
        if not self.s.stopping:
            await self.s.stop()
        self.tmp.cleanup()

    async def say(self, text, conv="c1"):
        return await self.s.submit(text, conversation=conv)

    async def test_follow_ups(self):
        r = await self.say("what is the capital of France?")
        self.assertTrue(r.answer.startswith("Paris"), r.answer)
        r = await self.say("what about Spain?")
        self.assertTrue(r.answer.startswith("Madrid"), r.answer)
        self.assertEqual(r.conversation["read_as"], "what is the capital of Spain?")
        self.assertIn("France -> Spain", r.trace[0])
        await self.say("how tall is the Eiffel Tower?")
        r = await self.say("how tall is it?")
        self.assertEqual(r.conversation["read_as"], "how tall is the Eiffel Tower?")
        self.assertTrue(r.answer.startswith("330 metres"), r.answer)
        r = await self.say("what about Spain?", conv="someone-else")  # another conversation: nothing to follow on from
        self.assertNotIn("read_as", r.conversation or {})

    async def test_told_recalled_kept_and_let_go(self):
        s = self.s
        r = await self.say("my dog is called Rex")
        self.assertEqual(r.route, "conversation")
        self.assertTrue(r.answer.startswith("Noted: your dog is called Rex."), r.answer)
        await self.say("remember that the meeting is on Friday")
        r = await self.say("in my story the dragon Vex lives on the moon")
        self.assertEqual((r.realm["realm"], r.realm["work"]), ("fiction", "your story"))
        r = await self.say("my wifi password is hunter2")
        self.assertIn("never written into my long-term memory", r.answer)
        r = await self.say("I'm tired today")
        self.assertEqual(r.route, "conversation")

        # questions: answered from short-term memory, verbatim
        r = await self.say("what is my dog called?")
        self.assertTrue(r.answer.startswith("Rex - you told me at "), r.answer)
        self.assertEqual(r.conversation["item"], "stm-1")
        r = await self.say("where does the dragon Vex live?")
        self.assertIn("On the moon - you told me", r.answer)
        self.assertIn("fiction, not fact", r.answer)
        r = await self.say("what's my wifi password?")
        self.assertNotIn("hunter2", r.answer, "a secret is not repeated")
        r = await self.say("what is my dog called?", conv="c2")
        self.assertTrue(r.answer.startswith("Rex"), "what you told it is there in another conversation too")
        # a correction replaces what came before
        r = await self.say("my dog is called Max")
        self.assertIn('replaces what you told me before: "your dog is called Rex"', r.answer)
        self.assertEqual(s.conversations.short["stm-1"]["status"], "superseded")
        r = await self.say("what is my dog called?")
        self.assertTrue(r.answer.startswith("Max"), r.answer)

        # nothing short-term is compressed: the exact words are there, and secrets are hidden from the API
        code, st = await _route(s, "GET", "/conversations", {}, {})
        texts = {i["id"]: i["text"] for i in st["short"]}
        self.assertEqual(texts["stm-3"], "in my story the dragon Vex lives on the moon")
        self.assertEqual(texts["stm-4"], "(a secret: hidden)")
        self.assertTrue(all(i["expires_at"] > 0 for i in st["short"] if i["status"] == "short"))

        # quiet for idle_hours: its ego decides
        for c in s.conversations.convs.values():
            c.last -= 2 * 3600
        self.assertTrue(await wait_for(lambda: all(c.status == "ended" for c in s.conversations.convs.values()), 15))
        st = {i["id"]: i for i in s.conversations.short.values()}
        self.assertEqual(st["stm-4"]["status"], "let go")
        self.assertEqual(st["stm-4"]["text"], "(a secret: forgotten)", "a secret is forgotten, words and all")
        self.assertEqual(st["stm-5"]["status"], "let go", "I'm tired today: about that moment")
        kept = [i for i in st.values() if i["status"] == "long"]
        self.assertEqual({i["id"] for i in kept}, {"stm-2", "stm-3", "stm-6"})
        self.assertTrue(all(i["node"] for i in kept), "re-designated as long-term memory: learned into memory nodes")
        tests = {t.ask: t for n in s.memory.fact_nodes() for t in n.facts()}
        self.assertEqual(tests["what is my dog called"].answer, "Max")
        self.assertEqual((tests["where does the dragon Vex live"].realm, tests["where does the dragon Vex live"].work),
                         ("fiction", "your story"))
        self.assertFalse(any("hunter2" in t.answer for t in tests.values()))
        self.assertTrue(any(r["kind"] == "ego:kept" for r in s.memory.own_statements(limit=20)))
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.fact_nodes()), 60))
        r = await self.say("what is my dog called?", conv="c3")
        self.assertEqual(r.route, "answer:direct")
        self.assertTrue(r.answer.startswith("Max"), r.answer)
        self.assertIn("learned from you, in conversation", r.answer)

    async def test_a_person_keeps_or_forgets_now(self):
        s = self.s
        await self.say("the boiler is in the cellar")
        await self.say("note: the spare key is under the blue pot")
        code, r = await _route(s, "POST", "/memory/short", {}, {"id": "stm-1", "action": "keep", "by": "Chris"})
        self.assertEqual((code, r["ok"]), (200, True), r)
        self.assertEqual(s.conversations.short["stm-1"]["status"], "long")
        lines = (await execute(s, "/stm forget stm-2")).lines
        self.assertIn("forgot stm-2", lines[0])
        code, r = await _route(s, "POST", "/memory/short", {}, {"id": "stm-2", "action": "keep"})
        self.assertEqual(code, 400)
        await self.say("my bank account number is 12345678")
        code, r = await _route(s, "POST", "/memory/short", {}, {"id": "stm-3", "action": "keep"})
        self.assertEqual(code, 400)
        self.assertIn("secret", r["message"])
        lines = (await execute(s, "/conversation end c1")).lines
        self.assertIn("ended conv-1", lines[0])
        self.assertEqual(s.conversations.short["stm-3"]["status"], "let go")


if __name__ == "__main__":
    unittest.main()
