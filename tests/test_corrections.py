"""Was the answer right? A person says yes, or no and picks from the next most likely answers (or none
of them, saying what is right if they know); the mesh learns from it and answers right later
(corrections.py, the web answer panel, /correct, GET /alternatives, POST /correct)."""
import json
import tempfile
import unittest
from pathlib import Path

from neuromesh import web
from neuromesh.commands import execute
from neuromesh.corrections import _same
from neuromesh.demo import FACTS
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import DIRECT

from test_system import FAST, make_root, wait_for

MEM = {**FAST, "memory": {"enabled": True}}
UNKNOWN = "Who painted the Zorblax mural?"


class Base(unittest.IsolatedAsyncioTestCase):
    overrides = FAST
    seed = 31

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.overrides, seed=self.seed)
        await self.system.start()
        self.events = []
        self.system.bus.listen(self.events.append, "answer.*")

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def memory_settled(self):
        s = self.system
        return await wait_for(lambda: s.memory.nodes and all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 90)


class CorrectionTests(Base):
    async def test_right_confirms_what_answered(self):
        s = self.system
        r = await s.submit("what is 6 * 7")
        self.assertEqual((r.route, str(r.output)), ("tool:calculator", "42"))
        ok, msg, learned = await s.corrections.right(r.order_id, by="tester")
        self.assertTrue(ok, msg)
        self.assertIn("the router learns that orders like this go to tool:calculator", learned)
        human = [e for e in s.store.examples("dispatch") if e.source == "human"]
        self.assertIn(("what is 6 * 7", "tool:calculator"), [(e.text, e.label) for e in human])
        self.assertEqual(s.corrections.counts, {"right": 1})
        self.assertEqual(json.loads(s.store.get_order(r.order_id)["feedback"])["verdict"], "right")
        self.assertEqual(s.corrections.status()["recent"][0]["order"], r.order_id)
        self.assertTrue(any(e.topic == "answer.right" and e.data["order"] == r.order_id for e in self.events))
        self.assertTrue(any(a["kind"] == "answer.right" for a in s.store.recent_audit(20)))
        # hand-offs (a task, an objective) are not answers to mark
        s.store.save_order("ord-handoff", "remind me to stretch", {"route": "tasks", "ok": True, "output": "ok", "steps": []})
        self.assertFalse((await s.corrections.right("ord-handoff"))[0])
        self.assertFalse((await s.corrections.right("ord-nothing"))[0])

    async def test_a_wrong_label_and_the_next_most_likely(self):
        s = self.system
        text = "sentiment: the pizza was lukewarm and the service was fine"
        r = await s.submit(text)
        self.assertEqual(r.route, "skill:sentiment")
        given = r.output.strip()
        alts = await s.corrections.alternatives(r.order_id)
        self.assertEqual((alts["given"]["answer"], alts["given"]["route"]), (given, "skill:sentiment"))
        labels = [a for a in alts["alternatives"] if a["kind"] == "label"]
        # every other sentiment label is offered, and never the answer it already gave
        self.assertEqual(sorted(a["answer"] for a in labels), sorted(lab for lab in ("positive", "negative", "neutral") if lab != given))
        self.assertNotIn(given, [a["answer"] for a in alts["alternatives"]])
        self.assertEqual([a["id"] for a in alts["alternatives"]], [f"a{i}" for i in range(1, len(alts["alternatives"]) + 1)])
        likes = [a["likelihood"] for a in alts["alternatives"]]
        self.assertEqual(likes, sorted(likes, reverse=True))
        pick = next(a for a in labels if a["answer"] == ("negative" if given != "negative" else "positive"))
        ok, msg, learned = await s.corrections.choose(r.order_id, pick["id"], by="tester")
        self.assertTrue(ok, msg)
        self.assertIn(f"the sentiment skill learns '{pick['answer']}' for this", learned)
        # a person's label stands at once for exactly that wording, and the node trains on it
        again = await s.submit(text)
        self.assertEqual(again.output.strip(), pick["answer"])
        self.assertIn(("the pizza was lukewarm and the service was fine", pick["answer"]),
                      [(e.text, e.label) for e in s.store.examples("sentiment") if e.source == "human"])
        self.assertEqual(s.corrections.marks[r.order_id]["verdict"], pick["id"])
        self.assertTrue(any(e.topic == "answer.corrected" for e in self.events))
        # what a person taught is still there after a restart
        await s.stop()
        s2 = System(self.root, overrides=self.overrides, seed=self.seed)
        try:
            self.assertEqual(s2.teachers.human.answers.get(("sentiment", "the pizza was lukewarm and the service was fine")), pick["answer"])
            self.assertEqual(s2.corrections.marks[r.order_id]["verdict"], pick["id"])
        finally:
            s2.store.close()

    async def test_another_route_worked_out_and_chosen(self):
        s = self.system
        text = 'how many words are in "the hotel was wonderful"'
        r = await s.submit(text)
        self.assertEqual(r.route, "tool:word_stats")
        alts = await s.corrections.alternatives(r.order_id)
        route = next(a for a in alts["alternatives"] if a["kind"] == "route" and a["teach"]["route"] == "skill:sentiment")
        self.assertTrue(route["how"].startswith("the router gave skill:sentiment "))
        self.assertIn(route["answer"], ("positive", "negative", "neutral"))  # worked out now, not just named
        ok, msg, learned = await s.corrections.choose(r.order_id, route["id"], by="tester")
        self.assertTrue(ok, msg)
        self.assertIn("the router learns that orders like this go to skill:sentiment", learned)
        self.assertIn(f"the sentiment skill learns '{route['answer']}' for it", learned)
        again = await s.submit(text)
        self.assertEqual((again.route, again.output.strip()), ("skill:sentiment", route["answer"]))
        # a route the person named themselves is not second-guessed with other routes
        named = await s.submit("sentiment: the pizza was lukewarm and the service was fine")
        self.assertEqual({a["kind"] for a in (await s.corrections.alternatives(named.order_id))["alternatives"]}, {"label"})
        # an alternative that was never offered is refused
        r2 = await s.submit("what is 3 * 3")
        ok, msg, _ = await s.corrections.choose(r2.order_id, "a9")
        self.assertFalse(ok)
        self.assertIn("no alternative a9", msg)

    async def test_the_command_and_the_api(self):
        s = self.system
        r = await s.submit("sentiment: the pizza was lukewarm and the service was fine")
        lines = "\n".join((await execute(s, f"/correct {r.order_id}")).lines)
        self.assertIn(f"{r.order_id}: 'sentiment: the pizza", lines)
        self.assertIn(f"/correct {r.order_id} yes | a1 | none", lines)
        self.assertIn("    a1  ", lines)
        lines = "\n".join((await execute(s, f"/correct {r.order_id} a1")).lines)
        self.assertIn("thanks - ", lines)
        self.assertIn("unknown order", "\n".join((await execute(s, "/correct ord-nope")).lines))
        r2 = await s.submit("what is 8 * 8")
        self.assertIn("thanks - the router learns", "\n".join((await execute(s, f"/correct {r2.order_id} yes")).lines))

        r3 = await s.submit("count the words in: the cat sat")
        status, body = await _route(s, "GET", "/alternatives", {"order": r3.order_id}, {})
        self.assertEqual((status, body["order"], body["given"]["route"]), (200, r3.order_id, "tool:word_stats"))
        status, _ = await _route(s, "GET", "/alternatives", {"order": "ord-nope"}, {})
        self.assertEqual(status, 404)
        status, body = await _route(s, "POST", "/correct", {}, {"order_id": r3.order_id, "verdict": "right", "by": "web"})
        self.assertEqual((status, body["ok"]), (200, True))
        self.assertTrue(body["learned"])
        status, body = await _route(s, "POST", "/correct", {}, {"order_id": r3.order_id, "verdict": "maybe"})
        self.assertEqual(status, 400)
        status, body = await _route(s, "POST", "/correct", {}, {"order_id": r3.order_id, "verdict": "wrong", "choice": "a7"})
        self.assertEqual((status, body["ok"]), (400, False))

    async def test_asking_again_after_a_correction_is_not_nagging(self):
        s = self.system
        text = "sentiment: the pizza was lukewarm and the service was fine"
        r = await s.submit(text)
        s.affect._repeat(text)  # as if it had been asked a moment ago
        await s.corrections.right(r.order_id, by="tester")
        self.assertFalse(s.affect._repeat(text))
        self.assertTrue(s.affect._repeat(text))  # a third time in a minute still is

    def test_the_web_answer_panel_asks(self):
        script = (web.WEB_DIR / "app.js").read_text()
        for part in ("function answerCheck", "Was this right?", "/alternatives?order=", '"/correct"', 'verdict: "right"',
                     'verdict: "wrong"', "None of these", "The right answer, if you know it", "Ask again", "S.checks"):
            self.assertIn(part, script)
        css = (web.WEB_DIR / "app.css").read_text()
        self.assertIn(".alt:has(input:checked)", css)

    def test_same_answers(self):
        self.assertTrue(_same("42", "42."))
        self.assertTrue(_same("Canberra is the capital of Australia", "canberra is the capital of australia."))
        self.assertFalse(_same("positive", "negative"))
        self.assertFalse(_same("", ""))


class MemoryCorrectionTests(Base):
    overrides = MEM
    seed = 32

    async def test_none_of_them_and_what_is_right(self):
        s = self.system
        r = await s.submit(UNKNOWN)
        self.assertNotIn("Lindqvist", str(r.output))
        alts = await s.corrections.alternatives(r.order_id)
        self.assertTrue(alts["question"])
        ok, msg, learned = await s.corrections.choose(r.order_id, "none", "Ada Lindqvist painted it.", by="tester")
        self.assertTrue(ok, msg)
        self.assertTrue(any(x.startswith("memory node mem-") and "Ada Lindqvist" in x for x in learned), learned)
        if r.route != DIRECT and not r.ok:
            self.assertTrue(any("answered directly" in x for x in learned), learned)
        self.assertTrue(await self.memory_settled())
        again = await s.submit(UNKNOWN)
        self.assertIn("Ada Lindqvist", again.output)
        self.assertEqual(s.corrections.marks[r.order_id]["answer"], "Ada Lindqvist painted it.")
        self.assertNotIn(r.order_id, s.corrections.wrong)

    async def test_none_of_them_and_no_one_knows(self):
        s = self.system
        r = await s.submit(UNKNOWN)
        await s.corrections.alternatives(r.order_id)
        ok, msg, learned = await s.corrections.choose(r.order_id, "none", by="tester")
        self.assertTrue(ok, msg)
        self.assertIn(r.order_id, s.corrections.wrong)
        self.assertIn(r.order_id, s.mind.unmet)
        asked = [q for q in s.mind.open_inquiries() if q.order_id == r.order_id]
        self.assertTrue(asked and asked[0].kind == "clarify", learned)
        self.assertIn("'answer: ...'", asked[0].question)  # one question per order: it may have asked already
        doubts = [d for d in s.objectives.doubts() if d["order"] == r.order_id]
        self.assertEqual(doubts[0]["why"], "you said it was wrong")
        # the person finds out later and replies: it keeps the answer
        ok, lines = await s.mind.reply(asked[0].id, "answer: Ada Lindqvist", by="tester")
        self.assertTrue(ok, lines)
        self.assertTrue(await self.memory_settled())
        self.assertIn("Ada Lindqvist", (await s.submit(UNKNOWN)).output)

    async def test_a_memory_node_learns_not_to_give_a_wrong_answer(self):
        s = self.system
        q = FACTS[1][0]
        for fq, fa in FACTS[:4]:
            s.memory.learn(fq, fa, "the operator", by="tester", kind="people")
        self.assertTrue(await self.memory_settled())
        r = await s.submit(q)
        self.assertIn("8,849", r.output)
        alts = await s.corrections.alternatives(r.order_id)
        self.assertEqual(alts["given"]["source"], "memory")
        ok, msg, learned = await s.corrections.choose(r.order_id, "none", by="tester")
        self.assertTrue(ok, msg)
        node_id = next(x for x in learned if x.startswith("memory node")).split()[2]
        node = s.memory.nodes[node_id]
        self.assertIn(q, [t.ask for t in node.unknowns])
        self.assertNotIn(q, [t.ask for t in node.tests])
        self.assertTrue(any(x["kind"] == "memory.must_not_answer" for x in s.store.recent_audit(30)))
        self.assertTrue(await self.memory_settled())
        self.assertNotIn("8,849", str((await s.submit(q)).output))
        # the other facts it holds are untouched
        self.assertIn("Canberra", (await s.submit("What is the capital of Australia?")).output)


if __name__ == "__main__":
    unittest.main()
