"""Objectives (P16): long-term aims it plans for, pursues by asking itself, ruminates over (checks,
echoes, contradictions, other words, reading, a person), follows up, and writes up; second thoughts."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.commands import execute
from neuromesh.demo import FACTS
from neuromesh.objectives import (PLANNER, PRACTICE, STRATEGIES, PlannerTeacher, fits, list_items, new_terms,
                                  objective_request, plan_questions, reword, split_questions, subject_of)
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import Result

from test_research import MockWikipedia, start_wiki
from test_system import FAST, make_root, wait_for

OBJ = {**FAST, "memory": {"enabled": True}, "objectives": {"step_seconds": 0, "second_thought_after_seconds": 0}}


async def drive(book, obj, limit=40, answer=None):
    """Work an objective to its end (answering a person's questions with answer(question))."""
    for _ in range(limit):
        if obj.status != "active":
            return
        obj.next_at = 0
        await book.work(obj, "tester")
        waiting = [q for q in obj.questions if q["status"] == "waiting"]
        if waiting and answer is not None:
            await book.sys.mind.reply(waiting[0]["inquiry"], answer(waiting[0]["ask"]), by="tester")


class PlanningTests(unittest.TestCase):
    def test_a_person_sets_an_objective(self):
        self.assertEqual(objective_request("objective: understand the Berlin Wall"), "understand the Berlin Wall")
        self.assertEqual(objective_request("Long-term goal: learn about jazz."), "learn about jazz")
        self.assertEqual(objective_request("make it your objective to compare tea and coffee"), "compare tea and coffee")
        self.assertEqual(objective_request("work towards understanding volcanoes"), "understanding volcanoes")
        for no in ("what is your goal", "the goal was scored", "objective lens prices"):
            self.assertIsNone(objective_request(no), no)

    def test_the_planner_teacher_knows_each_strategy(self):
        t = PlannerTeacher(seed=2)
        spec = type("Spec", (), {"name": PLANNER})()
        for strategy in STRATEGIES:
            texts = asyncio.run(t.generate(spec, strategy, 5))
            self.assertTrue(texts, strategy)
            self.assertTrue(all(t.label_sync(x).label == strategy for x in texts), (strategy, texts))
            self.assertTrue(all(t.label_sync(x.format(t="jazz", a="tea", b="coffee", s="the continents")).label == strategy
                                for x in PRACTICE[strategy][:3]))
        self.assertEqual(t.label_sync("frobnicate").label, "explain")  # no cue: understand it
        self.assertIsNone(asyncio.run(t.label(type("S", (), {"name": "sentiment"})(), "compare tea and coffee")))

    def test_plans(self):
        self.assertEqual(split_questions("find out who wrote Pride and Prejudice and when it was published"),
                         ["Who wrote Pride and Prejudice?", "When was Pride and Prejudice published?"])
        self.assertEqual(split_questions("work out how tall Mount Everest is, and how old it is"),
                         ["How tall is Mount Everest?", "How old is Mount Everest?"])
        ex = plan_questions("explain", "learn about volcanoes", subject_of("learn about volcanoes", "explain"))
        self.assertEqual([(q["kind"], q["ask"], q["core"]) for q in ex],
                         [("research", "volcanoes", False), ("ask", "What are volcanoes?", True),
                          ("ask", "What are volcanoes known for?", True), ("ask", "What is the history of volcanoes?", False)])
        self.assertEqual(plan_questions("explain", "understand Ada Lovelace", "Ada Lovelace")[1]["ask"], "Who was Ada Lovelace?")
        cmp = [q["ask"] for q in plan_questions("compare", "compare tea and coffee", "tea and coffee")]
        self.assertEqual(cmp, ["tea", "coffee", "What is tea?", "What is coffee?", "What is the difference between tea and coffee?"])
        self.assertEqual(plan_questions("survey", "list the noble gases", "the noble gases")[1]["ask"], "What are the noble gases?")
        for strategy, text in (("watch", "keep an eye on the price of gold"), ("improve", "get your sentiment skill to 95%"),
                               ("build", "be able to reverse text")):
            q = plan_questions(strategy, text, subject_of(text, strategy))[0]
            self.assertEqual((q["kind"], q["delegate"]), ("delegate", strategy))
        self.assertEqual(subject_of("be able to reverse text", "build"), "reverse text")

    def test_what_it_reads_into_answers(self):
        self.assertEqual(list_items("The inner planets are Mercury, Venus, Earth and Mars."), ["Mercury", "Venus", "Earth", "Mars"])
        self.assertEqual(list_items("It is a gas."), [])
        self.assertEqual(new_terms("The Berlin Wall was built by the German Democratic Republic on 13 August 1961 "
                                   "to stop people leaving for West Berlin.", "understand the Berlin Wall", 3),
                         ["German Democratic Republic", "West Berlin"])
        self.assertEqual(reword("When was Pride and Prejudice published?"), "in what year was Pride and Prejudice published?")
        self.assertEqual(reword("What is the Berlin Wall known for?"), "why is the Berlin Wall famous?")
        self.assertTrue(fits("When was it built?", "In 1961."))
        self.assertFalse(fits("When was it built?", "It is a wall."))
        self.assertTrue(fits("Who wrote Hamlet?", "William Shakespeare wrote it."))
        self.assertFalse(fits("Who wrote Hamlet?", "Hamlet is a play."))
        self.assertTrue(fits("How many bones are in the human body?", "An adult has 206 bones."))
        self.assertTrue(fits("What is jazz?", "Music."))


class ObjectiveTests(unittest.IsolatedAsyncioTestCase):
    overrides = OBJ

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.overrides, seed=13)
        await self.system.start()

    async def asyncTearDown(self):
        await self.system.stop()
        self.tmp.cleanup()

    async def teach_facts(self):
        s = self.system
        for q, a in FACTS:
            s.memory.learn(q, a, "the operator", by="tester", kind="people")
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 60))

    async def test_it_answers_by_asking_itself_and_writes_it_up(self):
        s, book = self.system, self.system.objectives
        await self.teach_facts()
        res = await s.submit("objective: find out who wrote Pride and Prejudice and when it was published")
        self.assertEqual((res.ok, res.route), (True, "objectives"))
        self.assertIn("Objective obj-1 set", res.output)
        self.assertFalse(any("attempt" in t for t in res.trace))  # setting it is not an order to retry
        obj = book.get("obj-1")
        self.assertEqual(obj.strategy, "answer")
        self.assertEqual([q["ask"] for q in obj.questions], ["Who wrote Pride and Prejudice?", "When was Pride and Prejudice published?"])
        await drive(book, obj)
        self.assertEqual(obj.status, "met", book.show("obj-1"))
        for q in obj.questions:
            self.assertEqual((q["status"], q["source"]), ("answered", "memory"))
            self.assertIn("Jane Austen", q["answer"])
            self.assertEqual(q["confirmed"], "single")  # nothing else to check it against
            self.assertTrue(q["order"].startswith("ord-"))
        report = book.report_path(obj).read_text()
        for part in ("# Find out who wrote Pride and Prejudice", "## What it found", "### Who wrote Pride and Prejudice?",
                     "only one source", "## How it got here"):
            self.assertIn(part, report)
        self.assertTrue(any(n.kind == "finding" and n.subject == "objective:obj-1" for n in s.mind.notes.values()))
        topics = [e.topic for e in s.bus.recent("objective.*", 200)]
        for t in ("objective.set", "objective.step", "objective.ruminated", "objective.ended"):
            self.assertIn(t, topics)
        self.assertTrue({"objective.set", "objective.met"} <= set(s.store.audit_counts()))
        self.assertIn("obj-1 [met]", "\n".join((await execute(s, "/objectives")).lines))
        self.assertIn("## Questions", "\n".join((await execute(s, "/objective report obj-1")).lines))

    async def test_it_doubts_an_echo_and_asks_a_person(self):
        s, book = self.system, self.system.objectives
        await self.teach_facts()
        obj = await book.add("understand the Berlin Wall", by="tester")
        self.assertEqual(obj.strategy, "explain")
        asked = []

        def person(q):
            asked.append(q)
            return "It was a guarded concrete barrier that divided Berlin from 1961 to 1989."
        await drive(book, obj, answer=person)
        self.assertEqual(obj.status, "met", book.show(obj.id))
        q3 = next(q for q in obj.questions if q["ask"] == "What is the Berlin Wall known for?")
        self.assertEqual((q3["source"], q3["confirmed"]), ("person", True))
        self.assertEqual(asked, ["What is the Berlin Wall known for?"])  # one question for a person, about what it lacked
        self.assertTrue(any(t[1] == "doubt" and "got the same answer" in t[2] for t in obj.thoughts))
        self.assertIsNotNone(s.memory.known("What is the Berlin Wall known for?"))  # a person's answer is kept (P1)
        report = book.report(obj)
        self.assertIn("## Still open", report)
        self.assertIn("What is the history of the Berlin Wall?", report)
        self.assertNotIn("### tell me about the Berlin Wall", report)  # stand-ins are folded into the question they stood for
        # a person who does not know: it stays open, and the objective ends short instead of guessing
        obj2 = await book.add("find out how many moons Neptune has", by="tester")
        await drive(book, obj2, answer=lambda q: "skip")
        self.assertEqual(obj2.status, "unmet", book.show(obj2.id))
        self.assertIn("could not find out: How many moons Neptune has?", obj2.note)

    async def test_disagreeing_sources_are_settled_not_hidden(self):
        s, book = self.system, self.system.objectives
        await self.teach_facts()
        orch = s.orchestrator
        real = orch.attempt

        async def attempt(order, method):
            if method == "knowledge" and "largest" in order.text:
                return Result(order.id, order.text, "answer:direct", "Saturn is the largest planet.", 0.8, knowledge="recalled")
            return await real(order, method)
        orch.attempt = attempt
        try:
            obj = await book.add("find out which planet is the largest", by="tester")
            await drive(book, obj, answer=lambda q: "Jupiter is the largest planet; Saturn is the second largest.")
        finally:
            orch.attempt = real
        q = obj.questions[0]
        self.assertTrue(any(t[1] == "contradiction" for t in obj.thoughts), obj.thoughts)
        self.assertEqual((q["source"], q["confirmed"], obj.status), ("person", True, "met"))
        self.assertIn("settled it", book.report(obj))
        self.assertTrue(any(e.topic == "objective.contradiction" for e in s.bus.recent("objective.*", 200)))

    async def test_watch_improve_and_build_are_handed_on(self):
        s, book = self.system, self.system.objectives
        w = await book.add("keep an eye on the price of gold", by="tester")
        await drive(book, w)
        self.assertEqual(w.status, "met")
        task = next(t for t in s.tasks.tasks.values() if t.kind == "run")
        self.assertEqual(task.target, "what is the price of gold")
        i = await book.add("get your sentiment skill to 95%", by="tester")
        await drive(book, i, limit=3)
        self.assertEqual(i.status, "active")  # until the task meets its target
        self.assertTrue(any(t.kind == "improve" and t.target == "sentiment" and abs(t.threshold - 0.95) < 1e-9
                            for t in s.tasks.tasks.values()))
        b = await book.add("be able to reverse text", by="tester")
        await drive(book, b, limit=3)
        self.assertEqual(b.questions[0]["status"], "waiting-tool")
        d = s.toolsmith.drafts["reverser"]
        s.guardian.decide(d["approval"], True, by="tester")
        self.assertTrue(await wait_for(lambda: s.toolsmith.drafts["reverser"]["status"] == "active", 20))
        await drive(book, b)
        self.assertEqual(b.status, "met", book.show(b.id))

    async def test_surveys_follow_the_list(self):
        s, book = self.system, self.system.objectives
        obj = await book.add("list the inner planets", by="tester")
        self.assertEqual(obj.strategy, "survey")
        await drive(book, obj, limit=12, answer=lambda q: "The inner planets are Mercury, Venus, Earth and Mars.")
        asks = [q["ask"] for q in obj.questions]
        for p in ("Mercury", "Venus", "Earth", "Mars"):
            self.assertIn(f"What is {p}?", asks)

    async def test_people_stay_in_charge(self):
        s, book = self.system, self.system.objectives
        out = "\n".join((await execute(s, "/objective compare tea and coffee")).lines)
        self.assertIn("set obj-1: compare tea and coffee (compare strategy", out)
        self.assertIn("added q6 to obj-1", "\n".join((await execute(s, "/objective obj-1 ask which has more caffeine")).lines))
        self.assertIn("paused", "\n".join((await execute(s, "/objective pause obj-1")).lines))
        self.assertIsNone(book.due())
        status, body = await _route(s, "POST", "/objectives", {}, {"id": "obj-1", "action": "resume"})
        self.assertEqual(status, 200, body)
        status, body = await _route(s, "POST", "/objectives", {}, {"text": "understand jazz"})
        self.assertEqual((status, body["id"]), (200, "obj-2"))
        status, body = await _route(s, "GET", "/objectives", {}, {})
        self.assertEqual([o["id"] for o in body["objectives"]], ["obj-1", "obj-2"])
        status, body = await _route(s, "GET", "/objectives", {"id": "obj-2"}, {})
        self.assertIn("# Understand jazz", body["report"])
        with self.assertRaises(ValueError):
            await book.add("understand jazz")  # it already works on that
        # nothing happens while it is paused (P1), and an order in flight comes first (P2)
        s.guardian.pause_background(True, "tester")
        steps = book.counts.get("steps", 0)
        await asyncio.sleep(0.6)
        self.assertEqual(book.counts.get("steps", 0), steps)
        s.guardian.pause_background(False, "tester")
        self.assertTrue(await wait_for(lambda: book.counts.get("steps", 0) > steps, 15))  # the Ruminator works when idle
        self.assertIn("removed", (await _route(s, "POST", "/objectives", {}, {"id": "obj-2", "action": "remove"}))[1]["message"])
        # objectives outlive a restart
        await s.stop()
        again = System(self.root, overrides=self.overrides, seed=14)
        self.assertEqual(list(again.objectives.items), ["obj-1"])
        self.assertEqual(again.objectives.get("obj-1").questions[-1]["ask"], "which has more caffeine?")
        self.system = again
        await again.start()

    async def test_second_thoughts(self):
        s, book = self.system, self.system.objectives
        res = await s.submit("What is the capital of Freedonia?")
        self.assertFalse(res.ok)
        self.assertEqual([d["text"] for d in book.doubts()], ["What is the capital of Freedonia?"])
        s.memory.learn("What is the capital of Freedonia?", "Fredville is the capital of Freedonia.", "a person", by="tester",
                       kind="people")
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 60))
        got = await book.second_thoughts(by="tester")
        self.assertEqual(got[0]["verdict"], "now answers", got)
        self.assertIn("Fredville", got[0]["line"])
        self.assertEqual(book.doubts(), [])  # it does not think the same order over twice
        self.assertTrue(any(n.kind == "update" and "Fredville" in n.text for n in s.mind.notes.values()))
        self.assertIn("now answers", "\n".join((await execute(s, "/objectives")).lines))

    async def test_its_own_initiative_pursues_and_ruminates(self):
        s, d = self.system, self.system.director
        s.mind.note("fact", "user", "interested in the Krebs cycle", private=False, source="tester")
        sit = d.situation()
        self.assertEqual(sit["flags"]["aims"], "some")
        d.set_mode("pursue", "tester")
        rec = await d.decide(force=True)
        self.assertEqual(rec["mode"], "pursue", rec)
        obj = s.objectives.get(rec["task"])
        self.assertTrue(obj.self_set)
        self.assertEqual(obj.text, "understand the Krebs cycle")
        self.assertIn(obj.id, d.self_objectives)
        self.assertEqual(d.situation()["flags"]["aims"], "none")  # it does not set the same aim twice
        s.objectives.set_status(obj.id, "removed", "tester")
        d.review()
        self.assertNotIn(obj.id, d.self_objectives)
        await s.submit("What is the capital of Freedonia?")
        self.assertEqual(d.situation()["flags"]["doubts"], "some")
        d.set_mode("ruminate", "tester")
        rec = await d.decide(force=True)
        self.assertEqual(rec["mode"], "ruminate", rec)
        self.assertIn("thought again about 1 recent answer", rec["summary"])


class PlannedByTeacherTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_llm_teacher_may_draft_the_questions(self):
        tmp = tempfile.TemporaryDirectory()
        s = System(make_root(Path(tmp.name)), overrides=OBJ, seed=15)
        try:
            async def plan(objective, strategy, skills, tools):
                return {"questions": [{"ask": "What is photosynthesis?", "core": True},
                                      {"ask": "Which pigment absorbs light in plants?", "core": True}]}, "llm"
            s.teachers.plan = plan
            obj = await s.objectives.add("understand photosynthesis", by="tester")
            self.assertIn("questions drafted by the llm teacher", obj.planned_by)
            self.assertEqual([q["ask"] for q in obj.questions if q["kind"] == "ask"],
                             ["What is photosynthesis?", "Which pigment absorbs light in plants?"])
        finally:
            tmp.cleanup()


class ReadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_it_reads_up_then_asks_itself_and_follows_up(self):
        MockWikipedia.requests.clear()
        MockWikipedia.outage = False
        server, api = start_wiki()
        tmp = tempfile.TemporaryDirectory()
        over = {**OBJ, "tools": {**FAST["tools"], "config": {"wikipedia": {"api_url": api}}}, "research": {"follow_links": 0}}
        s = System(make_root(Path(tmp.name), wikipedia=True), overrides=over, seed=16)
        await s.start()
        try:
            self.assertTrue(await wait_for(lambda: "wikipedia" in s.tools.tools and s.tools.tools["wikipedia"].approval_id))
            s.guardian.decide(s.tools.tools["wikipedia"].approval_id, True, by="tester")
            self.assertTrue(await wait_for(lambda: s.tools.tools["wikipedia"].state == "active", 20))
            book = s.objectives
            obj = await book.add("understand Ada Lovelace", by="tester")
            await drive(book, obj, limit=60, answer=lambda q: "skip")
            research = obj.questions[0]
            self.assertEqual((research["kind"], research["status"]), ("research", "answered"))
            self.assertIn("Ada Lovelace", research["answer"])
            first = obj.questions[1]
            self.assertEqual((first["ask"], first["status"], first["source"]), ("Who was Ada Lovelace?", "answered", "knowledge"))
            self.assertIn("mathematician", first["answer"])
            follow = [q["ask"] for q in obj.questions if q.get("role") == "follow"]
            self.assertIn("Who was Charles Babbage?", follow)
            self.assertIn(obj.status, ("met", "unmet"))
            self.assertIn("## Reading", book.report(obj))
        finally:
            await s.stop()
            server.shutdown()
            server.server_close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
