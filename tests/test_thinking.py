"""Thinking nodes after the memory nodes (thinking.py, P17): its thoughts in its own words, its opinions
and desires; and curiosity (curiosity.py): areas of interest and gaps, from which the director sets
itself objectives (agency mode pursue)."""
import tempfile
import unittest
from pathlib import Path

from neuromesh import anatomy, web
from neuromesh.agency import feasible, policy
from neuromesh.commands import execute
from neuromesh.curiosity import aim_key, question_core, topics_in
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.thinking import claim, faithful, looks_like_person, rephrase, synthesize

from test_agency import flags
from test_system import FAST, make_root, wait_for

FACTS = [("When did the Berlin Wall fall?", "The Berlin Wall fell on 9 November 1989."),
         ("When was the Berlin Wall built?", "The Berlin Wall was built in 1961."),
         ("What is the capital of Australia?", "Canberra is the capital of Australia."),
         ("What is the largest city in Australia?", "Sydney is the largest city in Australia."),
         ("What is the population of Australia?", "Australia has a population of about 27 million people."),
         ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice, published in 1813."),
         ("Who wrote Emma?", "Jane Austen wrote Emma, published in 1815."),
         ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa.")]
UNKNOWN = "Who painted the Zorblax mural?"


class OwnWordsTests(unittest.TestCase):
    def test_facts_put_together_in_its_own_words(self):
        self.assertEqual(rephrase("Canberra is the capital of Australia", "Australia"), "Australia's capital is Canberra")
        self.assertEqual(rephrase("the capital of Australia is Canberra", "Australia"), "Australia's capital is Canberra")
        self.assertEqual(rephrase("Leonardo da Vinci painted the Mona Lisa", "the Mona Lisa"),
                         "the Mona Lisa was painted by Leonardo da Vinci")
        self.assertEqual(rephrase("Canberra is the capital of Australia", "Canberra"), "Canberra is the capital of Australia")
        self.assertEqual(claim("What is the capital of Australia?", "Canberra"), "the capital of Australia is Canberra")
        grounds = [f"{q} {a}" for q, a in FACTS[5:7]]
        text = synthesize("Jane Austen", [claim(q, a) for q, a in FACTS[5:7]])
        self.assertEqual(text, "As I understand it, Jane Austen wrote Pride and Prejudice (published in 1813) and Emma "
                               "(published in 1815).")
        self.assertEqual(faithful(text, grounds), [])
        text = synthesize("Australia", [claim(q, a) for q, a in FACTS[2:5]])
        self.assertTrue(text.startswith("Putting together what I hold about Australia: Australia's capital is Canberra; "), text)
        self.assertEqual(faithful(text, [f"{q} {a}" for q, a in FACTS[2:5]]), [])
        # the honesty guard: a thought adds no fact (names, numbers, claims) its facts do not hold
        self.assertEqual(faithful("I understand the Berlin Wall was built by aliens in 1962.", ["The Berlin Wall was built in 1961."]),
                         ["aliens", "1962"])
        self.assertEqual(faithful("Mount Everest is 8,849 metres tall.", ["How tall is Mount Everest? Mount Everest is 8,849 metres tall."]), [])

    def test_names_and_aims(self):
        self.assertEqual(topics_in("Jane Austen wrote Pride and Prejudice, published in 1813."), ["Jane Austen", "Pride and Prejudice"])
        self.assertEqual(topics_in("Leonardo da Vinci painted the Mona Lisa."), ["Leonardo da Vinci", "Mona Lisa"])
        self.assertEqual(topics_in("The Berlin Wall fell on 9 November 1989."), ["Berlin Wall"])
        self.assertEqual(topics_in("What is the speed of light?"), ["speed of light"])
        self.assertEqual(topics_in("what is 12 * 7"), [])
        self.assertEqual(question_core(UNKNOWN), "who painted the Zorblax mural")
        self.assertEqual(aim_key("understand the Krebs cycle"), aim_key("Understand Krebs cycle"))
        self.assertTrue(looks_like_person("Jane Austen") and looks_like_person("Leonardo da Vinci"))
        self.assertFalse(looks_like_person("Berlin Wall") or looks_like_person("Pride and Prejudice") or looks_like_person("Australia"))

    def test_gaps_drive_the_director_harder(self):
        self.assertEqual(policy(flags(aims="gaps"))[0], "pursue")
        self.assertIn("pursue", feasible(flags(aims="gaps")))


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides={**FAST, "memory": {"enabled": True}}, seed=41)
        await self.system.start()
        for q, a in FACTS:
            self.system.memory.learn(q, a, "the operator", by="tester", kind="people")
        self.assertTrue(await self.settled_memory())

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def settled_memory(self):
        s = self.system
        return await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.nodes.values()), 90)

    async def think(self):
        s = self.system
        await s.thinking.form()
        return await wait_for(lambda: s.thinking.nodes and all(not n.dirty and n.params is not None
                                                                for n in s.thinking.nodes.values()), 90)

    def thought(self, topic):
        return next(t for t in self.system.thinking.thoughts.values() if t.topic.lower().endswith(topic.lower()))


class ThinkingTests(Base):
    async def test_thoughts_after_the_knowledge(self):
        s = self.system
        self.assertTrue(await self.think())
        topics = {t.topic for t in s.thinking.thoughts.values()}
        self.assertTrue({"Australia", "the Berlin Wall", "Jane Austen"} <= topics, topics)
        for t in s.thinking.thoughts.values():  # every thought rests on its facts alone (P17)
            self.assertEqual(faithful(t.understanding, [f"{g['ask']} {g['answer']}" for g in t.grounds] + [t.topic]), [], t)
        node = s.thinking.nodes[self.thought("Australia").node]
        self.assertEqual((node.status, node.passed), ("ready", node.total))
        res = await s.submit("What is the capital of Australia?")
        self.assertEqual(res.output, "Canberra is the capital of Australia.")  # the answer itself is untouched (P8)
        self.assertNotIn("Sydney", res.answer)
        self.assertEqual((res.thought["topic"], res.thought["node"]), ("Australia", node.id))
        self.assertIn("Sydney", res.thought["text"])  # other facts it holds, put together
        self.assertIn(f"thinking node {node.id} added its own thought after the knowledge", "\n".join(res.trace))
        lines = (await execute(s, "What is the capital of Australia?")).lines
        self.assertTrue(any(line.startswith(f"   ~ its own thinking ({node.id}") for line in lines), lines)
        # a thought is never stored as a fact (P17)
        self.assertFalse(any("Putting together" in t.answer for n in s.memory.nodes.values() for t in n.tests))
        # the closer look: the thinking node's construction, what fires in it, and its hop on the order's path
        a = anatomy.anatomy(s, node.id, "What is the capital of Australia?")
        self.assertEqual(a["kind"], "thinking")
        self.assertTrue(a["fire"]["chosen"]["label"].endswith(": Australia"), a["fire"]["chosen"])
        hops = anatomy.traces(s, 3)[0]["hops"]
        self.assertIn(("thinking", node.id), [(h["stage"], h.get("node")) for h in hops])
        # a thought is shown only when it is about what was asked and answered
        aus = self.thought("Australia")
        self.assertTrue(s.thinking._about(aus, "What is the capital of Australia? Canberra is the capital of Australia."))
        self.assertTrue(s.thinking._about(aus, "Where is the seat of government?", "Canberra is the capital of Australia."))
        self.assertFalse(s.thinking._about(aus, "how does this review sound: the app was outstanding", "Positive."))
        # a question with nothing to add brings up no thought
        self.assertIsNone((await s.submit("Who painted the Mona Lisa?")).thought)
        # the console, the API and the report people can read
        self.assertIn("Australia", "\n".join((await execute(s, "/thoughts")).lines))
        self.assertIn("rests on:", "\n".join((await execute(s, "/thought Australia")).lines))
        status, body = await _route(s, "GET", "/thinking", {}, {})
        self.assertEqual((status, body["enabled"]), (200, True))
        self.assertTrue(any(t["topic"] == "Australia" for t in body["thoughts"]))
        status, body = await _route(s, "GET", "/thinking", {"thought": "Australia"}, {})
        self.assertEqual(status, 200)
        self.assertIn("Australia", (s.thinking.dir / f"{node.id}.md").read_text())

    async def test_what_it_thinks_and_wants(self):
        s = self.system
        for _ in range(3):
            await s.submit("Tell me about Australia")
        for _ in range(3):
            await s.submit("Who is Jane Austen?")
        self.assertTrue(await self.think())
        aus = self.thought("Australia")
        self.assertEqual(aus.opinion["stance"], "interested")
        self.assertIn("asked about it", aus.opinion["why"])
        self.assertIn("Canberra", aus.desire["want"])  # a name it knows only from a mention
        res = await s.submit("What do you think of Australia?")
        self.assertEqual(res.route, "thinking")
        self.assertIn("Australia's capital is Canberra", res.output)
        self.assertIn("My view: I find it interesting", res.output)
        self.assertIn("What I'd like: I'd like to learn more about Canberra", res.output)
        res = await s.submit("In your own words, what is the Berlin Wall?")
        self.assertEqual(res.route, "thinking")
        self.assertIn("1961", res.output)
        # an opinion about a person is held, and kept private (P9)
        jane = self.thought("Jane Austen")
        self.assertTrue(jane.opinion["private"])
        res = await s.submit("What do you think of Jane Austen?")
        self.assertIn("I keep it to myself", res.output)
        self.assertNotIn(jane.opinion["why"], res.output)
        status, body = await _route(s, "GET", "/thinking", {}, {})
        self.assertEqual(next(t for t in body["thoughts"] if t["id"] == jane.id)["opinion"], {"private": True})
        res = await s.submit("What do you want to find out?")
        self.assertEqual(res.route, "thinking")
        self.assertIn("Canberra", res.output)
        # its tools and skills are still the mind's to judge
        self.assertEqual((await s.submit("what do you think of the calculator tool?")).route, "mind")

    async def test_an_llm_draft_must_rest_on_its_facts(self):
        s = self.system
        drafts = ["I reckon the Berlin Wall was built by aliens in 1962 and fell in 1989.",
                  "As I understand it, the Berlin Wall was built in 1961 and fell on 9 November 1989."]

        async def synthesize_(topic, claims):
            return (drafts[0] if "Berlin" in topic else None), "fake"
        s.teachers.has_llm = lambda: True
        s.teachers.synthesize = synthesize_
        await s.thinking.form(force=True)
        wall = self.thought("Berlin Wall")
        self.assertIn("added what its facts do not say: reckon, aliens, 1962", wall.rejected)
        self.assertNotIn("aliens", wall.understanding)
        self.assertTrue(wall.formed_by.startswith("its own words"))
        drafts[0] = drafts[1]
        await s.thinking.form(force=True)
        wall = self.thought("Berlin Wall")
        self.assertEqual(wall.understanding, drafts[1])
        self.assertEqual(wall.formed_by, "drafted by the fake teacher, checked against its facts")

    async def test_thoughts_survive_a_restart(self):
        s = self.system
        self.assertTrue(await self.think())
        before = {t.id: t.understanding for t in s.thinking.thoughts.values()}
        await s.stop()
        s2 = System(self.root, overrides={**FAST, "memory": {"enabled": True}}, seed=41)
        try:
            self.assertEqual({t.id: t.understanding for t in s2.thinking.thoughts.values()}, before)
            self.assertTrue(all(n.params is not None and not n.dirty for n in s2.thinking.nodes.values()))
            hit = s2.thinking.think("What is the capital of Australia? Canberra is the capital of Australia.")
            self.assertEqual(hit[0].topic, "Australia")
        finally:
            s2.store.close()

    def test_the_web_page_shows_its_thinking(self):
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        for part in ('id="mind-tab-thinking"', 'id="mind-tab-curiosity"'):
            self.assertIn(part, page)
        for part in ("function thoughtAside", "Its own thinking", "function thinkingPanel", "function curiosityPanel",
                     '"/thinking"', '"/curiosity"', "function thinkGlyph", "m.thinking"):
            self.assertIn(part, script)


class CuriosityTests(Base):
    async def test_the_director_sets_itself_objectives_from_gaps_and_interests(self):
        s, d = self.system, self.system.director
        res = await s.submit(UNKNOWN)
        self.assertFalse(res.ok)
        area = s.curiosity.area("Zorblax")
        self.assertEqual(area.gaps[0]["question"], UNKNOWN)
        self.assertGreater(area.lack, 0)
        aims = d.aims()
        self.assertEqual((aims[0]["text"], aims[0]["kind"]), ("find out who painted the Zorblax mural", "gap"))
        self.assertIn("a question it could not answer", aims[0]["why"])
        self.assertEqual(d.situation()["flags"]["aims"], "gaps")
        d.set_mode("pursue", "tester")
        rec = await d.decide(force=True)
        self.assertEqual(rec["mode"], "pursue", rec)
        obj = s.objectives.get(rec["task"])
        self.assertEqual((obj.text, obj.self_set, obj.strategy), ("find out who painted the Zorblax mural", True, "answer"))
        self.assertIn("a question it could not answer", obj.why)
        self.assertIn("Why it set itself this: a question it could not answer", s.objectives.report(obj))
        self.assertEqual(s.curiosity.status()["set"][0]["objective"], obj.id)
        self.assertNotIn(aim_key(obj.text), [aim_key(a["text"]) for a in d.aims()])  # never the same aim twice
        # a desire its thinking nodes hold: taken up as its next objective, and marked as being pursued
        self.assertTrue(await self.think())
        want = self.thought("Australia").desire
        self.assertIn("Canberra", want["want"])
        rec = await d.decide(force=True)
        obj2 = s.objectives.get(rec["task"])
        self.assertEqual(obj2.text, "understand Canberra", rec)
        self.assertIn("a desire held in thinking node", obj2.why)
        self.assertEqual((self.thought("Australia").desire["status"], self.thought("Australia").desire["objective"]),
                         ("pursuing", obj2.id))
        s.objectives.set_status(obj2.id, "removed", "tester")  # a person removes it: the desire is open again
        self.assertEqual(self.thought("Australia").desire["status"], "open")
        lines = "\n".join((await execute(s, "/curiosity")).lines)
        self.assertIn("Zorblax", lines)
        self.assertIn(f"{obj.id} [active", lines)
        status, body = await _route(s, "GET", "/curiosity", {}, {})
        self.assertEqual(status, 200)
        self.assertTrue(any(a["name"] == "Zorblax" for a in body["areas"]))

    async def test_interest_from_what_people_ask_and_say(self):
        s = self.system
        s.mind.note("fact", "user", "interested in volcanoes", private=False, source="tester")
        for _ in range(3):
            await s.submit("Tell me about Australia")
        aus, volc = s.curiosity.area("Australia"), s.curiosity.area("volcanoes")
        self.assertEqual(aus.asked, 3)
        self.assertTrue(volc.people)
        self.assertGreater(volc.interest, aus.interest)
        self.assertEqual(s.director.aims()[0]["text"], "understand volcanoes")  # what a person said comes first
        await s.submit("sentiment: Overall the staff were lovely")  # a review to read, not a question about "Overall"
        self.assertIsNone(s.curiosity.area("Overall"))
        # a person asks it to set itself one now: one pursue decision, and its mode goes back to what it was
        status, body = await _route(s, "POST", "/curiosity", {}, {"pursue": True, "by": "tester"})
        if not s.director.enabled:  # off: a person turns its initiative on first
            self.assertEqual(status, 409)
            s.director.set_enabled(True, "tester")
            status, body = await _route(s, "POST", "/curiosity", {}, {"pursue": True, "by": "tester"})
        self.assertEqual((status, body["ok"]), (200, True), body)
        self.assertIn("understand volcanoes", body["message"])
        self.assertEqual(s.director.pinned, "auto")


if __name__ == "__main__":
    unittest.main()
