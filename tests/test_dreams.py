"""Sleep and dreams: dreaming only from what it already holds, waking checks and the genetic algorithm,
orders that always wake it, and imagination that is always labelled (P13)."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from neuromesh.commands import execute
from neuromesh.dreams import Dream, cloze, spans, statement
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import imagine_prompt

from test_system import FAST, make_root, wait_for

FACTS = [("What is the capital of France?", "Paris is the capital and largest city of France."),
         ("What is the capital of Japan?", "Tokyo is the capital and largest city of Japan."),
         ("What is the capital of Italy?", "Rome is the capital and largest city of Italy."),
         ("What is the capital of Spain?", "Madrid is the capital and largest city of Spain."),
         ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
         ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet around 1600."),
         ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
         ("When did the Titanic sink?", "The Titanic sank in the North Atlantic in 1912."),
         ("Who was Ada Lovelace?", "Ada Lovelace was an English mathematician who worked with Charles Babbage."),
         ("How tall is the Eiffel Tower?", "The Eiffel Tower is 330 metres tall."),
         ("Who wrote Pride and Prejudice?", "Jane Austen")]  # a short answer: dreamt as a statement

DREAMS = {**FAST, "memory": {"enabled": True, "interval_seconds": 0.02},
          "dreams": {"enabled": True, "sleep_after_idle_seconds": 0.3, "check_every_seconds": 0.2, "sleep_chance": 1.0,
                     "objective_chance": 0.0,
                     "sleep_max_seconds": 3, "dreams_per_night": 300, "ga_epochs": 30}}


class SynthesisTests(unittest.TestCase):
    def test_questions_from_what_it_remembers(self):
        s = "Paris is the capital and largest city of France."
        qs = {sp.text: cloze(s, sp) for sp in spans(s)}
        self.assertEqual(qs, {"Paris": "What is the capital and largest city of France?",
                              "France": "Paris is the capital and largest city of what?"})
        s = "Alexander Fleming discovered penicillin in 1928."
        qs = {sp.text: (sp.kind, cloze(s, sp)) for sp in spans(s)}
        self.assertEqual(qs["Alexander Fleming"], ("name", "Who discovered penicillin in 1928?"))
        self.assertEqual(qs["1928"], ("year", "Alexander Fleming discovered penicillin in what year?"))
        self.assertEqual([sp.text for sp in spans("The Eiffel Tower is 330 metres tall.")], ["Eiffel Tower", "330"])
        self.assertEqual(cloze("Leonardo da Vinci painted the Mona Lisa.", spans("Leonardo da Vinci painted the Mona Lisa.")[1]),
                         "Leonardo da Vinci painted what?")

    def test_statements_from_short_answers(self):
        self.assertEqual(statement("Who painted the Mona Lisa?", "Leonardo da Vinci"), "Leonardo da Vinci painted the Mona Lisa.")
        self.assertEqual(statement("What is the capital of Australia?", "Canberra"), "Canberra is the capital of Australia.")
        self.assertEqual(statement("Which ocean is the largest?", "Pacific Ocean"), "Pacific Ocean is the largest ocean.")
        self.assertEqual(statement("In which country is the city of Kyoto?", "Japan"), "The city of Kyoto is in Japan.")
        self.assertIsNone(statement("What is 9 * 5?", "45"))
        self.assertIsNone(statement("Is the sky blue?", "yes"))
        # dates are one thing to ask about; months and question words are never names
        s = "The Berlin Wall fell on 9 November 1989."
        self.assertEqual({sp.text: cloze(s, sp) for sp in spans(s)},
                         {"9 November 1989": "The Berlin Wall fell on what date?", "Berlin Wall": "What fell on 9 November 1989?"})
        self.assertEqual(spans("What is the capital of Canada?")[0].text, "Canada")

    def test_imagine_orders(self):
        self.assertEqual(imagine_prompt("imagine a city on Mars"), "a city on Mars")
        self.assertEqual(imagine_prompt("What if Ada Lovelace painted the Mona Lisa?"), "Ada Lovelace painted the Mona Lisa?")
        self.assertEqual(imagine_prompt("dream up a new planet"), "a new planet")
        self.assertIsNone(imagine_prompt("what is the capital of France"))


class DreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=DREAMS, seed=3)
        await self.system.start()
        for q, a in FACTS:
            self.system.memory.learn(q, a, "the operator", "me")
        mem = self.system.memory
        self.assertTrue(await wait_for(lambda: mem.nodes and all(not n.dirty for n in mem.nodes.values()), timeout=30))

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_it_sleeps_dreams_wakes_checks_and_evolves(self):
        s, d = self.system, self.system.dreams
        outside = []

        async def no_teacher(*a, **k):
            outside.append("teacher")
            return None

        async def no_research(*a, **k):
            outside.append("research")
            return None, "no"
        s.teachers.answer, s.teachers.compose, s.research = no_teacher, no_teacher, no_research
        self.assertIn("Jane Austen wrote Pride and Prejudice.", [m["text"] for m in d.gather()])
        self.assertTrue(await wait_for(lambda: d.asleep, timeout=10), d.status())
        # asleep: all other background work waits; dreaming gets the capacity
        self.assertTrue(s.guardian.sleeping)
        self.assertFalse(s.guardian.background_allowed())
        self.assertTrue(s.guardian.background_allowed(dreaming=True))
        self.assertTrue(await wait_for(lambda: d.night_dreams > 0, timeout=10))
        self.assertTrue(await wait_for(lambda: any(e.topic == "dream.evolved" for e in s.bus.recent("dream.*", 200)),
                                       timeout=40), d.status())
        self.assertEqual(outside, [], "dreams use only what it already holds")
        reviewed = [e.data for e in s.bus.recent("dream.reviewed", 20)][0]
        self.assertGreater(reviewed["correct"], 0)
        self.assertGreater(reviewed["remembered"] + reviewed["imagined"], 0)
        self.assertGreater(len(d.verified), 0)
        self.assertTrue(all(v[2] in ("remembered", "imagined") for v in d.verified))
        self.assertLessEqual(len(d.nodes), d.pol.population)
        champ = d.champion()
        self.assertIsNotNone(champ.fitness)
        self.assertIn("dream.evolved", {a["kind"] for a in s.store.recent_audit(300)})
        lines = "\n".join((await execute(s, "/dreams")).lines)
        self.assertIn("last night:", lines)
        self.assertIn(f"* {champ.id}", lines)
        code, body = await _route(s, "GET", "/dreams", {}, {})
        self.assertEqual((code, body["nodes"][0]["champion"]), (200, True))
        # the dream nodes survive a restart
        nodes = set(d.nodes)
        await s.stop()
        again = System(self.root, overrides={**DREAMS, "dreams": {**DREAMS["dreams"], "enabled": True}}, seed=4)
        self.assertEqual(set(again.dreams.nodes), nodes)
        again.store.close()

    async def test_an_imagined_dream_checks_out_only_if_it_knows_the_same(self):
        d = self.system.dreams
        d.material = d.gather()

        def dream(q, anchor, answer):
            return Dream(q, None, "imagined", "test", "", [anchor], answer)
        self.assertTrue(d.confirm(dream("What is the capital and largest city of Japan?", "Japan", "Tokyo")))
        # the names appear together in something it knows, but not in this relation
        self.assertFalse(d.confirm(dream("Tokyo wrote what around 1600?", "Tokyo", "Japan")))
        self.assertFalse(d.confirm(dream("What is the capital and largest city of Japan?", "Japan", "Japan")))

    async def test_an_order_always_wakes_it(self):
        s, d = self.system, self.system.dreams
        d.pol.sleep_chance = 0.0
        d.wake("a test starts")
        ok, msg = d.fall_asleep("asked to", by="a test")
        self.assertTrue(ok, msg)
        self.assertTrue(d.asleep)
        res = await s.submit("what is 3 * 4")
        self.assertEqual((res.ok, d.asleep), (True, False))
        self.assertEqual(d.wake_reason, "an order arrived")
        self.assertFalse(s.guardian.sleeping)
        # /sleep and /wake, and a halt wakes it too
        self.assertIn("asleep", (await execute(s, "/sleep")).lines[0])
        self.assertEqual((await execute(s, "/wake")).lines[0], "  awake")
        d.fall_asleep("asked to", by="a test")
        s.guardian.halt("test", by="a test")
        self.assertTrue(await wait_for(lambda: not d.asleep, timeout=5))
        s.guardian.resume(by="a test")

    async def test_awake_it_imagines_and_says_so(self):
        s, d = self.system, self.system.dreams
        d.pol.sleep_chance = 0.0
        d.wake("a test starts")
        born = dict(d.nodes)
        d.nodes.clear()  # before its first dream there is nothing to imagine with
        res = await s.submit("imagine a new planet")
        self.assertFalse(res.ok)
        self.assertIn("sleep and dream first", res.note)
        d.nodes.update(born)
        d.pol.sleep_chance = 1.0
        self.assertTrue(await wait_for(lambda: any(e.topic == "dream.evolved" for e in s.bus.recent("dream.*", 200)),
                                       timeout=40), d.status())
        d.pol.sleep_chance = 0.0
        d.wake("test over")
        facts = sum(len(n.facts()) for n in s.memory.nodes.values())
        res = await s.submit("imagine the capital of Japan")
        self.assertEqual((res.ok, res.route, res.knowledge), (True, "dream:imagine", "imagined"))
        first, *_, source = res.answer.split("\n")
        self.assertTrue(first.startswith("I imagine "), res.answer)
        self.assertTrue(source.startswith("Source: imagined by dream node dream-") and source.endswith("not a fact"), source)
        res = await s.submit("what if Ada Lovelace painted the Mona Lisa")
        self.assertTrue(res.ok and "imagined" in res.answer)
        self.assertEqual(sum(len(n.facts()) for n in s.memory.nodes.values()), facts, "imagination is never stored as a fact")
        code, body = await _route(s, "POST", "/imagine", {}, {"prompt": "a city on Mars"})
        self.assertEqual((code, body["route"]), (200, "dream:imagine"))
        self.assertIn("imagined by dream node", "\n".join((await execute(s, "/imagine a city on Mars")).lines))


if __name__ == "__main__":
    unittest.main()
