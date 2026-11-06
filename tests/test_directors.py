"""The directors: always at least one, at both ends of every order, the only way into long-term memory,
duplicating themselves and collaborating when needed, and shaped by the mood (P14)."""
import asyncio
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.agency import situation_text
from neuromesh.commands import execute
from neuromesh.config import DirectorsPolicy
from neuromesh.directors import PRIMARY, agrees, stance
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import DIRECT, Order, Result
from neuromesh.web import mesh

from test_system import FAST, ROOT, make_root, wait_for

DIRECTORS = {**FAST, "memory": {"enabled": True, "interval_seconds": 0.02},
             "directors": {"consult": True, "retire_after_idle_seconds": 0.6, "beat_seconds": 0.05}}
BASE = {"valence": 0.2, "arousal": 0.3, "confidence": 0.6, "trust": 0.6, "patience": 0.7}


def feel(system, **values):
    """Set the mood directly (the temperament engines are off in these tests)."""
    system.affect.state.values.update(values)
    system.affect.state.updated = time.time()


class StanceTests(unittest.TestCase):
    def test_the_mood_shapes_the_stance(self):
        pol = DirectorsPolicy()
        calm = stance(BASE, BASE, "steady", pol)
        self.assertEqual((round(calm.review_confidence, 2), round(calm.memory_confidence, 2)), (0.6, 0.35))
        self.assertTrue(calm.consult and not calm.wary and not calm.impatient)
        self.assertEqual((calm.spawn_waiting, calm.pace, calm.explore), (2, 1.0, 1.0))
        # low trust and confidence: more second opinions, a higher bar for memory, and wary
        wary = stance({**BASE, "trust": 0.2, "confidence": 0.3}, BASE, "wary", pol)
        self.assertTrue(wary.wary and wary.consult)
        self.assertGreater(wary.review_confidence, calm.review_confidence)
        self.assertGreater(wary.memory_confidence, calm.memory_confidence)
        self.assertTrue(any(e.startswith("wary") for e in wary.effects))
        # impatient: no second opinions, and it duplicates itself sooner
        hasty = stance({**BASE, "patience": 0.1}, BASE, "steady", pol)
        self.assertTrue(hasty.impatient and not hasty.consult)
        self.assertEqual(hasty.spawn_waiting, 1)
        # stirred up: duplicates sooner and takes initiative sooner; good spirits explore more
        aroused = stance({**BASE, "arousal": 0.9, "valence": 0.7}, BASE, "upbeat", pol)
        self.assertEqual(aroused.spawn_waiting, 1)
        self.assertLess(aroused.pace, 1.0)
        self.assertGreater(aroused.explore, 1.0)
        # and the mood can be switched off
        pol.mood_influence = 0.0
        flat = stance({**BASE, "trust": 0.1, "patience": 0.05}, BASE, "wary", pol)
        self.assertEqual((flat.mood, flat.wary, flat.consult, round(flat.review_confidence, 2)), ("steady", False, True, 0.6))

    def test_agreement(self):
        self.assertTrue(agrees("Canberra", "The capital of Australia is Canberra."))
        self.assertTrue(agrees("Paris is the capital of France.", "The capital of France is Paris"))
        self.assertFalse(agrees("Lutz is the capital of Zubrowka.", "Pankrac is the capital of Zubrowka."))
        self.assertFalse(agrees("Sydney", "The capital of Australia is Canberra."))
        self.assertFalse(agrees("8,849 metres", "8,848.86 metres tall"))


class DirectorTests(unittest.IsolatedAsyncioTestCase):
    overrides = DIRECTORS

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.overrides, seed=7)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    def settled(self):
        mem = self.system.memory
        return all(not n.dirty for n in mem.nodes.values())

    async def test_there_is_always_at_least_one_director(self):
        s, pool = self.system, self.system.directors
        self.assertTrue(await wait_for(lambda: pool.nodes[PRIMARY].running, timeout=5))
        ok, msg = pool.retire(PRIMARY, "a test", by="a test")
        self.assertFalse(ok)
        self.assertIn("never retires", msg)
        made = [pool.spawn("a test") for _ in range(5)]
        self.assertEqual(len(pool.nodes), pool.limit)
        self.assertIsNone(made[-1], "never more than max_directors")
        self.assertTrue(all(n.running for n in pool.nodes.values()))
        # idle duplicates retire on their own; director-0 stays
        self.assertTrue(await wait_for(lambda: list(pool.nodes) == [PRIMARY], timeout=10), pool.lines())
        self.assertTrue(any(e.topic == "director.retired" for e in s.bus.recent("director.*", 50)))
        # if director-0 ever stops, it is restarted
        pool.nodes[PRIMARY].task.cancel()
        self.assertTrue(await wait_for(lambda: pool.restarts == 1 and pool.nodes[PRIMARY].running, timeout=5))
        self.assertIn("director.restarted", {a["kind"] for a in s.store.recent_audit(100)})
        # a halt stops what it does, not the directors: the order is refused, through a director
        s.guardian.halt("test", by="a test")
        res = await s.submit("what is 2 + 2")
        self.assertFalse(res.ok)
        self.assertEqual(res.director, PRIMARY)
        s.guardian.resume(by="a test")

    async def test_every_order_passes_a_director_on_the_way_in_and_out(self):
        s, pool = self.system, self.system.directors
        res = await s.submit("what is 12 * 7")
        self.assertEqual((res.ok, res.output), (True, 84))
        self.assertTrue(res.trace[0].startswith(f"{PRIMARY} took the order (steady"), res.trace)
        self.assertTrue(res.trace[-1].startswith(f"reviewed by {PRIMARY} (steady): approved"), res.trace)
        self.assertEqual(res.director, PRIMARY)
        self.assertEqual((pool.counts["intake"], pool.counts["review"]), (1, 1))
        self.assertEqual(s.store.get_order(res.order_id)["result"]["director"], PRIMARY)
        # orders run by background tasks and benchmark exams pass a director too
        res = await s.orchestrator.fulfil(Order(text="what is 3 + 4", source="challenge:x", background=True, learn=False))
        self.assertEqual(res.director, PRIMARY)
        self.assertIn("a benchmark question", res.trace[0])
        self.assertEqual(pool.counts["review"], 2)

    async def test_long_term_memory_only_through_a_directors_review(self):
        s, pool = self.system, self.system.directors

        async def answer(text):
            return {"why is the sky blue?": "The sky looks blue because air scatters blue light the most.",
                    "why is grass green?": "Grass is green because of chlorophyll.",
                    "what is the capital of zubrowka?": "Pankrac is the capital of Zubrowka."}.get(text.lower()), "mock"
        s.teachers.answer = answer
        res = await s.submit("why is the sky blue?")
        self.assertEqual((res.ok, res.served_by), (True, "teachers"))
        self.assertTrue(any(t.startswith(f"{PRIMARY} asked director-1 for a second opinion (a language model's answer")
                            for t in res.trace), res.trace)  # a duplicate was made to give it
        self.assertTrue(any("stored it in long-term memory" in t for t in res.trace), res.trace)
        self.assertIsNotNone(s.memory.known("why is the sky blue?"))
        # wary (low trust): a language model's answer nothing it holds agrees with is not stored
        feel(s, trust=0.2)
        res = await s.submit("why is grass green?")
        self.assertTrue(res.ok)
        self.assertTrue(any("did not store it in long-term memory: trust is low" in t for t in res.trace), res.trace)
        self.assertIsNone(s.memory.known("why is grass green?"))
        self.assertTrue(any(e.data["why"].startswith("trust is low") for e in s.bus.recent("director.memory", 10)))
        # ... but a person's word is always kept, and an imagined answer never
        self.assertIsNotNone(s.memory.learn("What is the capital of Zubrowka?", "Lutz is the capital of Zubrowka.",
                                            "the operator", by="a person"))
        self.assertIsNone(s.memory.learn("What is on Mars?", "a city of glass", "imagined by dream node dream-0001",
                                         by="dreams", kind="knowledge"))
        feel(s, **BASE)
        self.assertTrue(await wait_for(self.settled, timeout=30), s.memory.status())
        # a second opinion from what it already holds, never a teacher or the web
        fake = Result("ord-x", "What is the capital of Zubrowka?", DIRECT, "Pankrac is the capital of Zubrowka.", 0.9,
                      served_by="teachers")
        second = pool._second_opinion(Order(text="What is the capital of Zubrowka?"), fake)
        self.assertEqual(second["verdict"], "disagrees")
        self.assertIn("Lutz", second["answer"])
        self.assertTrue(second["source"].startswith("memory node "))
        self.assertEqual(pool.review_memory("What is the capital of Zubrowka?", "Pankrac is the capital of Zubrowka.",
                                            "the mock teacher (a language model)", "mock", "teachers", confidence=0.9)[1],
                         "a person told it otherwise ('Lutz is the capital of Zubrowka.')")

    async def test_a_disagreement_is_disclosed_and_kept_out_of_memory(self):
        s, pool = self.system, self.system.directors

        async def answer(text):
            return "The zorblax was invented by Ada in 1843.", "mock"
        s.teachers.answer = answer
        pool._from_nodes = lambda order, res: {"verdict": "disagrees", "answer": "Charles invented it in 1820.",
                                                   "source": "memory node mem-0042 (51% sure)"}
        res = await s.submit("who invented the zorblax?")
        first, *_, note = res.answer.split("\n")
        self.assertEqual(first, "The zorblax was invented by Ada in 1843.")  # the answer itself is unchanged (P8)
        self.assertEqual(note, "[director: director-1 checked memory node mem-0042 (51% sure) and found a different "
                               "answer: 'Charles invented it in 1820.'. Treat this answer with care]")
        self.assertLessEqual(res.confidence, 0.5)
        self.assertIn("approved with a note", res.review)
        self.assertTrue(any("did not store it in long-term memory: a second director found a different answer" in t
                            for t in res.trace), res.trace)
        self.assertIsNone(s.memory.known("who invented the zorblax?"))
        self.assertTrue(any(e.data["verdict"] == "disagrees" for e in s.bus.recent("director.consulted", 10)))
        # an exam answer is never given a note (it is graded as the mesh answered it)
        res = await s.orchestrator.fulfil(Order(text="who invented the zorblax?", source="challenge:x", background=True,
                                                learn=False))
        self.assertNotIn("[director:", res.answer)
        # impatient: no second opinions at all
        feel(s, patience=0.1)
        res = await s.submit("who invented the zorblax, really?")
        self.assertIn("no second opinions (impatient)", res.trace[0])
        self.assertNotIn("[director:", res.answer)

    async def test_it_duplicates_itself_when_work_waits_and_they_retire(self):
        s, pool = self.system, self.system.directors
        feel(s, arousal=0.9)  # stirred up: it duplicates itself as soon as work waits
        self.assertEqual(pool.stance().spawn_waiting, 1)
        # skills the teachers still label get second opinions: reviews wait on them, so work waits too
        orders = [f"sentiment: the hotel was {w} {i}" for i in range(4) for w in ("lovely", "awful", "fine")]
        results = await asyncio.gather(*(s.submit(o) for o in orders))
        self.assertTrue(all(r.director for r in results))
        self.assertGreater(pool.most, 1, pool.lines())
        self.assertGreater(len({r["director"] for r in pool.reviews} | {r["consulted"] for r in pool.reviews} - {""}), 1)
        spawned = [e.data for e in s.bus.recent("director.spawned", 20)]
        self.assertTrue(spawned and all("waiting for a director" in d["reason"] or "second opinion" in d["reason"]
                                        for d in spawned), spawned)
        self.assertTrue(await wait_for(lambda: list(pool.nodes) == [PRIMARY], timeout=10), pool.lines())

    async def test_commands_api_and_web(self):
        s, pool = self.system, self.system.directors
        await s.submit("what is 5 * 5")
        lines = "\n".join((await execute(s, "/directors")).lines)
        self.assertIn("directors: 1 running", lines)
        self.assertIn("stance steady:", lines)
        self.assertIn(f"* {PRIMARY} primary", lines)
        self.assertIn("recent reviews:", lines)
        self.assertIn("director-1 is running", (await execute(s, "/directors spawn")).lines[0])
        self.assertIn("never retires", (await execute(s, f"/directors retire {PRIMARY}")).lines[0])
        self.assertIn("retired", (await execute(s, "/directors retire director-1")).lines[0])
        code, body = await _route(s, "GET", "/directors", {}, {})
        self.assertEqual((code, body["nodes"][0]["id"], body["stance"]["mood"]), (200, PRIMARY, "steady"))
        code, body = await _route(s, "POST", "/directors", {}, {"action": "spawn"})
        self.assertEqual(code, 200)
        code, body = await _route(s, "POST", "/directors", {}, {"action": "retire", "id": body["director"]})
        self.assertEqual((code, body["ok"]), (200, True))
        m = mesh(s)["directors"]
        self.assertEqual((m["nodes"][0]["id"], m["mood"]), (PRIMARY, "steady"))
        self.assertIn("directors:", "\n".join((await execute(s, "/status")).lines))


class InitiativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_mood_and_a_duplicate_shape_its_initiative(self):
        tmp = tempfile.TemporaryDirectory()
        root = make_root(Path(tmp.name))
        shutil.copytree(ROOT / "challenges", root / "challenges")
        over = {**DIRECTORS, "agency": {"enabled": True, "decide_every_seconds": 3600, "idle_seconds": 3600,
                                        "explore": 0.0, "max_decisions_per_hour": 500}}
        s = System(root, overrides=over, seed=8)
        await s.start()
        try:
            d, pool = s.director, s.directors
            s.memory.learn("What is the capital of Zubrowka?", "Lutz is the capital of Zubrowka.", "the operator")
            self.assertTrue(await wait_for(lambda: all(not n.dirty for n in s.memory.nodes.values()), timeout=30))
            self.assertEqual(d.situation()["flags"]["mood"], "steady")
            feel(s, trust=0.2)  # wary: it checks what it holds instead of reaching out
            self.assertEqual(d.situation()["flags"]["mood"], "wary")
            feel(s, **BASE)
            # two things worth doing at once: memory nodes to check and an exam due
            real = d.situation

            def situation():
                sit = real()
                sit["flags"]["memory"] = "failing"
                sit["text"] = situation_text(sit["flags"])
                sit["feasible"] = sit["feasible"] | {"consolidate"}
                return sit
            d.situation = situation
            rec = await d.decide(force=True)
            self.assertEqual((rec["mode"], rec["by"]), ("consolidate", PRIMARY))
            second = d.history[-1]
            self.assertEqual(second["mode"], "examine")
            self.assertNotEqual(second["by"], PRIMARY)
            self.assertEqual(second["how"], f"a duplicate of {PRIMARY}, alongside its consolidate")
            self.assertIsNotNone(second["value"])
            self.assertTrue(any(e.data["reason"].startswith("a second mode of initiative")
                                for e in s.bus.recent("director.spawned", 10)))
            self.assertEqual(pool.counts["initiative"], 2)
        finally:
            await s.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
