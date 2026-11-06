"""Memory nodes: learned facts compressed into small neural networks, external human-readable test
files, and compression that only ever accepts a smaller node that passes every test (P11)."""
import random
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from neuromesh.commands import execute
from neuromesh.config import Directives
from neuromesh.memory import (UNKNOWN, MemoryBank, check_files, encode_many, fold, forward, init_params, key,
                              match_depth, near_misses, quantize, size_bytes, stack, subject, unfold_params, variants)
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import Order

from test_system import FAST, make_root, wait_for

FACTS = [("What is the capital of France?", "Paris is the capital and largest city of France."),
         ("Who was Ada Lovelace?", "Ada Lovelace was an English mathematician and writer."),
         ("When did Ada Lovelace die?", "Ada Lovelace died on 27 November 1852."),
         ("How tall is the Eiffel Tower?", "The Eiffel Tower is 330 metres tall."),
         ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
         ("Who won the 1990 World Cup?", "West Germany won the 1990 World Cup."),
         ("Who won the 1994 World Cup?", "Brazil won the 1994 World Cup."),
         ("What is the capital of Japan?", "Tokyo is the capital of Japan.")]

MEMORY = {**FAST, "memory": {"enabled": True, "interval_seconds": 0.02, "sync_seconds": 0.05}}


class Bus:
    def __init__(self):
        self.events = []

    def publish(self, topic, **data):
        self.events.append((topic, data))


def bank_at(root: Path, **pol) -> MemoryBank:
    d = Directives()
    for k, v in pol.items():
        setattr(d.memory, k, v)
    system = NS(directives=d, state_dir=root / "state", root=root, rng=random.Random(1), bus=Bus(),
                guardian=NS(audit=lambda *a, **k: None), store=NS(examples=lambda *a: []), tools=NS(tools={}))
    return MemoryBank(system)


def settle(bank: MemoryBank, limit: int = 80, check=None) -> int:
    """Run the compressor's jobs in-line until there is nothing left to do."""
    steps = 0
    while steps < limit and (job := bank.next_job()) is not None:
        bank.finish(job, job.run())
        steps += 1
        if check:
            check(bank.nodes[job.node_id])
    return steps


class QuestionTests(unittest.TestCase):
    def test_subjects_keys_and_variants(self):
        self.assertEqual(subject("What is the capital of France?"), "capital of France")
        self.assertEqual(subject("Tell me about Ada Lovelace"), "Ada Lovelace")
        self.assertEqual(subject("How tall is the Eiffel Tower?"), "")  # "how tall" carries meaning
        self.assertEqual(key("Who was Ada Lovelace?"), key("tell me about ada lovelace"))  # one fact
        self.assertEqual(key("What's the capital of france"), key("What is the capital of France?"))
        self.assertNotEqual(key("How tall is the Eiffel Tower?"), key("Tell me about the Eiffel Tower"))
        self.assertEqual(variants("What is the capital of France?"), ["What is the capital of France?", "capital of France"])
        self.assertEqual(variants("When did Ada Lovelace die?"), ["When did Ada Lovelace die?"])
        misses = near_misses("What is the capital of France?", 2)
        self.assertTrue(misses and all("France" not in m or "capital" not in m for m in misses), misses)

    def test_weights_storage_is_exact(self):
        rng = np.random.default_rng(0)
        p = init_params(rng, 64, (16,), 5)
        for bits in (16, 8):
            q = quantize(p, bits)
            for a, b in zip(q, quantize(q, bits)):  # idempotent: what was tested is what is stored
                np.testing.assert_array_equal(a, b)
        self.assertLess(size_bytes(p, 8), size_bytes(p, 16) < size_bytes(p, 32) and size_bytes(p, 16))
        X = encode_many(["what is the capital of france", "hello"], 256)
        small = init_params(rng, 64, (8,), 3)
        np.testing.assert_allclose(forward(unfold_params(small, 256), fold(X, 256))[0], forward(small, fold(X, 64))[0],
                                   rtol=1e-5, atol=1e-5)  # more inputs, same function


class BankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bank = bank_at(self.root)
        for q, a in FACTS:
            self.assertEqual(self.bank.learn(q, a, "a test", "me")[1], "new")

    def tearDown(self):
        self.tmp.cleanup()

    def node(self):
        return next(iter(self.bank.nodes.values()))

    def test_learn_write_tests_then_compress_only_while_every_test_passes(self):
        b = self.bank
        node = self.node()
        data = tomllib.loads(b.tests_path(node).read_text())  # external, human readable, valid TOML
        self.assertEqual(data["passing"], "not checked yet")
        asks = {t["ask"]: t["answer"] for t in data["test"]}
        self.assertEqual(asks["capital of France"], FACTS[0][1])
        self.assertTrue(any(u["reason"].startswith("auto: a near miss") for u in data["unknown"]))

        def every_test_passes(n):
            r = b.run_tests(n)
            self.assertEqual(r["passed"], r["total"], r["failing"])
        settle(b, check=every_test_passes)
        node = self.node()
        self.assertEqual(node.status, "compressed")
        self.assertLess(node.bytes * 20, node.first_bytes, node.history)
        events = [t for t, _ in b.sys.bus.events]
        self.assertIn("memory.learned", events)
        self.assertIn("memory.compressed", events)
        self.assertIn("memory.smallest", events)
        data = tomllib.loads(b.tests_path(node).read_text())
        self.assertTrue(data["passing"].startswith(f"{node.total}/{node.total}"))
        self.assertIn(f"{node.bytes:,} bytes", data["size"])
        self.assertTrue(any("compressed" in h for h in data["history"]))
        # what it learned answers questions, even asked another way; what it did not learn it leaves alone
        self.assertEqual(b.recall("what's the capital of France").answer, FACTS[0][1])
        self.assertEqual(b.recall("how tall is the eiffel tower").answer, FACTS[3][1])
        self.assertIsNone(b.recall("What is the capital of Germany?"))
        self.assertIsNone(b.recall("what is 12 * 7"))
        self.assertEqual(b.recall("who was Ada Lovelace?").source, "a test")

    def test_a_smaller_node_that_fails_a_test_is_refused(self):
        b = self.bank
        settle(b, limit=1)  # learned, not compressed yet
        node = self.node()
        before = [p.copy() for p in node.params], node.bytes
        job = b.next_job()
        self.assertEqual(job.kind, "compress")
        res = job.run()
        res["ok"], res["passed"], res["failing"] = False, res["total"] - 1, [0]  # one answer no longer matches
        b.finish(job, res)
        self.assertEqual(node.bytes, before[1])
        for a, c in zip(node.params, before[0]):
            np.testing.assert_array_equal(a, c)
        self.assertIn(job.step[0], node.tried)
        topic, data = b.sys.bus.events[-1]
        self.assertEqual((topic, data["failed"]), ("memory.rejected", 1))

    def test_people_edit_the_tests_and_the_node_follows(self):
        b = self.bank
        settle(b)
        node = self.node()
        path = b.tests_path(node)
        text = path.read_text().replace('answer = "Tokyo is the capital of Japan."', 'answer = "Tokyo (東京) is Japan\'s capital."')
        text = text.replace("[[test]]\nask = \"How tall is the Eiffel Tower?\"", "[[test]]\nask = \"How tall is the Eiffel Tower?\"", 1)
        path.write_text(text)
        b.sync_files(force=True)
        self.assertTrue(node.dirty)
        self.assertIsNone(b.recall("capital of Japan"), "the old answer is not served once a person changed it")
        settle(b)
        self.assertEqual(b.recall("What is the capital of Japan?").answer, "Tokyo (東京) is Japan's capital.")
        # deleting a test forgets that fact
        data = tomllib.loads(path.read_text())
        kept = [t for t in data["test"] if "Eiffel" not in t["ask"]]
        lines = ['node = "mem-0001"'] + [f'\n[[test]]\nask = "{t["ask"]}"\nanswer = "{t["answer"]}"' for t in kept]
        path.write_text("\n".join(lines) + "\n")
        b.sync_files(force=True)
        settle(b)
        self.assertNotIn("The Eiffel Tower is 330 metres tall.", node.answers)
        self.assertIsNone(b.recall("How tall is the Eiffel Tower?"))
        self.assertEqual(b.recall("Who won the 1994 World Cup?").answer, "Brazil won the 1994 World Cup.")
        # a test file a person writes is learned
        (b.tests_dir / "my-facts.toml").write_text('[[test]]\nask = "What is the name of my dog?"\nanswer = "Your dog is called Rex."\n')
        b.sync_files(force=True)
        settle(b)
        self.assertEqual(b.recall("What is the name of my dog?").answer, "Your dog is called Rex.")
        # deleting a file forgets the node; its weights are kept aside, not destroyed
        path.unlink()
        b.sync_files(force=True)
        self.assertNotIn("mem-0001", b.nodes)
        self.assertTrue(list((b.dir / "forgotten").glob("mem-0001-*.npz")))
        self.assertIsNone(b.recall("Who won the 1994 World Cup?"))

    def test_corrections_persistence_and_the_outside_check(self):
        b = self.bank
        self.assertEqual(b.learn("What's the capital of france", "Paris.", "a correction", "me")[1], "updated")
        self.assertEqual(b.learn("What is the capital of France?", "Paris.", "again", "me")[1], "known")
        self.assertIsNone(b.learn("what is 12 * 7", "84", "x", "me"))  # a tool's question, not a fact
        settle(b)
        node = self.node()
        again = bank_at(self.root)
        same = again.nodes[node.id]
        self.assertEqual((same.bytes, same.spec(), same.dirty), (node.bytes, node.spec(), False))
        self.assertEqual(again.recall("capital of France").answer, "Paris.")
        shutil.copy(Path(__file__).resolve().parent.parent / "directives.toml", self.root)
        lines = []
        self.assertEqual(check_files(self.root, out=lines.append), 0, lines)
        self.assertIn("pass", lines[0])
        path = b.tests_path(node)
        edited = path.read_text().replace('answer = "Paris."', 'answer = "Lyon."')
        path.write_text(edited)
        lines = []
        self.assertGreater(check_files(self.root, out=lines.append), 0)
        self.assertTrue(any("wanted 'Lyon.'" in ln for ln in lines), lines)
        self.assertEqual(path.read_text(), edited, "checking from outside changes nothing")


MORE = [("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
        ("What is the largest planet?", "Jupiter is the largest planet."),
        ("How many bones are in the human body?", "An adult has 206 bones."),
        ("What is the speed of light?", "Light travels at about 300,000 km per second.")]


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bank = bank_at(self.root, facts_per_node=4, merge=False)
        for q, a in FACTS + MORE:
            self.bank.learn(q, a, "a test", "me")
        settle(self.bank)  # learned and compressed, three nodes of four facts

    def tearDown(self):
        self.tmp.cleanup()

    def test_stacking_keeps_every_answer_exact(self):
        rng = np.random.default_rng(3)
        X = encode_many(["what is the capital of france", "hello", "who was ada lovelace"], 1024)
        pa, pb = init_params(rng, 64, (), 4), init_params(rng, 128, (16, 8), 3)  # linear vs two hidden layers
        aa, ab = [UNKNOWN, "a1", "a2", "shared"], [UNKNOWN, "b1", "shared"]
        answers = [UNKNOWN, "a1", "a2", "shared", "b1"]
        ua = unfold_params(pa, 128)
        merged = stack(match_depth(ua, 2, rng), aa, match_depth(pb, 2, rng), ab, answers)
        za, zb = forward(pa, fold(X, 64))[0], forward(pb, fold(X, 128))[0]
        z = forward(merged, fold(X, 128))[0]
        np.testing.assert_allclose(z[:, 1:3], za[:, 1:3], rtol=1e-4, atol=1e-4)    # A's answers, exactly
        np.testing.assert_allclose(z[:, 4], zb[:, 1], rtol=1e-4, atol=1e-4)        # B's answer, exactly
        np.testing.assert_allclose(z[:, 0], za[:, 0] + zb[:, 0], rtol=1e-4, atol=1e-4)  # unknown: both say so
        np.testing.assert_allclose(z[:, 3], za[:, 3] + zb[:, 2], rtol=1e-4, atol=1e-4)
        lin = stack(pa, aa, init_params(rng, 64, (), 3), ab, answers)
        self.assertEqual(len(lin), 2)  # two linear nodes stay one linear node

    def test_merge_replaces_both_and_merged_nodes_merge_again(self):
        b = self.bank
        self.assertEqual(len(b.nodes), 3)
        self.assertTrue(all(n.status == "compressed" for n in b.nodes.values()))
        b.pol.merge, b.pol.merge_every_seconds = True, 0
        settle(b)
        self.assertEqual(len(b.nodes), 1)
        node = next(iter(b.nodes.values()))
        self.assertEqual(len(node.facts()), 12)
        self.assertEqual(sorted(node.parts), ["mem-0001", "mem-0002", "mem-0003"])
        self.assertEqual(node.status, "compressed")  # compressed again after merging
        self.assertEqual((node.passed, node.dirty), (node.total, False))
        merged = [d for t, d in b.sys.bus.events if t == "memory.merged"]
        self.assertEqual(len(merged), 2)  # the second merge combined a merged node with another
        self.assertTrue(all(d["pos"] / d["pos_total"] >= 0.9 and d["neg"] / d["neg_total"] >= 0.9 for d in merged))
        self.assertEqual(merged[1]["parts"], 3)
        for q, a in FACTS + MORE:
            self.assertEqual(b.recall(q).answer, a, q)
        self.assertIsNone(b.recall("What is the capital of Germany?"))
        # the parents are gone: weights kept aside, test files folded into the merged node's
        self.assertEqual([p.name for p in b.tests_dir.glob("*.toml")], [f"{node.id}.toml"])
        self.assertEqual(len(list((b.dir / "merged").glob("*-into-*.npz"))), 4)
        data = tomllib.loads(b.tests_path(node).read_text())
        self.assertEqual(len([t for t in data["test"] if "variant_of" not in t]), 12)
        self.assertTrue(any("merged" in h for h in data["history"]))
        again = bank_at(self.root)
        self.assertEqual(again.nodes[node.id].parts, node.parts)  # persists

    def test_a_merge_below_90_percent_is_refused(self):
        b = self.bank
        a1, a2 = sorted(b.nodes.values(), key=lambda n: n.id)[:2]
        before = {n.id: n.bytes for n in b.nodes.values()}
        job = b.merge_job(a1, a2)
        res = job.run()
        res["pos"] = int(res["pos_total"] * 0.85)  # say only 85% of the answers matched
        b.finish(job, res)
        self.assertEqual({n.id: n.bytes for n in b.nodes.values()}, before)
        self.assertEqual(b.sys.bus.events[-1][0], "memory.merge_refused")
        self.assertNotIn((a1, a2), [(x, y) for x, y in b.merge_candidates()])  # not retried as they are

    def test_a_merged_node_that_fails_a_few_tests_relearns_them(self):
        b = self.bank
        a1, a2 = sorted(b.nodes.values(), key=lambda n: n.id)[:2]
        job = b.merge_job(a1, a2)
        res = job.run()
        failing = job.test_texts.index("Who was Ada Lovelace?")
        res["failing"], res["passed"], res["pos"], res["ok"] = [failing], res["passed"] - 1, res["pos"] - 1, False
        b.finish(job, res)  # 15/16 answers: accepted at >= 90%
        node = next(n for n in b.nodes.values() if n.merged_from)
        self.assertEqual((node.status, node.dirty, node.failing), ("failing", True, ["Who was Ada Lovelace?"]))
        self.assertIsNone(b.recall("Who was Ada Lovelace?"), "an answer whose test fails is not served")
        self.assertIn("failing: Who was Ada Lovelace?", tomllib.loads(b.tests_path(node).read_text())["passing"])
        settle(b)
        self.assertEqual((node.passed, node.failing), (node.total, []))
        self.assertEqual(b.recall("Who was Ada Lovelace?").answer, FACTS[1][1])


class MemorySystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=MEMORY, seed=3)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    def settled(self):
        m = self.system.memory
        return m.nodes and all(not n.dirty and n.status == "compressed" for n in m.nodes.values())

    async def test_memorize_compress_and_answer_from_the_node(self):
        s = self.system
        for q, a in FACTS[:5]:
            out = (await execute(s, f"/memorize {q} :: {a}")).lines[0]
            self.assertIn("memory node mem-0001", out)
        self.assertTrue(await wait_for(self.settled, timeout=30), s.memory.status())
        node = s.memory.nodes["mem-0001"]
        res = await s.submit("what's the capital of France?")
        self.assertEqual((res.knowledge, res.output), ("memory", FACTS[0][1]))
        self.assertIn(res.served_by, ("nodes", "mixed"))  # mixed while teachers still do the routing
        first, source = res.answer.split("\n")
        self.assertEqual(first, FACTS[0][1])  # the answer first, then where it came from
        self.assertTrue(source.startswith(f"Source: memory node mem-0001 ({node.bytes:,} bytes, from "), source)
        self.assertIn("tests pass), learned from the operator", source)
        self.assertEqual(s.memory.status()["recalled"], 1)
        lines = "\n".join((await execute(s, "/memory")).lines)
        self.assertIn("mem-0001 [compressed] 5 facts", lines)
        lines = "\n".join((await execute(s, "/memory test")).lines)
        self.assertIn("pass", lines)
        lines = "\n".join((await execute(s, "/memory mem-0001")).lines)
        self.assertIn("compressed:", lines)
        code, body = await _route(s, "GET", "/memory", {}, {})
        self.assertEqual((code, body["facts"]), (200, 5))
        code, body = await _route(s, "POST", "/memorize", {}, {"ask": "Who painted the Mona Lisa?",
                                                                 "answer": "Leonardo da Vinci painted the Mona Lisa."})
        self.assertEqual((code, body["status"]), (200, "new"))
        code, body = await _route(s, "POST", "/memory/test", {}, {})
        self.assertEqual(code, 200)
        self.assertIn("memory.compressed", {a["kind"] for a in s.store.recent_audit(200)})
        lines = "\n".join((await execute(s, "/memory forget Who painted the Mona Lisa?")).lines)
        self.assertIn("forgot", lines)
        self.assertIn("no pair to try", (await execute(s, "/memory merge")).lines[0])  # only one node

    async def test_what_teachers_and_research_say_is_learned_but_not_exams(self):
        s = self.system
        asked = []

        async def answer(text):
            asked.append(text)
            return "Sure! The sky looks blue because air scatters blue light the most.", "ollama"
        s.teachers.answer = answer
        res = await s.submit("why is the sky blue?")
        self.assertEqual(res.served_by, "teachers")
        await s.orchestrator.fulfil(Order(text="why is grass green?", source="challenge:x", background=True, learn=False))
        self.assertTrue(await wait_for(self.settled, timeout=30), s.memory.status())
        facts = [t.ask for n in s.memory.nodes.values() for t in n.facts()]
        self.assertEqual(facts, ["why is the sky blue?"])  # the exam question was not learned
        res = await s.submit("why is the sky blue")
        self.assertEqual(len(asked), 2, "the second time, the memory node answered")
        self.assertEqual(res.output, "The sky looks blue because air scatters blue light the most.")
        self.assertIn("learned from the ollama teacher", res.answer)
        # a retained article is compressed too
        doc = s.knowledge.add({"source": "kiwix", "title": "Ada Lovelace", "url": "http://kiwix/Ada", "summary":
                               "Augusta Ada King, Countess of Lovelace, was an English mathematician and writer.",
                               "passages": [], "links": [], "topic": "Ada Lovelace"}, "test")
        s.memory.learn_article(doc)
        self.assertTrue(await wait_for(self.settled, timeout=30))
        res = await s.submit("tell me about Ada Lovelace")
        self.assertEqual(res.knowledge, "memory")
        self.assertIn("English mathematician", res.output)
        self.assertIn("kiwix 'Ada Lovelace' http://kiwix/Ada", res.answer)

    async def test_people_can_tell_it_the_answer(self):
        s, m = self.system, self.system.mind
        m.pol.max_inquiries_per_hour = 50
        res = await s.submit("what is the name of my cat?")
        self.assertFalse(res.ok)
        self.assertIn("tell me the answer", res.inquiry["question"])
        ok, out = await m.reply(res.inquiry["id"], "answer: Your cat is called Miso.")
        self.assertTrue(ok, out)
        self.assertIn("memory node mem-0001", out[0])
        self.assertTrue(await wait_for(self.settled, timeout=30))
        res = await s.submit("what is the name of my cat?")
        self.assertEqual((res.ok, res.output), (True, "Your cat is called Miso."))
        self.assertIn("learned from you told me", res.answer)


if __name__ == "__main__":
    unittest.main()
