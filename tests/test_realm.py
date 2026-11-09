"""Fact and fiction: what the text says, names from works, its realm node; memory marked with it (and the column
added to older memory databases); answers from fiction framed as fiction; "is Gandalf real?"; the ego's part."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from neuromesh.memdb import MemoryDB
from neuromesh.realm import FACT, FICTION, frame, judge, known_work
from neuromesh.runtime import System
from neuromesh.server import _route

from test_system import FAST, make_root, wait_for


class JudgeTests(unittest.TestCase):
    def test_what_the_text_says_and_names_from_works(self):
        self.assertEqual((judge("In my story the dragon Vex lives on the moon").realm, judge("In my story x is y").work),
                         (FICTION, "your story"))
        self.assertEqual(judge("in the novel Dune, Paul rides a sandworm").work, "Dune")
        self.assertEqual(judge("In The Hobbit, Bilbo finds a ring").work, "The Hobbit")
        self.assertEqual(judge("I dreamed I could fly").work, "a dream")
        self.assertEqual(judge("who is Gandalf?").work, "The Lord of the Rings")
        self.assertEqual(judge("who wrote The Lord of the Rings?").realm, FACT, "about the work, not in it")
        self.assertEqual(judge("in real life, are whales mammals?").realm, FACT)
        for real in ("what is the capital of France", "Phoenix is the capital of Arizona", "the force of gravity",
                     "piglets are baby pigs", "Space Shuttle Atlantis launched in 1985", "Dune grass grows on beaches"):
            self.assertIsNone(judge(real), real)
        self.assertEqual(known_work("Kvothe plays the lute", {"Kvothe": "The Kingkiller Chronicle"}),
                         ("kvothe", "The Kingkiller Chronicle"))

    def test_framing(self):
        self.assertEqual(frame("Gandalf is a wizard.", "The Lord of the Rings"), "In The Lord of the Rings, Gandalf is a wizard.")
        self.assertEqual(frame("The ring is destroyed in Mount Doom.", "The Lord of the Rings"),
                         "In The Lord of the Rings, the ring is destroyed in Mount Doom.")
        self.assertEqual(frame("a wizard", "The Lord of the Rings"), "a wizard (in The Lord of the Rings: fiction, not fact).")
        self.assertEqual(frame("Gandalf is a character in Tolkien's novels.", "The Lord of the Rings"),
                         "Gandalf is a character in Tolkien's novels.", "already says so")

    def test_older_memory_databases_gain_the_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            db = sqlite3.connect(path)
            db.executescript("""CREATE TABLE nodes(id TEXT PRIMARY KEY, rev INTEGER NOT NULL DEFAULT 0, summary TEXT NOT NULL DEFAULT '{}',
                                updated REAL NOT NULL DEFAULT 0);
                CREATE TABLE tests(node TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL DEFAULT 'test', ask TEXT NOT NULL,
                                   answer TEXT NOT NULL DEFAULT '', variant_of TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '',
                                   learned TEXT NOT NULL DEFAULT '', by TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
                                   PRIMARY KEY(node, seq));
                CREATE TABLE own(id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, about TEXT NOT NULL DEFAULT '',
                                 ask TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, mood TEXT NOT NULL DEFAULT '',
                                 source TEXT NOT NULL DEFAULT '', node TEXT NOT NULL DEFAULT '');
                INSERT INTO nodes(id, rev) VALUES ('mem-0001', 1);
                INSERT INTO tests(node, seq, ask, answer) VALUES ('mem-0001', 0, 'who is Gandalf', 'a wizard');""")
            db.commit()
            db.close()
            ro = MemoryDB(path, readonly=True)  # read-only: it reads what is there
            self.assertEqual(ro.read("mem-0001")["tests"][0]["realm"], "")
            ro.close()
            rw = MemoryDB(path)
            self.assertIn("realm", rw.fields)
            rw.write("mem-0001", [{"ask": "who is Gandalf", "answer": "a wizard", "realm": "fiction",
                                   "work": "The Lord of the Rings"}], [], {})
            self.assertEqual(rw.read("mem-0001")["tests"][0]["work"], "The Lord of the Rings")
            rw.add_own("dream", "I dreamt of glass cities", realm="fiction")
            self.assertEqual(rw.own(kind="dream")[0]["realm"], "fiction")
            rw.close()


class RealmSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = System(make_root(Path(self.tmp.name)), overrides={
            **FAST, "memory": {"enabled": True, "interval_seconds": 0.05}, "ego": {"enabled": True},
            "realm": {"works": {"Kvothe": "The Kingkiller Chronicle"}}}, seed=1)
        await self.s.start()

    async def asyncTearDown(self):
        if not self.s.stopping:
            await self.s.stop()
        self.tmp.cleanup()

    async def test_memory_answers_and_the_ego(self):
        s = self.s
        self.assertIn("realm", s.capabilities, "the realm node is an internal capability")
        s.memory.learn("who is Gandalf", "a wizard who guides the Fellowship", "the operator", by="Chris", reviewed=True)
        s.memory.learn("what is the capital of Peru", "Lima", "the operator", by="Chris", reviewed=True)
        s.memory.learn("what instrument does Kvothe play", "the lute", "the operator", by="Chris", reviewed=True)
        facts = {t.ask: t for n in s.memory.fact_nodes() for t in n.facts()}
        self.assertEqual((facts["who is Gandalf"].realm, facts["who is Gandalf"].work), (FICTION, "The Lord of the Rings"))
        self.assertEqual((facts["what is the capital of Peru"].realm, facts["what is the capital of Peru"].work), (FACT, ""))
        self.assertEqual(facts["what instrument does Kvothe play"].work, "The Kingkiller Chronicle", "[realm.works]")
        row = s.memory.db.read(s.memory.fact_nodes()[0].id)["tests"][0]
        self.assertEqual(row["realm"], FICTION, "kept in the memory database")
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.fact_nodes()), 60))

        r = await s.submit("who is Gandalf?")
        self.assertEqual(r.realm["realm"], FICTION)
        self.assertTrue(r.answer.startswith("a wizard who guides the Fellowship (in The Lord of the Rings: fiction, not fact)."),
                        r.answer)
        self.assertTrue(any(t.startswith("ego (the outer layer)") and "as fiction, from The Lord of the Rings" in t for t in r.trace),
                        r.trace)
        r = await s.submit("what is the capital of Peru?")
        self.assertEqual(r.realm["realm"], FACT)
        self.assertTrue(r.answer.startswith("Lima"))
        r = await s.submit("is Gandalf real?")
        self.assertEqual(r.route, "realm")
        self.assertTrue(r.answer.startswith("No - Gandalf is from The Lord of the Rings: fiction, not fact."), r.answer)
        r = await s.submit("did Sherlock Holmes really exist?")
        self.assertIn("fiction, not fact", r.answer)
        code, st = await _route(s, "GET", "/realm", {"text": "In my story, the sea is purple"}, {})
        self.assertEqual(st["judged"]["realm"], FICTION)
        self.assertGreaterEqual(st["counts"][FICTION], 1)

    async def test_its_own_dreams_are_fiction(self):
        kept = self.s.memory.remember_own("dream", "I dreamt the library flooded with starlight", about="libraries")
        self.assertIsNotNone(kept)
        self.assertEqual(self.s.memory.own_statements(kind="dream")[0]["realm"], FICTION)


if __name__ == "__main__":
    unittest.main()
