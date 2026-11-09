"""More challenges: the catalog of benchmarks, any dataset on the Hugging Face Hub (search, look, choose the columns,
download), your own questions from a file, exporting a suite and adding it to another mesh, and removing one."""
import json
import tempfile
import unittest
from pathlib import Path

from neuromesh.challenges import guess_mapping, parse_items, rows_to_suite
from neuromesh.commands import execute
from neuromesh.runtime import System
from neuromesh.server import _route

from mock_servers import MOODS, MockDatasets
from test_system import FAST, make_root


class ReadingTests(unittest.TestCase):
    def test_guesses_and_reads_columns(self):
        feats = [{"name": "text", "kind": "string", "labels": []}, {"name": "label", "kind": "class", "labels": ["neg", "pos"]}]
        g = guess_mapping(feats, [{"text": "great film", "label": 1}])
        self.assertEqual((g["input"], g["answer"], g["kind"], g["labels"]), ("text", "label", "classify", ["neg", "pos"]))
        s = rows_to_suite([{"text": "great film", "label": 1}, {"text": "dull", "label": 0}], g, "films")
        self.assertEqual([i.answer for i in s.items], ["pos", "neg"])
        self.assertEqual((s.capability, s.labels), ("films", ["neg", "pos"]))
        # numbers written as text are a numeric suite
        feats = [{"name": "problem", "kind": "string", "labels": []}, {"name": "solution", "kind": "string", "labels": []}]
        rows = [{"problem": f"{a} + {a}?", "solution": str(2 * a)} for a in range(25)]
        g = guess_mapping(feats, rows)
        self.assertEqual((g["input"], g["answer"], g["kind"]), ("problem", "solution", "numeric"))
        self.assertEqual(rows_to_suite(rows, g, "sums").items[3].answer, 6.0)
        # options as a list, the answer as an index
        rows = [{"question": "Which is a mammal?", "options": ["shark", "whale"], "answer": 1}]
        s = rows_to_suite(rows, {"input": "question", "answer": "answer", "choices": "options", "kind": "choice"}, "m")
        self.assertEqual((s.items[0].answer, s.items[0].choices), ("B", [["A", "shark"], ["B", "whale"]]))
        with self.assertRaises(ValueError):
            rows_to_suite(rows, {"input": "question", "answer": "", "kind": "short"}, "m")

    def test_own_items_in_three_formats(self):
        jsonl = '{"input": "What is 2+2?", "answer": "4"}\n{"input": "Capital of France?", "answer": "Paris"}\n'
        self.assertEqual(len(parse_items(jsonl)), 2)
        self.assertEqual(parse_items('[{"question": "a?", "answer": "b"}]'), [{"question": "a?", "answer": "b"}])
        self.assertEqual(parse_items("Input,Answer\nWhat is 2+2?,4\n"), [{"input": "What is 2+2?", "answer": "4"}])


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.hub = MockDatasets()
        self.tmp = tempfile.TemporaryDirectory()
        root = make_root(Path(self.tmp.name))
        self.s = System(root, overrides={**FAST, "challenges": {"api_url": self.hub.url, "hub_url": self.hub.url,
                                                                "fetch_limit": 40}}, seed=1)
        await self.s.start()
        self.ch = self.s.challenges

    async def asyncTearDown(self):
        if not self.s.stopping:
            await self.s.stop()
        self.hub.stop()
        self.tmp.cleanup()

    async def test_browse_look_choose_and_download(self):
        code, cat = await _route(self.s, "GET", "/challenges/catalog", {}, {})
        self.assertEqual(code, 200)
        self.assertIn("arc_challenge", [b["name"] for b in cat["benchmarks"]])
        self.assertFalse(any(b["installed"] for b in cat["benchmarks"] if b["name"] == "arc_challenge"))
        code, found = await _route(self.s, "GET", "/challenges/search", {"q": "acme"}, {})
        self.assertEqual([d["id"] for d in found["datasets"]], ["acme/emotion-tiny", "acme/trivia-qa", "acme/odd-columns"])
        self.assertEqual(found["datasets"][0]["tasks"], ["text-classification"])
        code, found = await _route(self.s, "GET", "/challenges/search", {"q": "ai2_arc"}, {})
        self.assertEqual(found["datasets"][0]["adapter"], "arc_easy", "the catalog knows it")

        # look at a dataset: its splits, columns, first rows, and the questions as it would read them
        code, look = await _route(self.s, "GET", "/challenges/inspect", {"dataset": "acme/emotion-tiny"}, {})
        self.assertEqual(code, 200, look)
        self.assertEqual((look["config"], look["split"]), ("default", "test"))
        self.assertEqual(look["mapping"]["kind"], "classify")
        self.assertEqual(look["preview"][0], {"input": "i cried all night after the funeral", "answer": "sadness",
                                              "choices": [], "context": ""})
        asked = len(self.hub.requests)
        # read another way (the same rows: the Hub isn't asked again)
        code, look2 = await _route(self.s, "GET", "/challenges/inspect", {"dataset": "acme/emotion-tiny",
                                   "mapping": json.dumps({"kind": "short"})}, {})
        self.assertEqual(look2["preview"][0]["answer"], ["0"])
        self.assertEqual(len(self.hub.requests), asked)
        code, bad = await _route(self.s, "GET", "/challenges/inspect", {"dataset": "acme/emotion-tiny",
                                 "mapping": json.dumps({"answer": "nope"})}, {})
        self.assertEqual(code, 400)
        self.assertIn("no column 'nope'", bad["message"])

        # download it as a classification suite; it is a new skill to learn
        code, got = await _route(self.s, "POST", "/challenge/fetch", {}, {"dataset": "acme/emotion-tiny", "rows": 30,
                                                                         "name": "Moods", "capability": "mood", "by": "Chris"})
        self.assertEqual((code, got["suite"], got["items"], got["kind"]), (200, "moods", 30, "classify"))
        suite = self.ch.suites["moods"]
        self.assertEqual((suite.capability, suite.labels), ("mood", MOODS))
        self.assertTrue(suite.source.startswith("huggingface:acme/emotion-tiny default/test"))
        rep = await self.ch.run("moods", "practice", 10, by="Chris")
        self.assertIn("mood", self.s.capabilities, "practising a downloaded classification makes its skill")
        self.assertEqual(len(rep.results), 10)

        # columns it has to be told about
        code, look = await _route(self.s, "GET", "/challenges/inspect", {"dataset": "acme/odd-columns"}, {})
        self.assertEqual((look["mapping"]["input"], look["mapping"]["answer"], look["mapping"]["kind"]), ("q_text", "gold", "numeric"))
        code, got = await _route(self.s, "POST", "/challenge/fetch", {}, {"dataset": "acme/odd-columns", "mapping": {
            "input": "q_text", "answer": "gold", "kind": "short", "context": "notes"}})
        self.assertEqual(code, 200, got)
        item = self.ch.suites["odd-columns"].items[0]
        self.assertEqual((item.answer, item.context), (["6"], "x"))

        # a ready-made benchmark through the same door is read the usual way, with its subset
        code, got = await _route(self.s, "POST", "/challenge/fetch", {}, {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge"})
        self.assertEqual((code, got["suite"]), (200, "arc_challenge"))
        self.assertEqual(self.ch.suites["arc_challenge"].items[0].choices[1], ["B", "whale"])
        code, cat = await _route(self.s, "GET", "/challenges/catalog", {}, {})
        self.assertTrue(next(b for b in cat["benchmarks"] if b["name"] == "arc_challenge")["installed"])

        # the Hub is down: said plainly
        self.hub.down = True
        self.ch._looked.clear()
        code, err = await _route(self.s, "GET", "/challenges/inspect", {"dataset": "acme/trivia-qa"}, {})
        self.assertEqual(code, 400)
        self.assertIn("HTTP 503", err["message"])
        self.assertIn("temporarily unavailable", err["message"])

    async def test_own_file_export_import_and_remove(self):
        items = "\n".join(json.dumps({"input": q, "answer": a}) for q, a in [
            ("What is the capital of Peru?", "Lima"), ("Which planet is red?", "Mars"), ("Who wrote Emma?", "Jane Austen"),
            ("What is H2O?", "water")])
        code, got = await _route(self.s, "POST", "/challenges/import", {}, {"name": "My Quiz", "kind": "short", "items": items, "by": "Chris"})
        self.assertEqual((code, got["suite"], got["items"]), (200, "my-quiz", 4))
        code, got = await _route(self.s, "POST", "/challenges/import", {}, {"name": "feelings", "kind": "classify",
                                 "items": "text,label\nso happy today,joy\nawful news,sadness\nyay,joy\n"})
        self.assertEqual(code, 200, got)
        self.assertEqual(self.ch.suites["feelings"].labels, ["joy", "sadness"])
        code, bad = await _route(self.s, "POST", "/challenges/import", {}, {"name": "x", "kind": "short", "items": ""})
        self.assertEqual(code, 400)
        code, bad = await _route(self.s, "POST", "/challenges/import", {}, {"name": "x", "kind": "maths", "items": items})
        self.assertIn("the kind must be one of", bad["message"])

        # exported, then added again (as another mesh would)
        code, exp = await _route(self.s, "GET", "/challenges/export", {"suite": "my-quiz"}, {})
        self.assertEqual(code, 200)
        self.assertIn('kind = "short"', exp["manifest"])
        code, got = await _route(self.s, "POST", "/challenges/import", {}, {"name": "", "items": json.dumps(exp)})
        self.assertEqual((code, got["suite"], got["items"]), (200, "my-quiz", 4), "the same name replaces its own download")
        code, got = await _route(self.s, "POST", "/challenges/import", {}, {"name": "copy of quiz", "items": json.dumps(exp)})
        self.assertEqual(got["suite"], "copy-of-quiz")
        self.assertEqual([i.answer for i in self.ch.suites["copy-of-quiz"].items],
                         [i.answer for i in self.ch.suites["my-quiz"].items])

        # remove: downloaded and added suites only
        st = self.ch.status()
        self.assertTrue(next(x for x in st["suites"] if x["suite"] == "my-quiz")["downloaded"])
        ours = next((x for x in st["suites"] if not x["downloaded"]), None)
        if ours is not None:
            code, res = await _route(self.s, "POST", "/challenges/remove", {}, {"suite": ours["suite"]})
            self.assertEqual(code, 400)
            self.assertIn("challenges folder", res["message"])
        await self.ch.run("my-quiz", "exam")
        code, res = await _route(self.s, "POST", "/challenges/remove", {}, {"suite": "my-quiz", "by": "Chris"})
        self.assertEqual((code, res["ok"]), (200, True))
        self.assertNotIn("my-quiz", self.ch.suites)
        self.assertTrue(any(h["suite"] == "my-quiz" for h in self.ch.history), "its results stay in the history")
        self.assertTrue(any(a["kind"] == "challenge.removed" for a in self.s.store.recent_audit(50)))

    async def test_console(self):
        lines = (await execute(self.s, "/challenge search emotion")).lines
        self.assertIn("acme/emotion-tiny", lines[0])
        lines = (await execute(self.s, "/challenge look acme/trivia-qa")).lines
        self.assertIn("question (string)", lines[0])
        self.assertIn("read as: short - question question, answer answer", lines[1])
        lines = (await execute(self.s, "/challenge fetch acme/trivia-qa 3")).lines
        self.assertIn("fetched trivia-qa: 3 items", lines[0])
        lines = (await execute(self.s, "/challenge remove trivia-qa")).lines
        self.assertIn("removed trivia-qa", lines[0])


if __name__ == "__main__":
    unittest.main()
