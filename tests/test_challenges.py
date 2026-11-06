"""Open-source LLM challenges as a teacher: grading, fetching, practice that teaches, exams that don't."""
import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from neuromesh.challenges import ADAPTERS, Item, Suite, convert, fetch_rows, grade, load_suite, prompt_for
from neuromesh.commands import execute
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.tasks import DIRECT, DISPATCH, Inference

from test_system import FAST, make_root, wait_for

ROOT = Path(__file__).resolve().parent.parent
TEACHING = {**FAST, "teachers": {"order": ["human", "challenge", "heuristic"]},
            "tasks": {"min_interval_seconds": 0.1}}


def suite(kind, **kw):
    return Suite("t", kind, **kw)


class GradingTests(unittest.TestCase):
    def test_numeric(self):
        s, it = suite("numeric"), Item("1", "q", 60)
        self.assertEqual(grade(s, it, True, 60, "5 * 12 = 60.").verdict, "pass")
        self.assertEqual(grade(s, it, True, "x", "The answer is 60 eggs.").verdict, "pass")
        self.assertEqual(grade(s, it, True, "x", "It costs $1,060.").verdict, "fail")
        self.assertEqual(grade(s, it, True, "x", "There are 60 eggs in 12 boxes, so 5.").verdict, "close")
        self.assertEqual(grade(s, it, True, "x", "The answer is 60.4.").verdict, "close")  # within 1%
        self.assertEqual(grade(s, it, False, None, "I couldn't do that").verdict, "fail")

    def test_choice_yesno_short_classify(self):
        s = suite("choice")
        it = Item("1", "q", "C", choices=[["A", "coal"], ["B", "oil"], ["C", "wind"], ["D", "gas"]])
        self.assertEqual(grade(s, it, True, "", "C) wind").verdict, "pass")
        self.assertEqual(grade(s, it, True, "", "The answer is C.").verdict, "pass")
        self.assertEqual(grade(s, it, True, "", "Wind is renewable.").verdict, "pass")
        self.assertEqual(grade(s, it, True, "", "Either wind or coal.").verdict, "close")
        self.assertEqual(grade(s, it, True, "", "B").verdict, "fail")
        y = suite("yesno")
        self.assertEqual(grade(y, Item("1", "q", False), True, "", "No, penguins cannot fly.").verdict, "pass")
        self.assertEqual(grade(y, Item("1", "q", True), True, "", "No.").verdict, "fail")
        sh = suite("short")
        it = Item("1", "q", ["Leonardo da Vinci", "Leonardo"])
        self.assertEqual(grade(sh, it, True, "", "Leonardo da Vinci painted it.").verdict, "pass")
        self.assertEqual(grade(sh, it, True, "", "It was da Vinci.").verdict, "close")
        self.assertEqual(grade(sh, it, True, "", "Michelangelo.").verdict, "fail")
        c = suite("classify", capability="sentiment", labels=["negative", "positive"], close=[["neutral", "positive"]])
        self.assertEqual(grade(c, Item("1", "q", "positive"), True, "positive", "That reads as positive.").verdict, "pass")
        self.assertEqual(grade(c, Item("1", "q", "positive"), True, "neutral", "That reads as neutral.").verdict, "close")
        self.assertEqual(grade(c, Item("1", "q", "positive"), True, "negative", "").verdict, "fail")

    def test_prompts_and_adapters(self):
        it = Item("1", "is the sky blue", True, context="The sky looks blue.")
        self.assertEqual(prompt_for(suite("yesno"), it), "Is the sky blue?\nPassage: The sky looks blue.\nAnswer yes or no.")
        rows = {"gsm8k": {"question": "Janet has 3 eggs...", "answer": "3 - 1 = <<3-1=2>>2\n#### 1,200"},
                "sst2": {"idx": 0, "sentence": "it 's a charming journey . ", "label": 1},
                "ag_news": {"text": "Stocks rally", "label": 2},
                "boolq": {"question": "is it", "answer": False, "passage": "p"},
                "arc_easy": {"id": "M1", "question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
                             "answerKey": "B"},
                "commonsense_qa": {"id": "x", "question": "q", "question_concept": "c",
                                   "choices": {"label": ["A", "B"], "text": ["bank", "mall"]}, "answerKey": "A"},
                "squad": {"id": "s1", "title": "t", "context": "ctx", "question": "q",
                          "answers": {"text": ["Denver Broncos", "Denver Broncos"], "answer_start": [1, 1]}}}
        got = {name: convert(name, [row]).items[0] for name, row in rows.items()}
        self.assertEqual(got["gsm8k"].answer, 1200)
        self.assertEqual((got["sst2"].input, got["sst2"].answer), ("it 's a charming journey .", "positive"))
        self.assertEqual(got["ag_news"].answer, "business")
        self.assertEqual((got["boolq"].answer, got["boolq"].context), (False, "p"))
        self.assertEqual((got["arc_easy"].id, got["arc_easy"].choices, got["arc_easy"].answer), ("M1", [["A", "a"], ["B", "b"]], "B"))
        self.assertEqual(got["commonsense_qa"].choices[0], ["A", "bank"])
        self.assertEqual(got["squad"].answer, ["Denver Broncos"])
        self.assertEqual(convert("sst2", [{"sentence": "hidden", "label": -1}]).items[0].answer, "positive")  # -1 wraps
        self.assertEqual(set(ADAPTERS), set(rows) | {"arc_challenge"})

    def test_sample_suites_and_stable_split(self):
        for folder in (ROOT / "challenges").iterdir():
            if folder.is_dir():
                s = load_suite(folder)
                self.assertTrue(s.exam() and s.practice(), s.name)
                self.assertEqual({i.id for i in s.exam()}, {i.id for i in load_suite(folder).exam()})
                if s.kind == "classify":  # every label is examined
                    self.assertEqual({i.answer for i in s.exam()}, set(s.labels))


class MockHub(BaseHTTPRequestHandler):
    requests: list = []
    ROWS = [{"question": f"What is {i} plus {i}?", "answer": f"{i}+{i}=<<{i}+{i}={2 * i}>>{2 * i}\n#### {2 * i}"}
            for i in range(1, 151)]

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        MockHub.requests.append(q)
        if url.path != "/rows" or q.get("dataset") != "openai/gsm8k":
            self.send_response(404)
            self.end_headers()
            return
        off, n = int(q["offset"]), int(q["length"])
        body = json.dumps({"features": [], "rows": [{"row_idx": off + i, "row": r, "truncated_cells": []}
                                                    for i, r in enumerate(self.ROWS[off: off + n])],
                           "num_rows_total": len(self.ROWS), "num_rows_per_page": 100, "partial": False}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)


class FetchTests(unittest.TestCase):
    def test_paged_fetch(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        api = f"http://127.0.0.1:{server.server_port}"
        try:
            MockHub.requests.clear()
            rows = fetch_rows("openai/gsm8k", "main", "test", 130, api=api)
            self.assertEqual(len(rows), 130)
            self.assertEqual([(r["offset"], r["length"]) for r in MockHub.requests], [("0", "100"), ("100", "30")])
            rows = fetch_rows("openai/gsm8k", "main", "test", 1000, api=api)
            self.assertEqual(len(rows), 150, "stops at the end of the split")
            with self.assertRaises(RuntimeError):
                fetch_rows("nope/nope", "x", "y", 5, api=api)
        finally:
            server.shutdown()
            server.server_close()


class ChallengeSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        shutil.copytree(ROOT / "challenges", self.root / "challenges")
        self.system = System(self.root, overrides=TEACHING, seed=1)
        await self.system.start()
        self.ch = self.system.challenges

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_exams_measure_and_never_teach(self):
        s, suite = self.system, self.ch.suites["sample-sentiment"]
        before = {e.text for e in s.store.examples("sentiment")}
        rep = await self.ch.run("sample-sentiment", "exam")
        self.assertEqual(len(rep.results), len(suite.exam()))
        self.assertIn("close", rep.counts(), "neutral for a clear review is half right")
        exam_texts = {i.input for i in suite.exam()}
        self.assertFalse(exam_texts & ({e.text for e in s.store.examples("sentiment")} - before))
        spec = s.capabilities["sentiment"]
        teacher = s.teachers.teachers["challenge"]
        for item in suite.exam():
            self.assertIsNone(await teacher.label(spec, item.input), "the teacher never knows exam answers")
        practice = suite.practice()[0]
        lesson = await teacher.label(spec, practice.input)
        self.assertEqual((lesson.label, lesson.source), (practice.answer, "challenge"))
        self.assertTrue(set(await teacher.generate(spec, "negative", 50)) <= {i.input for i in suite.practice()})

    async def test_practice_grows_a_new_skill_until_the_exam_improves(self):
        s = self.system
        first = await self.ch.run("sample-topics", "exam")  # creates the 'topic' skill; nothing knows it yet
        self.assertIn("topic", s.capabilities)
        self.assertEqual(first.score, 0.0)
        rep = await self.ch.run("sample-topics", "practice", 100)
        self.assertEqual(rep.learned, len(self.ch.suites["sample-topics"].practice()))
        self.assertEqual(s.store.label_counts("topic"), {"world": 7, "sports": 7, "business": 7, "sci_tech": 7})
        scores = []

        async def improved():
            scores.append((await self.ch.run("sample-topics", "exam")).score)
            return scores[-1] >= 0.5
        for _ in range(30):
            if await improved() or (len(scores) > 3 and scores[-1] > first.score):
                break
            await wait_for(lambda: False, timeout=1)
        self.assertGreater(scores[-1], first.score, scores)  # 28 practice headlines is little to go on
        lines = (await execute(s, "/challenges")).lines
        self.assertTrue(any("sample-topics" in ln and "first 0%" in ln for ln in lines), lines)

    async def test_missed_questions_teach_the_router_a_better_route(self):
        s = self.system
        pick = next(i.input for i in self.ch.suites["sample-arithmetic"].practice() if "squared" in i.input
                    or "*" in i.input or " times " in i.input)
        real = s.infer

        async def misroute(cap, text, **kw):  # the router sends this one to the wrong place
            if cap == DISPATCH and text == pick:
                return Inference(DIRECT, 0.9, "teacher:heuristic", None, "teacher")
            return await real(cap, text, **kw)
        s.infer = misroute
        # a director deems it unanswered and tries the calculator; the router learns that
        res = await s.submit(pick)
        self.assertEqual((res.ok, res.route, res.attempts), (True, "tool:calculator", 2), res.trace)
        rows = s.store.db.execute("SELECT label, source FROM examples WHERE capability=? AND text=?", (DISPATCH, pick)).fetchall()
        self.assertIn(("tool:calculator", "director"), rows)
        # with no retries, practice finds the better route itself
        s.directors.pol.max_attempts = 1
        s.store.db.execute("DELETE FROM examples WHERE capability=? AND text=?", (DISPATCH, pick))
        rep = await self.ch.run("sample-arithmetic", "practice", 100)
        rerouted = [r for r in rep.results if "->" in r["route"]]
        self.assertTrue(rerouted and rep.rerouted == len(rerouted), rep.results)
        self.assertTrue(all(r["route"].endswith("tool:calculator (1)") for r in rerouted), rerouted)
        prompt = next(r["input"] for r in rerouted)
        self.assertTrue(any(e.text == prompt and e.label == "tool:calculator" and e.source == "challenge"
                            for e in s.store.examples(DISPATCH)))
        s.infer = real
        s.directors.pol.max_attempts = 3
        res = await s.submit(prompt)  # the router now sends it to the calculator
        self.assertTrue(res.ok and res.route == "tool:calculator", (res.route, res.answer))
        # an LLM teacher that answers word problems: the route search finds it and the router learns it
        wanted = {i.input: i.answer for i in self.ch.suites["sample-arithmetic"].items}

        async def answer(text):
            q = text.split("\n")[0]
            return (f"The answer is {wanted[q]}.", "ollama") if q in wanted else None
        s.teachers.answer = answer
        rep = await self.ch.run("sample-arithmetic", "practice", 100)
        self.assertGreaterEqual(rep.score, 0.9, rep.results)
        problem = "A box holds 12 eggs. How many eggs are in 5 boxes?"
        self.assertTrue(any(e.text == problem and e.label == DIRECT and e.source in ("challenge", "director")
                            for e in s.store.examples(DISPATCH)), "word problems are routed to the teacher that solves them")

    async def test_retained_answers_and_background_task(self):
        s = self.system
        rep = await self.ch.run("sample-trivia", "practice", 100)
        self.assertEqual(rep.learned, len(self.ch.suites["sample-trivia"].practice()))
        asked = next(i for i in self.ch.suites["sample-trivia"].practice() if i.answer == ["Canberra"]) \
            if any(i.answer == ["Canberra"] for i in self.ch.suites["sample-trivia"].practice()) else \
            self.ch.suites["sample-trivia"].practice()[0]
        res = await s.submit(asked.input)
        self.assertTrue(res.answer.startswith(asked.answer[0]), res.answer)  # recalled from what practice retained
        self.assertEqual(res.knowledge, "recalled")
        t = s.tasks.add("challenge sample-sentiment every 0.3s practice 10")
        self.assertTrue(await wait_for(lambda: t.runs >= 2 and "exam" in t.progress, timeout=30), t.note)
        self.assertTrue(any(n.text.startswith("sample-sentiment exam:") for n in s.mind.notes.values()
                            if n.kind == "update"))
        code, body = await _route(s, "GET", "/challenges", {}, {})
        self.assertTrue(any(r["suite"] == "sample-sentiment" and r["exams"] >= 2 for r in body["suites"]))
        code, body = await _route(s, "POST", "/challenge", {}, {"suite": "sample-yesno", "mode": "exam"})
        self.assertEqual((code, len(body["results"])), (200, 3))
        lines = (await execute(s, "/challenge run sample-science")).lines
        self.assertTrue(lines[0].startswith("  sample-science exam: "), lines)  # the result first


if __name__ == "__main__":
    unittest.main()
