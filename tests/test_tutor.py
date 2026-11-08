"""Lessons (tutor.py): when it is idle, its Ollama teachers set it orders, mark its answers, and teach it - as
teachers, never as people."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.runtime import System
from neuromesh.server import _route

from mock_servers import MockOllama
from test_system import FAST, make_root, wait_for

SET = {"maths": {"order": "what is 17 * 23", "expect": "391", "why": "multiplication"},
       "question": {"order": "What is the capital of Atlantis?", "expect": "Poseidonis", "why": "a legend"},
       "skill": {"order": "sentiment: the hotel was lovely and the staff were wonderful", "expect": "positive", "why": "praise"},
       "code": {"order": "write a python function that reverses the word hello", "expect": "it returns the text backwards"},
       "tool": {"order": "count the words in: the cat sat on the mat", "expect": "6"}}


def teacher(call):
    user, system = call["user"], call["system"]
    if "setting a short test" in system:
        kind = ("skill" if "'<skill>: <text>'" in user else "maths" if "arithmetic" in user else
                "code" if "python function" in user else "tool" if "tools can meet" in user else "question")
        return SET[kind]
    if "marking" in system:
        answer = user.split("The mesh answered:", 1)[1]
        if "Poseidonis" in user.split("The mesh answered:", 1)[0] and "Poseidonis" not in answer:
            return {"verdict": "wrong", "why": "the legend names Poseidonis", "correct": "Poseidonis"}
        if "def " in answer or "Its code" in user or "code it wrote" in user:
            return {"verdict": "right", "why": "the code reverses its text", "correct": ""}
        return {"verdict": "right", "why": "that is right", "correct": ""}
    if "fallback answerer" in system:
        return "I am not sure; Atlantis may not have had a capital."
    return {"label": "positive", "confidence": 0.9}


class TutorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = MockOllama(teacher)
        self.tmp = tempfile.TemporaryDirectory()
        over = {**FAST, "teachers": {"order": ["human", "ollama", "heuristic"],
                                     "ollama": {"provider": "ollama", "base_url": self.llm.url, "model": "llama3.1:8b"}},
                "memory": {"enabled": True, "interval_seconds": 0.05},
                "tutor": {"idle_seconds": 3600, "every_seconds": 0.5}}
        self.system = System(make_root(Path(self.tmp.name)), overrides=over, seed=8)
        await self.system.start()
        self.assertTrue(await wait_for(lambda: self.system.tools.tools.get("calculator") is not None
                                       and self.system.tools.tools["calculator"].state == "active", 20))

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.llm.stop()
        self.tmp.cleanup()

    async def test_teachers_set_orders_and_mark_them(self):
        s = self.system
        t = s.tutor
        self.assertEqual(t.due(), "people gave it orders lately")          # it has only just started
        rec = await t.lesson(kind="maths")
        self.assertEqual((rec["kind"], rec["verdict"]), ("maths", "right"))
        self.assertEqual(rec["marked_by"], "the teacher's expected number")  # no need to ask the teacher
        self.assertIn("391", rec["answer"])
        self.assertEqual(s.last_order_at, 0.0)                             # a teacher's order leaves it idle
        rec = await t.lesson(kind="question")
        self.assertEqual(rec["verdict"], "wrong")
        self.assertEqual(rec["marked_by"], "the ollama teacher")
        self.assertTrue(any("memory learns the teacher's answer" in x for x in rec["learned"]), rec["learned"])
        mark = next(c for c in self.llm.calls if "marking" in c["system"])
        self.assertIn("What you expected: Poseidonis", mark["user"])
        self.assertTrue(await wait_for(lambda: s.memory.known("What is the capital of Atlantis?") is not None, 30))
        fact = s.memory.known("What is the capital of Atlantis?")
        self.assertEqual((fact.answer, fact.by), ("Poseidonis", "ollama"))  # a teacher's lesson, not a person's
        rec = await t.lesson(kind="skill")
        self.assertIn(rec["verdict"], ("right", "wrong"))
        self.assertEqual(rec["marked_by"], "the teacher's expected label")
        self.assertTrue(any(e.source == "tutor" for e in s.store.examples("sentiment")) or rec["verdict"] == "wrong")
        rec = await t.lesson(kind="code")
        self.assertEqual((rec["route"], rec["verdict"]), ("scratch", "right"))
        self.assertIn("code it wrote", next(c for c in self.llm.calls[::-1] if "marking" in c["system"])["user"])
        st = t.status()
        self.assertEqual(st["count"], 4)
        self.assertTrue(any(e.topic == "tutor.lesson" for e in s.bus.recent("tutor.*", 10)))
        status, body = await _route(s, "GET", "/lessons", {}, {})
        self.assertEqual(body["count"], 4)
        self.assertIn("Lessons: 4", "\n".join(t.lines()))

    async def test_when_idle_a_teacher_sets_a_lesson(self):
        s = self.system
        s.directives.tutor.idle_seconds = 0.0
        self.assertIsNone(s.tutor.due())
        self.assertTrue(await wait_for(lambda: s.tutor.stats.get("lessons", 0) >= 1, 30))
        s.directives.tutor.enabled = False
        self.assertEqual(s.tutor.due(), "lessons are off")


if __name__ == "__main__":
    unittest.main()
