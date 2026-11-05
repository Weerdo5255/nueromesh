"""Replies lead with the result in the mesh's own words; sources and reasoning come after."""
import tempfile
import unittest
from pathlib import Path

from neuromesh.knowledge import Document, Hit, restate
from neuromesh.runtime import System
from neuromesh.tasks import CapabilitySpec
from neuromesh.temperament import TAG, Shaped, honest, shape
from neuromesh.tools import ToolRun
from neuromesh.voice import lead_first, skill_phrase, tool_phrase

from test_system import FAST, make_root

ADA = ("Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27 November 1852), was an English "
       "mathematician and writer.[1] She was the only legitimate child of the poet Lord Byron.[2]")


def doc(title, summary):
    return Document(1, "wikipedia", title, f"https://example.org/{title}", "1", summary, title, "test", "en", [], 1,
                    0.0, 0.0, 0)


class RestateTests(unittest.TestCase):
    def test_own_words_from_passages(self):
        d = doc("Ada Lovelace", ADA)
        text, quotes = restate("who was Ada Lovelace?", [Hit(d, "Introduction", ADA, 1.0, 1.0)])
        self.assertEqual(text, "Ada Lovelace was an English mathematician and writer. "
                               "She was the only legitimate child of the poet Lord Byron.")
        self.assertTrue(quotes[0].startswith("Augusta Ada King, Countess of Lovelace (née Byron"))  # verbatim
        death = "Lovelace died on 27 November 1852 at the age of 36."
        text, _ = restate("when did Ada Lovelace die?", [Hit(d, "Death", death, 1.0, 1.0)])
        self.assertEqual(text, "Ada Lovelace died on 27 November 1852 at the age of 36.")
        tower = doc("Eiffel Tower", "The Eiffel Tower is a wrought-iron lattice tower in Paris. It is 330 metres tall.")
        text, _ = restate("how tall is the Eiffel Tower?", [Hit(tower, "Introduction", "It is 330 metres tall.", 1, 1)])
        self.assertEqual(text, "The Eiffel Tower is 330 metres tall.")

    def test_preambles_and_phrasing(self):
        self.assertEqual(lead_first("Based on the passages provided, Paris is the capital."), "Paris is the capital.")
        self.assertEqual(lead_first("Sure! According to the article, it is 330 m tall."), "It is 330 m tall.")
        self.assertEqual(lead_first("Paris is the capital."), "Paris is the capital.")
        spec = CapabilitySpec("language", labels=["english", "french"], label_descriptions={"french": "written in French"})
        self.assertEqual(skill_phrase(spec, "french"), "I'd call that french: written in French.")
        spec.answer = "That is written in {label}."
        self.assertEqual(skill_phrase(spec, "english"), "That is written in english.")
        spec.answers = {"french": "C'est du français."}
        self.assertEqual(skill_phrase(spec, "french"), "C'est du français.")
        tool = type("T", (), {"name": "calc"})()
        self.assertEqual(tool_phrase(tool, ToolRun(True, 84, extra={"say": "12 * 7 = 84"})), "12 * 7 = 84.")
        self.assertEqual(tool_phrase(tool, ToolRun(True, 84)), "The answer is 84.")
        self.assertEqual(tool_phrase(tool, ToolRun(True, "3 km is 1.86 miles")), "3 km is 1.86 miles.")

    def test_soften_keeps_the_answer_first(self):
        ctx = {"order_id": "o", "temperament": {"trust": 0.5, "arousal": 0.5, "patience": 0.5}, "tone": "hostile",
               "confidence": 0.9, "provenance": "x", "defer_seconds": 1, "brief_chars": 50}
        shaped = shape("soften", "12 * 7 = 84.", ctx)
        self.assertTrue(shaped.text.startswith("12 * 7 = 84.\n"))
        self.assertTrue(honest("12 * 7 = 84.", shaped, "soften"))
        before = Shaped(f"Let's keep things calm.\n12 * 7 = 84.\n{TAG} gentler wording added; the answer itself is unchanged]")
        self.assertFalse(honest("12 * 7 = 84.", before, "soften"), "a line before the answer is refused")


class AnswerFirstSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides=FAST, seed=1)
        await self.system.start()

    async def asyncTearDown(self):
        await self.system.stop()
        self.tmp.cleanup()

    async def test_every_kind_of_reply_leads_with_the_result(self):
        s = self.system
        res = await s.submit("what is 12 * 7")
        self.assertEqual((res.answer, res.output), ("12 * 7 = 84.", 84))  # the raw value stays for programs
        res = await s.submit("count the words in: the quick brown fox")
        self.assertEqual(res.answer, "That text has 4 words, 19 characters and 1 sentence.")
        res = await s.submit("sentiment: the hotel was lovely")
        self.assertEqual((res.answer, res.output), ("That reads as positive.", "positive"))
        res = await s.submit("urgency: the server is down, fix it asap")
        self.assertEqual(res.answer, "That needs attention right away (urgent).")
        res = await s.submit("frobnicate the gizmo")
        self.assertTrue(res.answer.startswith("I couldn't do that: "), res.answer)
        self.assertTrue(res.summary().startswith(res.answer))  # then [route | confidence | served by]
        done = [e for e in s.bus.recent("order.done", 10) if e.data["order"] == res.order_id]
        self.assertTrue(done and done[0].data["output"].startswith("I couldn't do that"))

    async def test_teacher_answers_lead_and_sources_follow(self):
        s = self.system

        async def compose(question, hits):
            return "Based on the passages, Paris is the capital of France.", "ollama"

        async def answer(text):
            return "Sure! The answer is blue.\nI am fairly sure.", "ollama"
        s.teachers.compose, s.teachers.answer = compose, answer
        s.knowledge.add({"source": "kiwix", "title": "Paris", "url": "http://kiwix/Paris", "summary":
                         "Paris is the capital and largest city of France.", "passages":
                         [{"section": "Introduction", "text": "Paris is the capital and largest city of France."}],
                         "links": [], "topic": "Paris"}, "test")
        res = await s.submit("what is the capital of France? Paris")
        lines = res.answer.splitlines()
        self.assertEqual(lines[0], "Paris is the capital of France.")
        self.assertEqual(lines[1], "Source: kiwix 'Paris' http://kiwix/Paris")
        res = await s.submit("what colour is the sky on a clear day")
        lines = res.answer.splitlines()
        self.assertEqual(lines[:2], ["The answer is blue.", "I am fairly sure."])
        self.assertTrue(lines[-1].startswith("Source: the ollama teacher (a language model"), res.answer)


if __name__ == "__main__":
    unittest.main()
