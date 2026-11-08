"""Its optical nodes (vision.py, imaging.py): it takes in images, says what it sees, reads the text in them with its
own glyph node, and learns from both - from people, an LLM teacher that can see, tesseract and glyphs it draws."""
import asyncio
import base64
import io
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from neuromesh import imaging
from neuromesh.runtime import System
from neuromesh.server import _route
from neuromesh.vision import CHARS, char_of, label_of, similarity

from mock_servers import MockOllama
from test_system import FAST, make_root, wait_for

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

NODES = {k: v for k, v in FAST.items() if k != "nodes"}  # (the normal node settings: a glyph node needs time to train)


def text_image(lines, size=26, font=None) -> bytes:
    font = font or next((f for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                       "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf") if Path(f).exists()), None)
    f = ImageFont.truetype(font, size) if font else ImageFont.load_default()
    pil = Image.new("RGB", (40 + 18 * max(len(x) for x in lines), 30 + 48 * len(lines)), "white")
    d = ImageDraw.Draw(pil)
    for i, line in enumerate(lines):
        d.text((14, 12 + 48 * i), line, fill="black", font=f)
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    return buf.getvalue()


class ImagingTests(unittest.TestCase):
    def test_decoding_without_pillow(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 255, (21, 34, 3), dtype=np.uint8)
        self.assertTrue((imaging.decode(imaging.encode_png(a)) == a).all())
        # a palette PNG with 2-bit samples, written by hand (filter "Paeth" on every row)
        w, h = 5, 3
        idx = np.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 1], [1, 1, 2, 2, 3]], dtype=np.uint8)
        rows = []
        prev = np.zeros(2, dtype=np.int32)
        for y in range(h):
            packed = np.zeros(2, dtype=np.int32)
            for x in range(w):
                packed[x // 4] |= int(idx[y, x]) << (6 - 2 * (x % 4))
            filt = []
            for i in range(2):
                a_, b_, c_ = (packed[i - 1] if i else 0), prev[i], (prev[i - 1] if i else 0)
                filt.append((packed[i] - imaging._paeth(int(a_), int(b_), int(c_))) & 0xFF)
            rows.append(b"\x04" + bytes(filt))
            prev = packed
        pal = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])

        def chunk(tag, body):
            return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 2, 3, 0, 0, 0)) + chunk(b"PLTE", pal)
               + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))
        got = imaging.decode(png)
        colours = np.frombuffer(pal, dtype=np.uint8).reshape(4, 3)
        self.assertTrue((got == colours[idx]).all())
        ppm = b"P6\n# a comment\n34 21\n255\n" + a.tobytes()
        self.assertTrue((imaging.decode(ppm) == a).all())
        self.assertEqual(len(imaging.features(a)), 415)
        self.assertEqual(imaging.sniff(b"\xff\xd8\xff\xe0"), "jpeg")
        with self.assertRaises(imaging.ImageError):
            imaging.decode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        self.assertEqual(char_of(label_of("?")), "?")
        self.assertEqual(len({label_of(c) for c in CHARS}), len(CHARS))
        self.assertAlmostEqual(similarity("Meet me", "Meet  me"), 1.0)

    @unittest.skipUnless(HAVE_PIL, "needs Pillow to draw text")
    def test_finding_lines_and_glyphs(self):
        rgb = imaging.decode(text_image(["Hello World 42", "quick brown fox"]))
        found = imaging.lines(rgb)
        self.assertEqual(len(found), 2)
        self.assertEqual(len(found[0]["glyphs"]), 12)
        self.assertEqual(sum(g["space_before"] for g in found[0]["glyphs"]), 2)
        ink = imaging.binarize(rgb)
        t = imaging.glyph_text(ink, found[0]["glyphs"][0], found[0])
        self.assertTrue(t.startswith("glyph:") and len(t.split(":")[1]) == 64)
        v = imaging.glyph_vector(t, 1024)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)


class VisionSystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        if getattr(self, "system", None) is not None and not self.system.stopping:
            await self.system.stop()
        self.tmp.cleanup()

    async def test_it_sees_with_a_teacher_and_people_teach_it(self):
        def reply(call):
            if call["images"]:
                return {"label": "red square", "confidence": 0.9, "text": "", "why": "a red block"}
            return {"label": "positive", "confidence": 0.9}
        llm = MockOllama(reply)
        try:
            over = {**FAST, "teachers": {"order": ["human", "ollama", "heuristic"],
                                         "ollama": {"provider": "ollama", "base_url": llm.url, "model": "llama3.1:8b"}}}
            s = self.system = System(make_root(Path(self.tmp.name)), overrides=over, seed=4)
            await s.start()
            self.assertEqual(s.teachers.teachers["ollama"].vision, "llava:7b")  # the model that can see
            red = np.zeros((40, 60, 3), dtype=np.uint8)
            red[:, :] = (210, 30, 40)
            rec = await s.vision.take(imaging.encode_png(red), "red.png", "tester")
            self.assertEqual(rec["sees"]["label"], "red_square")
            self.assertIn("vision", rec["sees"]["by"])
            call = next(c for c in llm.calls if c["images"])
            self.assertEqual(call["model"], "llava:7b")
            self.assertEqual(base64.b64decode(call["images"][0])[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn("red_square", s.capabilities["sight"].labels)  # its sight node has a new label to learn
            blank = await s.vision.take(imaging.encode_png(np.full((30, 30, 3), 250, dtype=np.uint8)), "blank.png", "tester")
            self.assertTrue(blank["measures"]["blank"])
            ok, msg = s.vision.correct(rec["id"], "tester", label="flag")
            self.assertTrue(ok)
            self.assertEqual(s.teachers.human.answers[("sight", f"image:{rec['sha']}")], "flag")  # what a person says stands
            res = await s.submit(f"what do you see in {rec['id']}?")
            self.assertEqual(res.route, "vision")
            self.assertIn("shows flag", res.output)
            again = await s.vision.take(imaging.encode_png(red), "red again", "tester")  # the same picture
            self.assertEqual(again["sha"], rec["sha"])
            self.assertEqual(again["sees"]["label"], "flag")
            status, body = await _route(s, "GET", "/images/file", {"id": rec["id"]}, {})
            self.assertEqual((status, body.ctype), (200, "image/png"))
            status, body = await _route(s, "POST", "/images", {"name": "raw.png"}, {"_bytes": imaging.encode_png(red[:20])})
            self.assertEqual(status, 200)
            status, body = await _route(s, "POST", "/images", {}, {"data": base64.b64encode(b"not an image").decode()})
            self.assertEqual(status, 400)
            status, body = await _route(s, "GET", "/images", {}, {})
            self.assertGreaterEqual(body["count"], 3)
        finally:
            llm.stop()

    @unittest.skipUnless(HAVE_PIL, "needs Pillow (to draw glyphs to practise with) and fonts")
    async def test_it_learns_to_read(self):
        s = self.system = System(make_root(Path(self.tmp.name)), overrides=NODES, seed=6)
        await s.start()
        if not s.vision.fonts:
            self.skipTest("no fonts to practise with")
        for _ in range(5):
            await s.vision.practice(800)
        self.assertGreater(s.store.count("glyphs"), 2000)
        self.assertTrue(await wait_for(lambda: s.vision.glyph_node() is not None, 240), "no validated glyph node")
        rec = await s.vision.take(text_image(["Meet me at 7pm", "Bring the red kite!"]), "note.png", "tester")
        self.assertEqual(rec["sees"]["label"], "text")                       # its own measures
        own = rec["text"]
        self.assertTrue(own["by"].startswith("glyph node"), own)
        self.assertGreater(similarity(own["text"], "Meet me at 7pm\nBring the red kite!"), 0.8, own["text"])
        if s.vision.tesseract:                                               # tesseract teaches its glyph node too
            self.assertTrue(any("tesseract labelled" in x for x in rec["learned"]))
            self.assertIsNotNone(own.get("agrees"))
        res = await s.submit(f"what does {rec['id']} say?")
        self.assertIn("kite", res.output)
        self.assertIsNotNone(s.knowledge.get(f"note.png ({rec['id']})", "image"))  # it can answer from what it read
        ok, msg = s.vision.correct(rec["id"], "tester", text="Meet me at 7pm\nBring the red kite!")
        self.assertTrue(ok)
        self.assertIn("learns 27 character", msg)                            # a person's words label its glyphs
        lines = "\n".join(s.vision.lines())
        self.assertIn("Reader:", lines)


if __name__ == "__main__":
    unittest.main()
