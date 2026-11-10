"""Picture books (imagesets.py): image datasets from the Hugging Face Hub teach its optical nodes, practice teaches and
exams measure, and what they teach is merged into its memory nodes as picture memories (memory.py). And JPEG - what the
dataset viewer serves - decodes without Pillow (imaging.py)."""
import asyncio
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_servers import SHAPES, MockDatasets  # noqa: E402
from test_system import FAST, make_root, wait_for  # noqa: E402

from neuromesh import imagesets as books_mod  # noqa: E402
from neuromesh import imaging  # noqa: E402
from neuromesh.config import deep_merge  # noqa: E402
from neuromesh.imagesets import BOOKS, SAYS, guess_columns, label_name  # noqa: E402
from neuromesh.memory import is_picture  # noqa: E402
from neuromesh.runtime import System  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _picture(h, w):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // max(1, w - 1)), (y * 255 // max(1, h - 1)), ((x + y) * 3) % 256], axis=2).astype(np.uint8)
    a[h // 4:h // 2, w // 4:w // 2] = [200, 30, 40]
    return a


@unittest.skipIf(Image is None, "Pillow is not installed (it writes the JPEGs these compare with)")
class JpegTests(unittest.TestCase):
    """Its own baseline JPEG decoder reads what Pillow wrote, as Pillow reads it (within a couple of levels)."""

    def jpeg(self, a, **kw):
        buf = io.BytesIO()
        im = Image.fromarray(a)
        if kw.pop("grey", False):
            im = im.convert("L")
        im.save(buf, "JPEG", quality=kw.pop("quality", 90), **kw)
        return buf.getvalue()

    def test_it_reads_what_pillow_reads(self):
        for (h, w, kw) in [(32, 32, {}), (28, 28, {"grey": True}), (75, 326, {"subsampling": 0}), (120, 721, {"subsampling": 2}),
                           (37, 53, {"subsampling": 1}), (33, 47, {"grey": True}), (64, 96, {"restart_marker_blocks": 3})]:
            try:
                data = self.jpeg(_picture(h, w), **dict(kw))
            except TypeError:  # an older Pillow without restart markers
                continue
            mine = imaging._jpeg(data)
            ref = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            self.assertEqual(mine.shape, ref.shape, kw)
            d = np.abs(mine.astype(int) - ref.astype(int))
            self.assertLess(d.mean(), 1.0, kw)
            self.assertLessEqual(np.percentile(d, 99), 4, kw)

    def test_what_it_cannot_read_says_so(self):
        data = self.jpeg(_picture(40, 40), progressive=True)
        with self.assertRaises(imaging.ImageError) as cm:
            imaging._jpeg(data)
        self.assertIn("Pillow", str(cm.exception))
        with self.assertRaises(imaging.ImageError):
            imaging._jpeg(b"\xff\xd8\xff\xd9")  # no frame

    def test_without_pillow_jpeg_goes_to_its_own_decoder(self):
        data = self.jpeg(_picture(32, 32))
        real = sys.modules.get("PIL")
        sys.modules["PIL"] = None  # (as if it were not installed)
        try:
            rgb = imaging.decode(data)
        finally:
            sys.modules["PIL"] = real
        self.assertEqual(rgb.shape, (32, 32, 3))


class HelperTests(unittest.TestCase):
    def test_label_names(self):
        self.assertEqual([label_name(x) for x in ("T - shirt / top", "Ankle boot", "aquarium_fish", "Cat")],
                         ["t_shirt_top", "ankle_boot", "aquarium_fish", "cat"])

    def test_it_guesses_what_a_dataset_is(self):
        img = {"name": "img", "kind": "image", "labels": []}
        g = guess_columns([img, {"name": "label", "kind": "class", "labels": ["cat", "dog"]}], [])
        self.assertEqual((g["image"], g["answer"], g["kind"]), ("img", "label", "objects"))
        g = guess_columns([img, {"name": "label", "kind": "class", "labels": list("0123456789")}], [])
        self.assertEqual(g["kind"], "characters")
        g = guess_columns([img, {"name": "text", "kind": "string", "labels": []}], [{"text": "hello there"}])
        self.assertEqual((g["answer"], g["kind"]), ("text", "text"))
        g = guess_columns([{"name": "text", "kind": "string", "labels": []}], [])
        self.assertEqual(g["kind"], "")
        self.assertIn("no image", g["why"])
        self.assertTrue(all(b["kind"] in ("objects", "characters", "text") for b in BOOKS.values()))


class PictureBookTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.hub = MockDatasets()
        self.tmp = tempfile.TemporaryDirectory()
        self.over = deep_merge(FAST, {"challenges": {"api_url": self.hub.url, "hub_url": self.hub.url},
                                      "memory": {"enabled": True, "interval_seconds": 0.05},
                                      "imagesets": {"lesson_size": 30, "exam_every_lessons": 0, "idle_seconds": 0,
                                                    "lesson_every_seconds": 0.2}})
        self.root = make_root(Path(self.tmp.name))
        self.system = System(self.root, overrides=self.over, seed=4)
        await self.system.start()

    async def asyncTearDown(self):
        if not self.system.stopping:
            await self.system.stop()
        self.hub.stop()
        self.tmp.cleanup()

    async def test_a_book_of_objects_teaches_the_sight_node_and_exams_measure_it(self):
        s = self.system
        b = s.imagesets
        info = await asyncio.to_thread(b.inspect, "pics/shapes")
        self.assertEqual((info["guess"]["image"], info["guess"]["answer"], info["guess"]["kind"]), ("img", "label", "objects"))
        self.assertTrue(info["sample"][0]["src"].startswith(self.hub.url))
        self.hub.broken = {4, 5}  # two images the viewer can't serve: skipped, not fatal
        res = await b.get(dataset="pics/shapes", name="shapes", by="tester")
        self.assertEqual(res["skipped"], 2)
        book = res["book"]
        self.assertEqual((book["kind"], book["items"], book["labels"]), ("objects", 88, 3))
        self.assertEqual(book["practice"] + book["exam"], 88)
        self.assertGreater(book["exam"], 5)
        self.assertTrue(all(it["answer"] in SHAPES for it in b.items["shapes"]))
        # kept on disk: a new system finds the book again, split the same way
        split = {it["sha"]: it["split"] for it in b.items["shapes"]}
        again = books_mod.ImageSets(s)
        self.assertEqual({it["sha"]: it["split"] for it in again.items["shapes"]}, split)
        # the teacher names practice images only: never an exam image
        teacher = s.teachers.teachers["imagesets"]
        spec = s.capabilities["sight"]
        practice, exam = b.practice("shapes"), b.exam_items("shapes")
        lesson = await teacher.label(spec, f"image:{practice[0]['sha']}")
        self.assertEqual((lesson.label, lesson.source), (practice[0]["answer"], "imagesets"))
        self.assertIsNone(await teacher.label(spec, f"image:{exam[0]['sha']}"))
        more = await teacher.generate(spec, "circle", 5)
        self.assertTrue(more and all(b.index[t[6:]][1] == "circle" for t in more))
        # lessons: the sight node learns its practice images
        for _ in range(3):
            res = await b.lesson("shapes", by="tester")
            self.assertGreater(res["learned"], 0)
        taught = {e.text: e.label for e in s.store.examples("sight") if e.source == "imagesets"}
        self.assertEqual(len(taught), len(practice))
        self.assertFalse({f"image:{it['sha']}" for it in exam} & set(taught))  # exam images are never learned from
        self.assertTrue(set(SHAPES) <= set(s.capabilities["sight"].labels))
        # an exam grades the sight node on images it never learned
        self.assertTrue(await wait_for(lambda: s.registry.champion("sight") is not None, 60))
        ex = await b.exam("shapes", by="tester")
        self.assertEqual(ex["n"], len(exam))
        self.assertAlmostEqual(ex["chance"], 1 / 3, places=2)
        self.assertGreater(ex["score"], ex["chance"])
        self.assertIn("chance 33%", ex["line"])
        self.assertEqual(b.summary("shapes")["last"]["score"], ex["score"])

    async def test_what_the_books_teach_is_merged_into_memory_as_picture_memories(self):
        s = self.system
        b = s.imagesets
        await b.get(dataset="pics/shapes", name="shapes")
        res = await b.lesson("shapes")
        self.assertGreaterEqual(res["memories"], 6)  # the first two pictures of every label
        nodes = [n for n in s.memory.fact_nodes() if n.tests]
        self.assertTrue(nodes and all(n.pictures for n in nodes))
        held = [t for n in nodes for t in n.facts()]
        self.assertTrue(all(is_picture(t.ask) and t.answer in SHAPES and "picture book shapes" in t.source for t in held))
        # a question in words goes to its own node, never a picture node
        s.memory.learn("what is the capital of Peru?", "Lima", "test", by="tester", kind="people", reviewed=True)
        words = [n for n in s.memory.fact_nodes() if any(t.ask.startswith("what is the capital") for t in n.tests)]
        self.assertEqual(len(words), 1)
        self.assertFalse(words[0].pictures)
        self.assertTrue(await wait_for(lambda: all(n.params is not None and not n.dirty for n in s.memory.fact_nodes()), 90))
        pics = [n for n in s.memory.fact_nodes() if n.pictures]
        self.assertTrue(all(n.passed == n.total for n in pics), [(n.id, n.passed, n.total, n.failing[:3]) for n in pics])
        self.assertFalse([pair for pair in s.memory.merge_candidates() if pair[0].pictures != pair[1].pictures])
        # it recalls a picture it remembers, and says which memory node holds it
        t = held[0]
        r = s.memory.recall(t.ask)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, t.answer)
        self.assertTrue(r.node.pictures)
        self.assertIsNone(s.memory.recall("what is in image:0123456789abcdef0123"))  # words are not pictures
        rec = await s.vision.take(b.image_path("shapes", next(it for it in b.items["shapes"] if it["sha"] == t.ask[6:])).read_bytes(),
                                  name="one it remembers")
        self.assertEqual(rec["sees"]["label"], t.answer)
        self.assertTrue(rec["sees"].get("memory"))
        self.assertIn("remembers this picture", rec["sees"]["by"])
        self.assertIn("picture book shapes", rec["sees"]["by"])
        # a person's word about an image is kept as a picture memory too (P1)
        other = next(it for it in b.items["shapes"] if f"image:{it['sha']}" not in {x.ask for x in held})
        rec2 = await s.vision.take(b.image_path("shapes", other).read_bytes(), name="a new one")
        ok, msg = s.vision.correct(rec2["id"], "tester", label="red_block")
        self.assertTrue(ok)
        self.assertIn("memory keeps the picture", msg)
        self.assertEqual(s.memory.known(f"image:{other['sha']}").answer, "red_block")
        # and picture memories can be switched off
        s.directives.imagesets.memory = False
        self.assertFalse(s.memory.allows("pictures"))
        self.assertFalse(b.remember(b.books["shapes"], other, "circle"))

    async def test_characters_and_text_teach_its_reader(self):
        s = self.system
        b = s.imagesets
        info = await asyncio.to_thread(b.inspect, "pics/digits")
        self.assertEqual(info["guess"]["kind"], "characters")
        await b.get(dataset="pics/digits", name="digits")
        res = await b.lesson("digits")
        self.assertIn("found one character in 30 of 30", res["line"])
        glyphs = [e for e in s.store.examples("glyphs") if e.source == "imagesets"]
        self.assertGreaterEqual(len(glyphs), 10)
        self.assertTrue(all(e.text.startswith("glyph:") and e.label in "0123456789" for e in glyphs))
        self.assertTrue(all(s.vision.known.get(e.text) == e.label for e in glyphs))
        if Image is not None:  # printed words are drawn with Pillow's own font
            info = await asyncio.to_thread(b.inspect, "pics/printed-words")
            self.assertEqual(info["guess"]["kind"], "text")
            await b.get(dataset="pics/printed-words", name="words")
            res = await b.lesson("words")
            self.assertIn("exactly as many characters as the text has", res["line"])
            letters = {e.label for e in s.store.examples("glyphs") if e.source == "imagesets" and e.label.isalpha()}
            self.assertTrue(set("catsun") <= letters, letters)
        for _ in range(4):
            await b.lesson("digits")
        self.assertTrue(await wait_for(lambda: s.registry.champion("glyphs") is not None, 60))
        ex = await b.exam("digits")
        self.assertIsNotNone(ex["score"])
        self.assertIn("handwritten characters right", ex["line"])

    async def test_it_gets_its_first_books_by_itself_and_a_book_can_be_removed(self):
        s = self.system
        b = s.imagesets
        BOOKS["test-shapes"] = {"dataset": "pics/shapes", "config": "default", "split": "train", "image": "img", "answer": "label",
                                "kind": "objects", "license": "mit", "description": "shapes for a test"}
        try:
            self.assertTrue(any(c["name"] == "test-shapes" and not c["installed"] for c in b.catalog()))
            s.directives.imagesets.start_with = ("test-shapes",)
            s.directives.imagesets.fetch_limit = 40
            self.assertTrue(await wait_for(lambda: "test-shapes" in b.books, 30))  # the Picture-book agent, when idle
            self.assertEqual(b.books["test-shapes"]["by"], "its picture books")
            self.assertTrue(await wait_for(lambda: b.state.get("test-shapes", {}).get("lessons", 0) >= 1, 30))
            self.assertTrue(any(c["name"] == "test-shapes" and c["installed"] for c in b.catalog()))
            # more of a book: from where it stopped
            got = await b.get(name="test-shapes", more=True, limit=20)
            self.assertEqual(got["book"]["items"], 60)
            with self.assertRaises(ValueError):
                await b.get(dataset="acme/odd-columns", name="odd")  # no image column
            folder = b.folder("test-shapes")
            ok, msg = b.remove("test-shapes", "tester")
            self.assertTrue(ok)
            self.assertFalse(folder.exists())
            self.assertFalse(b.index)
            self.assertFalse([e for e in s.store.examples("sight") if e.source == "imagesets"])
            self.assertIn("removed test-shapes", msg)
        finally:
            BOOKS.pop("test-shapes", None)

    async def test_the_web_page_and_the_api(self):
        from neuromesh.server import _route
        s = self.system
        code, out = await _route(s, "POST", "/pictures", {}, {"dataset": "pics/shapes", "name": "shapes", "limit": 30, "by": "web"})
        self.assertEqual(code, 200, out)
        self.assertEqual(out["book"]["items"], 30)
        code, st = await _route(s, "GET", "/images", {}, {})
        self.assertEqual(st["books"]["books"][0]["name"], "shapes")
        code, found = await _route(s, "GET", "/pictures/search", {"q": "shapes"}, {})
        self.assertTrue(found["datasets"][0]["images"])
        sha = s.imagesets.items["shapes"][0]["sha"]
        code, raw = await _route(s, "GET", "/pictures/file", {"book": "shapes", "sha": sha}, {})
        self.assertEqual((code, raw.ctype), (200, "image/jpeg" if Image is not None else "image/png"))
        code, raw = await _route(s, "GET", "/images/file", {"sha": sha}, {})  # a picture memory's question, shown
        self.assertEqual(code, 200)
        code, out = await _route(s, "POST", "/pictures", {}, {"lesson": "shapes"})
        self.assertEqual(code, 200, out)
        code, out = await _route(s, "POST", "/pictures", {}, {"exam": "nope"})
        self.assertEqual(code, 400)
        from neuromesh.commands import execute
        out = await execute(s, "/pictures")
        self.assertTrue(any("shapes" in line for line in out.lines))


if __name__ == "__main__":
    unittest.main()
