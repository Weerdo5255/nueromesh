"""Its optical nodes: it takes in images, says what it sees, reads the text in them, and learns from both.

    an image (the web page, POST /images, /image <path>)
      --> kept in state/images/ by its contents (the same picture twice is one image)
      --> its memory nodes are asked first: a picture it remembers (a picture memory, memory.py) is named by the
          memory node that holds it
      --> otherwise its sight node (capability "sight", open labels) says what it shows. Until the node is validated,
          its teachers do, and teach it: a person ("it shows a cat"), an LLM teacher that can see (an Ollama vision
          model such as llava), its own measures ("text" when it reads words, "blank" when there is nothing), and the
          picture books it learns from (imagesets.py: image datasets from the Hugging Face Hub)
      --> its reader reads the text: lines and characters are found in the ink, and its glyph node (capability
          "glyphs") reads each character. The glyph node learns from glyphs it draws itself from the fonts it has
          (it knows what each one is: GlyphTeacher), from tesseract when it is installed (its character boxes
          label the glyphs in real images), and from people who say what an image's text says
      --> what it read is kept as knowledge it can answer from, with the image as its source; what it saw is kept
          with the image. Its own reading is compared with tesseract's, a teacher's or a person's, so how well it
          reads is measured, not assumed.

When it is idle (the Looker) it practises reading glyphs it draws, and re-reads images once its reader improves.
Nothing here runs code from an image or sends an image anywhere except to a teacher you configured.
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import glob
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path

import numpy as np

from . import imaging
from .agents import Agent
from .teachers import Lesson, LLMTeacher, Teacher, vision_model_of

log = logging.getLogger("neuromesh.vision")

SIGHT, GLYPHS = "sight", "glyphs"
PUNCT = {".": "dot", ",": "comma", ":": "colon", ";": "semicolon", "!": "exclamation", "?": "question",
         "'": "apostrophe", '"': "quote", "-": "hyphen", "(": "open-paren", ")": "close-paren", "&": "ampersand",
         "/": "slash", "%": "percent", "@": "at", "#": "hash", "+": "plus", "=": "equals", "$": "dollar"}
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + "".join(PUNCT)
CHAR_OF = {v: k for k, v in PUNCT.items()}
MIME = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
        "pnm": "image/x-portable-anymap"}
SKIP_FONTS = re.compile(r"italic|oblique|emoji|symbol|math|icon|dingbat|braille|wingding|ornament|music|cursive|"
                        r"script|handwrit|fontawesome|kacst|lohit|noto.*(cjk|arab|hebrew|deva|thai)|mono.*bold", re.I)
IMG_RE = re.compile(r"\bimg-\d+\b", re.I)


def label_of(ch: str) -> str:
    return PUNCT.get(ch, ch)


def char_of(label: str) -> str:
    return CHAR_OF.get(label, label if len(label) == 1 else "?")


def similarity(a: str, b: str) -> float:
    a, b = " ".join(str(a).split()), " ".join(str(b).split())
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ============================================================================ seeing with a teacher
async def teacher_sees(teacher: LLMTeacher, image: bytes, mime: str, labels: list[str]) -> dict | None:
    """Ask an LLM teacher that can see what an image shows, and what text it can read in it."""
    if not teacher.available():
        return None
    provider = teacher.provider
    model = getattr(teacher, "vision", "") if provider == "ollama" else teacher.model
    if not model:
        return None
    decision = teacher.guardian.authorize_llm_call(teacher.name, teacher.cfg, 1200 + teacher.cfg.max_tokens)
    if not decision.allowed:
        return None
    known = ", ".join(labels[:40])
    prompt = ("Look at the image. What does it mainly show? Give a short label in lower case, one or two words "
              "(like 'cat', 'street sign', 'receipt', 'bar chart'). " + (f"Use one of these if one fits: {known}. " if known else "")
              + "Then copy any text you can read in the image exactly, line by line (empty if there is none). "
              'Reply with JSON only: {"label": "...", "confidence": 0.0-1.0, "text": "...", "why": "<a few words>"}')
    b64 = base64.b64encode(image).decode()
    schema = {"type": "object", "required": ["label", "confidence"],
              "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}, "text": {"type": "string"},
                             "why": {"type": "string"}}}
    cfg = teacher.cfg

    def post() -> tuple[str, int]:
        if provider == "ollama":
            body = {"model": model, "stream": False, "keep_alive": cfg.keep_alive, "format": schema,
                    "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                    "options": {"temperature": cfg.temperature, "num_predict": max(cfg.max_tokens, 300)}}
            data = teacher._http("POST", "/api/chat", body, max(cfg.timeout_s, 120.0))
            if data.get("error"):
                raise RuntimeError(data["error"])
            return (data.get("message") or {}).get("content") or "", int(data.get("eval_count") or 0)
        if provider == "anthropic":
            data = teacher._http("POST", "/v1/messages", {
                "model": model, "max_tokens": max(cfg.max_tokens, 300), "temperature": cfg.temperature,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt}]}]},
                cfg.timeout_s, {"x-api-key": teacher._key(), "anthropic-version": cfg.anthropic_version})
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            u = data.get("usage", {})
            return text, u.get("input_tokens", 0) + u.get("output_tokens", 0)
        headers = {"authorization": f"Bearer {teacher._key()}"} if teacher._key() else {}
        data = teacher._http("POST", "/v1/chat/completions", {
            "model": model, "max_tokens": max(cfg.max_tokens, 300), "temperature": cfg.temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]},
            cfg.timeout_s, headers)
        return data["choices"][0]["message"]["content"] or "", data.get("usage", {}).get("total_tokens", 0)
    try:
        text, tokens = await asyncio.to_thread(post)
    except Exception as exc:
        teacher.errors += 1
        log.warning("teacher %s could not look at an image: %s", teacher.name, exc)
        return None
    teacher.calls += 1
    teacher.guardian.record_llm_usage(teacher.name, tokens)
    from .teachers import _extract_json
    data = _extract_json(text) or {}
    label = re.sub(r"[^a-z0-9 _-]", "", str(data.get("label", "")).lower()).strip().replace(" ", "_")[:40]
    if not label:
        return None
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
    except (TypeError, ValueError):
        conf = 0.7
    return {"label": label, "confidence": conf, "text": str(data.get("text") or "")[:4000], "why": str(data.get("why", ""))[:200],
            "teacher": teacher.name, "model": model}


class VisionTeacher(Teacher):
    """An LLM teacher that can see, for the sight node: what an image shows (and, kept with the image, its text)."""
    name = "vision"
    specialist = True

    def __init__(self, bank: "VisionBank"):
        self.bank = bank

    def available(self) -> bool:
        return self.bank.pol.vision_teacher and self.bank.seeing_teacher() is not None

    def status(self) -> dict:
        t = self.bank.seeing_teacher()
        return {"detail": f"specialist: {t.name} looks at images with {getattr(t, 'vision', '') or t.model}" if t else
                "specialist: needs an LLM teacher that can see (e.g. ollama pull llava)"}

    async def label(self, spec, text):
        if spec.name != SIGHT or not text.startswith("image:"):
            return None
        seen = await self.bank.ask_teacher(text[6:])
        if not seen:
            return None
        return Lesson(seen["label"], seen["confidence"], self.name, f"{seen['teacher']} ({seen['model']}): {seen['why']}")


class SightTeacher(Teacher):
    """Its own measures, for the sight node: an image with lines of text in it shows "text"; one with nothing in
    it is "blank". It knows nothing else, and says so (no label)."""
    name = "sight"
    specialist = True

    def __init__(self, bank: "VisionBank"):
        self.bank = bank

    def status(self) -> dict:
        return {"detail": "specialist: its own measures - text it can see, or a blank image"}

    async def label(self, spec, text):
        if spec.name != SIGHT or not text.startswith("image:"):
            return None
        rec = self.bank.by_sha(text[6:])
        if rec is None:
            return None
        m = rec.get("measures") or {}
        if m.get("blank"):
            return Lesson("blank", 0.9, self.name, "there is almost nothing in it")
        if m.get("text_lines", 0) >= 2 and m.get("glyphs", 0) >= 12 and m.get("text_share", 0) >= 0.25:
            return Lesson("text", 0.75, self.name, f"{m['text_lines']} lines of text")
        return None

    async def generate(self, spec, label, n):
        if spec.name != SIGHT:
            return []
        return [f"image:{r['sha']}" for r in self.bank.records.values() if (r.get("sees") or {}).get("label") == label][:n]


class GlyphTeacher(Teacher):
    """Glyphs it knows: ones it drew itself from its fonts, and ones tesseract or a person labelled in real images."""
    name = "glyphs"
    specialist = True

    def __init__(self, bank: "VisionBank"):
        self.bank = bank

    def available(self) -> bool:
        return True

    def status(self) -> dict:
        return {"detail": f"specialist: glyphs it draws from {len(self.bank.fonts)} fonts"
                + (" (needs Pillow to draw)" if not self.bank.can_draw else "") + f", {len(self.bank.known)} it knows"}

    async def label(self, spec, text):
        if spec.name != GLYPHS:
            return None
        ch = self.bank.known.get(text)
        return Lesson(label_of(ch), 0.95, self.name, "it drew it, or it was labelled") if ch else None

    async def generate(self, spec, label, n):
        if spec.name != GLYPHS:
            return []
        return await asyncio.to_thread(self.bank.draw_glyphs, char_of(label), n)


# ============================================================================ the bank
class VisionBank:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.vision
        self.dir = system.state_dir / "images"
        self.records: dict[str, dict] = {}
        for rec in system.store.load_records("images"):
            self.records[rec["id"]] = rec
        nums = [int(k.split("-")[1]) for k in self.records if re.fullmatch(r"img-\d+", k)]
        self._n = max(nums, default=0)
        raw = system.store.get_kv("vision.state")
        st = json.loads(raw) if raw else {}
        self.stats: Counter = Counter(st.get("stats", {}))
        self.reading: deque = deque(st.get("reading", []), maxlen=200)  # (own vs reference) similarity per reading
        self.log: deque = deque(st.get("log", []), maxlen=60)
        self._vec: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.known: dict[str, str] = {}  # glyph text -> the character (drawn, or labelled)
        self.rng = np.random.default_rng()
        self.prng = random.Random()
        self.fonts = self._fonts()
        try:
            import PIL  # noqa: F401
            self.can_draw = True
        except ImportError:
            self.can_draw = False
        t = self.pol.tesseract
        self.tesseract = (shutil.which("tesseract") if t == "auto" else t if t and os.path.exists(t) else None)

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def _fonts(self) -> list[str]:
        if self.pol.fonts:
            return [f for f in self.pol.fonts if os.path.exists(f)]
        found = []
        for pat in ("/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf", "/usr/local/share/fonts/**/*.ttf",
                    os.path.expanduser("~/.fonts/**/*.ttf"), os.path.expanduser("~/.local/share/fonts/**/*.ttf"),
                    "/Library/Fonts/*.ttf", "/System/Library/Fonts/*.ttf", "C:/Windows/Fonts/*.ttf"):
            found += glob.glob(pat, recursive=True)
        good = sorted({f for f in found if not SKIP_FONTS.search(os.path.basename(f))})
        return good[:80]

    def save(self) -> None:
        self.sys.store.set_kv("vision.state", json.dumps({"stats": dict(self.stats), "reading": list(self.reading),
                                                          "log": list(self.log)}))

    def _save(self, rec: dict) -> None:
        self.sys.store.save_record("images", rec["id"], rec.get("status", "seen"), rec)
        if len(self.records) > self.pol.keep_images:
            oldest = min(self.records.values(), key=lambda r: r.get("ts", 0))
            self.forget(oldest["id"])

    def forget(self, img_id: str) -> bool:
        rec = self.records.pop(img_id, None)
        if rec is None:
            return False
        self.sys.store.delete_record("images", img_id)
        if not any(r["sha"] == rec["sha"] for r in self.records.values()):
            for p in self.dir.glob(f"{rec['sha']}.*"):
                p.unlink(missing_ok=True)
        return True

    def by_sha(self, sha: str) -> dict | None:
        return next((r for r in self.records.values() if r["sha"] == sha), None)

    def path_of(self, rec: dict) -> Path:
        return self.dir / f"{rec['sha']}.{rec['ext']}"

    def file(self, img_id: str) -> tuple[bytes, str] | None:
        rec = self.records.get(img_id)
        if rec is None or not self.path_of(rec).exists():
            return None
        return self.path_of(rec).read_bytes(), MIME.get(rec["ext"], "application/octet-stream")

    # ------------------------------------------------------------------ what the nodes take in
    def vector(self, text: str) -> np.ndarray:
        dim = self.sys.encoder.dim
        sha = text[6:].strip()
        v = self._vec.get(sha)
        if v is None:
            p = self.dir / f"{sha}.npy"
            if not p.exists():  # an image from a picture book (imagesets.py)
                p = self.sys.state_dir / "imagesets" / "features" / f"{sha}.npy"
                if not p.exists():
                    return np.zeros(dim, dtype=np.float32)
            v = imaging.fold(np.load(p), dim)
            self._vec[sha] = v
            while len(self._vec) > 20000:
                self._vec.popitem(last=False)
        return v

    # ------------------------------------------------------------------ taking an image in
    async def take(self, data: bytes, name: str = "", by: str = "you") -> dict:
        """Take an image in: keep it, say what it shows, read its text, and learn from both."""
        s = self.sys
        if not self.enabled:
            raise ValueError("its image nodes are off ([vision] enabled)")
        if len(data) > self.pol.max_image_mb * 1048576:
            raise ValueError(f"the image is larger than {self.pol.max_image_mb:g} MB")
        kind = imaging.sniff(data)
        if not kind:
            raise ValueError("that is not an image it knows (PNG, JPEG, GIF, WebP, BMP or PPM)")
        rgb = await asyncio.to_thread(imaging.decode, data)
        sha = hashlib.sha256(data).hexdigest()[:20]
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{sha}.{kind}"
        if not path.exists():
            path.write_bytes(data)
        work = self._workable(rgb)
        feats = await asyncio.to_thread(imaging.features, work)
        np.save(self.dir / f"{sha}.npy", feats)
        self._vec.pop(sha, None)
        self._n += 1
        h, w = rgb.shape[:2]
        rec = {"id": f"img-{self._n}", "sha": sha, "ext": kind, "name": (name or f"image {self._n}")[:120], "w": int(w), "h": int(h),
               "bytes": len(data), "by": by, "ts": round(time.time(), 1), "status": "seen", "sees": None, "teacher": None,
               "text": None, "tesseract": None, "person": {}, "learned": [], "measures": {}}
        self.records[rec["id"]] = rec
        with s.activity.track("look", "eye", f"{rec['name']}: {w}x{h}"):
            await self.read(rec, work)
            await self.look(rec)
            r = self.remembers(rec["sha"])
            now = rec.get("text") or {}
            if r is not None and r.answer.startswith("says: ") and (not (now.get("text") or "").strip() or "unsure" in now.get("by", "")):
                rec["text"] = {"text": r.answer[6:], "confidence": round(float(r.confidence), 3), "node": r.node.id,
                               "glyphs": rec["measures"].get("glyphs", 0), "own_text": now.get("text", ""),
                               "by": f"memory node {r.node.id} remembers what this picture says" + (f" ({r.source})" if r.source else "")}
            seen = rec.get("teacher") or {}
            now = rec.get("text") or {}
            if seen.get("text") and (not (now.get("text") or "").strip() or "unsure" in now.get("by", "")):
                why = (f"its own reader was unsure: {now.get('confidence', 0):.0%}" if "unsure" in now.get("by", "")
                       else "its own reader is still learning")
                rec["text"] = {"text": seen["text"], "confidence": None, "node": None, "glyphs": rec["measures"].get("glyphs", 0),
                               "by": f"the {seen['teacher']} teacher ({seen['model']}; {why})", "own_text": now.get("text", "")}
            self.learn_text(rec)
        self.stats["images"] += 1
        self._save(rec)
        self.save()
        s.bus.publish("vision.seen", id=rec["id"], name=rec["name"], label=(rec["sees"] or {}).get("label"),
                      by=(rec["sees"] or {}).get("by", ""), words=len(((rec["text"] or {}).get("text") or "").split()),
                      read_by=(rec["text"] or {}).get("by", ""))
        return rec

    @staticmethod
    def _workable(rgb: np.ndarray, side: int = 2400) -> np.ndarray:
        h, w = rgb.shape[:2]
        if max(h, w) <= side:
            return rgb
        k = side / max(h, w)
        return imaging.shrink(rgb.astype(np.float32), int(h * k), int(w * k)).round().astype(np.uint8)

    def _png_of(self, rec: dict) -> tuple[bytes, str]:
        data = self.path_of(rec).read_bytes()
        if rec["ext"] in ("png", "jpeg", "gif", "webp"):
            return data, MIME[rec["ext"]]
        return imaging.encode_png(imaging.decode(data)), "image/png"

    def remembers(self, sha: str):
        """A picture memory of this image (memory.py), or None."""
        mem = getattr(self.sys, "memory", None)
        if mem is None or not mem.enabled:
            return None
        return mem.recall(f"image:{sha}")

    async def look(self, rec: dict) -> dict:
        """Its memory nodes (a picture it remembers), else its sight node (or, until it is validated, its teachers),
        say what the image shows."""
        s = self.sys
        r = self.remembers(rec["sha"])
        if r is not None and not r.answer.startswith("says: "):
            rec["sees"] = {"label": r.answer, "confidence": round(float(r.confidence), 3), "node": r.node.id, "memory": True,
                           "by": f"memory node {r.node.id} remembers this picture" + (f" ({r.source})" if r.source else "")}
            return rec["sees"]
        inf = await s.infer(SIGHT, f"image:{rec['sha']}")
        if inf.label is None:
            s.teachers.human.ask(SIGHT, f"image:{rec['sha']}", f"what does {rec['id']} ({rec['name']}) show?")
            rec["sees"] = {"label": None, "confidence": 0.0, "by": "nothing yet: it asks you", "node": None}
        else:
            rec["sees"] = {"label": inf.label, "confidence": round(float(inf.confidence), 3), "by": inf.provenance,
                           "node": inf.node_id}
        return rec["sees"]

    def seeing_teacher(self) -> LLMTeacher | None:
        for name in self.sys.teachers.order:
            t = self.sys.teachers.teachers.get(name)
            if isinstance(t, LLMTeacher) and t.available() and (t.provider != "ollama" or getattr(t, "vision", "")):
                return t
        return None

    async def ask_teacher(self, sha: str) -> dict | None:
        rec = self.by_sha(sha)
        t = self.seeing_teacher()
        if rec is None or t is None:
            return None
        if rec.get("teacher") and rec["teacher"].get("label"):
            return rec["teacher"]
        data, mime = await asyncio.to_thread(self._png_of, rec)
        spec = self.sys.capabilities[SIGHT]
        seen = await teacher_sees(t, data, mime, spec.active_labels())
        if seen:
            rec["teacher"] = seen
            self.stats["teacher looks"] += 1
        return seen

    # ------------------------------------------------------------------ reading
    def glyph_node(self):
        s = self.sys
        node = s.registry.champion(GLYPHS) if GLYPHS in s.capabilities else None
        if node is None or node.params is None:
            return None
        if (node.metrics.get("holdout_acc") or 0.0) < s.directives.nodes.serve_min_accuracy:
            return None
        return node

    def read_glyphs(self, rgb: np.ndarray) -> tuple[list[dict], np.ndarray]:
        ink = imaging.binarize(rgb)
        found = imaging.lines(rgb, ink)
        for line in found:
            for g in line["glyphs"]:
                g["text"] = imaging.glyph_text(ink, g, line)
        return found, ink

    def own_reading(self, found: list[dict], node=None, guess: bool = False) -> dict | None:
        """Its glyph node's reading of the glyphs found (the validated champion, or the node given: an exam). A
        character it is unsure of reads as "?", unless it is asked for its best guess (guess)."""
        node = node or self.glyph_node()
        if node is None or not found:
            return None
        texts = [g["text"] for line in found for g in line["glyphs"]]
        X = self.sys.encode_many(texts)
        P = node.predict_proba(X)
        labels = list(node.labels)
        out_lines, confs, k = [], [], 0
        for line in found:
            chars = []
            for g in line["glyphs"]:
                p = P[k]
                k += 1
                j = int(np.argmax(p))
                confs.append(float(p[j]))
                ch = char_of(labels[j]) if guess or p[j] >= self.pol.min_glyph_confidence else "?"
                chars.append((" " if g["space_before"] else "") + ch)
            out_lines.append("".join(chars))
        return {"text": "\n".join(out_lines), "confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
                "by": f"glyph node {node.id}", "node": node.id, "glyphs": len(texts)}

    def tesseract_read(self, rgb: np.ndarray) -> tuple[str, list[tuple]] | None:
        """tesseract's reading and its character boxes (image coordinates), when it is installed."""
        if not self.tesseract:
            return None
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "in.png")
            Path(p).write_bytes(imaging.encode_png(rgb))
            try:
                text = subprocess.run([self.tesseract, p, "stdout", "--psm", "6"], capture_output=True, text=True,
                                      timeout=60).stdout
                boxes_raw = subprocess.run([self.tesseract, p, "stdout", "--psm", "6", "makebox"], capture_output=True,
                                           text=True, timeout=60).stdout
            except Exception as exc:
                log.warning("tesseract failed: %s", exc)
                return None
        H = rgb.shape[0]
        boxes = []
        for line in boxes_raw.splitlines():
            parts = line.split(" ")
            if len(parts) >= 5 and len(parts[0]) == 1:
                try:
                    x0, b, x1, t = (int(v) for v in parts[1:5])
                except ValueError:
                    continue
                boxes.append((parts[0], x0, H - t, x1, H - b))
        return "\n".join(x.rstrip() for x in text.strip().splitlines() if x.strip()), boxes

    async def read(self, rec: dict, rgb: np.ndarray | None = None) -> dict | None:
        """Read the text in an image: its own reader; tesseract (to compare, and to teach its glyph node)."""
        s = self.sys
        if rgb is None:
            rgb = self._workable(await asyncio.to_thread(imaging.decode, self.path_of(rec).read_bytes()))
        found, ink = await asyncio.to_thread(self.read_glyphs, rgb)
        g = imaging.gray(rgb)
        n = sum(len(line["glyphs"]) for line in found)
        rec["measures"] = {"blank": bool(g.std() < 6.0), "text_lines": len(found), "glyphs": n,
                           "text_share": round(sum(line["bottom"] - line["top"] for line in found) / max(1, rgb.shape[0]), 3)}
        if not self.pol.read_text:
            return None
        own = self.own_reading(found)
        tess = await asyncio.to_thread(self.tesseract_read, rgb) if self.tesseract and n else None
        if tess is not None:
            rec["tesseract"] = tess[0]
            taught = self.teach_from_boxes(found, tess[1])
            if taught:
                rec["learned"].append(f"tesseract labelled {taught} glyph(s) for its glyph node")
        ref = (rec.get("person") or {}).get("text") or rec.get("tesseract") or (rec.get("teacher") or {}).get("text") or ""
        if own is not None:
            if ref:
                own["agrees"] = round(similarity(own["text"], ref), 3)
                self.reading.append({"ts": round(time.time(), 1), "img": rec["id"], "agrees": own["agrees"], "node": own["node"]})
            unsure = own["confidence"] < 0.55 or own["text"].count("?") > 0.3 * max(1, len(own["text"].replace(" ", "")))
            rec["own"] = own
            if unsure and rec.get("tesseract"):  # its reader was unsure: what tesseract read, and it says so
                rec["text"] = {**own, "text": rec["tesseract"], "by": f"tesseract (its own reader, {own['by']}, was unsure: "
                               f"{own['confidence']:.0%})", "own_text": own["text"]}
            else:
                rec["text"] = own if not unsure else {**own, "by": f"{own['by']} (unsure: {own['confidence']:.0%})"}
        elif rec.get("tesseract"):
            rec["text"] = {"text": rec["tesseract"], "confidence": None, "by": "tesseract (a teacher; its own reader is still learning)",
                           "node": None, "glyphs": n}
        elif n:
            rec["text"] = {"text": "", "confidence": None, "by": "its reader is still learning to read", "node": None, "glyphs": n}
        rec["read_with"] = own["node"] if own else None
        return rec.get("text")

    def teach_from_boxes(self, found: list[dict], boxes: list[tuple]) -> int:
        """tesseract's character boxes label its glyphs where they match (most of the box shared)."""
        n = 0
        for line in found:
            for g in line["glyphs"]:
                for ch, x0, y0, x1, y1 in boxes:
                    if ch not in CHARS:
                        continue
                    ix = max(0, min(g["x1"], x1) - max(g["x0"], x0))
                    iy = max(0, min(g["y1"], y1) - max(g["y0"], y0))
                    inter = ix * iy
                    union = (g["x1"] - g["x0"]) * (g["y1"] - g["y0"]) + (x1 - x0) * (y1 - y0) - inter
                    if union > 0 and inter / union >= 0.5:
                        if self.sys.learn(GLYPHS, g["text"], label_of(ch), "tesseract"):
                            self.known[g["text"]] = ch
                            n += 1
                        break
        self.stats["glyphs from tesseract"] += n
        return n

    def learn_text(self, rec: dict) -> None:
        """What it read becomes knowledge it can answer from, with the image as its source."""
        if not self.pol.learn_text:
            return
        t = rec.get("text") or {}
        text = (rec.get("person") or {}).get("text") or t.get("text") or ""
        words = re.findall(r"[A-Za-z]{2,}", text)
        if len(words) < 3 or text.count("?") > len(text) * 0.2:
            return
        sees = (rec.get("sees") or {}).get("label") or ""
        src = "you said" if (rec.get("person") or {}).get("text") else t.get("by", "")
        doc = self.sys.knowledge.add({"source": "image", "title": f"{rec['name']} ({rec['id']})", "url": f"/images/file?id={rec['id']}",
                                      "summary": " ".join(text.split())[:300], "topic": sees,
                                      "passages": [{"section": f"text in the image ({src})", "text": " ".join(text.split())}]},
                                     reason=f"read from an image ({src})")
        rec["learned"].append(f"kept what it read as knowledge it can answer from ('{doc.title}')")
        self.stats["texts kept"] += 1

    # ------------------------------------------------------------------ people
    def correct(self, img_id: str, by: str, label: str = "", text: str | None = None) -> tuple[bool, str]:
        """A person says what the image shows, or what its text says: that stands, and teaches the nodes (P1)."""
        s = self.sys
        rec = self.records.get(img_id)
        if rec is None:
            return False, f"no image {img_id}"
        out = []
        if label:
            lab = re.sub(r"[^a-z0-9 _-]", "", label.lower()).strip().replace(" ", "_")[:40]
            if not lab:
                return False, "say what it shows in a word or two"
            s.ensure_label(SIGHT, lab, by=by)
            s.teach(SIGHT, f"image:{rec['sha']}", lab, reason=f"{by} said {img_id} shows {lab}")
            rec["person"]["label"] = lab
            rec["sees"] = {"label": lab, "confidence": 1.0, "by": f"{by} said so", "node": None}
            out.append(f"its sight node learns that {img_id} shows {lab}")
            if lab != "text" and self._remember(rec, lab, f"{by} said {img_id} shows {lab}", by):
                out.append("its memory keeps the picture")
        if text is not None and text.strip():
            rec["person"]["text"] = text.strip()
            rgb = self._workable(imaging.decode(self.path_of(rec).read_bytes()))
            found, _ = self.read_glyphs(rgb)
            want = [ln.replace(" ", "") for ln in text.strip().splitlines() if ln.strip()]
            taught = 0
            if len(want) == len(found):
                for line, chars in zip(found, want):
                    if len(chars) == len(line["glyphs"]):
                        for g, ch in zip(line["glyphs"], chars):
                            if ch in CHARS:
                                s.teach(GLYPHS, g["text"], label_of(ch), reason=f"{by} said what {img_id} says")
                                self.known[g["text"]] = ch
                                taught += 1
            out.append(f"its glyph node learns {taught} character(s) from what you said" if taught else
                       "what you said is kept (its reader found a different number of characters, so it could not "
                       "learn them one by one)")
            if rec.get("text") and rec["text"].get("node"):
                agrees = similarity(rec["text"]["text"], text)
                self.reading.append({"ts": round(time.time(), 1), "img": img_id, "agrees": round(agrees, 3),
                                     "node": rec["text"]["node"], "by": "a person"})
            self.learn_text(rec)
            if self._remember(rec, "says: " + " ".join(text.split())[:500], f"{by} said what {img_id} says", by):
                out.append("its memory keeps what the picture says")
        self._save(rec)
        self.save()
        s.bus.publish("vision.corrected", id=img_id, by=by, label=label or "", text=(text or "")[:120])
        return True, "; ".join(out) or "nothing to learn"

    def _remember(self, rec: dict, answer: str, source: str, by: str) -> bool:
        """What a person said about a picture, kept as a picture memory (memory.py; a person's word, P1)."""
        mem = getattr(self.sys, "memory", None)
        if mem is None or not mem.allows("people"):
            return False
        res = mem.learn(f"image:{rec['sha']}", answer, source=source, by=by, kind="people", realm="fact")
        return bool(res)

    # ------------------------------------------------------------------ practice
    def draw_glyphs(self, ch: str, n: int) -> list[str]:
        """Glyphs of a character drawn in its fonts, inside a few letters of context, found the way its reader
        finds them: it knows what each one is."""
        if not self.can_draw or not self.fonts or ch not in CHARS:
            return []
        out, tries = [], 0
        context = "aeHnxdgpTk0"
        while len(out) < n and tries < n * 6:
            tries += 1
            word = "".join(self.prng.choice(context) for _ in range(self.prng.randint(1, 3)))
            k = self.prng.randint(0, len(word))
            word = word[:k] + ch + word[k:] + self.prng.choice("Hg")  # an ascender and a descender set the line
            rgb = imaging.render_text(word, self.prng.choice(self.fonts), self.prng.randint(14, 40), self.rng)
            if rgb is None:
                continue
            found, _ = self.read_glyphs(rgb)
            if len(found) != 1 or len(found[0]["glyphs"]) != len(word):
                continue
            t = found[0]["glyphs"][k]["text"]
            self.known[t] = ch
            out.append(t)
        while len(self.known) > 60000:
            self.known.pop(next(iter(self.known)))
        return out

    def draw_practice(self, counts: dict, n: int | None = None) -> list[tuple[str, str]]:
        """Glyphs of the characters it has fewest examples of (drawn; no learning yet: see practice)."""
        target = max(1, (n or self.pol.glyph_practice))
        order = sorted(CHARS, key=lambda c: (counts.get(label_of(c), 0), self.prng.random()))
        per = max(2, target // 20)
        return [(t, ch) for ch in order[:20] for t in self.draw_glyphs(ch, per)]

    async def practice(self, n: int | None = None) -> int:
        """Draw glyphs of the characters it has fewest examples of, and learn them."""
        s = self.sys
        if not self.can_draw or not self.fonts or GLYPHS not in s.capabilities:
            return 0
        counts = s.store.label_counts(GLYPHS)
        drawn = await asyncio.to_thread(self.draw_practice, counts, n)
        made = sum(1 for t, ch in drawn if s.learn(GLYPHS, t, label_of(ch), "glyphs"))
        order = list(dict.fromkeys(ch for _, ch in drawn))
        self.stats["glyphs practised"] += made
        if made:
            self.log.append({"ts": round(time.time(), 1), "line": f"practised reading: drew {made} glyph(s) of "
                             f"{', '.join(repr(c) for c in order[:6])}..."})
            self.save()
        return made

    async def reread(self) -> dict | None:
        """Read again an image it read with an older reader (or could not read yet)."""
        node = self.glyph_node()
        if node is None:
            return None
        stale = [r for r in self.records.values() if (r.get("measures") or {}).get("glyphs") and r.get("read_with") != node.id]
        if not stale:
            return None
        rec = min(stale, key=lambda r: r.get("ts", 0))
        before = (rec.get("text") or {}).get("text", "")
        await self.read(rec)
        self.learn_text(rec)
        self._save(rec)
        self.stats["re-read"] += 1
        self.save()
        after = (rec.get("text") or {}).get("text", "")
        self.sys.bus.publish("vision.reread", id=rec["id"], node=node.id, changed=before != after,
                             agrees=(rec.get("text") or {}).get("agrees"))
        return rec

    # ------------------------------------------------------------------ answering about images
    def answer(self, text: str) -> tuple[str, str] | None:
        """Questions about an image it took in: "what do you see in img-3?", "what does img-3 say?"."""
        m = IMG_RE.search(text)
        if not m:
            return None
        rec = self.records.get(m.group(0).lower())
        if rec is None:
            return f"I have no image called {m.group(0)}.", "no such image"
        sees = rec.get("sees") or {}
        t = rec.get("text") or {}
        words = (rec.get("person") or {}).get("text") or t.get("text") or ""
        low = text.lower()
        if re.search(r"\b(say|says|read|text|written|words)\b", low):
            if not words.strip():
                return f"I couldn't read any text in {rec['id']} ({rec['name']}).", "no text read"
            src = "you told me" if (rec.get("person") or {}).get("text") else t.get("by", "")
            return f"{rec['id']} says: {' '.join(words.split())[:600]} (read by {src})", f"read by {src}"
        if sees.get("label"):
            extra = f" I can read {len(words.split())} word(s) in it too." if words.strip() else ""
            return (f"{rec['id']} ({rec['name']}) shows {sees['label'].replace('_', ' ')} ({sees['by']}"
                    + (f", {sees['confidence']:.0%} sure" if sees.get("confidence") and sees["confidence"] < 1 else "") + f").{extra}",
                    sees["by"])
        return f"I don't know yet what {rec['id']} shows - I asked you.", "not known yet"

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        s = self.sys
        for r in self.records.values():  # a person answered its question about an image (/answer, the web page)
            if not (r.get("sees") or {}).get("label"):
                lab = s.teachers.human.answers.get((SIGHT, f"image:{r['sha']}"))
                if lab:
                    r["sees"] = {"label": lab, "confidence": 1.0, "by": "you said so", "node": None}
                    r.setdefault("person", {})["label"] = lab
                    self._save(r)
        recs = sorted(self.records.values(), key=lambda r: -r.get("ts", 0))
        reads = [r["agrees"] for r in self.reading if r.get("agrees") is not None]
        gnode = s.registry.champion(GLYPHS) if GLYPHS in s.capabilities else None
        snode = s.registry.champion(SIGHT) if SIGHT in s.capabilities else None
        st = self.seeing_teacher()
        return {"enabled": self.enabled, "images": recs[:60], "count": len(self.records),
                "reader": {"node": gnode.id if gnode else None, "accuracy": gnode.metrics.get("holdout_acc") if gnode else None,
                           "validated": self.glyph_node() is not None, "examples": s.store.count(GLYPHS),
                           "agrees": round(float(np.mean(reads[-30:])), 3) if reads else None, "readings": len(reads)},
                "sight": {"node": snode.id if snode else None, "accuracy": snode.metrics.get("holdout_acc") if snode else None,
                          "labels": list(s.capabilities[SIGHT].active_labels()) if SIGHT in s.capabilities else [],
                          "examples": s.store.count(SIGHT)},
                "teachers": {"tesseract": self.tesseract, "vision": f"{st.name}: {getattr(st, 'vision', '') or st.model}" if st else None,
                             "fonts": len(self.fonts), "can_draw": self.can_draw},
                "stats": dict(self.stats), "log": list(self.log)[-10:][::-1]}

    def lines(self) -> list[str]:
        st = self.status()
        r = st["reader"]
        out = [f"Images: {st['count']} taken in. Sight: {st['sight']['examples']} lessons, labels "
               f"{', '.join(st['sight']['labels'][:12]) or 'none yet'}"
               + (f"; node {st['sight']['node']} {st['sight']['accuracy']:.0%}" if st['sight']['node'] and st['sight']['accuracy'] is not None else "") + ".",
               f"Reader: {r['examples']} glyphs learned" + (f"; glyph node {r['node']} {r['accuracy']:.0%} on held-out glyphs"
                                                           if r["node"] and r["accuracy"] is not None else "; no glyph node yet")
               + ("" if r["validated"] else " (still learning: tesseract or a teacher reads for it)")
               + (f"; its readings agree {r['agrees']:.0%} with tesseract, a teacher or you" if r["agrees"] is not None else "") + ".",
               f"Teachers: tesseract {'at ' + st['teachers']['tesseract'] if st['teachers']['tesseract'] else 'not installed'}; "
               f"seeing teacher {st['teachers']['vision'] or 'none (ollama pull llava)'}; {st['teachers']['fonts']} fonts to practise with"
               + ("" if st["teachers"]["can_draw"] else " (install Pillow to draw them)") + "."]
        for rec in st["images"][:10]:
            sees = (rec.get("sees") or {}).get("label") or "?"
            words = len(((rec.get("person") or {}).get("text") or (rec.get("text") or {}).get("text") or "").split())
            out.append(f"  {rec['id']:<8} {rec['name'][:30]:<30} {rec['w']}x{rec['h']:<6} shows {sees}; {words} word(s) read")
        return out


class Looker(Agent):
    """When idle: it practises reading glyphs it draws, and re-reads images once its reader has improved."""
    name = "looker"

    @property
    def interval(self) -> float:
        return 3.0

    async def step(self) -> None:
        s = self.sys
        v = s.vision
        if not v.enabled or not s.guardian.background_allowed() or s.inflight:
            return
        if time.time() - (s.last_order_at or s.started_at) < v.pol.practice_after_idle_seconds:
            return
        counts = s.store.label_counts(GLYPHS) if GLYPHS in s.capabilities else {}
        if v.can_draw and v.fonts and min((counts.get(label_of(c), 0) for c in CHARS), default=0) < v.pol.glyphs_per_label:
            await s.background_slot()
            made = await v.practice()
            self.stats["practised"] += made or 0
            return
        rec = await v.reread()
        if rec is not None:
            self.stats["re-read"] += 1
