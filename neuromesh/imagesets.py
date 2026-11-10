"""Picture books: image datasets from the Hugging Face Hub, a teacher for its optical nodes (P21).

    a picture book (it gets the ones in [imagesets] start_with by itself; more from the Images view, /pictures get, or
    the API - the ready-made ones below, or any image dataset on the Hub, whose columns it guesses and you can change)
      --> rows from the Hugging Face dataset viewer (/rows): each an image - downloaded from the viewer's assets, which
          serves JPEG (imaging.py reads it, with or without Pillow) - and what it shows or the text in it
      --> kept in state/imagesets/<name>/: the images by their contents, items.jsonl, and what its sight node takes in
          for each (state/imagesets/features/)
      --> split once, stably, into practice (it learns from these) and exam (it is graded on these, never taught them,
          so an exam score is honest and comparable from exam to exam)

    three kinds of book:
      objects     what an image shows (CIFAR-10, CIFAR-100, Fashion-MNIST): its sight node learns the labels
      characters  one character an image (MNIST's handwritten digits): its reader finds the character's ink the way
                  it finds it on a page, and its glyph node learns what it is
      text        a printed word or line an image: where its reader finds as many characters as the text has, each
                  glyph is labelled for its glyph node; its reading is graded against the text

    a lesson (the Picture-book agent, when no one has given it an order for a while, or Practise now):
      practice    the next practice images of a book: its nodes learn them (as the "imagesets" teacher: weighted as
                  a set's gold answer, below a person). The same teacher answers for practice images when the Trainer
                  wants more examples of a label
      memory      what it is shown is merged into its memory nodes as picture memories (memory.py): the first images of
                  every label, and each practice image its sight node got wrong, are kept as memories its memory nodes
                  must recall (tested like any fact). When it looks at an image, memory is asked first: a picture it
                  remembers is named by the memory node that holds it
      exam        every [imagesets] exam_every_lessons lessons of a book: its sight node (or glyph node, or reader) is
                  graded on exam images, with chance level beside the score

What it learned names where it came from: the book, its dataset and split. Nothing in a book is run; images are only
decoded. A book can be removed (its images, features and practice examples go; picture memories are kept until you
remove them from memory).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import imaging
from .agents import Agent
from .challenges import _UA, _feature, _get_json, hub_search
from .teachers import Lesson, Teacher
from .vision import CHARS, GLYPHS, SIGHT, label_of, similarity

log = logging.getLogger("neuromesh.imagesets")

TEACHER = "imagesets"
KINDS = ("objects", "characters", "text")
SAYS = "says: "   # a picture memory of the text in an image: "says: <the text>"
_TEXT_COLS = ("text", "label", "transcription", "ground_truth", "word", "words", "ocr", "line", "sentence", "caption_text")
_ANSWER_COLS = ("label", "labels", "fine_label", "class", "category", "target", "digit", "letter")

BOOKS = {
    "cifar10": {"dataset": "uoft-cs/cifar10", "config": "plain_text", "split": "train", "image": "img", "answer": "label",
                "kind": "objects", "license": "unknown (free to use, Krizhevsky 2009)",
                "description": "small colour photos of ten things: airplanes, cars, birds, cats, deer, dogs, frogs, horses, "
                               "ships and trucks"},
    "cifar100": {"dataset": "uoft-cs/cifar100", "config": "cifar100", "split": "train", "image": "img", "answer": "fine_label",
                 "kind": "objects", "license": "unknown (free to use, Krizhevsky 2009)",
                 "description": "small colour photos of a hundred kinds of thing, from apples and bears to tractors and worms"},
    "fashion_mnist": {"dataset": "zalando-datasets/fashion_mnist", "config": "fashion_mnist", "split": "train",
                      "image": "image", "answer": "label", "kind": "objects", "license": "mit",
                      "description": "clothes and shoes in grey: T-shirts, trousers, pullovers, dresses, coats, sandals, "
                                     "shirts, sneakers, bags and ankle boots"},
    "mnist": {"dataset": "ylecun/mnist", "config": "mnist", "split": "train", "image": "image", "answer": "label",
              "kind": "characters", "license": "mit",
              "description": "handwritten digits 0 to 9: its reader learns to read handwriting"},
    "words": {"dataset": "priyank-m/trdg_dict_random_words_en_text_recognition", "config": "default", "split": "train",
              "image": "image", "answer": "label", "kind": "text", "license": "",
              "description": "printed words in many fonts and backgrounds (TextRecognitionDataGenerator): its reader learns "
                             "the characters it finds, and is graded on reading them"},
    "wiki_lines": {"dataset": "priyank-m/trdg_wikipedia_en_text_recognition", "config": "default", "split": "train",
                   "image": "image", "answer": "label", "kind": "text", "license": "",
                   "description": "printed lines of Wikipedia text, for reading whole lines"},
}


def label_name(name: str) -> str:
    """A class name as a sight label: "T - shirt / top" -> "t_shirt_top", "Ankle boot" -> "ankle_boot"."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).lower())).strip("_")[:40]


def book_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", str(name).lower()).strip("_-")[:40]


def guess_columns(features: list[dict], rows: list[dict]) -> dict:
    """Which column is the image, which says what it shows or what it says, and what kind of book that makes."""
    images = [f["name"] for f in features if f.get("kind") == "image"]
    if not images:
        return {"image": "", "answer": "", "kind": "", "why": "it has no image column"}
    by = {f["name"]: f for f in features}
    classes = [f["name"] for f in features if f.get("kind") == "class"]
    strings = [f["name"] for f in features if f.get("kind") == "string"]
    pick = next((c for want in _ANSWER_COLS for c in classes if c.lower() == want), classes[0] if classes else "")
    if pick:
        names = by[pick].get("labels") or []
        kind = "characters" if names and all(len(str(n)) == 1 and str(n) in CHARS for n in names) else "objects"
        return {"image": images[0], "answer": pick, "kind": kind, "why": f"'{pick}' names what each image shows"}
    text = next((c for want in _TEXT_COLS for c in strings if c.lower() == want), "")
    if text:
        vals = [str(r.get(text) or "") for r in rows[:30]]
        if vals and all(len(v) == 1 and v in CHARS for v in vals if v):
            return {"image": images[0], "answer": text, "kind": "characters", "why": f"'{text}' is one character an image"}
        return {"image": images[0], "answer": text, "kind": "text", "why": f"'{text}' is the text in each image"}
    return {"image": images[0], "answer": "", "kind": "", "why": "no column says what the images show or what they say"}


def _src(cell) -> str:
    if isinstance(cell, dict):
        return str(cell.get("src") or cell.get("url") or "")
    if isinstance(cell, str) and cell.startswith(("http://", "https://")):
        return cell
    return ""


# ============================================================================ the teacher
class ImageSetTeacher(Teacher):
    """A picture book's own answer, for its sight node: what a practice image shows (exam images are never answered)."""
    name = TEACHER
    specialist = True

    def __init__(self, books: "ImageSets"):
        self.books = books

    def available(self) -> bool:
        return self.books.enabled and bool(self.books.index)

    def status(self) -> dict:
        return {"detail": f"specialist: {len(self.books.books)} picture book(s) from the Hugging Face Hub, "
                          f"{len(self.books.index)} practice images it can name"}

    async def label(self, spec, text):
        if spec.name != SIGHT or not text.startswith("image:"):
            return None
        hit = self.books.index.get(text[6:].strip())
        if hit is None:
            return None
        book, answer = hit
        return Lesson(answer, 0.97, self.name, f"picture book {book}")

    async def generate(self, spec, label, n):
        if spec.name != SIGHT:
            return []
        out = [f"image:{sha}" for sha, (_, answer) in self.books.index.items() if answer == label]
        random.shuffle(out)
        return out[:n]


# ============================================================================ the books
class ImageSets:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.imagesets
        self.dir = system.state_dir / "imagesets"
        self.features_dir = self.dir / "features"
        self.books: dict[str, dict] = {}
        self.items: dict[str, list[dict]] = {}
        self.index: dict[str, tuple[str, str]] = {}   # sha -> (book, answer): practice images of objects books
        raw = system.store.get_kv("imagesets.state")
        st = json.loads(raw) if raw else {}
        self.state: dict[str, dict] = st.get("books", {})
        self.stats: Counter = Counter(st.get("stats", {}))
        self.log: deque = deque(st.get("log", []), maxlen=80)
        self.tried: dict[str, float] = st.get("tried", {})     # start_with books it tried to get, and when
        self.getting: dict[str, dict] = {}                     # downloads under way: progress
        self.errors: dict[str, str] = {}
        self.busy: set[str] = set()                            # books in a lesson right now
        self.rng = random.Random()
        self._lock = threading.Lock()
        self.load()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled) and self.sys.directives.vision.enabled

    # ------------------------------------------------------------------ storage
    def load(self) -> None:
        self.books.clear()
        self.items.clear()
        if not self.dir.is_dir():
            self._reindex()
            return
        for folder in sorted(p for p in self.dir.iterdir() if (p / "manifest.json").exists()):
            try:
                book = json.loads((folder / "manifest.json").read_text())
                items = [json.loads(line) for line in (folder / "items.jsonl").read_text().splitlines() if line.strip()]
            except (OSError, ValueError) as exc:
                self.errors[folder.name] = str(exc)
                continue
            self.books[book["name"]] = book
            self.items[book["name"]] = items
        self._reindex()

    def _reindex(self) -> None:
        self.index = {it["sha"]: (name, it["answer"]) for name, b in self.books.items() if b["kind"] == "objects"
                      for it in self.items.get(name, []) if it["split"] == "practice"}

    def save_state(self) -> None:
        self.sys.store.set_kv("imagesets.state", json.dumps({"books": self.state, "stats": dict(self.stats),
                                                             "log": list(self.log), "tried": self.tried}))

    def _note(self, line: str, **extra) -> None:
        self.log.append({"ts": round(time.time(), 1), "line": line, **extra})
        self.save_state()

    def folder(self, name: str) -> Path:
        return self.dir / name

    def image_path(self, name: str, item: dict) -> Path:
        return self.folder(name) / "images" / f"{item['sha']}.{item['ext']}"

    def _save_book(self, book: dict, items: list[dict]) -> None:
        f = self.folder(book["name"])
        f.mkdir(parents=True, exist_ok=True)
        book["items"] = len(items)
        (f / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
        (f / "manifest.json").write_text(json.dumps(book, indent=1))
        self.books[book["name"]] = book
        self.items[book["name"]] = items
        self._reindex()

    def split_of(self, name: str, sha: str) -> str:
        h = int(hashlib.sha256(f"{name}:{sha}".encode()).hexdigest()[:8], 16)
        return "exam" if h % 100 < self.pol.exam_percent else "practice"

    def practice(self, name: str) -> list[dict]:
        return [it for it in self.items.get(name, []) if it["split"] == "practice"]

    def exam_items(self, name: str) -> list[dict]:
        return [it for it in self.items.get(name, []) if it["split"] == "exam"]

    def has_features(self, sha: str) -> Path | None:
        p = self.features_dir / f"{sha}.npy"
        return p if p.exists() else None

    def rgb(self, name: str, item: dict) -> np.ndarray:
        return imaging.decode(self.image_path(name, item).read_bytes())

    # ------------------------------------------------------------------ the catalog and the Hub
    def catalog(self) -> list[dict]:
        return [{"name": k, **{x: b[x] for x in ("dataset", "config", "split", "kind", "description", "license")},
                 "installed": k in self.books, "getting": k in self.getting} for k, b in BOOKS.items()]

    def _api(self) -> tuple[str, str | None]:
        c = self.sys.directives.challenges
        import os
        return c.api_url, (os.environ.get(c.token_env) or None) if c.token_env else None

    def search(self, query: str) -> list[dict]:
        """Datasets on the Hub matching a search (look at one to see whether it has images)."""
        if not query.strip():
            return []
        c = self.sys.directives.challenges
        found = hub_search(query, hub=c.hub_url, token=self._api()[1])
        for d in found:
            d["book"] = next((k for k, b in BOOKS.items() if b["dataset"].lower() == d["id"].lower()), "")
            d["images"] = any(t in ("image-classification", "image-to-text", "optical-character-recognition",
                                    "object-detection", "image-feature-extraction") for t in d.get("tasks", []))
        return found

    def inspect(self, dataset: str, config: str = "", split: str = "") -> dict:
        """Its subsets and splits, its columns, its first images, and a guess at which column says what."""
        if not re.fullmatch(r"[\w.-]+(/[\w.-]+)?", dataset or ""):
            raise ValueError("a dataset is named like owner/name")
        api, token = self._api()
        base = api.rstrip("/")
        splits = _get_json(f"{base}/splits?{urllib.parse.urlencode({'dataset': dataset})}", token).get("splits") or []
        pairs = [{"config": s["config"], "split": s["split"]} for s in splits if isinstance(s, dict) and s.get("split")]
        if not pairs:
            raise RuntimeError(f"{dataset} has no splits the dataset viewer can read")
        pick = next((p for p in pairs if (not config or p["config"] == config) and (not split or p["split"] == split)), None)
        if pick is None:
            raise RuntimeError(f"{dataset} has no {config or '?'}/{split or '?'}")
        if not split:
            pick = next((p for want in ("train", "test", "validation") for p in pairs
                         if p["split"] == want and (not config or p["config"] == config)), pick)
        first = _get_json(f"{base}/first-rows?{urllib.parse.urlencode({'dataset': dataset, **pick})}", token)
        features = []
        for f in first.get("features") or []:
            kind, names = _feature(f.get("type"))
            features.append({"name": f.get("name"), "kind": kind, "labels": names})
        rows = [r["row"] for r in first.get("rows") or [] if isinstance(r, dict) and isinstance(r.get("row"), dict)]
        guess = guess_columns(features, rows)
        by = {f["name"]: f for f in features}
        sample = []
        for r in rows[:12]:
            ans = r.get(guess["answer"]) if guess["answer"] else None
            names = (by.get(guess["answer"]) or {}).get("labels") or []
            if isinstance(ans, int) and 0 <= ans < len(names):
                ans = names[ans]
            sample.append({"src": _src(r.get(guess["image"])) if guess["image"] else "", "answer": "" if ans is None else str(ans)[:120]})
        return {"dataset": dataset, **pick, "splits": pairs, "features": features, "guess": guess, "sample": sample,
                "book": next((k for k, b in BOOKS.items() if b["dataset"].lower() == dataset.lower()), "")}

    # ------------------------------------------------------------------ getting a book
    async def get(self, name: str = "", dataset: str = "", image: str = "", answer: str = "", kind: str = "",
                  config: str = "", split: str = "", limit: int = 0, more: bool = False, by: str = "operator") -> dict:
        """Get a ready-made book, more of a book it has, or any image dataset read as the columns say."""
        if not self.enabled:
            raise ValueError("picture books are off ([imagesets] enabled, or [vision] enabled)")
        if name in BOOKS and not dataset:
            spec = {"name": name, **BOOKS[name]}
        elif more and name in self.books:
            b = self.books[name]
            spec = {k: b[k] for k in ("name", "dataset", "config", "split", "image", "answer", "kind", "description", "license")}
        else:
            if not dataset:
                raise ValueError(f"no picture book '{name}' (ready-made: {', '.join(BOOKS)}), and no dataset named")
            info = await asyncio.to_thread(self.inspect, dataset, config, split)
            g = info["guess"]
            spec = {"name": book_name(name or dataset.split("/")[-1]), "dataset": dataset, "config": info["config"],
                    "split": info["split"], "image": image or g["image"], "answer": answer or g["answer"],
                    "kind": kind or g["kind"], "description": "", "license": ""}
            names = {f["name"] for f in info["features"]}
            if spec["image"] not in names or spec["answer"] not in names:
                raise ValueError(f"say which column is the image and which says what it shows ({g['why']})")
        if spec["kind"] not in KINDS:
            raise ValueError(f"a book's kind is one of {', '.join(KINDS)}")
        if spec["name"] in self.getting:
            raise ValueError(f"it is already getting {spec['name']}")
        if spec["name"] in self.books and not more and self.books[spec["name"]]["dataset"] != spec["dataset"]:
            raise ValueError(f"there is already a book called {spec['name']}")
        limit = max(1, min(int(limit or self.pol.fetch_limit), self.pol.max_items))
        self.getting[spec["name"]] = {"done": 0, "of": limit, "skipped": 0, "started": time.time()}
        try:
            with self.sys.activity.track("pictures", "picture book", f"getting {spec['name']} ({spec['dataset']}): {limit} images"):
                book, added, skipped = await asyncio.to_thread(self._get, spec, limit, by)
        except Exception as exc:
            self.errors[spec["name"]] = str(exc)
            self._note(f"could not get {spec['name']} ({spec['dataset']}): {exc}", book=spec["name"])
            raise
        finally:
            self.getting.pop(spec["name"], None)
        self.errors.pop(spec["name"], None)
        self.state.setdefault(book["name"], {"next": 0, "lessons": 0, "learned": 0, "memories": 0, "exams": []})
        self.stats["images got"] += added
        line = (f"got {added} image(s) of {book['name']} ({book['dataset']} {book['config']}/{book['split']}, {book['kind']})"
                + (f"; {skipped} skipped" if skipped else "") + f": {book['items']} in all")
        self._note(line, book=book["name"])
        self.sys.guardian.audit(by, "imagesets.get", book["name"], dataset=book["dataset"], images=added)
        self.sys.bus.publish("imagesets.got", book=book["name"], dataset=book["dataset"], images=added, kind=book["kind"], by=by)
        return {"book": self.summary(book["name"]), "added": added, "skipped": skipped, "line": line}

    def _rows(self, spec: dict, offset: int, length: int) -> dict:
        api, token = self._api()
        params = urllib.parse.urlencode({"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"],
                                         "offset": offset, "length": length})
        for attempt in range(3):
            try:
                return _get_json(f"{api.rstrip('/')}/rows?{params}", token, timeout=60)
            except RuntimeError as exc:
                if attempt == 2 or not re.search(r"HTTP (429|5\d\d)|could not reach", str(exc)):
                    raise
                time.sleep(2 * (attempt + 1))
        return {}

    def _fetch_image(self, url: str) -> bytes:
        api, token = self._api()
        host = urllib.parse.urlsplit(url).netloc.lower()
        auth = token and (host.endswith("huggingface.co") or host == urllib.parse.urlsplit(api).netloc.lower())
        req = urllib.request.Request(url, headers={"User-Agent": _UA, **({"Authorization": f"Bearer {token}"} if auth else {})})
        cap = int(self.pol.max_image_kb * 1024)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read(cap + 1)
                if len(data) > cap:
                    raise ValueError(f"larger than {self.pol.max_image_kb} KB")
                return data
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def _get(self, spec: dict, limit: int, by: str) -> tuple[dict, int, int]:
        """Rows, then their images (several at once), decoded, measured and kept. Network and disk only: a thread."""
        name = spec["name"]
        old = self.books.get(name)
        items = list(self.items.get(name, [])) if old else []
        have = {it["sha"] for it in items}
        offset = int((old or {}).get("offset", 0))
        progress = self.getting.get(name, {})
        labels: list = list((old or {}).get("labels") or [])
        added = skipped = 0
        (self.folder(name) / "images").mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        total = None
        while added < limit:
            want = min(100, limit - added + 10)
            data = self._rows(spec, offset, want)
            if total is None:
                total = int(data.get("num_rows_total") or 0)
            feats = {f.get("name"): f.get("type") for f in data.get("features") or []}
            if spec["answer"] in feats and not labels:
                labels = _feature(feats[spec["answer"]])[1]
            rows = [r for r in data.get("rows") or [] if isinstance(r, dict) and isinstance(r.get("row"), dict)]
            if not rows:
                break
            offset += len(rows)
            todo = []
            for r in rows:
                row = r["row"]
                src = _src(row.get(spec["image"]))
                ans = row.get(spec["answer"])
                if isinstance(ans, int) and labels and 0 <= ans < len(labels):
                    ans = labels[ans]
                ans = "" if ans is None else " ".join(str(ans).split())
                if spec["kind"] == "objects":
                    ans = label_name(ans)
                elif spec["kind"] == "characters":
                    ans = ans if len(ans) == 1 and ans in CHARS else ""
                else:
                    ans = ans[:200]
                if not src or not ans:
                    skipped += 1
                    continue
                todo.append((str(r.get("row_idx", offset)), src, ans))
            with ThreadPoolExecutor(max_workers=max(1, self.pol.download_workers)) as pool:
                got = list(pool.map(lambda t: self._take(name, t), todo))
            for res in got:
                if res is None:
                    skipped += 1
                    continue
                if res["sha"] in have:
                    continue
                have.add(res["sha"])
                res["split"] = self.split_of(name, res["sha"])
                items.append(res)
                added += 1
                if added >= limit:
                    break
            progress["done"], progress["skipped"] = added, skipped
            if total and offset >= total:
                break
        if not items:
            raise RuntimeError(f"no usable images in {spec['dataset']} {spec['config']}/{spec['split']}"
                               + (f" ({skipped} could not be read)" if skipped else ""))
        book = {**{k: spec[k] for k in ("name", "dataset", "config", "split", "image", "answer", "kind", "description", "license")},
                "labels": [label_name(x) if spec["kind"] == "objects" else x for x in labels], "offset": offset,
                "source": f"huggingface:{spec['dataset']} {spec['config']}/{spec['split']}", "by": (old or {}).get("by", by),
                "got": round(time.time(), 1), "total": total or 0}
        with self._lock:
            self._save_book(book, items)
        return book, added, skipped

    def _take(self, name: str, todo: tuple) -> dict | None:
        """One image: downloaded, decoded, its features worked out, kept by its contents."""
        row_id, src, answer = todo
        try:
            data = self._fetch_image(src)
            kind = imaging.sniff(data)
            if not kind:
                return None
            rgb = imaging.decode(data)
        except Exception as exc:
            log.debug("image %s of %s skipped: %s", row_id, name, exc)
            return None
        sha = hashlib.sha256(data).hexdigest()[:20]
        path = self.folder(name) / "images" / f"{sha}.{kind}"
        if not path.exists():
            path.write_bytes(data)
        fp = self.features_dir / f"{sha}.npy"
        if not fp.exists():
            np.save(fp, imaging.features(rgb))
        h, w = rgb.shape[:2]
        return {"id": row_id, "sha": sha, "ext": kind, "answer": answer, "w": int(w), "h": int(h)}

    def remove(self, name: str, by: str) -> tuple[bool, str]:
        if name not in self.books:
            return False, f"no picture book {name}"
        items = self.items.pop(name, [])
        book = self.books.pop(name)
        others = {it["sha"] for n, its in self.items.items() for it in its}
        for it in items:
            if it["sha"] not in others:
                (self.features_dir / f"{it['sha']}.npy").unlink(missing_ok=True)
        shutil.rmtree(self.folder(name), ignore_errors=True)
        gone, texts = 0, [f"image:{it['sha']}" for it in items]  # what its sight node learned from the book goes too
        for i in range(0, len(texts), 400):
            part = texts[i:i + 400]
            gone += self.sys.store.db.execute(f"DELETE FROM examples WHERE capability = ? AND source = ? AND text IN "
                                              f"({','.join('?' * len(part))})", (SIGHT, TEACHER, *part)).rowcount or 0
        self.state.pop(name, None)
        self._reindex()
        self._note(f"removed {name} ({book['dataset']}): {len(items)} images", book=name)
        self.sys.guardian.audit(by, "imagesets.remove", name, images=len(items))
        self.sys.bus.publish("imagesets.removed", book=name, by=by)
        return True, f"removed {name}: {len(items)} images" + (f", {gone} practice examples" if gone else "")

    # ------------------------------------------------------------------ lessons
    def _due_book(self) -> str | None:
        names = [n for n in self.books if self.practice(n)]
        if not names:
            return None
        return min(names, key=lambda n: (self.state.get(n, {}).get("lessons", 0), n))

    def _champion(self, cap: str):
        node = self.sys.registry.champion(cap) if cap in self.sys.capabilities else None
        return node if node is not None and node.params is not None else None

    def _predict(self, node, texts: list[str]) -> list[tuple[str, float]]:
        if node is None or not texts:
            return []
        P = node.predict_proba(self.sys.encode_many(texts))
        idx = P.argmax(axis=1)
        return [(node.labels[i], float(P[k, i])) for k, i in enumerate(idx)]

    def _glyphs_of(self, name: str, item: dict) -> list[dict]:
        rgb = self.rgb(name, item)
        found, _ = self.sys.vision.read_glyphs(rgb)
        return found

    async def lesson(self, name: str | None = None, n: int | None = None, by: str = "its picture books") -> dict | None:
        """The next practice images of a book: its nodes learn them, and picture memories are kept."""
        s = self.sys
        name = name or self._due_book()
        if not name or name not in self.books or name in self.busy:
            return None
        book = self.books[name]
        st = self.state.setdefault(name, {"next": 0, "lessons": 0, "learned": 0, "memories": 0, "exams": []})
        items = self.practice(name)
        if not items:
            return None
        n = max(1, int(n or self.pol.lesson_size))
        start = st["next"] % len(items)
        batch = (items[start:] + items[:start])[:n]
        self.busy.add(name)
        try:
            with s.activity.track("pictures", "picture book", f"a lesson from {name}: {len(batch)} images"):
                res = await asyncio.to_thread(self._lesson, book, batch, dict(st.get("first", {})))
            learned = 0
            for cap, text, label, char in res.pop("teach"):  # the store belongs to this thread
                if s.learn(cap, text, label, TEACHER):
                    learned += 1
                    if char:
                        s.vision.known[text] = char
            mem = 0
            for item, answer in res["remember"]:
                if self.remember(book, item, answer):
                    mem += 1
        finally:
            self.busy.discard(name)
        st["next"] = (start + len(batch)) % len(items)
        st["rounds"] = st.get("rounds", 0) + (1 if start + len(batch) >= len(items) else 0)
        st["first"] = res.pop("first", st.get("first", {}))
        st["lessons"] += 1
        st["learned"] += learned
        st["memories"] += mem
        res.update({"learned": learned, "memories": mem, "book": name})
        self.stats["lessons"] += 1
        self.stats["images learned"] += learned
        self.stats["picture memories"] += mem
        line = f"a lesson from {name}: {res['line']}; learned {learned}" + (f"; {mem} kept as picture memories" if mem else "")
        self._note(line, book=name, by=by)
        s.bus.publish("imagesets.lesson", book=name, learned=learned, memories=mem, by=by, right=res.get("right"), of=res.get("of"))
        res["line"] = line
        res["remember"] = len(res["remember"])
        if self.pol.exam_every_lessons and st["lessons"] % self.pol.exam_every_lessons == 0:
            res["exam"] = await self.exam(name, by=by)
        else:
            self.save_state()
        return res

    def _lesson(self, book: dict, batch: list[dict], firsts: dict) -> dict:
        """What a lesson teaches (worked out here, in a thread; the nodes are taught on the mesh's own thread)."""
        name, kind = book["name"], book["kind"]
        teach, remember = [], []
        if kind == "objects":
            node = self._champion(SIGHT)
            texts = [f"image:{it['sha']}" for it in batch]
            before = self._predict(node, texts)
            right = sum(1 for (lab, _), it in zip(before, batch) if lab == it["answer"])
            first, misses = Counter(firsts), 0
            for k, it in enumerate(batch):
                teach.append((SIGHT, texts[k], it["answer"], None))
                if first[it["answer"]] < self.pol.memory_per_label:  # the first pictures of every label
                    first[it["answer"]] += 1
                    remember.append((it, it["answer"]))
                elif before and before[k][0] != it["answer"] and misses < self.pol.memory_misses:  # what it got wrong
                    misses += 1
                    remember.append((it, it["answer"]))
            line = (f"its sight node {node.id} knew {right} of {len(batch)} before the lesson" if before
                    else f"{len(batch)} images (its sight node is not grown yet)")
            return {"teach": teach, "remember": remember, "first": dict(first), "line": line,
                    "right": right if before else None, "of": len(batch)}
        if kind == "characters":
            fitted = 0
            for it in batch:
                try:
                    found = self._glyphs_of(name, it)
                except Exception:
                    continue
                glyphs = [g for line_ in found for g in line_["glyphs"]]
                if len(glyphs) == 1:
                    fitted += 1
                    teach.append((GLYPHS, glyphs[0]["text"], label_of(it["answer"]), it["answer"]))
            self.stats["characters found"] += fitted
            return {"teach": teach, "remember": [], "of": len(batch),
                    "line": f"its reader found one character in {fitted} of {len(batch)} images"}
        node = self._champion(GLYPHS)
        lined, agree = 0, []
        for it in batch:
            try:
                found = self._glyphs_of(name, it)
            except Exception:
                continue
            chars = "".join(it["answer"].split())
            glyphs = [g for line_ in found for g in line_["glyphs"]]
            if node is not None and glyphs:
                own = self.sys.vision.own_reading(found, node=node, guess=True)
                if own is not None:
                    sim = similarity(own["text"].replace("\n", " "), it["answer"])
                    agree.append(sim)
                    if sim < 0.5 and len(remember) < self.pol.memory_misses:  # what it could not read
                        remember.append((it, SAYS + it["answer"]))
            if len(found) == 1 and len(glyphs) == len(chars):
                lined += 1
                teach += [(GLYPHS, g["text"], label_of(ch), ch) for g, ch in zip(glyphs, chars) if ch in CHARS]
        self.stats["text lined up"] += lined
        line = (f"its reader found exactly as many characters as the text has in {lined} of {len(batch)} images"
                + (f"; its reading agreed {np.mean(agree):.0%} with the text" if agree else ""))
        return {"teach": teach, "remember": remember, "line": line, "of": len(batch),
                "agrees": round(float(np.mean(agree)), 3) if agree else None}

    async def exam(self, name: str, n: int | None = None, by: str = "its picture books") -> dict | None:
        """Grade its node on exam images of a book (never learned from): the score, and chance beside it."""
        if name not in self.books:
            return None
        book = self.books[name]
        items = sorted(self.exam_items(name), key=lambda it: it["sha"])[: max(1, int(n or self.pol.exam_size))]
        if not items:
            return None
        with self.sys.activity.track("pictures", "picture book", f"an exam from {name}: {len(items)} images"):
            res = await asyncio.to_thread(self._exam, book, items)
        if res is None:
            return None
        res.update({"ts": round(time.time(), 1), "by": by, "book": name})
        st = self.state.setdefault(name, {"next": 0, "lessons": 0, "learned": 0, "memories": 0, "exams": []})
        st["exams"] = (st.get("exams", []) + [res])[-40:]
        self.stats["exams"] += 1
        line = f"an exam from {name}: {res['line']}"
        self._note(line, book=name, by=by)
        self.sys.bus.publish("imagesets.exam", book=name, score=res.get("score"), n=res.get("n"), node=res.get("node"),
                             chance=res.get("chance"), by=by)
        self.sys.guardian.audit(by, "imagesets.exam", name, score=res.get("score"), n=res.get("n"))
        res["line"] = line
        return res

    def _exam(self, book: dict, items: list[dict]) -> dict | None:
        name, kind = book["name"], book["kind"]
        if kind == "objects":
            node = self._champion(SIGHT)
            labels = {it["answer"] for it in self.items[name]}
            chance = 1 / max(1, len(labels))
            if node is None:
                return {"score": None, "n": len(items), "node": None, "chance": round(chance, 3),
                        "line": "its sight node is not grown yet: nothing to grade"}
            got = self._predict(node, [f"image:{it['sha']}" for it in items])
            right = sum(1 for (lab, _), it in zip(got, items) if lab == it["answer"])
            per = Counter(it["answer"] for (lab, _), it in zip(got, items) if lab == it["answer"])
            return {"score": round(right / len(items), 3), "n": len(items), "right": right, "node": node.id,
                    "chance": round(chance, 3), "best": [k for k, _ in per.most_common(3)],
                    "line": f"its sight node {node.id} named {right} of {len(items)} exam images right "
                            f"({right / len(items):.0%}; chance {chance:.0%})"}
        node = self._champion(GLYPHS)
        if node is None:
            return {"score": None, "n": len(items), "node": None, "chance": None, "line": "its glyph node is not grown yet"}
        if kind == "characters":
            texts, want = [], []
            for it in items:
                try:
                    found = self._glyphs_of(name, it)
                except Exception:
                    continue
                glyphs = [g for line_ in found for g in line_["glyphs"]]
                if len(glyphs) == 1:
                    texts.append(glyphs[0]["text"])
                    want.append(label_of(it["answer"]))
            got = self._predict(node, texts)
            right = sum(1 for (lab, _), w in zip(got, want) if lab == w)
            n = len(items)
            chance = 1 / max(1, len({it["answer"] for it in self.items[name]}))
            return {"score": round(right / n, 3), "n": n, "right": right, "node": node.id, "chance": round(chance, 3),
                    "found": len(texts), "line": f"its glyph node {node.id} read {right} of {n} handwritten characters right "
                                                 f"({right / n:.0%}; chance {chance:.0%}; it found one character in {len(texts)})"}
        sims, exact = [], 0
        for it in items:
            try:
                found = self._glyphs_of(name, it)
            except Exception:
                sims.append(0.0)
                continue
            own = self.sys.vision.own_reading(found, node=node, guess=True)  # its best guess at every character
            text = (own or {}).get("text", "").replace("\n", " ")
            sim = similarity(text, it["answer"])
            sims.append(sim)
            exact += 1 if " ".join(text.split()) == " ".join(it["answer"].split()) else 0
        score = float(np.mean(sims)) if sims else 0.0
        return {"score": round(score, 3), "n": len(items), "exact": exact, "node": node.id, "chance": None,
                "line": f"its reader (glyph node {node.id}, its best guess at every character) read {len(items)} exam images: "
                        f"{score:.0%} like the text on average, {exact} exactly right"}

    # ------------------------------------------------------------------ picture memories
    def remember(self, book: dict, item: dict, answer: str) -> bool:
        """Merge what a picture book taught into its memory nodes: this picture, and what it shows (or says)."""
        mem = getattr(self.sys, "memory", None)
        if not self.pol.memory or mem is None or not mem.allows("pictures"):
            return False
        st = self.state.get(book["name"], {})
        if st.get("memories", 0) >= self.pol.memory_max:
            return False
        res = mem.learn(f"image:{item['sha']}", answer, source=f"picture book {book['name']} ({book['source']})",
                        by=TEACHER, kind="pictures", realm="fact")
        return bool(res and res[1] == "new")

    def negatives(self, n: int, exclude: set) -> list[str]:
        """Pictures a picture-memory node must not answer for: practice images of the books it does not hold."""
        pool = [f"image:{it['sha']}" for name, its in self.items.items() for it in its[:400]
                if it["split"] == "practice" and f"image:{it['sha']}" not in exclude]
        r = random.Random(len(pool))
        r.shuffle(pool)
        return pool[:n]

    # ------------------------------------------------------------------ status
    def summary(self, name: str) -> dict:
        b = self.books[name]
        st = self.state.get(name, {})
        exams = st.get("exams", [])
        items = self.items.get(name, [])
        return {"name": name, "kind": b["kind"], "dataset": b["dataset"], "config": b["config"], "split": b["split"],
                "description": b.get("description", ""), "license": b.get("license", ""), "source": b.get("source", ""),
                "items": len(items), "practice": sum(1 for it in items if it["split"] == "practice"),
                "exam": sum(1 for it in items if it["split"] == "exam"), "labels": len({it["answer"] for it in items}),
                "lessons": st.get("lessons", 0), "learned": st.get("learned", 0), "memories": st.get("memories", 0),
                "rounds": st.get("rounds", 0), "exams": exams[-12:], "last": exams[-1] if exams else None,
                "total": b.get("total", 0), "by": b.get("by", ""), "got": b.get("got"),
                "sample": [{"sha": it["sha"], "answer": it["answer"]} for it in items[:8]]}

    def status(self) -> dict:
        return {"enabled": self.enabled, "books": [self.summary(n) for n in sorted(self.books)], "catalog": self.catalog(),
                "getting": {k: dict(v) for k, v in self.getting.items()}, "errors": dict(self.errors),
                "start_with": list(self.pol.start_with), "stats": dict(self.stats), "log": list(self.log)[-14:][::-1],
                "memory": bool(self.pol.memory), "fetch_limit": self.pol.fetch_limit, "kinds": list(KINDS)}

    def lines(self) -> list[str]:
        st = self.status()
        out = [f"Picture books: {len(st['books'])}" + ("" if st["enabled"] else " (off)")
               + f"; {st['stats'].get('lessons', 0)} lessons, {st['stats'].get('picture memories', 0)} picture memories."]
        for b in st["books"]:
            last = b["last"]
            out.append(f"  {b['name']:<14} {b['kind']:<10} {b['items']:>6} images ({b['practice']} practice, {b['exam']} exam), "
                       f"{b['labels']} labels; {b['lessons']} lessons"
                       + (f"; last exam {last['score']:.0%}" if last and last.get("score") is not None else ""))
        for k, v in st["getting"].items():
            out.append(f"  getting {k}: {v['done']} of {v['of']}")
        out += [f"  - {x['line']}" for x in st["log"][:5]]
        if not st["books"]:
            out.append("  ready-made: " + ", ".join(b["name"] for b in st["catalog"]) + "   (/pictures get <name>)")
        return out


class Picturebook(Agent):
    """When idle: it gets the books it starts with ([imagesets] start_with), then takes lessons from its books
    in turn, with an exam every few lessons."""
    name = "picturebook"

    def __init__(self, system):
        super().__init__(system)
        self.last_lesson = 0.0

    @property
    def interval(self) -> float:
        return 3.0

    async def step(self) -> None:
        s = self.sys
        b = s.imagesets
        if not b.enabled or not s.guardian.background_allowed() or s.inflight:
            return
        if time.time() - (s.last_order_at or s.started_at) < b.pol.idle_seconds:
            return
        now = time.time()
        for name in b.pol.start_with:
            if name in BOOKS and name not in b.books and name not in b.getting and now - b.tried.get(name, 0) > b.pol.retry_get_hours * 3600:
                b.tried[name] = now
                b.save_state()
                try:
                    await b.get(name, by="its picture books")
                    self.stats["got"] += 1
                except Exception as exc:
                    self.stats["could not get"] += 1
                    self.log.info("could not get picture book %s: %s", name, exc)
                return
        if now - self.last_lesson < b.pol.lesson_every_seconds:
            return
        self.last_lesson = now
        await s.background_slot()
        res = await b.lesson()
        self.stats["lessons" if res else "nothing to learn"] += 1
