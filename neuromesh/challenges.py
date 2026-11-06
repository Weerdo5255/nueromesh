"""Challenges: open-source LLM benchmarks as a teacher.

A challenge suite is a set of questions with gold answers (GSM8K, SST-2, AG News, BoolQ, ARC,
CommonsenseQA, SQuAD, or your own). Each suite is split once, stably, into

    practice  the mesh answers, is graded, and learns from the grade:
                - classify suites: the gold label becomes a training example for the skill node
                - a close or failed answer starts a route search: the mesh tries its other routes
                  (tools, skills, an LLM teacher, research) on the same question, and the router
                  node learns the route that scored best
                - a passed answer reinforces the route that produced it
                - short-answer suites can keep the gold answer as retained knowledge
              and, as the "challenge" teacher, the suite answers for its practice items when a
              node is unsure, and supplies them as real training data when the Trainer needs more.
    exam      the mesh answers and is graded. Nothing is learned from exam items, so exam scores
              measure the mesh honestly and are comparable from run to run.

Every answer is graded pass (1 point), close (half a point) or fail (0):
    classify  the label; close for pairs the suite lists (neutral for positive, say)
    numeric   the final number; close when the right number is in the reply but not given as the
              answer, or within close_tolerance
    choice    the option picked; close when the reply names the right option among others
    yesno     yes or no
    short     the gold text (or an alias) in the reply's first line; close for a partial overlap
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .tasks import DIRECT, DISPATCH, Order, skill_payload
from .teachers import Lesson, Teacher

KINDS = ("classify", "numeric", "choice", "yesno", "short")
VERDICT_SCORE = {"pass": 1.0, "close": 0.5, "fail": 0.0}


# ============================================================================ suites
@dataclass
class Item:
    id: str
    input: str
    answer: object                 # label | number | option letter | bool | list of acceptable strings
    choices: list = field(default_factory=list)   # [[letter, text], ...] for choice suites
    context: str = ""              # a passage to read (BoolQ, SQuAD)
    split: str = ""                # practice | exam ("" = decided by the suite's exam_percent)


@dataclass
class Suite:
    name: str
    kind: str
    description: str = ""
    capability: str = ""           # classify: the skill it tests and teaches
    labels: list = field(default_factory=list)
    label_descriptions: dict = field(default_factory=dict)
    close: list = field(default_factory=list)     # classify: [[got, want], ...] pairs worth half a point
    retain: bool = False           # short: keep practice answers as retained knowledge
    exam_percent: int = 30
    source: str = ""               # where it came from (e.g. "huggingface:openai/gsm8k main/test")
    license: str = ""
    folder: str = ""
    items: list = field(default_factory=list)

    def _hash(self, item: Item) -> int:
        return int(hashlib.sha1(f"{self.name}:{item.id}".encode()).hexdigest()[:8], 16)

    def exam_ids(self) -> set[str]:
        """exam_percent of the items, chosen by a stable hash of their ids (per label for classify
        suites, so every label is examined). Items with an explicit split keep it."""
        key = (len(self.items), self.exam_percent)
        if getattr(self, "_exam_key", None) == key:
            return self._exam
        free = [i for i in self.items if i.split not in ("practice", "exam")]
        groups: dict[str, list[Item]] = {}
        for i in free:
            groups.setdefault(str(i.answer) if self.kind == "classify" else "", []).append(i)
        exam = {i.id for i in self.items if i.split == "exam"}
        for group in groups.values():
            k = round(len(group) * self.exam_percent / 100)
            if self.exam_percent and len(group) > 1:
                k = max(1, k)
            exam |= {i.id for i in sorted(group, key=self._hash)[:k]}
        self._exam_key, self._exam = key, exam
        return exam

    def split_of(self, item: Item) -> str:
        if item.split in ("practice", "exam"):
            return item.split
        return "exam" if item.id in self.exam_ids() else "practice"

    def practice(self) -> list[Item]:
        return [i for i in self.items if self.split_of(i) == "practice"]

    def exam(self) -> list[Item]:
        return [i for i in self.items if self.split_of(i) == "exam"]


def load_suite(folder: Path) -> Suite:
    meta = tomllib.loads((folder / "manifest.toml").read_text(encoding="utf-8"))
    if meta.get("kind") not in KINDS:
        raise ValueError(f"{folder.name}: kind must be one of {KINDS}")
    suite = Suite(name=meta.get("name", folder.name), kind=meta["kind"], description=meta.get("description", ""),
                  capability=meta.get("capability", ""), labels=list(meta.get("labels", [])),
                  label_descriptions=dict(meta.get("label_descriptions", {})), close=list(meta.get("close", [])),
                  retain=bool(meta.get("retain", False)), exam_percent=int(meta.get("exam_percent", 30)),
                  source=meta.get("source", ""), license=meta.get("license", ""), folder=str(folder))
    if suite.kind == "classify" and not (suite.capability and suite.labels):
        raise ValueError(f"{folder.name}: a classify suite needs a capability and labels")
    for n, line in enumerate((folder / "items.jsonl").read_text(encoding="utf-8").splitlines()):
        if line.strip():
            d = json.loads(line)
            suite.items.append(Item(str(d.get("id", n)), d["input"], d["answer"], d.get("choices", []),
                                    d.get("context", ""), d.get("split", "")))
    return suite


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in value.items()) + " }"
    raise TypeError(type(value))


def save_suite(folder: Path, suite: Suite) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in asdict(suite).items() if k not in ("items", "folder") and v not in ("", [], {})}
    meta["retain"], meta["exam_percent"] = suite.retain, suite.exam_percent
    (folder / "manifest.toml").write_text("\n".join(f"{k} = {_toml(v)}" for k, v in meta.items()) + "\n",
                                          encoding="utf-8")
    with open(folder / "items.jsonl", "w", encoding="utf-8") as f:
        for it in suite.items:
            f.write(json.dumps({k: v for k, v in asdict(it).items() if v not in ("", [])}, ensure_ascii=False) + "\n")
    return folder


# ======================================================================== adapters
def _gsm8k_number(answer: str) -> float:
    tail = answer.rsplit("####", 1)[-1]
    return _num(re.findall(r"-?[\d,]*\.?\d+", tail)[-1])


def _choices(choices: dict) -> list:
    return [[str(lab), str(text)] for lab, text in zip(choices.get("label", []), choices.get("text", []))]


ADAPTERS: dict[str, dict] = {
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "kind": "numeric",
              "description": "Grade-school maths word problems (GSM8K)", "license": "MIT",
              "item": lambda r: {"input": r["question"], "answer": _gsm8k_number(r["answer"])}},
    "sst2": {"dataset": "stanfordnlp/sst2", "config": "default", "split": "validation", "kind": "classify",
             "capability": "sentiment", "labels": ["negative", "positive"], "close": [["neutral", "positive"],
                                                                                    ["neutral", "negative"]],
             "description": "Movie-review sentence sentiment (SST-2)",
             "item": lambda r: {"input": r["sentence"].strip(), "answer": ["negative", "positive"][int(r["label"])]}},
    "ag_news": {"dataset": "fancyzhx/ag_news", "config": "default", "split": "test", "kind": "classify",
                "capability": "topic", "labels": ["world", "sports", "business", "sci_tech"],
                "description": "News topic classification (AG News)",
                "label_descriptions": {"world": "world news and politics", "sports": "sport",
                                       "business": "business and the economy", "sci_tech": "science and technology"},
                "item": lambda r: {"input": r["text"].strip(),
                                   "answer": ["world", "sports", "business", "sci_tech"][int(r["label"])]}},
    "boolq": {"dataset": "google/boolq", "config": "default", "split": "validation", "kind": "yesno",
              "description": "Yes/no questions about a passage (BoolQ)", "license": "CC BY-SA 3.0",
              "item": lambda r: {"input": r["question"].strip(), "answer": bool(r["answer"]), "context": r["passage"]}},
    "arc_easy": {"dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test", "kind": "choice",
                 "description": "Grade-school science, multiple choice (ARC-Easy)", "license": "CC BY-SA 4.0",
                 "item": lambda r: {"input": r["question"], "answer": r["answerKey"], "choices": _choices(r["choices"]),
                                    "id": r.get("id")}},
    "arc_challenge": {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test", "kind": "choice",
                      "description": "Harder grade-school science, multiple choice (ARC-Challenge)",
                      "license": "CC BY-SA 4.0",
                      "item": lambda r: {"input": r["question"], "answer": r["answerKey"],
                                         "choices": _choices(r["choices"]), "id": r.get("id")}},
    "commonsense_qa": {"dataset": "tau/commonsense_qa", "config": "default", "split": "validation", "kind": "choice",
                       "description": "Commonsense reasoning, multiple choice (CommonsenseQA)", "license": "MIT",
                       "item": lambda r: {"input": r["question"], "answer": r["answerKey"],
                                          "choices": _choices(r["choices"]), "id": r.get("id")}},
    "squad": {"dataset": "rajpurkar/squad", "config": "plain_text", "split": "validation", "kind": "short",
              "description": "Reading comprehension: answer from a passage (SQuAD)", "license": "CC BY-SA 4.0",
              "item": lambda r: {"input": r["question"], "answer": list(dict.fromkeys(r["answers"]["text"])),
                                 "context": r["context"], "id": r.get("id")}},
}


def convert(adapter: str, rows: list[dict], name: str | None = None, source: str = "") -> Suite:
    a = ADAPTERS[adapter]
    suite = Suite(name=name or adapter, kind=a["kind"], description=a.get("description", ""),
                  capability=a.get("capability", ""), labels=list(a.get("labels", [])),
                  label_descriptions=dict(a.get("label_descriptions", {})), close=list(a.get("close", [])),
                  source=source, license=a.get("license", ""))
    for n, row in enumerate(rows):
        try:
            d = a["item"](row)
        except (KeyError, IndexError, ValueError, TypeError):
            continue  # skip malformed rows (e.g. hidden test labels)
        suite.items.append(Item(str(d.get("id") or n), d["input"], d["answer"], d.get("choices", []),
                                d.get("context", "")))
    return suite


def fetch_rows(dataset: str, config: str, split: str, limit: int, *, api: str, offset: int = 0,
               token: str | None = None, timeout: float = 30.0) -> list[dict]:
    """Rows from the Hugging Face dataset viewer API (/rows, at most 100 per request)."""
    rows: list[dict] = []
    while len(rows) < limit:
        params = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                         "offset": offset + len(rows), "length": min(100, limit - len(rows))})
        req = urllib.request.Request(f"{api.rstrip('/')}/rows?{params}",
                                     headers={"User-Agent": "NeuroMesh (+https://github.com/Weerdo5255/nueromesh)",
                                              **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"the dataset server returned HTTP {exc.code} for {dataset} ({config}/{split})") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"could not reach the dataset server at {api}: {getattr(exc, 'reason', exc)}") from None
        page = [r["row"] for r in data.get("rows", []) if isinstance(r, dict) and "row" in r]
        rows += page
        if not page or len(rows) >= int(data.get("num_rows_total", 1 << 30)) - offset:
            break
    return rows[:limit]


# ========================================================================= grading
@dataclass
class Grade:
    verdict: str                   # pass | close | fail
    got: str
    want: str
    detail: str = ""

    @property
    def score(self) -> float:
        return VERDICT_SCORE[self.verdict]


_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_STATED = re.compile(r"(?:answer is|answer:|=|equals|in total(?: is)?|total(?: of| is)?|altogether(?: is)?)"
                     r"\s*(?:about\s+|exactly\s+)?\$?(-?\d[\d,]*(?:\.\d+)?)", re.I)  # an answer stated as such


def _num(text) -> float:
    return float(str(text).replace("$", "").replace(",", ""))


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text):
        try:
            out.append(_num(m))
        except ValueError:
            pass
    return out


def _norm(text: str) -> str:
    t = str(text).lower()
    t = "".join(ch if ch not in string.punctuation else " " for ch in t)
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def _f1(got: str, want: str) -> float:
    g, w = _norm(got).split(), _norm(want).split()
    common = sum((Counter(g) & Counter(w)).values())
    if not common:
        return 0.0
    p, r = common / len(g), common / len(w)
    return 2 * p * r / (p + r)


def prompt_for(suite: Suite, item: Item) -> str:
    """How a challenge item is put to the mesh, as an order."""
    if suite.kind == "classify":
        return f"{suite.capability}: {item.input}"
    if suite.kind == "choice":
        options = "\n".join(f"{lab}) {text}" for lab, text in item.choices)
        return f"{item.input}\n{options}\nAnswer with the letter of the right option."
    text = item.input.strip()
    if suite.kind == "yesno":
        text = (text[:1].upper() + text[1:]).rstrip("?") + "?"
    if item.context:
        text += f"\nPassage: {item.context}"
    if suite.kind == "yesno":
        text += "\nAnswer yes or no."
    return text


def want_text(suite: Suite, item: Item) -> str:
    a = item.answer
    if suite.kind == "yesno":
        return "yes" if a else "no"
    if suite.kind == "short":
        return str(a[0] if isinstance(a, list) and a else a)
    if suite.kind == "numeric":
        return f"{_num(a):g}"
    return str(a)


def grade(suite: Suite, item: Item, ok: bool, output, reply: str, tolerance: float = 0.01,
          strict: bool = False) -> Grade:
    """pass / close / fail for one answer. `reply` is what a person would read (the answer first).
    strict (route search): a number only counts as the answer when it is the result itself or stated
    as the answer ("= 60", "the answer is 60"), not just the last number in some other reply."""
    want = item.answer
    first = (reply or "").strip().split("\n", 1)[0]
    if not ok:
        return Grade("fail", (reply or "")[:120], want_text(suite, item), "not fulfilled")
    kind = suite.kind
    if kind == "classify":
        got = output if isinstance(output, str) and output in suite.labels + [x[0] for x in suite.close] else ""
        if not got:
            words = set(_norm(first).split())
            got = next((lab for lab in suite.labels if lab in words), str(output))
        if got == want:
            return Grade("pass", got, want)
        close = [got, want] in suite.close or [want, got] in suite.close
        return Grade("close" if close else "fail", got, want)
    if kind == "numeric":
        target = _num(want)
        if isinstance(output, (int, float)) and not isinstance(output, bool):
            final, seen = float(output), [float(output)] + _numbers(first)
        else:
            stated = _STATED.findall(first)
            seen = _numbers(first)
            final = _num(stated[-1]) if stated else (seen[-1] if seen and not strict else None)
        same = lambda x: x is not None and abs(x - target) <= 1e-6 * max(1.0, abs(target))  # noqa: E731
        if same(final):
            return Grade("pass", f"{final:g}", f"{target:g}")
        if any(same(x) for x in seen):
            return Grade("close", f"{final:g}" if final is not None else first[:40], f"{target:g}",
                         "the right number is in the reply but not given as the answer")
        if final is not None and abs(final - target) <= tolerance * max(1.0, abs(target)):
            return Grade("close", f"{final:g}", f"{target:g}", f"within {tolerance:.0%}")
        return Grade("fail", f"{final:g}" if final is not None else first[:40], f"{target:g}")
    if kind == "choice":
        letters = [lab for lab, _ in item.choices]
        named = set()
        pats = [r"^\W*\(?([A-Z])\)?(?:[\s.:)]|$)", r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\b",
                r"^\s*\(?([A-Z])\)?\s*$"]
        for p in pats:
            for m in re.finditer(p, first):
                if m.group(1) in letters:
                    named.add(m.group(1))
        nf = f" {_norm(first)} "
        for lab, text in item.choices:
            if _norm(text) and f" {_norm(text)} " in nf:
                named.add(lab)
        if named == {want}:
            return Grade("pass", want, want)
        if want in named:
            return Grade("close", ",".join(sorted(named)), want, "the right option among others")
        return Grade("fail", ",".join(sorted(named)) or first[:40], want)
    if kind == "yesno":
        m = re.search(r"\b(yes|no|true|false|correct|incorrect)\b", first, re.I)
        got = None if not m else m.group(1).lower() in ("yes", "true", "correct")
        wanted = bool(want)
        return Grade("pass" if got is wanted else "fail", "?" if got is None else ("yes" if got else "no"),
                     "yes" if wanted else "no")
    # short
    golds = [str(g) for g in (want if isinstance(want, list) else [want])]
    nf = f" {_norm(first)} "
    if any(_norm(g) and f" {_norm(g)} " in nf for g in golds):
        return Grade("pass", first[:80], golds[0])
    best = max((_f1(first, g) for g in golds), default=0.0)
    if best >= 0.5 or any(_norm(g) and f" {_norm(g)} " in f" {_norm(reply)} " for g in golds):
        return Grade("close", first[:80], golds[0], f"partial overlap (F1 {best:.2f})")
    return Grade("fail", first[:80], golds[0])


# ========================================================================= teacher
def _key(text: str) -> str:
    return " ".join(str(text).split()).lower()


class ChallengeTeacher(Teacher):
    """The challenge suites as a teacher: gold labels for practice items of classify suites, the
    routes that passed practice questions, and real benchmark sentences when the Trainer needs
    more examples of a label. It never knows anything about exam items."""
    name = "challenge"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.gold: dict[str, dict[str, str]] = {}        # capability -> practice text key -> label
        self.pool: dict[str, dict[str, list[str]]] = {}  # capability -> label -> practice texts
        self.routes: dict[str, tuple[str, str]] = {}     # prompt key -> (prompt, route that passed)
        self.items = 0

    def index(self, suites: list[Suite]) -> None:
        self.gold.clear()
        self.pool.clear()
        self.items = 0
        for suite in suites:
            if suite.kind != "classify":
                continue
            for item in suite.practice():
                self.gold.setdefault(suite.capability, {})[_key(item.input)] = str(item.answer)
                self.pool.setdefault(suite.capability, {}).setdefault(str(item.answer), []).append(item.input)
                self.items += 1

    def available(self) -> bool:
        return bool(self.gold or self.routes)

    def status(self) -> dict:
        return {"detail": f"{self.items} practice answers, {len(self.routes)} routes that passed"}

    async def label(self, spec, text):
        if spec.name == DISPATCH:
            found = self.routes.get(_key(text))
            if found and found[1] in spec.labels and found[1] not in spec.inactive_labels:
                return Lesson(found[1], 0.9, self.name, "this route passed the challenge")
            return None
        label = self.gold.get(spec.name, {}).get(_key(text))
        if label and label in spec.labels:
            return Lesson(label, 0.99, self.name, "gold answer from a challenge suite")
        return None

    async def generate(self, spec, label, n):
        if spec.name == DISPATCH:
            texts = [p for p, r in self.routes.values() if r == label]
        else:
            texts = list(self.pool.get(spec.name, {}).get(label, []))
        self.rng.shuffle(texts)
        return texts[:n]


# ========================================================================== runner
@dataclass
class Report:
    suite: str
    mode: str                      # practice | exam
    started: float
    seconds: float = 0.0
    results: list = field(default_factory=list)   # [{id, verdict, got, want, route, detail}]
    learned: int = 0               # lessons given (practice)
    rerouted: int = 0              # questions the router learned a better route for

    def counts(self) -> Counter:
        return Counter(r["verdict"] for r in self.results)

    @property
    def score(self) -> float:
        return sum(VERDICT_SCORE[r["verdict"]] for r in self.results) / len(self.results) if self.results else 0.0

    def line(self) -> str:
        c = self.counts()
        return (f"{self.suite} {self.mode}: {self.score:.0%} ({c['pass']} pass, {c['close']} close, {c['fail']} fail "
                f"of {len(self.results)})")


class Challenges:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.challenges
        self.suites: dict[str, Suite] = {}
        self.errors: dict[str, str] = {}
        self.teacher: ChallengeTeacher = system.teachers.teachers.get("challenge")
        raw = system.store.get_kv("challenge.routes")
        if raw and self.teacher is not None:
            self.teacher.routes = {k: tuple(v) for k, v in json.loads(raw).items()}
        self.history: list[dict] = system.store.load_records("challenge_runs")[-200:]
        self.offsets: dict[str, int] = json.loads(system.store.get_kv("challenge.offsets") or "{}")
        self.running: set[str] = set()
        self.reload()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def folders(self) -> list[Path]:
        out = [Path(f) if Path(f).is_absolute() else self.sys.root / f for f in self.pol.folders]
        return out + [self.sys.state_dir / "challenges"]

    def reload(self) -> None:
        self.suites.clear()
        self.errors.clear()
        for base in self.folders():
            if not base.is_dir():
                continue
            for folder in sorted(p for p in base.iterdir() if (p / "manifest.toml").exists()):
                try:
                    suite = load_suite(folder)
                    suite.exam_percent = suite.exam_percent if suite.exam_percent else self.pol.exam_percent
                    self.suites[suite.name] = suite
                except (ValueError, KeyError, OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
                    self.errors[folder.name] = str(exc)
        if self.teacher is not None:
            self.teacher.index([s for s in self.suites.values()
                                if s.kind != "classify" or s.capability in self.sys.capabilities])

    def _save_routes(self) -> None:
        routes = dict(list(self.teacher.routes.items())[-5000:])
        self.sys.store.set_kv("challenge.routes", json.dumps(routes))

    # ------------------------------------------------------------- fetching
    def fetch(self, adapter: str, limit: int = 0, split: str = "", name: str = "", by: str = "operator") -> Suite:
        """Download a benchmark from the Hugging Face dataset viewer into state/challenges/."""
        if adapter not in ADAPTERS:
            raise ValueError(f"no adapter '{adapter}' (adapters: {', '.join(ADAPTERS)})")
        a = ADAPTERS[adapter]
        split = split or a["split"]
        limit = limit or self.pol.fetch_limit
        rows = fetch_rows(a["dataset"], a["config"], split, limit, api=self.pol.api_url, token=self.pol_token())
        name = name or (adapter if split == a["split"] else f"{adapter}-{split}")
        suite = convert(adapter, rows, name, f"huggingface:{a['dataset']} {a['config']}/{split} (first {len(rows)} rows)")
        if not suite.items:
            raise RuntimeError(f"no usable rows in {a['dataset']} {a['config']}/{split}")
        suite.exam_percent = self.pol.exam_percent
        save_suite(self.sys.state_dir / "challenges" / name, suite)
        self.sys.guardian.audit(by, "challenge.fetched", name, adapter=adapter, rows=len(suite.items),
                                source=suite.source)
        self.sys.bus.publish("challenge.fetched", suite=name, items=len(suite.items), source=suite.source)
        self.reload()
        return self.suites[name]

    def pol_token(self) -> str | None:
        import os
        return os.environ.get(self.pol.token_env) if self.pol.token_env else None

    # -------------------------------------------------------------- running
    def _prepare(self, suite: Suite, by: str) -> None:
        """A classify suite tests a skill; a person running it for a skill the mesh lacks creates it."""
        s = self.sys
        if suite.kind == "classify" and suite.capability not in s.capabilities:
            spec = s.define_capability(suite.capability, list(suite.labels), suite.description or f"from {suite.name}")
            spec.label_descriptions.update(suite.label_descriptions)
            s.save_capability(spec)
            self.reload()

    async def _ask(self, suite: Suite, item: Item, learn: bool):
        order = Order(text=prompt_for(suite, item), source=f"challenge:{suite.name}", background=True, learn=learn)
        return order, await self.sys.orchestrator.fulfil(order)

    async def run(self, name: str, mode: str = "exam", n: int | None = None, by: str = "operator") -> Report:
        if not self.enabled:
            raise ValueError("challenges are off ([challenges] enabled = false in the directives)")
        suite = self.suites.get(name)
        if suite is None:
            raise ValueError(f"no challenge suite '{name}' (/challenges lists them)")
        if mode not in ("practice", "exam"):
            raise ValueError("mode is practice or exam")
        if name in self.running:
            raise ValueError(f"{name} is already running")
        self._prepare(suite, by)
        pool = suite.practice() if mode == "practice" else suite.exam()
        if mode == "practice":  # rotate through the practice items
            n = n or self.pol.practice_batch
            start = self.offsets.get(name, 0) % max(1, len(pool))
            items = (pool[start:] + pool[:start])[: min(n, len(pool))]
            self.offsets[name] = start + len(items)
            self.sys.store.set_kv("challenge.offsets", json.dumps(self.offsets))
        else:  # the same exam every time (the first n), so runs are comparable
            items = pool[: n] if n else pool
        rep = Report(name, mode, time.time())
        self.running.add(name)
        try:
            with self.sys.activity.track("challenge", "challenges", f"{name} {mode} ({len(items)} items)"):
                for item in items:
                    if self.sys.guardian.halted:
                        break
                    result = await (self._practice(suite, item, rep) if mode == "practice" else self._exam(suite, item))
                    rep.results.append(result)
        finally:
            self.running.discard(name)
        rep.seconds = round(time.time() - rep.started, 2)
        self._record(rep, by)
        return rep

    async def _exam(self, suite: Suite, item: Item) -> dict:
        _, res = await self._ask(suite, item, learn=False)  # nothing is learned from exam items
        g = grade(suite, item, res.ok, res.output, res.answer, self.pol.close_tolerance)
        return {"id": item.id, "verdict": g.verdict, "got": g.got, "want": g.want, "route": res.route or "-",
                "detail": g.detail, "input": item.input[:120]}

    async def _practice(self, suite: Suite, item: Item, rep: Report) -> dict:
        s = self.sys
        if suite.kind == "classify":  # grade the node itself, then teach it the gold label
            cap = suite.capability
            node = s.registry.champion(cap)
            got, route = None, "no node yet"
            if node is not None:
                got, _ = node.predict(s.encode(item.input), frozenset(s.capabilities[cap].inactive_labels))
                route = f"node {node.id}"
            g = grade(suite, item, got is not None, got, got or "", self.pol.close_tolerance) if got else \
                Grade("fail", "-", want_text(suite, item), "no node to answer yet")
            if s.learn(cap, item.input, str(item.answer), "challenge"):
                rep.learned += 1
            return {"id": item.id, "verdict": g.verdict, "got": g.got, "want": g.want, "route": route,
                    "detail": g.detail, "input": item.input[:120]}
        order, res = await self._ask(suite, item, learn=True)
        g = grade(suite, item, res.ok, res.output, res.answer, self.pol.close_tolerance)
        route = res.route or "-"
        dispatched = route in s.dispatch.labels  # chosen by the router, not an explicit/mind/task route
        better = None
        if g.verdict != "pass":  # a close or failed answer: is there a route that gets it right?
            better = await self._route_search(suite, item, order.text, route, g.score)
        if better is None and g.verdict in ("pass", "close") and dispatched:
            if s.learn(DISPATCH, order.text, route, "challenge"):  # right, or close: keep routing it there
                rep.learned += 1
        if better is not None:  # the router learns the route that got it right
            route_found, score = better
            if s.learn(DISPATCH, order.text, route_found, "challenge"):
                rep.learned += 1
            self.teacher.routes[_key(order.text)] = (order.text, route_found)
            self._save_routes()
            rep.rerouted += 1
            route = f"{route} -> {route_found} ({score:g})"
        if suite.kind == "short" and suite.retain:
            self._retain(suite, item)
            rep.learned += 1
        return {"id": item.id, "verdict": g.verdict, "got": g.got, "want": g.want, "route": route,
                "detail": g.detail, "input": item.input[:120]}

    async def _route_search(self, suite: Suite, item: Item, prompt: str, current: str, current_score: float):
        """Try the other routes on a question the mesh missed; return (route, 1.0) for one that gets it
        right. Only a pass re-routes: a close answer from an unrelated route (a word count that happens
        to contain the number, say) is too weak a reason to send such questions there."""
        from . import voice
        s = self.sys
        best: tuple[float, str] | None = None
        tried = 0
        for route in s.dispatch.active_labels():
            if route == current or tried >= self.pol.max_route_trials:
                continue
            reply, output, ok = None, None, False
            if route.startswith("tool:"):
                tool = s.tools.tools.get(route[5:])
                if tool is None:
                    continue
                if tool.manifest.provides == "knowledge":
                    if suite.kind not in ("short", "yesno"):
                        continue
                    doc, _ = await s.research(item.input, reason=f"challenge {suite.name}", background=True)
                    if doc is not None:
                        hits = s.knowledge.search(item.input, doc_id=doc.id)
                        if hits:
                            body, _, _ = await s.answer_from_knowledge(item.input, hits)
                            reply, output, ok = body, body, True
                elif s.guardian.authorize_tool_run(tool).allowed:
                    run = await s.sandbox.run(tool, prompt, tool.manifest.timeout_s or s.directives.tools.default_timeout_s,
                                              s.tool_config(tool))
                    s.record_tool_run(tool, run, exploratory=True)
                    ok = run.ok
                    reply, output = (voice.tool_phrase(tool, run), run.output) if run.ok else (run.error, None)
            elif route == DIRECT:
                answered = await s.teachers.answer(prompt)
                if answered:
                    reply, output, ok = voice.lead_first(answered[0]), answered[0], True
            elif route.startswith("skill:") and suite.kind == "classify":
                cap = route[6:]
                inf = await s.infer(cap, skill_payload(prompt), learn=False)
                reply, output, ok = inf.label or "", inf.label, inf.label is not None
            else:
                continue
            tried += 1
            if reply is None:
                continue
            score = grade(suite, item, ok, output, reply, self.pol.close_tolerance, strict=True).score
            if score == 1.0 and score > current_score:
                best = (score, route)
                break
        return (best[1], best[0]) if best else None

    def _retain(self, suite: Suite, item: Item) -> None:
        golds = item.answer if isinstance(item.answer, list) else [item.answer]
        answer = str(golds[0])
        k = {"source": f"challenge:{suite.name}", "title": item.input.strip()[:120], "url": suite.source or suite.name,
             "summary": f"{answer}.", "topic": item.input, "links": [], "language": "en",
             "passages": [{"section": "Answer", "text": f"{item.input.strip()} {answer}."}]}
        self.sys.knowledge.add(k, f"practice answer from {suite.name}")

    def _record(self, rep: Report, by: str) -> None:
        c = rep.counts()
        prev = self.last(rep.suite, rep.mode)
        record = {"id": f"{rep.suite}:{rep.mode}:{int(rep.started * 1000)}", "suite": rep.suite, "mode": rep.mode,
                  "created_at": rep.started, "seconds": rep.seconds, "score": round(rep.score, 4), "n": len(rep.results),
                  "pass": c["pass"], "close": c["close"], "fail": c["fail"], "learned": rep.learned,
                  "rerouted": rep.rerouted, "by": by, "results": rep.results[:200]}
        self.history.append(record)
        del self.history[:-200]
        self.sys.store.save_record("challenge_runs", record["id"], rep.mode, record)
        self.sys.guardian.audit(by, f"challenge.{rep.mode}", rep.suite, score=record["score"], n=record["n"],
                                learned=rep.learned, rerouted=rep.rerouted)
        self.sys.bus.publish("challenge.done", suite=rep.suite, mode=rep.mode, score=record["score"], n=record["n"],
                             passed=c["pass"], close=c["close"], failed=c["fail"], learned=rep.learned,
                             rerouted=rep.rerouted, previous=prev["score"] if prev else None)

    def last(self, suite: str, mode: str = "exam") -> dict | None:
        return next((r for r in reversed(self.history) if r["suite"] == suite and r["mode"] == mode), None)

    # --------------------------------------------------------------- status
    def summary(self) -> list[dict]:
        out = []
        for name, suite in sorted(self.suites.items()):
            exam = [r for r in self.history if r["suite"] == name and r["mode"] == "exam"]
            out.append({"suite": name, "kind": suite.kind, "capability": suite.capability, "items": len(suite.items),
                        "practice": len(suite.practice()), "exam": len(suite.exam()), "source": suite.source,
                        "description": suite.description,
                        "last_exam": exam[-1]["score"] if exam else None,
                        "first_exam": exam[0]["score"] if exam else None,
                        "best_exam": max((r["score"] for r in exam), default=None), "exams": len(exam),
                        "practice_runs": sum(1 for r in self.history if r["suite"] == name and r["mode"] == "practice")})
        return out

    def status(self) -> dict:
        return {"enabled": self.enabled, "suites": self.summary(), "errors": self.errors,
                "routes_learned": len(self.teacher.routes) if self.teacher else 0}
