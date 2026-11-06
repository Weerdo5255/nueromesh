"""Sleep and dreams: dream nodes that make up questions and answers from what the mesh already knows.

When no one has given it an order for a while, the mesh has a chance to fall asleep. Asleep, all
other background work waits (training, evaluation, research, tasks, initiative) and the capacity
goes to dreaming. An order always wakes it at once (P2).

Dreaming uses only what the mesh already holds - the facts in its memory nodes and the passages it
has retained - never a research tool or a teacher. Each dream step picks random pieces of that
material and makes up questions and answers about them:

    remembered  a sentence with one name or number taken out ("Paris is the capital of what?" ->
                "France"): the answer is in the sentence, so it can be checked
    imagined    a remembered question with one name swapped for another from something else it
                knows ("What is the capital of Japan?"): nobody has told it the answer

The dream node rehearses on some remembered dreams and answers the rest, and all the imagined ones,
on its own. Upon waking, those answers are checked against its own memory: a remembered dream
against the sentence it came from, an imagined one against any fact or passage that says the same.
The dreams that check out are kept; if any did, the dream nodes go through a genetic algorithm:
children of the best dreamer, with more or fewer neurons and layers, are trained on every checked
dream and judged on dreams held out from training; the fittest stay (a larger one wins a tie).

Awake, the mesh can use its best dream node to imagine things ("imagine a city on Mars",
/imagine): the node suggests what comes to mind for the prompt, and the mesh blends that into
something it remembers to make a new response. An imagined response always says so, and names the
dream node it came from; it is never stored as a fact (P13).
"""
from __future__ import annotations

import json
import random
import re
import time
import zlib
from collections import Counter, deque
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agents import Agent
from .evolution import morph, mutate
from .memory import encode_many, fold, forward, hidden_of, init_params, remap_outputs, softmax, train
from .tasks import imagine_prompt  # noqa: F401  (orders like "imagine ..." are spotted in tasks.py)

NOTHING = "(nothing comes to mind)"
_SENT = re.compile(r"(?<=[.!?])\s+")
_NAME = re.compile(r"\b[A-Z][\w'’-]*(?:\s+(?:(?:da|de|del|di|du|von|van|der|of|the|and|y)\s+)?[A-Z][\w'’-]*)*")
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
STARTERS = {"The", "A", "An", "In", "It", "Its", "This", "That", "These", "Those", "He", "She", "They", "His", "Her",
            "Their", "On", "At", "As", "After", "Before", "During", "When", "While", "Although", "However", "There",
            "Some", "Many", "Most", "One", "For", "From", "By", "With", "If", "I", "We", "You", "Our", "Your",
            "What", "Who", "Whom", "Whose", "When", "Where", "Which", "Why", "How", "Is", "Are", "Was", "Were", "Do",
            "Does", "Did", "Can", "Could", "Should", "Would", "Will", "Yes", "No", "Answer", "Question", "True", "False"}
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = re.compile(rf"\b(?:\d{{1,2}}\s+{_MONTH}(?:,?\s+\d{{4}})?|{_MONTH}\s+\d{{1,2}}(?:,\s*\d{{4}})?|{_MONTH}\s+\d{{4}})\b")
CALENDAR = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
PROMPT_STOP = {"imagine", "what", "would", "could", "should", "there", "where", "which", "about", "perhaps", "were",
               "with", "that", "this", "from", "have", "dream", "into", "like", "then", "they", "will", "your"}
PERSON_VERBS = {"was", "wrote", "painted", "died", "born", "invented", "discovered", "composed", "founded",
                "led", "ruled", "said", "became", "married", "studied", "designed", "built", "walked", "flew", "won",
                "lived", "played", "sang", "directed", "proposed", "developed", "created", "explored", "reached"}

def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(" ".join(str(text).split())) if 20 <= len(s.strip()) <= 260]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


@dataclass
class Span:
    start: int
    end: int
    text: str
    kind: str  # name | year | number


def spans(sentence: str) -> list[Span]:
    """The names and numbers in a sentence a dream can be about."""
    out: list[Span] = [Span(m.start(), m.end(), m.group(0), "date") for m in _DATE.finditer(sentence)]
    dates = list(out)

    def in_date(a: int, b: int) -> bool:
        return any(a < d.end and d.start < b for d in dates)
    for m in _NAME.finditer(sentence):
        text, start = m.group(0), m.start()
        words = text.split()
        while words and words[0] in STARTERS:  # "The Eiffel Tower" -> "Eiffel Tower"; "In" is not a name
            start += len(words[0]) + 1
            words = words[1:]
        while words and words[-1] in ("of", "the", "and", "da", "de", "von", "van", "y"):
            words = words[:-1]
        if not words or (len(words) == 1 and (len(words[0]) < 2 or words[0] in CALENDAR)):
            continue
        text = " ".join(words)
        if in_date(start, start + len(text)):
            continue
        out.append(Span(start, start + len(text), text, "name"))
    for m in _NUMBER.finditer(sentence):
        if in_date(m.start(), m.end()):
            continue
        value = m.group(0).replace(",", "")
        kind = "year" if re.fullmatch(r"1[0-9]{3}|20[0-9]{2}", value) else "number"
        out.append(Span(m.start(), m.end(), m.group(0), kind))
    whole = len(sentence.rstrip(" .!?"))
    return [s for s in out if s.end - s.start < whole * 0.8]


def cloze(sentence: str, span: Span) -> str:
    """A question whose answer is the span: the sentence with the span taken out."""
    s = sentence.strip().rstrip(" .!?;:")
    before, after = s[:span.start].rstrip(), s[span.end:].strip()
    if span.kind == "date":
        before = re.sub(r"\s+(?:on|in)$", "", before) + " on"
        wh = "what date"
    elif span.kind == "year":
        wh = "what year"
    elif span.kind == "number":
        wh = "how many" if after[:1].isalpha() else "what number"
    else:
        nxt = after.split()[0].lower() if after else ""
        wh = "who" if len(span.text.split()) >= 2 and nxt in PERSON_VERBS else "what"
    before = re.sub(r"\s+(?:the|a|an)$", "", before)
    if not before or before in ("The", "A", "An"):
        q = f"{wh[0].upper()}{wh[1:]} {after}".strip()
    else:
        q = f"{before} {wh} {after}".strip()
    q = re.sub(r"\s+([,;:])", r"\1", re.sub(r"\s+", " ", q))
    return q.rstrip(" ,") + "?"


def statement(question: str, answer: str) -> str | None:
    """A remembered question and short answer as a statement it can dream about ("Who wrote Hamlet?" +
    "William Shakespeare" -> "William Shakespeare wrote Hamlet."); None when it does not read as one."""
    q, a = " ".join(str(question).split()).rstrip(" ?.!"), " ".join(str(answer).split()).rstrip(" .")
    if len(a) < 2 or a.lower() in ("yes", "no", "true", "false") or re.search(r"\d\s*[-+*/x×^]\s*\d|%|\b(?:plus|minus|times|squared|cubed|divided)\b", q):
        return None
    m = re.match(r"(?i)^(?:who|what)\s+(is|was|are|were)\s+(.+)$", q)
    if m:
        out = f"{a} {m.group(1).lower()} {m.group(2)}"
    elif (m := re.match(r"(?i)^(?:what|which)\s+([a-z]+)\s+(is|was|are|were)\s+(.+)$", q)):
        noun, verb, rest = m.group(1).lower(), m.group(2).lower(), m.group(3)
        out = f"{a} {verb} {rest} {noun}" if rest.lower().startswith("the ") else f"{a} {verb} the {noun} {rest}"
    elif (m := re.match(r"(?i)^in\s+(?:which|what)\s+[a-z]+\s+(is|was|are|were)\s+(.+)$", q)):
        out = f"{m.group(2)} {m.group(1).lower()} in {a}"
    elif (m := re.match(r"(?i)^(?:who|what)\s+([a-z]+)\s+(.+)$", q)) and (
            m.group(1).lower() in PERSON_VERBS or m.group(1).lower().endswith("ed")):
        out = f"{a} {m.group(1).lower()} {m.group(2)}"
    else:
        return None
    return out[:1].upper() + out[1:] + "."


# ============================================================================ nodes
@dataclass
class DreamNode:
    id: str
    params: list
    labels: list                 # what it can answer; labels[0] is NOTHING
    lr: float = 0.01
    generation: int = 0
    parent: str = ""
    fitness: float | None = None  # accuracy on checked dreams held out from its training
    created: float = field(default_factory=time.time)
    changes: list = field(default_factory=list)
    history: list = field(default_factory=list)

    @property
    def hidden(self) -> tuple:
        return hidden_of(self.params)

    @property
    def weights(self) -> int:
        return int(sum(p.size for p in self.params[::2]))

    def shape(self) -> str:
        return "x".join(map(str, self.hidden)) or "linear"

    def predict(self, X: np.ndarray, inputs: int) -> np.ndarray:
        return softmax(forward(self.params, fold(X, inputs))[0])


@dataclass
class Dream:
    question: str
    truth: str | None            # remembered dreams know their answer; imagined ones do not
    kind: str                    # remembered | imagined
    source: str
    sentence: str
    anchors: list = field(default_factory=list)  # imagined: the name swapped in
    answer: str | None = None    # what the dream node answered
    confidence: float = 0.0
    verdict: str = ""            # after waking: correct | wrong | unchecked | nothing


# ============================================================================= jobs
@dataclass
class DreamJob:
    """One dream step, run in a thread: rehearse some dreams, answer the rest."""
    params: list
    labels: list
    inputs: int
    dim: int
    lr: float
    epochs: int
    rehearse: list               # (question, answer)
    answer: list                 # Dream objects
    seed: int

    def describe(self) -> str:
        return f"rehearsing {len(self.rehearse)} dreams, answering {len(self.answer)}"

    def run(self) -> dict:
        idx = {a: i for i, a in enumerate(self.labels)}
        params = [p.copy() for p in self.params]
        rows = [(q, idx[a]) for q, a in self.rehearse if a in idx]
        if rows:
            X = fold(encode_many([q for q, _ in rows], self.dim), self.inputs)
            y = np.array([i for _, i in rows], dtype=np.int64)
            params, _, _ = train(params, X, y, np.ones(len(rows), dtype=np.float32), epochs=self.epochs, lr=self.lr,
                                 check=lambda p: False, check_every=10 ** 9)
        answers = []
        if self.answer:
            P = softmax(forward(params, fold(encode_many([d.question for d in self.answer], self.dim), self.inputs))[0])
            for d, row in zip(self.answer, P):
                top = int(row.argmax())
                answers.append((self.labels[top] if top else None, float(row[top])))
        return {"params": params, "answers": answers}


@dataclass
class EvolveJob:
    """Waking: the genetic algorithm over the dream nodes, on the dreams that checked out."""
    population: list             # (id, params, lr, generation, fitness)
    dreamer: tuple               # last night's dreaming node: (params, lr)
    labels: list
    inputs: int
    dim: int
    train_rows: list             # (question, answer)
    test_rows: list
    children: int
    keep: int
    epochs: int
    evo_policy: object
    max_hidden: int
    max_depth: int
    seed: int

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        idx = {a: i for i, a in enumerate(self.labels)}
        tr = [(q, idx[a]) for q, a in self.train_rows if a in idx]
        te = [(q, idx[a]) for q, a in self.test_rows if a in idx] or tr
        Xtr = fold(encode_many([q for q, _ in tr], self.dim), self.inputs) if tr else None
        ytr = np.array([i for _, i in tr], dtype=np.int64)
        Xte = fold(encode_many([q for q, _ in te], self.dim), self.inputs)
        yte = np.array([i for _, i in te], dtype=np.int64)

        def score(params) -> np.ndarray:
            return softmax(forward(params, Xte)[0]).argmax(axis=1) == yte

        def fit(params, lr):
            if Xtr is None:
                return params
            out, _, _ = train(params, Xtr, ytr, np.ones(len(ytr), dtype=np.float32), epochs=self.epochs, lr=lr,
                              check=lambda p: False, check_every=10 ** 9)
            return out
        pool = [{"id": i, "params": p, "lr": lr, "generation": g, "parent": "", "changes": [], "ok": score(p)}
                for i, p, lr, g, _ in self.population]
        dp, dlr = self.dreamer
        best_gen = max((g for _, _, _, g, _ in self.population), default=0)
        pool.append({"id": "", "params": dp, "lr": dlr, "generation": best_gen, "parent": "the night's dreamer",
                     "changes": ["dreamt last night"], "ok": score(dp)})
        parent = max(pool, key=lambda c: (c["ok"].mean(), sum(x.size for x in c["params"][::2])))
        for _ in range(self.children):
            g = mutate(hidden_of(parent["params"]) or (32,), parent["lr"], rng, self.evo_policy,
                       max_hidden=self.max_hidden, max_depth=self.max_depth, input_dim=self.inputs, outputs=len(self.labels))
            params = fit(morph(parent["params"], g.hidden, rng), g.lr) if hidden_of(parent["params"]) else \
                fit(init_params(rng, self.inputs, g.hidden, len(self.labels)), g.lr)
            pool.append({"id": "", "params": params, "lr": g.lr, "generation": parent["generation"] + 1,
                         "parent": parent["id"] or "the night's dreamer", "changes": g.changes, "ok": score(params)})
        size = lambda c: sum(x.size for x in c["params"][::2])  # noqa: E731
        # fittest first; between equally fit nodes the larger wins (the genetic algorithm trends larger)
        kept = sorted(pool, key=lambda c: (-float(c["ok"].mean()), -size(c)))[: self.keep]
        for c in kept:
            c["fitness"] = round(float(c["ok"].mean()), 4)
            c.pop("ok")
        return {"population": kept, "tested": len(te), "trained": len(tr)}


# ============================================================================= bank
class DreamBank:
    """The dream nodes, sleep and waking, and imagination."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.dreams
        self.dim = system.directives.nodes.input_dim
        self.dir = system.state_dir / "dreams"
        self.rng = random.Random(system.rng.randrange(1 << 30))
        self.nrng = np.random.default_rng(self.rng.randrange(1 << 30))
        self.nodes: dict[str, DreamNode] = {}
        self.asleep = False
        self.slept_at = 0.0
        self.woke_at = time.time()
        self.wake_reason = ""
        self.material: list[dict] = []
        self.journal: list[Dream] = []
        self.rehearsed: list[tuple] = []
        self.dreamer: DreamNode | None = None
        self.review_pending = False
        self.stats: Counter = Counter()
        raw = system.store.get_kv("dreams.state")
        state = json.loads(raw) if raw else {}
        self.verified: deque = deque([tuple(v) for v in state.get("verified", [])], maxlen=self.pol.keep_dreams)
        self.vocab: list = state.get("vocab", [NOTHING])
        self.last_night: dict = state.get("last_night", {})
        self.recent: deque = deque(state.get("recent", []), maxlen=40)
        self.stats.update(state.get("stats", {}))
        self._next_id = int(state.get("next_id", 1))
        if self.enabled:
            self._load()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    # ---------------------------------------------------------------- storage
    def _load(self) -> None:
        if not self.dir.exists():
            return
        for path in sorted(self.dir.glob("dream-*.npz")):
            try:
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z["meta"]))
                    params = [z[f"p{i}"].astype(np.float32) for i in range(meta["layers"] * 2)]
                if meta.get("dim") != self.dim:
                    continue
                self.nodes[meta["id"]] = DreamNode(meta["id"], params, meta["labels"], meta["lr"], meta["generation"],
                                                   meta.get("parent", ""), meta.get("fitness"), meta.get("created", 0.0),
                                                   meta.get("changes", []), meta.get("history", []))
            except Exception:
                continue

    def save(self) -> None:
        state = {"verified": list(self.verified), "vocab": self.vocab, "last_night": self.last_night,
                 "recent": list(self.recent), "stats": dict(self.stats), "next_id": self._next_id}
        self.sys.store.set_kv("dreams.state", json.dumps(state))

    def save_node(self, node: DreamNode) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {"id": node.id, "labels": node.labels, "lr": node.lr, "generation": node.generation, "parent": node.parent,
                "fitness": node.fitness, "created": node.created, "changes": node.changes, "history": node.history[-30:],
                "layers": len(node.params) // 2, "dim": self.dim}
        tmp = self.dir / f".{node.id}.tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(meta)), **{f"p{i}": p for i, p in enumerate(node.params)})
        tmp.replace(self.dir / f"{node.id}.npz")

    def champion(self) -> DreamNode | None:
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: (n.fitness if n.fitness is not None else -1, n.weights))

    def _new_id(self) -> str:
        nid = f"dream-{self._next_id:04d}"
        self._next_id += 1
        return nid

    # ---------------------------------------------------------------- material
    def gather(self) -> list[dict]:
        """What it can dream about: its memory nodes' facts and its retained passages (nothing external)."""
        s = self.sys
        out: list[dict] = []
        if s.memory.enabled:
            for node in s.memory.nodes.values():
                for t in node.facts():
                    said = sentences(t.answer) if len(t.answer.split()) >= 4 else [statement(t.ask, t.answer)]
                    for sent in said:  # statements only: a question is not something it knows
                        if sent and not sent.endswith("?"):
                            out.append({"text": sent, "source": f"memory node {node.id}"})
        try:
            for title, text in s.knowledge.sample_passages(self.pol.material_passages):
                for sent in [x for x in sentences(text) if not x.endswith("?")][:4]:
                    out.append({"text": sent, "source": f"retained '{title}'"})
        except Exception:
            pass
        seen, unique = set(), []
        for m in out:
            k = _norm(m["text"])
            if k not in seen and spans(m["text"]):
                seen.add(k)
                unique.append(m)
        self.rng.shuffle(unique)
        return unique[: self.pol.material_max]

    def _vocabulary(self, material: list[dict]) -> list[str]:
        counts: Counter = Counter()
        for m in material:
            for sp in spans(m["text"]):
                counts[sp.text] += 1
        for _, a, *_ in self.verified:
            counts[a] += 2
        keep = [a for a in self.vocab[1:] if a in counts]  # keep the order it already knows
        for a, _ in counts.most_common():
            if len(keep) >= self.pol.vocabulary:
                break
            if a not in keep:
                keep.append(a)
        return [NOTHING] + keep[: self.pol.vocabulary]

    def _remap_all(self, vocab: list[str]) -> None:
        if vocab == self.vocab:
            return
        for node in self.nodes.values():
            node.params = remap_outputs(node.params, node.labels, vocab, self.nrng)
            node.labels = list(vocab)
        self.vocab = vocab

    # ---------------------------------------------------------------- sleep
    def fall_asleep(self, reason: str = "idle", by: str = "dreamer") -> tuple[bool, str]:
        s = self.sys
        if not self.enabled:
            return False, "dreams are off ([dreams] enabled = false)"
        if self.asleep:
            return False, "already asleep"
        if s.guardian.halted:
            return False, "halted"
        material = self.gather()
        if len(material) < self.pol.min_material:
            return False, f"not enough to dream about yet ({len(material)} things it remembers; needs {self.pol.min_material})"
        self.material = material
        self._remap_all(self._vocabulary(material))
        if not self.nodes:
            node = DreamNode(self._new_id(), init_params(self.nrng, self.pol.inputs, tuple(self.pol.hidden), len(self.vocab)),
                             list(self.vocab), self.pol.learning_rate)
            node.history.append({"ts": time.time(), "note": "born, to dream"})
            self.nodes[node.id] = node
            self.save_node(node)
        champ = self.champion()
        self.dreamer = DreamNode("dreamer", [p.copy() for p in champ.params], list(champ.labels), champ.lr,
                                 champ.generation, champ.id)
        self.journal, self.rehearsed = [], []
        self.asleep, self.slept_at = True, time.time()
        s.guardian.sleeping = True
        self.stats["sleeps"] += 1
        idle = round(time.time() - getattr(s, "last_order_at", 0.0)) if getattr(s, "last_order_at", 0.0) else None
        s.guardian.audit(by, "dream.sleep", champ.id, reason=reason, material=len(material))
        s.bus.publish("dream.sleep", reason=reason, by=by, material=len(material), node=champ.id, idle=idle)
        return True, f"asleep: dreaming about {len(material)} things it remembers (everything else waits; an order wakes it)"

    def wake(self, reason: str, by: str = "dreamer") -> bool:
        if not self.asleep:
            return False
        s = self.sys
        self.asleep = False
        s.guardian.sleeping = False
        self.woke_at, self.wake_reason = time.time(), reason
        self.review_pending = bool(self.journal)
        answered = sum(1 for d in self.journal if d.answer)
        s.guardian.audit(by, "dream.wake", reason=reason, dreams=len(self.journal) + len(self.rehearsed))
        s.bus.publish("dream.wake", reason=reason, by=by, seconds=round(self.woke_at - self.slept_at, 1),
                      dreams=len(self.journal) + len(self.rehearsed), rehearsed=len(self.rehearsed),
                      answered=answered, imagined=sum(1 for d in self.journal if d.kind == "imagined"))
        return True

    @property
    def night_dreams(self) -> int:
        return len(self.journal) + len(self.rehearsed)

    # ---------------------------------------------------------------- dreaming
    def synthesize(self, k: int) -> list[Dream]:
        """Make up questions and answers about random pieces of what it remembers."""
        if not self.material:
            return []
        out: list[Dream] = []
        names = [(sp.text, m) for m in self.material for sp in spans(m["text"]) if sp.kind == "name"]
        for m in self.rng.sample(self.material, min(len(self.material), k)):
            sp_all = spans(m["text"])
            sp = self.rng.choice(sp_all)
            q = cloze(m["text"], sp)
            if self.rng.random() < self.pol.imagine_share and names:
                anchors = [x for x in sp_all if x.kind == "name" and x.text != sp.text and x.text in q]
                if anchors:
                    a = self.rng.choice(anchors)
                    other, om = self.rng.choice(names)
                    if other != a.text and other != sp.text and other not in q:
                        out.append(Dream(q.replace(a.text, other, 1), None, "imagined",
                                         f"{m['source']} + {om['source']}", m["text"], [other]))
                        continue
            out.append(Dream(q, sp.text, "remembered", m["source"], m["text"]))
        return out

    def dream_job(self) -> DreamJob | None:
        if self.dreamer is None or self.night_dreams >= self.pol.dreams_per_night:
            return None
        dreams = self.synthesize(self.pol.dreams_per_step)
        if not dreams:
            return None
        remembered = [d for d in dreams if d.kind == "remembered" and d.truth in self.vocab]
        cut = int(len(remembered) * self.pol.rehearse_share)
        rehearse = [(d.question, d.truth) for d in remembered[:cut]]
        replay = [(q, a) for q, a, *_ in self.rng.sample(list(self.verified), min(len(self.verified), len(rehearse)))]
        answer = remembered[cut:] + [d for d in dreams if d.kind == "imagined"]
        return DreamJob(self.dreamer.params, self.dreamer.labels, self.pol.inputs, self.dim, self.dreamer.lr,
                        self.pol.rehearse_epochs, rehearse + replay, answer, self.rng.randrange(1 << 30))

    def finish_dream(self, job: DreamJob, res: dict) -> None:
        if self.dreamer is None:
            return
        self.dreamer.params = res["params"]
        self.rehearsed += job.rehearse
        for d, (ans, conf) in zip(job.answer, res["answers"]):
            d.answer, d.confidence = ans, round(conf, 3)
            self.journal.append(d)
        self.stats["dreams"] += len(job.rehearse) + len(job.answer)
        if job.answer:
            d = self.rng.choice(job.answer)
            self.sys.bus.publish("dream.dreamt", question=d.question[:160], answer=d.answer or NOTHING, kind=d.kind,
                                 confidence=d.confidence, count=self.night_dreams)

    # ---------------------------------------------------------------- waking
    def confirm(self, d: Dream) -> bool:
        """An imagined answer is only right if something it already knows says so: the answer, the name
        swapped in, and the same relation (every other content word of the question)."""
        if not d.answer or f" {_norm(d.answer)} " in f" {_norm(d.question)} ":
            return False
        named = set(_norm(" ".join(d.anchors + [d.answer])).split())
        relation = [w for w in _norm(d.question).split()
                    if len(w) > 3 and w not in PROMPT_STOP and w not in named and w not in ("many", "much", "year", "date")]
        want = [_norm(d.answer)] + [_norm(a) for a in d.anchors] + relation
        texts = [m["text"] for m in self.material]
        s = self.sys
        try:
            texts += [h.text for h in s.knowledge.search(" ".join(d.anchors + [d.answer]), k=5)]
        except Exception:
            pass
        return any(all(f" {w} " in f" {_norm(t)} " for w in want if w) for t in texts)

    def review(self) -> EvolveJob | None:
        """Check last night's answers against its own memory; the dreams that check out are kept, and if
        any did, the dream nodes go through the genetic algorithm."""
        self.review_pending = False
        counts = Counter()
        good = []
        for d in self.journal:
            if not d.answer:
                d.verdict = "nothing"
            elif d.kind == "remembered":
                d.verdict = "correct" if _norm(d.answer) == _norm(d.truth) else "wrong"
            else:
                d.verdict = "correct" if self.confirm(d) else "unchecked"
            counts[(d.kind, d.verdict)] += 1
            if d.verdict == "correct":
                good.append(d)
                self.verified.append((d.question, d.answer, d.kind, d.source, round(time.time())))
        for d in self.rng.sample(self.journal, min(len(self.journal), 8)):
            self.recent.append({"question": d.question, "answer": d.answer or NOTHING, "kind": d.kind, "verdict": d.verdict,
                                "truth": d.truth, "confidence": d.confidence})
        rem = [d for d in self.journal if d.kind == "remembered"]
        ima = [d for d in self.journal if d.kind == "imagined"]
        self.last_night = {"slept": round(self.slept_at), "woke": round(self.woke_at), "reason": self.wake_reason,
                           "dreams": self.night_dreams, "rehearsed": len(self.rehearsed),
                           "remembered": len(rem), "remembered_correct": counts[("remembered", "correct")],
                           "imagined": len(ima), "imagined_confirmed": counts[("imagined", "correct")],
                           "correct": len(good), "evolved": False}
        self.stats["correct"] += len(good)
        self.stats["checked"] += len(self.journal)
        self.sys.bus.publish("dream.reviewed", **self.last_night)
        self.save()
        if not good or self.dreamer is None:
            self.dreamer = None
            return None
        rows = [(q, a) for q, a, *_ in self.verified]
        test = [r for r in rows if zlib.crc32(r[0].encode()) % 4 == 0]
        trainset = [r for r in rows if zlib.crc32(r[0].encode()) % 4 != 0] + list(self.rehearsed)[-400:]
        npol = self.sys.directives.nodes
        job = EvolveJob([(n.id, n.params, n.lr, n.generation, n.fitness) for n in self.nodes.values()],
                        (self.dreamer.params, self.dreamer.lr), list(self.vocab), self.pol.inputs, self.dim, trainset, test,
                        self.pol.children, self.pol.population, self.pol.ga_epochs, self.sys.directives.evolution,
                        min(npol.max_hidden, self.pol.max_hidden), npol.max_depth, self.rng.randrange(1 << 30))
        self.dreamer = None
        return job

    def finish_review(self, job: EvolveJob, res: dict) -> None:
        before = self.champion()
        old = (before.id, before.fitness) if before else (None, None)
        kept = {}
        for c in res["population"]:
            nid = c["id"] or self._new_id()
            node = self.nodes.get(nid) if c["id"] else None
            if node is None:
                node = DreamNode(nid, c["params"], list(job.labels), c["lr"], c["generation"], c["parent"],
                                 c["fitness"], changes=c["changes"])
                node.history.append({"ts": time.time(), "note": f"born on waking from {c['parent']}: "
                                                                f"{'; '.join(c['changes']) or 'as it dreamt'}"})
            node.fitness = c["fitness"]
            kept[nid] = node
        for nid in set(self.nodes) - set(kept):
            path = self.dir / f"{nid}.npz"
            if path.exists():
                path.unlink()
        self.nodes = kept
        for n in kept.values():
            self.save_node(n)
        champ = self.champion()
        self.last_night.update(evolved=True, champion=champ.id, fitness=champ.fitness, was=old[0], was_fitness=old[1],
                               generation=champ.generation, shape=champ.shape(), tested=res["tested"])
        self.stats["evolutions"] += 1
        self.save()
        self.sys.guardian.audit("dreamer", "dream.evolved", champ.id, fitness=champ.fitness, was=old[0],
                                was_fitness=old[1], population=list(kept))
        self.sys.bus.publish("dream.evolved", node=champ.id, fitness=champ.fitness, was=old[0], was_fitness=old[1],
                             generation=champ.generation, shape=champ.shape(), population=len(kept), tested=res["tested"])

    # ---------------------------------------------------------------- imagining
    def _blend(self, prompt: str, ideas: list[str]) -> tuple[str | None, str | None]:
        """Blend the prompt into something it remembers. If the nearest statement relates other things
        the same way ("Ottawa is the capital of Canada" for "the capital of Australia"), swap the prompt's
        name in: an analogy. If it already relates the prompt's things, the prompt is a what-if ("Jane
        Austen painted the Mona Lisa"): imagine what follows ("then perhaps Leonardo da Vinci wrote Pride
        and Prejudice")."""
        pool = [m["text"] for m in (self.material or self.gather())]  # what it remembers, as statements
        names = [sp.text for sp in spans(prompt[:1].upper() + prompt[1:]) if sp.kind == "name"]
        pw = _norm(prompt).split()
        name_words = set(_norm(" ".join(names)).split())

        def stems(t: str, skip: set = frozenset()) -> set:
            return {w[:5] for w in _norm(t).split() if len(w) > 3 and w not in PROMPT_STOP and w not in skip}
        want = stems(prompt, name_words)
        idea_names = [i for i in ideas if not _NUMBER.fullmatch(i)]
        best = None  # (score, blend, template, swapped_out, swapped_in, what_if)
        for t in pool:
            overlap = len(want & stems(t))
            if not overlap:
                continue
            t_names = [sp.text for sp in spans(t) if sp.kind == "name"]
            what_if = any(n in t for n in names)
            incoming = [n for n in names if n not in t] or [i for i in idea_names if i not in t][:1]
            bonus = 0.5 if any(i in t for i in ideas) else 0.0
            for n in incoming:
                for x in t_names:
                    if x in names or x == n or n in x or x in n:
                        continue
                    b = t.replace(x, n, 1)
                    score = (overlap + bonus, SequenceMatcher(None, _norm(b).split(), pw).ratio())
                    if best is None or score > best[0]:
                        best = (score, b, t, x, n, what_if and n in names)
        if best is None:
            return None, None
        _, b, t, x, n, what_if = best

        def sentence(text: str) -> str:
            return (text[:1].lower() + text[1:] if text.split()[0] in STARTERS else text).rstrip(" .") + "."
        if not what_if:
            return f"Perhaps {sentence(b)}", t
        for u in pool:  # what follows: the one it displaced takes the newcomer's place in turn
            if n in u and x not in u and u != t:
                return f"If {n} took {x}'s place, then perhaps {sentence(u.replace(n, x, 1))}", u
        return None, t

    def imagine(self, prompt: str) -> dict | None:
        """What its best dream node imagines for a prompt: what comes to mind, blended into something it
        remembers. Never a fact, and never stored as one."""
        champ = self.champion()
        prompt = " ".join(str(prompt).split()).strip(" ?.!")
        if champ is None or not prompt:
            return None
        P = champ.predict(encode_many([prompt], self.dim), self.pol.inputs)[0]
        said = f" {_norm(prompt)} "
        order = [int(i) for i in np.argsort(-P) if i != 0]
        fresh = [i for i in order if f" {_norm(champ.labels[i])} " not in said]  # not just the prompt back
        ideas = [(champ.labels[i], float(P[i])) for i in fresh[:3] if P[i] >= self.pol.imagine_threshold]
        if not ideas and (fresh or order):
            i = (fresh or order)[0]
            ideas = [(champ.labels[i], float(P[i]))]
        if not ideas:
            return None
        idea = ideas[0][0]
        blend, best = self._blend(prompt, [i for i, _ in ideas])
        pool = self.material or self.gather()

        def named(i: str) -> str:  # "the Solar System", as it is said in what it remembers
            return f"the {i}" if any(f"the {i}" in m["text"] for m in pool) else i
        also = ", ".join(named(i) for i, _ in ideas[1:])
        lines = [f"I imagine {named(idea)}" + (f" (or perhaps {also})" if also else "") + "."]
        if blend:
            lines.append(f"Or p{blend[1:]}" if blend.startswith("Perhaps ") and idea not in blend else blend)
        fit = f"{champ.fitness:.0%} of its checked dreams right" if champ.fitness is not None else "not checked yet"
        self.stats["imagined"] += 1
        return {"text": "\n".join(lines), "node": champ.id, "ideas": ideas, "blended_from": best,
                "source": f"imagined by dream node {champ.id} (generation {champ.generation}, {fit}); not a fact"}

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        champ = self.champion()
        return {"enabled": self.enabled, "asleep": self.asleep,
                "slept_for": round(time.time() - self.slept_at) if self.asleep else None,
                "awake_for": None if self.asleep else round(time.time() - self.woke_at),
                "dreams_tonight": self.night_dreams if self.asleep else 0,
                "review_pending": self.review_pending,
                "nodes": [{"id": n.id, "generation": n.generation, "shape": n.shape(), "weights": n.weights,
                           "fitness": n.fitness, "parent": n.parent, "champion": champ is not None and n.id == champ.id,
                           "changes": n.changes[-3:]}
                          for n in sorted(self.nodes.values(), key=lambda n: -(n.fitness or 0))],
                "verified": len(self.verified), "vocabulary": len(self.vocab) - 1, "last_night": self.last_night,
                "recent": list(self.recent)[-12:], **{k: v for k, v in self.stats.items()}}


# ============================================================================ agent
class Dreamer(Agent):
    """Sleep and dreams (see the module docstring)."""
    name = "dreamer"

    def __init__(self, system):
        super().__init__(system)
        self.next_check = time.time() + system.directives.dreams.check_every_seconds
        self.rng = random.Random(system.rng.randrange(1 << 30))

    @property
    def interval(self) -> float:
        return max(0.05, self.sys.directives.background.tick_seconds)

    async def bg(self, fn, *args):
        """Dreaming gets the background capacity everything else has given up while asleep."""
        await self.sys.background_slot(dreaming=True)
        t0 = time.monotonic()
        try:
            import asyncio
            return await asyncio.to_thread(fn, *args)
        finally:
            self.sys.guardian.note_background_busy(time.monotonic() - t0)

    async def step(self) -> None:
        s, d = self.sys, self.sys.dreams
        pol = d.pol
        if not d.enabled:
            if d.asleep:
                d.wake("dreams were turned off")
            return
        if s.guardian.halted or s.guardian.paused:
            if d.asleep:
                d.wake("halted" if s.guardian.halted else "background work was paused")
            return
        if d.asleep:
            if s.inflight or not s.orders.empty():
                d.wake("an order arrived")
                return
            if time.time() - d.slept_at >= pol.sleep_max_seconds or d.night_dreams >= pol.dreams_per_night:
                d.wake("rested")
                return
            job = d.dream_job()
            if job is None:
                d.wake("nothing more to dream about")
                return
            with s.activity.track("dream", "dreamer", job.describe()):
                res = await self.bg(job.run)
            d.finish_dream(job, res)
            self.stats["dream steps"] += 1
            return
        if d.review_pending:
            if s.inflight or not s.guardian.background_allowed():
                return
            job = d.review()
            if job is not None:
                with s.activity.track("dream", "dreamer", "waking: the genetic algorithm on the dream nodes"):
                    res = await self.bg(job.run)
                d.finish_review(job, res)
                self.stats["evolutions"] += 1
            return
        now = time.time()
        if now < self.next_check:
            return
        self.next_check = now + pol.check_every_seconds
        if s.inflight or now - getattr(s, "last_order_at", 0.0) < pol.sleep_after_idle_seconds:
            return
        if not s.guardian.background_allowed():
            return
        if self.rng.random() < pol.sleep_chance:
            ok, _ = d.fall_asleep("idle")
            if ok:
                self.stats["sleeps"] += 1
