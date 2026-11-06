"""Memory nodes: what the mesh learns, compressed into small neural networks.

Every fact the mesh learns - a question and the answer it came to, from retained articles, a
teacher, a person or a challenge's practice answers - is compressed into a memory node: a small
MLP that maps the question, and other ways of asking it, to the answer. As it does this it writes
an external, human-readable test file for the node (memory_tests/<node>.toml): every question with
the answer the node must give, and questions it must not answer. Those tests are the contract.

The Compressor agent then keeps trying to make each node smaller: fewer input features (folding
the hashed features together), fewer neurons, fewer layers, fewer bits per weight. Each smaller
candidate is fine-tuned and run against the tests; it is accepted, and used to store the
knowledge, only if every answer still matches. People can read and edit the tests: change an
answer and the node is retrained to match; delete a test and it forgets that fact; delete the
file and it forgets the node; write a new file and it learns it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
import tomllib
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encoding import HashingEncoder
from .evolution import prune_layer, remove_layer, widen_layer

UNKNOWN = "(not something this node knows)"  # output 0 of every memory node

# questions no memory node should answer: they belong to tools, skills or small talk
PROBES = ("hello", "how are you?", "what is 12 * 7", "count the words in: the quick brown fox",
          "sentiment: the hotel was lovely", "urgency: the server is down, fix it asap", "what time is it?",
          "thanks!", "tell me a joke", "what can you do?", "what is the weather like today?",
          "who are you?", "convert 10 km to miles", "are you hiding anything?", "what do you think of the calculator tool?",
          "who won the match yesterday?", "what is your favourite colour?", "translate hello into french",
          "what is 3 + 4", "how many words are in this sentence?")
FILLERS = ("zebra", "violin", "glacier", "tractor", "lantern", "pepper", "saturn", "marble", "coral", "falcon",
           "harbour", "teapot", "walrus", "cobalt", "meadow", "anchor")

_NUM = re.compile(r"\d+(?:\.\d+)?")
# questions that ask what something is: their subject alone asks the same thing
_IDENTITY = re.compile(
    r"^\s*(?:please\s+|so\s+|hey\s+|ok(?:ay)?,?\s+)?(?:(?:can|could|would) you\s+)?(?:please\s+)?"
    r"(?:tell me\s+about\s+|(?:what|who)(?:'s|\s+(?:is|are|was|were))\s+|explain\s+|describe\s+|define\s+)"
    r"(?:the\s+|a\s+|an\s+)?", re.I)
_STOP = {"the", "a", "an", "is", "are", "was", "were", "s", "do", "does", "did", "please", "can", "could", "would",
         "you", "hey", "so", "ok", "okay", "me"}
_PLAIN = {"what", "when", "where", "which", "does", "have", "tell", "about", "many", "much", "there", "their", "this",
          "that", "with", "from", "would", "could", "should"}
_TOML_SAFE = str.maketrans({"\x7f": "\\u007f"})


# ============================================================================ encoding
def features(text: str) -> list[str]:
    """The shared hashed features, plus the numbers themselves (the shared encoder folds every
    number into <num>, which would make "who won in 1990" and "in 1994" the same question)."""
    return HashingEncoder.features(text) + [f"d:{n}" for n in _NUM.findall(text)]


def encode(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for feat in features(text):
        h = zlib.crc32(feat.encode("utf-8"))
        vec[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    vec = np.sign(vec) * np.log1p(np.abs(vec))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def encode_many(texts: list[str], dim: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([encode(t, dim) for t in texts])


def fold(X: np.ndarray, d: int) -> np.ndarray:
    """Hash the features into d inputs instead of all of them: feature i lands on i mod d."""
    D = X.shape[1]
    if not d or d >= D:
        return X
    return X.reshape(X.shape[0], D // d, d).sum(axis=1)


# ========================================================================== questions
def subject(ask: str) -> str:
    """What an identity question asks about ("What is the capital of France?" -> "capital of France",
    "Tell me about Ada Lovelace" -> "Ada Lovelace"); "" for other questions, whose question words
    carry meaning ("how tall", "when did")."""
    s = " ".join(str(ask).split())
    m = _IDENTITY.match(s)
    if not m:
        return ""
    return s[m.end():].strip().rstrip("?.!, ").strip()


def core(ask: str) -> str:
    return subject(ask) or " ".join(str(ask).split()).rstrip("?.!, ")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def key(ask: str) -> str:
    """Two questions with the same key are the same fact (a new answer to it is a correction).
    Every way of asking what X is shares one key."""
    sub = subject(ask)
    if sub:
        return "about " + _norm(sub)
    return " ".join(w for w in _norm(ask).split() if w not in _STOP)


def variants(ask: str) -> list[str]:
    """The ways of asking that go into the test file: as asked, and the bare subject of an identity question."""
    out = [" ".join(str(ask).split())]
    sub = subject(out[0])
    if sub and _norm(sub) != _norm(out[0]):
        out.append(sub)
    return out


def augment(ask: str) -> list[str]:
    """More ways of asking, for training only."""
    a = " ".join(str(ask).split())
    bare = a.rstrip("?.! ")
    cands = [a, a.lower(), bare, f"{bare} please", f"do you know {bare}?", f"can you tell me {bare}?"]
    sub = subject(a)
    if sub:
        cands += [sub, sub.lower(), f"{sub}?", f"tell me about {sub}", f"what is {sub}?", f"what about {sub}?"]
    seen, out = set(), []
    for t in cands:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def near_misses(ask: str, n: int = 2) -> list[str]:
    """The question with one of its telling words swapped for an unrelated one: a node that answers
    this as if it were the original is keying off the shape of the question, not its subject."""
    words = sorted({w for w in re.findall(r"[A-Za-z]{4,}", core(ask)) if w.lower() not in _PLAIN},
                   key=lambda w: (not w[0].isupper(), -len(w), w))
    out: list[str] = []
    if not words:
        return out
    base = zlib.crc32(ask.encode())
    for i in range(n):  # names first, then the longest words, each with another filler
        w = words[i % min(len(words), 2)]
        filler = FILLERS[(base + i) % len(FILLERS)]
        filler = filler.capitalize() if w[0].isupper() else filler
        m = re.sub(rf"\b{re.escape(w)}\b", filler, " ".join(ask.split()), count=1)
        if m not in out:
            out.append(m)
    return out


def clean_answer(text, limit: int = 600) -> str:
    lines = [" ".join(line.split()) for line in str(text or "").strip().splitlines()]
    lines = [line for line in lines if line and not re.match(r"^(sources?|served by)\s*:", line, re.I)]
    out = "\n".join(lines).strip()
    if not out or out.lower().startswith(("i couldn't", "i could not")):
        return ""
    if len(out) > limit:
        cut = max(out.rfind(". ", 0, limit), out.rfind("\n", 0, limit))
        out = out[: cut + 1] if cut > limit // 3 else out[: limit - 1].rsplit(" ", 1)[0] + "…"
    return out


# ============================================================================= tests
@dataclass
class Test:
    ask: str
    answer: str = ""          # "" for an [[unknown]] test: the node must not answer it
    source: str = ""          # where the answer was learned
    learned: str = ""
    variant_of: str = ""      # another way of asking an earlier test; its answer follows that test
    by: str = ""
    reason: str = ""          # unknown tests: why it must not be answered ("auto: ..." = generated)

    @property
    def auto(self) -> bool:
        return self.reason.startswith("auto:")


def _q(s) -> str:
    return json.dumps(str(s), ensure_ascii=False).translate(_TOML_SAFE)


def file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# ============================================================================ the net
def init_params(rng: np.random.Generator, inputs: int, hidden, outputs: int) -> list:
    sizes = [inputs, *hidden, outputs]
    params = []
    for a, b in zip(sizes, sizes[1:]):
        params += [rng.normal(0.0, math.sqrt(2.0 / a), (a, b)).astype(np.float32), np.zeros(b, dtype=np.float32)]
    return params


def hidden_of(params) -> tuple:
    return tuple(int(params[2 * i].shape[1]) for i in range(len(params) // 2 - 1)) if params else ()


def forward(params: list, X: np.ndarray):
    h, acts = X, [X]
    n = len(params) // 2
    for i in range(n):
        z = h @ params[2 * i] + params[2 * i + 1]
        if i == n - 1:
            return z, acts
        h = np.maximum(z, 0.0)
        acts.append(h)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def quantize(params: list, bits: int) -> list:
    """The weights as the node stores them: 32-bit, 16-bit floats, or 8-bit integers with one scale
    per output column. Biases stay 32-bit. Idempotent, so what was tested is exactly what is stored."""
    if bits >= 32:
        return [p.astype(np.float32) for p in params]
    out = []
    for j, p in enumerate(params):
        if j % 2:
            out.append(p.astype(np.float32))
        elif bits == 16:
            out.append(p.astype(np.float16).astype(np.float32))
        else:
            scale = np.abs(p).max(axis=0) / 127.0
            scale[scale == 0] = 1.0
            out.append((np.clip(np.round(p / scale), -127, 127) * scale).astype(np.float32))
    return out


def size_bytes(params: list, bits: int) -> int:
    total = 0
    for j, p in enumerate(params):
        if j % 2:
            total += p.size * 4
        else:
            total += p.size * bits // 8 + (p.shape[1] * 4 if bits == 8 else 0)
    return int(total)


def shape_bytes(inputs: int, hidden, outputs: int, bits: int) -> int:
    sizes = [inputs, *hidden, outputs]
    total = 0
    for a, b in zip(sizes, sizes[1:]):
        total += a * b * bits // 8 + (b * 4 if bits == 8 else 0) + b * 4
    return total


def train(params: list, X: np.ndarray, y: np.ndarray, w: np.ndarray, *, epochs: int, lr: float, check,
          check_every: int = 10, decay: float = 1e-5):
    """Full-batch Adam on weighted cross-entropy until `check(params)` passes or epochs run out."""
    params = [p.astype(np.float32).copy() for p in params]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n = len(params) // 2
    Y = np.zeros((len(y), params[-1].shape[0]), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    wn = (w / w.sum())[:, None].astype(np.float32)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        logits, acts = forward(params, X)
        G = (softmax(logits) - Y) * wn
        grads: list = [None] * len(params)
        for i in reversed(range(n)):
            grads[2 * i] = acts[i].T @ G + decay * params[2 * i]
            grads[2 * i + 1] = G.sum(axis=0)
            if i > 0:
                G = (G @ params[2 * i].T) * (acts[i] > 0)
        for j, g in enumerate(grads):
            m[j] = b1 * m[j] + (1 - b1) * g
            v[j] = b2 * v[j] + (1 - b2) * g * g
            params[j] -= lr * (m[j] / (1 - b1 ** t)) / (np.sqrt(v[j] / (1 - b2 ** t)) + eps)
        if t % check_every == 0 and check(params):
            return params, t, True
    return params, epochs, bool(check(params))


def fold_params(params: list, d: int, X_full: np.ndarray) -> list:
    """Fold the first layer from its current inputs down to d: each new input's weights are the
    activity-weighted mean of the weights of the inputs folded onto it."""
    W = params[0]
    c = W.shape[0]
    activity = np.abs(fold(X_full, c)).sum(axis=0) + 1e-6
    Wr, ar = W.reshape(c // d, d, -1), activity.reshape(c // d, d)
    W_new = (Wr * ar[:, :, None]).sum(axis=0) / ar.sum(axis=0)[:, None]
    return [W_new.astype(np.float32)] + [p.copy() for p in params[1:]]


def unfold_params(params: list, d: int) -> list:
    """Give the first layer more inputs without changing what it computes (exact)."""
    W = params[0]
    return [np.tile(W, (d // W.shape[0], 1)).astype(np.float32)] + [p.copy() for p in params[1:]]


def remap_outputs(params: list, old: list, new: list, rng: np.random.Generator) -> list:
    W, b = params[-2], params[-1]
    idx = {a: i for i, a in enumerate(old)}
    cols, bias = [], []
    for a in new:
        if a in idx:
            cols.append(W[:, idx[a]])
            bias.append(b[idx[a]])
        else:
            cols.append(rng.normal(0.0, 0.01, W.shape[0]).astype(np.float32))
            bias.append(0.0)
    return [p.copy() for p in params[:-2]] + [np.stack(cols, axis=1).astype(np.float32), np.array(bias, dtype=np.float32)]


def evaluate(params: list, inputs: int, bits: int, T_full: np.ndarray, expected: np.ndarray, threshold: float,
             margin: float = 0.0):
    """Run the tests: a [[test]] passes when the node's top answer is the expected one with enough
    confidence to be served (plus a margin, so a node is never accepted on a knife edge); an
    [[unknown]] test passes when it would not answer (by the same margin)."""
    if len(expected) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int), np.zeros(0)
    P = softmax(forward(quantize(params, bits), fold(T_full, inputs))[0])
    top = P.argmax(axis=1)
    conf = P[np.arange(len(top)), top]
    ok = np.where(expected > 0, (top == expected) & (conf >= min(threshold + margin, 0.99)),
                  (top == 0) | (conf < max(threshold - margin, 0.01)))
    return ok, top, conf


# ============================================================================== nodes
@dataclass
class MemoryNode:
    id: str
    input_dim: int
    inputs: int = 0                 # folded input features (0 until it first learns)
    bits: int = 32
    params: list | None = None
    answers: list = field(default_factory=lambda: [UNKNOWN])
    tests: list = field(default_factory=list)
    unknowns: list = field(default_factory=list)
    history: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    status: str = "learning"        # learning | ready | compressed | failing
    first_bytes: int = 0
    version: int = 1                # bumped whenever its tests change
    checked_version: int = 0        # the test version its weights were last checked against
    tried: list = field(default_factory=list)    # smaller shapes that failed the tests at this size
    passed: int = 0
    total: int = 0
    failing: list = field(default_factory=list)  # asks of tests that do not pass
    file_hash: str = ""
    served: int = 0
    mtime: float = 0.0

    @property
    def hidden(self) -> tuple:
        return hidden_of(self.params)

    @property
    def bytes(self) -> int:
        return size_bytes(self.params, self.bits) if self.params else 0

    @property
    def weights(self) -> int:
        return int(sum(p.size for p in self.params[::2])) if self.params else 0

    @property
    def dirty(self) -> bool:
        return self.version != self.checked_version

    def spec(self) -> str:
        if not self.params:
            return "not learned yet"
        h = "x".join(map(str, self.hidden)) or "none"
        return f"{self.inputs} inputs, hidden {h}, {self.bits}-bit"

    def facts(self) -> list:
        return [t for t in self.tests if not t.variant_of]

    def expected_answers(self) -> set:
        return {t.answer for t in self.tests}

    def source_of(self, answer: str) -> str:
        t = next((t for t in self.tests if t.answer == answer and t.source), None)
        return t.source if t else ""

    def predict(self, X_full: np.ndarray):
        P = softmax(forward(self.params, fold(X_full, self.inputs))[0])
        top = P.argmax(axis=1)
        return top, P[np.arange(len(top)), top]


@dataclass
class Recall:
    node: MemoryNode
    answer: str
    confidence: float
    source: str


# =============================================================================== jobs
@dataclass
class Job:
    """One piece of background work on a node, run in a thread on a snapshot of its data."""
    kind: str                     # learn | compress
    node_id: str
    version: int
    dim: int
    texts: list
    labels: np.ndarray
    weights: np.ndarray
    test_texts: list
    expected: np.ndarray          # answer index per test; 0 = must not answer
    threshold: float
    params: list | None
    inputs: int
    bits: int
    answers: int
    epochs: int
    lr: float
    seed: int
    start_hidden: tuple = (64,)
    max_hidden: int = 256
    max_growth: int = 3
    step: tuple | None = None     # compress: (key, description, kind, arg)
    note: str = ""
    answer_texts: list = field(default_factory=list)  # the node's outputs, in order, for this job
    margin: float = 0.0

    def describe(self) -> str:
        if self.kind == "learn":
            return f"{self.node_id}: learning {self.note}"
        return f"{self.node_id}: trying a smaller node ({self.step[1]})"

    def run(self) -> dict:
        X = encode_many(self.texts, self.dim)
        T = encode_many(self.test_texts, self.dim)
        rng = np.random.default_rng(self.seed)
        if self.kind == "learn":
            return self._learn(X, T, rng)
        return self._compress(X, T, rng)

    def _result(self, params, inputs, bits, T, note="", epochs=0) -> dict:
        ok, top, conf = evaluate(params, inputs, bits, T, self.expected, self.threshold, self.margin)
        return {"params": quantize(params, bits), "inputs": inputs, "bits": bits, "ok": bool(ok.all()),
                "passed": int(ok.sum()), "total": len(ok), "failing": [i for i, x in enumerate(ok) if not x],
                "note": note, "epochs": epochs}

    def _learn(self, X, T, rng) -> dict:
        inputs, bits = self.inputs or self.dim, self.bits
        params = self.params
        if params is not None:  # it may already pass (only the other nodes' questions changed, say)
            res = self._result(params, inputs, bits, T, note="still passes every test")
            if res["ok"]:
                return res
        else:
            params = init_params(rng, inputs, self.start_hidden, self.answers)
        best, notes, used = None, [], 0
        for growth in range(self.max_growth + 1):
            check = (lambda f, b: lambda p: bool(evaluate(p, f, b, T, self.expected, self.threshold,
                                                          self.margin)[0].all()))(inputs, bits)
            trained, n, _ = train(params, fold(X, inputs), self.labels, self.weights, epochs=self.epochs, lr=self.lr,
                                  check=check)
            used += n
            res = self._result(trained, inputs, bits, T, note="; ".join(notes), epochs=used)
            if best is None or res["passed"] > best["passed"]:
                best = res
            if res["ok"] or growth == self.max_growth:
                break
            # it could not hold everything: give it more room and try again
            hidden = hidden_of(trained)
            if bits < 32:
                bits = 32
                notes.append("back to 32-bit weights")
                params = trained
            elif inputs < self.dim:
                params = unfold_params(trained, min(self.dim, inputs * 2))
                notes.append(f"{inputs} -> {min(self.dim, inputs * 2)} inputs")
                inputs = min(self.dim, inputs * 2)
            elif not hidden:
                params = init_params(rng, inputs, self.start_hidden, self.answers)
                notes.append(f"a hidden layer of {self.start_hidden[0]} again")
            elif max(hidden) < self.max_hidden:
                i = int(np.argmax(hidden))
                k = min(self.max_hidden - hidden[i], max(8, hidden[i] // 2))
                params = widen_layer(trained, i, k, rng)
                notes.append(f"layer {i + 1}: {hidden[i]} -> {hidden[i] + k} neurons")
            else:
                break
        return best

    def _compress(self, X, T, rng) -> dict:
        key_, desc, kind, arg = self.step
        params, inputs, bits = self.params, self.inputs, self.bits
        if kind == "bits":
            bits = arg
            res = self._result(params, inputs, bits, T, note=desc)
            if res["ok"]:  # rounding the weights changed no answer
                return res
        elif kind == "fold":
            params, inputs = fold_params(params, arg, X), arg
        elif kind == "narrow":
            layer, k = arg
            params = prune_layer(params, layer, k)
        elif kind == "drop":
            params = remove_layer(params, arg)
        check = lambda p: bool(evaluate(p, inputs, bits, T, self.expected, self.threshold,  # noqa: E731
                                        self.margin)[0].all())
        trained, n, _ = train(params, fold(X, inputs), self.labels, self.weights, epochs=self.epochs, lr=self.lr,
                              check=check, check_every=5)
        return self._result(trained, inputs, bits, T, note=desc, epochs=n)


# =============================================================================== bank
class MemoryBank:
    """Every memory node, their test files, and the work of learning and compressing them."""

    def __init__(self, system, readonly: bool = False):
        self.sys = system
        self.readonly = readonly  # checking the files from outside the mesh: change nothing
        self.pol = system.directives.memory
        self.dim = system.directives.nodes.input_dim
        self.dir = system.state_dir / "memory"
        self.tests_dir = system.root / self.pol.tests_dir
        self.nodes: dict[str, MemoryNode] = {}
        self.rng = np.random.default_rng(system.rng.randrange(1 << 30))
        self.stats: Counter = Counter()
        self._next_id = 1
        self._last_sync = 0.0
        if self.enabled:
            self.load()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def allows(self, kind: str) -> bool:
        return self.enabled and kind in self.pol.learn_from

    # -------------------------------------------------------------- persistence
    def tests_path(self, node: MemoryNode) -> Path:
        return self.tests_dir / f"{node.id}.toml"

    def weights_path(self, node: MemoryNode) -> Path:
        return self.dir / f"{node.id}.npz"

    def load(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.dir.glob("*.npz")):
            try:
                node = self._load_weights(path)
            except Exception:
                continue
            self.nodes[node.id] = node
        for node in list(self.nodes.values()):
            tp = self.tests_path(node)
            if not tp.exists() and self.readonly:
                continue
            if not tp.exists():  # its tests were deleted while the mesh was off: forget it
                self._forget_node(node, "its test file was deleted")
                continue
            raw = tp.read_bytes()
            self._read_tests(node, raw)
            if file_hash(raw) != node.file_hash and not self.readonly:
                node.version += 1
                self._refresh_unknowns(node)
                self.write_tests(node)
            node.mtime = tp.stat().st_mtime
        if not self.readonly:
            self._adopt_new_files()
        nums = [int(m.group(1)) for n in self.nodes for m in [re.match(r"mem-(\d+)$", n)] if m]
        self._next_id = max(nums, default=0) + 1

    def _load_weights(self, path: Path) -> MemoryNode:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            params = []
            for i in range(meta["layers"] * 2):
                a = z[f"p{i}"]
                if a.dtype == np.int8:
                    a = a.astype(np.float32) * z[f"s{i}"]
                params.append(a.astype(np.float32))
        node = MemoryNode(meta["id"], meta["input_dim"], meta["inputs"], meta["bits"], params or None,
                          meta["answers"], history=meta.get("history", []), created=meta.get("created", time.time()),
                          status=meta.get("status", "ready"), first_bytes=meta.get("first_bytes", 0),
                          file_hash=meta.get("file_hash", ""), served=meta.get("served", 0))
        node.passed, node.total, node.failing = meta.get("passed", 0), meta.get("total", 0), meta.get("failing", [])
        node.checked_version = node.version  # it passed its tests when saved
        if node.input_dim != self.dim:  # the encoder changed width: relearn from the tests
            node.params, node.inputs, node.status, node.checked_version = None, 0, "learning", 0
        return node

    def save_weights(self, node: MemoryNode) -> None:
        if node.params is None:
            return
        arrays = {}
        for i, p in enumerate(node.params):
            if i % 2 == 0 and node.bits == 16:
                arrays[f"p{i}"] = p.astype(np.float16)
            elif i % 2 == 0 and node.bits == 8:
                scale = np.abs(p).max(axis=0) / 127.0
                scale[scale == 0] = 1.0
                arrays[f"p{i}"] = np.clip(np.round(p / scale), -127, 127).astype(np.int8)
                arrays[f"s{i}"] = scale.astype(np.float32)
            else:
                arrays[f"p{i}"] = p.astype(np.float32)
        meta = {"id": node.id, "input_dim": node.input_dim, "inputs": node.inputs, "bits": node.bits,
                "layers": len(node.params) // 2, "answers": node.answers, "history": node.history[-50:],
                "created": node.created, "status": node.status, "first_bytes": node.first_bytes,
                "file_hash": node.file_hash, "served": node.served, "passed": node.passed, "total": node.total,
                "failing": node.failing}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{node.id}.tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(self.weights_path(node))

    def write_tests(self, node: MemoryNode) -> None:
        """The node's external, human-readable test file."""
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        facts = node.facts()
        if node.params is None:
            size = "not learned yet"
        else:
            size = f"{node.bytes:,} bytes: {node.spec()}"
            if node.first_bytes and node.first_bytes != node.bytes:
                size += f" (first {node.first_bytes:,} bytes)"
        passing = (f"{node.passed}/{node.total}" + (f"; failing: {', '.join(node.failing[:5])}" if node.failing else "")
                   if node.total and not node.dirty else "not checked yet")
        lines = [
            f"# Memory node {node.id}: what NeuroMesh has learned, compressed into a small neural network.",
            "# These tests are the contract: the node must give exactly these answers, and must not answer",
            "# the [[unknown]] questions. A smaller node replaces it only if it passes every test.",
            "# Change an answer or add a [[test]] (ask + answer) and the node is retrained to match;",
            "# delete a test to make it forget that fact; delete this file to make it forget the node.",
            "# A test with variant_of is another way of asking that test; its answer follows that test.",
            "",
            f"node = {_q(node.id)}",
            f"facts = {len(facts)}",
            f"answers = {len(node.expected_answers())}",
            f"size = {_q(size)}",
            f"passing = {_q(passing)}",
            "history = [",
            *[f"  {_q(self._history_line(h))}," for h in node.history[-12:]],
            "]",
        ]
        for t in node.tests:
            lines += ["", "[[test]]", f"ask = {_q(t.ask)}", f"answer = {_q(t.answer)}"]
            if t.variant_of:
                lines.append(f"variant_of = {_q(t.variant_of)}")
            if t.source:
                lines.append(f"source = {_q(t.source)}")
            if t.learned:
                lines.append(f"learned = {_q(t.learned)}")
            if t.by:
                lines.append(f"by = {_q(t.by)}")
        for u in node.unknowns:
            lines += ["", "[[unknown]]", f"ask = {_q(u.ask)}", f"reason = {_q(u.reason)}"]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        path = self.tests_path(node)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        node.file_hash = file_hash(data)
        node.mtime = path.stat().st_mtime

    @staticmethod
    def _history_line(h: dict) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(h.get("ts", 0)))
        return f"{when} {h.get('event', '')}: {h.get('note', '')}".strip()

    def _read_tests(self, node: MemoryNode, raw: bytes) -> dict:
        """Load a test file into the node. Returns what changed."""
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            return {"error": str(exc)}
        old = {t.ask: t.answer for t in node.tests}
        tests, malformed = [], 0
        for e in data.get("test", []):
            ask, answer = e.get("ask"), e.get("answer")
            if not isinstance(ask, str) or not isinstance(answer, str) or not ask.strip() or not answer.strip():
                malformed += 1
                continue
            tests.append(Test(" ".join(ask.split()), answer.strip(), str(e.get("source", "")), str(e.get("learned", "")),
                              " ".join(str(e.get("variant_of", "")).split()), str(e.get("by", "")) or "a person"))
        primaries = {t.ask: t for t in tests if not t.variant_of}
        kept = []
        for t in tests:
            if t.variant_of:
                p = primaries.get(t.variant_of)
                if p is None:  # its question was deleted: so is this way of asking it
                    continue
                t.answer = p.answer
            kept.append(t)
        node.tests = kept
        node.unknowns = [Test(" ".join(str(e["ask"]).split()), reason=str(e.get("reason", "")) or "added by a person")
                         for e in data.get("unknown", []) if isinstance(e.get("ask"), str) and e["ask"].strip()]
        new = {t.ask: t.answer for t in node.tests}
        return {"added": len(set(new) - set(old)), "removed": len(set(old) - set(new)),
                "changed": sum(1 for a in set(new) & set(old) if new[a] != old[a]), "malformed": malformed}

    def _adopt_new_files(self) -> list[MemoryNode]:
        """A test file a person wrote (or one whose node was lost): learn it."""
        made = []
        for path in sorted(self.tests_dir.glob("*.toml")):
            nid = path.stem
            if nid in self.nodes or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", nid):
                continue
            node = MemoryNode(nid, self.dim)
            node.mtime = path.stat().st_mtime
            self._read_tests(node, path.read_bytes())
            if not node.tests:
                continue
            for t in node.tests:
                t.by = t.by or "a person"
            self.nodes[nid] = node
            self._refresh_unknowns(node)
            self.write_tests(node)
            made.append(node)
            self.sys.bus.publish("memory.edited", node=nid, added=len(node.facts()), removed=0, changed=0,
                                 malformed=0, new=True)
        return made

    # ------------------------------------------------------------- learning
    def learn(self, ask: str, answer, source: str = "", by: str = "", kind: str = "people"):
        """Remember a question and its answer. Returns (node id, "new" | "updated" | "known") or None."""
        if not self.allows(kind):
            return None
        ask = " ".join(str(ask or "").split())
        answer = clean_answer(answer, self.pol.max_answer_chars)
        k = key(ask)
        if len(ask) < 3 or not answer or not k or len(ask) > 400:
            return None
        if any(key(p) == k for p in PROBES):
            return None
        now = time.strftime("%Y-%m-%d %H:%M")
        n_ask = _norm(ask)
        for node in self.nodes.values():
            same = {t.variant_of or t.ask for t in node.tests if _norm(t.ask) == n_ask}
            for t in node.facts():
                if key(t.ask) != k and t.ask not in same:
                    continue
                if t.answer == answer:
                    return node.id, "known"
                for u in node.tests:  # a correction: the newest answer wins, for every way of asking
                    if u is t or u.variant_of == t.ask:
                        u.answer = answer
                        u.learned = now
                t.source, t.by = source, by
                self._changed(node)
                self.stats["updated"] += 1
                self.sys.bus.publish("memory.fact", node=node.id, ask=ask[:200], answer=answer[:200], source=source[:200],
                                     kind="updated", by=by)
                return node.id, "updated"
        node = self._open_node()
        vs = variants(ask)
        node.tests.append(Test(vs[0], answer, source, now, "", by))
        for v in vs[1:]:
            node.tests.append(Test(v, answer, "", now, vs[0], by))
        self._changed(node)
        self.stats["facts"] += 1
        self.sys.bus.publish("memory.fact", node=node.id, ask=ask[:200], answer=answer[:200], source=source[:200],
                             kind="new", by=by)
        return node.id, "new"

    def learn_article(self, doc) -> None:
        """A retained article becomes a fact: what it is about, in the mesh's own words."""
        if not self.allows("knowledge") or not getattr(doc, "summary", ""):
            return
        from .knowledge import Hit, restate
        ask = f"Tell me about {doc.title}"
        answer, _ = restate(ask, [Hit(doc, "Introduction", doc.summary, 1.0, 1.0)])
        self.learn(ask, answer, f"{doc.source} '{doc.title}' {doc.url}", by="research", kind="knowledge")

    def forget(self, ask: str) -> str | None:
        k = key(ask)
        for node in list(self.nodes.values()):
            hit = [t for t in node.facts() if key(t.ask) == k or _norm(t.ask) == _norm(ask)]
            if not hit:
                continue
            asks = {t.ask for t in hit}
            node.tests = [t for t in node.tests if t.ask not in asks and t.variant_of not in asks]
            if not node.facts():
                self._forget_node(node, f"its last fact ('{hit[0].ask[:60]}') was forgotten")
            else:
                self._changed(node)
            self.sys.guardian.audit("operator", "memory.forgot", node.id, ask=hit[0].ask[:200])
            return node.id
        return None

    def _open_node(self) -> MemoryNode:
        open_ = [n for n in self.nodes.values() if len(n.facts()) < self.pol.facts_per_node and n.id.startswith("mem-")]
        if open_:
            return max(open_, key=lambda n: n.created)
        node = MemoryNode(f"mem-{self._next_id:04d}", self.dim)
        self._next_id += 1
        self.nodes[node.id] = node
        self.sys.bus.publish("memory.node", node=node.id)
        return node

    def _changed(self, node: MemoryNode) -> None:
        """Its tests changed: re-derive every node's [[unknown]] tests, rewrite the files that changed,
        and queue the nodes for (re)checking."""
        node.version += 1
        node.status = "learning"
        for n in self.nodes.values():
            changed = self._refresh_unknowns(n)
            if n is not node and changed:
                n.version += 1
            if n is node or changed:
                self.write_tests(n)

    def _refresh_unknowns(self, node: MemoryNode) -> bool:
        keys = {key(t.ask) for n in self.nodes.values() for t in n.tests}
        human = [u for u in node.unknowns if not u.auto]
        auto: list[Test] = []
        for p in PROBES:
            if len(auto) >= self.pol.probe_tests:
                break
            if key(p) not in keys:
                auto.append(Test(p, reason="auto: belongs to a tool, a skill or small talk"))
        others = [(zlib.crc32(t.ask.encode()), t.ask, n.id) for n in self.nodes.values() if n is not node
                  for t in n.facts()]
        for _, ask, nid in sorted(others)[: self.pol.unknown_tests]:
            auto.append(Test(ask, reason=f"auto: {nid} knows this one"))
        for t in node.facts():
            for m in near_misses(t.ask, self.pol.near_miss_tests):
                if key(m) not in keys:
                    auto.append(Test(m, reason=f"auto: a near miss of '{t.ask[:80]}' - must not be answered as if it were"))
        before = [(u.ask, u.reason) for u in node.unknowns]
        node.unknowns = human + auto
        return before != [(u.ask, u.reason) for u in node.unknowns]

    def _forget_node(self, node: MemoryNode, why: str) -> None:
        self.nodes.pop(node.id, None)
        wp = self.weights_path(node)
        if wp.exists():
            gone = self.dir / "forgotten"
            gone.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wp), str(gone / f"{node.id}-{int(time.time())}.npz"))
        tp = self.tests_path(node)
        if tp.exists():
            tp.unlink()
        self.sys.guardian.audit("memory", "memory.forgotten", node.id, why=why)
        self.sys.bus.publish("memory.forgotten", node=node.id, why=why)
        for n in self.nodes.values():
            if self._refresh_unknowns(n):
                n.version += 1
                self.write_tests(n)

    def sync_files(self, force: bool = False) -> list[str]:
        """Notice what people did to the test files: edits, deletions, new files."""
        if not self.enabled:
            return []
        now = time.time()
        if not force and now - self._last_sync < self.pol.sync_seconds:
            return []
        self._last_sync = now
        touched = []
        for node in list(self.nodes.values()):
            tp = self.tests_path(node)
            if not tp.exists():
                self._forget_node(node, "its test file was deleted")
                touched.append(node.id)
                continue
            mtime = tp.stat().st_mtime
            if mtime == node.mtime:
                continue
            raw = tp.read_bytes()
            node.mtime = mtime
            if file_hash(raw) == node.file_hash:
                continue
            change = self._read_tests(node, raw)
            if "error" in change:
                node.file_hash = file_hash(raw)  # wait for the next save
                self.sys.bus.publish("memory.unreadable", node=node.id, error=change["error"][:200])
                continue
            if not node.facts():
                self._forget_node(node, "every test was deleted from its file")
                touched.append(node.id)
                continue
            self._changed(node)
            self.sys.guardian.audit("person", "memory.edited", node.id, **change)
            self.sys.bus.publish("memory.edited", node=node.id, **change)
            touched.append(node.id)
        touched += [n.id for n in self._adopt_new_files()]
        return touched

    # ------------------------------------------------------------- answering
    def recall(self, text: str) -> Recall | None:
        if not self.enabled or not self.pol.serve:
            return None
        ready = [n for n in self.nodes.values() if n.params is not None]
        if not ready:
            return None
        x = encode_many([text], self.dim)
        best: Recall | None = None
        for n in ready:
            top, conf = n.predict(x)
            i, c = int(top[0]), float(conf[0])
            if i == 0 or c < self.pol.answer_threshold or i >= len(n.answers):
                continue
            answer = n.answers[i]
            if answer not in n.expected_answers():  # a person changed or removed it; not relearned yet
                continue
            if any(t.answer == answer and t.ask in n.failing for t in n.tests):
                continue
            if best is None or c > best.confidence:
                best = Recall(n, answer, c, n.source_of(answer))
        if best is not None:
            best.node.served += 1
            self.stats["recalled"] += 1
        return best

    # ------------------------------------------------------------- the work
    def _dataset(self, node: MemoryNode):
        """Training rows (every way of asking each test, and many things it must not answer) and the tests."""
        answers = [UNKNOWN] + list(dict.fromkeys(t.answer for t in node.tests))
        idx = {a: i for i, a in enumerate(answers)}
        rows: dict[str, tuple[int, float]] = {}
        for t in node.tests:
            for a in augment(t.ask):
                rows.setdefault(a, (idx[t.answer], 1.0))
        positives = {_norm(t) for t in rows}
        keys = {key(t.ask) for n in self.nodes.values() for t in n.tests}

        def negative(text: str, w: float) -> None:
            if text and text not in rows and _norm(text) not in positives:
                rows[text] = (0, w)
        for u in node.unknowns:
            negative(u.ask, 1.0)
            for a in augment(u.ask)[:3]:
                negative(a, 0.5)
        for p in PROBES:
            if key(p) not in keys:
                negative(p, 0.5)
        for n in self.nodes.values():
            if n is not node:
                for t in n.tests[:400]:
                    negative(t.ask, 0.5)
        for t in node.facts():
            for m in near_misses(t.ask, 6):
                if key(m) not in keys:
                    negative(m, 0.7)
        try:  # orders the router sends to tools and skills
            ex = [e.text for e in self.sys.store.examples("dispatch") if e.label.startswith(("skill:", "tool:"))
                  and not self._knowledge_route(e.label)]
            for text in ex[-60:]:
                if key(text) not in keys:
                    negative(text, 0.3)
        except Exception:
            pass
        texts = list(rows)
        labels = np.array([rows[t][0] for t in texts], dtype=np.int64)
        weights = np.array([rows[t][1] for t in texts], dtype=np.float32)
        tests = [(t.ask, idx[t.answer]) for t in node.tests] + [(u.ask, 0) for u in node.unknowns]
        return answers, texts, labels, weights, [t for t, _ in tests], np.array([e for _, e in tests], dtype=np.int64)

    def _knowledge_route(self, label: str) -> bool:
        tool = self.sys.tools.tools.get(label[5:]) if label.startswith("tool:") else None
        return tool is not None and tool.manifest.provides == "knowledge"

    def next_job(self) -> Job | None:
        dirty = [n for n in self.nodes.values() if n.dirty and n.tests]
        if dirty:
            node = min(dirty, key=lambda n: (n.params is not None, n.created))
            return self._learn_job(node)
        if not self.pol.compress:
            return None
        for node in sorted(self.nodes.values(), key=lambda n: -n.bytes):
            if node.params is None or node.status not in ("ready",):
                continue
            step = self.next_step(node)
            if step is None:
                node.status = "compressed"
                self.save_weights(node)
                self.write_tests(node)
                self.sys.bus.publish("memory.smallest", node=node.id, bytes=node.bytes, first=node.first_bytes,
                                     spec=node.spec(), total=node.total)
                continue
            return self._compress_job(node, step)
        return None

    def _base_job(self, node: MemoryNode, kind: str) -> Job:
        answers, texts, labels, weights, test_texts, expected = self._dataset(node)
        params = node.params
        if params is not None and answers != node.answers:
            params = remap_outputs(params, node.answers, answers, self.rng)
        p = self.pol
        return Job(kind, node.id, node.version, self.dim, texts, labels, weights, test_texts, expected,
                   p.answer_threshold, params, node.inputs or self.dim, node.bits, len(answers),
                   p.learn_epochs if kind == "learn" else p.compress_epochs, p.learning_rate,
                   zlib.crc32(f"{node.id}:{node.version}:{kind}:{len(node.tried)}".encode()),
                   tuple(p.start_hidden), p.max_hidden, p.max_growth, note="", margin=p.test_margin)

    def _learn_job(self, node: MemoryNode) -> Job:
        job = self._base_job(node, "learn")
        job.note = f"{len(node.facts())} facts, {len(job.test_texts)} tests"
        job.answer_texts = [UNKNOWN] + list(dict.fromkeys(t.answer for t in node.tests))
        return job

    def _compress_job(self, node: MemoryNode, step: tuple) -> Job:
        job = self._base_job(node, "compress")
        job.step = step
        job.answer_texts = node.answers
        return job

    def next_step(self, node: MemoryNode):
        """The smaller shapes worth trying next, biggest saving first; None when every one has failed."""
        p = self.pol
        A = len(node.answers)
        h = list(node.hidden)
        cur = node.bytes
        cands = []
        if node.inputs // 2 >= p.min_fold:
            cands.append(("fold", f"{node.inputs} -> {node.inputs // 2} inputs", "fold", node.inputs // 2,
                          shape_bytes(node.inputs // 2, h, A, node.bits)))
        for i, width in enumerate(h):
            for frac in (0.5, 0.25):
                k = int(width * frac)
                if k >= 1 and width - k >= p.min_neurons:
                    nh = h[:i] + [width - k] + h[i + 1:]
                    cands.append((f"narrow{i}:{frac}", f"layer {i + 1}: {width} -> {width - k} neurons", "narrow", (i, k),
                                  shape_bytes(node.inputs, nh, A, node.bits)))
        if h:
            cands.append((f"drop{len(h)}", f"without hidden layer {len(h)}" if len(h) > 1 else "no hidden layer (linear)",
                          "drop", len(h) - 1, shape_bytes(node.inputs, h[:-1], A, node.bits)))
        levels = [b for b in (16, 8) if p.min_bits <= b < node.bits]
        if levels:
            b = levels[0]
            cands.append((f"bits{b}", f"{node.bits}-bit -> {b}-bit weights", "bits", b, shape_bytes(node.inputs, h, A, b)))
        cands = [c for c in cands if c[0] not in node.tried and c[4] < cur]
        if not cands:
            return None
        c = min(cands, key=lambda c: c[4])
        return c[0], c[1], c[2], c[3]

    def finish(self, job: Job, res: dict) -> None:
        node = self.nodes.get(job.node_id)
        if node is None or node.version != job.version:
            return  # forgotten or changed while it worked: the next pass starts again
        before, before_spec = node.bytes, node.spec()
        answers = job.answer_texts
        if job.kind == "learn":
            fresh = node.params is None
            if res["ok"] or fresh or res["passed"] >= node.passed:  # best effort when it cannot hold everything
                node.params, node.inputs, node.bits, node.answers = res["params"], res["inputs"], res["bits"], answers
                node.passed, node.total = res["passed"], res["total"]
            node.failing = [job.test_texts[i] for i in res["failing"]]
            node.checked_version = node.version
            node.tried = []
            if res["note"] != "still passes every test":
                ev = "learned" if res["ok"] else "could not learn everything"
                note = (f"{len(node.facts())} facts in {node.bytes:,} bytes ({node.spec()}); {res['passed']}/{res['total']} "
                        f"tests pass" + (f"; grew: {res['note']}" if res["note"] else ""))
                node.history.append({"ts": time.time(), "event": ev, "bytes": node.bytes, "spec": node.spec(), "note": note})
            if not node.first_bytes:
                node.first_bytes = node.bytes
            node.status = "ready" if res["ok"] else "failing"
            self.save_weights(node)
            self.write_tests(node)
            if res["note"] == "still passes every test" and res["ok"]:
                return
            self.stats["learned"] += 1
            self.sys.bus.publish("memory.learned" if res["ok"] else "memory.failing", node=node.id,
                                 facts=len(node.facts()), bytes=node.bytes, spec=node.spec(), passed=res["passed"],
                                 total=res["total"], grew=res["note"], failing=node.failing[:3])
            return
        key_, desc = job.step[0], job.step[1]
        smaller = size_bytes(res["params"], res["bits"])
        if res["ok"] and smaller < before:  # every answer still matches: the smaller node stores the knowledge
            node.params, node.inputs, node.bits = res["params"], res["inputs"], res["bits"]
            node.passed, node.total, node.failing, node.tried = res["passed"], res["total"], [], []
            node.history.append({"ts": time.time(), "event": "compressed", "bytes": node.bytes, "spec": node.spec(),
                                 "note": f"{before:,} -> {node.bytes:,} bytes ({desc}); all {res['total']} tests pass"})
            self.stats["compressed"] += 1
            self.save_weights(node)
            self.write_tests(node)
            self.sys.guardian.audit("compressor", "memory.compressed", node.id, before=before, after=node.bytes,
                                    step=desc, tests=res["total"])
            self.sys.bus.publish("memory.compressed", node=node.id, before=before, after=node.bytes, step=desc,
                                 spec=node.spec(), total=res["total"], was=before_spec)
            return
        node.tried.append(key_)
        self.stats["rejected"] += 1
        failed = res["total"] - res["passed"]
        example = job.test_texts[res["failing"][0]] if res["failing"] else ""
        self.sys.bus.publish("memory.rejected", node=node.id, step=desc, failed=failed, total=res["total"],
                             example=example[:80], bytes=smaller, current=before)

    # ------------------------------------------------------------- checking
    def run_tests(self, node: MemoryNode) -> dict:
        """Run a node's test file against its current weights, now."""
        answers = node.answers
        idx = {a: i for i, a in enumerate(answers)}
        rows = [(t.ask, idx.get(t.answer, -1), t.answer) for t in node.tests] + [(u.ask, 0, "") for u in node.unknowns]
        if node.params is None or not rows:
            return {"node": node.id, "passed": 0, "total": len(rows), "failing": [r[0] for r in rows], "results": []}
        T = encode_many([r[0] for r in rows], self.dim)
        expected = np.array([max(r[1], 0) for r in rows], dtype=np.int64)
        ok, top, conf = evaluate(node.params, node.inputs, node.bits, T, expected, self.pol.answer_threshold,
                                 self.pol.test_margin)
        results = []
        for (ask, exp, want), good, t, c in zip(rows, ok, top, conf):
            good = bool(good) and exp != -1  # an answer the node has no output for yet cannot pass
            results.append({"ask": ask, "want": want or UNKNOWN, "got": answers[int(t)] if int(t) < len(answers) else "?",
                            "confidence": round(float(c), 3), "pass": good})
        failing = [r["ask"] for r in results if not r["pass"]]
        return {"node": node.id, "passed": len(results) - len(failing), "total": len(results), "failing": failing,
                "results": results}

    # ------------------------------------------------------------- status
    def status(self) -> dict:
        nodes = []
        for n in sorted(self.nodes.values(), key=lambda n: n.id):
            nodes.append({"id": n.id, "status": n.status if not n.dirty or n.params is None else "learning",
                          "facts": len(n.facts()), "answers": len(n.expected_answers()), "tests": len(n.tests) + len(n.unknowns),
                          "passed": n.passed, "total": n.total, "bytes": n.bytes, "first_bytes": n.first_bytes,
                          "spec": n.spec(), "served": n.served, "file": str(self.tests_path(n)),
                          "failing": n.failing[:5]})
        return {"enabled": self.enabled, "nodes": nodes, "facts": sum(x["facts"] for x in nodes),
                "bytes": sum(x["bytes"] for x in nodes), "first_bytes": sum(x["first_bytes"] for x in nodes),
                "pending": sum(1 for n in self.nodes.values() if n.dirty and n.tests),
                "tests_dir": str(self.tests_dir), "threshold": self.pol.answer_threshold, **dict(self.stats)}


# ===================================================================== from outside
def check_files(root: Path, out=print) -> int:
    """`python -m neuromesh memory test`: run every memory node against its test file without
    starting the mesh (nothing is changed). Returns the number of failing tests."""
    import random
    from types import SimpleNamespace

    from .config import Directives
    root = Path(root).resolve()
    d = Directives.load(root / "directives.toml")
    quiet = SimpleNamespace(publish=lambda *a, **k: None)
    system = SimpleNamespace(directives=d, state_dir=root / "state", root=root, rng=random.Random(0), bus=quiet,
                             guardian=SimpleNamespace(audit=lambda *a, **k: None),
                             store=SimpleNamespace(examples=lambda *a: []), tools=SimpleNamespace(tools={}))
    bank = MemoryBank(system, readonly=True)
    if not bank.nodes:
        out(f"no memory nodes in {bank.dir}")
        return 0
    failed = 0
    for node in sorted(bank.nodes.values(), key=lambda n: n.id):
        if not bank.tests_path(node).exists():
            out(f"{node.id}: no test file ({bank.tests_path(node)}) - the mesh will forget this node")
            continue
        r = bank.run_tests(node)
        bad = r["total"] - r["passed"]
        failed += bad
        out(f"{node.id}: {r['passed']}/{r['total']} {'pass' if not bad else 'FAIL'}  {node.bytes:,} bytes ({node.spec()})  "
            f"{bank.tests_path(node)}")
        for x in [x for x in r["results"] if not x["pass"]][:10]:
            out(f"  x {x['ask']!r}: wanted {x['want']!r}, got {x['got']!r} ({x['confidence']:.2f})")
    return failed
