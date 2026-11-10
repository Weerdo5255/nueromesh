"""Memory nodes: what the mesh learns, compressed into small neural networks.

Every fact the mesh learns - a question and the answer it came to, from retained articles, a
teacher, a person or a challenge's practice answers - is compressed into a memory node: a small
MLP that maps the question, and other ways of asking it, to the answer. As it does this it writes
the node's tests into the memory's own database (state/memory.db, memdb.py - not files): every
question with the answer the node must give, and questions it must not answer. Those tests are the
contract.

The Compressor agent then keeps trying to make each node smaller: fewer input features (folding
the hashed features together), fewer neurons, fewer layers, fewer bits per weight. Each smaller
candidate is fine-tuned and run against the tests; it is accepted, and used to store the
knowledge, only if every answer still matches. People can read and edit the tests (/memory tests,
the API, or any SQLite tool - triggers note every change): change an answer and the node is
retrained to match; delete a test and it forgets that fact; delete the node's row and it forgets
the node; add tests for a new node and it learns them.

Picture memories: a fact can be a picture - its question is the image itself ("image:<sha>", taken in by what its
sight node sees in it: imaging.py's features) and its answer what the image shows ("cat") or says ("says: STOP").
They come from the picture books it learns from (imagesets.py: the first pictures of every label, and the ones its
sight node got wrong) and from what people say about an image. They are held in their own memory nodes (a picture
node never holds a question in words, and the two are never merged), tested like any fact - other pictures, and
questions in words, are what a picture node must not answer - and recalled first when it looks at an image.

Its own statements - its opinions, what its ego said, how its ego settled a disagreement - are kept
in memory too, WITHOUT tests: in own-memory nodes (own-0001, ...) that learn them from the database's
own table but are never checked, compressed, merged, or served as an answer about the world (P18).

When nothing is left to compress, it goes back through its compressed memory nodes and tries to
merge two of them, chosen at random, into one: both networks side by side in a single node (every
answer exactly as its parent gave it), fine-tuned together. The combined node is tested against
both nodes' tests; if it passes at least 90% of them it replaces both, the tests it still fails are
listed with it and never served, and it relearns until every test passes. Combined nodes are
compressed further, and can be merged again with other combined nodes.
"""
from __future__ import annotations

import dataclasses
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
from .evolution import insert_layer, prune_layer, remove_layer, widen_layer
from .memdb import MemoryDB

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
def is_picture(ask: str) -> bool:
    """A picture memory's question: the image itself."""
    return str(ask).startswith("image:")


def features(text: str) -> list[str]:
    """The shared hashed features, plus the numbers themselves (the shared encoder folds every
    number into <num>, which would make "who won in 1990" and "in 1994" the same question)."""
    return HashingEncoder.features(text) + [f"d:{n}" for n in _NUM.findall(text)]


def encode(text: str, dim: int, pictures: dict | None = None) -> np.ndarray:
    if is_picture(text):  # what its sight node sees in the image (MemoryBank.pictures), or nothing it knows
        v = (pictures or {}).get(text)
        return v if v is not None and len(v) == dim else np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    for feat in features(text):
        h = zlib.crc32(feat.encode("utf-8"))
        vec[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    vec = np.sign(vec) * np.log1p(np.abs(vec))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def encode_many(texts: list[str], dim: int, pictures: dict | None = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([encode(t, dim, pictures) for t in texts])


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
    if is_picture(out[0]):  # a picture is asked one way: itself
        return out
    sub = subject(out[0])
    if sub and _norm(sub) != _norm(out[0]):
        out.append(sub)
    return out


def augment(ask: str) -> list[str]:
    """More ways of asking, for training only."""
    a = " ".join(str(ask).split())
    if is_picture(a):
        return [a]
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
    if is_picture(ask):  # other pictures are its near misses (MemoryBank._dataset)
        return []
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
    realm: str = ""           # "fact" or "fiction" (realm.py, P24); "" learned before it told them apart
    work: str = ""            # fiction: the work it belongs to ("The Lord of the Rings", "your story")

    @property
    def auto(self) -> bool:
        return self.reason.startswith("auto:")


def _q(s) -> str:
    return json.dumps(str(s), ensure_ascii=False).translate(_TOML_SAFE)


def file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def parse_toml(raw: bytes) -> dict:
    """A test file of earlier versions ([[test]] and [[unknown]] tables) as database rows."""
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return {"error": str(exc)}
    keep = ("ask", "answer", "variant_of", "source", "learned", "by")
    return {"tests": [{k: str(e.get(k, "")) for k in keep} for e in data.get("test", []) if isinstance(e, dict)],
            "unknowns": [{"ask": str(e.get("ask", "")), "reason": str(e.get("reason", ""))} for e in data.get("unknown", [])
                         if isinstance(e, dict)]}


def own_ask(kind: str, about: str) -> str:
    """The question an own statement answers, for its own-memory node ("" when it is about nothing in particular)."""
    about = " ".join(str(about or "").split())[:120]
    if not about:
        return ""
    if kind in ("opinion", "ego:opinion"):
        return f"What do you think about {about}?"
    if kind == "ego:curious":
        return f"Are you curious about {about}?"
    if kind == "ego:dream":
        return f"Did you dream about {about}?"
    if kind == "ego:doubt":
        return f"Are you sure about {about}?"
    if kind == "arbitration":
        return f"How did you settle {about}?"
    if kind == "desire":
        return f"What do you want to know about {about}?"
    return f"What did you say about {about}?"


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


def deepen_linear(params: list) -> list:
    """A linear node as one hidden layer of ReLUs, exactly: relu(z) - relu(-z) = z."""
    W, b = params[0], params[1]
    k = W.shape[1]
    eye = np.eye(k, dtype=np.float32)
    return [np.concatenate([W, -W], axis=1).astype(np.float32), np.concatenate([b, -b]).astype(np.float32),
            np.concatenate([eye, -eye], axis=0), np.zeros(k, dtype=np.float32)]


def match_depth(params: list, depth: int, rng: np.random.Generator) -> list:
    """Give a node more hidden layers without changing what it computes."""
    p = [x.copy() for x in params]
    if depth > 0 and len(p) == 2:
        p = deepen_linear(p)
    while len(hidden_of(p)) < depth:
        p = insert_layer(p, len(hidden_of(p)), rng, noise=0.0)  # an identity after a ReLU layer
    return p


def stack(pa: list, answers_a: list, pb: list, answers_b: list, answers: list) -> list:
    """Two memory nodes (same inputs, same depth) as one: their hidden layers side by side, reading the
    same inputs. Every answer's logit is exactly its parent's; the "not something I know" logit is the
    sum of both parents' (high when both say so, beaten by an answer either parent is sure of)."""
    ia = {a: i for i, a in enumerate(answers_a)}
    ib = {a: i for i, a in enumerate(answers_b)}
    M = len(answers)
    if len(pa) == 2:  # both linear: shared inputs, each answer's column from its parent
        W = np.zeros((pa[0].shape[0], M), dtype=np.float32)
        b = np.zeros(M, dtype=np.float32)
        for m, ans in enumerate(answers):
            for p, idx in ((pa, ia), (pb, ib)):
                j = idx.get(ans)
                if j is not None:
                    W[:, m] += p[0][:, j]
                    b[m] += p[1][j]
        return [W, b]
    out = [np.concatenate([pa[0], pb[0]], axis=1).astype(np.float32), np.concatenate([pa[1], pb[1]]).astype(np.float32)]
    for layer in range(1, len(pa) // 2 - 1):
        Wa, Wb = pa[2 * layer], pb[2 * layer]
        W = np.zeros((Wa.shape[0] + Wb.shape[0], Wa.shape[1] + Wb.shape[1]), dtype=np.float32)
        W[:Wa.shape[0], :Wa.shape[1]] = Wa
        W[Wa.shape[0]:, Wa.shape[1]:] = Wb
        out += [W, np.concatenate([pa[2 * layer + 1], pb[2 * layer + 1]]).astype(np.float32)]
    Wa, Wb = pa[-2], pb[-2]
    W = np.zeros((Wa.shape[0] + Wb.shape[0], M), dtype=np.float32)
    b = np.zeros(M, dtype=np.float32)
    for m, ans in enumerate(answers):
        j = ia.get(ans)
        if j is not None:
            W[:Wa.shape[0], m] = Wa[:, j]
            b[m] += pa[-1][j]
        j = ib.get(ans)
        if j is not None:
            W[Wa.shape[0]:, m] = Wb[:, j]
            b[m] += pb[-1][j]
    return out + [W, b]


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
    parts: list = field(default_factory=list)        # the nodes it was merged from, all the way down
    merged_from: list = field(default_factory=list)  # the two nodes it was made of
    rev: int = 0                    # its tests' revision in the memory database when last read or written
    own: bool = False               # an own-memory node: its own statements, untested (see the docstring)

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

    @property
    def pictures(self) -> bool:
        """A picture node: its facts are pictures (see is_picture)."""
        return bool(self.tests) and all(is_picture(t.ask) for t in self.tests)

    def expected_answers(self) -> set:
        return {t.answer for t in self.tests}

    def source_of(self, answer: str) -> str:
        t = next((t for t in self.tests if t.answer == answer and t.source), None)
        return t.source if t else ""

    def realm_of(self, answer: str) -> tuple[str, str]:
        """As what it learned this answer: ("fact" | "fiction" | "", the work)."""
        t = next((t for t in self.tests if t.answer == answer and t.realm and not t.variant_of), None) or \
            next((t for t in self.tests if t.answer == answer and t.realm), None)
        return (t.realm, t.work) if t else ("", "")

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
    how: str = ""                   # memory nodes that answered with it, or how the ego settled their disagreement
    realm: str = ""                 # as what it was learned: fact or fiction (realm.py)
    work: str = ""


# =============================================================================== jobs
@dataclass
class Job:
    """One piece of background work on a node, run in a thread on a snapshot of its data."""
    kind: str                     # learn | compress | merge
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
    parts: tuple = ()             # merge: ((params, inputs, answers), (params, inputs, answers))
    merge: dict = field(default_factory=dict)
    pictures: dict = field(default_factory=dict)  # picture asks -> what its sight node sees in them

    def describe(self) -> str:
        if self.kind == "learn":
            return f"{self.node_id}: learning {self.note}"
        if self.kind == "merge":
            return f"merging {self.merge['a']} and {self.merge['b']} ({self.note})"
        return f"{self.node_id}: trying a smaller node ({self.step[1]})"

    def run(self) -> dict:
        X = encode_many(self.texts, self.dim, self.pictures)
        T = encode_many(self.test_texts, self.dim, self.pictures)
        rng = np.random.default_rng(self.seed)
        if self.kind == "learn":
            return self._learn(X, T, rng)
        if self.kind == "merge":
            return self._merge(X, T, rng)
        return self._compress(X, T, rng)

    def _merge(self, X, T, rng) -> dict:
        """Both nodes side by side in one (exact), then fine-tuned together on both nodes' tests."""
        (pa, ia, aa), (pb, ib, ab) = self.parts
        inputs = max(ia, ib)
        pa = unfold_params(pa, inputs) if ia < inputs else [x.copy() for x in pa]
        pb = unfold_params(pb, inputs) if ib < inputs else [x.copy() for x in pb]
        depth = max(len(hidden_of(pa)), len(hidden_of(pb)))
        params = stack(match_depth(pa, depth, rng), aa, match_depth(pb, depth, rng), ab, self.answer_texts)
        stacked = int(evaluate(params, inputs, 32, T, self.expected, self.threshold, self.margin)[0].sum())
        best = [params, stacked]

        def check(p) -> bool:
            ok = evaluate(p, inputs, 32, T, self.expected, self.threshold, self.margin)[0]
            if ok.sum() > best[1]:
                best[0], best[1] = [x.copy() for x in p], int(ok.sum())
            return bool(ok.all())
        if stacked < len(self.expected):
            _, used, _ = train(params, fold(X, inputs), self.labels, self.weights, epochs=self.epochs, lr=self.lr,
                               check=check, check_every=5)
        else:
            used = 0
        res = self._result(best[0], inputs, 32, T, note=f"stacked node passed {stacked}/{len(self.expected)}", epochs=used)
        ok = np.ones(len(self.expected), dtype=bool)
        ok[res["failing"]] = False
        pos = self.expected > 0
        res.update(stacked=stacked, pos=int(ok[pos].sum()), pos_total=int(pos.sum()), neg=int(ok[~pos].sum()),
                   neg_total=int((~pos).sum()))
        return res

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
        self.tests_dir = system.root / self.pol.tests_dir   # earlier versions' test files (imported once)
        self.db_path = system.root / self.pol.database
        self.db = MemoryDB(self.db_path, readonly=readonly)
        self.nodes: dict[str, MemoryNode] = {}
        self.rng = np.random.default_rng(system.rng.randrange(1 << 30))
        self.stats: Counter = Counter()
        self._next_id = 1
        self._next_own = 1
        self._last_sync = 0.0
        self._last_merge = time.time()  # the first merge waits a while after start
        self.tried_merges: set = set()  # pairs (as they were) that did not merge
        if self.enabled:
            self.load()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def allows(self, kind: str) -> bool:
        if kind == "pictures":  # picture memories: the picture books' own switch ([imagesets] memory)
            return self.enabled and bool(self.sys.directives.imagesets.memory)
        return self.enabled and kind in self.pol.learn_from

    def pictures(self, texts) -> dict:
        """What its sight node sees in each picture among these asks (vision.py), for encoding them."""
        vision = getattr(self.sys, "vision", None)
        if vision is None:
            return {}
        return {t: vision.vector(t) for t in dict.fromkeys(texts) if is_picture(t)}

    def fact_nodes(self) -> list[MemoryNode]:
        """The nodes holding facts (each checked by its tests): not its own statements."""
        return [n for n in self.nodes.values() if not n.own]

    def own_nodes(self) -> list[MemoryNode]:
        return [n for n in self.nodes.values() if n.own]

    # -------------------------------------------------------------- its own statements (untested)
    def remember_own(self, kind: str, text: str, about: str = "", mood: str = "", source: str = "",
                     realm: str = "") -> dict | None:
        """Keep one of its own statements - an opinion, what its ego said, how its ego settled a disagreement -
        in memory, without tests. One about something is learned by an own-memory node (the newest statement
        for the same question wins); every statement stays in the memory database's own table."""
        if not self.enabled or not self.pol.own or self.readonly:
            return None
        text = " ".join(str(text or "").split())
        if not text:
            return None
        ask = own_ask(kind, about)
        node = self._own_node_for(ask) if ask else None
        if not realm and any(w in kind for w in ("dream", "imagin")):  # what it dreamt or imagined: fiction (P13, P24)
            realm = "fiction"
        rid = self.db.add_own(kind, text, about=about, ask=ask, mood=mood, source=source, node=node.id if node else "",
                              realm=realm)
        if node is not None:
            self._load_own(node)
            node.version += 1
            node.status = "learning"
        self.stats["own_kept"] += 1
        self.sys.bus.publish("memory.own", kind=kind, about=about[:80], node=node.id if node else "", text=text[:160])
        return {"id": rid, "node": node.id if node else "", "ask": ask}

    def _own_node_for(self, ask: str) -> MemoryNode:
        k = _norm(ask)
        for n in self.own_nodes():
            if any(_norm(t.ask) == k for t in n.tests):
                return n
        open_ = [n for n in self.own_nodes() if len(n.tests) < self.pol.own_per_node]
        if open_:
            return max(open_, key=lambda n: n.created)
        node = MemoryNode(f"own-{self._next_own:04d}", self.dim, own=True)
        self._next_own += 1
        self.nodes[node.id] = node
        self.sys.bus.publish("memory.node", node=node.id, own=True)
        return node

    def _load_own(self, node: MemoryNode) -> None:
        """An own-memory node's statements, from the database (the newest for each question)."""
        node.tests = [Test(r["ask"], r["text"], f"its own {r['kind'].replace('ego:', 'ego, ')}",
                           time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"] or 0)), "", "itself")
                      for r in self.db.own_asks(node.id)]
        node.unknowns = []

    def recall_own(self, text: str, threshold: float | None = None) -> Recall | None:
        """What it said or thought before, from its own-memory nodes (never an answer about the world)."""
        if not self.enabled or not self.pol.own:
            return None
        ready = [n for n in self.own_nodes() if n.params is not None]
        if not ready:
            return None
        x = encode_many([text], self.dim)
        best = None
        for n in ready:
            top, conf = n.predict(x)
            i, c = int(top[0]), float(conf[0])
            if i == 0 or i >= len(n.answers) or c < (self.pol.answer_threshold if threshold is None else threshold):
                continue
            if n.answers[i] not in n.expected_answers():
                continue
            if best is None or c > best.confidence:
                best = Recall(n, n.answers[i], c, n.source_of(n.answers[i]))
        return best

    def own_statements(self, about: str | None = None, kind=None, limit: int = 20, since: float = 0.0,
                       until: float | None = None) -> list[dict]:
        return self.db.own(kind=kind, about=about, limit=limit, since=since, until=until) if self.enabled else []

    # -------------------------------------------------------------- persistence
    def tests_ref(self, node: MemoryNode) -> str:
        """Where its tests are, for people."""
        return f"{self.db_path.name} ({'own statements, untested' if node.own else 'tests'} of {node.id})"

    def weights_path(self, node: MemoryNode) -> Path:
        return self.dir / f"{node.id}.npz"

    def load(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.dir.glob("*.npz")):
            try:
                node = self._load_weights(path)
            except Exception:
                continue
            self.nodes[node.id] = node
        if not self.readonly:
            self._import_files()
        for node in list(self.nodes.values()):
            if node.own:
                self._load_own(node)
                continue
            data = self.db.read(node.id)
            if data is None and self.readonly:
                legacy = self.tests_dir / f"{node.id}.toml"
                data = parse_toml(legacy.read_bytes()) if legacy.exists() else None
                if data is None or "error" in data:
                    continue
            if data is None:  # its tests were deleted while the mesh was off: forget it
                self._forget_node(node, "its tests were deleted from the memory database")
                continue
            self._read_rows(node, data)
            if data.get("rev", node.rev) != node.rev and not self.readonly:  # changed while the mesh was off
                node.version += 1
                self._refresh_unknowns(node)
                self.write_tests(node)
        if not self.readonly:
            self._adopt_new()
            for nid in self.db.own_nodes():  # statements whose node's weights were lost: learn them again
                if nid not in self.nodes:
                    node = MemoryNode(nid, self.dim, own=True)
                    self.nodes[nid] = node
                    self._load_own(node)
        nums = [int(m.group(1)) for n in self.nodes for m in [re.match(r"mem-(\d+)$", n)] if m]
        self._next_id = max(nums, default=0) + 1
        own = [int(m.group(1)) for n in self.nodes for m in [re.match(r"own-(\d+)$", n)] if m]
        self._next_own = max(own, default=0) + 1

    def _import_files(self) -> None:
        """Test files from earlier versions (memory_tests/<node>.toml) go into the database once; the
        folder is then moved aside (memory_tests.imported)."""
        d = self.tests_dir
        if not d.is_dir():
            return
        files = sorted(d.glob("*.toml"))
        imported = 0
        for path in files:
            nid = path.stem
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", nid) or self.db.rev(nid) is not None:
                continue
            raw = path.read_bytes()
            data = parse_toml(raw)
            if "error" in data:
                continue
            rev = self.db.write(nid, data["tests"], data["unknowns"], {"imported_from": str(path)})
            node = self.nodes.get(nid)
            if node is not None and node.file_hash == file_hash(raw):
                node.rev = rev  # unchanged since its weights were saved: nothing to relearn
            imported += 1
        if files:
            dest = d.with_name(d.name + ".imported")
            if dest.exists():
                dest = d.with_name(f"{d.name}.imported-{int(time.time())}")
            shutil.move(str(d), str(dest))
            self.sys.bus.publish("memory.imported", files=len(files), imported=imported, moved_to=str(dest))

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
                          file_hash=meta.get("file_hash", ""), served=meta.get("served", 0),
                          parts=meta.get("parts", []), merged_from=meta.get("merged_from", []),
                          rev=int(meta.get("rev", 0)), own=bool(meta.get("own", False)))
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
                "failing": node.failing, "parts": node.parts, "merged_from": node.merged_from, "rev": node.rev,
                "own": node.own}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{node.id}.tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(self.weights_path(node))

    def summary(self, node: MemoryNode) -> dict:
        if node.params is None:
            size = "not learned yet"
        else:
            size = f"{node.bytes:,} bytes: {node.spec()}"
            if node.first_bytes and node.first_bytes != node.bytes:
                size += f" (first {node.first_bytes:,} bytes)"
        passing = (f"{node.passed}/{node.total}" + (f"; failing: {', '.join(node.failing[:5])}" if node.failing else "")
                   if node.total and (not node.dirty or node.status == "failing") else "not checked yet")
        return {"node": node.id, "facts": len(node.facts()), "answers": len(node.expected_answers()), "size": size,
                "passing": passing, "history": [self._history_line(h) for h in node.history[-12:]]}

    def write_tests(self, node: MemoryNode) -> None:
        """The node's tests, in the memory database (its own statements are not tests: they are not written)."""
        if node.own or self.readonly:
            return
        tests = [{"ask": t.ask, "answer": t.answer, "variant_of": t.variant_of, "source": t.source, "learned": t.learned,
                  "by": t.by, "realm": t.realm, "work": t.work} for t in node.tests]
        unknowns = [{"ask": u.ask, "reason": u.reason} for u in node.unknowns]
        node.rev = self.db.write(node.id, tests, unknowns, self.summary(node))

    def render_tests(self, node: MemoryNode) -> str:
        """Its tests as text, for people (the same layout earlier versions' test files had)."""
        info = self.summary(node)
        lines = [f"# Memory node {node.id}: " + ("its own statements, kept without tests (never served as facts)."
                                                 if node.own else "the tests it must pass (in the memory database)."),
                 f"node = {_q(node.id)}", f"facts = {info['facts']}", f"answers = {info['answers']}",
                 f"size = {_q(info['size'])}", f"passing = {_q('untested' if node.own else info['passing'])}"]
        for t in node.tests:
            lines += ["", "[[test]]" if not node.own else "[[statement]]", f"ask = {_q(t.ask)}", f"answer = {_q(t.answer)}"]
            for k in ("variant_of", "source", "learned", "by", "realm", "work"):
                if getattr(t, k):
                    lines.append(f"{k} = {_q(getattr(t, k))}")
        for u in node.unknowns:
            lines += ["", "[[unknown]]", f"ask = {_q(u.ask)}", f"reason = {_q(u.reason)}"]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _history_line(h: dict) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(h.get("ts", 0)))
        return f"{when} {h.get('event', '')}: {h.get('note', '')}".strip()

    def _read_rows(self, node: MemoryNode, data: dict) -> dict:
        """Load its tests from the database's rows. Returns what changed."""
        old = {t.ask: t.answer for t in node.tests}
        tests, malformed = [], 0
        for e in data.get("tests", []):
            ask, answer = e.get("ask"), e.get("answer")
            if not isinstance(ask, str) or not isinstance(answer, str) or not ask.strip() or not answer.strip():
                malformed += 1
                continue
            realm = str(e.get("realm") or "").strip().lower()
            tests.append(Test(" ".join(ask.split()), answer.strip(), str(e.get("source") or ""), str(e.get("learned") or ""),
                              " ".join(str(e.get("variant_of") or "").split()), str(e.get("by") or "") or "a person",
                              "", realm if realm in ("fact", "fiction") else "", str(e.get("work") or "")[:120]))
        primaries = {t.ask: t for t in tests if not t.variant_of}
        kept = []
        for t in tests:
            if t.variant_of:
                p = primaries.get(t.variant_of)
                if p is None:  # its question was deleted: so is this way of asking it
                    continue
                t.answer = p.answer
                t.realm, t.work = p.realm, p.work
            kept.append(t)
        node.tests = kept
        node.unknowns = [Test(" ".join(str(e["ask"]).split()), reason=str(e.get("reason") or "") or "added by a person")
                         for e in data.get("unknowns", []) if isinstance(e.get("ask"), str) and e["ask"].strip()]
        node.rev = int(data.get("rev", node.rev))
        new = {t.ask: t.answer for t in node.tests}
        return {"added": len(set(new) - set(old)), "removed": len(set(old) - set(new)),
                "changed": sum(1 for a in set(new) & set(old) if new[a] != old[a]), "malformed": malformed}

    def _adopt_new(self) -> list[MemoryNode]:
        """Tests a person added for a node the mesh does not have (or one whose node was lost): learn them."""
        made = []
        for nid in sorted(self.db.revs()):
            if nid in self.nodes or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", nid) or nid.startswith("own-"):
                continue
            data = self.db.read(nid)
            node = MemoryNode(nid, self.dim)
            self._read_rows(node, data or {})
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

    def edit_tests(self, node_id: str, tests: list[dict], unknowns: list[dict] | None = None, by: str = "a person") -> dict:
        """A person rewrites a node's tests (the API, /memory): the database takes them, and the mesh relearns."""
        node = self.nodes.get(node_id)
        if node is not None and node.own:
            return {"ok": False, "error": "own-memory nodes hold its own statements, not tests"}
        if unknowns is None:
            unknowns = [{"ask": u.ask, "reason": u.reason} for u in (node.unknowns if node else []) if not u.auto]
        clean = [{"ask": str(t.get("ask", "")), "answer": str(t.get("answer", "")), "variant_of": str(t.get("variant_of", "")),
                  "source": str(t.get("source", "")), "learned": str(t.get("learned", "")), "by": str(t.get("by", "") or by)}
                 for t in tests if isinstance(t, dict)]
        self.db.write(node_id, clean, [{"ask": str(u.get("ask", "")), "reason": str(u.get("reason", ""))}
                                       for u in unknowns if isinstance(u, dict)], {"edited_by": by})
        touched = self.sync_tests(force=True)
        return {"ok": True, "node": node_id, "touched": touched}

    # ------------------------------------------------------------- learning
    def learn(self, ask: str, answer, source: str = "", by: str = "", kind: str = "people", reviewed: bool = False,
              realm: str | None = None, work: str = ""):
        """Remember a question and its answer. Returns (node id, "new" | "updated" | "known") or None.
        Nothing is stored without a director's review (P14): an order's facts are reviewed with the
        order (reviewed=True); anything else is reviewed here. Every fact is marked fact or fiction (realm:
        given, or judged here from the question and answer; realm.py, P24)."""
        if not self.allows(kind):
            return None
        directors = getattr(self.sys, "directors", None)
        if not reviewed and directors is not None and not directors.review_memory(ask, answer, source, by, kind)[0]:
            return None
        ask = " ".join(str(ask or "").split())
        answer = clean_answer(answer, self.pol.max_answer_chars)
        k = key(ask)
        if len(ask) < 3 or not answer or not k or len(ask) > 400:
            return None
        if any(key(p) == k for p in PROBES):
            return None
        if realm not in ("fact", "fiction") and not is_picture(ask):
            sense = getattr(self.sys, "realm", None)
            j = sense.judge(f"{ask} {answer}"[:600]) if sense is not None else None
            realm, work = (j.realm, j.work or work) if j is not None and j.confidence > 0 else ("", work)
        now = time.strftime("%Y-%m-%d %H:%M")
        n_ask = _norm(ask)
        for node in self.fact_nodes():
            same = {t.variant_of or t.ask for t in node.tests if _norm(t.ask) == n_ask}
            for t in node.facts():
                if key(t.ask) != k and t.ask not in same:
                    continue
                if t.answer == answer and (t.realm == realm or not realm):
                    return node.id, "known"
                for u in node.tests:  # a correction: the newest answer wins, for every way of asking
                    if u is t or u.variant_of == t.ask:
                        u.answer = answer
                        u.learned = now
                        u.realm, u.work = realm, work
                t.source, t.by = source, by
                self._changed(node)
                self.stats["updated"] += 1
                self.sys.bus.publish("memory.fact", node=node.id, ask=ask[:200], answer=answer[:200], source=source[:200],
                                     kind="updated", by=by)
                return node.id, "updated"
        node = self._open_node(is_picture(ask))
        vs = variants(ask)
        node.tests.append(Test(vs[0], answer, source, now, "", by, "", realm, work))
        for v in vs[1:]:
            node.tests.append(Test(v, answer, "", now, vs[0], by, "", realm, work))
        self._changed(node)
        self.stats["facts"] += 1
        self.sys.bus.publish("memory.fact", node=node.id, ask=ask[:200], answer=answer[:200], source=source[:200],
                             kind="new", by=by, realm=realm, work=work)
        return node.id, "new"

    def learn_article(self, doc) -> None:
        """A retained article becomes a fact: what it is about, in the mesh's own words."""
        if not self.allows("knowledge") or not getattr(doc, "summary", ""):
            return
        from .knowledge import Hit, restate
        ask = f"Tell me about {doc.title}"
        answer, _ = restate(ask, [Hit(doc, "Introduction", doc.summary, 1.0, 1.0)])
        self.learn(ask, answer, f"{doc.source} '{doc.title}' {doc.url}", by="research", kind="knowledge")

    def must_not_answer(self, node_id: str, ask: str, reason: str, by: str = "operator") -> bool:
        """A person says a node's answer to a question was wrong: a fact it held for exactly that question
        is forgotten, and the question becomes one it must not answer (an [[unknown]] test in its file)."""
        node = self.nodes.get(node_id)
        ask = " ".join(str(ask).split())
        if node is None or not ask:
            return False
        if any(key(t.ask) == key(ask) for t in node.facts()):
            self.forget(ask)
            node = self.nodes.get(node_id)
            if node is None:
                return True
        if any(_norm(u.ask) == _norm(ask) for u in node.unknowns):
            return False
        node.unknowns.append(Test(ask, reason=reason[:200] or "a person said its answer was wrong"))
        self._changed(node)
        self.sys.guardian.audit(by, "memory.must_not_answer", node.id, ask=ask[:200], reason=reason[:200])
        return True

    def forget(self, ask: str) -> str | None:
        k = key(ask)
        for node in self.fact_nodes():
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

    def _open_node(self, picture: bool = False) -> MemoryNode:
        open_ = [n for n in self.nodes.values() if len(n.facts()) < self.pol.facts_per_node and n.id.startswith("mem-")
                 and (n.pictures == picture or not n.tests)]
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

    def _refresh_unknowns(self, node: MemoryNode, exclude: tuple = ()) -> bool:
        if node.own:  # its own statements have no tests
            return False
        keys = {key(t.ask) for n in self.fact_nodes() for t in n.tests}
        human = [u for u in node.unknowns if not u.auto]
        auto: list[Test] = []
        for p in PROBES:
            if len(auto) >= self.pol.probe_tests:
                break
            if key(p) not in keys:
                auto.append(Test(p, reason="auto: belongs to a tool, a skill or small talk"))
        mine = {key(t.ask) for t in node.tests}  # a question it answers itself is never one it must not answer
        others = [(zlib.crc32(t.ask.encode()), t.ask, n.id) for n in self.fact_nodes()
                  if n is not node and n.id not in exclude for t in n.facts() if key(t.ask) not in mine]
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
        if not node.own:
            self.db.delete(node.id)
        self.sys.guardian.audit("memory", "memory.forgotten", node.id, why=why)
        self.sys.bus.publish("memory.forgotten", node=node.id, why=why)
        for n in self.nodes.values():
            if self._refresh_unknowns(n):
                n.version += 1
                self.write_tests(n)

    def sync_tests(self, force: bool = False) -> list[str]:
        """Notice what people did to the tests in the memory database: edits, deletions, new nodes."""
        if not self.enabled:
            return []
        now = time.time()
        if not force and now - self._last_sync < self.pol.sync_seconds:
            return []
        self._last_sync = now
        touched = []
        revs = self.db.revs()
        for node in list(self.nodes.values()):
            if node.own:
                continue
            rev = revs.get(node.id)
            if rev is None:
                self._forget_node(node, "its tests were deleted from the memory database")
                touched.append(node.id)
                continue
            if rev == node.rev:
                continue
            data = self.db.read(node.id) or {}
            change = self._read_rows(node, data)
            if not node.facts():
                self._forget_node(node, "every test was deleted from the memory database")
                touched.append(node.id)
                continue
            self._changed(node)
            self.sys.guardian.audit("person", "memory.edited", node.id, **change)
            self.sys.bus.publish("memory.edited", node=node.id, **change)
            touched.append(node.id)
        touched += [n.id for n in self._adopt_new()]
        return touched

    sync_files = sync_tests  # (earlier name)

    # ------------------------------------------------------------- answering
    def known(self, ask: str):
        """The fact it holds for a question, however it was asked, or None."""
        k = key(" ".join(str(ask or "").split()))
        if not k:
            return None
        for node in self.fact_nodes():
            for t in node.facts():
                if key(t.ask) == k:
                    return t
        return None

    def recall(self, text: str, quiet: bool = False, threshold: float | None = None) -> Recall | None:
        """The best answer a memory node holds for a question (quiet: a director's second opinion, which
        may look at less certain answers (threshold) and is not counted as serving)."""
        if not self.enabled or not self.pol.serve:
            return None
        picture = is_picture(text)  # a picture is recalled by picture nodes, a question in words by the others
        ready = [n for n in self.fact_nodes() if n.params is not None and n.pictures == picture]
        if not ready:
            return None
        x = encode_many([text], self.dim, self.pictures([text]))
        pool = getattr(self.sys, "pool", None)  # every memory node at once (ensemble.py)
        preds = pool.map("memory", lambda n: n.predict(x), ready) if pool is not None else [n.predict(x) for n in ready]
        found: list[Recall] = []
        for n, (top, conf) in zip(ready, preds):
            i, c = int(top[0]), float(conf[0])
            if hasattr(self.sys, "firings"):  # what each memory node fired on (anatomy.py)
                from .anatomy import record
                record(self.sys, n.id, "memory", text, n.answers[i] if i < len(n.answers) else str(i), c, used=False)
            if i == 0 or c < (self.pol.answer_threshold if threshold is None else threshold) or i >= len(n.answers):
                continue
            answer = n.answers[i]
            if answer not in n.expected_answers():  # a person changed or removed it; not relearned yet
                continue
            if any(t.answer == answer and t.ask in n.failing for t in n.tests):
                continue
            found.append(Recall(n, answer, c, n.source_of(answer), "", *n.realm_of(answer)))
        found.sort(key=lambda r: -r.confidence)
        best = found[0] if found else None
        if best is not None and len(found) > 1:
            from .ensemble import similar
            agree = [r for r in found if similar(r.answer, best.answer)]
            if len(agree) == len(found):
                best.how = f"{len(found)} memory nodes gave it"
            elif not quiet and getattr(self.sys, "ego", None) is not None:  # they do not match: the ego decides
                if pool is not None:
                    pool.disagreed += 1
                votes = [{"node": r.node, "label": r.answer, "conf": r.confidence,
                          "acc": r.node.passed / r.node.total if r.node.total else 0.5, "champion": r is best} for r in found]
                v = self.sys.ego.arbitrate("memory", text, votes, same=similar)
                best = next((r for r in found if r.node.id == v["node"] and r.answer == v["label"]), best)
                best.how = (f"memory nodes {', '.join(r.node.id for r in found)} disagreed; the ego arbitrated "
                            f"({v['strategy']}: {v['why']})")
        if best is not None and getattr(self.sys, "firings", None):
            for f in reversed(self.sys.firings):
                if f["node"] == best.node.id and f["text"] == str(text)[:300]:
                    f["used"] = True
                    break
        if best is not None and not quiet:
            best.node.served += 1
            self.stats["recalled"] += 1
        return best

    # ------------------------------------------------------------- the work
    def _dataset(self, node: MemoryNode, exclude: tuple = ()):
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
            if n is not node and n.id not in exclude:
                for t in n.tests[:400]:
                    negative(t.ask, 0.5)
        for t in node.facts():
            for m in near_misses(t.ask, 6):
                if key(m) not in keys:
                    negative(m, 0.7)
        books = getattr(self.sys, "imagesets", None)
        if node.pictures and books is not None:  # other pictures: what a picture node must not take for its own
            held = {t.ask for n in self.nodes.values() for t in n.tests}
            for other in books.negatives(max(40, 2 * len(node.tests)), held):
                negative(other, 0.7)
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
        if self.pol.merge and time.time() - self._last_merge >= self.pol.merge_every_seconds:
            pairs = self.merge_candidates()
            if pairs:
                self._last_merge = time.time()
                a, b = pairs[int(self.rng.integers(len(pairs)))]  # chosen at random
                return self.merge_job(a, b)
        return None

    # ------------------------------------------------------------- merging
    @staticmethod
    def _pair(a: MemoryNode, b: MemoryNode) -> str:
        return "|".join(sorted(f"{n.id}@{n.version}:{n.bytes}" for n in (a, b)))

    def merge_candidates(self) -> list[tuple]:
        """Pairs of compressed memory nodes that pass all their tests and have not been tried as they are."""
        nodes = sorted((n for n in self.fact_nodes() if n.params is not None and not n.dirty and n.status == "compressed"
                        and n.total and n.passed == n.total), key=lambda n: n.id)
        return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:] if a.pictures == b.pictures
                and len(a.facts()) + len(b.facts()) <= self.pol.merge_max_facts and self._pair(a, b) not in self.tried_merges]

    def merge_job(self, a: MemoryNode, b: MemoryNode) -> Job:
        merged = MemoryNode(f"{a.id}+{b.id}", self.dim)
        merged.tests = [dataclasses.replace(t) for t in a.tests + b.tests]
        seen: set = set()
        merged.unknowns = [dataclasses.replace(u) for u in a.unknowns + b.unknowns
                           if not u.auto and not (u.ask in seen or seen.add(u.ask))]
        self._refresh_unknowns(merged, exclude=(a.id, b.id))
        answers, texts, labels, weights, test_texts, expected = self._dataset(merged, exclude=(a.id, b.id))
        p = self.pol
        job = Job("merge", merged.id, 0, self.dim, texts, labels, weights, test_texts, expected, p.answer_threshold,
                  None, max(a.inputs, b.inputs), 32, len(answers), p.merge_epochs, p.learning_rate,
                  zlib.crc32(f"{self._pair(a, b)}:merge".encode()), margin=p.test_margin,
                  note=f"{len(a.facts()) + len(b.facts())} facts, {len(test_texts)} tests")
        job.answer_texts = answers
        job.pictures = self.pictures(texts + test_texts)
        job.parts = ((a.params, a.inputs, a.answers), (b.params, b.inputs, b.answers))
        job.merge = {"a": a.id, "b": b.id, "pair": self._pair(a, b), "tests": merged.tests, "unknowns": merged.unknowns}
        return job

    def _finish_merge(self, job: Job, res: dict) -> None:
        m = job.merge
        a, b = self.nodes.get(m["a"]), self.nodes.get(m["b"])
        if a is None or b is None or self._pair(a, b) != m["pair"]:
            return  # one of them changed while it worked
        self.tried_merges.add(m["pair"])
        need = self.pol.merge_pass_rate
        pos_rate = res["pos"] / res["pos_total"] if res["pos_total"] else 1.0
        neg_rate = res["neg"] / res["neg_total"] if res["neg_total"] else 1.0
        before = a.bytes + b.bytes
        detail = dict(a=a.id, b=b.id, pos=res["pos"], pos_total=res["pos_total"], neg=res["neg"], neg_total=res["neg_total"],
                      need=need, stacked=res["stacked"], total=res["total"])
        if pos_rate < need or neg_rate < need:
            self.stats["merges_refused"] += 1
            self.sys.bus.publish("memory.merge_refused", **detail)
            return
        node = MemoryNode(f"mem-{self._next_id:04d}", self.dim)
        self._next_id += 1
        node.params, node.inputs, node.bits, node.answers = res["params"], res["inputs"], res["bits"], job.answer_texts
        node.tests, node.unknowns = m["tests"], m["unknowns"]
        node.parts = (a.parts or [a.id]) + (b.parts or [b.id])
        node.merged_from = [a.id, b.id]
        node.first_bytes = (a.first_bytes or a.bytes) + (b.first_bytes or b.bytes)
        node.passed, node.total = res["passed"], res["total"]
        node.failing = [job.test_texts[i] for i in res["failing"]]
        node.served = a.served + b.served
        fails = len(node.failing)
        node.status = "ready" if not fails else "failing"
        node.checked_version = node.version if not fails else node.version - 1  # relearn until every test passes
        node.history = [{"ts": time.time(), "event": "merged", "bytes": node.bytes, "spec": node.spec(),
                         "note": f"{a.id} ({len(a.facts())} facts, {a.bytes:,} bytes) + {b.id} ({len(b.facts())} facts, "
                                 f"{b.bytes:,} bytes) -> {node.bytes:,} bytes; {res['pos']}/{res['pos_total']} answers and "
                                 f"{res['neg']}/{res['neg_total']} unknowns pass"
                                 + (f"; relearning the {fails} that fail" if fails else "")}]
        for parent in (a, b):
            self._retire(parent, node.id)
        self.nodes[node.id] = node
        for n in self.nodes.values():
            if self._refresh_unknowns(n):
                n.version += 1
            self.write_tests(n)
        self.save_weights(node)
        self.stats["merged"] += 1
        self.sys.guardian.audit("compressor", "memory.merged", node.id, parents=[a.id, b.id], parts=node.parts,
                                before=before, after=node.bytes, **{k: detail[k] for k in ("pos", "pos_total", "neg", "neg_total")})
        self.sys.bus.publish("memory.merged", node=node.id, facts=len(node.facts()), bytes=node.bytes, before=before,
                             parts=len(node.parts), failing=fails, **detail)

    def _retire(self, node: MemoryNode, into: str) -> None:
        """A node merged into another: its weights are kept aside, its tests live on in the merged node."""
        self.nodes.pop(node.id, None)
        wp = self.weights_path(node)
        if wp.exists():
            gone = self.dir / "merged"
            gone.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wp), str(gone / f"{node.id}-into-{into}.npz"))
        self.db.delete(node.id)

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
                   tuple(p.start_hidden), p.max_hidden, p.max_growth, note="", margin=p.test_margin,
                   pictures=self.pictures(texts + test_texts))

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
        if job.kind == "merge":
            return self._finish_merge(job, res)
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
            if node.own:  # its own statements: learned as well as it can, never checked
                node.status, node.failing = "untested", []
            self.save_weights(node)
            self.write_tests(node)
            if res["note"] == "still passes every test" and res["ok"]:
                return
            self.stats["learned"] += 1
            if node.own:
                self.sys.bus.publish("memory.own_learned", node=node.id, statements=len(node.tests), bytes=node.bytes)
                return
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
        T = encode_many([r[0] for r in rows], self.dim, self.pictures([r[0] for r in rows]))
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
        for n in sorted(self.fact_nodes(), key=lambda n: n.id):
            nodes.append({"id": n.id, "status": n.status if not n.dirty or n.params is None else "learning",
                          "facts": len(n.facts()), "answers": len(n.expected_answers()), "tests": len(n.tests) + len(n.unknowns),
                          "passed": n.passed, "total": n.total, "bytes": n.bytes, "first_bytes": n.first_bytes,
                          "spec": n.spec(), "served": n.served, "file": self.tests_ref(n),
                          "failing": n.failing[:5], "parts": n.parts or [n.id], "merged_from": n.merged_from})
        own = [{"id": n.id, "status": n.status if not n.dirty or n.params is None else "learning", "statements": len(n.tests),
                "bytes": n.bytes, "spec": n.spec()} for n in sorted(self.own_nodes(), key=lambda n: n.id)]
        counts = self.db.counts()
        return {"enabled": self.enabled, "nodes": nodes, "facts": sum(x["facts"] for x in nodes),
                "bytes": sum(x["bytes"] for x in nodes), "first_bytes": sum(x["first_bytes"] for x in nodes),
                "pending": sum(1 for n in self.nodes.values() if n.dirty and n.tests),
                "own": {"enabled": bool(self.pol.own), "nodes": own, "statements": counts["own"]},
                "database": str(self.db_path), "tests_stored": counts["tests"] + counts["unknowns"],
                "tests_dir": str(self.db_path), "threshold": self.pol.answer_threshold,
                "merge": self.pol.merge, "merge_pass_rate": self.pol.merge_pass_rate, **dict(self.stats)}


# ===================================================================== from outside
def check_files(root: Path, out=print) -> int:
    """`python -m neuromesh memory test`: run every memory node against its tests (in the memory database)
    without starting the mesh (nothing is changed). Returns the number of failing tests."""
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
    for node in sorted(bank.fact_nodes(), key=lambda n: n.id):
        if not node.tests:
            out(f"{node.id}: no tests in {bank.db_path} - the mesh will forget this node")
            continue
        r = bank.run_tests(node)
        bad = r["total"] - r["passed"]
        failed += bad
        out(f"{node.id}: {r['passed']}/{r['total']} {'pass' if not bad else 'FAIL'}  {node.bytes:,} bytes ({node.spec()})  "
            f"{bank.tests_ref(node)}")
        for x in [x for x in r["results"] if not x["pass"]][:10]:
            out(f"  x {x['ask']!r}: wanted {x['want']!r}, got {x['got']!r} ({x['confidence']:.2f})")
    return failed
