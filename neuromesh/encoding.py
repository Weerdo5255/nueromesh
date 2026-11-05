"""Text -> fixed-size vector encoding shared by every node.

A signed feature-hashing encoder (word unigrams, bigrams and character trigrams).
It needs no vocabulary, so new capabilities and tools never require re-fitting it.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sa-z\d]")
NEGATORS = {"not", "no", "never", "hardly", "barely", "isn't", "wasn't", "aren't", "don't", "didn't",
            "doesn't", "won't", "can't", "nothing", "neither", "nor"}
NEGATION_SCOPE = 3  # words after a negator are marked as negated ("not lovely" -> n:lovely)


class HashingEncoder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    @staticmethod
    def features(text: str) -> list[str]:
        raw = TOKEN_RE.findall(text.lower())
        norm = ["<num>" if tok[0].isdigit() else tok for tok in raw]
        feats = []
        negated = []
        scope = 0
        for w in norm:
            if w in NEGATORS:
                negated.append(False)
                scope = NEGATION_SCOPE
            elif scope and w.isalpha():
                negated.append(True)
                scope -= 1
            else:
                negated.append(False)
                if not w.isalpha():
                    scope = 0  # punctuation ends the negation scope
            feats.append(f"{'n' if negated[-1] else 'w'}:{w}")
        feats += [f"b:{a}_{b}" for a, b in zip(norm, norm[1:])]
        for word, neg in zip(raw, negated):
            if word.isalpha() and len(word) > 3:
                padded = f"<{word}>"
                prefix = "nc" if neg else "c"
                feats += [f"{prefix}:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in self.features(text):
            h = zlib.crc32(feat.encode("utf-8"))
            vec[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec
