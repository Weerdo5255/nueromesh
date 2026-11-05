"""External teachers. Nodes never learn from themselves; they learn from:

* HumanTeacher     - corrections and examples from people (highest trust)
* LLMTeacher       - an external LLM (Anthropic Messages API or any OpenAI-compatible
                     endpoint such as Ollama or vLLM) that labels data, writes synthetic
                     training examples, and answers orders no node can handle yet
* HeuristicTeacher - an offline rule-based stand-in so the system runs with no API key

The TeacherPool asks them in the order set by the directives.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass

from .config import LLMConfig
from .tasks import DISPATCH, CapabilitySpec

log = logging.getLogger("neuromesh.teachers")


@dataclass
class Lesson:
    label: str | None
    confidence: float
    source: str
    rationale: str = ""


class Teacher:
    name = "teacher"

    def available(self) -> bool:
        return True

    async def label(self, spec: CapabilitySpec, text: str) -> Lesson | None:
        return None

    async def generate(self, spec: CapabilitySpec, label: str, n: int) -> list[str]:
        return []

    async def answer(self, text: str) -> str | None:
        return None


# --------------------------------------------------------------------- human
class HumanTeacher(Teacher):
    name = "human"

    def __init__(self, max_questions: int = 200):
        self.answers: dict[tuple[str, str], str] = {}
        self.questions: deque[dict] = deque(maxlen=max_questions)
        self._qid = 0

    def ask(self, capability: str, text: str, reason: str) -> None:
        if (capability, text) in self.answers:
            return
        if any(q["capability"] == capability and q["text"] == text for q in self.questions):
            return
        self._qid += 1
        self.questions.append({"id": f"q{self._qid}", "capability": capability, "text": text,
                               "reason": reason, "asked_at": time.time()})

    def provide(self, capability: str, text: str, label: str) -> None:
        self.answers[(capability, text)] = label
        for q in list(self.questions):
            if q["capability"] == capability and q["text"] == text:
                self.questions.remove(q)

    def question(self, qid: str) -> dict | None:
        return next((q for q in self.questions if q["id"] == qid), None)

    async def label(self, spec, text):
        answer = self.answers.get((spec.name, text))
        return Lesson(answer, 1.0, "human") if answer else None


# ----------------------------------------------------------------------- llm
def _extract_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _clean_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_:\-]+", "_", label.strip().lower()).strip("_")


class LLMTeacher(Teacher):
    name = "llm"

    def __init__(self, cfg: LLMConfig, guardian):
        self.cfg = cfg
        self.guardian = guardian
        self.calls = 0
        self.errors = 0

    def _key(self) -> str:
        return os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""

    def available(self) -> bool:
        if self.cfg.provider == "anthropic":
            return bool(self._key())
        if self.cfg.provider == "openai":
            return bool(self.cfg.base_url or self._key())
        return False

    def _post(self, system: str, user: str) -> tuple[str, int]:
        cfg = self.cfg
        if cfg.provider == "anthropic":
            url = (cfg.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
            headers = {"x-api-key": self._key(), "anthropic-version": cfg.anthropic_version,
                       "content-type": "application/json"}
            body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}]}
        else:
            url = (cfg.base_url or "https://api.openai.com").rstrip("/") + "/v1/chat/completions"
            headers = {"content-type": "application/json"}
            if self._key():
                headers["authorization"] = f"Bearer {self._key()}"
            body = {"model": cfg.model, "max_tokens": cfg.max_tokens,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if cfg.provider == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            usage = data.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        else:
            text = data["choices"][0]["message"]["content"] or ""
            tokens = data.get("usage", {}).get("total_tokens", 0)
        return text, tokens

    async def _ask(self, system: str, user: str) -> str | None:
        decision = self.guardian.authorize_llm_call(len(system + user) // 4 + self.cfg.max_tokens)
        if not decision.allowed:
            return None
        try:
            text, tokens = await asyncio.to_thread(self._post, system, user)
        except Exception as exc:  # network errors, HTTP errors, malformed payloads
            self.errors += 1
            log.warning("LLM teacher call failed: %s", exc)
            return None
        self.calls += 1
        self.guardian.record_llm_usage(tokens)
        return text

    @staticmethod
    def _catalog(spec: CapabilitySpec) -> str:
        lines = []
        for lab in spec.active_labels():
            desc = spec.label_descriptions.get(lab, "")
            lines.append(f"- {lab}" + (f": {desc}" if desc else ""))
        return "\n".join(lines)

    async def label(self, spec, text):
        allow_new = spec.open_labels and spec.name != DISPATCH
        system = ("You are a teacher producing training labels for a small specialist neural "
                  "network. Reply with a single JSON object and nothing else.")
        user = (f"Capability: {spec.name}\nDescription: {spec.description}\n"
                f"Labels:\n{self._catalog(spec)}\n"
                + ("If no label fits, you may propose one new short snake_case label.\n" if allow_new else
                   "You must choose one of the labels above.\n")
                + f"\nInput:\n<<<\n{text}\n>>>\n\n"
                'Return {"label": "<label>", "confidence": <0..1>, "why": "<short reason>"}')
        reply = await self._ask(system, user)
        data = _extract_json(reply or "")
        if not data or not data.get("label"):
            return None
        lab = str(data["label"]).strip() if spec.name == DISPATCH else _clean_label(str(data["label"]))
        if lab not in spec.active_labels() and not allow_new:
            return None
        try:
            conf = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        return Lesson(lab, max(0.0, min(conf, 1.0)), "llm", str(data.get("why", ""))[:200])

    async def generate(self, spec, label, n):
        system = "You write realistic, varied training data. Reply with a single JSON object only."
        desc = spec.label_descriptions.get(label, "")
        user = (f"Capability: {spec.name} - {spec.description}\nAll labels:\n{self._catalog(spec)}\n\n"
                f"Write {n} diverse, realistic inputs whose correct label is '{label}'"
                + (f" ({desc})" if desc else "") + ". Vary length, wording and style; include "
                "a few hard cases near the boundary with other labels.\n"
                'Return {"examples": ["...", "..."]}')
        data = _extract_json(await self._ask(system, user) or "")
        if not data:
            return []
        return [str(x)[:400] for x in data.get("examples", []) if str(x).strip()][:n]

    async def answer(self, text):
        system = ("You are the fallback answerer inside a multi-agent system. No specialist node or "
                  "tool could handle this order. Answer concisely and say if you are unsure.")
        return await self._ask(system, text)


# ------------------------------------------------------------------ heuristic
_PLACEHOLDER = re.compile(r"\{([\w.]+)\}")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\S")


def fill_template(template: str, fillers: dict[str, list[str]], rng: random.Random,
                  namespace: str = "", extra: dict[str, str] | None = None) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if extra and key in extra:
            return extra[key]
        for k in (f"{namespace}.{key}" if namespace else None, key):
            if k and fillers.get(k):
                return rng.choice(fillers[k])
        return m.group(0)
    return _PLACEHOLDER.sub(sub, template)


class HeuristicTeacher(Teacher):
    """Keyword/lexicon teacher that works offline. Deliberately imperfect, like any teacher."""
    name = "heuristic"
    NEGATORS = {"not", "no", "never", "hardly", "isn't", "wasn't", "don't", "didn't", "nothing", "barely"}

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def label_sync(self, spec: CapabilitySpec, text: str) -> Lesson | None:
        lowered = " " + text.lower() + " "
        words = _WORD_RE.findall(text.lower())
        scores: dict[str, float] = defaultdict(float)
        for lab, terms in spec.lexicon.items():
            if lab in spec.inactive_labels or lab not in spec.labels:
                continue
            for term in terms:
                t = term.lower()
                if re.fullmatch(r"[a-z']+", t):
                    for i, w in enumerate(words):
                        if w != t:
                            continue
                        negated = any(words[j] in self.NEGATORS for j in range(max(0, i - 3), i))
                        target = spec.negation.get(lab, lab) if negated else lab
                        scores[target] += 1.0
                elif t in lowered:
                    scores[lab] += 1.0
        if not scores:
            if spec.default_label and spec.default_label not in spec.inactive_labels:
                return Lesson(spec.default_label, 0.6, "heuristic", "no cues")
            return None
        order = {lab: i for i, lab in enumerate(spec.labels)}
        best, score = max(scores.items(), key=lambda kv: (kv[1], -order.get(kv[0], 0)))
        conf = 0.55 + 0.4 * score / sum(scores.values())
        return Lesson(best, round(conf, 3), "heuristic", f"cues={dict(scores)}")

    async def label(self, spec, text):
        return self.label_sync(spec, text)

    def generate_sync(self, spec: CapabilitySpec, label: str, n: int) -> list[str]:
        out: list[str] = []
        own = spec.label_templates.get(label)
        flipped = [m for m, target in spec.negation.items() if target == label]
        for _ in range(n * 4):
            if len(out) >= n:
                break
            if own:
                text = fill_template(self.rng.choice(own), spec.fillers, self.rng, namespace=label)
            elif spec.templates and spec.lexicon.get(label):
                def words(lab: str) -> list[str]:  # template-friendly words if declared
                    return spec.fillers.get(f"word.{lab}") or spec.lexicon[lab]
                if flipped and self.rng.random() < 0.25:
                    word = "not " + self.rng.choice(words(self.rng.choice(flipped)))
                else:
                    word = self.rng.choice(words(label))
                text = fill_template(self.rng.choice(spec.templates), spec.fillers, self.rng,
                                     extra={"word": word})
            else:
                break
            if text not in out:
                out.append(text)
        return out

    async def generate(self, spec, label, n):
        return self.generate_sync(spec, label, n)


# ---------------------------------------------------------------------- pool
class TeacherPool:
    def __init__(self, teachers: dict[str, Teacher], order: tuple, weights: dict):
        self.teachers = teachers
        self.order = list(order)
        self.weights = weights
        self.lessons: dict[str, int] = defaultdict(int)

    @property
    def human(self) -> HumanTeacher:
        return self.teachers["human"]  # type: ignore[return-value]

    def ranked(self) -> list[Teacher]:
        return [self.teachers[n] for n in self.order if n in self.teachers and self.teachers[n].available()]

    def weight(self, source: str) -> float:
        return float(self.weights.get(source, 1.0))

    async def label(self, spec: CapabilitySpec, text: str) -> Lesson | None:
        for teacher in self.ranked():
            lesson = await teacher.label(spec, text)
            if lesson and lesson.label:
                self.lessons[teacher.name] += 1
                return lesson
        return None

    async def generate(self, spec: CapabilitySpec, label: str, n: int) -> tuple[list[str], str]:
        for teacher in self.ranked():
            texts = await teacher.generate(spec, label, n)
            if texts:
                return texts, teacher.name
        return [], ""

    async def answer(self, text: str) -> tuple[str, str] | None:
        for teacher in self.ranked():
            reply = await teacher.answer(text)
            if reply:
                return reply, teacher.name
        return None

    def describe(self) -> dict:
        return {name: {"available": t.available(), "lessons": self.lessons.get(name, 0)}
                for name, t in self.teachers.items()}
