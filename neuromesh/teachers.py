"""External teachers. Nodes never learn from themselves; they learn from:

* HumanTeacher     - corrections and examples from people (highest trust)
* LLMTeacher       - an external LLM that labels data, writes synthetic training examples
                     and answers orders no node can handle yet: Claude via the Anthropic API,
                     a local model on an Ollama server, or any OpenAI-compatible endpoint.
                     Several can be configured; each is its own named teacher with a budget
* HeuristicTeacher - an offline rule-based stand-in so the system runs with no API key

The TeacherPool asks them in the order set by the directives.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.parse
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
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
DEFAULT_ENDPOINTS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com"}


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


def ollama_base_url(configured: str = "") -> str:
    """Resolve the Ollama endpoint: the configured base_url, else $OLLAMA_HOST (same rules
    as the ollama CLI: bare host -> port 11434), else http://localhost:11434."""
    raw = (configured or os.environ.get("OLLAMA_HOST", "") or "http://localhost:11434").strip()
    default_port = 11434
    if "://" not in raw:
        raw = "http://" + raw
    elif raw.startswith("http://"):
        default_port = 80 if not configured and os.environ.get("OLLAMA_HOST") else 11434
    elif raw.startswith("https://"):
        default_port = 443
    parts = urllib.parse.urlsplit(raw)
    host = parts.hostname or "localhost"
    if host in ("0.0.0.0", "::"):  # a server bind address; connect locally
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"{parts.scheme}://{host}:{parts.port or default_port}{parts.path.rstrip('/')}"


def _is_chat_model(entry: dict) -> bool:
    name = str(entry.get("name") or entry.get("model") or "").lower()
    details = entry.get("details") or {}
    families = [details.get("family") or ""] + list(details.get("families") or [])
    return "embed" not in name and not any("bert" in str(f).lower() for f in families)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8", errors="replace"))
        return str(body.get("error") or body)
    except Exception:
        return f"HTTP {exc.code}"


class LLMTeacher(Teacher):
    """An external LLM as a teacher. Providers:

    anthropic - Anthropic Messages API (needs an API key)
    ollama    - a local Ollama server via its native /api/chat. Replies are constrained by a
                JSON schema (labels become an enum), which keeps small local models on-label.
                The server and model are probed periodically; while either is missing the
                teacher is skipped and the next one in the order is asked.
    openai    - any OpenAI-compatible /v1/chat/completions endpoint (vLLM, LM Studio, ...)

    Every call is authorised against this teacher's own hourly budget by the Guardian.
    """

    def __init__(self, name: str, cfg: LLMConfig, guardian, bus=None):
        self.name = name
        self.cfg = cfg
        self.guardian = guardian
        self.bus = bus
        self.provider = cfg.provider
        self.model = cfg.model
        self.calls = 0
        self.errors = 0
        self.installed: list[str] = []
        self._probed_at = 0.0
        if self.provider == "ollama":
            self.base_url = ollama_base_url(cfg.base_url)
            self.state, self.detail = "unknown", "not checked yet"
        else:
            self.base_url = (cfg.base_url or DEFAULT_ENDPOINTS.get(self.provider, "")).rstrip("/")
            self.state, self.detail = "ready", ""
        local = urllib.parse.urlsplit(self.base_url).hostname in LOCAL_HOSTS
        # a local server must never be reached through an HTTP(S)_PROXY from the environment
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}) if local
                                                   else urllib.request.ProxyHandler())

    # ------------------------------------------------------------ availability
    def _key(self) -> str:
        return os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""

    def available(self) -> bool:
        if not self.cfg.enabled:
            return False
        if self.provider == "anthropic":
            return bool(self._key())
        if self.provider == "openai":
            return bool(self.cfg.base_url or self._key())
        if self.provider == "ollama":
            return self.state == "ready"
        return False

    def status(self) -> dict:
        if not self.cfg.enabled:
            state, detail = "off", "disabled in directives"
        elif self.provider == "ollama":
            state, detail = self.state, self.detail
        elif self.available():
            state, detail = "ready", ""
        else:
            state, detail = "off", f"set {self.cfg.api_key_env}" if self.cfg.api_key_env else "no endpoint configured"
        return {"provider": self.provider, "model": self.model, "endpoint": self.base_url,
                "state": state, "detail": detail, "calls": self.calls, "errors": self.errors}

    def _set_state(self, state: str, detail: str) -> None:
        changed = (state, detail) != (self.state, self.detail)
        self.state, self.detail = state, detail
        if changed:
            log.info("teacher %s: %s %s", self.name, state, detail)
            if self.bus is not None:
                self.bus.publish("teacher.status", teacher=self.name, state=state, detail=detail,
                                 model=self.model, provider=self.provider)

    def _probe_sync(self) -> tuple[str, str, list[str], str | None]:
        try:
            data = self._http("GET", "/api/tags", None, timeout=min(5.0, self.cfg.timeout_s))
        except Exception as exc:
            reason = getattr(exc, "reason", exc)
            return "unreachable", f"no Ollama server at {self.base_url} ({reason}); start it with: ollama serve", [], None
        models = data.get("models") or []
        names = [str(m.get("name") or m.get("model")) for m in models]
        if self.cfg.model in ("", "auto"):
            chat = [m for m in models if _is_chat_model(m)]
            if not chat:
                return "no_model", "Ollama is running but has no chat model; run: ollama pull llama3.1", names, None
            picked = str(chat[0].get("name") or chat[0].get("model"))
            return "ready", f"auto-selected {picked} at {self.base_url}", names, picked
        wanted = self.cfg.model if ":" in self.cfg.model else f"{self.cfg.model}:latest"
        if wanted in names or self.cfg.model in names:
            return "ready", f"{self.cfg.model} at {self.base_url}", names, self.cfg.model
        installed = f" (installed: {', '.join(names[:6])})" if names else ""
        return "no_model", f"model {self.cfg.model} is not pulled; run: ollama pull {self.cfg.model}{installed}", names, None

    async def refresh(self, force: bool = False) -> None:
        """Re-check the Ollama server and model (no-op for hosted providers)."""
        if self.provider != "ollama" or not self.cfg.enabled:
            return
        if not force and time.time() - self._probed_at < self.cfg.probe_seconds:
            return
        self._probed_at = time.time()
        state, detail, names, model = await asyncio.to_thread(self._probe_sync)
        self.installed = names
        if model:
            self.model = model
        self._set_state(state, detail)

    # ------------------------------------------------------------------- calls
    def _http(self, method: str, path: str, body: dict | None, timeout: float, headers: dict | None = None) -> dict:
        req = urllib.request.Request(self.base_url + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"content-type": "application/json", **(headers or {})})
        with self._opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, system: str, user: str, schema: dict | None = None) -> tuple[str, int]:
        cfg = self.cfg
        if self.provider == "anthropic":
            data = self._http("POST", "/v1/messages", {
                "model": self.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
                "system": system, "messages": [{"role": "user", "content": user}]},
                cfg.timeout_s, {"x-api-key": self._key(), "anthropic-version": cfg.anthropic_version})
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            usage = data.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        elif self.provider == "ollama":
            body = {"model": self.model, "stream": False, "keep_alive": cfg.keep_alive,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}}
            if schema is not None:
                body["format"] = schema
            if cfg.think != "":
                body["think"] = cfg.think
            data = self._http("POST", "/api/chat", body, cfg.timeout_s)
            if data.get("error"):
                raise RuntimeError(data["error"])
            text = (data.get("message") or {}).get("content") or ""
            tokens = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
        else:
            headers = {"authorization": f"Bearer {self._key()}"} if self._key() else {}
            data = self._http("POST", "/v1/chat/completions", {
                "model": self.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                cfg.timeout_s, headers)
            text = data["choices"][0]["message"]["content"] or ""
            tokens = data.get("usage", {}).get("total_tokens", 0)
        return _THINK_RE.sub("", text).strip(), tokens

    async def _ask(self, system: str, user: str, schema: dict | None = None) -> str | None:
        if not self.available():
            return None
        decision = self.guardian.authorize_llm_call(self.name, self.cfg, len(system + user) // 4 + self.cfg.max_tokens)
        if not decision.allowed:
            return None
        try:
            text, tokens = await asyncio.to_thread(self._post, system, user, schema)
        except urllib.error.HTTPError as exc:
            self.errors += 1
            detail = _http_error_detail(exc)
            if self.provider == "ollama" and exc.code == 404:
                self._set_state("no_model", f"{detail}; run: ollama pull {self.model}")
            log.warning("teacher %s: HTTP %s %s", self.name, exc.code, detail)
            return None
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            self.errors += 1
            reason = getattr(exc, "reason", exc)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                log.warning("teacher %s: no reply within %ss (raise timeout_s for slow models)",
                            self.name, self.cfg.timeout_s)
            elif self.provider == "ollama":
                self._set_state("unreachable", f"lost contact with {self.base_url} ({reason})")
            else:
                log.warning("teacher %s: call failed: %s", self.name, reason)
            return None
        except Exception as exc:  # malformed payloads and the like
            self.errors += 1
            log.warning("teacher %s: call failed: %s", self.name, exc)
            return None
        self.calls += 1
        self.guardian.record_llm_usage(self.name, tokens)
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
        active = spec.active_labels()
        label_schema = {"type": "string"} if allow_new or not active else {"type": "string", "enum": active}
        schema = {"type": "object", "required": ["label", "confidence"],
                  "properties": {"label": label_schema, "confidence": {"type": "number"},
                                 "why": {"type": "string"}}}
        system = ("You are a teacher producing training labels for a small specialist neural "
                  "network. Reply with a single JSON object and nothing else.")
        user = (f"Capability: {spec.name}\nDescription: {spec.description}\n"
                f"Labels:\n{self._catalog(spec)}\n"
                + ("If no label fits, you may propose one new short snake_case label.\n" if allow_new else
                   "You must choose one of the labels above.\n")
                + f"\nInput:\n<<<\n{text}\n>>>\n\n"
                'Return {"label": "<label>", "confidence": <0..1>, "why": "<short reason>"}')
        data = _extract_json(await self._ask(system, user, schema) or "")
        if not data or not data.get("label"):
            return None
        lab = str(data["label"]).strip() if spec.name == DISPATCH else _clean_label(str(data["label"]))
        if lab not in active and not allow_new:
            return None
        try:
            conf = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        return Lesson(lab, max(0.0, min(conf, 1.0)), self.name, str(data.get("why", ""))[:200])

    async def generate(self, spec, label, n):
        schema = {"type": "object", "required": ["examples"],
                  "properties": {"examples": {"type": "array", "items": {"type": "string"}}}}
        system = "You write realistic, varied training data. Reply with a single JSON object only."
        desc = spec.label_descriptions.get(label, "")
        user = (f"Capability: {spec.name} - {spec.description}\nAll labels:\n{self._catalog(spec)}\n\n"
                f"Write {n} diverse, realistic inputs whose correct label is '{label}'"
                + (f" ({desc})" if desc else "") + ". Vary length, wording and style; include "
                "a few hard cases near the boundary with other labels.\n"
                'Return {"examples": ["...", "..."]}')
        data = _extract_json(await self._ask(system, user, schema) or "")
        if not data:
            return []
        return [str(x)[:400] for x in data.get("examples", []) if str(x).strip()][:n]

    async def answer(self, text):
        system = ("You are the fallback answerer inside a multi-agent system. No specialist node or "
                  "tool could handle this order. Start with the answer itself, in your own words; put any "
                  "reasoning, caveats or sources after it, never before. Be concise and say if you are unsure.")
        return await self._ask(system, text)

    async def write_tool(self, requests: list[str]) -> dict | None:
        """Write a small tool (Python, standard library only) for requests the mesh could not meet."""
        schema = {"type": "object", "required": ["name", "description", "code", "examples"],
                  "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                                 "code": {"type": "string"},
                                 "examples": {"type": "array", "items": {"type": "object", "required": ["input", "expect"],
                                              "properties": {"input": {"type": "string"}, "expect": {"type": "string"}}}}}}
        system = ("You write small, safe command-line tools for a multi-agent system. Reply with a single JSON object "
                  "and nothing else.")
        user = ("Write ONE Python tool that meets requests like these:\n" + "\n".join(f"- {r[:200]}" for r in requests)
                + "\n\nRules: the code reads one JSON object from stdin ({\"input\": \"<the request text>\"}) and prints "
                'one JSON object: {"output": <answer>, "say": "<one sentence>"} or {"error": "<why>"}. Use only these '
                "modules: json, re, sys, math, string, datetime, base64, collections, itertools, statistics, textwrap, "
                "unicodedata. No files, no network, no subprocesses, no eval or exec. Give 3 to 5 examples "
                '(input and the expected output). JSON: {"name": "<snake_case>", "description": "...", "code": "...", '
                '"examples": [{"input": "...", "expect": "..."}]}')
        data = _extract_json(await self._ask(system, user, schema) or "")
        if not data or not isinstance(data.get("code"), str):
            return None
        name = _clean_label(str(data.get("name", "")))
        examples = [{"input": str(e.get("input", ""))[:300], "expect": str(e.get("expect", ""))[:300]}
                    for e in (data.get("examples") or []) if isinstance(e, dict) and e.get("input")][:6]
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name or "") or not examples:
            return None
        return {"name": name, "description": str(data.get("description", ""))[:200], "code": data["code"][:20000],
                "examples": examples}

    async def propose(self, examples: list[str], known: list[str], hint: str = "") -> dict | None:
        """Draft one new capability from requests the mesh could not fulfil."""
        schema = {"type": "object", "required": ["kind", "name", "description"],
                  "properties": {"kind": {"type": "string", "enum": ["skill", "tool"]},
                                 "name": {"type": "string"}, "description": {"type": "string"},
                                 "labels": {"type": "array", "items": {"type": "string"}},
                                 "triggers": {"type": "array", "items": {"type": "string"}}}}
        system = ("You help a small multi-agent system decide which capability it is missing. "
                  "Reply with a single JSON object and nothing else.")
        user = ("These requests could not be fulfilled:\n" + "\n".join(f"- {e[:200]}" for e in examples)
                + f"\n\nIt already has: {', '.join(known)}.\n" + (f"The requests name {hint}.\n" if hint else "")
                + "Propose ONE new capability. If the requests ask to sort text into a fixed set of categories, "
                'propose a skill: {"kind": "skill", "name": "<snake_case>", "description": "...", '
                '"labels": ["<snake_case>", ...], "triggers": ["<word>", ...]}. If they need an action or outside '
                'data, propose a tool: {"kind": "tool", "name": "<snake_case>", "description": "...", '
                '"triggers": ["<word>", ...]}.')
        data = _extract_json(await self._ask(system, user, schema) or "")
        if not data:
            return None
        kind = str(data.get("kind", "")).lower()
        name = _clean_label(str(data.get("name", "")))
        if kind not in ("skill", "tool") or not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name or ""):
            return None
        labels = [x for x in (_clean_label(str(v)) for v in data.get("labels") or []) if x][:12]
        if kind == "skill" and len(set(labels)) < 2:
            labels = []
        triggers = [str(t).strip().lower()[:40] for t in data.get("triggers") or [] if str(t).strip()][:8]
        return {"kind": kind, "name": name, "description": str(data.get("description", ""))[:300],
                "labels": list(dict.fromkeys(labels)), "triggers": triggers}

    NOT_FOUND = "NOT IN PASSAGES"

    async def compose(self, question: str, hits) -> str | None:
        """Answer a question from retained passages only (no outside knowledge)."""
        doc = hits[0].doc
        excerpts = "\n".join(f"[{i + 1}] ({h.section}) {' '.join(h.text.split())[:700]}"
                             for i, h in enumerate(hits[:4]))
        system = ("Answer the question using only the passages provided. Write the answer in your own words "
                  "rather than copying sentences, and start with the answer itself - no preamble; the source is "
                  "added after your answer automatically. Be concise: at most three sentences. If the passages "
                  f"do not contain the answer, reply exactly: {self.NOT_FOUND}")
        user = f"Question: {question}\n\nPassages from the {doc.source} article '{doc.title}':\n{excerpts}"
        reply = await self._ask(system, user)
        if not reply or self.NOT_FOUND in reply.upper():
            return None
        return reply.strip()


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

    def scores(self, spec: CapabilitySpec, text: str) -> dict[str, float]:
        """How strongly each label's cues appear in the text (a director ranks other routes by this)."""
        lowered = " " + text.lower() + " "
        words = _WORD_RE.findall(text.lower())
        scores: dict[str, float] = defaultdict(float)
        for lab, terms in spec.lexicon.items():
            if lab in spec.inactive_labels or lab not in spec.labels:
                continue
            for term in terms:
                if term.startswith("re:"):
                    if re.search(term[3:], text, re.I):
                        scores[lab] += 2.0
                elif re.fullmatch(r"[a-z']+", term.lower()):
                    scores[lab] += sum(1.0 for w in words if w == term.lower())
                elif term.lower() in lowered:
                    scores[lab] += 1.0
        return dict(scores)

    def label_sync(self, spec: CapabilitySpec, text: str) -> Lesson | None:
        lowered = " " + text.lower() + " "
        words = _WORD_RE.findall(text.lower())
        scores: dict[str, float] = defaultdict(float)
        for lab, terms in spec.lexicon.items():
            if lab in spec.inactive_labels or lab not in spec.labels:
                continue
            for term in terms:
                if term.startswith("re:"):  # a structural cue, e.g. "re:\\d+\\s*[*/+x-]\\s*\\d+"
                    if re.search(term[3:], text, re.I):
                        scores[lab] += 2.0
                    continue
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
        self.activity = None  # an ActivityTracker, so a live view can show lessons in flight
        self.bus = None

    @property
    def human(self) -> HumanTeacher:
        return self.teachers["human"]  # type: ignore[return-value]

    def ranked(self, spec: CapabilitySpec | None = None) -> list[Teacher]:
        """Available teachers in the directives' order. A capability that lists its own `teachers`
        gets only those, in its order: ones from the global order, plus specialist teachers
        (like the temperament policy) that are never asked otherwise."""
        names = self.order
        if spec is not None and spec.teachers:
            names = [n for n in spec.teachers
                     if n in self.order or getattr(self.teachers.get(n), "specialist", False)]
        return [self.teachers[n] for n in names if n in self.teachers and self.teachers[n].available()]

    def weight(self, source: str) -> float:
        return float(self.weights.get(source, 1.0))

    def _track(self, teacher: Teacher, text: str):
        if self.activity is None or not isinstance(teacher, LLMTeacher):
            return contextlib.nullcontext()  # instant teachers are not worth showing in flight
        return self.activity.track("lesson", teacher.name, text, model=getattr(teacher, "model", ""))

    async def label(self, spec: CapabilitySpec, text: str) -> Lesson | None:
        for teacher in self.ranked(spec):
            t0 = time.time()
            with self._track(teacher, f"labelling {spec.name}: {' '.join(text.split())[:80]}"):
                lesson = await teacher.label(spec, text)
            if lesson and lesson.label:
                self.lessons[teacher.name] += 1
                if self.bus is not None:
                    self.bus.publish("lesson.label", teacher=teacher.name, capability=spec.name,
                                     text=text[:200], label=lesson.label, seconds=round(time.time() - t0, 3))
                return lesson
        return None

    async def generate(self, spec: CapabilitySpec, label: str, n: int) -> tuple[list[str], str]:
        for teacher in self.ranked(spec):
            with self._track(teacher, f"writing {n} examples of {spec.name}={label}"):
                texts = await teacher.generate(spec, label, n)
            if texts:
                return texts, teacher.name
        return [], ""

    async def compose(self, question: str, hits) -> tuple[str, str] | None:
        """Have the first available LLM teacher write an answer from retained passages."""
        for teacher in self.ranked():
            if not isinstance(teacher, LLMTeacher):
                continue
            with self._track(teacher, f"answering from knowledge: {' '.join(question.split())[:70]}"):
                reply = await teacher.compose(question, hits)
            if reply:
                return reply, teacher.name
        return None

    async def propose(self, examples: list[str], known: list[str], hint: str = "") -> tuple[dict, str] | None:
        """Have the first available LLM teacher draft a capability proposal (None offline)."""
        for teacher in self.ranked():
            if not isinstance(teacher, LLMTeacher):
                continue
            with self._track(teacher, f"drafting a capability proposal from {len(examples)} orders"):
                draft = await teacher.propose(examples, known, hint)
            if draft:
                return draft, teacher.name
        return None

    def has_llm(self) -> bool:
        """An LLM teacher is listed and available."""
        return any(isinstance(t, LLMTeacher) for t in self.ranked())

    async def write_tool(self, requests: list[str]) -> tuple[dict, str] | None:
        """Have the first available LLM teacher write a tool's code (None offline)."""
        for teacher in self.ranked():
            if not isinstance(teacher, LLMTeacher):
                continue
            with self._track(teacher, f"writing a tool for {len(requests)} requests"):
                made = await teacher.write_tool(requests)
            if made:
                return made, teacher.name
        return None

    async def answer(self, text: str) -> tuple[str, str] | None:
        for teacher in self.ranked():
            with self._track(teacher, f"answering: {' '.join(text.split())[:80]}"):
                reply = await teacher.answer(text)
            if reply:
                return reply, teacher.name
        return None

    async def refresh(self, force: bool = False) -> None:
        """Re-probe teachers that depend on a local server (Ollama)."""
        await asyncio.gather(*(t.refresh(force) for name, t in self.teachers.items()
                               if hasattr(t, "refresh") and name in self.order))

    def describe(self) -> dict:
        out = {}
        names = [n for n in self.order if n in self.teachers] + [n for n in self.teachers if n not in self.order]
        for name in names:
            t = self.teachers[name]
            listed = name in self.order
            specialist = getattr(t, "specialist", False)
            info = {"available": (listed or specialist) and t.available(), "in_order": listed,
                    "lessons": self.lessons.get(name, 0)}
            if hasattr(t, "status"):
                info.update(t.status())
            if not listed and not specialist:
                info["detail"] = "not listed in teachers.order"
            out[name] = info
        return out
