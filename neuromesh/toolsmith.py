"""The toolsmith: code nodes that write new tools, ask permission, and experiment with them (P15).

    a request no tool could meet
      --> [coder node]   which recipe from the code library (codelib.py) would meet it? (or none)
      --> fit            the recipe's parameters to the request's own wording, and to any answer a
                         person gave for it; recipes are tried until one fits every known answer
      --> write          tool.py (the recipe's own code, copied in), manifest.toml (its self-tests:
                         probes whose answers are known independently, and the requests themselves),
                         TESTS.md (for people) - into state/workshop/<tool>/<version>/
      --> check          the code: standard library only; no files, network, processes, eval or exec
      --> ask            an approval, and a question in conversation. Nothing it wrote has run yet.
      --> a person approves: it is installed in plugins/, and vetted and self-tested like any tool (P3)
      --> experiment     on its own initiative: it runs the tool on the requests that motivated it and on
                         fresh probes, teaches the router what works, and refines what fails - a new
                         version, which asks permission again
With an LLM teacher (and [toolsmith] llm_code), requests no recipe fits can be written by the teacher,
checked the same way, and are never run before a person approves them.

The coder node is an ordinary internal capability: grown, trained, evolved and promoted like any node,
taught by the CoderTeacher (the recipes' cues and practice requests), people, and what gets approved.
"""
from __future__ import annotations

import ast
import inspect
import itertools
import json
import logging
import pprint
import random
import re
import shutil
import time
from collections import deque
from pathlib import Path

from . import codelib
from .codelib import RECIPES, Recipe
from .teachers import Lesson, Teacher
from .tools import ACTIVE, folder_digest, outputs_match

log = logging.getLogger("neuromesh.toolsmith")

CODER = "coder"
NONE = "none"
ALLOWED_IMPORTS = {"json", "re", "sys", "math", "string", "datetime", "base64", "collections", "itertools",
                   "statistics", "textwrap", "unicodedata", "functools", "fractions", "decimal"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "input",
                   "breakpoint", "getattr", "setattr", "delattr", "memoryview"}
FORBIDDEN_ATTRS = {"system", "popen", "spawn", "fork", "remove", "unlink", "rmtree", "rename", "chmod", "kill",
                   "socket", "connect", "urlopen", "request", "load_module", "exec_module", "__subclasses__",
                   "__globals__", "__builtins__", "__code__", "write_text", "write_bytes"}
NOT_TOOLS = ["what is the capital of France", "how are you today", "tell me a joke", "who wrote Hamlet",
             "sentiment: the hotel was lovely", "what is 12 * 7", "urgency: the server is down",
             "what do you think of the calculator tool", "thanks, that helped", "keep learning about volcanoes",
             "imagine a city on Mars", "solve x^2 - 4 = 0", "count the words in: the quick brown fox"]
PRACTICE = {"w": codelib.WORDS, "W": [w.upper() for w in codelib.WORDS], "p": codelib.PHRASES,
            "n": ["7", "42", "115", "1994", "2026", "5, 3, 9, 1"], "r": ["XIV", "MCMXCIV", "XLII"],
            "l": ["pear, apple, fig", "zebra lion ant"], "d": ["1969-07-20", "2000-01-01", "14 March 2015"],
            "d1": ["2024-01-01", "1 June 2023"], "d2": ["2024-03-01", "25 December 2025"],
            "t": ["write to ann@example.com or bob@test.org", "see https://example.com and www.test.org"],
            "h": ["loving #python and #ai today"], "b": ["aGVsbG8=", "bmV1cm9tZXNo"], "v": ["5", "42.195", "100"],
            "a": ["cat", "red"], "x": ["dog", "blue"]}


# A person asks for a tool: "make a tool that reverses text", "write me a tool to count vowels".
TOOL_REQUEST = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
                          r"(?:make|build|write|create|code)\s+(?:me\s+|us\s+)?(?:a|an)\s+(?:new\s+|small\s+|little\s+)?"
                          r"tool\s+(?:that\s+(?:can\s+)?|to\s+|for\s+|which\s+(?:can\s+)?)(.+?)\s*[?.!]*$", re.I | re.S)


def tool_request(text: str) -> str | None:
    """What a person asked a tool to do, if the order asks the mesh to write one."""
    m = TOOL_REQUEST.match(str(text))
    return " ".join(m.group(1).split()) if m else None


# ============================================================== the coder teacher
def cue_scores(text: str) -> dict[str, float]:
    """How strongly each recipe's cues appear in a request."""
    low = " " + str(text).lower() + " "
    out: dict[str, float] = {}
    for r in RECIPES.values():
        score = 0.0
        for cue in r.cues + r.keywords:
            if cue.startswith("re:"):
                if re.search(cue[3:], text, re.I):
                    score += 2.0
            elif re.search(r"(?<![a-z])" + re.escape(cue.lower()) + r"(?:s|es|d|ed|ing)?(?![a-z])", low):
                score += 1.0 + 0.5 * (" " in cue)
        if score:
            out[r.id] = score
    return out


class CoderTeacher(Teacher):
    """Teaches the coder node which recipe a request needs, from the recipes' own cues, and writes
    practice requests for every recipe (and requests that need no new tool). A specialist."""
    name = "coder"
    specialist = True

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def status(self) -> dict:
        return {"detail": "specialist: teaches the coder node which recipe a request needs"}

    def label_sync(self, text: str) -> Lesson:
        scores = cue_scores(text)
        if not scores:
            return Lesson(NONE, 0.7, self.name, "no recipe's cues appear")
        best = max(scores.items(), key=lambda kv: (kv[1], -list(RECIPES).index(kv[0])))
        total = sum(scores.values())
        return Lesson(best[0], round(0.55 + 0.4 * best[1] / total, 3), self.name, f"cues={scores}")

    async def label(self, spec, text):
        if spec.name != CODER:
            return None
        return self.label_sync(text)

    async def generate(self, spec, label, n):
        if spec.name != CODER:
            return []
        if label == NONE:
            pool = list(NOT_TOOLS)
            self.rng.shuffle(pool)
            return pool[:n]
        r = RECIPES.get(label)
        if r is None:
            return []
        out: list[str] = []
        for _ in range(n * 6):
            tmpl = self.rng.choice(r.needs)
            text = re.sub(r"\{(\w+)\}", lambda m: self.rng.choice(PRACTICE.get(m.group(1), ["text"])), tmpl)
            if text not in out:
                out.append(text)
            if len(out) >= n:
                break
        return out


# ============================================================== writing the code
def _source(obj) -> str:
    return inspect.getsource(obj).rstrip() + "\n"


def tool_source(recipe: Recipe, params: dict, header: str) -> str:
    """tool.py for a recipe: the block's own code and its helpers, copied from codelib, with its parameters."""
    seen, parts, tables = set(), [], []
    for fn in tuple(recipe.helpers) + (recipe.block,):
        for dep in _deps(fn):
            if dep not in seen:
                seen.add(dep)
                parts.append(dep)
    for fn in parts:
        for t in codelib.BLOCK_TABLES.get(fn.__name__, ()):
            if t not in tables:
                tables.append(t)
    table_src = "".join(f"{t} = {_literal(codelib.TABLES[t])}\n" for t in tables)
    body = "\n\n".join(_source(fn) for fn in parts)
    return (f'"""{header}"""\nimport base64\nimport datetime\nimport json\nimport re\nimport sys\n\n'
            f"PARAMS = {_literal(params)}\n{table_src}\n\n{body}\n\n"
            f"def main():\n    request = json.load(sys.stdin)\n    print(json.dumps({recipe.block.__name__}(request['input'], PARAMS)))\n\n\n"
            "if __name__ == \"__main__\":\n    try:\n        main()\n    except Exception as exc:\n"
            "        print(json.dumps({\"error\": str(exc)}))\n")


def _deps(fn) -> list:
    """The codelib functions a function calls (helpers first), then itself."""
    out = []
    src = inspect.getsource(fn)
    for name in ("target", "numbers_in", "show_number", "say_number", "dates_in", "unit_of"):
        if name != fn.__name__ and re.search(rf"\b{name}\(", src):
            out += [d for d in _deps(getattr(codelib, name)) if d not in out]
    out.append(fn)
    return out


def _literal(value) -> str:
    """A Python literal for a table or the parameters, laid out for people to read."""
    return pprint.pformat(value, width=110, compact=True, sort_dicts=False)


def check_code(source: str) -> list[str]:
    """Problems that keep code from being offered: standard library only, nothing that reaches outside."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"it does not parse: {exc}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"imports {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"imports from {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"uses {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"uses .{node.attr}")
    if "def main" not in source and "json.dumps" not in source:
        problems.append("it never prints a JSON answer")
    return sorted(set(problems))


def _toml_str(s) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def manifest_text(name: str, version: str, description: str, keywords, templates, fillers, examples, made_for) -> str:
    lines = [f"name = {_toml_str(name)}", f"version = {_toml_str(version)}", f"description = {_toml_str(description)}",
             'entrypoint = "tool.py"', 'author = "neuromesh"                  # written by its toolsmith; runs only once a person approves (P15)',
             "permissions = []                       # made tools never get permissions",
             "keywords = [" + ", ".join(_toml_str(k) for k in keywords) + "]",
             "timeout_s = 5", "",
             "# what it was made for:"] + [f"#   {t[:110]}" for t in made_for[:6]] + [""]
    if templates:
        lines.append("templates = [" + ", ".join(_toml_str(t) for t in templates) + "]")
        lines.append("")
        lines.append("[fillers]")
        for k, vals in fillers.items():
            lines.append(f"{k} = [" + ", ".join(_toml_str(v) for v in vals) + "]")
        lines.append("")
    for ex in examples:
        lines += ["[[examples]]", f"input = {_toml_str(ex['input'])}", f"expect = {_toml_str(ex['expect'])}"]
        if ex.get("route") is False:
            lines.append("route = false")
        lines.append("")
    return "\n".join(lines)


# ================================================================== the toolsmith
class Toolsmith:
    """Writes tools from the code library (or with an LLM teacher), asks permission, installs what a
    person approves, and experiments with it (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.toolsmith
        self.workshop = system.state_dir / "workshop"
        raw = system.store.get_kv("toolsmith.state")
        st = json.loads(raw) if raw else {}
        self.drafts: dict[str, dict] = st.get("drafts", {})
        self.approved_digests: dict[str, str] = st.get("approved", {})
        self.cooldown: dict[str, float] = {k: v for k, v in st.get("cooldown", {}).items() if v > time.time()}
        self.history: deque = deque(st.get("history", []), maxlen=60)
        self.builds: deque = deque(t for t in st.get("builds", []) if time.time() - t < 3600)
        self.trials: deque = deque(t for t in st.get("trials", []) if time.time() - t < 3600)
        self.rng = random.Random(system.rng.randrange(1 << 30))
        self._classified: dict[tuple[str, str], tuple[str, float, str]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def save(self) -> None:
        st = {"drafts": self.drafts, "approved": self.approved_digests, "cooldown": self.cooldown,
              "history": list(self.history), "builds": list(self.builds),
              "trials": list(self.trials)}
        self.sys.store.set_kv("toolsmith.state", json.dumps(st))

    def _budget(self, which: deque, limit: int) -> int:
        now = time.time()
        while which and now - which[0] > 3600:
            which.popleft()
        return limit - len(which)

    # ------------------------------------------------------------ needs
    def classify(self, text: str) -> tuple[str, float, str]:
        """Which recipe a request needs (without waiting): the coder node when it is trusted, else its teacher."""
        s = self.sys
        node = s.registry.champion(CODER) if CODER in s.capabilities else None
        if node is not None and (node.params is None or
                                 (node.metrics.get("holdout_acc") or 0) < s.directives.nodes.serve_min_accuracy):
            node = None  # not trusted yet: the code library's cues decide
        key = (text, node.id if node is not None else "")
        if key in self._classified:
            return self._classified[key]
        out = None
        if node is not None:
            label, conf = node.predict(s.encode(text))
            if conf >= s.directives.orders.confidence_threshold:
                out = (label, conf, f"coder node {node.id}")
        if out is None:
            lesson = CoderTeacher().label_sync(text)
            out = (lesson.label, lesson.confidence, "the code library's cues")
        self._classified[key] = out
        while len(self._classified) > 500:
            self._classified.pop(next(iter(self._classified)))
        return out

    def needs(self) -> list[dict]:
        """Requests no tool could meet, grouped by the recipe that would meet them, not yet built or declined."""
        if not self.enabled:
            return []
        s = self.sys
        texts = [u.text for u in s.mind.unmet.values() if not tool_request(u.text)]
        groups: dict[str, dict] = {}
        made = {d["recipe"] for d in self.drafts.values() if d["status"] != "declined"}
        for text in dict.fromkeys(texts):
            recipe, conf, how = self.classify(text)
            if recipe == NONE or recipe not in RECIPES or conf < self.pol.need_confidence:
                continue
            if recipe in made or self.cooldown.get(recipe, 0) > time.time():
                continue
            g = groups.setdefault(recipe, {"recipe": recipe, "texts": [], "how": how})
            g["texts"].append(text)
        out = [g for g in groups.values() if len(g["texts"]) >= self.pol.min_needs]
        return sorted(out, key=lambda g: -len(g["texts"]))

    def fresh(self) -> list[str]:
        """Tools it made that are active and not yet tried out."""
        s = self.sys
        return [n for n, d in self.drafts.items() if d["status"] == "active" and n in s.tools.tools
                and s.tools.tools[n].state == ACTIVE]

    # ------------------------------------------------------------ building
    def _examples(self, texts: list[str]) -> list[tuple[str, str | None]]:
        """Each request, with the answer a person gave for it if memory holds one."""
        out = []
        for t in texts:
            known = self.sys.memory.known(t) if self.sys.memory.enabled else None
            out.append((t, known.answer if known is not None else None))
        return out

    def fit(self, recipe: Recipe, texts: list[str]) -> tuple[dict, list[str]]:
        """The recipe's parameters for these requests: its triggers, plus each cue as the requests word it
        (with the little words after it, so "reverse this word hello" reads "hello"); and what still does
        not fit: requests the block cannot do, or answers differently than a person said."""
        params = {"triggers": list(recipe.triggers), **recipe.params}
        for t in texts:
            low = t.lower()
            for cue in recipe.cues:
                if cue.startswith("re:"):
                    continue
                m = re.search(r"(?<![a-z])" + re.escape(cue.lower()) + r"(?![a-z])(?:\s+(?:the|of|this|these|word|text|me))*", low)
                if m and m.group(0).strip() not in params["triggers"]:
                    params["triggers"].append(m.group(0).strip())
        return params, self._problems(recipe, params, texts)

    def _problems(self, recipe: Recipe, params: dict, texts: list[str]) -> list[str]:
        """Requests the block cannot do with these parameters, or answers differently than a person said."""
        problems = []
        for text, want in self._examples(texts):
            try:
                got = recipe.block(text, params)["output"]
            except Exception as exc:
                problems.append(f"'{text[:60]}': {exc}")
                continue
            if want is not None and not outputs_match(got, want) and not outputs_match(want, got):
                problems.append(f"'{text[:60]}': gives {got!r}, a person said {want!r}")
        return problems

    async def build(self, texts: list[str], why: str, by: str, asked: str = "") -> dict | None:
        """Write a tool for requests no tool could meet (or that a person asked for), check it, and ask
        permission. None if it can't (the reason is in its history)."""
        s = self.sys
        texts = [t for t in dict.fromkeys(texts) if t and t.strip()]
        about = (texts or [asked])[0]
        if not self.enabled or not about:
            return None
        if self._budget(self.builds, self.pol.builds_per_hour) <= 0:
            self._log("build", about, f"its budget of builds this hour is spent ({self.pol.builds_per_hour})", by)
            return None
        if sum(1 for d in self.drafts.values() if d["status"] == "awaiting approval") >= self.pol.max_waiting:
            self._log("build", about, f"{self.pol.max_waiting} tools are already waiting for approval", by)
            return None
        self.builds.append(time.time())
        votes: dict[str, float] = {}
        for t in (texts + [asked] if asked else texts)[:8]:  # the coder node (or the teacher teaching it) says which recipe
            inf = await s.infer(CODER, t) if CODER in s.capabilities else None
            label, conf = (inf.label, inf.confidence) if inf is not None else self.classify(t)[:2]
            if label and label != NONE and label in RECIPES:
                votes[label] = votes.get(label, 0.0) + conf
        if texts:  # with requests to check against, recipes whose cues appear are worth trying too
            for rid, sc in cue_scores(" ".join(texts)).items():
                votes[rid] = votes.get(rid, 0.0) + 0.01 * sc
        ranked = sorted(votes, key=lambda r: -votes[r])[: 4 if texts else 1]
        tried = []
        for rid in ranked:
            params, problems = self.fit(RECIPES[rid], texts)
            tried.append(f"{rid}: {problems[0] if problems else 'fits'}")
            if not problems:
                return self.draft(RECIPES[rid], params, texts, why, by, asked)
        if self.pol.llm_code:
            made = await s.teachers.write_tool((texts + [asked] if asked else texts)[:8])
            if made:
                return self.draft_written(made[0], made[1], texts, why, by, asked)
        self._log("build", about, "no recipe in its code library fits" + (f" ({'; '.join(tried)})" if tried else "")
                  + ("" if self.pol.llm_code and s.teachers.has_llm() else "; no LLM teacher to write one"), by)
        return None

    def build_now(self, texts: list[str], why: str, by: str) -> dict | None:
        """A build without waiting on nodes or teachers (the coder node's classification, or cues): for
        a proposal a person adopts."""
        votes: dict[str, float] = {}
        for t in texts:
            label, conf, _ = self.classify(t)
            if label in RECIPES:
                votes[label] = votes.get(label, 0.0) + conf
        for rid in sorted(votes, key=lambda r: -votes[r])[:3]:
            params, problems = self.fit(RECIPES[rid], texts)
            if not problems:
                return self.draft(RECIPES[rid], params, texts, why, by)
        return None

    def _name(self, recipe_name: str, recipe_id: str) -> str:
        s = self.sys
        name, i = recipe_name, 2
        while (name in s.tools.tools and s.tools.tools[name].manifest.author != "neuromesh") or \
                (name in self.drafts and self.drafts[name]["recipe"] != recipe_id):
            name, i = f"{recipe_name}{i}", i + 1
        return name

    def draft(self, recipe: Recipe, params: dict, texts: list[str], why: str, by: str, asked: str = "") -> dict | None:
        """Write the files for a recipe tool, check them, and ask permission."""
        name = self._name(recipe.name, recipe.id)
        old = self.drafts.get(name)
        version = _bump(old["version"]) if old else "1.0.0"
        tests = self._tests(recipe, params, texts)
        made_for = texts or [f"(a person asked for a tool that {asked})"]
        header = (f"{name} v{version}: {recipe.title}.\n\nWritten by NeuroMesh's toolsmith from the '{recipe.id}' recipe "
                  f"(neuromesh/codelib.py) for:\n" + "".join(f"  {t[:100]}\n" for t in made_for[:3])
                  + "It runs only after a person approves it (P15). Its self-tests are in manifest.toml.\n")
        source = tool_source(recipe, params, header)
        templates, fillers = _templates(recipe)
        manifest = manifest_text(name, version, f"{recipe.title[0].upper()}{recipe.title[1:]} (made by NeuroMesh's toolsmith)",
                                 list(recipe.keywords), templates, fillers, tests, made_for)
        return self._offer(name, version, recipe.id, recipe.title, source, manifest, tests, texts, why, by, params, asked)

    def draft_written(self, made: dict, teacher: str, texts: list[str], why: str, by: str, asked: str = "") -> dict | None:
        """A tool an LLM teacher wrote: checked the same way, and never run before a person approves it."""
        name = self._name(made["name"], "custom")
        old = self.drafts.get(name)
        version = _bump(old["version"]) if old else "1.0.0"
        tests = [{"input": e["input"], "expect": e["expect"]} for e in made["examples"]]
        words = [w for w in re.findall(r"[a-z]{4,}", " ".join(texts + [asked]).lower()) if w not in _COMMON]
        manifest = manifest_text(name, version, f"{made['description'] or 'A tool'} (written by the {teacher} teacher for "
                                 "NeuroMesh's toolsmith)", list(dict.fromkeys(words))[:6], [], {}, tests,
                                 texts or [f"(a person asked for a tool that {asked})"])
        return self._offer(name, version, "custom", made["description"] or name, made["code"], manifest, tests, texts,
                           why, by, {"teacher": teacher}, asked)

    def _tests(self, recipe: Recipe, params: dict, texts: list[str]) -> list[dict]:
        """Self-tests: probes whose answers are computed independently of the block, then the requests
        it was made for (their answers worked out by the recipe; they also teach the router)."""
        out: list[dict] = []
        for tmpl, slots, expect in recipe.probes:
            keys = list(slots)
            for combo in list(itertools.product(*(slots[k] for k in keys)))[:2]:
                vals = dict(zip(keys, combo))
                out.append({"input": tmpl.format(**vals), "expect": str(expect(vals)), "route": False})
        out = out[:4]
        for text, want in self._examples(texts)[:4]:
            try:
                got = recipe.block(text, params)["output"]
            except Exception:
                continue
            out.append({"input": text, "expect": str(want if want is not None else got)})
        return out

    def _offer(self, name, version, recipe_id, title, source, manifest, tests, texts, why, by, params, asked) -> dict | None:
        s = self.sys
        about = (texts or [asked])[0]
        problems = check_code(source)
        if problems:
            self._log("build", about, f"{name} v{version} was not offered: {'; '.join(problems)}", by)
            s.bus.publish("tool.unsafe", tool=name, version=version, problems=problems[:5], by=by)
            return None
        folder = self.workshop / name / version
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tool.py").write_text(source)
        (folder / "manifest.toml").write_text(manifest)
        (folder / "TESTS.md").write_text(_tests_md(name, version, title, recipe_id, why, texts, asked, tests))
        approval = s.guardian.request_approval("tool.made", name, {
            "version": version, "recipe": recipe_id, "purpose": title, "why": why, "made_by": by,
            "for": [t[:120] for t in texts[:5]] or [asked[:120]], "tests": len(tests), "folder": str(folder),
            "permissions": []})
        old = self.drafts.get(name)
        record = {"name": name, "version": version, "recipe": recipe_id, "title": title, "params": params,
                  "needs": texts[:10], "asked": asked, "why": why, "by": by, "status": "awaiting approval",
                  "approval": approval.id, "folder": str(folder), "made_at": round(time.time(), 1), "tests": tests,
                  "experiments": [], "score": None, "live": None, "previous": None}
        if old is not None and old.get("live"):  # a refinement: the installed version stays until this one is approved
            record["live"] = old["live"]
            record["previous"] = {k: v for k, v in old.items() if k != "previous"}
            record["experiments"] = old.get("experiments", [])
        self.drafts[name] = record
        like = f"for requests like '{texts[0][:70]}'" if texts else f"as you asked ('{asked[:70]}')"
        q = s.mind.ask("approval", f"I wrote a tool, {name} v{version}, to {title} - {like}. Its code and tests are in "
                                   f"{folder}. May I install it and try it out? (yes/no)", f"I made it because {why}",
                       approval_id=approval.id)
        record["inquiry"] = q.id if q is not None else ""
        self._log("drafted", about, f"{name} v{version} ({recipe_id}): {len(tests)} tests; asked permission ({approval.id})", by)
        s.guardian.audit(by, "tool.drafted", name, version=version, recipe=recipe_id, approval=approval.id,
                         folder=str(folder))
        s.bus.publish("tool.drafted", tool=name, version=version, recipe=recipe_id, approval=approval.id, by=by,
                      why=why[:160], need=about[:120], title=title)
        self.save()
        return record

    # ------------------------------------------------- permission, install
    def on_event(self, ev) -> None:
        try:
            self._on_event(ev)
        except Exception:  # never break the bus; the failure is in the activity log
            log.exception("the toolsmith could not handle %s", ev.topic)

    def _on_event(self, ev) -> None:
        if ev.topic == "approval.decided":
            d = next((x for x in self.drafts.values() if x.get("approval") == ev.data.get("id")), None)
            if d is not None and d["status"] == "awaiting approval":
                if ev.data.get("status") == "approved":
                    self.install(d, ev.data.get("by", "a person"))
                else:
                    self.decline(d, ev.data.get("by", "a person"))
        elif ev.topic in ("tool.active", "tool.quarantined", "tool.retired"):
            d = self.drafts.get(ev.data.get("tool", ""))
            if d is None or d["status"] == "awaiting approval":
                return
            if ev.topic == "tool.active" and d["status"] == "installed":
                d["status"] = "active"
            elif ev.topic == "tool.quarantined":
                d["status"] = "quarantined"
                self._log("quarantined", d["name"], f"{d['name']} v{d['version']}: {ev.data.get('reason', '')}", "explorer")
            elif ev.topic == "tool.retired":
                d["status"] = "retired"
            self.save()

    def install(self, d: dict, by: str) -> None:
        """A person approved it: into plugins/, where it is vetted and self-tested like any tool."""
        s = self.sys
        dest = s.tools.dirs[0] / d["name"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(d["folder"], dest)
        self.approved_digests[d["name"]] = folder_digest(dest)
        d["status"], d["approved_by"], d["live"], d["previous"] = "installed", by, d["version"], None
        if CODER in s.capabilities and d["recipe"] in RECIPES:  # a person approved: the coder node chose well
            for text in d["needs"] + ([d["asked"]] if d.get("asked") else []):
                s.learn(CODER, text, d["recipe"], "approved")
        self._log("installed", d["name"], f"{d['name']} v{d['version']} approved by {by}; installed in {dest}", by)
        s.guardian.audit(by, "tool.installed", d["name"], version=d["version"], folder=str(dest))
        s.bus.publish("tool.installed", tool=d["name"], version=d["version"], by=by)
        self.save()

    def decline(self, d: dict, by: str) -> None:
        s = self.sys
        declined = d["version"]
        if d.get("previous"):  # a declined refinement: the installed version stays, and it is not refined again
            prev = d["previous"]
            self.drafts[d["name"]] = {**prev, "previous": None, "refine_declined": prev["version"]}
            line = f"{d['name']} v{declined} declined by {by}; v{prev['version']} stays installed"
        else:
            d["status"] = "declined"
            self.cooldown[d["recipe"]] = time.time() + self.pol.retry_after_seconds
            line = f"{d['name']} v{declined} declined by {by}; it won't offer it again for a while"
        self._log("declined", d["name"], line, by)
        s.guardian.audit(by, "tool.declined", d["name"], version=declined)
        s.bus.publish("tool.declined", tool=d["name"], version=declined, by=by)
        self.save()

    def approved(self, name: str, digest: str) -> bool:
        return self.approved_digests.get(name) == digest

    def record_approval(self, name: str, digest: str, by: str) -> None:
        """A person approved a made tool's files through the Explorer (e.g. ones placed by hand)."""
        self.approved_digests[name] = digest
        self.save()

    def resume(self) -> None:
        """After a restart, ask again for what was waiting (approvals are not kept across restarts)."""
        s = self.sys
        for d in self.drafts.values():
            if d["status"] == "awaiting approval" and d.get("approval") not in s.guardian.approvals:
                a = s.guardian.request_approval("tool.made", d["name"], {
                    "version": d["version"], "recipe": d["recipe"], "purpose": d["title"], "why": d["why"],
                    "made_by": d["by"], "for": d["needs"][:5] or [d.get("asked", "")], "tests": len(d["tests"]),
                    "folder": d["folder"], "permissions": []})
                d["approval"] = a.id
                q = s.mind.ask("approval", f"I wrote a tool, {d['name']} v{d['version']}, to {d['title']}. Its code and "
                                           f"tests are in {d['folder']}. May I install it and try it out? (yes/no)",
                               f"I made it because {d['why']}", approval_id=a.id)
                d["inquiry"] = q.id if q is not None else ""
        self.save()

    def claims(self, text: str) -> bool:
        """A request its code library could meet (so the mind need not propose a stub tool for it)."""
        if not self.enabled:
            return False
        recipe, conf, _ = self.classify(text)
        return recipe in RECIPES and conf >= self.pol.need_confidence

    # ------------------------------------------------------- experiments
    def probes(self, d: dict, n: int) -> list[tuple[str, object, str]]:
        """What to try a tool on: the requests it was made for, new ones like them that nothing met
        since, then fresh probes whose answers are known independently."""
        known = dict(self._examples(d["needs"][:4]))
        out = [(t, known.get(t), "made for") for t in d["needs"][:4]]
        if d["recipe"] in RECIPES:
            seen = set(d["needs"])
            for u in list(self.sys.mind.unmet.values())[-30:]:
                if u.text not in seen and self.classify(u.text)[0] == d["recipe"] and not tool_request(u.text):
                    seen.add(u.text)
                    out.append((u.text, dict(self._examples([u.text])).get(u.text), "new request"))
            r = RECIPES[d["recipe"]]
            for tmpl, slots, expect in r.probes:
                keys = list(slots)
                combos = list(itertools.product(*(slots[k] for k in keys)))
                self.rng.shuffle(combos)
                for combo in combos[:max(1, n // max(1, len(r.probes)))]:
                    vals = dict(zip(keys, combo))
                    out.append((tmpl.format(**vals), expect(vals), "probe"))
        else:
            out += [(t["input"], t["expect"], "its own test") for t in d["tests"]]
        return out[: n + 8]

    async def experiment(self, name: str, by: str) -> dict | None:
        """Try a tool it made, on its own initiative: score it, teach the router the requests it meets,
        and refine what fails (a new version, which asks permission again)."""
        s = self.sys
        d = self.drafts.get(name)
        tool = s.tools.tools.get(name)
        if d is None or tool is None or tool.state != ACTIVE or tool.manifest.author != "neuromesh":
            return None
        pending = d["status"] == "awaiting approval"  # a refinement waits for approval: the installed version is tried
        if pending:
            d = d.get("previous") or d
        if not s.guardian.authorize_tool_run(tool).allowed:
            return None
        if self._budget(self.trials, self.pol.experiments_per_hour) <= 0:
            return {"tool": name, "score": None, "line": "its budget of experiments this hour is spent", "results": []}
        self.trials.append(time.time())
        timeout = tool.manifest.timeout_s or s.directives.tools.default_timeout_s
        results, taught, met = [], 0, 0
        with s.activity.track("experiment", by, f"{name} v{tool.manifest.version}: trying it out"):
            for text, expect, kind in self.probes(d, self.pol.experiment_probes):
                run = await s.sandbox.run(tool, text, timeout, s.tool_config(tool))
                s.record_tool_run(tool, run, exploratory=True)
                ok = run.ok and (expect is None or outputs_match(run.output, expect))
                results.append({"input": text[:200], "kind": kind, "ok": ok,
                                "got": str(run.output if run.ok else run.error)[:160],
                                "want": None if expect is None else str(expect)})
                if ok and kind in ("made for", "new request"):
                    if s.learn("dispatch", text, tool.manifest.route, "experiment"):
                        taught += 1
                    for oid, u in list(s.mind.unmet.items()):  # a request it can meet now is no longer unmet
                        if u.text == text:
                            s.mind.unmet.pop(oid, None)
                            met += 1
        right = sum(r["ok"] for r in results)
        score = right / max(1, len(results))
        failed = [r for r in results if not r["ok"]]
        d["experiments"] = (d.get("experiments") or [])[-4:] + [{"ts": round(time.time(), 1), "version": tool.manifest.version,
                                                                  "score": round(score, 3), "runs": len(results),
                                                                  "failed": failed[:5]}]
        d["score"] = round(score, 3)
        proven = score >= self.pol.proven_score and not any(r["kind"] == "new request" for r in failed)
        d["status"] = "proven" if proven else "needs work"
        line = (f"{name} v{tool.manifest.version}: {right}/{len(results)} right"
                + (f"; the router learned {taught} request{'s' if taught != 1 else ''} it meets" if taught else "")
                + (f"; {met} unmet request{'s' if met != 1 else ''} met" if met else ""))
        refined = None
        checkable = [f for f in failed if f["want"] is not None]
        if pending:
            line += f"; v{self.drafts[name]['version']} is still waiting for approval"
        elif checkable and d.get("refine_declined") != d["version"]:
            refined = self.refine(d, checkable, by)
            if refined:
                line += f"; refined it into v{refined['version']}, which asks permission again"
            else:
                line += "; it could not find a way to refine it"
        if failed and len(checkable) < len(failed):
            n = len(failed) - len(checkable)
            line += (f"; {n} failed request{'s' if n != 1 else ''} had no known answer to check a fix against "
                     "(tell it the answer and it can learn from it)")
        if s.mind.enabled:
            s.mind.note("finding", f"tool:{name}", f"I tried {name} on my own: {line}.", private=False, source="toolsmith")
        self._log("experimented", name, line, by)
        s.guardian.audit(by, "tool.experimented", name, version=tool.manifest.version, score=round(score, 3),
                         runs=len(results))
        s.bus.publish("tool.experimented", tool=name, version=tool.manifest.version, score=round(score, 3),
                      runs=len(results), right=right, taught=taught, proven=proven, by=by,
                      refined=refined["version"] if refined else "")
        self.save()
        return {"tool": name, "score": round(score, 3), "line": line, "results": results, "refined": refined}

    def refine(self, d: dict, failed: list[dict], by: str) -> dict | None:
        """A new version that also reads the wordings that failed: for each, the shortest change to its
        triggers that makes the block give the right answer (the longest lead-in, when no answer is known).
        Checked against everything it already gets right; it asks permission again."""
        r = RECIPES.get(d["recipe"])
        if r is None or d["status"] == "awaiting approval":
            return None
        params = {**d["params"], "triggers": list(d["params"].get("triggers", []))}
        fixed = []
        for f in failed:
            trig = _lead_in(r, params, f["input"], f.get("want"))
            if trig is not None:
                params["triggers"].append(trig)
                fixed.append(f["input"])
        if not fixed:
            return None
        texts = list(dict.fromkeys(d["needs"] + [f["input"] for f in failed if f["kind"] == "new request"
                                                 and f["input"] in fixed]))
        problems = self._problems(r, params, texts)
        for tmpl, slots, expect in r.probes:  # nothing it got right may break
            keys = list(slots)
            for combo in itertools.product(*(slots[k] for k in keys)):
                vals = dict(zip(keys, combo))
                try:
                    if not outputs_match(r.block(tmpl.format(**vals), params)["output"], expect(vals)):
                        problems.append(tmpl.format(**vals))
                except Exception as exc:
                    problems.append(f"{tmpl.format(**vals)}: {exc}")
        if problems:
            self._log("refine", d["name"], f"a refinement of {d['name']} broke {len(problems)} checks; not offered", by)
            return None
        return self.draft(r, params, texts, f"its experiments with v{d['version']} failed on {len(failed)} "
                                            f"request{'s' if len(failed) != 1 else ''} ({len(fixed)} now handled)", by,
                          d.get("asked", ""))

    # ------------------------------------------------------------ status
    def _log(self, event: str, about: str, line: str, by: str) -> None:
        self.history.append({"ts": round(time.time(), 1), "event": event, "about": about[:120], "line": line[:300], "by": by})

    def status(self) -> dict:
        s = self.sys
        node = s.registry.champion(CODER) if CODER in s.capabilities else None
        acc = node.metrics.get("holdout_acc") if node else None
        return {"enabled": self.enabled,
                "node": ({"id": node.id, "holdout": acc, "params": node.param_count,
                          "trusted": acc is not None and acc >= s.directives.nodes.serve_min_accuracy} if node else None),
                "examples": s.store.count(CODER) if CODER in s.capabilities else 0,
                "recipes": [{"id": r.id, "title": r.title, "tool": r.name} for r in RECIPES.values()],
                "drafts": [{k: v for k, v in d.items() if k not in ("params", "previous")} for d in self.drafts.values()],
                "needs": self.needs(), "fresh": self.fresh(), "history": list(self.history)[-15:],
                "builds_left": self._budget(self.builds, self.pol.builds_per_hour),
                "experiments_left": self._budget(self.trials, self.pol.experiments_per_hour)}

    def lines(self) -> list[str]:
        st = self.status()
        node = st["node"]
        out = [f"  toolsmith: {'on' if self.enabled else 'off'}; {len(RECIPES)} recipes in its code library; coder node "
               + (f"{node['id']} (held-out {node['holdout']:.0%})" if node and node["holdout"] is not None else
                  f"not trusted yet ({st['examples']} examples; the code library's cues decide)")]
        if st["needs"]:
            out.append("  requests no tool can meet yet: " + "; ".join(f"{g['recipe']} x{len(g['texts'])} ('{g['texts'][0][:40]}')"
                                                                         for g in st["needs"][:4]))
        if not self.drafts:
            out.append("  no tools made yet (/toolsmith build <what it should do>)")
        for d in self.drafts.values():
            extra = (f" ({d['approval']}: /approve {d['approval']}, or /toolsmith show {d['name']})"
                     if d["status"] == "awaiting approval" else
                     f", scored {d['score']:.0%} in its last experiment" if d.get("score") is not None else "")
            live = f"; v{d['live']} is installed" if d.get("live") and d["live"] != d["version"] else ""
            made_for = f"made for '{d['needs'][0][:50]}'" if d["needs"] else f"asked for: {d.get('asked', '')[:50]}"
            out.append(f"  {d['name']} v{d['version']} [{d['status']}]{extra}{live} - {d['title']}; {made_for}")
        recent = list(self.history)[-5:]
        if recent:
            out.append("  recent:")
            out += [f"    {h['event']}: {h['line'][:150]}" for h in reversed(recent)]
        return out


_COMMON = {"that", "this", "with", "from", "into", "tool", "please", "make", "what", "which", "have", "them", "text"}


def _lead_in(recipe: Recipe, params: dict, text: str, want) -> str | None:
    """The shortest lead-in phrase that, as a trigger, makes the block give the known answer to this
    request. Without a known answer nothing can be checked, so nothing is learned."""
    if want is None:
        return None
    words = str(text).split()
    for n in range(1, len(words)):
        trig = " ".join(words[:n]).lower().strip(" ,:;")
        if not trig or trig in params["triggers"]:
            continue
        try:
            got = recipe.block(text, {**params, "triggers": params["triggers"] + [trig]})["output"]
        except Exception:
            continue
        if outputs_match(got, want):
            return trig
    return None


def _bump(version: str) -> str:
    try:
        a, b, c = (int(x) for x in version.split("."))
    except ValueError:
        return "1.1.0"
    return f"{a}.{b + 1}.0"


def _templates(recipe: Recipe) -> tuple[list[str], dict]:
    """Practice orders for routing, with the fillers they use (from the recipe's practice requests)."""
    used = sorted({m for t in recipe.needs for m in re.findall(r"\{(\w+)\}", t)})
    return list(recipe.needs), {k: list(PRACTICE.get(k, ["text"]))[:8] for k in used}


def _tests_md(name, version, title, recipe, why, texts, asked, tests) -> str:
    lines = [f"# {name} v{version}", "", f"{title[0].upper()}{title[1:]}.", "",
             f"Written by NeuroMesh's toolsmith ({'recipe ' + recipe if recipe != 'custom' else 'by an LLM teacher'}) "
             f"because {why}.", "", "It made it for:", ""] + [f"- {t}" for t in texts[:8]] + (
        [f"- a person asked for a tool that {asked}"] if asked else []) + [
        "", "It runs only after a person approves it (P15), in the sandbox, with no permissions. It will then be",
        "self-tested on these (all must pass), and tried out on more requests on its own initiative:", "",
        "| input | expected |", "|---|---|"]
    lines += [f"| {t['input'].replace('|', '/')} | {str(t['expect']).replace('|', '/')} |" for t in tests]
    lines += ["", "The code is in tool.py next to this file.", ""]
    return "\n".join(lines)
