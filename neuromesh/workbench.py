"""The workbench: whole programs it builds by itself from the code it studied in Gitea, with no teacher (P20).

    an experiment (the Tinkerer, when idle, every [workbench] every_seconds; or Experiment now in the Code view)
      --> a Python function it studied in Gitea (gitea.py) that it has not built a program from yet - ones whose
          doc comment shows examples first
      --> a whole program, put together with no LLM:
            its helpers     what the function uses that its own file sets up (imports and constants: gitea.py keeps
                            each Python file's header), the functions it calls that people wrote in the same repository,
                            and theirs in turn
            its imports     a standard module it names (re, math, json ...) or a standard name (Counter, Path ...)
            its checks      the examples in its doc comment (>>> slugify("A b")  /  'a-b') and the tests people wrote
                            for it (a test function that calls it), each printing PASS or FAIL
            its inputs      a demonstration call with inputs from its arguments' names and types (text -> a sentence,
                            n -> 3, numbers -> a list ...), printing what it returns
      --> checked by the scratch pad's validator (compiled; what it reaches for). A name nothing in the repository
          defines leaves it incomplete, and says which
      --> into the scratch pad as an experiment: it runs only as P20 allows. With [code] auto_run "harmless" (a
          person's switch) a program that reaches for nothing runs by itself; otherwise it waits, checked, for a person
          to run it. Its output says what passed, what failed and what each function returned
    and chains: once two single-argument functions from one repository have run, and what one returns is what the
    other takes (text, a number, a list), it builds a program that feeds one into the other and sees what comes out.

What it learns: which functions make whole programs, which run and pass their checks (the code-kind node learns the
kind of each one that ran, as an experiment's lesson), and the programs themselves. When it is asked to write code
with no recipe and no teacher, a program it built and checked comes before an example copied as it is.
Code from Gitea is never run as it is (P20): only a program it built, checked and that P20 lets run.
"""
from __future__ import annotations

import ast
import builtins
import doctest
import json
import logging
import random
import re
import sys
import textwrap
import time
from collections import Counter, deque

from .agents import Agent
from .gitea import CODEKIND

log = logging.getLogger("neuromesh.workbench")

BY = "its workbench"
BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__"}
STDLIB = set(getattr(sys, "stdlib_module_names", ())) or {
    "re", "math", "json", "string", "itertools", "collections", "functools", "datetime", "statistics", "textwrap",
    "unicodedata", "fractions", "decimal", "random", "operator", "heapq", "bisect", "hashlib", "base64", "time",
    "typing", "dataclasses", "enum", "copy", "pprint", "os", "sys", "pathlib", "csv", "io", "uuid", "calendar"}
COMMON = {**{n: f"from collections import {n}" for n in ("Counter", "defaultdict", "OrderedDict", "deque", "namedtuple")},
          **{n: f"from dataclasses import {n}" for n in ("dataclass", "field", "asdict")},
          **{n: f"from functools import {n}" for n in ("partial", "reduce", "lru_cache", "wraps", "cache", "total_ordering")},
          **{n: f"from itertools import {n}" for n in ("chain", "groupby", "islice", "product", "permutations",
                                                        "combinations", "zip_longest", "accumulate")},
          **{n: f"from typing import {n}" for n in ("List", "Dict", "Optional", "Tuple", "Set", "Any", "Union", "Iterable",
                                                     "Callable", "Iterator", "Sequence", "Mapping", "TypeVar")},
          **{n: f"from math import {n}" for n in ("sqrt", "pi", "floor", "ceil", "log", "exp", "sin", "cos", "tan", "gcd",
                                                   "isclose", "inf")},
          "Path": "from pathlib import Path", "Fraction": "from fractions import Fraction", "Decimal": "from decimal import Decimal",
          "Enum": "from enum import Enum", "date": "from datetime import date", "timedelta": "from datetime import timedelta",
          "datetime": "from datetime import datetime", "dedent": "from textwrap import dedent", "pprint": "from pprint import pprint"}
REACHES = re.compile(r"path|file|dir$|folder|url|uri|host|port|sock|conn|client|session|request|response|^db$|cursor|"
                     r"stream|^fp$|handle|^app$|^ctx$|context")
CALLABLE = re.compile(r"^(f|fn|func|function|callback|cb|key|predicate|pred|handler|op)$")
INT_NAMES = re.compile(r"^(n|num|number|count|k|i|j|size|length|width|height|limit|times|start|end|stop|index|idx|depth|"
                       r"level|step|a|b|x|y|year|month|day|base|exp|power|digits|places|precision|amount|total|radius|r|diameter|side|"
                       r"age|max_\w+|min_\w+|"
                       r"n_\w+|num_\w+|\w+_count)$")
LIST_NAMES = re.compile(r"^(numbers|nums|values|vals|items|xs|ys|lst|arr|array|list|seq|sequence|scores|elements|data|"
                        r"points|weights|sizes|counts|ints)$")
TEXTS_NAMES = re.compile(r"^(words|names|lines|strings|tokens|texts|titles|keys|tags|labels|parts|sentences)$")
DICT_NAMES = re.compile(r"^(d|dct|mapping|obj|config|settings|options|record|row|kwargs|params|meta|info|attrs|headers|env)$")
SCHEMA = """
CREATE TABLE IF NOT EXISTS code_programs(id INTEGER PRIMARY KEY, kind TEXT, example INTEGER, other INTEGER, repo TEXT,
    path TEXT, name TEXT, purpose TEXT, code_kind TEXT, ts REAL, item TEXT, status TEXT, helpers TEXT, unresolved TEXT,
    inputs TEXT, checks INTEGER DEFAULT 0, passed INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, returns TEXT,
    takes TEXT, output TEXT, code TEXT, why TEXT);
CREATE INDEX IF NOT EXISTS code_programs_example ON code_programs(example);
"""
GOOD = ("passed", "ran", "checks out")  # what a program it built may be: the best first
RUNNER = '''

def _report(kind, label, fn, want=None):
    """PASS or FAIL for a check, what a demonstration returned, or the error it raised."""
    try:
        got = fn()
    except Exception as exc:  # what went wrong is what it learns from
        print(f"{'ERROR' if kind == 'DEMO' else 'FAIL'} {label}: {type(exc).__name__}: {exc}")
        return
    if kind == "DEMO":
        print(f"DEMO {label} -> {got!r}"[:400])
    elif kind == "TEST":
        print(f"PASS {label}")
    else:
        ok = repr(got) == want or str(got) == want
        print(f"{'PASS' if ok else 'FAIL'} {label}: got {got!r}" + ("" if ok else f", expected {want}"))
'''


# ============================================================================ reading code
def free_names(tree: ast.AST) -> set[str]:
    """Names a piece of code uses that it does not define itself (scopes taken loosely: a name bound anywhere in it
    counts as defined)."""
    bound, used = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (used if isinstance(n.ctx, ast.Load) else bound).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.alias):
            bound.add(n.asname or n.name.split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return used - bound - BUILTINS


def standalone(source: str) -> tuple[str, ast.Module] | None:
    """A function's code on its own: methods and nested functions keep their body's indentation, so it is taken in."""
    for cut in (0, 4, 8, 12, 2, 6):
        lines = source.split("\n")
        if cut:
            if not all(not ln.strip() or ln.startswith(" " * cut) for ln in lines[1:]):
                continue
            lines = lines[:1] + [ln[cut:] for ln in lines[1:]]
        code = "\n".join(lines)
        try:
            return code, ast.parse(code)
        except SyntaxError:
            continue
    return None


def sample(arg: ast.arg, k: int) -> str | None:
    """An input for an argument, from its name and type (None: it needs something it can't make up - a file, the
    network, a function)."""
    n = arg.arg.lower()
    ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
    if REACHES.search(n) or "IO" in ann or "Path" in ann:
        return None
    if CALLABLE.match(n) or "Callable" in ann:
        return None
    if ann in ("bool",) or n.startswith(("is_", "has_", "use_", "allow", "show_", "with_")) or n in (
            "flag", "verbose", "strict", "reverse", "quiet", "debug", "force", "recursive"):
        return "True"
    if ann in ("float",) or n in ("ratio", "rate", "factor", "scale", "temperature", "price", "weight", "fraction"):
        return "2.5"
    if ann in ("int",) or INT_NAMES.match(n):
        return str((3, 4, 5, 2)[k % 4])
    if ann.startswith(("list[str]", "List[str]")) or TEXTS_NAMES.match(n):
        return '["pear", "apple", "fig"]'
    if ann.startswith(("list", "List", "Sequence", "Iterable", "tuple", "Tuple")) or LIST_NAMES.match(n):
        return "[3, 1, 4, 1, 5]"
    if ann.startswith(("dict", "Dict", "Mapping")) or DICT_NAMES.match(n):
        return '{"name": "ada", "year": 1815}'
    if n in ("date", "when", "day_str", "date_str"):
        return '"2024-03-01"'
    if "email" in n:
        return '"ann@example.com"'
    if "title" in n:
        return '"Hello World"'
    if n in ("char", "ch", "c", "letter", "sep", "separator", "delimiter"):
        return '"-"' if n in ("sep", "separator", "delimiter") else '"a"'
    if n in ("word", "token", "key", "name", "tag", "label"):
        return '"hello"'
    return '"Hello World, this is a test"'


def kind_of_value(text: str) -> str:
    """What a printed value is: text, a number, a list, a dict, a truth, or something else."""
    t = text.strip()
    if t[:1] in ("'", '"'):
        return "text"
    if re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", t):
        return "number"
    if t[:1] in ("[", "("):
        return "list"
    if t[:1] == "{":
        return "dict"
    if t in ("True", "False"):
        return "truth"
    return "other"


def takes_kind(arg: ast.arg) -> str:
    s = sample(arg, 0)
    return "" if s is None else kind_of_value(s)


def outcome(stdout: str) -> dict:
    """What a run of a program it built said: checks passed and failed, demonstrations and what they returned."""
    lines = [ln for ln in (stdout or "").splitlines() if ln.startswith(("PASS ", "FAIL ", "DEMO ", "ERROR "))]
    demos = [ln for ln in lines if ln.startswith("DEMO ")]
    returns = ""
    if demos:
        returns = kind_of_value(demos[0].split(" -> ", 1)[1]) if " -> " in demos[0] else ""
    return {"passed": sum(ln.startswith("PASS ") for ln in lines), "failed": sum(ln.startswith("FAIL ") for ln in lines),
            "errors": sum(ln.startswith("ERROR ") for ln in lines), "demos": len(demos), "returns": returns,
            "lines": lines[:30]}


# ============================================================================ the workbench
class Workbench:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.workbench
        self.db = system.store.db
        self.db.executescript(SCHEMA)
        raw = system.store.get_kv("workbench.state")
        st = json.loads(raw) if raw else {}
        self.stats: Counter = Counter(st.get("stats", {}))
        self.log: deque = deque(st.get("log", []), maxlen=60)
        self.rng = random.Random()
        self.busy = False

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled) and self.sys.scratch.enabled

    def save(self) -> None:
        self.sys.store.set_kv("workbench.state", json.dumps({"stats": dict(self.stats), "log": list(self.log)}))

    def _note(self, line: str, **extra) -> None:
        self.log.append({"ts": round(time.time(), 1), "line": line, **extra})
        self.save()

    # ------------------------------------------------------------------ what it has to work with
    def _fn(self, repo: str, name: str, path: str = "") -> dict | None:
        row = self.db.execute("SELECT id FROM code_examples WHERE repo = ? AND name = ? AND lang = 'python' "
                              "ORDER BY (path = ?) DESC, id DESC LIMIT 1", (repo, name, path)).fetchone()
        return self.sys.code_study.example(row[0]) if row else None

    def next_example(self) -> dict | None:
        """A Python function it has not built a program from: doc examples first, then any, the least practised."""
        row = self.db.execute(
            "SELECT e.id FROM code_examples e WHERE e.lang = 'python' AND e.kind != 'test' AND e.lines BETWEEN 2 AND ? "
            "AND NOT EXISTS (SELECT 1 FROM code_programs p WHERE p.example = e.id AND p.kind = 'whole') "
            "ORDER BY (e.doc LIKE '%>>>%') DESC, e.practised, RANDOM() LIMIT 1", (self.pol.max_lines,)).fetchone()
        return self.sys.code_study.example(row[0]) if row else None

    async def _header(self, repo: str, path: str, cache: dict) -> dict:
        if (repo, path) not in cache:
            head = self.sys.code_study.header(repo, path)
            if head is None and self.sys.code_study.enabled:
                try:
                    head = await self.sys.code_study.fetch_header(repo, path)
                except Exception:
                    head = None
            cache[(repo, path)] = head or {}
        return cache[(repo, path)]

    # ------------------------------------------------------------------ building a program
    async def compose(self, ex: dict, also: list[dict] | None = None) -> dict:
        """Everything a function (and any others given) needs, found in what it studied: helpers, imports, constants."""
        parts: dict[str, dict] = {}
        imports: list[str] = []
        constants: dict[str, str] = {}
        defined: set[str] = set()
        unresolved: list[str] = []
        heads: dict = {}
        queue: list[tuple[dict, str, ast.AST]] = []
        for fn in [ex] + list(also or []):
            got = standalone(fn["source"])
            if got is None:
                return {"ok": False, "why": f"{fn['name']} in {fn['path']} does not parse on its own"}
            parts.setdefault(fn["name"], {**fn, "source": got[0]})
            defined.add(fn["name"])
            queue.append((fn, got[0], got[1]))
        while queue:
            fn, _, tree = queue.pop(0)
            for name in sorted(free_names(tree)):
                if name in defined:
                    continue
                head = await self._header(fn["repo"], fn["path"], heads)
                imp = next((i for i in head.get("imports", []) if i["binds"] == name), None)
                if imp is not None and not imp.get("local") and imp["module"].split(".")[0] in STDLIB | {"typing_extensions"}:
                    imports.append(imp["code"])
                    defined.add(name)
                    continue
                if name in head.get("constants", {}):
                    src = head["constants"][name]
                    constants[name] = src
                    defined.add(name)
                    try:
                        queue.append(({**fn}, src, ast.parse(src)))
                    except SyntaxError:
                        pass
                    continue
                helper = self._fn(fn["repo"], name, fn["path"])
                if helper is not None and len(parts) < self.pol.max_helpers + 1 + len(also or []):
                    got = standalone(helper["source"])
                    args = got[1].body[0].args.args if got and got[1].body and isinstance(
                        got[1].body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) else []
                    if got is not None and not (args and args[0].arg in ("self", "cls")):
                        parts[name] = {**helper, "source": got[0]}
                        defined.add(name)
                        queue.append((helper, got[0], got[1]))
                        continue
                if imp is not None and not imp.get("local"):  # a module beyond the standard library: as the file had it
                    imports.append(imp["code"])
                    defined.add(name)
                    continue
                if name in STDLIB:
                    imports.append(f"import {name}")
                    defined.add(name)
                    continue
                if name in COMMON:
                    imports.append(COMMON[name])
                    defined.add(name)
                    continue
                unresolved.append(name)
                defined.add(name)  # (asked once)
        return {"ok": not unresolved, "parts": parts, "imports": list(dict.fromkeys(imports)), "constants": constants,
                "unresolved": unresolved, "why": "" if not unresolved else
                f"it found nothing in {ex['repo']} that defines {', '.join(unresolved[:6])}"}

    def _checks(self, ex: dict, tree: ast.FunctionDef, defined: set[str]) -> list[tuple[str, str, str]]:
        """The examples in its doc comment that call it, as checks: (label, expression, what it should print)."""
        out = []
        try:
            examples = doctest.DocTestParser().get_examples(ast.get_docstring(tree) or "")
        except ValueError:
            return out
        for e in examples:
            src, want = e.source.strip(), e.want.strip()
            if not want or ex["name"] not in src or "\n" in want:
                continue
            try:
                expr = ast.parse(src, mode="eval")
            except SyntaxError:
                continue
            if free_names(expr) - defined:
                continue
            out.append((src[:80], src, want))
        return out[: self.pol.max_checks]

    def _tests(self, ex: dict) -> list[dict]:
        """Tests people wrote that call it (test functions without arguments)."""
        rows = self.db.execute("SELECT id FROM code_examples WHERE repo = ? AND lang = 'python' AND kind = 'test' "
                               "AND calls LIKE ? LIMIT 12", (ex["repo"], f'%"{ex["name"]}"%')).fetchall()
        out = []
        for (tid,) in rows:
            t = self.sys.code_study.example(tid)
            got = standalone(t["source"]) if t else None
            if not got or not got[1].body or not isinstance(got[1].body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fn = got[1].body[0]
            if fn.args.args or fn.args.kwonlyargs or not fn.name.startswith("test") or isinstance(fn, ast.AsyncFunctionDef):
                continue
            out.append(t)
        return out[: self.pol.max_tests]

    def _call(self, fn: ast.FunctionDef, name: str) -> tuple[str, str] | None:
        """A demonstration call with inputs from its arguments' names, and what it takes (or None)."""
        args = fn.args.posonlyargs + fn.args.args
        required = args[: len(args) - len(fn.args.defaults)]
        vals = []
        for k, a in enumerate(required):
            v = sample(a, k)
            if v is None:
                return None
            vals.append(v)
        for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            if d is None:
                v = sample(a, len(vals))
                if v is None:
                    return None
                vals.append(f"{a.arg}={v}")
        call = f"{name}({', '.join(vals)})"
        if isinstance(fn, ast.AsyncFunctionDef):
            call = f"asyncio.run({call})"
        elif any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)):
            call = f"list({call})[:20]"
        return call, (takes_kind(required[0]) if len(required) == 1 else "")

    def program(self, title: str, purpose: str, origin: str, comp: dict, main: list[str], tests: list[dict]) -> str:
        imports = list(comp["imports"])
        if any(line.startswith("_report(\"DEMO\"") and "asyncio.run(" in line for line in main) and "import asyncio" not in imports:
            imports.append("import asyncio")
        head = (f'"""{title}\n\n{textwrap.fill(purpose, 100)}\n\nBuilt by its workbench from {origin}, with no teacher: '
                f"what it calls and imports was found in the code it studied; the checks below are the function's own "
                f'doc examples and the tests people wrote for it."""\n')
        body = [head, "from __future__ import annotations", *imports, ""]
        if comp["constants"]:
            body += list(comp["constants"].values()) + [""]
        body += [p["source"].rstrip() + "\n\n" for p in comp["parts"].values()]  # (the tests it runs are among them)
        body.append(RUNNER)
        body.append('\nif __name__ == "__main__":')
        body += [f"    {line}" for line in main] or ["    pass"]
        return "\n".join(body) + "\n"

    async def build(self, ex: dict) -> dict:
        """A whole program from one function it studied: helpers, imports, its checks and a demonstration."""
        got = standalone(ex["source"])
        rec = {"kind": "whole", "example": ex["id"], "repo": ex["repo"], "path": ex["path"], "name": ex["name"],
               "purpose": ex["purpose"], "code_kind": ex["kind"]}
        if got is None or not got[1].body or not isinstance(got[1].body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {**rec, "status": "skipped", "why": "its code does not parse on its own"}
        fn = got[1].body[0]
        if fn.args.args and fn.args.args[0].arg in ("self", "cls"):
            return {**rec, "status": "skipped", "why": "a method: it belongs to a class"}
        tests = self._tests(ex)
        comp = await self.compose(ex, also=tests)
        if not comp.get("parts"):
            return {**rec, "status": "skipped", "why": comp.get("why", "")}
        defined = set(comp["parts"]) | set(comp["constants"]) | {i.split(" as ")[-1].split()[-1] for i in comp["imports"]}
        checks = self._checks(ex, fn, defined | BUILTINS)
        main = [f"_report(\"CHECK\", {lab!r}, lambda: {src}, {want!r})" for lab, src, want in checks]
        main += [f"_report(\"TEST\", {t['name']!r}, {t['name']})" for t in tests]
        call = self._call(fn, ex["name"])
        if call is not None:
            main.append(f"_report(\"DEMO\", {call[0][:80]!r}, lambda: {call[0]})")
        helpers = [n for n in comp["parts"] if n != ex["name"] and n not in {t["name"] for t in tests}]
        inputs = ", ".join(x for x in (f"{len(checks)} doc example(s)" if checks else "", f"{len(tests)} test(s) people wrote"
                                       if tests else "", "a demonstration from its arguments' names" if call else "") if x)
        title = f"{ex['name']}, as a whole program (from {ex['repo']})"
        code = self.program(title, ex["purpose"], f"{ex['repo']}/{ex['path']} ({ex['name']}) on Gitea", comp, main, tests)
        return {**rec, "code": code, "title": title, "helpers": helpers, "unresolved": comp["unresolved"],
                "inputs": inputs or "nothing it could make up (it needs a file, the network or a function)",
                "checks": len(checks) + len(tests), "takes": call[1] if call else "", "why": comp.get("why", ""),
                "status": "incomplete" if comp["unresolved"] else "built", "runnable": bool(main)}

    async def library(self, examples: list[dict], title: str, purpose: str, origin: str) -> dict:
        """Several functions it built programs from, gathered into one module with all their checks (works.py: a
        long-term project of its own)."""
        fns, tests = [], []
        for ex in examples:
            got = standalone(ex["source"])
            if got is None or not got[1].body or not isinstance(got[1].body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fns.append((ex, got[1].body[0]))
            tests += [t for t in self._tests(ex) if t["name"] not in {x["name"] for x in tests}]
        if not fns:
            return {"ok": False, "why": "none of its parts parse on their own"}
        comp = await self.compose(fns[0][0], also=[ex for ex, _ in fns[1:]] + tests)
        if comp.get("unresolved") or not comp.get("parts"):
            return {"ok": False, "why": comp.get("why", "")}
        defined = set(comp["parts"]) | set(comp["constants"]) | {i.split(" as ")[-1].split()[-1] for i in comp["imports"]}
        main, checks = [], 0
        for ex, fn in fns:
            found = self._checks(ex, fn, defined | BUILTINS)
            checks += len(found)
            main += [f"_report(\"CHECK\", {lab!r}, lambda: {src}, {want!r})" for lab, src, want in found]
            call = self._call(fn, ex["name"])
            if call is not None:
                main.append(f"_report(\"DEMO\", {call[0][:80]!r}, lambda: {call[0]})")
        main += [f"_report(\"TEST\", {t['name']!r}, {t['name']})" for t in tests]
        code = self.program(title, purpose, origin, comp, main, tests)
        return {"ok": True, "code": code, "parts": [ex["name"] for ex, _ in fns], "checks": checks + len(tests),
                "helpers": [n for n in comp["parts"] if n not in {ex["name"] for ex, _ in fns} | {t["name"] for t in tests}]}

    async def chain(self) -> dict | None:
        """Two functions that ran, one fed into the other: a program that sees what comes out."""
        rows = self.db.execute("SELECT example, repo, name, returns, takes FROM code_programs WHERE kind = 'whole' AND "
                               "status IN ('ran', 'passed') AND takes != '' AND returns != ''").fetchall()
        tried = {(r[0], r[1]) for r in self.db.execute("SELECT example, other FROM code_programs WHERE kind = 'chain'").fetchall()}
        pairs = [(a, b) for a in rows for b in rows if a[0] != b[0] and a[1] == b[1] and a[3] == b[4] and (a[0], b[0]) not in tried]
        if not pairs:
            return None
        a, b = self.rng.choice(pairs)
        ea, eb = self.sys.code_study.example(a[0]), self.sys.code_study.example(b[0])
        if ea is None or eb is None:
            return None
        fa = standalone(ea["source"])[1].body[0]
        call = self._call(fa, ea["name"])
        if call is None:
            return None
        comp = await self.compose(ea, also=[eb])
        rec = {"kind": "chain", "example": ea["id"], "other": eb["id"], "repo": ea["repo"], "path": ea["path"],
               "name": f"{ea['name']} -> {eb['name']}", "purpose": f"{ea['purpose']}; then {eb['purpose']}",
               "code_kind": eb["kind"], "takes": a[4], "checks": 0}
        if comp["unresolved"]:
            return {**rec, "status": "incomplete", "unresolved": comp["unresolved"], "why": comp["why"], "helpers": []}
        inner = call[0]
        main = [f"_report(\"DEMO\", {inner[:60]!r}, lambda: {inner})",
                f"_report(\"DEMO\", {(eb['name'] + '(' + inner + ')')[:80]!r}, lambda: {eb['name']}({inner}))"]
        title = f"{ea['name']} then {eb['name']} (from {ea['repo']})"
        code = self.program(title, rec["purpose"], f"{ea['repo']}: {ea['name']} ({ea['path']}) fed into {eb['name']} ({eb['path']})",
                            comp, main, [])
        return {**rec, "code": code, "title": title, "helpers": [n for n in comp["parts"] if n not in (ea["name"], eb["name"])],
                "unresolved": [], "inputs": "what the first returns", "status": "built", "runnable": True}

    # ------------------------------------------------------------------ an experiment
    async def experiment(self, example: int | None = None, chain: bool | None = None, by: str = BY) -> dict | None:
        """Build a program, check it, put it in the scratch pad, and run it if P20 lets it run by itself."""
        s = self.sys
        if not self.enabled or self.busy:
            return None
        self.busy = True
        try:
            rec = None
            if chain or (chain is None and example is None and self.rng.random() < self.pol.chain_share):
                rec = await self.chain()
            if rec is None and not chain:
                ex = s.code_study.example(int(example)) if example else self.next_example()
                if ex is None or ex["lang"] != "python":
                    return None
                with s.activity.track("workbench", "coder", f"building a program from {ex['repo']}: {ex['name']}"):
                    rec = await self.build(ex)
            if rec is None:
                return None
            item = None
            if rec.get("code") and rec["status"] == "built":
                check = s.scratch.validator.check(rec["code"], "python")
                rec["status"] = "checks out" if check["ok"] else "does not check out"
                rec["touches"] = check.get("touches", [])
                if not check["ok"]:
                    rec["why"] = "; ".join(check.get("problems", [])[:2])
                else:
                    self._make_room()
                    item = s.scratch._new(rec["title"], rec["code"], "python", by, request=rec["purpose"],
                                          origin={"how": "workbench", "kind": rec["kind"], "repo": rec["repo"],
                                                  "path": rec["path"], "name": rec["name"], "helpers": rec.get("helpers", []),
                                                  "inputs": rec.get("inputs", ""), "example": rec["example"]},
                                          kind="experiment")
                    s.scratch._save(item)
                    rec["item"] = item["id"]
            pid = self._record(rec)
            rec["id"] = pid
            self.stats["experiments"] += 1
            self.stats[rec["status"]] += 1
            line = self._line(rec)
            if item is not None and rec.get("runnable") and s.scratch.auto_runs(item)[0]:
                run = await s.scratch.auto_run(item)  # a person switched auto-run on: it runs now (P20)
                if run:
                    rec.update(self._ran(pid, item["id"], run))
                    line = self._line(rec)
            elif item is not None:
                why = s.scratch.auto_runs(item)[1]
                line += f"; it waits in the scratch pad as {item['id']} ({why})"
            self._note(line, program=pid, item=rec.get("item"))
            s.bus.publish("workbench.experiment", id=pid, kind=rec["kind"], name=rec["name"], repo=rec["repo"],
                          status=rec["status"], item=rec.get("item"), by=by)
            rec["line"] = line
            return rec
        finally:
            self.busy = False

    def _line(self, rec: dict) -> str:
        what = f"built {rec['name']} ({rec['repo']}) into a whole program" if rec["kind"] == "whole" else \
            f"fed {rec['name'].replace(' -> ', ' into ')} ({rec['repo']})"
        if rec["status"] == "skipped":
            return f"skipped {rec['name']} ({rec['repo']}): {rec.get('why', '')}"
        if rec["status"] == "incomplete":
            return f"{what}, but it is incomplete: {rec.get('why', '')}"
        tail = {"checks out": "it checks out", "does not check out": f"it does not check out ({rec.get('why', '')})",
                "passed": f"it ran and passed {rec.get('passed', 0)} check(s)", "ran": "it ran",
                "failed": f"it ran: {rec.get('failed', 0)} check(s) failed" + (f", {rec.get('errors', 0)} error(s)" if rec.get("errors") else "")}
        helpers = rec.get("helpers") or []
        return (f"{what}" + (f" with {len(helpers)} helper(s) it found" if helpers else "") +
                (f", {rec['checks']} check(s)" if rec.get("checks") else "") + f": {tail.get(rec['status'], rec['status'])}")

    def _make_room(self) -> None:
        """Programs waiting in the scratch pad to be run: only the newest [workbench] max_waiting stay there."""
        pad = self.sys.scratch
        waiting = sorted((i for i in pad.items.values() if i.get("kind") == "experiment" and not i.get("runs")
                          and i.get("status") not in ("approved", "running", "awaiting approval")),
                         key=lambda i: i.get("created", 0))
        while len(waiting) >= self.pol.max_waiting:
            old = waiting.pop(0)
            pad.remove(old["id"], BY)
            self.db.execute("UPDATE code_programs SET item = '' WHERE item = ?", (old["id"],))

    def _record(self, rec: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO code_programs(kind, example, other, repo, path, name, purpose, code_kind, ts, item, status, helpers, "
            "unresolved, inputs, checks, takes, code, why) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec["kind"], rec["example"], rec.get("other"), rec["repo"], rec["path"], rec["name"], rec["purpose"][:300],
             rec.get("code_kind", ""), time.time(), rec.get("item", ""), rec["status"], json.dumps(rec.get("helpers", [])),
             json.dumps(rec.get("unresolved", [])), rec.get("inputs", ""), rec.get("checks", 0), rec.get("takes", ""),
             rec.get("code", "")[:40000], rec.get("why", "")[:400]))
        n = self.db.execute("SELECT COUNT(*) FROM code_programs").fetchone()[0]
        if n > self.pol.keep:
            self.db.execute("DELETE FROM code_programs WHERE id IN (SELECT id FROM code_programs ORDER BY id LIMIT ?)",
                            (n - self.pol.keep,))
        return int(cur.lastrowid)

    def _ran(self, pid: int, item_id: str, run: dict) -> dict:
        """What a run of a program it built showed: it learns from it."""
        out = outcome(run.get("stdout", ""))
        ok = bool(run.get("ok"))
        status = ("failed" if not ok or out["failed"] or (out["errors"] and not out["passed"]) else
                  "passed" if out["passed"] else "ran")
        self.db.execute("UPDATE code_programs SET status = ?, passed = ?, failed = ?, returns = ?, output = ? WHERE id = ?",
                        (status, out["passed"], out["failed"] + out["errors"], out["returns"],
                         ((run.get("stdout") or "") + (run.get("stderr") or ""))[:4000], pid))
        self.stats[f"runs {status}"] += 1
        row = self.db.execute("SELECT purpose, code_kind, kind FROM code_programs WHERE id = ?", (pid,)).fetchone()
        if status in ("passed", "ran") and row and row[2] == "whole" and CODEKIND in self.sys.capabilities and row[1]:
            if self.sys.learn(CODEKIND, row[0], row[1], "experiment"):  # what ran is what that kind of code is
                self.stats["lessons"] += 1
        return {"status": status, "passed": out["passed"], "failed": out["failed"], "errors": out["errors"],
                "returns": out["returns"]}

    def on_event(self, ev) -> None:
        """A person ran a program it built (or auto-run did): learn from what it printed."""
        if ev.topic != "code.ran":
            return
        row = self.db.execute("SELECT id, status FROM code_programs WHERE item = ? ORDER BY id DESC LIMIT 1",
                              (str(ev.data.get("id", "")),)).fetchone()
        item = self.sys.scratch.items.get(str(ev.data.get("id", "")))
        if row is None or item is None or not item.get("runs") or str(ev.data.get("by", "")).startswith("auto-run"):
            return  # (a run auto-run started is learned from by experiment(), which started it)
        res = self._ran(row[0], item["id"], item["runs"][-1])
        self._note(f"program {row[0]} ({item['id']}) ran: {res['status']}" + (f", {res['passed']} check(s) passed" if res["passed"] else "")
                   + (f", {res['failed']} failed" if res["failed"] else ""), program=row[0], item=item["id"])

    # ------------------------------------------------------------------ what it built, for the scratch pad
    def program_for(self, example_ids: list[int]) -> dict | None:
        """The best program it built from one of these examples (passed, ran, or checked), closest example first."""
        best = None
        for rank, eid in enumerate(example_ids):
            for status_rank, st in enumerate(GOOD):
                row = self.db.execute("SELECT id, code, status, name, repo, path FROM code_programs WHERE kind = 'whole' AND "
                                      "example = ? AND status = ? AND code != '' ORDER BY id DESC LIMIT 1", (eid, st)).fetchone()
                if row and (best is None or (status_rank, rank) < best[0]):
                    best = ((status_rank, rank), {"id": row[0], "code": row[1], "status": row[2], "name": row[3],
                                                  "repo": row[4], "path": row[5], "example": eid})
        return best[1] if best else None

    def programs(self, limit: int = 40, status: str = "") -> list[dict]:
        sql = ("SELECT id, kind, example, other, repo, path, name, purpose, ts, item, status, helpers, unresolved, inputs, checks, "
               "passed, failed, returns, takes, why FROM code_programs" + (" WHERE status = ?" if status else "") + " ORDER BY id DESC LIMIT ?")
        rows = self.db.execute(sql, ((status,) if status else ()) + (max(1, min(500, limit)),)).fetchall()
        keys = ("id", "kind", "example", "other", "repo", "path", "name", "purpose", "ts", "item", "status", "helpers",
                "unresolved", "inputs", "checks", "passed", "failed", "returns", "takes", "why")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["helpers"], d["unresolved"] = json.loads(d["helpers"] or "[]"), json.loads(d["unresolved"] or "[]")
            out.append(d)
        return out

    def get(self, pid: int) -> dict | None:
        row = self.db.execute("SELECT code, output FROM code_programs WHERE id = ?", (int(pid),)).fetchone()
        if not row:
            return None
        rec = next((p for p in self.programs(500) if p["id"] == int(pid)), None) or {"id": int(pid)}
        return {**rec, "code": row[0], "output": row[1]}

    def status(self) -> dict:
        counts = dict(self.db.execute("SELECT status, COUNT(*) FROM code_programs GROUP BY status").fetchall())
        left = self.db.execute("SELECT COUNT(*) FROM code_examples e WHERE e.lang = 'python' AND e.kind != 'test' AND "
                               "NOT EXISTS (SELECT 1 FROM code_programs p WHERE p.example = e.id AND p.kind = 'whole')").fetchone()[0]
        return {"enabled": self.enabled, "busy": self.busy, "every": self.pol.every_seconds, "counts": counts, "left": left,
                "auto_run": self.sys.scratch.mode("auto_run"), "programs": self.programs(30), "stats": dict(self.stats),
                "log": list(self.log)[-12:][::-1]}

    def lines(self) -> list[str]:
        st = self.status()
        c = st["counts"]
        out = [f"Workbench: {sum(c.values())} program(s) built from what it studied, {st['left']} function(s) to go; "
               + ", ".join(f"{n} {k}" for k, n in sorted(c.items(), key=lambda kv: -kv[1])) if c else
               f"Workbench: nothing built yet ({st['left']} function(s) it could build from)"]
        out.append(f"  auto-run is {st['auto_run']}" + ("" if st["auto_run"] != "ask" else
                                                         ": its programs wait for you to run them (/code autorun harmless lets "
                                                         "the ones that reach for nothing run by themselves)"))
        for p in st["programs"][:8]:
            out.append(f"  #{p['id']:<4} {p['status']:<18} {p['name'][:40]:<40} {p['repo']}" + (f" -> {p['item']}" if p["item"] else ""))
        out += [f"  - {x['line']}" for x in st["log"][:4]]
        return out


class Tinkerer(Agent):
    """When idle: an experiment on its workbench, every [workbench] every_seconds."""
    name = "tinkerer"

    def __init__(self, system):
        super().__init__(system)
        self.last = time.time() - system.directives.workbench.every_seconds + 45  # the first, a little after start

    @property
    def interval(self) -> float:
        return 5.0

    async def step(self) -> None:
        s = self.sys
        w = s.workbench
        if not w.enabled or not s.guardian.background_allowed() or s.inflight:
            return
        if time.time() - (s.last_order_at or s.started_at) < w.pol.idle_seconds:
            return
        if time.time() - self.last < w.pol.every_seconds:
            return
        self.last = time.time()
        await s.background_slot()
        res = await w.experiment()
        self.stats[res["status"] if res else "nothing to build"] += 1
