"""Gitea: its coder nodes learn to program from a local Gitea.

    a visit (when idle, every [gitea] study_every_seconds)
      --> pick a repository that changed since it last read it (Gitea says when each was updated), else one it has
          never studied, else one it has not studied for [gitea] restudy_days
      --> read its file tree (Gitea's API) and the source files that changed since it last read them
      --> take every function apart: its name, arguments, doc comment, the code, what it calls, and what kind
          of code it is (text, maths, parse, data, files, network, dates, test, cli, other - from what the
          code actually does: CodeTeacher)
      --> learn: its code-kind node (capability "codekind") learns what kind of code a purpose needs, from
          the purpose people wrote down (the name and doc comment); its coder node (which recipe from its
          code library a request needs) learns new wordings where a function's purpose matches a recipe
      --> keep the function as an example: the scratch pad (scratchpad.py) gives the closest examples to a
          teacher writing code, or adapts the closest one, and practice compares its own code with them; and what
          a Python file imports and defines at the top (its header) is kept, so its workbench (workbench.py) can build
          whole programs from the functions it studied, with no teacher

It only reads: it never writes to Gitea, and code from Gitea is never run (only code a person approves in
the scratch pad runs, P20).

The connection is a plugin, like Kiwix: plugins/gitea (a vetted, persistent tool that needs a person's approval
for the network, and is given $GITEA_TOKEN and nothing else). Its url and repositories are set in directives.toml
[tools.config.gitea], or with --gitea URL; the tool switch applies to it like any tool.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import random
import re
import time
from collections import Counter, deque

import numpy as np

from .agents import Agent
from .teachers import Lesson, Teacher

log = logging.getLogger("neuromesh.gitea")

CODEKIND = "codekind"
TOOL = "gitea"  # the plugin it reads Gitea through (plugins/gitea)
KINDS = ("text", "maths", "parse", "data", "files", "network", "dates", "test", "cli", "other")
LANGS = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust",
         ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".sh": "shell", ".rb": "ruby"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS code_repos(name TEXT PRIMARY KEY, branch TEXT, commit_sha TEXT, studied REAL, files INTEGER,
                                      functions INTEGER, description TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS code_files(repo TEXT, path TEXT, sha TEXT, lang TEXT, size INTEGER, studied REAL,
                                      functions INTEGER, PRIMARY KEY(repo, path));
CREATE TABLE IF NOT EXISTS code_examples(id INTEGER PRIMARY KEY, repo TEXT, path TEXT, sha TEXT, lang TEXT, name TEXT,
                                         kind TEXT, purpose TEXT, signature TEXT, doc TEXT, source TEXT, lines INTEGER,
                                         calls TEXT, ts REAL, practised INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS code_examples_kind ON code_examples(kind);
CREATE INDEX IF NOT EXISTS code_examples_name ON code_examples(repo, name);
CREATE INDEX IF NOT EXISTS code_examples_file ON code_examples(repo, path);
"""


# ============================================================================ reading code
def words_of(name: str) -> str:
    """snake_case and camelCase names as words: "parseHTTPDate" -> "parse http date"."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return " ".join(w for w in re.split(r"[\W_]+", s.lower()) if w)


def purpose_of(name: str, doc: str, args: list[str]) -> str:
    """What a function is for, in words: its name, the first sentence of its doc comment, its arguments."""
    first = re.split(r"(?<=[.!?])\s|\n\s*\n", " ".join(str(doc or "").split()), maxsplit=1)[0][:200].rstrip(" .")
    parts = [words_of(name)]
    if first:
        parts.append(first)
    if args:
        parts.append("given " + ", ".join(words_of(a) for a in args[:5] if a not in ("self", "cls")))
    return ". ".join(p for p in parts if p.strip()).strip()


_PY_MODULE_KIND = {"os": "files", "pathlib": "files", "shutil": "files", "glob": "files", "io": "files",
                   "csv": "files", "urllib": "network", "requests": "network", "http": "network", "socket": "network",
                   "httpx": "network", "aiohttp": "network", "json": "parse", "re": "parse", "xml": "parse",
                   "html": "parse", "datetime": "dates", "time": "dates", "calendar": "dates", "math": "maths",
                   "statistics": "maths", "decimal": "maths", "fractions": "maths", "numpy": "maths",
                   "argparse": "cli", "sys": "cli", "click": "cli", "unittest": "test", "pytest": "test",
                   "collections": "data", "itertools": "data", "heapq": "data", "bisect": "data"}
_TEXT_METHODS = {"upper", "lower", "title", "capitalize", "strip", "lstrip", "rstrip", "replace", "join",
                 "startswith", "endswith", "format", "center", "ljust", "rjust", "zfill", "casefold", "swapcase"}
_FILE_CALLS = {"open", "read_text", "write_text", "read_bytes", "write_bytes", "listdir", "walk", "makedirs", "mkdir",
               "exists", "isfile", "isdir", "unlink", "remove", "glob", "readFile", "writeFile", "readFileSync",
               "writeFileSync", "ReadFile", "WriteFile", "Open", "Create"}
_NET_CALLS = {"urlopen", "get", "post", "request", "connect", "fetch", "Dial", "Get", "Post", "socket", "send", "recv"}
_PARSE_CALLS = {"loads", "load", "parse", "match", "search", "findall", "fullmatch", "compile", "sub", "split",
                "parseInt", "parseFloat", "JSON.parse", "Unmarshal", "Atoi", "ParseFloat", "from_str"}
_DATE_CALLS = {"now", "today", "strftime", "strptime", "fromisoformat", "timedelta", "Date", "Since", "Parse"}
_MATH_CALLS = {"sqrt", "pow", "log", "exp", "sin", "cos", "tan", "floor", "ceil", "round", "abs", "sum", "max", "min",
               "mean", "median", "hypot", "gcd", "factorial", "Math.sqrt", "Abs", "Sqrt"}
_DATA_CALLS = {"sorted", "sort", "append", "extend", "pop", "insert", "items", "keys", "values", "filter", "map",
               "reduce", "zip", "enumerate", "reversed", "Counter", "defaultdict", "deque", "push", "slice", "splice"}


def kind_of(fn: dict) -> str:
    """What kind of code a function is, from what it actually does (its calls, imports and name). The
    code-kind node learns to tell this from the purpose alone."""
    name = str(fn.get("name", ""))
    calls = set(fn.get("calls") or [])
    mods = set(fn.get("modules") or [])
    if name.startswith(("test_", "Test", "test")) or fn.get("asserts", 0) >= 2:
        return "test"
    votes: Counter = Counter()
    for c in calls:  # the modules it actually uses (a file's imports serve many functions)
        prefix = c.split(".")[0]
        if "." in c and prefix in mods | set(_PY_MODULE_KIND) and prefix in _PY_MODULE_KIND:
            votes[_PY_MODULE_KIND[prefix]] += 1
    for c in calls:
        short = c.split(".")[-1]
        if c in _FILE_CALLS or short in _FILE_CALLS:
            votes["files"] += 2
        if c in _NET_CALLS or short in ("urlopen", "fetch", "Dial", "socket") or c.startswith(("requests.", "http.", "urllib")):
            votes["network"] += 2 if short in ("urlopen", "fetch", "Dial", "socket") or "." in c else 1
        if c in _PARSE_CALLS or short in _PARSE_CALLS:
            votes["parse"] += 1
        if c in _DATE_CALLS or short in _DATE_CALLS or c.startswith(("datetime", "time.")):
            votes["dates"] += 1
        if c in _MATH_CALLS or short in _MATH_CALLS or c.startswith("math."):
            votes["maths"] += 1
        if short in _TEXT_METHODS:
            votes["text"] += 1
        if c in _DATA_CALLS or short in _DATA_CALLS:
            votes["data"] += 1
        if short in ("add_argument", "parse_args", "argv", "exit", "Args", "Flag"):
            votes["cli"] += 2
    words = set(words_of(name).split())
    for w, k in (("parse", "parse"), ("read", "files"), ("write", "files"), ("load", "files"), ("save", "files"),
                 ("fetch", "network"), ("download", "network"), ("url", "network"), ("http", "network"),
                 ("date", "dates"), ("time", "dates"), ("day", "dates"), ("sum", "maths"), ("mean", "maths"),
                 ("average", "maths"), ("count", "data"), ("sort", "data"), ("format", "text"), ("slug", "text"),
                 ("reverse", "text"), ("upper", "text"), ("lower", "text"), ("slugify", "text"), ("capitalize", "text"),
                 ("main", "cli")):
        if w in words:
            votes[k] += 2  # what its author called it
    if fn.get("arith", 0) >= 3:
        votes["maths"] += 1
    if not votes:
        return "other"
    best = max(votes.values())
    order = ("test", "network", "files", "cli", "dates", "parse", "maths", "text", "data")
    return next(k for k in order if votes.get(k) == best)


def _call_name(node) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        base = f.value
        if isinstance(base, ast.Name):
            return f"{base.id}.{f.attr}"
        return f.attr
    return ""


def python_functions(source: str) -> list[dict]:
    """Every function in a Python file (top-level and methods): name, arguments, doc, code, calls."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        code = ast.get_source_segment(source, node) or ""
        lines = code.count("\n") + 1
        if lines < 2 or lines > 150 or len(node.body) < 1:
            continue
        calls = sorted({_call_name(n) for n in ast.walk(node) if isinstance(n, ast.Call)} - {""})
        args = [a.arg for a in node.args.args + node.args.kwonlyargs]
        arith = sum(1 for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)))
        asserts = sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert)) + sum(1 for c in calls if c.split(".")[-1].startswith("assert"))
        out.append({"name": node.name, "args": args, "doc": ast.get_docstring(node) or "", "source": code, "lines": lines,
                    "calls": calls, "modules": sorted(modules), "arith": arith, "asserts": asserts,
                    "signature": f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({', '.join(args)})"})
    return out


_HEADERS = {
    "javascript": re.compile(r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\(([^)]*)\)\s*\{|"
                             r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function\s*)?\(([^)]*)\)\s*(?:=>)?\s*\{", re.M),
    "go": re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(([^)]*)\)[^{\n]*\{", re.M),
    "rust": re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)[^{;\n]*\{", re.M),
    "java": re.compile(r"^[ \t]*(?:public|private|protected|static|final|\s)*[\w<>\[\],\s]+\s+(\w+)\s*\(([^)]*)\)\s*(?:throws [\w., ]+)?\s*\{", re.M),
    "c": re.compile(r"^[\w\s\*]+?\b(\w+)\s*\(([^)]*)\)\s*\{", re.M),
    "shell": re.compile(r"^[ \t]*(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", re.M),
    "ruby": None,
}
_HEADERS["typescript"] = _HEADERS["javascript"]
_HEADERS["cpp"] = _HEADERS["c"]


def brace_functions(source: str, lang: str) -> list[dict]:
    """Functions in a brace language, found by their headers and matched braces; the comment above is the doc."""
    pat = _HEADERS.get(lang)
    if pat is None:
        return []
    out = []
    for m in pat.finditer(source):
        groups = [g for g in m.groups()]
        name = next((g for g in groups[0::2] if g), None)
        args_raw = next((groups[i + 1] for i in range(0, len(groups), 2) if groups[i]), "") or ""
        if not name or name in ("if", "for", "while", "switch", "catch", "return", "else", "do"):
            continue
        start, depth, i = m.start(), 0, m.end() - 1
        end = None
        while i < len(source) and i - start < 12000:
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end is None:
            continue
        code = source[start:end].strip("\n")
        lines = code.count("\n") + 1
        if lines < 2 or lines > 150:
            continue
        above = source[:start].rstrip("\n").split("\n")[-6:]
        doc_lines = []
        for line in reversed(above):
            t = line.strip()
            if t.startswith(("//", "#", "*", "/*", "///")):
                doc_lines.insert(0, re.sub(r"^(/\*\*?|\*/|\*|///?|#)\s?", "", t).strip(" */"))
            else:
                break
        args = [a.strip().split()[-1].strip("*&") if a.strip() else "" for a in args_raw.split(",")]
        args = [re.sub(r"[^\w]", "", a) for a in args if a and re.sub(r"[^\w]", "", a)]
        calls = sorted(set(re.findall(r"\b([A-Za-z_][\w.]*)\s*\(", code[len(m.group(0)):])) - {"if", "for", "while", "switch", "return", "catch", "function"})
        arith = len(re.findall(r"[^=!<>+\-*/%]\s*[+\-*/%]\s*[\w(]", code))
        out.append({"name": name, "args": args, "doc": " ".join(x for x in doc_lines if x), "source": code, "lines": lines,
                    "calls": calls[:60], "modules": [], "arith": arith,
                    "asserts": len(re.findall(r"\bassert\w*\s*\(|\bexpect\s*\(", code)),
                    "signature": m.group(0).strip().rstrip("{").strip()})
    return out


def python_header(source: str) -> dict:
    """What a Python file sets up at the top: its imports (each with the names it binds) and its constants (simple
    assignments a function may use), so a function taken from it can be put back into a whole program."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    imports, constants = [], {}
    for node in tree.body:
        seg = ast.get_source_segment(source, node) or ""
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append({"code": f"import {a.name}" + (f" as {a.asname}" if a.asname else ""),
                                "binds": a.asname or a.name.split(".")[0], "module": a.name})
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                if a.name != "*":
                    imports.append({"code": f"from {node.module} import {a.name}" + (f" as {a.asname}" if a.asname else ""),
                                    "binds": a.asname or a.name, "module": node.module})
        elif isinstance(node, ast.ImportFrom) and node.level:  # from .utils import helper: the helper is in the repo
            for a in node.names:
                imports.append({"code": "", "binds": a.asname or a.name, "module": "." + (node.module or ""), "local": True})
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and len(seg) <= 600:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if names and node.value is not None and not any(isinstance(n, (ast.Lambda, ast.Await, ast.Yield)) for n in ast.walk(node.value)):
                for nm in names:
                    constants[nm] = seg
    return {"imports": imports[:80], "constants": dict(list(constants.items())[:80])}


def functions_in(source: str, lang: str) -> list[dict]:
    return python_functions(source) if lang == "python" else brace_functions(source, lang)


# ============================================================================ the teacher
class CodeTeacher(Teacher):
    """What people's code shows: the kind of code each purpose needed. A specialist, for the code-kind node."""
    name = "gitea"
    specialist = True

    def __init__(self, study: "CodeStudy | None" = None, seed: int | None = None):
        self.study = study
        self.rng = random.Random(seed)

    def available(self) -> bool:
        return self.study is not None and self.study.count() > 0

    def status(self) -> dict:
        n = self.study.count() if self.study is not None else 0
        return {"detail": f"specialist: teaches the code-kind node from {n} functions people wrote (Gitea)"}

    async def label(self, spec, text):
        if spec.name != CODEKIND or self.study is None:
            return None
        kind = self.study.kind_for(text)
        return Lesson(kind, 0.85, self.name, "what that code does") if kind else None

    async def generate(self, spec, label, n):
        if spec.name != CODEKIND or self.study is None:
            return []
        return self.study.purposes(label, n, self.rng)


# ============================================================================ studying
class CodeStudy:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.gitea
        self.db = system.store.db
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(code_files)").fetchall()}
        if "header" not in cols:  # (earlier versions did not keep a file's imports and constants)
            self.db.execute("ALTER TABLE code_files ADD COLUMN header TEXT DEFAULT ''")
        raw = system.store.get_kv("gitea.state")
        st = json.loads(raw) if raw else {}
        self.stats: Counter = Counter(st.get("stats", {}))
        self.log: deque = deque(st.get("log", []), maxlen=60)
        self.backlog: dict[str, int] = st.get("backlog", {})  # repositories it has not finished reading: files left
        self.version = ""
        self.reached: str | None = None  # what the last call to the tool said, if it failed
        self._vectors: tuple[int, np.ndarray, list[int]] | None = None
        self.busy = False

    # ------------------------------------------------------------------ the gitea tool (plugins/gitea)
    def tool(self):
        return self.sys.tools.tools.get(TOOL)

    @property
    def url(self) -> str:
        t = self.tool()
        return str(self.sys.tool_config(t).get("url", "")) if t is not None else ""

    @property
    def enabled(self) -> bool:
        """It studies when [gitea] enabled and the gitea tool is active and not switched off."""
        t = self.tool()
        return bool(self.pol.enabled and t is not None and t.state == "active" and not self.sys.tool_switch.refuses(t))

    @property
    def state(self) -> str:
        t = self.tool()
        if not self.pol.enabled:
            return "off"
        if t is None:
            return "no tool"
        if self.sys.tool_switch.refuses(t):
            return "switched off"
        if t.state != "active":
            return t.state
        return "unreachable" if self.reached else "ready" if self.version else "unknown"

    @property
    def detail(self) -> str:
        t = self.tool()
        st = self.state
        if st == "off":
            return "studying Gitea is off ([gitea] enabled)"
        if st == "no tool":
            return "no gitea tool (plugins/gitea)"
        if st == "switched off":
            return "the gitea tool is switched off (the tool switch)"
        if st in ("pending_approval", "discovered"):
            return f"the gitea tool waits for your approval (it reads {self.url} over the network, with $GITEA_TOKEN)"
        if st == "approved":
            return "the gitea tool is approved; its self-test runs next"
        if st in ("quarantined", "rejected"):
            return f"the gitea tool is {st}: {t.note}" + (f" ({t.last_error[:160]})" if t.last_error else "")
        if st == "unreachable":
            return f"no Gitea at {self.url} ({self.reached})"
        if st == "ready":
            return f"Gitea {self.version} at {self.url}"
        return f"the gitea tool is ready for {self.url}; not visited yet"

    def set_url(self, url: str, by: str = "console") -> str:
        """Point the gitea tool at another Gitea (for this run; [tools.config.gitea] url keeps it)."""
        s = self.sys
        s.directives.tools.config.setdefault(TOOL, {})["url"] = url.strip()
        self.version, self.reached = "", None
        t = self.tool()
        if t is not None and t.state == "quarantined":  # it may have failed on the old address: test it again
            t.state, t.note = "approved", f"its url changed to {url.strip()}: testing it again"
            s.tools.persist(t)
        s.guardian.audit(by, "gitea.url", TOOL, url=url.strip())
        return t.state if t is not None else "no tool"

    async def call(self, request: dict | str) -> tuple[dict | None, str]:
        """One request to the gitea tool, in its sandbox: (its answer, or None and why)."""
        s = self.sys
        t = self.tool()
        if t is None:
            return None, "no gitea tool"
        if t.state != "active":
            return None, self.detail
        decision = s.guardian.authorize_tool_run(t)
        if not decision.allowed:
            return None, decision.reason
        run = await s.sandbox.run(t, request if isinstance(request, str) else json.dumps(request),
                                  t.manifest.timeout_s or s.directives.tools.default_timeout_s, s.tool_config(t))
        s.record_tool_run(t, run)
        if run.refused:
            return None, run.error
        if not run.ok:
            if run.transient:
                self.reached = run.error[:200]
            return None, run.error
        self.reached = None
        return {"output": run.output, **(run.extra or {})}, ""

    def save(self) -> None:
        self.sys.store.set_kv("gitea.state", json.dumps({"stats": dict(self.stats), "log": list(self.log),
                                                         "backlog": self.backlog}))

    def _note(self, line: str, **extra) -> None:
        self.log.append({"ts": round(time.time(), 1), "line": line[:300], **extra})

    # ------------------------------------------------------------------ the store
    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM code_examples").fetchone()[0])

    def kind_for(self, purpose: str) -> str | None:
        row = self.db.execute("SELECT kind FROM code_examples WHERE purpose = ? LIMIT 1", (purpose,)).fetchone()
        return row[0] if row else None

    def purposes(self, kind: str, n: int, rng) -> list[str]:
        rows = self.db.execute("SELECT purpose FROM code_examples WHERE kind = ? ORDER BY RANDOM() LIMIT ?",
                               (kind, max(1, n))).fetchall()
        return [r[0] for r in rows if r[0]]

    def example(self, ex_id: int) -> dict | None:
        row = self.db.execute("SELECT id, repo, path, lang, name, kind, purpose, signature, doc, source, lines, calls, "
                              "practised FROM code_examples WHERE id = ?", (int(ex_id),)).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(r) -> dict:
        return {"id": r[0], "repo": r[1], "path": r[2], "lang": r[3], "name": r[4], "kind": r[5], "purpose": r[6],
                "signature": r[7], "doc": r[8], "source": r[9], "lines": r[10], "calls": json.loads(r[11] or "[]"),
                "practised": r[12]}

    def examples(self, limit: int = 50, kind: str | None = None, lang: str | None = None, q: str = "") -> list[dict]:
        where, args = [], []
        if kind:
            where.append("kind = ?"); args.append(kind)
        if lang:
            where.append("lang = ?"); args.append(lang)
        if q:
            where.append("(purpose LIKE ? OR name LIKE ? OR repo LIKE ?)"); args += [f"%{q}%"] * 3
        sql = ("SELECT id, repo, path, lang, name, kind, purpose, signature, doc, source, lines, calls, practised FROM "
               f"code_examples {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY id DESC LIMIT ?")
        return [self._row(r) for r in self.db.execute(sql, (*args, max(1, min(500, limit)))).fetchall()]

    def similar(self, text: str, k: int = 3, lang: str | None = None, kind: str | None = None,
                exclude: int | None = None) -> list[tuple[dict, float]]:
        """The examples whose purpose is closest to a request (a kind the code-kind node expects counts extra)."""
        n = self.count()
        if n == 0:
            return []
        if self._vectors is None or self._vectors[0] != n:
            rows = self.db.execute("SELECT id, purpose FROM code_examples ORDER BY id DESC LIMIT ?",
                                   (self.pol.max_examples,)).fetchall()
            ids = [r[0] for r in rows]
            M = self.sys.encode_many([r[1] or "" for r in rows]) if rows else np.zeros((0, 1), dtype=np.float32)
            self._vectors = (n, M, ids)
        _, M, ids = self._vectors
        if not ids:
            return []
        sims = M @ self.sys.encode(text)
        order = np.argsort(-sims)[: max(k * 12, 30)]
        out = []
        for i in order:
            ex = self.example(ids[int(i)])
            if ex is None or ex["id"] == exclude or (lang and ex["lang"] != lang):
                continue
            score = float(sims[int(i)]) + (0.08 if kind and ex["kind"] == kind else 0.0)
            out.append((ex, score))
        out.sort(key=lambda t: -t[1])
        return out[:k]

    # ------------------------------------------------------------------ a visit
    @staticmethod
    def _updated(repo: dict) -> float:
        """When Gitea says a repository last changed (0 if it doesn't say)."""
        raw = str(repo.get("updated_at") or "")
        try:
            from datetime import datetime
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() if raw else 0.0
        except ValueError:
            return 0.0

    def _pick(self, repos: list[dict]) -> dict | None:
        studied = {r[0]: r[1] or 0.0 for r in self.db.execute("SELECT name, studied FROM code_repos").fetchall()}
        stale = time.time() - self.pol.restudy_days * 86400
        unfinished = [r for r in repos if self.backlog.get(r["full_name"])]
        if unfinished:  # a repository it has not finished reading comes first
            return unfinished[0]
        changed = sorted((r for r in repos if r["full_name"] in studied and self._updated(r) > studied[r["full_name"]]),
                         key=lambda r: studied[r["full_name"]])
        if changed:  # something was pushed since it last read it: only the files that changed are read again
            return changed[0]
        fresh = [r for r in repos if r["full_name"] not in studied]
        if fresh:
            return fresh[0]
        old = sorted((r for r in repos if studied.get(r["full_name"], 0) < stale), key=lambda r: studied.get(r["full_name"], 0))
        return old[0] if old else None

    async def visit(self, by: str = "its coder nodes", repo: str | None = None) -> dict:
        """Study one repository: read the source files that changed, take their functions apart, learn from them."""
        if not self.enabled:
            return {"ok": False, "why": self.detail}
        for _ in range(600):  # one visit at a time: wait for the one under way
            if not self.busy:
                break
            await asyncio.sleep(0.1)
        if self.busy:
            return {"ok": False, "why": "it is already studying"}
        self.busy = True
        s = self.sys
        try:
            listed, why = await self.call({"op": "repos"})
            if listed is None:
                self.stats["unreachable" if self.reached else "tool failed"] += 1
                return {"ok": False, "why": why}
            self.version = str(listed.get("version", ""))
            repos = [r for r in listed.get("repos") or [] if not r.get("empty") and not r.get("archived")]
            target = next((r for r in repos if r["full_name"] == repo), None) if repo else self._pick(repos)
            if target is None:
                return {"ok": False, "why": f"no repository called {repo}" if repo else "it has studied every repository lately"}
            name, branch = target["full_name"], target.get("default_branch") or "main"
            with s.activity.track("study", "coder", f"reading {name} on Gitea"):
                got_tree, why = await self.call({"op": "tree", "repo": name, "branch": branch})
                if got_tree is None:
                    return {"ok": False, "why": why}
                commit, tree = str(got_tree.get("commit", branch)), list(got_tree.get("files") or [])
                known = {r[0]: r[1] for r in self.db.execute("SELECT path, sha FROM code_files WHERE repo = ?", (name,)).fetchall()}
                exts = tuple(self.pol.extensions)
                todo = [e for e in tree if e["path"].lower().endswith(exts) and int(e.get("size") or 0) <= self.pol.max_file_kb * 1024
                        and known.get(e["path"]) != e.get("sha") and not re.search(r"(^|/)(node_modules|vendor|dist|build|\.git)/", e["path"])]
                left = max(0, len(todo) - self.pol.files_per_visit)  # what this visit leaves for the next one
                todo = todo[: self.pol.files_per_visit]
                got: list[tuple[dict, object]] = []
                per_call = max(5, int(self.pol.files_per_call))
                for i in range(0, len(todo), per_call):  # many files a call; the tool reads them at once
                    part = todo[i:i + per_call]
                    answer, why = await self.call({"op": "files", "repo": name, "ref": commit, "paths": [e["path"] for e in part]})
                    texts = (answer or {}).get("files") or {}
                    errors = (answer or {}).get("errors") or {}
                    for e in part:
                        got.append((e, texts[e["path"]] if e["path"] in texts else RuntimeError(errors.get(e["path"], why))))
            learned, kinds, functions = 0, Counter(), 0
            for e, text in got:
                if isinstance(text, Exception):
                    self.stats["files failed"] += 1
                    continue
                lang = LANGS.get(os.path.splitext(e["path"])[1].lower(), "other")
                fns = await asyncio.to_thread(functions_in, text, lang)
                self.db.execute("DELETE FROM code_examples WHERE repo = ? AND path = ?", (name, e["path"]))
                for fn in fns:
                    kind = kind_of(fn)
                    purpose = purpose_of(fn["name"], fn["doc"], fn["args"])
                    self.db.execute("INSERT INTO code_examples(repo, path, sha, lang, name, kind, purpose, signature, doc, source, "
                                    "lines, calls, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                    (name, e["path"], e.get("sha", ""), lang, fn["name"], kind, purpose, fn["signature"][:300],
                                     fn["doc"][:1000], fn["source"][:20000], fn["lines"], json.dumps(fn["calls"][:60]), time.time()))
                    kinds[kind] += 1
                    learned += self.learn_from(purpose, kind)
                functions += len(fns)
                header = json.dumps(await asyncio.to_thread(python_header, text)) if lang == "python" else ""
                self.db.execute("INSERT OR REPLACE INTO code_files(repo, path, sha, lang, size, studied, functions, header) "
                                "VALUES (?,?,?,?,?,?,?,?)", (name, e["path"], e.get("sha", ""), lang, int(e.get("size") or 0),
                                                             time.time(), len(fns), header))
            total = self.db.execute("SELECT COUNT(*) FROM code_examples WHERE repo = ?", (name,)).fetchone()[0]
            nfiles = self.db.execute("SELECT COUNT(*) FROM code_files WHERE repo = ?", (name,)).fetchone()[0]
            self.db.execute("INSERT OR REPLACE INTO code_repos(name, branch, commit_sha, studied, files, functions, description) "
                            "VALUES (?,?,?,?,?,?,?)", (name, branch, commit, time.time(), nfiles, total, str(target.get("description") or "")[:200]))
            self._trim()
            if left:
                self.backlog[name] = left
            else:
                self.backlog.pop(name, None)
            self.stats["visits"] += 1
            self.stats["files read"] += sum(1 for _, t in got if not isinstance(t, Exception))
            self.stats["functions"] += functions
            line = (f"studied {name} ({branch}): read {len(todo)} changed file(s) of {len(tree)}, took apart {functions} "
                    f"function(s)" + (f" ({', '.join(f'{n} {k}' for k, n in kinds.most_common(4))})" if kinds else "")
                    + (f"; its coder nodes learned {learned} new wording(s)" if learned else "")
                    + (f"; {left} more file(s) to read on its next visit" if left else ""))
            self._note(line, repo=name)
            self.save()
            s.bus.publish("gitea.studied", repo=name, files=len(todo), functions=functions, learned=learned, by=by,
                          kinds=dict(kinds))
            return {"ok": True, "repo": name, "files": len(todo), "functions": functions, "learned": learned, "line": line,
                    "left": left}
        finally:
            self.busy = False

    def learn_from(self, purpose: str, kind: str) -> int:
        """Teach the code-kind node the kind this purpose needed, and the coder node a wording of a recipe."""
        s = self.sys
        n = 0
        if not purpose:
            return 0
        if CODEKIND in s.capabilities and s.learn(CODEKIND, purpose, kind, "gitea"):
            n += 1
        try:
            from .toolsmith import CODER, NONE, CoderTeacher
            lesson = CoderTeacher().label_sync(purpose)
            if lesson.label and lesson.label != NONE and lesson.confidence >= 0.7 and CODER in s.capabilities:
                if s.learn(CODER, purpose, lesson.label, "gitea"):
                    n += 1
        except Exception:
            pass
        return n

    def header(self, repo: str, path: str) -> dict | None:
        """A Python file's imports and constants, as it kept them (None: it has not kept them yet)."""
        row = self.db.execute("SELECT header FROM code_files WHERE repo = ? AND path = ?", (repo, path)).fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    async def fetch_header(self, repo: str, path: str) -> dict | None:
        """Read a file again for its header (files studied before headers were kept)."""
        row = self.db.execute("SELECT branch FROM code_repos WHERE name = ?", (repo,)).fetchone()
        got, _ = await self.call({"op": "files", "repo": repo, "ref": (row[0] if row else "") or "main", "paths": [path]})
        text = ((got or {}).get("files") or {}).get(path)
        if not isinstance(text, str):
            return None
        head = await asyncio.to_thread(python_header, text)
        self.db.execute("UPDATE code_files SET header = ? WHERE repo = ? AND path = ?", (json.dumps(head), repo, path))
        return head

    def _trim(self) -> None:
        n = self.count()
        if n > self.pol.max_examples:
            self.db.execute("DELETE FROM code_examples WHERE id IN (SELECT id FROM code_examples ORDER BY id LIMIT ?)",
                            (n - self.pol.max_examples,))

    # ------------------------------------------------------------------ status
    def repos(self) -> list[dict]:
        rows = self.db.execute("SELECT name, branch, studied, files, functions, description FROM code_repos "
                               "ORDER BY studied DESC").fetchall()
        return [{"name": r[0], "branch": r[1], "studied": r[2], "files": r[3], "functions": r[4], "description": r[5]}
                for r in rows]

    def status(self) -> dict:
        kinds = dict(self.db.execute("SELECT kind, COUNT(*) FROM code_examples GROUP BY kind").fetchall())
        langs = dict(self.db.execute("SELECT lang, COUNT(*) FROM code_examples GROUP BY lang").fetchall())
        node = self.sys.registry.champion(CODEKIND) if CODEKIND in self.sys.capabilities else None
        return {"enabled": self.enabled, "url": self.url, "state": self.state, "detail": self.detail,
                "version": self.version, "examples": self.count(), "kinds": kinds, "languages": langs,
                "repos": self.repos()[:30], "log": list(self.log)[-12:][::-1], "stats": dict(self.stats),
                "every": self.pol.study_every_seconds, "busy": self.busy, "backlog": dict(self.backlog),
                "node": {"id": node.id, "accuracy": node.metrics.get("holdout_acc")} if node is not None else None}

    def lines(self) -> list[str]:
        st = self.status()
        out = [f"Gitea: {st['detail']}; {st['examples']} functions kept as examples"
               + (f" ({', '.join(f'{n} {k}' for k, n in sorted(st['languages'].items(), key=lambda kv: -kv[1]))})" if st["languages"] else "")]
        if st["kinds"]:
            out.append("  kinds: " + ", ".join(f"{k} {n}" for k, n in sorted(st["kinds"].items(), key=lambda kv: -kv[1])))
        if st["node"]:
            acc = st["node"]["accuracy"]
            out.append(f"  code-kind node {st['node']['id']}" + (f": {acc:.0%} on held-out purposes" if acc is not None else ""))
        for r in st["repos"][:8]:
            out.append(f"  {r['name']:<30} {r['files']:>4} files {r['functions']:>5} functions, studied "
                       f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['studied']))}")
        out += [f"  - {x['line']}" for x in st["log"][:5]]
        return out


class Apprentice(Agent):
    """When idle, it visits a Gitea repository it has not studied (or not lately) and learns from its code; and
    now and then it practises in its scratch pad (scratchpad.py)."""
    name = "apprentice"

    def __init__(self, system):
        super().__init__(system)
        self.last_visit = time.time() - system.directives.gitea.study_every_seconds + 30  # first, half a minute in
        self.last_practice = time.time()

    @property
    def interval(self) -> float:
        pol = self.sys.directives.gitea
        return max(0.5, min(5.0, pol.study_every_seconds / 6, max(0.5, pol.catch_up_seconds)))

    async def step(self) -> None:
        s = self.sys
        pol = s.directives.gitea
        if not s.code_study.enabled or not s.guardian.background_allowed() or s.inflight:
            return
        now = time.time()
        if now - self.last_visit >= pol.study_every_seconds:
            self.last_visit = now
            res = await s.code_study.visit()
            self.stats["visits" if res.get("ok") else "skipped"] += 1
            if res.get("ok") and (res.get("files") or res.get("left")):
                # catching up: while a visit still finds files to read, the next one follows almost at once
                self.last_visit = now - pol.study_every_seconds + pol.catch_up_seconds
                self.stats["catching up"] += 1
            return
        if pol.practice and s.scratch.enabled and now - self.last_practice >= pol.practice_every_seconds and s.code_study.count():
            self.last_practice = now
            res = await s.scratch.practice()
            self.stats["practised" if res else "no practice"] += 1
