"""The scratch pad: where it writes code, a validator checks it, and nothing runs until a person approves it (P20).

    "write a python function that reverses each word in a sentence"   (an order, /code write ..., or the web page)
      --> its coder node: a recipe from its code library? then the recipe's own code, with a demonstration
      --> otherwise its code-kind node says what kind of code it needs, and the closest examples it studied in
          Gitea (gitea.py) are found: an LLM teacher writes the code with those examples to follow, or, with no
          teacher, it adapts the closest example (and says where it came from)
      --> the validator checks it: Python is compiled, JavaScript is checked by Node.js (node --check), shell by
          bash -n, JSON and TOML parsed. It also lists what the code reaches for - files, the network, other
          programs, eval - so whoever approves it knows. Code that does not check out goes back to the teacher
          once, with the problems.
      --> it waits in the scratch pad. A person approves exactly that code (an approval, /code approve, or the
          web page); then, and only then, it runs: in a fresh folder, with a CPU, memory and time limit and no
          keys in its environment, and the output is kept with it. Changing the code withdraws the approval.

Auto-run ([code] auto_run, or the switch in the Code view; only a person changes it): "ask" (the default) is the
above. "harmless": code it writes itself runs straight away, without asking, when it checks out and reaches for
nothing (no files, network, other programs or modules beyond the basics); anything else still asks. "always": code
it writes itself runs straight away whenever it checks out. Either way it runs within the same limits, and the run
says it ran by itself and who switched that on. Code people put in the scratch pad always waits for approval.

Its own code as tools: Python code that ran can become a tool it uses (codetools.py, [code] make_tools).

Practice: when idle (gitea.py's Apprentice) it takes the purpose of a function people wrote, writes code for it
without looking at theirs - on its own, with no LLM teacher unless [gitea] practice_teacher - checks it, and compares
the two (does it check out, is it the same kind of code, how much do they share). Practice drafts never run.

With no recipe and no teacher, a request is met from what its workbench built (workbench.py): a whole program it
made from the closest code people wrote, checked (and, where P20 let it run, run), before an example copied as it is.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import tomllib
from collections import Counter, deque
from pathlib import Path

from . import codelib
from .gitea import CODEKIND, functions_in, kind_of

AUTO_RUN = ("ask", "harmless", "always")
MAKE_TOOLS = ("off", "ask", "auto")
_MODE_WORDS = {"auto_run": {"off": "ask", "no": "ask", "safe": "harmless", "on": "always", "yes": "always", "all": "always"},
               "make_tools": {"no": "off", "offer": "ask", "on": "auto", "yes": "auto"}}
_NOT_PEOPLE = ("teacher:", "self", "node", "agency", "director", "neuromesh", "mesh", "auto", "tutor", "foreman",
               "toolsmith", "searcher", "the web")


def is_person(by: str) -> bool:
    """Switches only people may change: not a teacher, a node, a director or the mesh itself."""
    b = str(by or "").strip().lower()
    return bool(b) and not b.startswith(_NOT_PEOPLE)

LANGS = ("python", "javascript", "typescript", "shell", "go", "rust", "ruby", "java", "c", "cpp", "json", "toml")
EXT = {"python": "py", "javascript": "js", "typescript": "ts", "shell": "sh", "go": "go", "rust": "rs", "ruby": "rb",
       "java": "java", "c": "c", "cpp": "cpp", "json": "json", "toml": "toml"}
COMMENT = {"python": "#", "shell": "#", "ruby": "#", "toml": "#"}
_ALIASES = {"py": "python", "python3": "python", "js": "javascript", "node": "javascript", "nodejs": "javascript",
            "ts": "typescript", "bash": "shell", "sh": "shell", "golang": "go", "rs": "rust", "c++": "cpp"}
CODE_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:write|code|draft|make|create|program)\s+"
    r"(?:me\s+|us\s+)?(?:a|an|some)?\s*(?:(?:small|short|little|simple|quick)\s+)?"
    r"(?:(python|python3|py|javascript|js|node|typescript|ts|bash|shell|sh|go|golang|rust|ruby|java|c\+\+|cpp|c)\s+)?"
    r"(?:function|script|program|code|snippet|class|method)\b\s*(?:in\s+(\w+)\s+)?(?:that|to|which|for|:)?\s*(?P<what>.+?)\s*[?.!]*$",
    re.I | re.S)
_RUN_LAUNCHER = """
import os, sys
cpu, mem, prog = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if mem > 0:
        resource.setrlimit(resource.RLIMIT_AS, (mem * 1048576, mem * 1048576))
except Exception:
    pass
os.execv(prog[0], prog)
"""


def code_request(text: str) -> tuple[str, str] | None:
    """'write a python function that reverses each word' -> ('reverses each word', 'python'); else None."""
    m = CODE_REQUEST.match(str(text))
    if not m or re.search(r"\btool\b", text[: m.start("what")], re.I):
        return None
    what = m.group("what").strip()
    if len(what) < 4:
        return None
    lang = (m.group(1) or m.group(2) or "").lower()
    lang = _ALIASES.get(lang, lang)
    return what, lang if lang in LANGS else ""


def digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def shared(a: str, b: str) -> float:
    """How much two pieces of code share: their identifiers and calls (Jaccard)."""
    ta = set(re.findall(r"[A-Za-z_]\w{2,}", a)) - {"def", "return", "self", "None", "True", "False", "for", "while", "import"}
    tb = set(re.findall(r"[A-Za-z_]\w{2,}", b)) - {"def", "return", "self", "None", "True", "False", "for", "while", "import"}
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


# ============================================================================ the validator
class Validator:
    """Is the code syntactically correct? And what does it reach for (so the person approving it knows)?"""

    def __init__(self):
        self.node = shutil.which("node")
        self.bash = shutil.which("bash")

    def checkers(self) -> dict:
        return {"python": "compiled by Python", "javascript": "node --check" if self.node else "not available (no Node.js)",
                "shell": "bash -n" if self.bash else "not available (no bash)", "json": "parsed", "toml": "parsed"}

    def check(self, code: str, lang: str) -> dict:
        problems, notes, checker, ok = [], [], "", None
        if lang == "python":
            checker = "Python's compiler"
            try:
                compile(code, "<scratch>", "exec")
                ok = True
            except SyntaxError as exc:
                ok = False
                problems.append(f"line {exc.lineno}: {exc.msg}" + (f" ({(exc.text or '').strip()[:80]})" if exc.text else ""))
            except ValueError as exc:
                ok, problems = False, [str(exc)]
            if ok:
                notes = self.touches_python(code)
        elif lang == "javascript" and self.node:
            checker = "node --check"
            ok, problems = self._run([self.node, "--check"], code, ".js")
            notes = self.touches_text(code)
        elif lang == "shell" and self.bash:
            checker = "bash -n"
            ok, problems = self._run([self.bash, "-n"], code, ".sh")
            notes = self.touches_text(code)
        elif lang == "json":
            checker = "a JSON parser"
            try:
                json.loads(code)
                ok = True
            except json.JSONDecodeError as exc:
                ok, problems = False, [f"line {exc.lineno}: {exc.msg}"]
        elif lang == "toml":
            checker = "a TOML parser"
            try:
                tomllib.loads(code)
                ok = True
            except tomllib.TOMLDecodeError as exc:
                ok, problems = False, [str(exc)]
        else:
            checker = "matched brackets only (no checker for this language here)"
            ok, problems = self.brackets(code)
            ok = None if ok else False
            notes = self.touches_text(code)
        return {"ok": ok, "checker": checker, "problems": problems[:8], "touches": notes, "language": lang}

    def _run(self, cmd: list[str], code: str, suffix: str) -> tuple[bool, list[str]]:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
            f.write(code)
            path = f.name
        try:
            p = subprocess.run(cmd + [path], capture_output=True, text=True, timeout=15)
        except Exception as exc:
            return False, [f"the checker failed: {exc}"]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if p.returncode == 0:
            return True, []
        err = (p.stderr or p.stdout).replace(path, "<scratch>").strip().splitlines()
        return False, [" ".join(x.split())[:200] for x in err if x.strip()][:6] or [f"exit {p.returncode}"]

    @staticmethod
    def brackets(code: str) -> tuple[bool, list[str]]:
        pairs, stack = {")": "(", "]": "[", "}": "{"}, []
        text = re.sub(r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')", "''", code)
        for i, ch in enumerate(text):
            if ch in "([{":
                stack.append((ch, i))
            elif ch in pairs:
                if not stack or stack[-1][0] != pairs[ch]:
                    return False, [f"an unmatched '{ch}' at line {text[:i].count(chr(10)) + 1}"]
                stack.pop()
        if stack:
            return False, [f"an unclosed '{stack[-1][0]}' from line {text[:stack[-1][1]].count(chr(10)) + 1}"]
        return True, []

    @staticmethod
    def touches_python(code: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        mods, out = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
            elif isinstance(n, ast.Call):
                f = n.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                if name in ("open", "write_text", "write_bytes", "unlink", "remove", "rmtree", "mkdir", "makedirs", "rename"):
                    out.add("files: it opens or changes files")
                if name in ("eval", "exec", "__import__", "compile"):
                    out.add(f"it runs code it builds ({name})")
                if name in ("system", "popen", "run", "Popen", "call", "check_output") and "subprocess" in mods | {"os"}:
                    out.add("other programs: it starts processes")
        if mods & {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"}:
            out.add("the network: " + ", ".join(sorted(mods & {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"})))
        if mods & {"subprocess", "multiprocessing"}:
            out.add("other programs: it starts processes")
        if mods & {"os", "shutil", "pathlib"} and not any(x.startswith("files") for x in out):
            out.add("files: it may read or change files (" + ", ".join(sorted(mods & {"os", "shutil", "pathlib"})) + ")")
        extra = sorted(mods - {"json", "re", "sys", "math", "string", "datetime", "collections", "itertools", "functools",
                               "statistics", "textwrap", "random", "typing", "dataclasses", "enum", "decimal", "fractions",
                               "unicodedata", "base64", "hashlib", "heapq", "bisect", "time", "__future__", "abc", "copy",
                               "operator", "pprint", "os", "shutil", "pathlib", "socket", "urllib", "http", "subprocess"})
        if extra:
            out.add("modules beyond the basics: " + ", ".join(extra[:6]))
        return sorted(out)

    @staticmethod
    def touches_text(code: str) -> list[str]:
        out = []
        if re.search(r"\b(fs\.|readFile|writeFile|open\(|os\.|File\b|rm\s+-|>\s*/)", code):
            out.append("files: it may read or change files")
        if re.search(r"\b(fetch\(|http\.|https\.|net\.|curl|wget|socket|XMLHttpRequest)", code):
            out.append("the network")
        if re.search(r"\b(child_process|exec\(|spawn\(|subprocess|system\()", code):
            out.append("other programs: it starts processes")
        if re.search(r"\beval\(", code):
            out.append("it runs code it builds (eval)")
        return out


# ============================================================================ the pad
class ScratchPad:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.code
        self.validator = Validator()
        self.items: dict[str, dict] = {}
        for rec in system.store.load_records("scratch"):
            if rec.get("status") == "running":  # it stopped while the code ran: the approval stands, the run did not finish
                rec["status"] = "approved" if rec.get("approved_digest") else "valid"
            self.items[rec["id"]] = rec
        nums = [int(k.split("-")[1]) for k in self.items if re.fullmatch(r"code-\d+", k)]
        self._n = max(nums, default=0)
        raw = system.store.get_kv("scratch.state")
        st = json.loads(raw) if raw else {}
        self.stats: Counter = Counter(st.get("stats", {}))
        self.practice_log: deque = deque(st.get("practice", []), maxlen=self.pol.keep_practice)
        self.root = system.state_dir / "scratch"
        self.rng = random.Random()
        self._running: set[str] = set()
        self.modes = {"auto_run": {"mode": self.pol.auto_run, "by": "directives.toml", "ts": 0},
                      "make_tools": {"mode": self.pol.make_tools, "by": "directives.toml", "ts": 0},
                      **st.get("modes", {})}

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def _save(self, item: dict) -> None:
        item["updated"] = round(time.time(), 1)
        self.sys.store.save_record("scratch", item["id"], item["status"], item)
        drafts = sorted(self.items.values(), key=lambda i: i.get("created", 0))
        while len(drafts) > self.pol.keep_items:
            old = drafts.pop(0)
            self.items.pop(old["id"], None)
            self.sys.store.delete_record("scratch", old["id"])

    def _state(self) -> None:
        self.sys.store.set_kv("scratch.state", json.dumps({"stats": dict(self.stats), "practice": list(self.practice_log),
                                                          "modes": self.modes}))

    # ------------------------------------------------------------------ the switches
    def mode(self, which: str) -> str:
        allowed = AUTO_RUN if which == "auto_run" else MAKE_TOOLS
        m = str(self.modes.get(which, {}).get("mode", ""))
        return m if m in allowed else ("ask" if which in ("auto_run", "make_tools") else allowed[0])

    def set_mode(self, which: str, mode: str, by: str) -> tuple[bool, str]:
        """auto_run (ask | harmless | always) or make_tools (off | ask | auto): a person's decision, audited."""
        if which not in ("auto_run", "make_tools"):
            return False, "the switches are auto_run and make_tools"
        allowed = AUTO_RUN if which == "auto_run" else MAKE_TOOLS
        m = str(mode or "").strip().lower()
        m = _MODE_WORDS[which].get(m, m)
        if m not in allowed:
            return False, f"{which} is one of: {', '.join(allowed)}"
        if not is_person(by):
            return False, "only a person changes that switch (P20)"
        was = self.mode(which)
        self.modes[which] = {"mode": m, "by": by, "ts": round(time.time(), 1)}
        self._state()
        self.sys.guardian.audit(by, "code.mode", which, mode=m, was=was)
        self.sys.bus.publish("code.mode", which=which, mode=m, was=was, by=by)
        return True, {("auto_run", "ask"): "code it writes waits for your approval before it runs",
                      ("auto_run", "harmless"): "code it writes runs by itself when it checks out and reaches for nothing; "
                                                "anything else still asks",
                      ("auto_run", "always"): "code it writes runs by itself whenever it checks out (in a fresh folder, "
                                              "within limits, without keys); code you put in still asks",
                      ("make_tools", "off"): "it doesn't turn its code into tools (you still can: Make it a tool)",
                      ("make_tools", "ask"): "it offers code worth keeping as a tool, and you approve it",
                      ("make_tools", "auto"): "it installs code it judges useful as a tool by itself (Python, standard "
                                              "library only, nothing reaching outside; tested like any tool)"}[(which, m)]

    def auto_runs(self, item: dict) -> tuple[bool, str]:
        """May this code run without asking? Only code it wrote itself, that checks out, as the switch allows."""
        mode = self.mode("auto_run")
        if mode == "ask":
            return False, "auto-run is off: it asks"
        if (item.get("origin") or {}).get("how") in (None, "", "person"):
            return False, "code people put in always asks"
        if not self.runnable(item["language"]):
            return False, f"{item['language']} code does not run here"
        check = item.get("check") or {}
        if check.get("ok") is not True:
            return False, "it does not check out"
        if mode == "harmless" and check.get("touches"):
            return False, f"it reaches for {'; '.join(check['touches'])}, so it asks"
        return True, mode

    async def auto_run(self, item: dict) -> dict | None:
        """Run code it wrote, without asking, because a person switched auto-run on (and say so)."""
        ok, mode = self.auto_runs(item)
        if not ok:
            return None
        who = self.modes["auto_run"].get("by", "")
        item["approved_digest"], item["approval"], item["status"] = item["digest"], None, "approved"
        item["auto_run"] = {"mode": mode, "set_by": who, "ts": round(time.time(), 1)}
        self._save(item)
        by = f"auto-run: {mode}, switched on by {who}"
        self.sys.guardian.audit(by, "code.auto_approved", item["id"], digest=item["digest"], mode=mode,
                                touches=(item.get("check") or {}).get("touches", []))
        return await self.run(item["id"], by)

    def get(self, item_id: str) -> dict | None:
        return self.items.get(item_id)

    def runnable(self, lang: str) -> bool:
        return lang in self.pol.run_languages and (lang != "javascript" or self.validator.node is not None)

    # ------------------------------------------------------------------ writing
    def _guess_language(self, request: str) -> str:
        low = request.lower()
        for word, lang in (("javascript", "javascript"), ("node", "javascript"), ("typescript", "typescript"),
                           ("bash", "shell"), ("shell", "shell"), ("golang", "go"), (" go ", "go"), ("rust", "rust"),
                           ("ruby", "ruby"), ("json", "json"), ("toml", "toml")):
            if word in f" {low} ":
                return lang
        return "python"

    def _from_recipe(self, request: str) -> tuple[str, dict] | None:
        """A recipe from its code library, when its coder node is sure enough: the recipe's own code."""
        s = self.sys
        try:
            from .toolsmith import NONE, PRACTICE, _deps, _literal, _source
            from .teachers import fill_template
        except Exception:
            return None
        label, conf, how = s.toolsmith.classify(request)
        if label == NONE or label not in codelib.RECIPES or conf < self.pol.recipe_confidence:
            return None
        recipe = codelib.RECIPES[label]
        params, problems = s.toolsmith.fit(recipe, [])
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
        demo = fill_template(recipe.needs[0], PRACTICE, self.rng) if recipe.needs else "hello"
        code = (f'"""{" ".join(request.split())[:120]}\n\nWritten by its coder node ({how}, {conf:.0%} sure) from the '
                f'"{recipe.id}" recipe in its code library: {recipe.title}."""\n'
                "import base64\nimport datetime\nimport re\n\n"
                f"PARAMS = {_literal(params)}\n" + "".join(f"{t} = {_literal(codelib.TABLES[t])}\n" for t in tables) + "\n\n"
                + "\n\n".join(_source(fn) for fn in parts)
                + f"\n\nif __name__ == \"__main__\":\n    print({recipe.block.__name__}({demo!r}, PARAMS)[\"output\"])\n")
        return code, {"how": "recipe", "recipe": recipe.id, "by": how, "confidence": round(conf, 3)}

    def _kind(self, request: str) -> tuple[str | None, str]:
        """What kind of code it needs, from its code-kind node (when trusted) or the examples' teacher."""
        s = self.sys
        node = s.registry.champion(CODEKIND) if CODEKIND in s.capabilities else None
        if node is not None and node.params is not None and (node.metrics.get("holdout_acc") or 0) >= s.directives.nodes.serve_min_accuracy:
            label, conf = node.predict(s.encode(request))
            return label, f"code-kind node {node.id} ({conf:.0%})"
        return None, ""

    async def draft(self, request: str, lang: str, exclude: int | None = None, teacher: bool = True) -> tuple[str | None, dict]:
        """Write code for a request: a recipe, a teacher following the closest Gitea examples, or the closest example."""
        s = self.sys
        if lang == "python" and exclude is None:
            made = self._from_recipe(request)
            if made is not None:
                return made
        kind, kind_by = self._kind(request)
        found = s.code_study.similar(request, k=max(1, self.pol.examples), lang=lang, kind=kind, exclude=exclude)
        examples = [ex for ex, _ in found]
        origin = {"how": "", "kind": kind, "kind_by": kind_by,
                  "examples": [{"id": ex["id"], "repo": ex["repo"], "path": ex["path"], "name": ex["name"],
                                "score": round(sc, 3)} for ex, sc in found]}
        if teacher and s.teachers.has_llm():
            made = await s.teachers.write_code(request, lang, examples)
            if made is not None:
                code, who = made[0]["code"], made[1]
                check = self.validator.check(code, lang)
                if check["ok"] is False and self.pol.teacher_fix:
                    fixed = await s.teachers.write_code(request, lang, examples, previous=code, problems=check["problems"])
                    if fixed is not None:
                        code, origin["fixed"] = fixed[0]["code"], check["problems"][:3]
                return code, {**origin, "how": "teacher", "teacher": who, "explanation": made[0].get("explanation", "")}
        bench = getattr(s, "workbench", None)
        built = bench.program_for([ex["id"] for ex in examples]) if bench is not None and lang == "python" and examples else None
        if built is not None:  # a program its workbench built from code people wrote (and checked; maybe ran)
            what = {"passed": "it ran and passed its checks", "ran": "it ran", "checks out": "it checks out"}[built["status"]]
            head = (f"# Built by its workbench (program {built['id']}) from {built['repo']}/{built['path']} ({built['name']}) on\n"
                    f"# Gitea, the closest code people wrote to \"{' '.join(request.split())[:90]}\": {what}. With no teacher\n"
                    f"# to write it, it starts from a whole program it built itself.\n")
            return head + built["code"], {**origin, "how": "workbench", "program": built["id"], "program_status": built["status"]}
        if examples:
            ex = examples[0]
            mark = COMMENT.get(lang, "//")
            head = (f"{mark} Adapted from {ex['repo']}/{ex['path']} ({ex['name']}) on Gitea: the closest code people wrote\n"
                    f"{mark} to \"{' '.join(request.split())[:100]}\". With no teacher to write it, it starts from theirs.\n")
            return head + ex["source"] + "\n", {**origin, "how": "example"}
        return None, {**origin, "how": "nothing", "why": "no recipe fits, no teacher can write it, and it has no example close to it yet"}

    def _new(self, title: str, code: str, lang: str, by: str, request: str = "", origin: dict | None = None,
             kind: str = "draft") -> dict:
        self._n += 1
        item = {"id": f"code-{self._n}", "title": " ".join(title.split())[:120], "request": request[:500], "language": lang,
                "code": code, "origin": origin or {"how": "person"}, "by": by, "kind": kind,
                "created": round(time.time(), 1), "updated": round(time.time(), 1), "status": "draft",
                "check": None, "digest": digest(code), "approval": None, "approved_digest": None, "runs": []}
        self.items[item["id"]] = item
        self.validate(item)
        return item

    async def write(self, request: str, by: str, lang: str = "") -> dict:
        """A person (or an order) asked it to write code: draft it into the scratch pad and check it."""
        lang = _ALIASES.get(lang.lower(), lang.lower()) if lang else self._guess_language(request)
        if lang not in LANGS:
            lang = "python"
        with self.sys.activity.track("scratch", "coder", f"writing {lang}: {' '.join(request.split())[:70]}"):
            code, origin = await self.draft(request, lang)
        if code is None:
            self.stats["could not write"] += 1
            self._state()
            return {"id": None, "status": "nothing", "why": origin.get("why", ""), "origin": origin}
        item = self._new(request, code, lang, by, request=request, origin=origin)
        self.stats["written"] += 1
        self._state()
        self.sys.bus.publish("code.written", id=item["id"], title=item["title"], language=lang, how=origin.get("how"),
                             ok=item["check"]["ok"], by=by)
        if self.auto_runs(item)[0]:  # a person switched auto-run on: it runs now, without asking
            await self.auto_run(item)
        return item

    def add(self, code: str, lang: str, title: str, by: str) -> dict:
        """A person put code in the scratch pad."""
        lang = _ALIASES.get(lang.lower(), lang.lower()) or "python"
        item = self._new(title or "your code", code, lang if lang in LANGS else "python", by, origin={"how": "person", "by": by})
        self.sys.bus.publish("code.written", id=item["id"], title=item["title"], language=item["language"], how="person",
                             ok=item["check"]["ok"], by=by)
        return item

    def edit(self, item_id: str, code: str, by: str) -> tuple[bool, str]:
        item = self.items.get(item_id)
        if item is None:
            return False, f"no scratch pad item {item_id}"
        if item["status"] == "running":
            return False, f"{item_id} is running"
        item["code"], item["digest"] = code, digest(code)
        item.setdefault("edits", []).append({"ts": round(time.time(), 1), "by": by})
        self.validate(item)
        return True, f"{item_id} changed: {self.say_check(item)}"

    def validate(self, item: dict) -> dict:
        check = self.validator.check(item["code"], item["language"])
        item["check"] = check
        item["digest"] = digest(item["code"])
        if item.get("approved_digest") and item["approved_digest"] != item["digest"]:
            item["approved_digest"] = None  # the code changed: the approval was for other code
            item["approval"] = None
        pending = self.sys.guardian.approval_for(item.get("approval"), "code", item["id"])
        if pending is not None and pending.status == "pending" and pending.detail.get("digest") != item["digest"]:
            item["approval"] = None  # asked about other code: withdrawn (approve it again to run what is there now)
            self.sys.guardian.withdraw(pending.id, "its code changed after it asked")
        if item["status"] not in ("running",):
            item["status"] = ("approved" if item.get("approved_digest") else "valid" if check["ok"] else
                              "invalid" if check["ok"] is False else "unchecked")
        self.stats["checked"] += 1
        self.stats["checked ok" if check["ok"] else "checked bad" if check["ok"] is False else "unchecked"] += 1
        self._save(item)
        return check

    @staticmethod
    def say_check(item: dict) -> str:
        c = item["check"] or {}
        if c.get("ok"):
            return f"it checks out ({c['checker']})" + (f"; it reaches for {'; '.join(c['touches'])}" if c.get("touches") else "")
        if c.get("ok") is False:
            return f"it does not check out ({c['checker']}): " + "; ".join(c.get("problems") or [])
        return f"it could not be checked: {c.get('checker', '')}"

    # ------------------------------------------------------------------ approval and running
    def ask(self, item_id: str, by: str) -> tuple[bool, str]:
        """Ask a person to approve the code to run (an approval, like a tool's)."""
        s = self.sys
        item = self.items.get(item_id)
        if item is None:
            return False, f"no scratch pad item {item_id}"
        if not self.runnable(item["language"]):
            return False, f"{item['language']} code does not run here (it runs: {', '.join(l for l in self.pol.run_languages if self.runnable(l))})"
        if item["check"] and item["check"]["ok"] is False:
            return False, f"{item_id} does not check out, so it cannot run: " + "; ".join(item["check"]["problems"][:2])
        a = s.guardian.approval_for(item.get("approval"), "code", item_id)
        if a is not None and a.status == "pending":
            if a.detail.get("digest") == item["digest"]:
                return True, f"{item_id} is waiting for approval {item['approval']}"
            item["approval"] = None  # it asked about other code
            s.guardian.withdraw(a.id, "its code changed after it asked")
        a = s.guardian.request_approval("code", item_id, {
            "title": item["title"], "language": item["language"], "digest": item["digest"],
            "touches": (item["check"] or {}).get("touches", []), "lines": item["code"].count("\n") + 1,
            "description": f"run {item['language']} code from its scratch pad: {item['title'][:80]}"})
        item["approval"], item["status"] = a.id, "awaiting approval"
        self._save(item)
        s.bus.publish("code.asked", id=item_id, approval=a.id, title=item["title"], by=by)
        return True, f"{item_id} waits for your approval: /approve {a.id} (or /code approve {item_id})"

    def approve(self, item_id: str, by: str) -> tuple[bool, str]:
        """A person approves the code as it stands now; it then runs."""
        ok, msg = self.ask(item_id, by)
        if not ok:
            return ok, msg
        item = self.items[item_id]
        a = self.sys.guardian.decide(item["approval"], True, by=by, note=f"code {item['digest']}")
        if a is None:
            return False, f"approval {item['approval']} was already decided"
        return True, f"{item_id} approved by {by}; it runs now"

    def decline(self, item_id: str, by: str) -> tuple[bool, str]:
        item = self.items.get(item_id)
        if item is None:
            return False, f"no scratch pad item {item_id}"
        a = self.sys.guardian.approval_for(item.get("approval"), "code", item_id)
        if a is not None and a.status == "pending":
            self.sys.guardian.decide(a.id, False, by=by)
        item["status"], item["approval"], item["approved_digest"] = "declined", None, None
        self._save(item)
        return True, f"{item_id} declined: it will not run"

    def remove(self, item_id: str, by: str) -> tuple[bool, str]:
        if self.items.pop(item_id, None) is None:
            return False, f"no scratch pad item {item_id}"
        self.sys.store.delete_record("scratch", item_id)
        self.sys.guardian.audit(by, "code.removed", item_id)
        return True, f"{item_id} removed from the scratch pad"

    def resume(self) -> None:
        """After a restart, ask again for code that was waiting (approvals are not kept across restarts)."""
        for item in list(self.items.values()):
            if item["status"] != "awaiting approval":
                continue
            a = self.sys.guardian.approval_for(item.get("approval"), "code", item["id"])
            if a is not None and a.status == "pending":
                continue
            item["approval"] = None
            ok, _ = self.ask(item["id"], "neuromesh")
            if not ok:  # it can no longer run here, or no longer checks out
                item["status"] = "valid" if (item.get("check") or {}).get("ok") else "unchecked"
                self._save(item)

    def on_event(self, ev) -> None:
        if ev.topic != "approval.decided" or ev.data.get("kind", "code") != "code":
            return
        item = self.items.get(str(ev.data.get("subject", "")))
        if item is None or not item.get("approval") or item.get("approval") != ev.data.get("id"):
            return  # (an approval for something else, or one this item no longer waits on)
        if ev.data.get("status") == "withdrawn":
            item["approval"] = None
            self._save(item)
            return
        if ev.data.get("status") == "approved":
            a = self.sys.guardian.approvals.get(ev.data.get("id"))
            if a is not None and a.detail.get("digest") not in (None, item["digest"]):
                item["approval"] = None  # approved, but for code that has changed since: it does not run (P20)
                item["status"] = "valid" if (item.get("check") or {}).get("ok") else "unchecked"
                self._save(item)
                self.sys.guardian.audit(ev.data.get("by", "a person"), "code.not_run", item["id"],
                                        why="its code changed after it asked", approval=a.id)
                return
            item["approved_digest"], item["status"] = item["digest"], "approved"
            item["approved_by"] = ev.data.get("by", "a person")
            self._save(item)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            task = loop.create_task(self.run(item["id"], item["approved_by"]))
            self.sys._bg_tasks.add(task)
            task.add_done_callback(self.sys._bg_tasks.discard)
        else:
            item["status"], item["approval"] = "declined", None
            self._save(item)

    async def run(self, item_id: str, by: str) -> dict:
        """Run approved code: only exactly the code a person approved, in a fresh folder, within limits."""
        s = self.sys
        item = self.items.get(item_id)
        if item is None:
            return {"ok": False, "error": f"no scratch pad item {item_id}"}
        if not item.get("approved_digest") or item["approved_digest"] != digest(item["code"]):
            return {"ok": False, "error": "no person has approved this code as it stands (P20)"}
        if s.guardian.halted:
            return {"ok": False, "error": "the system is halted"}
        if not self.runnable(item["language"]) or item_id in self._running:
            return {"ok": False, "error": "it cannot run that here" if item_id not in self._running else "it is running"}
        self._running.add(item_id)
        item["status"] = "running"
        self._save(item)
        folder = self.root / "runs" / f"{item_id}-{int(time.time())}"
        folder.mkdir(parents=True, exist_ok=True)
        entry = folder / f"main.{EXT[item['language']]}"
        entry.write_text(item["code"], encoding="utf-8")
        if item["language"] == "python":
            prog, mem = [sys.executable, "-I", str(entry)], self.pol.run_memory_mb
        else:
            prog, mem = [self.validator.node, str(entry)], 0  # (V8 reserves more address space than it uses)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(folder), "LANG": "C.UTF-8",
               "PYTHONIOENCODING": "utf-8", "TMPDIR": str(folder)}
        t0 = time.perf_counter()
        limit = self.pol.max_output_kb * 1024
        rec = {"ts": round(time.time(), 1), "by": by, "digest": item["digest"], "folder": str(folder)}
        try:
            with s.activity.track("run", "scratch", f"running {item_id} ({item['language']}), approved by {by}"):
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-I", "-c", _RUN_LAUNCHER, str(self.pol.run_cpu_seconds), str(mem), *prog,
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    cwd=str(folder), env=env)
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), self.pol.run_timeout_s)
                    rec.update(exit=proc.returncode, timeout=False)
                except asyncio.TimeoutError:
                    proc.kill()
                    out, err = await proc.communicate()
                    rec.update(exit=None, timeout=True)
        except Exception as exc:
            out, err = b"", str(exc).encode()
            rec.update(exit=None, timeout=False)
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["stdout"] = out[:limit].decode("utf-8", errors="replace")
        rec["stderr"] = err[:limit].decode("utf-8", errors="replace")
        rec["ok"] = rec["exit"] == 0 and not rec["timeout"]
        item.setdefault("runs", []).append(rec)
        item["runs"] = item["runs"][-10:]
        item["status"] = "ran" if rec["ok"] else "failed"
        self._running.discard(item_id)
        self._save(item)
        self.stats["ran" if rec["ok"] else "run failed"] += 1
        self._state()
        s.guardian.audit(by, "code.ran", item_id, ok=rec["ok"], exit=rec["exit"], digest=item["digest"], ms=rec["ms"])
        s.bus.publish("code.ran", id=item_id, ok=rec["ok"], exit=rec["exit"], ms=rec["ms"], by=by,
                      output=(rec["stdout"] or rec["stderr"])[:200], timeout=rec["timeout"])
        return rec

    # ------------------------------------------------------------------ practice
    @staticmethod
    def challenge_key(lang: str, purpose: str) -> str:
        """What a practice challenge asks for, whatever repository it came from."""
        return hashlib.sha256(f"{lang}:{' '.join(str(purpose).lower().split())}".encode()).hexdigest()[:20]

    def practised_count(self) -> int:
        try:
            return self.sys.store.db.execute("SELECT COUNT(*) FROM code_practised").fetchone()[0]
        except Exception:
            return 0

    async def practice(self) -> dict | None:
        """Write code for the purpose of a function people wrote, without looking at it; check it; compare."""
        s = self.sys
        db = s.store.db  # never the same challenge twice: what it practised is remembered by what was asked for
        db.execute("CREATE TABLE IF NOT EXISTS code_practised(key TEXT PRIMARY KEY, ts REAL, ok INTEGER)")
        done = {r[0] for r in db.execute("SELECT key FROM code_practised").fetchall()}
        rows = db.execute("SELECT id, lang, purpose FROM code_examples WHERE practised = 0 ORDER BY (lines BETWEEN 3 AND 60 "
                          "AND doc != '') DESC, CASE lang WHEN 'python' THEN 0 ELSE 1 END, RANDOM() LIMIT 400").fetchall()
        pick = next(((i, self.challenge_key(lang, purpose)) for i, lang, purpose in rows
                     if self.challenge_key(lang, purpose) not in done), None)
        for i, lang, purpose in rows:  # (the same purpose studied again, or in another repository: done already)
            if self.challenge_key(lang, purpose) in done:
                db.execute("UPDATE code_examples SET practised = 1 WHERE id = ?", (i,))
        if pick is None:
            return None
        ex = s.code_study.example(pick[0])
        db.execute("UPDATE code_examples SET practised = practised + 1 WHERE id = ?", (ex["id"],))
        db.execute("INSERT OR REPLACE INTO code_practised(key, ts, ok) VALUES (?,?,0)", (pick[1], time.time()))
        with s.activity.track("practice", "coder", f"practising: {ex['purpose'][:70]}"):
            code, origin = await self.draft(ex["purpose"], ex["lang"], exclude=ex["id"],
                                            teacher=s.directives.gitea.practice_teacher)
        if code is None:
            rec = {"ts": round(time.time(), 1), "example": ex["id"], "purpose": ex["purpose"][:160], "repo": ex["repo"],
                   "name": ex["name"], "how": "nothing", "ok": None, "same_kind": None, "shared": 0.0}
        else:
            check = self.validator.check(code, ex["lang"])
            fns = functions_in(code, ex["lang"])
            mine = kind_of(fns[0]) if fns else None
            rec = {"ts": round(time.time(), 1), "example": ex["id"], "purpose": ex["purpose"][:160], "repo": ex["repo"],
                   "name": ex["name"], "how": origin.get("how"), "ok": check["ok"], "same_kind": mine == ex["kind"],
                   "kind": ex["kind"], "my_kind": mine, "shared": round(shared(code, ex["source"]), 3),
                   "problems": check["problems"][:2], "code": code[:4000], "lang": ex["lang"]}
            self.stats["practised"] += 1
            self.stats["practice ok"] += 1 if check["ok"] else 0
            self.stats["practice same kind"] += 1 if mine == ex["kind"] else 0
        self.practice_log.append(rec)
        self._state()
        s.bus.publish("code.practised", example=ex["id"], purpose=ex["purpose"][:120], how=rec["how"], ok=rec["ok"],
                      same_kind=rec["same_kind"], shared=rec["shared"], repo=ex["repo"], name=ex["name"])
        return rec

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        items = sorted(self.items.values(), key=lambda i: -i.get("created", 0))
        n = self.stats.get("practised", 0)
        return {"enabled": self.enabled, "items": items[:40], "count": len(self.items),
                "runnable": [l for l in self.pol.run_languages if self.runnable(l)], "checkers": self.validator.checkers(),
                "stats": dict(self.stats), "practice": list(self.practice_log)[-12:][::-1],
                "practice_summary": {"practised": n, "ok": (self.stats.get("practice ok", 0) / n) if n else None,
                                     "same_kind": (self.stats.get("practice same kind", 0) / n) if n else None},
                "waiting": [i["id"] for i in items if i["status"] == "awaiting approval"],
                "auto_run": self.mode("auto_run"), "make_tools": self.mode("make_tools"),
                "modes": {k: dict(v, mode=self.mode(k)) for k, v in self.modes.items()}}

    def lines(self, item_id: str | None = None) -> list[str]:
        if item_id:
            item = self.items.get(item_id)
            if item is None:
                return [f"no scratch pad item {item_id}"]
            o = item["origin"]
            how = {"recipe": f"its coder node, from the {o.get('recipe')} recipe", "teacher": f"the {o.get('teacher')} teacher"
                   + (f", following {len(o.get('examples', []))} example(s) from Gitea" if o.get("examples") else ""),
                   "example": "the closest example it studied in Gitea" if not o.get("examples") else
                   f"adapted from {o['examples'][0]['repo']}/{o['examples'][0]['path']} ({o['examples'][0]['name']})",
                   "person": f"{o.get('by', 'a person')}"}.get(o.get("how"), o.get("how"))
            out = [f"{item['id']}  {item['title']}  [{item['language']}, {item['status']}]", f"  written by {how}",
                   f"  {self.say_check(item)}"]
            out += ["  " + line for line in item["code"].splitlines()[:80]]
            for r in item.get("runs", [])[-2:]:
                out.append(f"  ran {time.strftime('%H:%M:%S', time.localtime(r['ts']))} "
                           + (f"(by itself: {r['by']}): " if str(r["by"]).startswith("auto-run") else f"(approved by {r['by']}): ")
                           + ("ok" if r["ok"] else "timed out" if r.get("timeout") else f"exit {r['exit']}"))
                out += ["    " + x for x in (r["stdout"] or r["stderr"]).splitlines()[:20]]
            return out
        st = self.status()
        ps = st["practice_summary"]
        run_say = {"ask": "code runs only once you approve it (P20)",
                   "harmless": "code it writes runs by itself when it reaches for nothing; the rest waits for you",
                   "always": "code it writes runs by itself whenever it checks out; yours waits for you"}[st["auto_run"]]
        out = [f"Scratch pad: {st['count']} item(s); {run_say}. Runs here: "
               f"{', '.join(st['runnable']) or 'nothing'}. /code write <what>, /code show <id>, /code approve <id>.",
               f"  auto-run: {st['auto_run']} (set by {self.modes['auto_run'].get('by')}); its code as tools: {st['make_tools']} "
               f"(/code autorun ask|harmless|always, /code tools off|ask|auto, /code tool <id>)"]
        for i in st["items"][:12]:
            out.append(f"  {i['id']:<8} {i['status']:<17} {i['language']:<10} {i['title'][:60]}")
        if ps["practised"]:
            out.append(f"  practice: {ps['practised']} rounds; its code checked out {ps['ok']:.0%} of the time and was the same "
                       f"kind as people's {ps['same_kind']:.0%}")
        return out
