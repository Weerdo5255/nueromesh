"""The tool doctor: when a tool fails, it works out why, and fixes what is its own code's fault (P3, P15).

    a tool fails    a run, its self-tests or a health check; offline and busy failures count here too
      --> notes     the error, the input, and the traceback its code printed (plugins send "trace")
      --> diagnose  after [repair] diagnose_after_failures failures in window_seconds, or when it is quarantined:
                      code         a fault in the tool's own code: a traceback through its files, a wrong
                                   answer to a self-test, output that isn't JSON
                      network      its server can't be reached (refused, unknown host, timed out): for a
                                   networked tool it checks whether the server answers at all
                      service      the server answered with an error (5xx, 429): busy, down, overloaded
                      config       its settings: the address, a token, the server's own settings (401, 403, 404)
                      environment  something missing on this machine: a module, a program, a permission
                      limits       it ran out of time, memory or CPU in the sandbox
                    with an LLM teacher it also reviews the code against the failures and says which it is
      --> fix       only a fault in the code: the teacher rewrites the code files (never the self-tests); the
                    fix must compile and may not do anything the old code didn't (no new network access,
                    files, processes, eval or exec); it is shown as a diff and a person approves it (P3, P15).
                    Approved: the old files are backed up (state/tool_backups/), the new ones installed with
                    the version bumped, and the tool is self-tested again. Still failing: the backup goes
                    back, and it tries again ([repair] attempts), then gives up and says so.
      --> report    a fault outside the code is reported, never "fixed" in the code: what failed, why it
                    thinks so, and what a person could do about it.
A person has the last word: any tool can be removed from the mesh (the web page's Tools, POST /tools/remove,
/tool <name> remove). Its files stay where they are and scans pass it by; restoring it vets and tests it again.
"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import logging
import re
import shutil
import time
import urllib.parse
from collections import deque
from pathlib import Path

from .tools import ACTIVE, PENDING, QUARANTINED, REMOVED, Tool, ToolRun, folder_digest

log = logging.getLogger("neuromesh.tooldoctor")

CAUSES = ("code", "network", "service", "config", "environment", "limits", "unknown")
OUTSIDE = ("network", "service", "config", "environment", "limits")
SAY = {"code": "a fault in its own code", "network": "the network: its server can't be reached",
       "service": "its server: it answered with an error", "config": "its settings (the address, a token, or the server's settings)",
       "environment": "this machine: something it needs is missing", "limits": "the sandbox's limits: it ran out of time or memory",
       "unknown": "something it can't pin down"}
ADVICE = {"network": "check that its server is running and reachable from here, then test it again",
          "service": "its server is having trouble; it will be tested again later, or test it once the server is back",
          "config": "check its address and token under [tools.config.{name}] (or its server's settings)",
          "environment": "install what it needs on this machine, or remove the tool",
          "limits": "give it more time ([tools] default_timeout_s, or timeout_s in its manifest), or a smaller job",
          "unknown": "look at its last error, test it again, or remove it",
          "code": "approve the fix, or remove the tool"}

_NETWORK = re.compile(r"connection refused|could not reach|couldn't reach|unreachable|name or service not known|"
                      r"nodename nor servname|getaddrinfo|temporary failure in name resolution|no route to host|"
                      r"network is unreachable|connection reset|remote end closed|urlopen error|urlerror|"
                      r"connectionrefusederror|connectionreseterror|timed out|timeouterror|ssl|certificate|"
                      r"is it running|offline", re.I)
_STATUS = r"(?:http(?: error)?|status|returned|answered|code)\s*:?\s*"
_SERVICE = re.compile(_STATUS + r"(?:5\d\d|429)\b|too many requests|service unavailable|bad gateway|"
                      r"gateway time-?out|internal server error|temporarily unavailable|overloaded", re.I)
_CONFIG = re.compile(_STATUS + r"(?:401|403|404)\b|unauthori[sz]ed|forbidden|not found at|no url|url is not set|"
                     r"not configured|missing (?:a |the |its )?(?:url|token|address|setting|api key)|"
                     r"(?:invalid|bad|expired|wrong) (?:api )?(?:token|key|url|address)|format not allowed", re.I)
_ENV = re.compile(r"modulenotfounderror|no module named|importerror|permission denied|filenotfounderror|"
                  r"command not found|no such file or directory", re.I)
_LIMITS = re.compile(r"timeout after|memoryerror|cannot allocate memory|cpu time limit|"
                     r"the tool's worker stopped|exit -9|exit -24|recursionerror", re.I)
_CODE_ERRORS = ("NameError", "TypeError", "AttributeError", "IndexError", "KeyError", "ValueError",
                "ZeroDivisionError", "SyntaxError", "IndentationError", "UnboundLocalError", "AssertionError",
                "JSONDecodeError", "UnicodeDecodeError", "UnicodeEncodeError", "StopIteration", "NotImplementedError",
                "OverflowError", "LookupError")
RISKY_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "input", "breakpoint",
               "getattr", "setattr", "delattr", "memoryview"}
RISKY_ATTRS = {"system", "popen", "spawn", "fork", "remove", "unlink", "rmtree", "rename", "chmod", "kill", "socket",
               "connect", "urlopen", "request", "load_module", "exec_module", "__subclasses__", "__globals__",
               "__builtins__", "__code__", "write_text", "write_bytes", "HTTPConnection", "HTTPSConnection", "Popen",
               "run", "call", "check_output"}
RISKY_MODULES = {"subprocess", "socket", "ctypes", "multiprocessing", "shutil", "os", "pty", "signal", "importlib",
                 "http", "urllib", "ftplib", "smtplib", "telnetlib", "asyncio", "pickle", "marshal"}


def classify(errors: list[str], traces: list[str], tool_files: list[str]) -> tuple[str, str, float]:
    """A first look at why a tool failed, from its errors and tracebacks: (cause, why, confidence)."""
    text = "\n".join(errors + traces)
    last = (errors or [""])[-1]
    exc_types = re.findall(r"^(\w+(?:\.\w+)*(?:Error|Exception|Exit|Interrupt))\b", "\n".join(traces), re.M)
    exc_types += re.findall(r"^(\w+Error)\b:", "\n".join(errors), re.M)
    through_code = any(re.search(rf'File "[^"]*{re.escape(f)}", line \d+', text) for f in tool_files)
    if _LIMITS.search(last) and not _NETWORK.search(last):
        return "limits", f"it ran out of room in the sandbox: {last[:160]}", 0.7
    if _ENV.search(text):
        m = re.search(r"No module named '([^']+)'", text)
        return "environment", (f"the module {m.group(1)} isn't installed here" if m else
                               f"something it needs is missing on this machine: {last[:160]}"), 0.8
    if _SERVICE.search(text):
        return "service", f"its server answered with an error: {last[:160]}", 0.75
    if _NETWORK.search(text):
        return "network", f"it couldn't reach its server: {last[:160]}", 0.75
    if _CONFIG.search(text) and not any(t.split(".")[-1] in _CODE_ERRORS for t in exc_types):
        return "config", f"its settings look wrong: {last[:160]}", 0.6
    code_exc = next((t for t in exc_types if t.split(".")[-1] in _CODE_ERRORS), "")
    if code_exc and (through_code or not traces):
        where = re.findall(r'File "([^"]+)", line (\d+), in (\w+)', text)
        at = next((f"{Path(f).name} line {n}, in {fn}" for f, n, fn in reversed(where)
                   if any(f.endswith(x) for x in tool_files)), "")
        return "code", f"its code raised {code_exc}" + (f" ({at})" if at else "") + f": {last[:160]}", 0.85
    if re.search(r"expected .+, got |invalid json|printed invalid|asked an empty question|entrypoint", text, re.I):
        return "code", f"its code gave a wrong or malformed answer: {last[:160]}", 0.7
    if through_code:
        return "code", f"its code failed: {last[:160]}", 0.55
    return "unknown", f"its last error: {last[:160]}" if last else "no error was recorded", 0.2


def _calls(tree: ast.AST) -> tuple[set, set, set]:
    names, attrs, mods = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return names, attrs, mods


def check_fix(old: str, new: str, path: str) -> list[str]:
    """A fix must compile and may not do anything riskier than the code it replaces."""
    try:
        new_tree = ast.parse(new, filename=path)
        compile(new_tree, path, "exec")
    except SyntaxError as exc:
        return [f"{path} doesn't compile: {exc.msg} (line {exc.lineno})"]
    try:
        old_tree = ast.parse(old, filename=path) if old.strip() else ast.parse("")
    except SyntaxError:
        old_tree = ast.parse("")
    on, oa, om = _calls(old_tree)
    nn, na, nm = _calls(new_tree)
    problems = [f"{path} now uses {x}()" for x in sorted((nn & RISKY_NAMES) - on)]
    problems += [f"{path} now uses .{x}" for x in sorted((na & RISKY_ATTRS) - oa)]
    problems += [f"{path} now imports {x}" for x in sorted((nm & RISKY_MODULES) - om)]
    if len(new.strip()) < 0.3 * len(old.strip()):
        problems.append(f"{path} lost most of its code ({len(old)} -> {len(new)} characters)")
    return problems


def diff_of(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        f"a/{path}", f"b/{path}", n=3))


def bump(version: str) -> str:
    parts = [int(p) if p.isdigit() else 0 for p in re.findall(r"\d+", version)[:3]] + [0, 0, 0]
    return f"{parts[0]}.{parts[1]}.{parts[2] + 1}"


async def reach(url: str, timeout: float = 3.0) -> str:
    """Does anything answer at url's host and port? ('answers', or why not.)"""
    u = urllib.parse.urlsplit(url if "://" in url else f"http://{url}")
    if not u.hostname:
        return "it has no address to try"
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(u.hostname, port), timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return "answers"
    except asyncio.TimeoutError:
        return f"no answer from {u.hostname}:{port} within {timeout:.0f}s"
    except ConnectionRefusedError:
        return f"{u.hostname}:{port} refused the connection (nothing is listening there)"
    except OSError as exc:
        if "name" in str(exc).lower() or "address" in str(exc).lower() or getattr(exc, "errno", 0) in (-2, -3, -5, 8):
            return f"{u.hostname} can't be found (unknown host)"
        return f"{u.hostname}:{port} can't be reached ({exc})"


class ToolDoctor:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.repair
        self.failures: dict[str, deque] = {}
        raw = system.store.get_kv("tooldoctor.state")
        st = json.loads(raw) if raw else {}
        self.diagnoses: dict[str, dict] = st.get("diagnoses", {})
        self.fixes: dict[str, dict] = st.get("fixes", {})
        self.preapproved: dict[str, list] = st.get("preapproved", {})
        self.attempts: dict[str, int] = st.get("attempts", {})       # "<tool>@<digest>" -> fixes tried
        self.seq = int(st.get("seq", 0))
        self.history: deque = deque(st.get("history", []), maxlen=80)
        self.backups = system.state_dir / "tool_backups"
        self.workshop = system.state_dir / "tool_fixes"
        self._busy: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def save(self) -> None:
        fixes = dict(sorted(self.fixes.items(), key=lambda kv: kv[1].get("made_at", 0))[-60:])
        self.sys.store.set_kv("tooldoctor.state", json.dumps({
            "diagnoses": self.diagnoses, "fixes": fixes, "preapproved": self.preapproved, "attempts": self.attempts,
            "seq": self.seq, "history": list(self.history)}))

    def _log(self, tool: str, line: str) -> None:
        self.history.append({"at": round(time.time(), 1), "tool": tool, "line": line})

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:  # no loop (a synchronous caller in a test): nothing to schedule on
            coro.close()
            return
        self.sys._bg_tasks.add(task)
        task.add_done_callback(self.sys._bg_tasks.discard)

    # ----------------------------------------------------------- noticing failures
    def note(self, tool: Tool, run: ToolRun, text: str = "", kind: str = "run", expected=None) -> None:
        """A failed run (or self-test). Enough of them, close together, and it looks into why."""
        if not self.enabled or run.ok or run.refused or tool.state == REMOVED:
            return
        if run.query is not None or run.not_found:  # asking a director, or no such article: the tool works
            return
        q = self.failures.setdefault(tool.name, deque(maxlen=20))
        q.append({"at": time.time(), "kind": kind, "input": text[:300], "error": run.error[:600], "trace": run.trace[-4000:],
                  "transient": run.transient, "expected": expected, "got": None if run.ok else run.output})
        recent = [f for f in q if time.time() - f["at"] <= self.pol.window_seconds]
        if len(recent) >= max(1, self.pol.diagnose_after_failures) and self._due(tool.name):
            self._spawn(self.diagnose(tool.name, f"{len(recent)} failed runs"))

    def _due(self, name: str) -> bool:
        """Not already looking at it, and nothing new learned since the last look (another version, a fix undone,
        or the cool-down over)."""
        if name in self._busy or self._applied(name) is not None:
            return False
        last = self.diagnoses.get(name)
        if last is None or last.get("again"):
            return True
        tool = self.sys.tools.tools.get(name)
        return time.time() - last.get("at", 0) >= self.pol.cooldown_seconds or (
            tool is not None and last.get("digest") != tool.manifest.digest)

    def on_event(self, ev) -> None:
        try:
            self._on_event(ev)
        except Exception:  # never break the bus
            log.exception("the tool doctor could not handle %s", ev.topic)

    def _on_event(self, ev) -> None:
        if not self.enabled:
            return
        name = str(ev.data.get("tool", ""))
        if ev.topic == "tool.quarantined":
            fix = self._applied(name)
            if fix is not None:
                self._spawn(self._rollback(fix, str(ev.data.get("reason", ""))))
            elif self._due(name):
                self._spawn(self.diagnose(name, f"it was quarantined: {ev.data.get('reason', '')}"))
        elif ev.topic == "tool.active":
            fix = self._applied(name)
            if fix is not None:
                fix["status"] = "worked"
                fix["settled_at"] = round(time.time(), 1)
                self.failures.pop(name, None)
                d = self.diagnoses.get(name)
                if d is not None:
                    d["outcome"] = f"fixed: v{fix['to_version']} passed its self-tests"
                self._log(name, f"the fix {fix['id']} worked: v{fix['to_version']} passed its self-tests")
                self.sys.guardian.audit("tooldoctor", "tool.fix_worked", name, fix=fix["id"], version=fix["to_version"])
                self.sys.bus.publish("tool.fix_worked", tool=name, fix=fix["id"], version=fix["to_version"])
                self.sys.mind.note("update", f"tool:{name}", f"I fixed the {name} tool (v{fix['to_version']}): "
                                   f"{fix['explanation'][:200]} It passes its self-tests again.", private=False,
                                   source="tooldoctor")
                self.save()
        elif ev.topic == "approval.decided" and ev.data.get("kind") == "tool.fix":
            fix = next((f for f in self.fixes.values() if f.get("approval") == ev.data.get("id")), None)
            if fix is None or fix["status"] != "awaiting approval":
                return
            if ev.data.get("status") == "approved":
                self.apply(fix["id"], ev.data.get("by") or "a person")
            elif ev.data.get("status") == "denied":
                fix["status"], fix["decided_by"] = "declined", ev.data.get("by") or "a person"
                self._log(name or fix["tool"], f"the fix {fix['id']} was declined by {fix['decided_by']}")
                self.save()

    def _applied(self, name: str) -> dict | None:
        tool = self.sys.tools.tools.get(name)
        return next((f for f in self.fixes.values() if f["tool"] == name and f["status"] == "applied"
                     and (tool is None or f.get("digest_to") in (None, tool.manifest.digest))), None)

    # ----------------------------------------------------------- diagnosing
    def _files(self, tool: Tool) -> dict[str, str]:
        folder = tool.manifest.path
        entry = folder / tool.manifest.entrypoint
        paths = [entry] + sorted(p for p in folder.glob("*.py") if p != entry)
        out = {}
        for p in paths[:4]:
            try:
                if p.is_file() and p.stat().st_size <= 200_000:
                    out[p.name] = p.read_text(encoding="utf-8")
            except OSError:
                continue
        return out

    def _url(self, tool: Tool) -> str:
        cfg = self.sys.tool_config(tool)
        for key in ("url", "base_url", "server", "host"):
            if isinstance(cfg.get(key), str) and cfg[key].strip():
                return cfg[key].strip()
        servers = cfg.get("servers")
        if isinstance(servers, list) and servers and isinstance(servers[0], str):
            return servers[0]
        return ""

    async def diagnose(self, name: str, why: str = "", by: str = "tooldoctor") -> dict:
        """Why does this tool fail? Its code, or something outside it - and, for its code, a fix to approve."""
        s = self.sys
        tool = s.tools.tools.get(name)
        if tool is None:
            return {"ok": False, "message": f"no tool {name}"}
        if tool.state == REMOVED:
            return {"ok": False, "message": f"{name} was removed from the mesh"}
        if name in self._busy:
            return {"ok": False, "message": f"already looking at {name}"}
        self._busy.add(name)
        try:
            with s.activity.track("diagnose", "tooldoctor", f"why the {name} tool fails"):
                return await self._diagnose(tool, why, by)
        finally:
            self._busy.discard(name)

    async def _diagnose(self, tool: Tool, why: str, by: str) -> dict:
        s, name = self.sys, tool.name
        seen = [f for f in self.failures.get(name, []) if time.time() - f["at"] <= max(self.pol.window_seconds, 86400)]
        if not seen and tool.last_error:
            seen = [{"at": time.time(), "kind": "last", "input": "", "error": tool.last_error, "trace": "",
                     "transient": False, "expected": None, "got": None}]
        if not seen and tool.manifest.examples:  # nothing recorded: try one of its self-tests now
            ex = tool.manifest.examples[0]
            run = await s.sandbox.run(tool, ex["input"], tool.manifest.timeout_s or s.directives.tools.default_timeout_s,
                                      s.tool_config(tool))
            if run.ok and not self._matches(run.output, ex.get("expect")):
                run.ok, run.error = False, f"expected {ex.get('expect')!r}, got {run.output!r}"
            if not run.ok:
                seen = [{"at": time.time(), "kind": "probe", "input": ex["input"], "error": run.error, "trace": run.trace,
                         "transient": run.transient, "expected": ex.get("expect"), "got": run.output}]
        files = self._files(tool)
        if not seen:
            d = {"tool": name, "version": tool.manifest.version, "digest": tool.manifest.digest, "at": round(time.time(), 1),
                 "cause": "unknown", "where": "none", "why": "it hasn't failed lately: nothing to look into",
                 "confidence": 0.0, "probe": "", "teacher": "", "fix": "", "advice": "", "failures": 0, "trigger": why}
            self.diagnoses[name] = d
            self.save()
            return {"ok": True, **d}
        cause, reason, conf = classify([f["error"] for f in seen], [f["trace"] for f in seen if f["trace"]], list(files))
        probe = ""
        url = self._url(tool) if "network" in tool.manifest.permissions else ""
        if self.pol.probe_network and url and cause in ("network", "service", "config", "limits", "unknown"):
            probe = await reach(url)
            if probe != "answers":
                cause, reason, conf = "network", f"its server at {url} can't be reached: {probe}", 0.95
            elif cause == "network":
                reason += f" - but {url} answers now, so it was probably a passing outage"
                conf = 0.5
        teacher, explanation, fixed = "", "", {}
        key = f"{name}@{tool.manifest.digest}"
        tries = self.attempts.get(key, 0)
        want_fix = cause in ("code", "unknown") and self.pol.write_fixes and tries < self.pol.attempts
        if want_fix and s.teachers.has_llm():
            prev = next((f for f in sorted(self.fixes.values(), key=lambda f: -f.get("made_at", 0))
                         if f["tool"] == name and f.get("digest_from") == tool.manifest.digest
                         and f["status"] in ("rolled back", "failed")), None)
            made = await s.teachers.fix_tool(
                {"name": name, "description": tool.manifest.description, "files": files,
                 "manifest": self._manifest_text(tool)},
                [{**f, "got": None if f.get("got") is None else str(f["got"])[:300]} for f in seen[-4:]],
                {"cause": cause, "why": reason, "probe": probe},
                previous=(prev or {}).get("diff", ""), problems=[(prev or {}).get("result", "")] if prev else None)
            if made is not None:
                verdict, teacher = made
                explanation = verdict["explanation"]
                if verdict["cause"] == "code" or cause == "unknown":
                    cause = verdict["cause"]
                    reason = explanation or reason
                    conf = max(conf, 0.7)
                if cause == "code":
                    fixed = {p: t for p, t in verdict.get("files", {}).items() if p in files}
        d = {"tool": name, "version": tool.manifest.version, "digest": tool.manifest.digest, "at": round(time.time(), 1),
             "cause": cause, "where": "code" if cause == "code" else "outside" if cause in OUTSIDE else "unsure",
             "why": reason, "confidence": round(conf, 2), "probe": probe, "teacher": teacher, "fix": "",
             "advice": ADVICE.get(cause, "").format(name=name), "failures": len(seen), "trigger": why,
             "errors": [f["error"][:200] for f in seen[-3:]], "outcome": ""}
        if cause == "code":
            if fixed:
                fix = self._draft(tool, fixed, files, d, explanation or reason, teacher, by)
                if fix is not None:
                    d["fix"] = fix["id"]
                    d["outcome"] = f"a fix is waiting for approval ({fix['id']})" if fix["status"] == "awaiting approval" \
                        else f"the fix it wrote was not offered: {fix.get('result', '')}"
            elif tries >= self.pol.attempts:
                d["outcome"] = f"it tried {tries} fixes for this version; it won't try again - fix it by hand or remove it"
            elif not s.teachers.has_llm() or not self.pol.write_fixes:
                d["outcome"] = "no LLM teacher to write a fix" if self.pol.write_fixes else "writing fixes is off ([repair] write_fixes)"
            else:
                d["outcome"] = "its teacher didn't write a fix this time"
        self.diagnoses[name] = d
        line = f"{name} v{d['version']}: {SAY.get(cause, cause)} - {reason[:200]}"
        self._log(name, line)
        s.guardian.audit(by, "tool.diagnosed", name, cause=cause, why=reason[:300], probe=probe, fix=d["fix"])
        s.bus.publish("tool.diagnosed", tool=name, cause=cause, where=d["where"], why=reason[:300], fix=d["fix"])
        s.mind.note("observation", f"tool:{name}", f"The {name} tool fails because of {SAY.get(cause, cause)}: {reason[:240]}"
                    + (f" What to do: {d['advice']}." if cause != "code" and d["advice"] else ""),
                    private=False, source="tooldoctor")
        self.save()
        return {"ok": True, **d}

    @staticmethod
    def _matches(output, expect) -> bool:
        from .tools import outputs_match
        return outputs_match(output, expect)

    @staticmethod
    def _manifest_text(tool: Tool) -> str:
        try:
            return (tool.manifest.path / "manifest.toml").read_text(encoding="utf-8")
        except OSError:
            return ""

    # ----------------------------------------------------------------- fixing
    def _draft(self, tool: Tool, fixed: dict, files: dict, d: dict, explanation: str, teacher: str, by: str) -> dict | None:
        s, name = self.sys, tool.name
        problems = []
        changed = {}
        for path, text in fixed.items():
            if text == files.get(path):
                continue
            problems += check_fix(files.get(path, ""), text, path)
            changed[path] = text
        self.seq += 1
        fid = f"fix-{self.seq}"
        key = f"{name}@{tool.manifest.digest}"
        to_version = bump(tool.manifest.version)
        diff = "".join(diff_of(p, files.get(p, ""), t) for p, t in changed.items())
        fix = {"id": fid, "tool": name, "from_version": tool.manifest.version, "to_version": to_version,
               "digest_from": tool.manifest.digest, "digest_to": None, "cause": d["cause"], "explanation": explanation[:600],
               "teacher": teacher, "files": list(changed), "diff": diff[:20000], "status": "", "approval": "",
               "made_at": round(time.time(), 1), "by": by, "attempt": self.attempts.get(key, 0) + 1, "folder": "",
               "backup": "", "result": ""}
        self.fixes[fid] = fix
        if not changed:
            fix["status"], fix["result"] = "failed", "the teacher's fix changed nothing"
        elif problems:
            fix["status"], fix["result"] = "failed", "; ".join(problems[:4])
        if fix["status"] == "failed":
            self.attempts[key] = self.attempts.get(key, 0) + 1
            self._log(name, f"the fix {fid} it wrote was not offered: {fix['result']}")
            s.guardian.audit(by, "tool.fix_unsafe", name, fix=fid, problems=problems[:4])
            self.save()
            return fix
        folder = self.workshop / name / fid
        folder.mkdir(parents=True, exist_ok=True)
        for path, text in changed.items():
            (folder / path).write_text(text, encoding="utf-8")
        (folder / "FIX.diff").write_text(diff, encoding="utf-8")
        (folder / "FIX.md").write_text(f"# {name}: {fid}\n\nv{tool.manifest.version} -> v{to_version}\n\n"
                                       f"Why it failed: {d['why']}\n\nThe fix ({teacher}): {explanation}\n\n"
                                       f"Errors:\n" + "\n".join(f"- {e}" for e in d.get("errors", [])) + "\n", encoding="utf-8")
        fix["folder"] = str(folder)
        approval = s.guardian.request_approval("tool.fix", name, {
            "fix": fid, "version": tool.manifest.version, "to_version": to_version, "cause": d["cause"],
            "why": d["why"][:400], "explanation": explanation[:400], "files": list(changed), "diff": diff[:6000],
            "folder": str(folder), "teacher": teacher, "permissions": list(tool.manifest.permissions)})
        fix["approval"], fix["status"] = approval.id, "awaiting approval"
        q = s.mind.ask("approval", f"The {name} tool fails because of a fault in its code: {explanation[:220]} "
                                   f"I wrote a fix (v{to_version}, {len(diff.splitlines())} lines of diff in {folder}). "
                                   "May I install it and test it? (yes/no)",
                       f"{d['failures']} failed runs; {d['why'][:160]}", approval_id=approval.id)
        fix["inquiry"] = q.id if q is not None else ""
        self._log(name, f"wrote {fid} (v{to_version}); asked permission ({approval.id})")
        s.guardian.audit(by, "tool.fix_drafted", name, fix=fid, approval=approval.id, files=list(changed))
        s.bus.publish("tool.fix_drafted", tool=name, fix=fid, approval=approval.id, version=to_version)
        self.save()
        return fix

    def apply(self, fid: str, by: str) -> tuple[bool, str]:
        """A person approved a fix: back up the files, install the new ones, bump the version; it is tested again."""
        s = self.sys
        fix = self.fixes.get(fid)
        if fix is None:
            return False, f"no fix {fid}"
        tool = s.tools.tools.get(fix["tool"])
        if tool is None or tool.state == REMOVED:
            fix["status"] = "withdrawn"
            self.save()
            return False, f"{fix['tool']} is no longer in the mesh"
        if tool.manifest.digest != fix["digest_from"]:
            fix["status"], fix["result"] = "withdrawn", "the tool's files changed since the fix was written"
            self.save()
            return False, f"{fix['tool']}'s files changed since {fid} was written; it will look again"
        folder = tool.manifest.path
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.backups / fix["tool"] / f"v{tool.manifest.version}-{fid}-{stamp}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, backup, ignore=shutil.ignore_patterns("__pycache__"))
        for path in fix["files"]:
            shutil.copyfile(Path(fix["folder"]) / path, folder / path)
        mf = folder / "manifest.toml"
        text = mf.read_text(encoding="utf-8")
        if re.search(r'^version\s*=\s*"[^"]*"', text, re.M):
            text = re.sub(r'^version\s*=\s*"[^"]*"', f'version = "{fix["to_version"]}"', text, count=1, flags=re.M)
        else:
            text = re.sub(r'^(name\s*=.*)$', rf'\1\nversion = "{fix["to_version"]}"', text, count=1, flags=re.M)
        mf.write_text(text, encoding="utf-8")
        digest = folder_digest(folder)
        self.preapproved.setdefault(fix["tool"], [])
        for dg in (digest, fix["digest_from"]):  # the fix, and the backup if it goes back
            if dg not in self.preapproved[fix["tool"]]:
                self.preapproved[fix["tool"]].append(dg)
        self.preapproved[fix["tool"]] = self.preapproved[fix["tool"]][-6:]
        if tool.manifest.author == "neuromesh":
            s.toolsmith.record_approval(fix["tool"], digest, by)
        fix.update(status="applied", digest_to=digest, backup=str(backup), approved_by=by, applied_at=round(time.time(), 1))
        key = f"{fix['tool']}@{fix['digest_from']}"
        self.attempts[key] = self.attempts.get(key, 0) + 1
        self._prune_backups(fix["tool"])
        self._log(fix["tool"], f"{fid} approved by {by}: v{fix['to_version']} installed; the old files are in {backup}")
        s.guardian.audit(by, "tool.fix_applied", fix["tool"], fix=fid, version=fix["to_version"], backup=str(backup))
        s.bus.publish("tool.fix_applied", tool=fix["tool"], fix=fid, version=fix["to_version"], by=by)
        self.save()
        return True, f"installed {fix['tool']} v{fix['to_version']}; it is self-tested next (the old files are kept)"

    async def _rollback(self, fix: dict, reason: str) -> None:
        """The fix didn't pass the self-tests: the backup goes back."""
        s = self.sys
        backup = Path(fix.get("backup") or "")
        tool = s.tools.tools.get(fix["tool"])
        fix.update(status="rolled back", result=f"v{fix['to_version']} failed its self-tests ({reason})",
                   settled_at=round(time.time(), 1))
        if tool is not None and backup.is_dir():
            folder = tool.manifest.path
            for p in backup.iterdir():
                dest = folder / p.name
                if p.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                    shutil.copytree(p, dest)
                else:
                    shutil.copyfile(p, dest)
            for p in folder.iterdir():  # files the fix added
                if p.name != "__pycache__" and not (backup / p.name).exists():
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
        key = f"{fix['tool']}@{fix['digest_from']}"
        left = self.pol.attempts - self.attempts.get(key, 0)
        d = self.diagnoses.get(fix["tool"])
        if d is not None:
            d["outcome"] = (f"the fix {fix['id']} didn't work and was undone; " +
                            ("it will try again" if left > 0 else "it has given up: fix it by hand or remove it"))
            d["again"] = left > 0  # when the old version fails its self-tests again, it looks again
        self._log(fix["tool"], f"{fix['id']} didn't pass the self-tests; v{fix['from_version']} is back")
        s.guardian.audit("tooldoctor", "tool.fix_rolled_back", fix["tool"], fix=fix["id"], reason=reason[:200])
        s.bus.publish("tool.fix_rolled_back", tool=fix["tool"], fix=fix["id"], reason=reason[:200])
        self.save()

    def _prune_backups(self, name: str) -> None:
        folder = self.backups / name
        if not folder.is_dir():
            return
        olds = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime)
        for p in olds[:-max(1, self.pol.keep_backups)]:
            shutil.rmtree(p, ignore_errors=True)

    def preapproved_digest(self, name: str, digest: str) -> bool:
        """Files a person approved through a fix (or that a fix replaced): no second approval for them."""
        return digest in self.preapproved.get(name, [])

    def withdraw(self, name: str, why: str, by: str) -> None:
        for fix in self.fixes.values():
            if fix["tool"] == name and fix["status"] == "awaiting approval":
                fix["status"], fix["result"] = "withdrawn", why
                if fix.get("approval"):
                    self.sys.guardian.withdraw(fix["approval"], why, by)
        self.save()

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {"enabled": self.enabled, "diagnoses": self.diagnoses,
                "fixes": sorted(self.fixes.values(), key=lambda f: -f.get("made_at", 0))[:30],
                "history": list(self.history)[-30:], "llm": self.sys.teachers.has_llm(),
                "attempts": self.pol.attempts}

    def lines(self, name: str = "") -> list[str]:
        out = []
        for tool, d in sorted(self.diagnoses.items()):
            if name and tool != name:
                continue
            out.append(f"  {tool} v{d['version']}: {SAY.get(d['cause'], d['cause'])} - {d['why'][:160]}")
            if d.get("probe"):
                out.append(f"    its server: {d['probe']}")
            if d.get("outcome"):
                out.append(f"    {d['outcome']}")
            elif d.get("advice") and d["cause"] != "code":
                out.append(f"    what to do: {d['advice']}")
        for f in sorted(self.fixes.values(), key=lambda f: -f.get("made_at", 0))[:8]:
            if name and f["tool"] != name:
                continue
            out.append(f"  {f['id']} {f['tool']} v{f['from_version']} -> v{f['to_version']}: {f['status']}"
                       + (f" ({f['result'][:100]})" if f.get("result") else ""))
        return out or ["  no tool has failed enough to look into"]
