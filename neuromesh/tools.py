"""Tools: plugins the system discovers, vets, tests, learns to route to, and retires.

A tool is a folder containing manifest.toml plus an entrypoint script. The script
reads {"input": "<order text>", "config": {...}} as JSON on stdin and prints {"output": ...} or
{"error": "..."} on stdout. Tools run in a subprocess with CPU/memory limits, a clean
environment (no API keys), and their own folder as working directory.

Any tool can loop back to a director: instead of an answer it prints
{"query": {"ask": "What is the speed of light?", "about": "speed of light", "why": "..."}}.
A director answers from what the mesh holds (and, within budgets, its research tools or teachers),
and the tool is run again with "context": [{"ask", "answer", "source"}, ...] in its input, so it
can use the answer (or see that no one could give one). A run may ask a few times
([directors] tool_queries); every question and answer is in the order's trace.

Note: this sandbox limits resources and environment, not syscalls. For untrusted
third-party tools run the system inside a container/VM (see README).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ACTIVE, PENDING, APPROVED, REJECTED, QUARANTINED, RETIRED, DISCOVERED = (
    "active", "pending_approval", "approved", "rejected", "quarantined", "retired", "discovered")


def folder_digest(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(path.relative_to(folder)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class ToolManifest:
    name: str
    path: Path
    digest: str
    version: str = "0.0.0"
    description: str = ""
    entrypoint: str = "tool.py"
    permissions: tuple = ()
    keywords: tuple = ()
    timeout_s: float | None = None
    supersedes: str | None = None
    provides: str = ""            # "knowledge": results are retained and recalled later
    examples: list = field(default_factory=list)
    templates: list = field(default_factory=list)
    fillers: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)  # passed to the tool on every run
    author: str = ""                            # "neuromesh": its own toolsmith wrote it (P15)
    persistent: bool = False                    # it can stay running (tool.py --serve): see Worker
    secrets: tuple = ()                         # environment variables it is given (e.g. GITEA_TOKEN), shown on its approval

    @property
    def route(self) -> str:
        return f"tool:{self.name}"

    @classmethod
    def load(cls, folder: Path) -> "ToolManifest":
        with open(folder / "manifest.toml", "rb") as fh:
            m = tomllib.load(fh)
        return cls(name=m["name"].lower(), path=folder, digest=folder_digest(folder),
                   version=m.get("version", "0.0.0"), description=m.get("description", ""),
                   entrypoint=m.get("entrypoint", "tool.py"), permissions=tuple(m.get("permissions", [])),
                   keywords=tuple(m.get("keywords", [])), timeout_s=m.get("timeout_s"),
                   supersedes=m.get("supersedes") or None, provides=m.get("provides", ""),
                   examples=list(m.get("examples", [])), templates=list(m.get("templates", [])),
                   fillers=dict(m.get("fillers", {})), config=dict(m.get("config", {})),
                   author=str(m.get("author", "")), persistent=bool(m.get("persistent", False)),
                   secrets=tuple(str(x) for x in m.get("secrets", []) if str(x).strip()))


@dataclass
class Tool:
    manifest: ToolManifest
    state: str = DISCOVERED
    approval_id: str | None = None
    runs: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    activated_at: float | None = None
    last_health: float = 0.0
    note: str = ""

    @property
    def name(self) -> str:
        return self.manifest.name

    def to_meta(self) -> dict:
        m = self.manifest
        return {"name": m.name, "version": m.version, "description": m.description,
                "permissions": list(m.permissions), "path": str(m.path), "state": self.state,
                "runs": self.runs, "successes": self.successes, "last_error": self.last_error,
                "activated_at": self.activated_at, "supersedes": m.supersedes, "note": self.note}


@dataclass
class ToolRun:
    ok: bool
    output: object = None
    error: str = ""
    ms: float = 0.0
    extra: dict = field(default_factory=dict)   # everything else the tool returned (e.g. "knowledge")
    transient: bool = False                     # e.g. network down: does not count towards quarantine
    not_found: bool = False                     # the tool works but has no answer (e.g. no such article)
    query: dict | None = None                   # the tool asks a director for context before it can answer
    refused: bool = False                       # the tool switch is off for it: it never ran (toolgate.py)


_LAUNCHER = """
import runpy, sys
cpu, mem, entry = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if mem > 0:
        resource.setrlimit(resource.RLIMIT_AS, (mem * 1048576, mem * 1048576))
except Exception:
    pass
sys.argv = [entry] + sys.argv[4:]
runpy.run_path(entry, run_name="__main__")
"""


class Worker:
    """A tool that stays running (manifest: persistent = true; it is started as tool.py --serve): requests go
    to it as JSON lines, many at once, each with an id, and its answers come back the same way. It keeps
    its connections and caches warm between requests - the Kiwix tool answers many times faster so. It runs
    in the same sandbox as a single run (limits, a clean environment, its own folder); its CPU limit is a
    budget for many requests, and when it is spent, the worker dies and a fresh one takes over."""

    def __init__(self, tool: "Tool", proc, digest: str):
        self.tool, self.proc, self.digest = tool, proc, digest
        self.pending: dict[int, asyncio.Future] = {}
        self.next_id = 1
        self.served = 0
        self.started = time.time()
        self.reader = asyncio.create_task(self._read())

    @property
    def alive(self) -> bool:
        return self.proc.returncode is None and not self.reader.done()

    async def _read(self) -> None:
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fut = self.pending.pop(int(data.pop("id", -1)), None) if isinstance(data, dict) else None
                if fut is not None and not fut.done():
                    fut.set_result(data)
        finally:
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("the tool's worker stopped"))
            self.pending.clear()

    async def call(self, request: dict, timeout: float) -> dict:
        rid = self.next_id
        self.next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        self.proc.stdin.write((json.dumps({**request, "id": rid}) + "\n").encode())
        await self.proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.pending.pop(rid, None)
            self.served += 1

    async def stop(self) -> None:
        if self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), 2)
            except Exception:
                self.proc.kill()
                await self.proc.wait()
        self.reader.cancel()


class Sandbox:
    def __init__(self, cpu_seconds: int = 5, memory_mb: int = 512, workers: bool = True, worker_cpu_seconds: int = 600,
                 worker_requests: int = 5000):
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.use_workers = workers                  # [tools] workers: persistent tools stay running
        self.gate = None                            # tool -> why it may not run (the tool switch), or None
        self.withheld: set[str] = set()             # secrets no tool is ever given (its teachers' API keys)
        self.worker_cpu_seconds = worker_cpu_seconds
        self.worker_requests = worker_requests      # a worker is replaced after this many requests
        self.workers: dict[str, Worker] = {}
        self._starting: dict[str, asyncio.Lock] = {}

    NETWORK_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
                   "SSL_CERT_FILE", "SSL_CERT_DIR")

    def _env(self, m: ToolManifest) -> dict:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONIOENCODING": "utf-8",
               "LANG": "C.UTF-8", "NEUROMESH_TOOL": m.name}
        if "network" in m.permissions:  # proxy/certificate settings only for approved network tools
            env.update({k: os.environ[k] for k in self.NETWORK_ENV if k in os.environ})
        for k in m.secrets:  # only the secrets its manifest lists, which a person saw on its approval
            if k in os.environ and k not in self.withheld:
                env[k] = os.environ[k]
        return env

    async def _worker(self, tool: Tool) -> Worker:
        m = tool.manifest
        w = self.workers.get(m.name)
        if w is not None and w.alive and w.digest == m.digest and w.served < self.worker_requests:
            return w
        lock = self._starting.setdefault(m.name, asyncio.Lock())
        async with lock:
            w = self.workers.get(m.name)
            if w is not None and w.alive and w.digest == m.digest and w.served < self.worker_requests:
                return w
            if w is not None:
                await w.stop()
            entry = (m.path / m.entrypoint).resolve()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c", _LAUNCHER, str(self.worker_cpu_seconds), str(self.memory_mb), str(entry),
                "--serve", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, cwd=str(m.path), env=self._env(m), limit=1 << 24)
            w = Worker(tool, proc, m.digest)
            self.workers[m.name] = w
            return w

    async def close(self) -> None:
        for w in list(self.workers.values()):
            await w.stop()
        self.workers.clear()

    def worker_status(self) -> dict:
        return {name: {"alive": w.alive, "served": w.served, "since": round(w.started, 1), "pid": w.proc.pid}
                for name, w in self.workers.items()}

    async def run(self, tool: Tool, text: str, timeout: float, config: dict | None = None,
                  context: list | None = None) -> ToolRun:
        m = tool.manifest
        why = self.gate(tool) if self.gate is not None else None
        if why:  # the tool switch is off for it: it does not run, for anything (toolgate.py)
            return ToolRun(False, error=why, refused=True)
        entry = (m.path / m.entrypoint).resolve()
        if m.path.resolve() not in entry.parents:
            return ToolRun(False, error="entrypoint escapes the tool folder")
        if m.persistent and self.use_workers:  # it stays running: no start-up, warm connections and caches
            t0 = time.perf_counter()
            request = {"input": text, "config": config if config is not None else dict(m.config)}
            if context:
                request["context"] = context
            try:
                w = await self._worker(tool)
                data = await w.call(request, timeout)
            except asyncio.TimeoutError:
                return ToolRun(False, error=f"timeout after {timeout}s", ms=(time.perf_counter() - t0) * 1000)
            except Exception as exc:  # the worker died (its CPU budget, a crash): a fresh one next time
                return ToolRun(False, error=f"the tool's worker stopped ({exc})", ms=(time.perf_counter() - t0) * 1000,
                               transient=True)
            return self._result(data, (time.perf_counter() - t0) * 1000)
        env = self._env(m)
        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", _LAUNCHER, str(self.cpu_seconds), str(self.memory_mb), str(entry),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(m.path), env=env)
        try:
            request = {"input": text, "config": config if config is not None else dict(m.config)}
            if context:
                request["context"] = context
            out, err = await asyncio.wait_for(proc.communicate(json.dumps(request).encode()), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolRun(False, error=f"timeout after {timeout}s", ms=(time.perf_counter() - t0) * 1000)
        ms = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            return ToolRun(False, error=(err.decode(errors="replace")[-400:] or f"exit {proc.returncode}"), ms=ms)
        lines = out.decode(errors="replace").strip().splitlines()
        try:
            data = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            return ToolRun(False, error="tool printed invalid JSON", ms=ms)
        return self._result(data, ms)

    @staticmethod
    def _result(data: dict, ms: float) -> ToolRun:
        if not isinstance(data, dict):
            return ToolRun(False, error="tool printed invalid JSON", ms=ms)
        if "error" in data:
            return ToolRun(False, error=str(data["error"]), ms=ms, transient=bool(data.get("transient")),
                           not_found=bool(data.get("not_found")))
        if data.get("query"):  # it asks a director for context (see the module docstring)
            q = data["query"] if isinstance(data["query"], dict) else {"ask": str(data["query"])}
            q = {k: str(v)[:300] for k, v in q.items() if k in ("ask", "about", "why")}
            if not q.get("ask"):
                return ToolRun(False, error="tool asked an empty question", ms=ms)
            return ToolRun(False, error=f"it needs to know: {q['ask']}", ms=ms, query=q, transient=True)
        extra = {k: v for k, v in data.items() if k != "output"}
        return ToolRun(True, output=data.get("output"), ms=ms, extra=extra)


def outputs_match(output: object, expect: object) -> bool:
    if expect is None:
        return True
    try:
        return abs(float(output) - float(expect)) <= 1e-6 * max(1.0, abs(float(expect)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(expect).strip().lower() in str(output).strip().lower()


class ToolRegistry:
    def __init__(self, root: Path, plugin_dirs: tuple, store):
        self.root = Path(root)
        self.dirs = [(self.root / d) for d in plugin_dirs]
        for d in self.dirs:
            d.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.tools: dict[str, Tool] = {}
        self._persisted = store.load_tools()

    def persist(self, tool: Tool) -> None:
        self.store.save_tool(tool.name, tool.state, tool.manifest.digest, tool.to_meta())

    def active(self) -> list[Tool]:
        return [t for t in self.tools.values() if t.state == ACTIVE]

    def scan(self) -> list[tuple[ToolManifest | None, str, str]]:
        """Return (manifest, change, name) for new, changed, restored or removed tools."""
        changes: list[tuple[ToolManifest | None, str, str]] = []
        seen: set[str] = set()
        for d in self.dirs:
            for folder in sorted(p for p in d.iterdir() if (p / "manifest.toml").is_file()):
                try:
                    manifest = ToolManifest.load(folder)
                except Exception as exc:  # malformed manifest: report, don't crash
                    changes.append((None, f"invalid: {exc}", folder.name))
                    continue
                seen.add(manifest.name)
                existing = self.tools.get(manifest.name)
                if existing is None:
                    prior = self._persisted.get(manifest.name)
                    if prior and prior["digest"] == manifest.digest and prior["state"] in (ACTIVE, REJECTED, RETIRED):
                        tool = Tool(manifest, prior["state"])
                        tool.runs = prior["meta"].get("runs", 0)
                        tool.successes = prior["meta"].get("successes", 0)
                        self.tools[manifest.name] = tool
                        changes.append((manifest, "restored", manifest.name))
                    else:
                        changes.append((manifest, "new", manifest.name))
                elif existing.manifest.digest != manifest.digest:
                    changes.append((manifest, "changed", manifest.name))
        for name, tool in self.tools.items():
            if name not in seen and tool.state not in (RETIRED, REJECTED):
                changes.append((None, "removed", name))
        return changes
