"""Code challenges: programming problems it downloads from the Hugging Face Hub and tries to solve (P20).

    a suite (it gets the ones in [codechallenges] start_with by itself; /codechallenges get <suite>)
      mbpp        google-research-datasets/mbpp (sanitized): "Write a python function to ...", a reference solution,
                  and the asserts that test it. Its train split is practice, its test split the exam
      humaneval   openai/openai_humaneval: a function's signature and doc comment to complete, and a check(candidate)
                  that tests it. Split once, stably, into practice and exam
    an attempt (the Solver, when idle, every [codechallenges] every_seconds; one challenge, never one it has done)
      --> it writes a solution in its own ways: a recipe, a program its workbench built, the closest code it has
          studied (an LLM teacher only with [codechallenges] teacher) - and names it what the challenge calls it
      --> the solution and the challenge's own tests become one program, checked by the scratch pad's validator, and
          it runs only as P20 allows: with Running its code "When harmless" it runs by itself when it reaches for
          nothing; otherwise it waits in the scratch pad for you to run it
      --> solved (every test passed), failed (which tests failed is kept), or no attempt (it knew no way to write it)
      --> a practice challenge then teaches it: its reference solution is studied like code from Gitea (its coder
          nodes learn the purpose, the function becomes an example it can draw on). An exam challenge's solution is
          never studied, so its exam score says what it can do on problems it has not seen
A challenge is attempted once. One it failed or could not attempt is gone back to only by its works (works.py:
revisit), when what it has studied has grown since. What it solved becomes a long-term project of its own: a
library of its solutions with their tests (works.py).
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
from collections import Counter, deque

from .agents import Agent
from .challenges import _get_json

log = logging.getLogger("neuromesh.codechallenges")

BY = "its code challenges"
SUITES = {
    "mbpp": {"dataset": "google-research-datasets/mbpp", "config": "sanitized", "practice": "train", "exam": "test",
             "license": "cc-by-4.0", "description": "Mostly Basic Python Problems: a sentence to implement, with asserts"},
    "humaneval": {"dataset": "openai/openai_humaneval", "config": "openai_humaneval", "practice": "test", "exam": "test",
                  "license": "mit", "description": "HumanEval: a function to complete from its signature and doc comment"},
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS code_challenges(id TEXT PRIMARY KEY, suite TEXT, task TEXT, prompt TEXT, entry TEXT, tests TEXT,
    setup TEXT, solution TEXT, split TEXT, status TEXT DEFAULT 'new', tries INTEGER DEFAULT 0, ts REAL, item TEXT DEFAULT '',
    how TEXT DEFAULT '', passed INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, code TEXT DEFAULT '', note TEXT DEFAULT '',
    studied INTEGER DEFAULT 0, basis TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS code_challenges_status ON code_challenges(status);
"""
KEYS = ("id", "suite", "task", "prompt", "entry", "tests", "setup", "solution", "split", "status", "tries", "ts", "item",
        "how", "passed", "failed", "code", "note", "studied", "basis")


def _defs(code: str) -> list[str]:
    try:
        return [n.name for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except SyntaxError:
        return []


def from_row(suite: str, row: dict, split: str) -> dict | None:
    """A dataset row as a challenge: what to write, what it is called, the tests, the reference solution."""
    if "test_list" in row:  # mbpp
        code = str(row.get("code") or "")
        tests = [t for t in row.get("test_list") or [] if isinstance(t, str) and t.strip().startswith("assert")]
        defs = _defs(code)
        entry = next((d for d in defs if any(re.search(rf"\b{re.escape(d)}\(", t) for t in tests)), defs[0] if defs else "")
        prompt = str(row.get("prompt") or row.get("text") or "").strip()
        setup = "\n".join(row.get("test_imports") or []) or str(row.get("test_setup_code") or "")
        if not (prompt and entry and tests):
            return None
        return {"task": str(row.get("task_id")), "prompt": prompt, "entry": entry, "tests": tests, "setup": setup,
                "solution": code, "split": split}
    if "canonical_solution" in row and row.get("entry_point"):  # humaneval
        head = str(row.get("prompt") or "")
        doc = ""
        try:
            fn = next(n for n in ast.parse(head + "    pass\n").body if isinstance(n, ast.FunctionDef) and n.name == row["entry_point"])
            doc = ast.get_docstring(fn) or ""
        except (SyntaxError, StopIteration):
            pass
        said = " ".join(doc.split(">>>")[0].split()) or f"complete the function {row['entry_point']}"
        task = str(row.get("task_id"))
        held = int(hashlib.sha256(task.encode()).hexdigest()[:6], 16) % 2 == 0  # split once, stably
        return {"task": task, "prompt": f"Write a python function {row['entry_point']}: {said}", "entry": str(row["entry_point"]),
                "tests": [str(row.get("test") or "")], "setup": "check", "solution": head + str(row["canonical_solution"]),
                "split": "exam" if held else "practice"}
    return None


class CodeChallenges:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.codechallenges
        self.db = system.store.db
        self.db.executescript(SCHEMA)
        raw = system.store.get_kv("codechallenges.state")
        st = json.loads(raw) if raw else {}
        self.stats: Counter = Counter(st.get("stats", {}))
        self.log: deque = deque(st.get("log", []), maxlen=60)
        self.tried: dict[str, float] = st.get("tried", {})
        self.offsets: dict[str, int] = st.get("offsets", {})
        self.getting: set[str] = set()
        self.busy = False

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled) and self.sys.scratch.enabled

    def save(self) -> None:
        self.sys.store.set_kv("codechallenges.state", json.dumps({"stats": dict(self.stats), "log": list(self.log),
                                                                  "tried": self.tried, "offsets": self.offsets}))

    def _note(self, line: str, **extra) -> None:
        self.log.append({"ts": round(time.time(), 1), "line": line[:300], **extra})
        self.save()

    def get(self, cid: str) -> dict | None:
        row = self.db.execute(f"SELECT {', '.join(KEYS)} FROM code_challenges WHERE id = ?", (cid,)).fetchone()
        if row is None:
            return None
        d = dict(zip(KEYS, row))
        d["tests"] = json.loads(d["tests"] or "[]")
        return d

    def count(self, where: str = "", args: tuple = ()) -> int:
        return self.db.execute("SELECT COUNT(*) FROM code_challenges" + (f" WHERE {where}" if where else ""), args).fetchone()[0]

    # ------------------------------------------------------------------ downloading
    async def fetch(self, suite: str, limit: int = 0, by: str = BY) -> dict:
        """Download more of a suite from the Hub's dataset viewer."""
        if suite not in SUITES:
            raise ValueError(f"no code challenge suite '{suite}' (suites: {', '.join(SUITES)})")
        if suite in self.getting:
            raise ValueError(f"it is already getting {suite}")
        self.getting.add(suite)
        try:
            with self.sys.activity.track("challenges", "coder", f"getting code challenges: {suite}"):
                added = await asyncio.to_thread(self._fetch, suite, max(1, int(limit or self.pol.fetch_limit)))
        except Exception as exc:
            self._note(f"could not get {suite}: {exc}", suite=suite)
            raise
        finally:
            self.getting.discard(suite)
        rows, n = added
        for r in rows:
            self.db.execute("INSERT OR IGNORE INTO code_challenges(id, suite, task, prompt, entry, tests, setup, solution, split, ts) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)", (f"{suite}:{r['task']}", suite, r["task"], r["prompt"][:2000], r["entry"],
                                                              json.dumps(r["tests"]), r["setup"], r["solution"][:20000], r["split"], time.time()))
        have = self.count("suite = ?", (suite,))
        line = f"got {len(rows)} code challenge(s) of {suite} ({SUITES[suite]['dataset']}): {have} in all"
        self.stats["downloaded"] += len(rows)
        self._note(line, suite=suite)
        self.sys.guardian.audit(by, "codechallenges.get", suite, challenges=len(rows))
        self.sys.bus.publish("codechallenges.got", suite=suite, challenges=len(rows), by=by)
        return {"suite": suite, "added": len(rows), "have": have, "line": line}

    def _fetch(self, suite: str, limit: int) -> tuple[list[dict], int]:
        c = self.sys.directives.challenges
        token = (os.environ.get(c.token_env) or None) if c.token_env else None
        spec = SUITES[suite]
        out: list[dict] = []
        splits = list(dict.fromkeys([("practice", spec["practice"]), ("exam", spec["exam"])]))
        if spec["practice"] == spec["exam"]:
            splits = [("", spec["practice"])]
        per = max(1, limit // len(splits))
        for role, split in splits:
            key = f"{suite}:{split}"
            offset, got = int(self.offsets.get(key, 0)), 0
            while got < per:
                q = urllib.parse.urlencode({"dataset": spec["dataset"], "config": spec["config"], "split": split,
                                            "offset": offset, "length": min(100, per - got)})
                data = _get_json(f"{c.api_url.rstrip('/')}/rows?{q}", token, timeout=60)
                rows = [r["row"] for r in data.get("rows") or [] if isinstance(r, dict) and isinstance(r.get("row"), dict)]
                if not rows:
                    break
                offset += len(rows)
                for row in rows:
                    ch = from_row(suite, row, role or "practice")
                    if ch is not None:
                        out.append(ch)
                        got += 1
                if offset >= int(data.get("num_rows_total") or 0):
                    break
            self.offsets[key] = offset
        return out, len(out)

    # ------------------------------------------------------------------ an attempt
    def next_new(self) -> str | None:
        """A challenge it has never attempted: practice ones first while it has studied few, then turn about."""
        want = "practice" if self.count("status != 'new' AND split = 'practice'") <= 2 * self.count("status != 'new' AND split = 'exam'") else "exam"
        for split in (want, "exam" if want == "practice" else "practice"):
            row = self.db.execute("SELECT id FROM code_challenges WHERE status = 'new' AND split = ? ORDER BY ts, id LIMIT 1", (split,)).fetchone()
            if row:
                return row[0]
        return None

    def _program(self, ch: dict, code: str) -> str:
        from .workbench import RUNNER
        lines = [f'"""{ch["id"]}: {ch["prompt"][:200]}\n\nIts own solution, with the challenge\'s tests ({ch["suite"]}, from the '
                 f'Hugging Face Hub)."""', code.rstrip(), ""]
        main = []
        if ch["setup"] == "check":  # humaneval: the test defines check(candidate)
            lines += [ch["tests"][0].strip(), ""]
            main.append(f'_report("TEST", "check({ch["entry"]})", lambda: check({ch["entry"]}))')
        else:
            if ch["setup"]:
                lines += [ch["setup"], ""]
            for i, t in enumerate(ch["tests"], 1):
                lines += [f"def _test_{i}():", f"    {t.strip()}", ""]
                main.append(f'_report("TEST", {("test %d: " % i + t.strip()[7:67])!r}, _test_{i})')
        return "\n".join(lines) + RUNNER + '\nif __name__ == "__main__":\n' + "\n".join(f"    {m}" for m in main) + "\n"

    async def _write(self, ch: dict, way: int = 0) -> tuple[str | None, str]:
        """Its own solution, named what the challenge calls it: (code, how it was written). way 0: its first choice
        (a recipe, a teacher if allowed, its workbench's programs, the closest example); way 1: without its recipes."""
        s = self.sys
        code, origin = await s.scratch.draft(ch["prompt"], "python", teacher=self.pol.teacher, exclude=None if way == 0 else -1)
        if not code:
            return None, "it knew no way to write it"
        how = (origin or {}).get("how", "its own way")
        defs = _defs(code)
        if not defs:
            return None, "what it wrote defines no function"
        if ch["entry"] not in defs:
            public = [d for d in defs if not d.startswith(("_", "test"))] or defs
            code = code.rstrip() + f"\n\n\n{ch['entry']} = {public[-1] if how == 'workbench' and len(public) > 1 else public[0]}  # what the challenge calls it\n"
        return code, how

    async def attempt(self, cid: str = "", by: str = BY, again: bool = False) -> dict | None:
        """Attempt one challenge: the one named, or the next it has never attempted."""
        s = self.sys
        if not self.enabled or self.busy:
            return None
        cid = cid or self.next_new() or ""
        ch = self.get(cid)
        if ch is None or (ch["status"] != "new" and not again) or ch["status"] in ("solved", "waiting"):
            return None
        self.busy = True
        try:
            with s.activity.track("challenges", "coder", f"attempting {cid}: {ch['prompt'][:60]}"):
                code, how = await self._write(ch)
                rec = {"id": cid, "suite": ch["suite"], "split": ch["split"], "prompt": ch["prompt"], "how": how}
                if code is None:
                    self._set(cid, status="no attempt", how=how, note=how)
                    rec.update(status="no attempt", line=f"{cid} ({ch['split']}): no attempt - {how}")
                else:
                    program = self._program(ch, code)
                    check = s.scratch.validator.check(program, "python")
                    if not check["ok"]:
                        self._set(cid, status="failed", how=how, code=code, note="; ".join(check.get("problems", [])[:2]))
                        rec.update(status="failed", line=f"{cid} ({ch['split']}): what it wrote ({how}) does not check out")
                    else:
                        self._room()
                        item = s.scratch._new(f"{cid}: {ch['prompt'][:60]}", program, "python", by, request=ch["prompt"],
                                              origin={"how": "challenge", "challenge": cid, "written": how}, kind="challenge")
                        s.scratch._save(item)
                        self._set(cid, status="waiting", how=how, code=code, item=item["id"])
                        rec["item"] = item["id"]
                        if s.scratch.auto_runs(item)[0]:
                            run = await s.scratch.auto_run(item)
                            rec.update(self._grade(cid, run or {}))
                            if rec["status"] == "failed":  # another way of its own, when the first did not pass
                                code2, how2 = await self._write(ch, way=1)
                                prog2 = self._program(ch, code2) if code2 and code2.strip() != code.strip() else None
                                if prog2 and s.scratch.validator.check(prog2, "python")["ok"]:
                                    s.scratch.edit(item["id"], prog2, by)
                                    item["origin"] = {"how": "challenge", "challenge": cid, "written": how2}
                                    if s.scratch.auto_runs(item)[0]:
                                        self.db.execute("UPDATE code_challenges SET how = ?, code = ? WHERE id = ?", (how2, code2, cid))
                                        run = await s.scratch.auto_run(item)
                                        second = self._grade(cid, run or {})
                                        second["line"] = second["line"].replace(": ", f": its first way ({how}) failed; ", 1)
                                        rec.update(second, how=how2)
                        else:
                            rec.update(status="waiting", line=f"{cid} ({ch['split']}): it wrote a solution ({how}); it waits in the "
                                                              f"scratch pad as {item['id']} to be run ({s.scratch.auto_runs(item)[1]})")
            self.stats["attempts"] += 1
            self.stats[rec["status"]] += 1
            if rec["status"] != "waiting":
                rec["line"] += self._study(cid)
            self._note(rec["line"], challenge=cid)
            s.bus.publish("codechallenges.attempted", id=cid, status=rec["status"], split=ch["split"], how=how, by=by)
            return rec
        finally:
            self.busy = False

    def _set(self, cid: str, **fields) -> None:
        fields.setdefault("ts", time.time())
        fields["basis"] = self.basis()
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE code_challenges SET {sets}, tries = tries + ? WHERE id = ?",
                        (*fields.values(), 1 if "how" in fields else 0, cid))

    def basis(self) -> str:
        """How much code it has studied: when this grows, a challenge it could not solve is worth another look."""
        n = self.db.execute("SELECT COUNT(*) FROM code_examples").fetchone()[0]
        return str(n if n < 100 else 100 + n // 20)

    def _room(self) -> None:
        pad = self.sys.scratch
        waiting = sorted((i for i in pad.items.values() if i.get("kind") == "challenge" and i["status"] not in ("running", "approved")
                          and not i.get("approved_digest")), key=lambda i: i.get("created", 0))
        while len(waiting) >= self.pol.max_in_pad:
            old = waiting.pop(0)
            pad.remove(old["id"], BY)
            self.db.execute("UPDATE code_challenges SET status = CASE status WHEN 'waiting' THEN 'new' ELSE status END, item = '' "
                            "WHERE item = ?", (old["id"],))

    def _grade(self, cid: str, run: dict) -> dict:
        from .workbench import outcome
        ch = self.get(cid)
        out = outcome(run.get("stdout", ""))
        bad = out["failed"] + out["errors"]
        solved = bool(run.get("ok")) and out["passed"] > 0 and bad == 0
        failing = [ln for ln in out["lines"] if ln.startswith(("FAIL", "ERROR"))][:2]
        note = "" if solved else ("; ".join(failing) or (run.get("stderr") or "it did not finish").strip().splitlines()[-1][:160])
        self._set(cid, status="solved" if solved else "failed", passed=out["passed"], failed=bad, note=note)
        return {"status": "solved" if solved else "failed", "passed": out["passed"], "failed": bad,
                "line": f"{cid} ({ch['split']}): " + (f"solved ({ch['how']}): all {out['passed']} test(s) passed" if solved else
                                                      f"not solved ({ch['how']}): {out['passed']} test(s) passed, {bad} failed")}

    def _study(self, cid: str) -> str:
        """A practice challenge teaches it once it has been attempted: its reference solution, studied like Gitea's code."""
        ch = self.get(cid)
        if ch is None or ch["split"] != "practice" or ch["studied"] or not self.pol.study_solutions or not ch["solution"]:
            return ""
        from .gitea import functions_in, kind_of, purpose_of
        s = self.sys
        repo, n = f"hub:{ch['suite']}", 0
        for fn in functions_in(ch["solution"], "python"):
            main = fn["name"] == ch["entry"]
            purpose = ch["prompt"] if main else purpose_of(fn["name"], fn["doc"], fn["args"])
            kind = kind_of(fn)
            self.db.execute("INSERT INTO code_examples(repo, path, sha, lang, name, kind, purpose, signature, doc, source, lines, "
                            "calls, ts, practised) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                            (repo, ch["task"], "", "python", fn["name"], kind, purpose[:300], fn["signature"][:300], fn["doc"][:1000],
                             fn["source"][:20000], fn["lines"], json.dumps(fn["calls"][:60]), time.time()))
            s.code_study.learn_from(purpose, kind)
            n += 1
        self.db.execute("UPDATE code_challenges SET studied = 1 WHERE id = ?", (cid,))
        s.code_study._vectors = None
        self.stats["solutions studied"] += 1 if n else 0
        return "; it then studied the reference solution" if n else ""

    def on_event(self, ev) -> None:
        """A person ran a waiting solution: grade it."""
        if ev.topic != "code.ran" or str(ev.data.get("by", "")).startswith("auto-run"):
            return
        item = self.sys.scratch.items.get(str(ev.data.get("id", "")))
        cid = ((item or {}).get("origin") or {}).get("challenge")
        if not cid or not item.get("runs"):
            return
        ch = self.get(cid)
        if ch is None or ch["status"] != "waiting":
            return
        res = self._grade(cid, item["runs"][-1])
        self.stats[res["status"]] += 1
        self.stats["waiting"] -= 1
        self._note(res["line"] + self._study(cid), challenge=cid)

    def retry_candidates(self) -> list[dict]:
        """Challenges it failed or could not attempt, now that it has studied more (for works.py)."""
        now = self.basis()
        rows = self.db.execute("SELECT id, prompt, status, note FROM code_challenges WHERE status IN ('failed', 'no attempt') "
                               "AND basis != ? ORDER BY ts LIMIT 20", (now,)).fetchall()
        return [{"id": r[0], "prompt": r[1], "status": r[2], "note": r[3]} for r in rows]

    def solved(self) -> list[dict]:
        rows = self.db.execute("SELECT id FROM code_challenges WHERE status = 'solved' ORDER BY id").fetchall()
        return [self.get(r[0]) for r in rows]

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        suites = []
        for name, spec in SUITES.items():
            c = dict(self.db.execute("SELECT split || ':' || status, COUNT(*) FROM code_challenges WHERE suite = ? GROUP BY 1", (name,)).fetchall())
            def n(split, *st):
                return sum(c.get(f"{split}:{x}", 0) for x in st)
            done_e = n("exam", "solved", "failed", "no attempt")
            suites.append({"name": name, **{k: spec[k] for k in ("dataset", "description", "license")},
                           "have": sum(c.values()), "new": n("practice", "new") + n("exam", "new"),
                           "practice": {"solved": n("practice", "solved"), "attempted": n("practice", "solved", "failed", "no attempt")},
                           "exam": {"solved": n("exam", "solved"), "attempted": done_e,
                                    "score": round(n("exam", "solved") / done_e, 3) if done_e else None},
                           "waiting": n("practice", "waiting") + n("exam", "waiting"), "getting": name in self.getting})
        recent = [dict(zip(("id", "split", "status", "how", "passed", "failed", "note", "item", "prompt", "ts"), r)) for r in self.db.execute(
            "SELECT id, split, status, how, passed, failed, note, item, prompt, ts FROM code_challenges WHERE status != 'new' "
            "ORDER BY ts DESC LIMIT 14").fetchall()]
        return {"enabled": self.enabled, "busy": self.busy, "suites": suites, "recent": recent, "stats": dict(self.stats),
                "auto_run": self.sys.scratch.mode("auto_run"), "teacher": bool(self.pol.teacher), "every": self.pol.every_seconds,
                "start_with": list(self.pol.start_with), "log": list(self.log)[-10:][::-1]}

    def lines(self) -> list[str]:
        st = self.status()
        out = ["Code challenges from the Hugging Face Hub" + ("" if st["enabled"] else " (off)") + f": {st['stats'].get('attempts', 0)} attempts, "
               f"{st['stats'].get('solved', 0)} solved."]
        for s_ in st["suites"]:
            e = s_["exam"]
            out.append(f"  {s_['name']:<10} {s_['have']:>5} downloaded, {s_['new']} not attempted; practice {s_['practice']['solved']}/"
                       f"{s_['practice']['attempted']} solved; exam {e['solved']}/{e['attempted']}"
                       + (f" ({e['score']:.0%})" if e["score"] is not None else "") + (f"; {s_['waiting']} waiting to be run" if s_["waiting"] else ""))
        out += [f"  - {x['line']}" for x in st["log"][:5]]
        return out


class Solver(Agent):
    """When idle: it gets the suites it starts with, then attempts one challenge it has never attempted."""
    name = "solver"

    def __init__(self, system):
        super().__init__(system)
        self.last = time.time() - system.directives.codechallenges.every_seconds + 40

    @property
    def interval(self) -> float:
        return 4.0

    async def step(self) -> None:
        s = self.sys
        c = s.codechallenges
        if not c.enabled or not s.guardian.background_allowed() or s.inflight:
            return
        if time.time() - (s.last_order_at or s.started_at) < c.pol.idle_seconds:
            return
        now = time.time()
        for suite in c.pol.start_with:
            low = c.count("suite = ? AND status = 'new'", (suite,)) < 5
            if suite in SUITES and low and suite not in c.getting and now - c.tried.get(suite, 0) > c.pol.retry_get_hours * 3600:
                c.tried[suite] = now
                c.save()
                try:
                    await c.fetch(suite)
                    self.stats["got"] += 1
                except Exception as exc:
                    self.stats["could not get"] += 1
                    self.log.info("could not get code challenges %s: %s", suite, exc)
                return
        if now - self.last < c.pol.every_seconds:
            return
        self.last = now
        await s.background_slot()
        rec = await c.attempt()
        self.stats[rec["status"] if rec else "nothing new"] += 1
