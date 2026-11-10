"""Its works: what it made before, which it goes back to and improves on its own initiative, and long-term
projects of its own that it keeps building (P20, P16).

Everything it makes is kept: programs its workbench built (workbench.py), code it wrote in the scratch pad
(scratchpad.py), tools it wrote (toolsmith.py), objectives that ended short (objectives.py). Until now it made each
once. Now its director (agency.py, the "revisit" mode) goes back to them when it is idle, the one most worth
another look first:

    a program    that was incomplete, failed its checks or did not check out: built again from what it has studied
                 since (a helper it could not find may be there now; people may have written a test for it)
    code         it wrote in the scratch pad that did not check out, or failed when it ran: drafted again from what
                 it knows now - its recipes, its workbench's programs, the closest example. The earlier version is
                 kept beside the new one; code a person approved is never changed
    a tool       it wrote that is in use: tried out again on fresh probes, and refined where it falls short (a new
                 version, which asks permission again, P15)
    an objective that ended short: tried again, planned afresh with what it knows now

A piece of work is due when it has never been revisited, when what it was made from changed (the repository was
studied again and has more or different functions), or after [works] revisit_after_hours. What a visit changed is
written down - before, after, better or not - so it does not keep trying what does not improve.

Long-term projects of its own. From a repository it studied it starts a project: a library of every function it
can build into a whole program, gathered into one module with all their checks (state/works/<project>/vN/). A
project is never finished in one go. Each session takes the next step and says what it means to do after:

    build a part      a function of the repository it has not built a program from yet
    mend a part       one that was incomplete or failed, built again
    put it together   a new version of the module when its parts changed: checked, kept, put in the scratch pad
                      (it runs only as P20 allows), and what it printed is noted against that version

You can start a project for a repository, pause it, have it work now, or remove it; it starts its own (at most
[works] max_projects) once a repository has enough parts that ran. Its journal says what every session did.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter, deque
from pathlib import Path

log = logging.getLogger("neuromesh.works")

BY = "its own initiative"
RANK = {"skipped": 0, "incomplete": 1, "does not check out": 2, "failed": 3, "checks out": 4, "ran": 5, "passed": 6}
WEAK = ("incomplete", "does not check out", "failed")
RAN = ("ran", "passed")


class Works:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.works
        self.dir = system.state_dir / "works"
        raw = system.store.get_kv("works.state")
        st = json.loads(raw) if raw else {}
        self.visits: dict[str, dict] = st.get("visits", {})      # "program:12" -> {ts, times, improved, basis, note}
        self.projects: dict[str, dict] = st.get("projects", {})
        self.journal: deque = deque(st.get("journal", []), maxlen=120)
        self.stats: Counter = Counter(st.get("stats", {}))
        self.busy = False

    @property
    def enabled(self) -> bool:
        return bool(self.pol.enabled)

    def save(self) -> None:
        self.sys.store.set_kv("works.state", json.dumps({"visits": self.visits, "projects": self.projects,
                                                         "journal": list(self.journal), "stats": dict(self.stats)}))

    # ------------------------------------------------------------------ what it could go back to
    def _basis(self, repo: str) -> str:
        """What a repository's studied code looks like now: when it changes, work made from it is due again."""
        row = self.sys.store.db.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM code_examples WHERE repo = ?", (repo,)).fetchone()
        return f"{row[0]}:{row[1]}"

    def _due(self, key: str, basis: str = "") -> tuple[float, str]:
        """How much a piece of work is worth another look (0: not now), and why."""
        v = self.visits.get(key)
        if v is None:
            return 1.0, "it has not gone back to it yet"
        if basis and v.get("basis") and v["basis"] != basis:
            return 0.9, "what it was made from has changed since"
        hours = (time.time() - v["ts"]) / 3600
        wait = self.pol.revisit_after_hours * (1 if v.get("improved") else min(8, 2 ** max(0, v.get("misses", 1) - 1)))
        if hours >= wait:
            return 0.5, f"it last went back to it {hours:.0f} h ago"
        return 0.0, ""

    def _latest_programs(self, repo: str = "") -> list[dict]:
        """The newest program its workbench built from each function."""
        seen, out = set(), []
        for p in self.sys.workbench.programs(500):
            if p["kind"] != "whole" or p["example"] in seen or (repo and p["repo"] != repo):
                continue
            seen.add(p["example"])
            out.append(p)
        return out

    def candidates(self) -> list[dict]:
        """Earlier work worth going back to, the most due first."""
        s = self.sys
        out = []

        def add(key, kind, title, state, basis=""):
            score, why = self._due(key, basis)
            if score > 0:
                out.append({"key": key, "kind": kind, "title": title, "state": state, "why": why, "score": score,
                            "basis": basis, "last": (self.visits.get(key) or {}).get("ts")})
        if s.workbench.enabled:
            for p in self._latest_programs():
                if p["status"] in WEAK and s.code_study.example(p["example"]) is not None:
                    add(f"program:{p['example']}", "program", f"{p['name']} ({p['repo']})",
                        p["status"] + (f": missing {', '.join(p['unresolved'][:3])}" if p["unresolved"] else ""), self._basis(p["repo"]))
        if s.scratch.enabled:
            for it in s.scratch.items.values():
                if it.get("kind") != "draft" or not it.get("request") or it.get("approved_digest") or it["status"] in ("running", "approved"):
                    continue
                if (it.get("origin") or {}).get("how") == "person" or it.get("approval"):
                    continue
                failed = bool(it.get("runs")) and not it["runs"][-1].get("ok")
                if it["status"] == "invalid" or failed:
                    add(f"code:{it['id']}", "code", f"{it['id']}: {it['title'][:60]}",
                        "it does not check out" if it["status"] == "invalid" else "it failed when it ran")
        smith = s.toolsmith
        if smith.enabled:
            for name, d in smith.drafts.items():
                if d.get("status") == "needs work" and name in s.tools.tools:
                    add(f"tool:{name}", "tool", f"{name} (a tool it wrote)", f"last tried: {d.get('score', '?')}")
        book = s.objectives
        if book.enabled:
            for o in book.archive.values():
                if not o.retried_as and not getattr(o, "retry_of", ""):
                    done, total = o.progress()
                    add(f"objective:{o.id}", "objective", f"{o.id}: {o.text[:70]}", f"ended short ({done} of {total} found)")
        return sorted(out, key=lambda c: (-c["score"], c["last"] or 0))

    # ------------------------------------------------------------------ going back to one
    def _visited(self, c: dict, improved: bool, before: str, after: str, line: str, by: str) -> dict:
        v = self.visits.get(c["key"]) or {"times": 0, "misses": 0}
        v.update(ts=round(time.time(), 1), times=v["times"] + 1, improved=bool(improved), basis=c.get("basis", ""),
                 misses=0 if improved else v.get("misses", 0) + 1, note=line[:200])
        self.visits[c["key"]] = v
        if len(self.visits) > 2000:
            for k in sorted(self.visits, key=lambda k: self.visits[k]["ts"])[:200]:
                del self.visits[k]
        rec = {"ts": v["ts"], "key": c["key"], "kind": c["kind"], "title": c["title"], "before": before, "after": after,
               "improved": bool(improved), "line": line, "by": by}
        self.journal.append(rec)
        self.stats["revisits"] += 1
        self.stats["improved" if improved else "no better"] += 1
        self.save()
        self.sys.guardian.audit(by, "works.revisited", c["key"], improved=bool(improved), before=before, after=after)
        self.sys.bus.publish("works.revisited", **rec)
        return rec

    async def revisit(self, key: str = "", by: str = BY) -> dict | None:
        """Go back to one piece of earlier work (the most due, or the one named) and try to improve it."""
        cands = self.candidates()
        c = next((x for x in cands if x["key"] == key), None) if key else (cands[0] if cands else None)
        if c is None:
            return None
        s = self.sys
        kind, ref = c["key"].split(":", 1)
        with s.activity.track("works", "director", f"going back to {c['title']}"):
            if kind == "program":
                old = next((p for p in self._latest_programs() if p["example"] == int(ref)), None)
                new = await s.workbench.experiment(example=int(ref), by=by)
                if old is None or new is None:
                    return None
                a = (RANK.get(old["status"], 0), old["checks"], -len(old["unresolved"]))
                b = (RANK.get(new["status"], 0), new.get("checks", 0), -len(new.get("unresolved", [])))
                better = b > a
                line = (f"went back to {c['title']}: {old['status']} -> {new['status']}" if better else
                        f"went back to {c['title']}: still {new['status']}" + (f" ({new.get('why', '')[:90]})" if new.get("why") else ""))
                return self._visited(c, better, old["status"], new["status"], line, by)
            if kind == "code":
                it = s.scratch.items.get(ref)
                if it is None:
                    return None
                code, origin = await s.scratch.draft(it["request"], it["language"], teacher=self.pol.teacher)
                before = "does not check out" if it["status"] == "invalid" else "failed when it ran"
                if not code or code.strip() == it["code"].strip():
                    return self._visited(c, False, before, before, f"went back to {c['title']}: it knows no better way to write it yet", by)
                check = s.scratch.validator.check(code, it["language"])
                if not check["ok"]:
                    return self._visited(c, False, before, before, f"went back to {c['title']}: what it wrote now does not check out either", by)
                it.setdefault("versions", []).append({"ts": round(time.time(), 1), "code": it["code"], "status": it["status"],
                                                      "origin": it.get("origin")})
                it["versions"] = it["versions"][-5:]
                it["origin"] = {**origin, "revised": len(it["versions"])}
                it["runs"] = []
                s.scratch.edit(ref, code, by)
                return self._visited(c, True, before, "checks out", f"went back to {c['title']}: wrote it again "
                                     f"({origin.get('how', 'its own way')}), and now it checks out; the earlier version is kept", by)
            if kind == "tool":
                d = s.toolsmith.drafts.get(ref) or {}
                was = d.get("score")
                rep = await s.toolsmith.experiment(ref, by)
                if rep is None or rep.get("score") is None:
                    return self._visited(c, False, str(was), str(was), f"went back to {c['title']}: could not try it out now", by)
                better = was is None or rep["score"] > float(was) + 1e-9 or bool(rep.get("refined"))
                return self._visited(c, better, f"{was}", f"{rep['score']:.2f}", f"went back to {c['title']}: {rep['line']}", by)
            if kind == "objective":
                ok, msg = await s.objectives.retry(ref, by)
                return self._visited(c, ok, "ended short", "being tried again" if ok else "ended short",
                                     f"went back to {c['title']}: {msg}", by)
        return None

    # ------------------------------------------------------------------ long-term projects of its own
    def _repos(self) -> dict[str, list[dict]]:
        by_repo: dict[str, list[dict]] = {}
        for p in self._latest_programs():
            by_repo.setdefault(p["repo"], []).append(p)
        return by_repo

    def _untried(self, repo: str) -> int | None:
        row = self.sys.store.db.execute(
            "SELECT e.id FROM code_examples e WHERE e.repo = ? AND e.lang = 'python' AND e.kind != 'test' AND e.lines BETWEEN 2 AND ? "
            "AND NOT EXISTS (SELECT 1 FROM code_programs p WHERE p.example = e.id AND p.kind = 'whole') "
            "ORDER BY (e.doc LIKE '%>>>%') DESC, e.id LIMIT 1", (repo, self.sys.directives.workbench.max_lines)).fetchone()
        return row[0] if row else None

    def start(self, repo: str, by: str = BY) -> tuple[bool, str]:
        """Start a long-term project: a library from a repository it studied."""
        if any(p["repo"] == repo for p in self.projects.values()):
            return False, f"it already has a project from {repo}"
        if self.sys.store.db.execute("SELECT 1 FROM code_examples WHERE repo = ? AND lang = 'python' LIMIT 1", (repo,)).fetchone() is None:
            return False, f"it has studied no Python in {repo} yet (/gitea study {repo})"
        n = 1 + max([int(k.split("-")[-1]) for k in self.projects] or [0])
        pid = f"work-{n}"
        name = re.sub(r"[^a-z0-9]+", "_", repo.split("/")[-1].lower()).strip("_") or "library"
        self.projects[pid] = {"id": pid, "repo": repo, "module": f"{name}_library", "title": f"A library from {repo}",
                              "goal": f"gather every function of {repo} that it can build into a whole program into one "
                                      f"module with all their checks, and keep improving it",
                              "created": round(time.time(), 1), "by": by, "status": "active", "version": 0, "parts": [],
                              "digest": "", "sessions": 0, "last": None, "item": "", "result": None, "next": "build its first parts",
                              "journal": []}
        self._log(self.projects[pid], f"started by {by}: {self.projects[pid]['goal']}")
        self.sys.guardian.audit(by, "works.project", pid, repo=repo)
        self.sys.bus.publish("works.project", id=pid, repo=repo, by=by)
        return True, f"{pid}: {self.projects[pid]['title']}"

    def _log(self, proj: dict, line: str) -> None:
        proj["journal"] = (proj.get("journal", []) + [{"ts": round(time.time(), 1), "line": line}])[-40:]
        self.save()

    def set_status(self, pid: str, status: str, by: str) -> tuple[bool, str]:
        proj = self.projects.get(pid)
        if proj is None:
            return False, f"no project {pid}"
        if status == "removed":
            del self.projects[pid]
            self.save()
            self.sys.guardian.audit(by, "works.removed", pid)
            return True, f"removed {pid} (its versions stay in state/works/{pid}/)"
        if status not in ("active", "paused"):
            return False, "a project is active or paused"
        proj["status"] = status
        self._log(proj, f"{status} by {by}")
        return True, f"{pid} is {status}"

    def _plan(self, proj: dict) -> tuple[str, object, str]:
        """The next step of a project, and what it means to do: (step, what, why)."""
        repo = proj["repo"]
        progs = self._latest_programs(repo)
        good = sorted(p["name"] for p in progs if p["status"] in RAN)
        weak = [p for p in progs if p["status"] in WEAK and self._due(f"program:{p['example']}", self._basis(repo))[0] > 0]
        untried = self._untried(repo)
        digest = hashlib.sha256(json.dumps([(p["name"], p["id"]) for p in progs if p["status"] in RAN]).encode()).hexdigest()[:12]
        if len(good) >= self.pol.min_parts and digest != proj.get("digest") and (proj["sessions"] % 3 == 2 or untried is None):
            return "assemble", digest, f"put its {len(good)} parts together as version {proj['version'] + 1}"
        if untried is not None:
            return "build", untried, "build a function it has not built from yet"
        if weak:
            return "mend", weak[0], f"mend {weak[0]['name']} ({weak[0]['status']})"
        if len(good) >= self.pol.min_parts and digest != proj.get("digest"):
            return "assemble", digest, f"put its {len(good)} parts together as version {proj['version'] + 1}"
        return "wait", None, "wait until the repository changes: every part is built and together"

    async def work(self, pid: str = "", by: str = BY) -> dict | None:
        """One session on a long-term project (the one named, or the one it has left longest)."""
        s = self.sys
        live = [p for p in self.projects.values() if p["status"] == "active"]
        proj = self.projects.get(pid) if pid else next(
            (p for p in sorted(live, key=lambda p: p["last"] or 0) if self._plan(p)[0] != "wait"), None)
        if proj is None:
            return None
        step, what, why = self._plan(proj)
        did, improved = "", False
        with s.activity.track("works", "director", f"{proj['id']} ({proj['title']}): {why}"):
            if step == "build":
                rec = await s.workbench.experiment(example=int(what), by=by)
                did = rec["line"] if rec else "could not build it now (its workbench is busy)"
                improved = bool(rec and rec["status"] in RAN + ("checks out",))
            elif step == "mend":
                rec = await self.revisit(f"program:{what['example']}", by)
                did = rec["line"] if rec else f"could not go back to {what['name']} now"
                improved = bool(rec and rec["improved"])
            elif step == "assemble":
                did, improved = await self._assemble(proj, what, by)
            else:
                did = "nothing to do until the repository changes"
        proj["sessions"] += 1
        proj["last"] = round(time.time(), 1)
        proj["next"] = self._plan(proj)[2]
        self._log(proj, f"{did}. Next: {proj['next']}")
        self.stats["sessions"] += 1
        rec = {"ts": proj["last"], "key": f"project:{proj['id']}", "kind": "project", "title": proj["title"], "before": "",
               "after": f"v{proj['version']}", "improved": improved, "line": f"{proj['id']}: {did}", "by": by, "step": step}
        self.journal.append(rec)
        self.save()
        s.guardian.audit(by, "works.session", proj["id"], step=step, did=did[:200])
        s.bus.publish("works.session", id=proj["id"], step=step, line=did[:300], version=proj["version"], by=by)
        return rec

    async def _assemble(self, proj: dict, digest: str, by: str) -> tuple[str, bool]:
        s = self.sys
        good = [p for p in self._latest_programs(proj["repo"]) if p["status"] in RAN]
        examples = [e for e in (s.code_study.example(p["example"]) for p in sorted(good, key=lambda p: p["name"])) if e]
        lib = await s.workbench.library(examples, f"{proj['module']}: {proj['title']}, version {proj['version'] + 1}",
                                        proj["goal"], f"{len(examples)} functions of {proj['repo']} on Gitea")
        if not lib.get("ok"):
            proj["digest"] = digest
            return f"could not put its parts together: {lib.get('why', '')}", False
        check = s.scratch.validator.check(lib["code"], "python")
        if not check["ok"]:
            proj["digest"] = digest
            return f"put its parts together, but the module does not check out: {'; '.join(check.get('problems', [])[:2])}", False
        added = sorted(set(lib["parts"]) - set(proj["parts"]))
        proj["version"] += 1
        proj["parts"], proj["digest"], proj["checks"] = lib["parts"], digest, lib["checks"]
        folder = self.dir / proj["id"] / f"v{proj['version']}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{proj['module']}.py").write_text(lib["code"])
        old = s.scratch.items.get(proj.get("item") or "")
        if old is not None and not old.get("approved_digest") and old["status"] != "running":
            s.scratch.remove(old["id"], by)  # the earlier version's draft makes room (its file stays in state/works/)
        item = s.scratch._new(f"{proj['module']} v{proj['version']} ({proj['id']})", lib["code"], "python", by, request=proj["goal"],
                              origin={"how": "works", "project": proj["id"], "version": proj["version"], "parts": lib["parts"]},
                              kind="project")
        s.scratch._save(item)
        proj["item"], proj["result"] = item["id"], None
        did = (f"put {len(lib['parts'])} parts together as version {proj['version']} ({lib['checks']} checks"
               + (f"; new: {', '.join(added[:6])}" if added and proj["version"] > 1 else "") + f"), kept as {item['id']}")
        if s.scratch.auto_runs(item)[0]:
            run = await s.scratch.auto_run(item)
            if run:
                did += f"; {self._result(proj, run)}"
        else:
            did += f"; it waits to be run ({s.scratch.auto_runs(item)[1]})"
        (folder / "NOTES.md").write_text(f"# {proj['title']}, version {proj['version']}\n\n{proj['goal']}\n\n"
                                         f"Parts: {', '.join(lib['parts'])}\n\n{did}\n")
        self.stats["versions"] += 1
        return did, True

    def _result(self, proj: dict, run: dict) -> str:
        from .workbench import outcome
        out = outcome(run.get("stdout", ""))
        proj["result"] = {"ok": bool(run.get("ok")), "passed": out["passed"], "failed": out["failed"] + out["errors"],
                          "version": proj["version"]}
        return (f"it ran: {out['passed']} check(s) passed" + (f", {out['failed'] + out['errors']} failed" if out["failed"] + out["errors"] else "")
                if run.get("ok") else "it failed when it ran")

    def on_event(self, ev) -> None:
        if ev.topic != "code.ran" or str(ev.data.get("by", "")).startswith("auto-run"):
            return
        item = self.sys.scratch.items.get(str(ev.data.get("id", "")))
        if item is None or not item.get("runs"):
            return
        for proj in self.projects.values():
            if proj.get("item") == item["id"]:
                self._log(proj, f"version {proj['version']} was run by {ev.data.get('by', 'a person')}: {self._result(proj, item['runs'][-1])}")

    def tend(self) -> None:
        """Start projects of its own: a repository with enough parts that ran, and room for another project."""
        if not self.pol.own_projects:
            return
        have = {p["repo"] for p in self.projects.values()}
        for repo, progs in self._repos().items():
            if len(self.projects) >= self.pol.max_projects:
                return
            if repo not in have and sum(p["status"] in RAN for p in progs) >= self.pol.min_parts:
                self.start(repo)

    # ------------------------------------------------------------------ for its director
    def due(self) -> dict:
        """What there is to go back to now: for the director's situation."""
        if not self.enabled:
            return {"projects": [], "candidates": []}
        self.tend()
        projects = [p for p in self.projects.values() if p["status"] == "active" and self._plan(p)[0] != "wait"]
        return {"projects": projects, "candidates": self.candidates()}

    async def session(self, by: str = BY) -> dict | None:
        """Its own initiative: a session on a long-term project, or a visit to earlier work - whichever it left longest."""
        if not self.enabled or self.busy:
            return None
        self.busy = True
        try:
            d = self.due()
            last_project = max((p["last"] or 0 for p in self.projects.values()), default=0)
            last_visit = max((v["ts"] for v in self.visits.values()), default=0)
            if d["projects"] and (not d["candidates"] or last_project <= last_visit):
                return await self.work(by=by)
            if d["candidates"]:
                return await self.revisit(by=by)
            return None
        finally:
            self.busy = False

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        cands = self.candidates() if self.enabled else []
        return {"enabled": self.enabled, "busy": self.busy, "stats": dict(self.stats),
                "projects": [{**{k: v for k, v in p.items() if k != "journal"}, "journal": p.get("journal", [])[-8:][::-1],
                              "plan": self._plan(p)[2] if p["status"] == "active" else "paused"}
                             for p in sorted(self.projects.values(), key=lambda p: p["id"])],
                "candidates": cands[:20], "waiting": len(cands), "journal": list(self.journal)[-20:][::-1],
                "revisit_after_hours": self.pol.revisit_after_hours, "max_projects": self.pol.max_projects}

    def lines(self) -> list[str]:
        st = self.status()
        out = [f"Its works: {len(st['projects'])} long-term project(s), {st['waiting']} earlier piece(s) of work worth going back to; "
               f"{st['stats'].get('revisits', 0)} revisits, {st['stats'].get('improved', 0)} improved."]
        for p in st["projects"]:
            out.append(f"  {p['id']:<8} {p['status']:<7} v{p['version']} {p['title']} ({len(p['parts'])} parts, {p['sessions']} sessions). Next: {p['plan']}")
        for c in st["candidates"][:6]:
            out.append(f"  due: {c['kind']:<9} {c['title'][:50]:<50} {c['state'][:40]} ({c['why']})")
        out += [f"  - {j['line']}" for j in st["journal"][:5]]
        return out
