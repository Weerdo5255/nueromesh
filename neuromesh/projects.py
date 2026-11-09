"""Projects: it focuses on one coding project in Gitea and works on the objectives a person sets for it (P20).

    /project focus ann/textkit            (or the Code view): the project it works on, and its branch
    /project objective add docstrings to every function in textkit/words.py
    /project objective write tests for textkit/stats.py
      --> when idle (every [projects] work_every_seconds), or when asked (/project work):
          it studies the project first (gitea.py, through the gitea plugin), takes the next open objective
          (people's rewards for objectives and their kinds come first, rewards.py), and finds the files it is
          about: files the objective names, otherwise the ones whose functions fit it best
      --> it changes them: an LLM teacher rewrites each file for the objective (or writes a new file, a test
          for one, say); with no teacher it can still do what needs no judgement (add missing docstrings from
          the functions' names, tidy whitespace) and says the rest waits for a teacher
      --> the validator checks every changed file (scratchpad.py); one that does not check out goes back to the
          teacher once; a change that still doesn't check out is kept, marked, and never proposed
      --> it proposes the change: a diff per file, waiting for a person's approval (/project approve ch-3, the
          Code view, /approve). Approved, the gitea plugin pushes it to a NEW branch ([projects] branch_prefix)
          and opens a pull request into the project's branch - never a commit to an existing branch, and only
          with its separate write token ($GITEA_WRITE_TOKEN)
      --> it watches its pull requests: merged, the objective is done (and that counts as a reward); closed
          without merging, it counts as a punishment, and the objective stays open for another try
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import time
from collections import Counter

from .agents import Agent
from .gitea import LANGS, TOOL, words_of
from .scratchpad import Validator

KINDS = ("docs", "tests", "types", "fix", "refactor", "cleanup", "feature", "other")
KIND_WORDS = (("docs", ("docstring", "document", "comment", "readme", "explain")), ("tests", ("test",)),
              ("types", ("type hint", "typing", "annotation", "type annotation")), ("fix", ("fix", "bug", "error", "crash", "broken")),
              ("cleanup", ("whitespace", "format", "lint", "tidy", "style")), ("refactor", ("refactor", "simplify", "rename", "clean up", "restructure")),
              ("feature", ("add", "implement", "support", "create", "make", "allow")))


def kind_of_objective(text: str) -> str:
    low = f" {text.lower()} "
    for kind, words in KIND_WORDS:
        if any(w in low for w in words):
            return kind
    return "other"


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")[:n].strip("-") or "change"


def add_docstrings(source: str) -> tuple[str, int]:
    """Python: a one-line docstring for each function that has none, from its name (no judgement needed)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0
    lines = source.splitlines(keepends=True)
    inserts = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node) and node.body:
            first = node.body[0]
            if first.lineno <= node.lineno:  # a one-line function ("def f(): return 1"): left alone
                continue
            indent = re.match(r"[ \t]*", lines[first.lineno - 1]).group(0)
            words = words_of(node.name).strip() or node.name
            text = words[0].upper() + words[1:] + "."
            inserts.append((first.lineno - 1, f'{indent}"""{text}"""\n'))
    for at, line in sorted(inserts, reverse=True):
        lines.insert(at, line)
    return "".join(lines), len(inserts)


def tidy(source: str) -> tuple[str, int]:
    """Trailing whitespace off every line, and one newline at the end."""
    out = "\n".join(line.rstrip() for line in source.splitlines())
    out = out.rstrip("\n") + "\n" if out.strip() else out
    changed = sum(1 for a, b in zip(source.splitlines(), out.splitlines()) if a != b) + (0 if source.endswith("\n") else 1)
    return out, changed


class Projects:
    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.projects
        raw = system.store.get_kv("projects.state")
        st = json.loads(raw) if raw else {}
        self.focus: dict | None = st.get("focus")
        self.stats: Counter = Counter(st.get("stats", {}))
        self.log: list = st.get("log", [])[-60:]
        self.changes: dict[str, dict] = {}
        for rec in system.store.load_records("changes"):
            self.changes[rec["id"]] = rec
        self._n = max((int(k.split("-")[1]) for k in self.changes if re.fullmatch(r"ch-\d+", k)), default=0)
        self._o = max((int(o["id"].split("-")[1]) for o in (self.focus or {}).get("objectives", [])), default=0)
        self.validator = Validator()
        self.busy = False
        self.last_poll = 0.0

    @property
    def enabled(self) -> bool:
        return self.pol.enabled

    def save(self) -> None:
        self.sys.store.set_kv("projects.state", json.dumps({"focus": self.focus, "stats": dict(self.stats), "log": self.log[-60:]}))

    def _save_change(self, ch: dict) -> None:
        self.sys.store.save_record("changes", ch["id"], ch["status"], ch)

    def _note(self, line: str) -> None:
        self.log.append({"ts": round(time.time(), 1), "line": line[:300]})
        self.log = self.log[-60:]

    # ------------------------------------------------------------------ what people set
    def set_focus(self, repo: str, branch: str = "", by: str = "web") -> tuple[bool, str]:
        repo = repo.strip().strip("/")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            return False, "name the project as owner/name"
        old = self.focus or {}
        keep = old.get("objectives", []) if old.get("repo") == repo else []
        self.focus = {"repo": repo, "branch": branch.strip() or old.get("branch") if old.get("repo") == repo else branch.strip(),
                      "set_by": by, "ts": round(time.time(), 1), "objectives": keep}
        self.save()
        self.sys.guardian.audit(by, "project.focus", repo, branch=self.focus["branch"] or "(its default branch)")
        self.sys.bus.publish("project.focus", repo=repo, branch=self.focus["branch"], by=by, objective="")
        return True, f"it focuses on {repo}" + (f" ({self.focus['branch']})" if self.focus["branch"] else "") + \
            ": give it objectives (/project objective <what it should do to the project>)"

    def clear(self, by: str) -> tuple[bool, str]:
        if not self.focus:
            return False, "it has no project"
        repo = self.focus["repo"]
        self.focus = None
        self.save()
        self.sys.guardian.audit(by, "project.cleared", repo)
        return True, f"it no longer focuses on {repo}"

    def add_objective(self, text: str, by: str) -> tuple[bool, str, dict | None]:
        if not self.focus:
            return False, "set the project first (/project focus owner/name)", None
        text = " ".join(str(text).split())[:500]
        if len(text) < 6:
            return False, "say what it should do to the project", None
        self._o += 1
        obj = {"id": f"po-{self._o}", "text": text, "kind": kind_of_objective(text), "status": "open", "by": by,
               "ts": round(time.time(), 1), "notes": [], "changes": []}
        self.focus["objectives"].append(obj)
        self.save()
        self.sys.guardian.audit(by, "project.objective", obj["id"], text=text, repo=self.focus["repo"])
        self.sys.bus.publish("project.objective", id=obj["id"], objective=text, kind=obj["kind"], by=by, repo=self.focus["repo"])
        return True, f"{obj['id']} ({obj['kind']}): {text}", obj

    def objective(self, oid: str) -> dict | None:
        return next((o for o in (self.focus or {}).get("objectives", []) if o["id"] == oid), None)

    def set_objective(self, oid: str, status: str, by: str) -> tuple[bool, str]:
        o = self.objective(oid)
        if o is None:
            return False, f"no objective {oid}"
        status = {"drop": "dropped", "done": "done", "reopen": "open", "open": "open", "dropped": "dropped"}.get(status, "")
        if not status:
            return False, "an objective is done, dropped, or reopened"
        o["status"] = status
        o["notes"].append(f"{by} marked it {status}")
        self.save()
        self.sys.bus.publish("project.objective", id=oid, objective=o["text"], kind=o["kind"], by=by, status=status,
                             repo=self.focus["repo"])
        return True, f"{oid} is {status}"

    # ------------------------------------------------------------------ working on it
    def _tool(self):
        return self.sys.tools.tools.get(TOOL)

    def next_objective(self) -> dict | None:
        if not self.focus:
            return None
        waiting = {c["objective_id"] for c in self.changes.values() if c["status"] in ("awaiting approval", "approved", "pushed")}
        open_ = [o for o in self.focus["objectives"] if o["status"] in ("open", "working") and o["id"] not in waiting]
        if not open_:
            return None
        rw = getattr(self.sys, "rewards", None)
        if rw is not None and rw.enabled:  # objectives (and kinds of objective) people rewarded come first
            open_.sort(key=lambda o: -(rw.score("project", o["kind"], o["text"]) + rw.direction("project", o["text"].lower())))
        return open_[0]

    async def targets(self, obj: dict, repo: str, tree: list[dict]) -> list[tuple[str, bool, str]]:
        """The files an objective is about: (path, new file?, the source a new file is for)."""
        s = self.sys
        paths = {e["path"]: e for e in tree}
        named = [p for p in paths if p in obj["text"] or os.path.basename(p) in obj["text"].split()]
        code = [p for p in paths if os.path.splitext(p)[1].lower() in LANGS and int(paths[p].get("size") or 0) <= self.pol.max_file_kb * 1024]
        if not named:
            if obj["kind"] == "docs":
                rows = s.store.db.execute("SELECT path, COUNT(*) FROM code_examples WHERE repo = ? AND doc = '' AND lang = 'python' "
                                          "GROUP BY path ORDER BY COUNT(*) DESC", (repo,)).fetchall()
                named = [r[0] for r in rows if r[0] in paths and not re.search(r"(^|/)tests?/|test_", r[0])]
            else:
                found = s.code_study.similar(obj["text"], k=12)
                named = list(dict.fromkeys(ex["path"] for ex, _ in found if ex["repo"] == repo and ex["path"] in paths))
                if obj["kind"] == "cleanup" and not named:
                    named = code
        named = [p for p in named if p in code][: self.pol.files_per_change]
        if obj["kind"] == "tests":
            out = []
            for p in named:
                if re.search(r"(^|/)tests?/|test_", p):
                    out.append((p, False, ""))
                    continue
                stem = os.path.splitext(os.path.basename(p))[0]
                test = f"tests/test_{stem}.py" if p.endswith(".py") else f"tests/{stem}.test{os.path.splitext(p)[1]}"
                out.append((test, test not in paths, p))
            return out
        return [(p, False, "") for p in named]

    async def work(self, by: str = "idle") -> dict | None:
        """Work on the next open objective: a proposed change, waiting for a person's approval."""
        s = self.sys
        if not self.focus or self.busy:
            return None
        tool = self._tool()
        if tool is None or tool.state != "active" or s.tool_switch.refuses(tool):
            return {"ok": False, "why": s.code_study.detail}
        obj = self.next_objective()
        if obj is None:
            return {"ok": False, "why": "no open objective is waiting for work"}
        self.busy = True
        repo = self.focus["repo"]
        try:
            with s.activity.track("project", "coder", f"{repo}: {obj['text'][:60]}"):
                listed, why = await s.code_study.call({"op": "repos"})
                info = next((r for r in (listed or {}).get("repos", []) if r.get("full_name") == repo), None)
                if info is None:
                    return {"ok": False, "why": why or f"it cannot see {repo} on Gitea"}
                base = self.focus.get("branch") or info.get("default_branch") or "main"
                studied = s.store.db.execute("SELECT studied FROM code_repos WHERE name = ?", (repo,)).fetchone()
                if not studied or time.time() - (studied[0] or 0) > 3600:
                    await s.code_study.visit(by="its project", repo=repo)
                got, why = await s.code_study.call({"op": "tree", "repo": repo, "branch": base})
                if got is None:
                    return {"ok": False, "why": why}
                tree, commit = got.get("files") or [], got.get("commit", base)
                targets = await self.targets(obj, repo, tree)
                if not targets:
                    obj["notes"].append("it found no file this objective is about: name one in the objective")
                    self.save()
                    return {"ok": False, "why": f"no file in {repo} fits '{obj['text']}' (name the file in the objective)"}
                obj["status"] = "working"
                sha = {e["path"]: e.get("sha", "") for e in tree}
                read = [p for p, new, _ in targets if not new] + [src for _, new, src in targets if new and src]
                answer, why = await s.code_study.call({"op": "files", "repo": repo, "ref": commit, "paths": list(dict.fromkeys(read))})
                texts = (answer or {}).get("files") or {}
                files, hows, notes = [], set(), []
                for path, new, src in targets:
                    before = "" if new else texts.get(path)
                    if before is None:
                        notes.append(f"could not read {path}")
                        continue
                    lang = LANGS.get(os.path.splitext(path)[1].lower(), "other")
                    after, how, summary = await self._change(obj, path, before, lang, new, texts.get(src, "") if src else "")
                    if after is None or after == before:
                        notes.append(summary or f"nothing to change in {path}")
                        continue
                    check = self.validator.check(after, lang)
                    if check["ok"] is False and how.startswith("the ") and s.teachers.has_llm():
                        fixed = await s.teachers.change_file(obj["text"], path, before, lang, new, texts.get(src, "") if src else "",
                                                             previous=after, problems=check["problems"])
                        if fixed is not None:
                            after = fixed[0]["content"]
                            check = self.validator.check(after, lang)
                    diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                                        f"a/{path}", f"b/{path}", n=3))
                    plus = sum(1 for x in diff.splitlines() if x.startswith("+") and not x.startswith("+++"))
                    minus = sum(1 for x in diff.splitlines() if x.startswith("-") and not x.startswith("---"))
                    files.append({"path": path, "new": new, "sha": "" if new else sha.get(path, ""), "lang": lang,
                                  "after": after, "diff": diff[:60000], "plus": plus, "minus": minus, "check": check,
                                  "summary": summary})
                    hows.add(how)
            if not files:
                obj["notes"].append("; ".join(notes)[:300] or "nothing to change")
                obj["status"] = "open"
                self.save()
                return {"ok": False, "why": "; ".join(notes) or "nothing to change"}
            self._n += 1
            ok = all(f["check"]["ok"] is not False for f in files)
            ch = {"id": f"ch-{self._n}", "ts": round(time.time(), 1), "repo": repo, "base": base, "base_commit": commit,
                  "objective": obj["text"], "objective_id": obj["id"], "objective_kind": obj["kind"], "how": sorted(hows),
                  "files": files, "summary": "; ".join(f["summary"] for f in files if f["summary"])[:600] or obj["text"],
                  "plus": sum(f["plus"] for f in files), "minus": sum(f["minus"] for f in files),
                  "status": "drafted" if ok else "does not check out", "approval": None, "pr": None, "by": by, "notes": notes}
            ch["digest"] = hashlib.sha256(json.dumps([(f["path"], f["after"]) for f in files]).encode()).hexdigest()[:16]
            self.changes[ch["id"]] = ch
            obj["changes"].append(ch["id"])
            if ok:
                a = s.guardian.request_approval("project-change", ch["id"], {
                    "repo": repo, "base": base, "objective": obj["text"], "files": [f["path"] for f in files],
                    "plus": ch["plus"], "minus": ch["minus"], "digest": ch["digest"], "how": ch["how"],
                    "description": f"push a branch and open a pull request on {repo}: {obj['text'][:80]}"})
                ch["approval"], ch["status"] = a.id, "awaiting approval"
                obj["status"] = "proposed"
            else:
                obj["status"] = "open"
                obj["notes"].append(f"{ch['id']} did not check out: " + "; ".join(p for f in files for p in f["check"]["problems"])[:200])
            self._save_change(ch)
            self.stats["changes"] += 1
            self._note(f"{ch['id']} for '{obj['text'][:60]}': {len(files)} file(s), +{ch['plus']} -{ch['minus']} ({', '.join(ch['how'])}); {ch['status']}")
            self.save()
            s.bus.publish("project.change", id=ch["id"], repo=repo, objective=obj["text"][:160], kind=obj["kind"],
                          files=len(files), plus=ch["plus"], minus=ch["minus"], status=ch["status"], how=", ".join(ch["how"]))
            return {"ok": True, "change": ch}
        finally:
            self.busy = False

    async def _change(self, obj: dict, path: str, before: str, lang: str, new: bool, source: str) -> tuple[str | None, str, str]:
        """One file changed for an objective: (the new text, how, what changed)."""
        s = self.sys
        if s.teachers.has_llm():
            made = await s.teachers.change_file(obj["text"], path, before, lang, new, source)
            if made is not None:
                return made[0]["content"], f"the {made[1]} teacher", made[0].get("summary") or f"{path} changed for the objective"
        if obj["kind"] == "docs" and lang == "python" and not new:
            after, n = add_docstrings(before)
            after, _ = tidy(after) if n else (after, 0)
            return (after, "mechanical", f"{path}: a docstring for {n} function(s) that had none, from their names") if n else \
                (None, "", f"every function in {path} has a docstring")
        if obj["kind"] == "cleanup" and not new:
            after, n = tidy(before)
            return (after, "mechanical", f"{path}: trailing whitespace and the final newline on {n} line(s)") if n else \
                (None, "", f"{path} is tidy")
        return None, "", f"{path} needs a teacher to change it for '{obj['text'][:60]}' (none is available; it tries again later)"

    # ------------------------------------------------------------------ approval, pushing, watching
    def approve(self, cid: str, by: str) -> tuple[bool, str]:
        ch = self.changes.get(cid)
        if ch is None:
            return False, f"no change {cid}"
        if ch["status"] != "awaiting approval" or not ch.get("approval"):
            return False, f"{cid} is {ch['status']}"
        a = self.sys.guardian.decide(ch["approval"], True, by=by, note=f"change {ch['digest']}")
        return (True, f"{cid} approved: it pushes a branch and opens a pull request") if a else (False, "that approval was already decided")

    def decline(self, cid: str, by: str) -> tuple[bool, str]:
        ch = self.changes.get(cid)
        if ch is None:
            return False, f"no change {cid}"
        if ch.get("approval") and ch["status"] == "awaiting approval":
            self.sys.guardian.decide(ch["approval"], False, by=by)
            return True, f"{cid} declined"
        ch["status"] = "declined"
        self._save_change(ch)
        return True, f"{cid} declined"

    def on_event(self, ev) -> None:
        if ev.topic != "approval.decided":
            return
        ch = next((c for c in self.changes.values() if c.get("approval") == ev.data.get("id")), None)
        if ch is None:
            return
        by = ev.data.get("by", "a person")
        obj = self.objective(ch["objective_id"])
        if ev.data.get("status") != "approved":
            ch["status"] = "declined"
            self._save_change(ch)
            if obj is not None:
                obj["status"] = "open"
                obj["notes"].append(f"{by} declined {ch['id']}")
                self.save()
            self._implicit(ch, -1, by, "declined the change")
            return
        ch["status"], ch["approved_by"] = "approved", by
        self._save_change(ch)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.push(ch["id"], by))
        self.sys._bg_tasks.add(task)
        task.add_done_callback(self.sys._bg_tasks.discard)

    async def push(self, cid: str, by: str) -> dict:
        """The approved change: a new branch with its files, and a pull request (the gitea plugin, its write token)."""
        s = self.sys
        ch = self.changes[cid]
        digest = hashlib.sha256(json.dumps([(f["path"], f["after"]) for f in ch["files"]]).encode()).hexdigest()[:16]
        if ch["status"] != "approved" or digest != ch["digest"]:
            return {"ok": False, "why": "only exactly the change a person approved is pushed"}
        branch = f"{self.pol.branch_prefix}{slug(ch['objective'])}-{cid}"
        body = (f"**Objective** ({ch['objective_kind']}): {ch['objective']}\n\n{ch['summary']}\n\n"
                f"Written by NeuroMesh ({', '.join(ch['how'])}); every file checked: "
                + "; ".join(f"`{f['path']}` {f['check']['checker']}: {'ok' if f['check']['ok'] else 'not checked' if f['check']['ok'] is None else 'problems'}" for f in ch["files"])
                + f".\n\nApproved by {by} in NeuroMesh ({cid}).")
        made, why = await s.code_study.call({"op": "change", "repo": ch["repo"], "base": ch["base"], "branch": branch,
                                             "message": f"{ch['objective'][:72]}\n\n{ch['summary']}\n\nProposed by NeuroMesh ({cid}), approved by {by}.",
                                             "files": [{"path": f["path"], "content": f["after"], "sha": f["sha"]} for f in ch["files"]]})
        if made is None:
            ch["status"], ch["error"] = "failed", why[:300]
            self._save_change(ch)
            obj = self.objective(ch["objective_id"])
            if obj is not None:  # it may try again (with a write token, say)
                obj["status"] = "open"
                obj["notes"].append(f"{cid} could not be pushed: {why[:120]}")
            self._note(f"{cid} could not be pushed: {why[:160]}")
            self.save()
            s.bus.publish("project.change", id=cid, repo=ch["repo"], objective=ch["objective"][:160], kind=ch["objective_kind"],
                          files=len(ch["files"]), plus=ch["plus"], minus=ch["minus"], status="failed", how=why[:120])
            return {"ok": False, "why": why}
        pr, why = await s.code_study.call({"op": "pull", "repo": ch["repo"], "base": ch["base"], "head": branch,
                                           "title": f"{ch['objective'][:200]}", "body": body})
        ch["branch"] = branch
        ch["pr"] = {"number": (pr or {}).get("number"), "url": (pr or {}).get("url", ""), "state": "open", "merged": False}
        ch["status"] = "pushed" if pr else "pushed (no pull request)"
        if not pr:
            ch["error"] = why[:300]
        self._save_change(ch)
        obj = self.objective(ch["objective_id"])
        if obj is not None and pr:  # it waits for a person to merge or close the pull request
            obj["status"] = "in review"
            obj["notes"].append(f"pull request #{ch['pr']['number']} is open, from {branch}")
        self.stats["pushed"] += 1
        self._note(f"{cid} pushed to {branch}" + (f": pull request #{ch['pr']['number']}" if pr else f" (no pull request: {why[:100]})"))
        self.save()
        s.guardian.audit(by, "project.pushed", cid, repo=ch["repo"], branch=branch, pr=ch["pr"]["number"])
        s.bus.publish("project.change", id=cid, repo=ch["repo"], objective=ch["objective"][:160], kind=ch["objective_kind"],
                      files=len(ch["files"]), plus=ch["plus"], minus=ch["minus"], status=ch["status"],
                      how=f"pull request #{ch['pr']['number']}" if pr else "a branch")
        return {"ok": True, "branch": branch, "pr": ch["pr"]}

    async def poll(self) -> int:
        """Its pull requests: merged (the objective is done, and that is a reward) or closed (a punishment)."""
        s = self.sys
        n = 0
        for ch in [c for c in self.changes.values() if c["status"] == "pushed" and (c.get("pr") or {}).get("number")]:
            st, _ = await s.code_study.call({"op": "pull_state", "repo": ch["repo"], "number": ch["pr"]["number"]})
            if not st:
                continue
            if st.get("merged") or st.get("state") == "closed":
                merged = bool(st.get("merged"))
                ch["status"] = "merged" if merged else "closed"
                ch["pr"].update(state=st.get("state"), merged=merged)
                self._save_change(ch)
                obj = self.objective(ch["objective_id"])
                if obj is not None:
                    obj["status"] = "done" if merged else "open"
                    obj["notes"].append(f"pull request #{ch['pr']['number']} was {'merged' if merged else 'closed without merging'}")
                self._implicit(ch, 1 if merged else -1, "a person on Gitea", "merged it" if merged else "closed it without merging")
                self._note(f"{ch['id']}: pull request #{ch['pr']['number']} {'merged' if merged else 'closed'}")
                n += 1
        if n:
            self.save()
        return n

    def _implicit(self, ch: dict, value: int, by: str, why: str) -> None:
        rw = getattr(self.sys, "rewards", None)
        if rw is not None and rw.enabled:
            rw.give(ch["id"], value, by=by.replace("teacher:", ""), why=why, kind="")

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        changes = sorted(self.changes.values(), key=lambda c: -c["ts"])
        slim = [{k: v for k, v in c.items() if k != "files"} | {"files": [{k: v for k, v in f.items() if k != "after"} for f in c["files"]]}
                for c in changes[:30]]
        tool = self._tool()
        write = bool(tool is not None and os.environ.get(str(self.sys.tool_config(tool).get("write_token_env") or "GITEA_WRITE_TOKEN")))
        return {"enabled": self.enabled, "focus": self.focus, "changes": slim, "busy": self.busy, "stats": dict(self.stats),
                "log": self.log[-10:][::-1], "can_write": write, "gitea": self.sys.code_study.detail,
                "studied": [r["name"] for r in self.sys.code_study.repos()]}

    def lines(self, cid: str | None = None) -> list[str]:
        if cid:
            ch = self.changes.get(cid)
            if ch is None:
                return [f"no change {cid}"]
            out = [f"{ch['id']}  {ch['repo']} ({ch['base']}): {ch['objective']}  [{ch['status']}]",
                   f"  {len(ch['files'])} file(s), +{ch['plus']} -{ch['minus']}, written by {', '.join(ch['how'])}: {ch['summary'][:200]}"]
            if ch.get("pr"):
                out.append(f"  pull request #{ch['pr']['number']} {ch['pr'].get('url', '')} from {ch.get('branch')}")
            for f in ch["files"]:
                out.append(f"  --- {f['path']} ({f['check']['checker']}: {'ok' if f['check']['ok'] else f['check']['problems']})")
                out += ["    " + x for x in f["diff"].splitlines()[:60]]
            return out
        st = self.status()
        if not st["focus"]:
            return ["Project: none. /project focus owner/name [branch], then /project objective <what it should do to it>."]
        f = st["focus"]
        out = [f"Project: {f['repo']}" + (f" ({f['branch']})" if f.get("branch") else "") + f", set by {f['set_by']}. "
               + ("It can open pull requests." if st["can_write"] else "It has no write token ($GITEA_WRITE_TOKEN): it proposes changes, and cannot push them.")]
        for o in f["objectives"]:
            out.append(f"  {o['id']:<6} {o['status']:<9} {o['kind']:<8} {o['text'][:70]}" + (f"  ({o['notes'][-1][:60]})" if o["notes"] else ""))
        for c in st["changes"][:8]:
            out.append(f"  {c['id']:<6} {c['status']:<18} +{c['plus']} -{c['minus']} {c['objective'][:50]}"
                       + (f"  PR #{c['pr']['number']}" if c.get("pr") and c["pr"].get("number") else ""))
        return out


class Foreman(Agent):
    """When idle, it works on its project's next objective, and watches its pull requests."""
    name = "foreman"

    def __init__(self, system):
        super().__init__(system)
        self.last = time.time()

    @property
    def interval(self) -> float:
        return max(1.0, min(10.0, self.sys.directives.projects.work_every_seconds / 6))

    async def step(self) -> None:
        s = self.sys
        p = s.projects
        if not p.enabled or not p.focus or not s.guardian.background_allowed() or s.inflight:
            return
        now = time.time()
        if now - p.last_poll >= p.pol.poll_seconds:
            p.last_poll = now
            await p.poll()
        if now - self.last >= p.pol.work_every_seconds and p.next_objective() is not None:
            self.last = now
            res = await p.work(by="idle")
            self.stats["changes" if res and res.get("ok") else "no change"] += 1
