"""Gitea tool: reads code from a local Gitea (https://gitea.com) for the mesh's coder nodes.

Only reads, through Gitea's API (/api/v1): its version, the repositories it can see, a branch's commit, a
repository's file tree, and source files (many at once). It never writes to Gitea and never runs what it reads.
The token comes from the one environment variable its manifest lists (GITEA_TOKEN), and is sent only to the
configured Gitea.

Requests (the "input"):
    "__selftest__"                          Gitea answers: "gitea ok: Gitea 1.22.3, 4 repositories"
    {"op": "repos"}                         the repositories it may read ([config] repos, or every one it can see)
    {"op": "tree", "repo", "branch"}        the branch's commit and every file in it (path, sha, size)
    {"op": "files", "repo", "ref", "paths"} the files' text, read at once ([config] parallel; at most 4 MB a call)
    anything else                           an order: the repositories whose name or description match its words

Speed: started with --serve (manifest: persistent = true) it stays running and answers requests as JSON lines,
many at once, over kept-alive connections.
"""
import http.client
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULTS = {"url": "http://localhost:3000", "token_env": "GITEA_TOKEN", "repos": [], "use_proxy": False, "timeout": 15,
            "parallel": 6, "max_file_kb": 256, "max_batch_kb": 4096, "user_agent": "neuromesh-coder"}
LOCAL = ("localhost", "127.0.0.1", "::1")
STOP = {"what", "which", "are", "is", "the", "on", "in", "my", "list", "show", "find", "search", "for", "gitea", "repos",
        "repositories", "repository", "repo", "have", "has", "with", "me", "all", "a", "an", "of", "any", "there"}


class Transient(Exception):
    """Gitea is down or unreachable: not a broken tool."""


class NotFound(Exception):
    pass


class Pool:
    """Kept-alive connections, one per thread."""

    def __init__(self):
        self.local = threading.local()

    def conn(self, scheme: str, netloc: str, timeout: float):
        conns = getattr(self.local, "conns", None)
        if conns is None:
            conns = self.local.conns = {}
        c = conns.get((scheme, netloc))
        if c is None:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            c = conns[(scheme, netloc)] = cls(netloc, timeout=timeout)
        return c

    def drop(self, scheme: str, netloc: str) -> None:
        c = getattr(self.local, "conns", {}).pop((scheme, netloc), None)
        if c is not None:
            c.close()


POOL = Pool()


class Gitea:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base = str(cfg["url"]).rstrip("/")
        if not self.base.startswith(("http://", "https://")):
            raise RuntimeError(f"Gitea's url should start with http:// or https:// (it is '{self.base}')")
        self.timeout = float(cfg.get("timeout", 15))
        self.token = os.environ.get(str(cfg.get("token_env") or ""), "") if cfg.get("token_env") else ""
        parts = urllib.parse.urlsplit(self.base)
        self.scheme, self.netloc, self.prefix = parts.scheme, parts.netloc, parts.path.rstrip("/")
        host = parts.hostname or ""
        local = host in LOCAL or host.endswith(".local") or re.match(r"^(10|192\.168|172\.(1[6-9]|2\d|3[01]))\.", host)
        proxy = os.environ.get("https_proxy" if self.scheme == "https" else "http_proxy") or \
            os.environ.get("HTTPS_PROXY" if self.scheme == "https" else "HTTP_PROXY")
        self.proxy = proxy if (proxy and cfg.get("use_proxy") and not local) else None

    def _request(self, path: str, params: dict | None = None, accept: str = "application/json") -> bytes:
        q = ("?" + urllib.parse.urlencode(params)) if params else ""
        target = f"{self.prefix}/api/v1{path}{q}"
        headers = {"accept": accept, "user-agent": self.cfg.get("user_agent", "neuromesh-coder")}
        if self.token:
            headers["authorization"] = f"token {self.token}"
        if self.proxy:  # through the proxy ([config] use_proxy, a Gitea that isn't local)
            return self._via_proxy(f"{self.base}/api/v1{path}{q}", headers, path)
        scheme, netloc = self.scheme, self.netloc
        for attempt in (0, 1):  # a kept-alive connection the server closed: once more, on a fresh one
            c = POOL.conn(scheme, netloc, self.timeout)
            try:
                c.request("GET", target, headers=headers)
                r = c.getresponse()
                body = r.read()
                if r.will_close:
                    POOL.drop(scheme, netloc)
                break
            except (http.client.HTTPException, OSError) as exc:
                POOL.drop(scheme, netloc)
                if attempt:
                    raise Transient(f"no Gitea at {self.base} ({exc})")
        if r.status in (401, 403):
            raise RuntimeError(f"Gitea refused it ({r.status}): set a read-only token in ${self.cfg.get('token_env')}")
        if r.status == 404:
            raise NotFound(f"not on Gitea: {path}")
        if r.status >= 500:
            raise Transient(f"Gitea answered {r.status}")
        if r.status != 200:
            raise RuntimeError(f"Gitea answered {r.status} for {path}")
        return body

    def _via_proxy(self, url: str, headers: dict, path: str) -> bytes:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({self.scheme: self.proxy}))
        try:
            with opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotFound(f"not on Gitea: {path}")
            if exc.code in (401, 403):
                raise RuntimeError(f"Gitea refused it ({exc.code}): set a read-only token in ${self.cfg.get('token_env')}")
            raise Transient(f"Gitea answered {exc.code}") if exc.code >= 500 else RuntimeError(f"Gitea answered {exc.code}")
        except (urllib.error.URLError, OSError) as exc:
            raise Transient(f"no Gitea at {self.base} ({getattr(exc, 'reason', exc)})")

    def get(self, path: str, params: dict | None = None):
        return json.loads(self._request(path, params).decode("utf-8"))

    def version(self) -> str:
        return str(self.get("/version").get("version", "?"))

    def repos(self, limit: int = 500) -> list[dict]:
        names = [str(n) for n in (self.cfg.get("repos") or []) if str(n).count("/") == 1]
        if names:
            out = []
            for n in names:
                try:
                    out.append(self.get(f"/repos/{n}"))
                except NotFound:
                    continue
            return [slim(r) for r in out]
        out, page = [], 1
        while len(out) < limit:
            data = self.get("/repos/search", {"limit": 50, "page": page})
            batch = data.get("data") if isinstance(data, dict) else data
            if not batch:
                break
            out += batch
            if len(batch) < 50:
                break
            page += 1
        return [slim(r) for r in out[:limit]]

    def allowed(self, repo: str) -> None:
        names = [str(n) for n in (self.cfg.get("repos") or [])]
        if names and repo not in names:
            raise RuntimeError(f"{repo} is not among the repositories it may read ([tools.config.gitea] repos)")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise RuntimeError(f"not a repository name: {repo}")

    def tree(self, repo: str, branch: str) -> dict:
        self.allowed(repo)
        try:
            commit = str(self.get(f"/repos/{repo}/branches/{urllib.parse.quote(branch, safe='')}")["commit"]["id"])
        except (NotFound, KeyError, TypeError):
            commit = branch
        files, page = [], 1
        while len(files) < 20000:
            data = self.get(f"/repos/{repo}/git/trees/{urllib.parse.quote(commit, safe='')}",
                            {"recursive": "true", "per_page": 1000, "page": page})
            files += [{"path": e["path"], "sha": e.get("sha", ""), "size": int(e.get("size") or 0)}
                      for e in data.get("tree") or [] if e.get("type") == "blob"]
            if not data.get("truncated"):
                break
            page += 1
        return {"commit": commit, "files": files}

    def files(self, repo: str, ref: str, paths: list[str]) -> dict:
        self.allowed(repo)
        limit = int(self.cfg.get("max_file_kb", 256)) * 1024
        budget = int(self.cfg.get("max_batch_kb", 4096)) * 1024

        def read(path: str):
            try:
                body = self._request(f"/repos/{repo}/raw/{urllib.parse.quote(path)}", {"ref": ref}, accept="*/*")
            except Exception as exc:
                return path, None, str(exc)
            if len(body) > limit:
                return path, None, f"larger than {limit // 1024} KB"
            return path, body.decode("utf-8", errors="replace"), None
        out, errors, used = {}, {}, 0
        with ThreadPoolExecutor(max_workers=max(1, int(self.cfg.get("parallel", 6)))) as pool:
            for path, text, err in pool.map(read, [str(p) for p in paths[:200]]):
                if err is not None:
                    errors[path] = err
                elif used + len(text) > budget:
                    errors[path] = "over this call's size budget: ask again"
                else:
                    out[path] = text
                    used += len(text)
        return {"files": out, "errors": errors}


def slim(r: dict) -> dict:
    return {"full_name": r.get("full_name"), "default_branch": r.get("default_branch") or "main",
            "description": str(r.get("description") or "")[:200], "empty": bool(r.get("empty")),
            "archived": bool(r.get("archived")), "updated_at": r.get("updated_at"), "language": r.get("language") or ""}


def handle(request: dict) -> dict:
    cfg = {**DEFAULTS, **(request.get("config") or {})}
    raw = request.get("input", "")
    g = Gitea(cfg)
    if raw == "__selftest__":
        version = g.version()
        repos = g.repos(limit=200)
        return {"output": f"gitea ok: Gitea {version} at {g.base}, {len(repos)} repositor{'y' if len(repos) == 1 else 'ies'}"
                          + (" (with a token)" if g.token else "")}
    req = None
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            req = json.loads(raw)
        except ValueError:
            req = None
    elif isinstance(raw, dict):
        req = raw
    if isinstance(req, dict) and req.get("op"):
        op = req["op"]
        if op == "version":
            return {"output": g.version()}
        if op == "repos":
            repos = g.repos()
            return {"output": f"{len(repos)} repositories", "repos": repos, "version": g.version()}
        if op == "tree":
            t = g.tree(str(req.get("repo", "")), str(req.get("branch") or "main"))
            return {"output": f"{len(t['files'])} files at {t['commit'][:12]}", **t}
        if op == "files":
            got = g.files(str(req.get("repo", "")), str(req.get("ref") or "main"), list(req.get("paths") or []))
            return {"output": f"{len(got['files'])} file(s) read", **got}
        raise RuntimeError(f"unknown request {op}")
    # an order: the repositories that match its words
    words = [w for w in re.findall(r"[a-z0-9_-]+", str(raw).lower()) if w not in STOP and len(w) > 1]
    repos = [r for r in g.repos() if not r["empty"]]
    if words:
        hits = [r for r in repos if any(w in (r["full_name"] + " " + r["description"]).lower() for w in words)]
        if not hits:
            raise NotFound(f"no repository on Gitea matches {' '.join(words)}")
    else:
        hits = repos
    lines = [f"{r['full_name']}" + (f": {r['description']}" if r["description"] else "") for r in hits[:20]]
    return {"output": f"{len(hits)} repositor{'y' if len(hits) == 1 else 'ies'} on Gitea: " + "; ".join(lines),
            "repos": hits[:50]}


def reply(request: dict) -> dict:
    try:
        return handle(request)
    except Transient as exc:
        return {"error": str(exc), "transient": True}
    except NotFound as exc:
        return {"error": str(exc), "not_found": True}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def serve() -> None:
    """Stay running: a request a line on stdin (with an "id"), its answer a line on stdout, many at once."""
    work = ThreadPoolExecutor(max_workers=8, thread_name_prefix="request")
    out = threading.Lock()

    def answer(req: dict) -> None:
        data = reply(req)
        data["id"] = req.get("id")
        line = json.dumps(data)
        with out:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if isinstance(req, dict):
            work.submit(answer, req)
    work.shutdown(wait=True)


def main() -> None:
    print(json.dumps(reply(json.load(sys.stdin))))


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        serve()
    else:
        main()
