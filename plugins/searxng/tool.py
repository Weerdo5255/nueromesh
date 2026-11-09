"""SearXNG tool: searches the web through your own SearXNG (https://docs.searxng.org), a metasearch engine that asks
Google, DuckDuckGo, Bing, Wikipedia and others at once and returns their results.

Only searches: GET /search?q=...&format=json on the configured SearXNG (JSON must be allowed in its settings.yml:
search: formats: [html, json]). It never opens the pages it finds and sends nothing else anywhere. Its user agent is
a plain "neuromesh" - no one's name or address. It sends the headers SearXNG's bot detection looks for (Accept with
text/html, Accept-Language, Accept-Encoding gzip/deflate), so an instance with its limiter on doesn't take it for a
scraper; with the limiter on, SearXNG still allows only a few JSON searches an hour from one address (add the mesh's
address to pass_ip in limiter.toml, or turn the limiter off for your own instance).

Its self-test: SearXNG answers (/config), and a search in JSON is allowed - each failure says what to change.

Requests (the "input"):
    "__selftest__"                              SearXNG answers in JSON: "searxng ok: <instance> <version>, 12 engines"
    {"op": "search", "q", "engines"?, "categories"?, "language"?, "results"?, "time_range"?}
                                                results (title, url, content, engines, score), SearXNG's own answers,
                                                infoboxes, suggestions, and engines that did not answer ("searched":
                                                what it searched for; a tool's "query" is a question for a director)
    anything else                               an order: what it found, the top results with where they came from

Speed: started with --serve (manifest: persistent = true) it stays running and answers requests as JSON lines, many at
once, over kept-alive connections.
"""
import gzip
import http.client
import json
import traceback
import zlib
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULTS = {"url": "http://localhost:8888", "engines": "", "categories": "general", "language": "en", "safesearch": 1,
            "results": 8, "use_proxy": False, "timeout": 20, "user_agent": "neuromesh"}
LOCAL = ("localhost", "127.0.0.1", "::1")
_JSON_OK: dict = {}  # SearXNG address -> when a search in JSON last worked (the health check need not search again)
LEADS = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for|about)|"
                   r"google|look\s+up|find|search\s+online\s+for|search\s+for|what\s+does\s+the\s+(?:web|internet)\s+say\s+about)\s+",
                   re.I)
TAILS = re.compile(r"\s+(?:online|on\s+the\s+(?:web|internet)|on\s+searxng)\s*[?.!]*\s*$", re.I)


class Transient(Exception):
    """SearXNG is down, busy, or its engines did not answer: not a broken tool."""


class Pool:
    """Kept-alive connections, one per thread."""

    def __init__(self):
        self.local = threading.local()

    def conn(self, scheme, netloc, timeout):
        conns = getattr(self.local, "conns", None)
        if conns is None:
            conns = self.local.conns = {}
        c = conns.get((scheme, netloc))
        if c is None:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            c = conns[(scheme, netloc)] = cls(netloc, timeout=timeout)
        return c

    def drop(self, scheme, netloc):
        c = getattr(self.local, "conns", {}).pop((scheme, netloc), None)
        if c is not None:
            c.close()


POOL = Pool()


class SearXNG:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base = str(cfg["url"]).rstrip("/")
        if not self.base.startswith(("http://", "https://")):
            raise RuntimeError(f"SearXNG's url should start with http:// or https:// (it is '{self.base}')")
        self.timeout = float(cfg.get("timeout", 20))
        parts = urllib.parse.urlsplit(self.base)
        self.scheme, self.netloc, self.prefix = parts.scheme, parts.netloc, parts.path.rstrip("/")
        host = parts.hostname or ""
        local = host in LOCAL or host.endswith(".local") or re.match(r"^(10|192\.168|172\.(1[6-9]|2\d|3[01]))\.", host)
        proxy = os.environ.get("https_proxy" if self.scheme == "https" else "http_proxy") or \
            os.environ.get("HTTPS_PROXY" if self.scheme == "https" else "HTTP_PROXY")
        self.proxy = proxy if (proxy and cfg.get("use_proxy") and not local) else None

    def headers(self) -> dict:
        lang = str(self.cfg.get("language") or "en").split("-")[0] or "en"
        return {"accept": "application/json, text/html;q=0.9, */*;q=0.8",       # SearXNG's bot detection wants text/html
                "accept-language": f"{lang},en;q=0.8" if lang != "en" else "en-US,en;q=0.9",  # ... a language
                "accept-encoding": "gzip, deflate",                                # ... and gzip or deflate
                "user-agent": str(self.cfg.get("user_agent") or "neuromesh")}

    def get(self, path: str, params: dict | None = None):
        q = ("?" + urllib.parse.urlencode(params)) if params else ""
        headers = self.headers()
        if self.proxy:
            status, body = self._via_proxy(f"{self.base}{path}{q}", headers)
        else:
            status, body = 0, b""
            for attempt in (0, 1):  # a kept-alive connection the server closed: once more, on a fresh one
                c = POOL.conn(self.scheme, self.netloc, self.timeout)
                try:
                    c.request("GET", f"{self.prefix}{path}{q}", headers=headers)
                    r = c.getresponse()
                    status, body = r.status, _decoded(r.read(), r.getheader("content-encoding", ""))
                    if r.will_close:
                        POOL.drop(self.scheme, self.netloc)
                    break
                except (http.client.HTTPException, OSError) as exc:
                    POOL.drop(self.scheme, self.netloc)
                    if attempt:
                        raise Transient(f"no SearXNG at {self.base} ({exc}): is it running there? Set its address with "
                                        "--searxng URL, [tools.config.searxng] url, or the web search panel")
        if status == 403 and path == "/search":
            raise RuntimeError("SearXNG refused JSON (403): allow it in its settings.yml - search: formats: [html, json] - "
                               "and restart SearXNG")
        if status == 429:
            raise Transient("SearXNG's limiter blocked it (429): for your own instance, set server: limiter: false in "
                            "settings.yml, or add the mesh's address to pass_ip in limiter.toml (with the limiter on, "
                            "SearXNG allows only a few JSON searches an hour)")
        if status == 404:
            raise RuntimeError(f"not on SearXNG: {path}")
        if status >= 500:
            raise Transient(f"SearXNG answered {status}")
        if status != 200:
            raise RuntimeError(f"SearXNG answered {status} for {path}")
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError:
            raise RuntimeError(f"SearXNG's answer to {path} wasn't JSON (is format=json allowed?)")

    def _via_proxy(self, url, headers):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({self.scheme: self.proxy}))
        try:
            with opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout) as resp:
                return resp.status, _decoded(resp.read(), resp.headers.get("content-encoding", ""))
        except urllib.error.HTTPError as exc:
            return exc.code, b""
        # (urllib does not undo gzip: the answer is read below by _decoded)
        except (urllib.error.URLError, OSError) as exc:
            raise Transient(f"no SearXNG at {self.base} ({getattr(exc, 'reason', exc)})")

    def config(self) -> dict:
        try:
            return self.get("/config")
        except Transient:
            raise
        except RuntimeError:
            return {}

    def search(self, q: str, engines: str = "", categories: str = "", language: str = "", results: int = 0,
               time_range: str = "") -> dict:
        q = " ".join(str(q).split())[:400]
        if not q:
            raise RuntimeError("nothing to search for")
        params = {"q": q, "format": "json", "safesearch": int(self.cfg.get("safesearch", 1))}
        for k, v in (("engines", engines or self.cfg.get("engines")), ("categories", categories or self.cfg.get("categories")),
                     ("language", language or self.cfg.get("language")), ("time_range", time_range)):
            if v:
                params[k] = v
        data = self.get("/search", params)
        _JSON_OK[self.base] = time.time()
        n = int(results or self.cfg.get("results") or 8)
        out = []
        for r in data.get("results") or []:
            if not r.get("url"):
                continue
            out.append({"title": " ".join(str(r.get("title") or "").split())[:300],
                        "url": str(r["url"])[:600],
                        "content": " ".join(str(r.get("content") or "").split())[:900],
                        "engines": list(r.get("engines") or ([r["engine"]] if r.get("engine") else []))[:8],
                        "score": round(float(r.get("score") or 0), 3),
                        "published": str(r.get("publishedDate") or "")[:40]})
            if len(out) >= n:
                break
        answers = []
        for a in data.get("answers") or []:
            text = a if isinstance(a, str) else (a.get("answer") or a.get("content") or a.get("text") or "") if isinstance(a, dict) else ""
            url = a.get("url", "") if isinstance(a, dict) else ""
            if str(text).strip():
                answers.append({"answer": " ".join(str(text).split())[:900], "url": str(url or "")[:600]})
        boxes = []
        for b in data.get("infoboxes") or []:
            if isinstance(b, dict):
                boxes.append({"infobox": str(b.get("infobox") or "")[:200], "content": " ".join(str(b.get("content") or "").split())[:1500],
                              "url": str(((b.get("urls") or [{}])[0] or {}).get("url") or b.get("id") or "")[:600],
                              "engine": str(b.get("engine") or "")})
        unresponsive = [str(u[0]) if isinstance(u, (list, tuple)) and u else str(u) for u in data.get("unresponsive_engines") or []]
        if not out and not answers and not boxes and unresponsive:
            raise Transient("SearXNG's engines did not answer: " + ", ".join(unresponsive[:6]))
        return {"searched": q, "results": out, "answers": answers[:4], "infoboxes": boxes[:2],
                "suggestions": [str(x) for x in (data.get("suggestions") or [])][:6], "unresponsive": unresponsive[:10],
                "total": int(data.get("number_of_results") or 0)}


def _decoded(body: bytes, encoding: str) -> bytes:
    enc = (encoding or "").lower()
    try:
        if "gzip" in enc:
            return gzip.decompress(body)
        if "deflate" in enc:
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
    except (OSError, zlib.error, EOFError):
        return body
    return body


def topic_of(text: str) -> str:
    t = LEADS.sub("", str(text).strip(), count=1)
    t = TAILS.sub("", t)
    return t.strip(" ?.!") or str(text).strip()


def handle(request: dict) -> dict:
    cfg = {**DEFAULTS, **(request.get("config") or {})}
    raw = request.get("input", "")
    sx = SearXNG(cfg)
    if raw == "__selftest__":  # SearXNG answers, and allows JSON; whether its engines answered doesn't matter here
        conf = sx.config()
        engines = [e for e in conf.get("engines") or [] if isinstance(e, dict) and e.get("enabled", True)]
        name = conf.get("instance_name") or "SearXNG"
        try:
            if time.time() - _JSON_OK.get(sx.base, 0) < 3600:  # checked within the hour: no search (the limiter counts them)
                got = "JSON checked within the hour"
            else:
                found = sx.search("searxng", results=1)
                got = f"{len(found['results'])} result(s) for a test search"
        except Transient as exc:
            if "429" in str(exc) or "no SearXNG" in str(exc):
                raise
            got = f"its engines didn't answer a test search ({exc}), but it answers in JSON"
        return {"output": f"searxng ok: {name} {conf.get('version', '')}".rstrip()
                          + (f", {len(engines)} engines" if engines else "") + f" at {sx.base}; JSON allowed; {got}"}
    req = None
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            req = json.loads(raw)
        except ValueError:
            req = None
    elif isinstance(raw, dict):
        req = raw
    if isinstance(req, dict) and req.get("op"):
        if req["op"] == "search":
            got = sx.search(str(req.get("q", "")), str(req.get("engines") or ""), str(req.get("categories") or ""),
                            str(req.get("language") or ""), int(req.get("results") or 0), str(req.get("time_range") or ""))
            return {"output": f"{len(got['results'])} results for {got['searched']}", **got}
        raise RuntimeError(f"unknown request {req['op']}")
    got = sx.search(topic_of(raw))  # an order: what it found
    lines = [f"{a['answer']}" + (f" ({a['url']})" if a["url"] else "") for a in got["answers"][:1]]
    lines += [f"{b['infobox']}: {b['content'][:300]}" + (f" ({b['url']})" if b["url"] else "") for b in got["infoboxes"][:1]]
    lines += [f"{r['title']}" + (f" - {r['content'][:200]}" if r["content"] else "") + f" ({r['url']})" for r in got["results"][:5]]
    if not lines:
        return {"error": f"nothing found for {got['searched']}", "not_found": True}
    return {"output": f"On the web ({', '.join(sorted({e for r in got['results'] for e in r['engines']})[:5]) or 'SearXNG'}): "
                      + " | ".join(lines), **got}


def reply(request: dict) -> dict:
    try:
        return handle(request)
    except Transient as exc:
        return {"error": str(exc), "transient": True}
    except Exception as exc:  # a fault in this code: the trace lets the mesh find it (tooldoctor.py)
        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}


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
