"""Kiwix research tool.

Looks a topic up on a local kiwix-serve (https://kiwix.org: offline Wikipedia and other ZIM
archives) and returns the article as knowledge for the mesh to retain, in the same shape as the
Wikipedia tool: a summary, the article split into passages by section, and the articles its
introduction links to.

Finding an article tries, in order, the server's /suggest (JSON), /search?format=xml (RSS) and
the HTML search page, then a direct title guess, so it works across kiwix-serve versions; with
"search": "fulltext" the full-text search comes first (it finds articles that mention a thing, not
only those named after it). Books come from each server's OPDS catalog (/catalog/v2/entries).

More than one server: "url" and the extra "servers" are all searched, in that order; a server that
is down is skipped. For one search of many (the mesh's Kiwix search, seeker.py) the request may
narrow things down: "only" [{"server", "book"}] searches just those books, "exclude" skips articles
already read. "__catalog__" lists every book on every server; "__random__" reads a random article
from the "only" book (for browsing a book it has not opened in a long time). Talks only to the
configured servers.

Speed: started with --serve (manifest: persistent = true) it stays running and answers requests as
JSON lines, many at once: connections to the servers are kept alive, the book catalog and what it read
are cached, and every book is searched at the same time (the best book's article still wins).
"""
import html
import http.client
import json
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

DEFAULTS = {"url": "http://localhost:8080", "servers": [], "books": [], "language": "eng", "use_proxy": False, "timeout": 15,
            "max_chars": 80000, "max_passages": 40, "links": 8,
            "user_agent": "NeuroMesh (+https://github.com/Weerdo5255/nueromesh) python-urllib"}
DROP_SECTIONS = {"references", "external links", "see also", "notes", "further reading", "bibliography",
                 "sources", "citations", "footnotes", "works cited", "notes and references", "gallery"}
LEADS = [r"^(please\s+)?(do\s+)?(some\s+)?(research(\s+on)?|look\s+up|lookup|search\s+(kiwix|wikipedia)\s+for|"
         r"kiwix|wikipedia|tell\s+me\s+about|what\s+do\s+you\s+know\s+about|learn\s+about|read\s+up\s+on|"
         r"give\s+me\s+(some\s+)?background\s+on|explain)\s+",
         r"^(who|what|where|when)\s+(is|was|are|were)\s+(the\s+|a\s+|an\s+)?"]
SKIP_TAGS = {"script", "style", "noscript", "table", "sup", "figure", "math", "nav", "footer", "header", "button"}
SKIP_CLASSES = ("navbox", "reflist", "references", "mw-references", "hatnote", "mw-editsection", "metadata",
                "infobox", "thumb", "toc", "mw-jump", "noprint")
VOID = {"br", "img", "hr", "meta", "link", "input", "wbr", "source", "area", "base", "col", "embed", "param",
        "track"}


class Transient(Exception):
    """The server is unreachable or busy: worth retrying, not a sign the tool is broken."""


class NotFound(Exception):
    """The server works but has no such article."""


class Pool:
    """--serve: keep-alive connections (one per thread and server) and a cache of the pages it read."""

    def __init__(self, cache_mb: int = 96):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.cache: OrderedDict = OrderedDict()
        self.size, self.limit = 0, cache_mb << 20
        self.hits = self.misses = 0

    def _conn(self, scheme: str, netloc: str, timeout: float):
        conns = getattr(self.local, "conns", None)
        if conns is None:
            conns = self.local.conns = {}
        c = conns.get((scheme, netloc))
        if c is None:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            c = conns[(scheme, netloc)] = cls(netloc, timeout=timeout)
        return c

    def _drop(self, scheme: str, netloc: str) -> None:
        c = getattr(self.local, "conns", {}).pop((scheme, netloc), None)
        if c is not None:
            c.close()

    def fetch(self, url: str, headers: dict, timeout: float) -> tuple[int, str, str]:
        with self.lock:
            hit = self.cache.get(url)
            if hit is not None:
                self.cache.move_to_end(url)
                self.hits += 1
                return 200, hit[0], hit[1]
            self.misses += 1
        start = url
        for _ in range(6):  # redirects (a random article, a moved title)
            parts = urllib.parse.urlsplit(url)
            path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
            for attempt in (0, 1):  # a kept-alive connection the server closed: once more, on a fresh one
                c = self._conn(parts.scheme, parts.netloc, timeout)
                try:
                    c.request("GET", path, headers=headers)
                    r = c.getresponse()
                    body = r.read()
                    if r.will_close:
                        self._drop(parts.scheme, parts.netloc)
                    break
                except (http.client.HTTPException, OSError):
                    self._drop(parts.scheme, parts.netloc)
                    if attempt:
                        raise
            loc = r.getheader("Location")
            if r.status in (301, 302, 303, 307, 308) and loc:
                url = urllib.parse.urljoin(url, loc)
                continue
            text = body.decode(r.msg.get_content_charset() or "utf-8", errors="replace")
            if r.status == 200 and "/random" not in start and "/catalog/" not in start:
                with self.lock:
                    for key in {start, url}:
                        if key not in self.cache:
                            self.cache[key] = (text, url)
                            self.size += len(text)
                    while self.size > self.limit and self.cache:
                        _, (old, _) = self.cache.popitem(last=False)
                        self.size -= len(old)
            return r.status, text, url
        return 508, "", url


POOL: Pool | None = None                  # --serve only
FAN: ThreadPoolExecutor | None = None     # --serve only: every book searched at once
CATALOG: dict = {}                        # --serve only: the catalog, per configuration


def topic_of(text: str) -> str:
    t = " ".join(text.split())
    head, sep, tail = t.partition(":")
    if sep and head.strip().lower() in ("kiwix", "wikipedia", "research", "look up") and tail.strip():
        t = tail.strip()
    t = re.sub(r"\s+on\s+(kiwix|wikipedia)$", "", t, flags=re.I)
    t = re.sub(r"[?.!]+$", "", t).strip()
    for pattern in LEADS:
        t = re.sub(pattern, "", t, flags=re.I).strip()
    return t or text.strip()


# --------------------------------------------------------------------------- http
class Server:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base = cfg["url"].rstrip("/")
        parts = urllib.parse.urlsplit(self.base)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise RuntimeError(f"not a server URL: {cfg['url']!r} (set [tools.config.kiwix] url)")
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.prefix = parts.path.rstrip("/")   # e.g. "/kiwix" when served under --urlRootLocation
        handlers = [] if cfg.get("use_proxy") else [urllib.request.ProxyHandler({})]
        self.opener = urllib.request.build_opener(*handlers)

    def url(self, path: str, params: dict | None = None, link: bool = False) -> str:
        """An API endpoint ("/suggest") is relative to the configured URL; a link the server gave
        us ("/kiwix/content/...") is relative to its origin."""
        if path.startswith(("http://", "https://")):
            url = path
        elif link:
            url = urllib.parse.urljoin(self.origin + "/", path)
        else:
            url = self.base + "/" + path.lstrip("/")
        return url + ("?" + urllib.parse.urlencode(params) if params else "")

    def get(self, path: str, params: dict | None = None, link: bool = False) -> tuple[str, str]:
        url = self.url(path, params, link)
        if POOL is not None and not self.cfg.get("use_proxy"):
            try:
                status, text, final = POOL.fetch(url, {"User-Agent": self.cfg["user_agent"], "Connection": "keep-alive"},
                                                 float(self.cfg["timeout"]))
            except (http.client.HTTPException, OSError) as exc:
                raise Transient(f"Kiwix server unreachable at {self.base} ({exc})") from None
            if status in (404, 400):
                raise NotFound(f"HTTP {status} for {url}")
            if status in (429, 500, 502, 503, 504):
                raise Transient(f"Kiwix server returned HTTP {status}")
            if status >= 300:
                raise RuntimeError(f"Kiwix server returned HTTP {status} for {url}")
            return text, final
        req = urllib.request.Request(url, headers={"User-Agent": self.cfg["user_agent"]})
        try:
            with self.opener.open(req, timeout=float(self.cfg["timeout"])) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace"), resp.geturl()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 400):
                raise NotFound(f"HTTP {exc.code} for {url}") from None
            if exc.code in (429, 500, 502, 503, 504):
                raise Transient(f"Kiwix server returned HTTP {exc.code}") from None
            raise RuntimeError(f"Kiwix server returned HTTP {exc.code} for {url}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise Transient(f"Kiwix server unreachable at {self.base} ({getattr(exc, 'reason', exc)})") from None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml(text: str) -> ET.Element:
    # some kiwix-serve versions emit unescaped "&" in attributes
    return ET.fromstring(re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", text))


def books(server: Server) -> list[dict]:
    text, _ = server.get("/catalog/v2/entries", {"count": "-1"})
    out = []
    for entry in _xml(text).iter():
        if _local(entry.tag) != "entry":
            continue
        book = {"name": "", "title": "", "language": "", "category": "", "href": ""}
        for child in entry:
            tag = _local(child.tag)
            if tag in ("name", "title", "language", "category") and child.text:
                book[tag] = child.text.strip()
            elif tag == "link" and (child.get("type") or "").startswith("text/html") and child.get("href"):
                book["href"] = child.get("href")
        if book["name"]:
            book["href"] = book["href"] or f"{server.prefix}/content/{book['name']}"
            out.append(book)
    return out


def choose(all_books: list[dict], cfg: dict) -> list[dict]:
    wanted = [b.strip() for b in cfg.get("books") or [] if b.strip()]
    if wanted:
        picked = [b for w in wanted for b in all_books if b["name"] == w or b["name"].startswith(w)]
        if not picked:
            raise RuntimeError(f"none of the configured books {wanted} is served "
                               f"(served: {', '.join(b['name'] for b in all_books) or 'none'})")
        return picked
    lang = str(cfg.get("language") or "").lower()

    def rank(b):
        wiki = "wikipedia" in (b["category"] + " " + b["name"]).lower()
        return (not wiki, bool(lang) and lang not in b["language"].lower(), b["name"])
    return sorted(all_books, key=rank)


# ------------------------------------------------------------------ finding articles
def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", " ", urllib.parse.unquote(s)).strip().lower()


def _title_from_path(path: str) -> str:
    last = urllib.parse.unquote(path.rstrip("/").rsplit("/", 1)[-1])
    return last.replace("_", " ")


def fulltext(server: Server, book: dict, topic: str) -> list[tuple[str, str]]:
    """Articles that mention a topic (the server's full-text search), best first."""
    found: list[tuple[str, str]] = []
    try:
        text, _ = server.get("/search", {"books.name": book["name"], "pattern": topic, "format": "xml", "pageLength": "8"})
        for item in _xml(text).iter():
            if _local(item.tag) == "item":
                title = link = ""
                for child in item:
                    if _local(child.tag) == "title" and child.text:
                        title = child.text.strip()
                    elif _local(child.tag) == "link" and child.text:
                        link = child.text.strip()
                if link:
                    found.append((title or _title_from_path(link), link))
    except (NotFound, ET.ParseError):
        pass
    return found


def candidates(server: Server, book: dict, topic: str, full: bool = False) -> list[tuple[str, str]]:
    """(title, article URL path) pairs, best first. full: the full-text search first."""
    found: list[tuple[str, str]] = fulltext(server, book, topic) if full else []
    if found:
        return found
    for params in ({"content": book["name"], "term": topic, "count": "5"},
                   {"books.name": book["name"], "term": topic, "count": "5"}):
        try:
            text, _ = server.get("/suggest", params)
            items = json.loads(text)
        except (NotFound, ValueError):
            continue
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get("path") and it.get("kind", "path") == "path":
                title = html.unescape(re.sub(r"<[^>]+>", "", str(it.get("value") or it.get("label") or "")))
                found.append((title or _title_from_path(it["path"]), f"{book['href'].rstrip('/')}/{it['path'].lstrip('/')}"))
        if found:
            break
    if not found:
        try:
            text, _ = server.get("/search", {"books.name": book["name"], "pattern": topic, "format": "xml",
                                            "pageLength": "5"})
            for item in _xml(text).iter():
                if _local(item.tag) == "item":
                    title = link = ""
                    for child in item:
                        if _local(child.tag) == "title" and child.text:
                            title = child.text.strip()
                        elif _local(child.tag) == "link" and child.text:
                            link = child.text.strip()
                    if link:
                        found.append((title or _title_from_path(link), link))
        except (NotFound, ET.ParseError):
            pass
    if not found:
        try:
            text, _ = server.get("/search", {"content": book["name"], "pattern": topic})
            content = book["href"].rstrip("/") + "/"
            for href, label in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
                href = html.unescape(href)
                if content in href or href.startswith(f"/{book['name']}/"):
                    found.append((html.unescape(re.sub(r"<[^>]+>", "", label)).strip(), href))
        except NotFound:
            pass
    guess = topic[:1].upper() + topic[1:]
    slug = urllib.parse.quote(guess.replace(" ", "_"))
    found += [(guess, f"{book['href'].rstrip('/')}/A/{slug}"), (guess, f"{book['href'].rstrip('/')}/{slug}")]
    wanted = _norm(topic)
    found.sort(key=lambda c: _norm(c[0]) != wanted)  # an exact title first (stable otherwise)
    return found


# -------------------------------------------------------------------- reading HTML
class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.sections: list[list] = [["Introduction", []]]
        self.links: list[str] = []
        self._stack: list[tuple[str, bool]] = []  # (tag, skipping)
        self._buf: list[str] = []
        self._in: str | None = None              # p | li | h | title | h1
        self._href: str | None = None

    @property
    def skipping(self) -> bool:
        return bool(self._stack) and self._stack[-1][1]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").lower()
        skip = self.skipping or tag in SKIP_TAGS or any(c in cls for c in SKIP_CLASSES) \
            or (a.get("role") == "navigation")
        if tag not in VOID:
            self._stack.append((tag, skip))
        if skip:
            return
        if tag == "title" and not self.title:
            self._in, self._buf = "title", []
        elif tag == "h1" and not self.title:
            self._in, self._buf = "h1", []
        elif tag in ("h2", "h3"):
            self._flush()
            self._in, self._buf = "h", []
        elif tag in ("p", "li", "dd") and self._in is None:
            self._in, self._buf = "p", []
        elif tag == "a" and self._in == "p" and len(self.sections) == 1:
            self._href = a.get("href")

    def handle_endtag(self, tag):
        while self._stack:
            top, _ = self._stack.pop()
            if top == tag:
                break
        if tag in ("title", "h1") and self._in in ("title", "h1"):
            self.title = " ".join("".join(self._buf).split())
            self._in, self._buf = None, []
        elif tag in ("h2", "h3") and self._in == "h":
            name = " ".join("".join(self._buf).split())
            self.sections.append([name or "Section", []])
            self._in, self._buf = None, []
        elif tag in ("p", "li", "dd") and self._in == "p":
            self._flush()
        elif tag == "a":
            self._href = None

    def handle_data(self, data):
        if self.skipping or self._in is None:
            return
        self._buf.append(data)
        if self._href and self._in == "p" and data.strip():
            self.links.append(self._href)
            self._href = None

    def _flush(self):
        if self._in == "p":
            text = " ".join("".join(self._buf).split())
            if text:
                self.sections[-1][1].append(text)
        self._in, self._buf = None, []


def article_links(hrefs: list[str], self_title: str) -> list[str]:
    out: list[str] = []
    for href in hrefs:
        if not href or href.startswith(("#", "mailto:")) or "://" in href:
            continue
        path = href.split("#", 1)[0].split("?", 1)[0]
        last = urllib.parse.unquote(path.rstrip("/").rsplit("/", 1)[-1])
        if not last or ":" in last or "." in last.rsplit("/", 1)[-1][-5:]:
            continue  # File:, Category:, images, stylesheets
        title = last.replace("_", " ")
        if title.lower() != self_title.lower() and title not in out:
            out.append(title)
    return out


def passages(parts: list[tuple[str, str]], max_passages: int, target: int = 700) -> list[dict]:
    out: list[dict] = []
    for section, paras in parts:
        chunk = ""
        for para in paras:
            if chunk and len(chunk) + len(para) > target:
                out.append({"section": section, "text": chunk})
                chunk = ""
            chunk = f"{chunk} {para}".strip()
        if chunk:
            out.append({"section": section, "text": chunk})
        if len(out) >= max_passages:
            break
    return out[:max_passages]


def summary_of(intro: str, limit: int = 700) -> str:
    text = ""
    for s in re.split(r"(?<=[.!?])\s+", " ".join(intro.split())):
        if text and len(text) + len(s) > limit:
            break
        text = f"{text} {s}".strip()
    return text or intro[:limit]


def read_article(server: Server, path: str, fallback_title: str) -> dict | None:
    page, final = server.get(path, link=True)
    return parse_article(server, page, final, fallback_title)


def parse_article(server: Server, page: str, final: str, fallback_title: str) -> dict | None:
    parser = ArticleParser()
    parser.feed(page[: 4 * int(server.cfg["max_chars"])])
    parser._flush()
    parts = [(name, paras) for name, paras in parser.sections
             if paras and name.strip().lower() not in DROP_SECTIONS]
    if not parts:
        return None
    title = re.sub(r"\s+[-–|]\s+(wikipedia|kiwix)\b.*$", "", parser.title, flags=re.I).strip() or fallback_title
    intro = " ".join(parts[0][1])  # the introduction, or the first section if there is none
    return {"title": title, "url": final, "intro": intro, "parts": parts,
            "links": article_links(parser.links, title)}


# ---------------------------------------------------------------------------- main
def servers_of(cfg: dict) -> list[Server]:
    """The configured server and the extra ones, each once, in order."""
    urls = [cfg["url"]] + [u for u in (cfg.get("servers") or []) if isinstance(u, str) and u.strip()]
    out, seen = [], set()
    for u in urls:
        key = u.strip().rstrip("/")
        if key and key not in seen:
            seen.add(key)
            out.append(Server({**cfg, "url": key}))
    return out


def catalog(cfg: dict, fresh: bool = False) -> tuple[list[tuple[Server, dict]], list[str]]:
    """Every book on every server, chosen and ranked per server ([books], [language]); and the servers that are
    down. --serve keeps it for "catalog_seconds" (a request for "__catalog__" always reads it again)."""
    key = json.dumps([cfg["url"], cfg.get("servers"), cfg.get("books"), cfg.get("language")], sort_keys=True)
    if POOL is not None and not fresh:
        hit = CATALOG.get(key)
        if hit and time.time() - hit[0] < float(cfg.get("catalog_seconds", 600)) and hit[1]:
            return hit[1], hit[2]
    out, down = [], []
    for server in servers_of(cfg):
        try:
            served = books(server)
        except Transient as exc:
            down.append(f"{server.base} ({exc})")
            continue
        try:
            picked = choose(served, cfg)
        except RuntimeError:
            picked = []  # this server serves none of the configured books
        out += [(server, b) for b in picked]
    if POOL is not None:
        CATALOG[key] = (time.time(), out, down)
    return out, down


def knowledge_of(cfg: dict, server: Server, book: dict, art: dict, topic: str) -> dict:
    summary = summary_of(art["intro"])
    knowledge = {
        "source": "kiwix", "language": book["language"] or cfg["language"], "title": art["title"],
        "url": art["url"], "revision": book["name"], "topic": topic, "summary": summary,
        "passages": passages(art["parts"], int(cfg["max_passages"])),
        "links": art["links"][: int(cfg["links"])], "book": book["title"] or book["name"], "server": server.base,
    }
    return {"output": f"{summary}\n(Kiwix: {art['title']}, {book['title'] or book['name']})", "knowledge": knowledge}


def random_article(cfg: dict, sources: list, exclude: set) -> dict:
    """A random article from the book asked for (browsing: something new from a book it has not opened lately)."""
    for server, book in sources:
        for _ in range(4):
            page = None
            for params in ({"content": book["name"]}, {"books.name": book["name"]}):
                try:
                    page, final = server.get("/random", params)
                    break
                except NotFound:
                    continue
            if page is None:
                break
            art = parse_article(server, page, final, _title_from_path(final))
            if art is None or art["url"] in exclude:
                continue
            return knowledge_of(cfg, server, book, art, art["title"])
    raise NotFound(f"no random article from {', '.join(b['name'] for _, b in sources)}")


def handle(request: dict) -> dict:
    cfg = {**DEFAULTS, **(request.get("config") or {})}
    what = str(request.get("input", "")).strip()
    sources, down = catalog(cfg, fresh=what in ("__catalog__", "__selftest__"))
    if what in ("__selftest__", "__catalog__"):
        if not sources:
            if down and len(down) == len(servers_of(cfg)):
                raise Transient("; ".join(down))
            raise RuntimeError("the Kiwix server is up but serves no books")
        if what == "__catalog__":
            listed = [{"server": srv.base, "name": b["name"], "title": b["title"], "language": b["language"],
                       "category": b["category"]} for srv, b in sources]
            return {"output": f"{len(listed)} book{'s' if len(listed) != 1 else ''} on "
                              f"{len({x['server'] for x in listed})} server(s)", "books": listed, "down": down}
        names = ", ".join(b["name"] for _, b in sources[:4]) + (", ..." if len(sources) > 4 else "")
        return {"output": f"kiwix ok: {len(sources)} book{'s' if len(sources) != 1 else ''} ({names})"}
    if not sources:
        if down:
            raise Transient("; ".join(down))
        raise NotFound("the Kiwix server serves no books")
    only = [(str(o.get("server", "")).rstrip("/"), str(o.get("book", ""))) for o in (cfg.get("only") or []) if isinstance(o, dict)]
    if only:
        sources = [(srv, b) for srv, b in sources if any((not s_ or s_ == srv.base) and b["name"] == n for s_, n in only)]
        if not sources:
            raise NotFound(f"none of the books asked for is served: {', '.join(n for _, n in only)}")
    exclude = {str(u) for u in (cfg.get("exclude") or [])}
    if what == "__random__":
        return random_article(cfg, sources, exclude)
    full = str(cfg.get("search", "title")).lower() == "fulltext"
    topic = topic_of(what)
    bare = re.sub(r"^(the|a|an)\s+", "", topic, flags=re.I)  # titles rarely start with an article

    def look(src):
        server, book = src
        try:
            return [c for term in dict.fromkeys([topic, bare]) for c in candidates(server, book, term, full)]
        except Transient:
            return None
    found_all = list(FAN.map(look, sources)) if FAN is not None and len(sources) > 1 else [look(x) for x in sources]
    unreachable = sum(1 for f in found_all if f is None)
    for (server, book), found in zip(sources, found_all):  # in the books' order: the best book's article wins
        if found is None:
            continue
        seen: set[str] = set()
        for title, path in found:
            if path in seen:
                continue
            seen.add(path)
            try:
                art = read_article(server, path, title)
            except NotFound:
                continue
            except Transient:
                break
            if art is None or art["url"] in exclude or path in exclude:
                continue
            return knowledge_of(cfg, server, book, art, topic)
    if unreachable and unreachable == len(sources):
        raise Transient(f"no Kiwix server could be searched for '{topic}'")
    raise NotFound(f"no {'article mentioning' if full else 'article matching'} '{topic}' in "
                   f"{', '.join(b['name'] for _, b in sources)}")


def reply(request: dict) -> dict:
    try:
        return handle(request)
    except Transient as exc:
        return {"error": str(exc), "transient": True}
    except NotFound as exc:
        return {"error": str(exc), "not_found": True}
    except Exception as exc:  # a fault in this code: the trace lets the mesh find it (tooldoctor.py)
        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}


def serve() -> None:
    """Stay running: a request a line on stdin (with an "id"), its answer a line on stdout, many at once."""
    global POOL, FAN
    POOL = Pool()
    FAN = ThreadPoolExecutor(max_workers=8, thread_name_prefix="book")
    work = ThreadPoolExecutor(max_workers=16, thread_name_prefix="request")
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
