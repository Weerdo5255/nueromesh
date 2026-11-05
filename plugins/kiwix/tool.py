"""Kiwix research tool.

Looks a topic up on a local kiwix-serve (https://kiwix.org: offline Wikipedia and other ZIM
archives) and returns the article as knowledge for the mesh to retain, in the same shape as the
Wikipedia tool: a summary, the article split into passages by section, and the articles its
introduction links to.

Finding an article tries, in order, the server's /suggest (JSON), /search?format=xml (RSS) and
the HTML search page, then a direct title guess, so it works across kiwix-serve versions.
Books come from the OPDS catalog (/catalog/v2/entries). Talks only to the configured server.
"""
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

DEFAULTS = {"url": "http://localhost:8080", "books": [], "language": "eng", "use_proxy": False, "timeout": 15,
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


def candidates(server: Server, book: dict, topic: str) -> list[tuple[str, str]]:
    """(title, article URL path) pairs, best first."""
    found: list[tuple[str, str]] = []
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
def main() -> None:
    request = json.load(sys.stdin)
    cfg = {**DEFAULTS, **(request.get("config") or {})}
    server = Server(cfg)
    served = books(server)
    if request["input"].strip() == "__selftest__":
        if not served:
            raise RuntimeError("the Kiwix server is up but serves no books")
        names = ", ".join(b["name"] for b in served[:4]) + (", ..." if len(served) > 4 else "")
        print(json.dumps({"output": f"kiwix ok: {len(served)} book{'s' if len(served) != 1 else ''} ({names})"}))
        return
    if not served:
        raise NotFound("the Kiwix server serves no books")
    topic = topic_of(request["input"])
    bare = re.sub(r"^(the|a|an)\s+", "", topic, flags=re.I)  # titles rarely start with an article
    for book in choose(served, cfg):
        seen: set[str] = set()
        found = [c for term in dict.fromkeys([topic, bare]) for c in candidates(server, book, term)]
        for title, path in found:
            if path in seen:
                continue
            seen.add(path)
            try:
                art = read_article(server, path, title)
            except NotFound:
                continue
            if art is None:
                continue
            summary = summary_of(art["intro"])
            knowledge = {
                "source": "kiwix", "language": book["language"] or cfg["language"], "title": art["title"],
                "url": art["url"], "revision": book["name"], "topic": topic, "summary": summary,
                "passages": passages(art["parts"], int(cfg["max_passages"])),
                "links": art["links"][: int(cfg["links"])],
            }
            print(json.dumps({"output": f"{summary}\n(Kiwix: {art['title']}, {book['title'] or book['name']})",
                              "knowledge": knowledge}))
            return
    raise NotFound(f"no article matching '{topic}' in {', '.join(b['name'] for b in served)}")


if __name__ == "__main__":
    try:
        main()
    except Transient as exc:
        print(json.dumps({"error": str(exc), "transient": True}))
    except NotFound as exc:
        print(json.dumps({"error": str(exc), "not_found": True}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
