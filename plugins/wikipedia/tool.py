"""Wikipedia research tool.

Finds the article that best matches an order ("tell me about Ada Lovelace", "who was ...",
"research the Krebs cycle") through the MediaWiki API and returns it as knowledge for the
mesh to retain: a short summary plus the article split into passages by section, and the
articles it links to from its introduction.

Talks only to the configured Wikipedia API. Settings arrive in the request's "config"
(from manifest.toml [config], overridable in directives.toml [tools.config.wikipedia]).
"""
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULTS = {
    "api_url": "https://{lang}.wikipedia.org/w/api.php",
    "language": "en",
    "user_agent": "NeuroMeshResearchBot/0.5 (https://github.com/Weerdo5255/nueromesh) python-urllib",
    "timeout": 15,
    "max_chars": 60000,
    "max_passages": 40,
    "links": 8,
}
DROP_SECTIONS = {"references", "external links", "see also", "notes", "further reading", "bibliography",
                 "sources", "citations", "footnotes", "works cited", "notes and references", "gallery"}
LEADS = [r"^(please\s+)?(do\s+)?(some\s+)?(research(\s+on)?|look\s+up|lookup|search\s+wikipedia\s+for|"
         r"wikipedia|tell\s+me\s+about|what\s+do\s+you\s+know\s+about|learn\s+about|read\s+up\s+on|"
         r"give\s+me\s+(some\s+)?background\s+on|explain)\s+",
         r"^(who|what|where|when)\s+(is|was|are|were)\s+(the\s+|a\s+|an\s+)?"]


class Transient(Exception):
    """Network trouble: worth retrying later, not a sign the tool is broken."""


def topic_of(text: str) -> str:
    t = " ".join(text.split())
    head, sep, tail = t.partition(":")
    if sep and head.strip().lower() in ("wikipedia", "research", "look up") and tail.strip():
        t = tail.strip()
    t = re.sub(r"\s+on\s+wikipedia$", "", t, flags=re.I)
    t = re.sub(r"[?.!]+$", "", t).strip()
    for pattern in LEADS:
        t = re.sub(pattern, "", t, flags=re.I).strip()
    return t or text.strip()


def api(cfg: dict, **params) -> dict:
    params.update(format="json", formatversion="2")
    url = cfg["api_url"].format(lang=cfg["language"]) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": cfg["user_agent"], "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(cfg["timeout"])) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 500, 502, 503, 504):
            raise Transient(f"Wikipedia returned HTTP {exc.code}") from None
        raise RuntimeError(f"Wikipedia returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise Transient(f"Wikipedia unreachable ({getattr(exc, 'reason', exc)})") from None


def find_title(cfg: dict, topic: str) -> str:
    data = api(cfg, action="query", list="search", srsearch=topic, srlimit=3, srprop="")
    hits = (data.get("query") or {}).get("search") or []
    if not hits:
        raise LookupError(f"Wikipedia has no article matching '{topic}'")
    return hits[0]["title"]


def fetch_page(cfg: dict, title: str) -> dict:
    data = api(cfg, action="query", prop="extracts|info|links", titles=title, redirects=1, explaintext=1,
               exsectionformat="wiki", inprop="url", pllimit="max", plnamespace=0)
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing") or not pages[0].get("extract"):
        raise LookupError(f"Wikipedia has no readable article '{title}'")
    return pages[0]


def sections(extract: str) -> list[tuple[str, str]]:
    """Split a wiki-format plain-text extract into (section, text), dropping reference-type sections."""
    out: list[tuple[str, str]] = []
    current, level, skipping, buf = "Introduction", 2, False, []
    for line in extract.splitlines():
        m = re.match(r"^(={2,})\s*(.+?)\s*\1\s*$", line)
        if m:
            if buf and not skipping:
                out.append((current, "\n".join(buf)))
            buf = []
            depth = len(m.group(1))
            name = m.group(2)
            if depth <= 2:
                skipping = name.strip().lower() in DROP_SECTIONS
            elif not skipping:
                skipping = name.strip().lower() in DROP_SECTIONS
            current, level = name, depth
            continue
        buf.append(line)
    if buf and not skipping:
        out.append((current, "\n".join(buf)))
    return [(name, text.strip()) for name, text in out if text.strip()]


def passages(parts: list[tuple[str, str]], max_passages: int, target: int = 700) -> list[dict]:
    out: list[dict] = []
    for section, text in parts:
        chunk = ""
        for para in [p.strip() for p in text.split("\n") if p.strip()]:
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
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(intro.split()))
    text = ""
    for s in sentences:
        if text and len(text) + len(s) > limit:
            break
        text = f"{text} {s}".strip()
    return text or intro[:limit]


def main() -> None:
    request = json.load(sys.stdin)
    cfg = {**DEFAULTS, **(request.get("config") or {})}
    topic = topic_of(request["input"])
    title = find_title(cfg, topic)
    page = fetch_page(cfg, title)
    extract = page["extract"][: int(cfg["max_chars"])]
    parts = sections(extract)
    intro = parts[0][1] if parts and parts[0][0] == "Introduction" else extract[:1500]
    summary = summary_of(intro)
    lowered = intro.lower()
    linked = [link["title"] for link in page.get("links") or [] if link.get("title")]
    linked = sorted((t for t in linked if t.lower() in lowered and t != page["title"]),
                    key=lambda t: lowered.find(t.lower()))
    knowledge = {
        "source": "wikipedia", "language": cfg["language"], "title": page["title"],
        "url": page.get("fullurl") or f"https://{cfg['language']}.wikipedia.org/wiki/"
                                      f"{urllib.parse.quote(page['title'].replace(' ', '_'))}",
        "revision": str(page.get("lastrevid", "")), "topic": topic, "summary": summary,
        "passages": passages(parts, int(cfg["max_passages"])), "links": linked[: int(cfg["links"])],
    }
    print(json.dumps({"output": f"{summary}\n(Wikipedia: {page['title']})", "knowledge": knowledge}))


if __name__ == "__main__":
    try:
        main()
    except Transient as exc:
        print(json.dumps({"error": str(exc), "transient": True}))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
