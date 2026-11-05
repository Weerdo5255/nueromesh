"""Retained knowledge: what the mesh has learned from research tools such as Wikipedia.

Articles are stored as documents plus passages (one per paragraph group) in the same
SQLite database as everything else, so they survive restarts. Passages are indexed with
SQLite FTS5 (stemmed, BM25-ranked); a plain scan is used if FTS5 is unavailable.

Whether retained knowledge is good enough to answer from is a *confidence* between 0 and
1: half for how fully the question names the article's subject (its title), half for how
much of the question a passage covers. "Who was Ada Lovelace?" against the article "Ada
Lovelace" scores 1.0; "population of France" against the article "Paris" stays below the
threshold, so the mesh goes and researches France instead of answering from the wrong page.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

STOPWORDS = set("""
a an the and or but if of in on at to for from by with about into over after before under as is are was were
be been being am do does did done have has had having it its it's this that these those there here what who whom
whose which when where why how whats who's what's whos tell me us you your i we they he she his her their our my
please research look up lookup find search wikipedia know knows known some any much many more most give show
explain learn read background info information can could would should will shall may might must also than then
""".split())
DOCS_SCHEMA = """
CREATE TABLE IF NOT EXISTS kn_documents(
    id INTEGER PRIMARY KEY, source TEXT, title TEXT, url TEXT, revision TEXT, summary TEXT, topic TEXT,
    reason TEXT, language TEXT, links TEXT, passages INTEGER, fetched_at REAL, last_used REAL,
    hits INTEGER DEFAULT 0, aliases TEXT DEFAULT '[]', UNIQUE(source, title));
"""
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def content_words(text: str) -> list[str]:
    seen, out = set(), []
    for t in tokens(text):
        if t not in STOPWORDS and t not in seen and (len(t) > 1 or t.isdigit()):
            seen.add(t)
            out.append(t)
    return out


def _same(a: str, b: str) -> bool:
    """Loose word match: equal, or one is a prefix of the other (died/die, capitals/capital)."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 3


def coverage(query: list[str], words: set[str]) -> float:
    if not query:
        return 0.0
    return sum(1 for q in query if any(_same(q, w) for w in words)) / len(query)


@dataclass
class Document:
    id: int
    source: str
    title: str
    url: str
    revision: str
    summary: str
    topic: str
    reason: str
    language: str
    links: list
    passages: int
    fetched_at: float
    last_used: float
    hits: int
    aliases: list = field(default_factory=list)  # questions/topics that led to this article

    @property
    def names(self) -> list[str]:
        return [self.title, *self.aliases]

    @property
    def age(self) -> float:
        return time.time() - self.fetched_at


@dataclass
class Hit:
    doc: Document
    section: str
    text: str
    passage_coverage: float
    title_coverage: float
    rank: float = 0.0

    @property
    def confidence(self) -> float:
        return round(0.5 * self.title_coverage + 0.5 * self.passage_coverage, 3)


@dataclass
class KnowledgeStats:
    documents: int
    passages: int
    recalled: int = 0
    researched: int = 0
    sources: dict = field(default_factory=dict)


class KnowledgeStore:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(DOCS_SCHEMA)
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS kn_passages USING fts5("
                            "title, section, text, doc_id UNINDEXED, tokenize='porter unicode61')")
            self.fts = True
        except sqlite3.OperationalError:  # SQLite built without FTS5
            self.db.execute("CREATE TABLE IF NOT EXISTS kn_passages_plain(doc_id INTEGER, title TEXT, "
                            "section TEXT, text TEXT)")
            self.fts = False
        self.recalled = 0
        self.researched = 0

    # ---------------------------------------------------------------- writing
    def add(self, k: dict, reason: str) -> Document:
        """Retain (or refresh) one article returned by a research tool."""
        now = time.time()
        source, title = k.get("source", "unknown"), k["title"]
        passages = [p for p in k.get("passages", []) if str(p.get("text", "")).strip()]
        row = self.db.execute("SELECT id, aliases FROM kn_documents WHERE source=? AND title=?",
                              (source, title)).fetchone()
        aliases = json.loads(row[1] or "[]") if row else []
        topic = " ".join(str(k.get("topic", "")).split())
        if topic and topic.lower() != title.lower() and topic.lower() not in (a.lower() for a in aliases):
            aliases = (aliases + [topic])[-12:]
        self.db.execute("BEGIN")
        try:
            values = (k.get("url", ""), str(k.get("revision", "")), k.get("summary", ""), k.get("topic", ""),
                      reason, k.get("language", ""), json.dumps(k.get("links", [])), len(passages), now,
                      json.dumps(aliases))
            if row:
                doc_id = row[0]
                self.db.execute("UPDATE kn_documents SET url=?, revision=?, summary=?, topic=?, reason=?, "
                                "language=?, links=?, passages=?, fetched_at=?, aliases=? WHERE id=?",
                                (*values, doc_id))
                self._delete_passages(doc_id)
            else:
                doc_id = self.db.execute(
                    "INSERT INTO kn_documents(url, revision, summary, topic, reason, language, links, passages, "
                    "fetched_at, aliases, source, title, last_used, hits) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                    (*values, source, title, now)).lastrowid
            table = "kn_passages" if self.fts else "kn_passages_plain"
            self.db.executemany(f"INSERT INTO {table}(title, section, text, doc_id) VALUES(?,?,?,?)",
                                [(title, p.get("section", ""), p["text"], doc_id) for p in passages])
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        self.researched += 1
        return self.get(title, source)

    def _delete_passages(self, doc_id: int) -> None:
        table = "kn_passages" if self.fts else "kn_passages_plain"
        self.db.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))

    def forget(self, title: str) -> bool:
        row = self.db.execute("SELECT id FROM kn_documents WHERE lower(title)=lower(?)", (title,)).fetchone()
        if not row:
            return False
        self._delete_passages(row[0])
        self.db.execute("DELETE FROM kn_documents WHERE id=?", (row[0],))
        return True

    def enforce_cap(self, max_documents: int) -> list[str]:
        """Forget the least recently used documents beyond the cap."""
        excess = self.count() - max_documents
        if excess <= 0:
            return []
        rows = self.db.execute("SELECT title FROM kn_documents ORDER BY last_used, fetched_at LIMIT ?",
                               (excess,)).fetchall()
        for (title,) in rows:
            self.forget(title)
        return [t for (t,) in rows]

    def touch(self, doc: Document) -> None:
        self.db.execute("UPDATE kn_documents SET hits=hits+1, last_used=? WHERE id=?", (time.time(), doc.id))
        self.recalled += 1

    # ---------------------------------------------------------------- reading
    _COLS = ("id, source, title, url, revision, summary, topic, reason, language, links, passages, "
             "fetched_at, last_used, hits, aliases")

    def _doc(self, row) -> Document:
        d = dict(zip([c.strip() for c in self._COLS.split(",")], row))
        d["links"] = json.loads(d["links"] or "[]")
        d["aliases"] = json.loads(d["aliases"] or "[]")
        return Document(**d)

    def get(self, title: str, source: str | None = None) -> Document | None:
        sql = f"SELECT {self._COLS} FROM kn_documents WHERE lower(title)=lower(?)"
        args: tuple = (title,)
        if source:
            sql += " AND source=?"
            args = (title, source)
        row = self.db.execute(sql, args).fetchone()
        return self._doc(row) if row else None

    def get_by_id(self, doc_id: int) -> Document | None:
        row = self.db.execute(f"SELECT {self._COLS} FROM kn_documents WHERE id=?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def documents(self, limit: int = 20) -> list[Document]:
        rows = self.db.execute(f"SELECT {self._COLS} FROM kn_documents ORDER BY fetched_at DESC LIMIT ?",
                               (limit,)).fetchall()
        return [self._doc(r) for r in rows]

    def stale(self, days: float, limit: int = 5) -> list[Document]:
        cutoff = time.time() - days * 86400
        rows = self.db.execute(f"SELECT {self._COLS} FROM kn_documents WHERE fetched_at < ? "
                               "ORDER BY hits DESC, fetched_at LIMIT ?", (cutoff, limit)).fetchall()
        return [self._doc(r) for r in rows]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM kn_documents").fetchone()[0]

    def stats(self) -> KnowledgeStats:
        passages = self.db.execute("SELECT COALESCE(SUM(passages), 0) FROM kn_documents").fetchone()[0]
        sources = dict(self.db.execute("SELECT source, COUNT(*) FROM kn_documents GROUP BY source").fetchall())
        return KnowledgeStats(self.count(), passages, self.recalled, self.researched, sources)

    def search(self, query: str, k: int = 8, doc_id: int | None = None) -> list[Hit]:
        """Best passages for a question, most confident first."""
        words = content_words(query)
        if not words:
            return []
        if self.fts:
            match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
            sql = ("SELECT doc_id, section, text, bm25(kn_passages, 4.0, 1.5, 1.0) AS rank FROM kn_passages "
                   "WHERE kn_passages MATCH ?" + (" AND doc_id = ?" if doc_id else "") + " ORDER BY rank LIMIT ?")
            args = (match, doc_id, k * 4) if doc_id else (match, k * 4)
            try:
                rows = self.db.execute(sql, args).fetchall()
            except sqlite3.OperationalError:
                return []
        else:
            sql = "SELECT doc_id, section, text, 0 FROM kn_passages_plain" + (" WHERE doc_id = ?" if doc_id else "")
            rows = self.db.execute(sql, (doc_id,) if doc_id else ()).fetchall()
        docs: dict[int, Document | None] = {}
        hits: list[Hit] = []
        for d_id, section, text, rank in rows:
            if d_id not in docs:
                docs[d_id] = self.get_by_id(d_id)
            doc = docs[d_id]
            if doc is None:
                continue
            # how fully the question names the article: by its title or a question that led to it
            named = max((coverage(content_words(n), set(words)) for n in doc.names if content_words(n)), default=0.0)
            passage_words = set(tokens(f"{doc.title} {section} {text}"))
            hit = Hit(doc, section, text, coverage(words, passage_words), named, float(rank))
            if hit.passage_coverage > 0:
                hits.append(hit)
        hits.sort(key=lambda h: (-h.confidence, h.rank))
        return hits[:k]

    def recall(self, query: str, threshold: float) -> tuple[list[Hit], float]:
        """Passages of the best-matching document and its confidence ([] if nothing matches)."""
        hits = self.search(query)
        if not hits:
            return [], 0.0
        best = hits[0]
        same_doc = [h for h in hits if h.doc.id == best.doc.id]
        return same_doc, best.confidence


def extract_answer(question: str, hits: list[Hit], max_chars: int = 480) -> str:
    """An answer built from retained sentences (used when no LLM teacher can write one).
    General questions ("who was X") get the article summary; specific ones get the
    sentences that best cover what was asked."""
    doc = hits[0].doc
    asked = content_words(question)
    title_words = set(content_words(doc.title))
    specific = [w for w in asked if not any(_same(w, t) for t in title_words)]
    if not specific:
        return _trim(doc.summary, max_chars)
    scored = []
    for i, hit in enumerate(hits[:5]):
        for j, sentence in enumerate(_SENT.split(" ".join(hit.text.split()))):
            score = coverage(specific, set(tokens(sentence)))
            if score > 0:
                scored.append((score - 0.02 * i - 0.001 * j, i, j, sentence))
    if not scored:
        return _trim(doc.summary, max_chars)
    best = sorted(scored, reverse=True)[:2]
    best.sort(key=lambda s: (s[1], s[2]))
    return _trim(" ".join(s[3] for s in best), max_chars)


def _trim(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"
