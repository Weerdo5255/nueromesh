"""Small stand-ins for the servers the mesh talks to in these tests: an Ollama server (with a vision model) whose
replies a test decides, and a Gitea with a few repositories."""
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Server:
    def __init__(self, handler):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.owner = self
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ----------------------------------------------------------------------------- Ollama
class _Ollama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, payload, code=200):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        owner = self.server.owner
        if self.path.startswith("/api/tags"):
            return self._send({"models": [{"name": m, "model": m, "details": {"families": fam}} for m, fam in owner.models]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        owner = self.server.owner
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        msgs = body.get("messages") or []
        system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        user_msg = next((m for m in msgs if m.get("role") == "user"), {})
        call = {"model": body.get("model"), "system": system, "user": user_msg.get("content", ""),
                "images": user_msg.get("images") or [], "format": body.get("format")}
        owner.calls.append(call)
        content = owner.responder(call)
        if content is None:
            return self._send({"error": "no reply"}, 500)
        self._send({"message": {"role": "assistant", "content": content if isinstance(content, str) else json.dumps(content)},
                    "prompt_eval_count": 10, "eval_count": 10})


class MockOllama(_Server):
    """An Ollama server: /api/tags lists models (a chat model and a vision model), /api/chat replies with what
    responder(call) returns (a dict is sent as JSON). call: {"model", "system", "user", "images", "format"}."""

    def __init__(self, responder, models=(("llama3.1:8b", ["llama"]), ("llava:7b", ["llama", "clip"]))):
        self.responder = responder
        self.models = list(models)
        self.calls: list[dict] = []
        super().__init__(_Ollama)


# ----------------------------------------------------------------------------- Gitea
TEXTKIT = {
    "textkit/words.py": '''"""Little helpers for words."""
import re


def slugify(title):
    """Turn a title into a URL slug: lower case, words joined by hyphens."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return "-".join(words)


def reverse_words(sentence):
    """Reverse the words in a sentence."""
    parts = sentence.split()
    return " ".join(reversed(parts))


def shout(text):
    """Make text upper case, with an exclamation mark."""
    return text.upper().strip() + "!"
''',
    "textkit/stats.py": '''import math
import statistics


def mean(numbers):
    """Calculate the average of a list of numbers."""
    total = sum(numbers)
    return total / len(numbers)


def hypotenuse(a, b):
    """The long side of a right triangle."""
    return math.sqrt(a * a + b * b)
''',
    "textkit/config.py": '''import json


def parse_settings(text):
    """Parse key=value lines into a dictionary."""
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_settings(path):
    """Read settings from a JSON file."""
    with open(path) as f:
        return json.load(f)
''',
    "tests/test_words.py": '''from textkit.words import slugify


def test_slugify():
    assert slugify("Hello World") == "hello-world"
    assert slugify("A  b") == "a-b"
''',
    "README.md": "# textkit\n",
}
WEBTOOLS = {
    "web/client.js": '''// Fetch a user record from the API as JSON.
async function fetchUser(id) {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}

// Capitalize the first letter of a word.
function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}
''',
    "cmd/days.go": '''package main

import "time"

// DaysUntil counts the days from now until a date.
func DaysUntil(t time.Time) int {
	return int(time.Until(t).Hours() / 24)
}
''',
}


class _Gitea(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, payload, code=200, raw=False):
        data = payload if raw else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "text/plain" if raw else "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        owner = self.server.owner
        owner.auth.append(self.headers.get("authorization"))
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        p = urllib.parse.unquote(u.path)
        owner.paths.append(p)
        if p == "/api/v1/version":
            return self._send({"version": "1.22.3"})
        if p == "/api/v1/repos/search":
            data = [{"full_name": n, "default_branch": "main", "description": f"{n} things", "empty": False,
                     "archived": False} for n in owner.repos]
            return self._send({"ok": True, "data": data if q.get("page", "1") == "1" else []})
        parts = p.split("/")
        if len(parts) >= 6 and parts[1:4] == ["api", "v1", "repos"]:
            name = f"{parts[4]}/{parts[5]}"
            files = owner.repos.get(name)
            if files is None:
                return self._send({"message": "not found"}, 404)
            rest = parts[6:]
            if rest[:1] == ["branches"]:
                return self._send({"name": "main", "commit": {"id": owner.commit(name)}})
            if rest[:2] == ["git", "trees"]:
                tree = [{"path": f, "type": "blob", "size": len(c), "sha": f"{abs(hash(c)) % 10**12:x}", "mode": "100644"}
                        for f, c in files.items()]
                return self._send({"sha": rest[2], "tree": tree, "truncated": False})
            if rest[:1] == ["raw"]:
                path = "/".join(rest[1:])
                if path in files:
                    return self._send(files[path].encode(), raw=True)
                return self._send({"message": "not found"}, 404)
            if not rest:
                return self._send({"full_name": name, "default_branch": "main", "description": ""})
        self._send({"message": "not found"}, 404)


class MockGitea(_Server):
    def __init__(self):
        self.repos = {"ann/textkit": dict(TEXTKIT), "ann/webtools": dict(WEBTOOLS)}
        self.auth: list = []
        self.paths: list = []
        super().__init__(_Gitea)

    def commit(self, name: str) -> str:
        return f"{abs(hash(json.dumps(self.repos[name], sort_keys=True))) % 10**12:012x}"
