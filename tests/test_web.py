"""The web interface: its files, the endpoints it is built on, and token protection."""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from neuromesh import web
from neuromesh.runtime import System
from neuromesh.server import serve_http

from test_system import FAST, make_root

WEB = {**FAST, "memory": {"enabled": True, "interval_seconds": 0.02, "sync_seconds": 0.05}}


async def http(port: int, method: str, path: str, body=None, token: str | None = None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(data)}\r\n"
    if token:
        head += f"Authorization: Bearer {token}\r\n"
    writer.write(head.encode() + b"\r\n" + data)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    header, _, payload = raw.partition(b"\r\n\r\n")
    lines = header.decode().split("\r\n")
    code = int(lines[0].split()[1])
    headers = {k.lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
    return code, headers, payload


class StaticFileTests(unittest.TestCase):
    def test_serves_the_page_and_nothing_else(self):
        page, ctype = web.static_file("/")
        self.assertIn(b"<title>NeuroMesh</title>", page)
        self.assertTrue(ctype.startswith("text/html"))
        for name, kind in (("app.js", "text/javascript"), ("app.css", "text/css"), ("mark.svg", "image/svg+xml")):
            self.assertTrue(web.static_file(f"/ui/{name}")[1].startswith(kind))
        for bad in ("/ui/../server.py", "/ui/..%2fserver.py", "/ui/.hidden", "/ui/web.py", "/status", "/ui/", "/ui/a/b.js"):
            if bad == "/ui/":
                self.assertIsNotNone(web.static_file(bad))  # the page
                continue
            self.assertIsNone(web.static_file(bad), bad)

    def test_the_mind_view_is_divided_into_tabs(self):
        import re
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        tabs = re.findall(r'id="mind-tab-(\w+)" data-mind="(\w+)" aria-controls="mind"', page)
        self.assertTrue(tabs and all(a == b for a, b in tabs), tabs)
        listed = re.search(r"const MIND_TABS = \[([^\]]+)\]", script).group(1)
        self.assertEqual([t for t, _ in tabs], re.findall(r'"(\w+)"', listed))  # every tab has a panel, in order
        for t, _ in tabs:
            self.assertRegex(script, rf"\n\s+{t}: \(\) =>", t)
        self.assertIn('role="tabpanel"', page)

    def test_the_code_and_images_views_and_the_tool_switch(self):
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        for view in ("code", "images"):
            self.assertIn(f'id="view-{view}"', page)
            self.assertIn(f'data-view="{view}"', page)
        self.assertIn('const VIEWS = ["live", "nodes", "memory", "mind", "code", "images", "logs", "status"]', script)
        for part in ('api("/tools/switch"', 'get("code", "/code")', 'api("/code", { write:', 'action: "approve"',
                     'get("images", "/images")', 'api("/images", { name: f.name, data', 'api("/images/correct"',
                     'get("lessons", "/lessons")', "Approve and run", "Let it decide", "/images/file?sha="):
            self.assertIn(part, script)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_script_parses(self):
        out = subprocess.run(["node", "--check", str(web.WEB_DIR / "app.js")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


class WebServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides=WEB, seed=2)
        await self.system.start()
        self.server = None

    async def asyncTearDown(self):
        os.environ.pop("NEUROMESH_TOKEN", None)
        if self.server:
            self.server.close()
        await self.system.stop()
        self.tmp.cleanup()

    async def serve(self) -> int:
        self.server = await serve_http(self.system, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def test_dashboard_log_and_commands(self):
        s = self.system
        port = await self.serve()
        code, headers, page = await http(port, "GET", "/")
        self.assertEqual(code, 200)
        self.assertIn(b'src="/ui/app.js"', page)
        self.assertEqual(headers["cache-control"], "no-store")
        await s.submit("what is 12 * 7")
        await s.submit("sentiment: the hotel was lovely")
        s.memory.learn("What is the capital of France?", "Paris is the capital of France.", "a test", "me")
        code, _, raw = await http(port, "GET", "/dashboard?after=0")
        d = json.loads(raw)
        self.assertEqual(code, 200)
        self.assertEqual(set(d), {"now", "status", "mesh", "work", "log", "orders"})
        routes = {r["route"]: r for r in d["mesh"]["routes"]}
        self.assertEqual(routes["tool:calculator"]["tool"]["state"], "active")
        self.assertIn("skill", routes["skill:sentiment"])
        self.assertIn("answer:direct", routes)
        self.assertEqual(d["mesh"]["memory"][0]["id"], "mem-0001")
        self.assertTrue(any(o["route"] == "tool:calculator" for o in d["orders"]))
        seq = d["log"]["seq"]
        self.assertTrue(d["log"]["entries"] and d["log"]["entries"][-1]["seq"] == seq)
        # only what is new travels
        code, _, raw = await http(port, "GET", f"/log?after={seq}")
        self.assertTrue(all(e["seq"] > seq for e in json.loads(raw)["entries"]))
        await s.submit("count the words in: a b c")
        code, _, raw = await http(port, "GET", f"/log?after={seq}")
        fresh = json.loads(raw)["entries"]
        self.assertTrue(fresh and any(e["tag"] == "order" for e in fresh))
        code, _, raw = await http(port, "GET", f"/log?after={seq}&routine=0")
        self.assertFalse(any(e["routine"] for e in json.loads(raw)["entries"]))
        # the command bar runs console commands; /quit cannot stop the daemon from a browser
        code, _, raw = await http(port, "POST", "/command", {"line": "/tools"})
        self.assertIn("calculator", "\n".join(json.loads(raw)["lines"]))
        code, _, raw = await http(port, "POST", "/command", {"line": "/quit"})
        self.assertIn("only works in the console", json.loads(raw)["lines"][0])
        self.assertFalse(s.stopping)
        # a memory node's detail
        for _ in range(100):
            if not s.memory.nodes["mem-0001"].dirty:
                break
            await asyncio.sleep(0.05)
        code, _, raw = await http(port, "GET", "/memory?node=mem-0001")
        node = json.loads(raw)
        self.assertEqual(node["facts"][0]["variants"], ["capital of France"])
        self.assertTrue(node["unknowns"])
        code, _, _ = await http(port, "GET", "/memory?node=nope")
        self.assertEqual(code, 404)

    async def test_the_token_protects_the_data_not_the_page(self):
        os.environ["NEUROMESH_TOKEN"] = "s3cret"
        port = await self.serve()
        self.assertEqual((await http(port, "GET", "/"))[0], 200)
        self.assertEqual((await http(port, "GET", "/ui/app.js"))[0], 200)
        self.assertEqual((await http(port, "GET", "/dashboard"))[0], 401)
        self.assertEqual((await http(port, "POST", "/command", {"line": "/status"}))[0], 401)
        self.assertEqual((await http(port, "GET", "/dashboard", token="wrong"))[0], 401)
        self.assertEqual((await http(port, "GET", "/dashboard", token="s3cret"))[0], 200)


if __name__ == "__main__":
    unittest.main()
