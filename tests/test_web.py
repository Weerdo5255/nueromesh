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

from test_system import FAST, make_root, wait_for

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
        self.assertIn('const VIEWS = ["live", "network", "nodes", "memory", "mind", "code", "images", "logs", "status"]', script)
        for part in ('api("/tools/switch"', 'get("code", "/code")', 'api("/code", { write:', 'action: "approve"',
                     'get("images", "/images")', 'api("/images", { name: f.name, data', 'api("/images/correct"',
                     'get("lessons", "/lessons")', "Approve and run", "Let it decide", "/images/file?sha="):
            self.assertIn(part, script)

    def test_code_switches_tools_and_web_answers(self):
        script = (web.WEB_DIR / "app.js").read_text()
        for part in ('api("/code", { [which]: m, by: "web" })', 'codeSwitch("auto_run"', 'codeSwitch("make_tools"',
                     'api("/code", { id: item.id, action: "tool", by: "web" })', '"Make it a tool"',
                     'api("/websearch", { inquiry: q.id, by: "web" })', '"Search the web now"', "md?.web_answers",
                     '"Correct it"', "function webPanel(w)", 'get("websearch", "/websearch")', '{ after_minutes: Number(mins.value) }',
                     '"Save and test"', 'api("/tools/retest", { tool: name, by: "web" })', '"Run code automatically"',
                     'h("h2", {}, "Running its code"), codeSwitches(sp)', 'st.scratch?.auto_run',
                     'api("/objectives", { id: o.id, action: "retry", by: "web" })', '"objectives:archive"', "ob.always"):
            self.assertIn(part, script)

    def test_rewards_and_the_project(self):
        script = (web.WEB_DIR / "app.js").read_text()
        for part in ('api("/reward", { target, value: v', 'get("rewards", "/rewards")', 'get("projects", "/projects")',
                     'rewardButtons(`log:${e.seq}`', "rewardButtons(orderId", "rewardButtons(ch.id", "rewardButtons(c.id",
                     'api("/projects", { change: c.id, action: "approve"', '"Approve and open a pull request"', 'a.kind === "project-change"',
                     'api("/projects", { focus:', 'api("/projects", { objective:', 'kind: "direction"'):
            self.assertIn(part, script)
        for name in ("Reward it", "Punish it"):  # every reward button says what it does
            self.assertIn(f'"aria-label": "{name}"', script)

    def test_the_whole_mesh_and_the_network_page(self):
        page = (web.WEB_DIR / "index.html").read_text()
        script = (web.WEB_DIR / "app.js").read_text()
        for part in ('id="view-network"', 'data-view="network"', 'id="whole"', 'id="whole-svg"', 'href="#network"'):
            self.assertIn(part, page)
        for part in ('get("net", "/network")', 'get("netFull", "/network?detail=1")', "function drawNet(", "function renderWhole(",
                     "function renderNetwork(", "function circuitSvg(", "Not grown yet", "Working now", "function workGroups("):
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

    async def test_the_network_endpoint(self):
        from neuromesh import network as net
        s = self.system
        port = await self.serve()
        for q in ["what is 12 * 7", "the hotel was lovely", "urgency: the server is down asap"]:
            await s.submit(q)
        self.assertTrue(await wait_for(lambda: s.registry.champion("dispatch") is not None, timeout=60))
        code, _, body = await http(port, "GET", "/network")
        self.assertEqual(code, 200)
        light = json.loads(body)
        self.assertFalse(light["detail"])
        fams = {f["id"]: f for f in light["families"]}
        self.assertEqual(fams["dispatch"]["kind"], "router")
        for f in light["families"]:  # every family has its place in the way an order travels
            self.assertIn(f["station"], light["stations"], f["id"])
            self.assertEqual(f["params"], sum(n["params"] for n in f["nodes"] if n["status"] != "retired"))
            for n in f["nodes"]:
                self.assertNotIn("layers", n)
        self.assertEqual(light["nodes"], sum(len(f["nodes"]) for f in light["families"]))
        self.assertIn("calculator", {t["id"] for t in light["tools"]})
        router = fams["dispatch"]["nodes"]
        self.assertTrue(router and all(n["sizes"][0] > 0 and n["params"] > 0 for n in router))
        # with detail: the strongest neurons of every layer and the strongest synapses between them
        code, _, body = await http(port, "GET", "/network?detail=1")
        full = json.loads(body)
        node = next(f for f in full["families"] if f["id"] == "dispatch")["nodes"][0]
        layers = node["layers"]
        self.assertEqual([l["size"] for l in layers], node["sizes"])
        self.assertEqual((layers[0]["kind"], layers[-1]["kind"]), ("input", "output"))
        self.assertEqual(len(layers[-1]["labels"]), len(layers[-1]["shown"]))
        for a, b in zip(layers, layers[1:]):
            self.assertLessEqual(len(a["shown"]), max(16, 2 * 16))
            self.assertTrue(a["synapses"])
            for r, c, w in a["synapses"]:  # indices into what is shown, strongest first
                self.assertLess(r, len(a["shown"]))
                self.assertLess(c, len(b["shown"]))
            ws = [abs(w) for _, _, w in a["synapses"]]
            self.assertEqual(ws, sorted(ws, reverse=True))
        # what the background is doing, to which node
        nid = node["id"]
        with s.activity.track("training", "trainer", f"training: {nid} slice 1 on 9 examples"):
            busy = net.network(s)
        mine = next(n for f in busy["families"] for n in f["nodes"] if n["id"] == nid)
        self.assertTrue(any(b["actor"] == "trainer" and b["now"] for b in mine["busy"]))
        self.assertTrue(any(w["actor"] == "trainer" and nid in w["nodes"] for w in busy["work"]))

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
