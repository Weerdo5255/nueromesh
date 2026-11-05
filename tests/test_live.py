"""Live view: panel builders, the activity log, and the curses view itself driven through a
pseudo-terminal (POSIX only)."""
import asyncio
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

from neuromesh.activity import ActivityLog, ActivityTracker, format_event
from neuromesh.bus import Bus, Event
from neuromesh.live import ASCII, UNICODE, network_lines, teacher_goal_lines, work_lines, wrap_log

ROOT = Path(__file__).resolve().parent.parent


class ActivityTests(unittest.TestCase):
    def test_tracker_shows_work_in_flight_then_recent(self):
        t = ActivityTracker()
        with t.track("lesson", "ollama", "labelling sentiment: great food"):
            self.assertEqual([a.actor for a in t.current()], ["ollama"])
        self.assertEqual(t.current(), [])
        self.assertEqual(t.recent()[0].text, "labelling sentiment: great food")

    def test_events_become_log_lines_and_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            log = ActivityLog(Path(d) / "activity.log")
            bus = Bus()
            bus.listen(log.on_event)
            bus.publish("order.done", order="o1", route="tool:calculator", ok=True, served_by="nodes",
                        latency_ms=12.0, text="what is 6 * 7", output="42", note="")
            bus.publish("lesson.label", teacher="ollama", capability="sentiment", text="lovely", label="positive",
                        seconds=1.2)
            bus.publish("node.trained", node="n1", capability="c", status="candidate", loss=0.1, steps=3, examples=40)
            bus.publish("audit", actor="guardian", kind="deny.tool", subject="shell_exec",
                        detail={"reason": "forbidden permissions"})
            bus.publish("audit", actor="system", kind="system.start", subject="x", detail={})  # not logged
            bus.publish("node.promoted", node="n2", capability="c", replaced="n1", reason="better", evidence={})
            tags = [e.tag for e in log.tail(10)]
            self.assertEqual(tags, ["order", "lesson", "train", "guard", "promote"])
            self.assertEqual([e.tag for e in log.tail(10, routine=False)], ["order", "guard", "promote"])
            written = (Path(d) / "activity.log").read_text().splitlines()
            self.assertEqual(len(written), 5)
            self.assertIn("[lesson] ollama: sentiment('lovely') = positive", written[1])

    def test_log_wraps_under_the_text_column(self):
        entry = format_event(Event(time.time(), "lesson.label", {
            "teacher": "ollama", "capability": "sentiment", "text": "a " * 40, "label": "neutral", "seconds": 0.5}))
        rows = wrap_log([entry], 40, UNICODE)
        self.assertGreater(len(rows), 1)
        self.assertTrue(rows[1][0][0].isspace() and len(rows[1][0][0]) == 18)


class PanelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from neuromesh.runtime import System
        from test_system import FAST, make_root, orders, wait_for
        self.tmp = tempfile.TemporaryDirectory()
        self.system = System(make_root(Path(self.tmp.name)), overrides=FAST, seed=1, device="cpu")
        await self.system.start()
        for text in orders(60):
            await self.system.submit(text)
        await wait_for(lambda: all(self.system.registry.champion(c) for c in ("dispatch", "sentiment")))

    async def asyncTearDown(self):
        await self.system.stop()
        self.tmp.cleanup()

    async def test_network_shows_router_routes_and_node_families(self):
        s = self.system
        champ = s.registry.champion("sentiment")
        s.registry.add(champ.spawn_wider(1.5), "test challenger")
        text = "\n".join(line for line, _ in network_lines(s, UNICODE))
        self.assertIn("router", text)
        self.assertIn(s.registry.champion("dispatch").id, text)
        for route in ("skill:sentiment", "skill:urgency", "tool:calculator", "tool:word_stats", "answer:direct"):
            self.assertIn(route, text)
        self.assertIn(f"challenger {s.registry.candidates('sentiment')[0].id}", text)
        self.assertIn("calculator v1.0.0 [active]", text)
        ascii_text = "\n".join(line for line, _ in network_lines(s, ASCII))
        self.assertTrue(all(ord(ch) < 128 for ch in ascii_text), "ASCII mode must not emit Unicode")

    async def test_work_panel_and_teachers(self):
        s = self.system
        with s.activity.track("lesson", "ollama", "labelling sentiment: the soup was cold"):
            lines = [line for line, _ in work_lines(s, UNICODE)]
        self.assertTrue(any("ollama" in line and "the soup was cold" in line for line in lines))
        self.assertIn("lessons given so far:", lines[-1])
        tg = [line for line, _ in teacher_goal_lines(s, UNICODE)]
        self.assertTrue(tg[0].startswith("teachers:") and "heuristic" in tg[0])
        self.assertTrue(tg[1].startswith("goals:") and "G1:" in tg[1])


@unittest.skipUnless(os.name == "posix", "pseudo-terminals are POSIX-only")
class TerminalTests(unittest.TestCase):
    """Runs `neuromesh run` in a pseudo-terminal, types an order and a command, quits."""

    def test_live_view_in_a_terminal(self):
        import fcntl
        import pty
        import struct
        import termios
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            for name in ("directives.toml", "capabilities.toml"):
                shutil.copy(ROOT / name, root / name)
            shutil.copytree(ROOT / "plugins", root / "plugins")
            snap = Path(tmp) / "screen.txt"
            pid, fd = pty.fork()
            if pid == 0:
                path = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
                env = dict(os.environ, TERM="xterm-256color", LANG="C.UTF-8", LC_ALL="C.UTF-8",
                           PYTHONPATH=path, NEUROMESH_LIVE_SNAPSHOT=str(snap))
                os.execvpe(sys.executable, [sys.executable, "-m", "neuromesh", "run", "--root", str(root)], env)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 36, 130, 0, 0))
            output = bytearray()

            def drain():
                while True:
                    try:
                        data = os.read(fd, 65536)
                    except OSError:
                        return
                    if not data:
                        return
                    output.extend(data)
            threading.Thread(target=drain, daemon=True).start()

            def screen_has(text, timeout=30.0):
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if snap.exists() and text in snap.read_text(encoding="utf-8"):
                        return True
                    time.sleep(0.1)
                return False

            try:
                self.assertTrue(screen_has("network:"), "live view never drew")
                self.assertIn("activity log", snap.read_text(encoding="utf-8"))
                os.write(fd, b"what is 6 * 7\r")
                self.assertTrue(screen_has("] 42"), "order result never shown")
                os.write(fd, b"/tools\r")
                self.assertTrue(screen_has("> /tools"), "command output never shown")
                os.write(fd, b"\t/quit\r")
                deadline = time.time() + 20
                while time.time() < deadline:
                    done, status = os.waitpid(pid, os.WNOHANG)
                    if done:
                        break
                    time.sleep(0.1)
                else:
                    os.kill(pid, 9)
                    self.fail("live view did not quit")
                self.assertEqual(os.waitstatus_to_exitcode(status), 0)
                self.assertNotIn(b"Traceback", bytes(output))
                self.assertTrue((root / "state" / "activity.log").exists())
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
