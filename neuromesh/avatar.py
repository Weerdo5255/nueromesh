"""The avatar: an optional face for the mesh in the web interface, with its mind at a glance - and a
voice, spoken by a text-to-speech workflow on your ComfyUI server.

    glance()    everything about its mind that matters right now, in one place: its mood (the five
                temperament numbers), what it is doing, awake or dreaming, the objective it is working
                on, its latest thought and what it wants to find out, what it lacks, what waits for
                you (questions, approvals, tools it wrote), its initiative, and how much it holds -
                with the face it should make, and the same summary as a few spoken sentences in the
                first person. Built only from its real state: the avatar never says what is not so.
    ComfyTTS    speaks a text with your own ComfyUI workflow, exported in API format ("Save (API
                Format)"). The text goes where the workflow says {{text}} (or into the first text
                input of a text-to-speech node), the workflow is queued with POST /prompt, its result
                read from /history/<id>, and the audio fetched from /view. Audio is cached by text.
                Only the server set in [avatar] comfyui_url is ever contacted, as client "neuromesh".

The avatar's look is drawn by the web interface (an original character, not anyone's); its
expression follows its mood: [avatar] in directives.toml turns it on and sets up the voice.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
               ".opus": "audio/ogg", ".m4a": "audio/mp4", ".webm": "audio/webm"}
TEXT_INPUTS = ("text", "tts_text", "input_text", "prompt", "sentence", "speech", "gen_text", "content")


class SpeechError(Exception):
    """ComfyUI could not speak the text (not set up, unreachable, or the workflow failed)."""


# ------------------------------------------------------------------------- the voice
class ComfyTTS:
    """Text to speech with a ComfyUI workflow (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        self.pol = system.directives.avatar
        self.dir = system.state_dir / "avatar"
        self.spoken = 0
        self.failures = 0
        self.last_error = ""
        self._inflight: dict[str, asyncio.Future] = {}

    def workflow_path(self) -> Path | None:
        raw = str(self.pol.tts_workflow or "").strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        return p if p.is_absolute() else self.sys.root / p

    def configured(self) -> tuple[bool, str]:
        if not self.pol.enabled:
            return False, "the avatar is off ([avatar] enabled = false)"
        if not str(self.pol.comfyui_url or "").strip():
            return False, "no ComfyUI server set ([avatar] comfyui_url)"
        wf = self.workflow_path()
        if wf is None:
            return False, "no text-to-speech workflow set ([avatar] tts_workflow)"
        if not wf.is_file():
            return False, f"the workflow file {wf} does not exist (export yours from ComfyUI: Save (API Format))"
        return True, "ready"

    def status(self) -> dict:
        ok, why = self.configured()
        return {"ready": ok, "why": why, "server": self.pol.comfyui_url if ok else "", "spoken": self.spoken,
                "failures": self.failures, "last_error": self.last_error, "max_chars": int(self.pol.max_chars)}

    # ----------------------------------------------------------- the workflow
    def load(self) -> dict:
        wf = self.workflow_path()
        try:
            data = json.loads(wf.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SpeechError(f"could not read the workflow {wf}: {exc}") from None
        if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
            data = data["prompt"]
        if not isinstance(data, dict) or not data or not all(isinstance(v, dict) and "class_type" in v for v in data.values()):
            raise SpeechError("the workflow is not in ComfyUI's API format (use Save (API Format), not Save)")
        return data

    def fill(self, workflow: dict, text: str) -> tuple[dict, str]:
        """The workflow with the text in it: everywhere it says the placeholder, or else in the first text
        input of a node that looks like text to speech. Returns (workflow, where the text went)."""
        wf = copy.deepcopy(workflow)
        mark = str(self.pol.text_placeholder or "{{text}}")
        where = []
        for nid, node in wf.items():
            for key, val in (node.get("inputs") or {}).items():
                if isinstance(val, str) and mark in val:
                    node["inputs"][key] = val.replace(mark, text)
                    where.append(f"{node.get('class_type', '?')}.{key}")
        if where:
            return wf, ", ".join(where)
        ranked = sorted(wf.items(), key=lambda kv: (not re.search(r"tts|speech|voice|speak|kokoro|f5|chatterbox|bark|zonos",
                                                                  str(kv[1].get("class_type", "")), re.I), kv[0]))
        for nid, node in ranked:
            for key in TEXT_INPUTS:
                val = (node.get("inputs") or {}).get(key)
                if isinstance(val, str):
                    node["inputs"][key] = text
                    return wf, f"{node.get('class_type', '?')}.{key} (no {mark} in the workflow)"
        raise SpeechError(f"the workflow has no {mark} and no text input to put the words in")

    # --------------------------------------------------------------- talking
    def _open(self, url: str, data: bytes | None = None, timeout: float = 30.0) -> bytes:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                              "User-Agent": "neuromesh"} if data else {"User-Agent": "neuromesh"})
        opener = urllib.request.build_opener() if self.pol.use_proxy else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise SpeechError(f"ComfyUI said HTTP {exc.code} for {urllib.parse.urlsplit(url).path}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SpeechError(f"ComfyUI is unreachable at {self.pol.comfyui_url} ({getattr(exc, 'reason', exc)})") from None

    def _audio_of(self, outputs: dict) -> dict | None:
        """The audio a finished workflow made: an "audio" output (Save Audio), or any output file with an audio extension."""
        found = []
        for nid, out in (outputs or {}).items():
            for key, items in (out or {}).items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, dict) and it.get("filename") and Path(it["filename"]).suffix.lower() in AUDIO_TYPES:
                        found.append((key != "audio", it.get("type") != "output", it))
        found.sort(key=lambda f: (f[0], f[1]))
        return found[0][2] if found else None

    def _speak_sync(self, text: str) -> tuple[bytes, str, str]:
        base = str(self.pol.comfyui_url).rstrip("/")
        wf, where = self.fill(self.load(), text)
        pid = str(uuid.uuid4())
        body = json.dumps({"prompt": wf, "client_id": "neuromesh", "prompt_id": pid}).encode()
        reply = self._open(f"{base}/prompt", body, timeout=30)
        try:
            got = json.loads(reply)
        except ValueError:
            got = {}
        if got.get("error") or got.get("node_errors"):
            raise SpeechError(f"ComfyUI refused the workflow: {got.get('error') or got.get('node_errors')}")
        pid = got.get("prompt_id") or pid
        deadline = time.time() + float(self.pol.timeout_seconds)
        delay = 0.3
        while time.time() < deadline:
            hist = json.loads(self._open(f"{base}/history/{urllib.parse.quote(pid)}", timeout=15) or b"{}")
            entry = hist.get(pid) if isinstance(hist, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    why = msgs[-1][1].get("exception_message", "") if msgs and isinstance(msgs[-1][1], dict) else "it failed"
                    raise SpeechError(f"the workflow failed in ComfyUI: {str(why)[:200]}")
                audio = self._audio_of(entry.get("outputs") or {})
                if audio is not None:
                    q = urllib.parse.urlencode({"filename": audio["filename"], "subfolder": audio.get("subfolder", ""),
                                                "type": audio.get("type", "output")})
                    data = self._open(f"{base}/view?{q}", timeout=60)
                    if not data:
                        raise SpeechError("ComfyUI returned an empty audio file")
                    return data, Path(audio["filename"]).suffix.lower(), where
                if status.get("completed"):
                    raise SpeechError("the workflow finished but made no audio (does it end in a Save Audio node?)")
            time.sleep(delay)
            delay = min(1.5, delay * 1.5)
        raise SpeechError(f"ComfyUI did not finish within {self.pol.timeout_seconds:g}s")

    def clean(self, text: str) -> str:
        """What is worth saying aloud: no markdown, links or long numbers of digits, within max_chars."""
        t = re.sub(r"https?://\S+", "", str(text))
        t = re.sub(r"[`*_#>\[\]{}|]", "", t)
        t = re.sub(r"\(/[^)]*\)", "", t)
        t = " ".join(t.split())
        limit = int(self.pol.max_chars)
        if len(t) > limit:
            cut = t[:limit]
            t = cut[: max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))+1] or cut.rsplit(" ", 1)[0]
        return t.strip()

    async def speak(self, text: str, by: str = "web") -> dict:
        """Speak a text: {"file", "type", "cached", "text", "where"}. Raises SpeechError."""
        ok, why = self.configured()
        if not ok:
            raise SpeechError(why)
        said = self.clean(text)
        if not said:
            raise SpeechError("nothing to say")
        wf = self.workflow_path()
        key = hashlib.sha1(f"{said}|{wf.stat().st_mtime_ns}|{self.pol.comfyui_url}".encode()).hexdigest()[:20]
        self.dir.mkdir(parents=True, exist_ok=True)
        have = next((p for p in self.dir.glob(f"{key}.*") if p.suffix in AUDIO_TYPES), None)
        if have is not None:
            have.touch()
            return {"file": have.name, "type": AUDIO_TYPES[have.suffix], "cached": True, "text": said, "where": ""}
        if key in self._inflight:  # the same words are being spoken already: wait for them
            return await asyncio.shield(self._inflight[key])
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            with self.sys.activity.track("speech", "avatar", f"speaking with ComfyUI: {said[:60]}"):
                data, ext, where = await asyncio.to_thread(self._speak_sync, said)
            ext = ext if ext in AUDIO_TYPES else ".wav"
            path = self.dir / f"{key}{ext}"
            path.write_bytes(data)
            self._prune()
            self.spoken += 1
            self.last_error = ""
            self.sys.guardian.audit(by, "avatar.spoke", path.name, chars=len(said), where=where)
            out = {"file": path.name, "type": AUDIO_TYPES[ext], "cached": False, "text": said, "where": where}
            fut.set_result(out)
            return out
        except SpeechError as exc:
            self.failures += 1
            self.last_error = str(exc)
            fut.set_exception(exc)
            fut.exception()  # retrieved: no "never retrieved" warning when nobody else waits
            raise
        finally:
            self._inflight.pop(key, None)

    def _prune(self) -> None:
        files = sorted((p for p in self.dir.iterdir() if p.suffix in AUDIO_TYPES), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - int(self.pol.cache_files))]:
            try:
                p.unlink()
            except OSError:
                pass

    def audio(self, name: str) -> tuple[bytes, str] | None:
        """A spoken file, by name (only files it made; nothing outside state/avatar)."""
        if not re.fullmatch(r"[0-9a-f]{20}\.[a-z0-9]{2,5}", name or ""):
            return None
        p = self.dir / name
        if p.suffix not in AUDIO_TYPES or not p.is_file():
            return None
        return p.read_bytes(), AUDIO_TYPES[p.suffix]


# ------------------------------------------------------------------------ the glance
def _cut(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def expression(s, doing: list[dict]) -> tuple[str, str]:
    """The face it should make, and why: from its mood, whether it sleeps, and what it is doing."""
    if s.guardian.halted:
        return "still", "it is halted"
    if s.dreams.asleep:
        return "asleep", "it is asleep, dreaming"
    v = s.affect.state.values
    mood = s.affect.state.mood()
    kinds = {d["kind"] for d in doing}
    if mood == "wary":
        return "wary", "its trust is low"
    if mood == "tense":
        return "frustrated", "things have gone badly lately"
    if mood == "subdued":
        return "sad", "its spirits are low"
    if v["patience"] < 0.25:
        return "impatient", "its patience is short"
    if kinds & {"thinking", "research", "synthesis", "speech"}:
        return "thinking", "it is thinking something over" if "thinking" in kinds else "it is looking something up"
    if mood == "upbeat":
        return "happy", "it is in good spirits"
    if mood == "alert":
        return "curious", "it is alert"
    if v["confidence"] < 0.35:
        return "unsure", "its confidence is low"
    if mood == "content":
        return "content", "it is content"
    return "calm", "it is steady"


def glance(s) -> dict:
    """Its whole mind at a glance (see the module docstring)."""
    snap = s.affect.state.snapshot()  # with the temperament off its mood rests at the baseline
    doing = [{"kind": a.kind, "actor": a.actor, "text": _cut(a.text, 90), "for": round(time.time() - a.started, 1)}
             for a in s.activity.current() if a.kind not in ("order",)][:4]
    face, face_why = expression(s, doing)
    book = s.objectives
    active = [o for o in book.items.values() if o.status == "active"]
    focus = None
    if active:
        o = sorted(active, key=lambda o: (o.self_set, o.created_at))[0]  # yours first
        done, total = o.progress()
        focus = {"id": o.id, "text": o.text, "done": done, "total": total, "self_set": o.self_set, "why": o.why}
    thinking = s.thinking
    thought = None
    if thinking.enabled and thinking.thoughts:
        th = max(thinking.thoughts.values(), key=lambda t: t.updated)
        thought = {"id": th.id, "topic": th.topic, "text": _cut(th.understanding, 220), "node": th.node}
    desires = thinking.desires() if thinking.enabled else []
    want = {"want": desires[0]["want"], "status": desires[0].get("status", "open")} if desires else None
    cur = s.curiosity
    gaps = sum(len(a.gaps) for a in cur.areas().values()) if cur.enabled else 0
    aims = cur.aims(1) if cur.enabled else []
    inquiries = s.mind.open_inquiries() if s.mind.enabled else []
    approvals = s.guardian.pending_approvals()
    drafts = [d for d in s.toolsmith.drafts.values() if d.get("status") == "awaiting approval"] if s.toolsmith.enabled else []
    d = s.director
    last = d.history[-1] if d.history else None
    facts = sum(len(n.facts()) for n in s.memory.nodes.values()) if s.memory.enabled else 0
    g = {
        "mood": snap.get("mood", "steady"), "temperament": bool(s.affect.enabled),
        "feelings": {k: snap[k] for k in ("valence", "arousal", "confidence", "trust", "patience") if k in snap},
        "face": face, "face_why": face_why,
        "state": "halted" if s.guardian.halted else "asleep" if s.dreams.asleep else
                 "paused" if s.guardian.paused else "working" if doing else "idle",
        "doing": doing,
        "focus": focus,
        "thought": thought,
        "want": want,
        "lacks": {"gaps": gaps, "aim": aims[0]["text"] if aims else ""},
        "waiting": {"questions": len(inquiries), "approvals": len(approvals), "tools": len(drafts),
                    "first": inquiries[0].question if inquiries else ""},
        "initiative": {"on": d.enabled, "mode": d.pinned, "last": (last or {}).get("summary", "") if last else "",
                       "last_mode": (last or {}).get("mode", "") if last else ""},
        "holds": {"facts": facts, "thoughts": len(thinking.thoughts) if thinking.enabled else 0,
                  "articles": s.knowledge.count(),
                  "nodes": sum(1 for c in s.capabilities if s.registry.champion(c) is not None)},
    }
    g["line"] = spoken_line(g)
    return g


MOOD_WORDS = {"upbeat": "I'm in good spirits", "content": "I'm content", "steady": "I'm steady",
              "alert": "I'm alert", "subdued": "I'm feeling a little low", "tense": "I'm a bit tense",
              "wary": "I'm feeling wary"}


def spoken_line(g: dict) -> str:
    """The glance as a few sentences in the first person: what it would say if you asked how it is."""
    if g["state"] == "halted":
        return "I'm halted. I won't do anything until you resume me."
    if g["state"] == "asleep":
        out = ["I'm asleep, dreaming about what I've learned. An order will wake me."]
    else:
        out = [MOOD_WORDS.get(g["mood"], "I'm steady") + "."]
        if g["doing"]:
            out.append(f"Right now I'm {_doing_words(g['doing'][0])}.")
        elif g["state"] == "paused":
            out.append("My background work is paused.")
    f = g["focus"]
    if f:
        out.append(f"I'm working on {'an objective I set myself' if f['self_set'] else 'your objective'}: "
                   f"{_cut(f['text'], 90)}, {f['done']} of {f['total']} steps done.")
    if g["thought"]:
        out.append(f"I've been thinking about {g['thought']['topic']}.")
    if g["want"] and g["want"]["status"] == "open":
        out.append(g["want"]["want"] + ".")
    w = g["waiting"]
    asks = []
    if w["questions"]:
        asks.append(f"{w['questions']} question{'s' if w['questions'] != 1 else ''} for you")
    if w["approvals"]:
        asks.append(f"{w['approvals']} thing{'s' if w['approvals'] != 1 else ''} waiting for your approval")
    if w["tools"]:
        asks.append(f"{w['tools']} tool{'s' if w['tools'] != 1 else ''} I wrote for you to check")
    if asks:
        out.append("I have " + (", ".join(asks[:-1]) + " and " + asks[-1] if len(asks) > 1 else asks[0]) + ".")
    return " ".join(out)


def _doing_words(a: dict) -> str:
    k, t = a["kind"], a["text"]
    if k == "research":
        return f"looking up {_cut(t.split(':', 1)[-1].strip(), 60)}"
    if k == "thinking":
        return "putting what I know into my own words"
    if k == "training":
        return "training one of my nodes"
    if k == "speech":
        return "finding my voice"
    if k == "evaluation":
        return "checking how well my nodes do"
    if k in ("lesson", "synthesis"):
        return "learning from a teacher"
    return f"busy with {_cut(t, 60)}"
