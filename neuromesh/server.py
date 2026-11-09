"""A small JSON-over-HTTP API (stdlib only) so other programs and agents can send orders,
teach the system and approve tools while it runs as a daemon - and the web interface at /,
which is built on the same API (web.py, neuromesh/web/).

Binds to 127.0.0.1 by default. Set NEUROMESH_TOKEN to require "Authorization: Bearer <token>" on
every API call (the page itself holds no data and asks for the token).
"""
from __future__ import annotations

import asyncio
import re
import time
import json
import os
from dataclasses import asdict
from urllib.parse import parse_qs, urlparse

from . import web
from .report import render_lineage

REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed",
           409: "Conflict", 410: "Gone", 500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable"}


class Raw:
    """A response that is not JSON: bytes and their content type (the avatar's spoken audio)."""

    def __init__(self, data: bytes, ctype: str):
        self.data, self.ctype = data, ctype


async def _route(system, method: str, path: str, query: dict, body: dict):
    g = system.guardian
    if method == "GET":
        if path == "/dashboard":  # the web interface's live view, in one request
            return 200, web.dashboard(system, int(query.get("after", 0) or 0))
        if path == "/log":  # the activity log; ?after=<seq> returns only newer entries
            return 200, web.log_since(system, int(query.get("after", 0) or 0), int(query.get("n", 500)),
                                      query.get("routine", "1") != "0")
        if path == "/mesh":
            return 200, web.mesh(system)
        if path == "/status":
            return 200, system.status()
        if path == "/lineage":
            return 200, {"lineage": render_lineage(system.registry.lineage, query.get("capability"))}
        if path == "/events":
            n = int(query.get("n", 50))
            return 200, [{"ts": e.ts, "topic": e.topic, "data": e.data} for e in system.bus.recent("*", n)]
        if path == "/audit":
            return 200, system.store.recent_audit(int(query.get("n", 50)))
        if path == "/questions":
            return 200, list(system.teachers.human.questions)
        if path == "/temperament":
            return 200, system.affect.status()
        if path in ("/opinions", "/notes"):  # what users may see; private notes are only counted
            m = system.mind
            if path == "/opinions":
                notes = [n for n in m.opinions() if m.shareable(n)]
                withheld = len(m.opinions()) - len(notes)
            else:
                notes = [n for n in m.public_notes(50) if n.kind != "opinion"]
                withheld = len(m.private_notes())
            return 200, {"notes": [asdict(n) for n in notes], "withheld": withheld}
        if path == "/mind":  # the operator's full view (P1, P6); audited
            return 200, {**system.mind.status(),
                         "notes": [asdict(n) for n in system.mind.inspect(query.get("by", "api-operator"))]}
        if path == "/tasks":
            return 200, system.tasks.status()
        if path == "/challenges":
            return 200, system.challenges.status()
        if path == "/tools/doctor":
            return 200, system.tooldoctor.status()
        if path == "/conversations":  # conversations, and its short-term memory nodes
            return 200, system.conversations.status()
        if path == "/realm":  # its sense of fact and fiction; ?text= judges one
            st = system.realm.status()
            if query.get("text"):
                st["judged"] = system.realm.judge(query["text"], teach=False).to_dict()
            return 200, st
        if path == "/challenges/catalog":
            return 200, system.challenges.catalog()
        if path in ("/challenges/search", "/challenges/inspect"):
            try:
                if path == "/challenges/search":
                    return 200, {"query": query.get("q", ""),
                                 "datasets": await asyncio.to_thread(system.challenges.search, query.get("q", ""))}
                mapping = json.loads(query["mapping"]) if query.get("mapping") else None
                return 200, await asyncio.to_thread(system.challenges.inspect, query.get("dataset", ""),
                                                    query.get("config", ""), query.get("split", ""), mapping)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                return 400, {"ok": False, "message": str(exc)}
        if path == "/challenges/export":
            try:
                return 200, system.challenges.export(query.get("suite", ""))
            except (ValueError, OSError) as exc:
                return 400, {"ok": False, "message": str(exc)}
        if path == "/evolution":
            return 200, {**system.evolution.status(), "archive": system.evolution.archive}
        if path == "/director":
            return 200, system.director.status()
        if path == "/dreams":
            return 200, system.dreams.status()
        if path == "/ego":
            return 200, system.ego.status()
        if path == "/directors":
            return 200, system.directors.status()
        if path == "/anatomy":  # ?node=<id or skill>&text=<input>: how a node is built, and what fires in it
            from . import anatomy
            ref = query.get("node", "")
            a = anatomy.anatomy(system, ref, query.get("text") or None, latest=bool(query.get("latest")))
            if a is not None:
                return 200, a
            meta = system.registry.lineage.get(ref)
            if meta:  # it was here, and evolution moved on: its weights are gone
                cap = meta.get("capability", "")
                champ = system.registry.champion(cap)
                return 410, {"error": f"{ref} has been {meta.get('status', 'disposed')} since; its weights are gone."
                             + (f" {champ.id} serves {cap} now." if champ else ""), "capability": cap if champ else None}
            return 404, {"error": "no such node (a node id, a memory node id, or a skill)"}
        if path == "/alternatives":  # ?order=<id>: the next most likely answers to an order, worked out now
            alts = await system.corrections.alternatives(query.get("order", ""))
            return (200, alts) if alts is not None else (404, {"error": "unknown order"})
        if path == "/traces":  # the paths recent orders took through the mesh, hop by hop
            from . import anatomy
            return 200, {"traces": anatomy.traces(system, min(50, int(query.get("n", 15))))}
        if path == "/tools/switch":  # the tool switch: on, off, or it decides; per tool; what it decided lately
            return 200, system.tool_switch.status()
        if path == "/lessons":  # orders its teachers set it while it was idle, how it answered, how they marked it
            return 200, system.tutor.status()
        if path == "/rewards":  # people's rewards and punishments, the directions they push, its reward node
            return 200, system.rewards.status()
        if path == "/projects":  # its coding project in Gitea: objectives, proposed changes, pull requests
            return 200, system.projects.status()
        if path == "/projects/change":  # ?id=ch-3: one proposed change with its diffs
            ch = system.projects.changes.get(str(query.get("id", "")))
            if ch is None:
                return 404, {"error": "no such change"}
            return 200, {k: v for k, v in ch.items() if k != "files"} | {"files": [{k: v for k, v in f.items() if k != "after"} for f in ch["files"]]}
        if path == "/images":  # what its optical nodes took in: what each image shows, the text it read
            return 200, system.vision.status()
        if path == "/images/file":  # ?id=img-3 or ?sha=...: the image itself
            v = system.vision
            img_id = str(query.get("id", ""))
            if not img_id and query.get("sha"):
                rec = v.by_sha(str(query["sha"]))
                img_id = rec["id"] if rec else ""
            got = v.file(img_id)
            if got is None:
                return 404, {"error": "no such image"}
            return 200, Raw(*got)
        if path == "/code":  # Gitea study, the scratch pad and its practice
            return 200, {"gitea": system.code_study.status(), "scratch": system.scratch.status(),
                         "tools": system.codetools.status()}
        if path == "/websearch":  # questions nobody answered in time, searched with your SearXNG
            return 200, system.websearch.status()
        if path == "/code/item":  # ?id=code-3
            item = system.scratch.get(str(query.get("id", "")))
            return (200, item) if item is not None else (404, {"error": "no such scratch pad item"})
        if path == "/code/examples":  # ?q=&kind=&lang=&n=: functions it studied in Gitea
            return 200, {"examples": system.code_study.examples(int(query.get("n", 40)), query.get("kind") or None,
                                                                query.get("lang") or None, str(query.get("q", "")))}
        if path == "/toolsmith":
            return 200, system.toolsmith.status()
        if path == "/avatar":  # the avatar: its mind at a glance, and whether it has a voice
            from .avatar import glance
            return 200, {**glance(system), "enabled": system.directives.avatar.enabled, "voice": system.voice_tts.status()}
        if path.startswith("/memory/tests/"):  # a node's tests, from the memory database
            node = system.memory.nodes.get(path.rsplit("/", 1)[-1])
            if node is None:
                return 404, {"error": "no such memory node"}
            data = system.memory.db.read(node.id) or {}
            return 200, {"node": node.id, "own": node.own, "rev": node.rev, "text": system.memory.render_tests(node),
                         "tests": data.get("tests", []) if not node.own else
                         [{"ask": t.ask, "answer": t.answer, "learned": t.learned} for t in node.tests],
                         "unknowns": data.get("unknowns", [])}
        if path == "/mind/archive":  # ?tab=ego&at=<epoch seconds>&hours=6: what happened in its mind around then
            try:
                at = float(query.get("at") or time.time())
                hours = float(query.get("hours") or 6)
            except ValueError:
                return 400, {"error": "at is seconds since 1970, hours a number"}
            tab = query.get("tab") or None
            return 200, system.journal.around(tab, at, hours, limit=int(query.get("limit", 200) or 200))
        if path == "/mind/archive/days":  # ?tab=ego&tz=-300: the days that have entries (tz: getTimezoneOffset)
            try:
                tz = int(query.get("tz") or 0)
            except ValueError:
                tz = 0
            return 200, {"tab": query.get("tab") or "", "days": system.journal.days(query.get("tab") or None, tz),
                         "counts": system.journal.counts()}
        if path == "/memory/own":  # its own statements, kept without tests
            return 200, {"statements": system.memory.own_statements(about=query.get("about") or None,
                                                                    limit=min(200, int(query.get("limit", 50) or 50)))}
        if path.startswith("/avatar/audio/"):  # what it said, as audio (only files it made)
            got = system.voice_tts.audio(path.rsplit("/", 1)[-1])
            return (200, Raw(*got)) if got else (404, {"error": "no such audio"})
        if path == "/thinking":  # its thinking nodes: thoughts in its own words, opinions, desires (P17)
            if query.get("thought"):
                th = system.thinking.thoughts.get(query["thought"]) or system.thinking.find(query["thought"])
                if th is None:
                    return 404, {"error": "no such thought"}
                return 200, {"id": th.id, "lines": system.thinking.show(th.id, query.get("by", "api-operator"))}
            return 200, system.thinking.status()
        if path == "/kiwix":  # its Kiwix sources and recent searches (?refresh=1 reads the book list again)
            await system.seeker.catalog(refresh=bool(query.get("refresh")))
            return 200, system.seeker.status()
        if path == "/curiosity":  # areas of interest, gaps in what it knows, the objectives it could set itself
            return 200, system.curiosity.status()
        if path == "/objectives":
            if query.get("id"):
                obj = system.objectives.get(query["id"])
                if obj is None:
                    return 404, {"error": "no such objective"}
                return 200, {"id": obj.id, "report": system.objectives.report(obj),
                             "path": str(system.objectives.report_path(obj))}
            return 200, system.objectives.status()
        if path == "/memory":
            if query.get("node"):
                node = web.memory_node(system, query["node"])
                return (200, node) if node else (404, {"error": f"no memory node {query['node']}"})
            return 200, system.memory.status()
        if path == "/inquiries":
            return 200, [asdict(q) for q in system.mind.open_inquiries()]
        if path == "/proposals":
            return 200, [asdict(p) for p in system.mind.proposals.values()]
        if path == "/why":
            return 200, {"order": query.get("order", ""), "lines": system.affect.explain(query.get("order", ""))}
        if path == "/knowledge":
            q = query.get("q", "")
            if q:
                return 200, [{"title": h.doc.title, "section": h.section, "confidence": h.confidence,
                              "text": h.text, "url": h.doc.url} for h in system.knowledge.search(q)]
            return 200, {"stats": vars(system.knowledge.stats()),
                         "documents": [{"title": d.title, "url": d.url, "passages": d.passages, "hits": d.hits,
                                        "fetched_at": d.fetched_at, "reason": d.reason}
                                       for d in system.knowledge.documents(int(query.get("n", 50)))]}
    if method == "POST":
        if path == "/command":  # {"line": "/status"}: any console command (the web interface's command bar)
            from .commands import execute
            out = await execute(system, str(body.get("line", "")))
            if out.quit:
                return 200, {"lines": ["  /quit only works in the console; stop the daemon with Ctrl-C where it runs, "
                                       "or /halt to stop all activity"], "is_order": False}
            return 200, {"lines": out.lines, "is_order": out.is_order}
        if path == "/order":
            result = await system.submit(body["text"], source=body.get("source", "api"),
                                         conversation=str(body.get("conversation", ""))[:80])
            return 200, result.public_dict()  # a held answer is only shown through /release
        if path == "/research":
            doc, note = await system.research(body["topic"], reason=body.get("reason", "api request"))
            if doc is None:
                return 400, {"ok": False, "message": note}
            return 200, {"ok": True, "title": doc.title, "url": doc.url, "summary": doc.summary,
                         "passages": doc.passages}
        if path == "/release":
            ok, msg, output = system.affect.release(body["order_id"], by=body.get("by", "api-user"))
            return (200 if ok else 400), {"ok": ok, "message": msg, "output": output if ok else None}
        if path == "/calm":
            return 200, system.affect.calm(by=body.get("by", "api-operator"))
        if path == "/tasks":  # set a background task: {"spec": "learn the Krebs cycle every 5m"}
            try:
                t = system.tasks.add(body["spec"], by=body.get("by", "api-user"))
            except ValueError as exc:
                return 400, {"ok": False, "message": str(exc)}
            return 200, asdict(t)
        if path == "/task":  # {"id": "task-1", "action": "pause" | "resume" | "remove"}
            status = {"pause": "paused", "resume": "active", "remove": "removed"}.get(body.get("action", ""))
            if status is None:
                return 400, {"ok": False, "message": "action is pause, resume or remove"}
            ok, msg = system.tasks.set_status(body["id"], status, by=body.get("by", "api-user"))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/challenge":  # {"suite": "...", "mode": "exam" | "practice", "n": 20}
            try:
                rep = await system.challenges.run(body["suite"], body.get("mode", "exam"), body.get("n"),
                                                  by=body.get("by", "api-operator"))
            except ValueError as exc:
                return 400, {"ok": False, "message": str(exc)}
            return 200, {"summary": rep.line(), "score": rep.score, "learned": rep.learned, "rerouted": rep.rerouted,
                         "results": rep.results}
        if path == "/director":  # {"mode": "auto"|..., "enabled": true|false, "decide": true}
            d, by = system.director, body.get("by", "api-operator")
            out: dict = {}
            if "enabled" in body:
                out["message"] = d.set_enabled(bool(body["enabled"]), by)
            if "mode" in body:
                ok, msg = d.set_mode(str(body["mode"]), by)
                if not ok:
                    return 400, {"ok": False, "message": msg}
                out["message"] = msg
            if body.get("decide"):
                if not d.enabled or g.halted:
                    return 400, {"ok": False, "message": "its initiative is off or the mesh is halted"}
                out["decision"] = await d.decide(force=True)
            return 200, {"ok": True, **out, "status": d.status()}
        if path == "/ego":  # {"enabled": bool}
            if not isinstance(body.get("enabled"), bool):
                return 400, {"ok": False, "message": "send {\"enabled\": true | false}"}
            msg = system.ego.set_enabled(body["enabled"], by=body.get("by", "api-operator"))
            return 200, {"ok": True, "message": msg, "status": system.ego.status()}
        if path == "/dreams":  # {"action": "sleep" | "wake" | "flip"} or {"allow_sleep": bool}
            action, by = body.get("action", ""), body.get("by", "api-operator")
            if isinstance(body.get("allow_sleep"), bool):
                msg = system.dreams.set_allow_sleep(body["allow_sleep"], by=by)
                return 200, {"ok": True, "message": msg, "status": system.dreams.status()}
            if action == "flip":
                rec = await system.dreamer.flip()
                return 200, {"ok": True, "flip": rec, "status": system.dreams.status()}
            if action == "sleep":
                ok, msg = system.dreams.fall_asleep("asked to", by=by)
                return (200 if ok else 400), {"ok": ok, "message": msg, "status": system.dreams.status()}
            if action == "wake":
                ok = system.dreams.wake("woken by a person", by=by)
                return (200 if ok else 400), {"ok": ok, "message": "awake" if ok else "it is not asleep"}
            return 400, {"ok": False, "message": "action is sleep, wake or flip (or send allow_sleep: true | false)"}
        if path == "/directors":  # {"action": "spawn"} or {"action": "retire", "id": "director-1"}
            pool, by = system.directors, body.get("by", "api")
            if body.get("action") == "spawn":
                node = pool.spawn(f"asked for by {by}", by="director-0")
                return (200, {"director": node.id}) if node else (409, {"error": f"at most {pool.limit} directors"})
            if body.get("action") == "retire":
                ok, msg = pool.retire(str(body.get("id", "")), f"retired by {by}", by=by)
                return (200 if ok else 409), {"ok": ok, "message": msg}
            if body.get("attempts") is not None:
                try:
                    return 200, {"message": pool.set_attempts(int(body["attempts"]), by)}
                except (TypeError, ValueError):
                    return 400, {"error": "attempts is a whole number from 1 to 10"}
            return 400, {"error": 'action is "spawn" or "retire", or give "attempts"'}
        if path == "/tools/retest":  # {"tool": "searxng"}: run its self-tests again (after fixing what failed)
            ok, msg = system.retest_tool(str(body.get("tool", "")))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/memory/short":  # {"id": "stm-3", "action": "keep" | "forget"}
            ok, msg = system.conversations.decide(str(body.get("id", "")), str(body.get("action", "")),
                                                  str(body.get("by") or "web"))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/conversations/end":  # {"id": "conv-2"} (or the conversation key): its ego decides what to keep now
            ok, msg = system.conversations.end_now(str(body.get("id", "")), str(body.get("by") or "web"))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/tools/remove":  # {"tool": "kiwix", "why": "..."}: out of the mesh (its files stay)
            ok, msg = await system.remove_tool(str(body.get("tool", "")), str(body.get("by") or "web"), str(body.get("why", "")))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/tools/restore":  # {"tool": "kiwix"}: back in, vetted and tested again
            ok, msg = system.restore_tool(str(body.get("tool", "")), str(body.get("by") or "web"))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/tools/diagnose":  # {"tool": "kiwix"}: why does it fail? (and a fix for its code, to approve)
            d = await system.tooldoctor.diagnose(str(body.get("tool", "")), "a person asked", str(body.get("by") or "web"))
            if not d.get("ok"):
                return 400, d
            from .tooldoctor import SAY
            return 200, {**d, "message": f"{d['tool']}: {SAY.get(d['cause'], d['cause'])} - {d['why'][:200]}"
                                         + (f" ({d['outcome']})" if d.get("outcome") else "")}
        if path == "/tools/switch":  # {"mode": "on"|"off"|"decide"} for all tools; {"tool": "calculator", "mode": ...|"default"}
            by = str(body.get("by") or "web")
            if body.get("tool"):
                ok, msg = system.tool_switch.set_tool(str(body["tool"]), str(body.get("mode", "")), by)
            else:
                ok, msg = system.tool_switch.set_mode(str(body.get("mode", "")), by)
            return (200 if ok else 400), {"ok": ok, "message": msg, **system.tool_switch.status()}
        if path == "/images":  # {"name", "data": base64} or the image itself (content-type image/*; ?name=)
            import base64
            data = body.get("_bytes")
            if data is None:
                data = base64.b64decode(str(body.get("data", "")).split(",", 1)[-1], validate=False)
            name = str(body.get("name") or query.get("name") or "")
            try:
                rec = await system.vision.take(data, name=name, by=str(body.get("by") or query.get("by") or "web"))
            except ValueError as exc:
                return 400, {"error": str(exc)}
            return 200, rec
        if path == "/reward":  # {"target": "ord-12" | "last" | "log:123" | a node | a tool | code-3 | img-2 | lesson-4 |
            #                    ch-1 | "a direction in words", "value": -3..3, "why": "..."}
            ok, msg, rec = system.rewards.give(str(body.get("target", "")), body.get("value", 1), str(body.get("by") or "web"),
                                               str(body.get("why") or ""), str(body.get("kind") or ""))
            return (200 if ok else 400), {"ok": ok, "message": msg, "reward": rec}
        if path == "/projects":  # {"focus": "owner/name", "branch"} | {"clear": true} | {"objective": "..."} |
            #                      {"objective_id", "action": done|drop|reopen} | {"work": true} | {"change", "action": approve|decline}
            p, by = system.projects, str(body.get("by") or "web")
            if body.get("focus"):
                ok, msg = p.set_focus(str(body["focus"]), str(body.get("branch") or ""), by)
            elif body.get("clear"):
                ok, msg = p.clear(by)
            elif body.get("objective"):
                ok, msg, _ = p.add_objective(str(body["objective"]), by)
            elif body.get("objective_id"):
                ok, msg = p.set_objective(str(body["objective_id"]), str(body.get("action", "")), by)
            elif body.get("work"):
                res = await p.work(by=by)
                ok = bool(res and res.get("ok"))
                msg = f"{res['change']['id']}: {res['change']['status']}" if ok else (res or {}).get("why", "nothing to do")
            elif body.get("change"):
                fn = {"approve": p.approve, "decline": p.decline}.get(str(body.get("action", "")))
                if fn is None:
                    return 400, {"error": "a change is approved or declined"}
                ok, msg = fn(str(body["change"]), by)
            else:
                return 400, {"error": "say what to do: focus, clear, objective, objective_id+action, work, change+action"}
            return (200 if ok else 400), {"ok": ok, "message": msg, **p.status()}
        if path == "/websearch":  # {"inquiry": "inq-3"}: search the web for that question now; or its settings:
            #                       {"after_minutes": 30} | {"enabled": false} | {"url": "http://localhost:8080"}
            ws, by = system.websearch, str(body.get("by") or "web")
            if any(k in body for k in ("after_minutes", "enabled", "url")):
                ok, msg = ws.configure(by, after_minutes=body.get("after_minutes"), enabled=body.get("enabled"),
                                       url=body.get("url"))
                return (200 if ok else 400), {"ok": ok, "message": msg, **ws.status()}
            rec = await ws.now(str(body.get("inquiry", "")), by)
            msg = (f"answered {rec.get('inquiry')}: {rec['answer'][:160]}" if rec.get("ok") else
                   f"no answer from the web: {rec.get('why', '')}")
            return (200 if rec.get("ok") else 400), {**rec, "message": msg}
        if path == "/lessons":  # {"kind": "question"|...}: a teacher sets it a lesson now
            rec = await system.tutor.lesson(by=str(body.get("by") or "web"), kind=body.get("kind"))
            if rec is None:
                return 400, {"error": f"no lesson: {system.tutor.due() or 'the teacher set nothing it could use'}"}
            return 200, rec
        if path == "/images/correct":  # {"id": "img-3", "label": "cat"} and/or {"text": "what its text says"}
            ok, msg = system.vision.correct(str(body.get("id", "")), str(body.get("by") or "web"),
                                            label=str(body.get("label") or ""), text=body.get("text"))
            return (200 if ok else 400), {"ok": ok, "message": msg, "image": system.vision.records.get(str(body.get("id", "")))}
        if path == "/images/forget":
            ok = system.vision.forget(str(body.get("id", "")))
            return (200 if ok else 404), {"ok": ok}
        if path == "/images/practice":  # draw and learn glyphs now
            made = await system.vision.practice(int(body.get("n") or 0) or None)
            return 200, {"glyphs": made, "message": f"it drew and learned {made} glyph(s)" if made else
                         "it could not draw glyphs (it needs Pillow and fonts)"}
        if path == "/code":  # {"write": "...", "language"} | {"add": code, "language", "title"} | {"id", "code"} |
            #                  {"id", "action": ask|approve|decline|remove|check|tool} | {"study": true, "repo"} | {"practice": true}
            #                  {"auto_run": "ask"|"harmless"|"always"} | {"make_tools": "off"|"ask"|"auto"}
            sp, by = system.scratch, str(body.get("by") or "web")
            for which in ("auto_run", "make_tools"):
                if body.get(which):
                    ok, msg = sp.set_mode(which, str(body[which]), by)
                    return (200 if ok else 400), {"ok": ok, "message": msg, which: sp.mode(which)}
            if body.get("write"):
                item = await sp.write(str(body["write"]), by=by, lang=str(body.get("language") or ""))
                return (200 if item.get("id") else 400), item
            if "add" in body:
                return 200, sp.add(str(body["add"]), str(body.get("language") or "python"), str(body.get("title") or ""), by)
            if body.get("id") and "code" in body:
                ok, msg = sp.edit(str(body["id"]), str(body["code"]), by)
                return (200 if ok else 400), {"ok": ok, "message": msg, "item": sp.get(str(body["id"]))}
            if body.get("id") and body.get("action"):
                act, iid = str(body["action"]), str(body["id"])
                if act == "check":
                    item = sp.get(iid)
                    if item is None:
                        return 404, {"error": "no such scratch pad item"}
                    sp.validate(item)
                    return 200, {"ok": True, "message": f"{iid}: {sp.say_check(item)}", "item": item}
                if act == "tool":  # a person makes this code a tool the mesh can use
                    ok, msg = system.codetools.make(iid, by, how="person")
                    return (200 if ok else 400), {"ok": ok, "message": msg, "item": sp.get(iid)}
                fn = {"ask": sp.ask, "approve": sp.approve, "decline": sp.decline, "remove": sp.remove}.get(act)
                if fn is None:
                    return 400, {"error": f"unknown action {act}"}
                ok, msg = fn(iid, by)
                return (200 if ok else 400), {"ok": ok, "message": msg, "item": sp.get(iid)}
            if body.get("study"):
                res = await system.code_study.visit(by=by, repo=str(body.get("repo") or "") or None)
                return (200 if res.get("ok") else 400), {**res, "message": res.get("line") or res.get("why", "")}
            if body.get("practice"):
                rec = await sp.practice()
                return (200 if rec else 400), rec or {"error": "nothing to practise on yet (it studies Gitea first)"}
            return 400, {"error": "say what to do: write, add, id+code, id+action, study or practice"}
        if path == "/toolsmith":  # {"build": "reverse a word"} or {"try": "reverser"}: it asks before a tool runs (P15)
            smith, by = system.toolsmith, body.get("by", "api")
            if not smith.enabled:
                return 409, {"error": "the toolsmith is off ([toolsmith] enabled)"}
            if body.get("build"):
                from .commands import toolsmith_command
                lines = await toolsmith_command(system, f"build {body['build']}")
                made = smith.history[-1] if smith.history else None
                ok = bool(made and made["event"] == "drafted")
                return (200 if ok else 422), {"ok": ok, "message": " ".join(x.strip() for x in lines),
                                              "status": smith.status()}
            if body.get("try"):
                rep = await smith.experiment(str(body["try"]), by)
                if rep is None:
                    return 409, {"ok": False, "message": f"{body['try']} is not an active tool it made"}
                return 200, {"ok": True, "message": rep["line"], "results": rep.get("results", [])}
            return 400, {"error": 'give "build": "<what the tool should do>" or "try": "<tool>"'}
        if path == "/correct":  # {"order_id", "verdict": "right"} or {"order_id", "verdict": "wrong", "choice": "a2"|"none", "answer"}
            book, by, oid = system.corrections, body.get("by", "api"), str(body.get("order_id", ""))
            if body.get("verdict") == "right":
                ok, msg, learned = await book.right(oid, by)
            elif body.get("verdict") == "wrong":
                ok, msg, learned = await book.choose(oid, str(body.get("choice") or "none"), str(body.get("answer") or ""), by)
            else:
                return 400, {"error": 'give "verdict": "right", or "wrong" with "choice" (an alternative id, or "none")'}
            return (200 if ok else 400), {"ok": ok, "message": msg, "learned": learned}
        if path == "/kiwix":  # {"question": "..."}: search Kiwix for its answer now
            q = str(body.get("question", "")).strip()
            if not q:
                return 400, {"error": "question is required"}
            found = await system.seeker.seek(q, "", background=False)
            return 200, {**found.to_dict(), "summary": found.summary()}
        if path == "/avatar/speak":  # {"text": "..."} says it with ComfyUI; no text: how it is (its glance)
            from .avatar import SpeechError, glance
            text = str(body.get("text") or "").strip() or glance(system)["line"]
            try:
                said = await system.voice_tts.speak(text, by=body.get("by", "api"))
            except SpeechError as exc:
                return 503, {"ok": False, "error": str(exc), "text": system.voice_tts.clean(text)}
            return 200, {"ok": True, "url": f"/avatar/audio/{said['file']}", **said}
        if path == "/curiosity":  # {"pursue": true}: its director sets itself an objective from its curiosity now
            rec = await system.director.pursue_now(body.get("by", "api"))
            if rec is None:
                return 409, {"ok": False, "message": "its initiative is off or the mesh is halted (turn it on in Mind > Initiative)"}
            return 200, {"ok": rec.get("mode") == "pursue", "message": rec.get("summary", ""), "decision": rec}
        if path == "/thinking":  # {"form": true} re-forms its thoughts now; {"forget": "th-3"} drops one
            by = body.get("by", "api")
            if body.get("forget"):
                ok, msg = system.thinking.forget(str(body["forget"]), by)
                return (200 if ok else 404), {"ok": ok, "message": msg}
            changed = await system.thinking.form(force=True)
            return 200, {"ok": True, "message": f"{changed['formed']} new, {changed['reformed']} re-formed, "
                                                f"{changed['dropped']} dropped", **changed}
        if path == "/objectives":  # {"text": "..."} sets one; {"id", "action": pause|resume|remove|think}; {"id", "ask"}
            book, by = system.objectives, body.get("by", "api")
            if not book.enabled:
                return 409, {"error": "objectives are off ([objectives] enabled)"}
            if body.get("text"):
                try:
                    obj = await book.add(str(body["text"]), by=by)
                except ValueError as exc:
                    return 400, {"ok": False, "message": str(exc)}
                return 200, {"ok": True, "message": f"set {obj.id}: {obj.text} ({obj.strategy})", "id": obj.id}
            if body.get("ruminate"):
                got = await book.second_thoughts(by=by)
                return 200, {"ok": True, "message": "; ".join(g["line"] for g in got) or "nothing to think over again right now",
                             "thoughts": got}
            oid = str(body.get("id", ""))
            if body.get("ask"):
                ok, msg = book.ask(oid, str(body["ask"]), by)
                return (200 if ok else 400), {"ok": ok, "message": msg}
            action = body.get("action")
            if action == "think":
                obj = book.get(oid)
                if obj is None:
                    return 404, {"ok": False, "message": f"no objective {oid}"}
                return 200, {"ok": True, "message": await book.ruminate(obj, by)}
            if action == "retry":  # try an archived objective again
                ok, msg = await book.retry(oid, by)
                return (200 if ok else 400), {"ok": ok, "message": msg}
            if action in ("pause", "resume", "remove"):
                ok, msg = book.set_status(oid, {"pause": "paused", "resume": "active", "remove": "removed"}[action], by)
                return (200 if ok else 400), {"ok": ok, "message": msg}
            return 400, {"error": 'give "text", or "id" with "action" (pause|resume|remove|think) or "ask", or "ruminate": true'}
        if path == "/imagine":  # {"prompt": "a city on Mars"}
            result = await system.submit(f"imagine {body.get('prompt', '')}", source=body.get("source", "api"))
            return 200, result.public_dict()
        if path == "/memorize":  # {"ask": "...", "answer": "..."}
            made = system.memory.learn(body.get("ask", ""), body.get("answer", ""), body.get("source") or "the operator",
                                       by=body.get("by", "api"), kind="people")
            return (200, {"node": made[0], "status": made[1]}) if made else (400, {"error": "not kept (memory off, or "
                                                                                           "not a question with an answer)"})
        if path.startswith("/memory/tests/"):  # {"tests": [{"ask", "answer"}...], "unknowns": [...]}: a person's edit
            nid = path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", nid) or not isinstance(body.get("tests"), list):
                return 400, {"ok": False, "error": "send {\"tests\": [{\"ask\": ..., \"answer\": ...}, ...]}"}
            res = system.memory.edit_tests(nid, body["tests"], body.get("unknowns"), by=body.get("by", "api"))
            return (200 if res["ok"] else 400), res
        if path == "/memory/test":
            system.memory.sync_tests(force=True)
            results = [await asyncio.to_thread(system.memory.run_tests, n) for n in system.memory.fact_nodes()]
            return 200, {"nodes": results, "passing": all(r["passed"] == r["total"] for r in results)}
        if path == "/challenge/fetch":
            # {"benchmark": "gsm8k", "rows": 200, "split": "test"}, or any dataset on the Hub:
            # {"dataset": "owner/name", "config": "...", "split": "...", "rows": 300, "name": "...",
            #  "mapping": {"input": "question", "answer": "label", "kind": "classify", "context": "", "choices": ""}}
            by = body.get("by", "api-operator")
            try:
                suite = await system.challenges.get(str(body.get("benchmark", "")) if not body.get("dataset") else "",
                                                    str(body.get("dataset", "")), body.get("mapping") or {},
                                                    int(body.get("rows", 0) or 0), body.get("config", ""), body.get("split", ""),
                                                    body.get("name", ""), body.get("capability", ""), by)
            except (ValueError, RuntimeError, KeyError) as exc:
                return 400, {"ok": False, "message": str(exc)}
            return 200, {"ok": True, "suite": suite.name, "items": len(suite.items), "source": suite.source,
                         "kind": suite.kind, "message": f"downloaded {len(suite.items)} items as {suite.name}"}
        if path == "/challenges/import":  # {"name", "kind", "items": "<JSONL or CSV text>", "capability", "labels"}
            try:
                suite = system.challenges.add_own(str(body.get("name", "")), str(body.get("kind", "")),
                                                  str(body.get("items", "")), str(body.get("capability", "")),
                                                  body.get("labels") or [], str(body.get("description", "")),
                                                  body.get("by", "api-operator"))
            except (ValueError, json.JSONDecodeError) as exc:
                return 400, {"ok": False, "message": str(exc)}
            return 200, {"ok": True, "suite": suite.name, "items": len(suite.items),
                         "message": f"added {suite.name}: {len(suite.items)} items"}
        if path == "/challenges/remove":  # {"suite": "..."}
            ok, msg = system.challenges.remove(str(body.get("suite", "")), body.get("by", "api-operator"))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/reply":
            ok, lines = await system.mind.reply(body["inquiry_id"], body["answer"], by=body.get("by", "api-user"))
            return (200 if ok else 400), {"ok": ok, "message": "\n".join(x.strip() for x in lines)}
        if path in ("/adopt", "/decline"):
            by = body.get("by", "api-operator")
            ok, msg = (system.mind.adopt(body["proposal_id"], by, body.get("labels")) if path == "/adopt"
                       else system.mind.decline(body["proposal_id"], by, body.get("note", "")))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/investigate":
            from .mind import Goal
            rec = await system.investigator.investigate(Goal(body["goal"], body.get("reason", "api request"), "operator"))
            return 200, rec
        if path == "/feedback":
            ok, msg = system.feedback(body["order_id"], body["label"])
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/teach":
            system.teach(body["capability"], body["text"], body["label"])
            return 200, {"ok": True}
        if path == "/answer":
            ok, msg = system.answer_question(body["question_id"], body["label"])
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/define":
            spec = system.define_capability(body["name"], body.get("labels", []), body.get("description", ""),
                                            body.get("triggers"))
            return 200, spec.to_dict()
        if path == "/approve":
            a = g.decide(body["id"], bool(body.get("approve", True)), by=body.get("by", "api-operator"),
                         note=body.get("note", ""))
            return (200, a.to_dict()) if a else (404, {"error": "no pending approval with that id"})
        if path == "/halt":
            g.halt(body.get("reason", "operator request"), by=body.get("by", "api-operator"))
            return 200, {"halted": True}
        if path == "/resume":
            ok, msg = g.resume(by=body.get("by", "api-operator"))
            return (200 if ok else 400), {"ok": ok, "message": msg}
        if path == "/pause":
            g.pause_background(bool(body.get("paused", True)), by=body.get("by", "api-operator"))
            return 200, {"paused": g.paused}
        if path == "/rollback":
            node = system.registry.rollback(body["capability"])
            return (200, {"champion": node.id}) if node else (404, {"error": "nothing to roll back to"})
    return 404, {"error": f"no route {method} {path}"}


async def serve_http(system, host: str = "127.0.0.1", port: int = 8765):
    token = os.environ.get("NEUROMESH_TOKEN")

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        code, payload = 500, {"error": "internal error"}
        page: tuple[bytes, str] | None = None
        try:
            request_line = (await reader.readline()).decode("latin-1").strip()
            method, target, _ = request_line.split(" ", 2)
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length", "0") or 0)
            if length > 64 * 1048576:
                raise ValueError("the request is larger than 64 MB")
            raw = await reader.readexactly(length) if length else b""
            if headers.get("content-type", "").startswith(("image/", "application/octet-stream")):
                body = {"_bytes": raw}  # an image sent as it is (POST /images)
            else:
                body = json.loads(raw) if raw else {}
            url = urlparse(target)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            page = web.static_file(url.path) if method.upper() in ("GET", "HEAD") else None
            if page is not None:  # the web interface's files: no data in them, so no token needed
                pass
            elif token and headers.get("authorization") != f"Bearer {token}":
                code, payload = 401, {"error": "missing or wrong bearer token"}
            else:
                code, payload = await _route(system, method.upper(), url.path, query, body)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            code, payload = 400, {"error": f"bad request: {exc}"}
        except Exception as exc:  # keep the server alive
            code, payload = 500, {"error": repr(exc)}
        if page is not None:
            code, (data, ctype) = 200, page
        elif isinstance(payload, Raw):
            data, ctype = payload.data, payload.ctype
        else:
            data, ctype = json.dumps(payload, default=str).encode(), "application/json"
        writer.write(f"HTTP/1.1 {code} {REASONS.get(code, 'Error')}\r\nContent-Type: {ctype}\r\n"
                     f"Content-Length: {len(data)}\r\nCache-Control: no-store\r\n"
                     f"X-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n".encode() + data)
        try:
            await writer.drain()
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port)
