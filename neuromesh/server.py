"""A small JSON-over-HTTP API (stdlib only) so other programs and agents can send orders,
teach the system and approve tools while it runs as a daemon - and the web interface at /,
which is built on the same API (web.py, neuromesh/web/).

Binds to 127.0.0.1 by default. Set NEUROMESH_TOKEN to require "Authorization: Bearer <token>" on
every API call (the page itself holds no data and asks for the token).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from urllib.parse import parse_qs, urlparse

from . import web
from .report import render_lineage

REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed",
           500: "Internal Server Error"}


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
        if path == "/evolution":
            return 200, {**system.evolution.status(), "archive": system.evolution.archive}
        if path == "/director":
            return 200, system.director.status()
        if path == "/dreams":
            return 200, system.dreams.status()
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
        if path == "/traces":  # the paths recent orders took through the mesh, hop by hop
            from . import anatomy
            return 200, {"traces": anatomy.traces(system, min(50, int(query.get("n", 15))))}
        if path == "/toolsmith":
            return 200, system.toolsmith.status()
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
            result = await system.submit(body["text"], source=body.get("source", "api"))
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
        if path == "/dreams":  # {"action": "sleep" | "wake"}
            action, by = body.get("action", ""), body.get("by", "api-operator")
            if action == "sleep":
                ok, msg = system.dreams.fall_asleep("asked to", by=by)
                return (200 if ok else 400), {"ok": ok, "message": msg, "status": system.dreams.status()}
            if action == "wake":
                ok = system.dreams.wake("woken by a person", by=by)
                return (200 if ok else 400), {"ok": ok, "message": "awake" if ok else "it is not asleep"}
            return 400, {"ok": False, "message": "action is sleep or wake"}
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
        if path == "/memory/test":
            system.memory.sync_files(force=True)
            results = [await asyncio.to_thread(system.memory.run_tests, n) for n in list(system.memory.nodes.values())]
            return 200, {"nodes": results, "passing": all(r["passed"] == r["total"] for r in results)}
        if path == "/challenge/fetch":  # {"benchmark": "gsm8k", "rows": 200, "split": "test"}
            try:
                suite = await asyncio.to_thread(system.challenges.fetch, body["benchmark"], int(body.get("rows", 0)),
                                                body.get("split", ""), body.get("name", ""), body.get("by", "api-operator"))
            except (ValueError, RuntimeError) as exc:
                return 400, {"ok": False, "message": str(exc)}
            return 200, {"suite": suite.name, "items": len(suite.items), "source": suite.source}
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
            raw = await reader.readexactly(length) if length else b""
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
