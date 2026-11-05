"""A small JSON-over-HTTP API (stdlib only) so other programs and agents can send orders,
teach the system and approve tools while it runs as a daemon.

Binds to 127.0.0.1 by default. Set NEUROMESH_TOKEN to require "Authorization: Bearer <token>".
"""
from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import parse_qs, urlparse

from .report import render_lineage

REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 500: "Internal Server Error"}


async def _route(system, method: str, path: str, query: dict, body: dict):
    g = system.guardian
    if method == "GET":
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
    if method == "POST":
        if path == "/order":
            result = await system.submit(body["text"], source=body.get("source", "api"))
            return 200, result.to_dict()
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
            if token and headers.get("authorization") != f"Bearer {token}":
                code, payload = 401, {"error": "missing or wrong bearer token"}
            else:
                code, payload = await _route(system, method.upper(), url.path, query, body)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            code, payload = 400, {"error": f"bad request: {exc}"}
        except Exception as exc:  # keep the server alive
            code, payload = 500, {"error": repr(exc)}
        data = json.dumps(payload, default=str).encode()
        writer.write(f"HTTP/1.1 {code} {REASONS.get(code, 'Error')}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
        try:
            await writer.drain()
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port)
