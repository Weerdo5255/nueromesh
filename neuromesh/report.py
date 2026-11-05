"""Human-readable rendering of events, status and node lineage."""
from __future__ import annotations

from .bus import Event


def describe_event(ev: Event) -> str | None:
    d, t = ev.data, ev.topic
    if t == "node.spawned":
        return f"[spawn]    {d['node']} ({d['origin']}, {d['arch']}) - {d['reason']}"
    if t == "node.promoted":
        rep = f", replacing {d['replaced']}" if d.get("replaced") else ""
        return f"[promote]  {d['node']} now serves '{d['capability']}'{rep} - {d['reason']}"
    if t == "node.retired":
        return f"[retire]   {d['node']} - {d['reason']}"
    if t == "node.disposed":
        return f"[dispose]  {d['node']} - {d['reason']}"
    if t == "tool.discovered":
        perms = ", ".join(d["permissions"]) or "none"
        return f"[tool]     discovered '{d['tool']}' v{d['version']} (permissions: {perms})"
    if t == "tool.approved":
        return f"[tool]     '{d['tool']}' approved ({d['by']})"
    if t == "tool.rejected":
        return f"[tool]     '{d['tool']}' REJECTED - {d['reason']}"
    if t == "tool.active":
        return f"[tool]     '{d['tool']}' active - {d['reason']}; {d['seeded']} routing examples seeded"
    if t == "tool.quarantined":
        return f"[tool]     '{d['tool']}' quarantined - {d['reason']}"
    if t == "tool.retired":
        return f"[tool]     '{d['tool']}' retired - {d['reason']}"
    if t == "tool.invalid":
        return f"[tool]     invalid plugin '{d['tool']}': {d['error']}"
    if t == "approval.requested":
        perms = ", ".join(d["detail"].get("permissions", []))
        return f"[approval] {d['id']}: {d['kind']} '{d['subject']}' needs a human decision (permissions: {perms})"
    if t == "approval.decided":
        return f"[approval] {d['id']} {d['status']} by {d['by']}"
    if t == "capability.labels_changed":
        return f"[label]    '{d['capability']}' learned label '{d['label']}' (from {d['by']})"
    if t == "capability.new":
        return f"[skill]    new capability '{d['capability']}' labels={d['labels']}"
    if t == "teacher.lesson":
        return f"[lesson]   {d['source']} taught {d['examples']} x {d['capability']}={d['label']} ({d['reason']})"
    if t == "knowledge.learned":
        why = "background" if d.get("background") else "for an order"
        n = d["passages"]
        return f"[learn]    {d['source']} '{d['title']}' retained ({n} passage{'s' if n != 1 else ''}, {why}: {d['reason']})"
    if t == "knowledge.recalled":
        return f"[recall]   '{d['question'][:60]}' answered from '{d['title']}' ({d['confidence']:.2f})"
    if t == "knowledge.forgotten":
        return f"[forget]   '{d['title']}' - {d['reason']}"
    if t == "knowledge.failed":
        return f"[research] {d['tool']} could not look up '{d['topic'][:60]}': {d['error']}"
    if t == "teacher.status":
        verb = "ready" if d["state"] == "ready" else f"unavailable ({d['state']})"
        return f"[teacher]  {d['teacher']} {verb}: {d['detail']}"
    if t == "system.halt":
        return f"[HALT]     {d['reason']} (by {d['by']})"
    if t == "system.resume":
        return f"[resume]   by {d['by']}"
    return None


def _acc(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def render_status(st: dict) -> str:
    lines = [f"{st['name']}  up {st['uptime_s']:.0f}s  orders={st['orders']}  inflight={st['inflight']}  "
             f"background duty={st['background_duty_cycle']:.0%}"
             + ("  ** HALTED: " + st["halt_reason"] if st["halted"] else "")
             + ("  (background paused)" if st["background_paused"] else "")]
    lines.append(f"compute: {st['compute']}")
    n = st["nodes"]
    lines.append(f"nodes: {n['live']} live, {n['ever']} ever built, {n['disposed']} disposed   "
                 f"recent orders served by: {st['served_recent']}")
    lines.append("")
    lines.append(f"{'capability':<12} {'examples':>8}  {'champion':<24} {'arch':<16} {'held-out':>8}  candidates")
    for name, c in st["capabilities"].items():
        ch = c["champion"]
        cands = ", ".join(f"{x['id']}({_acc(x['holdout_acc']).strip()})" for x in c["candidates"]) or "-"
        lines.append(f"{name:<12} {c['examples']:>8}  {(ch['id'] if ch else '-'):<24} "
                     f"{(ch['arch'] if ch else ''):<16} {_acc(ch['holdout_acc'] if ch else None):>8}  {cands}")
    lines.append("")
    lines.append("tools: " + ", ".join(f"{k}={v['state']}" for k, v in st["tools"].items()))
    if st["approvals"]:
        lines.append("pending approvals: " + ", ".join(f"{a['id']} ({a['kind']} {a['subject']})"
                                                       for a in st["approvals"]))
    if st["questions"]:
        lines.append(f"questions for a human teacher: {len(st['questions'])}  (/questions)")
    lines.append("teachers (in order asked):")
    for name, v in st["teachers"].items():
        model = f" {v['provider']}:{v['model']}" if v.get("provider") else ""
        usage = st["llm_usage"].get(name)
        used = f", {usage['calls_last_hour']} calls/{usage['tokens_last_hour']} tokens this hour" if usage else ""
        state = "ready" if v["available"] else f"off ({v.get('detail') or 'unavailable'})"
        lines.append(f"  {name:<10}{model:<32} {state}; {v['lessons']} lessons{used}")
    k = st.get("knowledge")
    if k:
        r = k["research"]
        lines.append(f"knowledge: {k['documents']} articles / {k['passages']} passages retained; "
                     f"{k['recalled']} recalled, {k['researched']} researched this run; research "
                     f"{r['last_hour']}/{r['limit']} this hour; tool: {k['tool'] or 'none active'}")
    lines.append("")
    lines.append("directive scorecard:")
    for g in st["scorecard"]:
        mark = "met " if g["met"] else "open"
        op = ">=" if g["direction"] == "at_least" else "<="
        lines.append(f"  [{mark}] {g['id']:<16} {g['metric']} = {g['value']}  (target {op} {g['target']})")
    return "\n".join(lines)


def render_lineage(lineage: dict[str, dict], capability: str | None = None) -> str:
    children: dict[str | None, list[dict]] = {}
    for meta in sorted(lineage.values(), key=lambda m: m["created_at"]):
        if capability and meta["capability"] != capability:
            continue
        parent = meta.get("parent_id") if meta.get("parent_id") in lineage else None
        children.setdefault(parent, []).append(meta)
    lines: list[str] = []

    def walk(meta: dict, prefix: str, last: bool) -> None:
        acc = meta.get("metrics", {}).get("holdout_acc")
        status = meta["status"].upper() if meta["status"] == "champion" else meta["status"]
        lines.append(f"{prefix}{'`-- ' if last else '|-- '}{meta['id']}  [{status}]  {meta['origin']} "
                     f"{meta['arch']['hidden']} lr={meta['arch']['lr']:g}  held-out={_acc(acc).strip()}  "
                     f"steps={meta['train_steps']}  labels={len(meta['labels'])}")
        kids = children.get(meta["id"], [])
        for i, kid in enumerate(kids):
            walk(kid, prefix + ("    " if last else "|   "), i == len(kids) - 1)

    roots = children.get(None, [])
    by_cap: dict[str, list[dict]] = {}
    for r in roots:
        by_cap.setdefault(r["capability"], []).append(r)
    for cap, rs in by_cap.items():
        lines.append(f"{cap}:")
        for i, r in enumerate(rs):
            walk(r, "  ", i == len(rs) - 1)
    return "\n".join(lines)
