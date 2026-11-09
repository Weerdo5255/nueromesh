"""Its own code as tools: a Python function it wrote in the scratch pad, kept as a tool it can use from then on.

    code that ran      a function that takes something (a parameter), called at the top of the code to show what it
                       does - print(reverse_words("the cat sat")) - and what those calls printed when it ran: the
                       examples (input -> output) that become the tool's self-tests
    the wrapper        tool.py: the code without its demonstration, and a main() that hands the function an order's
                       words (after "name:" if the order names it), its arguments split at commas and typed like the
                       examples' (numbers as numbers, lists as lists), and prints the answer as JSON
    checked            like any tool the toolsmith makes (toolsmith.check_code): standard library only, and nothing that
                       reaches outside - no files, network, other programs, eval or exec. Made tools get no permissions.
    installed          by a person (Make it a tool in the Code view, /code tool code-3); or, after code it wrote ran
                       cleanly, offered by the mesh ([code] make_tools = "ask": a person approves, as for any tool it
                       makes) or installed by itself ("auto"). Either way it is vetted and self-tested like any tool,
                       the router learns the orders it meets, the toolsmith experiments with it, and the tool switch
                       decides each time whether it is worth using.

Code that is not Python, reaches outside, takes nothing, or never showed what it does stays in the scratch pad (it
says why).
"""
from __future__ import annotations

import ast
import json
import re
import time

from .toolsmith import _COMMON, _toml_str, check_code, manifest_text

_TYPES = (str, int, float, bool, list, tuple)


def _literal(node):
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None, False
    if isinstance(value, _TYPES) and (not isinstance(value, (list, tuple)) or all(isinstance(v, (str, int, float, bool)) for v in value)):
        return value, True
    return None, False


def _demo(node: ast.AST, functions: set[str]) -> dict | None:
    """print(f(<literal>, ...)) or print(f(<literal>, NAME)["output"]): a demonstration of a function it defines."""
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print" and len(node.value.args) == 1 and not node.value.keywords):
        return None
    call, suffix = node.value.args[0], None
    if isinstance(call, ast.Subscript) and isinstance(call.slice, ast.Constant) and isinstance(call.slice.value, str):
        call, suffix = call.value, call.slice.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in functions
            and call.args and not call.keywords):
        return None
    values, literal = [], []
    for a in call.args:
        v, ok = _literal(a)
        if not ok and not isinstance(a, ast.Name):
            return None
        values.append(v)
        literal.append(ok)
    if not literal[0]:
        return None
    return {"function": call.func.id, "values": values, "literal": literal,
            "rest": [None if ok else a.id for a, ok in zip(call.args, literal)], "suffix": suffix}


def as_input(d: dict) -> str:
    """The text a tool is given for a demonstration's arguments (the literal ones)."""
    def one(v):
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return str(v)
    lits = [v for v, ok in zip(d["values"], d["literal"]) if ok]
    if len(lits) == 1:
        return one(lits[0])
    return ", ".join(one(v) for v in lits)


def plan(code: str, stdout: str) -> tuple[dict | None, str]:
    """What tool this code would make: its function, the demonstrations and what they printed. (plan, why not)"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"it does not parse ({exc.msg})"
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    if not functions:
        return None, "it defines no function to keep as a tool"
    body = list(tree.body)
    for n in tree.body:  # an `if __name__ == "__main__":` block holds the demonstration
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__"):
            body += n.body
    prints = [n for n in body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Name) and n.value.func.id == "print"]
    demos = [d for d in (_demo(n, set(functions)) for n in prints) if d is not None]
    if not demos:
        return None, "it never shows what its function does (a print of the function called on something)"
    lines = [x for x in stdout.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(demos) != len(prints) or len(lines) != len(demos):
        return None, "its output can't be matched to its demonstrations (one printed line for each)"
    fn = demos[0]["function"]
    mine = [(d, out) for d, out in zip(demos, lines) if d["function"] == fn and d["literal"] == demos[0]["literal"]
            and d["suffix"] == demos[0]["suffix"]]
    params = functions[fn].args
    if not (params.args or params.vararg):
        return None, f"{fn} takes nothing, so there is nothing to give it"
    cut = []  # the demonstration is left out of the tool
    for n in tree.body:
        if n in prints or (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                           and n.test.left.id == "__name__"):
            cut.append((n.lineno, n.end_lineno))
    keep = [line for i, line in enumerate(code.splitlines(), 1) if not any(a <= i <= b for a, b in cut)]
    return {"function": fn, "doc": ast.get_docstring(functions[fn]) or "", "demo": demos[0],
            "examples": [{"input": as_input(d), "expect": out} for d, out in mine],
            "source": "\n".join(keep).rstrip() + "\n"}, ""


def wrapper(p: dict, name: str, header: str) -> str:
    d = p["demo"]
    like = [v for v, ok in zip(d["values"], d["literal"]) if ok]
    args, li = [], 0
    for ok, rest in zip(d["literal"], d["rest"]):
        if ok:
            args.append(f"_args[{li}]")
            li += 1
        else:
            args.append(rest)
    call = f"{p['function']}({', '.join(args)})" + (f"[{d['suffix']!r}]" if d["suffix"] else "")
    colon = any(":" in e["input"] for e in p["examples"])
    return (f'"""{header}"""\n' + p["source"] + f'''

# ---------------------------------------------------------------- the tool (NeuroMesh's wrapper around the code above)
import json as _json
import re
import sys as _sys

_LIKE = {like!r}  # a demonstration's arguments: what it is given is read like these
_NAMES = {[name, name.replace("_", " "), p["function"], p["function"].replace("_", " ")]!r}


def _typed(raw, like):
    raw = raw.strip()
    if isinstance(like, bool):
        return raw.lower() in ("true", "yes", "1")
    if isinstance(like, int):
        return int(float(raw)) if re.fullmatch(r"-?\\d+(\\.0+)?", raw) else int(raw)
    if isinstance(like, float):
        return float(raw)
    if isinstance(like, (list, tuple)):
        inner = raw.strip("[]()")
        parts = [x.strip() for x in inner.split(",")] if "," in inner else inner.split()
        return type(like)(_typed(x, like[0] if like else "") for x in parts if x)
    if len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "'\\"":
        return raw[1:-1]
    return raw


def _call(text):
    text = str(text).strip()
    head, sep, tail = text.partition(":")
    if sep and tail.strip() and (head.strip().lower() in _NAMES
                                 or ({not colon!r} and len(head) < 60 and re.fullmatch(r"[A-Za-z][A-Za-z '-]*", head.strip()))):
        text = tail.strip()  # "{name}: ..." - the words it was given
    if len(_LIKE) == 1:
        _args = [_typed(text, _LIKE[0])]
    else:
        parts = [x.strip() for x in text.split(",", len(_LIKE) - 1)]
        if len(parts) != len(_LIKE):
            raise ValueError(f"it needs {{len(_LIKE)}} things, separated by commas")
        _args = [_typed(x, like) for x, like in zip(parts, _LIKE)]
    return {call}


if __name__ == "__main__":
    try:
        _request = _json.load(_sys.stdin)
        print(_json.dumps({{"output": str(_call(_request.get("input", "")))}}))
    except Exception as exc:
        print(_json.dumps({{"error": f"{{type(exc).__name__}}: {{exc}}"}}))
''')


class CodeTools:
    """Code from its scratch pad, kept as tools (see the module docstring)."""

    def __init__(self, system):
        self.sys = system
        raw = system.store.get_kv("codetools.state")
        st = json.loads(raw) if raw else {}
        self.made: dict[str, dict] = st.get("made", {})   # item id -> {tool, version, digest, how, by, ts}
        self.considered: dict[str, str] = st.get("considered", {})  # item id -> digest it already thought about

    def save(self) -> None:
        self.sys.store.set_kv("codetools.state", json.dumps({"made": self.made, "considered": self.considered}))

    def candidate(self, item: dict) -> tuple[dict | None, str]:
        """Could this code be a tool? (plan, why not)"""
        if item is None:
            return None, "no such code"
        if item["language"] != "python":
            return None, "only Python code becomes a tool"
        runs = [r for r in item.get("runs", []) if r.get("ok") and r.get("digest") == item["digest"]]
        if not runs:
            return None, "it hasn't run cleanly as it stands (run it first)"
        p, why = plan(item["code"], runs[-1]["stdout"])
        if p is None:
            return None, why
        problems = [x for x in check_code(wrapper(p, p["function"], "")) if x != "it never prints a JSON answer"]
        if problems:
            return None, "a tool may not: " + "; ".join(problems[:4])
        return p, ""

    def make(self, item_id: str, by: str, how: str = "person") -> tuple[bool, str]:
        """Make code into a tool. how: "person" (installed now: a person chose it), "ask" (offered: a person approves)
        or "auto" (installed by itself: [code] make_tools = "auto", switched on by a person)."""
        s = self.sys
        item = s.scratch.get(item_id)
        p, why = self.candidate(item)
        if p is None:
            return False, f"{item_id} can't be a tool: {why}"
        ts = s.toolsmith
        recipe = f"code:{item_id}"
        name = ts._name(re.sub(r"[^a-z0-9_]", "_", p["function"].lower()).strip("_")[:40] or "made_tool", recipe)
        old = ts.drafts.get(name)
        if old is not None and old.get("status") == "awaiting approval":
            return True, f"{name} is already waiting for approval ({old.get('approval')})"
        from .toolsmith import _bump
        version = _bump(old["version"]) if old else "1.0.0"
        title = (p["doc"].splitlines()[0].strip().rstrip(".") if p["doc"] else "") or item["title"]
        if title[:1].isupper() and title[1:2].islower():
            title = title[0].lower() + title[1:]  # "to write a text backwards", in its question
        made_how = {"person": f"{by} made it a tool", "ask": "it offered it as a tool",
                    "auto": f"it made it a tool by itself (make_tools: auto, switched on by {s.scratch.modes['make_tools'].get('by')})"}[how]
        header = (f"{name} v{version}: {title}.\n\nMade by NeuroMesh from its own code in the scratch pad ({item_id}: "
                  f"{item['title'][:80]}), which ran cleanly; {made_how}. Its self-tests (manifest.toml) are what the code's "
                  f"own demonstration printed. It runs only as a tool, within a tool's limits, with no permissions.\n")
        source = wrapper(p, name, header)
        tests = [{"input": e["input"], "expect": e["expect"], "route": False} for e in p["examples"][:4]]
        tests += [{"input": f"{name}: {e['input']}", "expect": e["expect"]} for e in p["examples"][:3]]
        words = [w for w in re.findall(r"[a-z]{4,}", f"{title} {item['title']} {p['function'].replace('_', ' ')}".lower())
                 if w not in _COMMON]
        templates = [f"{name}: {{x}}", f"use {name} on {{x}}", f"{p['function'].replace('_', ' ')} {{x}}"]
        fillers = {"x": [e["input"] for e in p["examples"][:6]]}
        manifest = manifest_text(name, version, f"{title[:1].upper()}{title[1:]} (made by NeuroMesh from its own code, {item_id})",
                                 list(dict.fromkeys(words))[:6], templates, fillers, tests,
                                 [item.get("request") or item["title"]])
        manifest = manifest.replace('author = "neuromesh"                  # written by its toolsmith; runs only once a person approves (P15)',
                                    f'author = "neuromesh"                  # made from its own code ({item_id}); see tool.py')
        auto_by = by if how == "person" else (f"make_tools: auto (switched on by {s.scratch.modes['make_tools'].get('by')})"
                                              if how == "auto" else "")
        d = ts._offer(name, version, recipe, title, source, manifest, tests, [item.get("request") or item["title"]],
                      f"{made_how} ({item_id} ran cleanly and shows what it does)", by,
                      {"from": item_id, "digest": item["digest"], "function": p["function"]}, item["title"],
                      auto_by=auto_by)
        if d is None:
            return False, f"{item_id} can't be a tool: its wrapper didn't pass the toolsmith's checks"
        self.made[item_id] = {"tool": name, "version": version, "digest": item["digest"], "how": how, "by": by,
                              "ts": round(time.time(), 1), "status": d["status"]}
        item["tool"] = {"name": name, "version": version, "status": d["status"]}
        s.scratch._save(item)
        self.save()
        s.bus.publish("code.tool", id=item_id, tool=name, version=version, how=how, by=by, status=d["status"])
        if d["status"] == "awaiting approval":
            return True, f"{item_id} is offered as the tool {name}; it waits for approval ({d['approval']})"
        return True, f"{item_id} is now the tool {name} v{version}: it is self-tested next, then the router can send orders to it"

    def consider(self, item: dict) -> None:
        """Code it wrote ran cleanly: is it worth keeping as a tool? ([code] make_tools: off | ask | auto)"""
        s = self.sys
        mode = s.scratch.mode("make_tools")
        if (mode == "off" or item is None or (item.get("origin") or {}).get("how") in (None, "", "person")
                or self.considered.get(item["id"]) == item["digest"] or item["id"] in self.made):
            return
        self.considered[item["id"]] = item["digest"]
        self.save()
        p, why = self.candidate(item)
        if p is None:
            return
        rw = getattr(s, "rewards", None)  # people discouraged making tools of its code: it doesn't
        if rw is not None and rw.enabled and rw.score("code", "tool") < -0.4:
            return
        if any(t.manifest.author == "neuromesh" and (t.name == p["function"] or t.name.startswith(p["function"]))
               for t in s.tools.tools.values()):
            return  # it has one like it
        if any((d.get("params") or {}).get("function") == p["function"] and d["status"] not in ("retired",)
               for d in s.toolsmith.drafts.values()):
            return  # it offered one like it already (waiting, or declined: it doesn't ask again)
        self.make(item["id"], "neuromesh", how="auto" if mode == "auto" else "ask")

    def on_event(self, ev) -> None:
        if ev.topic == "code.ran" and ev.data.get("ok"):
            try:
                self.consider(self.sys.scratch.get(ev.data.get("id", "")))
            except Exception:  # never break the bus
                pass
        elif ev.topic in ("tool.installed", "tool.active", "tool.declined", "tool.quarantined"):
            for item_id, m in self.made.items():
                if m["tool"] == ev.data.get("tool"):
                    m["status"] = ev.topic.split(".")[1]
                    item = self.sys.scratch.get(item_id)
                    if item is not None and item.get("tool"):
                        item["tool"]["status"] = m["status"]
                        self.sys.scratch._save(item)
            self.save()

    def status(self) -> dict:
        return {"made": self.made, "mode": self.sys.scratch.mode("make_tools")}


def toml_ok(text: str) -> bool:
    import tomllib
    try:
        tomllib.loads(text)
        return True
    except Exception:
        return False


__all__ = ["CodeTools", "plan", "wrapper", "as_input", "toml_ok", "_toml_str"]
