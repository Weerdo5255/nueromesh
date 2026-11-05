"""The mesh's voice: every reply leads with the result, in the mesh's own words. Sources come
after it, and the reasoning (the trace) after that.

    12 * 7 = 84.
    That reads as positive.
    Ada Lovelace died on 27 November 1852 at the age of 36.
    Source: kiwix 'Ada Lovelace' http://localhost:8080/... ("Lovelace died on 27 November 1852 ...")

Result.output keeps the raw result (84, "positive", ...) for programs; Result.answer is what a
person reads. With an LLM teacher the answer to a question is written by the teacher from the
retained passages; without one it is restated from them (see knowledge.restate).
"""
from __future__ import annotations

import re

_PREAMBLE = re.compile(r"^\s*(?:(?:based on|according to|from|using)\s+(?:the\s+)?(?:provided\s+|given\s+|retained\s+)?"
                       r"(?:passages?|articles?|texts?|sources?|information|context)[^,:]*[,:]\s*|"
                       r"(?:sure|certainly|of course|great question)[,!.]\s*|answer:\s*)", re.I)


def lead_first(text: str) -> str:
    """Drop a preamble an LLM may put before its answer ("Based on the passages, ...")."""
    t = str(text).strip()
    for _ in range(2):
        m = _PREAMBLE.match(t)
        if not m:
            break
        t = t[m.end():].lstrip()
    return t[:1].upper() + t[1:] if t else str(text).strip()


def sentence(text: str) -> str:
    t = " ".join(str(text).split())
    return t if not t or t[-1] in ".!?)\"'" else t + "."


def tool_phrase(tool, run) -> str:
    """A tool says its result in its own words with "say"; otherwise the mesh phrases it."""
    say = run.extra.get("say")
    if say:
        return sentence(say)
    out = run.output
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        return f"The answer is {out}."
    return sentence(out) if isinstance(out, str) and " " in out.strip() else f"The {tool.name} tool gives {out}."


def skill_phrase(spec, label: str) -> str:
    """A label as a sentence: the capability's own phrasing ([<skill>.answers] / answer), else a plain one."""
    desc = spec.label_descriptions.get(label, "")
    if spec.answers.get(label):
        return sentence(spec.answers[label].format(label=label, description=desc))
    if spec.answer:
        return sentence(spec.answer.format(label=label, description=desc))
    return sentence(f"I'd call that {label}" + (f": {desc}" if desc else ""))


def failure(note: str) -> str:
    return sentence(f"I couldn't do that: {note}" if note else "I couldn't do that")


def finish(res) -> None:
    """The reply as a person reads it: the result first, then its sources."""
    if not res.ok:
        res.answer = failure(res.note)
        return
    lead = res.answer or (sentence(res.output) if res.output is not None else "")
    if res.sources:
        label = "Source" if len(res.sources) == 1 else "Sources"
        lead = f"{lead}\n{label}: " + "; ".join(res.sources)
    res.answer = lead
