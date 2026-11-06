"""Capabilities (what the system can learn to do), orders and results."""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

DISPATCH = "dispatch"
DIRECT = "answer:direct"


@dataclass
class CapabilitySpec:
    """A learnable skill. Neural nodes are grown per capability.

    lexicon / negation / templates / fillers only drive the offline heuristic teacher;
    an LLM teacher needs just the description and labels.
    """
    name: str
    description: str = ""
    labels: list[str] = field(default_factory=list)
    label_descriptions: dict[str, str] = field(default_factory=dict)
    inactive_labels: list[str] = field(default_factory=list)
    open_labels: bool = True
    default_label: str | None = None
    triggers: list[str] = field(default_factory=list)
    lexicon: dict[str, list[str]] = field(default_factory=dict)
    negation: dict[str, str] = field(default_factory=dict)
    templates: list[str] = field(default_factory=list)
    label_templates: dict[str, list[str]] = field(default_factory=dict)
    fillers: dict[str, list[str]] = field(default_factory=dict)
    kind: str = "classify"
    answer: str = ""         # how a result is said, e.g. "That reads as {label}." ({label}, {description})
    answers: dict[str, str] = field(default_factory=dict)  # per-label phrasing, overriding `answer`
    internal: bool = False   # used by the mesh itself (emotional engines): never a route for orders
    teachers: list[str] = field(default_factory=list)  # only these teachers, in this order ([] = all)
    created_at: float = field(default_factory=time.time)

    def active_labels(self) -> list[str]:
        return [lab for lab in self.labels if lab not in self.inactive_labels]

    def add_label(self, label: str, description: str | None = None) -> bool:
        changed = False
        if label not in self.labels:
            self.labels.append(label)
            changed = True
        if label in self.inactive_labels:
            self.inactive_labels.remove(label)
            changed = True
        if description:
            self.label_descriptions[label] = description
        return changed

    def deactivate(self, label: str) -> None:
        if label in self.labels and label not in self.inactive_labels:
            self.inactive_labels.append(label)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CapabilitySpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Order:
    text: str
    source: str = "user"
    id: str = field(default_factory=lambda: f"ord-{uuid.uuid4().hex[:8]}")
    background: bool = False   # run by a background task: uses background budgets, not a user order
    learn: bool = True         # False for challenge exams: answering teaches the mesh nothing
    created_at: float = field(default_factory=time.time)
    future: Any = None


@dataclass
class Inference:
    label: str | None
    confidence: float
    provenance: str
    node_id: str | None = None
    kind: str = "none"  # node | teacher | node-lowconf | none


@dataclass
class Result:
    order_id: str
    text: str
    route: str | None
    output: Any
    confidence: float
    trace: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    served_by: str = "none"  # nodes | teachers | mixed | none
    ok: bool = True
    note: str = ""
    latency_ms: float = 0.0
    knowledge: str = ""      # "recalled" (answered from retained knowledge) | "researched" | ""
    answer: str = ""         # what a person reads: the result first, in the mesh's own words, then sources
    sources: list = field(default_factory=list)  # where the result came from (shown after the answer)
    disposition: str = ""    # what the temperament did to the answer ("" = nothing; see temperament.py)
    original: Any = None     # the unchanged answer when the temperament changed or held it (operator-only)
    inquiry: dict | None = None  # a question the mesh asks the user about this order ({"id", "question"})

    def to_dict(self) -> dict:
        return asdict(self)

    def public_dict(self) -> dict:
        """What a user may see: everything but the unchanged answer of a held or withheld reply."""
        d = self.to_dict()
        d.pop("original", None)
        if self.disposition in ("brief", "confirm", "defer", "block"):
            d["output"] = None  # the raw result is held back with the answer
        return d

    def summary(self) -> str:
        """The reply, then where it came from: [route | confidence | served by]."""
        out = self.answer or (self.output if self.ok else f"(not fulfilled) {self.note}")
        return f"{out}  [{self.route or '-'} | conf {self.confidence:.2f} | {self.served_by}]"


_SKILL_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.+)$", re.S)
_TOOL_RE = re.compile(r"^\s*tool\s+([\w-]+)\s*:\s*(.+)$", re.S | re.I)


def parse_order(text: str, capabilities: dict[str, CapabilitySpec]) -> tuple[str | None, str]:
    """Explicit routing: 'sentiment: <text>' or 'tool calculator: <text>'. Otherwise the
    dispatch capability decides (and is itself a learned neural node)."""
    m = _TOOL_RE.match(text)
    if m:
        return f"tool:{m.group(1).lower()}", m.group(2).strip()
    m = _SKILL_RE.match(text)
    if m:
        name = m.group(1).lower()
        spec = capabilities.get(name)
        if spec is not None and spec.name != DISPATCH:
            return f"skill:{name}", m.group(2).strip()
    return None, text


def skill_payload(text: str) -> str:
    """For implicitly routed skill orders ('how does this review sound: ...') keep the part
    after the first colon if there is one."""
    if ":" in text:
        head, tail = text.split(":", 1)
        if len(head) < 60 and tail.strip():
            return tail.strip()
    return text
