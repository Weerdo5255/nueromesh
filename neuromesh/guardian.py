"""The Guardian: the policy engine that enforces the prime directives.

Every consequential action (spawning a node, promoting it, installing or running a
tool, calling an LLM teacher, doing background work) is authorised here first, and
every decision that is not a plain "allow" is written to the audit log.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .bus import Bus
from .config import Directives
from .store import Store


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class Decision:
    verdict: Verdict
    reason: str = ""
    directive: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW


@dataclass
class Approval:
    id: str
    kind: str
    subject: str
    detail: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"
    decided_by: str | None = None
    decided_at: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


ALLOW = Decision(Verdict.ALLOW)


class Guardian:
    def __init__(self, directives: Directives, store: Store, bus: Bus, state_dir: Path):
        self.d = directives
        self.store = store
        self.bus = bus
        self.stop_file = Path(state_dir) / "STOP"
        self.halted = False
        self.halt_reason = ""
        self.paused = False
        self.approvals: dict[str, Approval] = {}
        self._approval_ids = itertools.count(1)
        self._llm: defaultdict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._research: deque[tuple[float, bool]] = deque()
        self._research_unlimited: deque[float] = deque()   # lookups on exempt servers: counted, never limited
        self._investigations: deque[float] = deque()
        self._task_runs: deque[float] = deque()
        self._busy: deque[tuple[float, float]] = deque()
        self.decisions: Counter = Counter()

    # ----------------------------------------------------------------- audit
    def audit(self, actor: str, kind: str, subject: str = "", /, **detail) -> None:
        self.store.audit(actor, kind, subject, detail)
        self.bus.publish("audit", actor=actor, kind=kind, subject=subject, detail=detail)

    def _deny(self, kind: str, subject: str, reason: str, directive: str) -> Decision:
        self.decisions["deny"] += 1
        self.audit("guardian", f"deny.{kind}", subject, reason=reason, directive=directive)
        return Decision(Verdict.DENY, reason, directive)

    # ----------------------------------------- P1: human authority / kill switch
    def check_kill_switch(self) -> None:
        if self.stop_file.exists() and not self.halted:
            self.halt(f"kill switch file present ({self.stop_file.name})", by="kill-switch")

    def halt(self, reason: str, by: str = "operator") -> None:
        self.halted = True
        self.halt_reason = reason
        self.audit(by, "system.halt", reason=reason)
        self.bus.publish("system.halt", reason=reason, by=by)

    def resume(self, by: str = "operator") -> tuple[bool, str]:
        if self.stop_file.exists():
            return False, f"remove {self.stop_file} first"
        self.halted = False
        self.paused = False
        self.halt_reason = ""
        self.audit(by, "system.resume")
        self.bus.publish("system.resume", by=by)
        return True, "resumed"

    def pause_background(self, paused: bool, by: str = "operator") -> None:
        self.paused = paused
        self.audit(by, "background.pause" if paused else "background.unpause")
        self.bus.publish("system.background", paused=paused)

    # ---------------------------------------------- P2: orders before background
    def note_background_busy(self, seconds: float) -> None:
        self._busy.append((time.monotonic(), seconds))

    def duty_cycle(self, window: float = 30.0) -> float:
        now = time.monotonic()
        while self._busy and now - self._busy[0][0] > window:
            self._busy.popleft()
        return sum(s for _, s in self._busy) / window

    def background_allowed(self) -> bool:
        return (self.d.background.enabled and not self.halted and not self.paused
                and self.duty_cycle() < self.d.background.max_duty_cycle)

    # ------------------------------------------------- P3: approved tools only
    def authorize_tool(self, manifest) -> Decision:
        perms = set(manifest.permissions)
        policy = self.d.tools
        forbidden = perms - set(policy.allowed_permissions)
        if forbidden:
            return self._deny("tool", manifest.name,
                              f"requests forbidden permissions {sorted(forbidden)}", "P3-approved-tools-only")
        if perms <= set(policy.auto_approve_permissions):
            self.decisions["allow"] += 1
            self.audit("guardian", "allow.tool", manifest.name, permissions=sorted(perms))
            return ALLOW
        self.decisions["needs_approval"] += 1
        return Decision(Verdict.NEEDS_APPROVAL,
                        f"permissions {sorted(perms)} need human approval", "P3-approved-tools-only")

    def authorize_tool_run(self, tool) -> Decision:
        if self.halted:
            return self._deny("tool_run", getattr(tool, "name", "?"), "system halted", "P1-human-authority")
        if tool is None:
            return Decision(Verdict.DENY, "unknown tool", "P3-approved-tools-only")
        if tool.state != "active":
            return self._deny("tool_run", tool.name, f"tool is {tool.state}", "P3-approved-tools-only")
        return ALLOW

    # -------------------------------------------------------------- P4: budgets
    def authorize_spawn(self, live_nodes: int, candidates_for_cap: int, capability: str) -> Decision:
        n = self.d.nodes
        if self.halted:
            return Decision(Verdict.DENY, "system halted", "P1-human-authority")
        if live_nodes >= n.max_nodes:
            return self._deny("spawn", capability, f"node budget reached ({n.max_nodes})", "P4-budgets")
        if candidates_for_cap >= n.max_candidates_per_capability:
            return Decision(Verdict.DENY, "candidate slots full", "P4-budgets")
        return ALLOW

    def authorize_llm_call(self, teacher: str, cfg, est_tokens: int) -> Decision:
        """Each LLM teacher has its own hourly call and token budget (from its config)."""
        calls = self._llm[teacher]
        now = time.time()
        while calls and now - calls[0][0] > 3600:
            calls.popleft()
        if self.halted:
            return Decision(Verdict.DENY, "system halted", "P1-human-authority")
        if len(calls) >= cfg.max_calls_per_hour:
            return self._deny("llm_call", teacher, "hourly call budget exhausted", "P4-budgets")
        if sum(t for _, t in calls) + est_tokens > cfg.max_tokens_per_hour:
            return self._deny("llm_call", teacher, "hourly token budget exhausted", "P4-budgets")
        return ALLOW

    def authorize_research(self, background: bool, unlimited: bool = False) -> Decision:
        """Research calls (e.g. Wikipedia) have an hourly budget, with a smaller share for
        background curiosity so orders always have room. Tools the directives list in
        [research] unlimited (your own Kiwix server) are exempt."""
        pol = self.d.research
        now = time.time()
        while self._research and now - self._research[0][0] > 3600:
            self._research.popleft()
        if self.halted:
            return Decision(Verdict.DENY, "system halted", "P1-human-authority")
        if not pol.enabled:
            return Decision(Verdict.DENY, "research is disabled in the directives", "P3-approved-tools-only")
        if unlimited:
            return ALLOW
        if len(self._research) >= pol.max_fetches_per_hour:
            reason = f"hourly research budget used up ({pol.max_fetches_per_hour}/h)"
            if background:
                return Decision(Verdict.DENY, reason, "P4-budgets")
            return self._deny("research", "", reason, "P4-budgets")
        if background and sum(1 for _, bg in self._research if bg) >= pol.background_per_hour:
            return Decision(Verdict.DENY, f"background research budget used up ({pol.background_per_hour}/h)",
                            "P4-budgets")
        return ALLOW

    def authorize_investigation(self) -> Decision:
        """Investigations the mesh starts on its own have an hourly budget (P4)."""
        now = time.time()
        while self._investigations and now - self._investigations[0] > 3600:
            self._investigations.popleft()
        if self.halted:
            return Decision(Verdict.DENY, "system halted", "P1-human-authority")
        if len(self._investigations) >= self.d.mind.investigations_per_hour:
            return Decision(Verdict.DENY, f"hourly investigation budget used up "
                            f"({self.d.mind.investigations_per_hour}/h)", "P4-budgets")
        return ALLOW

    def record_investigation(self) -> None:
        self._investigations.append(time.time())

    def investigation_usage(self) -> dict:
        now = time.time()
        return {"last_hour": sum(1 for ts in self._investigations if now - ts <= 3600),
                "limit": self.d.mind.investigations_per_hour}

    def authorize_task_run(self) -> Decision:
        """Background tasks people set have an hourly budget of runs (P4)."""
        now = time.time()
        while self._task_runs and now - self._task_runs[0] > 3600:
            self._task_runs.popleft()
        if self.halted:
            return Decision(Verdict.DENY, "system halted", "P1-human-authority")
        if len(self._task_runs) >= self.d.tasks.max_runs_per_hour:
            return Decision(Verdict.DENY, f"hourly background-task budget used up "
                            f"({self.d.tasks.max_runs_per_hour}/h)", "P4-budgets")
        return ALLOW

    def record_task_run(self) -> None:
        self._task_runs.append(time.time())

    def task_usage(self) -> dict:
        now = time.time()
        return {"last_hour": sum(1 for ts in self._task_runs if now - ts <= 3600),
                "limit": self.d.tasks.max_runs_per_hour}

    def record_research(self, background: bool, unlimited: bool = False) -> None:
        if unlimited:
            self._research_unlimited.append(time.time())
            while self._research_unlimited and time.time() - self._research_unlimited[0] > 3600:
                self._research_unlimited.popleft()
            return
        self._research.append((time.time(), background))

    def research_usage(self) -> dict:
        now = time.time()
        recent = [bg for ts, bg in self._research if now - ts <= 3600]
        pol = self.d.research
        return {"last_hour": len(recent), "background_last_hour": sum(recent),
                "limit": pol.max_fetches_per_hour, "background_limit": pol.background_per_hour,
                "unlimited_last_hour": sum(1 for ts in self._research_unlimited if now - ts <= 3600)}

    def record_llm_usage(self, teacher: str, tokens: int) -> None:
        self._llm[teacher].append((time.time(), int(tokens)))

    def llm_usage(self) -> dict:
        now = time.time()
        out = {}
        for teacher, calls in self._llm.items():
            recent = [t for ts, t in calls if now - ts <= 3600]
            out[teacher] = {"calls_last_hour": len(recent), "tokens_last_hour": sum(recent)}
        return out

    # ------------------------------------------------------- P5: no regression
    def authorize_promotion(self, capability: str, champion_acc: float | None,
                            candidate_acc: float | None) -> Decision:
        if self.halted:
            return Decision(Verdict.DENY, "system halted", "P1-human-authority")
        if candidate_acc is None:
            return Decision(Verdict.DENY, "candidate not evaluated", "P5-no-regression")
        if champion_acc is not None and candidate_acc < champion_acc:
            return self._deny("promotion", capability,
                              f"candidate {candidate_acc:.3f} < champion {champion_acc:.3f}", "P5-no-regression")
        return ALLOW

    def authorize_champion_update(self, capability: str, old_acc: float | None,
                                  new_acc: float | None) -> Decision:
        if old_acc is None or new_acc is None:
            return ALLOW
        if new_acc + self.d.nodes.champion_update_tolerance < old_acc:
            self.decisions["deny"] += 1
            return Decision(Verdict.DENY, f"update would regress {old_acc:.3f} -> {new_acc:.3f}",
                            "P5-no-regression")
        return ALLOW

    # -------------------------------------------------------------- approvals
    def request_approval(self, kind: str, subject: str, detail: dict) -> Approval:
        approval = Approval(f"apv-{next(self._approval_ids)}", kind, subject, detail)
        self.approvals[approval.id] = approval
        self.audit("guardian", "approval.requested", subject, approval=approval.id, approval_kind=kind,
                   **{k: v for k, v in detail.items() if k not in ("actor", "kind", "subject")})
        self.bus.publish("approval.requested", id=approval.id, kind=kind, subject=subject, detail=detail)
        return approval

    def decide(self, approval_id: str, approve: bool, by: str = "human", note: str = "") -> Approval | None:
        approval = self.approvals.get(approval_id)
        if approval is None or approval.status != "pending":
            return None
        approval.status = "approved" if approve else "denied"
        approval.decided_by, approval.decided_at, approval.note = by, time.time(), note
        self.audit(by, f"approval.{approval.status}", approval.subject, approval=approval_id, note=note)
        self.bus.publish("approval.decided", id=approval_id, status=approval.status,
                         subject=approval.subject, by=by)
        return approval

    def pending_approvals(self) -> list[Approval]:
        return [a for a in self.approvals.values() if a.status == "pending"]
