"""Directive loading.

The directives file (directives.toml) is the system's constitution: immutable prime
directives, measurable goals, and the operating policy every agent must respect.
"""
from __future__ import annotations

import copy
import tomllib
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def build(cls, data: dict | None):
    """Build a dataclass from a dict, ignoring unknown keys and coercing lists/dicts."""
    data = data or {}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            default = None
        if is_dataclass(default) and isinstance(value, dict):
            value = build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        elif isinstance(default, dict) and isinstance(value, dict):
            value = {**default, **value}
        kwargs[f.name] = value
    return cls(**kwargs)


@dataclass(frozen=True)
class PrimeDirective:
    id: str
    text: str


@dataclass(frozen=True)
class Goal:
    id: str
    text: str
    metric: str
    target: float
    direction: str = "at_least"  # or "at_most"

    def met(self, value: float | None) -> bool:
        if value is None:
            return False
        return value >= self.target if self.direction == "at_least" else value <= self.target


@dataclass
class OrderPolicy:
    confidence_threshold: float = 0.75
    audit_sample_rate: float = 0.1
    max_concurrent_orders: int = 16


@dataclass
class NodePolicy:
    max_nodes: int = 40
    max_candidates_per_capability: int = 2
    default_hidden: tuple = (64,)
    default_lr: float = 0.01
    input_dim: int = 1024
    min_examples_to_spawn: int = 12
    serve_min_accuracy: float = 0.8
    target_accuracy: float = 0.95
    promotion_margin: float = 0.01
    promotion_z: float = 1.0
    min_eval_samples: int = 10
    min_train_steps_before_eval: int = 3
    max_candidate_slices: int = 12
    batch_size: int = 32
    max_hidden: int = 512           # widest hidden layer the architect may breed
    max_depth: int = 3              # most hidden layers the architect may breed
    challenger_cooldown_seconds: float = 30.0
    retired_grace_seconds: float = 120.0
    candidate_max_age_seconds: float = 600.0
    champion_update_tolerance: float = 0.0
    holdout_percent: int = 20


@dataclass
class BackgroundPolicy:
    enabled: bool = True
    max_duty_cycle: float = 0.5
    tick_seconds: float = 1.0
    train_epochs_per_slice: int = 8
    synthesis_cooldown_seconds: float = 20.0
    synthesis_batch: int = 8
    min_examples_per_label: int = 8
    health_check_seconds: float = 300.0


@dataclass
class LLMConfig:
    provider: str = "anthropic"  # "anthropic", "ollama", or "openai" (any OpenAI-compatible endpoint)
    enabled: bool = True
    model: str = "claude-haiku-5-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = ""
    anthropic_version: str = "2023-06-01"
    timeout_s: float = 30.0
    max_tokens: int = 400
    temperature: float = 0.2
    max_calls_per_hour: int = 300
    max_tokens_per_hour: int = 200_000
    keep_alive: str = "10m"         # ollama: how long the model stays loaded between calls
    think: object = ""              # ollama: true/false or "low"/"medium"/"high" for reasoning models; "" = model default
    probe_seconds: float = 30.0     # ollama: how often to re-check the server and model

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        provider = str(d.get("provider", "anthropic")).lower()
        return build(cls, {**PROVIDER_DEFAULTS.get(provider, {}), **d, "provider": provider})


PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic": {"model": "claude-haiku-5-5", "api_key_env": "ANTHROPIC_API_KEY", "timeout_s": 30.0},
    "openai": {"model": "", "api_key_env": "OPENAI_API_KEY", "timeout_s": 60.0},
    # local models: no key, generous timeouts and budgets (the cost is your own hardware)
    "ollama": {"model": "auto", "api_key_env": "", "timeout_s": 120.0, "temperature": 0.1,
               "max_calls_per_hour": 3000, "max_tokens_per_hour": 3_000_000},
}


@dataclass
class TeacherPolicy:
    order: tuple = ("human", "ollama", "llm", "heuristic")
    weights: dict = field(default_factory=lambda: {
        "human": 3.0, "tool-manifest": 2.0, "llm": 1.5, "ollama": 1.2, "heuristic": 1.0, "temperament": 1.0})
    models: dict = field(default_factory=dict)  # teacher name -> LLMConfig, from [teachers.<name>]

    @property
    def llm(self) -> LLMConfig | None:
        return self.models.get("llm")

    @classmethod
    def from_dict(cls, d: dict | None) -> "TeacherPolicy":
        """Every [teachers.<name>] table with a `provider` key defines an LLM teacher called
        <name>. `order` is authoritative: a teacher that is not listed there is never asked."""
        d = d or {}
        is_model = lambda k, v: isinstance(v, dict) and ("provider" in v or k == "llm")  # noqa: E731
        policy = build(cls, {k: v for k, v in d.items() if not is_model(k, v) and k != "models"})
        policy.models = {k: LLMConfig.from_dict(v) for k, v in d.items() if is_model(k, v)}
        return policy


@dataclass
class ToolPolicy:
    plugin_dirs: tuple = ("plugins",)
    auto_approve_permissions: tuple = ()
    allowed_permissions: tuple = ("network", "fs_read")
    quarantine_after_failures: int = 3
    scan_seconds: float = 2.0
    self_test_pass_rate: float = 1.0
    sandbox_cpu_seconds: int = 5
    sandbox_memory_mb: int = 512
    default_timeout_s: float = 10.0
    config: dict = field(default_factory=dict)  # [tools.config.<tool>] overrides of manifest [config]


@dataclass
class ResearchPolicy:
    enabled: bool = True                # research tools (wikipedia) may be used at all
    recall_threshold: float = 0.8       # how well retained knowledge must match to answer from memory
    max_fetches_per_hour: int = 60      # every research call: orders and background together
    background_per_hour: int = 20       # of which background curiosity and refreshing
    follow_links: int = 2               # after learning an article for an order, also read N linked ones
    max_depth: int = 1                  # how far curiosity may wander from what was asked
    refresh_days: float = 30.0          # re-read retained articles older than this when idle
    max_documents: int = 2000           # retention cap; the least recently used are forgotten first
    research_questions: bool = True     # research unroutable questions ("who/what/...") on Wikipedia


@dataclass
class TemperamentPolicy:
    """The emotional engines (P8): how the mesh feels, and how far that may shape answers."""
    enabled: bool = True
    # the personality the mood relaxes back to (valence -1..1, the rest 0..1)
    baseline: dict = field(default_factory=lambda: {
        "valence": 0.2, "arousal": 0.3, "confidence": 0.6, "trust": 0.6, "patience": 0.7})
    half_life_seconds: float = 600.0    # how fast feelings fade back to the baseline
    sensitivity: float = 1.0            # scales every emotional reaction (0 = unflappable)
    low_confidence: float = 0.6         # answers below this confidence count as uncertain
    long_answer_chars: int = 400        # answers this long count as long
    brief_chars: int = 220              # how much of a long answer "brief" shows
    defer_seconds: float = 20.0         # how long a deferred answer waits before /release works


@dataclass
class MindPolicy:
    """The mesh's private mind (P9, P10): opinions and notes, questions for the user,
    capability proposals, and investigations it starts on its own."""
    enabled: bool = True
    share_trust: float = 0.5                    # opinions are shared only while trust is at least this
    private_subjects: tuple = ("user", "teacher")  # opinions about these are always private
    reflect_seconds: float = 20.0               # how often opinions are re-formed and proposals considered
    min_evidence: int = 3                       # unmet orders with a common theme before proposing a tool or skill
    min_skill_requests: int = 2                 # orders naming a skill that does not exist before proposing it
    min_opinion_evidence: int = 3               # observations before an opinion is formed
    max_open_inquiries: int = 5                 # questions waiting for the user at once
    max_inquiries_per_hour: int = 6             # so the mesh does not nag
    ask_interests_after_orders: int = 8         # ask what to read up on after this many orders (0 = never)
    investigate: bool = True                    # investigate goals on its own (background, own budget)
    investigations_per_hour: int = 30
    retry_unmet_after_seconds: float = 120.0    # look into an unfulfilled order again after this long
    max_unmet_attempts: int = 2
    retest_quarantined_after_seconds: float = 900.0  # re-run self-tests of a quarantined tool (0 = never)
    max_notes: int = 2000


@dataclass
class ComputePolicy:
    device: str = "cpu"     # "cpu" | "auto" | "cuda" | "cuda:N" | "mps"
    library: str = "auto"   # "auto" | "numpy" | "torch" | "cupy"


@dataclass
class Directives:
    name: str = "NeuroMesh"
    mission: str = ""
    prime: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    compute: ComputePolicy = field(default_factory=ComputePolicy)
    research: ResearchPolicy = field(default_factory=ResearchPolicy)
    orders: OrderPolicy = field(default_factory=OrderPolicy)
    nodes: NodePolicy = field(default_factory=NodePolicy)
    background: BackgroundPolicy = field(default_factory=BackgroundPolicy)
    teachers: TeacherPolicy = field(default_factory=TeacherPolicy)
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    temperament: TemperamentPolicy = field(default_factory=TemperamentPolicy)
    mind: MindPolicy = field(default_factory=MindPolicy)

    @classmethod
    def from_dict(cls, d: dict) -> "Directives":
        ident = d.get("identity", {})
        return cls(
            name=ident.get("name", "NeuroMesh"),
            mission=ident.get("mission", ""),
            prime=[PrimeDirective(**p) for p in d.get("prime", [])],
            goals=[Goal(**g) for g in d.get("goals", [])],
            compute=build(ComputePolicy, d.get("compute")),
            research=build(ResearchPolicy, d.get("research")),
            orders=build(OrderPolicy, d.get("orders")),
            nodes=build(NodePolicy, d.get("nodes")),
            background=build(BackgroundPolicy, d.get("background")),
            teachers=TeacherPolicy.from_dict(d.get("teachers")),
            tools=build(ToolPolicy, d.get("tools")),
            temperament=build(TemperamentPolicy, d.get("temperament")),
            mind=build(MindPolicy, d.get("mind")),
        )

    @classmethod
    def load(cls, path: Path, overrides: dict | None = None) -> "Directives":
        data = load_toml(path)
        if overrides:
            data = deep_merge(data, overrides)
        return cls.from_dict(data)
