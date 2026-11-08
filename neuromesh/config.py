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
    max_running: int = 8            # node instances that may run at the same time, across the mesh (ensemble.py)
    instances: int = 3              # nodes of one kind that answer together: the champion and its instances
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
    max_depth: int = 4              # most hidden layers the architect may breed
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
    vision_model: str = "auto"      # ollama: a model that can see images ("auto": an installed one, e.g. llava; "": none)

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
    order: tuple = ("human", "challenge", "ollama", "llm", "heuristic")
    weights: dict = field(default_factory=lambda: {
        "human": 3.0, "challenge": 2.5, "tool-manifest": 2.0, "outcome": 2.0, "llm": 1.5, "ollama": 1.2,
        "heuristic": 1.0, "temperament": 1.0, "agency": 1.0, "director": 1.8,
        "coder": 1.0, "experiment": 2.0, "approved": 2.5, "planner": 1.0, "ego": 1.0, "toolgate": 1.0,
        "gitea": 1.0, "vision": 1.3, "sight": 1.0, "glyphs": 1.0, "tesseract": 1.5, "tutor": 1.2})
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
    mode: str = "on"                  # the tool switch: "on", "off", or "decide" (it decides each time; toolgate.py)
    plugin_dirs: tuple = ("plugins",)
    auto_approve_permissions: tuple = ()
    allowed_permissions: tuple = ("network", "fs_read")
    quarantine_after_failures: int = 3
    scan_seconds: float = 2.0
    self_test_pass_rate: float = 1.0
    sandbox_cpu_seconds: int = 5
    sandbox_memory_mb: int = 512
    default_timeout_s: float = 10.0
    workers: bool = True              # tools that can stay running (persistent = true: the Kiwix tool) do
    worker_cpu_seconds: int = 600     # a worker's CPU budget, for many requests (then a fresh one starts)
    worker_requests: int = 5000       # a worker is replaced after this many requests
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
    prefer: tuple = ("kiwix", "wikipedia")  # research tools in order: the first that has the article wins
    unlimited: tuple = ("kiwix",)       # research tools exempt from the budgets above (your own server)
    kiwix_attempts: int = 5             # Kiwix searches for one question (books, wordings, title and full text)
                                        # before it gives up and asks you (1 = a single look-up, as before)
    kiwix_coverage: float = 0.6         # how much of a question an article must cover to count as its answer
    catalog_seconds: float = 600.0      # how long the list of Kiwix books is kept before it is read again
    kiwix_parallel: int = 4             # Kiwix searches run at once for one question (then the first that answers wins)
    browse: bool = True                 # now and then it reads a Kiwix book it has not opened in a long time
    browse_every_seconds: float = 300.0 # how often it may pick such a book (in the background, within budgets)
    browse_stale_days: float = 7.0      # a book unopened for this long counts as "not looked at in a long time"
    browse_articles: int = 2            # random articles it reads from the book each time


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
class TaskPolicy:
    """Background tasks people set: learn a topic, improve a skill, run an order on an interval.
    They only use background capacity (P2) and have their own budget (P4)."""
    enabled: bool = True
    max_tasks: int = 20
    max_runs_per_hour: int = 240           # all task runs together
    min_interval_seconds: float = 30.0     # the shortest interval a task may ask for
    learn_interval_seconds: float = 60.0   # default pace of a learn task (one article per run)
    learn_max_articles: int = 25           # how much a learn task reads before it switches to reviewing
    learn_depth: int = 2                   # how far from the topic it follows links
    learn_links_per_article: int = 3
    learn_burst_unlimited: int = 10        # articles per run when the research tool is your own server (kiwix)
    improve_interval_seconds: float = 30.0
    challenge_interval_seconds: float = 1800.0  # default pace of a challenge task (practice, then the exam)
    pause_after_failures: int = 5          # a run task that keeps failing is paused


@dataclass
class ChallengePolicy:
    """Open-source LLM benchmarks as a teacher: practice items teach, exam items measure."""
    enabled: bool = True
    folders: tuple = ("challenges",)        # suite folders (relative to the root); fetched suites go to state/challenges
    exam_percent: int = 30                  # share of each suite held out as the exam (never learned from)
    practice_batch: int = 20                # practice items per run (rotating through the suite)
    max_route_trials: int = 4               # other routes tried on a question the mesh missed
    close_tolerance: float = 0.01           # numeric answers within 1% count as close
    api_url: str = "https://datasets-server.huggingface.co"  # where /challenge fetch downloads benchmarks
    fetch_limit: int = 500                  # rows fetched per benchmark by default
    token_env: str = "HF_TOKEN"             # environment variable with a Hugging Face token (optional)


@dataclass
class EvolutionPolicy:
    """The genetic algorithm that breeds each capability's challengers (see evolution.py)."""
    enabled: bool = True
    growth_bias: float = 0.7          # chance a mutation grows (neurons or a layer) rather than shrinks
    layer_rate: float = 0.3           # chance a mutation changes the number of layers rather than neurons
    double_mutation: float = 0.3      # chance of a second mutation in the same child
    crossover_rate: float = 0.25      # chance a child recombines its parent with another good genome
    lr_rate: float = 0.2              # chance the learning rate mutates too
    tournament: int = 2               # parents are the fittest of this many live nodes
    complexity_weight: float = 0.004  # fitness bonus per doubling of size: equal accuracy -> the larger is fitter
    promote_larger_on_tie: bool = True  # a larger challenger at least as accurate as the champion replaces it
    keep_evolving: bool = True        # keep breeding (more slowly) once the champion meets its target
    idle_cooldown_factor: float = 4.0 # how much more slowly, then
    min_neurons: int = 8
    max_params: int = 1_500_000       # largest node it may breed (with [nodes] max_hidden and max_depth)
    archive_size: int = 40            # genomes remembered per capability


@dataclass
class MemoryPolicy:
    """Memory nodes: what the mesh learns, compressed into small neural networks, each checked by its
    tests in the memory's own database (state/memory.db, see memory.py and memdb.py)."""
    enabled: bool = True
    learn_from: tuple = ("knowledge", "teachers", "people", "challenges")
    serve: bool = True                # answer questions from memory nodes before anything else
    answer_threshold: float = 0.6     # confidence a memory node needs to answer
    test_margin: float = 0.1          # a test passes only with this margin: >= 0.7 for an answer, < 0.5 to not answer
    database: str = "state/memory.db" # the memory's own database (its tests, its own statements), from the root
    tests_dir: str = "memory_tests"   # test files of earlier versions: imported into the database once, then moved aside
    own: bool = True                  # keep its opinions and its ego's statements in memory too (untested)
    own_per_node: int = 40            # statements an own-memory node holds
    facts_per_node: int = 40
    start_hidden: tuple = (64,)       # the first shape a node learns in (then it is compressed)
    max_hidden: int = 256             # how far a node may grow when it cannot hold its facts
    max_growth: int = 3
    compress: bool = True
    min_fold: int = 32                # fewest input features a node may fold down to
    min_neurons: int = 4
    min_bits: int = 8                 # 32, 16 or 8 bits per weight
    learn_epochs: int = 400
    compress_epochs: int = 250
    learning_rate: float = 0.01
    unknown_tests: int = 30           # other nodes' questions in each test file as [[unknown]]
    probe_tests: int = 6              # tool / skill / small-talk questions as [[unknown]]
    near_miss_tests: int = 1          # near misses of each of its questions as [[unknown]]
    interval_seconds: float = 0.5
    sync_seconds: float = 2.0         # how often the tests are checked for people's edits
    max_answer_chars: int = 600
    merge: bool = True                # go back through compressed memory nodes and merge pairs, chosen at random
    merge_pass_rate: float = 0.9      # a merged node replaces both when it passes this share of their tests
    merge_every_seconds: float = 30.0 # how often it tries a merge when there is nothing else to do
    merge_max_facts: int = 200        # largest merged node
    merge_epochs: int = 300


@dataclass
class AgencyPolicy:
    """Independent agency: the Director and the agency node in front of it (see agency.py)."""
    enabled: bool = True
    decide_every_seconds: float = 120.0   # how often it considers taking initiative
    idle_seconds: float = 15.0            # quiet time (no orders from people) before it does
    max_decisions_per_hour: int = 12      # its own budget (P4)
    max_self_tasks: int = 3               # tasks it has set itself at once
    explore: float = 0.1                  # chance of trying another possible mode, to learn what it is worth
    modes: tuple = ("research", "practise", "examine", "improve", "consolidate", "build", "experiment", "pursue",
                    "ruminate", "rest")
    max_self_objectives: int = 4          # objectives it has set itself at once
    research_articles: int = 3            # a self-set learn task reads this many articles
    practice_items: int = 20
    max_task_runs: int = 5                # a self-set task is scored and retired after this many runs
    improve_window_seconds: float = 600.0 # how long a self-set improve task gets before it is scored
    task_max_age_seconds: float = 1800.0
    verify_after_seconds: float = 3600.0  # memory nodes count as unverified after this long
    reexam_after_seconds: float = 21600.0 # a suite's exam is due again after this long
    low_score: float = 0.8                # exam scores below this make practice worthwhile
    good_value: float = 0.5               # an outcome this good reinforces the choice
    poor_value: float = 0.15              # an outcome this poor teaches the policy's next best instead
    retry_after_seconds: float = 1800.0   # a target that did not pay off rests this long before it is chosen again


@dataclass
class DreamsPolicy:
    """Sleep and dreams: dream nodes that make up questions and answers from what the mesh already
    knows, checked on waking, evolved by a genetic algorithm, and used to imagine (see dreams.py)."""
    enabled: bool = True
    sleep_after_idle_seconds: float = 60.0  # quiet time (no orders from people) before the idle coin flip
    objective_chance: float = 0.75          # idle coin flip: the chance it works on an objective; otherwise it sleeps
    sleep_chance: float = 1.0               # when the flip lands on sleep, the chance it does fall asleep then
    allow_sleep: bool = True                # false: it never sleeps (a person can toggle it: /sleep off, /sleep on)
    check_every_seconds: float = 30.0       # how often it tosses the coin while idle
    sleep_max_seconds: float = 120.0        # longest sleep; an order always wakes it at once
    duty_cycle: float = 0.9                 # the background capacity dreaming may use while asleep
    dreams_per_step: int = 24
    dreams_per_night: int = 600
    rehearse_share: float = 0.6             # remembered dreams it rehearses on; it answers the rest itself
    imagine_share: float = 0.3              # dreams that swap in a name from something else it knows
    rehearse_epochs: int = 25
    min_material: int = 5                   # things it must remember before it can dream
    material_passages: int = 60             # retained passages sampled for a night's dreams
    material_max: int = 400
    vocabulary: int = 400                   # most answers a dream node can give
    inputs: int = 512
    hidden: tuple = (64,)
    max_hidden: int = 256
    learning_rate: float = 0.01
    population: int = 4                     # dream nodes kept by the genetic algorithm
    children: int = 3                       # children bred on each waking
    ga_epochs: int = 60
    keep_dreams: int = 1500                 # checked dreams remembered
    imagine_threshold: float = 0.2          # lowest confidence for an idea it imagines


@dataclass
class ComputePolicy:
    device: str = "cpu"     # "cpu" | "auto" | "cuda" | "cuda:N" | "mps"
    library: str = "auto"   # "auto" | "numpy" | "torch" | "cupy"


@dataclass
class DirectorsPolicy:
    """The directors: at least one director node at both ends of every order, duplicating itself when
    it needs to, its stance shaped by the mood (see directors.py)."""
    max_directors: int = 4                  # director-0 and its duplicates (P4); never fewer than one
    spawn_after_waiting: int = 2            # pieces of work waiting for a director before it duplicates itself
    retire_after_idle_seconds: float = 30.0 # a duplicate idle this long retires
    beat_seconds: float = 0.5
    consult: bool = True                    # ask a duplicate for a second opinion when unsure
    consult_wait_seconds: float = 0.25      # how long a review waits for a free director to consult
    second_opinion_threshold: float = 0.35  # a second opinion also weighs what memory nodes hold less surely
    review_confidence: float = 0.6          # answers below this get a second opinion (the mood moves it)
    memory_confidence: float = 0.35         # facts from orders below this confidence are not stored (the mood moves it)
    wary_trust: float = 0.35                # below this trust it is wary: a language model's answer needs corroboration
    impatient_patience: float = 0.25        # below this patience it skips second opinions (unless wary)
    mood_influence: float = 1.0             # how much the mood shapes the directors (0 = not at all)
    collaborate: bool = True                # a duplicate may take a second mode of initiative alongside director-0
    second_mode_min: float = 0.55           # the drive a second mode needs
    max_attempts: int = 3                   # ways a director tries an unanswered order before it gives up (1 = no retries)
    tool_queries: int = 3                   # times one tool run may ask a director for context
    query_outside: bool = True              # a director may research or ask a teacher to answer a tool (within budgets)


@dataclass
class ToolsmithPolicy:
    """The toolsmith: coder nodes that write new tools from the code library (or with an LLM teacher),
    ask permission, and experiment with what a person approves (see toolsmith.py, P15)."""
    enabled: bool = True
    min_needs: int = 1                      # requests no tool could meet before it writes a tool for them
    need_confidence: float = 0.55           # how sure the coder node (or its cues) must be that a recipe fits
    builds_per_hour: int = 4                # tools it may write in an hour (P4)
    max_waiting: int = 3                    # tools waiting for approval at once; it writes no more until they are decided
    experiments_per_hour: int = 6           # experiments with its tools in an hour (P4)
    experiment_probes: int = 8              # fresh probes per experiment (as well as the requests it was made for)
    proven_score: float = 0.8               # an experiment this good proves a tool; below it, it refines the tool
    retry_after_seconds: float = 3600.0     # a recipe a person declined is not offered again for this long
    llm_code: bool = True                   # an LLM teacher may write tools no recipe fits (still asked for, still checked)


@dataclass
class ObjectivesPolicy:
    """Objectives: long-term aims it plans for, pursues by asking itself, ruminates over and writes up
    (see objectives.py, P16)."""
    enabled: bool = True
    max_objectives: int = 8                 # open at once (active or paused)
    steps_per_hour: int = 120               # queries against itself, checks and readings, for all objectives (P4)
    step_seconds: float = 2.0               # rest between two steps of one objective
    max_questions: int = 24                 # questions one objective may have, follow-ups included
    max_rounds: int = 4                     # rounds of rumination that may add follow-ups
    checks_per_round: int = 4               # answers checked against a second source per round
    follow_ups: int = 3                     # new questions a round may raise (explain objectives)
    confirm: bool = True                    # check each answer against a second source
    met_share: float = 1.0                  # share of core questions answered to call it met
    ask_people: bool = True                 # ask a person what nothing it has could answer (one question at a time)
    person_wait_seconds: float = 86400.0    # how long a question to a person stays open before it moves on
    watch_every_seconds: float = 3600.0     # how often a watch objective's background task looks
    llm_plan: bool = True                   # an LLM teacher may draft an objective's questions
    llm_summary: bool = True                # an LLM teacher may write the summary of what it found
    second_thoughts: int = 3                # recent answers it thinks over again in one go (agency mode ruminate)
    doubt_confidence: float = 0.6           # answers less sure than this are worth second thoughts
    second_thought_after_seconds: float = 120.0  # how long after an order before it thinks it over again


@dataclass
class ThinkingPolicy:
    """Thinking nodes: after the memory nodes, its thoughts in its own words, with its opinions and desires
    (see thinking.py, P17)."""
    enabled: bool = True
    express: bool = True                    # add its thought after an answer from what it knows, when it adds something
    form_seconds: float = 10.0              # how often it re-forms its thoughts from what its memory nodes hold
    min_facts: int = 2                      # facts a topic needs for a thought (fewer if it wants to know more about it)
    max_thoughts: int = 200
    max_grounds: int = 12                   # facts one thought rests on
    max_claims: int = 4                     # facts it puts together in one understanding
    thoughts_per_node: int = 24
    llm: bool = True                        # an LLM teacher may draft the understanding (checked against its facts)
    threshold: float = 0.6                  # confidence a thinking node needs to bring up a thought
    test_margin: float = 0.1
    start_hidden: list = field(default_factory=lambda: [48])
    max_hidden: int = 256
    epochs: int = 400
    probe_tests: int = 6                    # small talk, tools and skills its tests say it must not think about
    max_negatives: int = 300
    min_feelings: int = 2                   # facts read by the sentiment skill before it forms a favourable view
    interested_after: int = 3               # times people ask about a topic before it says it is interested
    desire_interest: float = 0.4            # interest at which it wants to know more about a topic it knows little of
    enough_facts: int = 5
    show: int = 40


@dataclass
class CuriosityPolicy:
    """Curiosity: areas of interest and gaps in what it knows; the objectives the director may set itself
    (see curiosity.py, P12, P16)."""
    enabled: bool = True
    interest_weight: float = 0.45
    lack_weight: float = 0.45
    both_weight: float = 0.4                # an area it is interested in and lacks information on matters most
    min_interest: float = 0.3               # interest an area needs before "understand <area>" is an aim
    min_lack: float = 0.25                  # what an area must lack (with no interest) before it is an aim
    enough_facts: int = 5                   # facts about an area it is interested in that are enough for now
    documents: int = 50                     # recent articles it looks at
    links_per_document: int = 5
    cache_seconds: float = 5.0
    show: int = 20


@dataclass
class AvatarPolicy:
    """The avatar: an optional face for the mesh in the web interface, its mind at a glance, and a voice
    from a ComfyUI text-to-speech workflow (see avatar.py)."""
    enabled: bool = True                    # the web interface offers it (each viewer shows or hides it)
    comfyui_url: str = ""                   # your ComfyUI server, e.g. "http://127.0.0.1:8188" (empty: no voice)
    tts_workflow: str = ""                  # a text-to-speech workflow exported from ComfyUI with Save (API Format)
    text_placeholder: str = "{{text}}"      # where the words go in that workflow
    timeout_seconds: float = 120.0          # how long one sentence may take to speak
    max_chars: int = 600                    # longer texts are cut at a sentence
    cache_files: int = 60                   # spoken audio kept in state/avatar (the same words are not spoken twice)
    use_proxy: bool = False                 # talk to ComfyUI directly, even if HTTP(S)_PROXY is set


@dataclass
class EgoPolicy:
    """The ego: the nodes wrapped around all the others. It reads the mood as an order comes in, and on the
    way out says what it makes of things - from its curiosity, dreams and temperament - labelled (see ego.py, P18)."""
    enabled: bool = True
    express: bool = True                    # its remarks after answers (questions about itself are answered either way)
    quiet_for: int = 1                      # answers it stays quiet after a remark
    min_confidence: float = 0.5             # below this an answer counts as unsure (doubt)
    fascinate: bool = True                  # what it says it is curious about becomes an area it wants to understand
    fascination_weight: float = 0.35        # interest its ego adds to such an area (curiosity.py; the director's aims)
    fascination_days: float = 7.0           # how long that lasts
    keep_remarks: int = 60


@dataclass
class GiteaPolicy:
    """A local Gitea its coder nodes learn to program from (gitea.py): it studies the code in its repositories."""
    enabled: bool = True
    url: str = ""                       # e.g. "http://localhost:3000" (or --gitea URL, or NEUROMESH_GITEA_URL); "" = none
    token_env: str = "GITEA_TOKEN"      # an access token (read-only is enough) for private repositories
    repos: tuple = ()                   # "owner/name" to study; empty = every repository it can see
    extensions: tuple = (".py", ".js", ".mjs", ".ts", ".go", ".rs", ".sh", ".rb", ".java", ".c", ".cpp")
    max_file_kb: int = 256              # larger files are skipped
    files_per_visit: int = 40           # files it reads per visit to a repository
    parallel: int = 4                   # files fetched at once
    study_every_seconds: float = 300.0  # how often it visits a repository (when idle)
    restudy_days: float = 7.0           # a repository it studied is visited again after this long
    max_examples: int = 20000           # functions it keeps as examples (the oldest go first)
    timeout_s: float = 15.0
    practice: bool = True               # when idle, it practises: writes code for an example's purpose without
    practice_every_seconds: float = 600.0  # looking at it, checks it, and compares it with what a person wrote


@dataclass
class CodePolicy:
    """The scratch pad (scratchpad.py): where it writes code, a validator checks it, and it runs only when a person
    approves exactly that code (P20)."""
    enabled: bool = True
    run_languages: tuple = ("python", "javascript")  # what may run once approved (javascript needs Node.js)
    run_timeout_s: float = 20.0
    run_cpu_seconds: int = 10
    run_memory_mb: int = 512
    max_output_kb: int = 64
    recipe_confidence: float = 0.6      # the coder node's recipe is used when it is at least this sure
    examples: int = 3                   # Gitea examples given to a teacher (or adapted) for a request
    teacher_fix: bool = True            # code that does not check out goes back to the teacher once, with the problems
    keep_items: int = 200
    keep_practice: int = 60


@dataclass
class VisionPolicy:
    """Image nodes (vision.py): it takes in images, says what it sees, reads the text in them, and learns from both."""
    enabled: bool = True
    max_image_mb: float = 20.0
    read_text: bool = True              # read text in images: its own glyph reader, taught as below
    tesseract: str = "auto"             # "auto": use tesseract (if installed) as a teacher for reading; "": never; or a path
    vision_teacher: bool = True         # ask an LLM teacher that can see (an Ollama vision model) what an image shows
    learn_text: bool = True             # text it reads is kept as knowledge it can answer from (and its source)
    glyph_practice: int = 600           # practice glyphs it renders from fonts per round (it knows what each one is)
    fonts: tuple = ()                   # font files to render practice glyphs with ( empty: the system's fonts + its own)
    min_glyph_confidence: float = 0.35  # below this a character reads as "?"
    glyphs_per_label: int = 40          # it practises until it has drawn at least this many glyphs of each character
    practice_after_idle_seconds: float = 60.0  # practice waits until no one has given it an order for this long
    keep_images: int = 2000


@dataclass
class TutorPolicy:
    """When it is idle, its Ollama teachers set it orders, judge its answers and teach it (tutor.py, P22)."""
    enabled: bool = True
    providers: tuple = ("ollama",)      # which teachers may set orders (by provider)
    idle_seconds: float = 120.0         # no orders from people for this long first
    every_seconds: float = 180.0        # between two teacher-set orders
    max_per_hour: int = 20
    kinds: tuple = ("question", "skill", "maths", "tool", "code")
    learn: bool = True                  # what a teacher says is right is taught (as a teacher, never as a person)
    keep: int = 200


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
    tasks: TaskPolicy = field(default_factory=TaskPolicy)
    challenges: ChallengePolicy = field(default_factory=ChallengePolicy)
    evolution: EvolutionPolicy = field(default_factory=EvolutionPolicy)
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    agency: AgencyPolicy = field(default_factory=AgencyPolicy)
    dreams: DreamsPolicy = field(default_factory=DreamsPolicy)
    directors: DirectorsPolicy = field(default_factory=DirectorsPolicy)
    toolsmith: ToolsmithPolicy = field(default_factory=ToolsmithPolicy)
    objectives: ObjectivesPolicy = field(default_factory=ObjectivesPolicy)
    thinking: ThinkingPolicy = field(default_factory=ThinkingPolicy)
    curiosity: CuriosityPolicy = field(default_factory=CuriosityPolicy)
    avatar: AvatarPolicy = field(default_factory=AvatarPolicy)
    ego: EgoPolicy = field(default_factory=EgoPolicy)
    gitea: GiteaPolicy = field(default_factory=GiteaPolicy)
    code: CodePolicy = field(default_factory=CodePolicy)
    vision: VisionPolicy = field(default_factory=VisionPolicy)
    tutor: TutorPolicy = field(default_factory=TutorPolicy)

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
            tasks=build(TaskPolicy, d.get("tasks")),
            challenges=build(ChallengePolicy, d.get("challenges")),
            evolution=build(EvolutionPolicy, d.get("evolution")),
            memory=build(MemoryPolicy, d.get("memory")),
            agency=build(AgencyPolicy, d.get("agency")),
            dreams=build(DreamsPolicy, d.get("dreams")),
            directors=build(DirectorsPolicy, d.get("directors")),
            toolsmith=build(ToolsmithPolicy, d.get("toolsmith")),
            objectives=build(ObjectivesPolicy, d.get("objectives")),
            thinking=build(ThinkingPolicy, d.get("thinking")),
            curiosity=build(CuriosityPolicy, d.get("curiosity")),
            avatar=build(AvatarPolicy, d.get("avatar")),
            ego=build(EgoPolicy, d.get("ego")),
            gitea=build(GiteaPolicy, d.get("gitea")),
            code=build(CodePolicy, d.get("code")),
            vision=build(VisionPolicy, d.get("vision")),
            tutor=build(TutorPolicy, d.get("tutor")),
        )

    @classmethod
    def load(cls, path: Path, overrides: dict | None = None) -> "Directives":
        data = load_toml(path)
        if overrides:
            data = deep_merge(data, overrides)
        return cls.from_dict(data)
