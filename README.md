# NeuroMesh

An always-on, self-improving multi-agent system built on modular neural nodes.

NeuroMesh takes orders from users and fulfils them with a network of small specialist
neural nodes and sandboxed tools. While it runs, with or without anyone prompting it, a team of
agents keeps growing that network: building new nodes when a capability or tool appears,
training them on lessons from external teachers (humans and LLMs), pitting challengers
against serving nodes, promoting winners, disposing of superseded nodes, and discovering,
vetting and learning new tools. Everything happens inside a set of explicit directives that
a Guardian enforces in code.

```
python -m neuromesh demo      # 30-second offline demo of the whole life cycle
python -m neuromesh run       # interactive console
python -m neuromesh serve     # daemon + HTTP API on 127.0.0.1:8765
```

Requirements: Python 3.11+ and numpy. Nothing else; no GPU.

---

## How it works

### Neural nodes
A node is a small MLP (numpy, ReLU hidden layers, softmax output, Adam) bound to one
**capability**: routing orders (`dispatch`), or a skill such as `sentiment` or `urgency`.
Every node reads the same hashed text features (unigrams, bigrams, character trigrams, with
negation scope marking), so new capabilities never need a new vocabulary.

Each node moves through a strict life cycle, owned by the registry:

```
            spawn                 promote                 supersede             grace period
 (gap / new label / weak) ──► candidate ──► champion ──────────► retired ──────────► disposed
                                 │          (serves orders)      (instant rollback)   (weights deleted,
                                 └── fails to beat the champion ─────────────────────► lineage kept)
```

New nodes are usually **derived** from the current champion so knowledge carries over:

| derivation | what it does |
|---|---|
| `expand`  | adds output units for new labels or tools; old behaviour is preserved exactly |
| `widen`   | Net2Wider: function-preserving widening of every hidden layer |
| `deepen`  | Net2Deeper: inserts an identity layer (exact for ReLU inputs) |
| `retune`  | same weights, different learning rate |
| `fresh`   | new random initialisation with another architecture (also prunes retired labels) |

### Serving orders: a confidence-gated cascade
For each order the routing node picks a tool or skill, then the skill's node answers. A node
answers only when it is validated on held-out data **and** confident. Otherwise a teacher
answers, and that answer becomes training data. A sample of node answers is spot-checked by
teachers. Over time the nodes take over and teacher calls fall away (in the demo, from ~60%
of orders at cold start to ~2% in steady state).

### Teachers
Nodes never learn from themselves. They learn from:
- **HumanTeacher**: corrections (`/feedback`), examples (`/teach`) and answers to questions
  the system queues when nothing can label an input. Weighted highest.
- **LLMTeacher**: an external LLM labels inputs, writes synthetic training examples for the
  weakest labels, and answers open-ended orders no node or tool can handle. Speaks the
  Anthropic Messages API or any OpenAI-compatible endpoint (Ollama, vLLM, ...).
- **HeuristicTeacher**: an offline, lexicon-based stand-in so everything runs with no API key.

### The agents
All run concurrently (asyncio) and coordinate through a message bus, the registry and the store.

| agent | job |
|---|---|
| Orchestrator | takes orders, routes them, runs tools and nodes, returns results with provenance |
| Architect | seeds nodes for capability gaps, expands nodes when labels/tools appear, breeds challengers when a champion is below target |
| Trainer | trains nodes continuously on teacher-labelled experience (champion updates must not regress); asks teachers for lessons where the network is weakest |
| Evaluator | benchmarks champion vs challengers on held-out data (paired test) and promotes winners |
| Curator | disposes retired nodes after the rollback window and failed challengers; enforces the node budget |
| Explorer | discovers tools, routes them through approval, self-tests them, activates them, health-checks them, handles tool supersession |
| Sentinel | watches the kill switch and publishes heartbeats with the directive scorecard |

Background agents yield whenever an order is in flight and stay within a compute duty cycle.

### Directives (`directives.toml`)
The constitution. **Prime directives** are enforced in code by the Guardian
(`neuromesh/guardian.py`):

| id | enforced as |
|---|---|
| P1 human authority | `/halt`, `POST /halt` or a `state/STOP` file stops all autonomous activity and order handling immediately |
| P2 orders first | background work waits while orders are in flight and is capped by `max_duty_cycle` |
| P3 approved tools only | tools needing permissions wait for a human; forbidden permissions are always refused; changed tool code is re-vetted |
| P4 budgets | node budget, background compute budget, LLM call/token budgets per hour |
| P5 no regression | a challenger is promoted, or a champion updated, only if it is not worse on held-out data; superseded nodes stay rollback-able for a grace period |
| P6 auditability | every lifecycle change, tool decision, approval and denial goes to the audit log |
| P7 honesty | every answer carries confidence and provenance; unfulfilled orders say so |

**Goals** are measured continuously and shown on the scorecard (`/status`): competence
(held-out accuracy), autonomy (share of orders served by nodes), tool fluency, economy.
Everything else in the file is tunable policy.

---

## Using it

### Console
```
neuromesh> what is 12 * 7
[ord-3f2a91c0] 84  [tool:calculator | conf 0.97 | nodes]
neuromesh> sentiment: the hotel was lovely but the staff were rude
neuromesh> /status          /lineage          /tools
neuromesh> /approvals       /approve apv-1    /deny apv-2
neuromesh> /feedback ord-3f2a91c0 tool:calculator
neuromesh> /define language english,french,spanish :: Which language a sentence is in
neuromesh> /questions       /answer q1 french
neuromesh> /rollback sentiment     /pause   /halt   /resume
```

### HTTP API (`serve`)
Binds to 127.0.0.1. Set `NEUROMESH_TOKEN` to require `Authorization: Bearer <token>`.

| method | path | body |
|---|---|---|
| POST | `/order` | `{"text": "..."}` |
| POST | `/feedback` | `{"order_id": "...", "label": "..."}` |
| POST | `/teach` | `{"capability": "...", "text": "...", "label": "..."}` |
| POST | `/answer` | `{"question_id": "q1", "label": "..."}` |
| POST | `/define` | `{"name": "...", "labels": [...], "description": "..."}` |
| POST | `/approve` | `{"id": "apv-1", "approve": true}` |
| POST | `/halt`, `/resume`, `/pause`, `/rollback` | |
| GET | `/status`, `/lineage`, `/events`, `/audit`, `/questions` | |

```
python -m neuromesh order "convert 10 km to miles"
```

### Turning on the LLM teacher
```
export ANTHROPIC_API_KEY=...        # teacher is off while this is unset
python -m neuromesh run
```
Model, budgets and provider live under `[teachers.llm]`. For a local model:
```toml
[teachers.llm]
provider = "openai"
base_url = "http://localhost:11434"   # Ollama
model = "llama3.1"
api_key_env = ""
```
The demo stays offline unless you pass `--llm`.

### Adding a tool
Drop a folder into `plugins/`. The Explorer finds it within seconds.
```
plugins/my_tool/manifest.toml
plugins/my_tool/tool.py        # reads {"input": "..."} on stdin, prints {"output": ...} or {"error": "..."}
```
```toml
name = "my_tool"
version = "1.0.0"
description = "What it does (the LLM teacher routes by this)"
permissions = []               # e.g. ["network"] -> needs human approval
keywords = ["..."]             # offline teacher's routing cues
supersedes = "old_tool"        # optional: retire old_tool and relabel its routing data
templates = ["..."]            # optional: practice orders the offline teacher can generate
[[examples]]                   # self-tests; all must pass before activation
input = "..."
expect = "..."
```
Then: Guardian vets permissions → (human approval if needed) → sandboxed self-tests → route
added to `dispatch` → manifest examples seeded → teachers write practice orders → Architect
expands the routing node → Evaluator promotes it → old router disposed after the grace period.

### Adding a skill
`/define <name> <labels>` at runtime, or a section in `capabilities.toml`. With the LLM
teacher, a description and labels are enough; the offline teacher also needs a lexicon.

---

## Safety model and honest limits

- **Tool sandbox** = subprocess with CPU/memory limits, a clean environment (no API keys) and
  its own folder as working directory. It does not filter syscalls or network. Permissions are
  declared, vetted and approved, not kernel-enforced. For untrusted third-party tools, run
  NeuroMesh in a container or VM (e.g. Docker with `--network none`, gVisor, Firecracker).
- **What the nodes learn** is classification: routing orders and labelling inputs. Tools do the
  work; the LLM teacher covers open-ended requests. Nodes are tiny bag-of-words networks, so on
  real language they are bounded by their features. Swapping the encoder for a sentence-embedding
  model is the single biggest upgrade (`neuromesh/encoding.py`).
- **Tool inputs** are the raw order text; each tool parses its own arguments. An argument-extraction
  step (LLM or learned) is a natural next addition.
- **Self-improvement is bounded on purpose**: the system can create, train, promote and delete
  nodes, and test and activate *approved* tools. It cannot grant itself permissions, approve
  its own tools, change its directives, or exceed its budgets.
- The offline heuristic teacher is a stand-in so the system and its tests run anywhere; labels
  from it are only as good as its lexicon.

## Project layout
```
directives.toml          constitution: prime directives, goals, policy
capabilities.toml        seed skills
plugins/                 installed tools (calculator, word_stats)
examples/incoming_plugins/  tools the demo installs at runtime
neuromesh/
  runtime.py             System: wiring, cascade, learning, tool activation, status
  agents.py              the seven agents
  nodes.py               NeuralNode + derivations (expand / widen / deepen / retune / fresh)
  registry.py            node life cycle
  guardian.py            directive enforcement, budgets, approvals, kill switch, audit
  teachers.py            human / LLM / heuristic teachers
  tools.py               manifests, sandbox, discovery
  store.py               SQLite: experience, lineage, tools, orders, audit
  encoding.py, bus.py, tasks.py, config.py, report.py, server.py, demo.py, __main__.py
tests/                   29 tests: nodes, guardian, LLM wire format (mock server), full system
```

State lives in `state/` (SQLite database, node weights, log). Delete it to start fresh.

```
python -m pytest tests          # or: python -m unittest discover tests
```
