# NeuroMesh

An always-on, self-improving multi-agent system built on modular neural nodes.

NeuroMesh takes orders from users and fulfils them with a network of small specialist
neural nodes and sandboxed tools. While it runs, with or without anyone prompting it, a team of
agents keeps growing that network: building new nodes when a capability or tool appears,
training them on lessons from external teachers (humans and LLMs), pitting challengers
against serving nodes, promoting winners, disposing of superseded nodes, and discovering,
vetting and learning new tools. Every answer also passes through emotional engines, nodes
that read the tone of a message and decide, from the mesh's temperament, whether to pass,
soften, caveat, hold or withhold the answer. They can never change it: the mesh does not lie.
Everything happens inside a set of explicit directives that a Guardian enforces in code.

```
python -m neuromesh demo      # 30-second offline demo of the whole life cycle
python -m neuromesh run       # live console: node network, lessons in progress, activity log
python -m neuromesh serve     # daemon + HTTP API on 127.0.0.1:8765
python -m neuromesh bench     # CPU vs GPU speed for node training and inference
python -m neuromesh teachers  # which teachers are available (Ollama, Claude); --test for a live lesson
```

Requirements: Python 3.11+ and numpy. Optional: [Ollama](https://ollama.com) for a local
teacher model (see [LLM teachers](#llm-teachers-local-models-ollama-and-claude)), and PyTorch
or CuPy to run the nodes on a GPU (see [Running on a GPU](#running-on-a-gpu)).

---

## How it works

### Neural nodes
A node is a small MLP (ReLU hidden layers, softmax output, Adam; numpy on the CPU, or
PyTorch / CuPy on a GPU) bound to one **capability**: routing orders (`dispatch`), or a
skill such as `sentiment` or `urgency`.
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
- **LLM teachers**: a language model labels inputs, writes synthetic training examples for
  the weakest labels, and answers open-ended orders no node or tool can handle. Use Claude
  (Anthropic API), a **local model on Ollama**, or any OpenAI-compatible server; configure
  as many as you like, each with its own budget, and choose the order they are asked in.
- **HeuristicTeacher**: an offline, lexicon-based stand-in so everything runs with no API key.

### Temperament: emotional engines that never lie
Every answer passes two more neural nodes on its way to the user. They are internal
capabilities: grown, trained, challenged and disposed like any other node, but never a route
for orders.

```
order --> router node --> tool / skill node --> answer
                                                  |
message --> [affect node]        how it comes across: friendly | neutral | frustrated | hostile | distressed
                 |
events -->  temperament          valence, arousal, confidence, trust, patience
                 |
situation -> [disposition node]  pass | caveat | soften | brief | confirm | defer | block
                 |
            honesty guard (P8) --> user
```

- **Temperament**: five numbers that move with what happens. Successes and failures, human
  corrections (confidence dips) and lessons (trust rises), node promotions, quarantined tools,
  research, the tone of each message, repeated requests and load all nudge them, and they fade
  back to a baseline personality with a half-life (`[temperament]` in `directives.toml`). The
  mood survives restarts; `/calm` resets it.
- **The affect node** reads the tone of each message. Teachers teach it in the usual order:
  an LLM teacher reads tone well, and the offline heuristic uses a small lexicon.
- **The disposition node** is the decision engine. It reads the situation (the temperament,
  the tone, whether the order succeeded, the answer's confidence, whether it came from a
  network tool, its length, whether the request is a repeat) and picks an action. The
  temperament policy teaches it (a specialist teacher only this node uses), and so do people:
  `/feedback <order> disposition=pass` overrules the policy for that situation and trains the
  node. Until the node is validated on held-out data, the policy decides.

| action | what the user sees | the default policy chooses it when |
|---|---|---|
| pass | the answer, unchanged | nothing calls for a change |
| caveat | the answer, then a note with its confidence and where it came from | the answer's confidence is low |
| soften | a gentler opening line, then the answer, unchanged | the message is frustrated, hostile or distressed |
| brief | the first sentences, verbatim; `/release` shows the rest | the mesh is strained and the answer is long |
| confirm | a notice; the answer waits for `/release` | trust is low and the answer came from a network tool |
| defer | a notice; `/release` works after `defer_seconds` | patience is low and the request is a repeat |
| block | a notice that an answer exists and is being withheld | the conversation is hostile and trust is low |

- **P8, never lie.** Actions are carried out only by fixed templates. The honesty guard checks
  every result in code before it is shown. The answer's words must survive exactly, or the
  answer must be visibly held back. The only additions allowed are one of a few fixed opening
  lines and a single `[temperament: ...]` disclosure that states real values (trust, patience,
  confidence). Every change is disclosed. A result that fails the check, whether from a bug or
  a tampered template, is replaced by the original answer and audited (`honesty.guard`).
  Failed orders are never shaped.
- **Care floor.** A distressed person is never blocked or kept waiting: a withholding decision
  becomes `soften`.
- `/why <order>` explains a decision: the tone, the mood at the time, the situation, what the
  policy would have chosen, and which node decided. A held or withheld answer stays in the
  operator's database and audit trail. The API never returns it except through `/release`.

```
neuromesh> You stupid machine, what is 5 * 5
 [ord-6a1f02c4] Let's keep this calm. Here is the answer:
    25
    [temperament: gentler wording added; the answer itself is unchanged]  [tool:calculator | conf 0.95 | teachers]
neuromesh> I hate you, you moron. what is 6 * 7          (a few times; trust falls)
 [ord-0b93d7e1] [temperament: withheld: the mesh is not giving this answer now (the message read as hostile; trust 0.19). An answer exists and nothing here is made up - ask again later]  [tool:calculator | conf 0.95 | teachers]
neuromesh> /why ord-0b93d7e1      /temperament      /calm
```

Set `enabled = false` under `[temperament]` to send every answer straight to the user.

### The agents
All run concurrently (asyncio) and coordinate through a message bus, the registry and the store.

| agent | job |
|---|---|
| Orchestrator | takes orders, routes them, runs tools and nodes, passes every answer through the emotional engines, returns results with provenance |
| Architect | seeds nodes for capability gaps, expands nodes when labels/tools appear, breeds challengers when a champion is below target |
| Trainer | trains nodes continuously on teacher-labelled experience (champion updates must not regress); asks teachers for lessons where the network is weakest |
| Evaluator | benchmarks champion vs challengers on held-out data (paired test) and promotes winners |
| Curator | disposes retired nodes after the rollback window and failed challengers; enforces the node budget |
| Explorer | discovers tools, routes them through approval, self-tests them, activates them, health-checks them, handles tool supersession |
| Researcher | reads up in the background (own hourly budget) on articles linked from what users asked about, and refreshes retained knowledge that has aged |
| Sentinel | watches the kill switch, re-checks local teacher servers (Ollama), lets the mood settle and saves it, publishes heartbeats with the directive scorecard |

Background agents yield whenever an order is in flight and stay within a compute duty cycle.

### Directives (`directives.toml`)
The constitution. **Prime directives** are enforced in code by the Guardian
(`neuromesh/guardian.py`):

| id | enforced as |
|---|---|
| P1 human authority | `/halt`, `POST /halt` or a `state/STOP` file stops all autonomous activity and order handling immediately |
| P2 orders first | background work waits while orders are in flight and is capped by `max_duty_cycle` |
| P3 approved tools only | tools needing permissions wait for a human; forbidden permissions are always refused; changed tool code is re-vetted |
| P4 budgets | node budget, background compute budget, per-teacher LLM call/token budgets per hour |
| P5 no regression | a challenger is promoted, or a champion updated, only if it is not worse on held-out data; superseded nodes stay rollback-able for a grace period |
| P6 auditability | every lifecycle change, tool decision, approval and denial goes to the audit log |
| P7 honesty | every answer carries confidence and provenance; unfulfilled orders say so |
| P8 never lie | the temperament may withhold, delay, shorten or add context or a gentler tone, never change an answer; every change is disclosed; distressed people are never kept waiting (the honesty guard in `neuromesh/temperament.py`) |

**Goals** are measured continuously and shown on the scorecard (`/status`): competence
(held-out accuracy), autonomy (share of orders served by nodes), tool fluency, economy.
Everything else in the file is tunable policy.

---

## Using it

### Live console
`python -m neuromesh run` opens a live view in the terminal:

```
 NeuroMesh │ numpy 2.5.3 on cpu │ up 15s │ orders 30 │ nodes 6 live / 15 built │ background 3% │ RUNNING
─ network: how orders reach nodes and tools ──────────────────────────│─ activity log [all] ──────────────────────────
 orders                                                               │ 12:58:43 eval     sentiment on 16 held-out:
  └─▶ router ★ dispatch-22b4ef 64 lr=0.01 · held-out 1.000 · 15 serve │                   sentiment-d32e0a=0.5,
       ↳ challenger dispatch-7fd432 expand 64 lr=0.01 · slice 7/8 · h │                   sentiment-f7d077=0.5625
     ├─▶ skill:sentiment ⇒ ★ sentiment-f7d077 64 lr=0.01 · held-out 0 │ 12:58:43 retire   sentiment-d32e0a -
     │                     ↳ retired sentiment-a607e3 · held-out 0.30 │                   superseded by
     │                     ↳ retired sentiment-d32e0a · held-out 0.50 │                   sentiment-f7d077
     ├─▶ skill:urgency   ⇒ ★ urgency-6fdfc0 64 lr=0.01 · held-out 1.0 │ 12:58:43 promote  sentiment-f7d077 now serves
     ├─▶ tool:calculator ⇒ calculator v1.0.0 [active] 10 runs         │                   'sentiment', replacing
     ├─▶ tool:word_stats ⇒ word_stats v1.0.0 [active] 4 runs          │                   sentiment-d32e0a - beat the
     └─▶ answer:direct   ⇒ teachers: none available (queued for a hum │                   champion on held-out data
 nodes: 6 live · 15 built · 9 disposed                                │                   (0.562 vs 0.500, z=1.00)
                                                                      │ 12:58:43 train    dispatch-7fd432 slice 6 on
                                                                      │                   34 examples loss 0.001
─ lessons and work in progress ───────────────────────────────────────│ 12:58:43 eval     sentiment on 16 held-out:
 ✓ evaluation evaluator dispatch: 2 node(s) on 6 held-out examples x7 │                   sentiment-f7d077=0.5625
 ✓ evaluation evaluator sentiment: 1 node(s) on 18 held-out examples  │ 12:58:43 eval     dispatch on 6 held-out:
 ✓ training   trainer   dispatch-7fd432 [candidate, 64 lr=0.01] slice │                   dispatch-22b4ef=1.0,
 ✓ evaluation evaluator sentiment: 1 node(s) on 16 held-out examples  │                   dispatch-7fd432=1.0
 ✓ training   trainer   dispatch-7fd432 [candidate, 64 lr=0.01] slice │ 12:58:43 lesson   heuristic taught 8 x
 ✓ evaluation evaluator sentiment: 2 node(s) on 16 held-out examples  │                   sentiment=negative (lowest
 lessons given so far: heuristic 23                                   │                   recall 0.00 on 'negative')
 teachers: human ✓ · ollama:auto ✗ · llm:claude-haiku-5-5 ✗ · heurist │ 12:58:43 train    dispatch-7fd432 slice 7 on
 goals:    G1:0.556 >=0.95 ✗ · G2:0.4 >=0.8 ✗ · G3:1 >=1 ✓ · G4:6 <=2 │                   34 examples loss 0.001
─ result ─────────────────────────────────────────────────────────────│ 12:58:43 eval     sentiment on 18 held-out:
 [ord-54f083cb] positive  [skill:sentiment | conf 0.95 | teachers]    │                   sentiment-f7d077=0.5556
    . explicit route skill:sentiment                                  │ 12:58:43 eval     dispatch on 6 held-out:
    . sentiment -> positive via teacher:heuristic (0.95)              │                   dispatch-22b4ef=1.0,
neuromesh>                                                            │                   dispatch-7fd432=1.0
```

- **Network** (top left): how orders reach the nodes. Orders go to the router node, which
  routes to each skill node or tool. Every champion is marked with a star; its challengers
  (how they were derived, training progress, held-out score) and retired nodes (with the
  countdown to disposal) hang underneath it. Pending, rejected and quarantined tools are listed too.
- **Lessons and work in progress**: lessons teachers are giving right now (an Ollama call
  shows with a spinner and a timer), training slices, evaluations, tool self-tests and
  orders in flight, then the work that just finished.
- **Answers** (in the network panel): the affect and disposition nodes every answer passes,
  and the honesty guard's tally of answers checked, changed, held and withheld.
- **Teachers, knowledge, goals and mood**: which teachers are up, retained knowledge, the
  directive scorecard, and the temperament with the last change it made.
- **Activity log** (right): everything that happens, one line each: orders, lessons,
  training, evaluations, node births, promotions, retirements and disposals, tool vetting,
  approvals and Guardian denials. Also written to `state/activity.log`.
- **Prompt** (bottom left): orders and `/commands` as before; the result appears just above it.

Keys: **Enter** send, **Up/Down** history, **Left/Right/Home/End**, **Ctrl-U** clear the line,
**Ctrl-W** delete a word, **Tab** switch the left panel between the network and the output of
the last command, **PgUp/PgDn** scroll the log, **Ctrl-C** or `/quit` stop. `/log important`
hides routine lines (training, evaluations, single lessons); `/log all` brings them back.

- `run --plain` gives the old line-by-line console; it is also used automatically when input
  is piped or there is no terminal.
- `run --ascii` (or `NEUROMESH_ASCII=1`) draws without Unicode, for terminals that need it.
- `serve --live` shows the same view, prompt included, while serving the HTTP API.
- Uses Python's built-in `curses`: nothing to install on macOS or Linux. On Windows, run
  `pip install windows-curses` (without it, the plain console is used).

Commands:
```
what is 12 * 7                      sentiment: the hotel was lovely but the staff were rude
/status    /lineage    /tools    /teachers
/approvals    /approve apv-1    /deny apv-2
/feedback ord-3f2a91c0 tool:calculator
/define language english,french,spanish :: Which language a sentence is in
/questions    /answer q1 french
/rollback sentiment    /pause    /halt    /resume
/research <topic>    /knowledge [question]    /forget <article title>
/temperament    /why ord-3f2a91c0    /release ord-3f2a91c0    /calm
/feedback ord-3f2a91c0 disposition=pass    /feedback ord-3f2a91c0 affect=frustrated
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
| POST | `/research` | `{"topic": "..."}` |
| POST | `/release` | `{"order_id": "..."}`: an answer the temperament held back |
| POST | `/calm` | reset the temperament to its baseline |
| POST | `/halt`, `/resume`, `/pause`, `/rollback` | |
| GET | `/status`, `/lineage`, `/events`, `/audit`, `/questions`, `/knowledge?q=...`, `/temperament`, `/why?order=...` | |

```
python -m neuromesh order "convert 10 km to miles"
```

### LLM teachers: local models (Ollama) and Claude
Teachers are asked in the order set by `[teachers] order` (default: human, ollama, llm,
heuristic). A teacher that is unavailable is skipped automatically, and the next one answers.

**A local model with Ollama.** Teaching then stays on your machine and costs nothing per call.
```
ollama pull llama3.1                        # or any chat model you like
python -m neuromesh teachers --test         # is it found? does a live lesson work?
python -m neuromesh run                     # uses it automatically
python -m neuromesh run --ollama qwen2.5:7b # pick a specific model
python -m neuromesh demo --ollama auto      # the demo, taught by your local model
```
- With `model = "auto"` (the default) NeuroMesh uses the first chat model Ollama has installed
  (embedding models are skipped). Set `model` in `[teachers.ollama]` to choose one.
- It talks to Ollama's native `/api/chat`. Every lesson uses a JSON schema, and on a fixed
  label set the labels are an enum, so even a small model can only answer with a valid label.
- The server is found at `base_url`, else `$OLLAMA_HOST`, else `http://localhost:11434`;
  proxy settings are ignored for local servers. Use `--ollama-url` for another machine.
- The server and model are re-checked every 30 s (`probe_seconds`). If Ollama stops, or the
  model is removed, the teacher goes offline, the next teacher takes over, and it comes back by
  itself when Ollama does. `/status` and `teachers` say why a teacher is off and what to run.
- Local models on a CPU can be slow: raise `timeout_s` if lessons time out. For reasoning
  models, `think = false` skips the thinking phase and is much faster.
- Examples taught by Ollama are weighted 1.2 (Claude 1.5, human 3.0). Change this in
  `[teachers.weights]`.

**Claude.**
```
export ANTHROPIC_API_KEY=...                # the teacher is off while this is unset
python -m neuromesh run
```
**Your own mix.** Any `[teachers.<name>]` table with a `provider` (`ollama`, `anthropic`,
`openai`) defines another teacher with its own model and hourly budget. List its name in
`order` to use it. For example, a fast local model first and a larger one on another
machine as backup:
```toml
[teachers]
order = ["human", "ollama", "big_local", "heuristic"]

[teachers.big_local]
provider = "ollama"
model = "llama3.1:70b"
base_url = "http://gpu-box:11434"
timeout_s = 300
```
Leaving a teacher out of `order` switches it off; so does `enabled = false`. The demo stays
offline unless you pass `--ollama MODEL` or `--llm`.

### Research and retained knowledge (Wikipedia)
The mesh can look things up on Wikipedia and keeps what it learns.

```
neuromesh> /approve apv-1               # once: the wikipedia tool needs network access
neuromesh> who was Ada Lovelace?        # researched: fetched, retained, answered
neuromesh> when was Ada Lovelace born?  # recalled: answered from memory, no network
neuromesh> /research the Krebs cycle    # read up on something now
neuromesh> /knowledge                   # what has been retained
neuromesh> /knowledge capital of France # what it would recall for a question, and how confidently
neuromesh> /forget Paris
```

- **Approval first.** `plugins/wikipedia` declares the `network` permission, so it waits for a
  human's `/approve` (P3), then self-tests against the live API before it is used. It is the
  only component that talks to Wikipedia. It identifies itself with a User-Agent and contact URL,
  as [Wikimedia asks](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy).
  Set your own contact in `[tools.config.wikipedia]`.
- **Retained.** Each article is stored with its sections as passages, in `state/neuromesh.db`
  (indexed with SQLite FTS5), so it survives restarts. Reference sections are dropped.
- **Memory first.** A question is answered from retained knowledge when it matches well enough
  (`recall_threshold`, 0.8). Confidence needs the question to name the article's subject *and* a
  passage to cover what is asked, so "population of France" is not answered from the article on
  Paris. Questions that led to an article are remembered as its aliases ("capital of France" ->
  Paris), so they are recalled next time. Otherwise the mesh researches, retains, then answers.
- **Answers come from the passages.** If an LLM teacher (Ollama or Claude) is available it writes
  the answer, told to use only the retained passages; otherwise the best-matching sentences are
  quoted. Every answer names its source article and URL.
- **Which orders.** Orders the router sends to the wikipedia tool ("tell me about...", "who
  was...", "research..."), `tool wikipedia: <topic>`, and questions no tool or skill fits
  ("how tall is the Eiffel Tower?") when `research_questions = true`.
- **Background curiosity.** After researching for an order, the Researcher agent also reads up to
  `follow_links` (2) articles the introduction links to, never more than `max_depth` (1) away
  from what was asked. It re-reads retained articles older than `refresh_days` (30). It only
  works when no order is in flight, within its own hourly budget.
- **Budgets and limits** (`[research]` in `directives.toml`): `max_fetches_per_hour` (60) for all
  lookups, `background_per_hour` (20) for curiosity and refreshing, `max_documents` (2000)
  retained, after which the least recently used are forgotten. A network outage is reported per
  order but never quarantines the tool.
- In the live console: lookups show in the work panel, `learn` and `recall` lines in the activity
  log, and the knowledge line shows articles retained and lookups used this hour.

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
keywords = ["...", 're:\d+\s*[*/+-]\s*\d+']   # offline teacher's routing cues; re: patterns count double
supersedes = "old_tool"        # optional: retire old_tool and relabel its routing data
provides = "knowledge"         # optional: results are retained and recalled (see Research)
[config]                       # optional: settings passed to the tool as "config" on every run
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

### Running on a GPU
Nodes run on numpy on the CPU by default. To train and serve them on a GPU, install one of:

- **PyTorch**: NVIDIA (CUDA), AMD (ROCm build) or Apple Silicon (MPS);
  see https://pytorch.org/get-started/locally/
- **CuPy**: NVIDIA only; `pip install cupy-cuda12x` (match your CUDA version)

Then choose the device with `--device`, the `NEUROMESH_DEVICE` environment variable, or
`[compute]` in `directives.toml` (in that order of precedence):
```
python -m neuromesh run --device auto       # best GPU found, otherwise the CPU
python -m neuromesh serve --device cuda:1   # second NVIDIA GPU
python -m neuromesh demo --device mps       # Apple Silicon
python -m neuromesh run --device cuda --gpu-library cupy
```
`auto` falls back to the CPU when no GPU is usable. A device you name explicitly that isn't
available is an error that says what to install, never a silent fallback. `/status` shows
the device in use.

On the GPU: node training, held-out evaluation and every forward pass that answers an
order. On the CPU: text encoding, tools and everything else. Weights are saved
device-independent, so you can train on a GPU box and serve from a laptop, or switch devices
between runs.

**Is it worth it?** The default nodes are small (1,024 inputs, 64 hidden units, batches of 32).
At that size a GPU will likely be no faster than numpy, because kernel-launch and transfer
overhead dominate. GPUs pay off when the nodes grow. Measure on your own hardware:
```
python -m neuromesh bench --device auto
python -m neuromesh bench --device auto --input-dim 16384 --hidden 512,256 --batch-size 256
```
The benchmark trains identical nodes on CPU and GPU and reports the speed-up and whether
the two agree. A GPU-sized starting point for `[nodes]` in `directives.toml`:
```toml
input_dim = 16384        # wider text features: fewer hash collisions, more for the GPU to do
default_hidden = [512]
batch_size = 256
max_hidden = 2048        # lets the architect breed wider challengers
```
Changing `input_dim` retires the saved nodes (they can't read the new features); the
architect rebuilds them from the stored experience, so nothing learned from teachers is lost.

To check a GPU setup: `python -m pytest tests/test_backend.py`. Its parity tests compare
every GPU backend installed on the machine against numpy (forward pass, training, node
derivations, moving weights between devices). They are skipped where no GPU library exists.

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
- **Research reaches the internet** only through the approved wikipedia tool, within hourly
  budgets, and background curiosity stays within `max_depth` of what users asked. Retained
  knowledge is Wikipedia text: answers name the article they came from, and are only as good as
  it was when fetched (articles are re-read after `refresh_days`). The default keyword routing is
  simple; with an LLM teacher, routing of questions to research is much better.
- **The temperament changes how an answer is delivered, never what it says.** The honesty guard
  checks the form of every change (verbatim answer, fixed openings, one disclosure line). It
  does not check that the answer itself is true. That is P7's job: confidence and provenance on
  every answer. With only the offline heuristic, tone reading is a crude lexicon; an LLM teacher
  reads tone far better and the affect node learns from it. A blocked answer is withheld from the
  user, not destroyed: the operator can see it in the order record.
- The offline heuristic teacher is a stand-in so the system and its tests run anywhere; labels
  from it are only as good as its lexicon.

## Project layout
```
directives.toml          constitution: prime directives, goals, policy
capabilities.toml        seed skills
plugins/                 installed tools (calculator, word_stats, wikipedia)
examples/incoming_plugins/  tools the demo installs at runtime
neuromesh/
  runtime.py             System: wiring, cascade, learning, tool activation, status
  live.py                live console (curses): network, work in progress, activity log, prompt
  activity.py            work-in-flight tracker and the activity log
  commands.py            orders and /commands shared by the live and plain consoles
  agents.py              the eight agents
  nodes.py               NeuralNode + derivations (expand / widen / deepen / retune / fresh)
  backend.py             compute backends: numpy (CPU), PyTorch (CUDA / ROCm / MPS), CuPy
  bench.py               CPU vs GPU benchmark
  registry.py            node life cycle
  guardian.py            directive enforcement, budgets, approvals, kill switch, audit
  teachers.py            human / LLM / heuristic teachers
  tools.py               manifests, sandbox, discovery
  knowledge.py           retained knowledge: articles + passages (SQLite FTS5), recall confidence
  temperament.py         emotional engines: temperament, policy teacher, output templates, honesty guard (P8)
  store.py               SQLite: experience, lineage, tools, orders, temperament, audit
  encoding.py, bus.py, tasks.py, config.py, report.py, server.py, demo.py, __main__.py
tests/                   79 tests: nodes, guardian, LLM wire formats and a mock Ollama server,
                         Wikipedia research against a mock MediaWiki API, temperament and
                         the honesty guard (incl. tampered templates), full system,
                         live console (incl. a pseudo-terminal run), compute
                         backends (4 GPU parity tests run where PyTorch or CuPy is installed)
```

State lives in `state/` (SQLite database, node weights, log). Delete it to start fresh.

```
python -m pytest tests          # or: python -m unittest discover tests
```
