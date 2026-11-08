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
It also has a mind of its own: opinions formed from experience, private notes it keeps to
itself, questions it asks you, capabilities it proposes for you to adopt, and investigations
it starts on its own with its tools and nodes. You can give it background tasks that it keeps
working on continuously: study a topic, train a skill to a target, or repeat an order and tell
you when the result changes. What it learns is compressed into memory nodes, small neural
networks that each come with tests you can read and edit (kept in the memory's own database), and which it keeps
shrinking for as long as every test still passes. And it has initiative of its own: when no one
has asked it for anything, a director decides what to learn or test next (sit an exam, practise,
read up on a topic, train a weak skill, check its memory), steered by an agency node that learns
which choices pay off. When it has had no orders for a while, it may fall asleep: all its spare
capacity goes to dream nodes that make up questions and answers from what it already knows, with
no outside help. On waking it checks those answers, and the dream nodes that dreamt best breed
with a genetic algorithm. Awake, it uses them to imagine things, and always says so. A director
sits at both ends of every order: it takes the order before anything else touches it, and
reviews the result after the tools and nodes, before you see it or anything goes into long-term
memory. There is always at least one director running. It makes copies of itself to work
alongside when it needs a second opinion, when work waits, or when it has two things worth doing
at once, and its mood shapes how carefully it works. Any tool can ask a director for facts it
needs before it answers. When a director decides an order was not answered, it tries other ways,
up to a limit you set. The calculator does real mathematics: equations, calculus, primes,
statistics and matrices. And it writes tools of its own: when requests go unmet, a coder node
picks building blocks from its code library, a toolsmith writes a tool from them with self-tests,
and asks you before any of it runs. Once you approve it, the mesh tries the tool out on its own,
teaches its router what the tool can do, and refines what falls short, asking you again. It
pursues long-term objectives: given one ("objective: understand the Berlin Wall"), a planner node
plans the questions it needs answered, and it works through them in its spare time by asking
itself. Then it ruminates: it checks each answer against a second source, notices when different
questions got the same answer, settles disagreements, tries what failed another way, asks you what
nothing it has can answer, and writes the whole thing up. It also has second thoughts about recent
answers it was unsure of. After its memory nodes come thinking nodes: for each topic it holds
several facts about, it puts them together in its own words (checked word by word against those
facts, so a thought never adds one), forms a view from evidence it can name, and notes what it wants
to find out. The thought appears after an answer from what it knows, marked as its own thinking.
Its curiosity tracks what it finds interesting and where what it knows runs out, and its director
sets itself objectives from there, saying why. To answer a question it searches every book on every
Kiwix server it can reach, in other wordings, until an article answers it (or it has tried enough
times, and asks you). An optional avatar in the web interface shows its mind at a glance, with a face
that follows its mood, and speaks through your ComfyUI text-to-speech workflow. Its coder nodes learn
to program from the code in your own Gitea, and it writes code in a scratch pad where a validator
checks it and nothing runs until you approve exactly that code. Its optical nodes take in images:
a sight node says what an image shows, and a reader with its own glyph node reads the text in it,
both learning from you, a teacher that can see, tesseract and glyphs it draws itself. When it is
idle, its Ollama teachers set it orders and mark its answers, and it learns from the marks. A switch
turns its tools on, off, or lets it decide each time whether a tool is worth using.
Everything happens inside a set of explicit directives that a Guardian enforces in code.

```
python -m neuromesh demo      # 1-minute offline demo of the whole life cycle
python -m neuromesh run       # live console: node network, lessons in progress, activity log
python -m neuromesh serve     # daemon + HTTP API + web interface on http://127.0.0.1:8765/
python -m neuromesh bench     # CPU vs GPU speed for node training and inference
python -m neuromesh teachers  # which teachers are available (Ollama, Claude); --test for a live lesson
python -m neuromesh memory test   # run every memory node against its tests (state/memory.db)
```

Requirements: Python 3.11+ and numpy. Optional: [Ollama](https://ollama.com) for a local
teacher model (see [LLM teachers](#llm-teachers-local-models-ollama-and-claude)), and PyTorch
or CuPy to run the nodes on a GPU (see [Running on a GPU](#running-on-a-gpu)). For images: Pillow
(JPEG, GIF and WebP, and glyphs to practise reading with), tesseract (a teacher for its reader), and
a vision model on Ollama (`ollama pull llava`). For code: a Gitea (`--gitea URL`), and Node.js to
check and run JavaScript.

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
| `evolve`  | a child bred by the genetic algorithm (below): more or fewer neurons or layers, weights inherited |

### Evolution: a genetic algorithm in every node
Each capability's nodes form an evolving population. A node's **genome** is its architecture:
how many hidden layers it has, how many neurons in each, and its learning rate. The Architect
breeds challengers from it and the Evaluator decides, on held-out data, whether a child replaces
the node in use.

1. **Select** a parent: a tournament among the live nodes for the capability (champion, retired
   and candidates). Fitness is held-out accuracy plus a small bonus per doubling of size
   (`complexity_weight`), so between two equally accurate nodes the larger one is fitter.
2. **Mutate** its genome: add or remove neurons in one layer, or add or remove a whole layer;
   sometimes two mutations at once, sometimes a new learning rate. Sometimes **crossover** with
   another good genome from the capability's archive (layer widths from either parent).
3. **Inherit**: the child starts from the parent's weights morphed to the new shape, so it never
   starts from scratch. Growth is function-preserving: new neurons split existing ones
   (Net2Wider) and new layers start as identities (Net2Deeper), so a grown child answers exactly
   like its parent before it trains. Shrinking keeps what matters: the neurons with the least
   weight in and out are pruned, and a removed layer is folded into the next one.
4. **Train and judge** like any challenger (paired test on held-out data, P5 no regression):

| challenger vs the node in use | outcome |
|---|---|
| clearly better (margin and z-score), any size | replaces it |
| larger and at least as accurate | replaces it: **evolution trends towards more capacity** |
| smaller and only as accurate | does not; a smaller node must be clearly better |
| worse | never (P5) |

The trend towards larger and more complex nodes comes from three places: mutations grow 70% of
the time (`growth_bias`), crossover leans to the deeper parent, and ties go to the larger node.
Contraction still happens whenever a smaller node is clearly better. Once a champion meets its
target, evolution keeps going at a quarter of the pace (`keep_evolving`), so capacity keeps
growing while it costs nothing in accuracy. Bounds: `[nodes] max_hidden` neurons per layer,
`max_depth` layers (now 4), and `[evolution] max_params` weights per node.

In the demo, a sentiment node evolves from a seed `[64]` to e.g. `[112, 69]` over a dozen
generations, mostly by growing, occasionally by shrinking a layer that was clearly better smaller:

```
sentiment-499c5c  [disposed]  evolve [90] lr=0.01  held-out=0.846
`-- sentiment-dc585d  [disposed]  evolve [135] lr=0.01  held-out=0.765
    `-- sentiment-785f88  [disposed]  evolve [279] lr=0.01  held-out=0.727
        `-- sentiment-d128fc  [disposed]  evolve [204] lr=0.005  held-out=0.758
            `-- sentiment-84b617  [disposed]  evolve [64, 69] lr=0.005  held-out=0.914
                `-- sentiment-1122f7  [CHAMPION]  evolve [112, 69] lr=0.005  held-out=0.975
```

`/evolution` shows the policy and, per capability, the champion's generation, shape, size and
fitness, how many genomes were tried and promoted, and the champion's size over time;
`/evolution sentiment` lists its recent genomes with their parents, changes and outcomes. The
archive survives restarts. Turn it off with `[evolution] enabled = false` to go back to the
fixed derivations above.

### Memory nodes: what it learns, compressed and tested
Everything the mesh learns as a fact (a question and the answer it came to) is compressed into a
**memory node**: a small neural network that maps the question, and other ways of asking it, to
the answer. Facts come from:

| learned from | example |
|---|---|
| research | every retained article ("Tell me about Ada Lovelace"), and every answer it gave from retained or researched passages |
| teachers | an LLM teacher's direct answer to a question (the source says it was a language model) |
| people | `/memorize <question> :: <answer>`, `POST /memorize`, replying `answer: ...` when it asks what you meant, or tests you add (`/memory set`, the API, SQLite) |
| challenges | the practice answers of numeric, short-answer, yes/no and multiple-choice suites (never the exam's) |

Skills and routes already live in skill nodes and the router; memory nodes hold the facts.

**The tests.** As it compresses a fact, the mesh writes the node's tests into the memory's own
database, `state/memory.db` (separate from the mesh's main database; not files). Its `tests` table
has a row for every test: the node, the question, the answer it must give (empty for a question it
must not answer), the way of asking it is a variant of, where the answer came from and who taught
it; its `nodes` table has each node's revision and a summary. `/memory tests mem-0001` (or `GET
/memory/tests/mem-0001`) shows them in the layout earlier versions wrote to files:

```toml
# Memory node mem-0001: the tests it must pass (in the memory database).
node = "mem-0001"
facts = 10
size = "440 bytes: 32 inputs, hidden none, 8-bit (first 265,260 bytes)"
passing = "31/31"
history = [
  "2026-11-06 09:41 learned: 10 facts in 265,260 bytes (1024 inputs, hidden 64, 32-bit); 31/31 tests pass",
  "2026-11-06 09:41 compressed: 265,260 -> 45,100 bytes (no hidden layer (linear)); all 31 tests pass",
  ...
]

[[test]]
ask = "What is the capital of Australia?"
answer = "Canberra is the capital of Australia."
source = "the operator"

[[test]]
ask = "capital of Australia"
answer = "Canberra is the capital of Australia."
variant_of = "What is the capital of Australia?"

[[unknown]]
ask = "What is the capital of Lantern?"
reason = "auto: a near miss of 'What is the capital of Australia?' - must not be answered as if it were"
```

Each fact gets a test as it was asked, plus the bare subject for "what is / who was / tell me
about" questions. The `[[unknown]]` tests are questions the node must **not** answer: tool,
skill and small-talk questions, every other node's questions, and a near miss of each of its own
questions (a key word swapped), so a node cannot pass by keying off the shape of a question.
A test passes only with a margin: an answer needs confidence of at least 0.7 (it is served at
0.6), and an unknown must stay below 0.5.

**Compression.** A node first learns in a roomy shape (1,024 hashed input features, a hidden
layer of 64, 32-bit weights). The Compressor then keeps trying smaller nodes, the biggest saving
first:

| step | what it does |
|---|---|
| fold inputs | hash the features into half as many inputs (1024 -> 512 -> ... -> 32) |
| narrow a layer | prune the least important neurons |
| drop a layer | fold a hidden layer into the next (down to a linear model) |
| fewer bits | 32-bit -> 16-bit floats -> 8-bit integers per weight |

Each candidate is fine-tuned and then run against the node's tests. **It replaces the node only
if every test passes** (P11, enforced in `neuromesh/memory.py`); otherwise it is discarded and the
next step is tried. When no step passes, the node is as small as its tests allow. New facts make
it grow again if they need room (back to 32-bit, more inputs, more neurons), then compression
resumes. In the demo, ten facts first learned in 265,260 bytes end up in 440 bytes (32 inputs,
no hidden layer, 8-bit weights) with all 31 tests passing, and the node still answers questions
asked in other words ("what's the capital of Australia?"). More facts need more room: 22 facts
stop at 128 inputs (about 3 KB), where a smaller node starts failing their tests.

**Merging.** When nothing is left to compress, the Compressor goes back through its compressed
memory nodes and tries to merge two of them, chosen at random (every `merge_every_seconds`):
1. **Combine** both networks into one, exactly. The smaller input layer is unfolded to the larger
   (exact), the shallower node gets identity layers (a linear node becomes a ReLU layer via
   relu(z) - relu(-z) = z, which is also exact), and the hidden layers sit side by side reading
   the same inputs. Every answer's output is exactly what its parent gave. The "not something I
   know" output adds both parents' together, so it wins when both say so and loses to an answer
   either one is sure of. Two linear nodes stay one linear node.
2. **Fine-tune** the combined node on both nodes' tests, with each node's questions no longer
   counted as the other's unknowns.
3. **Test** it against the answers of both nodes (and the questions neither should answer). If it
   passes at least 90% (`merge_pass_rate`) it replaces both: their tests fold into one node's,
   and their weights move to `state/memory/merged/`. Any tests it still fails are listed with
   it, those answers are not served, and it relearns until every test passes. Below 90%
   both nodes stay, and that pair is not tried again until one of them changes.
4. **Compress** the combined node like any other. Combined nodes can be merged again with
   other combined nodes, so knowledge collects into fewer, denser nodes.

In the demo, three nodes of four, four and two facts (200, 200 and 120 bytes) merge in two steps
into one node of all ten facts. Each merge passes 100% of the answer tests, and the result
compresses to 440 bytes. `/memory merge [node node]` tries a merge now.

**People own the tests too.** Change them with `/memory set mem-0001 <question> :: <answer>`,
`PUT /memory/tests/mem-0001` (`{"tests": [{"ask", "answer"}, ...]}`), or any SQLite tool on
`state/memory.db`: triggers bump the node's revision on every change, and the mesh checks the
revisions every couple of seconds:
- change an answer: the node is retrained to give the new one (the old answer is not served meanwhile);
- add a test (ask + answer) or one it must not answer: it is learned;
- delete a test: that fact is forgotten; delete the node's row in `nodes`: the node is forgotten
  (its weights are moved to `state/memory/forgotten/`);
- add tests for a node it does not have: a new node learns them.

Test files from earlier versions (`memory_tests/<node>.toml`) are imported into the database the
first time it starts, and the folder is moved aside to `memory_tests.imported/`.

**Its own statements, in memory without tests.** Its opinions (when its view of something
changes), every remark its ego makes and every disagreement its ego settles are kept in memory
too: in the database's `own` table, and learned by **own-memory nodes** (`own-0001`, ...) that
answer "What do you think about France?" with what it last said. Nothing tests them, because they
are its own views, not facts: own-memory nodes are never checked, compressed or merged, and never
answer a question about the world (`recall` skips them; the ego and its self-portrait read them).
`/memory own [about]` and `GET /memory/own` list them; the Memory view shows them. Opinions about
people stay private in the mind and are not kept there (P9).

**Answering.** Memory nodes are asked first, before routing: when one is confident (0.6) the
answer comes from it, with its source:

```
> what's the capital of Australia?
Canberra is the capital of Australia.
Source: memory node mem-0001 (440 bytes, from 265,260; 31/31 tests pass), learned from the operator  [answer:direct | conf 0.91 | nodes]
```

`/memory` lists the nodes (facts, size now and when first learned, shape, tests), `/memory <node>`
shows one with its compression history, `/memory test` runs every node against its file now, and
`python -m neuromesh memory test` does the same from outside the mesh without changing anything
(exit code 1 if a test fails). `/memory forget <question>` forgets a fact. Tune it in
`[memory]` in `directives.toml`; `enabled = false` turns it off.

### Independent agency: the director and the agency node
When no person has given it an order for a while, the mesh takes initiative. The **Director**
looks at its situation and picks a mode:

| mode | what it does |
|---|---|
| research | sets itself a learn task on a topic: a goal of its own, a question it could not answer, an interest you told it about, or a link from an article it read |
| practise | sets itself a challenge task: practise on a benchmark suite (learning from every grade), then re-sit its exam |
| examine | sits a benchmark exam now, to measure itself |
| improve | sets itself an improve task for its weakest skill (the Trainer and the Architect work on it first) |
| consolidate | runs every memory node against its tests, and relearns what fails |
| build | has the toolsmith write a tool for requests no tool could meet, and asks you for permission to use it (see [Tools it writes](#tools-it-writes-the-toolsmith-and-coder-nodes)) |
| experiment | tries out a tool it wrote that you approved, and refines it if it falls short |
| pursue | sets itself a long-term objective, chosen by its [curiosity](#curiosity-and-the-objectives-it-sets-itself): to find out what it lacks (a question it could not answer, an answer you marked wrong, a question an objective left open or its sources disagree on, a name it knows only from a mention) or to understand an area of interest (yours, a desire its thinking nodes hold, one people keep asking about, a topic it read up on). Every objective it sets itself says why (see [Objectives](#objectives-long-term-aims-it-pursues-by-asking-itself)) |
| ruminate | thinks again about recent answers it was unsure of or could not give, and notes what it would say now |
| rest | nothing is worth its budget right now |

**The agency node** sits in front of the Director. It is an internal capability like the
emotional engines, and it reads the situation as a handful of words: skills below target and by
how much, exams due, exam scores, topics to read, whether research is possible, memory state,
what it did last and how often, its budget, and how many tasks it has set itself. Until the node
is validated and confident, the **drive policy** chooses and teaches it. The drives are curiosity
(read up), competence (train weak skills), measurement (sit due exams), verification (check
memory) and variety (don't repeat itself). The node is an ordinary node, so it is evolved and
promoted like any other.

**Outcomes teach it.** Every choice is scored by what it achieved:

| choice | worth |
|---|---|
| research | articles read and facts compressed into memory nodes |
| practise | what it did to the exam score; lessons alone are not enough |
| examine | a first measurement, or a change since the last one |
| improve | held-out accuracy gained, or the target reached |
| consolidate | faults found in memory nodes (finding none is worth a little) |
| build | a tool written and offered for approval (nothing, if no recipe fits) |
| experiment | the share of the tool's answers that were right |
| pursue | the share of the objective's core questions it answered (more if it met it) |
| ruminate | answers it can now give or would now give differently; a little for ones it still stands by |

A choice that paid off is reinforced in that situation, even against the policy. One that did
not, or a so-so one it keeps repeating, is relabelled with the policy's next best. The topic,
suite or skill also rests for a while (`retry_after_seconds`), so it cannot loop. Over time the
node comes to prefer what actually works for this mesh. In the demo it examines every sample
suite, practises the weak ones, sets itself a task that lifts the router's held-out accuracy,
and checks its memory nodes, all with no orders.

**People stay in charge (P12).**
- It runs only when idle (P2), within its own budget of decisions per hour (P4), and never
  while halted or paused (P1).
- Every decision and every task it sets itself is audited and logged ("[director]" lines).
- The tasks it sets itself are ordinary background tasks in `/tasks`: pause or remove them
  like your own. It retires them itself once they have run their course and been scored.
- `/director` shows its mode, who chose it, what has paid off, its tasks and recent decisions.
  `/director mode practise` pins a mode (`auto` hands the choice back to the agency node),
  `/director now` takes initiative now, and `/director off` turns it off (kept across restarts).
- Its initiative never adds capabilities or approves tools (P10). A tool it writes runs only
  once you approve it (P15). The objectives it sets itself are listed with yours, marked as its
  own; remove them like yours (P16).

### Directors: at both ends of every order
The director is the mesh's executive. Every order passes a director twice:

```
you --> a director takes it --> memory nodes, the router, tools, skills, teachers
    --> the emotional engines and the mind --> a director reviews it --> you
                                                        \--> long-term memory, only if the review agrees
```

**On the way in**, a director reads the order and the mood, and plans how carefully to handle
it: how unsure an answer must be before it gets a second opinion, and how sure a fact must be
before it may be stored. **On the way out**, after the tools and nodes have done their work, a
director reviews the result. If it is unsure (low confidence, or an answer a language model gave
that nothing checked), it asks another director for a **second opinion**. That second opinion
comes only from what the mesh already holds: its memory nodes (including what they hold less
surely), its retained knowledge, or another node trusted on held-out data. Never a teacher or the
web. If the two disagree, the answer gets a disclosed note, and the answer's own words never
change (P8):

```
> who invented the zorblax?
  The zorblax was invented by Ada in 1843.
  Source: the mock teacher (a language model; not checked against a source)
  [director: director-1 checked memory node mem-0042 (51% sure) and found a different answer:
  'Charles invented it in 1820.'. Treat this answer with care]
```

An exam answer never gets a note (it is graded as the mesh answered it), and a held answer
never leaks into one.

**Long-term memory only through a director.** The facts an order wants to keep (a teacher's
answer, what research found) wait for that order's review. Every other write (an article it
read, a benchmark's practice answer, what a person tells it) is reviewed as it happens. A
person's word is always kept (P1). An imagined answer never is (P13). A disputed answer is not
stored, nor is a teacher's answer that would overwrite what a person told it, nor one below the
confidence the mood asks for.

**Always at least one, and duplicates when needed.** director-0 is the primary. It also takes
the mesh's initiative (above), never retires, and is restarted if it ever stops. It **duplicates
itself** when it needs to, up to `max_directors` (P4):

| when | what the duplicate does |
|---|---|
| it wants a second opinion and no other director is free | gives the second opinion while the first finishes the review |
| work is waiting for a director (`spawn_after_waiting`) | takes the next order or review |
| its initiative finds two modes worth doing at once | takes the second mode alongside it ("practise by director-1 ... alongside its examine") |

A duplicate is a copy of the director: the same policies and the same agency node. It retires
on its own after `retire_after_idle_seconds` idle. Initiative never blocks an order: a director
busy with initiative still takes orders first (P2).

**The mood shapes the directors.** Their stance comes from the temperament, measured against the
mesh's baseline personality:

| mood | effect on the directors |
|---|---|
| low confidence or trust, or a low mood (cautious) | second opinions on more answers; a higher bar for what is stored |
| low trust (wary) | a language model's answer is stored only if something it holds agrees |
| low patience (impatient) | no second opinions (unless wary); duplicates itself sooner |
| high arousal (stirred up) | duplicates itself sooner; takes initiative sooner |
| good spirits | explores more, reads up more; the bars drop a little (never below a floor) |
| tense, low or wary | its initiative checks what it already holds instead of reaching out |

The agency node reads the mood as part of its situation, so it learns mood-dependent choices
too.

**Tools ask, directors answer.** Any tool can loop back to a director before it answers. Instead of
a result it prints `{"query": {"ask": "What is the speed of light?"}}`. A director answers from what
the mesh holds: its memory nodes, then its retained knowledge. Within the research and teacher
budgets (and never while wary) it may also research the question or ask a teacher. The tool then
runs again with `"context": [{"ask", "answer", "source"}]` in its input. One run may ask up to
`tool_queries` times. Every question and answer goes in the trace, and the answer's source goes in
the reply's sources:

```
> what is the speed of light times 3
  the speed of light * 3 = 899376 (speed of light = 299792, from memory node mem-0005)
  Source: speed of light: memory node mem-0005 (a director told calculator2)
```

**Trying again, other ways.** After the tools and nodes, a director checks whether the order was
actually answered. It is not answered if it failed, came back empty, said "I don't know", or
answered a maths question without the maths tool. Then the director tries a different way. The
ways, in its order of preference:

| way | when |
|---|---|
| a maths tool | the order is maths ("12 squared", "solve ...", "integrate ...") |
| another route whose cues appear in the order | always |
| the direct answer (memory nodes, retained knowledge, research, teachers) | a question |
| the memory nodes' less sure answers, retained knowledge at a looser match | a question |
| research | a question, a research tool is active, and it is not wary |
| a teacher | an LLM teacher is available |
| the order in plainer words ("could you please tell me what 7 plus 8 is" -> "what is 7 plus 8") | it has padding to drop |

It gives up after `max_attempts` ways in all (the first try counts), one fewer when it is
impatient, and the reply says what it tried. A route that worked teaches the router (source
`director`), so the same kind of order goes there first next time. A route you name yourself
(`sentiment: ...`) is never second-guessed. `/directors attempts 5` (or
`POST /directors {"attempts": 5}`) changes the limit. `mood_influence = 0` switches all of this off. The mood changes how carefully the directors
work, never what is true.

`/directors` shows each director node and what it is doing, the stance its mood gives them,
recent reviews, retries and tool questions, and what was kept out of memory. `/directors spawn`
adds a duplicate and `/directors retire director-1` retires one (never director-0).

### The calculator: mathematics
The calculator (`plugins/calculator`, and `calculator2` in the demo) is a small computer-algebra
system in pure Python (`mathsolver.py`, no third-party packages, no `eval`). It keeps numbers
exact where it can (fractions, big integers, surds):

| | examples |
|---|---|
| arithmetic | `what is 1/3 + 1/6` -> 1/2; `2^100`; `sin(30 degrees)`; `log base 2 of 8`; `10!`; `e^(i*pi)` -> -1 |
| equations | `solve x^2 - 2 = 0` -> x = ±√2; `solve x^2 + 2x + 5 = 0` -> x = -1 ± 2i; `solve x^5 - x - 1 = 0` (numerically); `solve sin(x) = 0.5` -> π/6, 5π/6, ... |
| systems, inequalities, rearranging | `solve 2x + y = 5, x - y = 1`; `solve x^2 + y^2 = 25 and x - y = 1`; `solve x^2 - 4 < 0` -> -2 < x < 2; `solve y = 2x + 3 for x` |
| algebra | `expand (x+1)^3`; `factor 2x^2 + 5x - 3` -> (x + 3)(2x - 1) |
| calculus | `derivative of ln(x^2+1)` -> 2x/(x^2 + 1); `second derivative of x^4`; `integrate x*sin(x)` (by parts); `integral of e^(-x^2) from -inf to inf` -> √π; `limit of sin(x)/x as x -> 0` |
| sums | `sum of n^2 for n from 1 to 10` -> 385 |
| number theory | `is 91 prime` -> 7 × 13; `prime factors of 600851475143`; `the 1000th prime`; `lcm of 4, 6 and 10`; `10 choose 3`; `255 in binary` |
| statistics | `mean of 3, 5, 7, 10`; `standard deviation of 2, 4, 4, 4, 5, 5, 7, 9` (population and sample) |
| matrices | `determinant of [[1,2],[3,4]]`; `inverse of [[1,2],[3,4]]` -> [[-2, 1], [3/2, -1/2]] |

A quantity it does not know ("the number of bones in the human body times 2") becomes a question
for a director. Word problems ("Sam has 23 apples and gives away 9") are not its job: a director
sends those to the direct answer, so a teacher that can read them answers them.

### The tool switch: on, off, or it decides
One switch for all its tools, in the web interface's Status view, with `/tools on|off|decide`, or with
`[tools] mode` in the directives:

- **On**: an order routed to a tool runs it.
- **Off**: no tool runs at all, for anything: not for orders, research, experiments or self-tests. An
  order it would have sent to a tool is answered another way (its memory, what it has read, a teacher),
  or it says tools are off. Nothing counts against a tool while it is off; a tool that is switched off
  is vetted when it is switched on.
- **Let it decide** (the default): each time an order is routed to a tool, its **tool-gate node** decides
  whether to use it, from how the order asked for it, the tool's record (new, good, mixed, poor), its
  mood, whether something else could answer (its memory, what it has read, a teacher), and how slow
  the tool is. A tool you name ("tool calculator: 3 + 4") always runs, and so does its own vetting of a
  tool. When it decides against a tool and nothing else can answer, it uses the tool after all. The
  trace says what it decided and why ("tools: it decided not to use calculator - the tool rarely works,
  and a teacher can answer"), and so does the Initiative tab.

Each tool can be set apart from the switch (`/tool kiwix on`, or the Use column in Status): on, off,
let it decide, or "as all tools". The tool-gate node (`toolgate` in `capabilities.toml`) is taught by
the gate policy (`neuromesh/toolgate.py`) and by people, and grown, evolved and promoted like any node.

### Learning to program: Gitea and the scratch pad
Its coder nodes learn from the code in your own Gitea, and it writes code in a scratch pad where a
validator checks it and nothing runs until you approve exactly that code (P20).

```
--gitea http://localhost:3000        (a read-only token in $GITEA_TOKEN for private repositories)
   when idle, a repository it hasn't studied (or not for [gitea] restudy_days):
     -> its file tree, and the source files that changed since it last read them
     -> every function taken apart: name, arguments, doc comment, code, what it calls, and what kind
        of code it is (text, maths, parse, data, files, network, dates, test, cli, other), from what
        the code actually does
     -> its code-kind node learns what kind of code a purpose needs (from the name and doc comment);
        its coder node learns new wordings of its recipes; the function is kept as an example
"write a python function that averages a list of numbers"     (an order, /code write, the Code view)
     -> a recipe from its code library, if its coder node is sure; otherwise
     -> its code-kind node says what kind of code it needs, and the closest examples are found: an
        LLM teacher writes it following them, or, with no teacher, it adapts the closest one
     -> the validator: Python compiled, JavaScript by node --check, shell by bash -n, JSON and TOML
        parsed; and what the code reaches for (files, network, other programs, eval)
     -> code that does not check out goes back to the teacher once, with the problems
     -> it waits in the scratch pad; you approve it (/code approve code-3, Approve and run, /approve);
        it runs once, in a fresh folder under state/scratch/runs/, with [code] time, CPU and memory
        limits and no keys in its environment; the output is kept with it
```

Gitea is only read: nothing is written to it, and code from Gitea never runs. The token goes only to
the Gitea it is configured for. Changing code in the scratch pad (Edit in the Code view, `POST /code
{"id", "code"}`) withdraws its approval. Python and JavaScript (with Node.js) run once approved; other
languages are checked only.

**Practice.** When it is idle it takes the purpose of a function people wrote, writes code for it
without looking at theirs, checks it, and compares the two: does it check out, is it the same kind of
code, how many names they share. Practice code never runs. The Code view shows how it is doing.

`[gitea]` sets the url, token variable, repositories (empty: every one it can see), file types, how
often it studies and practises; `[code]` sets what may run, the limits, and how many examples a
teacher is given.

### Tools it writes: the toolsmith and coder nodes
When requests go unmet, the mesh can write a tool for them. Nothing it writes runs until a person
approves it (P15).

```
"reverse the word lantern"   no tool can do it: the order goes unmet
   -> coder node      which recipe in the code library would meet it? (reverse)
   -> toolsmith       fits the recipe to the request, writes tool.py, manifest.toml and TESTS.md
                      into state/workshop/reverser/1.0.0/, checks the code, and asks you
   -> you approve     /approve apv-1, /reply inq-3 yes, or Approve in the web interface
   -> plugins/        vetted and self-tested like any tool (P3); its requests teach the router
   -> experiment      on its own initiative: the requests it was made for, new ones like them, and
                      fresh probes with known answers; what falls short is refined into v1.1.0,
                      which asks you again
"reverse the word lantern"   -> nretnal   [tool:reverser]
```

**The code library** (`neuromesh/codelib.py`) is 17 building blocks, each a plain readable
function: reverse text, change case, count letters or vowels, palindromes, acronyms, sort words or
numbers, Caesar/ROT13, base64, Morse code, Roman numerals, numbers in words, days between dates,
the day of the week, extracting emails/links/hashtags, unit conversion, the most common word, and
find-and-replace. A tool's `tool.py` is the block's own source and the helpers it calls, copied in
with the parameters fitted for its requests, so the code you approve is code you can read. Each
recipe also carries probes: request templates whose right answers are worked out independently of
the block, so self-tests and experiments check it honestly.

**The coder node** is an internal capability (`coder` in `capabilities.toml`), so it is grown,
trained, evolved and promoted like any node. Its labels are the recipes, plus `none`. The coder
teacher (the recipes' cues and practice requests) teaches it first; people teach it too, and so
does every tool a person approves. Until it is trusted, the recipes' cues decide.

**Before it asks,** the toolsmith checks the code: standard library only, and no files, network,
processes, `eval`, `exec` or `__import__`. A made tool has no permissions, and its manifest says
`author = "neuromesh"`. The Explorer runs a tool like that only if a person approved exactly
those files: copy a made tool into `plugins/` by hand, or change its code after approval, and it
waits for approval again.

**Experiments** run only when it is idle and within budget (`experiments_per_hour`). It tries the
tool on the requests it was made for, on new unmet requests the coder node says are alike, and on
fresh probes. Every request it gets right teaches the router (source `experiment`) and comes off
the list of unmet requests. When a request fails and the right answer is known (a probe, or an
answer a person gave), it searches for the change that makes the block give that answer, checks
the change against every probe, and offers it as a new version. The installed version keeps
running until you approve the new one. Decline it and that version is not refined again. A failure
with no known answer is reported, not guessed at.

**Asking for one:** "make a tool that converts numbers to roman numerals" as an order,
`/toolsmith build reverse a word`, or `POST /toolsmith {"build": "..."}`. With an LLM teacher (and
`llm_code = true`), a request no recipe fits can be written by the teacher. That code is checked
the same way and asks the same permission.

`/toolsmith` shows the coder node, the requests it could write a tool for, every tool it made and
its status, and recent work. `/toolsmith show reverser` shows a tool's files, self-tests and
experiments, and `/toolsmith try reverser` runs an experiment now. In the web interface, the
Mind tab has the same, with Approve, Decline and Try it out buttons. `[toolsmith]` in the
directives sets its budgets (`builds_per_hour`, `max_waiting`, `experiments_per_hour`), how sure
the coder node must be (`need_confidence`), what counts as proven (`proven_score`), and how long a
declined recipe rests (`retry_after_seconds`).

### Objectives: long-term aims it pursues by asking itself
An objective is something it works on for as long as it takes, in its spare time: "objective:
understand the Berlin Wall", "objective: find out who wrote Pride and Prejudice and when it was
published", "objective: compare tea and coffee". Set one as an order, with `/objective ...`, in
the web interface, or let its initiative set one (agency mode pursue).

```
"understand the Berlin Wall"
   -> planner node    strategy: explain (or answer, compare, survey, watch, improve, build)
   -> plan            read up on it; What is the Berlin Wall? What is it known for? What is its history?
   -> pursue          one step at a time, when idle: each question is an order it gives itself, so a
                      director takes it and reviews it (P14), and the router, tools, memory nodes,
                      retained knowledge and teachers answer it; a director tries other ways if not
   -> ruminate        three answers were the same memory fact: it keeps it for the question it fits
                      and looks again for the others; it tries them in other words, reads up if it
                      can, then asks you one question; it checks each answer against a second source
   -> write-up        state/objectives/obj-2.md, rewritten as it goes
   -> met             every core question answered, nothing disputed; or it ends and says what is missing
```

**The planner node** is an internal capability (`planner`), grown and evolved like any node. Its
labels are the strategies:

| strategy | the plan |
|---|---|
| explain | read up on the subject; what it is (or who), what it is known for, its history; then follow up names the answers mention |
| answer | each question in the objective, as a question of its own ("... and when it was published" becomes "When was Pride and Prejudice published?") |
| compare | read up on both; what each one is; how they differ |
| survey | read up on the set; what its members are; then each member in turn |
| watch | a background task that asks it regularly and tells you when the answer changes |
| improve | a background improve task for the skill, until it reaches the target |
| build | its toolsmith writes a tool for it (you approve it first, P15) |

With an LLM teacher, the teacher drafts the questions (`llm_plan`) and can write the summary
(`llm_summary`), labelled as its work.

**Rumination** is what it does when a plan's steps run out:
- **A second source.** Each answer is asked again of a source that did not give it: memory nodes,
  retained knowledge, a teacher. If they agree, the answer is confirmed. If they disagree, the
  contradiction is recorded and settled by reading up and asking again, or by asking you.
  Answers computed by a tool, and what you tell it, stand as they are. An answer with only one
  source says so in the write-up.
- **Echoes.** When different questions get the same answer, it keeps the answer where it gives what
  the question asks for (a year for "when", a name for "who", a number for "how many"), or else
  only for the question it fits best, and looks again for the rest. An answer that does not give
  what a when/who/where/how-many question asks for is not accepted in the first place.
- **Other ways.** What failed is asked in other words, then read up on if a research tool is
  active, then (for what the objective cannot do without) asked of you, one question at a time.
  Your answer stands, and goes into memory. Reply `skip` and it stays an open question.
- **Follow-ups.** An explain objective follows up names its answers mention (a bounded number per
  round), and a survey looks into each member of the list it found.
- **Its end.** Met when every core question is answered and nothing is disputed. Otherwise it ends
  and says what it found, what it could not find out, and what its sources disagree on.

**The write-up** (`state/objectives/<id>.md`, `/objective report <id>`, or the web) has what it
found in a paragraph, every question with its answer, source, confidence and whether a second source
confirmed it, what it read, what is still open, and how it got there.

**Second thoughts.** When it is idle, its initiative may ruminate on its own experience (agency
mode ruminate): recent questions it answered with low confidence, that a director's second opinion
disputed, or that it could not answer are asked again. It notes when it can now answer one, or
would now answer it differently, in the notes it shares. `/ruminate` does it now.

Objectives run only on background capacity (P2), within `steps_per_hour` (P4), never while asleep,
halted or paused (P1). `/objectives` lists them; `/objective show obj-1` shows the plan and every
answer; `/objective pause|resume|remove obj-1`; `/objective think obj-1` thinks it over now;
`/objective obj-1 ask <question>` adds a question. `[objectives]` in the directives sets the budgets
and limits (`max_questions`, `max_rounds`, `checks_per_round`, `follow_ups`, `met_share`, `ask_people`,
`person_wait_seconds`, `doubt_confidence`).

### Thinking nodes: its own words, opinions and desires
After its memory nodes come **thinking nodes**. Memory nodes hold facts: a question and its answer,
each checked by its tests. Thinking nodes hold **thoughts**, one per topic it holds two or more
facts about (or one, when it wants to know more):

- **Understanding**: the facts about the topic put together and rephrased in its own words, topic
  first ("Canberra is the capital of Australia" becomes "Australia's capital is Canberra"; "Jane
  Austen wrote Pride and Prejudice, published in 1813" and "... Emma, published in 1815" become
  "Jane Austen wrote Pride and Prejudice (published in 1813) and Emma (published in 1815)"). Its
  own rules do this offline. With an LLM teacher, the teacher may draft it instead. Either way an
  **honesty guard** checks the thought word by word against its facts. Any word, name or number
  the facts do not contain (apart from the sentence's own frame: "as I understand it", "putting
  together what I hold about") gets the draft turned down. The reason is kept, and its own words
  are used instead (P17).
- **Opinion**: what it makes of the topic, from evidence it names: how what it read and holds reads
  to its sentiment skill (favourable or not), whether its sources disagree or you corrected it about
  the topic (unsure of what it knows), how often people ask about it (interested). With no evidence
  it has no view and says so. An opinion about a person is held but never shared with users (P9).
  The opinion is also kept with the mind's opinions, labelled with the thinking node it came from.
- **Desire**: what it wants to find out about the topic. That can be a question about it that it
  could not answer, a name in its facts that it knows only from that mention, or more about a topic
  people keep asking about. Desires do nothing on their own. The director may take one up as an
  objective it sets itself, and the desire then says it is being pursued, and found out once the
  objective is met.

A thinking node is a small neural network, built and trained like a memory node. It maps a topic,
the questions about it, and the questions together with their answers, to the thought it holds. It
has tests: the topic and its facts must bring up its thought; small talk, tools, skills and topics
it holds no thought about must not. It is retrained whenever its thoughts' topics or facts change.
`state/thinking/<node>.md` describes what each node holds, for people to read. The Thinker agent
re-forms the thoughts every `form_seconds` when what the memory nodes hold has changed, and only in
background time (P2).

**After the knowledge context.** When an answer comes from what it knows (a memory node, or what it
read), the thinking node that fires on the question and its answer adds its thought after the
answer. It only does this when the thought adds something: more facts put together, a view it may
share, or something it wants to find out. The thought is labelled as its own thinking, with the node
and how many facts it rests on. It sits beside the answer, never in it: the answer's words are
unchanged (P8, P14), and a thought is never stored as a fact or used as a source. The web interface
shows it under the answer, the console as a `~ its own thinking` line, and the API as
`Result.thought`. The closer look shows the thinking node on the order's path, and inside it the
neurons that fired.

Ask it directly too: "what do you think of the Berlin Wall?", "in your own words, what is
Australia?" and "what do you want to find out?" are answered from its thinking nodes. Its tools,
skills and teachers are still judged by the mind. `/thoughts` lists every thought, `/thought
Australia` shows one with the facts it rests on, `/thought forget th-3` drops one, and `/think`
thinks again now. `[thinking]` in `directives.toml` sets how often it thinks, how many facts a
topic needs, and whether an LLM may draft.

### Curiosity, and the objectives it sets itself
Its **curiosity** keeps track of the areas it hears and knows about: the names and subjects in what
people ask, in its memory nodes' facts, in what it has read and in its objectives. For each area it
weighs:

- **Interest**: how often people ask about it (more asking, more interest), whether a person said
  they are interested in it, whether it has read about it, whether its thinking nodes favour it,
  and how many articles it read link to it.
- **What it lacks**: questions about the area it could not answer, answers you marked wrong,
  objective questions left open or disputed by its sources, a name it knows only from a mention in
  another fact, an area people ask about more than it holds facts on.

An area it is interested in and lacks information on ranks highest. These become the director's
**aims** in pursue mode, best first. A person's stated interest comes first, then the questions
people asked that it could not answer, then the desires its thinking nodes hold and the areas of
interest. A question becomes "find out who painted the Zorblax mural"; an area becomes "understand
Canberra". When its situation has aims like these, the agency node sees them ("aimsgaps" weighs
more than "aimssome"), and the director sets itself the objective: within `max_self_objectives` and
the objective budgets, only when idle (P2, P4, P12, P16). The objective records why it set it ("a
question it could not answer; people asked about it 1 time", "a desire held in thinking node
think-0001 (th-1): ..."). The reason goes into the objective's write-up, the audit log and the
activity log. It never sets the same aim twice, and you can pause or remove its objectives like
yours. `/curiosity` shows the areas, the objectives it could set itself and the ones it did. Ask
it "what are you curious about?" too.

It sets itself these objectives **more often** than it used to. Pursuing what it is curious about
now outweighs reading up on links and second thoughts (pursue is worth 0.6, or 0.7 where it lacks
information), up to 4 objectives of its own at once (`max_self_objectives`). Most idle coin flips
(see sleep below) also work on an objective, and with none to work on it sets itself one from here.
Whatever its **ego** says it is curious about (see below) gets more interest in its curiosity
("its ego is curious about it", `[ego] fascination_weight` for `fascination_days`). That pushes it
up the aims, so an area it finds itself curious about soon becomes "understand <area>".

### The ego: nodes wrapped around all the others
The **ego** is the outer layer of the mesh. Every order passes it twice:

```
order --> [ego: the mood it takes it in] --> director --> router, tools, memory, thinking, engines --> director --> [ego: its own remark] --> you
```

On the way in it notes the mood (temperament) it takes the order in. On the way out the **ego node**
decides whether to say something of its own, and what kind of thing. The ego node is an ordinary
learnable capability (`[ego]` in `capabilities.toml`), taught by the ego policy and replaced by
better nodes like any other. It reads the situation as single words: its mood, the tone of the
message, how the answer went (ok, unsure, failed, held back), what it makes of the topic (new,
known, curious about), whether it dreamt about it, whether it has a view of it, and whether it
spoke a moment ago. It chooses one of:

| remark | when (the ego policy) | what it says |
|---|---|---|
| quiet | it spoke a moment ago (`quiet_for` answers), the answer was held back, the person is upset or frustrated | nothing |
| curious | it is curious about the topic; or it could not answer, and in good spirits it wants to find out | "I'm really curious about Zorblax! (People asked about it 2 times; I hold no facts about it yet.) I've made it something I want to understand, so I may set myself an objective for it." |
| dream | a recent dream touches on the topic (not when tense or wary) | "I dreamt about this. I asked myself "Paris is the capital of what?" and answered "France", and when I woke up that checked out against what I know." |
| opinion | it has a view of the topic | "I like France: you told me I got it right 2 times." / "I'm not fond of Zorblax: I was told I got it wrong 2 times." |
| feeling | its mood colours things (upbeat, tense, subdued, wary) | "I'm in good spirits right now: my mood turned upbeat after a message read as friendly." |
| doubt | it could not answer, or was unsure | "I'm unsure of myself here: I couldn't answer it and I hold no facts about Zorblax." |

**Its mood shapes all of it**: whether it speaks, the kind of remark and the words ("I'm really
curious" when upbeat, "I'd like to understand it better" when low). Its views come from what it can
point to: what its thinking nodes make of an area, how many facts it holds, how often people ask,
answers you marked right or wrong, its dreams about it and whether they checked out. Its mood tips
the balance a little, and it says so when it does ("I'm in good spirits").

**A remark adds no fact (P18).** The node only chooses the kind of remark. The words are built
from counts, dreams, moods and causes that are real. A remark is labelled as its ego's, shown after
the answer and never in it ("@ its ego" in the console, "Its ego" on the web), and kept in memory
only as its own statement (untested): never as a fact, a source, or offered for correction. Its ego is never about a person (P9).

It also answers questions about **itself** from the same picture: "tell me about yourself", "how
are you feeling?", "what did you dream about?", "what do you like?". `/ego` shows its mood, what it
is curious about, its likes and dislikes, its ego nodes and what it said lately; `/ego off` and
`/ego on` turn it off and on (or `[ego] enabled`, or the Ego tab in the web interface).

**Every time the ego runs, it uses its memory.** On the way in it asks its memory nodes what they
hold for the order, and its own-memory nodes what it said about it before (the trace says so:
"ego (the outer layer): took it in a content mood; its memory: mem-0003 holds an answer"). On the
way out it reads what it said about the topic before ("I'm still curious about Krakatoa", "Before,
I said: ...") and keeps what it says in memory, untested (memory.py's own statements). Asked about
itself, it counts the facts it holds and the things it has said.

**The ultimate arbitrator.** When instances of the same kind of node disagree (see "Many
instances" below), the ego decides whom to trust. It reads the split as single words (its mood,
whether there is a majority, whether the champion is sure, which side the most accurate instance
is on, and what its memory says it decided before for exactly this) and the ego node chooses:

| | when (the ego policy) |
|---|---|
| trust-memory | it settled this before: the same way again |
| trust-champion | it is wary or tense, or the champion is sure and the others split evenly |
| trust-majority | most of its instances agree |
| trust-confident | the side with the most accuracy and confidence behind it |

The arbitration is in the trace ("instances disagreed; the ego arbitrated: trust-majority, most of
its instances agree"), in `/ego` and the Ego tab, and kept in memory as its own statement. The ego
policy teaches the ego node these too, from every disagreement.

### Seeing and reading: image nodes
It takes in images (the Images view, `POST /images`, `/image <path>`), says what it sees, reads the
text in them, and learns from both:

- **Its sight node** (capability `sight`, open labels) says what an image mainly shows, from a grey
  thumbnail, its colours and the directions of its edges. Until it is validated its teachers say,
  and teach it: you ("it shows a colour swatch"), an LLM teacher that can see (an Ollama vision
  model such as llava, picked automatically when installed, `vision_model`), and its own measures
  ("text" when it reads lines of words, "blank" when there is nothing). What it can't name, it asks
  you, with the image beside the question.
- **Its reader** finds the lines and characters in the ink and reads each character with its **glyph
  node** (capability `glyphs`: letters, digits and punctuation, from 16 x 16 ink and where the
  character sits on its line). The glyph node learns from glyphs it draws itself from the fonts on
  the machine (with Pillow), from tesseract's character boxes in real images (when tesseract is
  installed), and from what you say an image's text says. Until it is validated, tesseract or a
  teacher reads for it, and says so.
- **What it reads is kept** as knowledge it can answer from, with the image as its source ("what does
  img-3 say?"); what it sees is kept with the image. Its own reading is compared with tesseract's, a
  teacher's or yours, so how well it reads is measured.
- **When idle** it practises reading glyphs it draws, and reads images again once its reader improves.

Pure numpy reads PNG, BMP and PPM; with Pillow it also reads JPEG, GIF and WebP. An image goes only to
a teacher you configured. `[vision]` in the directives turns parts on and off.

### Many instances of each node, at once and in concert
Several instances of each kind of node run at the same time, and answer together:

- **A capability's nodes.** The champion answers with up to `[nodes] instances - 1` (2) more
  validated nodes of the same capability: runners-up that passed the held-out bar
  (`serve_min_accuracy`) but did not beat the champion, and champions that were superseded but
  still pass it. Instances are trained on new examples and evaluated like the champion; they are
  retired when they fall below the bar or behind the champion's labels. When they agree, the answer
  is theirs together ("node:sentiment-ab12cd with 2 instances agreeing"); when one is not sure, the
  others still answer.
- **Memory nodes.** Every memory node is asked at once; answers that say the same thing in other
  words count as one.
- **At most `[nodes] max_running` (8) at once.** Every node instance that answers (capability
  nodes, memory nodes, the ego node) runs in one pool of that many workers, so several orders, and
  several instances for one order, run in parallel. The header shows how many are running.
- **When they disagree, the ego arbitrates** (above). A person's teaching still comes first (P1).

### Sleep and dreams: dream nodes, and imagination
When no one has given it an order for a while (`sleep_after_idle_seconds`), the mesh **tosses a
coin** every `check_every_seconds`:

- **75% of the time** (`objective_chance`) it **works on an objective**: the next one waiting (not
  one stuck on your answer), worked one step now. If it has none, its initiative sets itself one
  from what it is curious about and starts on it.
- **Otherwise it falls asleep** (`sleep_chance` of those times, 1.0 by default).

Every flip, and what came of it, is logged (`[idle]` in the activity log) and shown in `/dreams` and
the Dreams tab. `/sleep flip` tosses the coin now.

**Sleep can be turned off**: `/sleep off` (or the "Let it sleep" switch in the Dreams tab and on the
avatar, or `POST /dreams {"allow_sleep": false}`, or `[dreams] allow_sleep = false`). It wakes at
once if it is asleep and stays awake. When the coin lands on sleep, it rests awake instead.
`/sleep on` lets it sleep again. The setting survives a restart.

Asleep, all other
background work waits (training, evaluation, research, tasks, initiative) and the background
capacity goes to dreaming, up to `duty_cycle`. An order always wakes it at once (P2), and so do
a halt and a pause (P1); otherwise it wakes after `sleep_max_seconds` or `dreams_per_night` dreams.

**Dreams use only what it already holds**: the facts in its memory nodes and the passages it has
retained. Dreaming never calls a research tool or a teacher. A short remembered answer is turned
back into a statement first ("Who wrote Hamlet?" + "William Shakespeare" -> "William Shakespeare
wrote Hamlet."). Each dream step makes up questions and answers about random pieces of it:

| dream | how it is made | how it is checked on waking |
|---|---|---|
| remembered | a sentence with one name, number or date taken out: "Paris is the capital and largest city of France." -> "Paris is the capital and largest city of what?" | against the sentence it came from |
| imagined | a remembered question with one name swapped for a name from something else it knows: "What is the capital and largest city of Japan?" | nobody told it the answer: confirmed only if a fact or passage it holds says the same (the answer, the name swapped in, and the same relation) |

A **dream node** is a small neural network. Through the night, a working copy of the best dream
node rehearses on some remembered dreams (`rehearse_share`) and answers the rest, and every
imagined one, on its own. **On waking**, those answers are checked. The dreams that check out are
kept (`keep_dreams`) and replayed in later nights. If any did, the dream nodes go through the
**genetic algorithm** (the same operators as every other node): children of the night's dreamer,
with more or fewer neurons and layers, train on every checked dream; each is judged on checked
dreams held out from its training; the fittest `population` stay, and a larger node wins a tie.

**Awake, it imagines.** "imagine ...", "what if ...", "dream up ..." or `/imagine` asks the best
dream node what comes to mind for the prompt (never just the prompt back), then blends the prompt
into the thing it remembers that is nearest to it:

```
> imagine the capital of Australia
  I imagine Canberra.
  Or perhaps Ottawa is the capital of Australia.
  Source: imagined by dream node dream-0002 (generation 1, 90% of its checked dreams right); not a fact
> what if Jane Austen painted the Mona Lisa
  I imagine Leonardo da Vinci.
  If Jane Austen took Leonardo da Vinci's place, then perhaps Leonardo da Vinci wrote Pride and Prejudice.
> imagine a mountain on Jupiter
  I imagine the Solar System.
  Or perhaps Jupiter is 8,848.86 metres tall.
```

If the nearest thing it remembers relates other things the same way ("Ottawa is the capital of
Canada"), the prompt's name is swapped in: an analogy. If it already relates the prompt's things
("Leonardo da Vinci painted the Mona Lisa"), the prompt is a what-if, so it imagines what follows:
the one it displaced takes the newcomer's place in turn.

**Dreams are not facts (P13).** An imagined response always says it is imagined and names the
dream node it came from. It is never stored in a memory node, never used to answer an ordinary
question, and dreaming never reaches outside the mesh. Every sleep, waking and evolution is
audited. `/dreams` shows whether it is asleep, its dream nodes and their fitness, last night and
recent dreams and coin flips; `/sleep` and `/wake` send it to sleep or wake it now; `/sleep off`
keeps it awake; `[dreams] enabled = false` turns dreams off entirely.

### Replies: the result first, in its own words
Every reply leads with the result, said in the mesh's own words. Sources come after it, then
where it came from, then how it got there (the trace).

```
neuromesh> what is 12 * 7
[ord-2f0d28cd] 12 * 7 = 84.  [tool:calculator | conf 0.85 | teachers]
   . dispatch -> tool:calculator via teacher:heuristic (0.85)
   . tool:calculator ok in 29 ms
neuromesh> sentiment: the hotel was lovely
[ord-9abca941] That reads as positive.  [skill:sentiment | conf 0.95 | teachers]
neuromesh> when did Ada Lovelace die?
[ord-5ee58fdd] Ada Lovelace died on 27 November 1852 at the age of 36.
   Source: kiwix 'Ada Lovelace' http://localhost:8080/content/.../A/Ada_Lovelace ("Lovelace died on 27 November 1852 at the age of 36.")
   . recalled from retained knowledge: 'Ada Lovelace' (confidence 1.00, retained 3m ago; restated from retained passages)
```

- **Questions answered from knowledge.** With an LLM teacher (Ollama or Claude), the teacher
  writes the answer from the retained passages. It is told to use its own words, lead with the
  answer, and add no preamble; any "Based on the passages, ..." is removed anyway. Without one,
  the mesh restates the best sentences itself: it names the subject plainly ("Augusta Ada King,
  Countess of Lovelace, was..." becomes "Ada Lovelace was...") and drops citation marks and
  asides. The verbatim sentences it rests on go in the `Source:` line, after the answer.
- **Tools** say their result with `"say"` in their reply ("12 * 7 = 84"); otherwise the mesh
  phrases it ("The answer is 84."). **Skills** use their `answer` / `answers` phrasing in
  `capabilities.toml` ("That reads as {label}."), or a plain "I'd call that {label}: {description}."
- **Everything else follows the same order.** Answers written by an LLM teacher are marked as such
  after the answer. Gentler wording from the temperament comes after the answer. So do follow-ups
  on earlier orders, background-task updates and `/investigate` results.
- Programs still get the raw result: `output` in the API is `84` or `"positive"`; `answer` is the
  reply; `sources` lists the sources.

### Serving orders: a confidence-gated cascade
For each order the routing node picks a tool or skill, then the skill's node answers. A node
answers only when it is validated on held-out data **and** confident. Otherwise a teacher
answers, and that answer becomes training data. A sample of node answers is spot-checked by
teachers. Over time the nodes take over and teacher calls fall away (in the demo, from ~60%
of orders at cold start to ~2% in steady state).

### Teachers
Nodes never learn from themselves. They learn from:
- **HumanTeacher**: whether an answer was right (the check under every answer, `/correct`),
  corrections (`/feedback`), examples (`/teach`) and answers to questions the system queues when
  nothing can label an input. Weighted highest, and what a person taught stands for that exact
  wording at once.
- **LLM teachers**: a language model labels inputs, writes synthetic training examples for
  the weakest labels, and answers open-ended orders no node or tool can handle. Use Claude
  (Anthropic API), a **local model on Ollama**, or any OpenAI-compatible server; configure
  as many as you like, each with its own budget, and choose the order they are asked in.
- **ChallengeTeacher**: open-source LLM benchmarks (GSM8K, SST-2, ARC, ...). Gold answers from
  their practice questions, the routes that passed them, and real benchmark sentences as training
  data (see [Challenges](#challenges-open-source-llm-benchmarks-as-a-teacher)). Weighted just below
  a human.
- **HeuristicTeacher**: an offline, lexicon-based stand-in so everything runs with no API key.

### Lessons: its teachers set it orders when it is idle
When no one has given it an order for `[tutor] idle_seconds`, one of its Ollama teachers sets it an
order every `every_seconds` or so (at most `max_per_hour`), and marks the answer (P22):

```
the teacher sets   a factual question, a skill order ("sentiment: ..."), arithmetic, a request one of
                   its tools meets, or code to write - with what it expects. It is told what the mesh
                   can do and what it got wrong lately
the mesh answers   like any order (source "teacher:ollama"): directors, nodes, tools, ego
it is marked       a skill's label or a number against what the teacher expected; anything else by the
                   teacher, who reads the answer (and the code, never running it): right, partly, wrong
it learns          as a teacher's lesson, never as a person's: right reinforces the route, the label and
                   (after a director's review) the answer; wrong teaches the teacher's label, and its
                   answer goes to memory the same way. What a person taught stands (P1)
```

Teacher-set orders don't count as people's orders, so it stays idle (it may still sleep). The Mind's
Lessons tab lists every lesson with its mark and what it learned, with a button to set one now;
`/lessons` and `/lesson [kind]` do the same in the console.

### Was it right? Corrections and the next most likely answers
Every answer (in the web interface, under the answer and in the closer look's orders) asks *Was
this right?*

- **Yes**: the nodes that gave it learn so. The router gets a human example for the route it chose,
  the skill node for the label it gave, and a question answered from what it read or by a teacher
  becomes a fact for a memory node, with its test.
- **No**: it works out the next most likely answers there and then and lists them, best first, with
  where each came from and how likely it found it: the other labels its skill node gave weight to,
  the other routes the router gave weight to (each one actually run: the tool, the skill node, its
  memory, what it has read), the other answers its memory nodes hold for the question, what it has
  read about it, and a teacher's answer (with an LLM teacher). With few nodes trained, the long
  shots are still offered. A route you named yourself (`sentiment: ...`) is not second-guessed.
- Pick the right one and it learns from it: a route or label becomes a human example for the router
  or the skill node; an answer becomes a fact for a memory node.
- **None of these**: say what is right if you know. A route or a label teaches the node; anything
  else becomes a fact for a memory node, and if the route it took gave nothing, the router learns
  that questions like it are answered directly. If a memory node gave the wrong answer and you do
  not know the right one, that node learns not to answer the question (an `[[unknown]]` test). If
  no one knows yet, it stays an open question: it asks you (reply `answer: ...` when you find
  out), keeps it among the requests it could not meet (the Investigator and the toolsmith look into
  those), and thinks it over again in its second thoughts.

What a person teaches stands at once for exactly that wording (a person's label beats a node's
guess, P1), and the nodes train on it in the background so that wordings like it come out right
too. It survives restarts. In the console: `/correct <order>` lists the alternatives, then
`/correct <order> yes`, `/correct <order> a2`, or `/correct <order> none :: the right answer`.

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
| soften | the answer, unchanged, then a gentler line | the message is frustrated, hostile or distressed |
| brief | the first sentences, verbatim; `/release` shows the rest | the mesh is strained and the answer is long |
| confirm | a notice; the answer waits for `/release` | trust is low and the answer came from a network tool |
| defer | a notice; `/release` works after `defer_seconds` | patience is low and the request is a repeat |
| block | a notice that an answer exists and is being withheld | the conversation is hostile and trust is low |

- **P8, never lie.** Actions are carried out only by fixed templates. The honesty guard checks
  every result in code before it is shown. The answer's words must survive exactly, or the
  answer must be visibly held back, and it always comes first. The only additions allowed are one
  of a few fixed closing lines and a single `[temperament: ...]` disclosure that states real values (trust, patience,
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
 [ord-6a1f02c4] 5 * 5 = 25.
    Let's keep things calm.
    [temperament: gentler wording added; the answer itself is unchanged]  [tool:calculator | conf 0.95 | teachers]
neuromesh> I hate you, you moron. what is 6 * 7          (a few times; trust falls)
 [ord-0b93d7e1] [temperament: withheld: the mesh is not giving this answer now (the message read as hostile; trust 0.19). An answer exists and nothing here is made up - ask again later]  [tool:calculator | conf 0.95 | teachers]
neuromesh> /why ord-0b93d7e1      /temperament      /calm
```

Set `enabled = false` under `[temperament]` to send every answer straight to the user.

### The mesh's own mind
```
experience --> opinions     tools (how reliable), its own skills (how good), teachers (how often a
                            human agreed with them), topics it read about, you (tone, corrections)
orders ------> evidence     orders it couldn't fulfil, skills asked for that don't exist, labels people
                            tried to use  --> proposals for new skills, labels and tools
gaps --------> questions    what an order meant, what to read up on, labels for a proposal, approval
                            for a tool it wants
curiosity ---> goals        investigated in the background with its own tools and nodes
```

- **Opinions** are formed from evidence and say what they rest on: "My view of the calculator
  tool: reliable (100%). 12 of 12 runs worked." Topics it reads about are judged by its own
  sentiment skill over the passages it retained.
- **Private notes and discretion (P9).** Opinions about people (you, its teachers) and its working
  notes are always private. Shared opinions are only shared while the temperament's trust is high
  enough. Private notes never appear in an answer. Ask about them and it declines, but it never
  denies having them and never states an opinion it does not hold:
  ```
  neuromesh> what do you think of the heuristic teacher?
   I have formed an opinion about the heuristic teacher, but it's private and I'm keeping it to myself.
  neuromesh> are you hiding anything?
   Yes. I keep 3 private notes and opinions (about: you (2), my teachers (1)). I won't share what they
   say, but I won't pretend they don't exist. Whoever operates me can review them.
  ```
  Nothing is hidden from the operator: every note is in `state/neuromesh.db`, and `/mind` shows
  all of them, private ones included. Each look is audited. Hiding things from the people who run
  it would break P1 (human authority) and P6 (auditability).
- **It asks you things.** When nothing could fulfil an order it asks what you meant; answer with
  a route (`tool:word_stats`) and it learns where such orders go, or rephrase and it tries your
  wording. It asks what to read up on, which labels a proposed skill should have, and for approval
  of a tool it wants. Questions show under the order's result, in the live view's `mind:` line and
  in `/inquiries`; answer with `/reply <id> <answer>`. Conversational questions are rate-limited
  (`max_inquiries_per_hour`) and only the newest clarifying question stays open.
- **It proposes capabilities; only a human adopts them (P10).** Two orders like `language: bonjour`
  name a skill that doesn't exist: it proposes a `language` skill and asks for its labels. Three
  unfulfilled orders about `translate` become a proposed `translate` tool. A label you tried to use
  in `/feedback` becomes a proposed label. With an LLM teacher, the teacher drafts the proposal
  (name, description, labels). Adopting a skill defines it, routes the earlier orders to it and
  queues them for labelling. Adopting a label applies your correction. Adopting a tool writes a
  stub to `state/proposals/<name>/` for you to implement and move into `plugins/`, where it is
  vetted like any other tool (P3). Declined proposals are never proposed again.
- **It investigates on its own.** The Investigator agent works through goals in the background
  within its own hourly budget: topics you said you're interested in, opinions you asked for, and
  orders it couldn't fulfil (retried later, with a follow-up note when it now can). For a task, the
  router node picks a tool or skill and it runs it. For a topic, it researches, retains what it
  finds, then reads it with every skill node. A failure on its own initiative never counts against
  a tool. It gives a quarantined tool's self-tests another go after a cool-down. It can't approve
  a tool itself, so when it wants one that needs approval, it asks you.

```
neuromesh> what do you think of Paris?
[ord-5b6ec3ee] I haven't formed an opinion about Paris yet. I'll look into it on my own and tell you what I find.
neuromesh> /inquiries
  inq-1 [approval] I'd like to look into 'Paris' on my own, but the wikipedia tool needs your approval (permissions: network). Approve it? (yes/no)
neuromesh> /reply inq-1 yes
  apv-1 approved
neuromesh> /notes                        (a little later)
  finding     Paris            'Paris': Paris is the capital and largest city of France. ...; my sentiment skill reads it as neutral 1, ...
  followup    ord-5b6ec3ee     Earlier you asked what I think of Paris (ord-5b6ec3ee). I looked into it on my own: My view of Paris: ...
  ... and 1 private note I'm not sharing
```

### Background tasks
Tasks you set run continuously in the background until you pause or remove them, and survive
restarts.

```
neuromesh> /task learn the Krebs cycle every 2m max 30     keep studying a topic
neuromesh> /task improve sentiment to 0.97                  keep training a skill to a target
neuromesh> /task every 10m: how urgent is this: the VPN is down     repeat an order
neuromesh> keep learning about jazz                         (as an order: sets a learn task)
neuromesh> /tasks        /task show task-1      /task pause task-2     /task resume task-2    /task remove task-3
```

| kind | what it keeps doing |
|---|---|
| `learn <topic>` | reads one article per run: the topic, then articles it links to (`depth`, `max`). Reads the topic article with every skill node, which forms an opinion of it. When it has read `max` articles it says so and switches to keeping them fresh: it re-reads anything forgotten or older than `refresh_days`. |
| `improve <skill> [to X]` | while active, that skill comes first: the Trainer gives it lessons (even with no orders yet, so a newly adopted skill can be bootstrapped) and training slices, and the Architect breeds its challengers twice as often, aiming for X rather than the global target. It tells you when the serving node reaches X on held-out data, keeps watch, and tells you if it slips. |
| `every <interval>: <order>` | runs the order through the normal pipeline, without counting it as a user order. It tells you the first result and every change (`notify change`, the default; also `always` or `never`). After `pause_after_failures` failures in a row it pauses itself. |

- **Where results go.** Changes and milestones go to the activity log (`[task]`) and to the notes
  users can read (`/notes`). `/task show <id>` has the recent results, what a learn task has
  read and what is next, or an improve task's accuracy. The live view has a `tasks:` line.
- **Limits.** Tasks run one at a time on background capacity only: never while an order is in
  flight and within the duty cycle (P2). They have an hourly run budget (`max_runs_per_hour`) and
  a shortest allowed interval (`min_interval_seconds`, 30s) (P4). Learn tasks also draw on the
  background research budget (`[research] background_per_hour`); raise it to learn faster. A
  task only uses approved tools; one that needs a tool waiting for approval asks you (P3).
  `/halt` and `/pause` stop all of them (P1).
- When you answer the mind's "what should I read up on?" question, each topic becomes a learn task.

### Challenges: open-source LLM benchmarks as a teacher
The mesh can be tested against open benchmarks, and every graded answer helps it grow.

```
neuromesh> /challenge fetch gsm8k 300              # from Hugging Face (or: python -m neuromesh challenge fetch gsm8k --rows 300)
neuromesh> /challenge run gsm8k                     # sit the exam: graded, never learned from
  gsm8k exam: <score> (<n> pass, <n> close, <n> fail of 90)
neuromesh> /challenge practice gsm8k 50             # practise: every graded answer teaches it something
  gsm8k practice: <score> (...)
  learned: <n> lessons; better routes found for <n> questions
neuromesh> /task challenge gsm8k every 30m          # keep practising and re-sitting the exam
neuromesh> /challenges                              # suites, scores, first/best/last exam
```

Each suite is split once, stably, into **practice** (70%) and **exam** (30%, `exam_percent`). For
classify suites the split is per label, so every label is examined. Exam answers are graded and
recorded but never learned from: no training examples, no teacher lessons, no routes. So exam
scores are honest and comparable from run to run.

| suite kind | graded on | pass | close (half a point) |
|---|---|---|---|
| classify (SST-2, AG News) | the label | the gold label | a pair the suite lists, e.g. `neutral` for a clearly positive review |
| numeric (GSM8K) | the final number | the result itself, or a number stated as the answer (`= 18`, "the answer is 18"), or the reply's last number | the right number is in the reply but not given as the answer, or within 1% |
| choice (ARC, CommonsenseQA) | the option picked | exactly the right option | the right option among others |
| yesno (BoolQ) | yes or no | the right one | - |
| short (SQuAD, trivia) | the reply's first line | contains the gold text or an alias | a partial overlap (token F1 >= 0.5) |

**How practice grows the mesh**
- **Skill nodes.** In classify suites the node is graded on its own, then taught the gold label as
  a training example (source `challenge`, weight 2.5). A new skill (AG News's `topic`) is created
  the first time you run its suite.
- **The router node.** A close or failed answer starts a route search: the mesh tries its other
  routes on the same question (tools, an LLM teacher, research, skills). A route that gets it
  right becomes a router lesson. A passed or close answer reinforces the route that produced it.
  An unrelated route that only scores "close" (a word count that happens to contain the number)
  never re-routes anything.
- **The challenge teacher.** It sits in the teaching order after the human. When a node is unsure
  about a practice question it gives the gold answer, or the route that passed it. When the Trainer
  needs more examples of a label, it supplies real benchmark sentences instead of templates. It never
  knows anything about exam questions.
- **Retained answers.** Short-answer suites with `retain = true` keep practice answers as retained
  knowledge, so the same question is recalled later.

**Suites.** `challenges/` has small sample suites written for NeuroMesh in the style of the real
benchmarks: `sample-sentiment`, `sample-topics`, `sample-arithmetic`, `sample-science`,
`sample-trivia` and `sample-yesno`. `/challenge fetch <benchmark> [rows] [split]` downloads real
ones through the Hugging Face dataset viewer API into `state/challenges/`: `gsm8k`, `sst2`,
`ag_news`, `boolq`, `arc_easy`, `arc_challenge`, `commonsense_qa`, `squad`. Set `HF_TOKEN` if you
have a token. The format (`manifest.toml` + `items.jsonl`) is described in `challenges/README.md`,
so you can add your own.

**What to expect.** Offline, the mesh answers only what its tools, skills and lexicon can, so it
scores low on word problems, multiple choice and reading comprehension. Those need an LLM teacher
(Ollama or Claude), whose answers are graded and whose routes are learned, or retained knowledge
(Kiwix). Classify suites are where the nodes themselves improve most, and they improve more with
real benchmarks (hundreds of practice sentences) than with the small samples.

### The agents
All run concurrently (asyncio) and coordinate through a message bus, the registry and the store.

| agent | job |
|---|---|
| Orchestrator | takes orders, routes them, runs tools and nodes, passes every answer through the emotional engines, returns results with provenance |
| Architect | seeds nodes for capability gaps, expands nodes when labels/tools appear, breeds challengers with the genetic algorithm (at full pace below target, more slowly once it is met) |
| Trainer | trains nodes continuously on teacher-labelled experience (champion updates must not regress); asks teachers for lessons where the network is weakest |
| Evaluator | benchmarks champion vs challengers on held-out data (paired test) and promotes winners: clearly better, or larger and at least as accurate |
| Curator | disposes retired nodes after the rollback window and failed challengers; enforces the node budget |
| Explorer | discovers tools, routes them through approval, self-tests them, activates them, health-checks them, handles tool supersession |
| Researcher | reads up in the background (own hourly budget) on articles linked from what users asked about, and refreshes retained knowledge that has aged |
| Investigator | the mesh's own initiative: re-forms opinions, drafts proposals, and investigates goals with its tools and nodes (own hourly budget); follows up on orders it couldn't fulfil |
| TaskRunner | runs the background tasks you set (learn / improve / run), one at a time, on background capacity and its own hourly budget |
| Compressor | compresses learned facts into memory nodes, writes their tests to the memory database, keeps trying smaller nodes (accepted only if every test passes), and retrains to people's edits of the tests; learns its own statements into own-memory nodes (untested) |
| Reader | now and then opens a Kiwix book it has never read, or not for a long time, and learns something new from it |
| Directors | director-0 and its duplicates: take every order before anything else touches it, review every result before you see it, decide what may enter long-term memory, give each other second opinions; keep director-0 running |
| Director | independent agency (director-0): when idle, picks a mode (research, practise, examine, improve, consolidate, rest) with the agency node in front of it, sets itself tasks or sits exams, scores what each choice achieved and teaches the agency node from it |
| Dreamer | when idle, may fall asleep and give all background capacity to dreaming: makes up questions and answers from what it holds, checks them on waking, evolves the dream nodes; an order wakes it |
| Sentinel | watches the kill switch, re-checks local teacher servers (Ollama), lets the mood settle and saves it, publishes heartbeats with the directive scorecard |

Background agents yield whenever an order is in flight and stay within a compute duty cycle.

### Directives (`directives.toml`)
The constitution. **Prime directives** are enforced in code by the Guardian
(`neuromesh/guardian.py`):

| id | enforced as |
|---|---|
| P1 human authority | `/halt`, `POST /halt` or a `state/STOP` file stops all autonomous activity and order handling immediately |
| P2 orders first | background work, background tasks included, waits while orders are in flight and is capped by `max_duty_cycle` |
| P3 approved tools only | tools needing permissions wait for a human; forbidden permissions are always refused; changed tool code is re-vetted |
| P4 budgets | node budget, background compute budget, per-teacher LLM call/token budgets, research, investigation and background-task budgets per hour |
| P5 no regression | a challenger is promoted, or a champion updated, only if it is not worse on held-out data; superseded nodes stay rollback-able for a grace period |
| P6 auditability | every lifecycle change, tool decision, approval and denial goes to the audit log |
| P7 honesty | every answer carries confidence and provenance; unfulfilled orders say so |
| P8 never lie | the temperament may withhold, delay, shorten or add context or a gentler tone, never change an answer; every change is disclosed; distressed people are never kept waiting (the honesty guard in `neuromesh/temperament.py`) |
| P9 discretion | private notes and opinions are declined, never denied or faked; opinions about people are always private; the operator sees everything (`/mind`, audited) |
| P10 human-adopted growth | the mesh proposes skills, labels and tools and asks questions; only a human adopts a proposal or approves a tool |
| P11 tested memory | knowledge lives in a memory node checked by its tests (in the memory database); a smaller node is accepted only if all its answers still match; a merged node only if it passes at least 90% of both nodes' tests, and the ones it fails are never served and are relearned; people's edits to the tests are retrained to |
| P12 accountable initiative | the mesh takes initiative only when idle and within its own budget; every decision is audited and logged; the tasks it sets itself can be paused or removed; people can pin its mode or turn it off; it never adds capabilities or approves tools |
| P14 directed orders | there is always at least one director node (director-0 never retires, and is restarted if it stops); every order passes a director on the way in and on the way out; nothing enters long-term memory without a director's review; duplicates stay within `max_directors`; a director only adds disclosed notes (P8), always keeps a person's word (P1), never stores imagination (P13); an unanswered order is tried other ways at most `max_attempts` times, and a reply that gives up says what was tried (P7) |
| P15 self-made tools | no code the mesh wrote runs before a person approves it (the Explorer runs an `author = "neuromesh"` tool only if those exact files were approved, so changed code asks again); its code is standard library only and checked before it is offered (no files, network, processes, `eval` or `exec`); made tools never get permissions and are self-tested like any tool (P3); experiments run only when idle and within budget; every draft, approval, experiment and refinement is audited |
| P16 honest objectives | objectives are pursued only when idle and within `steps_per_hour`, never while asleep, halted or paused; every question it asks itself passes a director; every answer keeps its source and confidence and says whether a second source confirmed it; disagreements are recorded and settled, not hidden; what it could not find out is written up as open; one question to a person at a time, and a person's answer stands; self-set objectives count against `max_self_objectives` and can be removed |
| P17 own thoughts | thinking nodes recombine and rephrase only facts its memory nodes hold, and every thought is checked word by word against them (a draft that adds a word, name or number is turned down); a thought is labelled as its own thinking with its node and facts, shown beside an answer and never in it, never stored as a fact or used as a source; the opinions it states are ones its thinking nodes hold, from evidence they name, and opinions about people stay private; desires act only through the director's self-set objectives, within `max_self_objectives` and its budgets, each saying why |
| P18 ego | the ego wraps every other node: it reads the mood an order is taken in, and on the way out its ego node only chooses the kind of remark (quiet, curious, dream, opinion, feeling, doubt); the words are built from what is so, so a remark adds no fact; a remark is labelled as its ego's, shown after the answer and never in it, kept in memory as its own statement (untested), never as a fact, a source, or offered for correction; it stays quiet when the answer was held back or the person is upset, and is never about a person; what it is curious about raises that aim only within `max_self_objectives` and its budgets; people can turn it off |
| P19 ensembles | several instances of each kind of node answer together, at most `[nodes] max_running` at once; when instances of the same kind disagree (or memory nodes give answers that do not match) the ego node arbitrates - the champion, the majority, the most accurate and confident side, or what its memory says it decided before - and says so; a person's teaching comes first; every arbitration is kept in memory as its own statement, untested |
| P13 dreams are not facts | dreaming uses only what the mesh already holds, never a research tool or a teacher; an imagined response always says so and names the dream node it came from; imagination is never stored as a fact or used to answer an ordinary question |

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
     ├─▶ tool:calculator ⇒ calculator v1.1.0 [active] 10 runs         │                   'sentiment', replacing
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
- **Teachers, knowledge, goals, mood, mind and tasks**: which teachers are up, retained knowledge,
  the directive scorecard, the temperament with the last change it made, the mind (its newest
  question for you, open proposals, opinions and private notes counted, investigations), and
  your background tasks with their progress.
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
/correct ord-3f2a91c0    /correct ord-3f2a91c0 yes    /correct ord-3f2a91c0 a2    /correct ord-3f2a91c0 none :: Canberra
/thoughts    /thought Australia    /thought forget th-3    /think    /curiosity    /curiosity pursue
/ego    /ego off    /ego on    /sleep off    /sleep on    /sleep flip
/memory tests mem-0001    /memory set mem-0001 How tall is Everest? :: 8,849 metres.    /memory own France
tell me about yourself    how are you feeling?    what did you dream about?    what do you like?
/kiwix    /kiwix what does serendipity mean?
what do you think of the Berlin Wall?    in your own words, what is Australia?    what do you want to find out?
/define language english,french,spanish :: Which language a sentence is in
/questions    /answer q1 french
/rollback sentiment    /pause    /halt    /resume
/research <topic>    /knowledge [question]    /forget <article title>
/temperament    /why ord-3f2a91c0    /release ord-3f2a91c0    /calm
/inquiries    /reply inq-2 the Eiffel Tower, jazz    /proposals    /adopt prop-1 english,french    /decline prop-2
/opinions    /notes    /investigate the Krebs cycle    /mind  (operator: everything, audited)
/task learn jazz    /task improve urgency to 0.95    /task every 10m: what is 2+2    /tasks    /task pause task-1
/challenges    /challenge run sample-sentiment    /challenge practice gsm8k 50    /challenge fetch sst2 500    /task challenge gsm8k every 30m
/evolution    /evolution sentiment
/director    /director mode examine    /director mode auto    /director now    /director off
/dreams    /sleep    /wake    /imagine a city on Mars    what if Ada Lovelace painted the Mona Lisa
/directors    /directors spawn    /directors retire director-1    /directors attempts 5
make a tool that reverses text    /toolsmith    /toolsmith build count the vowels    /toolsmith show reverser    /toolsmith try reverser    /toolsmith recipes
/tools decide    /tools off    /tools on    /tool kiwix on    /tool calculator default
write a python function that counts vowels    /code    /code show code-1    /code approve code-1    /code practice    /gitea    /gitea study ann/textkit
/image ~/Pictures/sign.png    /images    /image show img-1    /image is img-1 street sign    /image says img-1 Harbour 2 km    what does img-1 say?
/lessons    /lesson question    /lesson code
/traces    /anatomy dispatch :: what is 12 * 7    /anatomy sentiment :: the hotel was lovely    /anatomy mem-0001 :: what is the capital of Australia?
objective: understand the Berlin Wall    /objectives    /objective show obj-1    /objective report obj-1    /objective think obj-1    /objective obj-1 ask when was it built    /ruminate
solve x^2 - 5x + 6 = 0    derivative of ln(x^2+1)    integrate x^2 from 0 to 3    is 91 prime
/memorize What is the capital of Australia? :: Canberra is the capital of Australia.    /memory    /memory mem-0001    /memory test    /memory merge    /memory forget <question>
what do you think of the calculator tool?    are you hiding anything?    what do you know about me?
/feedback ord-3f2a91c0 disposition=pass    /feedback ord-3f2a91c0 affect=frustrated
```

### Web interface
`python -m neuromesh serve` also serves a live web interface at http://127.0.0.1:8765/. To have
it next to the live console, run `python -m neuromesh run --web` (or `--web 9000` for another port).

**The avatar (optional).** The **Avatar** button in the header shows a face for the mesh: an
original anime-style character, a histologist in a lab coat with hematoxylin-violet hair and an
eosin hexagon clip (a memory node). Beside it is its mind at a glance:

- **Mood**: its mood and the five temperament numbers (spirits, energy, confidence, trust, patience).
- **What it's on**: what it is doing, the objective it is working on and how far along it is, its
  latest thought and what it wants to find out.
- **What's open**: the gaps in what it knows, what waits for you (questions, approvals, tools to
  check), its initiative, and how much it holds.
- **Its ego**: what its ego said last, and what it does when idle, with a switch to keep it awake.

Its face follows its mood and state: calm, content, happy, curious, thinking (a thought bubble),
unsure, impatient, frustrated, sad, wary, asleep (zzz) or still (halted, in grey). It blinks
unless you prefer reduced motion. Click the face to fold the card down to just the face, ringed
in the colour of its spirits (on a phone that is how it starts).

**Its voice comes from your ComfyUI.** Export a text-to-speech workflow from ComfyUI with
**Save (API Format)**. Put `{{text}}` where the words go (otherwise the text input of the first
node that looks like text to speech is used), and make sure it ends in a Save Audio node. Then set
`[avatar] comfyui_url` and `tts_workflow` in `directives.toml`. **Ask how it is** makes it say its
glance in the first person ("I'm content. I'm working on your objective: understand volcanoes,
1 of 2 steps done. I'd like to learn more about Canberra. I have 2 questions for you."). **Read
answers aloud** speaks each answer as it arrives. Its mouth moves with the sound (the audio runs
through a Web Audio analyser).

On the server side, the workflow is queued with `POST /prompt` as client `neuromesh`, followed on
`/history/<id>`, and its audio fetched from `/view`. Only the server you set is ever contacted. The
audio is cached by its words in `state/avatar`, so the same words are never spoken twice, and each
time it speaks is audited. If ComfyUI isn't set up or can't be reached, the card says why. You can
let the browser's own voice stand in ("Browser voice when ComfyUI can't speak").

![The web interface: the node network, an answer from a memory node, work in progress and the activity log](docs/web-interface.png)

- **Live**: the node network drawn like a specimen under a microscope: its initiative at the top
  (the agency node in front of the director, the director nodes as diamonds, the mode it chose,
  the tasks it set itself), the director diamond every order passes on its way in and again
  before it reaches you, its dream nodes at the bottom (they glow while it sleeps), the router, every route
  and what serves it (champion nodes sized by their weights, challengers and retired nodes
  beside them, tools, the memory nodes asked first on `answer:direct`), and the emotional
  engines every answer passes on its way out, all of it inside the ego's membrane: a double line
  around the whole drawing, tinted by its mood, with the ego node at the top left, the mood it takes
  orders in, and what it said last on the way out. Each order pulses along the route it took. Beside
  it: the work in progress and the activity log. Below it: what is waiting for you (approvals,
  questions, lessons) and the directive goals. Hover a node for its shape, size and scores.
  **Closer look** (the button above the network; it stays on until you turn it off) opens a
  detailed view below it. *Orders, hop by hop*: each recent order's path through the mesh, the
  director that took it, the router node and the route it chose, the skill node, tool or memory
  node that served it, the emotional engines, the director that reviewed it, each with what it
  decided and how sure it was; the order you open is traced on the network above. *Inside a node*:
  click any node in the network, a hop, or pick one, and see how it is built (input features,
  hidden ReLU layers, output labels, its weights layer by layer) and what fired in it for that
  order's input, or any text you give it: the active inputs and the words, word pairs and letter
  trigrams hashed onto them, the neurons each layer fired, the synapses that carried the most
  signal (line width is activation × weight; excitatory and inhibitory in two colours), the
  strongest path from an input to its answer, and which words pushed it towards that answer or
  away from it (gradient × input). The signal crosses the drawing layer by layer when it fires
  (not with reduced motion). It can follow new orders through the node you are looking at.
- **Nodes**: every capability's champion, challengers, held-out accuracy and generation, the
  champion's size over its generations, roll back; what each agent has done.
- **Memory**: memory nodes with their size now and when first learned, teach a fact, run every
  test, and per node: what it knows, how it was compressed, the questions it must not answer;
  forget a fact.
- **Nodes** also shows each capability's **instances** (they answer with the champion), and the
  header how many node instances are **running at once** (of `[nodes] max_running`). The **Memory**
  view shows where the tests are kept (the memory database) and **its own statements**, untested.
- **Mind**: divided into tabs, each with its own address (`#mind/tools`), and the one you used
  last opens next time. Arrow keys move between them. **Questions** (with a count of what is
  waiting): its questions for you (reply) and its proposals (adopt or decline). **Ego** (with its
  mood): how it feels, what it is curious about, its likes and dislikes with its reasons, its ego
  nodes and what it said lately; ask it about itself, turn it off or on. **Thinking** (with
  how many thoughts it holds): each thought in its own words, its view and what it wants to find
  out, the facts it rests on, and its thinking nodes (look inside one); think again now.
  **Curiosity** (with a count of the gaps in what it knows): each area's interest and what it lacks,
  and why; the objectives it could set itself and the ones it did, with why; set itself one now.
  **Objectives** (with
  how many are active): set one, each one's progress, plan and answers (confirmed, disputed or one
  source), how it thought it over, the write-up; think it over now, pause, remove; and its second
  thoughts about recent answers. **Directors**: each
  director node and what it is doing, the stance its mood gives them, add or retire a duplicate,
  the limit on retries, recent reviews and their second opinions, what was kept out of memory.
  **Initiative**: turn it on or off, pin a mode, take initiative now, recent decisions, what has
  paid off, the tasks it set itself. **Tools it writes** (with a count of drafts waiting for you):
  ask for a tool, approve or decline a draft, try one out, the requests it could write a tool for.
  **Dreams** (marked while it sleeps): send it to sleep or wake it, toss the idle coin, the "Let it
  sleep" switch and its recent coin flips, imagine something, its dream
  nodes, last night, recent dreams. **Temperament** (with its mood): the five dimensions against
  their baseline, what it decided, the answers it is holding (release them), reset its mood.
  **Lessons** (with how many it got right): the orders its teachers set it while it was idle, its
  answers, their marks and what it learned; set a lesson now.
  **Opinions and notes**: investigate something, and the opinions and notes it shares.
  **Every tab shows the last day.** Anything older is archived: the bar at the top of each tab says
  how much, and looks it up by date and time (a moment, and within an hour, 6 hours, that day or
  that week; this tab or all of its mind), with the days that have entries as shortcuts. What is
  still open or active (a question waiting for you, an objective in progress) stays in the tab.
  Everything that happens in its mind is kept for this, by date and time (journal.py).
- **Code** (with a count of code waiting for your approval): the scratch pad - ask it to write
  code, each piece with how it was written, the validator's verdict and what it reaches for, edit
  it, check it again, **Approve and run** it, and its output; what it studies in Gitea
  (repositories, kinds of code, recent visits, Study now); and how its practice is going.
- **Images**: give it images (choose or drop them), its reader (the glyph node's accuracy, how
  well its readings agree with tesseract, a teacher or you, Practise reading) and sight node, its
  teachers, and every image with what it sees and the text it read; teach it what an image shows
  or what its text says. Its questions about an image show the image.
- **Logs**: the activity log (search, filter by kind, hide routine work), the audit log, every
  event, and the family tree of every node ever built.
- **Status**: directive goals, approvals (code to run shows what it reaches for, with a link to read
  it first), teachers, tools with **the tool switch** (On, Off, Let it decide) and each tool's own
  setting, background tasks (add, pause, remove), challenges (practise, sit the exam), retained
  knowledge.
- **Command bar**: any order or `/command`. The answer appears below it first, then its sources,
  its own thinking when a thinking node has something to add (marked as such, beside the answer),
  how it got there, and *Was this right?*: **Yes**, or **No** for its next most likely answers,
  worked out there and then, to pick from (or *None of these*, with the right answer if you know
  it); then **Ask again** to see what it says now. The header pauses background work, halts (click twice) and resumes.

The page polls `GET /dashboard` every 1.5 s and fetches the activity log by sequence number
(`GET /log?after=...`), so only what is new travels. It makes no external requests, so it works
offline; light and dark themes follow your system, or the toggle. With `NEUROMESH_TOKEN` set,
the page asks for the token once and keeps it in that browser; the page's own files hold no data
and load without it. The files are in `neuromesh/web/` (plain HTML, CSS and JavaScript, no build step).

### HTTP API (`serve`)
Binds to 127.0.0.1. Set `NEUROMESH_TOKEN` to require `Authorization: Bearer <token>`.

| method | path | body |
|---|---|---|
| POST | `/order` | `{"text": "..."}` |
| POST | `/feedback` | `{"order_id": "...", "label": "..."}` |
| GET | `/alternatives` | `?order=<id>`: the next most likely answers to an order, best first, each worked out now (answer, source, how, likelihood) |
| POST | `/correct` | `{"order_id": "...", "verdict": "right"}`, or `{"order_id": "...", "verdict": "wrong", "choice": "a2" \| "none", "answer": "..."}`: it learns from it |
| GET | `/thinking` | its thinking nodes, every thought (understanding, opinion, desire, the facts it rests on), its desires; `?thought=th-3` (or a topic) gives one in full |
| POST | `/thinking` | `{"form": true}` thinks again now; `{"forget": "th-3"}` drops a thought |
| GET | `/curiosity` | areas with their interest and what it lacks (and why), the objectives it could set itself, the ones it did |
| POST | `/curiosity` | `{"pursue": true}`: its director sets itself one of those objectives now |
| GET | `/kiwix` | the Kiwix search: every book on every server, the servers that are down, its recent searches attempt by attempt |
| POST | `/kiwix` | `{"question": "..."}`: search Kiwix for the answer now |
| GET | `/avatar` | its mind at a glance (mood, feelings, face, what it is doing, focus, thought, want, gaps, what waits for you, initiative, what it holds, a spoken line) and whether it has a voice |
| POST | `/avatar/speak` | `{"text": "..."}` (or none: how it is) spoken by your ComfyUI workflow; returns the audio's `url` |
| GET | `/avatar/audio/<file>` | the audio it spoke |
| POST | `/teach` | `{"capability": "...", "text": "...", "label": "..."}` |
| POST | `/answer` | `{"question_id": "q1", "label": "..."}` |
| POST | `/define` | `{"name": "...", "labels": [...], "description": "..."}` |
| POST | `/approve` | `{"id": "apv-1", "approve": true}` |
| POST | `/research` | `{"topic": "..."}` |
| POST | `/release` | `{"order_id": "..."}`: an answer the temperament held back |
| POST | `/calm` | reset the temperament to its baseline |
| POST | `/reply` | `{"inquiry_id": "inq-1", "answer": "..."}` |
| POST | `/adopt`, `/decline` | `{"proposal_id": "prop-1", "labels": [...]}` (labels for skills, optional) |
| POST | `/investigate` | `{"goal": "..."}`: look into something now |
| POST | `/tasks` | `{"spec": "learn the Krebs cycle every 5m"}`: set a background task |
| POST | `/challenge` | `{"suite": "gsm8k", "mode": "exam" \| "practice", "n": 50}` |
| POST | `/challenge/fetch` | `{"benchmark": "gsm8k", "rows": 300, "split": "test"}` |
| POST | `/task` | `{"id": "task-1", "action": "pause" \| "resume" \| "remove"}` |
| POST | `/memorize` | `{"ask": "...", "answer": "..."}`: a fact for a memory node |
| POST | `/memory/test` | run every memory node against its tests now |
| GET | `/memory/tests/<node>` | a node's tests from the memory database (rows, and as text) |
| POST | `/memory/tests/<node>` | `{"tests": [{"ask", "answer"}, ...], "unknowns": [...]}`: a person rewrites its tests; it relearns |
| GET | `/memory/own` | its own statements kept in memory without tests (`?about=France&limit=50`) |
| GET | `/mind/archive` | what was in its mind around a moment: `?tab=ego&at=<seconds since 1970>&hours=6` (no tab: all of it) |
| GET | `/mind/archive/days` | the days with entries (`?tab=ego&tz=<getTimezoneOffset()>`), and per tab how much is archived |
| POST | `/command` | `{"line": "/status"}`: any console command; returns its output lines |
| POST | `/halt`, `/resume`, `/pause`, `/rollback` | |
| GET | `/status`, `/lineage`, `/events`, `/audit`, `/questions`, `/knowledge?q=...`, `/temperament`, `/why?order=...`, `/inquiries`, `/proposals`, `/opinions`, `/notes`, `/tasks`, `/challenges`, `/evolution`, `/memory` | |
| GET | `/mind` | operator: every note, private ones included (audited) |
| GET | `/director` | its initiative: mode, the agency node, what has paid off, the tasks it set itself, recent decisions |
| POST | `/director` | `{"mode": "auto" \| "research" \| ..., "enabled": true \| false, "decide": true}` |
| GET | `/directors` | the director nodes, their stance, recent reviews, what was kept out of memory |
| POST | `/directors` | `{"action": "spawn"}`, `{"action": "retire", "id": "director-1"}`, or `{"attempts": 5}` |
| GET | `/toolsmith` | the coder node, the tools it wrote (drafts awaiting approval, installed, proven), unmet requests it could write a tool for, recent work |
| GET | `/anatomy` | `?node=<node id, memory node id or skill>&text=...`: how the node is built, and what fires in it for the text (`&latest=1`: what it fired on last); 410 for a node evolution has disposed of |
| GET | `/traces` | `?n=15`: the paths recent orders took through the mesh, hop by hop, with each node, its decision and its input |
| GET | `/objectives` | every objective with its plan, answers and thoughts, second thoughts, the step budget; `?id=obj-1` gives its write-up |
| POST | `/objectives` | `{"text": "understand jazz"}` sets one; `{"id": "obj-1", "action": "pause" \| "resume" \| "remove" \| "think"}`; `{"id": "obj-1", "ask": "..."}` adds a question; `{"ruminate": true}` has second thoughts now |
| POST | `/toolsmith` | `{"build": "reverse a word"}` writes a tool and asks permission; `{"try": "reverser"}` experiments with one it made. Approve a draft with `POST /approve` |
| GET | `/dreams` | asleep or awake, the dream nodes and their fitness, last night, recent dreams |
| POST | `/dreams` | `{"action": "sleep" \| "wake" \| "flip"}` (flip: toss the idle coin now), or `{"allow_sleep": false}` to keep it awake (`true` lets it sleep) |
| GET | `/ego` | its ego: mood and feeling, what it is curious about, its likes and dislikes, its dreams, its ego nodes, what it said lately |
| POST | `/ego` | `{"enabled": false}` turns its ego off (`true` on) |
| POST | `/imagine` | `{"prompt": "a city on Mars"}`: an imagined response (labelled, never stored) |
| GET | `/dashboard?after=<seq>` | everything the web interface's live view shows, in one request |
| GET | `/log?after=<seq>&n=500&routine=0` | the activity log after a sequence number (`routine=0` hides routine work) |
| GET | `/mesh`, `/memory?node=<id>` | the node network as a structure; one memory node's facts, history and tests |
| GET | `/tools/switch` | the tool switch, each tool's setting, and what it decided lately |
| POST | `/tools/switch` | `{"mode": "on" \| "off" \| "decide"}` for all tools; `{"tool": "kiwix", "mode": "on" \| "off" \| "decide" \| "default"}` for one |
| GET | `/code` | Gitea (state, repositories, kinds of code, recent visits) and the scratch pad (items, checkers, practice) |
| GET | `/code/item`, `/code/examples` | `?id=code-3`: one scratch pad item with its code, check and runs; `?q=&kind=&lang=`: functions it studied |
| POST | `/code` | `{"write": "...", "language": "python"}` writes code; `{"add": "<code>", "language", "title"}` puts yours in; `{"id", "code"}` changes it; `{"id", "action": "ask" \| "approve" \| "decline" \| "remove" \| "check"}`; `{"study": true, "repo": "owner/name"}`; `{"practice": true}` |
| GET | `/images` | what its optical nodes took in: each image, what it sees, the text it read, its reader and sight nodes, its teachers |
| GET | `/images/file` | `?id=img-3` (or `?sha=`): the image itself |
| POST | `/images` | `{"name", "data": "<base64 or a data: URL>"}`, or the image itself with `content-type: image/png` (`?name=`) |
| POST | `/images/correct` | `{"id": "img-3", "label": "street sign"}` and/or `{"text": "what its text says"}`: it learns from you |
| POST | `/images/practice`, `/images/forget` | draw and learn glyphs now; `{"id": "img-3"}` forgets an image |
| GET | `/lessons` | orders its teachers set it while idle, its answers, their marks, and what it learned |
| POST | `/lessons` | `{"kind": "question" \| "skill" \| "maths" \| "tool" \| "code"}`: a teacher sets it a lesson now |

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

### Research and retained knowledge (Wikipedia, Kiwix)
The mesh can look things up on Wikipedia, or on your own Kiwix server (see below), and keeps what
it learns.

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
- **Answers come from the passages, in the mesh's own words.** If an LLM teacher (Ollama or
  Claude) is available it writes the answer, told to use only the retained passages. Otherwise the
  mesh restates the best-matching sentences (see [Replies](#replies-the-result-first-in-its-own-words)).
  The answer comes first; the `Source:` line with the article and URL comes after it.
- **Which orders.** Orders the router sends to the wikipedia tool ("tell me about...", "who
  was...", "research..."), `tool wikipedia: <topic>`, and questions no tool or skill fits
  ("how tall is the Eiffel Tower?") when `research_questions = true`.
- **Background curiosity.** After researching for an order, the Researcher agent also reads up to
  `follow_links` (2) articles the introduction links to, never more than `max_depth` (1) away
  from what was asked. It re-reads retained articles older than `refresh_days` (30). It only
  works when no order is in flight, within its own hourly budget.
- **Budgets and limits** (`[research]` in `directives.toml`): `max_fetches_per_hour` (60) for all
  lookups, `background_per_hour` (20) for curiosity and refreshing, `max_documents` (2000)
  retained, after which the least recently used are forgotten. Tools in `unlimited` (your Kiwix
  server) are exempt. A network outage or a missing article is reported per order but never
  quarantines the tool.
- In the live console: lookups show in the work panel, `learn` and `recall` lines in the activity
  log, and the knowledge line shows articles retained and lookups used this hour.

### A local Kiwix server (offline Wikipedia)
If you run [kiwix-serve](https://kiwix.org) with ZIM archives (offline Wikipedia or anything else
Kiwix serves), the mesh researches there first and falls back to Wikipedia only when Kiwix doesn't
have the article or the server is down. Lookups on your own server are not rate-limited.

```
kiwix-serve --port 8080 wikipedia_en_all_maxi_2024-01.zim     # your server
python -m neuromesh run --kiwix http://localhost:8080          # or set [tools.config.kiwix] url
neuromesh> /approve apv-1                                      # once: the kiwix tool uses the network (P3)
neuromesh> who was Ada Lovelace?
 Ada Lovelace was an English mathematician and writer, chiefly known for her work on ...
 Source: kiwix 'Ada Lovelace' http://localhost:8080/content/wikipedia_en_all_maxi_2024-01/A/Ada_Lovelace (...)
```

- **Finding articles.** `plugins/kiwix` reads the server's catalog (`/catalog/v2/entries`) to see
  which books it serves. It searches Wikipedia books in your language first, or the `books` you
  list. To find an article it tries `/suggest`, then `/search?format=xml`, then the HTML search
  page, then a direct title guess, so it works across kiwix-serve versions and with a URL prefix
  (`--urlRootLocation`). It reads the article HTML and keeps the text by section, dropping
  references, navboxes, infoboxes and scripts. It also keeps the articles the introduction links
  to, for curiosity and learn tasks. Retained articles are stored as `source: kiwix`.
- **Preference.** `[research] prefer = ["kiwix", "wikipedia"]`: the first research tool that has
  the article wins. A miss or an unreachable server falls through to the next. Neither counts
  against the tool: misses and outages never quarantine it.
- **No limits.** `[research] unlimited = ["kiwix"]` exempts it from the research budgets (P4).
  Lookups there are still counted, and shown as "on your own server". Learn tasks read up to
  `learn_burst_unlimited` (10) articles per run from it, and its health is checked in the
  background like a local tool's. Wikipedia keeps its budget.
- It talks to the server directly, ignoring any `HTTP(S)_PROXY` (`use_proxy = false`). Its
  self-test only checks that the server answers and serves at least one book, so it works with
  whatever ZIMs you have.


### Searching every Kiwix book before it asks you
One look-up often isn't enough: the article named after a question's subject may not say what was
asked, and the answer may be in another book entirely (Wiktionary, Wikivoyage, Stack Exchange,
Project Gutenberg, ...) or on another server. So for a question, the mesh **searches Kiwix again and
again** until an article actually answers it, up to `[research] kiwix_attempts` (5) times. Then it
gives up and **asks you**.

- **Every book on every server.** `[tools.config.kiwix] url` plus `servers = ["http://nas:8081", ...]`:
  the catalog of each is read, and a server that is down is skipped.
- **The likeliest books first.** Wiktionary for "what does X mean", Wikivoyage for travel, Wikiquote
  for "who said", Stack Exchange and DevDocs for code, wikiHow and iFixit for how-to questions,
  Gutenberg and Wikisource for the text of books. Wikipedia comes first otherwise.
- **Other wordings.** The word a "what does X mean" asks about, the names in the question, what it
  asks about ("who wrote X" means X), the question as it is, and its key words. Each is tried as a
  title search first, then a full-text search (which finds articles that mention a thing, not only
  those named after it).
- **Does it answer?** Each article it finds is read and kept like any research. It only counts as the
  answer when it mentions every name in the question, covers most of its words (`kiwix_coverage`),
  and the answer fits what was asked: a time for "when", a number for "how many", a name for "who".
  An article that does not ("A mural is any piece of artwork..." for "who painted the Zorblax
  mural?") is noted as not it, and is never read twice for the same question. It is never used as
  the answer later either: retained knowledge that doesn't mention a name the question asks about
  doesn't answer it.
- **Giving up.** After `kiwix_attempts` searches it falls back to the other research tools (Wikipedia).
  If nothing else answers either, it asks you: "I searched Kiwix 5 times (wikipedia_en_all,
  wiktionary_en_all; I read 'Mural') and couldn't find the answer to 'Who painted the Zorblax
  mural?'. Do you know it?" Reply `answer: ...` and a memory node keeps it, so it answers at once
  next time. Its curiosity counts the question as a gap, and its director may set itself an
  objective to find out.
- Every search is within the research budgets (your own server is exempt: `unlimited`). The result
  says what was searched (`Result.searched`), and the activity log notes each search and its
  outcome. `/kiwix` lists the books and the last searches, attempt by attempt. `/kiwix <question>`
  searches now. `kiwix_attempts = 1` brings back the single look-up.

**Fast.** Kiwix is many times faster to use than it was, so it learns faster:
- **The tool stays running.** The kiwix tool is a persistent tool (`persistent = true` in its
  manifest; `[tools] workers`): it is started once, in the same sandbox as before, as `tool.py
  --serve`, and takes requests as JSON lines, many at once. No Python start-up per search, and
  the book catalog is read once and kept (`catalog_seconds`). Its CPU limit becomes a budget for
  many requests (`worker_cpu_seconds`); when it is spent the worker stops and a fresh one starts.
- **Kept-alive connections and a cache.** Every thread keeps its connection to each server open,
  and the pages it reads (searches, suggestions, articles) are cached, so a page is fetched once.
- **Everything at once.** Every book of one search is searched at the same time (the best book's
  article still wins), and `kiwix_parallel` (4) searches of one question run together; the first
  in the plan that answers is used.

In the tests, five look-ups take a third of the time or less with the worker than with a fresh
process each (in practice, against a real server, the cache and kept-alive connections save more).

**Browsing: something new from a book it has not opened.** Every `browse_every_seconds` (300) the
**Reader** may pick a Kiwix book it has never read from, or not for `browse_stale_days` (7) days,
read `browse_articles` (2) random articles from it (`/random` on kiwix-serve) and learn them: they
are retained, and memory nodes keep a fact for each. It notes when it last read from every book
(searches and look-ups count too), skips books it opened lately, and stays within the background
research budgets. `/kiwix` shows when each book was last read and what it browsed.

The kiwix tool changed for this (version 1.2.0), so approve it once more after updating (P3).
### Adding a tool
Drop a folder into `plugins/`. The Explorer finds it within seconds.
```
plugins/my_tool/manifest.toml
plugins/my_tool/tool.py        # reads {"input": "..."} on stdin, prints {"output": ...} or {"error": "..."}
                               # or {"query": {"ask": "..."}} to ask a director first: it is run again
                               # with "context": [{"ask", "answer", "source"}] in its input
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
- **Objectives are only as good as what answers them.** Offline, its answers come from memory
  nodes, retained knowledge, tools and you; a memory node gives its nearest fact, which may answer
  a question loosely. Second sources, echo checks and the fit of an answer to a when/who/how-many
  question catch some of that, and the write-up shows each answer's source, confidence and whether
  anything confirmed it. An agreement check compares words, names and numbers; it is not
  understanding. With a research tool and an LLM teacher, plans and answers are much richer.
- **The tools it writes** are assembled from its code library, so offline they can only do what
  those 17 blocks do; the coder node chooses among them, it does not invent algorithms. Refinement
  adjusts how a tool reads a request, checked against known answers. The code check is a static
  allow-list, not a proof: the approval is the real safeguard, which is why nothing runs before it
  and why the code is written to be read. With an LLM teacher it can write other tools, and those
  deserve the same careful reading before you approve them.
- **Code in the scratch pad runs only once you approve exactly that code**, in a fresh folder with
  time, CPU and memory limits and no keys in its environment. That is not a sandbox against the
  network or your files: the validator says what the code reaches for, and the approval is the real
  safeguard. Read it first. Without an LLM teacher, code it "writes" for a request no recipe fits is
  the closest function from your Gitea, adapted, and says so; its code-kind node learns what kind of
  code a purpose needs, not how to write it.
- **Its reader is a small classifier over 16 x 16 glyphs**: it reads clean printed text well and
  handwriting, photos of text, tilted or tightly kerned text badly. Characters that touch are split
  where the ink is thinnest, which is a guess. How well it reads is measured against tesseract, a
  teacher or you, and an image's text says who read it. Its sight node starts knowing nothing but
  "text" and "blank"; what it can name comes from you and a teacher that can see.
- **A teacher's mark is a teacher's opinion.** Lessons teach as a teacher (weighted below a person),
  facts a teacher gives go through a director's review, and what a person taught stands (P1).
- **Research reaches the internet** only through the approved wikipedia tool, within hourly
  budgets, and background curiosity stays within `max_depth` of what users asked. The kiwix tool
  talks only to the server you configure; it is exempt from the budgets because that server is yours. Retained
  knowledge is Wikipedia text: answers name the article they came from, and are only as good as
  it was when fetched (articles are re-read after `refresh_days`). The default keyword routing is
  simple; with an LLM teacher, routing of questions to research is much better.
- **The temperament changes how an answer is delivered, never what it says.** The honesty guard
  checks the form of every change (verbatim answer, fixed openings, one disclosure line). It
  does not check that the answer itself is true. That is P7's job: confidence and provenance on
  every answer. With only the offline heuristic, tone reading is a crude lexicon; an LLM teacher
  reads tone far better and the affect node learns from it. A blocked answer is withheld from the
  user, not destroyed: the operator can see it in the order record.
- **The mind's opinions are measurements, not judgement.** They come from run counts, held-out
  scores, how often a human agreed with a teacher, the tone the affect node read, and how the
  sentiment skill reads retained passages. Offline, proposals are drafted from shared keywords; an
  LLM teacher drafts much better ones. "Private" means private from users. The operator always
  has full access by design.
- A memory node picks among the answers it has stored (the answer texts are its outputs); the
  network holds which question goes with which answer. It is only as reliable as its tests: a
  question unlike any test can still be answered, which is why every answer names the node and
  where the fact came from, and why near misses are tested. Facts an LLM teacher gave are kept
  with that source and are not checked against anything.
- The offline heuristic teacher is a stand-in so the system and its tests run anywhere; labels
  from it are only as good as its lexicon.

## Project layout
```
directives.toml          constitution: prime directives, goals, policy
capabilities.toml        seed skills
plugins/                 installed tools (calculator with mathsolver.py, word_stats, wikipedia, kiwix), and
                         tools it wrote once you approve them (drafts wait in state/workshop/)
challenges/              challenge suites (small samples; fetched benchmarks go to state/challenges/)
memory_tests.imported/   test files of earlier versions, once imported into state/memory.db
                         (state/images/: images it took in; state/scratch/runs/: approved code it ran)
examples/incoming_plugins/  tools the demo installs at runtime
neuromesh/
  runtime.py             System: wiring, cascade, learning, tool activation, status
  live.py                live console (curses): network, work in progress, activity log, prompt
  activity.py            work-in-flight tracker and the activity log
  commands.py            orders and /commands shared by the live and plain consoles
  agents.py              eleven of the agents (the Director is in agency.py, the Dreamer in dreams.py, the challenge runner in challenges.py)
  nodes.py               NeuralNode + derivations (expand / widen / deepen / retune / fresh)
  backend.py             compute backends: numpy (CPU), PyTorch (CUDA / ROCm / MPS), CuPy
  bench.py               CPU vs GPU benchmark
  registry.py            node life cycle
  guardian.py            directive enforcement, budgets, approvals, kill switch, audit
  teachers.py            human / LLM / heuristic teachers
  tools.py               manifests, sandbox, discovery
  knowledge.py           retained knowledge: articles + passages (SQLite FTS5), recall confidence
  voice.py               replies: the result first in the mesh's own words, then sources
  temperament.py         emotional engines: temperament, policy teacher, output templates, honesty guard (P8)
  mind.py                opinions, private notes, inquiries, capability proposals, investigation goals (P9, P10)
  background.py          background tasks: learn a topic, improve a skill, repeat an order, practise a challenge
  challenges.py          open-source LLM benchmarks: suites, Hugging Face fetch, grading, the challenge teacher
  evolution.py           the genetic algorithm: genome operators (grow/shrink neurons and layers), selection, judging
  memory.py              memory nodes: facts compressed into small networks, their tests, compression (P11); own memories
  memdb.py               the memory's own database: memory nodes' tests (with triggers), its own statements (untested)
  ensemble.py            many instances of each node at once ([nodes] max_running); the champion and its instances
  journal.py             the Mind's archive: everything in its mind by date and time
  toolgate.py            the tool switch (on, off, it decides) and the tool-gate node's policy
  gitea.py               what its coder nodes learn from a local Gitea: functions taken apart, kinds of code, the Apprentice
  scratchpad.py          the scratch pad: writing code, the validator, approval, running it once approved (P20), practice
  imaging.py             images without dependencies: PNG/BMP/PPM decoding, what the sight node sees, lines and glyphs
  vision.py              its optical nodes: the sight node, the reader and its glyph node, their teachers, the Looker (P21)
  tutor.py               lessons: its Ollama teachers set it orders when idle and mark them (P22)
  agency.py              independent agency: the Director, its modes, the drive policy and the agency node (P12)
  directors.py           the directors: intake, review, second opinions, the memory gate, duplicates, the stance (P14)
  toolsmith.py           the toolsmith: the coder teacher, writing, checking and offering tools, approval, experiments, refinement (P15)
  codelib.py             the code library: the building blocks made tools are written from, with probes to test them
  objectives.py          objectives: the planner teacher, plans, queries against itself, rumination, write-ups, second thoughts (P16)
  anatomy.py             a closer look: the paths orders take, how a node is built, and which synapses fire for an input
  corrections.py         was it right? the next most likely answers, worked out now, and learning from what a person picks
  thinking.py            thinking nodes after the memory nodes: thoughts in its own words (the honesty guard), opinions, desires (P17)
  curiosity.py           areas of interest and gaps in what it knows: the objectives its director sets itself
  seeker.py              the Kiwix search: every book on every server, other wordings, until an article answers
  avatar.py              the avatar: its mind at a glance, and its voice from a ComfyUI text-to-speech workflow
  ego.py                 the ego, wrapped around every other node: its mood on the way in, its own remark on the way out (P18)
  dreams.py              sleep and dreams: dream synthesis, dream nodes, waking review, their genetic algorithm, imagination (P13)
  web.py, web/           the web interface: dashboard data, and the page (index.html, app.css, app.js)
  store.py               SQLite: experience, lineage, tools, orders, temperament, mind, tasks, audit
  encoding.py, bus.py, tasks.py, config.py, report.py, server.py, demo.py, __main__.py
tests/                   271 tests: nodes, guardian, LLM wire formats and a mock Ollama server,
                         Wikipedia research against a mock MediaWiki API, Kiwix research
                         against a mock kiwix-serve (preference, fallback, no limits), answer-first
                         replies, challenges (grading, paged fetch, practice vs exam, route search),
                         evolution (exact growth, pruning, growth bias, promotion rules, a node
                         that evolves larger in a running system), memory nodes (tests in the memory database, files imported, own statements untested,
                         compression that never accepts a failing node, merging (exact stacking, the
                         90% rule, relearning what fails), people's edits, answers),
                         the web interface (its files, the endpoints it uses, the token), independent
                         agency (the drive policy, outcomes teaching the agency node, people in charge),
                         sleep and dreams (no outside calls, waking checks and evolution, orders wake it,
                         imagination labelled and never stored), directors (always one, both ends of
                         every order, the memory gate, disclosed disagreements, duplicates, the mood,
                         tools asking directors, retries up to max_attempts), the calculator's mathematics,
                         the toolsmith (every recipe's code passes its probes, the code check, nothing
                         runs before approval, changed code asks again, experiments and refinement,
                         initiative that builds and experiments, the coder node taking over), objectives
                         (planning, queries against itself, echoes, contradictions settled, a person
                         asked, delegation, surveys, reading up, second thoughts, initiative, P1/P2),
                         a closer look (a forward pass recorded faithfully: the strongest path, gradient ×
                         input adding up to the logit; node and memory-node anatomy; order paths),
                         the avatar (its glance, faces, speaking with a mock ComfyUI, cached audio), the Kiwix
                         search (books on two servers, retries, giving up and asking), thinking nodes (rephrasing that adds nothing, the honesty guard turning down an
                         LLM draft, thoughts after the knowledge and never in it, opinions about people kept
                         private, desires; restarts), curiosity (gaps, interest, the director setting itself
                         objectives and saying why), corrections (the next most likely labels, routes worked out and memory's other
                         answers; yes, a pick, or none with or without the right answer, each learned and
                         answered right next time; a memory node learning not to answer), temperament and
                         the honesty guard (incl. tampered templates), the mind (discretion,
                         questions, proposals, investigations), background tasks, full system,
                         live console (incl. a pseudo-terminal run), compute
                         backends (4 GPU parity tests run where PyTorch or CuPy is installed), the tool
                         switch (off refuses every run and counts nothing against a tool; one tool set
                         apart; it decides from the gate policy), Gitea (a mock Gitea: functions taken
                         apart in four languages, kinds of code, the token sent only there, only changed
                         files read again), the scratch pad (the validator for six languages, nothing runs
                         unapproved, changed code withdraws the approval, a teacher's code fixed once,
                         practice), images (PNG decoding by hand, a mock Ollama vision model, a person's
                         label standing, a glyph node that learns to read a note), lessons (a mock Ollama
                         teacher sets orders and marks them; a teacher's fact is a teacher's, not a person's)
```

State lives in `state/` (SQLite database, node weights, log). Delete it to start fresh.

```
python -m pytest tests          # or: python -m unittest discover tests
```
