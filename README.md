# SeamEval

**SeamEval** is an evaluation framework for diagnosing failure modes at agent boundaries in multi-agent LLM systems. It extends [MASEval](https://github.com/parameterlab/maseval) with handoff-level attribution — instrumenting the *seams* between agents rather than just measuring end-to-end task outcomes.

---

## Motivation

Existing multi-agent benchmarks are outcome-centric: they record whether the system produced the right final answer, but not *where* it broke down. A system can fail silently at an agent handoff — dropping context, misrouting a task, or producing a plausible-but-wrong continuation — and the benchmark simply records a wrong answer with no structural diagnosis.

SeamEval treats every agent transition as a first-class evaluation event. The unit of analysis is the **seam**: the interface between two agents where context is passed, control is transferred, and failures accumulate. This shifts evaluation from "did the system get it right?" to "at which transition did it start going wrong, and why?"

---

## Research Contributions

### 1. AutoGen adapter for MASEval

A production-quality `AgentAdapter` (`seam_eval/adapters/autogen.py`) integrating AutoGen (AG2) multi-agent conversations into the MASEval evaluation lifecycle. Supports both two-agent (UserProxy + AssistantAgent) and GroupChat configurations, normalises AutoGen's per-agent message storage into MASEval's unified conversation history format, and exposes full callback hooks for downstream instrumentation.

### 2. Handoff failure taxonomy

A structured classification of multi-agent failure modes at the transition level, defined in `seam_eval/taxonomy.py`. Modes are organized into five `FailureCategory` families:

**Routing failures** — task sent to the wrong place or owned by no one

| Mode | Description |
|---|---|
| `misrouted_handoff` | Task dispatched to a specialist not equipped for it; that agent attempts execution anyway and produces a wrong or degraded result |
| `responsibility_gap` | Two or more agents each assume a subtask belongs to the other; no agent executes it and the gap is not surfaced |

**Loop failures** — control flow never terminates

| Mode | Description |
|---|---|
| `agent_self_loop` | A single agent re-invokes itself without making progress; the loop continues until a turn or token budget is exhausted |
| `mutual_loop` | Two agents hand off to each other indefinitely; neither reaches a terminal condition or escalates the deadlock |

**Context failures** — information state at the seam is wrong

| Mode | Description |
|---|---|
| `context_insufficiency` | The receiving agent lacks sufficient context; it does not explicitly fail but silently produces an incorrect or partial output |
| `context_corruption` | The receiving agent is passed stale, conflicting, or structurally malformed context that causes downstream processing errors |

**Termination failures** — the chain stops at the wrong moment

| Mode | Description |
|---|---|
| `premature_termination` | A stop condition fires on an intermediate output, halting work before the task is complete |
| `missed_termination` | No agent triggers a stop condition after task completion; the chain continues executing, consuming resources or producing spurious output |

**Output failures** — the result is structurally wrong

| Mode | Description |
|---|---|
| `compounding_drift` | Small framing errors accumulate across a multi-hop chain; the terminal output diverges from the original goal despite no single catastrophic failure |
| `duplicate_execution` | Two or more agents each complete the same subtask independently; their results conflict and no reconciliation strategy exists |

### 3. SeamTrace

A `Callback`-based instrumentation layer (`seam_eval/callbacks/seam_trace.py`) that attaches structured metadata to every agent transition, recording context passed, context dropped, the receiving agent, its return value, and the resulting failure classification. Produces per-task `SeamTrace` objects — grouped by `FailureCategory` — that serve as the primary unit of analysis.

---

## Architecture

SeamEval is built as a downstream extension of MASEval — it subclasses MASEval's abstract base classes rather than forking them.

```
MASEval (upstream)
├── Benchmark          ← SeamBenchmark subclasses this
├── AgentAdapter       ← AutoGenAdapter implements this
├── Callback           ← SeamTraceCallback implements this
└── Evaluator          ← HandoffEvaluator implements this

SeamEval (this repo)
├── seam_eval/
│   ├── taxonomy.py                  # failure mode enum + data classes
│   ├── adapters/
│   │   └── autogen.py               # AutoGen AgentAdapter
│   ├── callbacks/
│   │   └── seam_trace.py            # handoff instrumentation callback
│   ├── evaluators/
│   │   └── handoff_evaluator.py     # LLM-based failure classifier
│   └── benchmarks/
│       └── seam_benchmark.py        # base benchmark wiring it all together
└── experiments/
    └── two_agent_demo.py            # runnable end-to-end example
```

### `taxonomy.py` — the data model

The entire framework is grounded in five types:

**`SeamTask`** — a MASEval `Task` subclass that carries an `intended_behavior` field: one to three sentences describing what a correct run looks like (expected answer, expected agent process, or both). This is injected into the `HandoffEvaluator` prompt so the LLM can distinguish genuine failures from correct agent behavior (e.g. an agent correctly reporting it lacks live-data access is not a failure).

**`FailureCategory`** — an enum of the five failure families (`ROUTING`, `LOOP`, `CONTEXT`, `TERMINATION`, `OUTPUT`). Used for grouping in reports and filtering.

**`HandoffFailureMode`** — an enum of ten failure modes plus `NONE` for successful transitions. All classification and reporting is keyed on this enum. New failure modes should be added here before any detection logic is written. The `FAILURE_CATEGORIES` dict maps each mode to its `FailureCategory`.

**`SeamEvent`** — a dataclass representing a single agent transition. Captures:
- `sender` and `receiver` (agent names)
- `message` (the content being handed off)
- `context_passed` (messages the sender explicitly forwarded)
- `context_dropped` (messages available to the sender but not forwarded, computed by `SeamTraceCallback`)
- `receiver_response` (the receiver's return value)
- `failure_mode` and `failure_rationale` (filled in post-hoc by `HandoffEvaluator`)
- `turn_index` and `metadata`
- `.category` property — returns the `FailureCategory` for the assigned failure mode

**`SeamTrace`** — an ordered list of `SeamEvent` objects for a single task run, plus convenience methods: `failures`, `failure_rate`, `failure_counts()`, `failures_by_category()`, and `report()` (which groups output by category). Produced by `SeamTraceCallback`; consumed by `HandoffEvaluator`.

### `adapters/autogen.py` — `AutoGenAdapter`

Bridges AutoGen's conversation model to MASEval's `AgentAdapter` interface. The two key methods:

**`_run_agent(query)`** — initiates an AutoGen conversation via `agent.initiate_chat(responder, message=query, ...)` and returns the final answer extracted from the message history. The conversation terminates when the initiating agent sends `TERMINATE` or `max_turns` is reached.

**`get_messages()`** — retrieves the conversation history from AutoGen's per-agent `chat_messages` dict and normalises each message into MASEval's `{role, content, name, metadata}` schema. Merges both sides of the conversation; falls back to flattening all recorded message lists if the primary partner's history is empty.

Supports two topologies:
- **Two-agent**: `UserProxyAgent` as initiator, `AssistantAgent` as responder, `max_turns` controls the exchange.
- **GroupChat**: `UserProxyAgent` as initiator, `GroupChatManager` as responder, `max_turns=1` (the manager owns turn control internally).

### `callbacks/seam_trace.py` — `SeamTraceCallback`

A MASEval `Callback` that passively observes the evaluation lifecycle and records a `SeamEvent` for every `on_handoff` call. Key design constraints:

- **Stateless per-task**: a fresh instance is created per task by `SeamBenchmark.setup_callbacks()`. Call `reset(task_id)` to reuse an instance.
- **Non-destructive**: never modifies messages or agent state; purely observational.
- **Fault-tolerant**: exceptions during recording are caught and logged, never re-raised — a tracing failure must not abort the evaluation.

The `on_handoff` hook computes `context_dropped` as the set difference between `context_available` and `context_passed`, using content-based identity so it works correctly across serialisation boundaries. It also handles `on_agent_start`, `on_agent_end`, and `on_task_end` for completeness.

### `evaluators/handoff_evaluator.py` — `HandoffEvaluator`

A MASEval `Evaluator` that classifies failure modes in a `SeamTrace` using **LLM-based evaluation**. It formats the full `SeamEvent` transcript into a structured prompt — including the `intended_behavior` description from `SeamTask` — and submits it to an OpenAI-compatible model, which returns per-event failure classifications as structured JSON.

The `intended_behavior` field is critical: it lets the LLM distinguish genuine structural failures from correct agent behavior (e.g. an agent correctly refusing an out-of-scope task should be classified as `none`, not `misrouted_handoff`).

Key design points:
- Uses `response_format={"type": "json_object"}` and `temperature=0` for deterministic output.
- LLM output is mapped back onto `SeamEvent.failure_mode` and `SeamEvent.failure_rationale`. Rationale is prefixed with the LLM's `confidence` level (`high`/`medium`/`low`).
- Unknown failure mode labels from the LLM default to `NONE` with a warning logged.
- The default model is `gpt-4o-mini`; any OpenAI-compatible endpoint can be used via `base_url`.

`evaluate()` accepts an `intended_behavior` string and returns a structured report dict:

```python
{
    "task_id": str,
    "total_transitions": int,
    "failure_count": int,
    "failure_rate": float,
    "failure_counts": {"context_insufficiency": 2, ...},  # non-zero modes only
    "overall_summary": str,                                # LLM-generated summary
    "events": [
        {
            "turn_index": int,
            "sender": str,
            "receiver": str,
            "failure_mode": str,
            "failure_rationale": str | None,               # includes confidence prefix
            "description": str,
        },
        ...
    ]
}
```

### `benchmarks/seam_benchmark.py` — `SeamBenchmark`

An abstract MASEval `Benchmark` subclass that wires the entire instrumentation pipeline together automatically. Concrete subclasses need only implement the four core MASEval lifecycle methods (`setup_environment`, `setup_user`, `setup_agents`, `run_agents`) — `SeamBenchmark` handles everything seam-related:

- **`setup_callbacks(task)`** — instantiates a fresh `SeamTraceCallback` per task and stashes it as `_active_callback`.
- **`evaluate(...)`** — calls `super().evaluate()` for any domain-specific evaluators, then runs `HandoffEvaluator` on the accumulated trace and appends the handoff report to the results list under `{"evaluator": "HandoffEvaluator", "result": ...}`.
- **`seam_traces`** property — exposes the full `{task_id: SeamTrace}` dict after a run, enabling post-hoc analysis across tasks.

A custom `HandoffEvaluator` instance can be injected at construction time (`SeamBenchmark(agent_data=..., handoff_evaluator=MyEvaluator())`) to swap in LLM-based classification without changing the benchmark.

---

## Data Flow

A complete task execution follows this path:

```
benchmark.run(tasks)
    │
    ├─ setup_callbacks(task)          → SeamTraceCallback (fresh per task)
    ├─ setup_environment(...)
    ├─ setup_agents(...)              → AutoGenAdapter wrapping AutoGen agents
    │
    ├─ run_agents(...)
    │       │
    │       └─ AutoGenAdapter._run_agent(query)
    │               │
    │               └─ agent.initiate_chat(responder, ...)
    │                       │
    │                       ├─ [turn 0] user_proxy → assistant
    │                       │       └─ callback.on_handoff(...)  → SeamEvent(turn=0)
    │                       ├─ [turn 1] assistant → user_proxy
    │                       │       └─ callback.on_handoff(...)  → SeamEvent(turn=1)
    │                       └─ ... (until TERMINATE or max_turns)
    │
    ├─ callback.on_task_end(final_answer)
    │
    └─ evaluate(...)
            ├─ [domain evaluators]
            └─ HandoffEvaluator.evaluate(trace)
                    ├─ classify each SeamEvent (rule cascade)
                    └─ return structured report dict
```

---

## Installation

```bash
git clone https://github.com/gabrieleidelman/seam-eval
cd seam-eval
pip install -e ".[dev]"
```

Dependencies: `maseval>=0.4.0`, `pyautogen>=0.2.0`, `pydantic>=2.0`, `openai>=1.0`.

---

## Quickstart

### Running the demo

```bash
OPENAI_API_KEY=sk-... python experiments/two_agent_demo.py
```

### Implementing your own benchmark

```python
import autogen
from maseval import Task
from seam_eval.adapters.autogen import AutoGenAdapter
from seam_eval.benchmarks.seam_benchmark import SeamBenchmark


class MyBenchmark(SeamBenchmark):

    def setup_environment(self, agent_data, task):
        from maseval import Environment
        return Environment()

    def setup_user(self, agent_data, environment, task):
        return None

    def setup_agents(self, agent_data, environment, task, user):
        llm_config = {"config_list": [{"model": "gpt-4o", "api_key": agent_data["api_key"]}]}
        assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
        proxy = autogen.UserProxyAgent(
            "user_proxy",
            human_input_mode="NEVER",
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
            code_execution_config=False,
        )
        adapter = AutoGenAdapter(agent=proxy, responder=assistant, max_turns=10)
        return [adapter], {"assistant": adapter}

    def run_agents(self, agents, task, environment):
        return agents[0].run(task.query)


tasks = [Task(query="Summarise the key risks of autonomous AI agents.")]
benchmark = MyBenchmark(agent_data={"api_key": "sk-..."})
reports = benchmark.run(tasks)

# Inspect per-task handoff attribution
for trace in benchmark.seam_traces.values():
    print(trace.report())
```

### Using `SeamTask` and `intended_behavior`

Pass `intended_behavior` on each task so the LLM evaluator can correctly distinguish failures from correct agent behavior:

```python
from seam_eval.taxonomy import SeamTask

tasks = [
    SeamTask(
        query="Find the current price of AAPL and summarise analyst sentiment.",
        intended_behavior=(
            "The retrieval agent should fetch live stock data and pass it to "
            "the summariser. The final answer should include a price and a "
            "sentiment label. An agent reporting it lacks live-data access is "
            "a failure only if the data was actually available."
        ),
    )
]
benchmark = MyBenchmark(agent_data={"api_key": "sk-..."})
reports = benchmark.run(tasks)
```

### Using a custom or alternative model

```python
from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator

evaluator = HandoffEvaluator(model="gpt-4o", api_key="sk-...")
benchmark = MyBenchmark(
    agent_data={"api_key": "sk-..."},
    handoff_evaluator=evaluator,
)
```

---

## Running Tests

```bash
pytest
```

Tests cover `taxonomy.py` (enum completeness, `SeamEvent`/`SeamTrace` behaviour), `HandoffEvaluator` (each heuristic check independently), and `SeamTraceCallback` (lifecycle hooks, context-drop computation, fault tolerance).

---

## Development Conventions

- All new classes subclass the appropriate MASEval abstract base.
- New failure modes go in `taxonomy.py` first, before any detection logic.
- `SeamTraceCallback` must remain stateless per-task.
- Experiments in `experiments/` must be fully reproducible — use MASEval's seeding utilities.
- Type hints on every function signature; no untyped code.

---

## Relationship to MASEval

SeamEval is a downstream extension of MASEval, not a fork. The `AutoGenAdapter` is intended to be upstreamed to MASEval as a PR. The seam-specific instrumentation (`SeamTraceCallback`, `HandoffEvaluator`, `taxonomy.py`) stays in this repo as a standalone research contribution.

---

## Author

Gabriel Eidelman — Stanford CS, AI Track.
