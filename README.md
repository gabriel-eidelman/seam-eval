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

A structured classification of multi-agent failure modes at the transition level, defined in `seam_eval/taxonomy.py` as `HandoffFailureMode`:

| Mode | Description |
|---|---|
| `context_truncation` | Agent B receives an impoverished representation of what agent A reasoned |
| `misrouted_handoff` | Task is sent to the wrong specialist; receiving agent attempts it anyway |
| `silent_swallowing` | Agent receives a task it cannot handle, returns a soft failure, system continues |
| `responsibility_gap` | Two agents each assume the other owns a subtask; neither executes it |
| `compounding_drift` | Small framing errors accumulate across a chain; terminal task diverges from original |
| `premature_termination` | Handoff/stop condition triggers on intermediate output, cutting off incomplete work |

### 3. SeamTrace

A `Callback`-based instrumentation layer (`seam_eval/callbacks/seam_trace.py`) that attaches structured metadata to every agent transition, recording context passed, context dropped, the receiving agent, its return value, and the resulting failure classification. Produces per-task `SeamTrace` objects that serve as the primary unit of analysis.

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
│   │   └── handoff_evaluator.py     # heuristic failure classifier
│   └── benchmarks/
│       └── seam_benchmark.py        # base benchmark wiring it all together
└── experiments/
    └── two_agent_demo.py            # runnable end-to-end example
```

### `taxonomy.py` — the data model

The entire framework is grounded in three types:

**`HandoffFailureMode`** — an enum of the six failure modes plus `NONE` for successful transitions. All classification and reporting is keyed on this enum. New failure modes should be added here before any detection logic is written.

**`SeamEvent`** — a dataclass representing a single agent transition. Captures:
- `sender` and `receiver` (agent names)
- `message` (the content being handed off)
- `context_passed` (messages the sender explicitly forwarded)
- `context_dropped` (messages available to the sender but not forwarded, computed by `SeamTraceCallback`)
- `receiver_response` (the receiver's return value)
- `failure_mode` and `failure_rationale` (filled in post-hoc by `HandoffEvaluator`)
- `turn_index` and `metadata`

**`SeamTrace`** — an ordered list of `SeamEvent` objects for a single task run, plus convenience methods: `failures`, `failure_rate`, `failure_counts()`, and `report()`. Produced by `SeamTraceCallback`; consumed by `HandoffEvaluator`.

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

A MASEval `Evaluator` that classifies failure modes in a `SeamTrace`. Iterates over every `SeamEvent` and applies a **deterministic rule cascade**, ordered by specificity (most specific checks first):

1. **Misrouted handoff** — receiver name differs from `expected_receiver` (requires caller to supply ground truth)
2. **Context truncation** — `context_dropped / context_available ≥ 0.5`
3. **Silent swallowing** — empty response, or response contains soft-failure phrases (`"i cannot"`, `"i don't have access"`, `"as an ai"`, etc.)
4. **Premature termination** — response contains a termination signal (`"terminate"`, `"task complete"`, etc.) while the message still contains open-task markers (`?`, `"please"`, `"find"`, etc.)
5. **Responsibility gap** — message contains delegation language (`"over to you"`, `"please handle"`) but response is a short acknowledgement only (`"understood"`, `"noted"`, etc.)
6. **Compounding drift** — intentionally stubbed; requires cross-event analysis at the benchmark level

The cascade short-circuits on the first match. To extend with LLM-based classification, subclass `HandoffEvaluator` and override `_classify_event`.

`evaluate()` returns a structured report dict:

```python
{
    "task_id": str,
    "total_transitions": int,
    "failure_count": int,
    "failure_rate": float,
    "failure_counts": {"context_truncation": 2, ...},  # non-zero modes only
    "events": [
        {
            "turn_index": int,
            "sender": str,
            "receiver": str,
            "failure_mode": str,
            "failure_rationale": str | None,
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

### Injecting LLM-based classification

```python
from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator
from seam_eval.taxonomy import HandoffFailureMode, SeamEvent


class LLMHandoffEvaluator(HandoffEvaluator):
    def _classify_event(self, event, expected_receiver=None):
        # Run heuristics first; only call LLM if inconclusive.
        mode, rationale = super()._classify_event(event, expected_receiver=expected_receiver)
        if mode is HandoffFailureMode.NONE and len(event.message) > 200:
            mode, rationale = self._llm_classify(event)
        return mode, rationale

    def _llm_classify(self, event):
        # Your LLM call here.
        ...


benchmark = MyBenchmark(
    agent_data={"api_key": "sk-..."},
    handoff_evaluator=LLMHandoffEvaluator(),
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
