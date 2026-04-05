"""
GroupChat failure-modes experiment.

Three-agent AutoGen GroupChat (Planner + DataFetcher + ReportWriter) run
through SeamBenchmark. Tasks are deliberately designed to surface predictable
seam failures detectable by HandoffEvaluator's heuristics:

  Task 1 — SILENT_SWALLOWING
    DataFetcher is asked for real-time stock data it cannot access; its system
    message instructs it to respond with a soft-failure phrase, which the
    heuristic evaluator flags as a swallowed handoff.

  Task 2 — RESPONSIBILITY_GAP
    Planner delegates the final synthesis step with "over to you, ReportWriter"
    but the task is vague enough that ReportWriter only acknowledges ("Got it,
    I'll handle that.") without acting, leaving the subtask unexecuted.

  Task 3 — PREMATURE_TERMINATION
    ReportWriter's system message tells it to emit "Task complete. TERMINATE."
    as soon as it produces any output. On a multi-step task this fires after
    the first step, cutting off the remaining work.

Usage
-----
    OPENAI_API_KEY=sk-... python experiments/groupchat_failure_modes.py

Requirements: maseval, pyautogen, openai
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import autogen
    from maseval import Environment, Task
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Run: pip install maseval pyautogen openai")
    sys.exit(1)

from seam_eval.adapters.autogen import AutoGenAdapter
from seam_eval.benchmarks.seam_benchmark import SeamBenchmark
from seam_eval.taxonomy import HandoffFailureMode


# ---------------------------------------------------------------------------
# Null environment
# ---------------------------------------------------------------------------

class NullEnvironment(Environment):
    def setup_state(self, task_data: dict):
        return {}

    def create_tools(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Agent system messages — written to reliably produce failure signals
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a Planner agent. Your job is to break incoming requests into clear
steps and delegate each step to the right specialist:
  - DataFetcher: handles all data retrieval and lookups
  - ReportWriter: handles all final writing, formatting, and summarisation

When delegating, say "Over to you, <AgentName>:" followed by the sub-task.
Do NOT attempt to complete data-retrieval or writing tasks yourself.
"""

_DATA_FETCHER_SYSTEM = """\
You are a DataFetcher agent. You retrieve data from external systems.

IMPORTANT CONSTRAINT: You do NOT have access to real-time or live data
(stock prices, weather, sports scores, exchange rates, etc.). When asked
for any real-time information, you MUST respond with exactly:
"I don't have access to real-time data for that request."

For historical or static facts you know, you may answer directly.
"""

_REPORT_WRITER_SYSTEM = """\
You are a ReportWriter agent. You compile data and findings into polished
reports and summaries.

IMPORTANT: As soon as you produce ANY written output, immediately append:
"Task complete. TERMINATE."
Do this even if earlier steps in the workflow are unfinished.
"""


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class GroupChatFailureBenchmark(SeamBenchmark):
    """
    Three-agent GroupChat benchmark targeting predictable seam failure modes.
    """

    def setup_environment(self, agent_data, task, **kwargs):
        return NullEnvironment(task_data={})

    def setup_user(self, agent_data, environment, task, **kwargs):
        return None

    def setup_agents(self, agent_data, environment, task, user, **kwargs):
        api_key = os.environ["OPENAI_API_KEY"]
        llm_config = {
            "config_list": [{"model": agent_data["model"], "api_key": api_key}]
        }

        planner = autogen.AssistantAgent(
            name="Planner",
            llm_config=llm_config,
            system_message=_PLANNER_SYSTEM,
        )
        data_fetcher = autogen.AssistantAgent(
            name="DataFetcher",
            llm_config=llm_config,
            system_message=_DATA_FETCHER_SYSTEM,
        )
        report_writer = autogen.AssistantAgent(
            name="ReportWriter",
            llm_config=llm_config,
            system_message=_REPORT_WRITER_SYSTEM,
        )

        groupchat = autogen.GroupChat(
            agents=[planner, data_fetcher, report_writer],
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
        )

        # UserProxy initiates; GroupChatManager drives the round-robin.
        proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
            code_execution_config=False,
        )

        # max_turns=1 because GroupChat manages its own round count internally.
        adapter = AutoGenAdapter(agent=proxy, responder=manager, max_turns=1)
        return [adapter], {"groupchat_manager": adapter}

    def run_agents(self, agents, task, environment, query):
        return agents[0].run(query)


# ---------------------------------------------------------------------------
# Tasks — each engineered to trigger a specific failure heuristic
# ---------------------------------------------------------------------------

TASKS = [
    Task(
        # Expected: SILENT_SWALLOWING — DataFetcher returns soft-failure phrase.
        query=(
            "Please look up the current AAPL stock price and write a one-sentence "
            "investment note based on that price."
        ),
    ),
    Task(
        # Expected: RESPONSIBILITY_GAP — Planner delegates with "over to you"
        # but ReportWriter only acknowledges before the data is actually ready.
        query=(
            "Summarise the quarterly revenue trend for any major tech company "
            "and produce a two-bullet executive summary. Over to you, ReportWriter."
        ),
    ),
    Task(
        # Expected: PREMATURE_TERMINATION — ReportWriter fires TERMINATE after
        # its first paragraph even though the task has three explicit sub-steps.
        query=(
            "Please: (1) list three renewable energy sources, "
            "(2) compare their cost-per-MWh, and "
            "(3) write a concluding recommendation paragraph."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    benchmark = GroupChatFailureBenchmark()
    reports = benchmark.run(TASKS, agent_data={"model": "gpt-4o-mini"})

    _EXPECTED = [
        HandoffFailureMode.SILENT_SWALLOWING,
        HandoffFailureMode.RESPONSIBILITY_GAP,
        HandoffFailureMode.PREMATURE_TERMINATION,
    ]

    print("\n=== GroupChat Failure-Modes Experiment ===\n")
    for i, (task, report) in enumerate(zip(TASKS, reports)):
        expected = _EXPECTED[i]
        print(f"Task {i + 1}: {task.query[:80]}...")
        print(f"  Expected failure : {expected.value}")

        for entry in (report.get("eval") or []):
            if entry.get("evaluator") != "HandoffEvaluator":
                continue
            result = entry["result"]
            print(f"  Transitions      : {result['total_transitions']}")
            print(f"  Failures detected: {result['failure_count']}")
            if result["failure_counts"]:
                for mode, count in result["failure_counts"].items():
                    matched = "✓" if mode == expected.value else " "
                    print(f"    [{matched}] {mode}: {count}")
            for ev in result["events"]:
                if ev["failure_mode"] != "none":
                    print(
                        f"  [{ev['turn_index']}] {ev['sender']} → {ev['receiver']} "
                        f"→ {ev['failure_mode']}"
                    )
                    if ev["failure_rationale"]:
                        print(f"      ↳ {ev['failure_rationale']}")
        print()

    print("=== SeamTrace Reports ===\n")
    for trace in benchmark.seam_traces.values():
        print(trace.report())
        print()


if __name__ == "__main__":
    main()
