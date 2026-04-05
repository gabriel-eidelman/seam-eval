"""
Minimal two-agent SeamEval demo.

Runs a simple UserProxy + AssistantAgent conversation through SeamBenchmark
and prints the handoff attribution report.

Usage
-----
    OPENAI_API_KEY=sk-... python experiments/two_agent_demo.py

Requirements: maseval, pyautogen, openai
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Guard: ensure the package is importable from the repo root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

try:
    import autogen
    from maseval import Environment
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Run: pip install maseval pyautogen openai")
    sys.exit(1)

from seam_eval.adapters.autogen import AutoGenAdapter
from seam_eval.taxonomy import SeamTask
from seam_eval.benchmarks.seam_benchmark import SeamBenchmark
from seam_eval.callbacks.seam_trace import SeamTraceCallback
from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator


class NullEnvironment(Environment):
    """Minimal no-op environment for demos that don't need external tools."""

    def setup_state(self, task_data: dict):
        return {}

    def create_tools(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Concrete benchmark for this demo
# ---------------------------------------------------------------------------

class TwoAgentDemoBenchmark(SeamBenchmark):
    """Minimal benchmark: single UserProxy + AssistantAgent pair."""

    def setup_environment(self, agent_data, task, **kwargs):
        return NullEnvironment(task_data={})

    def setup_user(self, agent_data, environment, task, **kwargs):
        return None

    def setup_agents(self, agent_data, environment, task, user, **kwargs):
        llm_config = {"config_list": [{"model": agent_data["model"], "api_key": os.environ["OPENAI_API_KEY"]}]}

        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="You are a helpful assistant. Reply TERMINATE when done.",
        )
        proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
            code_execution_config=False,
        )
        adapter = AutoGenAdapter(agent=proxy, responder=assistant, max_turns=6)
        return [adapter], {"assistant": adapter}

    def run_agents(self, agents, task, environment, query):
        adapter = agents[0]
        return adapter.run(query)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tasks = [
        SeamTask(
            query="What is 2 + 2? Be brief.",
            intended_behavior="The assistant should return '4' or 'The answer is 4.' with no additional commentary.",
        ),
        SeamTask(
            query="Name the capital of France. One word only.",
            intended_behavior="The assistant should return 'Paris' and nothing else.",
        ),
    ]

    benchmark = TwoAgentDemoBenchmark()
    reports = benchmark.run(tasks, agent_data={"model": "gpt-4o-mini"})

    print("\n=== Handoff Attribution Reports ===\n")
    for task, report in zip(tasks, reports):
        print(f"Task: {task.query!r}")
        for entry in (report.get("eval") or []):
            if entry.get("evaluator") == "HandoffEvaluator":
                result = entry["result"]
                print(f"  Transitions : {result['total_transitions']}")
                print(f"  Failures    : {result['failure_count']}")
                for ev in result["events"]:
                    status = ev["failure_mode"] if ev["failure_mode"] != "none" else "ok"
                    print(f"  [{ev['turn_index']}] {ev['sender']} → {ev['receiver']} ({status})")
                    if ev["failure_rationale"]:
                        print(f"      ↳ {ev['failure_rationale']}")
        print()

    # Also print raw SeamTrace reports.
    print("=== SeamTrace Reports ===\n")
    for trace in benchmark.seam_traces.values():
        print(trace.report())
        print()


if __name__ == "__main__":
    main()
