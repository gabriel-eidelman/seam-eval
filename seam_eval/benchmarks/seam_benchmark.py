"""
SeamBenchmark — base benchmark with seam instrumentation built in.

Subclass this instead of MASEval's Benchmark directly when you want
handoff-level failure attribution wired up automatically.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from maseval import Benchmark, Task

from seam_eval.callbacks.seam_trace import SeamTraceCallback
from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator
from seam_eval.taxonomy import SeamTrace

logger = logging.getLogger(__name__)


class SeamBenchmark(Benchmark):
    """
    Abstract base benchmark that automatically instruments agent seams.

    Concrete subclasses must implement:
        setup_environment(agent_data, task) -> Environment
        setup_user(agent_data, environment, task) -> user | None
        setup_agents(agent_data, environment, task, user) -> (agents_to_run, agents_dict)
        run_agents(agents, task, environment) -> final_answer

    SeamBenchmark handles:
        - Instantiating a SeamTraceCallback per task
        - Injecting the callback into the MASEval lifecycle
        - Running HandoffEvaluator on the resulting trace
        - Merging handoff attribution into the standard evaluation report

    For custom evaluators, override `setup_evaluators` as normal — the
    handoff evaluator is always appended automatically.
    """

    def __init__(
        self,
        handoff_evaluator: Optional[HandoffEvaluator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._handoff_evaluator = handoff_evaluator or HandoffEvaluator()
        # Keyed by task_id; populated during run.
        self._seam_traces: dict[str, SeamTrace] = {}

    # ------------------------------------------------------------------
    # SeamBenchmark-specific API
    # ------------------------------------------------------------------

    @property
    def seam_traces(self) -> dict[str, SeamTrace]:
        """Accumulated SeamTraces from all tasks run so far."""
        return self._seam_traces

    def get_trace(self, task_id: str) -> Optional[SeamTrace]:
        """Return the SeamTrace for a specific task, if available."""
        return self._seam_traces.get(task_id)

    # ------------------------------------------------------------------
    # MASEval Benchmark overrides
    # ------------------------------------------------------------------

    def get_model_adapter(self, model_id: str, **kwargs: Any):
        """Not used by SeamEval — no LLM-based simulators or judges."""
        raise NotImplementedError(
            "SeamBenchmark does not use a ModelAdapter. "
            "Override get_model_adapter() in your subclass if needed."
        )

    def setup_evaluators(
        self,
        environment: Any,
        task: Task,
        agents: Any,
        user: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Create a fresh SeamTraceCallback for this task, then delegate to
        subclass evaluator setup.

        Subclasses should override this method calling super() to preserve
        seam instrumentation, or rely on the base return value of [].
        """
        task_id = str(getattr(task, "id", id(task)))
        self._active_callback = SeamTraceCallback(task_id=task_id)
        return []

    def _populate_seam_trace(self, traces: Any) -> None:
        """
        Extract agent handoffs from MASEval's collected message traces and
        feed them into the active SeamTraceCallback.

        MASEval stores per-agent message histories in
        traces["agents"][<name>]["messages"]. For each adapter we walk the
        message list and emit an on_handoff call whenever the speaker
        changes (identified by the "name" field, falling back to "role").
        """
        if not hasattr(self, "_active_callback"):
            return
        agent_traces = (traces or {}).get("agents", {})
        for agent_data in agent_traces.values():
            messages = agent_data.get("messages", [])
            if not messages:
                continue
            for i in range(len(messages) - 1):
                cur = messages[i]
                nxt = messages[i + 1]
                cur_name = cur.get("name") or cur.get("role", "unknown")
                nxt_name = nxt.get("name") or nxt.get("role", "unknown")
                if cur_name == nxt_name:
                    continue
                self._active_callback.on_handoff(
                    sender=cur_name,
                    receiver=nxt_name,
                    message=cur.get("content", ""),
                    context_passed=list(messages[: i + 1]),
                    context_available=list(messages[: i + 1]),
                    receiver_response=nxt.get("content"),
                )

    def evaluate(
        self,
        evaluators: list[Any],
        agents: Any,
        final_answer: Any,
        traces: Any,
    ) -> list[dict[str, Any]]:
        """
        Run all evaluators, then append handoff-level attribution.

        The handoff report is appended as an extra entry in the results list
        under the key "evaluator": "HandoffEvaluator".
        """
        results = super().evaluate(evaluators, agents, final_answer, traces) or []

        # Populate the seam trace from the collected message histories.
        self._populate_seam_trace(traces)

        # Run seam attribution if a trace was collected for this task.
        if hasattr(self, "_active_callback"):
            seam_trace = self._active_callback.trace
            self._seam_traces[seam_trace.task_id] = seam_trace

            # expected_agents can be provided via task metadata if available.
            expected_agents: list[str] | None = None
            handoff_report = self._handoff_evaluator.evaluate(
                seam_trace,
                expected_agents=expected_agents,
            )
            results.append(
                {
                    "evaluator": "HandoffEvaluator",
                    "result": handoff_report,
                }
            )
            logger.info(
                "SeamBenchmark: task=%s transitions=%d failures=%d",
                seam_trace.task_id,
                handoff_report["total_transitions"],
                handoff_report["failure_count"],
            )

        return results
