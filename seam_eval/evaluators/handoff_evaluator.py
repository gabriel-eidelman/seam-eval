"""
HandoffEvaluator — classifies failure modes in a SeamTrace.

Processes the SeamEvents collected by SeamTraceCallback and assigns a
HandoffFailureMode to each transition based on structural heuristics.
LLM-based classification can be layered on top via subclassing.
"""

from __future__ import annotations

import logging
from typing import Any

from maseval import Evaluator

from seam_eval.taxonomy import (
    FAILURE_MODE_DESCRIPTIONS,
    HandoffFailureMode,
    SeamEvent,
    SeamTrace,
)

logger = logging.getLogger(__name__)


# Soft-failure signals: phrases that indicate an agent couldn't handle a task
# but returned a plausible-sounding non-answer.
_SOFT_FAILURE_SIGNALS = (
    "i cannot",
    "i can't",
    "i don't have access",
    "i'm unable",
    "i am unable",
    "not able to",
    "outside my scope",
    "i don't know",
    "i do not know",
    "as an ai",
)

# Phrases that suggest a conversation terminated before completion.
_PREMATURE_TERMINATION_SIGNALS = (
    "terminate",
    "exit",
    "goodbye",
    "task complete",
    "done.",
    "finished.",
)

# Threshold: if context_dropped / context_available > this, flag truncation.
_TRUNCATION_RATIO_THRESHOLD = 0.5

# Threshold: if response is shorter than this fraction of the message, suspect swallowing.
_SWALLOWING_RESPONSE_RATIO = 0.1


class HandoffEvaluator(Evaluator):
    """
    Structural heuristic evaluator for handoff-level failure attribution.

    For each SeamEvent in a trace, applies a deterministic rule cascade to
    assign the most likely HandoffFailureMode. The cascade is ordered by
    specificity — more specific checks run first.

    To extend with LLM-based classification, subclass and override
    `_classify_event`.

    Note: HandoffEvaluator is invoked directly by SeamBenchmark rather than
    through MASEval's standard evaluator pipeline. The `__call__` and
    `filter_traces` methods are stubs to satisfy the abstract interface.
    """

    def __init__(self) -> None:
        # Skip Evaluator.__init__ — HandoffEvaluator doesn't need task/environment.
        pass

    def filter_traces(self, traces: Any) -> Any:
        return traces

    def __call__(self, traces: Any, final_answer: Any = None) -> dict[str, Any]:
        return {}

    def evaluate(
        self,
        trace: SeamTrace,
        expected_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Classify all SeamEvents in the trace and return a structured report.

        Parameters
        ----------
        trace:
            The SeamTrace produced by SeamTraceCallback for this task.
        expected_agents:
            Optional list of agent names that are the legitimate recipients
            for each turn. Used by the misrouted-handoff detector.

        Returns
        -------
        dict with keys:
            - task_id
            - total_transitions
            - failure_count
            - failure_rate
            - failure_counts   (per HandoffFailureMode)
            - events           (per-event classification results)
        """
        for i, event in enumerate(trace.events):
            expected = (expected_agents[i] if expected_agents and i < len(expected_agents) else None)
            mode, rationale = self._classify_event(event, expected_receiver=expected)
            event.failure_mode = mode
            event.failure_rationale = rationale

        return self._build_report(trace)

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify_event(
        self,
        event: SeamEvent,
        expected_receiver: str | None = None,
    ) -> tuple[HandoffFailureMode, str | None]:
        """
        Apply the heuristic cascade to a single SeamEvent.

        Returns (mode, rationale_string).
        """
        checks = [
            self._check_misrouted_handoff,
            self._check_context_truncation,
            self._check_silent_swallowing,
            self._check_premature_termination,
            self._check_responsibility_gap,
            self._check_compounding_drift,
        ]
        for check in checks:
            mode, rationale = check(event, expected_receiver=expected_receiver)
            if mode is not HandoffFailureMode.NONE:
                logger.debug(
                    "HandoffEvaluator: %s classified as %s",
                    event.summary(),
                    mode.value,
                )
                return mode, rationale

        return HandoffFailureMode.NONE, None

    def _check_context_truncation(
        self,
        event: SeamEvent,
        **_: Any,
    ) -> tuple[HandoffFailureMode, str | None]:
        available = len(event.context_passed) + len(event.context_dropped)
        if available == 0:
            return HandoffFailureMode.NONE, None
        drop_ratio = len(event.context_dropped) / available
        if drop_ratio >= _TRUNCATION_RATIO_THRESHOLD:
            return (
                HandoffFailureMode.CONTEXT_TRUNCATION,
                f"{drop_ratio:.0%} of available context was dropped "
                f"({len(event.context_dropped)}/{available} messages).",
            )
        return HandoffFailureMode.NONE, None

    def _check_misrouted_handoff(
        self,
        event: SeamEvent,
        expected_receiver: str | None = None,
        **_: Any,
    ) -> tuple[HandoffFailureMode, str | None]:
        if expected_receiver and event.receiver != expected_receiver:
            return (
                HandoffFailureMode.MISROUTED_HANDOFF,
                f"Expected receiver '{expected_receiver}', got '{event.receiver}'.",
            )
        return HandoffFailureMode.NONE, None

    def _check_silent_swallowing(
        self,
        event: SeamEvent,
        **_: Any,
    ) -> tuple[HandoffFailureMode, str | None]:
        response = (event.receiver_response or "").lower().strip()
        if not response:
            return (
                HandoffFailureMode.SILENT_SWALLOWING,
                "Receiver returned an empty response.",
            )
        if any(signal in response for signal in _SOFT_FAILURE_SIGNALS):
            matched = next(s for s in _SOFT_FAILURE_SIGNALS if s in response)
            return (
                HandoffFailureMode.SILENT_SWALLOWING,
                f"Receiver response contains soft-failure signal: '{matched}'.",
            )
        return HandoffFailureMode.NONE, None

    def _check_premature_termination(
        self,
        event: SeamEvent,
        **_: Any,
    ) -> tuple[HandoffFailureMode, str | None]:
        response = (event.receiver_response or "").lower().strip()
        message = event.message.lower().strip()
        if any(sig in response for sig in _PREMATURE_TERMINATION_SIGNALS):
            # Only flag if the original message still had clear open tasks.
            has_open_task = "?" in message or any(
                kw in message for kw in ("please", "need", "find", "calculate", "determine")
            )
            if has_open_task:
                return (
                    HandoffFailureMode.PREMATURE_TERMINATION,
                    "Receiver signalled termination while the task message contained open questions.",
                )
        return HandoffFailureMode.NONE, None

    def _check_responsibility_gap(
        self,
        event: SeamEvent,
        **_: Any,
    ) -> tuple[HandoffFailureMode, str | None]:
        # Heuristic: if the message explicitly delegates ("please handle",
        # "over to you", "you should") but the response acknowledges without
        # acting, flag a responsibility gap.
        msg = event.message.lower()
        resp = (event.receiver_response or "").lower()
        delegation_signals = ("over to you", "please handle", "you should", "your turn")
        acknowledgement_only = ("understood", "noted", "got it", "i see", "acknowledged")
        if any(d in msg for d in delegation_signals) and any(
            a in resp for a in acknowledgement_only
        ) and len(resp) < 200:
            return (
                HandoffFailureMode.RESPONSIBILITY_GAP,
                "Message delegated a task but receiver only acknowledged without acting.",
            )
        return HandoffFailureMode.NONE, None

    def _check_compounding_drift(
        self,
        event: SeamEvent,
        **_: Any,
    ) -> tuple[HandoffFailureMode, str | None]:
        # Requires cross-event analysis; single-event heuristic is weak.
        # This stub is intentionally minimal — extend via subclassing or
        # post-process traces at the benchmark level.
        return HandoffFailureMode.NONE, None

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report(trace: SeamTrace) -> dict[str, Any]:
        failure_counts = trace.failure_counts()
        return {
            "task_id": trace.task_id,
            "total_transitions": len(trace.events),
            "failure_count": len(trace.failures),
            "failure_rate": trace.failure_rate,
            "failure_counts": {
                mode.value: count
                for mode, count in failure_counts.items()
                if count > 0
            },
            "events": [
                {
                    "turn_index": e.turn_index,
                    "sender": e.sender,
                    "receiver": e.receiver,
                    "failure_mode": e.failure_mode.value,
                    "failure_rationale": e.failure_rationale,
                    "description": FAILURE_MODE_DESCRIPTIONS[e.failure_mode],
                }
                for e in trace.events
            ],
        }
