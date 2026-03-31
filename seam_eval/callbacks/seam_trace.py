"""
SeamTraceCallback — instrumentation layer for agent handoffs.

Attaches to the MASEval evaluation lifecycle and records a SeamEvent for
every agent transition observed during a task run. Stateless per-task:
a fresh instance should be created for each task execution.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from seam_eval.taxonomy import (
    HandoffFailureMode,
    SeamEvent,
    SeamTrace,
)

logger = logging.getLogger(__name__)


class SeamTraceCallback:
    """
    Callback that records structured metadata for every agent transition.

    Design constraints
    ------------------
    - Stateless per-task: create a new instance per task run.
    - Non-destructive: never modifies messages or agent state.
    - Fault-tolerant: recording errors are logged, not raised.

    Collected data
    --------------
    Each SeamEvent captures:
    - sender / receiver agent names
    - the message content being handed off
    - context_passed: messages the sender forwarded
    - context_dropped: messages present in sender history but not forwarded
    - receiver_response: what the receiver returned
    - failure_mode: filled in by HandoffEvaluator post-hoc
    """

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._trace = SeamTrace(task_id=task_id)
        self._turn_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def trace(self) -> SeamTrace:
        """The accumulated trace for this task run."""
        return self._trace

    def reset(self, task_id: str) -> None:
        """Re-initialise the callback for a new task (avoids re-instantiation)."""
        self._task_id = task_id
        self._trace = SeamTrace(task_id=task_id)
        self._turn_index = 0

    # ------------------------------------------------------------------
    # MASEval Callback hooks
    # ------------------------------------------------------------------

    def on_agent_start(
        self,
        agent_name: str,
        query: str,
        **kwargs: Any,
    ) -> None:
        """Called when an agent begins processing a message."""
        logger.debug("SeamTrace: agent_start agent=%s turn=%d", agent_name, self._turn_index)

    def on_agent_end(
        self,
        agent_name: str,
        response: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes its turn."""
        logger.debug(
            "SeamTrace: agent_end agent=%s response_len=%d",
            agent_name,
            len(response or ""),
        )

    def on_handoff(
        self,
        sender: str,
        receiver: str,
        message: str,
        context_passed: Optional[list[dict[str, Any]]] = None,
        context_available: Optional[list[dict[str, Any]]] = None,
        receiver_response: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called at each agent transition.

        Parameters
        ----------
        sender:
            Name of the agent handing off.
        receiver:
            Name of the agent receiving.
        message:
            The message being passed at the seam.
        context_passed:
            Messages/context explicitly forwarded to the receiver.
        context_available:
            Full context that was available to the sender. Used to compute
            context_dropped = context_available - context_passed.
        receiver_response:
            The receiver's return value, if already known.
        metadata:
            Arbitrary extra data (token counts, timestamps, etc.).
        """
        try:
            passed = context_passed or []
            available = context_available or []
            dropped = self._compute_dropped(passed, available)

            event = SeamEvent(
                sender=sender,
                receiver=receiver,
                message=message,
                context_passed=passed,
                context_dropped=dropped,
                receiver_response=receiver_response,
                turn_index=self._turn_index,
                metadata=metadata or {},
            )
            self._trace.events.append(event)
            self._turn_index += 1
            logger.debug("SeamTrace: recorded %s", event.summary())
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("SeamTrace: failed to record handoff: %s", exc)

    def on_task_end(
        self,
        final_answer: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the task run completes (before evaluation)."""
        self._trace.metadata.update(metadata or {})
        if final_answer is not None:
            self._trace.metadata["final_answer"] = final_answer
        logger.debug(
            "SeamTrace: task_end task_id=%s events=%d failures=%d",
            self._task_id,
            len(self._trace.events),
            len(self._trace.failures),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dropped(
        passed: list[dict[str, Any]],
        available: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Return messages that were available to the sender but not forwarded.

        Uses content-based identity (not object identity) so this works
        across serialisation boundaries.
        """
        if not available:
            return []
        passed_contents = {m.get("content", "") for m in passed}
        return [m for m in available if m.get("content", "") not in passed_contents]
