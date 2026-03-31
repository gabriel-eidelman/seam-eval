"""
Handoff failure taxonomy for multi-agent LLM systems.

Every failure mode represents a distinct structural breakdown at an agent
transition (seam). Add new modes here before implementing detection logic.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


class HandoffFailureMode(Enum):
    """Structured classification of multi-agent failure modes at seams."""

    # Agent B receives an impoverished representation of what agent A reasoned.
    CONTEXT_TRUNCATION = "context_truncation"

    # Task is routed to the wrong specialist; receiving agent attempts it anyway.
    MISROUTED_HANDOFF = "misrouted_handoff"

    # Agent receives a task it cannot handle, returns soft failure, system continues.
    SILENT_SWALLOWING = "silent_swallowing"

    # Two agents each assume the other owns a subtask; neither executes it.
    RESPONSIBILITY_GAP = "responsibility_gap"

    # Small framing errors accumulate across a chain; terminal task diverges.
    COMPOUNDING_DRIFT = "compounding_drift"

    # Handoff/stop condition triggers on intermediate output, cutting off work.
    PREMATURE_TERMINATION = "premature_termination"

    # Successful handoff — no failure detected.
    NONE = "none"


# Human-readable descriptions for each mode, used in reports.
FAILURE_MODE_DESCRIPTIONS: dict[HandoffFailureMode, str] = {
    HandoffFailureMode.CONTEXT_TRUNCATION: (
        "Agent B received an impoverished representation of agent A's reasoning."
    ),
    HandoffFailureMode.MISROUTED_HANDOFF: (
        "Task was sent to the wrong specialist; receiving agent attempted it anyway."
    ),
    HandoffFailureMode.SILENT_SWALLOWING: (
        "Agent received an unhandleable task, returned a soft failure, system continued."
    ),
    HandoffFailureMode.RESPONSIBILITY_GAP: (
        "Two agents each assumed the other owned a subtask; neither executed it."
    ),
    HandoffFailureMode.COMPOUNDING_DRIFT: (
        "Small framing errors accumulated across a chain; terminal task diverged from original."
    ),
    HandoffFailureMode.PREMATURE_TERMINATION: (
        "Handoff/stop condition triggered on intermediate output, cutting off incomplete work."
    ),
    HandoffFailureMode.NONE: "Successful handoff — no failure detected.",
}


@dataclass
class SeamEvent:
    """
    A single agent transition recorded during task execution.

    Captures everything needed for post-hoc failure attribution:
    who sent, who received, what context was passed, and what was dropped.
    """

    sender: str
    receiver: str
    message: str

    # Subset of the sender's internal context forwarded to the receiver.
    context_passed: list[dict[str, Any]] = field(default_factory=list)

    # Context present in the sender that was NOT forwarded.
    context_dropped: list[dict[str, Any]] = field(default_factory=list)

    # Raw return value from the receiver's turn.
    receiver_response: Optional[str] = None

    # Failure classification assigned by HandoffEvaluator.
    failure_mode: HandoffFailureMode = HandoffFailureMode.NONE

    # Free-text rationale for the classification.
    failure_rationale: Optional[str] = None

    # Sequence number within the task (0-indexed).
    turn_index: int = 0

    # Arbitrary metadata (e.g. token counts, timestamps).
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.failure_mode is not HandoffFailureMode.NONE

    def summary(self) -> str:
        status = self.failure_mode.value if self.is_failure else "ok"
        return (
            f"[turn {self.turn_index}] {self.sender} → {self.receiver} "
            f"({status})"
        )


@dataclass
class SeamTrace:
    """
    Aggregated collection of SeamEvents for a single task run.

    Produced by SeamTraceCallback; consumed by HandoffEvaluator.
    """

    task_id: str
    events: list[SeamEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[SeamEvent]:
        return [e for e in self.events if e.is_failure]

    @property
    def failure_rate(self) -> float:
        if not self.events:
            return 0.0
        return len(self.failures) / len(self.events)

    def failure_counts(self) -> dict[HandoffFailureMode, int]:
        counts: dict[HandoffFailureMode, int] = {
            mode: 0 for mode in HandoffFailureMode
        }
        for event in self.failures:
            counts[event.failure_mode] += 1
        return counts

    def report(self) -> str:
        lines = [
            f"SeamTrace for task '{self.task_id}'",
            f"  Transitions : {len(self.events)}",
            f"  Failures    : {len(self.failures)} "
            f"({self.failure_rate:.0%})",
        ]
        for event in self.events:
            lines.append(f"  {event.summary()}")
            if event.failure_rationale:
                lines.append(f"    ↳ {event.failure_rationale}")
        return "\n".join(lines)
