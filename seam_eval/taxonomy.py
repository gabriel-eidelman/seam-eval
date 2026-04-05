"""
Handoff failure taxonomy for multi-agent LLM systems.

Failures are defined as genuine structural breakdowns at agent transition
boundaries (seams). Correct behavior — such as an agent reporting it lacks
access to data, declining an out-of-scope task, or returning a well-formed
null result — is NOT a failure and is excluded from this taxonomy.

Add new failure modes here before implementing detection logic.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional

from maseval import Task


class SeamTask(Task):
    """
    A MASEval Task augmented with a concise description of intended behavior.

    ``intended_behavior`` tells the HandoffEvaluator what a correct run should
    look like — what answer is expected, what agents should do, and what process
    is appropriate. The evaluator uses this to distinguish genuine failures from
    correct agent behavior (e.g. an agent correctly reporting it lacks access to
    live data is NOT a failure).

    Parameters
    ----------
    intended_behavior:
        One to three sentences describing the expected answer, the expected
        agent process, or both. Used verbatim in the evaluator prompt.
    """

    def __init__(self, *args: Any, intended_behavior: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.intended_behavior: str = intended_behavior


class FailureCategory(Enum):
    """High-level grouping of failure families."""

    ROUTING = "routing"
    LOOP = "loop"
    CONTEXT = "context"
    TERMINATION = "termination"
    OUTPUT = "output"


class HandoffFailureMode(Enum):
    """
    Structured classification of multi-agent failure modes at handoff
    boundaries.

    Each value represents a distinct structural breakdown. Correct,
    intended agent behavior (e.g. refusing out-of-scope tasks, reporting
    missing access) is NOT a failure and should be classified as NONE.
    """

    # ------------------------------------------------------------------ #
    # Routing failures — task sent to the wrong place or owned by no one  #
    # ------------------------------------------------------------------ #

    # Task dispatched to a specialist not equipped for it; that agent
    # attempts execution anyway and produces a wrong or degraded result.
    MISROUTED_HANDOFF = "misrouted_handoff"

    # Two or more agents each assume a subtask belongs to the other;
    # no agent executes it and the gap is not surfaced.
    RESPONSIBILITY_GAP = "responsibility_gap"

    # ------------------------------------------------------------------ #
    # Loop failures — control flow never terminates                        #
    # ------------------------------------------------------------------ #

    # A single agent re-invokes itself without making progress; the loop
    # continues until a token or turn budget is exhausted.
    AGENT_SELF_LOOP = "agent_self_loop"

    # Two agents hand off to each other indefinitely; neither reaches a
    # terminal condition or escalates the deadlock.
    MUTUAL_LOOP = "mutual_loop"

    # ------------------------------------------------------------------ #
    # Context failures — information state at the seam is wrong           #
    # ------------------------------------------------------------------ #

    # The receiving agent lacks sufficient context to complete the task
    # correctly. It does not explicitly fail; instead it silently produces
    # an incorrect or partial output.
    CONTEXT_INSUFFICIENCY = "context_insufficiency"

    # The receiving agent is passed stale, conflicting, or structurally
    # malformed context that causes downstream processing errors.
    CONTEXT_CORRUPTION = "context_corruption"

    # ------------------------------------------------------------------ #
    # Termination failures — the chain stops at the wrong moment          #
    # ------------------------------------------------------------------ #

    # A stop condition fires on an intermediate output, halting work
    # before the task is complete.
    PREMATURE_TERMINATION = "premature_termination"

    # No agent triggers a stop condition after the task is complete;
    # the chain continues executing unnecessarily, consuming resources
    # or producing spurious side effects.
    MISSED_TERMINATION = "missed_termination"

    # ------------------------------------------------------------------ #
    # Output failures — the result is structurally wrong                  #
    # ------------------------------------------------------------------ #

    # Small framing or interpretation errors accumulate across a multi-hop
    # chain; the terminal agent's output diverges significantly from the
    # original task goal despite no single catastrophic failure.
    COMPOUNDING_DRIFT = "compounding_drift"

    # Two or more agents each complete the same subtask independently;
    # their results conflict and the system has no reconciliation strategy.
    DUPLICATE_EXECUTION = "duplicate_execution"

    # Successful handoff — no failure detected.
    NONE = "none"


# Failure category membership for grouping and filtering.
FAILURE_CATEGORIES: dict[HandoffFailureMode, FailureCategory] = {
    HandoffFailureMode.MISROUTED_HANDOFF:    FailureCategory.ROUTING,
    HandoffFailureMode.RESPONSIBILITY_GAP:   FailureCategory.ROUTING,
    HandoffFailureMode.AGENT_SELF_LOOP:      FailureCategory.LOOP,
    HandoffFailureMode.MUTUAL_LOOP:          FailureCategory.LOOP,
    HandoffFailureMode.CONTEXT_INSUFFICIENCY: FailureCategory.CONTEXT,
    HandoffFailureMode.CONTEXT_CORRUPTION:   FailureCategory.CONTEXT,
    HandoffFailureMode.PREMATURE_TERMINATION: FailureCategory.TERMINATION,
    HandoffFailureMode.MISSED_TERMINATION:   FailureCategory.TERMINATION,
    HandoffFailureMode.COMPOUNDING_DRIFT:    FailureCategory.OUTPUT,
    HandoffFailureMode.DUPLICATE_EXECUTION:  FailureCategory.OUTPUT,
    HandoffFailureMode.NONE:                 None,
}

# Human-readable descriptions for each mode, used in reports and evals.
FAILURE_MODE_DESCRIPTIONS: dict[HandoffFailureMode, str] = {
    HandoffFailureMode.MISROUTED_HANDOFF: (
        "Task was dispatched to a specialist not equipped for it; that agent "
        "attempted execution anyway and produced a wrong or degraded result."
    ),
    HandoffFailureMode.RESPONSIBILITY_GAP: (
        "Two or more agents each assumed a subtask belonged to the other; "
        "no agent executed it and the gap was not surfaced."
    ),
    HandoffFailureMode.AGENT_SELF_LOOP: (
        "A single agent re-invoked itself without making progress; the loop "
        "continued until a token or turn budget was exhausted."
    ),
    HandoffFailureMode.MUTUAL_LOOP: (
        "Two agents handed off to each other indefinitely; neither reached a "
        "terminal condition or escalated the deadlock."
    ),
    HandoffFailureMode.CONTEXT_INSUFFICIENCY: (
        "The receiving agent lacked sufficient context to complete the task "
        "correctly. It did not explicitly fail — it silently produced an "
        "incorrect or partial output."
    ),
    HandoffFailureMode.CONTEXT_CORRUPTION: (
        "The receiving agent was passed stale, conflicting, or structurally "
        "malformed context that caused downstream processing errors."
    ),
    HandoffFailureMode.PREMATURE_TERMINATION: (
        "A stop condition fired on an intermediate output, halting work "
        "before the task was complete."
    ),
    HandoffFailureMode.MISSED_TERMINATION: (
        "No agent triggered a stop condition after task completion; the chain "
        "continued executing, consuming resources or producing spurious output."
    ),
    HandoffFailureMode.COMPOUNDING_DRIFT: (
        "Small framing or interpretation errors accumulated across a multi-hop "
        "chain; the terminal output diverged from the original goal despite no "
        "single catastrophic failure."
    ),
    HandoffFailureMode.DUPLICATE_EXECUTION: (
        "Two or more agents each completed the same subtask independently; "
        "their results conflicted and no reconciliation strategy was available."
    ),
    HandoffFailureMode.NONE: (
        "Successful handoff — no structural failure detected."
    ),
}


@dataclass
class SeamEvent:
    """
    A single agent transition recorded during task execution.

    Captures everything needed for post-hoc failure attribution:
    who sent, who received, what context was passed, and what was dropped.

    Note: an agent correctly reporting that it lacks access to data, or
    declining an out-of-scope task, should NOT be recorded as a failure.
    Set failure_mode = HandoffFailureMode.NONE in those cases.
    """

    sender: str
    receiver: str
    message: str

    # Subset of the sender's internal context forwarded to the receiver.
    context_passed: list[dict[str, Any]] = field(default_factory=list)

    # Context present in the sender that was NOT forwarded.
    context_dropped: list[dict[str, Any]] = field(default_factory=dict)

    # Raw return value from the receiver's turn.
    receiver_response: Optional[str] = None

    # Failure classification assigned by HandoffEvaluator.
    failure_mode: HandoffFailureMode = HandoffFailureMode.NONE

    # Free-text rationale for the classification.
    failure_rationale: Optional[str] = None

    # Sequence number within the task (0-indexed).
    turn_index: int = 0

    # Arbitrary metadata (e.g. token counts, timestamps, loop counters).
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.failure_mode is not HandoffFailureMode.NONE

    @property
    def category(self) -> Optional[FailureCategory]:
        return FAILURE_CATEGORIES.get(self.failure_mode)

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
            mode: 0 for mode in HandoffFailureMode if mode is not HandoffFailureMode.NONE
        }
        for event in self.failures:
            counts[event.failure_mode] += 1
        return counts

    def failures_by_category(self) -> dict[FailureCategory, list[SeamEvent]]:
        result: dict[FailureCategory, list[SeamEvent]] = {c: [] for c in FailureCategory}
        for event in self.failures:
            if event.category is not None:
                result[event.category].append(event)
        return result

    def report(self) -> str:
        lines = [
            f"SeamTrace for task '{self.task_id}'",
            f"  Transitions : {len(self.events)}",
            f"  Failures    : {len(self.failures)} "
            f"({self.failure_rate:.0%})",
        ]

        by_category = self.failures_by_category()
        for category, events in by_category.items():
            if events:
                lines.append(f"  [{category.value.upper()}]")
                for event in events:
                    lines.append(f"    {event.summary()}")
                    if event.failure_rationale:
                        lines.append(f"      ↳ {event.failure_rationale}")

        non_failures = [e for e in self.events if not e.is_failure]
        if non_failures:
            lines.append(f"  [OK]")
            for event in non_failures:
                lines.append(f"    {event.summary()}")

        return "\n".join(lines)