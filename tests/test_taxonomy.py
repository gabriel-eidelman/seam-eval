"""Tests for taxonomy.py — SeamEvent, SeamTrace, HandoffFailureMode."""

import pytest

from seam_eval.taxonomy import (
    FAILURE_MODE_DESCRIPTIONS,
    HandoffFailureMode,
    SeamEvent,
    SeamTrace,
)


def make_event(
    sender: str = "agent_a",
    receiver: str = "agent_b",
    message: str = "do the thing",
    failure_mode: HandoffFailureMode = HandoffFailureMode.NONE,
    turn_index: int = 0,
) -> SeamEvent:
    return SeamEvent(
        sender=sender,
        receiver=receiver,
        message=message,
        failure_mode=failure_mode,
        turn_index=turn_index,
    )


class TestHandoffFailureMode:
    def test_all_modes_have_descriptions(self) -> None:
        for mode in HandoffFailureMode:
            assert mode in FAILURE_MODE_DESCRIPTIONS
            assert FAILURE_MODE_DESCRIPTIONS[mode]

    def test_none_is_not_failure(self) -> None:
        event = make_event(failure_mode=HandoffFailureMode.NONE)
        assert not event.is_failure

    def test_any_other_mode_is_failure(self) -> None:
        for mode in HandoffFailureMode:
            if mode is HandoffFailureMode.NONE:
                continue
            event = make_event(failure_mode=mode)
            assert event.is_failure, f"{mode} should be a failure"


class TestSeamEvent:
    def test_summary_includes_turn_sender_receiver(self) -> None:
        event = make_event(sender="alice", receiver="bob", turn_index=3)
        summary = event.summary()
        assert "alice" in summary
        assert "bob" in summary
        assert "3" in summary

    def test_summary_shows_ok_for_none_mode(self) -> None:
        event = make_event(failure_mode=HandoffFailureMode.NONE)
        assert "ok" in event.summary()

    def test_summary_shows_failure_mode_value(self) -> None:
        event = make_event(failure_mode=HandoffFailureMode.CONTEXT_TRUNCATION)
        assert "context_truncation" in event.summary()


class TestSeamTrace:
    def test_empty_trace(self) -> None:
        trace = SeamTrace(task_id="t0")
        assert trace.failures == []
        assert trace.failure_rate == 0.0

    def test_failure_rate_calculation(self) -> None:
        trace = SeamTrace(task_id="t1")
        trace.events = [
            make_event(failure_mode=HandoffFailureMode.NONE),
            make_event(failure_mode=HandoffFailureMode.SILENT_SWALLOWING),
            make_event(failure_mode=HandoffFailureMode.CONTEXT_TRUNCATION),
            make_event(failure_mode=HandoffFailureMode.NONE),
        ]
        assert trace.failure_rate == pytest.approx(0.5)
        assert len(trace.failures) == 2

    def test_failure_counts(self) -> None:
        trace = SeamTrace(task_id="t2")
        trace.events = [
            make_event(failure_mode=HandoffFailureMode.CONTEXT_TRUNCATION),
            make_event(failure_mode=HandoffFailureMode.CONTEXT_TRUNCATION),
            make_event(failure_mode=HandoffFailureMode.SILENT_SWALLOWING),
        ]
        counts = trace.failure_counts()
        assert counts[HandoffFailureMode.CONTEXT_TRUNCATION] == 2
        assert counts[HandoffFailureMode.SILENT_SWALLOWING] == 1
        assert counts[HandoffFailureMode.NONE] == 0

    def test_report_is_string(self) -> None:
        trace = SeamTrace(task_id="t3")
        trace.events = [make_event(failure_mode=HandoffFailureMode.MISROUTED_HANDOFF)]
        trace.events[0].failure_rationale = "Expected 'specialist', got 'generalist'."
        report = trace.report()
        assert isinstance(report, str)
        assert "t3" in report
        assert "misrouted_handoff" in report
