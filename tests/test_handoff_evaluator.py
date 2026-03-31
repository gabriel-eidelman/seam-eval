"""Tests for HandoffEvaluator — heuristic failure classification."""

import pytest

from seam_eval.evaluators.handoff_evaluator import HandoffEvaluator
from seam_eval.taxonomy import HandoffFailureMode, SeamEvent, SeamTrace


def make_trace(task_id: str = "task-0", events: list[SeamEvent] | None = None) -> SeamTrace:
    trace = SeamTrace(task_id=task_id)
    trace.events = events or []
    return trace


def make_event(
    sender: str = "a",
    receiver: str = "b",
    message: str = "do something",
    receiver_response: str = "done",
    context_passed: list | None = None,
    context_dropped: list | None = None,
    turn_index: int = 0,
) -> SeamEvent:
    return SeamEvent(
        sender=sender,
        receiver=receiver,
        message=message,
        receiver_response=receiver_response,
        context_passed=context_passed or [],
        context_dropped=context_dropped or [],
        turn_index=turn_index,
    )


@pytest.fixture()
def evaluator() -> HandoffEvaluator:
    return HandoffEvaluator()


class TestContextTruncation:
    def test_detects_majority_drop(self, evaluator: HandoffEvaluator) -> None:
        msgs = [{"content": f"msg{i}"} for i in range(4)]
        event = make_event(
            context_passed=msgs[:1],
            context_dropped=msgs[1:],
        )
        trace = make_trace(events=[event])
        report = evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.CONTEXT_TRUNCATION

    def test_no_flag_when_drop_is_minority(self, evaluator: HandoffEvaluator) -> None:
        msgs = [{"content": f"msg{i}"} for i in range(4)]
        event = make_event(
            context_passed=msgs[:3],
            context_dropped=msgs[3:],
        )
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.NONE

    def test_no_flag_when_no_context(self, evaluator: HandoffEvaluator) -> None:
        event = make_event()
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.NONE


class TestMisroutedHandoff:
    def test_detects_wrong_receiver(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(receiver="specialist_b")
        trace = make_trace(events=[event])
        report = evaluator.evaluate(trace, expected_agents=["specialist_a"])
        assert event.failure_mode is HandoffFailureMode.MISROUTED_HANDOFF

    def test_no_flag_when_receiver_matches(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(receiver="specialist_a")
        trace = make_trace(events=[event])
        evaluator.evaluate(trace, expected_agents=["specialist_a"])
        assert event.failure_mode is HandoffFailureMode.NONE


class TestSilentSwallowing:
    def test_detects_empty_response(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(receiver_response="")
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.SILENT_SWALLOWING

    def test_detects_soft_failure_phrase(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(receiver_response="I cannot help with that.")
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.SILENT_SWALLOWING

    def test_no_flag_for_normal_response(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(receiver_response="The answer is 42.")
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.NONE


class TestPrematureTermination:
    def test_detects_terminate_on_open_task(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(
            message="Please find the best route?",
            receiver_response="TERMINATE",
        )
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        assert event.failure_mode is HandoffFailureMode.PREMATURE_TERMINATION

    def test_no_flag_when_task_is_closed(self, evaluator: HandoffEvaluator) -> None:
        event = make_event(
            message="Here is the summary.",
            receiver_response="TERMINATE",
        )
        trace = make_trace(events=[event])
        evaluator.evaluate(trace)
        # No open task markers → should not flag.
        assert event.failure_mode is HandoffFailureMode.NONE


class TestReport:
    def test_report_structure(self, evaluator: HandoffEvaluator) -> None:
        trace = make_trace(
            task_id="task-report",
            events=[
                make_event(receiver_response="I cannot do that."),
                make_event(receiver_response="Sure, here's the result.", turn_index=1),
            ],
        )
        report = evaluator.evaluate(trace)
        assert report["task_id"] == "task-report"
        assert report["total_transitions"] == 2
        assert "failure_rate" in report
        assert isinstance(report["events"], list)
        assert len(report["events"]) == 2
