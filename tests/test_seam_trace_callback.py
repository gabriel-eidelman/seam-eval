"""Tests for SeamTraceCallback — lifecycle instrumentation."""

from seam_eval.callbacks.seam_trace import SeamTraceCallback
from seam_eval.taxonomy import HandoffFailureMode


class TestSeamTraceCallback:
    def test_initial_state(self) -> None:
        cb = SeamTraceCallback(task_id="t0")
        assert cb.trace.task_id == "t0"
        assert cb.trace.events == []

    def test_on_handoff_records_event(self) -> None:
        cb = SeamTraceCallback(task_id="t1")
        cb.on_handoff(
            sender="agent_a",
            receiver="agent_b",
            message="do x",
            receiver_response="done x",
        )
        assert len(cb.trace.events) == 1
        event = cb.trace.events[0]
        assert event.sender == "agent_a"
        assert event.receiver == "agent_b"
        assert event.message == "do x"
        assert event.receiver_response == "done x"
        assert event.failure_mode is HandoffFailureMode.NONE

    def test_turn_index_increments(self) -> None:
        cb = SeamTraceCallback(task_id="t2")
        for _ in range(3):
            cb.on_handoff(sender="a", receiver="b", message="msg")
        indices = [e.turn_index for e in cb.trace.events]
        assert indices == [0, 1, 2]

    def test_context_dropped_computed(self) -> None:
        cb = SeamTraceCallback(task_id="t3")
        available = [{"content": f"m{i}"} for i in range(4)]
        passed = available[:2]
        cb.on_handoff(
            sender="a",
            receiver="b",
            message="msg",
            context_passed=passed,
            context_available=available,
        )
        event = cb.trace.events[0]
        assert len(event.context_passed) == 2
        assert len(event.context_dropped) == 2

    def test_reset_clears_state(self) -> None:
        cb = SeamTraceCallback(task_id="old")
        cb.on_handoff(sender="a", receiver="b", message="x")
        cb.reset(task_id="new")
        assert cb.trace.task_id == "new"
        assert cb.trace.events == []

    def test_on_task_end_stores_metadata(self) -> None:
        cb = SeamTraceCallback(task_id="t4")
        cb.on_task_end(final_answer="42", metadata={"tokens": 100})
        assert cb.trace.metadata["final_answer"] == "42"
        assert cb.trace.metadata["tokens"] == 100

    def test_bad_handoff_does_not_raise(self) -> None:
        cb = SeamTraceCallback(task_id="t5")
        # Pass a deliberately broken context list to exercise fault tolerance.
        cb.on_handoff(
            sender="a",
            receiver="b",
            message="msg",
            context_passed=[None],  # type: ignore[list-item]
            context_available=[None],  # type: ignore[list-item]
        )
        # Should have recorded the event despite the bad input.
        assert len(cb.trace.events) == 1
