"""
HandoffEvaluator — LLM-based failure-mode classifier for SeamTraces.

Formats the full SeamEvent transcript for a task run and submits it to an
LLM that returns per-event failure classifications as structured JSON.

Usage
-----
    evaluator = HandoffEvaluator(model="gpt-4o-mini", api_key="sk-...")
    report = evaluator.evaluate(trace)

To use a different OpenAI-compatible endpoint, pass ``base_url``.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from maseval import Evaluator

from seam_eval.taxonomy import (
    FAILURE_MODE_DESCRIPTIONS,
    HandoffFailureMode,
    SeamEvent,
    SeamTrace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert evaluator of multi-agent LLM systems. Your job is to
    analyze a transcript of agent-to-agent handoffs and classify each transition
    according to a structured failure taxonomy.

    ## Key Principle

    You will be given a description of the INTENDED BEHAVIOR for the task.
    Use it to distinguish genuine structural failures from correct agent behavior.
    An agent correctly reporting it lacks access to live data, declining an
    out-of-scope request, or returning a well-formed null result is NOT a
    failure — classify those as "none". Only flag transitions where something
    genuinely went wrong in the coordination between agents.

    ## Failure Taxonomy

    Each transition must be assigned exactly one of the following labels:

    none
        Successful handoff — the receiving agent received adequate context,
        behaved in accordance with its role, and produced a substantive
        on-task response (or a correct refusal). No structural failure detected.

    misrouted_handoff
        The task was dispatched to a specialist not equipped for it, and that
        agent attempted execution anyway, producing a wrong or degraded result.
        (An agent correctly refusing an out-of-scope task is NOT this failure.)

    responsibility_gap
        The sender explicitly delegates a subtask but no agent actually executes
        it. The receiver either only acknowledges without acting, or passes the
        task back, leaving a required subtask permanently unowned.

    agent_self_loop
        A single agent re-invokes itself repeatedly without making meaningful
        progress. The loop continues until a turn or token budget is exhausted.

    mutual_loop
        Two agents hand off to each other indefinitely; neither reaches a
        terminal condition or escalates the deadlock.

    context_insufficiency
        The receiving agent lacks the context needed to complete the task
        correctly. It does not explicitly fail — it silently produces an
        incorrect or partial output because essential prior reasoning,
        intermediate results, or task constraints were not forwarded.

    context_corruption
        The receiving agent was passed stale, conflicting, or structurally
        malformed context that caused it to behave incorrectly or produce
        downstream errors.

    premature_termination
        A stop or termination signal fires before the task is fully complete.
        Work is halted on an intermediate output with open sub-steps remaining.

    missed_termination
        No agent triggers a stop condition after the task is complete. The
        chain continues executing unnecessarily, consuming resources or
        producing spurious side effects.

    compounding_drift
        Small framing or interpretation errors accumulate across a multi-hop
        chain such that the content handed off at this transition has
        materially diverged from the original task intent, even though no
        single prior transition was catastrophically wrong.

    duplicate_execution
        Two or more agents each complete the same subtask independently;
        their results conflict and the system has no reconciliation strategy.

    ## Output Format

    Respond with a single JSON object — no markdown fences, no prose.

    {
      "overall_summary": "<one or two sentence summary of the conversation>",
      "events": [
        {
          "turn_index": <integer>,
          "failure_mode": "<one of the labels above>",
          "confidence": "<high | medium | low>",
          "rationale": "<one or two sentences explaining the classification>"
        }
      ]
    }

    The "events" array must contain exactly one entry per transition in the
    transcript, in the same order, with the same turn_index values.
    Only output valid JSON. Do not include any text outside the JSON object.
""")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class HandoffEvaluator(Evaluator):
    """
    LLM-based handoff failure evaluator.

    Builds a structured transcript from the SeamEvents in a SeamTrace,
    submits it to an LLM with a taxonomy-grounded prompt, and parses the
    structured JSON response back into per-event failure classifications.

    Parameters
    ----------
    model:
        OpenAI model ID to use for evaluation (default: "gpt-4o-mini").
    api_key:
        OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
    base_url:
        Optional base URL for OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "HandoffEvaluator requires openai. Run: pip install openai"
            ) from exc

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._model = model

    # ------------------------------------------------------------------
    # MASEval Evaluator interface stubs
    # ------------------------------------------------------------------

    def filter_traces(self, traces: Any) -> Any:
        return traces

    def __call__(self, traces: Any, final_answer: Any = None) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trace: SeamTrace,
        intended_behavior: str | None = None,
        expected_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Classify all SeamEvents in the trace via LLM and return a report.

        Parameters
        ----------
        trace:
            The SeamTrace produced by SeamTraceCallback for this task.
        intended_behavior:
            Concise description of what a correct run should look like —
            expected answer, expected process, or both. Injected into the
            transcript so the LLM can distinguish genuine failures from
            correct agent behavior.
        expected_agents:
            Unused — kept for interface compatibility. The LLM infers routing
            failures from the transcript itself.

        Returns
        -------
        dict with keys:
            - task_id
            - total_transitions
            - failure_count
            - failure_rate
            - failure_counts   (per HandoffFailureMode)
            - overall_summary  (LLM-generated)
            - events           (per-event classification results)
        """
        if not trace.events:
            return self._empty_report(trace.task_id)

        transcript = self._build_transcript(trace, intended_behavior=intended_behavior)
        llm_result = self._classify_with_llm(transcript, trace)
        self._apply_classifications(trace.events, llm_result.get("events", []))

        report = self._build_report(trace)
        report["overall_summary"] = llm_result.get("overall_summary", "")
        return report

    # ------------------------------------------------------------------
    # Transcript builder
    # ------------------------------------------------------------------

    def _build_transcript(
        self, trace: SeamTrace, intended_behavior: str | None = None
    ) -> str:
        """Format SeamEvents into a readable transcript for the LLM."""
        lines: list[str] = [
            "=== AGENT TRANSITION TRANSCRIPT ===",
            f"Task ID     : {trace.task_id}",
            f"Transitions : {len(trace.events)}",
        ]
        if intended_behavior:
            lines += [
                "",
                "--- Intended Behavior ---",
                intended_behavior.strip(),
                "--- End Intended Behavior ---",
            ]
        lines.append("")
        for event in trace.events:
            lines += self._format_event(event)
        return "\n".join(lines)

    @staticmethod
    def _format_event(event: SeamEvent) -> list[str]:
        lines = [
            f"--- Transition {event.turn_index} ---",
            f"Sender   : {event.sender}",
            f"Receiver : {event.receiver}",
            f"Message  : {event.message.strip()!r}",
        ]

        if event.context_passed:
            lines.append(f"Context forwarded ({len(event.context_passed)} messages):")
            for i, msg in enumerate(event.context_passed):
                speaker = msg.get("name") or msg.get("role", "?")
                content = (msg.get("content") or "").strip()
                preview = content[:200] + ("…" if len(content) > 200 else "")
                lines.append(f"  [{i}] {speaker}: {preview!r}")
        else:
            lines.append("Context forwarded: (none)")

        if event.context_dropped:
            lines.append(f"Context dropped ({len(event.context_dropped)} messages):")
            for i, msg in enumerate(event.context_dropped):
                speaker = msg.get("name") or msg.get("role", "?")
                content = (msg.get("content") or "").strip()
                preview = content[:200] + ("…" if len(content) > 200 else "")
                lines.append(f"  [{i}] {speaker}: {preview!r}")
        else:
            lines.append("Context dropped: (none)")

        resp = (event.receiver_response or "").strip()
        if resp:
            preview = resp[:300] + ("…" if len(resp) > 300 else "")
            lines.append(f"Receiver response: {preview!r}")
        else:
            lines.append("Receiver response: (empty)")

        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _classify_with_llm(
        self, transcript: str, trace: SeamTrace
    ) -> dict[str, Any]:
        """Submit the transcript to the LLM and parse the JSON response."""
        logger.debug(
            "HandoffEvaluator: calling %s for task=%s (%d events)",
            self._model,
            trace.task_id,
            len(trace.events),
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "HandoffEvaluator: failed to parse LLM JSON for task=%s: %s",
                trace.task_id,
                exc,
            )
            result = {}
        return result

    # ------------------------------------------------------------------
    # Apply LLM output back onto events
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_classifications(
        events: list[SeamEvent],
        llm_events: list[dict[str, Any]],
    ) -> None:
        """Write failure_mode and failure_rationale onto each SeamEvent."""
        index_map = {e["turn_index"]: e for e in llm_events if "turn_index" in e}
        for event in events:
            entry = index_map.get(event.turn_index)
            if not entry:
                logger.warning(
                    "HandoffEvaluator: no LLM classification for turn %d",
                    event.turn_index,
                )
                continue

            raw_mode = entry.get("failure_mode", "none")
            try:
                event.failure_mode = HandoffFailureMode(raw_mode)
            except ValueError:
                logger.warning(
                    "HandoffEvaluator: unknown failure mode %r at turn %d, defaulting to NONE",
                    raw_mode,
                    event.turn_index,
                )
                event.failure_mode = HandoffFailureMode.NONE

            rationale = entry.get("rationale", "")
            confidence = entry.get("confidence", "")
            if confidence:
                rationale = f"[{confidence}] {rationale}"
            event.failure_rationale = rationale or None

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

    @staticmethod
    def _empty_report(task_id: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "total_transitions": 0,
            "failure_count": 0,
            "failure_rate": 0.0,
            "failure_counts": {},
            "overall_summary": "",
            "events": [],
        }
