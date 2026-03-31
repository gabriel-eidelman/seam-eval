"""
AutoGen (AG2) AgentAdapter for MASEval.

Bridges AutoGen's multi-agent conversation API to MASEval's AgentAdapter
interface. Supports both two-agent (UserProxy + Assistant) and GroupChat
configurations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from maseval import AgentAdapter, MessageHistory

try:
    import autogen
    from autogen import ConversableAgent
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyautogen is required for AutoGenAdapter. "
        "Install it with: pip install pyautogen"
    ) from exc

logger = logging.getLogger(__name__)


class AutoGenAdapter(AgentAdapter):
    """
    AgentAdapter wrapping an AutoGen ConversableAgent or GroupChat.

    Usage
    -----
    For a two-agent setup (the most common case):

        initiator = autogen.UserProxyAgent("user_proxy", ...)
        responder = autogen.AssistantAgent("assistant", ...)
        adapter = AutoGenAdapter(
            agent=initiator,
            responder=responder,
            max_turns=10,
        )
        result = adapter.run(query)

    For a GroupChat:

        agents = [agent_a, agent_b, agent_c]
        groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=12)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=...)
        initiator = autogen.UserProxyAgent("user_proxy", ...)
        adapter = AutoGenAdapter(
            agent=initiator,
            responder=manager,
            max_turns=1,  # GroupChat manages its own turn count
        )
    """

    def __init__(
        self,
        agent: ConversableAgent,
        responder: ConversableAgent,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        agent:
            The initiating agent (typically a UserProxyAgent). MASEval calls
            _run_agent on this adapter; we initiate the conversation from here.
        responder:
            The agent or GroupChatManager that receives the initial message.
        max_turns:
            Maximum number of back-and-forth turns before the conversation
            is forced to terminate.
        """
        super().__init__(agent, agent.name, **kwargs)
        self._responder = responder
        self._max_turns = max_turns
        self._last_chat_result: Optional[Any] = None

    # ------------------------------------------------------------------
    # AgentAdapter interface
    # ------------------------------------------------------------------

    def _run_agent(self, query: str) -> str:
        """
        Initiate the AutoGen conversation and return the final answer.

        AutoGen conversations terminate when the initiating agent sends
        TERMINATE or the turn limit is reached. The last non-empty message
        from the responder is used as the final answer.
        """
        logger.debug(
            "AutoGenAdapter._run_agent: initiating conversation with query=%r",
            query[:120],
        )
        self._last_chat_result = self.agent.initiate_chat(
            self._responder,
            message=query,
            max_turns=self._max_turns,
            silent=True,
        )
        return self._extract_final_answer()

    def get_messages(self) -> MessageHistory:
        """
        Return the full conversation history in MASEval's expected format.

        AutoGen stores messages per-agent in agent.chat_messages. We merge
        both sides of the conversation and sort by position to reconstruct
        the interleaved history.
        """
        raw: dict[ConversableAgent, list[dict[str, Any]]] = (
            self.agent.chat_messages
        )
        if not raw:
            return MessageHistory()

        # Grab messages from the primary conversation partner.
        messages = raw.get(self._responder, [])
        if not messages:
            # Fallback: flatten all recorded message lists.
            messages = [msg for msgs in raw.values() for msg in msgs]

        return MessageHistory([self._normalise_message(m) for m in messages])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_final_answer(self) -> str:
        """Return the last substantive message from the responder."""
        messages = self.get_messages()
        # Walk backwards to find the last non-empty assistant message.
        for msg in reversed(messages):
            if msg.get("role") in ("assistant", "tool") and msg.get("content"):
                content = msg["content"]
                if isinstance(content, str) and content.strip():
                    return content.strip()
        # Fallback: return the last message content regardless of role.
        if messages:
            return str(messages[-1].get("content", ""))
        return ""

    @staticmethod
    def _normalise_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Normalise an AutoGen message dict to MASEval's expected schema."""
        return {
            "role": msg.get("role", "unknown"),
            "content": msg.get("content") or "",
            "name": msg.get("name"),
            "metadata": {
                k: v
                for k, v in msg.items()
                if k not in {"role", "content", "name"}
            },
        }
