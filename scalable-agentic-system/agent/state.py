"""
Agent state — the object that flows through every LangGraph node.

Two tiers of memory, matching the blueprint's "Memory & State Manager":
  - Short-term / working state (this TypedDict): lives for the duration
    of a single graph run, held in-memory by LangGraph.
  - Long-term / cross-turn state: conversation history & session context
    persisted in Redis (fast) and Postgres (durable), keyed by session_id.
    See services/executor.py + database/models.py.
"""
from __future__ import annotations
from typing import Any, Optional, TypedDict
from app.schemas import RankedTool, ExecutionResult


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    user_permissions: list[str]

    raw_message: str
    rewritten_query: str
    conversation_summary: str

    candidate_tools: list[RankedTool]
    plan: list[dict[str, Any]]        # ordered list of {tool_id, args, depends_on}
    execution_results: list[ExecutionResult]

    needs_clarification: bool
    clarification_question: Optional[str]

    final_response: Optional[str]
    error: Optional[str]
