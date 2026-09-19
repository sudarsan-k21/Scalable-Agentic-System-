"""
System Search Tool — lets the agent (and, transitively, the user) query
the AGENTIC SYSTEM ITSELF: which tools/capabilities exist, and what
happened on a past request. This is distinct from the RAG tool (product
docs) and from the Tool Router (which selects tools to call, but doesn't
answer meta-questions about the system).

Two sub-capabilities, both exposed under one tool_id so the router still
only has to pick ONE tool for "what can you do" / "what happened" style
questions:
  - capability_search: semantic search over the SAME tool_registry table
    the router uses, but returns human-readable summaries instead of
    JSON schemas for the LLM to call.
  - log_search: looks up past execution results for this session/user
    from the durable execution log (Postgres), e.g. "what's the status
    of my last request?".
"""
from __future__ import annotations
from sqlalchemy import create_engine, text
from app.config import settings
from router.embeddings import embed_text
from router.tool_registry import vector_search

_engine = create_engine(settings.postgres_url)


def capability_search(query: str, top_k: int = 10) -> dict:
    vec = embed_text(query)
    hits = vector_search(vec, top_k=top_k)
    return {
        "query": query,
        "capabilities": [
            {"name": h["name"], "description": h["description"], "category": h["category"]}
            for h in hits
        ],
    }


def log_search(user_id: str, session_id: str | None = None, limit: int = 10) -> dict:
    query = """
        SELECT tool_id, success, error, created_at
        FROM execution_log
        WHERE user_id = :user_id
          AND (:session_id IS NULL OR session_id = :session_id)
        ORDER BY created_at DESC
        LIMIT :limit
    """
    with _engine.connect() as conn:
        rows = conn.execute(
            text(query), {"user_id": user_id, "session_id": session_id, "limit": limit}
        ).mappings().all()
    return {"recent_activity": [dict(r) for r in rows]}


TOOL_HANDLERS = {
    "system.capability_search": capability_search,
    "system.log_search": log_search,
}

TOOL_METADATA = [
    {
        "tool_id": "system.capability_search",
        "name": "Search Available Tools",
        "description": (
            "Search this system's own catalog of tools/capabilities, e.g. "
            "'what tools are available for managing invoices?'."
        ),
        "category": "system",
        "permissions_required": [],
        "parameters_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}},
            "required": ["query"],
        },
    },
    {
        "tool_id": "system.log_search",
        "name": "Check Request Status / Logs",
        "description": (
            "Look up the status or result of a past request for this user or session, "
            "e.g. 'what's the status of my last request?'."
        ),
        "category": "system",
        "permissions_required": [],
        "parameters_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "session_id": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["user_id"],
        },
    },
]
