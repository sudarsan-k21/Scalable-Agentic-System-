"""
Tool Router & Selector — the heart of the "1000+ tools" story.

Instead of binding every registered tool into the LLM call (which is what
tanks accuracy and blows the context window), we:

  1. Rewrite/enrich the user's query with conversation context.
  2. Embed the query.
  3. ANN-search the tool registry (pgvector) for the top ~30 candidates.
  4. Re-rank those candidates with success-rate / permission / latency
     signals (router/ranking.py).
  5. Truncate to TOP_K (default 8) above a similarity floor.
  6. Convert only those TOP_K rows into JSON-schema tool defs and hand
     them to the LLM.

The LLM therefore only ever "sees" a handful of tools per turn, no matter
how large the registry grows — accuracy stays roughly constant as the
system scales from 50 to 5000 tools.
"""
from __future__ import annotations
from app.config import settings
from app.schemas import RankedTool
from router.embeddings import embed_text
from router.tool_registry import vector_search
from router.ranking import rerank

CANDIDATE_POOL_SIZE = 30  # cast a wide net before re-ranking down to Top-K


def rewrite_query(user_message: str, conversation_summary: str = "") -> str:
    """Cheap query rewriting: fold in recent context so a follow-up like
    'now refund it' resolves against 'the invoice I just created'.
    In production this is a small, fast LLM call or a rule-based expander,
    not the same heavy model used for planning."""
    if conversation_summary:
        return f"{conversation_summary}\nCurrent request: {user_message}"
    return user_message


def select_tools(
    user_message: str,
    user_permissions: set[str],
    conversation_summary: str = "",
    category_hint: str | None = None,
    top_k: int | None = None,
) -> list[RankedTool]:
    top_k = top_k or settings.top_k_tools

    query = rewrite_query(user_message, conversation_summary)
    query_vec = embed_text(query)

    raw_hits = vector_search(query_vec, top_k=CANDIDATE_POOL_SIZE, category=category_hint)
    ranked = rerank(raw_hits, user_permissions)

    filtered = [r for r in ranked if r.similarity >= settings.router_similarity_floor]
    return filtered[:top_k]


def to_llm_tool_defs(ranked_tools: list[RankedTool]) -> list[dict]:
    """Convert the Top-K RankedTool objects into OpenAI/Anthropic-style
    function-calling tool definitions. Parameter schemas are loaded here,
    lazily — the rest of the registry never pays this cost."""
    defs = []
    for r in ranked_tools:
        defs.append({
            "type": "function",
            "function": {
                "name": r.tool.tool_id,
                "description": r.tool.description,
                "parameters": r.tool.parameters_schema,
            },
        })
    return defs
