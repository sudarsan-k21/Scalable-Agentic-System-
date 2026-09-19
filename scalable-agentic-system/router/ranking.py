"""
Re-ranks the raw ANN hits from pgvector using more than just cosine
similarity, so tool selection improves over time and isn't purely a
static embedding match.

final_score = w1 * similarity
            + w2 * historical_success_rate
            + w3 * permission_match
            - w4 * normalized_latency_penalty
"""
from __future__ import annotations
from app.schemas import ToolMetadata, RankedTool

WEIGHTS = {"similarity": 0.65, "success_rate": 0.20, "permission": 0.10, "latency": 0.05}


def permission_ok(required: list[str], user_permissions: set[str]) -> bool:
    return set(required).issubset(user_permissions)


def rerank(raw_hits: list[dict], user_permissions: set[str]) -> list[RankedTool]:
    ranked: list[RankedTool] = []
    max_latency = max((h["avg_latency_ms"] for h in raw_hits), default=1) or 1

    for hit in raw_hits:
        tool = ToolMetadata(**hit)
        allowed = permission_ok(tool.permissions_required, user_permissions)
        if not allowed:
            # Filtered out entirely rather than down-weighted: a tool the
            # user isn't authorized for should never reach the LLM.
            continue

        latency_penalty = hit["avg_latency_ms"] / max_latency
        score = (
            WEIGHTS["similarity"] * hit["similarity"]
            + WEIGHTS["success_rate"] * tool.historical_success_rate
            + WEIGHTS["permission"] * 1.0
            - WEIGHTS["latency"] * latency_penalty
        )
        ranked.append(RankedTool(tool=tool, similarity=hit["similarity"], final_score=score))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
