"""
Tool Registry: the single source of truth for every tool the agent can
call, whether it's one of 50 PayPal endpoints or one of 1000+ tools
spanning many external systems.

Design choice: tools are stored as ROWS IN POSTGRES (metadata + pgvector
embedding column), not as an in-memory list bound into the LLM's system
prompt. This is what lets the registry grow to thousands of entries
without ever touching the LLM's context window — only the Top-K survivors
of `search()` are turned into JSON-schema tool defs for the LLM call.

Adding a new integration (e.g. a 6th, 7th... Nth API collection) is a
data-plane operation: insert rows here. No prompt engineering, no code
change in the agent/planner.
"""
from __future__ import annotations
from typing import Optional
from sqlalchemy import create_engine, text
from app.config import settings
from app.schemas import ToolMetadata
from router.embeddings import embed_text, embed_batch

_engine = create_engine(settings.postgres_url)

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS tool_registry (
    tool_id                 TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    description             TEXT NOT NULL,
    category                TEXT NOT NULL,
    permissions_required    TEXT[] DEFAULT '{}',
    parameters_schema       JSONB NOT NULL,
    embedding               VECTOR(1536),
    historical_success_rate REAL DEFAULT 1.0,
    avg_latency_ms          INT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS tool_registry_embedding_idx
    ON tool_registry USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


def init_db() -> None:
    with _engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def register_tool(tool: ToolMetadata) -> None:
    """Idempotent upsert. Called at startup for static collections (e.g.
    the PayPal Postman collection) and at runtime for dynamically
    discovered / hot-loaded tools."""
    vec = embed_text(f"{tool.name}: {tool.description}")
    with _engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO tool_registry
                    (tool_id, name, description, category, permissions_required,
                     parameters_schema, embedding, historical_success_rate, avg_latency_ms)
                VALUES
                    (:tool_id, :name, :description, :category, :permissions_required,
                     :parameters_schema, :embedding, :hsr, :latency)
                ON CONFLICT (tool_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    category = EXCLUDED.category,
                    permissions_required = EXCLUDED.permissions_required,
                    parameters_schema = EXCLUDED.parameters_schema,
                    embedding = EXCLUDED.embedding
            """),
            {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "permissions_required": tool.permissions_required,
                "parameters_schema": tool.parameters_schema,
                "embedding": str(vec),
                "hsr": tool.historical_success_rate,
                "latency": tool.avg_latency_ms,
            },
        )


def bulk_register(tools: list[ToolMetadata]) -> None:
    """Bulk path for loading an entire API collection (e.g. 500 endpoints
    from an OpenAPI spec) in one shot using batched embeddings."""
    texts = [f"{t.name}: {t.description}" for t in tools]
    vectors = embed_batch(texts)
    with _engine.begin() as conn:
        for tool, vec in zip(tools, vectors):
            conn.execute(
                text("""
                    INSERT INTO tool_registry
                        (tool_id, name, description, category, permissions_required,
                         parameters_schema, embedding, historical_success_rate, avg_latency_ms)
                    VALUES
                        (:tool_id, :name, :description, :category, :permissions_required,
                         :parameters_schema, :embedding, :hsr, :latency)
                    ON CONFLICT (tool_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding, description = EXCLUDED.description
                """),
                {
                    "tool_id": tool.tool_id, "name": tool.name, "description": tool.description,
                    "category": tool.category, "permissions_required": tool.permissions_required,
                    "parameters_schema": tool.parameters_schema, "embedding": str(vec),
                    "hsr": tool.historical_success_rate, "latency": tool.avg_latency_ms,
                },
            )


def vector_search(query_embedding: list[float], top_k: int, category: Optional[str] = None) -> list[dict]:
    """ANN search via pgvector's ivfflat index. This is O(log n)-ish, not
    O(n) — the mechanism that keeps retrieval fast at 1000+ tools."""
    where_clause = "WHERE category = :category" if category else ""
    query = f"""
        SELECT tool_id, name, description, category, permissions_required,
               parameters_schema, historical_success_rate, avg_latency_ms,
               1 - (embedding <=> :qvec) AS similarity
        FROM tool_registry
        {where_clause}
        ORDER BY embedding <=> :qvec
        LIMIT :top_k
    """
    params = {"qvec": str(query_embedding), "top_k": top_k}
    if category:
        params["category"] = category
    with _engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]
