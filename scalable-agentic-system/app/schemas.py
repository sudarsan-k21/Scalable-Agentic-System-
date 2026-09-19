"""Shared request/response and internal data contracts."""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict] = Field(default_factory=list)
    trace_url: Optional[str] = None


class ToolMetadata(BaseModel):
    """
    One row per tool in the Tool Registry. This is the unit of retrieval
    for semantic search — NOT the tool's full JSON schema (that is loaded
    lazily, only for the tools that survive Top-K filtering).
    """
    tool_id: str
    name: str
    description: str          # rich NL description, used for embedding
    category: str             # e.g. "paypal.invoices", "paypal.disputes"
    permissions_required: list[str] = Field(default_factory=list)
    parameters_schema: dict[str, Any]  # JSON schema, loaded lazily
    embedding: Optional[list[float]] = None
    historical_success_rate: float = 1.0
    avg_latency_ms: int = 0


class RankedTool(BaseModel):
    tool: ToolMetadata
    similarity: float
    final_score: float


class ExecutionResult(BaseModel):
    tool_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    retries: int = 0
    latency_ms: int = 0
