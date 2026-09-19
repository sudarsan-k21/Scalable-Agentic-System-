"""
FastAPI entrypoint — the "chat" surface a user or frontend hits.
Wraps the LangGraph agent, loads/saves conversation state around it,
and returns the LangSmith trace URL alongside the reply for debugging.
"""
from __future__ import annotations
import os
from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.config import settings
from agent.agent import agent_graph
from database.connection import cache_get, cache_set

os.environ.setdefault("LANGCHAIN_TRACING_V2", str(settings.langchain_tracing_v2).lower())
os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

app = FastAPI(title="Scalable Agentic System")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    cached = cache_get(f"session:{req.session_id}") or {}
    initial_state = {
        "session_id": req.session_id,
        "user_id": req.user_id,
        "raw_message": req.message,
        "conversation_summary": cached.get("conversation_summary", ""),
    }

    final_state = agent_graph.invoke(initial_state)

    updated_summary = (
        f"{initial_state['conversation_summary']}\n"
        f"User: {req.message}\nAgent: {final_state.get('final_response')}"
    ).strip()[-4000:]  # keep the rolling summary bounded
    cache_set(f"session:{req.session_id}", {"conversation_summary": updated_summary})

    return ChatResponse(
        session_id=req.session_id,
        reply=final_state.get("final_response", ""),
        tool_calls=[r.dict() for r in final_state.get("execution_results", [])],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
