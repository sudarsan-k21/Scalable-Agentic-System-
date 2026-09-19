"""
Planner — takes the Top-K candidate tools (already filtered by the
router) plus the user's intent, and produces an ordered execution plan.

Handles both:
  - Single-tool turns ("send an invoice for $50 to user_123")
  - Multi-tool / multi-hop turns ("find the open dispute for user_123
    and then email them the resolution steps") -> RAG tool + PayPal
    dispute tool + notification tool, in sequence, with the output of
    step N feeding parameters into step N+1.

Kept as a *thin* LLM call: the model only reasons over the handful of
tools the router already selected, never the full registry. This is
what keeps planning latency and hallucination rate flat as the tool
catalog grows.
"""
from __future__ import annotations
import json
from openai import OpenAI
from app.config import settings
from app.schemas import RankedTool

_client = OpenAI(api_key=settings.openai_api_key)

PLANNER_SYSTEM_PROMPT = """You are the planning module of an agentic system.
You are given:
- The user's request
- A short list of candidate tools (already filtered for relevance and permissions)

Produce a JSON execution plan: a list of steps, each with
  tool_id, args (best-effort, may reference {{step_N.output.field}} for
  dependent steps), and a one-line reason.

Rules:
- Use ONLY tool_ids from the candidate list. Never invent a tool.
- If the request is ambiguous or required parameters are missing (e.g. no
  amount, no recipient), do NOT guess — instead return
  {"needs_clarification": true, "question": "..."}.
- Prefer the smallest number of steps that satisfies the request.
- Output strict JSON, no prose.
"""


def build_plan(user_message: str, candidate_tools: list[RankedTool]) -> dict:
    tool_summaries = [
        {"tool_id": r.tool.tool_id, "name": r.tool.name, "description": r.tool.description}
        for r in candidate_tools
    ]

    if not tool_summaries:
        return {"needs_clarification": True, "question": None,
                "plan": [], "reason": "no_relevant_tools"}

    completion = _client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "user_request": user_message,
                "candidate_tools": tool_summaries,
            })},
        ],
    )
    return json.loads(completion.choices[0].message.content)
