"""
Agent orchestration using LangGraph.

Graph shape (mirrors the blueprint's 4-stage Agentic System box):

    input_processor -> tool_router -> planner -> [needs_clarification?]
        -> yes -> ask_user (END)
        -> no  -> [high_risk_step?]
              -> yes -> confirm_with_user (END, resumes on next turn)
              -> no  -> executor -> responder (END)

Why LangGraph over a plain LangChain AgentExecutor or a hand-rolled loop:
  - Explicit state machine: each node's inputs/outputs are typed
    (agent/state.py), so behavior is inspectable and testable in
    isolation — critical once you have branches like "needs
    clarification" or "requires confirmation" that a simple ReAct loop
    tends to bury inside free-form reasoning text.
  - Native support for interrupts/human-in-the-loop (confirm_with_user),
    which a bare LangChain AgentExecutor doesn't give you cleanly.
  - Cycles are first-class, which matters if a future version adds a
    self-correction loop (retry planning after a validation failure)
    without restructuring the whole thing.
  - LangSmith tracing is automatic for every node, satisfying the
    Observability & Monitoring requirement with near-zero extra code.

Trade-off acknowledged: LangGraph adds a learning curve and another
dependency versus just calling the OpenAI SDK directly. We accept that
cost because multi-step, conditional, stateful flows are exactly the
regime this system lives in once you're past a handful of tools.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.planner import build_plan
from router.tool_router import select_tools, to_llm_tool_defs
from services.executor import execute_plan
from services.security import get_user_permissions, requires_confirmation


def node_input_processor(state: AgentState) -> AgentState:
    # Intent detection / entity extraction is implicit here: we lean on
    # the router's embedding search + the planner's structured JSON
    # output rather than a separate NLU pass, keeping latency down.
    state["rewritten_query"] = state["raw_message"]
    return state


def node_tool_router(state: AgentState) -> AgentState:
    permissions = set(state.get("user_permissions") or get_user_permissions(state["user_id"]))
    candidates = select_tools(
        user_message=state["rewritten_query"],
        user_permissions=permissions,
        conversation_summary=state.get("conversation_summary", ""),
    )
    state["candidate_tools"] = candidates
    return state


def node_planner(state: AgentState) -> AgentState:
    plan_result = build_plan(state["rewritten_query"], state["candidate_tools"])
    if plan_result.get("needs_clarification"):
        state["needs_clarification"] = True
        state["clarification_question"] = plan_result.get("question") or (
            "Could you give me a bit more detail so I can find the right tool for that?"
        )
        state["plan"] = []
    else:
        state["needs_clarification"] = False
        state["plan"] = plan_result.get("plan", [])
    return state


def node_executor(state: AgentState) -> AgentState:
    tool_schemas = {r.tool.tool_id: r.tool.parameters_schema for r in state["candidate_tools"]}
    results = execute_plan(
        state["plan"], tool_schemas, state["session_id"], state["user_id"]
    )
    state["execution_results"] = results
    return state


def node_responder(state: AgentState) -> AgentState:
    if state.get("needs_clarification"):
        state["final_response"] = state["clarification_question"]
        return state

    results = state.get("execution_results", [])
    if not results:
        state["final_response"] = "I couldn't find a tool that matches that request."
        return state

    summaries = []
    for r in results:
        if r.success:
            summaries.append(f"{r.tool_id} succeeded: {r.output}")
        else:
            summaries.append(f"{r.tool_id} failed: {r.error}")
    state["final_response"] = " | ".join(summaries)
    return state


def route_after_planner(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "responder"
    # High-risk / confirmation gate: if any planned step is high-risk,
    # short-circuit to responder with a confirmation prompt instead of
    # executing immediately. (Full implementation would set a "pending
    # confirmation" flag in Redis and resume on the user's next "yes".)
    for step in state.get("plan", []):
        if requires_confirmation(step["tool_id"], step.get("args", {})):
            state["final_response"] = (
                f"This will call {step['tool_id']} with {step.get('args')} — "
                "please confirm to proceed."
            )
            return "responder"
    return "executor"


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input_processor", node_input_processor)
    graph.add_node("tool_router", node_tool_router)
    graph.add_node("planner", node_planner)
    graph.add_node("executor", node_executor)
    graph.add_node("responder", node_responder)

    graph.set_entry_point("input_processor")
    graph.add_edge("input_processor", "tool_router")
    graph.add_edge("tool_router", "planner")
    graph.add_conditional_edges("planner", route_after_planner, {
        "responder": "responder",
        "executor": "executor",
    })
    graph.add_edge("executor", "responder")
    graph.add_edge("responder", END)

    return graph.compile()


agent_graph = build_agent_graph()
