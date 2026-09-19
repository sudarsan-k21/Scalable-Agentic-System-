"""Tests for agent graph branching logic (clarification / confirmation gates)."""
from agent.agent import route_after_planner


def test_route_to_responder_when_clarification_needed():
    state = {"needs_clarification": True, "plan": []}
    assert route_after_planner(state) == "responder"


def test_route_to_executor_for_low_risk_plan():
    state = {
        "needs_clarification": False,
        "plan": [{"tool_id": "paypal.reports.sales", "args": {"period": "LAST_MONTH"}}],
    }
    assert route_after_planner(state) == "executor"


def test_route_to_responder_for_high_risk_plan():
    state = {
        "needs_clarification": False,
        "plan": [{"tool_id": "paypal.payments.send", "args": {"amount": 5000, "currency": "USD"}}],
    }
    result = route_after_planner(state)
    assert result == "responder"
    assert "confirm" in state["final_response"].lower()
