"""
Unit tests for router/ranking.py and router/tool_router.py's filtering
logic. Embedding/DB calls are mocked so these run without live Postgres
or OpenAI credentials (CI-friendly).
"""
from unittest.mock import patch
from router.ranking import rerank, permission_ok


def _hit(tool_id, similarity, latency=100, hsr=0.9, perms=None):
    return {
        "tool_id": tool_id, "name": tool_id, "description": "desc", "category": "paypal",
        "permissions_required": perms or [], "parameters_schema": {},
        "historical_success_rate": hsr, "avg_latency_ms": latency, "similarity": similarity,
    }


def test_permission_ok():
    assert permission_ok(["paypal.read"], {"paypal.read", "paypal.write"})
    assert not permission_ok(["paypal.admin"], {"paypal.read"})


def test_rerank_filters_unauthorized_tools():
    hits = [
        _hit("paypal.invoices.create", 0.9, perms=["paypal.invoices.write"]),
        _hit("paypal.admin.delete_account", 0.95, perms=["paypal.admin"]),
    ]
    ranked = rerank(hits, user_permissions={"paypal.invoices.write"})
    ids = [r.tool.tool_id for r in ranked]
    assert "paypal.invoices.create" in ids
    assert "paypal.admin.delete_account" not in ids


def test_rerank_orders_by_final_score():
    hits = [
        _hit("low_success", 0.9, hsr=0.2),
        _hit("high_success", 0.85, hsr=0.99),
    ]
    ranked = rerank(hits, user_permissions=set())
    assert ranked[0].tool.tool_id == "high_success"


@patch("router.tool_router.vector_search")
@patch("router.tool_router.embed_text", return_value=[0.1] * 1536)
def test_select_tools_respects_similarity_floor(mock_embed, mock_search):
    mock_search.return_value = [
        _hit("paypal.invoices.create", 0.9, perms=[]),
        _hit("irrelevant.tool", 0.10, perms=[]),
    ]
    from router.tool_router import select_tools
    results = select_tools("send an invoice for $50", user_permissions=set())
    ids = [r.tool.tool_id for r in results]
    assert "paypal.invoices.create" in ids
    assert "irrelevant.tool" not in ids  # below default similarity floor of 0.55
