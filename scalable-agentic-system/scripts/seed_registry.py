"""
Bulk-loads the tool registry: PayPal endpoints (from a Postman/OpenAPI
collection, conceptually), plus the RAG and System Search tools.

Run once at deploy time, and again whenever a new API collection /
integration is added. This is the "add 500 more tools" growth path:
parse the new collection into ToolMetadata rows, call bulk_register().
No changes needed anywhere else in the system.

Usage:
    python -m scripts.seed_registry
"""
from app.schemas import ToolMetadata
from router.tool_registry import init_db, bulk_register
from tools import rag_tool, system_search

# In production this list is generated programmatically by walking the
# Postman collection / OpenAPI spec (endpoint -> name, description,
# required scopes, JSON schema of params). A handful shown here.
PAYPAL_TOOLS = [
    ToolMetadata(
        tool_id="paypal.invoices.create",
        name="Create Invoice",
        description="Create and send a PayPal invoice for a given amount to a recipient email.",
        category="paypal.invoices",
        permissions_required=["paypal.invoices.write"],
        parameters_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string", "default": "USD"},
                "recipient_email": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["amount", "currency", "recipient_email"],
        },
    ),
    ToolMetadata(
        tool_id="paypal.payments.send",
        name="Send Payment",
        description="Send a payout/payment of a given amount to a recipient's PayPal email.",
        category="paypal.payments",
        permissions_required=["paypal.payments.write"],
        parameters_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string", "default": "USD"},
                "recipient_email": {"type": "string"},
            },
            "required": ["amount", "currency", "recipient_email"],
        },
    ),
    ToolMetadata(
        tool_id="paypal.disputes.get_status",
        name="Get Dispute Status",
        description="Check whether there is an open dispute for a given customer and its status.",
        category="paypal.disputes",
        permissions_required=["paypal.disputes.read"],
        parameters_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    ),
    ToolMetadata(
        tool_id="paypal.reports.sales",
        name="Get Sales Report",
        description="Retrieve total sales volume / transaction report for a given time period.",
        category="paypal.reports",
        permissions_required=["paypal.reports.read"],
        parameters_schema={
            "type": "object",
            "properties": {"period": {"type": "string", "enum": ["LAST_WEEK", "LAST_MONTH", "LAST_QUARTER"]}},
            "required": ["period"],
        },
    ),
    # ... remaining 46+ PayPal endpoints follow the same shape.
]

SYSTEM_TOOLS = [ToolMetadata(**m) for m in system_search.TOOL_METADATA]
RAG_TOOLS = [ToolMetadata(**rag_tool.TOOL_METADATA)]


def main():
    init_db()
    bulk_register(PAYPAL_TOOLS + SYSTEM_TOOLS + RAG_TOOLS)
    print(f"Registered {len(PAYPAL_TOOLS) + len(SYSTEM_TOOLS) + len(RAG_TOOLS)} tools.")


if __name__ == "__main__":
    main()
