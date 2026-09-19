"""
Security: authn/authz + risk classification.

Kept as a distinct module (rather than folded into the executor) so it
can be unit-tested and audited independently, and so "risky" tool calls
(payments, refunds, deletes) can be routed through an extra confirmation
step regardless of which agent framework sits on top.
"""
from __future__ import annotations
import jwt
from app.config import settings

HIGH_RISK_CATEGORIES = {"paypal.payments", "paypal.refunds", "paypal.disputes.resolve"}


def decode_user_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


def get_user_permissions(user_id: str) -> set[str]:
    """Stub: in production, pulled from an IAM/RBAC service or DB table
    and cached in Redis with a short TTL."""
    return {"paypal.invoices.write", "paypal.payments.write",
            "paypal.disputes.read", "paypal.reports.read", "rag.read", "system.read"}


def is_high_risk(tool_id: str) -> bool:
    return any(tool_id.startswith(cat) for cat in HIGH_RISK_CATEGORIES)


def requires_confirmation(tool_id: str, args: dict) -> bool:
    """Extra guardrail: large monetary amounts always require an explicit
    user confirmation step in the agent graph, even if permissions pass."""
    if is_high_risk(tool_id):
        amount = args.get("amount")
        if amount is not None and float(amount) > 500:
            return True
    return False
