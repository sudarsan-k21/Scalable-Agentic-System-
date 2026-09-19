"""
PayPal tools — a thin, generic wrapper generated (conceptually) from the
Postman/OpenAPI collection. Each PayPal endpoint becomes ONE row in the
tool registry (see scripts under `Step 2/3` of the implementation plan)
and one entry in TOOL_HANDLERS below. This pattern is what lets us go
from 50 PayPal endpoints to 500+ endpoints across many services without
restructuring the agent: same registry, same executor, just more rows
and more handler entries (often auto-generated from the spec).
"""
from __future__ import annotations
import httpx
from app.config import settings

_BASE = settings.paypal_base_url


def _auth_headers() -> dict:
    # In production: cache the OAuth2 token in Redis and refresh on expiry.
    return {"Authorization": "Bearer <cached_oauth_token>", "Content-Type": "application/json"}


def create_invoice(amount: float, currency: str, recipient_email: str, note: str = "") -> dict:
    payload = {
        "detail": {"currency_code": currency, "note": note},
        "primary_recipients": [{"billing_info": {"email_address": recipient_email}}],
        "amount": {"value": str(amount), "currency_code": currency},
    }
    with httpx.Client(base_url=_BASE, headers=_auth_headers(), timeout=15) as client:
        r = client.post("/v2/invoicing/invoices", json=payload)
        r.raise_for_status()
        return r.json()


def send_payment(amount: float, currency: str, recipient_email: str) -> dict:
    payload = {
        "sender_batch_header": {"sender_batch_id": "batch_auto", "email_subject": "Payment"},
        "items": [{
            "recipient_type": "EMAIL",
            "amount": {"value": str(amount), "currency": currency},
            "receiver": recipient_email,
        }],
    }
    with httpx.Client(base_url=_BASE, headers=_auth_headers(), timeout=15) as client:
        r = client.post("/v1/payments/payouts", json=payload)
        r.raise_for_status()
        return r.json()


def get_dispute_status(customer_id: str) -> dict:
    with httpx.Client(base_url=_BASE, headers=_auth_headers(), timeout=15) as client:
        r = client.get("/v1/customer/disputes", params={"customer_id": customer_id})
        r.raise_for_status()
        return r.json()


def get_sales_report(period: str) -> dict:
    with httpx.Client(base_url=_BASE, headers=_auth_headers(), timeout=15) as client:
        r = client.get("/v1/reporting/transactions", params={"period": period})
        r.raise_for_status()
        return r.json()


# Maps tool_id (as stored in the registry) -> callable. Populated in bulk
# for all 50+ PayPal endpoints; only 4 shown here for brevity.
TOOL_HANDLERS = {
    "paypal.invoices.create": create_invoice,
    "paypal.payments.send": send_payment,
    "paypal.disputes.get_status": get_dispute_status,
    "paypal.reports.sales": get_sales_report,
}
