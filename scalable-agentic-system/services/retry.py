"""
Retry policy, centralized so every tool handler gets consistent behavior
without each PayPal (or other) wrapper re-implementing backoff logic.

Idempotent calls (GET-style: reports, dispute status, RAG search) retry
aggressively. Non-idempotent calls (payments, invoice creation) retry
conservatively and only on clearly transient errors (timeouts, 5xx),
never on 4xx, to avoid double-charging.
"""
from __future__ import annotations
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
import httpx

IDEMPOTENT_RETRY = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)

NON_IDEMPOTENT_RETRY = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, max=4),
    retry=retry_if_exception_type(httpx.TimeoutException),
    reraise=True,
)

IDEMPOTENT_PREFIXES = ("paypal.disputes.get", "paypal.reports", "rag.", "system.")


def is_idempotent(tool_id: str) -> bool:
    return tool_id.startswith(IDEMPOTENT_PREFIXES)
