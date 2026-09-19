"""Tests for services/validation.py and services/retry.py classification logic."""
import pytest
from services.validation import validate_args, ValidationError
from services.retry import is_idempotent


def test_validate_args_passes_for_valid_input():
    schema = {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}
    validate_args({"amount": 50}, schema)  # should not raise


def test_validate_args_raises_for_missing_required_field():
    schema = {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}
    with pytest.raises(ValidationError):
        validate_args({}, schema)


@pytest.mark.parametrize("tool_id,expected", [
    ("paypal.disputes.get_status", True),
    ("paypal.reports.sales", True),
    ("rag.knowledge_base.search", True),
    ("system.capability_search", True),
    ("paypal.invoices.create", False),
    ("paypal.payments.send", False),
])
def test_is_idempotent_classification(tool_id, expected):
    assert is_idempotent(tool_id) == expected
