"""Validates planned tool-call arguments against each tool's JSON schema
before execution — catches hallucinated/missing parameters early, rather
than letting a malformed call hit a real external API."""
from __future__ import annotations
import jsonschema


class ValidationError(Exception):
    pass


def validate_args(args: dict, schema: dict) -> None:
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Invalid arguments: {e.message}") from e


def validate_plan(plan: list[dict], tool_schemas: dict[str, dict]) -> list[str]:
    """Returns a list of human-readable errors (empty if the plan is valid)."""
    errors = []
    for step in plan:
        tool_id = step.get("tool_id")
        schema = tool_schemas.get(tool_id)
        if schema is None:
            errors.append(f"Unknown tool_id in plan: {tool_id}")
            continue
        try:
            validate_args(step.get("args", {}), schema)
        except ValidationError as e:
            errors.append(f"{tool_id}: {e}")
    return errors
