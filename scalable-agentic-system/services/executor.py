"""
Execution Layer (Step 4 of the blueprint): calls the selected tool(s),
handles retries/errors, and persists an audit trail.

This module is deliberately dumb about *which* tool to call — that
decision was already made by the router + planner. Its job is purely:
validate -> call -> retry-on-transient-failure -> log -> return.
"""
from __future__ import annotations
import json
import time
from app.schemas import ExecutionResult
from services.validation import validate_args, ValidationError
from services.retry import IDEMPOTENT_RETRY, NON_IDEMPOTENT_RETRY, is_idempotent
from database.connection import SessionLocal
from database.models import ExecutionLog

from tools import paypal_tools, rag_tool, system_search

TOOL_HANDLERS: dict = {
    **paypal_tools.TOOL_HANDLERS,
    **rag_tool.TOOL_HANDLERS,
    **system_search.TOOL_HANDLERS,
}


def _call_with_retry(tool_id: str, handler, args: dict):
    wrapper = IDEMPOTENT_RETRY if is_idempotent(tool_id) else NON_IDEMPOTENT_RETRY

    @wrapper
    def _call():
        return handler(**args)

    return _call()


def execute_step(
    tool_id: str,
    args: dict,
    schema: dict,
    session_id: str,
    user_id: str,
) -> ExecutionResult:
    start = time.time()
    handler = TOOL_HANDLERS.get(tool_id)

    if handler is None:
        result = ExecutionResult(tool_id=tool_id, success=False, error="Unknown tool_id")
        _log(session_id, user_id, tool_id, args, result)
        return result

    try:
        validate_args(args, schema)
    except ValidationError as e:
        result = ExecutionResult(tool_id=tool_id, success=False, error=str(e))
        _log(session_id, user_id, tool_id, args, result)
        return result

    retries = 0
    try:
        output = _call_with_retry(tool_id, handler, args)
        result = ExecutionResult(
            tool_id=tool_id, success=True, output=output,
            retries=retries, latency_ms=int((time.time() - start) * 1000),
        )
    except Exception as e:  # noqa: BLE001 - surfaced to the agent as a controlled failure
        result = ExecutionResult(
            tool_id=tool_id, success=False, error=str(e),
            latency_ms=int((time.time() - start) * 1000),
        )

    _log(session_id, user_id, tool_id, args, result)
    return result


def execute_plan(plan: list[dict], tool_schemas: dict[str, dict], session_id: str, user_id: str) -> list[ExecutionResult]:
    """Executes a plan sequentially, resolving {{step_N.output.field}}
    references between dependent steps before each call."""
    results: list[ExecutionResult] = []
    outputs_by_index: dict[int, dict] = {}

    for i, step in enumerate(plan):
        args = _resolve_references(step.get("args", {}), outputs_by_index)
        result = execute_step(
            step["tool_id"], args, tool_schemas.get(step["tool_id"], {}),
            session_id, user_id,
        )
        results.append(result)
        outputs_by_index[i] = result.output if result.success else {}

        if not result.success and step.get("critical", True):
            break  # stop the chain on a critical failure; agent surfaces the error

    return results


def _resolve_references(args: dict, outputs_by_index: dict[int, dict]) -> dict:
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("{{step_") and v.endswith("}}"):
            # e.g. "{{step_0.output.dispute_id}}"
            path = v.strip("{}").split(".")
            step_idx = int(path[0].replace("step_", ""))
            field = path[-1]
            resolved[k] = outputs_by_index.get(step_idx, {}).get(field)
        else:
            resolved[k] = v
    return resolved


def _log(session_id: str, user_id: str, tool_id: str, args: dict, result: ExecutionResult) -> None:
    db = SessionLocal()
    try:
        db.add(ExecutionLog(
            session_id=session_id, user_id=user_id, tool_id=tool_id,
            args=json.dumps(args), success=result.success,
            output=json.dumps(result.output) if result.output is not None else None,
            error=result.error, retries=result.retries, latency_ms=result.latency_ms,
        ))
        db.commit()
    finally:
        db.close()
