"""engine/errors.py: hierarchy and the API_CONTRACT.md error mapping (as data)."""

from __future__ import annotations

from http import HTTPStatus

import pytest

from engine import errors
from engine.errors import (
    BackendError,
    ConfigError,
    EngineError,
    LLMBudgetExceeded,
    LLMError,
    LLMOutputError,
    PlannerValidationError,
    RunTimeout,
    StateInvariantError,
    TaskConflict,
    TaskExists,
    TaskNotActive,
    TaskNotFound,
)

# Written from API_CONTRACT.md "Errors" and SDS §6.2. Exceptions with no row there map to
# 500 INTERNAL_ERROR.
EXPECTED_MAPPING: dict[type[EngineError], tuple[int, str]] = {
    TaskNotFound: (404, "TASK_NOT_FOUND"),
    TaskExists: (409, "TASK_EXISTS"),
    TaskNotActive: (409, "TASK_NOT_ACTIVE"),
    TaskConflict: (409, "TASK_CONFLICT"),
    LLMError: (502, "LLM_ERROR"),
    BackendError: (503, "BACKEND_ERROR"),
    RunTimeout: (504, "RUN_TIMEOUT"),
    LLMBudgetExceeded: (429, "RUN_BUDGET_EXCEEDED"),
    LLMOutputError: (500, "INTERNAL_ERROR"),
    PlannerValidationError: (500, "INTERNAL_ERROR"),
    ConfigError: (500, "INTERNAL_ERROR"),
    StateInvariantError: (500, "INTERNAL_ERROR"),
}


def test_every_exported_error_has_an_expected_mapping() -> None:
    exported = {getattr(errors, name) for name in errors.__all__} - {EngineError}
    assert exported == set(EXPECTED_MAPPING)


@pytest.mark.parametrize(("exc_type", "expected"), list(EXPECTED_MAPPING.items()))
def test_error_maps_to_api_contract_code(
    exc_type: type[EngineError], expected: tuple[int, str]
) -> None:
    status, code = expected
    assert exc_type.http_status == HTTPStatus(status)
    assert exc_type.code == code


@pytest.mark.parametrize("exc_type", list(EXPECTED_MAPPING))
def test_every_error_derives_directly_from_engine_error(exc_type: type[EngineError]) -> None:
    assert exc_type.__bases__ == (EngineError,)


def test_llm_output_and_budget_errors_are_not_caught_as_llm_error() -> None:
    # A planner `except LLMError` must not swallow SCHEMA failures or the per-run cap.
    assert not issubclass(LLMOutputError, LLMError)
    assert not issubclass(LLMBudgetExceeded, LLMError)


def test_error_carries_message_and_optional_task_id() -> None:
    exc = TaskNotFound("no such task", task_id="t-1")
    assert str(exc) == "no such task"
    assert exc.message == "no such task"
    assert exc.task_id == "t-1"
    assert BackendError("down").task_id is None


def test_base_error_defaults_to_internal_error() -> None:
    assert EngineError.http_status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert EngineError.code == "INTERNAL_ERROR"
