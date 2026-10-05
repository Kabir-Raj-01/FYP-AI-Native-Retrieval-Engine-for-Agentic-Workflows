"""Python client for the retrieval engine's HTTP API (API_CONTRACT "Python client").

Host apps import only ``client`` and ``client.models``.
"""

from client.retrieval_client import (
    BackendUnavailable,
    InternalError,
    LLMUnavailable,
    RetrievalClient,
    RetrievalClientError,
    RunBudgetExceeded,
    RunTimeout,
    TaskConflict,
    TaskExists,
    TaskNotActive,
    TaskNotFound,
    ValidationFailed,
)

__all__ = [
    "BackendUnavailable",
    "InternalError",
    "LLMUnavailable",
    "RetrievalClient",
    "RetrievalClientError",
    "RunBudgetExceeded",
    "RunTimeout",
    "TaskConflict",
    "TaskExists",
    "TaskNotActive",
    "TaskNotFound",
    "ValidationFailed",
]
