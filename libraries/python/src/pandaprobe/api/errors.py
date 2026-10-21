"""Errors distinguish operation failure from an unknown creation outcome."""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for platform API errors."""


class ApiError(PlatformError):
    def __init__(self, message: str, *, status_code: int, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(ApiError):
    pass


class NotFoundError(ApiError):
    pass


class RateLimitError(ApiError):
    pass


class TransportError(PlatformError):
    pass


class ApiDeadlineExceeded(PlatformError):
    pass


class InvalidResponseError(PlatformError):
    pass


class CreationOutcomeUnknown(PlatformError):
    """A create may have reached the server. Retrying could create a billable duplicate."""


class EvaluationFailed(PlatformError):
    def __init__(self, run_id: str, status: str) -> None:
        super().__init__(f"Evaluation {run_id} ended with status {status}")
        self.run_id = run_id


class InsufficientEvidence(PlatformError):
    def __init__(self, run_id: str, missing: list[tuple[str, str]]) -> None:
        super().__init__(f"Evaluation {run_id} has {len(missing)} missing or unusable scores")
        self.run_id = run_id
        self.missing = missing
