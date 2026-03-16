"""Pydantic models matching the PandaProbe backend API schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpanKind(StrEnum):
    AGENT = "AGENT"
    TOOL = "TOOL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    CHAIN = "CHAIN"
    EMBEDDING = "EMBEDDING"
    OTHER = "OTHER"


class SpanStatusCode(StrEnum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class TraceStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class ScoreDataType(StrEnum):
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    CATEGORICAL = "CATEGORICAL"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class SpanData(BaseModel):
    model_config = {"populate_by_name": True}

    span_id: UUID = Field(default_factory=uuid4)
    parent_span_id: UUID | None = None
    name: str = Field(min_length=1, max_length=512)
    kind: SpanKind = SpanKind.OTHER
    status: SpanStatusCode = SpanStatusCode.UNSET
    input: Any | None = None
    output: Any | None = None
    model: str | None = Field(default=None, max_length=255)
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime | None = None
    error: str | None = None
    completion_start_time: datetime | None = None
    model_parameters: dict[str, Any] | None = None
    cost: dict[str, float] | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to a dict matching the backend POST /traces span schema."""
        d: dict[str, Any] = {}
        d["span_id"] = str(self.span_id)
        if self.parent_span_id is not None:
            d["parent_span_id"] = str(self.parent_span_id)
        d["name"] = self.name
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        if self.input is not None:
            d["input"] = _safe_json(self.input)
        if self.output is not None:
            d["output"] = _safe_json(self.output)
        if self.model is not None:
            d["model"] = self.model
        if self.token_usage is not None:
            d["token_usage"] = self.token_usage
        if self.metadata:
            d["metadata"] = self.metadata
        d["started_at"] = _iso(self.started_at)
        if self.ended_at is not None:
            d["ended_at"] = _iso(self.ended_at)
        if self.error is not None:
            d["error"] = self.error
        if self.completion_start_time is not None:
            d["completion_start_time"] = _iso(self.completion_start_time)
        if self.model_parameters is not None:
            d["model_parameters"] = self.model_parameters
        if self.cost is not None:
            d["cost"] = self.cost
        return d


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TraceData(BaseModel):
    model_config = {"populate_by_name": True}

    trace_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=512)
    status: TraceStatus = TraceStatus.COMPLETED
    input: Any | None = None
    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime | None = None
    session_id: str | None = Field(default=None, max_length=255)
    user_id: str | None = Field(default=None, max_length=255)
    tags: list[str] = Field(default_factory=list)
    environment: str | None = Field(default=None, max_length=255)
    release: str | None = Field(default=None, max_length=255)
    spans: list[SpanData] = Field(default_factory=list, max_length=500)

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to a dict matching the backend POST /traces schema."""
        d: dict[str, Any] = {
            "trace_id": str(self.trace_id),
            "name": self.name,
            "status": self.status.value,
            "started_at": _iso(self.started_at),
        }
        if self.input is not None:
            d["input"] = _safe_json(self.input)
        if self.output is not None:
            d["output"] = _safe_json(self.output)
        if self.metadata:
            d["metadata"] = self.metadata
        if self.ended_at is not None:
            d["ended_at"] = _iso(self.ended_at)
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.user_id is not None:
            d["user_id"] = self.user_id
        if self.tags:
            d["tags"] = self.tags
        if self.environment is not None:
            d["environment"] = self.environment
        if self.release is not None:
            d["release"] = self.release
        if self.spans:
            d["spans"] = [s.to_api_dict() for s in self.spans]
        return d


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


class ScoreData(BaseModel):
    trace_id: UUID
    name: str
    value: str
    data_type: ScoreDataType = ScoreDataType.NUMERIC
    source: str = "PROGRAMMATIC"
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trace_id": str(self.trace_id),
            "name": self.name,
            "value": self.value,
            "data_type": self.data_type.value,
            "source": self.source,
        }
        if self.reason is not None:
            d["reason"] = self.reason
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """Format a datetime as an ISO 8601 string with timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _safe_json(value: Any) -> Any:
    """Ensure a value is JSON-serializable; fall back to repr()."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    try:
        import json

        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)
