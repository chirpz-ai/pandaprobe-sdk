"""Public platform read models, independent of tracing and Reflect policy."""

from __future__ import annotations

import math
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    # Preserve additive server fields without requiring a simultaneous SDK release.
    model_config = ConfigDict(extra="allow")


class Span(ApiModel):
    span_id: UUID
    trace_id: UUID
    parent_span_id: UUID | None = None
    name: str
    kind: str
    status: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TraceSummary(ApiModel):
    trace_id: UUID
    name: str
    status: Literal["PENDING", "RUNNING", "COMPLETED", "ERROR"]
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    started_at: str
    ended_at: str | None = None


class Trace(TraceSummary):
    project_id: UUID
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[Span] = Field(default_factory=list)


class TracePage(ApiModel):
    items: list[TraceSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class Metric(ApiModel):
    name: str
    description: str
    category: str


class EvaluationRequest(BaseModel):
    trace_ids: list[UUID] = Field(min_length=1)
    metrics: list[str] = Field(min_length=1)
    name: str | None = None
    model: str | None = None


class EvaluationRun(ApiModel):
    id: UUID
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    metric_names: list[str]
    total_targets: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    failed_count: int = Field(default=0, ge=0)
    target_type: str
    error_message: str | None = None


class Score(ApiModel):
    id: UUID
    trace_id: UUID
    name: str
    value: str | None
    status: Literal["PENDING", "SUCCESS", "FAILED"]
    data_type: str
    eval_run_id: UUID | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def numeric_value(self) -> float | None:
        if self.status != "SUCCESS" or self.data_type != "NUMERIC" or self.value is None:
            return None
        try:
            number = float(self.value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None


class EvaluationResult(BaseModel):
    """A completed run with a complete set of usable scores, not proof of improvement."""

    run: EvaluationRun
    scores: list[Score]
