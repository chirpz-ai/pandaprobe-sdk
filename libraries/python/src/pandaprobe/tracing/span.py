"""SpanContext — manages the lifecycle of a single span within a trace."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode
from pandaprobe.tracing.context import get_span_stack
from pandaprobe.validation import validate_span_input, validate_span_output

if TYPE_CHECKING:
    from pandaprobe.tracing.context import TraceContext

logger = logging.getLogger("pandaprobe")


class SpanContext:
    """Context manager for an individual span.

    For LLM spans (kind="LLM"), input must be
    ``{"messages": [{"role": "...", "content": "..."}]}`` (full conversation
    history) and output must be ``{"messages": [{"role": "...", "content": "..."}]}``.
    Other span kinds accept arbitrary input/output.

    Automatically manages timing, parent-child relationships via the span
    stack (contextvar), and error capture.
    """

    def __init__(
        self,
        trace_ctx: TraceContext,
        name: str,
        *,
        kind: SpanKind = SpanKind.OTHER,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._trace_ctx = trace_ctx
        self._span_id = str(uuid4())
        self._name = name
        self._kind = kind
        self._model = model
        self._metadata = metadata or {}
        self._input: Any = None
        self._output: Any = None
        self._status: SpanStatusCode = SpanStatusCode.UNSET
        self._error: str | None = None
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None
        self._token_usage: dict[str, int] | None = None
        self._model_parameters: dict[str, Any] | None = None
        self._cost: dict[str, float] | None = None
        self._completion_start_time: datetime | None = None
        self._parent_span_id: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def span_id(self) -> str:
        return self._span_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SpanContext:
        self._started_at = datetime.now(timezone.utc)
        stack = get_span_stack()
        if stack:
            self._parent_span_id = stack[-1]
        stack.append(self._span_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self._ended_at = datetime.now(timezone.utc)
        if exc_type is not None:
            self._status = SpanStatusCode.ERROR
            self._error = str(exc_val)
        elif self._status == SpanStatusCode.UNSET:
            self._status = SpanStatusCode.OK

        self._finalize()

        stack = get_span_stack()
        if stack and stack[-1] == self._span_id:
            stack.pop()

        return None

    async def __aenter__(self) -> SpanContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        return self.__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_input(self, input: Any) -> None:
        if self._kind == SpanKind.LLM:
            validate_span_input(input)
        self._input = input

    def set_output(self, output: Any) -> None:
        if self._kind == SpanKind.LLM:
            validate_span_output(output)
        self._output = output

    def set_token_usage(self, *, prompt_tokens: int = 0, completion_tokens: int = 0, **extra: int) -> None:
        self._token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, **extra}

    def set_model(self, model: str) -> None:
        self._model = model

    def set_model_parameters(self, params: dict[str, Any]) -> None:
        self._model_parameters = params

    def set_cost(self, *, total: float, **extra: float) -> None:
        self._cost = {"total": total, **extra}

    def set_completion_start_time(self, ts: datetime) -> None:
        self._completion_start_time = ts

    def set_error(self, error: str) -> None:
        self._error = error
        self._status = SpanStatusCode.ERROR

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self._metadata.update(metadata)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        try:
            span = SpanData(
                span_id=self._span_id,
                parent_span_id=self._parent_span_id,
                name=self._name,
                kind=self._kind,
                status=self._status,
                input=self._input,
                output=self._output,
                model=self._model,
                token_usage=self._token_usage,
                metadata=self._metadata,
                started_at=self._started_at,
                ended_at=self._ended_at,
                error=self._error,
                completion_start_time=self._completion_start_time,
                model_parameters=self._model_parameters,
                cost=self._cost,
            )
            self._trace_ctx.add_span_data(span)
        except Exception as exc:
            logger.error("PandaProbe failed to finalize span: %s", exc)
