"""TraceContext — manages the lifecycle of a single trace."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pandaprobe.schemas import SpanKind, TraceData, TraceStatus
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import warn_if_invalid_messages

if TYPE_CHECKING:
    from pandaprobe.client import Client
    from pandaprobe.tracing.span import SpanContext

logger = logging.getLogger("pandaprobe")

# Global contextvar tracking the active trace context.
_current_trace: ContextVar[TraceContext | None] = ContextVar("_current_trace", default=None)
# Global contextvar tracking the stack of active span IDs.
_span_stack: ContextVar[list[str]] = ContextVar("_span_stack", default=[])


def get_current_trace() -> TraceContext | None:
    return _current_trace.get(None)


def get_span_stack() -> list[str]:
    return _span_stack.get()


class TraceContext:
    """Context manager that wraps a trace lifecycle.

    Input must be ``{"messages": [{"role": "user", "content": "..."}]}``
    containing only the current turn's user message.
    Output must be ``{"messages": [{"role": "assistant", "content": "..."}]}``.

    Usage::

        with client.trace(
            "my-trace",
            input={"messages": [{"role": "user", "content": "hello"}]},
        ) as t:
            with t.span("llm-call", kind="LLM") as s:
                s.set_input({"messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                ]})
                ...
                s.set_output({"messages": [{"role": "assistant", "content": "hi!"}]})
            t.set_output({"messages": [{"role": "assistant", "content": "hi!"}]})
    """

    def __init__(
        self,
        client: Client,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._client = client
        self._trace_id = str(uuid4())
        self._name = name
        self._input = input
        warn_if_invalid_messages(input, "trace input")
        self._output: Any = None
        self._session_id = session_id if session_id is not None else get_current_session_id()
        self._user_id = user_id if user_id is not None else get_current_user_id()
        self._tags = tags or []
        self._metadata = metadata or {}
        self._status: TraceStatus = TraceStatus.COMPLETED
        self._error: str | None = None
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None
        self._spans: list = []
        self._token: Token[TraceContext | None] | None = None
        self._stack_token: Token[list[str]] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trace_id(self) -> str:
        return self._trace_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TraceContext:
        self._started_at = datetime.now(timezone.utc)
        self._token = _current_trace.set(self)
        self._stack_token = _span_stack.set([])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self._ended_at = datetime.now(timezone.utc)
        if exc_type is not None:
            self._status = TraceStatus.ERROR
            self._error = str(exc_val)

        self._finalize()

        if self._token is not None:
            _current_trace.reset(self._token)
        if self._stack_token is not None:
            _span_stack.reset(self._stack_token)

        return None  # don't suppress exceptions

    async def __aenter__(self) -> TraceContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        return self.__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Span factory
    # ------------------------------------------------------------------

    def span(
        self,
        name: str,
        *,
        kind: str | SpanKind = SpanKind.OTHER,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        from pandaprobe.tracing.span import SpanContext

        return SpanContext(
            trace_ctx=self,
            name=name,
            kind=SpanKind(kind) if isinstance(kind, str) else kind,
            model=model,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_output(self, output: Any) -> None:
        warn_if_invalid_messages(output, "trace output")
        self._output = output

    def set_input(self, input: Any) -> None:
        warn_if_invalid_messages(input, "trace input")
        self._input = input

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self._metadata.update(metadata)

    def set_status(self, status: TraceStatus | str) -> None:
        self._status = TraceStatus(status)

    def add_span_data(self, span_data) -> None:  # noqa: ANN001
        self._spans.append(span_data)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        """Build TraceData and enqueue for background submission."""
        try:
            trace = TraceData(
                trace_id=self._trace_id,
                name=self._name,
                status=self._status,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                started_at=self._started_at,
                ended_at=self._ended_at,
                session_id=self._session_id,
                user_id=self._user_id,
                tags=self._tags,
                spans=self._spans,
            )
            self._client.log_trace(trace)
        except Exception as exc:
            logger.error("PandaProbe failed to finalize trace: %s", exc)
