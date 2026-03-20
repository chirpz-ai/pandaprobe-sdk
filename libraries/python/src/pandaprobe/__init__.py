"""PandaProbe — Python SDK for open-source agent tracing and evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from uuid import UUID

from pandaprobe._version import __version__
from pandaprobe.client import Client, get_client, init
from pandaprobe.decorators import span, trace
from pandaprobe.schemas import (
    ScoreDataType,
    SpanKind,
    SpanStatusCode,
    TraceStatus,
)
from pandaprobe.tracing.session import (
    reset_current_session_id,
    set_current_session_id,
)

__all__ = [
    "__version__",
    "Client",
    "init",
    "get_client",
    "trace",
    "span",
    "start_trace",
    "SpanKind",
    "SpanStatusCode",
    "TraceStatus",
    "ScoreDataType",
    "flush",
    "shutdown",
    "score",
    "set_session",
    "session",
]


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def flush(timeout: float = 30.0) -> None:
    """Block until all queued items are sent (delegates to the global client)."""
    client = get_client()
    if client is not None:
        client.flush(timeout)


def shutdown() -> None:
    """Flush remaining items and release resources (delegates to the global client)."""
    client = get_client()
    if client is not None:
        client.shutdown()


def score(
    trace_id: str | UUID,
    name: str,
    value: str,
    *,
    data_type: str = "NUMERIC",
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Submit a programmatic score for a trace (delegates to the global client)."""
    client = get_client()
    if client is not None:
        client.score(trace_id, name, value, data_type=data_type, reason=reason, metadata=metadata)


# ---------------------------------------------------------------------------
# Trace context manager
# ---------------------------------------------------------------------------


def start_trace(
    name: str,
    *,
    input: Any = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Start a new trace and return a :class:`TraceContext` context manager.

    Input must be ``{"messages": [{"role": "user", "content": "..."}]}``
    containing only the current turn's user message.
    Output must be ``{"messages": [{"role": "assistant", "content": "..."}]}``.

    Convenience wrapper that resolves the global client automatically::

        with pandaprobe.start_trace(
            "my-trace",
            input={"messages": [{"role": "user", "content": "hello"}]},
        ) as t:
            with t.span("llm-call", kind="LLM") as s:
                s.set_input({"messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                ]})
                ...
                s.set_output({"messages": [{"role": "assistant", "content": "world"}]})
            t.set_output({"messages": [{"role": "assistant", "content": "world"}]})
    """
    client = get_client()
    if client is None:
        raise RuntimeError(
            "No PandaProbe client available. "
            "Set PANDAPROBE_API_KEY and PANDAPROBE_PROJECT_NAME environment variables, "
            "or call pandaprobe.init() first."
        )
    return client.trace(
        name,
        input=input,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        metadata=metadata,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Session propagation
# ---------------------------------------------------------------------------


def set_session(session_id: str) -> None:
    """Set the session ID for the current context.

    All traces created after this call (via decorators, wrappers, or
    integrations) will inherit this session ID unless overridden explicitly.
    """
    set_current_session_id(session_id)


@contextmanager
def session(session_id: str):
    """Context manager that sets a session ID for its scope.

    Usage::

        with pandaprobe.session("conv-123"):
            run_agent(query)  # traces inherit session_id="conv-123"
        # session_id is reset after the block
    """
    token = set_current_session_id(session_id)
    try:
        yield
    finally:
        reset_current_session_id(token)
