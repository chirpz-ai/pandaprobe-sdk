"""LangChain/LangGraph CallbackHandler for PandaProbe tracing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pandaprobe.integrations._base import BaseIntegrationAdapter
from pandaprobe.integrations.langgraph.utils import extract_name, safe_output
from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus
from pandaprobe.tracing.session import get_current_session_id

logger = logging.getLogger("pandaprobe")

try:
    from langchain_core.callbacks import BaseCallbackHandler as _LCBase
except ImportError:  # pragma: no cover
    _LCBase = object  # type: ignore[assignment,misc]


class LangGraphCallbackHandler(BaseIntegrationAdapter, _LCBase):  # type: ignore[misc]
    """LangChain ``BaseCallbackHandler`` that maps graph events to PandaProbe traces.

    Usage::

        from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

        handler = LangGraphCallbackHandler()
        result = graph.invoke(input, config={"callbacks": [handler]})
    """

    def __init__(self, **kwargs: Any) -> None:
        # _LCBase (BaseCallbackHandler) expects no args when it's object,
        # but the real class needs to be initialised too.
        BaseIntegrationAdapter.__init__(self, **kwargs)
        if _LCBase is not object:
            _LCBase.__init__(self)

        # run_id → SpanData mapping for all active spans
        self._spans: dict[str, SpanData] = {}
        # run_id → parent_run_id
        self._parents: dict[str, str | None] = {}
        # root run_id (the outermost chain)
        self._root_run_id: str | None = None
        self._trace_input: Any = None
        self._trace_output: Any = None
        self._trace_started_at: datetime | None = None
        self._trace_name: str = "LangGraph"

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        self._parents[rid] = pid
        name = name or extract_name(serialized, "chain")

        if pid is None:
            self._root_run_id = rid
            self._trace_input = safe_output(inputs)
            self._trace_started_at = datetime.now(timezone.utc)
            self._trace_name = name

        kind = SpanKind.AGENT if pid is not None else SpanKind.CHAIN
        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=kind,
            input=safe_output(inputs),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.output = safe_output(outputs)
        span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)

        if rid == self._root_run_id:
            self._trace_output = safe_output(outputs)
            self._finalize_trace()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.error = str(error)
        span.status = SpanStatusCode.ERROR
        span.ended_at = datetime.now(timezone.utc)

        if rid == self._root_run_id:
            self._finalize_trace(error=True)

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        invocation_params: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        self._parents[rid] = pid
        name = name or extract_name(serialized, "llm")
        model = (invocation_params or {}).get("model") or (invocation_params or {}).get("model_name")
        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=SpanKind.LLM,
            input=safe_output(prompts),
            model=model,
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return

        try:
            if hasattr(response, "generations") and response.generations:
                span.output = safe_output(response.generations)
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage")
                if token_usage:
                    span.token_usage = {
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                    }
        except Exception as exc:
            logger.debug("Error extracting LLM response: %s", exc)

        span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.error = str(error)
        span.status = SpanStatusCode.ERROR
        span.ended_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        self._parents[rid] = pid
        name = name or extract_name(serialized, "tool")
        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=SpanKind.TOOL,
            input=safe_output(input_str),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.output = safe_output(output)
        span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.error = str(error)
        span.status = SpanStatusCode.ERROR
        span.ended_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Retriever callbacks
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        self._parents[rid] = pid
        name = name or extract_name(serialized, "retriever")
        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=SpanKind.RETRIEVER,
            input=safe_output(query),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.output = safe_output(documents)
        span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_trace(self, *, error: bool = False) -> None:
        """Build a TraceData from collected spans and submit."""
        try:
            client = self._resolve_client()
            spans = list(self._spans.values())
            session_id = self._session_id if self._session_id is not None else get_current_session_id()
            trace = TraceData(
                name=self._trace_name,
                status=TraceStatus.ERROR if error else TraceStatus.COMPLETED,
                input=self._trace_input,
                output=self._trace_output,
                metadata=dict(self._metadata),
                started_at=self._trace_started_at or datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                session_id=session_id,
                user_id=self._user_id,
                tags=list(self._tags),
                spans=spans,
            )
            client.log_trace(trace)
        except Exception as exc:
            logger.error("PandaProbe LangGraph callback failed to submit trace: %s", exc)
        finally:
            self._spans.clear()
            self._parents.clear()
            self._root_run_id = None
