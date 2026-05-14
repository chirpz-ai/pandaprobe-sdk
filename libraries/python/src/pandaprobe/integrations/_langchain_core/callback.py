"""Generic ``langchain-core`` callback handler shared by every LangChain-based integration.

This base class encapsulates the LangChain ``BaseCallbackHandler`` event
mapping to the PandaProbe trace/span schema. Concrete framework integrations
(``langgraph``, ``langchain``, future ``deepagents``) subclass it and
override only the small set of hooks below.

Override surface:

* ``DEFAULT_TRACE_NAME`` — the trace name used until the root chain start
  callback overwrites it with the framework's chain name.
* ``_classify_chain_kind(has_parent, serialized)`` — decides the
  :class:`~pandaprobe.schemas.SpanKind` for chain spans. Default keeps every
  integration aligned: root chain ⇒ ``CHAIN``, every nested chain ⇒
  ``AGENT``.
* ``_on_root_chain_start(...)`` — optional no-op hook for subclass-specific
  bookkeeping at the start of the outermost run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pandaprobe.integrations._base import BaseIntegrationAdapter
from pandaprobe.integrations._langchain_core.utils import (
    extract_model_parameters,
    extract_name,
    extract_reasoning_from_generation,
    extract_token_usage,
    normalize_langchain_input,
    normalize_langchain_output,
    normalize_llm_generation_output,
    normalize_type_to_role,
    safe_output,
)
from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message

logger = logging.getLogger("pandaprobe")

try:
    from langchain_core.callbacks import BaseCallbackHandler as _LCBase
except ImportError:  # pragma: no cover
    _LCBase = object  # type: ignore[assignment,misc]


class BaseLangChainCallbackHandler(BaseIntegrationAdapter, _LCBase):  # type: ignore[misc]
    """Shared ``langchain-core`` callback handler.

    Maps LangChain run events keyed by ``run_id`` / ``parent_run_id`` to
    PandaProbe :class:`~pandaprobe.schemas.SpanData` and submits a single
    :class:`~pandaprobe.schemas.TraceData` via :meth:`Client.log_trace` when
    the outermost chain finishes.
    """

    DEFAULT_TRACE_NAME: str = "LangChain"

    def __init__(self, **kwargs: Any) -> None:
        BaseIntegrationAdapter.__init__(self, **kwargs)
        if _LCBase is not object:
            _LCBase.__init__(self)

        self._spans: dict[str, SpanData] = {}
        self._parents: dict[str, str | None] = {}
        self._root_run_id: str | None = None
        self._trace_input: Any = None
        self._trace_output: Any = None
        self._trace_started_at: datetime | None = None
        self._trace_name: str = self.DEFAULT_TRACE_NAME

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    def _classify_chain_kind(self, has_parent: bool, serialized: dict[str, Any] | None) -> SpanKind:
        """Decide the :class:`SpanKind` for a chain span.

        Default heuristic: root chain ⇒ ``CHAIN``, every nested chain ⇒
        ``AGENT``. Subclasses may override for framework-specific accuracy.
        """
        return SpanKind.AGENT if has_parent else SpanKind.CHAIN

    def _filter_root_chain_name(self, name: str, serialized: dict[str, Any] | None) -> str:
        """Remap framework-internal default root names. Default identity.

        Subclasses can use this to suppress framework-leaked names (e.g.
        LangChain's ``create_agent`` emits the internal graph name
        ``"LangGraph"`` as the root chain name) in favour of the
        integration's :attr:`DEFAULT_TRACE_NAME`.
        """
        return name

    def _on_root_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        run_id: str,
        name: str,
    ) -> None:
        """Hook fired the first time we observe a root chain start. Default no-op."""

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
            name = self._filter_root_chain_name(name, serialized)
            self._root_run_id = rid
            normalized = normalize_langchain_input(safe_output(inputs))
            self._trace_input = extract_last_user_message(normalized)
            self._trace_started_at = datetime.now(timezone.utc)
            self._trace_name = name
            self._on_root_chain_start(serialized, inputs, rid, name)

        kind = self._classify_chain_kind(pid is not None, serialized)
        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=kind,
            input=normalize_langchain_input(safe_output(inputs)),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.output = normalize_type_to_role(safe_output(outputs))
        span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)

        if rid == self._root_run_id:
            self._trace_output = normalize_langchain_output(safe_output(outputs))
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
        params = invocation_params or {}
        model = params.get("model") or params.get("model_name")
        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=SpanKind.LLM,
            input=safe_output(prompts),
            model=model,
            model_parameters=extract_model_parameters(params),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        self._parents[rid] = pid
        name = name or extract_name(serialized, "llm")
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name")

        serialized_msgs: list[Any] = []
        if messages and messages[0]:
            for msg in messages[0]:
                serialized_msgs.append(safe_output(msg))

        span = SpanData(
            span_id=rid,
            parent_span_id=pid,
            name=name,
            kind=SpanKind.LLM,
            input=normalize_type_to_role({"messages": serialized_msgs}),
            model=model,
            model_parameters=extract_model_parameters(params),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return

        try:
            normalized = normalize_llm_generation_output(response)
            if normalized is not None:
                span.output = normalized
            span.token_usage = extract_token_usage(response)
            reasoning = extract_reasoning_from_generation(response)
            if reasoning:
                span.metadata["reasoning_summary"] = reasoning
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
            input=normalize_type_to_role(safe_output(input_str)),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.output = normalize_type_to_role(safe_output(output))
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
            input=normalize_type_to_role(safe_output(query)),
            started_at=datetime.now(timezone.utc),
        )
        self._spans[rid] = span

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        span = self._spans.get(rid)
        if span is None:
            return
        span.output = normalize_type_to_role(safe_output(documents))
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
            user_id = self._user_id if self._user_id is not None else get_current_user_id()
            trace = TraceData(
                name=self._trace_name,
                status=TraceStatus.ERROR if error else TraceStatus.COMPLETED,
                input=self._trace_input,
                output=self._trace_output,
                metadata=dict(self._metadata),
                started_at=self._trace_started_at or datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                session_id=session_id,
                user_id=user_id,
                tags=list(self._tags),
                spans=spans,
            )
            client.log_trace(trace)
        except Exception as exc:
            logger.error("PandaProbe %s callback failed to submit trace: %s", self.DEFAULT_TRACE_NAME, exc)
        finally:
            self._spans.clear()
            self._parents.clear()
            self._root_run_id = None
