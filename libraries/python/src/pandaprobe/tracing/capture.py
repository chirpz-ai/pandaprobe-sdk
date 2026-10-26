"""Explicit capture of existing trace submissions within one application boundary.

This does not create traces, instrument frameworks, or wait for persistence.
The scope is owned by a Client and propagated with ContextVar, so concurrent
invocations and different clients cannot route into a last-active runtime.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any

from typing_extensions import Self


class TraceCapture:
    def __init__(
        self,
        variable: ContextVar,
        *,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        max_traces: int = 1000,
        on_submit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_traces < 1:
            raise ValueError("max_traces must be positive")
        self.trace_ids: list[str] = []
        self.overflow = False
        self.closed = False
        self._variable = variable
        self._token: Token | None = None
        self._metadata = dict(metadata or {})
        self._session_id = session_id
        self._max_traces = max_traces
        self._entered = False
        self._on_submit = on_submit

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("Trace capture scopes cannot be reused")
        self._entered = True
        self._token = self._variable.set(self)
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True
        if self._token is not None:
            self._variable.reset(self._token)

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(self, *_: object) -> None:
        self.__exit__()

    def _apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["metadata"] = {**payload.get("metadata", {}), **self._metadata}
        if self._session_id is not None:
            result["session_id"] = self._session_id
        if self.closed:
            # A child task may defer cancellation. Keep its inherited attribution
            # while refusing to add late work to an already completed boundary.
            return result
        trace_id = str(payload["trace_id"])
        if trace_id not in self.trace_ids:
            if len(self.trace_ids) < self._max_traces:
                self.trace_ids.append(trace_id)
            else:
                self.overflow = True
        if self._on_submit is not None:
            try:
                self._on_submit(result)
            except Exception:
                logging.getLogger(__name__).exception("Trace submission observer failed")
        return result
