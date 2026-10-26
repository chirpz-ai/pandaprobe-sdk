"""Trace discovery and evaluation resources using the backend's HTTP routes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import UUID

from .errors import ApiDeadlineExceeded, EvaluationFailed, InsufficientEvidence, InvalidResponseError, NotFoundError
from .models import EvaluationRequest, EvaluationResult, EvaluationRun, Metric, Score, Trace, TracePage, TraceSummary

if TYPE_CHECKING:
    from .client import PlatformClient


class Traces:
    def __init__(self, client: PlatformClient) -> None:
        self._client = client

    async def get(self, trace_id: str | UUID, *, timeout: float | None = None) -> Trace:
        value = await self._client.request("GET", f"traces/{UUID(str(trace_id))}", timeout=timeout)
        trace = self._client.decode(Trace, value)
        if trace.trace_id != UUID(str(trace_id)):
            raise InvalidResponseError("Retrieved trace identity does not match the request")
        return trace

    async def list(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        timeout: float | None = None,
    ) -> TracePage:
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("limit must be 1–200 and offset nonnegative")
        params = {
            k: v
            for k, v in {"session_id": session_id, "status": status, "limit": limit, "offset": offset}.items()
            if v is not None
        }
        value = await self._client.request("GET", "traces", params=params, timeout=timeout)
        return self._client.decode(TracePage, value)

    async def iterate(self, *, session_id: str | None = None, max_items: int = 1000) -> AsyncIterator[TraceSummary]:
        offset = 0
        while offset < max_items:
            page = await self.list(session_id=session_id, limit=min(200, max_items - offset), offset=offset)
            for item in page.items:
                yield item
            offset += len(page.items)
            if not page.items or offset >= page.total:
                break

    async def wait_visible(
        self,
        trace_id: str | UUID,
        *,
        timeout: float = 30,
        poll_interval: float = 0.5,
    ) -> Trace:
        """Wait for persisted, finished evidence, including task ERROR traces."""
        _check_poll(timeout, poll_interval)
        deadline = time.monotonic() + timeout
        while True:
            try:
                trace = await self.get(trace_id, timeout=_remaining(deadline))
                if trace.status in ("COMPLETED", "ERROR"):
                    return trace
            except NotFoundError:
                pass
            await _pause(deadline, poll_interval)


class Evaluations:
    def __init__(self, client: PlatformClient) -> None:
        self._client = client

    async def metrics(self) -> list[Metric]:
        data = await self._client.request("GET", "evaluations/trace-metrics")
        return self._client.decode_many(Metric, data)

    async def create(
        self,
        trace_ids: list[str | UUID],
        metrics: list[str],
        *,
        name: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> EvaluationRun:
        request = EvaluationRequest(
            trace_ids=list(dict.fromkeys(UUID(str(t)) for t in trace_ids)),
            metrics=list(dict.fromkeys(metrics)),
            name=name,
            model=model,
        )
        data = await self._client.request(
            "POST",
            "evaluations/trace-runs/batch",
            timeout=timeout,
            json=request.model_dump(mode="json", exclude_none=True),
        )
        return self._client.decode(EvaluationRun, data, creation=True)

    async def get(self, run_id: str | UUID, *, timeout: float | None = None) -> EvaluationRun:
        data = await self._client.request("GET", f"evaluations/trace-runs/{UUID(str(run_id))}", timeout=timeout)
        run = self._client.decode(EvaluationRun, data)
        if run.id != UUID(str(run_id)):
            raise InvalidResponseError("Retrieved evaluation identity does not match the request")
        return run

    async def scores(self, run_id: str | UUID, *, timeout: float | None = None) -> list[Score]:
        data = await self._client.request("GET", f"evaluations/trace-runs/{UUID(str(run_id))}/scores", timeout=timeout)
        return self._client.decode_many(Score, data)

    async def wait(
        self,
        run_id: str | UUID,
        *,
        trace_ids: list[str | UUID],
        metrics: list[str],
        timeout: float = 120,
        poll_interval: float = 1.0,
    ) -> EvaluationResult:
        """Poll one existing run; terminal status alone is not usable evidence."""
        _check_poll(timeout, poll_interval)
        expected = {(str(UUID(str(t))), m) for t in trace_ids for m in metrics}
        if not expected:
            raise ValueError("Expected traces and metrics must not be empty")
        deadline = time.monotonic() + timeout
        while True:
            run = await self.get(run_id, timeout=_remaining(deadline))
            if run.status in ("FAILED", "CANCELLED"):
                raise EvaluationFailed(str(run.id), run.status)
            if run.status == "COMPLETED":
                scores = await self.scores(run_id, timeout=_remaining(deadline))
                usable: dict[tuple[str, str], Score] = {}
                duplicates: set[tuple[str, str]] = set()
                seen: set[tuple[str, str]] = set()
                for score in scores:
                    key = (str(score.trace_id), score.name)
                    if key in seen:
                        duplicates.add(key)
                    seen.add(key)
                    if score.eval_run_id == run.id and score.numeric_value is not None:
                        usable[key] = score
                missing = sorted(expected - (usable.keys() - duplicates))
                if missing:
                    raise InsufficientEvidence(str(run.id), missing)
                return EvaluationResult(run=run, scores=[usable[key] for key in sorted(expected)])
            await _pause(deadline, poll_interval)


def _check_poll(timeout: float, interval: float) -> None:
    import math

    if timeout <= 0 or interval <= 0 or not math.isfinite(timeout + interval):
        raise ValueError("Polling timeout and interval must be finite and positive")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ApiDeadlineExceeded("Polling deadline exceeded")
    return remaining


async def _pause(deadline: float, interval: float) -> None:
    await asyncio.sleep(min(_remaining(deadline), interval))
