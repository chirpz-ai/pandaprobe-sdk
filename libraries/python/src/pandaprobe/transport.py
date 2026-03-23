"""Background transport layer for the PandaProbe SDK.

Manages a background thread with a synchronous httpx client
and a thread-safe queue to batch and send items efficiently,
ensuring reliable shutdown without async-related race conditions.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

import httpx

from pandaprobe._version import __version__
from pandaprobe.config import SdkConfig

logger = logging.getLogger("pandaprobe")

_SHUTDOWN = object()
_FLUSH = object()

_NO_RETRY_STATUSES = {401, 403, 422}
_RETRIABLE_STATUSES = {429, 500, 502, 503, 504}

MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5  # seconds


@dataclass
class _QueueItem:
    """Wraps a unit of work to be executed on the background loop."""

    kind: str  # "trace", "spans", "update_trace", "update_span", "score"
    payload: dict[str, Any]
    trace_id: str | None = None
    span_id: str | None = None


class Transport:
    """Non-blocking transport that batches and sends payloads in the background."""

    def __init__(self, config: SdkConfig, on_error: Callable[[Exception], None] | None = None) -> None:
        self._config = config
        self._on_error = on_error
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=config.max_queue_size)

        self._base_headers: dict[str, str] = {
            "X-API-Key": config.api_key or "",
            "X-Project-Name": config.project_name or "",
            "Content-Type": "application/json",
            "User-Agent": f"pandaprobe-python/{__version__}",
        }

        self._thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()

        if config.enabled:
            self._start_worker()
            atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Public enqueue helpers
    # ------------------------------------------------------------------

    def enqueue_trace(self, payload: dict[str, Any]) -> None:
        self._put(_QueueItem(kind="trace", payload=payload))

    def enqueue_spans(self, trace_id: str, payload: list[dict[str, Any]]) -> None:
        self._put(_QueueItem(kind="spans", payload={"spans": payload}, trace_id=trace_id))

    def enqueue_update_trace(self, trace_id: str, payload: dict[str, Any]) -> None:
        self._put(_QueueItem(kind="update_trace", payload=payload, trace_id=trace_id))

    def enqueue_update_span(self, trace_id: str, span_id: str, payload: dict[str, Any]) -> None:
        self._put(_QueueItem(kind="update_span", payload=payload, trace_id=trace_id, span_id=span_id))

    def enqueue_score(self, payload: dict[str, Any]) -> None:
        self._put(_QueueItem(kind="score", payload=payload))

    # ------------------------------------------------------------------
    # Flush / shutdown
    # ------------------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> None:
        """Block until all queued items have been sent (up to *timeout* seconds)."""
        if not self._config.enabled or self._shutdown_event.is_set():
            return
        self._flush_event.clear()
        self._queue.put(_FLUSH)
        self._flush_event.wait(timeout=timeout)

    def shutdown(self) -> None:
        """Flush remaining items and stop the background worker."""
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)

    # ------------------------------------------------------------------
    # Internal: background worker
    # ------------------------------------------------------------------

    def _put(self, item: _QueueItem) -> None:
        if not self._config.enabled or self._shutdown_event.is_set():
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("PandaProbe queue full — dropping oldest item")
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _start_worker(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pandaprobe-transport")
        self._thread.start()

    def _run_loop(self) -> None:
        """Background thread entry point.  Runs a synchronous worker loop."""
        http = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )
        try:
            self._worker(http)
        finally:
            http.close()

    def _worker(self, http: httpx.Client) -> None:
        batch: list[_QueueItem] = []
        last_flush = time.monotonic()

        while True:
            deadline = last_flush + self._config.flush_interval
            while True:
                wait = max(0.0, deadline - time.monotonic())
                try:
                    item = self._queue.get(timeout=wait if wait > 0 else 0.01)
                except queue.Empty:
                    break

                if item is _SHUTDOWN:
                    if batch:
                        self._flush_batch(http, batch)
                        batch.clear()
                    self._flush_event.set()
                    return

                if item is _FLUSH:
                    if batch:
                        self._flush_batch(http, batch)
                        batch.clear()
                        last_flush = time.monotonic()
                    self._flush_event.set()
                    continue

                batch.append(item)
                if len(batch) >= self._config.batch_size:
                    break

            if batch:
                self._flush_batch(http, batch)
                batch.clear()
                last_flush = time.monotonic()

    def _flush_batch(self, http: httpx.Client, batch: list[_QueueItem]) -> None:
        for item in batch:
            try:
                self._send(http, item)
            except Exception as exc:
                logger.error("PandaProbe transport error: %s", exc)
                if self._on_error:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass

    def _send(self, http: httpx.Client, item: _QueueItem) -> None:
        url, method, body = self._build_request(item)
        headers = {**self._base_headers, "X-Request-ID": str(uuid4())}

        for attempt in range(MAX_RETRIES + 1):
            try:
                if method == "POST":
                    resp = http.post(url, json=body, headers=headers)
                elif method == "PATCH":
                    resp = http.patch(url, json=body, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if resp.status_code < 300:
                    logger.debug("PandaProbe %s %s → %s", method, url, resp.status_code)
                    return

                if resp.status_code in _NO_RETRY_STATUSES:
                    logger.error(
                        "PandaProbe %s %s → %s (not retrying): %s",
                        method,
                        url,
                        resp.status_code,
                        resp.text[:500],
                    )
                    return

                if resp.status_code in _RETRIABLE_STATUSES:
                    retry_after = _parse_retry_after(resp)
                    backoff = retry_after or (INITIAL_BACKOFF * (2**attempt))
                    logger.warning(
                        "PandaProbe %s %s → %s, retrying in %.1fs (attempt %d/%d)",
                        method,
                        url,
                        resp.status_code,
                        backoff,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(backoff)
                    continue

                logger.error("PandaProbe %s %s → %s: %s", method, url, resp.status_code, resp.text[:500])
                return

            except httpx.HTTPError as exc:
                if attempt < MAX_RETRIES:
                    backoff = INITIAL_BACKOFF * (2**attempt)
                    logger.warning(
                        "PandaProbe connection error: %s, retrying in %.1fs (attempt %d/%d)",
                        exc,
                        backoff,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(backoff)
                else:
                    logger.error("PandaProbe connection error after %d retries: %s", MAX_RETRIES, exc)
                    raise

    def _build_request(self, item: _QueueItem) -> tuple[str, str, Any]:
        base = self._config.endpoint.rstrip("/")
        if item.kind == "trace":
            return f"{base}/traces", "POST", item.payload
        if item.kind == "spans":
            return f"{base}/traces/{item.trace_id}/spans", "POST", item.payload["spans"]
        if item.kind == "update_trace":
            return f"{base}/traces/{item.trace_id}", "PATCH", item.payload
        if item.kind == "update_span":
            return f"{base}/traces/{item.trace_id}/spans/{item.span_id}", "PATCH", item.payload
        if item.kind == "score":
            return f"{base}/evaluations/trace-scores", "POST", item.payload
        raise ValueError(f"Unknown item kind: {item.kind}")


def _parse_retry_after(resp: httpx.Response) -> float | None:
    val = resp.headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
