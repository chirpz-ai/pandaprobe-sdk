"""PandaProbe Client — main entry point for the SDK."""

from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import UUID

from pandaprobe.config import SdkConfig, _env_bool, resolve_config
from pandaprobe.schemas import ScoreData, ScoreDataType, TraceData
from pandaprobe.transport import Transport

logger = logging.getLogger("pandaprobe")

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_client: Client | None = None
_auto_init_attempted: bool = False


def init(
    *,
    api_key: str | None = None,
    project_name: str | None = None,
    endpoint: str | None = None,
    environment: str | None = None,
    release: str | None = None,
    enabled: bool | None = None,
    batch_size: int | None = None,
    flush_interval: float | None = None,
    max_queue_size: int | None = None,
    debug: bool | None = None,
) -> Client:
    """Configure and set the global PandaProbe client singleton.

    Returns the newly created :class:`Client` instance.
    """
    global _global_client
    if _global_client is not None:
        _global_client.shutdown()
    _global_client = Client(
        api_key=api_key,
        project_name=project_name,
        endpoint=endpoint,
        environment=environment,
        release=release,
        enabled=enabled,
        batch_size=batch_size,
        flush_interval=flush_interval,
        max_queue_size=max_queue_size,
        debug=debug,
    )
    return _global_client


def get_client() -> Client | None:
    """Return the current global client.

    On first call, if no client has been created via :func:`init`, attempts
    auto-initialization from environment variables.  Auto-init is gated by
    ``PANDAPROBE_ENABLED`` (defaults to ``true``).
    """
    global _global_client, _auto_init_attempted
    if _global_client is None and not _auto_init_attempted:
        _auto_init_attempted = True
        _global_client = _try_auto_init()
    return _global_client


def _try_auto_init() -> Client | None:
    """Attempt to create a Client from environment variables.

    Returns ``None`` silently when ``PANDAPROBE_ENABLED=false`` or when
    required env vars (API key / project name) are missing.
    """
    if not _env_bool("PANDAPROBE_ENABLED", True):
        return None
    try:
        return Client()
    except ValueError as exc:
        logger.debug("PandaProbe auto-init skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    """PandaProbe SDK client.

    Can be used directly (``pandaprobe.Client(...)``) or via the module-level
    singleton (``pandaprobe.init(...)``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_name: str | None = None,
        endpoint: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        enabled: bool | None = None,
        batch_size: int | None = None,
        flush_interval: float | None = None,
        max_queue_size: int | None = None,
        debug: bool | None = None,
    ) -> None:
        self._config: SdkConfig = resolve_config(
            api_key=api_key,
            project_name=project_name,
            endpoint=endpoint,
            environment=environment,
            release=release,
            enabled=enabled,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            debug=debug,
        )
        self._error_callbacks: list[Callable[[Exception], None]] = []
        self._transport = Transport(config=self._config, on_error=self._dispatch_error)

        if self._config.enabled:
            logger.debug("PandaProbe client initialised (endpoint=%s)", self._config.endpoint)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> SdkConfig:
        return self._config

    # ------------------------------------------------------------------
    # Trace operations
    # ------------------------------------------------------------------

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Start a new trace and return a :class:`TraceContext` context manager."""
        from pandaprobe.tracing.context import TraceContext

        return TraceContext(
            client=self,
            name=name,
            input=input,
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def log_trace(self, trace_data: TraceData) -> None:
        """Enqueue a fully-formed trace for background submission."""
        if not self._config.enabled:
            return
        if self._config.environment and not trace_data.environment:
            trace_data.environment = self._config.environment
        if self._config.release and not trace_data.release:
            trace_data.release = self._config.release
        self._transport.enqueue_trace(trace_data.to_api_dict())

    # ------------------------------------------------------------------
    # Score operations
    # ------------------------------------------------------------------

    def score(
        self,
        trace_id: str | UUID,
        name: str,
        value: str,
        *,
        data_type: str = "NUMERIC",
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Submit a programmatic score for a trace."""
        if not self._config.enabled:
            return
        score = ScoreData(
            trace_id=trace_id if isinstance(trace_id, UUID) else UUID(trace_id),
            name=name,
            value=value,
            data_type=ScoreDataType(data_type),
            reason=reason,
            metadata=metadata or {},
        )
        self._transport.enqueue_score(score.to_api_dict())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> None:
        """Block until all queued items are sent."""
        self._transport.flush(timeout=timeout)

    def shutdown(self) -> None:
        """Flush remaining items and release resources."""
        self._transport.shutdown()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register an error callback invoked when a transport error occurs."""
        self._error_callbacks.append(callback)

    def _dispatch_error(self, exc: Exception) -> None:
        for cb in self._error_callbacks:
            try:
                cb(exc)
            except Exception:
                pass
