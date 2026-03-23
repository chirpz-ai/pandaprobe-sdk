"""Base adapter protocol shared by all framework integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandaprobe.client import Client


class BaseIntegrationAdapter:
    """Common foundation for framework integration adapters.

    Subclasses implement framework-specific hooking logic but share the same
    trace/span lifecycle management via the Tracing Core layer.
    """

    def __init__(
        self,
        *,
        client: Client | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._user_id = user_id
        self._tags = tags or []
        self._metadata = metadata or {}

    def _resolve_client(self) -> Client:
        """Return the explicit client or fall back to the global singleton."""
        if self._client is not None:
            return self._client
        from pandaprobe.client import get_client

        client = get_client()
        if client is None:
            raise RuntimeError(
                "No PandaProbe client available. "
                "Set PANDAPROBE_API_KEY and PANDAPROBE_PROJECT_NAME environment variables, "
                "or call pandaprobe.init() first, or pass client= to the adapter."
            )
        return client
