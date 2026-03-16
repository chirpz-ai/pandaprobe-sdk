"""SessionManager — convenience wrapper that pins a session_id to all traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandaprobe.client import Client
    from pandaprobe.tracing.context import TraceContext


class SessionManager:
    """Thin helper returned by :pymeth:`Client.session`.

    Every trace started through this manager inherits the given *session_id*.
    """

    def __init__(self, client: Client, session_id: str) -> None:
        self._client = client
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TraceContext:
        return self._client.trace(
            name,
            input=input,
            session_id=self._session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
