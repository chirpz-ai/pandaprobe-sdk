"""ContextVar-based session ID propagation.

Provides a universal mechanism for setting and retrieving the current
session ID across all SDK layers (decorators, wrappers, integrations,
context managers) without requiring explicit parameter passing.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_current_session_id: ContextVar[str | None] = ContextVar("_current_session_id", default=None)


def get_current_session_id() -> str | None:
    """Return the session ID set in the current context, or ``None``."""
    return _current_session_id.get(None)


def set_current_session_id(session_id: str | None) -> Token[str | None]:
    """Set the session ID for the current context. Returns a reset token."""
    return _current_session_id.set(session_id)


def reset_current_session_id(token: Token[str | None]) -> None:
    """Reset the session ID to its previous value using the given token."""
    _current_session_id.reset(token)
