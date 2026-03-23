"""ContextVar-based session ID and user ID propagation.

Provides a universal mechanism for setting and retrieving the current
session ID and user ID across all SDK layers (decorators, wrappers,
integrations, context managers) without requiring explicit parameter passing.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# User ID
# ---------------------------------------------------------------------------

_current_user_id: ContextVar[str | None] = ContextVar("_current_user_id", default=None)


def get_current_user_id() -> str | None:
    """Return the user ID set in the current context, or ``None``."""
    return _current_user_id.get(None)


def set_current_user_id(user_id: str | None) -> Token[str | None]:
    """Set the user ID for the current context. Returns a reset token."""
    return _current_user_id.set(user_id)


def reset_current_user_id(token: Token[str | None]) -> None:
    """Reset the user ID to its previous value using the given token."""
    _current_user_id.reset(token)
