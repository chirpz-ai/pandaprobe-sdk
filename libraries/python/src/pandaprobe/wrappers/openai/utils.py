"""OpenAI-specific utilities."""

from __future__ import annotations

from typing import Any


def strip_not_given(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove OpenAI ``NotGiven``/``Omit`` sentinel types from kwargs.

    OpenAI's SDK uses these sentinel types for unset parameters. They must
    be stripped before logging, otherwise captured inputs are polluted with
    non-serialisable objects.
    """
    try:
        from openai._types import NotGiven, Omit

        return {k: v for k, v in kwargs.items() if not isinstance(v, (NotGiven, Omit))}
    except ImportError:
        return kwargs
