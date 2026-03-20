"""Shared validation and utility functions for the standard trace/span schema.

The standard schema requires ``input`` and ``output`` to be a dict with a
``messages`` key whose value is a list of message objects.  Each message must
have at least ``role`` (str) and ``content`` (str | None).
"""

from __future__ import annotations

from typing import Any

_SCHEMA_EXAMPLE = (
    '{"messages": [{"role": "user", "content": "hello"}]}'
)


def validate_messages_format(data: Any, label: str) -> None:
    """Validate that *data* follows the standard messages schema.

    Rules:
    - ``None`` is accepted silently (no-op).
    - Otherwise *data* must be a ``dict`` with a ``"messages"`` key whose
      value is a ``list`` of dicts, each containing ``"role"`` (str) and
      ``"content"`` (str or None).

    Raises ``ValueError`` with a clear message on failure.
    """
    if data is None:
        return

    if not isinstance(data, dict):
        raise ValueError(
            f"{label} must be a dict with a 'messages' key, got {type(data).__name__}. "
            f"Expected format: {_SCHEMA_EXAMPLE}"
        )

    if "messages" not in data:
        raise ValueError(
            f"{label} dict must contain a 'messages' key. "
            f"Got keys: {list(data.keys())}. Expected format: {_SCHEMA_EXAMPLE}"
        )

    messages = data["messages"]
    if not isinstance(messages, list):
        raise ValueError(
            f"{label}['messages'] must be a list, got {type(messages).__name__}. "
            f"Expected format: {_SCHEMA_EXAMPLE}"
        )

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(
                f"{label}['messages'][{i}] must be a dict, got {type(msg).__name__}. "
                f"Each message must have at least 'role' and 'content' keys."
            )
        if "role" not in msg:
            raise ValueError(
                f"{label}['messages'][{i}] is missing required key 'role'. "
                f"Got keys: {list(msg.keys())}."
            )
        if not isinstance(msg["role"], str):
            raise ValueError(
                f"{label}['messages'][{i}]['role'] must be a string, "
                f"got {type(msg['role']).__name__}."
            )
        if "content" not in msg:
            raise ValueError(
                f"{label}['messages'][{i}] is missing required key 'content'. "
                f"Got keys: {list(msg.keys())}."
            )
        content = msg["content"]
        if content is not None and not isinstance(content, str):
            raise ValueError(
                f"{label}['messages'][{i}]['content'] must be a string or None, "
                f"got {type(content).__name__}."
            )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def validate_trace_input(value: Any) -> None:
    """Validate trace-level input (last user message only)."""
    validate_messages_format(value, "trace input")


def validate_trace_output(value: Any) -> None:
    """Validate trace-level output (assistant response)."""
    validate_messages_format(value, "trace output")


def validate_span_input(value: Any) -> None:
    """Validate span-level input (full conversation history for LLM spans)."""
    validate_messages_format(value, "span input")


def validate_span_output(value: Any) -> None:
    """Validate span-level output (model response for LLM spans)."""
    validate_messages_format(value, "span output")


# ---------------------------------------------------------------------------
# Extraction utility
# ---------------------------------------------------------------------------


def extract_last_user_message(input_data: Any) -> Any:
    """Extract only the last user message from a messages structure.

    Returns ``{"messages": [last_user_message]}`` if a user/human message
    is found.  Returns *input_data* unchanged when the structure doesn't
    match or no user message exists.
    """
    if not isinstance(input_data, dict):
        return input_data

    messages = input_data.get("messages")
    if not isinstance(messages, list):
        return input_data

    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            return {"messages": [msg]}

    return input_data


def extract_last_assistant_message(output_data: Any) -> Any:
    """Extract only the last assistant message from a messages structure.

    Returns ``{"messages": [last_assistant_message]}`` if an assistant/ai
    message is found.  Returns *output_data* unchanged when the structure
    doesn't match or no assistant message exists.
    """
    if not isinstance(output_data, dict):
        return output_data

    messages = output_data.get("messages")
    if not isinstance(messages, list):
        return output_data

    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
            return {"messages": [msg]}

    return output_data
