"""Claude Agent SDK-specific normalization and extraction utilities."""

from __future__ import annotations

import json
import logging
from typing import Any

from pandaprobe.integrations._base import SAFE_MODEL_PARAM_KEYS, config_to_dict

logger = logging.getLogger("pandaprobe")


def safe_serialize(obj: Any) -> Any:
    """Best-effort JSON-safe serialization of an arbitrary object.

    Handles Claude Agent SDK dataclass types, Pydantic models, and plain dicts.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    return repr(obj)


def flatten_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert SDK content blocks (dataclasses) to serializable dicts.

    Handles ``TextBlock``, ``ThinkingBlock``, ``ToolUseBlock``, and
    ``ToolResultBlock`` from the Claude Agent SDK.  Unknown block types
    are serialized via ``safe_serialize``.
    """
    if not isinstance(content, list):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return []

    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            result.append(block)
            continue

        block_type = type(block).__name__

        if block_type == "TextBlock":
            result.append({"type": "text", "text": getattr(block, "text", "")})
        elif block_type == "ThinkingBlock":
            result.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    "signature": getattr(block, "signature", ""),
                }
            )
        elif block_type == "ToolUseBlock":
            result.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", None),
                }
            )
        elif block_type == "ToolResultBlock":
            raw_content = getattr(block, "content", None)
            text = _extract_tool_result_text(raw_content)
            result.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "tool_use_id", None),
                    "content": text,
                    "is_error": getattr(block, "is_error", False),
                }
            )
        else:
            result.append(safe_serialize(block))

    return result


def _extract_tool_result_text(content: Any) -> str:
    """Extract text from tool result content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                texts.append(getattr(item, "text", ""))
        return "\n".join(texts) if texts else str(content)
    return str(content)


def extract_thinking_from_content(content: Any) -> str | None:
    """Extract thinking/reasoning text from content blocks.

    Looks for ``ThinkingBlock`` dataclasses or dicts with
    ``"type": "thinking"``.  Returns concatenated thinking text or ``None``.
    """
    if not isinstance(content, list):
        return None

    thinking_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                text = block.get("thinking") or block.get("text")
                if text:
                    thinking_parts.append(str(text))
        elif type(block).__name__ == "ThinkingBlock":
            text = getattr(block, "thinking", None)
            if text:
                thinking_parts.append(str(text))

    return "\n\n".join(thinking_parts) if thinking_parts else None


def strip_thinking_blocks(content: Any) -> Any:
    """Remove thinking blocks from content and collapse text-only lists.

    1. If *content* is not a list, return as-is.
    2. Filter out ``{"type": "thinking", ...}`` dicts.
    3. If the remaining blocks are all text (strings or ``{"type": "text"}``),
       collapse into a plain string.
    4. Otherwise return the filtered list.
    """
    if not isinstance(content, list):
        return content

    filtered = [block for block in content if not (isinstance(block, dict) and block.get("type") == "thinking")]

    if not filtered:
        return content

    text_parts: list[str] = []
    for block in filtered:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and "text" in block:
            text_parts.append(str(block["text"]))
        else:
            return filtered

    if not text_parts:
        return filtered
    return " ".join(text_parts) if len(text_parts) > 1 else text_parts[0]


def normalize_content_to_text(content: Any) -> str | None:
    """Extract visible text from content, excluding thinking blocks.

    Handles plain strings, lists of content blocks (``TextBlock`` dataclasses
    or dicts), and ``None``.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "thinking":
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            elif type(block).__name__ == "TextBlock":
                text_parts.append(str(getattr(block, "text", "")))
            elif type(block).__name__ == "ThinkingBlock":
                continue
        return " ".join(text_parts) if text_parts else None

    return str(content)


def extract_token_usage(usage: Any) -> dict[str, int] | None:
    """Map Claude Agent SDK ``ResultMessage.usage`` to universal format.

    Input keys (from SDK):
      - ``input_tokens``
      - ``output_tokens``
      - ``cache_read_input_tokens``
      - ``cache_creation_input_tokens``

    Output keys (PandaProbe universal):
      - ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
      - ``cache_read_tokens``, ``reasoning_tokens``
    """
    if not usage:
        return None

    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)

    result: dict[str, int] = {}

    input_tokens = get("input_tokens")
    output_tokens = get("output_tokens")
    cache_read = get("cache_read_input_tokens")

    if input_tokens is not None:
        result["prompt_tokens"] = int(input_tokens)
    if output_tokens is not None:
        result["completion_tokens"] = int(output_tokens)
    if input_tokens is not None and output_tokens is not None:
        result["total_tokens"] = int(input_tokens) + int(output_tokens)
    if cache_read is not None and int(cache_read):
        result["cache_read_tokens"] = int(cache_read)

    return result if result else None


def extract_model_parameters(options: Any) -> dict[str, Any] | None:
    """Extract safe model parameters from ``ClaudeAgentOptions``.

    Filters through ``SAFE_MODEL_PARAM_KEYS`` and also captures
    ``thinking_config`` (the ``thinking`` field on options) if present.
    """
    if not options:
        return None

    config_dict = config_to_dict(options)
    params: dict[str, Any] = {}
    for key in SAFE_MODEL_PARAM_KEYS:
        if key == "thinking":
            continue
        val = config_dict.get(key)
        if val is not None:
            params[key] = safe_serialize(val)

    thinking = config_dict.get("thinking")
    if thinking:
        params["thinking_config"] = safe_serialize(thinking)

    return params if params else None


def serialize_tool_response(resp: Any) -> str:
    """Serialize a tool response to a string for span output.

    Uses explicit ``None`` check instead of truthiness to preserve
    ``0``, ``False``, and ``[]``.
    """
    if isinstance(resp, (dict, list)):
        return json.dumps(resp)
    if resp is None:
        return ""
    return str(resp)
