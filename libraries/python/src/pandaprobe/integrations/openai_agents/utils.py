"""OpenAI Agents SDK-specific normalization and extraction utilities."""

from __future__ import annotations

import json
import logging
from typing import Any

from pandaprobe.integrations._base import SAFE_MODEL_PARAM_KEYS, config_to_dict, safe_serialize

logger = logging.getLogger("pandaprobe")


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def collapse_content(content: Any) -> Any:
    """Collapse single-element text content lists to plain strings.

    ``[{"type": "output_text", "text": "Hello"}]`` becomes ``"Hello"``.
    Multi-element or mixed-type lists are returned as-is.
    """
    if not isinstance(content, list):
        return content
    if not content:
        return content

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type in ("output_text", "text"):
                text = item.get("text")
                if text is not None:
                    text_parts.append(str(text))
                    continue
        return content

    if not text_parts:
        return content
    return " ".join(text_parts) if len(text_parts) > 1 else text_parts[0]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve a value from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _normalize_input_item(item: Any) -> dict[str, Any] | None:
    """Normalize a single input item to a message dict."""
    if isinstance(item, str):
        return {"role": "user", "content": item}

    if isinstance(item, dict):
        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            content = collapse_content(item.get("content"))
            return {"role": role, "content": content}

        if item_type == "function_call_output":
            output = item.get("output")
            content = json.dumps(output) if isinstance(output, (dict, list)) else ("" if output is None else str(output))
            return {
                "role": "tool",
                "content": content,
                "tool_call_id": item.get("call_id"),
            }

        if item_type == "item_reference":
            return None

        if item_type == "function_call":
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    }
                ],
            }

        if "role" in item and not item_type:
            content = item.get("content")
            if isinstance(content, list):
                content = collapse_content(content)
            result: dict[str, Any] = {"role": item["role"], "content": content}
            if "tool_calls" in item:
                result["tool_calls"] = item["tool_calls"]
            if "name" in item:
                result["name"] = item["name"]
            if "tool_call_id" in item:
                result["tool_call_id"] = item["tool_call_id"]
            return result

        return {"role": "user", "content": safe_serialize(item)}

    if hasattr(item, "model_dump"):
        try:
            return _normalize_input_item(item.model_dump(exclude_none=True, mode="json"))
        except Exception:
            pass

    return {"role": "user", "content": safe_serialize(item)}


def normalize_response_input(span_data: Any) -> dict[str, Any]:
    """Build ``{"messages": [...]}`` from ``ResponseSpanData``.

    Combines ``response.instructions`` (system message) with
    ``span_data.input`` (user/tool messages).
    """
    messages: list[dict[str, Any]] = []

    response = _get(span_data, "response")
    if response is not None:
        instructions = _get(response, "instructions")
        if instructions and isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

    raw_input = _get(span_data, "input")
    if raw_input is not None:
        if isinstance(raw_input, str):
            messages.append({"role": "user", "content": raw_input})
        elif isinstance(raw_input, list):
            for item in raw_input:
                msg = _normalize_input_item(item)
                if msg is not None:
                    messages.append(msg)
        else:
            msg = _normalize_input_item(raw_input)
            if msg is not None:
                messages.append(msg)

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------


def _normalize_output_item(item: Any) -> dict[str, Any] | None:
    """Normalize a single output item to a message dict.

    Reasoning items are skipped (handled separately by ``extract_reasoning``).
    """
    if isinstance(item, dict):
        item_type = item.get("type", "")

        if item_type == "reasoning":
            return None

        if item_type == "message":
            role = item.get("role", "assistant")
            content = collapse_content(item.get("content"))
            msg: dict[str, Any] = {"role": role, "content": content}
            return msg

        if item_type == "function_call":
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    }
                ],
            }

        return {"role": "assistant", "content": safe_serialize(item)}

    if hasattr(item, "model_dump"):
        try:
            return _normalize_output_item(item.model_dump(exclude_none=True, mode="json"))
        except Exception:
            pass

    return {"role": "assistant", "content": safe_serialize(item)}


def normalize_response_output(span_data: Any) -> dict[str, Any]:
    """Build ``{"messages": [...]}`` from ``ResponseSpanData.response.output``.

    Reasoning items are stripped from the output messages.
    """
    response = _get(span_data, "response")
    if response is None:
        return {"messages": []}

    output_items = _get(response, "output")
    if not isinstance(output_items, list):
        if hasattr(response, "model_dump"):
            try:
                response_dict = response.model_dump(exclude_none=True, mode="json")
                output_items = response_dict.get("output", [])
            except Exception:
                return {"messages": []}
        else:
            return {"messages": []}

    messages: list[dict[str, Any]] = []
    for item in output_items:
        msg = _normalize_output_item(item)
        if msg is not None:
            messages.append(msg)

    return {"messages": messages}


# ---------------------------------------------------------------------------
# GenerationSpanData normalization
# ---------------------------------------------------------------------------


def _normalize_generation_message(msg: Any) -> dict[str, Any]:
    """Normalize a single message from GenerationSpanData input/output."""
    if isinstance(msg, dict):
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, list):
            content = collapse_content(content)
        result: dict[str, Any] = {"role": role, "content": content}
        if "tool_calls" in msg:
            result["tool_calls"] = msg["tool_calls"]
        if "name" in msg:
            result["name"] = msg["name"]
        return result

    if hasattr(msg, "model_dump"):
        try:
            return _normalize_generation_message(msg.model_dump(exclude_none=True, mode="json"))
        except Exception:
            pass

    return {"role": "user", "content": safe_serialize(msg)}


def normalize_generation_input(span_data: Any) -> dict[str, Any]:
    """Normalize ``GenerationSpanData.input`` to ``{"messages": [...]}``."""
    raw = _get(span_data, "input")
    if not isinstance(raw, list):
        if isinstance(raw, str):
            return {"messages": [{"role": "user", "content": raw}]}
        return {"messages": []}

    messages: list[dict[str, Any]] = []
    for item in raw:
        messages.append(_normalize_generation_message(item))

    return {"messages": messages}


def normalize_generation_output(span_data: Any) -> dict[str, Any]:
    """Normalize ``GenerationSpanData.output`` to ``{"messages": [...]}``."""
    raw = _get(span_data, "output")
    if not isinstance(raw, list):
        if isinstance(raw, str):
            return {"messages": [{"role": "assistant", "content": raw}]}
        return {"messages": []}

    messages: list[dict[str, Any]] = []
    for item in raw:
        msg = _normalize_generation_message(item)
        if msg.get("role") not in ("assistant", "system", "tool"):
            msg["role"] = "assistant"
        messages.append(msg)

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Reasoning extraction
# ---------------------------------------------------------------------------


def extract_reasoning(response: Any) -> str | None:
    """Extract reasoning summary from ``response.output`` reasoning items.

    Looks for items with ``type == "reasoning"`` and extracts the
    ``summary`` list's ``summary_text`` parts.
    """
    output = _get(response, "output")
    if not isinstance(output, list):
        if hasattr(response, "model_dump"):
            try:
                response_dict = response.model_dump(exclude_none=True, mode="json")
                output = response_dict.get("output", [])
            except Exception:
                return None
        else:
            return None

    reasoning_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            item_dict = None
            if hasattr(item, "model_dump"):
                try:
                    item_dict = item.model_dump(exclude_none=True, mode="json")
                except Exception:
                    continue
            elif hasattr(item, "__dict__"):
                item_dict = {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
            if item_dict is None:
                continue
            item = item_dict

        if item.get("type") != "reasoning":
            continue

        summary = item.get("summary", [])
        if isinstance(summary, list):
            for part in summary:
                if isinstance(part, dict) and part.get("type") == "summary_text":
                    text = part.get("text")
                    if text:
                        reasoning_parts.append(str(text))

    return "\n\n".join(reasoning_parts) if reasoning_parts else None


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------


def _safe_int(val: Any) -> int | None:
    """Convert to int or return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def extract_token_usage(usage: Any) -> dict[str, int] | None:
    """Map Responses API / Chat Completions usage to universal format.

    Input keys (Responses API):
      - ``input_tokens`` -> ``prompt_tokens``
      - ``output_tokens`` -> ``completion_tokens``
      - ``total_tokens`` -> ``total_tokens``
      - ``input_tokens_details.cached_tokens`` -> ``cache_read_tokens``
      - ``output_tokens_details.reasoning_tokens`` -> ``reasoning_tokens``

    Also accepts already-mapped keys (``prompt_tokens``, etc.).
    """
    if not usage:
        return None

    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)

    result: dict[str, int] = {}

    prompt = _safe_int(get("input_tokens")) or _safe_int(get("prompt_tokens"))
    completion = _safe_int(get("output_tokens")) or _safe_int(get("completion_tokens"))
    total = _safe_int(get("total_tokens"))

    if prompt is not None:
        result["prompt_tokens"] = prompt
    if completion is not None:
        result["completion_tokens"] = completion
    if total is not None:
        result["total_tokens"] = total
    elif prompt is not None and completion is not None:
        result["total_tokens"] = prompt + completion

    input_details = get("input_tokens_details")
    if input_details:
        cached = _safe_int(_get(input_details, "cached_tokens"))
        if cached is not None and cached > 0:
            result["cache_read_tokens"] = cached

    output_details = get("output_tokens_details")
    if output_details:
        reasoning = _safe_int(_get(output_details, "reasoning_tokens"))
        if reasoning is not None and reasoning > 0:
            result["reasoning_tokens"] = reasoning

    cached_flat = _safe_int(get("cached_prompt_tokens")) or _safe_int(get("cache_read_tokens"))
    if cached_flat is not None and cached_flat > 0 and "cache_read_tokens" not in result:
        result["cache_read_tokens"] = cached_flat

    reasoning_flat = _safe_int(get("reasoning_tokens"))
    if reasoning_flat is not None and reasoning_flat > 0 and "reasoning_tokens" not in result:
        result["reasoning_tokens"] = reasoning_flat

    return result if result else None


# ---------------------------------------------------------------------------
# Model parameter extraction
# ---------------------------------------------------------------------------


def extract_response_model_parameters(response: Any) -> dict[str, Any] | None:
    """Extract safe model parameters from a Responses API ``Response`` object."""
    if response is None:
        return None

    config_dict = config_to_dict(response)
    params: dict[str, Any] = {}
    for key in SAFE_MODEL_PARAM_KEYS:
        val = config_dict.get(key)
        if val is not None:
            params[key] = safe_serialize(val)

    return params if params else None


def extract_generation_model_parameters(span_data: Any) -> dict[str, Any] | None:
    """Extract safe model parameters from ``GenerationSpanData.model_config``."""
    model_config = _get(span_data, "model_config")
    if not model_config:
        return None

    config_dict = config_to_dict(model_config)
    params: dict[str, Any] = {}
    for key in SAFE_MODEL_PARAM_KEYS:
        val = config_dict.get(key)
        if val is not None:
            params[key] = safe_serialize(val)

    return params if params else None


# ---------------------------------------------------------------------------
# Tool I/O serialization
# ---------------------------------------------------------------------------


def serialize_tool_io(value: Any) -> Any:
    """Serialize tool input or output.

    Uses explicit ``None`` check instead of truthiness to preserve
    ``0``, ``False``, and ``[]``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return safe_serialize(parsed)
        except (json.JSONDecodeError, TypeError):
            return value
    return safe_serialize(value)
