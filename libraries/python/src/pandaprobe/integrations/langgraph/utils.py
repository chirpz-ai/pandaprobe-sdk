"""LangGraph-specific helper utilities."""

from __future__ import annotations

import logging
from typing import Any

from pandaprobe.integrations._base import SAFE_MODEL_PARAM_KEYS, config_to_dict

logger = logging.getLogger("pandaprobe")

_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
}


def _normalize_role(role: str) -> str:
    """Map LangChain role names to the OpenAI-style equivalents."""
    return _ROLE_MAP.get(role, role)


def _normalize_content_blocks(content: Any) -> Any:
    """Normalize provider-specific content block lists to a universal format.

    Different LLM providers return message content in different formats when
    thinking/reasoning is enabled:

    * **Gemini** (``langchain-google-genai``): ``[{"type": "thinking", ...},
      {"type": "text", "text": "...", "extras": {"signature": "..."}}]``
    * **Anthropic** (``langchain-anthropic``): ``[{"type": "thinking",
      "thinking": "..."}, {"type": "text", "text": "..."}]``
    * **OpenAI**: plain string (no block list)

    This function:
    1. Strips ``{"type": "thinking", ...}`` blocks (reasoning is captured
       separately via ``extract_reasoning_from_generation``).
    2. If the remaining blocks are all text, collapses them to a plain string,
       discarding provider metadata (e.g. Gemini's ``extras.signature``).
    3. Passes through string content and mixed-type lists unchanged.
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


def extract_model_parameters(invocation_params: Any) -> dict[str, Any] | None:
    """Extract whitelisted model parameters, dropping ``None`` values and secrets.

    Converts the input (dict or config object) via ``config_to_dict``, then
    filters by ``SAFE_MODEL_PARAM_KEYS``.  Also captures ``thinking_config``
    if present.
    """
    if not invocation_params:
        return None
    config_dict = config_to_dict(invocation_params)
    params: dict[str, Any] = {}
    for key in SAFE_MODEL_PARAM_KEYS:
        val = config_dict.get(key)
        if val is not None:
            params[key] = val
    thinking = config_dict.get("thinking_config")
    if thinking:
        params["thinking_config"] = thinking
    return params if params else None


def extract_token_usage(response: Any) -> dict[str, int] | None:
    """Extract token usage from LangChain's ``usage_metadata`` and normalise
    to PandaProbe's flat ``dict[str, int]`` standard (matching the ADK,
    Gemini, OpenAI, and Anthropic wrappers):

      - input_tokens                        -> prompt_tokens
      - output_tokens - reasoning            -> completion_tokens
      - total_tokens                         -> total_tokens
      - output_token_details.reasoning       -> reasoning_tokens
      - input_token_details.cache_read       -> cache_read_tokens
    """
    try:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is None:
            return None

        meta = getattr(msg, "usage_metadata", None)
        if not meta:
            return None

        md = config_to_dict(meta)
        if not md:
            return None

        usage: dict[str, int] = {}

        if (v := md.get("input_tokens")) is not None:
            usage["prompt_tokens"] = int(v)
        if (v := md.get("total_tokens")) is not None:
            usage["total_tokens"] = int(v)

        output_tokens = md.get("output_tokens")
        output_details = md.get("output_token_details")
        reasoning = (
            int(output_details["reasoning"])
            if isinstance(output_details, dict) and output_details.get("reasoning") is not None
            else 0
        )
        if reasoning:
            usage["reasoning_tokens"] = reasoning
        if output_tokens is not None:
            usage["completion_tokens"] = max(0, int(output_tokens) - reasoning)

        input_details = md.get("input_token_details")
        if isinstance(input_details, dict) and input_details.get("cache_read") is not None:
            cache_val = int(input_details["cache_read"])
            if cache_val:
                usage["cache_read_tokens"] = cache_val

        return usage if usage else None
    except Exception:
        return None


def extract_name(serialized: dict[str, Any] | None, fallback: str = "unknown") -> str:
    """Extract a human-readable name from a LangChain serialized dict."""
    if not serialized:
        return fallback
    if "name" in serialized and serialized["name"]:
        return str(serialized["name"])
    if "id" in serialized:
        ids = serialized["id"]
        if isinstance(ids, list) and ids:
            return str(ids[-1])
    return fallback


def safe_output(value: Any) -> Any:
    """Ensure a callback output is JSON-serialisable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_output(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_output(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "page_content"):
        return {"page_content": value.page_content, "metadata": getattr(value, "metadata", {})}
    return repr(value)


def normalize_langchain_input(inputs: Any) -> Any:
    """Convert LangChain input formats to standard ``role``/``content`` dicts.

    Handles list-of-tuples (``[("human", "hi")]``) and dicts with ``type``
    instead of ``role``.  Returns *inputs* unchanged if the structure doesn't
    match.
    """
    if not isinstance(inputs, dict) or "messages" not in inputs:
        return inputs

    messages = inputs["messages"]
    if not isinstance(messages, list):
        return inputs

    normalized: list[Any] = []
    for item in messages:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append(
                {
                    "role": _normalize_role(str(item[0])),
                    "content": _normalize_content_blocks(item[1]),
                }
            )
        elif isinstance(item, dict) and "type" in item and "content" in item:
            new_item = {k: v for k, v in item.items() if k != "type"}
            new_item["role"] = _normalize_role(str(item["type"]))
            new_item["content"] = _normalize_content_blocks(new_item["content"])
            normalized.append(new_item)
        elif isinstance(item, dict) and "role" in item and isinstance(item["role"], str):
            new_item = dict(item)
            new_item["role"] = _normalize_role(item["role"])
            if "content" in new_item:
                new_item["content"] = _normalize_content_blocks(new_item["content"])
            normalized.append(new_item)
        else:
            normalized.append(item)

    result = {k: v for k, v in inputs.items() if k != "messages"}
    result["messages"] = normalized
    return result


def normalize_langchain_output(outputs: Any) -> Any:
    """Extract only the last message from outputs, renaming ``type`` to ``role``.

    Returns ``{"messages": [last_item]}`` or *outputs* unchanged if the
    structure doesn't match.
    """
    if not isinstance(outputs, dict) or "messages" not in outputs:
        return outputs

    messages = outputs["messages"]
    if not isinstance(messages, list) or not messages:
        return outputs

    last_item = messages[-1]
    if isinstance(last_item, dict):
        if "type" in last_item:
            last_item = {k: v for k, v in last_item.items() if k != "type"}
            last_item["role"] = _normalize_role(str(messages[-1]["type"]))
        elif "role" in last_item and isinstance(last_item["role"], str):
            last_item = dict(last_item)
            last_item["role"] = _normalize_role(last_item["role"])
        if "content" in last_item:
            last_item["content"] = _normalize_content_blocks(last_item["content"])

    return {"messages": [last_item]}


def normalize_type_to_role(data: Any) -> Any:
    """Recursively rename ``type`` to ``role``, normalize roles, and strip thinking blocks."""
    if isinstance(data, dict):
        has_content = "content" in data
        if "type" in data and has_content:
            new_dict: dict[str, Any] = {}
            for k, v in data.items():
                if k == "type":
                    new_dict["role"] = _normalize_role(str(v))
                elif k == "content":
                    new_dict[k] = _normalize_content_blocks(normalize_type_to_role(v))
                else:
                    new_dict[k] = normalize_type_to_role(v)
            return new_dict
        if "role" in data and has_content and isinstance(data["role"], str):
            return {
                k: (
                    _normalize_role(v)
                    if k == "role"
                    else _normalize_content_blocks(normalize_type_to_role(v))
                    if k == "content"
                    else normalize_type_to_role(v)
                )
                for k, v in data.items()
            }
        return {k: normalize_type_to_role(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize_type_to_role(item) for item in data]
    return data


def extract_reasoning_from_generation(response: Any) -> str | None:
    """Extract reasoning/thinking text from a LangChain LLM response.

    Handles two provider patterns:
      - **Anthropic**: ``message.content`` is a list of blocks; those with
        ``type == "thinking"`` contain the reasoning text.
      - **OpenAI**: reasoning may appear in ``message.additional_kwargs``
        under ``reasoning_content`` or ``reasoning``.
    """
    if not hasattr(response, "generations") or not response.generations:
        return None
    try:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is None:
            return None

        content = getattr(msg, "content", None)
        if isinstance(content, list):
            thinking_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    text = block.get("thinking") or block.get("text")
                    if text:
                        thinking_parts.append(str(text))
            if thinking_parts:
                return "\n\n".join(thinking_parts)

        additional = getattr(msg, "additional_kwargs", None) or {}
        reasoning = additional.get("reasoning_content") or additional.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            return reasoning
    except Exception:
        pass
    return None


def normalize_llm_generation_output(response: Any) -> Any | None:
    """Extract the actual message from LangChain's ``response.generations``.

    Returns ``{"messages": [message_dict]}`` or ``None`` if generations are
    empty/missing.
    """
    if not hasattr(response, "generations") or not response.generations:
        return None

    try:
        gen = response.generations[0][0]
        if hasattr(gen, "message"):
            serialized = safe_output(gen.message)
            normalized = normalize_type_to_role(serialized)
            return {"messages": [normalized]}
        return {"messages": [{"role": "assistant", "content": gen.text}]}
    except Exception:
        logger.debug("Error normalizing LLM generation output, falling back")
        return safe_output(response.generations)
