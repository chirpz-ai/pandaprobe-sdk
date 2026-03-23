"""LangGraph-specific helper utilities."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("pandaprobe")


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
            normalized.append({"role": item[0], "content": item[1]})
        elif isinstance(item, dict) and "type" in item and "content" in item:
            new_item = {k: v for k, v in item.items() if k != "type"}
            new_item["role"] = item["type"]
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
    if isinstance(last_item, dict) and "type" in last_item:
        last_item = {k: v for k, v in last_item.items() if k != "type"}
        last_item["role"] = messages[-1]["type"]

    return {"messages": [last_item]}


def normalize_type_to_role(data: Any) -> Any:
    """Recursively rename ``type`` to ``role`` in dicts that also have ``content``."""
    if isinstance(data, dict):
        if "type" in data and "content" in data:
            new_dict: dict[str, Any] = {}
            for k, v in data.items():
                if k == "type":
                    new_dict["role"] = normalize_type_to_role(v)
                else:
                    new_dict[k] = normalize_type_to_role(v)
            return new_dict
        return {k: normalize_type_to_role(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize_type_to_role(item) for item in data]
    return data


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
