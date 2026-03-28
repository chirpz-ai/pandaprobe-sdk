"""CrewAI-specific normalization and extraction utilities."""

from __future__ import annotations

import logging
from typing import Any

from pandaprobe.integrations._base import SAFE_MODEL_PARAM_KEYS, config_to_dict, safe_serialize

logger = logging.getLogger("pandaprobe")

_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "model": "assistant",
    "function": "tool",
}


def _normalize_role(role: str) -> str:
    """Map CrewAI/provider-specific role names to standard roles."""
    return _ROLE_MAP.get(role, role)


def normalize_messages(messages: Any) -> dict[str, Any]:
    """Convert a CrewAI message list to the standard ``{"messages": [...]}`` schema.

    Each message dict gets its role normalized and thinking blocks stripped
    from content.  Text-only content lists are collapsed to plain strings.
    """
    if not isinstance(messages, list):
        return {"messages": []}

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg:
            new_msg = dict(msg)
            new_msg["role"] = _normalize_role(str(msg["role"]))
            if "content" in new_msg:
                new_msg["content"] = _normalize_content(new_msg["content"])
            normalized.append(new_msg)
        elif isinstance(msg, str):
            normalized.append({"role": "user", "content": msg})
        else:
            normalized.append(safe_serialize(msg))

    return {"messages": normalized}


def _normalize_content(content: Any) -> Any:
    """Normalize message content: strip thinking blocks, collapse text-only lists."""
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


def strip_thinking_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages with thinking blocks stripped from assistant content."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            new_msg = dict(msg)
            content = msg.get("content")
            new_msg["content"] = _normalize_content(content) if isinstance(content, list) else content
            result.append(new_msg)
        else:
            result.append(msg)
    return result


def extract_reasoning_from_messages(messages: Any) -> str | None:
    """Extract thinking/reasoning text from the last assistant message.

    Looks for dicts with ``"type": "thinking"`` inside the content list
    of the last assistant message.
    """
    if not isinstance(messages, list):
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") not in ("assistant", "model", "ai"):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        thinking_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = block.get("thinking") or block.get("text")
                if text:
                    thinking_parts.append(str(text))
        return "\n\n".join(thinking_parts) if thinking_parts else None

    return None


def extract_token_usage(source: Any) -> dict[str, int] | None:
    """Extract token usage from a CrewAI usage source.

    Accepts:
      - A dict with token usage keys (e.g. from ``CrewOutput.token_usage``)
      - An LLM instance with a ``_token_usage`` dict attribute
      - Any object with standard token count attributes

    Maps CrewAI field names to PandaProbe's flat ``dict[str, int]`` standard:
      - prompt_tokens         -> prompt_tokens
      - completion_tokens     -> completion_tokens
      - total_tokens          -> total_tokens
      - cached_prompt_tokens  -> cache_read_tokens
      - reasoning_tokens      -> reasoning_tokens
    """
    usage = getattr(source, "_token_usage", None) or source

    if not usage:
        return None

    if isinstance(usage, dict):
        get = usage.get
    else:
        get = lambda k: getattr(usage, k, None)  # noqa: E731

    result: dict[str, int] = {}

    prompt = _safe_int(get("prompt_tokens"))
    completion = _safe_int(get("completion_tokens"))
    total = _safe_int(get("total_tokens"))
    cached = _safe_int(get("cached_prompt_tokens"))
    reasoning = _safe_int(get("reasoning_tokens"))

    if prompt is not None:
        result["prompt_tokens"] = prompt
    if completion is not None:
        result["completion_tokens"] = completion
    if total is not None:
        result["total_tokens"] = total
    elif prompt is not None and completion is not None:
        result["total_tokens"] = prompt + completion
    if cached is not None and cached > 0:
        result["cache_read_tokens"] = cached
    if reasoning is not None and reasoning > 0:
        result["reasoning_tokens"] = reasoning

    return result if result else None


def _safe_int(val: Any) -> int | None:
    """Convert to int or return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def extract_model_parameters(llm: Any) -> dict[str, Any] | None:
    """Extract safe model parameters from a CrewAI ``LLM`` instance.

    Reads instance attributes and filters through ``SAFE_MODEL_PARAM_KEYS``.
    """
    if llm is None:
        return None

    config_dict = config_to_dict(llm)
    params: dict[str, Any] = {}
    for key in SAFE_MODEL_PARAM_KEYS:
        val = config_dict.get(key)
        if val is not None:
            params[key] = safe_serialize(val)

    thinking = config_dict.get("thinking_config")
    if thinking:
        params["thinking_config"] = safe_serialize(thinking)

    return params if params else None


def extract_model_name(llm: Any) -> str | None:
    """Extract the model name from a CrewAI ``LLM`` instance or agent."""
    if llm is None:
        return None
    model = getattr(llm, "model", None)
    if model:
        return str(model)
    return None


def build_agent_system_message(agent: Any) -> str | None:
    """Build a synthetic system-prompt string from an agent's role, goal, and backstory."""
    parts: list[str] = []
    role = getattr(agent, "role", None)
    goal = getattr(agent, "goal", None)
    backstory = getattr(agent, "backstory", None)

    if role:
        parts.append(f"Role: {role}")
    if goal:
        parts.append(f"Goal: {goal}")
    if backstory:
        parts.append(f"Backstory: {backstory}")

    return "\n".join(parts) if parts else None


def build_crew_system_message(crew: Any) -> str | None:
    """Build a synthetic system message describing the crew configuration."""
    agents = getattr(crew, "agents", None) or []
    if not agents:
        return None

    agent_summaries: list[str] = []
    for agent in agents:
        role = getattr(agent, "role", None) or "unnamed"
        goal = getattr(agent, "goal", None)
        summary = f"- {role}"
        if goal:
            summary += f" (goal: {goal})"
        agent_summaries.append(summary)

    return "Crew agents:\n" + "\n".join(agent_summaries)
