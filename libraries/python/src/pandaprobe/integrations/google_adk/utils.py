"""Google ADK-specific normalization and extraction utilities."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

logger = logging.getLogger("pandaprobe")


def safe_serialize(obj: Any) -> Any:
    """Best-effort JSON-safe serialization of an arbitrary object.

    Handles ADK-specific types (Pydantic models, proto-like objects).
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


def extract_text_from_content(content: Any) -> str | None:
    """Extract plain text from an ADK Content object, skipping thinking parts."""
    if content is None:
        return None
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    text_parts = []
    for p in parts:
        if getattr(p, "thought", False):
            continue
        text = getattr(p, "text", None)
        if text:
            text_parts.append(str(text))
    return " ".join(text_parts) if text_parts else None


def extract_thinking_from_content(content: Any) -> str | None:
    """Extract thinking/reasoning text from an ADK Content object."""
    if content is None:
        return None
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    thinking_parts = []
    for p in parts:
        if getattr(p, "thought", False):
            text = getattr(p, "text", None)
            if text:
                thinking_parts.append(str(text))
    return "\n\n".join(thinking_parts) if thinking_parts else None


def _serialize_part(part: Any) -> dict[str, Any]:
    """Serialize a single ADK Part into a dict."""
    if isinstance(part, dict):
        return part

    if hasattr(part, "inline_data") and part.inline_data:
        data = getattr(part.inline_data, "data", None)
        mime_type = getattr(part.inline_data, "mime_type", "application/octet-stream")
        if data is not None:
            encoded = base64.b64encode(data).decode("utf-8") if isinstance(data, bytes) else str(data)
            return {"type": "image", "data": encoded, "mime_type": mime_type}

    if hasattr(part, "file_data") and part.file_data:
        return {
            "type": "file",
            "file_uri": getattr(part.file_data, "file_uri", None),
            "mime_type": getattr(part.file_data, "mime_type", None),
        }

    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call
        return {
            "type": "function_call",
            "name": getattr(fc, "name", "unknown"),
            "arguments": safe_serialize(dict(getattr(fc, "args", None) or {})),
        }

    if hasattr(part, "function_response") and part.function_response:
        fr = part.function_response
        return {
            "type": "function_response",
            "name": getattr(fr, "name", "unknown"),
            "response": safe_serialize(getattr(fr, "response", None)),
        }

    text = getattr(part, "text", None)
    if text is not None:
        if getattr(part, "thought", False):
            return {"type": "thinking", "thinking": str(text)}
        return {"type": "text", "text": str(text)}

    if hasattr(part, "executable_code") and part.executable_code:
        code = part.executable_code
        return {
            "type": "executable_code",
            "language": getattr(code, "language", "python"),
            "code": getattr(code, "code", ""),
        }

    if hasattr(part, "code_execution_result") and part.code_execution_result:
        result = part.code_execution_result
        return {
            "type": "code_execution_result",
            "outcome": getattr(result, "outcome", "unknown"),
            "output": getattr(result, "output", ""),
        }

    return safe_serialize(part)


def _content_to_message(content: Any) -> dict[str, Any]:
    """Convert a single ADK Content object to a standard message dict.

    Maps ``role="model"`` to ``role="assistant"`` and separates text from
    tool_calls.
    """
    role = getattr(content, "role", "user") or "user"
    if role == "model":
        role = "assistant"

    parts = getattr(content, "parts", None) or []
    serialized = [_serialize_part(p) for p in parts]

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    other_parts: list[dict[str, Any]] = []

    for sp in serialized:
        ptype = sp.get("type")
        if ptype == "text":
            text_parts.append(sp.get("text", ""))
        elif ptype == "thinking":
            pass
        elif ptype == "function_call":
            tool_calls.append(sp)
        elif ptype == "function_response":
            tool_results.append(sp)
        else:
            other_parts.append(sp)

    if tool_results:
        messages: list[dict[str, Any]] = []
        for tr in tool_results:
            resp = tr.get("response")
            messages.append(
                {
                    "role": "tool",
                    "name": tr.get("name", ""),
                    "content": json.dumps(resp) if isinstance(resp, dict) else str(resp or ""),
                }
            )
        return messages  # type: ignore[return-value]

    msg: dict[str, Any] = {"role": role}
    content_text = " ".join(text_parts) if text_parts else None
    if other_parts and not content_text:
        msg["content"] = other_parts
    else:
        msg["content"] = content_text

    if tool_calls:
        msg["tool_calls"] = [
            {
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", {}),
            }
            for tc in tool_calls
        ]

    return msg


def normalize_contents_to_messages(contents: Any, system_instruction: Any = None) -> dict[str, Any]:
    """Convert ADK's ``list[Content]`` to the standard messages schema.

    Optionally prepends a system message extracted from config.system_instruction.
    Returns ``{"messages": [...]}``.
    """
    messages: list[dict[str, Any]] = []

    if system_instruction is not None:
        sys_text = extract_text_from_content(system_instruction)
        if sys_text is None and isinstance(system_instruction, str):
            sys_text = system_instruction
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    if contents:
        for content_item in contents:
            converted = _content_to_message(content_item)
            if isinstance(converted, list):
                messages.extend(converted)
            else:
                messages.append(converted)

    return {"messages": messages}


def normalize_llm_response_to_messages(content: Any) -> dict[str, Any]:
    """Convert a single ADK Content response to the standard output messages schema.

    Returns ``{"messages": [message_dict]}``.
    """
    if content is None:
        return {"messages": [{"role": "assistant", "content": None}]}

    msg = _content_to_message(content)
    if isinstance(msg, list):
        return {"messages": msg}
    return {"messages": [msg]}


def extract_token_usage(event_or_response: Any) -> dict[str, int] | None:
    """Extract token usage from an ADK event or LlmResponse.

    Maps ADK field names to PandaProbe's flat ``dict[str, int]`` standard
    (matching the Gemini / OpenAI / Anthropic wrappers):
      - prompt_token_count         -> prompt_tokens
      - candidates_token_count     -> completion_tokens
      - total_token_count          -> total_tokens
      - cached_content_token_count -> cache_read_tokens
      - thoughts_token_count       -> reasoning_tokens
    """
    usage_metadata = getattr(event_or_response, "usage_metadata", None)
    if not usage_metadata:
        return None

    usage: dict[str, int] = {}
    if (v := getattr(usage_metadata, "prompt_token_count", None)) is not None:
        usage["prompt_tokens"] = int(v)
    if (v := getattr(usage_metadata, "candidates_token_count", None)) is not None:
        usage["completion_tokens"] = int(v)
    if (v := getattr(usage_metadata, "total_token_count", None)) is not None:
        usage["total_tokens"] = int(v)
    if (v := getattr(usage_metadata, "cached_content_token_count", None)) is not None:
        usage["cache_read_tokens"] = int(v)
    if (v := getattr(usage_metadata, "thoughts_token_count", None)) is not None:
        usage["reasoning_tokens"] = int(v)

    return usage if usage else None


def extract_model_name(llm_request: Any) -> str | None:
    """Extract the model name from an ADK LlmRequest."""
    config = getattr(llm_request, "config", None)
    if config:
        model = getattr(config, "model", None)
        if model:
            return str(model)
    model = getattr(llm_request, "model", None)
    if model:
        return str(model)
    return None


_SAFE_MODEL_PARAM_KEYS: set[str] = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_output_tokens",
    "max_completion_tokens",
    "stop",
    "stop_sequences",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "response_format",
    "reasoning_effort",
    "thinking_level",
    "thinking_budget",
    "response_modalities",
    "response_mime_type",
}


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a GenerateContentConfig to a plain dict, dropping None values."""
    try:
        if hasattr(config, "model_dump"):
            return config.model_dump(exclude_none=True)
    except Exception:
        pass
    try:
        if hasattr(config, "__dict__"):
            return {k: v for k, v in vars(config).items() if not k.startswith("_") and v is not None}
    except Exception:
        pass
    return {}


def extract_model_parameters(llm_request: Any) -> dict[str, Any] | None:
    """Extract safe model parameters from an ADK LlmRequest config."""
    config = getattr(llm_request, "config", None)
    if not config:
        return None

    config_dict = _config_to_dict(config)
    params: dict[str, Any] = {}
    for key in _SAFE_MODEL_PARAM_KEYS:
        val = config_dict.get(key)
        if val is not None:
            params[key] = safe_serialize(val)

    thinking = config_dict.get("thinking_config")
    if thinking:
        params["thinking_config"] = safe_serialize(thinking)

    return params if params else None
