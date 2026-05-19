"""AWS Bedrock-specific utilities for the wrap_bedrock instrumentation."""

from __future__ import annotations

import json
import logging
from typing import Any

from pandaprobe.client import get_client
from pandaprobe.schemas import SpanKind
from pandaprobe.tracing.context import get_current_trace
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message
from pandaprobe.wrappers._base import safe_serialize

logger = logging.getLogger("pandaprobe")


# ``inferenceConfig`` keys live nested under that dict in the Converse API,
# but we promote them to span model_parameters as flat keys.  The remaining
# keys live at the top level of the call.
SAFE_BEDROCK_PARAMS: set[str] = {
    "temperature",
    "topP",
    "maxTokens",
    "stopSequences",
    "guardrailConfig",
    "additionalModelRequestFields",
    "toolConfig",
}


# ---------------------------------------------------------------------------
# Role + message normalisation
# ---------------------------------------------------------------------------


def _normalize_role(role: str) -> str:
    """Map Bedrock-Converse roles onto the canonical PandaProbe role set."""
    if role == "model":
        return "assistant"
    return role


def _flatten_content_blocks(content: Any) -> Any:
    """Reduce a list of Converse content blocks into a single string when possible.

    Converse messages have ``content`` as a list of typed blocks
    (``{"text": "..."}``, ``{"image": {...}}``, ``{"toolUse": {...}}``,
    ``{"toolResult": {...}}``, ``{"reasoningContent": {...}}``).  When every
    block is a ``{"text": ...}`` block we collapse them into a single string
    for readability; otherwise the structured list is preserved via
    :func:`safe_serialize`.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return safe_serialize(content)

    text_only: list[str] = []
    all_text = True
    for block in content:
        if isinstance(block, dict) and set(block.keys()) == {"text"} and isinstance(block["text"], str):
            text_only.append(block["text"])
        else:
            all_text = False
            break

    if all_text:
        return "".join(text_only)
    return safe_serialize(content)


def _flatten_system_blocks(system: Any) -> str | Any:
    """Flatten Converse ``system`` blocks (a list of ``{"text": "..."}`` dicts)."""
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        return safe_serialize(system)

    text_only: list[str] = []
    all_text = True
    for block in system:
        if isinstance(block, dict) and "text" in block and isinstance(block["text"], str):
            text_only.append(block["text"])
        else:
            all_text = False
            break

    if all_text:
        return "\n".join(text_only)
    return safe_serialize(system)


def normalize_converse_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert Converse API ``system`` + ``messages`` into the universal schema.

    Hoists the top-level ``system`` argument (a list of blocks) to the front
    of the messages list as a ``role="system"`` entry, normalises Converse
    roles, and flattens text-only content blocks into a single string.
    """
    messages: list[dict[str, Any]] = []

    system = cleaned_kwargs.get("system")
    if system is not None:
        flattened = _flatten_system_blocks(system)
        messages.append({"role": "system", "content": flattened})

    raw_messages = cleaned_kwargs.get("messages") or []
    if isinstance(raw_messages, list):
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = _normalize_role(msg.get("role", "user"))
            content = _flatten_content_blocks(msg.get("content"))
            messages.append({"role": role, "content": content})

    return {"messages": messages}


def normalize_invoke_model_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Best-effort universal-schema normalisation for InvokeModel calls.

    The body shape varies per provider — Anthropic/Mistral on Bedrock use
    ``{"messages": [...]}`` (and Anthropic adds a separate ``system``), while
    Llama / Titan models use ``{"prompt": "..."}`` or ``{"inputText": "..."}``.
    This helper does a best-effort translation; unknown shapes fall through
    as a single user message holding the serialized body.
    """
    body = cleaned_kwargs.get("body")
    if body is None:
        return {"messages": []}

    parsed: Any
    if isinstance(body, (bytes, bytearray)):
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            parsed = body.decode("utf-8", errors="replace")
    elif isinstance(body, str):
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
    else:
        parsed = body

    if isinstance(parsed, str):
        return {"messages": [{"role": "user", "content": parsed}]}

    if isinstance(parsed, dict):
        messages: list[dict[str, Any]] = []

        system = parsed.get("system")
        if system:
            messages.append({"role": "system", "content": _flatten_system_blocks(system)})

        if isinstance(parsed.get("messages"), list):
            for msg in parsed["messages"]:
                if not isinstance(msg, dict):
                    continue
                role = _normalize_role(msg.get("role", "user"))
                content = _flatten_content_blocks(msg.get("content"))
                messages.append({"role": role, "content": content})
            return {"messages": messages}

        for key in ("prompt", "inputText", "input"):
            value = parsed.get(key)
            if isinstance(value, str):
                messages.append({"role": "user", "content": value})
                return {"messages": messages}

        messages.append({"role": "user", "content": safe_serialize(parsed)})
        return {"messages": messages}

    return {"messages": [{"role": "user", "content": safe_serialize(parsed)}]}


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


def extract_bedrock_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull safe Converse parameters from kwargs and the nested ``inferenceConfig`` dict."""
    params: dict[str, Any] = {}

    inference_config = kwargs.get("inferenceConfig")
    if isinstance(inference_config, dict):
        for k, v in inference_config.items():
            if k in SAFE_BEDROCK_PARAMS:
                params[k] = safe_serialize(v)

    for k, v in kwargs.items():
        if k in SAFE_BEDROCK_PARAMS:
            params[k] = safe_serialize(v)

    return params


# ---------------------------------------------------------------------------
# Span entry — supports Converse and InvokeModel call shapes
# ---------------------------------------------------------------------------


def enter_bedrock_span(
    cleaned_kwargs: dict[str, Any],
    method_name: str = "bedrock-converse",
    *,
    api: str = "converse",
):
    """Open an LLM span for an AWS Bedrock API call.

    ``api`` selects the input-normalisation strategy:

    * ``"converse"`` — Converse / ConverseStream API (system + messages).
    * ``"invoke_model"`` — legacy InvokeModel API (provider-specific JSON body).
    """
    if api == "invoke_model":
        input_data = normalize_invoke_model_input(cleaned_kwargs)
    else:
        input_data = normalize_converse_input(cleaned_kwargs)

    model_params = extract_bedrock_params(cleaned_kwargs)
    model_id = cleaned_kwargs.get("modelId")
    trace_ctx = get_current_trace()

    if trace_ctx is not None:
        span_ctx = trace_ctx.span(method_name, kind=SpanKind.LLM, model=model_id)
        span_ctx.__enter__()
        span_ctx.set_input(input_data)
        if model_params:
            span_ctx.set_model_parameters(model_params)
        return span_ctx

    client = get_client()
    if client is None or not client.enabled:
        return None

    standalone = client.trace(
        method_name,
        input=extract_last_user_message(input_data),
        session_id=get_current_session_id(),
        user_id=get_current_user_id(),
    )
    standalone.__enter__()

    span_ctx = standalone.span(method_name, kind=SpanKind.LLM, model=model_id)
    span_ctx.__enter__()
    span_ctx.set_input(input_data)
    if model_params:
        span_ctx.set_model_parameters(model_params)
    span_ctx._standalone_trace = standalone  # type: ignore[attr-defined]
    return span_ctx


# ---------------------------------------------------------------------------
# Token usage mapping (Converse + InvokeModel responses)
# ---------------------------------------------------------------------------


def map_converse_usage(usage: Any) -> dict[str, int]:
    """Map Converse usage (``inputTokens`` / ``outputTokens`` / ``totalTokens``)
    onto canonical PandaProbe names.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    prompt = usage.get("inputTokens") or 0
    completion = usage.get("outputTokens") or 0
    total = usage.get("totalTokens") or 0
    if isinstance(prompt, int) and prompt > 0:
        out["prompt_tokens"] = prompt
    if isinstance(completion, int) and completion > 0:
        out["completion_tokens"] = completion
    if isinstance(total, int) and total > 0:
        out["total_tokens"] = total
    cached = usage.get("cacheReadInputTokens") or 0
    if isinstance(cached, int) and cached > 0:
        out["cache_read_tokens"] = cached
    cache_write = usage.get("cacheWriteInputTokens") or 0
    if isinstance(cache_write, int) and cache_write > 0:
        out["cache_creation_tokens"] = cache_write
    return out


def map_invoke_model_usage(parsed_body: Any) -> dict[str, int]:
    """Best-effort token-usage extraction from an InvokeModel response body.

    Handles the most common shapes:

    * Anthropic-on-Bedrock: ``{"usage": {"input_tokens": .., "output_tokens": ..}}``
    * Mistral-on-Bedrock: ``{"usage": {"prompt_tokens": .., "completion_tokens": ..}}``
    * Amazon Titan: ``{"inputTextTokenCount": .., "results": [{"tokenCount": ..}]}``
    * Cohere: ``{"meta": {"billed_units": {"input_tokens": .., "output_tokens": ..}}}``
    """
    if not isinstance(parsed_body, dict):
        return {}

    out: dict[str, int] = {}
    usage = parsed_body.get("usage")
    if isinstance(usage, dict):
        for src, dst in (
            ("input_tokens", "prompt_tokens"),
            ("prompt_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            v = usage.get(src)
            if isinstance(v, int) and v > 0:
                out.setdefault(dst, v)
        for src, dst in (
            ("cache_read_input_tokens", "cache_read_tokens"),
            ("cache_creation_input_tokens", "cache_creation_tokens"),
        ):
            v = usage.get(src)
            if isinstance(v, int) and v > 0:
                out[dst] = v
        if out:
            return out

    titan_input = parsed_body.get("inputTextTokenCount")
    if isinstance(titan_input, int) and titan_input > 0:
        out["prompt_tokens"] = titan_input
        results = parsed_body.get("results")
        if isinstance(results, list) and results:
            tc = results[0].get("tokenCount") if isinstance(results[0], dict) else None
            if isinstance(tc, int) and tc > 0:
                out["completion_tokens"] = tc
        return out

    meta = parsed_body.get("meta")
    if isinstance(meta, dict):
        billed = meta.get("billed_units")
        if isinstance(billed, dict):
            inp = billed.get("input_tokens")
            outp = billed.get("output_tokens")
            if isinstance(inp, int) and inp > 0:
                out["prompt_tokens"] = inp
            if isinstance(outp, int) and outp > 0:
                out["completion_tokens"] = outp
    return out
