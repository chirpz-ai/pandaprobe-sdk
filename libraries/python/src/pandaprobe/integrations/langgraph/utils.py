"""LangGraph-specific helper utilities."""

from __future__ import annotations

from typing import Any


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
