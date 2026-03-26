"""Base adapter protocol shared by all framework integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandaprobe.client import Client

# ---------------------------------------------------------------------------
# Shared model-parameter utilities
# ---------------------------------------------------------------------------

SAFE_MODEL_PARAM_KEYS: set[str] = {
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "n",
    "candidate_count",
    "max_tokens",
    "max_output_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "stop_sequences",
    "response_format",
    "response_modalities",
    "response_mime_type",
    "reasoning_effort",
    "reasoning",
    "thinking",
    "thinking_level",
    "thinking_budget",
    "top_logprobs",
    "stream_options",
    "service_tier",
    "truncation",
}


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a config object or plain dict to a dict, dropping ``None`` values.

    Handles Pydantic models (``model_dump``), plain dicts, and objects with
    ``__dict__``.
    """
    if isinstance(config, dict):
        return {k: v for k, v in config.items() if v is not None}
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


# ---------------------------------------------------------------------------
# Base adapter class
# ---------------------------------------------------------------------------


class BaseIntegrationAdapter:
    """Common foundation for framework integration adapters.

    Subclasses implement framework-specific hooking logic but share the same
    trace/span lifecycle management via the Tracing Core layer.
    """

    def __init__(
        self,
        *,
        client: Client | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._user_id = user_id
        self._tags = tags or []
        self._metadata = metadata or {}

    def _resolve_client(self) -> Client:
        """Return the explicit client or fall back to the global singleton."""
        if self._client is not None:
            return self._client
        from pandaprobe.client import get_client

        client = get_client()
        if client is None:
            raise RuntimeError(
                "No PandaProbe client available. "
                "Set PANDAPROBE_API_KEY and PANDAPROBE_PROJECT_NAME environment variables, "
                "or call pandaprobe.init() first, or pass client= to the adapter."
            )
        return client
