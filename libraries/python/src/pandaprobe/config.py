"""Configuration management for the PandaProbe SDK.

Reads from environment variables with ``PANDAPROBE_`` prefix, with constructor
kwargs taking precedence.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("pandaprobe")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    return float(val)


@dataclass(frozen=True)
class SdkConfig:
    """Immutable SDK configuration resolved from env vars + constructor kwargs."""

    api_key: str | None = None
    project_name: str | None = None
    endpoint: str = "http://localhost:8000"
    environment: str | None = None
    release: str | None = None
    enabled: bool = True
    batch_size: int = 10
    flush_interval: float = 5.0
    max_queue_size: int = 1000
    debug: bool = False

    def __post_init__(self) -> None:
        if self.enabled and not self.api_key:
            raise ValueError(
                "PandaProbe API key is required. Set PANDAPROBE_API_KEY or pass api_key= to Client / init()."
            )
        if self.enabled and not self.project_name:
            raise ValueError(
                "PandaProbe project name is required. "
                "Set PANDAPROBE_PROJECT_NAME or pass project_name= to Client / init()."
            )
        if self.debug:
            logging.getLogger("pandaprobe").setLevel(logging.DEBUG)


def resolve_config(
    *,
    api_key: str | None = None,
    project_name: str | None = None,
    endpoint: str | None = None,
    environment: str | None = None,
    release: str | None = None,
    enabled: bool | None = None,
    batch_size: int | None = None,
    flush_interval: float | None = None,
    max_queue_size: int | None = None,
    debug: bool | None = None,
) -> SdkConfig:
    """Build an :class:`SdkConfig` by merging constructor kwargs over env vars."""

    return SdkConfig(
        api_key=api_key if api_key is not None else _env("PANDAPROBE_API_KEY"),
        project_name=project_name if project_name is not None else _env("PANDAPROBE_PROJECT_NAME"),
        endpoint=endpoint if endpoint is not None else (_env("PANDAPROBE_ENDPOINT") or "http://localhost:8000"),
        environment=environment if environment is not None else _env("PANDAPROBE_ENVIRONMENT"),
        release=release if release is not None else _env("PANDAPROBE_RELEASE"),
        enabled=enabled if enabled is not None else _env_bool("PANDAPROBE_ENABLED", True),
        batch_size=batch_size if batch_size is not None else _env_int("PANDAPROBE_BATCH_SIZE", 10),
        flush_interval=flush_interval if flush_interval is not None else _env_float("PANDAPROBE_FLUSH_INTERVAL", 5.0),
        max_queue_size=max_queue_size if max_queue_size is not None else _env_int("PANDAPROBE_MAX_QUEUE_SIZE", 1000),
        debug=debug if debug is not None else _env_bool("PANDAPROBE_DEBUG", False),
    )
