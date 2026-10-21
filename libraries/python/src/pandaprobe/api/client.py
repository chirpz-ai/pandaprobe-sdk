"""Native asynchronous platform API with safe reads and single-attempt creates."""

from __future__ import annotations

import asyncio
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import Self

from pandaprobe.client import Client, get_client
from pandaprobe.config import SdkConfig, resolve_config

from .errors import (
    ApiDeadlineExceeded,
    ApiError,
    AuthenticationError,
    CreationOutcomeUnknown,
    InvalidResponseError,
    NotFoundError,
    RateLimitError,
    TransportError,
)

T = TypeVar("T", bound=BaseModel)
_RETRYABLE = {408, 429, 500, 502, 503, 504}


class PlatformClient:
    """Use an existing SDK client's resolved identity, with an independent HTTP lifecycle.

    Evaluation creation is deliberately never retried: the current server does
    not support idempotency keys. Safe GETs have bounded retries and deadlines.
    """

    def __init__(
        self,
        client: Client | None = None,
        *,
        config: SdkConfig | None = None,
        timeout: float = 30.0,
        read_attempts: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if client is not None and config is not None:
            raise ValueError("Pass client or config, not both")
        if timeout <= 0 or not math.isfinite(timeout) or read_attempts < 1:
            raise ValueError("timeout must be finite and positive; read_attempts must be positive")
        if config is None:
            client = client if client is not None else get_client()
            config = client.config if client is not None else resolve_config()
        if not config.api_key or not config.project_name:
            raise ValueError("Platform access requires the SDK API key and project name")
        url = httpx.URL(config.endpoint)
        if url.scheme not in ("http", "https") or url.userinfo or url.query or url.fragment:
            raise ValueError("The platform endpoint must be an HTTP(S) URL without credentials, query, or fragment")
        self.config = config
        self.timeout = timeout
        self.read_attempts = read_attempts
        self._http = httpx.AsyncClient(
            base_url=config.endpoint.rstrip("/") + "/",
            headers={"X-API-Key": config.api_key, "X-Project-Name": config.project_name},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )
        from .resources import Evaluations, Traces

        self.traces = Traces(self)
        self.evaluations = Evaluations(self)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    def decode(model: type[T], value: Any, *, creation: bool = False) -> T:
        try:
            return model.model_validate(value)
        except ValidationError as exc:
            error = CreationOutcomeUnknown if creation else InvalidResponseError
            raise error("The platform returned an invalid response") from exc

    @staticmethod
    def decode_many(model: type[T], value: Any) -> list[T]:
        try:
            return TypeAdapter(list[model]).validate_python(value)
        except ValidationError as exc:
            raise InvalidResponseError("The platform returned an invalid collection") from exc

    async def request(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any) -> Any:
        budget = self.timeout if timeout is None else timeout
        if budget <= 0 or not math.isfinite(budget):
            raise ApiDeadlineExceeded("Platform operation deadline exceeded")
        deadline = time.monotonic() + budget
        attempts = self.read_attempts if method == "GET" else 1
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ApiDeadlineExceeded("Platform operation deadline exceeded")
            try:
                response = await asyncio.wait_for(self._http.request(method, path, **kwargs), timeout=remaining)
            except asyncio.CancelledError:
                # Never retry cancellation. Callers persist an uncertain create
                # before dispatch so cancellation cannot erase that uncertainty.
                raise
            except (httpx.TransportError, asyncio.TimeoutError) as exc:
                if method != "GET":
                    raise CreationOutcomeUnknown("Evaluation creation outcome is unknown; do not retry") from exc
                if time.monotonic() >= deadline:
                    raise ApiDeadlineExceeded("Platform operation deadline exceeded") from exc
                if attempt + 1 == attempts:
                    raise TransportError("Platform transport failed") from exc
                await self._delay(attempt, deadline)
                continue
            status = response.status_code
            if 200 <= status < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    error = CreationOutcomeUnknown if method != "GET" else InvalidResponseError
                    raise error("The platform returned invalid JSON") from exc
            if method != "GET" and (status >= 500 or status == 408):
                raise CreationOutcomeUnknown(f"Creation returned HTTP {status}; the run may exist")
            if status in _RETRYABLE and attempt + 1 < attempts:
                await self._delay(attempt, deadline, response.headers.get("Retry-After"))
                continue
            error_type = {
                401: AuthenticationError,
                403: AuthenticationError,
                404: NotFoundError,
                429: RateLimitError,
            }.get(status, ApiError)
            raise error_type(
                f"Platform request failed with HTTP {status}",
                status_code=status,
                request_id=response.headers.get("X-Request-ID"),
            )
        raise AssertionError("unreachable")

    @staticmethod
    async def _delay(attempt: int, deadline: float, retry_after: str | None = None) -> None:
        delay = random.uniform(0.1, min(2**attempt, 5.0))
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    pass
        if not math.isfinite(delay):
            delay = 1.0
        delay = max(0.0, min(delay, 30.0))
        if delay >= deadline - time.monotonic():
            raise ApiDeadlineExceeded("Retry exceeds platform operation deadline")
        await asyncio.sleep(delay)
