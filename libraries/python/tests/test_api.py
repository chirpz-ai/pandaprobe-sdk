"""API contracts and failure boundaries against the real backend response shapes."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from pandaprobe.api import (
    ApiDeadlineExceeded,
    AuthenticationError,
    CreationOutcomeUnknown,
    EvaluationFailed,
    InsufficientEvidence,
    PlatformClient,
    RateLimitError,
)
from pandaprobe.api.models import EvaluationRequest, EvaluationRun, Metric, Score, Span, Trace, TracePage, TraceSummary
from pandaprobe.config import SdkConfig

TRACE_ID, RUN_ID, PROJECT_ID = (str(uuid4()) for _ in range(3))


def run(status="COMPLETED"):
    return {
        "id": RUN_ID,
        "status": status,
        "metric_names": ["task_completion"],
        "total_targets": 1,
        "evaluated_count": 1,
        "failed_count": 0,
        "target_type": "TRACE",
    }


def score(**changes):
    return dict(
        {
            "id": str(uuid4()),
            "trace_id": TRACE_ID,
            "name": "task_completion",
            "value": "0.8",
            "status": "SUCCESS",
            "data_type": "NUMERIC",
            "eval_run_id": RUN_ID,
        },
        **changes,
    )


def trace(status="COMPLETED"):
    return {
        "trace_id": TRACE_ID,
        "project_id": PROJECT_ID,
        "name": "task",
        "status": status,
        "started_at": "2026-10-19T00:00:00Z",
        "session_id": "session",
        "spans": [],
    }


def api(handler, **kwargs):
    config = SdkConfig(api_key="explicit-key", project_name="explicit-project", endpoint="https://api.example/root")
    return PlatformClient(config=config, transport=httpx.MockTransport(handler), **kwargs)


async def test_resolved_identity_endpoint_and_separate_lifecycle(monkeypatch):
    monkeypatch.setenv("PANDAPROBE_API_KEY", "wrong")
    monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "wrong")

    def handler(request):
        assert request.headers["X-API-Key"] == "explicit-key"
        assert request.headers["X-Project-Name"] == "explicit-project"
        assert str(request.url) == f"https://api.example/root/traces/{TRACE_ID}"
        return httpx.Response(200, json=trace())

    async with api(handler) as client:
        assert str((await client.traces.get(TRACE_ID)).trace_id) == TRACE_ID
    assert client._http.is_closed


async def test_retrieval_rejects_a_different_trace_or_run_identity():
    from pandaprobe.api.errors import InvalidResponseError

    async with api(lambda _: httpx.Response(200, json=trace())) as client:
        with pytest.raises(InvalidResponseError, match="identity"):
            await client.traces.get(str(uuid4()))
    async with api(lambda _: httpx.Response(200, json=run())) as client:
        with pytest.raises(InvalidResponseError, match="identity"):
            await client.evaluations.get(str(uuid4()))


async def test_eventual_visibility_includes_failed_tasks():
    states = iter([404, "RUNNING", "ERROR"])

    def handler(_):
        state = next(states)
        return httpx.Response(404) if state == 404 else httpx.Response(200, json=trace(state))

    async with api(handler) as client:
        result = await client.traces.wait_visible(TRACE_ID, timeout=1, poll_interval=0.001)
    assert result.status == "ERROR"


async def test_slow_run_is_created_once_and_polled_to_completion():
    created, polls = [], []

    def handler(request):
        if request.method == "POST":
            created.append(json.loads(request.content))
            return httpx.Response(202, json=run("PENDING"))
        if request.url.path.endswith("/scores"):
            return httpx.Response(200, json=[score()])
        polls.append(1)
        return httpx.Response(200, json=run("RUNNING" if len(polls) < 3 else "COMPLETED"))

    async with api(handler) as client:
        result = await client.evaluations.create([TRACE_ID, TRACE_ID], ["task_completion", "task_completion"])
        evaluated = await client.evaluations.wait(
            result.id, trace_ids=[TRACE_ID], metrics=["task_completion"], timeout=1, poll_interval=0.001
        )
    assert len(created) == 1 and len(polls) == 3
    assert created[0] == {"trace_ids": [TRACE_ID], "metrics": ["task_completion"]}
    assert evaluated.scores[0].numeric_value == 0.8


@pytest.mark.parametrize(
    "values",
    [
        [],
        [score(value=None)],
        [score(status="FAILED")],
        [score(value="NaN")],
        [score(value="inf")],
        [score(value="bad")],
        [score(eval_run_id=None)],
        [score(data_type="BOOLEAN")],
        [score(), score()],
    ],
)
async def test_missing_failed_invalid_or_duplicate_scores_are_insufficient(values):
    def handler(request):
        return httpx.Response(200, json=values if request.url.path.endswith("/scores") else run())

    async with api(handler) as client:
        with pytest.raises(InsufficientEvidence):
            await client.evaluations.wait(RUN_ID, trace_ids=[TRACE_ID], metrics=["task_completion"])


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED"])
async def test_failed_run_is_not_score_success(status):
    async with api(lambda _: httpx.Response(200, json=run(status))) as client:
        with pytest.raises(EvaluationFailed):
            await client.evaluations.wait(RUN_ID, trace_ids=[TRACE_ID], metrics=["task_completion"])


@pytest.mark.parametrize("failure", ["lost", "server", "invalid_json", "invalid_shape"])
async def test_ambiguous_creation_is_never_retried(failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "lost":
            raise httpx.ReadError("response lost")
        if failure == "server":
            return httpx.Response(503)
        if failure == "invalid_json":
            return httpx.Response(202, content=b"not-json")
        return httpx.Response(202, json={})

    async with api(handler) as client:
        with pytest.raises(CreationOutcomeUnknown):
            await client.evaluations.create([TRACE_ID], ["task_completion"])
    assert len(calls) == 1


async def test_safe_get_retries_honor_retry_after():
    calls = []

    def handler(_):
        calls.append(1)
        return (
            httpx.Response(429, headers={"Retry-After": "0"}) if len(calls) == 1 else httpx.Response(200, json=run())
        )

    async with api(handler) as client:
        assert (await client.evaluations.get(RUN_ID)).status == "COMPLETED"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "status,error", [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitError)]
)
async def test_auth_and_rate_limit_errors(status, error):
    async with api(lambda _: httpx.Response(status), read_attempts=1) as client:
        with pytest.raises(error):
            await client.traces.get(TRACE_ID)


async def test_persistence_and_running_deadlines():
    async with api(lambda _: httpx.Response(404)) as client:
        with pytest.raises(ApiDeadlineExceeded):
            await client.traces.wait_visible(TRACE_ID, timeout=0.01, poll_interval=0.001)
    async with api(lambda _: httpx.Response(200, json=run("RUNNING"))) as client:
        with pytest.raises(ApiDeadlineExceeded):
            await client.evaluations.wait(
                RUN_ID, trace_ids=[TRACE_ID], metrics=["task_completion"], timeout=0.01, poll_interval=0.001
            )


async def test_cancellation_reaches_http_request_without_retry():
    calls, entered = [], asyncio.Event()

    async def handler(_):
        calls.append(1)
        entered.set()
        await asyncio.Event().wait()

    async with api(handler) as client:
        task = asyncio.create_task(client.evaluations.create([TRACE_ID], ["task_completion"]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == 1


async def test_redirect_does_not_forward_credentials():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://other.example"})

    from pandaprobe.api import ApiError

    async with api(handler) as client:
        with pytest.raises(ApiError):
            await client.traces.get(TRACE_ID)
    assert len(calls) == 1


def test_models_match_pinned_backend_contract():
    contract = json.loads((Path(__file__).parent / "fixtures/platform_contract.json").read_text())
    mapping = {
        "TraceResponse": Trace,
        "TraceScoreResponse": Score,
        "EvalRunResponse": EvaluationRun,
        "CreateBatchEvalRunRequest": EvaluationRequest,
        "PaginatedResponse_TraceListItem_": TracePage,
        "TraceListItem": TraceSummary,
        "SpanResponse": Span,
        "MetricSummary": Metric,
    }

    def branches(schema, definitions):
        if "$ref" in schema:
            schema = definitions[schema["$ref"].rsplit("/", 1)[-1]]
        return [
            branch
            for item in schema.get("anyOf", [schema])
            for branch in (branches(item, definitions) if "anyOf" in schema else [item])
        ]

    def accepts(consumer, producer, consumer_defs, producer_defs):
        def compatible(left, right):
            kind = left.get("type")
            if kind is None:
                return True  # Any JSON value.
            if kind != right.get("type"):
                return False
            if "format" in left and left["format"] != right.get("format"):
                return False
            if "enum" in left and ("enum" not in right or not set(right["enum"]) <= set(left["enum"])):
                return False
            if kind == "array":
                return accepts(left["items"], right["items"], consumer_defs, producer_defs)
            if kind == "object" and "properties" in left:
                return (
                    set(left.get("required", [])) <= set(right.get("required", []))
                    and set(left["properties"]) <= set(right.get("properties", {}))
                    and all(
                        accepts(value, right["properties"][name], consumer_defs, producer_defs)
                        for name, value in left["properties"].items()
                    )
                )
            return True

        return all(
            any(compatible(left, right) for left in branches(consumer, consumer_defs))
            for right in branches(producer, producer_defs)
        )

    for name, model in mapping.items():
        remote = contract["schemas"][name]
        local = model.model_json_schema()
        assert set(local["properties"]) <= set(remote["properties"]), name
        assert set(local.get("required", [])) <= set(remote.get("required", [])), name
        if model is EvaluationRequest:
            # Every request value we emit must be accepted by the server.
            assert set(remote["required"]) <= set(local["required"])
            for field, schema in local["properties"].items():
                assert accepts(remote["properties"][field], schema, contract["schemas"], local.get("$defs", {})), field
        else:
            # Includes nested spans/pages, nullable fields, UUIDs, and enum values.
            assert accepts(local, remote, local.get("$defs", {}), contract["schemas"]), name
    assert contract["paths"]["/evaluations/trace-runs/batch"] == ["post"]
    assert "Idempotency-Key" not in contract["creation_headers"]
    params = contract["operations"]["GET /traces"]["parameters"]
    assert {"session_id", "status", "limit", "offset"} <= {p["name"] for p in params if p["in"] == "query"}
