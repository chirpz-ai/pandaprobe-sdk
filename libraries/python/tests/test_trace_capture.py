import asyncio
from unittest.mock import MagicMock

import pytest

from pandaprobe import Client


def client():
    result = Client(api_key="key", project_name="project", enabled=False)
    # Exercise the public log path without a worker or external requests.
    object.__setattr__(result.config, "enabled", True)
    result._transport = MagicMock()
    return result


async def test_capture_is_scoped_per_invocation_and_client():
    a, b = client(), client()

    async def invoke(name):
        with a.capture_traces(metadata={"execution": name}) as captured:
            await asyncio.sleep(0)
            with a.trace(name) as trace:
                pass
            with b.trace("unrelated"):
                pass
        return captured, trace.trace_id

    results = await asyncio.gather(invoke("a"), invoke("b"))
    for captured, trace_id in results:
        assert captured.trace_ids == [trace_id]
    payloads = [call.args[0] for call in a._transport.enqueue_trace.call_args_list]
    assert {p["metadata"]["execution"] for p in payloads} == {"a", "b"}
    assert all("metadata" not in p.args[0] for p in b._transport.enqueue_trace.call_args_list)


def test_capture_does_not_create_traces_and_restores_attributes():
    sdk = client()
    with (
        sdk.capture_traces(metadata={"role": "reflection"}, session_id="reflection-only") as capture,
        sdk.trace("reflect", session_id="developer-session"),
    ):
        pass
    with sdk.trace("ordinary", session_id="developer-session"):
        pass
    assert len(capture.trace_ids) == 1
    first, second = [call.args[0] for call in sdk._transport.enqueue_trace.call_args_list]
    assert first["session_id"] == "reflection-only" and first["metadata"]["role"] == "reflection"
    assert second["session_id"] == "developer-session" and "metadata" not in second
    assert len(sdk._transport.enqueue_trace.call_args_list) == 2


def test_capture_preserves_task_exception_and_records_error_trace():
    sdk = client()
    error = RuntimeError("original task failure")
    with pytest.raises(RuntimeError) as exc, sdk.capture_traces() as capture, sdk.trace("task"):
        raise error
    assert exc.value is error and capture.closed
    assert len(capture.trace_ids) == 1
    assert sdk._transport.enqueue_trace.call_args.args[0]["status"] == "ERROR"


def test_nested_capture_and_overflow_are_explicit():
    sdk = client()
    with sdk.capture_traces(max_traces=1) as outer:
        with sdk.trace("first"):
            pass
        with sdk.capture_traces() as inner, sdk.trace("inner"):
            pass
        with sdk.trace("second"):
            pass
    assert len(outer.trace_ids) == len(inner.trace_ids) == 1
    assert outer.overflow and not inner.overflow


def test_observer_failure_cannot_change_trace_export_or_task_result():
    sdk = client()

    def observer(payload):
        raise RuntimeError("observer failure")

    with sdk.capture_traces(on_submit=observer), sdk.trace("task"):
        result = "original"
    assert result == "original"
    assert sdk._transport.enqueue_trace.call_args.args[0]["status"] == "COMPLETED"


async def test_late_child_retains_internal_attribution_without_extending_the_closed_boundary():
    sdk = client()
    release = asyncio.Event()
    observed = MagicMock()

    async def child():
        await release.wait()
        with sdk.trace("late reflection"):
            pass

    with sdk.capture_traces(metadata={"role": "reflection"}, session_id="internal", on_submit=observed) as scope:
        task = asyncio.create_task(child())
    release.set()
    await task
    assert scope.closed and scope.trace_ids == []
    payload = sdk._transport.enqueue_trace.call_args.args[0]
    assert payload["metadata"]["role"] == "reflection" and payload["session_id"] == "internal"
    observed.assert_not_called()
