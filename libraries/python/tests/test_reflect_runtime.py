"""Task preservation, phase attribution, late feedback, and concurrent instances."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from pandaprobe import Client
from pandaprobe.api.errors import ApiDeadlineExceeded
from pandaprobe.api.models import EvaluationResult, EvaluationRun, Score, Trace
from pandaprobe.reflect import Reflect, ReflectConfig


def setup(tmp_path, *, replay=None, cases=None, **config):
    sdk = Client(api_key="key", project_name="p", enabled=False)
    object.__setattr__(sdk.config, "enabled", True)
    sdk._transport = MagicMock()
    traces, runs = {}, {}

    def enqueue(payload):
        traces[payload["trace_id"]] = Trace(project_id=uuid4(), **payload)

    sdk._transport.enqueue_trace.side_effect = enqueue

    async def visible(trace_id, **kwargs):
        return traces[trace_id]

    async def create(trace_ids, metrics, **kwargs):
        run = EvaluationRun(
            id=uuid4(),
            status="COMPLETED",
            metric_names=metrics,
            total_targets=len(trace_ids),
            evaluated_count=len(trace_ids),
            target_type="TRACE",
        )
        runs[str(run.id)] = run
        return run

    async def wait(run_id, *, trace_ids, metrics, **kwargs):
        return EvaluationResult(
            run=runs[run_id],
            scores=[
                Score(
                    id=uuid4(),
                    trace_id=t,
                    name=m,
                    value="0",
                    status="SUCCESS",
                    data_type="NUMERIC",
                    eval_run_id=run_id,
                )
                for t in trace_ids
                for m in metrics
            ],
        )

    api = SimpleNamespace(
        traces=SimpleNamespace(wait_visible=AsyncMock(side_effect=visible)),
        evaluations=SimpleNamespace(create=AsyncMock(side_effect=create), wait=AsyncMock(side_effect=wait)),
        aclose=AsyncMock(),
    )
    runtime = Reflect(
        sdk,
        config=ReflectConfig(workspace=tmp_path, diagnostic_metrics=(), **config),
        api=api,
        replay=replay,
        cases=cases,
    )
    return runtime, sdk, api, traces


async def test_run_returns_original_result_and_same_agent_reflects_with_separate_attribution(tmp_path):
    runtime, sdk, api, traces = setup(tmp_path)
    original = {"answer": ["original task object"]}
    phases = []

    async def agent(invocation):
        phases.append(invocation.phase)
        with sdk.trace(invocation.phase, session_id="fixed-framework-session"):
            if invocation.phase == "task":
                return original
            assert invocation.previous_result is original
            assert "task_completion" in await runtime.bash("cat /inbox/notice.json")
            await runtime.bash("cat > /rules/math.md <<'EOF'\n# Math\nApplies to: arithmetic tasks\nCheck sums.\nEOF")
            return "Proposed checking sums"

    result = await runtime.run(agent, "1+1", session_id="customer", execution_id="task-one")
    assert result is original and phases == ["task", "reflection"]
    roles = {trace.metadata["pandaprobe.reflect"]["role"]: trace for trace in traces.values()}
    assert roles["task"].session_id == "customer"
    assert roles["reflection"].session_id == "reflect:reflection:task-one"
    assert api.evaluations.create.await_args.args[0] == [str(roles["task"].trace_id)]
    outcome = await runtime.settle_reflection("task-one")
    assert outcome.status == "proposed" and outcome.proposal.evidence_ids
    report = runtime.report("task-one")
    assert report.task_status == "completed" and report.reflection.status == "proposed"
    assert report.evaluation.status == "evaluated" and report.proposals[0].status == "candidate"
    assert not runtime.workspace.rules()
    await runtime.aclose()
    api.aclose.assert_not_awaited()  # Injected API lifecycle belongs to its owner.


@pytest.mark.parametrize("fault", ["error", "timeout", "cancelled", "invalid_rule", "unchanged"])
async def test_reflection_failures_and_no_proposals_never_replace_task_result(tmp_path, fault):
    runtime, sdk, _api, _traces = setup(tmp_path, reflection_timeout=0.02)

    async def agent(invocation):
        if invocation.phase == "task":
            with sdk.trace("task"):
                return "original"
        if fault == "error":
            raise RuntimeError("reflection failed")
        if fault == "timeout":
            await asyncio.Event().wait()
        if fault == "cancelled":
            raise asyncio.CancelledError
        if fault == "invalid_rule":
            await runtime.bash("echo invalid > /rules/a.md")
        return "no change"

    assert await runtime.run(agent, "task", session_id="s", execution_id="one") == "original"
    assert (await runtime.settle_reflection("one")).status == {
        "error": "failed",
        "timeout": "timed_out",
        "cancelled": "cancelled",
        "invalid_rule": "invalid",
        "unchanged": "no_proposal",
    }[fault]
    if fault in ("unchanged", "invalid_rule"):
        assert runtime.report("one").reflection.rationale == "no change"
    assert runtime._current.get() is None
    await runtime.aclose()


async def test_original_task_exception_is_preserved_and_can_be_evaluated_without_continuation(tmp_path):
    runtime, sdk, _api, traces = setup(tmp_path)
    original = RuntimeError("task exception")

    async def agent(invocation):
        with sdk.trace("task"):
            raise original

    with pytest.raises(RuntimeError) as caught:
        await runtime.run(agent, "task", session_id="s", execution_id="failed")
    assert caught.value is original
    assert (await runtime.settle_evaluation("failed")).notice is not None
    assert not runtime._reflection_tasks
    assert next(iter(traces.values())).status == "ERROR"
    await runtime.aclose()


async def test_pending_feedback_is_resumed_explicitly_after_restart(tmp_path):
    runtime, sdk, api, _traces = setup(tmp_path)
    async with runtime.execution(session_id="s", execution_id="late"):
        with sdk.trace("task"):
            pass
    api.evaluations.wait.side_effect = ApiDeadlineExceeded("pending")
    assert (await runtime.settle_evaluation("late")).status == "pending"
    await runtime.aclose()
    restarted, _sdk2, _api2, _traces2 = setup(tmp_path)
    # Reuse the actual original API/run to verify no second creation.
    restarted.api = api
    restarted._evaluator.api = api
    next(iter(runtime.workspace.state()["evaluations"].values()))["run_id"]

    async def completed(run, *, trace_ids, metrics, **kwargs):
        record = EvaluationRun(
            id=run, status="COMPLETED", metric_names=metrics, total_targets=1, evaluated_count=1, target_type="TRACE"
        )
        return EvaluationResult(
            run=record,
            scores=[
                Score(
                    id=uuid4(),
                    trace_id=trace_ids[0],
                    name=m,
                    value="0",
                    status="SUCCESS",
                    data_type="NUMERIC",
                    eval_run_id=run,
                )
                for m in metrics
            ],
        )

    api.evaluations.wait.side_effect = completed
    calls = []

    async def continuation(invocation):
        calls.append(invocation.phase)
        return "Insufficient actionable guidance"

    outcome = await restarted.continue_reflection("late", continuation)
    assert outcome.status == "no_proposal" and calls == ["reflection"]
    assert api.evaluations.create.await_count == 1
    await restarted.aclose()


async def test_concurrent_executions_and_instances_have_independent_filesystems(tmp_path):
    runtime, _sdk, _api, _traces = setup(tmp_path)
    other, _, _, _ = setup(tmp_path / "other")
    entered = asyncio.Event()

    async def work(owner, name):
        async with owner.execution(session_id=name, execution_id=name) as execution:
            await owner.bash(f"echo '{name}' > /context.md")
            entered.set()
            await asyncio.sleep(0.01)
            assert name in await owner.bash("cat /context.md")
            assert execution.instructions().count("<reflect_active_memory>") == 1
            # The other instance has no active context in this task.
            if owner is runtime:
                assert "requires an active" in await other.bash("cat /context.md")

    await asyncio.gather(work(runtime, "a"), work(runtime, "b"), work(other, "c"))
    assert runtime.workspace.load_execution("a")["context"] == "a\n"
    assert runtime.workspace.load_execution("b")["context"] == "b\n"
    await runtime.aclose()
    await other.aclose()


async def test_local_active_memory_persists_without_promoting_unvalidated_rule_changes(tmp_path):
    runtime, _sdk, _api, _traces = setup(tmp_path)
    async with runtime.execution(session_id="s"):
        await runtime.bash("echo 'Remember my formatting' > /context.md")
    async with runtime.execution(session_id="s") as second:
        assert "Remember my formatting" in second.instructions()
        await runtime.bash("echo 'Untested rule guidance' > /context.md")
        await runtime.bash("cat > /rules/new.md <<'EOF'\n# New\nApplies to: tasks\nDo something untested.\nEOF")
    async with runtime.execution(session_id="s") as third:
        assert "Untested" not in third.instructions()
        assert "Remember my formatting" in third.instructions()
    await runtime.aclose()


async def test_wait_timeout_does_not_cancel_evaluation_and_shutdown_does(tmp_path):
    runtime, _sdk, api, _traces = setup(tmp_path)
    entered = asyncio.Event()

    async def visible(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    api.traces.wait_visible.side_effect = visible
    task = runtime.submit_traces([str(uuid4())], execution_id="one", session_id="s")
    await entered.wait()
    assert (await runtime.settle_evaluation("one", timeout=0.001)).status == "pending"
    assert not task.done()
    await runtime.aclose()
    assert task.cancelled()
    assert runtime.workspace.state()["executions"]["one"]["outcome"]["status"] == "cancelled"


async def test_trace_export_is_a_separate_wait(tmp_path):
    runtime, sdk, api, _traces = setup(tmp_path)
    sdk.flush = MagicMock()
    await runtime.wait_export(timeout=0.1)
    sdk.flush.assert_called_once_with(timeout=0.1)
    api.evaluations.create.assert_not_awaited()
    await runtime.aclose()


async def test_host_cancellation_delay_does_not_hold_reflection_or_shutdown(tmp_path):
    runtime, sdk, _api, _traces = setup(tmp_path, reflection_timeout=0.01, shutdown_timeout=0.01)
    stop = asyncio.Event()

    async def agent(request):
        if request.phase == "task":
            with sdk.trace("task"):
                return "task result"
        while not stop.is_set():
            try:
                await stop.wait()
            except asyncio.CancelledError:
                pass  # Deliberately exercise a badly behaved host callback.
        return "late"

    assert await asyncio.wait_for(runtime.run(agent, "task", session_id="s", execution_id="one"), 0.5) == "task result"
    assert (await runtime.settle_reflection("one")).status == "timed_out"
    await asyncio.wait_for(runtime.aclose(), 0.2)
    stop.set()
    await asyncio.gather(*runtime._host_tasks)


async def test_explicit_trace_submission_can_receive_a_host_continuation(tmp_path):
    runtime, sdk, _api, _traces = setup(tmp_path)
    with sdk.trace("external task") as trace:
        pass
    await runtime.submit_traces([trace.trace_id], execution_id="external", session_id="s")

    async def continuation(request):
        assert "external" in await runtime.bash("cat /inbox/notice.json")
        return "No actionable proposal"

    assert (await runtime.continue_reflection("external", continuation)).status == "no_proposal"
    await runtime.aclose()


@pytest.mark.parametrize("authoring_phase", ["task", "reflection"])
async def test_registered_replay_is_scheduled_without_holding_the_task_barrier(tmp_path, authoring_phase):
    from pandaprobe.reflect import Outcome, ReplayCase

    resource = asyncio.Lock()
    entered = asyncio.Event()

    async def replay(request):
        entered.set()
        async with resource:
            output = await request.execution.runtime.bash("cat /rules/*.md")
            request.execution.submitted(output, channel="tool_output")
            success = request.case.protected or "Check sums" in output
            return Outcome(scores={"task_completion": float(success), "coherence": 1}, success=success)

    runtime, sdk, _api, _traces = setup(
        tmp_path, replay=replay, cases=[ReplayCase(str(i), i, protected=i == 2) for i in range(3)]
    )

    async def agent(request):
        with sdk.trace(request.phase):
            if request.phase == authoring_phase:
                await runtime.bash(
                    "cat > /rules/math.md <<'EOF'\n# Math\nApplies to: arithmetic tasks\nCheck sums.\nEOF"
                )
            return "original task result"

    async with resource:
        assert (
            await asyncio.wait_for(runtime.run(agent, "task", session_id="s", execution_id="one"), 1)
            == "original task result"
        )
        await asyncio.wait_for(entered.wait(), 1)
        assert not await runtime.settle_validation(timeout=0.001)
        assert not runtime.workspace.rules()
    assert await runtime.settle_validation(timeout=1)
    assert runtime.proposals()[0].status == "accepted"
    await runtime.aclose()


async def test_reflection_continuation_retains_the_execution_output_budget(tmp_path):
    runtime, sdk, _api, _traces = setup(tmp_path, execution_output_max_bytes=1024, output_max_bytes=1024)

    async def agent(request):
        if request.phase == "task":
            with sdk.trace("task"):
                await runtime.bash("printf '" + "x" * 900 + "'")
            return "result"
        assert request.execution.shell.usage["output_bytes"] > 768
        assert "output budget reached" in await runtime.bash("cat /context.md")
        return "No output budget remains"

    assert await runtime.run(agent, "task", session_id="s", execution_id="one") == "result"
    assert (await runtime.settle_reflection("one")).status == "no_proposal"
    await runtime.aclose()


async def test_reflection_memory_only_edits_reach_the_next_session_invocation(tmp_path):
    runtime, sdk, _api, _traces = setup(tmp_path)

    async def agent(request):
        if request.phase == "task":
            with sdk.trace("task"):
                return "original"
        await runtime.bash("echo 'Remember the requested response format' > /context.md")
        return "Updated local memory without proposing shared guidance"

    await runtime.run(agent, "task", session_id="s", execution_id="one")
    assert (await runtime.settle_reflection("one")).status == "no_proposal"
    async with runtime.execution(session_id="s") as next_turn:
        assert "Remember the requested response format" in next_turn.instructions()
    async with runtime.execution(session_id="other") as other_session:
        assert "Remember the requested response format" not in other_session.instructions()
    assert not runtime.workspace.rules()
    assert runtime.workspace.load_execution("one")["active_memory_unvalidated"]
    await runtime.aclose()
