"""Durable evaluation scheduling and signal semantics, independent of models."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from pandaprobe.api.errors import ApiDeadlineExceeded, CreationOutcomeUnknown, InsufficientEvidence
from pandaprobe.api.models import EvaluationResult, EvaluationRun, Score, Trace
from pandaprobe.reflect.config import ReflectConfig
from pandaprobe.reflect.evaluation import Evaluator
from pandaprobe.reflect.signals import SignalGate
from pandaprobe.reflect.workspace import Workspace


def setup(tmp_path, **options):
    options.setdefault("diagnostic_metrics", ())
    config = ReflectConfig(workspace=tmp_path, **options)
    workspace = Workspace(tmp_path, config)
    trace = Trace(trace_id=uuid4(), name="task", status="COMPLETED", started_at="2026-10-20", project_id=uuid4())
    run = EvaluationRun(
        id=uuid4(),
        status="COMPLETED",
        metric_names=list(config.thresholds),
        total_targets=1,
        evaluated_count=1,
        target_type="TRACE",
    )
    result = EvaluationResult(
        run=run,
        scores=[
            Score(
                id=uuid4(),
                trace_id=trace.trace_id,
                eval_run_id=run.id,
                name=name,
                value="0.2",
                status="SUCCESS",
                data_type="NUMERIC",
            )
            for name in config.thresholds
        ],
    )
    api = SimpleNamespace(
        traces=SimpleNamespace(wait_visible=AsyncMock(return_value=trace)),
        evaluations=SimpleNamespace(create=AsyncMock(return_value=run), wait=AsyncMock(return_value=result)),
    )
    return config, workspace, api, [str(trace.trace_id)]


async def test_duplicate_submissions_are_single_flight_and_durable(tmp_path):
    config, store, api, ids = setup(tmp_path)
    runtime = Evaluator(api, store, config)
    a = runtime.submit(ids, execution_id="one", session_id="session")
    b = runtime.submit(ids, execution_id="one", session_id="session")
    assert a is b
    result = await a
    assert result.status == "evaluated" and result.notice
    assert (store.path / "evidence" / f"{result.notice.id}.txt").exists()
    restarted = Evaluator(api, Workspace(tmp_path, config), config)
    repeated = await restarted.submit(ids, execution_id="one", session_id="session")
    assert repeated == result
    api.evaluations.create.assert_awaited_once()
    with pytest.raises(ValueError, match="different evidence"):
        runtime.submit([str(uuid4())], execution_id="one", session_id="session")


async def test_lost_create_response_and_crash_claim_never_create_again(tmp_path):
    config, store, api, ids = setup(tmp_path)
    api.evaluations.create.side_effect = CreationOutcomeUnknown("lost response")
    result = await Evaluator(api, store, config).submit(ids, execution_id="one", session_id="session")
    assert result.status == "creation_unknown"
    # Simulate a crash after dispatch but before the outcome is saved.
    store.transaction(lambda state: state["executions"]["one"].pop("outcome"))
    api.evaluations.create.side_effect = None
    tasks = Evaluator(api, store, config).resume()
    assert (await tasks[0]).status == "creation_unknown"
    api.evaluations.create.assert_awaited_once()


async def test_poll_deadline_restart_resumes_existing_run(tmp_path):
    config, store, api, ids = setup(tmp_path, evaluation_max_runs=1)
    api.evaluations.wait.side_effect = ApiDeadlineExceeded("not ready")
    first = await Evaluator(api, store, config).submit(ids, execution_id="one", session_id="s")
    assert first.status == "pending" and len(first.run_ids) == 1
    api.evaluations.wait.side_effect = None
    tasks = Evaluator(api, store, config).resume()
    assert (await tasks[0]).status == "evaluated"
    api.evaluations.create.assert_awaited_once()
    assert api.evaluations.wait.await_count == 2
    assert store.state()["evaluation_count"] == 1


async def test_cancel_during_create_is_durably_uncertain(tmp_path):
    config, store, api, ids = setup(tmp_path)
    entered = asyncio.Event()

    async def hanging_create(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    api.evaluations.create.side_effect = hanging_create
    task = Evaluator(api, store, config).submit(ids, execution_id="one", session_id="s")
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = Evaluator(api, store, config).resume()
    assert (await resumed[0]).status == "creation_unknown"
    api.evaluations.create.assert_awaited_once()


async def test_missing_scores_do_not_produce_failure_notice_or_success(tmp_path):
    config, store, api, ids = setup(tmp_path)
    api.evaluations.wait.side_effect = InsufficientEvidence(str(uuid4()), [(ids[0], "coherence")])
    outcome = await Evaluator(api, store, config).submit(ids, execution_id="one", session_id="s")
    assert outcome.status == "insufficient" and outcome.notice is None and not outcome.scores
    assert not store.state()["gates"]


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "100"])
def test_daily_budget_requires_positive_integer_or_none(limit):
    with pytest.raises(ValueError, match="evaluation_max_runs must be a positive integer or None"):
        ReflectConfig(evaluation_max_runs=limit)


@pytest.mark.parametrize("prior_count", [100, 1000])
async def test_default_has_no_daily_cap_and_preserves_usage_across_restart(tmp_path, prior_count):
    config, store, api, ids = setup(tmp_path)
    assert config.evaluation_max_runs is None
    store.transaction(
        lambda state: state.update(
            evaluation_day=datetime.now(timezone.utc).date().isoformat(), evaluation_count=prior_count
        )
    )
    first = await Evaluator(api, store, config).submit(ids, execution_id="a", session_id="s")
    restarted = Evaluator(api, Workspace(tmp_path, config), config)
    second = await restarted.submit([str(uuid4())], execution_id="b", session_id="s")
    assert first.status == second.status == "evaluated"
    assert api.evaluations.create.await_count == 2
    assert store.state()["evaluation_count"] == prior_count + 2

    # Opting into a cap includes usage already incurred in the same UTC day.
    capped = ReflectConfig(evaluation_max_runs=prior_count + 2, diagnostic_metrics=())
    outcome = await Evaluator(api, store, capped).submit([str(uuid4())], execution_id="c", session_id="s")
    assert outcome.status == "skipped"
    assert api.evaluations.create.await_count == 2


async def test_daily_budget_is_shared_between_instances(tmp_path, caplog):
    config, store, api, ids = setup(tmp_path, evaluation_max_runs=1)
    a, b = Evaluator(api, store, config), Evaluator(api, Workspace(tmp_path, config), config)
    results = await asyncio.gather(
        a.submit(ids, execution_id="a", session_id="s"), b.submit([str(uuid4())], execution_id="b", session_id="s")
    )
    assert sorted(r.status for r in results) == ["evaluated", "skipped"]
    skipped = next(r for r in results if r.status == "skipped")
    assert skipped.reason.startswith("Daily evaluation-run budget reached (1/1 creation attempts on ")
    assert "UTC; resets at 00:00 UTC" in skipped.reason
    assert skipped.reason in caplog.text and "detect evaluation skipped" in caplog.text
    events = [json.loads(path.read_text()) for path in (store.path / "events").glob("*.json")]
    assert any(event.get("reason") == skipped.reason for event in events)
    api.evaluations.create.assert_awaited_once()


async def test_daily_budget_resets_at_utc_midnight_after_restart(tmp_path, monkeypatch):
    import pandaprobe.reflect.evaluation as evaluation

    class Clock:
        current = datetime(2026, 10, 21, 23, 59, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz):
            assert tz == timezone.utc
            return cls.current

    monkeypatch.setattr(evaluation, "datetime", Clock)
    config, store, api, ids = setup(tmp_path, evaluation_max_runs=1)
    runtime = Evaluator(api, store, config)
    assert (await runtime.submit(ids, execution_id="a", session_id="s")).status == "evaluated"
    assert (await runtime.submit([str(uuid4())], execution_id="b", session_id="s")).status == "skipped"

    Clock.current += timedelta(minutes=1)
    restarted = Evaluator(api, Workspace(tmp_path, config), config)
    assert (await restarted.submit([str(uuid4())], execution_id="c", session_id="s")).status == "evaluated"
    assert store.state()["evaluation_count"] == 1
    assert store.state()["evaluation_day"] == "2026-10-22"
    assert api.evaluations.create.await_count == 2


@pytest.mark.parametrize("limit", [1, 2])
async def test_diagnostics_share_budget_without_discarding_primary_evidence(tmp_path, caplog, limit):
    config, store, api, ids = setup(tmp_path, evaluation_max_runs=limit, diagnostic_metrics=("tool_correctness",))
    outcome = await Evaluator(api, store, config).submit(ids, execution_id="a", session_id="s")
    assert outcome.status == "evaluated" and outcome.notice is not None
    assert outcome.scores == {"task_completion": 0.2, "coherence": 0.2}
    evidence = json.loads((store.path / "evidence" / f"{outcome.notice.id}.txt").read_text())
    assert "primary" in evidence
    if limit == 1:
        assert outcome.reason.startswith("Diagnostic evidence unavailable: Daily evaluation-run budget reached (1/1")
        assert "UTC; resets at 00:00 UTC" in outcome.reason
        assert evidence["diagnostic_error"] == outcome.reason
        assert "diagnose evaluation skipped" in caplog.text
    else:
        assert outcome.reason == "" and "diagnostic" in evidence
    assert api.evaluations.create.await_count == limit
    assert store.state()["evaluation_count"] == limit


async def test_uncertain_creation_still_consumes_opt_in_budget(tmp_path):
    config, store, api, ids = setup(tmp_path, evaluation_max_runs=1)
    api.evaluations.create.side_effect = CreationOutcomeUnknown("lost response")
    runtime = Evaluator(api, store, config)
    assert (await runtime.submit(ids, execution_id="a", session_id="s")).status == "creation_unknown"
    assert (await runtime.submit([str(uuid4())], execution_id="b", session_id="s")).status == "skipped"
    api.evaluations.create.assert_awaited_once()
    assert store.state()["evaluation_count"] == 1


async def test_internal_activity_sampling_and_empty_evidence_do_not_evaluate(tmp_path):
    config, store, api, ids = setup(tmp_path)
    api.traces.wait_visible.return_value = api.traces.wait_visible.return_value.model_copy(
        update={"metadata": {"pandaprobe.reflect": {"role": "reflection"}}}
    )
    runtime = Evaluator(api, store, config)
    assert (await runtime.submit(ids, execution_id="a", session_id="s")).status == "skipped"
    assert (await runtime.submit([], execution_id="b", session_id="s")).status == "insufficient"
    api.evaluations.create.assert_not_awaited()
    unsampled = Evaluator(
        api, store, ReflectConfig(workspace=tmp_path, evaluation_sample_rate=0, diagnostic_metrics=())
    )
    assert (await unsampled.submit([str(uuid4())], execution_id="c", session_id="s")).reason == "Sampling"


async def test_same_trace_cannot_trigger_new_billable_runs_under_another_execution_id(tmp_path):
    config, store, api, ids = setup(tmp_path)
    runtime = Evaluator(api, store, config)
    assert (await runtime.submit(ids, execution_id="one", session_id="s")).status == "evaluated"
    second = await runtime.submit(ids, execution_id="two", session_id="s")
    assert second.status == "skipped" and "already belongs" in second.reason
    api.evaluations.create.assert_awaited_once()


async def test_large_trace_input_cannot_silently_hide_diagnostic_scores(tmp_path):
    config, store, api, ids = setup(tmp_path, evidence_max_bytes=2048)
    api.traces.wait_visible.return_value.input = "large task input" * 10000
    result = await Evaluator(api, store, config).submit(ids, execution_id="one", session_id="s")
    body = (store.path / "evidence" / f"{result.notice.id}.txt").read_text()
    assert len(body.encode("utf-8")) <= config.evidence_max_bytes
    assert ids[0] in body and "task_completion" in body and "coherence" in body
    assert body.endswith("[EVIDENCE TRUNCATED: original trace/run IDs identify the complete platform records.]")


def test_intermediate_low_score_is_not_final_failure_but_stalls_and_regressions_trigger(tmp_path):
    config = ReflectConfig(workspace=tmp_path, stall_window=3, cooldown_seconds=0)
    gate = SignalGate(Workspace(tmp_path, config), config)

    def observe(index, score, session="s", completed=False):
        return gate.observe(
            execution_id=str(index),
            session_id=session,
            trace_ids=[],
            scores={"task_completion": score, "coherence": 1.0},
            reasons={},
            completed=completed,
        )

    assert observe(0, 0.1) is None
    assert observe(1, 0.1) is None
    assert observe(2, 0.1).signatures == ["task_completion:stalled"]
    assert observe(3, 0.8, session="other") is None
    assert observe(4, 0.2, session="other").signatures == ["task_completion:regression"]
    assert observe(5, 0.2, completed=True).signatures == ["task_completion:below_threshold"]


def test_cooldown_and_missing_scores(tmp_path):
    config = ReflectConfig(workspace=tmp_path)
    gate = SignalGate(Workspace(tmp_path, config), config)
    values = {
        "session_id": "s",
        "trace_ids": [],
        "scores": {"task_completion": 0, "coherence": 0},
        "reasons": {},
        "completed": True,
    }
    first = gate.observe(execution_id="one", **values)
    assert first
    assert gate.observe(execution_id="one", **values) == first
    assert gate.observe(execution_id="two", **values) is None
    with pytest.raises(ValueError, match="complete"):
        gate.observe(execution_id="three", **{**values, "scores": {}})
