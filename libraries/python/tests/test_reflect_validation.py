"""Behavioral acceptance requires independent, paired, delivered evidence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pandaprobe import Client
from pandaprobe.reflect import Outcome, Reflect, ReflectConfig, ReplayCase, Rule, ValidationVerdict

TEXT = "# Arithmetic\nApplies to: addition tasks\nCheck addition with a calculator.\n"


@pytest.mark.parametrize(
    "values", [{"success": "false"}, {"scores": {"task_completion": True}}, {"scores": {"task_completion": "1"}}]
)
def test_host_outcomes_reject_coerced_boolean_or_numeric_evidence(values):
    from pandaprobe.reflect import TrialArm

    with pytest.raises(ValueError):
        Outcome(**values)
    with pytest.raises(ValueError):
        TrialArm(execution_id="one", **values)


def setup(tmp_path):
    sdk = Client(api_key="key", project_name="p", enabled=False)
    runtime = Reflect(sdk, config=ReflectConfig(workspace=tmp_path), api=SimpleNamespace(aclose=AsyncMock()))
    store = runtime.workspace
    proposal = store.propose(
        store.state()["head"], {"math.md": Rule(path="math.md", text=TEXT)}, session_id="author", execution_id="author"
    ).proposal
    cases = [ReplayCase(id=str(i), input={"a": i, "b": 1}, protected=i == 2) for i in range(3)]
    return runtime, proposal, cases


async def measured(request):
    """Independent scripted verifier; assertions don't rely on the reflection's text."""
    output = await request.execution.runtime.bash("cat /rules/*.md")
    request.execution.submitted(output, channel="tool_output")
    success = request.case.protected or "calculator" in output
    return Outcome(
        scores={"task_completion": float(success), "coherence": 1.0},
        success=success,
        evidence={"expected": request.case.input["a"] + 1},
    )


async def test_replay_fresh_controls_delivery_and_acceptance(tmp_path):
    runtime, proposal, cases = setup(tmp_path)
    verdict = await runtime.validate(proposal.id, cases=cases, replay=measured)
    assert verdict.outcome == "accepted"
    assert "math.md" in runtime.workspace.rules()
    ids = [arm["execution_id"] for row in verdict.cases for arm in (row["control"], row["candidate"])]
    assert len(set(ids)) == 6
    assert all(not row["control"]["candidate_ids"] for row in verdict.cases)
    assert not runtime._evaluator.tasks
    assert await runtime.settle_validation()
    await runtime.aclose()


@pytest.mark.parametrize(
    "fault", ["empty", "missing_success", "missing_metric", "undelivered", "same_outcome", "no_protected"]
)
async def test_incomplete_unexposed_or_unimproved_is_pending(tmp_path, fault):
    runtime, proposal, cases = setup(tmp_path)

    async def replay(request):
        if fault == "empty":
            return Outcome()
        if fault == "undelivered":
            # Making a file readable or calling cat is not model submission.
            await request.execution.runtime.bash("cat /rules/*.md")
            return Outcome(
                scores={
                    "task_completion": float(request.arm == "candidate" or request.case.protected),
                    "coherence": 1,
                },
                success=True,
            )
        outcome = await measured(request)
        if fault == "missing_success":
            return Outcome(scores=outcome.scores)
        if fault == "missing_metric":
            return Outcome(scores={"coherence": 1}, success=True)
        if fault == "same_outcome":
            return Outcome(scores={"task_completion": 1, "coherence": 1}, success=True)
        return outcome

    if fault == "no_protected":
        cases = [ReplayCase(id=c.id, input=c.input) for c in cases]
    verdict = await runtime.validate(proposal.id, cases=cases, replay=replay)
    assert verdict.outcome == "pending"
    assert not runtime.workspace.rules()
    await runtime.aclose()


async def test_protected_regression_rejects_even_when_failure_cases_improve(tmp_path):
    runtime, proposal, cases = setup(tmp_path)

    async def replay(request):
        outcome = await measured(request)
        if request.case.protected and request.arm == "candidate":
            return Outcome(scores={"task_completion": 0, "coherence": 1}, success=False)
        return outcome

    verdict = await runtime.validate(proposal.id, cases=cases, replay=replay)
    assert verdict.outcome == "rejected" and not runtime.workspace.rules()
    await runtime.aclose()


async def test_changed_snapshot_during_replay_is_not_evidence_for_original_candidate(tmp_path):
    runtime, proposal, cases = setup(tmp_path)

    async def replay(request):
        result = await measured(request)
        await runtime.bash("rm -rf /rules")
        return result

    verdict = await runtime.validate(proposal.id, cases=cases, replay=replay)
    assert verdict.outcome == "pending" and "failed" in verdict.reason.lower()
    await runtime.aclose()


async def test_concurrent_accepted_revision_prevents_stale_publication(tmp_path):
    runtime, proposal, cases = setup(tmp_path)
    changed = False

    async def replay(request):
        nonlocal changed
        if not changed:
            changed = True
            other = runtime.workspace.propose(
                proposal.base_revision,
                {"other.md": Rule(path="other.md", text=TEXT.replace("Arithmetic", "Other"))},
                session_id="other",
                execution_id="other",
            ).proposal
            runtime.workspace.decide(
                ValidationVerdict(proposal_id=other.id, outcome="accepted", mode="replay", reason="fixture")
            )
        return await measured(request)

    verdict = await runtime.validate(proposal.id, cases=cases, replay=replay)
    assert verdict.outcome == "pending" and "rebase" in verdict.reason
    assert set(runtime.workspace.rules()) == {"other.md"}
    await runtime.aclose()


async def test_replay_runs_after_task_resource_barrier_and_is_cancellable(tmp_path):
    runtime, proposal, cases = setup(tmp_path)
    resource = asyncio.Lock()

    async def replay(request):
        async with resource:
            return await measured(request)

    async with resource:
        task = runtime.validate(proposal.id, cases=cases, replay=replay)
        assert not await runtime.settle_validation(timeout=0.01)
    assert (await task).outcome == "accepted"
    await runtime.aclose()


async def test_cancelled_replay_persists_pending_and_releases_context(tmp_path):
    runtime, proposal, cases = setup(tmp_path)
    started = asyncio.Event()

    async def replay(request):
        started.set()
        await asyncio.Event().wait()

    task = runtime.validate(proposal.id, cases=cases, replay=replay)
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.workspace.proposal(proposal.id).status == "pending"
    assert runtime._current.get() is None
    await runtime.aclose()


async def test_removal_requires_old_rule_exposure_and_noninferiority(tmp_path):
    runtime, proposal, cases = setup(tmp_path)
    runtime.workspace.decide(
        ValidationVerdict(proposal_id=proposal.id, outcome="accepted", mode="replay", reason="fixture")
    )
    removal = runtime.workspace.propose(
        runtime.workspace.state()["head"], {}, session_id="s", execution_id="remove"
    ).proposal

    async def replay(request):
        output = await runtime.bash("cat /rules/*.md")
        request.execution.submitted(output, channel="tool_output")
        return Outcome(scores={"task_completion": 1, "coherence": 1}, success=True)

    verdict = await runtime.validate(removal.id, cases=cases, replay=replay)
    assert verdict.outcome == "accepted" and "no behavioral improvement claimed" in verdict.reason
    assert not runtime.workspace.rules()
    await runtime.aclose()


async def test_live_trials_require_explicit_paired_assignment_and_cannot_reuse_execution(tmp_path):
    runtime, proposal, cases = setup(tmp_path)
    from pandaprobe.reflect import TrialArm, TrialCase

    records = []
    for case in cases:
        arms = {}
        for arm in ("control", "candidate"):
            async with runtime.trial_execution(proposal.id, arm=arm) as execution:
                output = await runtime.bash("cat /rules/*.md")
                execution.submitted(output, channel="tool_output")
                success = arm == "candidate" or case.protected
            arms[arm] = TrialArm(
                execution_id=execution.id,
                scores={"task_completion": float(success), "coherence": 1},
                success=success,
                delivered_versions=sorted(execution.delivered_versions),
                candidate_ids=execution.shell.candidate_ids,
            )
        record = TrialCase(case_id=case.id, base_revision=proposal.base_revision, protected=case.protected, **arms)
        records.append(record)
    first = runtime.validator.record_trial(proposal.id, records[0])
    assert first.outcome == "pending"
    assert runtime.validator.record_trial(proposal.id, records[0]) == first
    with pytest.raises(ValueError, match="different evidence"):
        runtime.validator.record_trial(proposal.id, records[0].model_copy(update={"protected": True}))
    assert runtime.validator.record_trial(proposal.id, records[1]).outcome == "pending"
    final = runtime.validator.record_trial(proposal.id, records[2])
    assert final.outcome == "accepted" and final.mode == "trial"
    with pytest.raises(ValueError, match="decided"):
        runtime.trial_execution(proposal.id, arm="candidate")
    await runtime.aclose()


def test_live_trials_reject_bad_shapes_without_persisting_them(tmp_path):
    runtime, proposal, _cases = setup(tmp_path)
    with pytest.raises(ValueError):
        runtime.validator.record_trial(
            proposal.id,
            {
                "case_id": "one",
                "base_revision": proposal.base_revision,
                "control": {"execution_id": "x", "scores": {"task_completion": float("nan")}},
            },
        )
    assert not runtime.workspace.state()["trials"]
