"""Independent paired validation of proposed file changes, including deletions.

The host supplies reproducible inputs and an external verifier. The proposing
agent cannot call these operations through its isolated Bash filesystem.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from .execution import Execution
from .models import RuleProposal, TrialCase, ValidationVerdict, digest
from .workspace import RevisionConflict

if TYPE_CHECKING:
    from .runtime import Reflect


@dataclass(frozen=True)
class ReplayCase:
    id: str
    input: Any
    protected: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Replay case identity must be nonempty")
        # Persist a reproducible input fingerprint, never repr(object) or memory addresses.
        json.dumps(self.input, allow_nan=False)


@dataclass(frozen=True)
class Outcome:
    """Host-verifier evidence. None/empty is insufficient, never a success."""

    scores: dict[str, float] = field(default_factory=dict)
    success: bool | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success is not None and not isinstance(self.success, bool):
            raise ValueError("Outcome success must be a boolean or None")
        if any(
            not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for name, value in self.scores.items()
        ):
            raise ValueError("Outcome scores must be named, finite values in [0, 1]")
        json.dumps(self.evidence, allow_nan=False)


@dataclass(frozen=True)
class ReplayRequest:
    case: ReplayCase
    execution: Execution
    arm: Literal["control", "candidate"]

    async def run(self, agent: Any, **kwargs: Any) -> Any:
        """Invoke the same host agent on fresh input, without recursive evaluation."""
        from .integrations import invoker_for
        from .runtime import Invocation

        return await invoker_for(agent, **kwargs)(
            Invocation(input=self.case.input, execution=self.execution, phase="validation")
        )


Replay = Callable[[ReplayRequest], Awaitable[Outcome]]


class Validator:
    def __init__(self, runtime: Reflect) -> None:
        self.runtime = runtime

    async def replay(self, proposal_id: str, cases: list[ReplayCase], replay: Replay) -> ValidationVerdict:
        proposal = self.runtime.workspace.proposal(proposal_id)
        if proposal.status in ("accepted", "rejected"):
            return ValidationVerdict.model_validate(proposal.decision)
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("Replay case IDs must be unique")
        rows: list[dict[str, Any]] = []
        try:
            await asyncio.wait_for(
                self._collect(proposal, cases, replay, rows), self.runtime.config.validation_timeout
            )
            verdict = self.assess(proposal, rows, mode="replay")
        except asyncio.TimeoutError:
            verdict = ValidationVerdict(
                proposal_id=proposal_id,
                outcome="pending",
                mode="replay",
                reason="Replay deadline exceeded",
                cases=rows,
            )
        except asyncio.CancelledError:
            self._publish(
                ValidationVerdict(
                    proposal_id=proposal_id, outcome="pending", mode="replay", reason="Replay cancelled", cases=rows
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001 - arbitrary host replay errors are evidence gaps, not repair
            verdict = ValidationVerdict(
                proposal_id=proposal_id,
                outcome="pending",
                mode="replay",
                reason=f"Replay failed: {type(exc).__name__}",
                cases=rows,
            )
        return self._publish(verdict)

    async def _collect(self, proposal: RuleProposal, cases: list[ReplayCase], replay: Replay, rows: list) -> None:
        baseline = self.runtime.workspace.rules(proposal.base_revision)
        candidate = self.runtime.workspace.candidate_rules(proposal)
        for case in cases:
            row: dict[str, Any] = {
                "case_id": case.id,
                "input_hash": digest(json.dumps(case.input, sort_keys=True)),
                "protected": case.protected,
                "base_revision": proposal.base_revision,
            }
            rows.append(row)
            # Alternate order deterministically, avoiding a fixed candidate-last bias.
            arms = ("candidate", "control") if int(digest(case.id)[0], 16) % 2 else ("control", "candidate")
            for arm in arms:
                execution = Execution(
                    self.runtime,
                    session_id=f"reflect:validation:{uuid4().hex}",
                    role="validation",
                    revision=proposal.base_revision,
                    rules=baseline if arm == "control" else candidate,
                    candidate_ids=[proposal.id] if arm == "candidate" else [],
                )
                try:
                    async with execution:
                        outcome = await self.runtime._call_host(
                            replay(ReplayRequest(case=case, execution=execution, arm=arm)),
                            timeout=self.runtime.config.validation_timeout,
                        )
                    if not isinstance(outcome, Outcome):
                        raise TypeError("Replay must return an externally verified Outcome")
                    if execution.shell.rules() != (baseline if arm == "control" else candidate):
                        raise ValueError("Validation changed the assigned rule snapshot")
                    row[arm] = {
                        "execution_id": execution.id,
                        "scores": outcome.scores,
                        "success": outcome.success,
                        "evidence": outcome.evidence,
                        "delivered_versions": sorted(execution.delivered_versions),
                        "candidate_ids": execution.shell.candidate_ids,
                    }
                finally:
                    await execution.shell.aclose()

    def assess(self, proposal: RuleProposal, rows: list[dict[str, Any]], *, mode: str) -> ValidationVerdict:
        config = self.runtime.config

        def verdict(outcome: str, reason: str) -> ValidationVerdict:
            return ValidationVerdict(proposal_id=proposal.id, outcome=outcome, reason=reason, mode=mode, cases=rows)

        if len(rows) < config.validation_min_cases or len({r["case_id"] for r in rows}) != len(rows):
            return verdict("pending", "Insufficient distinct paired cases")
        if not any(row["protected"] for row in rows):
            return verdict("pending", "At least one protected successful case is required")
        required = set(config.thresholds)
        seen_executions: set[str] = set()
        deltas: list[float] = []
        control_versions: set[str] = set()
        candidate_versions: set[str] = set()
        for row in rows:
            if row.get("base_revision") != proposal.base_revision:
                return verdict("pending", "Case baseline does not match the proposed revision")
            if "control" not in row or "candidate" not in row:
                return verdict("pending", "Missing paired outcome")
            control, candidate = row["control"], row["candidate"]
            for arm in (control, candidate):
                if not arm.get("execution_id") or arm["execution_id"] in seen_executions:
                    return verdict("pending", "Execution evidence was reused across trial arms or cases")
                seen_executions.add(arm["execution_id"])
                scores = arm.get("scores", {})
                if (
                    required - scores.keys()
                    or not isinstance(arm.get("success"), bool)
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        or not 0 <= v <= 1
                        for v in scores.values()
                    )
                ):
                    return verdict("pending", "Missing or invalid independently verified outcomes")
            if control.get("candidate_ids") or candidate.get("candidate_ids") != [proposal.id]:
                return verdict("pending", "Trial exposure/assignment does not match control and candidate")
            if set(control["scores"]) != set(candidate["scores"]):
                return verdict("pending", "Paired metric sets differ")
            if row["protected"] and not control["success"]:
                return verdict("pending", "Protected case no longer succeeds under the frozen control")
            if control["success"] and not candidate["success"]:
                return verdict("rejected", "Candidate regressed a successful outcome")
            for name, before in control["scores"].items():
                change = candidate["scores"][name] - before
                if change < -config.validation_regression_tolerance:
                    return verdict("rejected", f"Candidate regressed {name} on case {row['case_id']}")
                if not row["protected"]:
                    deltas.append(change)
            control_versions.update(control.get("delivered_versions", []))
            candidate_versions.update(candidate.get("delivered_versions", []))
        baseline = self.runtime.workspace.rules(proposal.base_revision)
        for path, rule in proposal.changes.items():
            if rule is not None and rule.version not in candidate_versions:
                return verdict("pending", f"Changed guidance was not delivered: {path}")
            if rule is None and baseline[path].version not in control_versions:
                return verdict("pending", f"Removed guidance was not exercised by controls: {path}")
        if deltas and sum(deltas) / len(deltas) >= config.validation_min_improvement:
            return verdict(
                "accepted", "Observed paired improvement within configured regression tolerances in tested cases"
            )
        candidate_rules = self.runtime.workspace.candidate_rules(proposal)
        if sum(len(r.text.encode()) for r in candidate_rules.values()) < sum(
            len(r.text.encode()) for r in baseline.values()
        ):
            return verdict(
                "accepted", "Reduced guidance size with measured non-inferiority; no behavioral improvement claimed"
            )
        return verdict("pending", "No sufficient measured improvement")

    def _publish(self, verdict: ValidationVerdict) -> ValidationVerdict:
        try:
            proposal = self.runtime.workspace.decide(verdict)
            return ValidationVerdict.model_validate(proposal.decision)
        except RevisionConflict:
            pending = verdict.model_copy(
                update={"outcome": "pending", "reason": "Accepted guidance changed; rebase and revalidate"}
            )
            self.runtime.workspace.decide(pending)
            return pending

    def record_trial(self, proposal_id: str, case: TrialCase | dict[str, Any]) -> ValidationVerdict:
        """Record a host-assigned matched live trial; unrelated sessions never count.

        The host attests independent outcomes and actual submitted rule versions.
        Repeated case IDs with different data are rejected, not silently replaced.
        """
        proposal = self.runtime.workspace.proposal(proposal_id)
        if proposal.status in ("accepted", "rejected"):
            return ValidationVerdict.model_validate(proposal.decision)
        case = TrialCase.model_validate(case).model_dump(mode="json", exclude_none=True)
        json.dumps(case, allow_nan=False)

        def record(state):
            trials = state["trials"].setdefault(proposal_id, {})
            if case["case_id"] in trials and trials[case["case_id"]] != case:
                raise ValueError("A trial case cannot be replaced with different evidence")
            inserted = case["case_id"] not in trials
            trials[case["case_id"]] = case
            return list(trials.values()), inserted

        rows, inserted = self.runtime.workspace.transaction(record)
        if not inserted and proposal.decision is not None:
            return ValidationVerdict.model_validate(proposal.decision)
        return self._publish(self.assess(proposal, rows, mode="trial"))
