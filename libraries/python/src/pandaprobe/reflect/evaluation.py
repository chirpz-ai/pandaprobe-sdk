"""Bounded native evaluation, with durable single-attempt run creation."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pandaprobe.api import PlatformClient
from pandaprobe.api.errors import (
    ApiDeadlineExceeded,
    CreationOutcomeUnknown,
    InsufficientEvidence,
    PlatformError,
)
from pandaprobe.api.models import EvaluationResult, Trace

from .config import ReflectConfig
from .models import EvaluationOutcome, digest
from .shell import clip_bytes
from .signals import SignalGate
from .workspace import Workspace, atomic_write

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    pass


class Evaluator:
    """One runtime's scheduler. Workspace transactions coordinate billable claims."""

    def __init__(self, api: PlatformClient, workspace: Workspace, config: ReflectConfig) -> None:
        self.api, self.workspace, self.config = api, workspace, config
        self.gate = SignalGate(workspace, config)
        self._slots = asyncio.Semaphore(config.max_concurrency)
        self.tasks: dict[str, asyncio.Task[EvaluationOutcome]] = {}

    def submit(
        self,
        trace_ids: list[str],
        *,
        execution_id: str,
        session_id: str,
        completed: bool = True,
    ) -> asyncio.Task[EvaluationOutcome]:
        ids = sorted({str(UUID(str(value))) for value in trace_ids})
        if not execution_id or not session_id:
            raise ValueError("Execution and session identities must be nonempty")
        request = {
            "trace_ids": ids,
            "session_id": session_id,
            "completed": completed,
            "thresholds": dict(self.config.thresholds),
            "diagnostics": list(self.config.diagnostic_metrics),
        }

        def register(state: dict[str, Any]) -> None:
            requests = state.setdefault("executions", {})
            existing = requests.get(execution_id)
            if existing and existing["request"] != request:
                raise ValueError("An execution ID cannot identify different evidence or policies")
            requests.setdefault(execution_id, {"request": request})
            owners = state.setdefault("trace_owners", {})
            if any(trace_id in owners and owners[trace_id] != execution_id for trace_id in ids):
                requests[execution_id]["outcome"] = EvaluationOutcome(
                    execution_id=execution_id,
                    status="skipped",
                    reason="Trace evidence already belongs to another execution",
                ).model_dump(mode="json")
            else:
                owners.update({trace_id: execution_id for trace_id in ids})

        self.workspace.transaction(register)
        existing = self.tasks.get(execution_id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(self._evaluate(ids, execution_id, session_id, completed))
        self.tasks[execution_id] = task
        return task

    def _save(self, outcome: EvaluationOutcome) -> EvaluationOutcome:
        def update(state: dict[str, Any]) -> None:
            state["executions"][outcome.execution_id]["outcome"] = outcome.model_dump(mode="json")

        self.workspace.transaction(update)
        self.workspace.event("evaluation_outcome", **outcome.model_dump(mode="json"))
        return outcome

    async def _run(self, trace_ids: list[str], metrics: list[str], execution_id: str, stage: str) -> EvaluationResult:
        key = digest(json.dumps([execution_id, stage, trace_ids, metrics], sort_keys=True))

        def claim(state: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
            runs = state["evaluations"]
            if key in runs:
                return False, runs[key]
            day = datetime.now(timezone.utc).date().isoformat()
            if state.get("evaluation_day") != day:
                state["evaluation_day"], state["evaluation_count"] = day, 0
            limit = self.config.evaluation_max_runs
            if limit is not None and state["evaluation_count"] >= limit:
                raise BudgetExceeded(
                    f"Daily evaluation-run budget reached ({state['evaluation_count']}/{limit} "
                    f"creation attempts on {day} UTC; resets at 00:00 UTC)"
                )
            record = {
                "status": "creating",
                "execution_id": execution_id,
                "stage": stage,
                "trace_ids": trace_ids,
                "metrics": metrics,
            }
            runs[key] = record
            state["evaluation_count"] += 1
            return True, record

        try:
            owner, record = self.workspace.transaction(claim)
        except BudgetExceeded as exc:
            logger.warning("Reflect %s evaluation skipped for execution %s: %s", stage, execution_id, exc)
            raise
        if record.get("status") == "completed":
            return EvaluationResult.model_validate(record["result"])

        def save(**values: Any) -> None:
            self.workspace.transaction(lambda state: state["evaluations"][key].update(values))

        if owner:
            try:
                run = await self.api.evaluations.create(
                    trace_ids,
                    metrics,
                    name=f"Reflect {stage}: {execution_id[:128]}",
                    timeout=self.config.evaluation_timeout,
                )
            except (CreationOutcomeUnknown, asyncio.CancelledError):
                save(status="creation_unknown")
                raise
            except PlatformError as exc:
                save(status="failed", error=type(exc).__name__)
                raise
            # There is no await between receiving the ID and its durable write.
            save(status="running", run_id=str(run.id))
            record["run_id"] = str(run.id)
        elif "run_id" not in record:
            if record["status"] == "failed":
                raise PlatformError("The original creation request failed; a new execution is required")
            raise CreationOutcomeUnknown("A run was already dispatched; its ID is unknown. Never recreate it")
        result = await self.api.evaluations.wait(
            record["run_id"],
            trace_ids=trace_ids,
            metrics=metrics,
            timeout=self.config.evaluation_timeout,
            poll_interval=self.config.poll_interval,
        )
        save(status="completed", result=result.model_dump(mode="json"))
        return result

    def _run_ids(self, execution_id: str) -> list[str]:
        return [
            row["run_id"]
            for row in self.workspace.state()["evaluations"].values()
            if row["execution_id"] == execution_id and "run_id" in row
        ]

    async def _evaluate(
        self, ids: list[str], execution_id: str, session_id: str, completed: bool
    ) -> EvaluationOutcome:
        previous = self.workspace.state()["executions"][execution_id].get("outcome")
        if previous and previous["status"] not in ("pending", "cancelled"):
            return EvaluationOutcome.model_validate(previous)
        scores: dict[str, float] = {}
        try:
            if not ids:
                return self._save(
                    EvaluationOutcome(
                        execution_id=execution_id,
                        status="insufficient",
                        reason="No completed task trace references were submitted",
                    )
                )
            if len(ids) > self.config.evaluation_max_traces:
                return self._save(
                    EvaluationOutcome(
                        execution_id=execution_id,
                        status="insufficient",
                        reason="Execution exceeds the trace evaluation limit",
                    )
                )
            if int(digest(execution_id)[:16], 16) / 2**64 >= self.config.evaluation_sample_rate:
                return self._save(EvaluationOutcome(execution_id=execution_id, status="skipped", reason="Sampling"))
            async with self._slots:
                gathered = await asyncio.gather(
                    *[
                        self.api.traces.wait_visible(
                            t, timeout=self.config.trace_visibility_timeout, poll_interval=self.config.poll_interval
                        )
                        for t in ids
                    ],
                    return_exceptions=True,
                )
                traces: list[Trace] = []
                for item in gathered:
                    if isinstance(item, BaseException):
                        raise item
                    traces.append(item)
                if any((t.metadata or {}).get("pandaprobe.reflect", {}).get("role", "task") != "task" for t in traces):
                    return self._save(
                        EvaluationOutcome(
                            execution_id=execution_id,
                            status="skipped",
                            reason="Internal reflection/validation evidence is excluded",
                        )
                    )
                primary = await self._run(ids, list(self.config.thresholds), execution_id, "detect")
                reasons: dict[str, str] = {}
                for score in primary.scores:
                    value = score.numeric_value
                    if value is None or not 0 <= value <= 1:
                        raise ValueError("Configured signal scores must be finite and normalized to [0, 1]")
                    # Worst trace preserves a meaningful failure within the application boundary.
                    if score.name not in scores or value < scores[score.name]:
                        scores[score.name] = value
                        reasons[score.name] = score.reason or ""
                notice = self.gate.observe(
                    execution_id=execution_id,
                    session_id=session_id,
                    trace_ids=ids,
                    scores=scores,
                    reasons=reasons,
                    completed=completed,
                )
                evidence: dict[str, Any] = {
                    "trace_ids": ids,
                    "primary": primary.model_dump(mode="json"),
                }
                diagnostic_reason = ""
                if notice and self.config.diagnostic_metrics:
                    try:
                        diagnostic = await self._run(
                            ids, list(self.config.diagnostic_metrics), execution_id, "diagnose"
                        )
                        evidence["diagnostic"] = diagnostic.model_dump(mode="json")
                    except (PlatformError, BudgetExceeded) as exc:
                        detail = str(exc) if isinstance(exc, BudgetExceeded) else type(exc).__name__
                        diagnostic_reason = f"Diagnostic evidence unavailable: {detail}"
                        evidence["diagnostic_error"] = diagnostic_reason
                if notice:
                    # Preserve diagnostic findings before potentially very large raw inputs.
                    # Presentation may be clipped; original trace/run IDs remain in state.
                    evidence["traces"] = [t.model_dump(mode="json") for t in traces]
                    body = json.dumps(evidence, ensure_ascii=False)
                    limit = self.config.evidence_max_bytes
                    if len(body.encode("utf-8")) > limit:
                        marker = (
                            "\n[EVIDENCE TRUNCATED: original trace/run IDs identify the complete platform records.]"
                        )
                        body = clip_bytes(body, limit - len(marker.encode("utf-8"))) + clip_bytes(marker, limit)
                    atomic_write(self.workspace.path / "evidence" / f"{notice.id}.txt", body)
                return self._save(
                    EvaluationOutcome(
                        execution_id=execution_id,
                        status="evaluated",
                        scores=scores,
                        run_ids=self._run_ids(execution_id),
                        notice=notice,
                        reason=diagnostic_reason,
                    )
                )
        except asyncio.CancelledError:
            self._save(
                EvaluationOutcome(
                    execution_id=execution_id,
                    status="cancelled",
                    run_ids=self._run_ids(execution_id),
                    reason="Evaluation wait cancelled",
                )
            )
            raise
        except (PlatformError, BudgetExceeded, ValueError) as exc:
            status = "failed"
            if isinstance(exc, CreationOutcomeUnknown):
                status = "creation_unknown"
            elif isinstance(exc, ApiDeadlineExceeded):
                status = "pending"
            elif isinstance(exc, (InsufficientEvidence, ValueError)):
                status = "insufficient"
            elif isinstance(exc, BudgetExceeded):
                status = "skipped"
            return self._save(
                EvaluationOutcome(
                    execution_id=execution_id,
                    status=status,
                    scores=scores,
                    run_ids=self._run_ids(execution_id),
                    reason=str(exc),
                )
            )

    def resume(self) -> list[asyncio.Task[EvaluationOutcome]]:
        """Resume only pending records; unknown creations remain explicitly unknown."""
        tasks = []
        for execution_id, row in self.workspace.state().get("executions", {}).items():
            if row.get("outcome", {}).get("status") in (None, "pending", "cancelled"):
                request = row["request"]
                if request["thresholds"] != dict(self.config.thresholds) or request["diagnostics"] != list(
                    self.config.diagnostic_metrics
                ):
                    continue
                tasks.append(
                    self.submit(
                        request["trace_ids"],
                        execution_id=execution_id,
                        session_id=request["session_id"],
                        completed=request["completed"],
                    )
                )
        return tasks
