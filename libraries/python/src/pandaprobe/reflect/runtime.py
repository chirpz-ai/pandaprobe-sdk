"""Explicit Reflect orchestration; no provider patches or package-owned model."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pandaprobe.api import PlatformClient
from pandaprobe.client import Client, get_client

from .config import ReflectConfig
from .evaluation import Evaluator
from .execution import Execution
from .models import EvaluationOutcome, ExecutionReport, ProposalOutcome, RuleProposal, digest
from .workspace import Workspace

logger = logging.getLogger(__name__)


class _HostCancelled(Exception):
    """A host continuation cancelled itself, independently of its caller."""


@dataclass(frozen=True)
class Invocation:
    """Generic host integration. The host retains its model, tools and agent loop."""

    input: Any
    execution: Execution
    phase: str = "task"
    previous_result: Any = None
    feedback: str | None = None

    @property
    def instructions(self) -> str:
        return self.execution.instructions()


class Reflect:
    def __init__(
        self,
        client: Client | None = None,
        *,
        config: ReflectConfig | None = None,
        api: PlatformClient | None = None,
        replay=None,
        cases=None,
    ) -> None:
        self.client = client if client is not None else get_client()
        if (replay is None) != (cases is None):
            raise ValueError("Provide replay and its protected cases together")
        self._replay, self._cases = replay, list(cases) if cases is not None else None
        if self.client is None:
            raise ValueError("Initialize PandaProbe first, or pass its Client to Reflect")
        self.config = config or ReflectConfig()
        # Instantiate only after explicit opt-in. Reuse the exact resolved identity.
        identity = self.client.config
        namespace = digest(f"{identity.endpoint.rstrip('/')}\n{identity.project_name}\n{identity.api_key}")[:24]
        self.workspace = Workspace(Path(self.config.workspace) / namespace, self.config)
        self.api = api if api is not None else PlatformClient(self.client)
        self._owns_api = api is None
        self._current: ContextVar[Execution | None] = ContextVar(f"reflect_execution_{id(self)}", default=None)
        self._evaluator = Evaluator(self.api, self.workspace, self.config)
        self._reflection_tasks: dict[str, asyncio.Task[ProposalOutcome]] = {}
        self._validation_tasks: set[asyncio.Task] = set()
        self._validation_by_proposal: dict[str, asyncio.Task] = {}
        self._host_tasks: set[asyncio.Task] = set()
        self._closed = False
        from .validation import Validator

        self.validator = Validator(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def execution(self, *, session_id: str, execution_id: str | None = None, completed: bool = True) -> Execution:
        """Explicit application boundary; framework callbacks do not define it."""
        return Execution(self, session_id=session_id, execution_id=execution_id, completed=completed)

    async def bash(self, command: str) -> str:
        """The single supplied tool, bound to this instance and invocation context."""
        execution = self._current.get()
        if execution is None:
            return "Reflect Bash requires an active reflect.run(...) or reflect.execution(...) boundary."
        return await execution.shell.execute(command)

    def tool(self, framework: str = "generic") -> Any:
        from .integrations import make_tool

        return make_tool(self, framework)

    def agent(self, factory, *args: Any, **kwargs: Any) -> Any:
        """Construct a native agent with Reflect's supplied Bash already attached."""
        from .integrations import build_agent

        return build_agent(self, factory, *args, **kwargs)

    def submit_traces(
        self,
        trace_ids: list[str],
        *,
        execution_id: str,
        session_id: str,
        completed: bool = True,
    ) -> asyncio.Task[EvaluationOutcome]:
        if self._closed:
            raise RuntimeError("Reflect is closed")
        return self._evaluator.submit(trace_ids, execution_id=execution_id, session_id=session_id, completed=completed)

    async def wait_export(self, *, timeout: float = 30) -> None:
        """Wait for SDK export separately from persistence and evaluation."""
        await asyncio.to_thread(self.client.flush, timeout=timeout)

    async def settle_evaluation(self, execution_id: str, *, timeout: float | None = None) -> EvaluationOutcome:
        task = self._evaluator.tasks.get(execution_id)
        if task is None or (
            task.done()
            and not task.cancelled()
            and task.exception() is None
            and task.result().status in ("pending", "cancelled")
        ):
            row = self.workspace.state().get("executions", {}).get(execution_id)
            if row is None:
                raise KeyError(execution_id)
            request = row["request"]
            task = self.submit_traces(
                request["trace_ids"],
                execution_id=execution_id,
                session_id=request["session_id"],
                completed=request["completed"],
            )
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            return EvaluationOutcome(execution_id=execution_id, status="pending", reason="Caller wait deadline")

    def resume_evaluations(self) -> list[asyncio.Task[EvaluationOutcome]]:
        if self._closed:
            raise RuntimeError("Reflect is closed")
        return self._evaluator.resume()

    async def continue_reflection(
        self,
        execution_id: str,
        continuation: Callable[[Invocation], Awaitable[Any]],
        *,
        input: Any = None,
        previous_result: Any = None,
    ) -> ProposalOutcome:
        """Deliver late feedback only with an explicit host continuation."""
        if self._closed:
            raise RuntimeError("Reflect is closed")
        existing = self._reflection_tasks.get(execution_id)
        if (
            existing is not None
            and existing.done()
            and not existing.cancelled()
            and existing.exception() is None
            and existing.result().reason.startswith("No assigned failure notice")
        ):
            existing = None
        if existing is None:
            task = asyncio.create_task(self._reflect(execution_id, continuation, input, previous_result))
            self._reflection_tasks[execution_id] = task
        else:
            task = existing
        return await asyncio.shield(task)

    async def _reflect(self, execution_id: str, continuation, input: Any, previous_result: Any) -> ProposalOutcome:
        evaluation = await self.settle_evaluation(execution_id)
        if evaluation.notice is None:
            return ProposalOutcome(status="no_proposal", reason=f"No assigned failure notice ({evaluation.status})")
        notice = evaluation.notice

        # A durable claim prevents concurrent processes/restarts from repeating administration.
        def claim(state):
            row = state["executions"][execution_id]
            if "reflection" in row:
                return False, row["reflection"]
            row["reflection"] = {"status": "running"}
            return True, row["reflection"]

        owner, record = self.workspace.transaction(claim)
        if not owner:
            if "outcome" in record:
                return ProposalOutcome.model_validate(record["outcome"])
            return ProposalOutcome(
                status="failed", reason="Reflection already claimed; interrupted work requires review"
            )
        execution = None
        try:
            execution = Execution(
                self,
                session_id=notice.session_id,
                execution_id=execution_id,
                restore=self.workspace.load_execution(execution_id) is not None,
                role="reflection",
            )
            evidence_path = self.workspace.path / "evidence" / f"{notice.id}.txt"
            evidence = evidence_path.read_text(encoding="utf-8") if evidence_path.exists() else "Evidence unavailable"
            execution.shell.set_evidence(notice.model_dump(mode="json"), evidence)
            execution.shell.begin_reflection()
            feedback = (
                "The task invocation has finished. Reflect on the external evidence in /inbox now. "
                "Judge explanations may be wrong: compare them with execution evidence. Search existing /rules, "
                "then make at most one coherent improvement or maintenance change set, or leave files unchanged "
                "if the issue is already covered, uncertain, or unactionable. Keep /context.md concise. "
                "Do not repeat task actions or poll for validation. Finish within this checkpoint."
            )
            async with execution.activate("reflection"):
                result = await self._call_host(
                    continuation(
                        Invocation(
                            input=input,
                            execution=execution,
                            phase="reflection",
                            previous_result=previous_result,
                            feedback=feedback,
                        )
                    ),
                    timeout=self.config.reflection_timeout,
                )
            from .integrations import result_text

            outcome = execution.seal(evidence_ids=[notice.id], rationale=result_text(result))
        except asyncio.TimeoutError:
            outcome = ProposalOutcome(status="timed_out", reason="Reflection checkpoint deadline exceeded")
        except _HostCancelled:
            outcome = ProposalOutcome(status="cancelled", reason="Host reflection continuation cancelled itself")
        except asyncio.CancelledError:
            outcome = ProposalOutcome(status="cancelled", reason="Reflection checkpoint cancelled")
            self._save_reflection(execution_id, outcome)
            raise
        except Exception as exc:
            logger.exception("Reflect checkpoint failed for %s", execution_id)
            outcome = ProposalOutcome(status="failed", reason=type(exc).__name__)
        finally:
            if execution is not None:
                await execution.shell.aclose()
        self._save_reflection(execution_id, outcome)
        return outcome

    def _save_reflection(self, execution_id: str, outcome: ProposalOutcome) -> None:
        self.workspace.transaction(
            lambda state: state["executions"][execution_id].update(
                {"reflection": {"status": "completed", "outcome": outcome.model_dump(mode="json")}}
            )
        )
        self.workspace.event("reflection_outcome", execution_id=execution_id, **outcome.model_dump(mode="json"))
        self._schedule_validation(outcome)

    def _schedule_validation(self, outcome: ProposalOutcome) -> None:
        if outcome.status == "proposed" and outcome.proposal is not None and self._replay is not None:
            self.validate(outcome.proposal.id, cases=self._cases, replay=self._replay)

    def report(self, execution_id: str) -> ExecutionReport:
        """Inspect lifecycle and evidence without conflating proposals with repair."""
        snapshot = self.workspace.load_execution(execution_id) or {}
        state = self.workspace.state()
        record = state.get("executions", {}).get(execution_id)
        if record is None and not snapshot:
            raise KeyError(execution_id)
        record = record or {}
        return ExecutionReport(
            execution_id=execution_id,
            session_id=snapshot.get("session_id", record.get("request", {}).get("session_id", "")),
            task_status=snapshot.get("task_status", "externally_submitted"),
            trace_ids=snapshot.get("trace_ids", record.get("request", {}).get("trace_ids", [])),
            evaluation=record.get("outcome"),
            reflection=record.get("reflection", {}).get("outcome"),
            proposals=[p for p in state["proposals"].values() if p["execution_id"] == execution_id],
            deliveries=snapshot.get("deliveries", []),
        )

    def proposals(self) -> list[RuleProposal]:
        """Return versioned proposals and their current independent decisions."""
        return [RuleProposal.model_validate(row) for row in self.workspace.state()["proposals"].values()]

    async def _call_host(self, awaitable, *, timeout: float) -> Any:
        """Bound our wait even if host code defers cancellation; never force-kill it."""
        task = asyncio.create_task(awaitable)
        self._host_tasks.add(task)

        def finished(done):
            self._host_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if not done:
                task.cancel()
                raise asyncio.TimeoutError
            if task.cancelled():
                raise _HostCancelled("Host operation cancelled itself")
            return task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def settle_reflection(self, execution_id: str, *, timeout: float | None = None) -> ProposalOutcome:
        task = self._reflection_tasks.get(execution_id)
        if task is None:
            return ProposalOutcome(status="no_proposal", reason="No host continuation was scheduled")
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            return ProposalOutcome(status="timed_out", reason="Caller wait deadline; checkpoint is still running")

    async def run(
        self,
        agent: Any,
        input: Any,
        *,
        session_id: str,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the developer's agent, then offer it a bounded reflection checkpoint.

        Native adapters supply memory and continuation semantics. Custom agents
        accept an Invocation and remain responsible for actual model submission.
        The original task result is always returned unchanged.
        """
        from .integrations import invoker_for

        invoke = invoker_for(agent, **kwargs)
        active = self._current.get()
        if active is not None and active.role == "validation":
            return await invoke(Invocation(input=input, execution=active, phase="validation"))
        async with self.execution(session_id=session_id, execution_id=execution_id) as execution:
            result = await invoke(Invocation(input=input, execution=execution))
        try:
            await self.continue_reflection(execution.id, invoke, input=input, previous_result=result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Reflect settlement failed for %s", execution.id)
        return result

    async def settle_validation(self, *, timeout: float | None = None) -> bool:
        """Wait separately, after the host releases resources replay may need."""
        if not self._validation_tasks:
            return True
        _, pending = await asyncio.wait(self._validation_tasks, timeout=timeout)
        return not pending

    def validate(self, proposal_id: str, *, cases, replay) -> asyncio.Task:
        """Schedule independent replay. Never await this while holding task resources."""
        if self._closed:
            raise RuntimeError("Reflect is closed")
        existing = self._validation_by_proposal.get(proposal_id)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(self.validator.replay(proposal_id, cases, replay))
        self._validation_by_proposal[proposal_id] = task
        self._validation_tasks.add(task)
        return task

    def trial_execution(self, proposal_id: str, *, arm: str, session_id: str | None = None) -> Execution:
        """Fresh explicit trial exposure; no candidate enters ordinary accepted views."""
        if arm not in ("control", "candidate"):
            raise ValueError("Trial arm must be control or candidate")
        proposal = self.workspace.proposal(proposal_id)
        if proposal.status in ("accepted", "rejected"):
            raise ValueError("A decided proposal cannot start a new trial")
        from uuid import uuid4

        return Execution(
            self,
            session_id=session_id or f"reflect:trial:{uuid4().hex}",
            role="validation",
            revision=proposal.base_revision,
            rules=self.workspace.rules(proposal.base_revision)
            if arm == "control"
            else self.workspace.candidate_rules(proposal),
            candidate_ids=[proposal.id] if arm == "candidate" else [],
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = (
            list(self._evaluator.tasks.values())
            + list(self._reflection_tasks.values())
            + list(self._validation_tasks)
            + list(self._host_tasks)
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.config.shutdown_timeout)
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                logger.warning("Reflect closed with %d host operations still cancelling", len(pending))
        if self._owns_api:
            await self.api.aclose()
