"""Explicit application boundaries and execution-local, editable memory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .models import DeliveryReceipt, ProposalOutcome, Rule, digest
from .shell import ReflectShell
from .workspace import WorkspaceError

if TYPE_CHECKING:
    from .runtime import Reflect

logger = logging.getLogger(__name__)

INSTRUCTIONS = """Reflect memory is available through reflect_bash.
Search /rules for guidance relevant to the task; read matching files as needed.
Use /context.md for concise active memory. Its current contents are incorporated
at invocation checkpoints. Follow the task's instructions over stored guidance.
At a reflection checkpoint, inspect /inbox and improve or consolidate /rules.
Each rule is a Markdown file with a '# Title', an 'Applies to: ...' line, and
actionable guidance. Add/edit/move/remove files normally; the SDK captures the
change set and independently validates it. Files are proposals, not proof of
success. /decisions contains outcomes. Avoid duplicates and discard obsolete
working guidance. Task work comes first; no polling or acknowledgement is needed.
"""


class Execution:
    """One pinned filesystem and identity. It is never a process-global router."""

    def __init__(
        self,
        runtime: Reflect,
        *,
        session_id: str,
        execution_id: str | None = None,
        completed: bool = True,
        role: str = "task",
        revision: str | None = None,
        rules: dict[str, Rule] | None = None,
        candidate_ids: list[str] | None = None,
        restore: bool = False,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must be nonempty")
        self.runtime, self.session_id = runtime, session_id
        self.id = execution_id or uuid4().hex
        self.role, self.completed = role, completed
        self.base_revision = revision or runtime.workspace.state()["head"]
        self.trace_ids: list[str] = []
        self.proposal = ProposalOutcome(status="no_proposal", reason="No checkpoint completed")
        self.status = "created"
        self.task_status = "created"
        self.phase_traces: dict[str, list[str]] = {}
        self._entered = False
        self._scope = None
        self._token = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._candidate_ids = list(candidate_ids or [])
        prior = runtime.workspace.load_execution(self.id)
        context = ""
        if prior and not restore:
            raise ValueError("Execution IDs must be unique; use continue_reflection to resume pending work")
        if restore:
            if prior is None or prior["session_id"] != session_id:
                raise ValueError("Cannot resume an unknown execution/session")
            self.base_revision = prior["base_revision"]
            rules = {path: Rule(path=path, text=text) for path, text in prior["files"].items()}
            context, self.trace_ids = prior["context"], prior["trace_ids"]
            self.task_status = prior.get("task_status", prior["status"])
            self.phase_traces = prior.get("phase_traces", {})
            self._candidate_ids = prior.get("candidate_ids", [])
        elif role == "task":
            memory = runtime.workspace.state().get("memory", {}).get(digest(session_id), {})
            if memory.get("revision") == self.base_revision:
                context = memory.get("context", "")
        self.shell = ReflectShell(
            runtime.config,
            rules if rules is not None else runtime.workspace.rules(self.base_revision),
            execution_id=self.id,
            context=context,
            candidate_ids=self._candidate_ids,
            on_checkpoint=self._checkpoint,
            on_receipt=self._receipt,
            usage=prior.get("usage") if restore else None,
        )
        if restore:
            self.shell.receipts = [DeliveryReceipt.model_validate(row) for row in prior.get("deliveries", [])]
        decisions = runtime.workspace.state()["proposals"]
        self.shell.set_decisions(
            {
                key: {"status": row["status"], "reason": (row["decision"] or {}).get("reason", "")}
                for key, row in list(decisions.items())[-20:]
            }
        )

    def _receipt(self, receipt: Any, text: str) -> None:
        self.runtime.workspace.event("guidance_delivery", **receipt.model_dump(mode="json"))

    def _checkpoint(self, files: dict[str, str], context: str) -> None:
        self.runtime.workspace.save_execution(
            self.id,
            {
                "execution_id": self.id,
                "session_id": self.session_id,
                "role": self.role,
                "base_revision": self.base_revision,
                "files": files,
                "context": context,
                "active_memory_hash": digest(context),
                "active_memory_unvalidated": bool(context),
                "trace_ids": self.trace_ids,
                "status": self.status,
                "task_status": self.task_status,
                "phase_traces": self.phase_traces,
                "usage": self.shell.usage,
                "candidate_ids": self.shell.candidate_ids,
                "proposal": self.proposal.model_dump(mode="json"),
                "deliveries": [receipt.model_dump(mode="json") for receipt in self.shell.receipts],
            },
        )

    def instructions(self) -> str:
        """Build one replaceable memory block. Reading this does not prove delivery."""
        context = self.shell.context()
        return INSTRUCTIONS + (f"\n<reflect_active_memory>\n{context}\n</reflect_active_memory>" if context else "")

    def submitted(self, content: str, *, channel: str = "active_memory") -> None:
        """Host attestation: this exact content was included in a model request.

        Tool execution alone is not this attestation. Native adapters attest the
        input they actually supply; custom hosts call this at their model boundary.
        """
        self.shell.receipt(content, channel=channel, submitted=True)

    def observe_model_input(self, value: Any) -> None:
        """Record exact strings seen at an explicit model-input boundary."""
        from .integrations import input_text

        self.submitted(input_text(value), channel="tool_output")

    def _observe_trace(self, payload: dict[str, Any]) -> None:
        for span in payload.get("spans", []):
            if span.get("kind") == "LLM" and span.get("input") is not None:
                self.observe_model_input(span["input"])

    @property
    def delivered_versions(self) -> set[str]:
        return {
            version
            for receipt in self.shell.receipts
            if receipt.submitted
            for version in receipt.rule_versions.values()
        }

    def seal(self, *, evidence_ids: list[str] | None = None, rationale: str = "") -> ProposalOutcome:
        try:
            self.proposal = self.runtime.workspace.propose(
                self.base_revision,
                self.shell.rules(),
                session_id=self.session_id,
                execution_id=self.id,
                evidence_ids=evidence_ids,
                rationale=rationale,
            )
            if self.proposal.proposal:
                self.shell.candidate_ids = list(dict.fromkeys([*self.shell.candidate_ids, self.proposal.proposal.id]))
            # Unvalidated authoring memory never leaks into a fresh control/task.
            if self.role in ("task", "reflection") and self.proposal.status == "no_proposal":
                memory = {"revision": self.base_revision, "context": self.shell.context()}
                self.runtime.workspace.transaction(
                    lambda state: state.setdefault("memory", {}).update({digest(self.session_id): memory})
                )
        except (ValueError, WorkspaceError) as exc:
            self.proposal = ProposalOutcome(status="invalid", reason=str(exc))
        self.proposal = self.proposal.model_copy(update={"rationale": rationale[:4096]})
        self._checkpoint(self.shell.files(), self.shell.context())
        return self.proposal

    @asynccontextmanager
    async def activate(self, role: str):
        """Host-controlled phase scope. Reflection cannot attribute itself as task."""
        token = self.runtime._current.set(self)
        self.loop = asyncio.get_running_loop()
        previous_role = self.role
        self.role = role
        scope = self.runtime.client.capture_traces(
            metadata={
                "pandaprobe.reflect": {"role": role, "execution_id": self.id, "task_session_id": self.session_id}
            },
            session_id=self.session_id if role == "task" else f"reflect:{role}:{self.id}",
            max_traces=self.runtime.config.evaluation_max_traces + 1,
            on_submit=self._observe_trace,
        )
        try:
            with scope:
                yield self
        finally:
            self.runtime._current.reset(token)
            self.role = previous_role
            if role == "task":
                self.trace_ids = list(dict.fromkeys([*self.trace_ids, *scope.trace_ids]))
            self.phase_traces[role] = list(dict.fromkeys([*self.phase_traces.get(role, []), *scope.trace_ids]))
            self.runtime.workspace.event(
                "execution_phase",
                execution_id=self.id,
                role=role,
                trace_ids=scope.trace_ids,
                capture_overflow=scope.overflow,
            )

    async def __aenter__(self):
        if self._entered or self.runtime._closed:
            raise RuntimeError("Execution cannot be entered twice or after Reflect is closed")
        self._entered, self.status = True, "running"
        if self.role == "task":
            self.task_status = "running"
        self._scope = self.activate(self.role)
        await self._scope.__aenter__()
        self._checkpoint(self.shell.files(), self.shell.context())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Cleanup must not overwrite the agent's result or its original exception.
        try:
            if self._scope is not None:
                await self._scope.__aexit__(exc_type, exc, tb)
            self.status = "failed" if exc is not None else "completed"
            if self.role == "task":
                self.task_status = self.status
                self.seal()
                self.runtime._schedule_validation(self.proposal)
                self.runtime.submit_traces(
                    self.trace_ids, execution_id=self.id, session_id=self.session_id, completed=self.completed
                )
            else:
                self._checkpoint(self.shell.files(), self.shell.context())
        except Exception:
            logger.exception("Reflect could not finish execution %s", self.id)
        finally:
            try:
                await self.shell.aclose()
            except Exception:
                logger.exception("Reflect shell cleanup failed for %s", self.id)
