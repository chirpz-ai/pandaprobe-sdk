"""Durable filesystem authority, outside the agent's isolated Bash filesystem.

Immutable revisions are written before a single atomic state-pointer update.
Cross-process locking prevents two Reflect instances from publishing over one
another. A crash can leave an unreferenced revision, never a half-published one.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from .config import ReflectConfig
from .models import ProposalOutcome, Rule, RuleProposal, ValidationVerdict, digest, utc_now

T = TypeVar("T")


class WorkspaceError(Exception):
    pass


class RevisionConflict(WorkspaceError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".reflect-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Workspace:
    def __init__(self, path: Path, config: ReflectConfig) -> None:
        try:
            from filelock import FileLock
        except ImportError as exc:
            raise ImportError('Install Reflect with pip install "pandaprobe[reflect]"') from exc
        self.path = path.resolve()
        self.config = config
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.path / ".lock"), timeout=10)
        with self._lock:
            if not (self.path / "state.json").exists():
                for manifest in (self.path / "revisions").glob("*/manifest.json"):
                    if json.loads(manifest.read_text(encoding="utf-8")):
                        raise WorkspaceError(
                            "Workspace state is missing but nonempty history exists; restore its authoritative state"
                        )
                revision = self._write_revision({})
                self._write_state(
                    {
                        "schema_version": 1,
                        "head": revision,
                        "proposals": {},
                        "evaluations": {},
                        "trials": {},
                        "gates": {},
                        "notices": {},
                        "evaluation_count": 0,
                    }
                )
            self._read_state()

    def _read_state(self) -> dict[str, Any]:
        try:
            state = json.loads((self.path / "state.json").read_text(encoding="utf-8"))
            if state["schema_version"] != 1:
                raise WorkspaceError("Unsupported Reflect workspace version")
            return state
        except (ValueError, KeyError, TypeError) as exc:
            raise WorkspaceError("Reflect workspace state is corrupt; refusing to overwrite it") from exc

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write(self.path / "state.json", _json(state))

    def transaction(self, update: Callable[[dict[str, Any]], T]) -> T:
        """SDK-internal state transaction; no agent-facing administrative command."""
        with self._lock:
            state = self._read_state()
            result = update(state)
            self._write_state(state)
            return result

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._read_state()

    def _write_revision(self, rules: Mapping[str, Rule]) -> str:
        manifest = {path: rule.version for path, rule in sorted(rules.items())}
        revision = digest(_json(manifest))
        target = self.path / "revisions" / revision
        if target.exists():
            self.rules(revision)  # Reusing an orphan revision must still verify its integrity.
            return revision
        target.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".revision-", dir=target.parent))
        try:
            for path, rule in rules.items():
                atomic_write(temporary / "rules" / path, rule.text)
            atomic_write(temporary / "manifest.json", _json(manifest))
            os.replace(temporary, target)
            _sync_directory(target.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return revision

    def rules(self, revision: str | None = None) -> dict[str, Rule]:
        if revision is None:
            revision = self.state()["head"]
        if len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
            raise WorkspaceError("Invalid revision")
        folder = self.path / "revisions" / revision
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if digest(_json(manifest)) != revision:
            raise WorkspaceError("Revision manifest integrity check failed")
        result: dict[str, Rule] = {}
        for path, version in manifest.items():
            Rule.safe_path(path)
            file = folder / "rules" / path
            if file.is_symlink() or not file.resolve().is_relative_to(folder.resolve()):
                raise WorkspaceError("Revision contains a link outside its snapshot")
            text = file.read_text(encoding="utf-8")
            rule = Rule(path=path, text=text)
            if rule.version != version:
                raise WorkspaceError("Rule integrity check failed")
            result[path] = rule
        return result

    def check_rules(self, rules: Mapping[str, Rule]) -> None:
        if len(rules) > self.config.rules_max_count:
            raise WorkspaceError("Rule count limit reached; consolidate or remove obsolete rules")
        size = 0
        for path, rule in rules.items():
            if path != rule.path:
                raise WorkspaceError("Rule path and record disagree")
            count = len(rule.text.encode("utf-8"))
            if count > self.config.rule_max_bytes:
                raise WorkspaceError(f"Rule {path} exceeds its byte budget")
            size += count
        if size > self.config.rules_max_bytes:
            raise WorkspaceError("Rule corpus exceeds its byte budget")

    def propose(
        self,
        base_revision: str,
        rules: Mapping[str, Rule],
        *,
        session_id: str,
        execution_id: str,
        evidence_ids: list[str] | None = None,
        rationale: str = "",
    ) -> ProposalOutcome:
        self.check_rules(rules)
        baseline = self.rules(base_revision)
        changes = {
            path: rules.get(path)
            for path in sorted(baseline.keys() | rules.keys())
            if baseline.get(path) != rules.get(path)
        }
        if not changes:
            return ProposalOutcome(status="no_proposal", reason="The rule revision is unchanged")
        # Identical guidance under another filename is duplication, not a new rule.
        texts: dict[str, str] = {}
        for path, rule in rules.items():
            normalized = " ".join(rule.text.lower().split())
            if normalized in texts and (path in changes or texts[normalized] in changes):
                return ProposalOutcome(status="duplicate", reason=f"Guidance duplicates {texts[normalized]}")
            texts[normalized] = path
        fingerprint = digest(
            _json(
                {
                    "base": base_revision,
                    "changes": {path: rule.model_dump() if rule else None for path, rule in changes.items()},
                }
            )
        )
        with self._lock:
            state = self._read_state()
            if state["head"] != base_revision:
                return ProposalOutcome(status="conflict", reason="Accepted guidance changed; refresh before proposing")
            for record in state["proposals"].values():
                if record["fingerprint"] == fingerprint:
                    return ProposalOutcome(
                        status="duplicate",
                        proposal=RuleProposal.model_validate(record),
                        reason="This exact change set has already been considered",
                    )
            proposal = RuleProposal(
                base_revision=base_revision,
                changes=changes,
                fingerprint=fingerprint,
                session_id=session_id,
                execution_id=execution_id,
                evidence_ids=evidence_ids or [],
                rationale=rationale[:4096],
            )
            state["proposals"][proposal.id] = proposal.model_dump(mode="json")
            self._write_state(state)
        self.event("proposal_created", proposal_id=proposal.id, execution_id=execution_id)
        return ProposalOutcome(status="proposed", proposal=proposal)

    def proposal(self, proposal_id: str) -> RuleProposal:
        return RuleProposal.model_validate(self.state()["proposals"][proposal_id])

    def candidate_rules(self, proposal: RuleProposal) -> dict[str, Rule]:
        rules = self.rules(proposal.base_revision)
        for path, rule in proposal.changes.items():
            if rule is None:
                rules.pop(path, None)
            else:
                rules[path] = rule
        self.check_rules(rules)
        return rules

    def decide(self, verdict: ValidationVerdict) -> RuleProposal:
        """Called by the independent validator; never exposed inside Bash."""
        with self._lock:
            state = self._read_state()
            proposal = RuleProposal.model_validate(state["proposals"][verdict.proposal_id])
            if proposal.status in ("accepted", "rejected"):
                return proposal
            if verdict.outcome == "accepted":
                if state["head"] != proposal.base_revision:
                    raise RevisionConflict(
                        "Accepted revision changed during validation; candidate must be revalidated"
                    )
                revision = self._write_revision(self.candidate_rules(proposal))
                state["head"] = revision
            proposal = proposal.model_copy(
                update={"status": verdict.outcome, "decision": verdict.model_dump(mode="json")}
            )
            state["proposals"][proposal.id] = proposal.model_dump(mode="json")
            self._write_state(state)
        self.event("validation_decided", **verdict.model_dump(mode="json"))
        return proposal

    def event(self, kind: str, **data: Any) -> str:
        event_id = uuid4().hex
        record = {"id": event_id, "type": kind, "created_at": utc_now(), **data}
        atomic_write(self.path / "events" / f"{event_id}.json", _json(record))
        return event_id

    def save_execution(self, execution_id: str, data: dict[str, Any]) -> None:
        atomic_write(self.path / "executions" / f"{digest(execution_id)}.json", _json(data))

    def load_execution(self, execution_id: str) -> dict[str, Any] | None:
        path = self.path / "executions" / f"{digest(execution_id)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
