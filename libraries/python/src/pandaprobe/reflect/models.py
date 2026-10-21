"""Versioned evidence and lifecycle values; proposal completion is not repair."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rule(Record):
    """One Markdown file. Its directory is its scope; applicability is explicit."""

    path: str
    text: str

    @field_validator("path")
    @classmethod
    def safe_path(cls, path: str) -> str:
        if len(path) > 200 or not re.fullmatch(r"[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*\.md", path):
            raise ValueError("Rule paths must be relative Markdown names without traversal or special characters")
        return path

    @field_validator("text")
    @classmethod
    def applicable_markdown(cls, text: str) -> str:
        text = text.replace("\r\n", "\n").strip() + "\n"
        if "\x00" in text or not re.search(r"^Applies to:\s*\S.+$", text, re.MULTILINE):
            raise ValueError("Each rule needs an 'Applies to: ...' line with its applicability")
        if len(text.splitlines()) < 3 or not text.startswith("# "):
            raise ValueError("Each rule needs a Markdown title, applicability, and behavioral guidance")
        return text

    @property
    def version(self) -> str:
        return digest(self.text)

    @property
    def scope(self) -> str:
        parent = str(PurePosixPath(self.path).parent)
        return "global" if parent == "." else parent

    @property
    def applicability(self) -> str:
        return next(
            line.removeprefix("Applies to:").strip()
            for line in self.text.splitlines()
            if line.startswith("Applies to:")
        )


class RuleProposal(Record):
    id: str = Field(default_factory=lambda: uuid4().hex)
    base_revision: str
    changes: dict[str, Rule | None]
    fingerprint: str
    session_id: str
    execution_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    status: Literal["candidate", "pending", "accepted", "rejected"] = "candidate"
    rationale: str = ""
    decision: dict[str, Any] | None = None


class ProposalOutcome(Record):
    status: Literal["proposed", "duplicate", "no_proposal", "invalid", "conflict", "timed_out", "failed", "cancelled"]
    proposal: RuleProposal | None = None
    reason: str = ""
    rationale: str = ""  # Reported by the agent, not an independently verified conclusion.


class DeliveryReceipt(Record):
    id: str = Field(default_factory=lambda: uuid4().hex)
    execution_id: str
    channel: Literal["tool_output", "active_memory"]
    content_hash: str
    byte_count: int
    rule_versions: dict[str, str] = Field(default_factory=dict)
    candidate_ids: list[str] = Field(default_factory=list)
    submitted: bool = False
    created_at: str = Field(default_factory=utc_now)


class ValidationVerdict(Record):
    proposal_id: str
    outcome: Literal["accepted", "rejected", "pending"]
    reason: str
    mode: Literal["replay", "trial"]
    cases: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class FailureNotice(Record):
    id: str = Field(default_factory=lambda: uuid4().hex)
    execution_id: str
    session_id: str
    trace_ids: list[str]
    scores: dict[str, float]
    reasons: dict[str, str] = Field(default_factory=dict)
    signatures: list[str]
    created_at: str = Field(default_factory=utc_now)


class EvaluationOutcome(Record):
    execution_id: str
    status: Literal["evaluated", "skipped", "pending", "insufficient", "failed", "creation_unknown", "cancelled"]
    run_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    notice: FailureNotice | None = None
    reason: str = ""


class TrialArm(Record):
    """Host-verified outcome and exact exposure for one assigned trial arm."""

    execution_id: str = Field(min_length=1)
    scores: dict[str, StrictFloat] = Field(default_factory=dict)
    success: StrictBool | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    delivered_versions: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)

    @field_validator("scores")
    @classmethod
    def normalized_scores(cls, scores: dict[str, float]) -> dict[str, float]:
        if any(not name or not math.isfinite(value) or not 0 <= value <= 1 for name, value in scores.items()):
            raise ValueError("Trial scores must be finite values in [0, 1]")
        return scores


class TrialCase(Record):
    case_id: str = Field(min_length=1)
    base_revision: str = Field(min_length=64, max_length=64)
    input_hash: str = ""
    protected: bool = False
    control: TrialArm | None = None
    candidate: TrialArm | None = None


class ExecutionReport(Record):
    execution_id: str
    session_id: str
    task_status: str
    trace_ids: list[str]
    evaluation: EvaluationOutcome | None = None
    reflection: ProposalOutcome | None = None
    proposals: list[RuleProposal] = Field(default_factory=list)
    deliveries: list[DeliveryReceipt] = Field(default_factory=list)
