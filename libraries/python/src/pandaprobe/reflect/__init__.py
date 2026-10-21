"""Opt-in, filesystem-first task-agent reflection.

Importing this module does not activate Reflect or import model/framework code.
"""

from .config import ReflectConfig
from .execution import Execution
from .models import (
    DeliveryReceipt,
    EvaluationOutcome,
    ExecutionReport,
    ProposalOutcome,
    Rule,
    RuleProposal,
    TrialArm,
    TrialCase,
    ValidationVerdict,
)
from .runtime import Invocation, Reflect
from .validation import Outcome, ReplayCase, ReplayRequest

__all__ = [
    "DeliveryReceipt",
    "EvaluationOutcome",
    "Execution",
    "ExecutionReport",
    "Invocation",
    "Outcome",
    "ProposalOutcome",
    "Reflect",
    "ReflectConfig",
    "ReplayCase",
    "ReplayRequest",
    "Rule",
    "RuleProposal",
    "TrialArm",
    "TrialCase",
    "ValidationVerdict",
]
