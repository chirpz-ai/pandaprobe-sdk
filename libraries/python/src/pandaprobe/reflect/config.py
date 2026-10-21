"""Reflect policy defaults. Shared platform identity remains in SdkConfig."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ReflectConfig:
    workspace: Path | str | None = None
    trace_visibility_timeout: float = 30.0
    evaluation_timeout: float = 120.0
    poll_interval: float = 1.0
    evaluation_max_runs: int = 100
    evaluation_max_traces: int = 32
    evaluation_sample_rate: float = 1.0
    max_concurrency: int = 4
    cooldown_seconds: float = 60.0
    reflection_timeout: float = 60.0
    shutdown_timeout: float = 5.0
    reflection_max_commands: int = 8
    shell_timeout: float = 5.0
    shell_max_commands: int = 2000
    shell_max_loop_iterations: int = 5000
    shell_memory_bytes: int = 16 * 1024 * 1024
    command_max_bytes: int = 16 * 1024
    output_max_bytes: int = 12 * 1024
    execution_output_max_bytes: int = 96 * 1024
    context_max_bytes: int = 8 * 1024
    rule_max_bytes: int = 8 * 1024
    rules_max_bytes: int = 512 * 1024
    rules_max_count: int = 256
    evidence_max_bytes: int = 64 * 1024
    thresholds: dict[str, float] = field(default_factory=lambda: {"task_completion": 0.5, "coherence": 0.5})
    diagnostic_metrics: tuple[str, ...] = ("tool_correctness", "argument_correctness")
    stall_window: int = 5
    minimum_progress: float = 0.02
    regression_drop: float = 0.15
    validation_min_cases: int = 3
    validation_min_improvement: float = 0.05
    validation_regression_tolerance: float = 0.02
    validation_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.workspace is None:
            value = os.environ.get("PANDAPROBE_REFLECT_WORKSPACE", ".pandaprobe/reflect")
            object.__setattr__(self, "workspace", Path(value))
        else:
            object.__setattr__(self, "workspace", Path(self.workspace))
        probabilities = (
            "evaluation_sample_rate",
            "minimum_progress",
            "regression_drop",
            "validation_min_improvement",
            "validation_regression_tolerance",
        )
        for name in probabilities:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "trace_visibility_timeout",
            "evaluation_timeout",
            "poll_interval",
            "reflection_timeout",
            "shutdown_timeout",
            "shell_timeout",
            "validation_timeout",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.cooldown_seconds) or self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be finite and nonnegative")
        for name in (
            "evaluation_max_runs",
            "evaluation_max_traces",
            "max_concurrency",
            "reflection_max_commands",
            "shell_max_commands",
            "shell_max_loop_iterations",
            "shell_memory_bytes",
            "command_max_bytes",
            "output_max_bytes",
            "execution_output_max_bytes",
            "context_max_bytes",
            "rule_max_bytes",
            "rules_max_bytes",
            "rules_max_count",
            "evidence_max_bytes",
            "stall_window",
            "validation_min_cases",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.thresholds or any(
            not name or not math.isfinite(v) or not 0 <= v <= 1 for name, v in self.thresholds.items()
        ):
            raise ValueError("thresholds must contain metric names and finite values between zero and one")
        # Do not retain the caller's mutable mapping.
        object.__setattr__(self, "thresholds", dict(self.thresholds))
