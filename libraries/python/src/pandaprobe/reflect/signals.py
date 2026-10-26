"""Deterministic signal gates; missing measurements are never success."""

from __future__ import annotations

import math
import time
from typing import Any

from .config import ReflectConfig
from .models import FailureNotice, digest
from .workspace import Workspace


class SignalGate:
    def __init__(self, workspace: Workspace, config: ReflectConfig) -> None:
        self.workspace, self.config = workspace, config

    def observe(
        self,
        *,
        execution_id: str,
        session_id: str,
        trace_ids: list[str],
        scores: dict[str, float],
        reasons: dict[str, str],
        completed: bool,
    ) -> FailureNotice | None:
        if set(self.config.thresholds) - scores.keys() or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()
        ):
            raise ValueError("Signal gates require complete finite scores in [0, 1]")

        def update(state: dict[str, Any]) -> FailureNotice | None:
            for record in state["notices"].values():
                if record["execution_id"] == execution_id:
                    return FailureNotice.model_validate(record)
            key = digest(session_id)
            gate = state["gates"].setdefault(key, {"history": [], "last_notices": {}})
            history = gate["history"]
            # Retried evaluation must not turn one execution into a trajectory.
            if any(row["execution_id"] == execution_id for row in history):
                return None
            signatures = []
            for metric, threshold in self.config.thresholds.items():
                value = scores[metric]
                previous = [row["scores"][metric] for row in history if metric in row["scores"]]
                if completed and value < threshold:
                    signatures.append(f"{metric}:below_threshold")
                elif previous and previous[-1] - value >= self.config.regression_drop and value < threshold:
                    signatures.append(f"{metric}:regression")
                elif len(previous) >= self.config.stall_window - 1 and value < threshold:
                    window = (previous + [value])[-self.config.stall_window :]
                    if max(window) - min(window) < self.config.minimum_progress:
                        signatures.append(f"{metric}:stalled")
            history.append({"execution_id": execution_id, "scores": scores})
            del history[: -self.config.stall_window]
            now = time.time()
            signatures = [
                signature
                for signature in signatures
                if now - gate["last_notices"].get(signature, 0) >= self.config.cooldown_seconds
            ]
            if not signatures:
                return None
            for signature in signatures:
                gate["last_notices"][signature] = now
            notice = FailureNotice(
                execution_id=execution_id,
                session_id=session_id,
                trace_ids=trace_ids,
                scores=scores,
                reasons=reasons,
                signatures=signatures,
            )
            state["notices"][notice.id] = notice.model_dump(mode="json")
            return notice

        return self.workspace.transaction(update)
