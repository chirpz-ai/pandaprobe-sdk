"""Live model/platform exercise shared by seven native framework examples.

This is application code, not SDK orchestration. Gold outcomes inspect the local
support ledger. Platform judges independently decide whether to trigger Reflect.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pandaprobe import Client
from pandaprobe.reflect import Reflect, ReflectConfig

from .scenario import SupportDesk, Ticket, replay_cases, tickets


@dataclass
class AgentBinding:
    agent: Any
    kwargs: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    history_mode: str = "native"

    def input(self, message: str):
        if self.history_mode == "messages":
            return [*self.history, {"role": "user", "content": message}]
        return message

    def remember(self, message: str, result: Any) -> None:
        if self.history_mode == "messages":
            self.history.extend(
                [{"role": "user", "content": message}, {"role": "assistant", "content": reply_text(result)}]
            )
        elif self.history_mode == "claude":
            # Resume the actual native conversation, including Reflect's checkpoint.
            session = next(
                (getattr(item, "session_id", None) for item in reversed(result) if getattr(item, "session_id", None)),
                None,
            )
            if session:
                self.agent = replace(self.agent, resume=session)


def reply_text(result: Any) -> str:
    """Materialize the actual final response, not an event/object repr."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "messages" in result:
        return reply_text(result["messages"][-1])
    for field_name in ("final_output", "raw", "result"):
        value = getattr(result, field_name, None)
        if isinstance(value, str):
            return value
    if isinstance(result, list):
        for item in reversed(result):
            if getattr(item, "is_error", False):
                raise RuntimeError("Claude returned an error result")
            if callable(getattr(item, "is_final_response", None)) and not item.is_final_response():
                continue
            text = reply_text(item)
            if text:
                return text
        return ""
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", content)
    if isinstance(parts, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") or "" for part in parts
        )
    return ""


def arguments(framework: str, default_model: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Live {framework} Reflect customer-support pressure test")
    parser.add_argument("--model", default=os.environ.get("REFLECT_EXAMPLE_MODEL", default_model))
    parser.add_argument(
        "--rounds", type=int, default=1, help="Six three-turn tickets per round (default: 18 task turns)"
    )
    parser.add_argument("--max-tickets", type=int, help="Optional short smoke run; 1 means three task turns")
    parser.add_argument(
        "--max-validations", type=int, default=2, help="Maximum candidates to replay (six model tasks each)"
    )
    parser.add_argument(
        "--task-timeout", type=float, default=360, help="Seconds per task including evaluation/reflection"
    )
    parser.add_argument(
        "--output", type=Path, help="Run directory; defaults to .pandaprobe/reflect-examples/<framework>/<id>"
    )
    parser.add_argument(
        "--workspace", type=Path, help="Explicit reuse of previous accepted guidance; default is a fresh run"
    )
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.max_validations < 0 or not math.isfinite(args.task_timeout) or args.task_timeout <= 0:
        parser.error("rounds/task-timeout must be positive; max-validations must be nonnegative")
    if args.max_tickets is not None and args.max_tickets < 1:
        parser.error("max-tickets must be positive")
    return args


def save_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(report: dict) -> dict:
    turns = report["turns"]
    return {
        "task_turns": len(turns),
        "verified_successes": sum(row.get("verification", {}).get("success") is True for row in turns),
        "task_errors": sum("error" in row for row in turns),
        "failure_notices": sum(
            bool(row.get("reflect", {}).get("evaluation", {}).get("notice"))
            for row in turns
            if row.get("reflect", {}).get("evaluation")
        ),
        "reflection_outcomes": [
            row["reflect"]["reflection"]["status"] for row in turns if row.get("reflect", {}).get("reflection")
        ],
        "validation_outcomes": [row["outcome"] for row in report["validations"]],
        "limitation": "Completion/delivery do not prove improvement. Only paired validation measures the tested change.",
    }


async def trace_usage(runtime, report: dict) -> list[dict]:
    """Best-effort observed model usage, separated by task/reflection/validation.

    Missing telemetry remains unknown. This does not price evaluation-model runs.
    """
    ids = {}
    for row in report["turns"]:
        for role, traces in row.get("phase_traces", {}).items():
            ids.update({trace_id: role for trace_id in traces})
    for row in report["replay_arms"]:
        snapshot = runtime.workspace.load_execution(row["execution_id"]) or {}
        for traces in snapshot.get("phase_traces", {}).values():
            ids.update({trace_id: "validation" for trace_id in traces})
    slots = asyncio.Semaphore(4)

    async def retrieve(trace_id, role):
        async with slots:
            try:
                trace = await runtime.api.traces.get(trace_id, timeout=5)
                return {
                    "trace_id": trace_id,
                    "role": role,
                    "llm_spans": [
                        {
                            "span_id": str(span.span_id),
                            "model": getattr(span, "model", None),
                            "token_usage": getattr(span, "token_usage", None),
                            "latency_ms": getattr(span, "latency_ms", None),
                            "cost": getattr(span, "cost", None),
                        }
                        for span in trace.spans
                        if span.kind == "LLM"
                    ],
                }
            except Exception as exc:
                return {"trace_id": trace_id, "role": role, "unavailable": type(exc).__name__}

    return await asyncio.gather(*(retrieve(trace_id, role) for trace_id, role in ids.items()))


async def exercise(runtime, build: Callable, args, framework: str, output: Path) -> dict:
    """Run fresh customer conversations, retaining only shared accepted rules between them."""
    report = {
        "framework": framework,
        "model": args.model,
        "run_id": output.name,
        "versions": {name: importlib.metadata.version(name) for name in ("pandaprobe", "bashkit")},
        "python": platform.python_version(),
        "initial_revision": runtime.workspace.state()["head"],
        "settings": {
            "rounds": args.rounds,
            "max_tickets": args.max_tickets,
            "max_validations": args.max_validations,
            "task_timeout": args.task_timeout,
            "thresholds": runtime.config.thresholds,
        },
        "turns": [],
        "validations": [],
        "replay_arms": [],
        "status": "running",
        "workload": "Synthetic store; real provider calls and platform evaluations. No external business mutations.",
        "workload_version": 1,
    }
    if framework in (
        "langchain",
        "langgraph",
        "deepagents",
        "openai_agents",
        "google_adk",
        "crewai",
        "claude_agent_sdk",
    ):
        distribution = framework.replace("_", "-")
        report["versions"][distribution] = importlib.metadata.version(distribution)
    report_path = output / "report.json"
    checked: set[str] = set()
    queue = [ticket for index in range(args.rounds) for ticket in tickets(index)]
    if args.max_tickets:
        queue = queue[: args.max_tickets]
    started = time.monotonic()
    save_report(report_path, report)
    try:
        for ticket in queue:
            desk = SupportDesk(ticket)
            session_id = f"reflect-support:{output.name}:{ticket.id}"
            binding = build(runtime, desk, args.model, output, validation=False)
            for stage in ("inspect", "resolve", "repeat"):
                desk.begin(stage)
                message = ticket.prompt(stage)
                execution_id = uuid4().hex
                turn = {
                    "ticket": ticket.id,
                    "policy": ticket.policy,
                    "stage": stage,
                    "session_id": session_id,
                    "execution_id": execution_id,
                }
                tick = time.monotonic()
                print(f"\n[{ticket.id}/{stage}] {message}", flush=True)
                try:
                    result = await asyncio.wait_for(
                        runtime.run(
                            binding.agent,
                            binding.input(message),
                            session_id=session_id,
                            execution_id=execution_id,
                            **binding.kwargs,
                        ),
                        timeout=args.task_timeout,
                    )
                    text = reply_text(result)
                    binding.remember(message, result)
                    turn["reply"] = text
                    turn["verification"] = desk.verify(text).__dict__
                    print(text, flush=True)
                except Exception as exc:
                    # Preserve the failure and partial artifacts, then stop. Never retry paid calls blindly.
                    turn["error"] = type(exc).__name__
                    raise
                finally:
                    turn["seconds"] = round(time.monotonic() - tick, 3)
                    try:
                        turn["reflect"] = runtime.report(execution_id).model_dump(mode="json")
                        snapshot = runtime.workspace.load_execution(execution_id) or {}
                        turn["phase_traces"] = snapshot.get("phase_traces", {})
                        turn["shell_usage"] = snapshot.get("usage", {})
                        turn["active_memory_bytes"] = len(snapshot.get("context", "").encode())
                    except KeyError:
                        pass
                    turn["domain_events"] = list(desk.events[desk._stage_start :])
                    report["turns"].append(turn)
                    save_report(report_path, report)
                evaluation = turn.get("reflect", {}).get("evaluation") or {}
                if evaluation.get("status") != "evaluated":
                    raise RuntimeError(
                        f"Platform evidence unavailable: {evaluation.get('status', 'missing')}; see report"
                    )
                print(
                    f"Verified: {turn['verification']['success']}; evaluation: {evaluation.get('status')}", flush=True
                )

                # No auto-replay callback: validation is explicitly settled after task resources are released.
                for proposal in runtime.proposals():
                    if proposal.id in checked or proposal.status not in ("candidate", "pending"):
                        continue
                    if len(checked) >= args.max_validations:
                        break
                    checked.add(proposal.id)

                    async def replay(request, proposal_id=proposal.id):
                        isolated = SupportDesk(Ticket(**request.case.input))
                        isolated.begin("resolve")
                        fresh = build(runtime, isolated, args.model, output, validation=True)
                        # run() reuses the active validation context without scheduling another evaluation.
                        # The case stores order data; the agent receives only the customer prompt.
                        replay_started = time.monotonic()
                        answer = await runtime.run(
                            fresh.agent,
                            isolated.ticket.prompt("resolve"),
                            session_id=request.execution.session_id,
                            **fresh.kwargs,
                        )
                        outcome = isolated.verify(reply_text(answer))
                        report["replay_arms"].append(
                            {
                                "proposal_id": proposal_id,
                                "case_id": request.case.id,
                                "arm": request.arm,
                                "execution_id": request.execution.id,
                                "seconds": round(time.monotonic() - replay_started, 3),
                                "reply": reply_text(answer),
                                "verification": outcome.__dict__,
                            }
                        )
                        save_report(report_path, report)
                        return outcome

                    verdict = await runtime.validate(proposal.id, cases=replay_cases(ticket.policy), replay=replay)
                    report["validations"].append(verdict.model_dump(mode="json"))
                    print(f"Validation: {verdict.outcome} — {verdict.reason}", flush=True)
                    save_report(report_path, report)
        report["status"] = "completed"
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        report["summary"] = summarize(report)
        report["proposals"] = [proposal.model_dump(mode="json") for proposal in runtime.proposals()]
        rules = runtime.workspace.rules()
        report["accepted_rules"] = {
            path: {"version": rule.version, "bytes": len(rule.text.encode())} for path, rule in rules.items()
        }
        report["workspace"] = str(runtime.workspace.path)
        try:
            await runtime.wait_export(timeout=10)
            report["trace_usage"] = await asyncio.wait_for(trace_usage(runtime, report), timeout=30)
        except Exception as exc:
            report["trace_usage_error"] = type(exc).__name__
        if report["status"] != "completed":
            report["status"] = "interrupted"
        save_report(report_path, report)
    return report


def main(framework, build, instrument, default_model, required_key, argv=None) -> None:
    args = arguments(framework, default_model, argv)
    required = ["PANDAPROBE_API_KEY", "PANDAPROBE_PROJECT_NAME", required_key]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Export these test credentials before running: " + ", ".join(missing))
    output = args.output or Path(".pandaprobe/reflect-examples") / framework / uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {output.resolve()}\nLive model/evaluation calls; fictional customer data only.", flush=True)
    sdk = Client()
    if not sdk.config.enabled:
        sdk.shutdown()
        raise SystemExit("These live examples require PANDAPROBE_ENABLED=true")

    async def run():
        config = ReflectConfig(
            workspace=args.workspace or output / "workspace",
            thresholds={"task_completion": 0.8},  # Replay verifies this metric, not subjective coherence.
            reflection_timeout=90,
            validation_timeout=600,
        )
        async with Reflect(sdk, config=config) as runtime:
            # Fail before model spend if identity, API access, or required metrics are unavailable.
            metrics = {metric.name for metric in await runtime.api.evaluations.metrics()}
            required_metrics = set(config.thresholds) | set(config.diagnostic_metrics)
            if required_metrics - metrics:
                raise RuntimeError(f"Platform lacks required metrics: {sorted(required_metrics - metrics)}")
            instrument(sdk)
            try:
                report = await exercise(runtime, build, args, framework, output)
                print(json.dumps(report["summary"], indent=2), flush=True)
            finally:
                await runtime.wait_export()

    try:
        asyncio.run(run())
    finally:
        sdk.shutdown()
    print(f"Report: {output / 'report.json'}", flush=True)
