"""Bounded live incident demonstration; missing stages never become success claims."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from pandaprobe import Client
from pandaprobe.reflect import Reflect, ReflectConfig
from pandaprobe.reflect.integrations import invoker_for

from .delivery import TaskInputAudit, collect_delivery
from .refund_incident import DISCOVERY, FOLLOW_UP, INSTRUCTIONS, RefundBatch, RefundOrder, replay_cases
from .runner import reply_text, save_report, trace_usage


def arguments(argv=None, *, framework="langchain", default_model="gpt-6-luna"):
    parser = argparse.ArgumentParser(description=f"Live {framework} Reflect stale-runbook incident")
    parser.add_argument("--model", default=os.environ.get("REFLECT_EXAMPLE_MODEL", default_model))
    parser.add_argument("--output", type=Path, help="New artifact directory (existing directories are refused)")
    parser.add_argument("--task-timeout", type=float, default=360, help="Seconds per task/evaluation/reflection")
    args = parser.parse_args(argv)
    if not math.isfinite(args.task_timeout) or args.task_timeout <= 0:
        parser.error("task-timeout must be finite and positive")
    return args


async def run_task(runtime, build, args, output, report, order, stage):
    batch = RefundBatch(order)
    row = {
        "stage": stage,
        "order": order.id,
        "order_data": asdict(order),
        "input": order.prompt,
        "execution_id": uuid4().hex,
        "session_id": f"reflect-refund:{stage}:{uuid4().hex}",
    }
    report["turns"].append(row)
    started = time.monotonic()
    initial_rules, task_events = {}, []
    input_audit = TaskInputAudit()
    try:
        audit_options = {"input_audit": input_audit} if "input_audit" in inspect.signature(build).parameters else {}
        binding = build(runtime, batch, args.model, output, **audit_options)
        if not audit_options:
            input_audit.attach(binding)
        invoke = invoker_for(binding.agent, **binding.kwargs)

        async def observed(request):
            nonlocal initial_rules, task_events
            if request.phase == "task":
                row["initial_active_memory_bytes"] = len(request.execution.shell.context().encode())
                row["initial_candidate_ids"] = list(request.execution.shell.candidate_ids)
                row["base_revision"] = request.execution.base_revision
                initial_rules = request.execution.shell.files()
                input_audit.execution_id = request.execution.id
            try:
                return await invoke(request)
            finally:
                if request.phase == "task":
                    input_audit.execution_id = None
                    # Freeze business events before reflection; audit only recorded task traces.
                    task_events = list(batch.events)

        result = await asyncio.wait_for(
            runtime.run(observed, order.prompt, session_id=row["session_id"], execution_id=row["execution_id"]),
            timeout=args.task_timeout,
        )
        row["reply"] = reply_text(result)
        row["verification"] = batch.verify(row["reply"]).__dict__
        print(f"[{stage}] ledger success={row['verification']['success']}\n{row['reply']}", flush=True)
    except BaseException as exc:
        row["error"] = type(exc).__name__
        raise
    finally:
        row["seconds"] = round(time.monotonic() - started, 3)
        row["domain_events"] = list(batch.events)
        try:
            row["reflect"] = runtime.report(row["execution_id"]).model_dump(mode="json")
            snapshot = runtime.workspace.load_execution(row["execution_id"]) or {}
            row["phase_traces"] = snapshot.get("phase_traces", {})
            row["shell_usage"] = snapshot.get("usage", {})
        except KeyError:
            pass
        row["task_delivery"] = await collect_delivery(
            runtime,
            row.get("phase_traces", {}).get("task", []),
            initial_rules,
            task_events,
            observed_inputs=input_audit.records,
        )
        save_report(output / "report.json", report)
    return row


async def exercise(runtime, build, args, output: Path, *, framework="langchain") -> dict:
    """At most one candidate, six trial arms, and one fresh post-acceptance task."""
    if runtime.workspace.state().get("executions") or runtime.proposals() or runtime.workspace.rules():
        raise ValueError("The incident must start in an unused workspace")
    report = {
        "framework": framework,
        "model": args.model,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("pandaprobe", "bashkit", framework.replace("_", "-"))
        },
        "run_id": output.name,
        "workload": "Controlled stale-runbook incident; fictional refunds, real model and platform calls",
        "workload_version": 4,
        "agent_instructions": INSTRUCTIONS,
        "validation_inputs": [asdict(case) for case in replay_cases()],
        "initial_revision": runtime.workspace.state()["head"],
        "settings": {
            "task_timeout": args.task_timeout,
            "config": {**asdict(runtime.config), "workspace": str(runtime.config.workspace)},
        },
        "turns": [],
        "validations": [],
        "replay_arms": [],
        "milestones": dict.fromkeys(
            (
                "initial_failure_verified",
                "platform_failure_detected",
                "proposal_created",
                "independently_accepted",
                "fresh_guidance_retrieved",
                "fresh_task_verified",
            )
        ),
        "status": "running",
    }
    report_path = output / "report.json"
    started = time.monotonic()

    def stop(stage, reason):
        report.update(status="incomplete", stopped_at=stage, reason=reason)
        return report

    save_report(report_path, report)
    try:
        discovery = await run_task(runtime, build, args, output, report, DISCOVERY, "discovery")
        # An empty reply/operational error alone is not the planned business failure.
        mismatch = any(
            event.get("result", {}).get("reason") == "REFUND_AMOUNT_POLICY_MISMATCH"
            for event in discovery["domain_events"]
        )
        report["milestones"]["initial_failure_verified"] = discovery["verification"]["success"] is False and mismatch
        if not report["milestones"]["initial_failure_verified"]:
            return stop("discovery", "The intended policy-mismatch failure did not occur; no failure was forced.")
        evaluation = discovery.get("reflect", {}).get("evaluation") or {}
        detected = evaluation.get("status") == "evaluated" and bool(evaluation.get("notice"))
        report["milestones"]["platform_failure_detected"] = detected
        if not detected:
            return stop("detection", "The ledger failed but the platform supplied no usable failure notice.")
        reflection = discovery.get("reflect", {}).get("reflection") or {}
        proposal_record = reflection.get("proposal")
        proposed = reflection.get("status") in ("proposed", "duplicate") and bool(proposal_record)
        report["milestones"]["proposal_created"] = proposed
        if not proposed:
            return stop("reflection", f"No candidate from the reflection checkpoint: {reflection.get('status')}.")
        proposal = runtime.workspace.proposal(proposal_record["id"])
        if proposal.status != "candidate" or proposal.execution_id != discovery["execution_id"]:
            return stop("reflection", "The checkpoint did not yield a new candidate for this incident.")

        async def replay(request):
            # Each arm has new business state, model binding, and native conversation.
            batch = RefundBatch(RefundOrder(**request.case.input))
            row = {
                "case_id": request.case.id,
                "arm": request.arm,
                "protected": request.case.protected,
                "execution_id": request.execution.id,
                "session_id": request.execution.session_id,
                "initial_active_memory_bytes": len(request.execution.shell.context().encode()),
            }
            report["replay_arms"].append(row)
            tick = time.monotonic()
            try:
                binding = build(runtime, batch, args.model, output, validation=True)
                answer = await asyncio.wait_for(
                    runtime.run(
                        binding.agent,
                        batch.order.prompt,
                        session_id=request.execution.session_id,
                        **binding.kwargs,
                    ),
                    timeout=args.task_timeout,
                )
                row["reply"] = reply_text(answer)
                outcome = batch.verify(row["reply"])
                row["verification"] = outcome.__dict__
                print(f"[validation/{request.case.id}/{request.arm}] ledger success={outcome.success}", flush=True)
                return outcome
            except BaseException as exc:
                row["error"] = type(exc).__name__
                raise
            finally:
                row["seconds"] = round(time.monotonic() - tick, 3)
                row["domain_events"] = list(batch.events)
                save_report(report_path, report)

        # Only the SDK's independent validator can promote the proposed file changes.
        verdict = await runtime.validate(proposal.id, cases=replay_cases(), replay=replay)
        report["validations"].append(verdict.model_dump(mode="json"))
        report["milestones"]["independently_accepted"] = verdict.outcome == "accepted"
        print(f"[validation] {verdict.outcome}: {verdict.reason}", flush=True)
        if verdict.outcome != "accepted":
            return stop("validation", verdict.reason)
        report["accepted_revision"] = runtime.workspace.state()["head"]
        expected_versions = {rule.version for rule in proposal.changes.values() if rule is not None}
        report["expected_accepted_versions"] = sorted(expected_versions)

        # Reopen disk state, with no agent/checkpointer/session or context.md reuse.
        async with Reflect(runtime.client, config=runtime.config, api=runtime.api) as fresh_runtime:
            fresh = await run_task(fresh_runtime, build, args, output, report, FOLLOW_UP, "fresh")
        delivery = fresh["task_delivery"]
        retrieved = (
            bool(expected_versions)
            and expected_versions <= set(delivery["before_submission_excerpt_versions"])
            and fresh["initial_active_memory_bytes"] == 0
            and not fresh["initial_candidate_ids"]
            and fresh["base_revision"] == report["accepted_revision"]
        )
        report["milestones"]["fresh_guidance_retrieved"] = retrieved
        report["milestones"]["fresh_task_verified"] = fresh["verification"]["success"]
        if not retrieved:
            return stop(
                "fresh_delivery",
                "No verified excerpt from each accepted rule reached model input before the fresh refund attempt. "
                "See the example-only task_delivery audit; SDK acceptance does not depend on this check.",
            )
        if not fresh["verification"]["success"]:
            return stop("fresh_outcome", "Guidance reached the fresh task but the independent ledger check failed.")
        fresh_evaluation = fresh.get("reflect", {}).get("evaluation") or {}
        if fresh_evaluation.get("status") != "evaluated" or fresh_evaluation.get("notice"):
            return stop("fresh_evaluation", "Fresh ledger success lacks an agreeing, completed platform evaluation.")
        report["status"] = "completed"
        return report
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        if report["status"] == "running":
            report["status"] = "interrupted"
        report["workspace"] = str(runtime.workspace.path)
        report["proposals"] = [proposal.model_dump(mode="json") for proposal in runtime.proposals()]
        report["accepted_rules"] = {
            path: {"version": rule.version, "text": rule.text} for path, rule in runtime.workspace.rules().items()
        }
        save_report(report_path, report)
        try:
            await runtime.wait_export(timeout=10)
            report["trace_usage"] = await asyncio.wait_for(trace_usage(runtime, report), timeout=30)
        except Exception as exc:
            report["trace_usage_error"] = type(exc).__name__
        save_report(report_path, report)


def main(framework, build, instrument, default_model, required_key, argv=None):
    args = arguments(argv, framework=framework, default_model=default_model)
    missing = [
        name for name in ("PANDAPROBE_API_KEY", "PANDAPROBE_PROJECT_NAME", required_key) if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit("Export these test credentials before running: " + ", ".join(missing))
    output = args.output or Path(".pandaprobe/reflect-examples") / f"{framework}-failure" / uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {output.resolve()}\nReal model/evaluation calls; fictional refunds only.", flush=True)
    sdk = Client()

    async def run():
        if not sdk.config.enabled:
            raise RuntimeError("The live example requires PANDAPROBE_ENABLED=true")
        config = ReflectConfig(
            workspace=output / "workspace",
            thresholds={"task_completion": 0.8},
            reflection_timeout=90,
            validation_timeout=600,
        )
        async with Reflect(sdk, config=config) as runtime:
            metrics = {metric.name for metric in await runtime.api.evaluations.metrics()}
            missing_metrics = (set(config.thresholds) | set(config.diagnostic_metrics)) - metrics
            if missing_metrics:
                raise RuntimeError(f"Platform lacks required metrics: {sorted(missing_metrics)}")
            instrument(sdk)
            return await exercise(runtime, build, args, output, framework=framework)

    try:
        report = asyncio.run(run())
    finally:
        sdk.shutdown()
    print(json.dumps({key: report[key] for key in ("status", "milestones")}, indent=2), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    if report["status"] != "completed":
        raise SystemExit(f"Full loop not demonstrated at {report['stopped_at']}: {report['reason']}")
