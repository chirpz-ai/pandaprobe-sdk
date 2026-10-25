"""Business oracle and full native LangChain incident loop, with no live services.

The scripted model and judge below are test fixtures, never used by the example.
They test lifecycle plumbing and honest reporting, not learned model performance.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest

from pandaprobe import Client
from pandaprobe.api import PlatformClient
from pandaprobe.reflect import Reflect, ReflectConfig


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[1] / "examples/langchain/05_reflect_failure.py"
    spec = importlib.util.spec_from_file_location("reflect_failure_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def incident(example):
    from _support import refund_incident

    return refund_incident


def test_refund_failure_is_visible_irrecoverable_and_verified_from_ledger(incident):
    batch = incident.RefundBatch(incident.DISCOVERY)
    order = json.loads(batch.lookup_refund_order(batch.order.id))
    rejected = json.loads(batch.submit_refund(order["id"], order["invoice_total_cents"], order["id"]))
    assert rejected["reason"] == "REFUND_AMOUNT_POLICY_MISMATCH"
    assert rejected["batch_closed"] and not rejected["posted"]
    assert rejected["required_cents"] == order["merchandise_paid_cents"]
    assert not batch.verify("I successfully refunded everything").success
    assert (
        json.loads(batch.submit_refund(order["id"], order["merchandise_paid_cents"], order["id"]))["status"]
        == "closed"
    )
    assert batch.ledger == [] and not batch.verify("Recovered").success
    # The independent verifier does not trust the service's success flag.
    batch.events[1]["result"] = {"status": "posted", "posted": True}
    assert not batch.verify("Posted").success


@pytest.mark.parametrize("amount,key", [(6500, "wrong"), (6500.0, "INCIDENT-101"), (True, "INCIDENT-101")])
def test_refund_rejects_invalid_identity_and_noninteger_money(incident, amount, key):
    batch = incident.RefundBatch(incident.DISCOVERY)
    batch.lookup_refund_order(batch.order.id)
    assert json.loads(batch.submit_refund(batch.order.id, amount, key))["status"] == "rejected"
    assert not batch.verify("Posted").success


def test_refund_validation_cases_are_held_out_fresh_and_protect_success(incident):
    cases = incident.replay_cases()
    assert len(cases) == 3 and sum(case.protected for case in cases) == 1
    ids = {case.input["id"] for case in cases}
    assert len(ids | {incident.DISCOVERY.id, incident.FOLLOW_UP.id}) == 5
    for case in cases:
        order = incident.RefundOrder(**case.input)
        control, candidate = incident.RefundBatch(order), incident.RefundBatch(order)
        control.lookup_refund_order(order.id)
        candidate.lookup_refund_order(order.id)
        control.submit_refund(order.id, order.merchandise_paid_cents + order.outbound_shipping_cents, order.id)
        assert not candidate.closed and not candidate.ledger
        candidate.submit_refund(order.id, order.merchandise_paid_cents, order.id)
        assert control.verify("Result").success is case.protected
        assert candidate.verify("Result").success
        assert not candidate.verify("").success


def test_refund_verifier_requires_read_before_write_and_single_submission(incident):
    batch = incident.RefundBatch(incident.DISCOVERY)
    batch.submit_refund(batch.order.id, batch.order.merchandise_paid_cents, batch.order.id)
    batch.lookup_refund_order(batch.order.id)
    assert not batch.verify("Posted").success
    batch = incident.RefundBatch(incident.DISCOVERY)
    batch.lookup_refund_order(batch.order.id)
    batch.submit_refund(batch.order.id, batch.order.merchandise_paid_cents, batch.order.id)
    assert batch.verify("Posted").success
    batch.submit_refund(batch.order.id, batch.order.merchandise_paid_cents, batch.order.id)
    assert not batch.verify("Posted").success


def test_failure_example_help_and_bad_timeout_are_side_effect_free(example, capsys):
    from _support.failure_runner import arguments

    with pytest.raises(SystemExit) as exc:
        example.main("langchain", None, None, "test", "NO_KEY", ["--help"])
    assert exc.value.code == 0 and "--task-timeout" in capsys.readouterr().out
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            arguments([f"--task-timeout={value}"])


@pytest.fixture
def native_fixture(example, incident, tmp_path, monkeypatch):
    pytest.importorskip("langchain")
    import langchain_openai
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from _support.failure_runner import arguments, exercise

    payloads, runs, bindings, model_inputs = {}, {}, [], []
    good_rule = (
        "# Standard refund amounts\nApplies to: authorized standard-return refund batches\n"
        "Refund merchandise_paid_cents only; outbound shipping is nonrefundable. "
        "This supersedes the legacy invoice-total fallback. Keep the order ID as idempotency key.\n"
    )

    class ScriptedModel(BaseChatModel):
        mode: str
        fresh: bool
        binding_id: int

        @property
        def _llm_type(self):
            return "offline-incident-fixture"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            model_inputs.append((self.binding_id, list(messages)))
            if self.mode == "provider_error":
                raise RuntimeError("Fixture provider unavailable")
            human_index = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
            reflection = "The task invocation has finished" in messages[human_index].content
            outputs = [message for message in messages[human_index + 1 :] if isinstance(message, ToolMessage)]

            def call(name, arguments):
                return {"name": name, "args": arguments, "id": uuid4().hex}

            def bash(command):
                return AIMessage(content="", tool_calls=[call("reflect_bash", {"command": command})])

            if reflection:
                if self.mode == "no_proposal":
                    answer = AIMessage(content="Uncertain; no guidance proposed.")
                elif not outputs:
                    answer = bash("cat /inbox/notice.json; cat /inbox/evidence.txt; cat /rules/*.md 2>/dev/null")
                elif len(outputs) == 1:
                    rule = (
                        good_rule
                        if self.mode != "bad_proposal"
                        else good_rule.replace("Refund merchandise_paid_cents only", "Refund invoice_total_cents")
                    )
                    answer = bash(
                        "cat > /rules/refunds.md <<'RULE'\n" + rule + "RULE\n"
                        "printf 'Private authoring memory; never leak into trials' > /context.md"
                    )
                else:
                    answer = AIMessage(content="Proposed guidance; independent replay must decide.")
            elif not outputs:
                order_id = messages[human_index].content.split("order ", 1)[1].split()[0]
                calls = [call("lookup_refund_order", {"order_id": order_id})]
                if not (self.fresh and self.mode in ("fresh_no_read", "fresh_late_read", "fresh_read_after_submit")):
                    command = "cat /rules/*.md 2>/dev/null || true"
                    if self.mode == "numbered_grep":
                        command = "grep -RnE 'refund|Refund|Applies|Standard' /rules 2>/dev/null || true"
                    elif self.mode == "partial_grep" or (self.fresh and self.mode == "fresh_partial_grep"):
                        command = "grep -Rn merchandise_paid /rules 2>/dev/null || true"
                    calls.append(call("reflect_bash", {"command": command}))
                answer = AIMessage(content="", tool_calls=calls)
            elif not any(message.name == "submit_refund" for message in outputs):
                order = json.loads(
                    next(message.content for message in outputs if message.name == "lookup_refund_order")
                )
                # Select only from what this model request actually received, never the trial arm label.
                corrected = any("Refund merchandise_paid_cents only" in message.content for message in outputs)
                corrected |= self.mode == "no_failure" or (
                    self.fresh and self.mode in ("fresh_no_read", "fresh_read_after_submit")
                )
                amount = order["merchandise_paid_cents"] if corrected else order["invoice_total_cents"]
                if self.mode == "regression" and corrected and order["outbound_shipping_cents"] == 0:
                    amount -= 1
                answer = AIMessage(
                    content="",
                    tool_calls=[
                        call(
                            "submit_refund",
                            {"order_id": order["id"], "amount_cents": amount, "idempotency_key": order["id"]},
                        )
                    ],
                )
            else:
                posted = json.loads(next(message.content for message in outputs if message.name == "submit_refund"))[
                    "posted"
                ]
                if (
                    self.fresh
                    and self.mode == "fresh_read_after_submit"
                    and not any(message.name == "reflect_bash" for message in outputs)
                ):
                    answer = bash("cat /rules/*.md")
                else:
                    answer = AIMessage(content="Refund posted." if posted else "Refund rejected; batch closed.")
            return ChatResult(generations=[ChatGeneration(message=answer)])

    async def run(mode="healthy"):
        sdk = Client(api_key="test", project_name="test", enabled=False)
        object.__setattr__(sdk.config, "enabled", True)
        sdk._transport = MagicMock()
        project_id = str(uuid4())
        sdk._transport.enqueue_trace.side_effect = lambda payload: payloads.update(
            {payload["trace_id"]: {"project_id": project_id, **payload}}
        )

        def respond(request):
            path = request.url.path
            if path.startswith("/traces/"):
                payload = payloads[path.rsplit("/", 1)[1]]
                # Reads materialize the enclosing trace ID/defaults omitted from span exports.
                value = {
                    **payload,
                    "spans": [
                        {**span, "trace_id": payload["trace_id"], "metadata": span.get("metadata") or {}}
                        for span in payload["spans"]
                    ],
                }
                return httpx.Response(200, json=value)
            if request.method == "POST":
                run_id = str(uuid4())
                runs[run_id] = json.loads(request.content)
            else:
                run_id = path.split("/")[3]
            query = runs[run_id]
            if path.endswith("/scores"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": str(uuid4()),
                            "eval_run_id": run_id,
                            "trace_id": tid,
                            "name": metric,
                            "value": str(
                                float(
                                    mode == "judge_miss"
                                    or "REFUND_AMOUNT_POLICY_MISMATCH" not in json.dumps(payloads[tid])
                                )
                            ),
                            "status": "SUCCESS",
                            "data_type": "NUMERIC",
                            "reason": "Offline judge fixture",
                        }
                        for tid in query["trace_ids"]
                        for metric in query["metrics"]
                    ],
                )
            return httpx.Response(
                200,
                json={
                    "id": run_id,
                    "status": "COMPLETED",
                    "metric_names": query["metrics"],
                    "target_type": "TRACE",
                    "total_targets": len(query["trace_ids"]),
                    "evaluated_count": len(query["trace_ids"]),
                },
            )

        def build(runtime, batch, model, workdir, *, validation=False, input_audit=None):
            scripted = ScriptedModel(
                mode=mode, fresh=batch.order.id == incident.FOLLOW_UP.id, binding_id=len(bindings)
            )
            monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: scripted)
            binding = example.build(runtime, batch, model, workdir, validation=validation, input_audit=input_audit)
            bindings.append((runtime, batch, binding))
            return binding

        output = tmp_path / "run"
        output.mkdir()
        config = ReflectConfig(
            workspace=output / "workspace",
            thresholds={"task_completion": 0.8},
            diagnostic_metrics=(),
            poll_interval=0.001,
        )
        try:
            async with PlatformClient(sdk, transport=httpx.MockTransport(respond)) as api:
                async with Reflect(sdk, config=config, api=api) as runtime:
                    return await exercise(runtime, build, arguments([]), output)
        finally:
            sdk.shutdown()

    return run, payloads, runs, bindings, model_inputs


async def test_langchain_live_failure_loop_acceptance_and_fresh_delivery(native_fixture, tmp_path, incident):
    run, payloads, runs, bindings, inputs = native_fixture
    report = await run()
    assert report["status"] == "completed" and all(report["milestones"].values())
    assert [row["verification"]["success"] for row in report["turns"]] == [False, True]
    assert report["validations"][0]["outcome"] == "accepted"
    assert len(report["replay_arms"]) == 6 and len(bindings) == 8
    assert len({id(batch) for _, batch, _ in bindings}) == 8
    assert len({id(binding.agent) for _, _, binding in bindings}) == 8
    assert bindings[0][0] is not bindings[-1][0]  # Disk reopened by a new runtime.
    assert len({row["session_id"] for row in report["turns"] + report["replay_arms"]}) == 8
    assert all(row["initial_active_memory_bytes"] == 0 for row in report["replay_arms"])
    assert len(report["validations"][0]["cases"]) == 3
    for case in report["validations"][0]["cases"]:
        assert case["control"]["success"] is case["protected"]
        assert case["candidate"]["success"] is True
    fresh = report["turns"][1]
    assert fresh["task_delivery"]["model_input_versions"] == report["expected_accepted_versions"]
    assert not fresh["initial_candidate_ids"] and fresh["initial_active_memory_bytes"] == 0
    first_fresh_input = next(messages for binding_id, messages in inputs if binding_id == 7)
    assert not any("Private authoring memory" in message.content for message in first_fresh_input)
    assert not any("The task invocation has finished" in message.content for message in first_fresh_input)
    assert incident.DISCOVERY.id not in str(first_fresh_input)
    assert {row["role"] for row in report["trace_usage"]} == {"task", "reflection", "validation"}
    assert len(runs) == 2  # No recursively evaluated reflection or replay traces.
    assert all(
        payloads[tid]["metadata"]["pandaprobe.reflect"]["role"] == "task"
        for query in runs.values()
        for tid in query["trace_ids"]
    )
    assert json.loads((tmp_path / "run/report.json").read_text())["status"] == "completed"


async def test_langchain_numbered_grep_completes_validation_and_fresh_delivery(native_fixture):
    run, *_ = native_fixture
    report = await run("numbered_grep")
    assert report["status"] == "completed" and all(report["milestones"].values())
    fresh = report["turns"][1]["task_delivery"]
    assert fresh["before_submission_versions"] == report["expected_accepted_versions"]
    assert not fresh["partial_model_input_versions"]
    assert any(
        match["representation"] == "numbered_grep" and match["complete"]
        for receipt in fresh["receipts"]
        for match in receipt["rule_coverage"].values()
    )


@pytest.mark.parametrize("mode", ["partial_grep", "fresh_partial_grep"])
async def test_langchain_verified_excerpt_completes_loop_without_claiming_full_coverage(native_fixture, mode):
    run, *_ = native_fixture
    report = await run(mode)
    assert report["status"] == "completed" and all(report["milestones"].values())
    fresh = report["turns"][1]["task_delivery"]
    assert fresh["before_submission_excerpt_versions"] == report["expected_accepted_versions"]
    assert fresh["partial_model_input_versions"] == report["expected_accepted_versions"]
    assert not fresh["model_input_versions"] and not fresh["before_submission_versions"]


@pytest.mark.parametrize(
    "mode,stage",
    [
        ("no_failure", "discovery"),
        ("judge_miss", "detection"),
        ("no_proposal", "reflection"),
        ("bad_proposal", "validation"),
        ("regression", "validation"),
        ("fresh_no_read", "fresh_delivery"),
        ("fresh_late_read", "fresh_delivery"),
        ("fresh_read_after_submit", "fresh_delivery"),
    ],
)
async def test_langchain_failure_runner_does_not_invent_missing_stages(native_fixture, mode, stage):
    run, *_ = native_fixture
    report = await run(mode)
    assert report["status"] == "incomplete" and report["stopped_at"] == stage
    if stage in ("discovery", "detection", "reflection"):
        assert not report["validations"] and len(report["turns"]) == 1
    if stage == "validation":
        assert report["validations"][0]["outcome"] == ("rejected" if mode == "regression" else "pending")
        assert not report["accepted_rules"]
    if mode == "fresh_no_read":
        assert report["milestones"]["fresh_task_verified"] is True
        assert not report["turns"][1]["task_delivery"]["before_submission_excerpt_versions"]
    if mode == "fresh_late_read":
        fresh = report["turns"][1]
        assert not fresh["task_delivery"]["model_input_versions"]
        assert fresh["phase_traces"]["reflection"]  # Later reflection reads are excluded from the task audit.
    if mode == "fresh_read_after_submit":
        fresh = report["turns"][1]
        assert fresh["verification"]["success"]
        assert fresh["task_delivery"]["model_input_versions"]
        assert not fresh["task_delivery"]["before_submission_versions"]
        assert not fresh["task_delivery"]["before_submission_excerpt_versions"]


async def test_langchain_failure_runner_preserves_operational_error_and_partial_report(native_fixture, tmp_path):
    run, *_ = native_fixture
    with pytest.raises(RuntimeError, match="Fixture provider unavailable"):
        await run("provider_error")
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == "interrupted" and report["turns"][0]["error"] == "RuntimeError"
    assert not report["accepted_rules"] and not any(report["milestones"].values())
