"""Native framework loops with local model fixtures and a host trace boundary.

The live examples use their existing framework tracing. These tests additionally
trace the independent ledger outcome to supply deterministic evaluation fixtures.
No provider or PandaProbe network calls; Claude's CLI transport is mocked.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from pandaprobe import Client
from pandaprobe.api.models import EvaluationResult, EvaluationRun, Score, Trace
from pandaprobe.reflect import Reflect, ReflectConfig
from pandaprobe.reflect.integrations import input_text, invoker_for

FRAMEWORKS = ("langchain", "langgraph", "deepagents", "openai_agents", "google_adk", "crewai", "claude_agent_sdk")
RULE = "# Refund amounts\nApplies to: authorized standard-return batches\nRefund merchandise_paid_cents only.\n"


def load(framework):
    path = Path(__file__).parents[1] / "examples" / framework / "05_reflect_failure.py"
    spec = importlib.util.spec_from_file_location(f"failure_{framework}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_failure_examples_help_without_credentials_or_frameworks(framework, capsys):
    module = load(framework)
    with pytest.raises(SystemExit) as exc:
        module.main(framework, None, None, "test-model", "NO_KEY", ["--help"])
    assert exc.value.code == 0 and framework in capsys.readouterr().out


class Policy:
    """A fixture, not a reflection implementation; decisions inspect actual model input."""

    def __init__(self, batch, *, observable=True, excerpt=False):
        self.batch = batch
        self.observable = observable
        self.excerpt = excerpt
        self.trace = None
        self.phase = "task"
        self.steps = {}
        self.inputs = []

    def choose(self, text):
        self.inputs.append(text)
        if self.observable:
            with self.trace.span("Local fixture model", kind="LLM") as span:
                span.set_input({"messages": [{"role": "user", "content": text}]})
        step = self.steps.get(self.phase, 0)
        self.steps[self.phase] = step + 1
        if self.phase == "reflection":
            if step == 0:
                return "reflect_bash", {"command": "cat /inbox/notice.json"}
            if step == 1:
                return "reflect_bash", {"command": "cat > /rules/refunds.md <<'RULE'\n" + RULE + "RULE"}
            return "Proposed guidance for independent validation."
        if step == 0:
            pattern = "merchandise_paid" if self.excerpt else "Refund|Applies"
            return "reflect_bash", {"command": f"grep -RnE '{pattern}' /rules 2>/dev/null || true"}
        if step == 1:
            return "lookup_refund_order", {"order_id": self.batch.order.id}
        if step == 2:
            assert "merchandise_paid_cents" in text  # Actual order response reached this request.
            order = self.batch.order
            amount = order.merchandise_paid_cents
            if "Refund merchandise_paid_cents only." not in text:
                amount += order.outbound_shipping_cents
            return "submit_refund", {"order_id": order.id, "amount_cents": amount, "idempotency_key": order.id}
        return "Refund posted." if self.batch.ledger else "Refund rejected; batch closed."


def local_binding(framework, module, runtime, batch, output, validation, monkeypatch, *, excerpt=False):
    """Swap only provider transport/model behavior, preserving native agent/tool loops."""
    policy = Policy(batch, observable=framework != "claude_agent_sdk", excerpt=excerpt)
    if framework in ("langchain", "langgraph", "deepagents"):
        import langchain_openai
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class LocalModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "refund-fixture"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                pending = set()
                for message in messages:
                    if isinstance(message, ToolMessage):
                        assert message.tool_call_id in pending, "Tool reply has no preceding unanswered call"
                        pending.remove(message.tool_call_id)
                    else:
                        assert not pending, "Tool calls must be answered before the next message"
                        pending = (
                            {call["id"] for call in message.tool_calls} if isinstance(message, AIMessage) else set()
                        )
                assert not pending
                value = policy.choose(input_text(messages))
                answer = (
                    AIMessage(content=value)
                    if isinstance(value, str)
                    else AIMessage(content="", tool_calls=[{"name": value[0], "args": value[1], "id": uuid4().hex}])
                )
                return ChatResult(generations=[ChatGeneration(message=answer)])

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: LocalModel())
    elif framework == "crewai":
        import crewai
        from crewai.llms.base_llm import BaseLLM

        class LocalModel(BaseLLM):
            def supports_function_calling(self):
                return False

            def call(self, messages, **kwargs):
                value = policy.choose(input_text(messages))
                if isinstance(value, str):
                    return "Final Answer: " + value
                return f"Thought: Use the tool.\nAction: {value[0]}\nAction Input: {json.dumps(value[1])}"

            async def acall(self, messages, **kwargs):
                return self.call(messages, **kwargs)

        monkeypatch.setattr(crewai, "LLM", lambda **kwargs: LocalModel(model="fixture"))
    elif framework == "claude_agent_sdk":
        import claude_agent_sdk as sdk

        handlers = {}
        original_server = sdk.create_sdk_mcp_server

        def server(**kwargs):
            handlers.update({tool.name: tool.handler for tool in kwargs["tools"]})
            return original_server(**kwargs)

        class LocalClient:
            def __init__(self, *, options):
                self.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def query(self, message):
                self.message = message

            async def receive_response(self):
                text = str(self.options.system_prompt) + self.message
                while True:
                    value = policy.choose(text)
                    if isinstance(value, str):
                        break
                    result = await handlers[value[0]](value[1])
                    text += "\n" + input_text(result)
                yield sdk.ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="mock-native-session",
                    result=value,
                )

        monkeypatch.setattr(sdk, "create_sdk_mcp_server", server)
        monkeypatch.setattr(sdk, "ClaudeSDKClient", LocalClient)

    binding = module.build(runtime, batch, "fixture", output, validation=validation)
    if framework in ("langchain", "langgraph", "deepagents"):
        binding.kwargs["config"]["callbacks"] = []  # Host boundary below supplies fixture traces.
    elif framework == "openai_agents":
        from agents import Model, ModelResponse, RunConfig, Usage
        from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

        class LocalModel(Model):
            async def get_response(self, system_instructions, input, *args, **kwargs):
                value = policy.choose(input_text([system_instructions, input]))
                if isinstance(value, str):
                    message = ResponseOutputMessage(
                        id=uuid4().hex,
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[ResponseOutputText(type="output_text", text=value, annotations=[])],
                    )
                else:
                    message = ResponseFunctionToolCall(
                        id=uuid4().hex,
                        call_id=uuid4().hex,
                        type="function_call",
                        name=value[0],
                        arguments=json.dumps(value[1]),
                    )
                return ModelResponse(output=[message], usage=Usage(), response_id=None)

            def stream_response(self, *args, **kwargs):
                raise NotImplementedError

        binding.agent.model = LocalModel()
        binding.kwargs["run_config"] = RunConfig(tracing_disabled=True)
    elif framework == "google_adk":
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        class LocalModel(BaseLlm):
            model: str = "fixture"

            async def generate_content_async(self, llm_request, stream=False):
                value = policy.choose(input_text(llm_request))
                part = (
                    types.Part(text=value)
                    if isinstance(value, str)
                    else types.Part(function_call=types.FunctionCall(name=value[0], args=value[1]))
                )
                yield LlmResponse(content=types.Content(role="model", parts=[part]))

        binding.agent.agent.model = LocalModel()
    return binding, policy


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("excerpt", [False, True])
async def test_failure_native_framework_lifecycle_and_numbered_delivery(framework, excerpt, tmp_path, monkeypatch):
    dependency = {"openai_agents": "agents", "google_adk": "google.adk"}.get(framework, framework)
    pytest.importorskip(dependency)
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crew"))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    module = load(framework)
    from _support.failure_runner import arguments, exercise
    from _support.runner import AgentBinding, reply_text

    sdk = Client(api_key="test", project_name="test", enabled=False)
    object.__setattr__(sdk.config, "enabled", True)
    sdk._transport = MagicMock()
    traces, runs, batches = {}, {}, []

    def enqueue(payload):
        value = {
            **payload,
            "spans": [
                {**span, "trace_id": payload["trace_id"], "metadata": span.get("metadata") or {}}
                for span in payload.get("spans", [])
            ],
        }
        traces[payload["trace_id"]] = Trace(project_id=uuid4(), **value)

    sdk._transport.enqueue_trace.side_effect = enqueue

    async def visible(trace_id, **kwargs):
        return traces[trace_id]

    async def create(trace_ids, metrics, **kwargs):
        run = EvaluationRun(
            id=uuid4(),
            status="COMPLETED",
            metric_names=metrics,
            target_type="TRACE",
            total_targets=len(trace_ids),
            evaluated_count=len(trace_ids),
        )
        runs[str(run.id)] = run
        return run

    async def wait(run_id, *, trace_ids, metrics, **kwargs):
        return EvaluationResult(
            run=runs[run_id],
            scores=[
                Score(
                    id=uuid4(),
                    trace_id=tid,
                    name=metric,
                    value=str(traces[tid].metadata["verified_score"]),
                    status="SUCCESS",
                    data_type="NUMERIC",
                    eval_run_id=run_id,
                )
                for tid in trace_ids
                for metric in metrics
            ],
        )

    api = SimpleNamespace(
        traces=SimpleNamespace(wait_visible=AsyncMock(side_effect=visible), get=AsyncMock(side_effect=visible)),
        evaluations=SimpleNamespace(create=AsyncMock(side_effect=create), wait=AsyncMock(side_effect=wait)),
    )

    def build(runtime, batch, model, workdir, *, validation=False):
        batches.append(batch)
        binding, policy = local_binding(
            framework, module, runtime, batch, workdir, validation, monkeypatch, excerpt=excerpt
        )
        invoke = invoker_for(binding.agent, **binding.kwargs)

        async def traced(request):
            policy.phase = request.phase
            with sdk.trace("Native fixture refund") as trace:
                policy.trace = trace
                result = await invoke(request)
                trace.set_metadata({"verified_score": float(batch.verify(reply_text(result)).success)})
                trace.set_output({"messages": [{"role": "assistant", "content": reply_text(result)}]})
                return result

        return AgentBinding(traced)

    output = tmp_path / "run"
    output.mkdir()
    config = ReflectConfig(workspace=output / "workspace", thresholds={"task_completion": 0.8}, diagnostic_metrics=())
    try:
        async with Reflect(sdk, config=config, api=api) as runtime:
            report = await exercise(runtime, build, arguments([]), output, framework=framework)
    finally:
        sdk.shutdown()
    if framework == "claude_agent_sdk":
        # CLI output is not proof of the exact subsequent model input. Never invent that attestation.
        assert report["status"] == "incomplete" and report["stopped_at"] == "fresh_delivery"
        assert report["validations"][0]["outcome"] == "accepted"
        assert report["accepted_rules"] and report["milestones"]["fresh_task_verified"]
    else:
        assert report["status"] == "completed", report.get("reason")
        assert all(report["milestones"].values())
        assert len(batches) == 8 and len({id(batch) for batch in batches}) == 8
        delivery = report["turns"][1]["task_delivery"]
        assert delivery["before_submission_excerpt_versions"] == report["expected_accepted_versions"]
        assert bool(delivery["before_submission_versions"]) is not excerpt
        if excerpt:
            assert delivery["partial_model_input_versions"] == report["expected_accepted_versions"]
    assert len(report["replay_arms"]) == 6
    assert all(row["initial_active_memory_bytes"] == 0 for row in report["replay_arms"])
    assert api.evaluations.create.await_count == 2
    assert {t.metadata["pandaprobe.reflect"]["role"] for t in traces.values()} == {"task", "reflection", "validation"}
