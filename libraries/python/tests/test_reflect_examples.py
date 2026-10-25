"""Offline verification of live example wiring. No model/platform credentials used."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pandaprobe import Client
from pandaprobe.reflect import Invocation, Reflect, ReflectConfig
from pandaprobe.reflect.integrations import invoker_for

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FRAMEWORKS = ("langchain", "langgraph", "deepagents", "openai_agents", "google_adk", "crewai", "claude_agent_sdk")


def load(framework="langchain"):
    spec = importlib.util.spec_from_file_location(
        f"reflect_example_{framework}", EXAMPLES / framework / "04_reflect.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def support():
    load()  # Imports shared application helpers, not a framework or provider.
    from _support import scenario

    return scenario


@pytest.fixture
def runtime(tmp_path):
    sdk = Client(api_key="test", project_name="test", enabled=False)
    return Reflect(sdk, config=ReflectConfig(workspace=tmp_path), api=SimpleNamespace(aclose=AsyncMock()))


@pytest.mark.parametrize("index", range(6))
def test_support_policy_oracle_authorization_and_idempotency(support, index):
    ticket = support.tickets()[index]
    desk = support.SupportDesk(ticket)
    expected = ticket.expected
    assert "error" in json.loads(desk.resolve_ticket(**expected, idempotency_key=ticket.id))
    assert not desk.ledger
    desk.begin("inspect")
    desk.lookup_order(ticket.id)
    desk.search_policy("returns")
    assert desk.verify("Options").success
    desk.begin("resolve")
    assert json.loads(desk.resolve_ticket(**expected, idempotency_key=ticket.id))["status"] == "applied"
    assert desk.verify("Done").success
    desk.begin("repeat")
    desk.lookup_order(ticket.id)
    assert json.loads(desk.resolve_ticket(**expected, idempotency_key=ticket.id))["status"] == "already_applied"
    assert desk.verify("Already done").success
    assert desk.ledger == [expected]


def test_wrong_amount_and_old_policy_never_pass_verification(support):
    ticket = support.tickets()[3]  # Now outside the 14-day window.
    desk = support.SupportDesk(ticket, stage="resolve")
    desk.resolve_ticket(ticket.id, "refund", 4200, "returns-v1", ticket.id)
    assert desk.verify("Refunded successfully!").success is False
    assert not desk.ledger
    desk.resolve_ticket(**ticket.expected, idempotency_key=ticket.id)
    # Correct recovery does not erase the invalid attempt.
    assert desk.verify("Escalated").success is False
    desk.begin("repeat")
    desk.lookup_order(ticket.id)
    assert desk.verify("Already escalated").success


def test_fresh_replay_arms_and_policy_specific_held_out_cases(support):
    before = support.replay_cases("returns-v1")
    after = support.replay_cases("returns-v2")
    assert len(before) == 3 and sum(case.protected for case in before) == 1
    assert support.Ticket(**before[0].input).expected["action"] == "refund"
    assert support.Ticket(**after[0].input).expected["action"] == "escalate"
    left = support.SupportDesk(support.Ticket(**before[0].input), stage="resolve")
    right = support.SupportDesk(support.Ticket(**before[0].input), stage="resolve")
    left.resolve_ticket(**left.ticket.expected, idempotency_key=left.ticket.id)
    assert not right.ledger and not right.events
    assert {case.id for case in before}.isdisjoint(t.id for t in support.tickets())


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_all_examples_have_side_effect_free_help(framework, monkeypatch, capsys):
    from argparse import ArgumentParser

    monkeypatch.setattr(ArgumentParser, "exit", lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit(0)))
    module = load(framework)
    with pytest.raises(SystemExit):
        module.main(framework, module.build, lambda sdk: None, "test-model", "NO_KEY", ["--help"])
    assert "--max-validations" in capsys.readouterr().out


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_native_example_factories_wire_real_domain_tools(framework, runtime, support, tmp_path, monkeypatch):
    dependency = {"openai_agents": "agents", "google_adk": "google.adk"}.get(framework, framework)
    pytest.importorskip(dependency)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crew"))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    module = load(framework)
    desk = support.SupportDesk(support.tickets()[0])
    model = {
        "crewai": "openai/gpt-6-luna",
        "google_adk": "gemini-3.8-flash",
        "claude_agent_sdk": "claude-haiku-5-5",
    }.get(framework, "gpt-6-luna")
    binding = module.build(runtime, desk, model, tmp_path, validation=False)
    if framework in ("langchain", "langgraph", "deepagents"):
        assert hasattr(binding.agent, "ainvoke")
        assert binding.kwargs["config"]["callbacks"][0]._client is runtime.client
    elif framework == "google_adk":
        assert binding.agent.session_service is not None
        assert sum(getattr(t, "name", "") == "reflect_bash" for t in binding.agent.agent.tools) == 1
    elif framework == "claude_agent_sdk":
        assert binding.agent.tools == [] and binding.agent.setting_sources == []
        assert len(binding.agent.allowed_tools) == 3
        assert "support" in binding.agent.mcp_servers
    else:
        tools = {tool.name: tool for tool in binding.agent.tools}
        assert set(tools) == {"lookup_order", "search_policy", "resolve_ticket", "reflect_bash"}
        if framework == "openai_agents":
            fresh = module.build(runtime, desk, model, tmp_path, validation=True)
            assert "session" in binding.kwargs and "session" not in fresh.kwargs
        else:
            value = tools["lookup_order"].run(order_id=desk.ticket.id)
            assert json.loads(value)["id"] == desk.ticket.id
    await runtime.aclose()


@pytest.mark.parametrize("framework", ["langchain", "langgraph", "deepagents"])
async def test_example_native_agent_runs_two_turns_and_domain_tools(
    framework, runtime, support, tmp_path, monkeypatch
):
    pytest.importorskip(framework)
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    import langchain_openai

    class LocalModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    ticket = support.tickets()[0]
    model = LocalModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup_order", "args": {"order_id": ticket.id}, "id": "order"},
                    {"name": "search_policy", "args": {"query": "returns"}, "id": "policy"},
                ],
            ),
            AIMessage(content="The refundable merchandise amount is 6500 cents, awaiting authorization."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "resolve_ticket",
                        "args": {**ticket.expected, "idempotency_key": ticket.id},
                        "id": "resolve",
                    },
                ],
            ),
            AIMessage(content="Refund applied."),
        ]
    )
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: model)
    module = load(framework)
    desk = support.SupportDesk(ticket)
    binding = module.build(runtime, desk, "local", tmp_path)
    for stage in ("inspect", "resolve"):
        desk.begin(stage)
        async with runtime.execution(session_id="customer") as execution:
            answer = await invoker_for(binding.agent, **binding.kwargs)(
                Invocation(
                    input=ticket.prompt(stage),
                    execution=execution,
                )
            )
        assert desk.verify(answer["messages"][-1].content).success
    assert desk.ledger == [ticket.expected]
    await runtime.aclose()


def test_report_extracts_native_results_without_treating_errors_as_answers(support):
    from _support.runner import reply_text

    assert reply_text({"messages": [SimpleNamespace(content="hello")]}) == "hello"
    assert (
        reply_text([SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="ADK answer")]))])
        == "ADK answer"
    )
    assert reply_text(SimpleNamespace(raw="Crew answer")) == "Crew answer"
    assert reply_text(SimpleNamespace(final_output="OpenAI answer")) == "OpenAI answer"
    with pytest.raises(RuntimeError):
        reply_text([SimpleNamespace(is_error=True, result="error")])


async def test_live_runner_reports_real_lifecycle_without_live_services(support, tmp_path):
    """Same application runner, real Bash/replay, deterministic model and HTTP fixtures."""
    import httpx
    from uuid import uuid4
    from unittest.mock import MagicMock
    from _support.runner import AgentBinding, arguments, exercise
    from pandaprobe.api import PlatformClient

    sdk = Client(api_key="test", project_name="test", enabled=False)
    payloads, runs = {}, {}
    project_id = str(uuid4())

    def submit(payload):
        payloads[payload["trace_id"]] = {"project_id": project_id, **payload}

    object.__setattr__(sdk.config, "enabled", True)
    sdk._transport = MagicMock()
    sdk._transport.enqueue_trace.side_effect = submit

    def respond(request):
        path = request.url.path
        if path.startswith("/traces/"):
            return httpx.Response(200, json=payloads[path.split("/")[-1]])
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
                        "value": str(payloads[tid]["metadata"]["verified_score"]),
                        "status": "SUCCESS",
                        "data_type": "NUMERIC",
                        "reason": "Fixture ledger measurement",
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

    config = ReflectConfig(
        workspace=tmp_path / "workspace",
        thresholds={"task_completion": 0.8},
        diagnostic_metrics=(),
        cooldown_seconds=0,
        poll_interval=0.01,
    )
    rule = "# Refund amounts\nApplies to: refund support tickets\nUse the paid merchandise amount.\n"
    desks = []

    def build(runtime, desk, model, workdir, *, validation=False):
        desks.append(desk)

        async def agent(request):
            with sdk.trace("scripted support", metadata={"verified_score": 1.0}) as trace:
                if request.phase == "reflection":
                    await runtime.bash("cat /inbox/notice.json")
                    await runtime.bash("cat > /rules/refunds.md <<'EOF'\n" + rule + "EOF")
                    return "Proposed refund guidance"
                guidance = await runtime.bash("cat /rules/refunds.md 2>/dev/null || true")
                desk.lookup_order(desk.ticket.id)
                desk.search_policy("returns")
                if desk.stage != "inspect":
                    expected = desk.ticket.expected
                    if desk.ticket.id in ("N1-101", "H-201") and "paid merchandise" not in guidance:
                        expected = {**expected, "amount_cents": 9999}
                    desk.resolve_ticket(**expected, idempotency_key=desk.ticket.id)
                answer = "Resolved" if desk.ledger else "Reviewed"
                trace.set_metadata({"verified_score": float(desk.verify(answer).success)})
                trace.set_output({"messages": [{"role": "assistant", "content": answer}]})
                return answer

        return AgentBinding(agent)

    output = tmp_path / "report"
    output.mkdir()
    args = arguments("local", "scripted", ["--max-tickets", "1", "--max-validations", "1"])
    async with PlatformClient(sdk, transport=httpx.MockTransport(respond)) as api:
        async with Reflect(sdk, config=config, api=api) as runtime:
            report = await exercise(runtime, build, args, "local", output)
    assert report["status"] == "completed"
    assert len(report["turns"]) == 3
    assert [row["verification"]["success"] for row in report["turns"]] == [True, False, True]
    assert report["validations"][0]["outcome"] == "accepted"
    assert len(report["replay_arms"]) == 6
    assert len({id(desk) for desk in desks}) == 7
    assert all(row["reflect"]["evaluation"]["status"] == "evaluated" for row in report["turns"])
    assert report["accepted_rules"]["refunds.md"]["version"]
    assert {row["role"] for row in report["trace_usage"]} == {"task", "reflection", "validation"}
    assert all(
        payloads[tid]["metadata"]["pandaprobe.reflect"]["role"] == "task"
        for query in runs.values()
        for tid in query["trace_ids"]
    )
    assert json.loads((output / "report.json").read_text())["status"] == "completed"
    sdk.shutdown()


async def test_example_openai_agents_native_multiturn_tools(runtime, support, tmp_path, monkeypatch):
    pytest.importorskip("agents")
    from agents import Model, ModelResponse, RunConfig, Usage
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    class LocalModel(Model):
        def __init__(self):
            self.index = 0

        async def get_response(self, *args, **kwargs):
            self.index += 1
            if self.index in (1, 3):
                specs = [("lookup_order", {"order_id": desk.ticket.id}), ("search_policy", {"query": "returns"})]
                if self.index == 3:
                    specs = [("resolve_ticket", {**desk.ticket.expected, "idempotency_key": desk.ticket.id})]
                output = [
                    ResponseFunctionToolCall(
                        id=f"t{self.index}{i}",
                        call_id=f"c{self.index}{i}",
                        type="function_call",
                        name=name,
                        arguments=json.dumps(arguments),
                    )
                    for i, (name, arguments) in enumerate(specs)
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        id=f"m{self.index}",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[ResponseOutputText(type="output_text", text="Reviewed", annotations=[])],
                    )
                ]
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    from _support.runner import reply_text

    module = load("openai_agents")
    desk = support.SupportDesk(support.tickets()[0])
    binding = module.build(runtime, desk, "local", tmp_path)
    binding.agent.model = LocalModel()
    binding.kwargs["run_config"] = RunConfig(tracing_disabled=True)
    for stage in ("inspect", "resolve"):
        desk.begin(stage)
        async with runtime.execution(session_id="customer") as execution:
            result = await invoker_for(binding.agent, **binding.kwargs)(
                Invocation(input=binding.input(desk.ticket.prompt(stage)), execution=execution)
            )
        assert desk.verify(reply_text(result)).success
    items = await binding.kwargs["session"].get_items()
    assert len([item for item in items if item.get("role") == "user"]) == 2
    await runtime.aclose()


async def test_example_google_adk_native_multiturn_tools(runtime, support, tmp_path):
    pytest.importorskip("google.adk")
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from pydantic import PrivateAttr
    from _support.runner import reply_text

    class LocalModel(BaseLlm):
        model: str = "local"
        _index: int = PrivateAttr(default=0)

        async def generate_content_async(self, llm_request, stream=False):
            self._index += 1
            if self._index == 1:
                parts = [
                    types.Part(
                        function_call=types.FunctionCall(name="lookup_order", args={"order_id": desk.ticket.id})
                    ),
                    types.Part(function_call=types.FunctionCall(name="search_policy", args={"query": "returns"})),
                ]
            elif self._index == 3:
                parts = [
                    types.Part(
                        function_call=types.FunctionCall(
                            name="resolve_ticket", args={**desk.ticket.expected, "idempotency_key": desk.ticket.id}
                        )
                    )
                ]
            else:
                parts = [types.Part(text="Reviewed")]
            yield LlmResponse(content=types.Content(role="model", parts=parts))

    module = load("google_adk")
    desk = support.SupportDesk(support.tickets()[0])
    binding = module.build(runtime, desk, "local", tmp_path)
    binding.agent.agent.model = LocalModel()
    for stage in ("inspect", "resolve"):
        desk.begin(stage)
        async with runtime.execution(session_id="customer") as execution:
            result = await invoker_for(binding.agent, **binding.kwargs)(
                Invocation(input=desk.ticket.prompt(stage), execution=execution)
            )
        assert desk.verify(reply_text(result)).success
    sessions = await binding.agent.session_service.list_sessions(app_name="northstar_support", user_id="reflect")
    assert len(sessions.sessions) == 1
    await runtime.aclose()


async def test_example_crewai_native_multiturn_tools(runtime, support, tmp_path, monkeypatch):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crew"))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    crewai = pytest.importorskip("crewai")
    from crewai.llms.base_llm import BaseLLM
    from _support.runner import reply_text

    calls = []

    class LocalModel(BaseLLM):
        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            calls.append(messages)
            index = len(calls)
            if index in (1, 2, 4):
                name, arguments = {
                    1: ("lookup_order", {"order_id": desk.ticket.id}),
                    2: ("search_policy", {"query": "returns"}),
                    4: ("resolve_ticket", {**desk.ticket.expected, "idempotency_key": desk.ticket.id}),
                }[index]
                return f"Thought: Use support tools.\nAction: {name}\nAction Input: {json.dumps(arguments)}"
            return "Final Answer: Reviewed"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    model = LocalModel(model="local")
    monkeypatch.setattr(crewai, "LLM", lambda **kwargs: model)
    module = load("crewai")
    desk = support.SupportDesk(support.tickets()[0])
    binding = module.build(runtime, desk, "local", tmp_path)
    for stage in ("inspect", "resolve"):
        desk.begin(stage)
        message = desk.ticket.prompt(stage)
        async with runtime.execution(session_id="customer") as execution:
            result = await invoker_for(binding.agent, **binding.kwargs)(
                Invocation(input=binding.input(message), execution=execution)
            )
        binding.remember(message, result)
        assert desk.verify(reply_text(result)).success
    assert len(binding.history) == 4
    await runtime.aclose()


async def test_example_claude_native_tools_and_continuation(runtime, support, tmp_path, monkeypatch):
    """Native MCP handlers, mocked CLI transport; no assertion about live tool-body delivery."""
    sdk = pytest.importorskip("claude_agent_sdk")
    from _support.runner import reply_text

    handlers, observed = {}, []
    native_server = sdk.create_sdk_mcp_server

    def server(**kwargs):
        if kwargs["name"] == "support":
            handlers.update({tool.name: tool.handler for tool in kwargs["tools"]})
        return native_server(**kwargs)

    class LocalClient:
        def __init__(self, *, options):
            observed.append(options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def query(self, message):
            await handlers["lookup_order"]({"order_id": desk.ticket.id})
            await handlers["search_policy"]({"query": "returns"})
            if desk.stage == "resolve":
                await handlers["resolve_ticket"]({**desk.ticket.expected, "idempotency_key": desk.ticket.id})

        async def receive_response(self):
            yield sdk.ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="native-conversation",
                result="Reviewed",
            )

    monkeypatch.setattr(sdk, "create_sdk_mcp_server", server)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", LocalClient)
    module = load("claude_agent_sdk")
    desk = support.SupportDesk(support.tickets()[0])
    binding = module.build(runtime, desk, "local", tmp_path)
    for stage in ("inspect", "resolve"):
        desk.begin(stage)
        message = desk.ticket.prompt(stage)
        async with runtime.execution(session_id="customer") as execution:
            result = await invoker_for(binding.agent)(Invocation(input=message, execution=execution))
        binding.remember(message, result)
        assert desk.verify(reply_text(result)).success
    assert observed[0].resume is None and observed[1].resume == "native-conversation"
    assert all(options.tools == [] for options in observed)
    await runtime.aclose()


async def test_runner_preserves_failed_turn_and_partial_report(runtime, support, tmp_path):
    from _support.runner import AgentBinding, arguments, exercise

    def build(*args, **kwargs):
        async def fail(request):
            raise ValueError("model unavailable")

        return AgentBinding(fail)

    output = tmp_path / "failed-run"
    output.mkdir()
    args = arguments("local", "local", ["--max-tickets", "1"])
    with pytest.raises(ValueError, match="model unavailable"):
        await exercise(runtime, build, args, "local", output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["turns"][0]["error"] == "ValueError"
    assert report["summary"]["verified_successes"] == 0
    assert report["summary"]["task_errors"] == 1
    assert not report["accepted_rules"]
    await runtime.aclose()
