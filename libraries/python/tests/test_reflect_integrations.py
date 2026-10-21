"""Installed-framework tests with deterministic local models, never paid calls."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from pandaprobe import Client
from pandaprobe.reflect import Invocation, Reflect, ReflectConfig, Rule, ValidationVerdict
from pandaprobe.reflect.integrations import invoker_for

TEXT = "# Math\nApplies to: addition tasks\nCheck sums with a calculator.\n"


def setup(tmp_path):
    sdk = Client(api_key="key", project_name="p", enabled=False)
    runtime = Reflect(sdk, config=ReflectConfig(workspace=tmp_path), api=SimpleNamespace(aclose=AsyncMock()))
    proposal = runtime.workspace.propose(
        runtime.workspace.state()["head"],
        {"math.md": Rule(path="math.md", text=TEXT)},
        session_id="s",
        execution_id="author",
    ).proposal
    runtime.workspace.decide(
        ValidationVerdict(proposal_id=proposal.id, outcome="accepted", reason="fixture", mode="replay")
    )
    return runtime


async def test_langchain_real_agent_tool_delivery_memory_replacement_and_continuation(tmp_path):
    pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from pandaprobe.integrations.langchain import LangChainCallbackHandler

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "reflect_bash", "args": {"command": "cat /rules/math.md"}, "id": "read"}],
            ),
            AIMessage(content="Task answer"),
            AIMessage(content="Reflection complete"),
        ]
    )
    runtime = setup(tmp_path)
    object.__setattr__(runtime.client.config, "enabled", True)
    runtime.client._transport = MagicMock()
    agent = runtime.agent(
        create_agent, model=model, system_prompt="Keep this developer instruction.", checkpointer=InMemorySaver()
    )
    invoke = invoker_for(agent, config={"callbacks": [LangChainCallbackHandler(client=runtime.client)]})
    async with runtime.execution(session_id="customer") as execution:
        await runtime.bash("echo 'old memory' > /context.md")
        first = await invoke(Invocation(input="Do the task", execution=execution))
        assert first["messages"][-1].content == "Task answer"
        assert any(isinstance(m, ToolMessage) and TEXT.strip() in m.content for m in first["messages"])
        assert Rule(path="math.md", text=TEXT).version in execution.delivered_versions
        await runtime.bash("echo 'new memory' > /context.md")
        async with execution.activate("reflection"):
            second = await invoke(
                Invocation(
                    input="Do the task",
                    execution=execution,
                    phase="reflection",
                    previous_result=first,
                    feedback="Review the result",
                )
            )
        memory = [m for m in second["messages"] if m.id == "pandaprobe-reflect-active-memory"]
        assert len(memory) == 1 and "new memory" in memory[0].content and "old memory" not in memory[0].content
    payloads = [call.args[0] for call in runtime.client._transport.enqueue_trace.call_args_list]
    assert len(payloads) == 2  # Reuse the existing tracing callback; no second tracing patch.
    assert [p["metadata"]["pandaprobe.reflect"]["role"] for p in payloads] == ["task", "reflection"]
    assert execution.trace_ids == [payloads[0]["trace_id"]]
    await runtime.aclose()


async def test_native_langgraph_without_langchain_supplies_tool_and_blocks_domain_actions_in_reflection(tmp_path):
    pytest.importorskip("langgraph")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    called = []

    @tool
    def domain_action() -> str:
        """Perform a domain action."""
        called.append(True)
        return "performed"

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    runtime = setup(tmp_path)
    model = Model(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "domain_action", "args": {}, "id": "domain"}]),
            AIMessage(
                content="",
                tool_calls=[{"name": "reflect_bash", "args": {"command": "cat /rules/math.md"}, "id": "read"}],
            ),
            AIMessage(content="Done"),
        ]
    )
    agent = runtime.agent(create_react_agent, model=model, tools=[domain_action])
    async with runtime.execution(session_id="s") as execution:
        async with execution.activate("reflection"):
            result = await invoker_for(agent)(Invocation(input="Reflect", execution=execution))
        assert not called
        assert any(isinstance(item, ToolMessage) and item.status == "error" for item in result["messages"])
        assert Rule(path="math.md", text=TEXT).version in execution.delivered_versions
    await runtime.aclose()


async def test_openai_agents_real_runner_tools_history_and_existing_hooks(tmp_path):
    pytest.importorskip("agents")
    from agents import Agent, Model, ModelResponse, RunConfig, RunHooks, Usage
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    class LocalModel(Model):
        def __init__(self):
            self.calls = []

        async def get_response(self, system_instructions, input, *args, **kwargs):
            self.calls.append((system_instructions, input))
            if len(self.calls) == 1:
                output = [
                    ResponseFunctionToolCall(
                        id="tool1",
                        call_id="read",
                        type="function_call",
                        name="reflect_bash",
                        arguments=json.dumps({"command": "cat /rules/math.md"}),
                    )
                ]
            else:
                output = [
                    ResponseOutputMessage(
                        id=str(uuid4()),
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[ResponseOutputText(type="output_text", text="Answer", annotations=[])],
                    )
                ]
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    class Hooks(RunHooks):
        calls = 0

        async def on_llm_start(self, *args):
            self.calls += 1

    runtime = setup(tmp_path)
    model, hooks = LocalModel(), Hooks()
    agent = Agent(name="task", instructions="Original instructions", model=model)
    invoke = invoker_for(agent, hooks=hooks, run_config=RunConfig(tracing_disabled=True))
    async with runtime.execution(session_id="s") as execution:
        result = await invoke(Invocation(input="Do task", execution=execution))
        assert result.final_output == "Answer" and hooks.calls == 2
        assert Rule(path="math.md", text=TEXT).version in execution.delivered_versions
        assert model.calls[0][0].startswith("Original instructions")
        await invoke(
            Invocation(
                input="Do task",
                execution=execution,
                phase="reflection",
                previous_result=result,
                feedback="Inspect failure evidence",
            )
        )
        assert any(item.get("type") == "function_call_output" for item in model.calls[-1][1])
        assert model.calls[-1][1][-1]["content"] == "Inspect failure evidence"
    assert not agent.tools and agent.instructions == "Original instructions"
    await runtime.aclose()


async def test_adk_real_model_loop_supplied_tool_memory_and_session_history(tmp_path):
    pytest.importorskip("google.adk")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from pydantic import PrivateAttr

    class LocalModel(BaseLlm):
        model: str = "local-model"
        _calls: list = PrivateAttr(default_factory=list)

        async def generate_content_async(self, llm_request, stream=False):
            self._calls.append(llm_request.model_copy(deep=True))
            if len(self._calls) == 1:
                part = types.Part(
                    function_call=types.FunctionCall(name="reflect_bash", args={"command": "cat /rules/math.md"})
                )
            else:
                part = types.Part(text="Answer")
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    runtime = setup(tmp_path)
    model = LocalModel()
    agent = runtime.agent(LlmAgent, name="task_agent", model=model, instruction="Original instructions")
    invoke = invoker_for(agent)
    async with runtime.execution(session_id="s") as execution:
        events = await invoke(Invocation(input="Task", execution=execution))
        assert any(event.is_final_response() for event in events)
        assert len(model._calls) == 2
        assert model._calls[0].config.system_instruction.count("Reflect memory is available") == 1
        assert Rule(path="math.md", text=TEXT).version in execution.delivered_versions
        await invoke(
            Invocation(
                input="Task",
                execution=execution,
                phase="reflection",
                previous_result=events,
                feedback="Reflect on the evidence",
            )
        )
        assert any(
            part.function_response is not None for item in model._calls[-1].contents for part in item.parts or []
        )
    await runtime.aclose()


async def test_deepagents_factory_supplies_only_one_reflect_tool(tmp_path):
    pytest.importorskip("deepagents")
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert sum(getattr(t, "name", "") == "reflect_bash" for t in tools) == 1
            return self

    runtime = setup(tmp_path)
    agent = runtime.agent(create_deep_agent, model=Model(responses=[AIMessage(content="Done")]))
    async with runtime.execution(session_id="s") as execution:
        result = await invoker_for(agent)(Invocation(input="Task", execution=execution))
        assert result["messages"][-1].content == "Done"
    await runtime.aclose()


async def test_crewai_native_async_agent_and_tool_preserve_host_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crew-data"))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    pytest.importorskip("crewai")
    from crewai import Agent
    from crewai.llms.base_llm import BaseLLM

    seen = []

    class LocalModel(BaseLLM):
        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            seen.append(messages)
            if len(seen) == 1:
                return 'Thought: Read guidance.\nAction: reflect_bash\nAction Input: {"command": "cat /rules/math.md"}'
            return "Final Answer: Done"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    runtime = setup(tmp_path)
    tool = runtime.tool("crewai")
    model = LocalModel(model="local")
    agent = Agent(role="Task agent", goal="Solve the task", backstory="Use evidence", llm=model, verbose=False)
    async with runtime.execution(session_id="s") as execution:
        output = await tool._arun(command="cat /rules/math.md")
        assert TEXT.strip() in output
        threaded = await __import__("asyncio").to_thread(tool.run, command="cat /rules/math.md")
        assert TEXT.strip() in threaded
        result = await invoker_for(agent)(Invocation(input="Task", execution=execution))
        assert result.raw == "Done"
        assert len(seen) == 2 and TEXT.strip() in str(seen[-1]).replace("\\n", "\n")
        assert any(receipt.submitted for receipt in execution.shell.receipts)
    assert not agent.tools and agent.backstory == "Use evidence"
    await runtime.aclose()


async def test_claude_native_mcp_tool_and_options_preserve_existing_settings(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions

    from pandaprobe.reflect.adapters.claude_agent_sdk import QUALIFIED_TOOL, configure

    runtime = setup(tmp_path)
    options = ClaudeAgentOptions(system_prompt="Original", model="host-chosen-model", allowed_tools=["Read"])
    configured = configure(runtime, options)
    assert configured.model == options.model and configured.allowed_tools == ["Read", QUALIFIED_TOOL]
    assert options.allowed_tools == ["Read"] and not options.mcp_servers
    tool = runtime.tool("claude_agent_sdk")
    async with runtime.execution(session_id="s"):
        output = await tool.handler({"command": "cat /rules/math.md"})
        assert TEXT.strip() in output["content"][0]["text"]
    await runtime.aclose()


async def test_claude_native_invocation_scope_continuation_and_trial_isolation(tmp_path, monkeypatch):
    import asyncio

    sdk = pytest.importorskip("claude_agent_sdk")
    from pandaprobe.reflect.adapters.claude_agent_sdk import QUALIFIED_TOOL, SERVER

    runtime = setup(tmp_path)
    observed, queries = [], []

    class LocalClient:
        def __init__(self, *, options):
            observed.append(options)

        async def __aenter__(self):
            # Native MCP dispatch runs in child tasks created by this client.
            self.read = asyncio.create_task(runtime.bash("cat /rules/math.md"))
            return self

        async def __aexit__(self, *_):
            assert TEXT.strip() in await self.read

        async def query(self, value):
            queries.append(value)

        async def receive_response(self):
            yield sdk.ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="native-task-session",
                result="Answer",
            )

    monkeypatch.setattr(sdk, "ClaudeSDKClient", LocalClient)
    options = sdk.ClaudeAgentOptions(
        system_prompt="Original", tools=["Read"], allowed_tools=["Read"], resume="host-session"
    )
    invoke = invoker_for(options)
    async with runtime.execution(session_id="s") as execution:
        first = await invoke(Invocation(input="Task", execution=execution))
        async with execution.activate("reflection"):
            await invoke(
                Invocation(
                    input="Task",
                    execution=execution,
                    phase="reflection",
                    previous_result=first,
                    feedback="Review evidence",
                )
            )
        async with execution.activate("validation"):
            await invoke(Invocation(input="Fresh task", execution=execution, phase="validation"))
    assert observed[0].tools == ["Read"] and observed[0].resume == "host-session"
    assert observed[1].resume == "native-task-session" and observed[1].tools == []
    assert observed[1].allowed_tools == [QUALIFIED_TOOL] and list(observed[1].mcp_servers) == [SERVER]
    assert getattr(observed[1], "strict_mcp_config", False) or "strict-mcp-config" in observed[1].extra_args
    assert observed[2].resume is None and not observed[2].continue_conversation
    assert queries == ["Task", "Review evidence", "Fresh task"]
    assert all(p.system_prompt.count("Reflect memory is available") == 1 for p in observed)
    assert options.system_prompt == "Original" and options.allowed_tools == ["Read"]
    await runtime.aclose()


async def test_crewai_native_crew_reflects_with_final_task_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crew-data"))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    pytest.importorskip("crewai")
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM

    seen, stops = [], []

    class LocalModel(BaseLLM):
        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            seen.append(messages)
            stops.append(list(self.stop))
            return "Final Answer: Done"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    runtime = setup(tmp_path)
    model = LocalModel(model="local")
    model.stop = ["HOST_END"]
    member = Agent(role="Analyst", goal="Solve tasks", backstory="Use evidence", llm=model)
    task = Task(description="Solve {problem}", expected_output="Answer", agent=member)
    crew = Crew(agents=[member], tasks=[task], verbose=False)
    invoke = invoker_for(crew)
    async with runtime.execution(session_id="s") as execution:
        first = await invoke(Invocation(input={"problem": "addition"}, execution=execution))
        assert first.raw == "Done"
        async with execution.activate("reflection"):
            second = await invoke(
                Invocation(
                    input={},
                    execution=execution,
                    phase="reflection",
                    previous_result=first,
                    feedback="Reflect on evidence",
                )
            )
        assert second.raw == "Done"
    assert len(seen) == 2 and "Reflect memory is available" in str(seen[0])
    assert "HOST_END" in stops[0]
    assert "Reflect on evidence" in str(seen[1])
    assert task.description == "Solve {problem}" and not member.tools
    await runtime.aclose()
