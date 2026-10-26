"""Native agent factories shared by the two live Reflect workloads.

Only the selected factory imports its optional framework/provider. Each call
creates fresh native conversation state; Reflect still owns its single Bash tool.
"""

from __future__ import annotations

from uuid import uuid4

from .runner import AgentBinding


def build_langchain(runtime, desk, model, workdir, *, validation=False, instructions, input_audit=None):
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    from pandaprobe.integrations.langchain import LangChainCallbackHandler

    agent = runtime.agent(
        create_agent,
        model=ChatOpenAI(model=model, reasoning_effort="none", max_retries=0, timeout=60),
        tools=[StructuredTool.from_function(fn) for fn in desk.functions],
        system_prompt=instructions,
        checkpointer=InMemorySaver(),
        middleware=[input_audit.middleware()] if input_audit else [],
    )
    return AgentBinding(
        agent, {"config": {"callbacks": [LangChainCallbackHandler(client=runtime.client)], "recursion_limit": 32}}
    )


def build_langgraph(runtime, desk, model, workdir, *, validation=False, instructions):
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

    # Native LangGraph factory; LangChain's create_agent is an alternative if installed.
    agent = runtime.agent(
        create_react_agent,
        model=ChatOpenAI(model=model, reasoning_effort="none", max_retries=0, timeout=60),
        tools=[StructuredTool.from_function(fn) for fn in desk.functions],
        prompt=instructions,
        checkpointer=InMemorySaver(),
    )
    return AgentBinding(
        agent, {"config": {"callbacks": [LangGraphCallbackHandler(client=runtime.client)], "recursion_limit": 32}}
    )


def build_deepagents(runtime, desk, model, workdir, *, validation=False, instructions, input_audit=None):
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler

    agent = runtime.agent(
        create_deep_agent,
        model=ChatOpenAI(model=model, reasoning_effort="none", max_retries=0, timeout=60),
        tools=[StructuredTool.from_function(fn) for fn in desk.functions],
        system_prompt=instructions
        + "\nUse reflect_bash for /rules and /context.md; other files are task scratch space.",
        checkpointer=InMemorySaver(),
        backend=StateBackend(),
        middleware=[input_audit.middleware()] if input_audit else [],
    )
    return AgentBinding(
        agent, {"config": {"callbacks": [DeepAgentsCallbackHandler(client=runtime.client)], "recursion_limit": 40}}
    )


def build_openai_agents(runtime, desk, model, workdir, *, validation=False, instructions):
    from agents import Agent, SQLiteSession, function_tool

    agent = runtime.agent(
        Agent,
        name="Northstar Support",
        model=model,
        instructions=instructions,
        tools=[function_tool(fn) for fn in desk.functions],
    )
    kwargs = {"max_turns": 16}
    if not validation:
        kwargs["session"] = SQLiteSession(uuid4().hex)  # Includes the reflection continuation.
    return AgentBinding(agent, kwargs)


def instrument_openai_agents(sdk, *, tag="reflect-support"):
    from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(client=sdk, tags=[tag, "example"])
    adapter.instrument()
    return adapter


def build_google_adk(runtime, desk, model, workdir, *, validation=False, instructions):
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    agent = runtime.agent(
        LlmAgent, name="northstar_support", model=model, instruction=instructions, tools=desk.functions
    )
    runner = Runner(agent=agent, app_name="northstar_support", session_service=InMemorySessionService())
    return AgentBinding(runner, {"run_config": RunConfig(max_llm_calls=16)})


def instrument_google_adk(sdk, *, tag="reflect-support"):
    from pandaprobe.integrations.google_adk import GoogleADKAdapter

    adapter = GoogleADKAdapter(client=sdk, tags=[tag, "example"])
    adapter.instrument()
    return adapter


def build_crewai(runtime, desk, model, workdir, *, validation=False, instructions):
    from crewai import Agent, LLM
    from crewai.tools import tool

    # GPT-6 Luna requires "none" for Chat Completions tool calls. Use extra_body
    # because the locked CrewAI version forwards reasoning_effort only for o1.
    model_options = (
        {"extra_body": {"reasoning_effort": "none"}} if model in ("openai/gpt-6-luna", "gpt-6-luna") else {}
    )
    agent = runtime.agent(
        Agent,
        role="Northstar Support",
        goal="Resolve the customer's authorized support ticket correctly.",
        backstory=instructions,
        llm=LLM(model=model, **model_options),
        tools=[tool(fn.__name__)(fn) for fn in desk.functions],
        max_iter=16,
        allow_delegation=False,
        allow_code_execution=False,
        cache=False,
        verbose=False,
    )
    return AgentBinding(agent, history_mode="messages")


def instrument_crewai(sdk, *, tag="reflect-support"):
    from pandaprobe.integrations.crewai import CrewAIAdapter

    adapter = CrewAIAdapter(client=sdk, tags=[tag, "example"])
    adapter.instrument()
    return adapter


def build_claude_agent_sdk(runtime, desk, model, workdir, *, validation=False, instructions):
    import inspect
    from typing import get_type_hints

    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

    def mcp_tool(fn):
        hints = get_type_hints(fn)
        schema = {name: hints[name] for name in inspect.signature(fn).parameters}

        async def call(arguments):
            return {"content": [{"type": "text", "text": fn(**arguments)}]}

        return tool(fn.__name__, fn.__doc__, schema)(call)

    tools = [mcp_tool(fn) for fn in desk.functions]
    server = create_sdk_mcp_server(name="support", tools=tools)
    cwd = workdir / "claude" / uuid4().hex
    cwd.mkdir(parents=True)
    # No host Bash/filesystem tools. Reflect supplies its isolated Bash through MCP.
    # Empty settings prevent local project plugins/instructions changing the exercise.
    options = ClaudeAgentOptions(
        model=model,
        thinking={"type": "disabled"},
        effort="low",
        system_prompt=instructions,
        mcp_servers={"support": server},
        allowed_tools=[f"mcp__support__{fn.__name__}" for fn in desk.functions],
        tools=[],
        setting_sources=[],
        cwd=str(cwd),
        max_turns=16,
        extra_args={"strict-mcp-config": None},
    )
    return AgentBinding(options, history_mode="claude")


def instrument_claude_agent_sdk(sdk, *, tag="reflect-support"):
    from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

    adapter = ClaudeAgentSDKAdapter(client=sdk, tags=[tag, "example"])
    adapter.instrument()
    return adapter
