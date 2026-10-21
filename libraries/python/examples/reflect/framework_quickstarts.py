"""Small integration functions for existing, developer-owned agents/models.

These functions use the existing PandaProbe tracing integrations explicitly.
Reflect does not install those integrations. Pass the same resolved SDK Client
you already use, and your own model and domain tools. See README.md for installs.
"""

from pandaprobe.reflect import Reflect


async def langchain(sdk, model, domain_tools, message, session_id):
    """LangChain's create_agent compiles a LangGraph message-state agent."""
    from langchain.agents import create_agent

    from pandaprobe.integrations.langchain import LangChainCallbackHandler

    async with Reflect(sdk) as reflect:
        agent = reflect.agent(create_agent, model=model, tools=domain_tools)
        return await reflect.run(
            agent, message, session_id=session_id, config={"callbacks": [LangChainCallbackHandler(client=sdk)]}
        )


async def langgraph(sdk, model, domain_tools, message, session_id):
    """A persistent LangGraph conversation with the existing tracing callback."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

    async with Reflect(sdk) as reflect:
        agent = reflect.agent(create_react_agent, model=model, tools=domain_tools, checkpointer=InMemorySaver())
        return await reflect.run(
            agent,
            {"messages": [{"role": "user", "content": message}]},
            session_id=session_id,
            config={"callbacks": [LangGraphCallbackHandler(client=sdk)]},
        )


async def deepagents(sdk, model, domain_tools, message, session_id):
    """Deep Agents requires Python 3.11+ independently of Reflect's Python 3.10 support."""
    from deepagents import create_deep_agent

    from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler

    async with Reflect(sdk) as reflect:
        agent = reflect.agent(create_deep_agent, model=model, tools=domain_tools)
        return await reflect.run(
            agent, message, session_id=session_id, config={"callbacks": [DeepAgentsCallbackHandler(client=sdk)]}
        )


async def openai_agents(sdk, existing_agent, message, session_id):
    """Reuse the existing Agent; Reflect adds its tool to an invocation copy."""
    from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

    OpenAIAgentsAdapter(client=sdk).instrument()
    async with Reflect(sdk) as reflect:
        return await reflect.run(existing_agent, message, session_id=session_id)


async def google_adk(sdk, model, domain_tools, message, session_id):
    """Use the host's ADK model; no separately configured reflection model."""
    from google.adk.agents import LlmAgent

    from pandaprobe.integrations.google_adk import GoogleADKAdapter

    GoogleADKAdapter(client=sdk).instrument()
    async with Reflect(sdk) as reflect:
        agent = reflect.agent(LlmAgent, name="task_agent", model=model, tools=domain_tools)
        return await reflect.run(agent, message, session_id=session_id)


async def crewai(sdk, existing_agent_or_crew, input, session_id):
    """A Crew takes an inputs dictionary; a standalone Agent takes messages/text."""
    from pandaprobe.integrations.crewai import CrewAIAdapter

    CrewAIAdapter(client=sdk).instrument()
    async with Reflect(sdk) as reflect:
        return await reflect.run(existing_agent_or_crew, input, session_id=session_id)


async def claude_agent_sdk(sdk, existing_options, message, session_id):
    """Pass ClaudeAgentOptions before a client connects, allowing SDK MCP registration."""
    from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

    ClaudeAgentSDKAdapter(client=sdk).instrument()
    async with Reflect(sdk) as reflect:
        return await reflect.run(existing_options, message, session_id=session_id)
