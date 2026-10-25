"""Lazy framework tool formatting and explicit invocation adapters.

This module never installs tracing patches. Framework-specific helpers are
imported only when the host explicitly requests their integration.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from .shell import TOOL_DESCRIPTION, TOOL_NAME

if TYPE_CHECKING:
    from .runtime import Reflect


def input_text(value: Any) -> str:
    """Extract text from nested framework messages without JSON-escaping newlines."""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return "\n".join(input_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(input_text(item) for item in value)
    return ""


def result_text(value: Any) -> str:
    """Retain the final agent explanation, separate from deterministic outcomes."""
    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]
    if isinstance(value, (list, tuple)):
        value = value[-1] if value else ""
    for name in ("final_output", "raw", "result", "content"):
        content = getattr(value, name, None)
        if content is not None:
            value = content
            break
    return (input_text(value) or str(value))[:4096]


def make_tool(runtime: Reflect, framework: str) -> Any:
    async def reflect_bash(command: str) -> str:
        return await runtime.bash(command)

    reflect_bash.__doc__ = TOOL_DESCRIPTION
    if framework == "generic":
        return reflect_bash
    if framework in ("langchain", "langgraph", "deepagents"):
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(coroutine=reflect_bash, name=TOOL_NAME, description=TOOL_DESCRIPTION)
    if framework == "openai_agents":
        from agents import function_tool

        return function_tool(reflect_bash)
    if framework == "google_adk":
        from google.adk.tools import FunctionTool

        return FunctionTool(reflect_bash)
    if framework == "crewai":
        from .adapters.crewai import make_tool as native_tool

        return native_tool(runtime)
    if framework == "claude_agent_sdk":
        from .adapters.claude_agent_sdk import make_tool as native_tool

        return native_tool(runtime)
    raise ValueError(f"Unsupported Reflect tool framework: {framework}")


def invoker_for(agent: Any, **kwargs: Any):
    modules = {cls.__module__.split(".")[0] for cls in type(agent).__mro__}
    if "agents" in modules:
        from .adapters.openai_agents import invoker

        return invoker(agent, **kwargs)
    if "google" in modules:
        from .adapters.google_adk import invoker

        return invoker(agent, **kwargs)
    if "crewai" in modules:
        from .adapters.crewai import invoker

        return invoker(agent, **kwargs)
    if "claude_agent_sdk" in modules:
        from .adapters.claude_agent_sdk import invoker

        return invoker(agent, **kwargs)
    if hasattr(agent, "ainvoke"):
        from .adapters.langchain import invoker

        return invoker(agent, **kwargs)
    if callable(agent) and not hasattr(agent, "ainvoke"):

        async def invoke(request):
            value = agent(request, **kwargs)
            if not inspect.isawaitable(value):
                raise TypeError("A generic Reflect agent must be async and accept an Invocation")
            return await value

        return invoke
    raise TypeError("No explicit Reflect invocation adapter exists for this agent")


def build_agent(runtime: Reflect, factory, *args: Any, **kwargs: Any) -> Any:
    """Add the supplied tool while preserving the host's native constructor."""
    module = factory.__module__
    if module.startswith(("langchain", "langgraph", "deepagents")):
        framework = "langchain"
        if "middleware" in inspect.signature(factory).parameters:
            from .adapters.langchain import middleware

            kwargs = {**kwargs, "middleware": [*kwargs.get("middleware", []), middleware(runtime)]}
    elif module.startswith("agents"):
        framework = "openai_agents"
    elif module.startswith("google.adk"):
        framework = "google_adk"
        from .adapters.google_adk import configure

        kwargs = configure(runtime, kwargs)
    elif module.startswith("crewai"):
        framework = "crewai"
    elif module.startswith("claude_agent_sdk"):
        from .adapters.claude_agent_sdk import configure

        return configure(runtime, factory(*args, **kwargs))
    else:
        raise ValueError("Unsupported agent constructor; use reflect.tool() and an explicit Invocation callback")
    supplied = runtime.tool(framework)
    if any(getattr(tool, "name", None) == TOOL_NAME for tool in kwargs.get("tools", [])):
        raise ValueError("reflect.agent supplies reflect_bash automatically; do not also pass it in tools")
    if module.startswith("langgraph") and "prompt" in inspect.signature(factory).parameters:
        from .adapters.langgraph import configure

        return configure(runtime, factory, args, kwargs, [*kwargs.get("tools", []), supplied])
    return factory(*args, **{**kwargs, "tools": [*kwargs.get("tools", []), supplied]})
