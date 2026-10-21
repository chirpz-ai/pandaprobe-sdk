"""ADK agent configuration and materialized event-stream invocation."""

from __future__ import annotations

import inspect
from typing import Any


def configure(runtime, kwargs: dict[str, Any]) -> dict[str, Any]:
    original = kwargs.get("instruction", "")

    async def instruction(context):
        prefix = original(context) if callable(original) else original
        if inspect.isawaitable(prefix):
            prefix = await prefix
        execution = runtime._current.get()
        return (prefix or "") + ("\n\n" + execution.instructions() if execution else "")

    async def receipt(callback_context, llm_request):
        execution = runtime._current.get()
        if execution is not None:
            if execution.role == "reflection":
                from google.genai import types

                declarations = [
                    declaration
                    for tool in llm_request.config.tools or []
                    for declaration in tool.function_declarations or []
                    if declaration.name == "reflect_bash"
                ]
                llm_request.config.tools = [types.Tool(function_declarations=declarations)]
            execution.observe_model_input(llm_request)

    async def permissions(tool, args, tool_context):
        execution = runtime._current.get()
        if execution is not None and execution.role == "reflection" and tool.name != "reflect_bash":
            return {"error": "Reflection checkpoints permit only Reflect filesystem administration"}

    callbacks = kwargs.get("before_model_callback")
    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, list):
        callbacks = [callbacks]
    before_tool = kwargs.get("before_tool_callback") or []
    if not isinstance(before_tool, list):
        before_tool = [before_tool]
    return {
        **kwargs,
        "instruction": instruction,
        "before_model_callback": [*callbacks, receipt],
        "before_tool_callback": [permissions, *before_tool],
    }


def invoker(agent: Any, **kwargs: Any):
    from google.adk.agents.run_config import RunConfig
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    supplied_runner = agent if isinstance(agent, Runner) else None
    sessions = InMemorySessionService() if supplied_runner is None else supplied_runner.session_service

    async def invoke(request):
        runtime = request.execution.runtime
        if supplied_runner is None:
            fields = {"instruction": agent.instruction, "before_model_callback": agent.before_model_callback}
            fields["before_tool_callback"] = agent.before_tool_callback
            tools = list(agent.tools)
            attached = any(getattr(tool, "name", None) == "reflect_bash" for tool in tools)
            if not attached:
                tools.append(runtime.tool("google_adk"))
            configured = agent if attached else agent.model_copy(update={**configure(runtime, fields), "tools": tools})
            runner = Runner(agent=configured, app_name=agent.name, session_service=sessions)
        else:
            runner = supplied_runner
            if not any(getattr(tool, "name", None) == "reflect_bash" for tool in getattr(runner.agent, "tools", [])):
                raise ValueError(
                    "Construct the ADK runner's agent with reflect.agent(LlmAgent, ...) to supply Bash and memory"
                )
        user_id = kwargs.get("user_id", "reflect")
        session_id = request.execution.id if request.phase == "validation" else request.execution.session_id
        session = await sessions.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
        if session is None:
            await sessions.create_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
        config = kwargs.get("run_config") or RunConfig()
        if request.phase == "reflection":
            config = config.model_copy(
                update={"max_llm_calls": min(config.max_llm_calls, runtime.config.reflection_max_commands + 1)}
            )
        value = request.feedback if request.phase == "reflection" else request.input
        if isinstance(value, str):
            value = types.Content(role="user", parts=[types.Part(text=value)])
        options = {key: value for key, value in kwargs.items() if key not in ("user_id", "run_config")}
        return [
            event
            async for event in runner.run_async(
                user_id=user_id, session_id=session_id, new_message=value, run_config=config, **options
            )
        ]

    return invoke
