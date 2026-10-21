"""OpenAI Agents: same host agent/model, explicit continuation and input receipts."""

from __future__ import annotations

import inspect
from typing import Any


def invoker(agent: Any, **kwargs: Any):
    from agents import RunHooks, Runner

    async def invoke(request):
        runtime = request.execution.runtime
        original = agent.instructions

        async def instructions(context, current_agent):
            prefix = original(context, current_agent) if callable(original) else original
            if inspect.isawaitable(prefix):
                prefix = await prefix
            return (prefix or "") + "\n\n" + request.execution.instructions()

        original_hooks = kwargs.get("hooks")

        class DeliveryHooks(RunHooks):
            def __getattribute__(self, name):
                if name.startswith("on_") and name != "on_llm_start" and original_hooks is not None:
                    return getattr(original_hooks, name)
                return super().__getattribute__(name)

            async def on_llm_start(self, context, current_agent, system_prompt, input_items):
                if original_hooks is not None:
                    await original_hooks.on_llm_start(context, current_agent, system_prompt, input_items)
                request.execution.observe_model_input([system_prompt, input_items])

        tools = list(agent.tools)
        if not any(getattr(tool, "name", None) == "reflect_bash" for tool in tools):
            tools.append(runtime.tool("openai_agents"))
        if request.phase == "reflection":
            tools = [tool for tool in tools if getattr(tool, "name", None) == "reflect_bash"]
        configured = agent.clone(
            instructions=instructions, tools=tools, handoffs=[] if request.phase == "reflection" else agent.handoffs
        )
        options = {**kwargs, "hooks": DeliveryHooks()}
        if request.phase == "validation" and any(
            options.get(key) for key in ("session", "conversation_id", "previous_response_id")
        ):
            raise ValueError(
                "Validation needs a fresh conversation; persistent sessions/response IDs are not accepted"
            )
        if request.phase == "reflection":
            options["max_turns"] = min(options.get("max_turns") or 10, runtime.config.reflection_max_commands + 1)
            feedback = {"role": "user", "content": request.feedback}
            if options.get("session") is not None:
                value = [feedback]
            else:
                value = [*request.previous_result.to_input_list(), feedback]
                options.pop("previous_response_id", None)
                options.pop("conversation_id", None)
                options.pop("auto_previous_response_id", None)
        else:
            value = request.input
        return await Runner.run(configured, value, **options)

    return invoke
