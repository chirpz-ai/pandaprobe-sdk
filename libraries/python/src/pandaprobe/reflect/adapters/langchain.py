"""Message-state LangChain, LangGraph and Deep Agents invocation boundaries."""

from __future__ import annotations

from typing import Any

MEMORY_ID = "pandaprobe-reflect-active-memory"


def middleware(runtime):
    """Observe explicit model input, including Python 3.10 async loops."""
    from langchain.agents.middleware import AgentMiddleware

    class ReflectMiddleware(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            execution = runtime._current.get()
            if execution is not None:
                if execution.role == "reflection":
                    request = request.override(
                        tools=[tool for tool in request.tools if getattr(tool, "name", None) == "reflect_bash"]
                    )
                execution.observe_model_input([request.system_message, request.messages])
            return await handler(request)

        async def awrap_tool_call(self, request, handler):
            execution = runtime._current.get()
            if (
                execution is not None
                and execution.role == "reflection"
                and request.tool_call["name"] != "reflect_bash"
            ):
                from langchain_core.messages import ToolMessage

                return ToolMessage(
                    content="Reflection checkpoints permit only Reflect filesystem administration.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            return await handler(request)

    return ReflectMiddleware()


def invoker(agent: Any, **kwargs: Any):
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages

    class DeliveryCallback(AsyncCallbackHandler):
        def __init__(self, execution):
            self.execution = execution

        async def on_chat_model_start(self, serialized, messages, **options):
            self.execution.observe_model_input(messages)

        async def on_llm_start(self, serialized, prompts, **options):
            self.execution.observe_model_input(prompts)

    async def invoke(request):
        config = dict(kwargs.get("config") or {})
        configurable = dict(config.get("configurable") or {})
        if request.phase == "validation":
            configurable["thread_id"] = request.execution.id
        else:
            configurable.setdefault("thread_id", request.execution.session_id)
        config["configurable"] = configurable
        existing_callbacks = config.get("callbacks")
        callback = DeliveryCallback(request.execution)
        if existing_callbacks is not None and not isinstance(existing_callbacks, list):
            callbacks = existing_callbacks.copy()
            callbacks.add_handler(callback)
        else:
            callbacks = [*(existing_callbacks or []), callback]
        config["callbacks"] = callbacks
        if request.phase == "reflection":
            config["recursion_limit"] = min(
                config.get("recursion_limit", 25), request.execution.runtime.config.reflection_max_commands * 2 + 2
            )
            value = dict(request.previous_result) if isinstance(request.previous_result, dict) else {"messages": []}
            messages = convert_to_messages(value.get("messages", []))
            messages.append(HumanMessage(content=request.feedback))
        else:
            value = (
                {"messages": [HumanMessage(content=request.input)]}
                if isinstance(request.input, str)
                else dict(request.input)
            )
            if "messages" not in value:
                raise ValueError(
                    "Native Reflect LangChain integration requires message state; use Invocation for custom chains"
                )
            messages = convert_to_messages(value["messages"])
        messages = [message for message in messages if message.id != MEMORY_ID]
        value["messages"] = [SystemMessage(content=request.instructions, id=MEMORY_ID), *messages]
        options = {key: value for key, value in kwargs.items() if key != "config"}
        return await agent.ainvoke(value, config=config, **options)

    return invoke
