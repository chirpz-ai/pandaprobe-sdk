"""Native prebuilt LangGraph integration without a langchain dependency."""

from __future__ import annotations

import inspect

from .langchain import MEMORY_ID


def configure(runtime, factory, args, kwargs, tools):
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, convert_to_messages
    from langchain_core.runnables import RunnableLambda
    from langgraph.prebuilt import ToolNode

    if "awrap_tool_call" not in inspect.signature(ToolNode).parameters:
        raise ImportError('This Reflect integration needs a newer LangGraph; upgrade "pandaprobe[reflect,langgraph]"')
    original = kwargs.get("prompt")

    async def prompt(state, config):
        messages = state.get("llm_input_messages", state["messages"])
        if original is None:
            result = messages
        elif isinstance(original, str):
            result = [SystemMessage(content=original), *messages]
        elif isinstance(original, SystemMessage):
            result = [original, *messages]
        elif hasattr(original, "ainvoke"):
            result = await original.ainvoke(state, config=config)
        else:
            result = await RunnableLambda(original).ainvoke(state, config=config)
        if hasattr(result, "to_messages"):
            result = result.to_messages()
        if isinstance(result, str):
            result = [HumanMessage(content=result)]
        result = [item for item in convert_to_messages(result) if item.id != MEMORY_ID]
        execution = runtime._current.get()
        if execution is not None:
            result.insert(0, SystemMessage(content=execution.instructions(), id=MEMORY_ID))
            execution.observe_model_input(result)
        return result

    async def permissions(request, handler):
        execution = runtime._current.get()
        if execution is not None and execution.role == "reflection" and request.tool_call["name"] != "reflect_bash":
            return ToolMessage(
                content="Reflection checkpoints permit only Reflect filesystem administration.",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return await handler(request)

    node = ToolNode(tools, awrap_tool_call=permissions)
    return factory(*args, **{**kwargs, "tools": node, "prompt": RunnableLambda(prompt)})
