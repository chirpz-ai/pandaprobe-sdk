"""CrewAI native agents and crews, with isolated invocation copies."""

from __future__ import annotations

import asyncio
from typing import Any

from ..shell import TOOL_DESCRIPTION, TOOL_NAME


def _never_cache(*_):
    return False


def make_tool(runtime):
    from crewai.tools import BaseTool
    from pydantic import BaseModel

    class Arguments(BaseModel):
        command: str

    class BashTool(BaseTool):
        name: str = TOOL_NAME
        description: str = TOOL_DESCRIPTION
        args_schema: type[BaseModel] = Arguments

        async def _arun(self, command: str) -> str:
            execution = runtime._current.get()
            if execution is not None and execution.loop is not asyncio.get_running_loop():
                future = asyncio.run_coroutine_threadsafe(runtime.bash(command), execution.loop)
                return await asyncio.wrap_future(future)
            return await runtime.bash(command)

        def to_structured_tool(self):
            structured = super().to_structured_tool()
            # Older CrewAI's async sync-function executor loses ContextVars.
            # Supply its native async function path instead of a global router.
            structured.func = self._arun
            return structured

        def _run(self, command: str) -> str:
            execution = runtime._current.get()
            if execution is None or execution.loop is None:
                return "Reflect Bash requires an active Reflect invocation"
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is execution.loop:
                raise RuntimeError("Use CrewAI's asynchronous tool execution with Reflect")
            future = asyncio.run_coroutine_threadsafe(runtime.bash(command), execution.loop)
            return future.result(timeout=runtime.config.shell_timeout + 3)

    return BashTool(cache_function=_never_cache)


def observed_llm(llm, execution):
    """Explicit instance delegation; no global provider wrapping or hooks."""
    from crewai.llms.base_llm import BaseLLM
    from pydantic import PrivateAttr

    class ObservedLLM(BaseLLM):
        _delegate: Any = PrivateAttr()

        def call(self, messages, *args, **kwargs):
            execution.observe_model_input(messages)
            self._delegate.stop = self.stop
            return self._delegate.call(messages, *args, **kwargs)

        async def acall(self, messages, *args, **kwargs):
            execution.observe_model_input(messages)
            self._delegate.stop = self.stop
            return await self._delegate.acall(messages, *args, **kwargs)

        def supports_function_calling(self):
            return self._delegate.supports_function_calling()

        def supports_stop_words(self):
            return self._delegate.supports_stop_words()

        def get_context_window_size(self):
            return self._delegate.get_context_window_size()

        def get_token_usage_summary(self):
            return self._delegate.get_token_usage_summary()

    wrapper = ObservedLLM(model=llm.model, temperature=getattr(llm, "temperature", None))
    wrapper._delegate = llm
    wrapper.stop = getattr(llm, "stop", [])
    return wrapper


def invoker(agent: Any, **kwargs: Any):
    from crewai import Crew

    async def invoke(request):
        runtime = request.execution.runtime
        configured = agent.copy()
        is_crew = isinstance(configured, Crew)
        members = configured.agents if is_crew else [configured]
        for member in members:
            tools = [tool for tool in member.tools if tool.name != TOOL_NAME]
            member.tools = ([*tools] if request.phase != "reflection" else []) + [make_tool(runtime)]
            member.llm = observed_llm(member.llm, request.execution)
            if member.function_calling_llm is not None:
                member.function_calling_llm = observed_llm(member.function_calling_llm, request.execution)
            if request.phase == "reflection":
                member.max_iter = min(member.max_iter, runtime.config.reflection_max_commands + 1)
                member.allow_delegation = False
                member.allow_code_execution = False
        if request.phase == "reflection":
            if is_crew:
                if not configured.tasks or configured.tasks[-1].agent is None:
                    raise ValueError("Crew reflection needs an explicit agent on the final task")
                target = configured.tasks[-1].agent
                # The final task's agent reflects; the whole crew is not rerun.
                history = [{"role": "assistant", "content": str(request.previous_result)}]
            else:
                target = configured
                history = list(getattr(request.previous_result, "messages", []))
            messages = [
                {"role": "system", "content": request.instructions},
                *history,
                {"role": "user", "content": request.feedback},
            ]
            return await target.kickoff_async(messages)
        if is_crew:
            values = dict(request.input or {})
            key = "pandaprobe_reflect_active_memory"
            if key in values:
                raise ValueError(f"{key} is reserved by Reflect")
            values[key] = request.instructions
            for task in configured.tasks:
                task.description += "\n\n{" + key + "}"
                if task.tools:
                    task.tools = [t for t in task.tools if t.name != TOOL_NAME] + [make_tool(runtime)]
            return await configured.kickoff_async(inputs=values, **kwargs)
        messages = [{"role": "system", "content": request.instructions}]
        messages.extend(
            [{"role": "user", "content": request.input}] if isinstance(request.input, str) else request.input
        )
        return await configured.kickoff_async(messages, **kwargs)

    return invoke
