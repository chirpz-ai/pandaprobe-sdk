"""Claude Agent SDK MCP tool and per-invocation native client lifecycle."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from ..shell import TOOL_DESCRIPTION, TOOL_NAME

SERVER = "pandaprobe_reflect"
QUALIFIED_TOOL = f"mcp__{SERVER}__{TOOL_NAME}"


def make_tool(runtime):
    from claude_agent_sdk import tool

    @tool(TOOL_NAME, TOOL_DESCRIPTION, {"command": str})
    async def reflect_bash(arguments):
        output = await runtime.bash(arguments["command"])
        return {"content": [{"type": "text", "text": output}]}

    return reflect_bash


def configure(runtime, options):
    from claude_agent_sdk import create_sdk_mcp_server

    if not isinstance(options.mcp_servers, dict):
        raise TypeError("Reflect requires a dictionary of Claude MCP servers so it can add its supplied tool")
    if QUALIFIED_TOOL in options.disallowed_tools:
        raise ValueError("Claude options explicitly disallow Reflect's supplied Bash tool")
    server = create_sdk_mcp_server(name=SERVER, tools=[make_tool(runtime)])
    return replace(
        options,
        mcp_servers={**options.mcp_servers, SERVER: server},
        allowed_tools=list(dict.fromkeys([*options.allowed_tools, QUALIFIED_TOOL])),
    )


def invoker(options: Any, **kwargs: Any):
    from claude_agent_sdk import ClaudeSDKClient, ResultMessage

    if kwargs:
        raise ValueError("Configure Claude invocation settings in ClaudeAgentOptions")

    async def invoke(request):
        runtime = request.execution.runtime
        configured = configure(runtime, options)
        prompt = configured.system_prompt
        memory = request.instructions
        if isinstance(prompt, str) or prompt is None:
            prompt = (prompt or "") + "\n\n" + memory
        elif isinstance(prompt, dict) and prompt.get("type") == "preset":
            prompt = {**prompt, "append": prompt.get("append", "") + "\n\n" + memory}
        else:
            raise ValueError("Reflect supports string or preset Claude system prompts")
        changes: dict[str, Any] = {"system_prompt": prompt}
        option_fields = {field.name for field in fields(configured)}
        if request.phase == "validation":
            changes.update(resume=None, continue_conversation=False)
            if "session_id" in option_fields:
                changes["session_id"] = None
            changes["extra_args"] = {
                key: value
                for key, value in configured.extra_args.items()
                if key not in ("resume", "continue", "session-id", "fork-session")
            }
        if request.phase == "reflection":
            session = next(
                (
                    message.session_id
                    for message in reversed(request.previous_result or [])
                    if isinstance(message, ResultMessage)
                ),
                None,
            )
            if session is None:
                raise ValueError("Claude reflection needs the completed task's session ID")
            changes.update(
                resume=session,
                continue_conversation=False,
                max_turns=min(configured.max_turns or 10, runtime.config.reflection_max_commands + 1),
                tools=[],
                allowed_tools=[QUALIFIED_TOOL],
                mcp_servers={SERVER: configured.mcp_servers[SERVER]},
            )
            if "strict_mcp_config" in option_fields:
                changes["strict_mcp_config"] = True
            else:
                # Older supported SDKs expose this CLI setting through extra_args.
                changes["extra_args"] = {**configured.extra_args, "strict-mcp-config": None}
        configured = replace(configured, **changes)
        # Creating the client inside this scope propagates the execution context
        # into its MCP dispatch tasks; no last-active-client router is used.
        async with ClaudeSDKClient(options=configured) as client:
            await client.query(request.feedback if request.phase == "reflection" else request.input)
            # These instructions were sent to the SDK. Actual tool-body delivery
            # is confirmed through traced model input or an explicit host receipt.
            request.execution.submitted(memory)
            return [message async for message in client.receive_response()]

    return invoke
