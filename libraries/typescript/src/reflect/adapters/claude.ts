import { type NativeAgent, optionalModule } from "../integrations.js";
import type { Invocation, Reflect as ReflectRuntime } from "../runtime.js";
import { TOOL_DESCRIPTION, TOOL_NAME } from "../shell.js";
const SERVER = "pandaprobe_reflect";
const QUALIFIED = `mcp__${SERVER}__${TOOL_NAME}`;
export async function claudeAgent(
  runtime: ReflectRuntime,
  query: any,
  options: Record<string, any>,
  defaults: Record<string, any> = {},
): Promise<NativeAgent> {
  const { tool, createSdkMcpServer } = await optionalModule("@anthropic-ai/claude-agent-sdk");
  const { z } = await optionalModule("zod");
  if (options.disallowedTools?.includes(QUALIFIED))
    throw new TypeError("Claude options explicitly disallow Reflect's supplied Bash tool");
  return {
    framework: "claude-agent-sdk",
    async invokeReflect(request: Invocation, overrides = {}) {
      // Bind dispatch directly as well as inheriting ALS; MCP tasks may have their own async resources.
      const supplied = tool(
        TOOL_NAME,
        TOOL_DESCRIPTION,
        { command: z.string() },
        async ({ command }: { command: string }) => ({
          content: [
            { type: "text", text: await request.execution.shell.execute(command, { signal: request.signal }) },
          ],
        }),
      );
      const server = createSdkMcpServer({ name: SERVER, tools: [supplied] });
      const configured: Record<string, any> = {
        ...options,
        ...defaults,
        ...overrides,
        mcpServers: { ...options.mcpServers, [SERVER]: server },
        allowedTools: [...new Set([...(options.allowedTools ?? []), QUALIFIED])],
      };
      const prompt = configured.systemPrompt;
      if (typeof prompt === "string" || prompt == null)
        configured.systemPrompt = `${prompt ?? ""}\n\n${request.instructions}`;
      else if (prompt.type === "preset")
        configured.systemPrompt = { ...prompt, append: `${prompt.append ?? ""}\n\n${request.instructions}` };
      else throw new TypeError("Reflect supports string or preset Claude system prompts");
      if (request.phase === "validation") {
        for (const key of ["resume", "continue", "sessionId", "forkSession"]) delete configured[key];
        configured.extraArgs = { ...configured.extraArgs };
        for (const key of ["resume", "continue", "session-id", "fork-session"]) delete configured.extraArgs[key];
      }
      if (request.phase === "reflection") {
        const previous = request.previousResult as any[];
        const result = [...(previous ?? [])].reverse().find((message) => message.type === "result");
        if (!result?.session_id) throw new TypeError("Claude reflection needs the completed task's session ID");
        Object.assign(configured, {
          resume: result.session_id,
          continue: false,
          maxTurns: Math.min(configured.maxTurns ?? 10, runtime.config.reflectionMaxCommands + 1),
          tools: [],
          allowedTools: [QUALIFIED],
          mcpServers: { [SERVER]: server },
          strictMcpConfig: true,
        });
        // Match Python's restricted tool/server surface. Do not add canUseTool:
        // allowedTools approves Reflect first, shadowing that redundant callback.
      }
      const controller = new AbortController();
      const abort = () => controller.abort(request.signal.reason);
      request.signal.addEventListener("abort", abort, { once: true });
      if (request.signal.aborted) abort();
      configured.abortController = controller;
      let stream: any;
      const messages: any[] = [];
      try {
        stream = query({
          prompt: request.phase === "reflection" ? request.feedback : request.input,
          options: configured,
        });
        for await (const message of stream) {
          request.signal.throwIfAborted();
          messages.push(message);
          if (message.type === "result" && (message.is_error || message.subtype !== "success"))
            throw new Error(
              `Claude Agent SDK failed (${message.subtype}): ${message.result ?? message.errors?.join("; ") ?? "No error details returned"}`,
            );
        }
        if (!messages.some((message) => message.type === "result"))
          throw new Error("Claude Agent SDK ended without a result");
        return messages;
      } finally {
        request.signal.removeEventListener("abort", abort);
        controller.abort();
        await stream?.return?.();
        await stream?.close?.();
        await server.instance?.close?.();
      }
    },
  };
}
