import { type NativeAgent, optionalModule } from "../integrations.js";
import type { Invocation, Reflect as ReflectRuntime } from "../runtime.js";
import { TOOL_NAME } from "../shell.js";
export async function openaiAgent(
  runtime: ReflectRuntime,
  agent: any,
  defaults: Record<string, any> = {},
): Promise<NativeAgent> {
  const { run } = await optionalModule("@openai/agents");
  return {
    framework: "openai-agents",
    native: agent,
    async invokeReflect(request: Invocation, overrides = {}) {
      const options = { ...defaults, ...overrides, signal: request.signal };
      if (
        request.phase === "validation" &&
        ["session", "conversationId", "previousResponseId", "autoPreviousResponseId"].some((key) =>
          Boolean((options as any)[key]),
        )
      )
        throw new TypeError(
          "Validation needs a fresh conversation; persistent sessions/response IDs are not accepted",
        );
      const original = agent.instructions;
      const instructions = async (context: any, current: any) =>
        `${typeof original === "function" ? await original(context, current) : (original ?? "")}\n\n${request.execution.instructions()}`;
      const configured = agent.clone({
        instructions,
        tools:
          request.phase === "reflection" ? agent.tools.filter((t: any) => t.name === TOOL_NAME) : [...agent.tools],
        handoffs: request.phase === "reflection" ? [] : agent.handoffs,
        ...(request.phase === "reflection" ? { mcpServers: [] } : {}),
      });
      let input = request.input;
      if (request.phase === "reflection") {
        (options as any).maxTurns = Math.min(
          (options as any).maxTurns ?? 10,
          runtime.config.reflectionMaxCommands + 1,
        );
        const feedback = { role: "user", content: request.feedback };
        if ((options as any).session) input = [feedback];
        else {
          const prior = request.previousResult as any;
          if (!Array.isArray(prior?.history))
            throw new TypeError("OpenAI reflection requires the completed result history or its session");
          input = [...prior.history, feedback];
          for (const key of ["previousResponseId", "conversationId", "autoPreviousResponseId"])
            delete (options as any)[key];
        }
      }
      const result = await run(configured, input, options);
      if ((options as any).stream) {
        for await (const event of result) {
          if (event?.type === "error") throw event.error;
        }
        await result.completed;
        if (result.error) throw result.error;
      }
      return result;
    },
  };
}
