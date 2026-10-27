import { ADMIN_ONLY, type NativeAgent, optionalModule } from "../integrations.js";
import type { Invocation, Reflect as ReflectRuntime } from "../runtime.js";
import { TOOL_NAME } from "../shell.js";
function appendInstructions(original: any, memory: string): any {
  if (original == null || typeof original === "string") return `${original ?? ""}\n\n${memory}`;
  const messages = Array.isArray(original) ? original : [original];
  return [...messages, { role: "system", content: memory }];
}
export async function vercelAgent(
  runtime: ReflectRuntime,
  Factory: any,
  settings: Record<string, any>,
  supplied: any,
  defaults: Record<string, any> = {},
): Promise<NativeAgent> {
  const { stepCountIs } = await optionalModule("ai");
  if (settings.tools?.[TOOL_NAME]) throw new TypeError("reflect.agent supplies reflect_bash automatically");
  return {
    framework: "vercel-ai",
    async invokeReflect(request: Invocation, overrides = {}) {
      const reflection = request.phase === "reflection";
      const options = { ...defaults, ...overrides };
      const guard = (tool: any) => ({
        ...tool,
        ...(tool.execute
          ? {
              execute: (...args: any[]) => {
                if (reflection) throw new Error(ADMIN_ONLY);
                return tool.execute(...args);
              },
            }
          : {}),
      });
      const tools = reflection
        ? { [TOOL_NAME]: supplied }
        : {
            ...Object.fromEntries(Object.entries(settings.tools ?? {}).map(([name, tool]) => [name, guard(tool)])),
            [TOOL_NAME]: supplied,
          };
      let effectiveInstructions = settings.instructions;
      let prepareStep = settings.prepareStep;
      const configured: Record<string, any> = {
        ...settings,
        tools,
        instructions: appendInstructions(settings.instructions, request.instructions),
        prepareCall: async (value: any) => {
          const prepared = (await settings.prepareCall?.(value)) ?? value;
          effectiveInstructions =
            prepared.instructions === configured.instructions
              ? settings.instructions
              : (prepared.instructions ?? settings.instructions);
          prepareStep = prepared.prepareStep === configured.prepareStep ? settings.prepareStep : prepared.prepareStep;
          return {
            ...prepared,
            tools: reflection ? { [TOOL_NAME]: supplied } : { ...(prepared.tools ?? tools), [TOOL_NAME]: supplied },
            instructions: appendInstructions(effectiveInstructions, request.execution.instructions()),
            prepareStep: configured.prepareStep,
            ...(reflection
              ? {
                  activeTools: [TOOL_NAME],
                  toolChoice: "auto",
                  stopWhen: stepCountIs(runtime.config.reflectionMaxCommands + 1),
                }
              : prepared.activeTools
                ? { activeTools: [...new Set([...prepared.activeTools, TOOL_NAME])] }
                : {}),
          };
        },
        prepareStep: async (step: any) => {
          const prepared = (await prepareStep?.(step)) ?? {};
          return {
            ...prepared,
            system: appendInstructions(prepared.system ?? effectiveInstructions, request.execution.instructions()),
            ...(reflection
              ? { activeTools: [TOOL_NAME], toolChoice: "auto" }
              : prepared.activeTools
                ? { activeTools: [...new Set([...prepared.activeTools, TOOL_NAME])] }
                : {}),
          };
        },
      };
      const agent = new Factory(configured);
      let input: any = typeof request.input === "string" ? { prompt: request.input } : request.input;
      if (reflection) {
        const previous = request.previousResult as any;
        const response = await previous?.response;
        if (!response?.messages) throw new TypeError("Vercel reflection requires a completed native result");
        const prior =
          typeof request.input === "string"
            ? [{ role: "user", content: request.input }]
            : ((request.input as any)?.messages ?? []);
        input = { messages: [...prior, ...response.messages, { role: "user", content: request.feedback }] };
      }
      const stream = options.stream;
      options.stream = undefined;
      if (stream) {
        const result = await agent.stream({ ...input, ...options, abortSignal: request.signal });
        for await (const event of result.fullStream) {
          if (event.type === "error") throw event.error;
          if (event.type === "abort") throw request.signal.reason ?? new Error("Vercel stream aborted");
        }
        await result.response;
        return result;
      }
      return agent.generate({ ...input, ...options, abortSignal: request.signal });
    },
  };
}
