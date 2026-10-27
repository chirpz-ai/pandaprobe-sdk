import type { Invocation, Reflect as ReflectRuntime } from "./runtime.js";
import { TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS } from "./shell.js";
export type Framework =
  | "generic"
  | "langchain"
  | "langgraph"
  | "deepagents"
  | "openai-agents"
  | "claude-agent-sdk"
  | "vercel-ai";
export type Agent<I = any, R = any> = ((request: Invocation<I>) => Promise<R>) | NativeAgent<I, R>;
export interface AgentOptions {
  framework: Exclude<Framework, "generic">;
  options?: Record<string, any>;
  invokeOptions?: Record<string, any>;
}
export interface NativeAgent<I = any, R = any> {
  invokeReflect(request: Invocation<I>, options?: Record<string, any>): Promise<R>;
  readonly native?: any;
  readonly framework: Framework;
}
export const MEMORY_ID = "pandaprobe-reflect-active-memory";
export const ADMIN_ONLY = "Reflection checkpoints permit only Reflect filesystem administration.";
export async function optionalModule(name: string): Promise<any> {
  try {
    return await import(name);
  } catch (error) {
    throw new Error(
      `Reflect requires the optional ${name} package; install the documented framework dependencies. ${String(error)}`,
    );
  }
}
export function resultText(input: any): string {
  let value = input;
  if (value?.messages) value = value.messages;
  if (Array.isArray(value)) value = value[value.length - 1];
  for (const name of ["finalOutput", "raw", "result", "content", "text"])
    if (value?.[name] !== undefined) {
      value = value[name];
      break;
    }
  return (typeof value === "string" ? value : (JSON.stringify(value) ?? String(value))).slice(0, 4096);
}
export function invokerFor<I = any, R = any>(
  agent: Agent<I, R>,
  options: Record<string, any> = {},
): (request: Invocation<I>) => Promise<R> {
  if (agent && typeof (agent as NativeAgent).invokeReflect === "function")
    return (request) => (agent as NativeAgent<I, R>).invokeReflect(request, options);
  if (typeof agent === "function")
    return async (request) => {
      const result = agent(request);
      if (!result || typeof result.then !== "function")
        throw new TypeError("A generic Reflect agent must be async and accept an Invocation");
      return result;
    };
  throw new TypeError("Use reflect.agent(factory, { framework, options }) or an async Invocation callback");
}
export async function makeTool(runtime: ReflectRuntime, framework: Framework): Promise<any> {
  const execute = async ({ command }: { command: string }) => {
    if (typeof command !== "string") throw new TypeError("command must be a string");
    return runtime.bash(command);
  };
  if (framework === "generic") return (command: string) => runtime.bash(command);
  if (["langchain", "langgraph", "deepagents"].includes(framework)) {
    const { tool } = await optionalModule("@langchain/core/tools");
    return tool(execute, { name: TOOL_NAME, description: TOOL_DESCRIPTION, schema: TOOL_PARAMETERS });
  }
  if (framework === "openai-agents") {
    const { tool } = await optionalModule("@openai/agents");
    return tool({
      name: TOOL_NAME,
      description: TOOL_DESCRIPTION,
      parameters: TOOL_PARAMETERS,
      strict: true,
      execute,
    });
  }
  if (framework === "vercel-ai") {
    const { tool, jsonSchema } = await optionalModule("ai");
    return tool({ description: TOOL_DESCRIPTION, inputSchema: jsonSchema(TOOL_PARAMETERS), execute });
  }
  if (framework === "claude-agent-sdk") {
    const { tool } = await optionalModule("@anthropic-ai/claude-agent-sdk");
    const { z } = await optionalModule("zod");
    return tool(TOOL_NAME, TOOL_DESCRIPTION, { command: z.string() }, async (args: { command: string }) => ({
      content: [{ type: "text", text: await execute(args) }],
    }));
  }
  throw new TypeError(`Unsupported Reflect tool framework: ${framework}`);
}
export async function buildAgent(runtime: ReflectRuntime, factory: any, settings: AgentOptions): Promise<NativeAgent> {
  if (typeof factory !== "function") throw new TypeError("Pass the native agent factory or constructor");
  const options = { ...settings.options };
  const framework = settings.framework;
  if (!["langchain", "langgraph", "deepagents", "openai-agents", "claude-agent-sdk", "vercel-ai"].includes(framework))
    throw new TypeError(`Unsupported Reflect framework: ${framework}`);
  if (framework === "claude-agent-sdk") {
    const { claudeAgent } = await import("./adapters/claude.js");
    return claudeAgent(runtime, factory, options, settings.invokeOptions);
  }
  const supplied = await makeTool(runtime, framework);
  if (framework === "vercel-ai") {
    const { vercelAgent } = await import("./adapters/vercel.js");
    return vercelAgent(runtime, factory, options, supplied, settings.invokeOptions);
  }
  const tools = [...(options.tools ?? [])];
  if (tools.some((tool) => tool.name === TOOL_NAME))
    throw new TypeError("reflect.agent supplies reflect_bash automatically; do not also pass it in tools");
  tools.push(supplied);
  if (framework === "openai-agents") {
    const { openaiAgent } = await import("./adapters/openai.js");
    return openaiAgent(runtime, new factory({ ...options, tools }), settings.invokeOptions);
  }
  const { langchainAgent } = await import("./adapters/langchain.js");
  return langchainAgent(runtime, factory, { ...options, tools }, framework, settings.invokeOptions);
}
