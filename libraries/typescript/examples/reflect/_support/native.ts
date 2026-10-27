import { AsyncLocalStorage } from "node:async_hooks";
import { mkdirSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { wrapClaudeAgentQuery } from "../../../src/integrations/claude-agent-sdk/index.js";
import { DeepAgentsCallbackHandler } from "../../../src/integrations/deepagents/index.js";
import { LangChainCallbackHandler } from "../../../src/integrations/langchain/index.js";
import { LangGraphCallbackHandler } from "../../../src/integrations/langgraph/index.js";
import type { TraceProcessor } from "../../../src/integrations/openai-agents/adapter.js";
import { OpenAIAgentsAdapter } from "../../../src/integrations/openai-agents/index.js";
import { pandaProbeMiddleware, wrapVercelAgent } from "../../../src/integrations/vercel-ai/index.js";
import {
  type Framework,
  Invocation,
  type NativeAgent,
  type Reflect as ReflectRuntime,
} from "../../../src/reflect/index.js";
import { optionalModule } from "../../../src/reflect/integrations.js";
import type { TaskInputAudit } from "./delivery.js";
import { nativeTelemetry } from "./telemetry.js";

// The Agents SDK registry is shared by all factories. Register one dispatcher,
// then route each trace to the invoking factory's client/tags (including fresh
// runtimes and simultaneous examples). Leave host-owned processors untouched.
const openaiProcessors = new WeakSet<object>();
const openaiInvocation = new AsyncLocalStorage<TraceProcessor>();
function registerOpenAIProcessor(sdk: any): void {
  if (openaiProcessors.has(sdk)) return;
  const owners = new Map<string, TraceProcessor>();
  const id = (event: any) => String(event.traceId ?? event.trace_id ?? "");
  sdk.addTraceProcessor({
    onTraceStart(trace: any) {
      const processor = openaiInvocation.getStore();
      if (!processor) return;
      owners.set(id(trace), processor);
      processor.onTraceStart(trace);
    },
    onSpanStart(span: any) {
      owners.get(id(span))?.onSpanStart(span);
    },
    onSpanEnd(span: any) {
      owners.get(id(span))?.onSpanEnd(span);
    },
    onTraceEnd(trace: any) {
      const processor = owners.get(id(trace));
      owners.delete(id(trace));
      processor?.onTraceEnd(trace);
    },
    forceFlush() {},
    shutdown() {
      owners.clear();
      openaiProcessors.delete(sdk);
    },
  });
  openaiProcessors.add(sdk);
}
export interface BusinessTool {
  name: string;
  description: string;
  fields: Record<string, "string" | "number" | "integer">;
  execute(args: any): string | Promise<string>;
}
export interface AgentFactory {
  (
    tools: BusinessTool[],
    instructions: string,
    options?: { validation?: boolean; inputAudit?: TaskInputAudit },
  ): Promise<NativeAgent>;
  forRuntime(runtime: ReflectRuntime): Promise<AgentFactory>;
  framework: Exclude<Framework, "generic">;
  modelName: string;
}
export interface NativeOptions {
  model: any;
  /** ChatOpenAI request deadline in milliseconds; Python's default is 60 seconds. */
  modelTimeoutMs?: number;
  query?: any;
  modelProvider?: string;
  stream?: boolean;
  workdir?: string;
  tag?: string;
  onResult?: (record: Record<string, unknown>) => void;
}
function conversation(agent: NativeAgent): NativeAgent {
  const saved = new Map<string, any>();
  return {
    ...agent,
    async invokeReflect(request) {
      const prior = saved.get(request.execution.sessionId);
      let active = request;
      const options: any = {};
      if (request.phase === "task" && prior) {
        if (agent.framework === "claude-agent-sdk") options.resume = prior;
        else
          active = new Invocation(
            agent.framework === "openai-agents"
              ? [...prior, { role: "user", content: request.input }]
              : { messages: [...prior, { role: "user", content: request.input }] },
            request.execution,
            request.phase,
            request.previousResult,
            request.feedback,
            request.signal,
          );
      }
      const result = await agent.invokeReflect(active, options);
      if (request.phase !== "validation") {
        if (agent.framework === "openai-agents") saved.set(request.execution.sessionId, result.history);
        else if (agent.framework === "claude-agent-sdk")
          saved.set(
            request.execution.sessionId,
            [...result].reverse().find((m: any) => m.type === "result")?.session_id,
          );
        else {
          const input =
            typeof active.input === "string"
              ? [{ role: "user", content: active.input }]
              : (active.input as any).messages;
          saved.set(request.execution.sessionId, [...input, ...(await result.response).messages]);
        }
      }
      return result;
    },
  };
}
export async function nativeFactory(
  runtime: ReflectRuntime,
  framework: Exclude<Framework, "generic">,
  options: NativeOptions,
): Promise<AgentFactory> {
  let openaiProcessor: TraceProcessor | undefined;
  let model = options.model;
  if (["langchain", "langgraph", "deepagents"].includes(framework) && typeof model === "string") {
    const { ChatOpenAI } = await optionalModule("@langchain/openai");
    model = new ChatOpenAI({
      model,
      reasoning: { effort: "none" },
      maxRetries: 0,
      timeout: options.modelTimeoutMs ?? 60_000,
    });
  }
  if (framework === "vercel-ai" && typeof model === "string") {
    const { createOpenAI } = await optionalModule("@ai-sdk/openai");
    model = createOpenAI()(model);
  }
  if (framework === "openai-agents") {
    const sdk = await optionalModule("@openai/agents");
    registerOpenAIProcessor(sdk);
    openaiProcessor = new OpenAIAgentsAdapter({
      client: runtime.client,
      tags: [options.tag ?? "reflect-support", "example"],
    }).createTraceProcessor();
  }
  const create = async (
    definitions: BusinessTool[],
    instructions: string,
    setup: { validation?: boolean; inputAudit?: TaskInputAudit } = {},
  ): Promise<NativeAgent> => {
    const schema = (t: BusinessTool) => ({
      type: "object",
      properties: Object.fromEntries(Object.entries(t.fields).map(([k, type]) => [k, { type }])),
      required: Object.keys(t.fields),
      additionalProperties: false,
    });
    if (["langchain", "langgraph", "deepagents"].includes(framework)) {
      const { tool } = await optionalModule("@langchain/core/tools");
      const { MemorySaver } = await optionalModule("@langchain/langgraph");
      const factory =
        framework === "langgraph"
          ? (await optionalModule("@langchain/langgraph/prebuilt")).createReactAgent
          : framework === "deepagents"
            ? (await optionalModule("deepagents")).createDeepAgent
            : (await optionalModule("langchain")).createAgent;
      const middleware = [];
      if (setup.inputAudit && framework !== "langgraph") {
        const { createMiddleware } = await optionalModule("langchain");
        middleware.push(
          createMiddleware({
            name: "ReflectExampleInputAudit",
            wrapModelCall: async (request: any, handler: any) => {
              setup.inputAudit!.record([request.systemMessage, request.messages]);
              return handler(request);
            },
          }),
        );
      }
      const Handler =
        framework === "langgraph"
          ? LangGraphCallbackHandler
          : framework === "deepagents"
            ? DeepAgentsCallbackHandler
            : LangChainCallbackHandler;
      return runtime.agent(factory, {
        framework: framework as "langchain" | "langgraph" | "deepagents",
        options: {
          ...(framework === "langgraph"
            ? { llm: model, prompt: instructions }
            : {
                model,
                systemPrompt:
                  instructions +
                  (framework === "deepagents"
                    ? "\nUse reflect_bash for /rules and /context.md; other files are task scratch space."
                    : ""),
              }),
          tools: definitions.map((t) =>
            tool(t.execute, { name: t.name, description: t.description, schema: schema(t) }),
          ),
          checkpointer: new MemorySaver(),
          ...(framework === "deepagents" ? { backend: new (await optionalModule("deepagents")).StateBackend() } : {}),
          ...(framework === "langgraph" ? {} : { middleware }),
        },
        invokeOptions: {
          callbacks: [
            new Handler({ client: runtime.client }),
            ...(setup.inputAudit ? [setup.inputAudit.callback()] : []),
          ],
          recursionLimit: framework === "deepagents" ? 40 : 32,
          stream: options.stream ?? false,
        },
      });
    }
    if (framework === "openai-agents") {
      const { Agent, tool, MemorySession } = await optionalModule("@openai/agents");
      return runtime.agent(Agent, {
        framework,
        options: {
          name: "Northstar Support",
          model,
          instructions,
          tools: definitions.map((t) =>
            tool({
              name: t.name,
              description: t.description,
              parameters: schema(t),
              strict: true,
              execute: t.execute,
            }),
          ),
        },
        invokeOptions: {
          maxTurns: 16,
          ...(setup.validation ? {} : { session: new MemorySession() }),
          stream: options.stream ?? false,
        },
      });
    }
    if (framework === "vercel-ai") {
      const ai = await optionalModule("ai");
      const wrapped = ai.wrapLanguageModel({ model, middleware: pandaProbeMiddleware() });
      const agent = await runtime.agent(wrapVercelAgent(ai.ToolLoopAgent, { client: runtime.client }), {
        framework,
        options: {
          model: wrapped,
          instructions,
          stopWhen: ai.stepCountIs(12),
          tools: Object.fromEntries(
            definitions.map((t) => [
              t.name,
              ai.tool({ description: t.description, inputSchema: ai.jsonSchema(schema(t)), execute: t.execute }),
            ]),
          ),
        },
        invokeOptions: { stream: options.stream ?? false },
      });
      return conversation(agent);
    }
    const sdk = await optionalModule("@anthropic-ai/claude-agent-sdk");
    const { z } = await optionalModule("zod");
    // Create a business server per task/arm and close it after the native query.
    const query = options.query ?? sdk.query;
    const scoped = async function* (args: any) {
      const server = sdk.createSdkMcpServer({
        name: "support",
        tools: definitions.map((t) =>
          sdk.tool(
            t.name,
            t.description,
            Object.fromEntries(
              Object.entries(t.fields).map(([k, type]) => [
                k,
                type === "integer" ? z.number().int() : type === "number" ? z.number() : z.string(),
              ]),
            ),
            async (a: any) => ({ content: [{ type: "text", text: await t.execute(a) }] }),
          ),
        ),
      });
      try {
        if (runtime.currentExecution?.role !== "reflection") {
          args.options.mcpServers = { ...args.options.mcpServers, support: server };
          args.options.allowedTools = [
            ...args.options.allowedTools,
            ...definitions.map((t) => `mcp__support__${t.name}`),
          ];
        }
        yield* query(args);
      } finally {
        await server.instance.close();
      }
    };
    const directory = join(options.workdir ?? runtime.config.workspace, "claude");
    mkdirSync(directory, { recursive: true });
    const cwd = mkdtempSync(join(directory, "invocation-"));
    return conversation(
      await runtime.agent(wrapClaudeAgentQuery(scoped, { client: runtime.client }), {
        framework: "claude-agent-sdk",
        options: {
          model,
          thinking: { type: "disabled" },
          effort: "low",
          systemPrompt: instructions,
          tools: [],
          settingSources: [],
          cwd,
          maxTurns: 16,
          strictMcpConfig: true,
        },
      }),
    );
  };
  const seen = new Set<string>();
  const build = async (
    definitions: BusinessTool[],
    instructions: string,
    setup?: { validation?: boolean; inputAudit?: TaskInputAudit },
  ) => {
    const agent = await create(definitions, instructions, setup);
    return {
      ...agent,
      async invokeReflect(request: Invocation, overrides?: Record<string, any>) {
        try {
          const invoke = () => agent.invokeReflect(request, overrides);
          const result = await (openaiProcessor ? openaiInvocation.run(openaiProcessor, invoke) : invoke());
          if (options.onResult)
            options.onResult({
              execution_id: request.execution.id,
              phase: request.phase,
              ...(await nativeTelemetry(framework, result, seen)),
            });
          return result;
        } catch (error) {
          options.onResult?.({
            execution_id: request.execution.id,
            phase: request.phase,
            model_calls: null,
            tokens: null,
            native_reported_cost_usd: null,
            error: String(error),
          });
          throw error;
        }
      },
    };
  };
  return Object.assign(build, {
    framework,
    modelName: typeof options.model === "string" ? options.model : "scripted",
    forRuntime: (fresh: ReflectRuntime) => nativeFactory(fresh, framework, options),
  });
}
