import { ADMIN_ONLY, MEMORY_ID, type NativeAgent, optionalModule } from "../integrations.js";
import type { Invocation, Reflect as ReflectRuntime } from "../runtime.js";
import { TOOL_NAME } from "../shell.js";
export async function langchainAgent(
  runtime: ReflectRuntime,
  factory: any,
  settings: Record<string, any>,
  framework: "langchain" | "langgraph" | "deepagents",
  defaults: Record<string, any> = {},
): Promise<NativeAgent> {
  const { SystemMessage, HumanMessage, ToolMessage, coerceMessageLikeToMessage } =
    await optionalModule("@langchain/core/messages");
  const messages = (values: any[]) => values.map((value) => coerceMessageLikeToMessage(value));
  const memory = (values: any[], instructions: string) => [
    new SystemMessage({ content: instructions, id: MEMORY_ID }),
    ...messages(values).filter((m: any) => m.id !== MEMORY_ID),
  ];
  let options = { ...settings };
  if (framework === "langgraph") {
    const { ToolNode } = await optionalModule("@langchain/langgraph/prebuilt");
    const { RunnableLambda } = await optionalModule("@langchain/core/runnables");
    // JS ToolNode has no Python awrap_tool_call hook. Guard its actual invoke path.
    const guarded = settings.tools.map(
      (tool: any) =>
        new Proxy(tool, {
          get(target, key) {
            if (key === "invoke")
              return async (input: any, config: any) => {
                if (runtime.currentExecution?.role === "reflection" && target.name !== TOOL_NAME) {
                  const call = input?.tool_call ?? input;
                  return new ToolMessage({ content: ADMIN_ONLY, tool_call_id: call.id ?? "", status: "error" });
                }
                return target.invoke(input, config);
              };
            return globalThis.Reflect.get(target, key, target);
          },
        }),
    );
    const original = settings.prompt;
    const originalModel = settings.llm;
    const llm = async (state: any, config: any) => {
      const model = typeof originalModel === "function" ? await originalModel(state, config) : originalModel;
      if (!model?.bindTools)
        throw new TypeError(
          "Reflect LangGraph requires a chat model with native bindTools; use an explicit Invocation for custom graphs",
        );
      return model.bindTools(
        runtime.currentExecution?.role === "reflection" ? guarded.filter((t: any) => t.name === TOOL_NAME) : guarded,
      );
    };
    const prompt = RunnableLambda.from(async (state: any, config: any) => {
      let value: any = state.llm_input_messages ?? state.messages;
      if (typeof original === "string") value = [new SystemMessage(original), ...value];
      else if (original?.invoke) value = await original.invoke(state, config);
      else if (typeof original === "function") value = await original(state, config);
      else if (original) value = [original, ...value];
      if (value?.toChatMessages) value = value.toChatMessages();
      if (typeof value === "string") value = [new HumanMessage(value)];
      const execution = runtime.currentExecution;
      return execution ? memory(value, execution.instructions()) : value;
    });
    options = { ...settings, llm, tools: new ToolNode(guarded), prompt };
  } else {
    const { createMiddleware } = await optionalModule("langchain");
    if (!createMiddleware) throw new Error("Reflect requires LangChain 1.x middleware support");
    const middleware = createMiddleware({
      name: "PandaProbeReflect",
      wrapModelCall: async (request: any, handler: any) => {
        const execution = runtime.currentExecution;
        if (!execution) return handler(request);
        return handler({
          ...request,
          messages: memory(request.messages, execution.instructions()),
          ...(execution.role === "reflection"
            ? { tools: request.tools.filter((t: any) => t.name === TOOL_NAME) }
            : {}),
        });
      },
      wrapToolCall: async (request: any, handler: any) => {
        if (runtime.currentExecution?.role === "reflection" && request.toolCall.name !== TOOL_NAME)
          return new ToolMessage({ content: ADMIN_ONLY, tool_call_id: request.toolCall.id, status: "error" });
        return handler(request);
      },
    });
    options = { ...settings, middleware: [...(settings.middleware ?? []), middleware] };
  }
  const agent = await factory(options);
  return {
    framework,
    native: agent,
    async invokeReflect(request: Invocation, overrides = {}) {
      const invokeOptions = { ...defaults, ...overrides };
      const config: Record<string, any> = {
        ...invokeOptions,
        configurable: { ...invokeOptions.configurable },
        signal: request.signal,
      };
      config.configurable.thread_id =
        request.phase === "validation"
          ? request.execution.id
          : (config.configurable.thread_id ?? request.execution.sessionId);
      let value: any;
      if (request.phase === "reflection") {
        config.recursionLimit = Math.min(config.recursionLimit ?? 25, runtime.config.reflectionMaxCommands * 2 + 2);
        let history: any;
        if (agent.checkpointer) {
          if (typeof agent.getState !== "function")
            throw new TypeError("The native agent does not expose checkpoint state; supply an explicit continuation");
          history = (await agent.getState(config)).values?.messages;
        }
        value = history?.length ? {} : { ...((request.previousResult as any) ?? {}) };
        value.messages = [...messages(value.messages ?? []), new HumanMessage(request.feedback!)];
      } else {
        value =
          typeof request.input === "string"
            ? { messages: [new HumanMessage(request.input)] }
            : { ...(request.input as any) };
        if (!Array.isArray(value.messages))
          throw new TypeError(
            "Native Reflect LangChain integration requires message state; use Invocation for custom chains",
          );
      }
      value.messages = memory(value.messages, request.instructions);
      if (config.stream) {
        config.stream = undefined;
        const stream = await agent.stream(value, { ...config, streamMode: "values" });
        let result: any;
        for await (const item of stream) result = item;
        return result;
      }
      return agent.invoke(value, config);
    },
  };
}
