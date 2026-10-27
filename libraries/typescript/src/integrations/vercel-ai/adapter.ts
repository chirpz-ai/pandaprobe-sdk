import { type Client, getClient } from "../../client.js";
import type { AdapterOptions } from "../base.js";
import { VercelRun, vercelRuns } from "./run.js";

type Any = any;
export interface VercelAgentOptions extends AdapterOptions {
  name?: string;
}
const wrapped = new WeakSet<object>();

/** Instrument a native ToolLoopAgent instance or constructor without changing its public types. */
export function wrapVercelAgent<T extends object>(agent: T, options: VercelAgentOptions = {}): T {
  if (wrapped.has(agent)) return agent;
  const proxy = new Proxy(agent, {
    construct(target, args, newTarget) {
      return wrapVercelAgent(Reflect.construct(target as Any, args, newTarget), options);
    },
    get(target, key) {
      const value = Reflect.get(target, key, target);
      if (typeof value !== "function" || (key !== "generate" && key !== "stream"))
        return typeof value === "function" ? value.bind(target) : value;
      return async (input: Any) => {
        let client: Client | null = null;
        try {
          client = options.client ?? getClient();
        } catch {
          /* unavailable client */
        }
        if (!client?.enabled) return value.call(target, input);
        const run = new VercelRun(client, options, input, (target as Any).id);
        const call = { ...input };
        const callback = (names: string[], observe: (event: Any) => void, finish = false) => {
          const original = names.map((name) => input[name]).find((fn) => typeof fn === "function");
          const handler = async (event: Any) => {
            try {
              observe(event);
            } catch {
              /* Observation must not prevent native callbacks. */
            }
            // Let the native SDK decide whether callback failures propagate.
            await original?.(event);
            if (finish && key === "stream") run.finish();
          };
          for (const name of names) call[name] = handler;
        };
        // AI SDK 6 names and their AI SDK 7 replacements; native SDK chooses one.
        callback(["onStart", "experimental_onStart"], (event) => run.start(event));
        callback(["onStepStart", "experimental_onStepStart"], (event) => run.stepStart(event));
        callback(["onToolExecutionStart", "experimental_onToolCallStart"], (event) => run.toolStart(event));
        callback(["onToolExecutionEnd", "experimental_onToolCallFinish"], (event) => run.toolEnd(event));
        callback(["onStepEnd", "onStepFinish"], (event) => run.stepEnd(event));
        callback(["onEnd", "onFinish"], (event) => run.result(event), true);
        run.watch(input.abortSignal);
        return vercelRuns.run(run, async () => {
          try {
            const result = await value.call(target, call);
            if (key === "generate") {
              run.result(result);
              run.finish();
              return result;
            }
            return streamResult(result, run);
          } catch (error) {
            run.finish(error);
            throw error;
          }
        });
      };
    },
  });
  wrapped.add(proxy);
  return proxy;
}

function streamResult<T extends object>(result: T, run: VercelRun): T {
  return new Proxy(result, {
    get(target, key) {
      const value = Reflect.get(target, key, target);
      if (value instanceof ReadableStream) {
        const reader = value.getReader();
        return new ReadableStream({
          async pull(controller) {
            try {
              const { done, value } = await reader.read();
              if (done) {
                run.finish();
                reader.releaseLock();
                controller.close();
                return;
              }
              if (value?.type === "error") run.finish(value.error ?? new Error("Vercel agent stream failed"));
              if (value?.type === "abort") run.finish(new Error("Vercel agent aborted"));
              controller.enqueue(value);
            } catch (error) {
              run.finish(error);
              reader.releaseLock();
              controller.error(error);
            }
          },
          cancel(reason) {
            run.finish(reason ?? new Error("Vercel agent stream cancelled"));
            return reader.cancel(reason).finally(() => reader.releaseLock());
          },
        });
      }
      if (value instanceof Promise)
        return value.catch((error) => {
          run.finish(error);
          throw error;
        });
      if (key === "consumeStream" && typeof value === "function")
        return async (...args: Any[]) => {
          try {
            return await value.apply(target, args);
          } catch (error) {
            run.finish(error);
            throw error;
          } finally {
            run.finish();
          }
        };
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}
