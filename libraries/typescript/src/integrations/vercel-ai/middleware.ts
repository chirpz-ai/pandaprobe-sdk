/**
 * Vercel AI SDK model middleware that emits a
 * PandaProbe LLM span for every `doGenerate` / `doStream`.
 *
 * Net-new vs the Python SDK, but follows the same schema/normalization contract
 * so traces are cross-compatible. Reuses the wrapper streaming-reducer engine.
 * Use wrapVercelAgent for native agent lifecycle and tool execution spans.
 *
 * Usage:
 *   import { wrapLanguageModel } from "ai";
 *   import { pandaProbeMiddleware } from "pandaprobe/integrations/vercel-ai";
 *   const model = wrapLanguageModel({ model: openai("gpt-4o"), middleware: pandaProbeMiddleware() });
 */

import { closeLlmSpan, errorLlmSpan, openLlmSpan, setSpanUsage } from "../../wrappers/base.js";
import { vercelRuns } from "./run.js";
import { extractVercelOutput, extractVercelUsage, normalizeVercelInput } from "./utils.js";

type Any = any;

export interface VercelMiddlewareOptions {
  /** Span/trace name (default "vercel-ai-generate"). */
  name?: string;
}

function spanModel(model: Any): string | null {
  return model?.modelId ?? model?.modelID ?? null;
}

function openSpan(params: Any, model: Any, name: string) {
  const run = vercelRuns.getStore();
  if (run) {
    const span = run.model(params, model, name);
    return {
      span,
      usage: (usage: Record<string, number> | null) => {
        if (usage) span?.setTokenUsage(usage);
      },
      end: (error?: unknown) => span?.end(error),
    };
  }
  const span = openLlmSpan({
    methodName: name,
    inputData: normalizeVercelInput(params),
    modelParams: {},
    model: spanModel(model),
  });
  return {
    span,
    usage: (usage: Record<string, number> | null) => {
      if (span) setSpanUsage(span, usage);
    },
    end: (error?: unknown) => (error == null ? closeLlmSpan(span) : errorLlmSpan(span, error)),
  };
}

/** Build a PandaProbe middleware object for the AI SDK's `wrapLanguageModel`. */
export function pandaProbeMiddleware(options: VercelMiddlewareOptions = {}) {
  const name = options.name ?? "vercel-ai-generate";

  return {
    wrapGenerate: async ({ doGenerate, params, model }: Any): Promise<Any> => {
      const observation = openSpan(params, model, name);
      const { span } = observation;
      try {
        const result = await doGenerate();
        if (span !== null) {
          const output = extractVercelOutput(result);
          if (output !== null) span.setOutput(output);
          observation.usage(extractVercelUsage(result?.usage));
        }
        observation.end();
        return result;
      } catch (exc) {
        observation.end(exc);
        throw exc;
      }
    },

    wrapStream: async ({ doStream, params, model }: Any): Promise<Any> => {
      const observation = openSpan(params, model, name);
      const { span } = observation;
      let result: Any;
      try {
        result = await doStream();
      } catch (exc) {
        observation.end(exc);
        throw exc;
      }

      const textParts: string[] = [];
      const toolCalls: Any[] = [];
      let usage: Record<string, number> | null = null;
      let first = true;
      let finalized = false;

      // Finalize the span exactly once, on ANY stream termination: normal
      // completion, a read error, or downstream cancellation / partial read.
      // (A TransformStream's `flush` only runs on normal completion, leaking the
      // span — and any standalone trace — on the other paths.)
      const finalize = (error?: unknown): void => {
        if (finalized || span === null) {
          return;
        }
        finalized = true;
        if (error !== undefined) {
          vercelRuns.getStore()?.streamError(error);
          observation.end(error);
          return;
        }
        const output = extractVercelOutput({ text: textParts.join(""), toolCalls });
        if (output) span.setOutput(output);
        observation.usage(usage);
        observation.end();
      };

      const reader = (result.stream as ReadableStream<Any>).getReader();
      const wrapped = new ReadableStream<Any>({
        async pull(controller) {
          try {
            const { done, value } = await reader.read();
            if (done) {
              finalize();
              controller.close();
              return;
            }
            if (span !== null && first) {
              span.setCompletionStartTime(new Date());
              first = false;
            }
            if (value?.type === "text-delta" && typeof value.delta === "string") {
              textParts.push(value.delta);
            } else if (value?.type === "text" && typeof value.text === "string") {
              textParts.push(value.text);
            }
            if (value?.type === "tool-call") toolCalls.push(value);
            if (value?.type === "error") finalize(value.error ?? new Error("Vercel model stream failed"));
            if (value?.type === "finish" && value.usage) {
              usage = extractVercelUsage(value.usage);
            }
            controller.enqueue(value);
          } catch (exc) {
            finalize(exc);
            controller.error(exc);
          }
        },
        cancel(reason) {
          // Downstream cancelled / didn't read to completion — finalize with
          // whatever was collected, then release the source.
          finalize();
          return reader.cancel(reason);
        },
      });

      return { ...result, stream: wrapped };
    },
  };
}
