/**
 * Vercel AI SDK integration — a `LanguageModelV2Middleware` that emits a
 * PandaProbe LLM span for every `doGenerate` / `doStream`.
 *
 * Net-new vs the Python SDK, but follows the same schema/normalization contract
 * so traces are cross-compatible. Reuses the wrapper streaming-reducer engine.
 *
 * Usage:
 *   import { wrapLanguageModel } from "ai";
 *   import { pandaProbeMiddleware } from "pandaprobe/integrations/vercel-ai";
 *   const model = wrapLanguageModel({ model: openai("gpt-4o"), middleware: pandaProbeMiddleware() });
 */

import type { SpanContext } from "../../tracing/span.js";
import { closeLlmSpan, errorLlmSpan, openLlmSpan, setSpanUsage } from "../../wrappers/base.js";
import { extractVercelText, extractVercelUsage, normalizeVercelInput } from "./utils.js";

type Any = any;

export interface VercelMiddlewareOptions {
  /** Span/trace name (default "vercel-ai-generate"). */
  name?: string;
}

function spanModel(model: Any): string | null {
  return model?.modelId ?? model?.modelID ?? null;
}

function openSpan(params: Any, model: Any, name: string): SpanContext | null {
  return openLlmSpan({
    methodName: name,
    inputData: normalizeVercelInput(params),
    modelParams: {},
    model: spanModel(model),
  });
}

/** Build a PandaProbe middleware object for the AI SDK's `wrapLanguageModel`. */
export function pandaProbeMiddleware(options: VercelMiddlewareOptions = {}) {
  const name = options.name ?? "vercel-ai-generate";

  return {
    wrapGenerate: async ({ doGenerate, params, model }: Any): Promise<Any> => {
      const span = openSpan(params, model, name);
      try {
        const result = await doGenerate();
        if (span !== null) {
          const text = extractVercelText(result);
          if (text !== null) {
            span.setOutput({ messages: [{ role: "assistant", content: text }] });
          }
          setSpanUsage(span, extractVercelUsage(result?.usage));
        }
        closeLlmSpan(span);
        return result;
      } catch (exc) {
        errorLlmSpan(span, exc);
        throw exc;
      }
    },

    wrapStream: async ({ doStream, params, model }: Any): Promise<Any> => {
      const span = openSpan(params, model, name);
      let result: Any;
      try {
        result = await doStream();
      } catch (exc) {
        errorLlmSpan(span, exc);
        throw exc;
      }

      const textParts: string[] = [];
      let usage: Record<string, number> | null = null;
      let first = true;

      const transform = new TransformStream({
        transform(chunk: Any, controller: Any) {
          if (span !== null && first) {
            span.setCompletionStartTime(new Date());
            first = false;
          }
          if (chunk?.type === "text-delta" && typeof chunk.delta === "string") {
            textParts.push(chunk.delta);
          } else if (chunk?.type === "text" && typeof chunk.text === "string") {
            textParts.push(chunk.text);
          }
          if (chunk?.type === "finish" && chunk.usage) {
            usage = extractVercelUsage(chunk.usage);
          }
          controller.enqueue(chunk);
        },
        flush() {
          if (span !== null) {
            if (textParts.length > 0) {
              span.setOutput({ messages: [{ role: "assistant", content: textParts.join("") }] });
            }
            setSpanUsage(span, usage);
            closeLlmSpan(span);
          }
        },
      });

      return { ...result, stream: result.stream.pipeThrough(transform) };
    },
  };
}
