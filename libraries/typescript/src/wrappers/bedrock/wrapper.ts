/** wrapBedrock — automatic LLM span instrumentation for the AWS Bedrock JS client. */

import type { SpanContext } from "../../tracing/span.js";
import { closeLlmSpan, errorLlmSpan, safeSerialize, setSpanUsage, wrapAsyncStream } from "../base.js";
import { decodeBody, enterBedrockSpan, mapConverseUsage, mapInvokeModelUsage } from "./utils.js";

type Any = any;

/**
 * Instrument an `@aws-sdk/client-bedrock-runtime` `BedrockRuntimeClient`.
 *
 * The JS SDK is command-based, so this patches `client.send` and branches on
 * the command type: ConverseCommand, ConverseStreamCommand, InvokeModelCommand,
 * InvokeModelWithResponseStreamCommand. Returns the same client (mutated).
 */
export function wrapBedrock<T>(client: T): T {
  const c = client as Any;
  if (!c || typeof c.send !== "function") {
    return client;
  }
  const originalSend = c.send.bind(c);

  c.send = async (command: Any, ...rest: Any[]): Promise<Any> => {
    const name = command?.constructor?.name;
    const input: Record<string, unknown> = command?.input ?? {};
    const modelId = (input as Any).modelId as string | undefined;

    if (name === "ConverseCommand") {
      const span = enterBedrockSpan(input, "bedrock-converse", "converse");
      try {
        const response = await originalSend(command, ...rest);
        finishConverseSpan(span, response, modelId);
        return response;
      } catch (exc) {
        errorLlmSpan(span, exc);
        throw exc;
      }
    }

    if (name === "ConverseStreamCommand") {
      const span = enterBedrockSpan(input, "bedrock-converse", "converse");
      try {
        const response = await originalSend(command, ...rest);
        return wrapConverseStreamResponse(response, span, modelId);
      } catch (exc) {
        errorLlmSpan(span, exc);
        throw exc;
      }
    }

    if (name === "InvokeModelCommand") {
      const span = enterBedrockSpan(input, "bedrock-invoke-model", "invoke_model");
      try {
        const response = await originalSend(command, ...rest);
        finishInvokeModelSpan(span, response, modelId);
        return response;
      } catch (exc) {
        errorLlmSpan(span, exc);
        throw exc;
      }
    }

    if (name === "InvokeModelWithResponseStreamCommand") {
      const span = enterBedrockSpan(input, "bedrock-invoke-model-stream", "invoke_model");
      try {
        const response = await originalSend(command, ...rest);
        return wrapInvokeStreamResponse(response, span, modelId);
      } catch (exc) {
        errorLlmSpan(span, exc);
        throw exc;
      }
    }

    return originalSend(command, ...rest);
  };

  return client;
}

// ---------------------------------------------------------------------------
// Converse — blocking
// ---------------------------------------------------------------------------

function splitConverseContent(content: Any): { textParts: string[]; reasoningParts: string[] } {
  const textParts: string[] = [];
  const reasoningParts: string[] = [];
  if (!Array.isArray(content)) {
    return { textParts, reasoningParts };
  }
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    if (typeof block.text === "string") {
      textParts.push(block.text);
    } else if (block.reasoningContent?.reasoningText?.text) {
      reasoningParts.push(block.reasoningContent.reasoningText.text);
    }
  }
  return { textParts, reasoningParts };
}

function finishConverseSpan(span: SpanContext | null, response: Any, modelId?: string): void {
  if (span === null) {
    return;
  }
  try {
    const message = response?.output?.message;
    if (message && typeof message === "object") {
      const { textParts, reasoningParts } = splitConverseContent(message.content);
      if (textParts.length > 0) {
        span.setOutput({ messages: [{ role: "assistant", content: textParts.join("") }] });
      } else {
        span.setOutput({ messages: [safeSerialize(message)] });
      }
      if (reasoningParts.length > 0) {
        span.setMetadata({ reasoning_summary: reasoningParts.join("\n\n") });
      }
    }
  } catch {
    // ignore
  }
  if (modelId) {
    span.setModel(modelId);
  }
  setSpanUsage(span, mapConverseUsage(response?.usage));
  closeLlmSpan(span);
}

// ---------------------------------------------------------------------------
// Converse — streaming
// ---------------------------------------------------------------------------

function wrapConverseStreamResponse(response: Any, span: SpanContext | null, modelId?: string): Any {
  if (!response || typeof response !== "object" || !("stream" in response)) {
    closeLlmSpan(span);
    return response;
  }
  const inner = response.stream as AsyncIterable<Any>;
  const textParts: string[] = [];
  const reasoningParts: string[] = [];
  let usage: Record<string, number> | null = null;

  const reduce = (s: SpanContext): void => {
    if (textParts.length > 0) {
      s.setOutput({ messages: [{ role: "assistant", content: textParts.join("") }] });
    }
    if (reasoningParts.length > 0) {
      s.setMetadata({ reasoning_summary: reasoningParts.join("\n\n") });
    }
    if (modelId) {
      s.setModel(modelId);
    }
    setSpanUsage(s, usage);
    closeLlmSpan(s);
  };

  async function* tap(): AsyncGenerator<Any> {
    for await (const event of inner) {
      const delta = event?.contentBlockDelta?.delta;
      if (delta) {
        if (typeof delta.text === "string" && delta.text) {
          textParts.push(delta.text);
        }
        const rtext = delta.reasoningContent?.text;
        if (typeof rtext === "string" && rtext) {
          reasoningParts.push(rtext);
        }
      }
      if (event?.metadata?.usage) {
        usage = mapConverseUsage(event.metadata.usage) ?? usage;
      }
      yield event;
    }
  }

  return { ...response, stream: wrapAsyncStream(tap(), span, reduce) };
}

// ---------------------------------------------------------------------------
// InvokeModel — blocking
// ---------------------------------------------------------------------------

function extractInvokeModelText(parsed: Any): string | null {
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  if (Array.isArray(parsed.content)) {
    const parts = parsed.content.filter((b: Any) => b && typeof b.text === "string").map((b: Any) => b.text as string);
    if (parts.length > 0) {
      return parts.join("");
    }
  }
  for (const key of ["completion", "generation", "output_text"]) {
    if (typeof parsed[key] === "string") {
      return parsed[key];
    }
  }
  const r0 = parsed.results?.[0];
  if (r0 && typeof r0 === "object") {
    for (const key of ["outputText", "output_text", "text"]) {
      if (typeof r0[key] === "string") {
        return r0[key];
      }
    }
  }
  const c0 = parsed.choices?.[0];
  if (c0 && typeof c0 === "object") {
    if (typeof c0.message?.content === "string") {
      return c0.message.content;
    }
    if (typeof c0.text === "string") {
      return c0.text;
    }
  }
  const g0 = parsed.generations?.[0];
  if (g0 && typeof g0.text === "string") {
    return g0.text;
  }
  return null;
}

function finishInvokeModelSpan(span: SpanContext | null, response: Any, modelId?: string): void {
  if (span === null) {
    return;
  }
  const parsed = decodeBody(response?.body);
  try {
    const text = extractInvokeModelText(parsed);
    if (text !== null) {
      span.setOutput({ messages: [{ role: "assistant", content: text }] });
    } else if (parsed != null) {
      span.setOutput({ messages: [{ role: "assistant", content: safeSerialize(parsed) }] });
    }
  } catch {
    // ignore
  }
  if (modelId) {
    span.setModel(modelId);
  }
  setSpanUsage(span, mapInvokeModelUsage(parsed));
  closeLlmSpan(span);
}

// ---------------------------------------------------------------------------
// InvokeModel — streaming (minimal span; provider bodies differ wildly)
// ---------------------------------------------------------------------------

function wrapInvokeStreamResponse(response: Any, span: SpanContext | null, modelId?: string): Any {
  if (!response || typeof response !== "object" || !("body" in response)) {
    closeLlmSpan(span);
    return response;
  }
  const inner = response.body as AsyncIterable<Any>;
  const reduce = (s: SpanContext): void => {
    if (modelId) {
      s.setModel(modelId);
    }
    closeLlmSpan(s);
  };
  return { ...response, body: wrapAsyncStream(inner, span, reduce) };
}
