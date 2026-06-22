/** AWS Bedrock-specific utilities for the wrapBedrock instrumentation. */

import type { SpanContext } from "../../tracing/span.js";
import { openLlmSpan, safeSerialize, tokenUsage } from "../base.js";

type Any = any;

export const SAFE_BEDROCK_PARAMS = new Set([
  "temperature",
  "topP",
  "maxTokens",
  "stopSequences",
  "guardrailConfig",
  "additionalModelRequestFields",
  "toolConfig",
]);

function normalizeRole(role: string): string {
  return role === "model" ? "assistant" : role;
}

function flattenContentBlocks(content: Any): Any {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return safeSerialize(content);
  }
  const textOnly: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object" && Object.keys(block).length === 1 && typeof block.text === "string") {
      textOnly.push(block.text);
    } else {
      return safeSerialize(content);
    }
  }
  return textOnly.join("");
}

function flattenSystemBlocks(system: Any): Any {
  if (typeof system === "string") {
    return system;
  }
  if (!Array.isArray(system)) {
    return safeSerialize(system);
  }
  const textOnly: string[] = [];
  for (const block of system) {
    if (block && typeof block === "object" && typeof block.text === "string") {
      textOnly.push(block.text);
    } else {
      return safeSerialize(system);
    }
  }
  return textOnly.join("\n");
}

/** Convert Converse API `system` + `messages` into the universal schema. */
export function normalizeConverseInput(input: Record<string, unknown>): { messages: unknown[] } {
  const messages: unknown[] = [];
  const system = (input as Any).system;
  if (system != null) {
    messages.push({ role: "system", content: flattenSystemBlocks(system) });
  }
  const raw = (input as Any).messages;
  if (Array.isArray(raw)) {
    for (const msg of raw) {
      if (!msg || typeof msg !== "object") {
        continue;
      }
      messages.push({ role: normalizeRole(msg.role ?? "user"), content: flattenContentBlocks(msg.content) });
    }
  }
  return { messages };
}

/** Decode an InvokeModel body (Uint8Array | string | object) to a value. */
export function decodeBody(body: Any): Any {
  if (body == null) {
    return null;
  }
  let text: string | undefined;
  if (body instanceof Uint8Array) {
    text = new TextDecoder().decode(body);
  } else if (typeof body === "string") {
    text = body;
  } else {
    return body;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** Best-effort universal-schema normalization for InvokeModel input bodies. */
export function normalizeInvokeModelInput(input: Record<string, unknown>): { messages: unknown[] } {
  const parsed = decodeBody((input as Any).body);
  if (parsed == null) {
    return { messages: [] };
  }
  if (typeof parsed === "string") {
    return { messages: [{ role: "user", content: parsed }] };
  }
  if (parsed && typeof parsed === "object") {
    const messages: unknown[] = [];
    const system = parsed.system;
    if (system) {
      messages.push({ role: "system", content: flattenSystemBlocks(system) });
    }
    if (Array.isArray(parsed.messages)) {
      for (const msg of parsed.messages) {
        if (!msg || typeof msg !== "object") {
          continue;
        }
        messages.push({ role: normalizeRole(msg.role ?? "user"), content: flattenContentBlocks(msg.content) });
      }
      return { messages };
    }
    for (const key of ["prompt", "inputText", "input"]) {
      if (typeof parsed[key] === "string") {
        messages.push({ role: "user", content: parsed[key] });
        return { messages };
      }
    }
    messages.push({ role: "user", content: safeSerialize(parsed) });
    return { messages };
  }
  return { messages: [{ role: "user", content: safeSerialize(parsed) }] };
}

export function extractBedrockParams(input: Record<string, unknown>): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  const inferenceConfig = (input as Any).inferenceConfig;
  if (inferenceConfig && typeof inferenceConfig === "object") {
    for (const [k, v] of Object.entries(inferenceConfig)) {
      if (SAFE_BEDROCK_PARAMS.has(k) && v !== undefined) {
        params[k] = safeSerialize(v);
      }
    }
  }
  for (const [k, v] of Object.entries(input)) {
    if (SAFE_BEDROCK_PARAMS.has(k) && v !== undefined) {
      params[k] = safeSerialize(v);
    }
  }
  return params;
}

export type BedrockApi = "converse" | "invoke_model";

/** Open an LLM span for an AWS Bedrock command. */
export function enterBedrockSpan(
  input: Record<string, unknown>,
  methodName: string,
  api: BedrockApi,
): SpanContext | null {
  return openLlmSpan({
    methodName,
    inputData: api === "invoke_model" ? normalizeInvokeModelInput(input) : normalizeConverseInput(input),
    modelParams: extractBedrockParams(input),
    model: ((input as Any).modelId as string | undefined) ?? null,
  });
}

// ---------------------------------------------------------------------------
// Usage mapping
// ---------------------------------------------------------------------------

export function mapConverseUsage(usage: Any): Record<string, number> | null {
  if (!usage || typeof usage !== "object") {
    return null;
  }
  return tokenUsage({
    prompt_tokens: usage.inputTokens || undefined,
    completion_tokens: usage.outputTokens || undefined,
    total_tokens: usage.totalTokens || undefined,
    cache_read_tokens: usage.cacheReadInputTokens || undefined,
    cache_creation_tokens: usage.cacheWriteInputTokens || undefined,
  });
}

export function mapInvokeModelUsage(parsed: Any): Record<string, number> | null {
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  const usage = parsed.usage;
  if (usage && typeof usage === "object") {
    const out: Record<string, number> = {};
    const prompt = usage.input_tokens ?? usage.prompt_tokens;
    const completion = usage.output_tokens ?? usage.completion_tokens;
    if (typeof prompt === "number" && prompt > 0) {
      out.prompt_tokens = prompt;
    }
    if (typeof completion === "number" && completion > 0) {
      out.completion_tokens = completion;
    }
    if (typeof usage.total_tokens === "number" && usage.total_tokens > 0) {
      out.total_tokens = usage.total_tokens;
    }
    if (typeof usage.cache_read_input_tokens === "number" && usage.cache_read_input_tokens > 0) {
      out.cache_read_tokens = usage.cache_read_input_tokens;
    }
    if (typeof usage.cache_creation_input_tokens === "number" && usage.cache_creation_input_tokens > 0) {
      out.cache_creation_tokens = usage.cache_creation_input_tokens;
    }
    if (Object.keys(out).length > 0) {
      return out;
    }
  }
  if (typeof parsed.inputTextTokenCount === "number" && parsed.inputTextTokenCount > 0) {
    const out: Record<string, number> = { prompt_tokens: parsed.inputTextTokenCount };
    const tc = parsed.results?.[0]?.tokenCount;
    if (typeof tc === "number" && tc > 0) {
      out.completion_tokens = tc;
    }
    return out;
  }
  const billed = parsed.meta?.billed_units;
  if (billed && typeof billed === "object") {
    return tokenUsage({ prompt_tokens: billed.input_tokens, completion_tokens: billed.output_tokens });
  }
  return null;
}
