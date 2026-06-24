/** wrapOpenAI — automatic LLM span instrumentation for the OpenAI JS client. */

import { SpanKind } from "../../schemas.js";
import type { SpanContext } from "../../tracing/span.js";
import { closeLlmSpan, enterLlmSpan, errorLlmSpan, safeSerialize, wrapAsyncStream } from "../base.js";
import { enterResponsesSpan, stripUndefined } from "./utils.js";

type Any = any;

/**
 * Instrument an `OpenAI` or `AzureOpenAI` client (sync API is async in JS).
 *
 * Patches `chat.completions.create` and `responses.create` so every call emits
 * an LLM span. For the Responses API, built-in tool calls are captured as child
 * TOOL spans. Returns the same client instance (mutated).
 */
export function wrapOpenAI<T>(client: T): T {
  patchChatCompletions(client as Any);
  patchResponses(client as Any);
  return client;
}

function patchChatCompletions(client: Any): void {
  const completions = client?.chat?.completions;
  if (!completions || typeof completions.create !== "function") {
    return;
  }
  const original = completions.create.bind(completions);
  completions.create = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const cleaned = stripUndefined(params ?? {});
    const span = enterLlmSpan(cleaned, "openai-chat");
    try {
      const response = await original(params, ...rest);
      if (cleaned.stream) {
        return wrapAsyncStream(response as AsyncIterable<Any>, span, reduceOpenAIStream);
      }
      finishSpanFromChatResponse(span, response);
      return response;
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

function patchResponses(client: Any): void {
  const responses = client?.responses;
  if (!responses || typeof responses.create !== "function") {
    return;
  }
  const original = responses.create.bind(responses);
  responses.create = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const cleaned = stripUndefined(params ?? {});
    const span = enterResponsesSpan(cleaned);
    try {
      const response = await original(params, ...rest);
      if (cleaned.stream) {
        return wrapAsyncStream(response as AsyncIterable<Any>, span, reduceResponsesStream);
      }
      finishFromResponse(span, response);
      return response;
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

// ---------------------------------------------------------------------------
// Token usage helpers
// ---------------------------------------------------------------------------

function extractTokenDetails(usage: Any): Record<string, number> {
  const extra: Record<string, number> = {};
  const total = usage?.total_tokens;
  if (typeof total === "number" && total > 0) {
    extra.total_tokens = total;
  }
  for (const attr of ["completion_tokens_details", "output_tokens_details"]) {
    const details = usage?.[attr];
    if (!details || typeof details !== "object") {
      continue;
    }
    for (const [k, v] of Object.entries(details)) {
      if (typeof v === "number" && v > 0) {
        extra[k] = v;
      }
    }
  }
  return extra;
}

// ---------------------------------------------------------------------------
// Chat Completions extraction
// ---------------------------------------------------------------------------

function finishSpanFromChatResponse(span: SpanContext | null, response: Any): void {
  if (span === null) {
    return;
  }
  try {
    const choice = response?.choices?.[0];
    if (choice?.message) {
      span.setOutput({ messages: [safeSerialize(choice.message)] });
    } else if (choice?.text != null) {
      span.setOutput({ messages: [{ role: "assistant", content: choice.text }] });
    }
    if (response?.model) {
      span.setModel(response.model);
    }
    const usage = response?.usage;
    if (usage) {
      span.setTokenUsage({
        promptTokens: usage.prompt_tokens ?? 0,
        completionTokens: usage.completion_tokens ?? 0,
        ...extractTokenDetails(usage),
      });
    }
  } catch {
    // best-effort extraction
  }
  closeLlmSpan(span);
}

function reduceOpenAIStream(span: SpanContext, chunks: Any[]): void {
  const contentParts: string[] = [];
  let model: string | undefined;
  let usage: Record<string, number> | undefined;

  for (const chunk of chunks) {
    if (chunk?.model) {
      model = chunk.model;
    }
    const delta = chunk?.choices?.[0]?.delta;
    if (delta?.content) {
      contentParts.push(delta.content);
    }
    if (chunk?.usage) {
      usage = {
        prompt_tokens: chunk.usage.prompt_tokens ?? 0,
        completion_tokens: chunk.usage.completion_tokens ?? 0,
        ...extractTokenDetails(chunk.usage),
      };
    }
  }

  try {
    if (contentParts.length > 0) {
      span.setOutput({ messages: [{ role: "assistant", content: contentParts.join("") }] });
    }
    if (model) {
      span.setModel(model);
    }
    if (usage) {
      const { prompt_tokens, completion_tokens, ...extra } = usage;
      span.setTokenUsage({ promptTokens: prompt_tokens, completionTokens: completion_tokens, ...extra });
    }
  } catch {
    // best-effort
  }
  closeLlmSpan(span);
}

// ---------------------------------------------------------------------------
// Responses API extraction
// ---------------------------------------------------------------------------

function getField(obj: Any, field: string): Any {
  if (obj == null) {
    return undefined;
  }
  return obj[field];
}

function reduceResponsesStream(span: SpanContext, events: Any[]): void {
  let completed: Any;
  for (const ev of events) {
    if (getField(ev, "type") === "response.completed") {
      completed = getField(ev, "response");
    }
  }
  if (completed) {
    finishFromResponse(span, completed);
  } else {
    closeLlmSpan(span);
  }
}

function finishFromResponse(span: SpanContext | null, response: Any): void {
  if (span === null) {
    return;
  }
  const outputItems: Any[] = getField(response, "output") ?? [];

  try {
    const messageItems = outputItems
      .filter((item) => getField(item, "type") === "message")
      .map((item) => safeSerialize(item));
    if (messageItems.length > 0) {
      span.setOutput({ messages: messageItems });
    }
  } catch {
    // ignore
  }

  try {
    if (response?.model) {
      span.setModel(response.model);
    }
  } catch {
    // ignore
  }

  try {
    const usage = response?.usage;
    if (usage) {
      span.setTokenUsage({
        promptTokens: usage.input_tokens ?? 0,
        completionTokens: usage.output_tokens ?? 0,
        ...extractTokenDetails(usage),
      });
    }
  } catch {
    // ignore
  }

  try {
    extractReasoningSummary(span, outputItems);
  } catch {
    // ignore
  }

  try {
    createToolChildSpans(span, outputItems);
  } catch {
    // ignore
  }

  closeLlmSpan(span);
}

function extractReasoningSummary(span: SpanContext, outputItems: Any[]): void {
  const summaries: string[] = [];
  for (const item of outputItems) {
    if (getField(item, "type") !== "reasoning") {
      continue;
    }
    const summaryList = getField(item, "summary");
    if (!Array.isArray(summaryList)) {
      continue;
    }
    for (const s of summaryList) {
      const text = getField(s, "text");
      if (text) {
        summaries.push(text);
      }
    }
  }
  if (summaries.length > 0) {
    span.setMetadata({ reasoning_summary: summaries.join("\n\n") });
  }
}

function createToolChildSpans(span: SpanContext, outputItems: Any[]): void {
  const traceCtx = span.getTraceContext();
  for (const item of outputItems) {
    const itemType = getField(item, "type");
    if (itemType === "function_call") {
      const name = getField(item, "name") ?? "function_call";
      const toolSpan = traceCtx.span(`function_call:${name}`, { kind: SpanKind.TOOL });
      toolSpan.start();
      const args = getField(item, "arguments");
      if (args) {
        toolSpan.setInput(safeSerialize(args));
      }
      toolSpan.end();
    } else if (
      itemType === "web_search_call" ||
      itemType === "file_search_call" ||
      itemType === "code_interpreter_call"
    ) {
      const toolName =
        itemType === "web_search_call"
          ? "web_search"
          : itemType === "file_search_call"
            ? "file_search"
            : "code_interpreter";
      const toolSpan = traceCtx.span(toolName, { kind: SpanKind.TOOL });
      toolSpan.start();
      toolSpan.setInput(safeSerialize(item));
      const status = getField(item, "status");
      if (status) {
        toolSpan.setOutput({ status });
      }
      toolSpan.end();
    }
  }
}
