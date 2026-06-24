/** OpenAI Agents SDK normalization and extraction utilities (ports openai_agents/utils.py). */

import { SpanKind } from "../../schemas.js";
import { SAFE_MODEL_PARAM_KEYS, configToDict, safeSerialize } from "../base.js";

type Any = any;

// ---------------------------------------------------------------------------
// Content helpers
// ---------------------------------------------------------------------------

/** Collapse single-element text content lists to plain strings. */
export function collapseContent(content: Any): Any {
  if (!Array.isArray(content) || content.length === 0) {
    return content;
  }
  const parts: string[] = [];
  for (const item of content) {
    if (item && typeof item === "object") {
      const t = item.type ?? "";
      if (t === "output_text" || t === "text") {
        const text = item.text;
        if (text != null) {
          parts.push(String(text));
          continue;
        }
      }
    }
    return content;
  }
  if (parts.length === 0) {
    return content;
  }
  return parts.length > 1 ? parts.join(" ") : parts[0];
}

/** Retrieve a value from a dict/object (plain-object access). */
function get(obj: Any, key: string): Any {
  return obj == null ? undefined : obj[key];
}

// ---------------------------------------------------------------------------
// Input normalization
// ---------------------------------------------------------------------------

function normalizeInputItem(item: Any): Record<string, Any> | null {
  if (typeof item === "string") {
    return { role: "user", content: item };
  }
  if (item && typeof item === "object") {
    const itemType = item.type ?? "";

    if (itemType === "message") {
      return { role: item.role ?? "user", content: collapseContent(item.content) };
    }
    if (itemType === "function_call_output") {
      const output = item.output;
      const content =
        output && typeof output === "object" ? JSON.stringify(output) : output == null ? "" : String(output);
      return { role: "tool", content, tool_call_id: item.call_id };
    }
    if (itemType === "item_reference") {
      return null;
    }
    if (itemType === "function_call") {
      return {
        role: "assistant",
        content: null,
        tool_calls: [{ id: item.id, name: item.name, arguments: item.arguments }],
      };
    }
    if ("role" in item && !itemType) {
      let content = item.content;
      if (Array.isArray(content)) {
        content = collapseContent(content);
      }
      const result: Record<string, Any> = { role: item.role, content };
      if ("tool_calls" in item) {
        result.tool_calls = item.tool_calls;
      }
      if ("name" in item) {
        result.name = item.name;
      }
      if ("tool_call_id" in item) {
        result.tool_call_id = item.tool_call_id;
      }
      return result;
    }
    return { role: "user", content: safeSerialize(item) };
  }
  return { role: "user", content: safeSerialize(item) };
}

/** Build `{messages: [...]}` from a ResponseSpanData (instructions + input). */
export function normalizeResponseInput(spanData: Any): { messages: Any[] } {
  const messages: Any[] = [];
  const response = get(spanData, "response");
  if (response != null) {
    const instructions = get(response, "instructions");
    if (typeof instructions === "string" && instructions.trim()) {
      messages.push({ role: "system", content: instructions });
    }
  }
  const rawInput = get(spanData, "input");
  if (rawInput != null) {
    if (typeof rawInput === "string") {
      messages.push({ role: "user", content: rawInput });
    } else if (Array.isArray(rawInput)) {
      for (const item of rawInput) {
        const msg = normalizeInputItem(item);
        if (msg !== null) {
          messages.push(msg);
        }
      }
    } else {
      const msg = normalizeInputItem(rawInput);
      if (msg !== null) {
        messages.push(msg);
      }
    }
  }
  return { messages };
}

// ---------------------------------------------------------------------------
// Output normalization
// ---------------------------------------------------------------------------

function normalizeOutputItem(item: Any): Record<string, Any> | null {
  if (item && typeof item === "object") {
    const itemType = item.type ?? "";
    if (itemType === "reasoning") {
      return null;
    }
    if (itemType === "message") {
      return { role: item.role ?? "assistant", content: collapseContent(item.content) };
    }
    if (itemType === "function_call") {
      return {
        role: "assistant",
        content: null,
        tool_calls: [{ id: item.id, name: item.name, arguments: item.arguments }],
      };
    }
    return { role: "assistant", content: safeSerialize(item) };
  }
  return { role: "assistant", content: safeSerialize(item) };
}

/** Build `{messages: [...]}` from ResponseSpanData.response.output (reasoning stripped). */
export function normalizeResponseOutput(spanData: Any): { messages: Any[] } {
  const response = get(spanData, "response");
  if (response == null) {
    return { messages: [] };
  }
  const outputItems = get(response, "output");
  if (!Array.isArray(outputItems)) {
    return { messages: [] };
  }
  const messages: Any[] = [];
  for (const item of outputItems) {
    const msg = normalizeOutputItem(item);
    if (msg !== null) {
      messages.push(msg);
    }
  }
  return { messages };
}

// ---------------------------------------------------------------------------
// GenerationSpanData normalization
// ---------------------------------------------------------------------------

function normalizeGenerationMessage(msg: Any): Record<string, Any> {
  if (msg && typeof msg === "object" && !Array.isArray(msg)) {
    let content = msg.content;
    if (Array.isArray(content)) {
      content = collapseContent(content);
    }
    const result: Record<string, Any> = { role: msg.role ?? "user", content };
    if ("tool_calls" in msg) {
      result.tool_calls = msg.tool_calls;
    }
    if ("name" in msg) {
      result.name = msg.name;
    }
    return result;
  }
  return { role: "user", content: safeSerialize(msg) };
}

export function normalizeGenerationInput(spanData: Any): { messages: Any[] } {
  const raw = get(spanData, "input");
  if (!Array.isArray(raw)) {
    if (typeof raw === "string") {
      return { messages: [{ role: "user", content: raw }] };
    }
    return { messages: [] };
  }
  return { messages: raw.map((item) => normalizeGenerationMessage(item)) };
}

export function normalizeGenerationOutput(spanData: Any): { messages: Any[] } {
  const raw = get(spanData, "output");
  if (!Array.isArray(raw)) {
    if (typeof raw === "string") {
      return { messages: [{ role: "assistant", content: raw }] };
    }
    return { messages: [] };
  }
  const messages: Any[] = [];
  for (const item of raw) {
    const msg = normalizeGenerationMessage(item);
    if (!["assistant", "system", "tool"].includes(msg.role)) {
      msg.role = "assistant";
    }
    messages.push(msg);
  }
  return { messages };
}

// ---------------------------------------------------------------------------
// Reasoning extraction
// ---------------------------------------------------------------------------

/** Extract reasoning summary text from response.output reasoning items. */
export function extractReasoning(response: Any): string | null {
  const output = get(response, "output");
  if (!Array.isArray(output)) {
    return null;
  }
  const parts: string[] = [];
  for (const item of output) {
    if (!item || typeof item !== "object" || item.type !== "reasoning") {
      continue;
    }
    const summary = item.summary;
    if (Array.isArray(summary)) {
      for (const part of summary) {
        if (part && typeof part === "object" && part.type === "summary_text" && part.text) {
          parts.push(String(part.text));
        }
      }
    }
  }
  return parts.length > 0 ? parts.join("\n\n") : null;
}

// ---------------------------------------------------------------------------
// Token usage extraction
// ---------------------------------------------------------------------------

function safeInt(val: Any): number | undefined {
  if (val == null) {
    return undefined;
  }
  const n = typeof val === "number" ? val : Number.parseInt(String(val), 10);
  return Number.isNaN(n) ? undefined : Math.trunc(n);
}

/** Map Responses API / Chat Completions usage to the universal token-usage format. */
export function extractTokenUsage(usage: Any): Record<string, number> | null {
  if (!usage) {
    return null;
  }
  const g = (k: string): Any => (usage == null ? undefined : usage[k]);
  const result: Record<string, number> = {};

  let prompt = safeInt(g("input_tokens"));
  if (prompt === undefined) {
    prompt = safeInt(g("prompt_tokens"));
  }
  let completion = safeInt(g("output_tokens"));
  if (completion === undefined) {
    completion = safeInt(g("completion_tokens"));
  }
  const total = safeInt(g("total_tokens"));

  if (prompt !== undefined) {
    result.prompt_tokens = prompt;
  }
  if (completion !== undefined) {
    result.completion_tokens = completion;
  }
  if (total !== undefined) {
    result.total_tokens = total;
  } else if (prompt !== undefined && completion !== undefined) {
    result.total_tokens = prompt + completion;
  }

  const inputDetails = g("input_tokens_details");
  if (inputDetails) {
    const cached = safeInt(get(inputDetails, "cached_tokens"));
    if (cached !== undefined && cached > 0) {
      result.cache_read_tokens = cached;
    }
  }
  const outputDetails = g("output_tokens_details");
  if (outputDetails) {
    const reasoning = safeInt(get(outputDetails, "reasoning_tokens"));
    if (reasoning !== undefined && reasoning > 0) {
      result.reasoning_tokens = reasoning;
    }
  }

  const cachedFlat = safeInt(g("cached_prompt_tokens")) ?? safeInt(g("cache_read_tokens"));
  if (cachedFlat !== undefined && cachedFlat > 0 && result.cache_read_tokens === undefined) {
    result.cache_read_tokens = cachedFlat;
  }
  const reasoningFlat = safeInt(g("reasoning_tokens"));
  if (reasoningFlat !== undefined && reasoningFlat > 0 && result.reasoning_tokens === undefined) {
    result.reasoning_tokens = reasoningFlat;
  }

  return Object.keys(result).length > 0 ? result : null;
}

// ---------------------------------------------------------------------------
// Model parameter extraction
// ---------------------------------------------------------------------------

function pickModelParams(source: Any): Record<string, unknown> | null {
  if (!source) {
    return null;
  }
  const dict = configToDict(source);
  const params: Record<string, unknown> = {};
  for (const key of SAFE_MODEL_PARAM_KEYS) {
    if (dict[key] !== undefined && dict[key] !== null) {
      params[key] = safeSerialize(dict[key]);
    }
  }
  return Object.keys(params).length > 0 ? params : null;
}

export function extractResponseModelParameters(response: Any): Record<string, unknown> | null {
  return pickModelParams(response);
}

export function extractGenerationModelParameters(spanData: Any): Record<string, unknown> | null {
  return pickModelParams(get(spanData, "model_config"));
}

// ---------------------------------------------------------------------------
// Tool I/O serialization
// ---------------------------------------------------------------------------

/** Serialize tool input/output; JSON-parse string payloads, else safe-serialize. */
export function serializeToolIo(value: Any): Any {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value === "string") {
    try {
      return safeSerialize(JSON.parse(value));
    } catch {
      return value;
    }
  }
  return safeSerialize(value);
}

// ---------------------------------------------------------------------------
// Span kind + name resolution
// ---------------------------------------------------------------------------

const SPAN_TYPE_TO_KIND: Record<string, SpanKind> = {
  agent: SpanKind.AGENT,
  handoff: SpanKind.AGENT,
  response: SpanKind.LLM,
  generation: SpanKind.LLM,
  function: SpanKind.TOOL,
  guardrail: SpanKind.OTHER,
  custom: SpanKind.OTHER,
};

/** Map an OpenAI Agents span-data type to a PandaProbe SpanKind. */
export function mapSpanKind(type: string | undefined): SpanKind {
  return (type && SPAN_TYPE_TO_KIND[type]) || SpanKind.OTHER;
}

/** Derive a human-readable span name (span_data.name → span.name → Capitalized(type) → "Span"). */
export function resolveSpanName(span: Any): string {
  const spanData = span?.spanData ?? span?.span_data;
  const dataName = spanData?.name;
  if (dataName) {
    return String(dataName);
  }
  if (span?.name) {
    return String(span.name);
  }
  const type = spanData?.type;
  if (type) {
    const s = String(type);
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
  return "Span";
}
