/** Helper utilities shared by every LangChain-core-based integration. */

import { SAFE_MODEL_PARAM_KEYS, configToDict } from "../base.js";

type Any = any;

const ROLE_MAP: Record<string, string> = {
  human: "user",
  ai: "assistant",
  HumanMessage: "user",
  AIMessage: "assistant",
  SystemMessage: "system",
  ToolMessage: "tool",
};

export function normalizeRole(role: string): string {
  return ROLE_MAP[role] ?? role;
}

function isObject(v: unknown): v is Record<string, Any> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Strip thinking blocks and collapse text-only content lists to a string. */
export function normalizeContentBlocks(content: Any): Any {
  if (!Array.isArray(content)) {
    return content;
  }
  const filtered = content.filter((block) => !(isObject(block) && block.type === "thinking"));
  if (filtered.length === 0) {
    return content;
  }
  const textParts: string[] = [];
  for (const block of filtered) {
    if (typeof block === "string") {
      textParts.push(block);
    } else if (isObject(block) && block.type === "text" && "text" in block) {
      textParts.push(String(block.text));
    } else {
      return filtered;
    }
  }
  if (textParts.length === 0) {
    return filtered;
  }
  return textParts.length > 1 ? textParts.join(" ") : textParts[0];
}

/** Extract whitelisted model parameters, dropping null values and secrets. */
export function extractModelParameters(invocationParams: unknown): Record<string, unknown> | null {
  if (!invocationParams) {
    return null;
  }
  const dict = configToDict(invocationParams);
  const params: Record<string, unknown> = {};
  for (const key of SAFE_MODEL_PARAM_KEYS) {
    if (dict[key] !== undefined && dict[key] !== null) {
      params[key] = dict[key];
    }
  }
  if (dict.thinking_config) {
    params.thinking_config = dict.thinking_config;
  }
  return Object.keys(params).length > 0 ? params : null;
}

/** Extract + normalize token usage from a LangChain LLM result. */
export function extractTokenUsage(response: Any): Record<string, number> | null {
  try {
    const gen = response?.generations?.[0]?.[0];
    const msg = gen?.message;
    const meta = msg?.usage_metadata ?? msg?.usageMetadata;
    if (meta) {
      const md = configToDict(meta);
      const usage: Record<string, number> = {};
      if (md.input_tokens != null) {
        usage.prompt_tokens = Number(md.input_tokens);
      }
      if (md.total_tokens != null) {
        usage.total_tokens = Number(md.total_tokens);
      }
      const outputDetails = md.output_token_details as Any;
      const reasoning =
        isObject(outputDetails) && outputDetails.reasoning != null ? Number(outputDetails.reasoning) : 0;
      if (reasoning) {
        usage.reasoning_tokens = reasoning;
      }
      if (md.output_tokens != null) {
        usage.completion_tokens = Math.max(0, Number(md.output_tokens) - reasoning);
      }
      const inputDetails = md.input_token_details as Any;
      if (isObject(inputDetails) && inputDetails.cache_read != null) {
        const cache = Number(inputDetails.cache_read);
        if (cache) {
          usage.cache_read_tokens = cache;
        }
      }
      if (Object.keys(usage).length > 0) {
        return usage;
      }
    }
  } catch {
    // fall through to legacy path
  }

  try {
    const llmOutput = response?.llmOutput ?? response?.llm_output;
    const legacy = isObject(llmOutput) ? (llmOutput.tokenUsage ?? llmOutput.token_usage) : null;
    if (isObject(legacy)) {
      const usage: Record<string, number> = {};
      if (legacy.promptTokens ?? legacy.prompt_tokens) {
        usage.prompt_tokens = Number(legacy.promptTokens ?? legacy.prompt_tokens);
      }
      if (legacy.completionTokens ?? legacy.completion_tokens) {
        usage.completion_tokens = Number(legacy.completionTokens ?? legacy.completion_tokens);
      }
      if (legacy.totalTokens ?? legacy.total_tokens) {
        usage.total_tokens = Number(legacy.totalTokens ?? legacy.total_tokens);
      }
      return Object.keys(usage).length > 0 ? usage : null;
    }
  } catch {
    // ignore
  }
  return null;
}

/** Extract a human-readable name from a LangChain serialized object or run name. */
export function extractName(serialized: Any, fallback = "unknown"): string {
  if (!serialized) {
    return fallback;
  }
  if (serialized.name) {
    return String(serialized.name);
  }
  if (Array.isArray(serialized.id) && serialized.id.length > 0) {
    return String(serialized.id[serialized.id.length - 1]);
  }
  return fallback;
}

/** Detect a LangChain message object (BaseMessage instance). */
function asLangChainMessage(value: Any): { role: string; content: unknown } | null {
  if (!isObject(value)) {
    return null;
  }
  let type: string | undefined;
  if (typeof value._getType === "function") {
    try {
      type = value._getType();
    } catch {
      type = undefined;
    }
  }
  type = type ?? value.type ?? value.role;
  if (type && "content" in value) {
    return { role: normalizeRole(String(type)), content: value.content };
  }
  return null;
}

/** Ensure a callback value is JSON-serializable; handles LangChain messages/documents. */
export function safeOutput(value: Any): Any {
  if (value === null || value === undefined) {
    return value;
  }
  const t = typeof value;
  if (t === "string" || t === "number" || t === "boolean") {
    return value;
  }
  if (t === "bigint") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map((v) => safeOutput(v));
  }
  const lcMessage = asLangChainMessage(value);
  if (lcMessage) {
    return { role: lcMessage.role, content: normalizeContentBlocks(lcMessage.content) };
  }
  if (value.pageContent !== undefined) {
    return { page_content: value.pageContent, metadata: value.metadata ?? {} };
  }
  if (typeof value.toJSON === "function") {
    try {
      return safeOutput(value.toJSON());
    } catch {
      // fall through
    }
  }
  if (isObject(value)) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      if (!k.startsWith("_")) {
        out[k] = safeOutput(v);
      }
    }
    return out;
  }
  return String(value);
}

/** Recursively rename `type` to `role`, normalize roles, strip thinking blocks. */
export function normalizeTypeToRole(data: Any): Any {
  if (Array.isArray(data)) {
    return data.map((item) => normalizeTypeToRole(item));
  }
  if (isObject(data)) {
    const hasContent = "content" in data;
    if ("type" in data && hasContent) {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(data)) {
        if (k === "type") {
          out.role = normalizeRole(String(v));
        } else if (k === "content") {
          out.content = normalizeContentBlocks(normalizeTypeToRole(v));
        } else {
          out[k] = normalizeTypeToRole(v);
        }
      }
      return out;
    }
    if ("role" in data && hasContent && typeof data.role === "string") {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(data)) {
        if (k === "role") {
          out.role = normalizeRole(String(v));
        } else if (k === "content") {
          out.content = normalizeContentBlocks(normalizeTypeToRole(v));
        } else {
          out[k] = normalizeTypeToRole(v);
        }
      }
      return out;
    }
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(data)) {
      out[k] = normalizeTypeToRole(v);
    }
    return out;
  }
  return data;
}

/** Convert LangChain input formats to standard role/content dicts. */
export function normalizeLangchainInput(inputs: Any): Any {
  if (!isObject(inputs) || !("messages" in inputs)) {
    return inputs;
  }
  const messages = inputs.messages;
  if (!Array.isArray(messages)) {
    return inputs;
  }
  const normalized: unknown[] = [];
  for (const item of messages) {
    if (Array.isArray(item) && item.length >= 2) {
      normalized.push({ role: normalizeRole(String(item[0])), content: normalizeContentBlocks(item[1]) });
    } else if (isObject(item) && "type" in item && "content" in item) {
      const newItem: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(item)) {
        if (k !== "type") {
          newItem[k] = v;
        }
      }
      newItem.role = normalizeRole(String(item.type));
      newItem.content = normalizeContentBlocks(newItem.content);
      normalized.push(newItem);
    } else if (isObject(item) && typeof item.role === "string") {
      const newItem = { ...item };
      newItem.role = normalizeRole(item.role);
      if ("content" in newItem) {
        newItem.content = normalizeContentBlocks(newItem.content);
      }
      normalized.push(newItem);
    } else {
      normalized.push(item);
    }
  }
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(inputs)) {
    if (k !== "messages") {
      result[k] = v;
    }
  }
  result.messages = normalized;
  return result;
}

/** Extract only the last message from outputs, renaming `type` to `role`. */
export function normalizeLangchainOutput(outputs: Any): Any {
  if (!isObject(outputs) || !("messages" in outputs)) {
    return outputs;
  }
  const messages = outputs.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    return outputs;
  }
  let last = messages[messages.length - 1];
  if (isObject(last)) {
    if ("type" in last) {
      const copy: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(last)) {
        if (k !== "type") {
          copy[k] = v;
        }
      }
      copy.role = normalizeRole(String(last.type));
      last = copy;
    } else if (typeof last.role === "string") {
      last = { ...last, role: normalizeRole(last.role) };
    }
    if ("content" in last) {
      last.content = normalizeContentBlocks(last.content);
    }
  }
  return { messages: [last] };
}

/** Extract reasoning/thinking text from a LangChain LLM response. */
export function extractReasoningFromGeneration(response: Any): string | null {
  if (!response?.generations) {
    return null;
  }
  try {
    const gen = response.generations[0]?.[0];
    const msg = gen?.message;
    const content = msg?.content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        if (isObject(block) && block.type === "thinking") {
          const text = block.thinking ?? block.text;
          if (text) {
            parts.push(String(text));
          }
        }
      }
      if (parts.length > 0) {
        return parts.join("\n\n");
      }
    }
    const additional = msg?.additional_kwargs ?? msg?.additionalKwargs ?? {};
    const reasoning = additional.reasoning_content ?? additional.reasoning;
    if (typeof reasoning === "string" && reasoning) {
      return reasoning;
    }
  } catch {
    // ignore
  }
  return null;
}

/** Extract the message from `response.generations` into `{messages: [...]}`. */
export function normalizeLlmGenerationOutput(response: Any): Any {
  if (!response?.generations) {
    return null;
  }
  try {
    const gen = response.generations[0]?.[0];
    if (gen?.message !== undefined) {
      const serialized = safeOutput(gen.message);
      return { messages: [normalizeTypeToRole(serialized)] };
    }
    return { messages: [{ role: "assistant", content: gen?.text ?? "" }] };
  } catch {
    return safeOutput(response.generations);
  }
}
