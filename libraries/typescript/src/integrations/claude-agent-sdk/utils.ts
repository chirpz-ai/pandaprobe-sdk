/** Claude Agent SDK normalization and extraction utilities (ports claude_agent_sdk/utils.py). */

import { SAFE_MODEL_PARAM_KEYS, configToDict, safeSerialize } from "../base.js";

type Any = any;

/** Extract the user prompt text from a query() argument. */
export function extractPromptText(prompt: Any): string {
  if (typeof prompt === "string") {
    return prompt;
  }
  if (prompt && typeof prompt === "object" && typeof prompt.content === "string") {
    return prompt.content;
  }
  return "";
}

/** Extract a string system prompt from ClaudeAgentOptions (ignores preset objects). */
export function extractSystemPrompt(options: Any): string | null {
  const sys = options?.systemPrompt ?? options?.system_prompt;
  if (typeof sys === "string" && sys) {
    return sys;
  }
  return null;
}

/** Visible text from content blocks, excluding thinking. */
export function normalizeContentToText(content: Any): string | null {
  if (content == null) {
    return null;
  }
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (typeof block === "string") {
        parts.push(block);
      } else if (block && typeof block === "object") {
        if (block.type === "thinking") {
          continue;
        }
        if (block.type === "text") {
          parts.push(String(block.text ?? ""));
        }
      }
    }
    return parts.length > 0 ? parts.join(" ") : null;
  }
  return String(content);
}

/** Concatenated thinking text from content blocks, or null. */
export function extractThinkingFromContent(content: Any): string | null {
  if (!Array.isArray(content)) {
    return null;
  }
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object" && block.type === "thinking") {
      const text = block.thinking ?? block.text;
      if (text) {
        parts.push(String(text));
      }
    }
  }
  return parts.length > 0 ? parts.join("\n\n") : null;
}

/** True when content has thinking blocks but no text or tool_use (SDK emits a thinking-only turn). */
export function isThinkingOnly(content: Any): boolean {
  if (!Array.isArray(content) || content.length === 0) {
    return false;
  }
  let hasThinking = false;
  for (const block of content) {
    if (!block || typeof block !== "object") {
      return false;
    }
    if (block.type === "thinking") {
      hasThinking = true;
    } else {
      return false; // text / tool_use / other → not thinking-only
    }
  }
  return hasThinking;
}

/** Collect tool_use blocks from an assistant message. */
export function extractToolUses(content: Any): Array<{ id: string; name: string; input: unknown }> {
  const uses: Array<{ id: string; name: string; input: unknown }> = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block && typeof block === "object" && block.type === "tool_use") {
        uses.push({ id: String(block.id ?? ""), name: String(block.name ?? "tool"), input: block.input });
      }
    }
  }
  return uses;
}

/** Collect tool_result blocks from a user message. */
export function extractToolResults(content: Any): Array<{ toolUseId: string; content: unknown; isError: boolean }> {
  const results: Array<{ toolUseId: string; content: unknown; isError: boolean }> = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block && typeof block === "object" && block.type === "tool_result") {
        results.push({
          toolUseId: String(block.tool_use_id ?? ""),
          content: block.content,
          isError: Boolean(block.is_error),
        });
      }
    }
  }
  return results;
}

/** Map Claude Agent SDK ResultMessage.usage to the universal token-usage format. */
export function extractTokenUsage(usage: Any): Record<string, number> | null {
  if (!usage) {
    return null;
  }
  const g = (k: string): Any => (usage == null ? undefined : usage[k]);
  const result: Record<string, number> = {};
  const input = g("input_tokens");
  const output = g("output_tokens");
  const cacheRead = g("cache_read_input_tokens");
  const cacheCreation = g("cache_creation_input_tokens");
  if (input != null) {
    result.prompt_tokens = Math.trunc(Number(input));
  }
  if (output != null) {
    result.completion_tokens = Math.trunc(Number(output));
  }
  if (input != null && output != null) {
    result.total_tokens = Math.trunc(Number(input)) + Math.trunc(Number(output));
  }
  if (cacheRead != null && Math.trunc(Number(cacheRead)) > 0) {
    result.cache_read_tokens = Math.trunc(Number(cacheRead));
  }
  if (cacheCreation != null && Math.trunc(Number(cacheCreation)) > 0) {
    result.cache_creation_tokens = Math.trunc(Number(cacheCreation));
  }
  return Object.keys(result).length > 0 ? result : null;
}

/** Extract safe model parameters from ClaudeAgentOptions (captures thinking as thinking_config). */
export function extractModelParameters(options: Any): Record<string, unknown> | null {
  if (!options) {
    return null;
  }
  const dict = configToDict(options);
  const params: Record<string, unknown> = {};
  for (const key of SAFE_MODEL_PARAM_KEYS) {
    if (key === "thinking") {
      continue;
    }
    if (dict[key] !== undefined && dict[key] !== null) {
      params[key] = safeSerialize(dict[key]);
    }
  }
  if (dict.thinking) {
    params.thinking_config = safeSerialize(dict.thinking);
  }
  return Object.keys(params).length > 0 ? params : null;
}

/** Serialize a tool response to a string for span output (JSON for objects/arrays). */
export function serializeToolResponse(resp: Any): string {
  if (resp && typeof resp === "object") {
    return JSON.stringify(resp);
  }
  if (resp === null || resp === undefined) {
    return "";
  }
  return String(resp);
}
