/** Claude Agent SDK normalization helpers. */

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

/** Pull text and thinking from an assistant message's content blocks. */
export function splitAssistantContent(content: Any): { text: string; thinking: string } {
  const textParts: string[] = [];
  const thinkingParts: string[] = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!block || typeof block !== "object") {
        continue;
      }
      if (block.type === "text" && typeof block.text === "string") {
        textParts.push(block.text);
      } else if (block.type === "thinking" && typeof block.thinking === "string") {
        thinkingParts.push(block.thinking);
      }
    }
  } else if (typeof content === "string") {
    textParts.push(content);
  }
  return { text: textParts.join(""), thinking: thinkingParts.join("\n\n") };
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
export function extractToolResults(content: Any): Array<{ toolUseId: string; content: unknown }> {
  const results: Array<{ toolUseId: string; content: unknown }> = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block && typeof block === "object" && block.type === "tool_result") {
        results.push({ toolUseId: String(block.tool_use_id ?? ""), content: block.content });
      }
    }
  }
  return results;
}

/** Map Claude usage fields to PandaProbe token usage. */
export function extractClaudeUsage(usage: Any): Record<string, number> | null {
  if (!usage || typeof usage !== "object") {
    return null;
  }
  const out: Record<string, number> = {};
  if (typeof usage.input_tokens === "number") {
    out.prompt_tokens = usage.input_tokens;
  }
  if (typeof usage.output_tokens === "number") {
    out.completion_tokens = usage.output_tokens;
  }
  if (typeof usage.cache_read_input_tokens === "number" && usage.cache_read_input_tokens > 0) {
    out.cache_read_tokens = usage.cache_read_input_tokens;
  }
  if (typeof usage.cache_creation_input_tokens === "number" && usage.cache_creation_input_tokens > 0) {
    out.cache_creation_tokens = usage.cache_creation_input_tokens;
  }
  return Object.keys(out).length > 0 ? out : null;
}
