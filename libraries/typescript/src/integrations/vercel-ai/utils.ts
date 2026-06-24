/** Vercel AI SDK normalization helpers. */

import { safeSerialize } from "../base.js";

type Any = any;

/** Flatten an AI SDK message `content` (string or array of parts) to text or structured. */
function normalizeContent(content: Any): Any {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const textParts: string[] = [];
    let allText = true;
    for (const part of content) {
      if (part && typeof part === "object" && part.type === "text" && typeof part.text === "string") {
        textParts.push(part.text);
      } else {
        allText = false;
        break;
      }
    }
    if (allText && textParts.length > 0) {
      return textParts.join("");
    }
    return safeSerialize(content);
  }
  return safeSerialize(content);
}

/** Convert AI SDK call params (`params.prompt`) to the standard messages schema. */
export function normalizeVercelInput(params: Any): { messages: unknown[] } {
  const messages: unknown[] = [];
  const prompt = params?.prompt;
  if (typeof prompt === "string") {
    messages.push({ role: "user", content: prompt });
    return { messages };
  }
  if (Array.isArray(prompt)) {
    for (const msg of prompt) {
      if (msg && typeof msg === "object") {
        messages.push({ role: msg.role ?? "user", content: normalizeContent(msg.content) });
      }
    }
  }
  return { messages };
}

/** Extract assistant text from an AI SDK generate result (v4 `.text` or v5 content parts). */
export function extractVercelText(result: Any): string | null {
  if (typeof result?.text === "string") {
    return result.text;
  }
  if (Array.isArray(result?.content)) {
    const parts = result.content
      .filter((p: Any) => p && p.type === "text" && typeof p.text === "string")
      .map((p: Any) => p.text as string);
    if (parts.length > 0) {
      return parts.join("");
    }
  }
  return null;
}

/** Map AI SDK usage (`{inputTokens, outputTokens}` or `{promptTokens, completionTokens}`). */
export function extractVercelUsage(usage: Any): Record<string, number> | null {
  if (!usage || typeof usage !== "object") {
    return null;
  }
  const prompt = usage.inputTokens ?? usage.promptTokens;
  const completion = usage.outputTokens ?? usage.completionTokens;
  const total = usage.totalTokens;
  const out: Record<string, number> = {};
  if (typeof prompt === "number") {
    out.prompt_tokens = prompt;
  }
  if (typeof completion === "number") {
    out.completion_tokens = completion;
  }
  if (typeof total === "number") {
    out.total_tokens = total;
  }
  return Object.keys(out).length > 0 ? out : null;
}
