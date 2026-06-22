/** OpenAI Agents SDK normalization helpers. */

import { SpanKind } from "../../schemas.js";
import { safeSerialize } from "../base.js";

type Any = any;

/** Map an OpenAI Agents span-data type to a PandaProbe SpanKind. */
export function mapSpanKind(type: string | undefined): SpanKind {
  switch (type) {
    case "agent":
      return SpanKind.AGENT;
    case "function":
    case "mcp_tools":
    case "guardrail":
      return SpanKind.TOOL;
    case "generation":
    case "response":
      return SpanKind.LLM;
    case "handoff":
      return SpanKind.CHAIN;
    default:
      return SpanKind.OTHER;
  }
}

/** Best-effort extraction of input/output from an Agents spanData object. */
export function extractSpanIo(spanData: Any): { input?: unknown; output?: unknown; model?: string | null } {
  if (!spanData || typeof spanData !== "object") {
    return {};
  }
  const out: { input?: unknown; output?: unknown; model?: string | null } = {};
  if (spanData.input !== undefined) {
    out.input = safeSerialize(spanData.input);
  }
  if (spanData.output !== undefined) {
    out.output = safeSerialize(spanData.output);
  }
  if (typeof spanData.model === "string") {
    out.model = spanData.model;
  }
  return out;
}

/** Map Agents usage to PandaProbe token usage. */
export function extractAgentsUsage(usage: Any): Record<string, number> | null {
  if (!usage || typeof usage !== "object") {
    return null;
  }
  const out: Record<string, number> = {};
  const prompt = usage.inputTokens ?? usage.input_tokens ?? usage.promptTokens ?? usage.prompt_tokens;
  const completion = usage.outputTokens ?? usage.output_tokens ?? usage.completionTokens ?? usage.completion_tokens;
  const total = usage.totalTokens ?? usage.total_tokens;
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
