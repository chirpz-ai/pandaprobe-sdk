/** Gemini-specific utilities for the wrapGemini instrumentation (@google/genai). */

import type { SpanContext } from "../../tracing/span.js";
import { openLlmSpan, safeSerialize } from "../base.js";

// The JS @google/genai SDK uses camelCase config keys.
export const SAFE_GEMINI_PARAMS = new Set([
  "temperature",
  "topP",
  "topK",
  "maxOutputTokens",
  "stopSequences",
  "candidateCount",
  "presencePenalty",
  "frequencyPenalty",
]);

type Any = any;

export function extractGeminiParams(kwargs: Record<string, unknown>): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  const config = kwargs.config;
  if (config && typeof config === "object") {
    for (const [k, v] of Object.entries(config as Record<string, unknown>)) {
      if (SAFE_GEMINI_PARAMS.has(k) && v !== undefined) {
        params[k] = safeSerialize(v);
      }
    }
  }
  for (const [k, v] of Object.entries(kwargs)) {
    if (SAFE_GEMINI_PARAMS.has(k) && v !== undefined) {
      params[k] = safeSerialize(v);
    }
  }
  return params;
}

function normalizeRole(role: string): string {
  return role === "model" ? "assistant" : role;
}

function toRecord(obj: unknown): Any {
  if (obj && typeof obj === "object") {
    return obj;
  }
  return obj;
}

function extractTextFromParts(parts: unknown): string {
  if (!Array.isArray(parts)) {
    return "";
  }
  const texts: string[] = [];
  for (const part of parts) {
    if (typeof part === "string") {
      texts.push(part);
    } else if (part && typeof part === "object" && typeof (part as Any).text === "string") {
      if ((part as Any).text) {
        texts.push((part as Any).text);
      }
    }
  }
  return texts.join("\n");
}

/** Convert Gemini `contents` + config system instruction into standard messages. */
export function normalizeGeminiInput(kwargs: Record<string, unknown>): { messages: unknown[] } {
  const messages: unknown[] = [];

  const config = kwargs.config as Any;
  if (config && typeof config === "object") {
    const sysInst = config.systemInstruction ?? config.system_instruction;
    if (sysInst) {
      if (typeof sysInst === "string") {
        messages.push({ role: "system", content: sysInst });
      } else if (sysInst.parts) {
        const text = extractTextFromParts(sysInst.parts);
        if (text) {
          messages.push({ role: "system", content: text });
        }
      } else if (Array.isArray(sysInst)) {
        const text = extractTextFromParts(sysInst);
        if (text) {
          messages.push({ role: "system", content: text });
        }
      }
    }
  }

  const contents = kwargs.contents as Any;
  if (contents == null) {
    return { messages };
  }
  if (typeof contents === "string") {
    messages.push({ role: "user", content: contents });
    return { messages };
  }
  if (Array.isArray(contents)) {
    if (contents.every((item) => typeof item === "string")) {
      for (const item of contents) {
        messages.push({ role: "user", content: item });
      }
      return { messages };
    }
    for (const c of contents) {
      const content = toRecord(c);
      if (!content || typeof content !== "object") {
        continue;
      }
      const role = normalizeRole(content.role ?? "user");
      const text = extractTextFromParts(content.parts ?? []);
      messages.push({ role, content: text || "" });
    }
  }
  return { messages };
}

/** Open an LLM span for a Gemini API call. */
export function enterGeminiSpan(kwargs: Record<string, unknown>, methodName = "gemini-generate"): SpanContext | null {
  return openLlmSpan({
    methodName,
    inputData: normalizeGeminiInput(kwargs),
    modelParams: extractGeminiParams(kwargs),
    model: (kwargs.model as string | undefined) ?? null,
  });
}
