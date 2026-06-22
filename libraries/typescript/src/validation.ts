/**
 * Shared validation and utility functions for the standard trace/span schema.
 *
 * The standard schema requires `input` and `output` to be an object with a
 * `messages` key whose value is a list of message objects. Each message must
 * have at least `role` (string) and `content` (string, array, or null).
 */

import { logger } from "./logger.js";

const SCHEMA_EXAMPLE = '{"messages": [{"role": "user", "content": "hello"}]}';

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Validate that *data* follows the standard messages schema.
 *
 * Rules:
 * - `null`/`undefined` is accepted silently (no-op).
 * - Otherwise *data* must be an object with a `messages` key whose value is an
 *   array of objects, each containing `role` (string) and `content`
 *   (string, array, or null). An array content is used by the Responses API.
 *
 * Throws `Error` with a clear message on failure.
 */
export function validateMessagesFormat(data: unknown, label: string): void {
  if (data === null || data === undefined) {
    return;
  }

  if (!isPlainObject(data)) {
    throw new Error(
      `${label} must be an object with a 'messages' key, got ${typeName(data)}. Expected format: ${SCHEMA_EXAMPLE}`,
    );
  }

  if (!("messages" in data)) {
    throw new Error(
      `${label} object must contain a 'messages' key. Got keys: ${JSON.stringify(
        Object.keys(data),
      )}. Expected format: ${SCHEMA_EXAMPLE}`,
    );
  }

  const messages = data.messages;
  if (!Array.isArray(messages)) {
    throw new Error(
      `${label}['messages'] must be an array, got ${typeName(messages)}. Expected format: ${SCHEMA_EXAMPLE}`,
    );
  }

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!isPlainObject(msg)) {
      throw new Error(
        `${label}['messages'][${i}] must be an object, got ${typeName(
          msg,
        )}. Each message must have at least 'role' and 'content' keys.`,
      );
    }
    if (!("role" in msg)) {
      throw new Error(
        `${label}['messages'][${i}] is missing required key 'role'. Got keys: ${JSON.stringify(Object.keys(msg))}.`,
      );
    }
    if (typeof msg.role !== "string") {
      throw new Error(`${label}['messages'][${i}]['role'] must be a string, got ${typeName(msg.role)}.`);
    }
    if (!("content" in msg)) {
      throw new Error(
        `${label}['messages'][${i}] is missing required key 'content'. Got keys: ${JSON.stringify(Object.keys(msg))}.`,
      );
    }
    const content = msg.content;
    if (content !== null && content !== undefined && typeof content !== "string" && !Array.isArray(content)) {
      throw new Error(
        `${label}['messages'][${i}]['content'] must be a string, array, or null, got ${typeName(content)}.`,
      );
    }
  }
}

/**
 * Log a warning if *data* doesn't follow the messages schema.
 *
 * Unlike `validateMessagesFormat` this **never throws** — it is designed for
 * the context layer where an observability SDK must not crash user code.
 */
export function warnIfInvalidMessages(data: unknown, label: string): void {
  if (data === null || data === undefined) {
    return;
  }
  try {
    validateMessagesFormat(data, label);
  } catch (exc) {
    logger.warning(
      `${label} does not follow the messages schema and will be stored as-is. Details: ${
        (exc as Error).message
      }  Expected format: ${SCHEMA_EXAMPLE}`,
    );
  }
}

// ---------------------------------------------------------------------------
// Extraction utilities
// ---------------------------------------------------------------------------

/**
 * Extract only the last user message from a messages structure.
 *
 * Returns `{messages: [lastUserMessage]}` if a user/human message is found.
 * Returns *inputData* unchanged when the structure doesn't match.
 */
export function extractLastUserMessage(inputData: unknown): unknown {
  if (!isPlainObject(inputData)) {
    return inputData;
  }
  const messages = inputData.messages;
  if (!Array.isArray(messages)) {
    return inputData;
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (isPlainObject(msg) && (msg.role === "user" || msg.role === "human")) {
      return { messages: [msg] };
    }
  }
  return inputData;
}

/**
 * Extract only the last assistant message from a messages structure.
 *
 * Returns `{messages: [lastAssistantMessage]}` if an assistant/ai message is
 * found. Returns *outputData* unchanged when the structure doesn't match.
 */
export function extractLastAssistantMessage(outputData: unknown): unknown {
  if (!isPlainObject(outputData)) {
    return outputData;
  }
  const messages = outputData.messages;
  if (!Array.isArray(messages)) {
    return outputData;
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (isPlainObject(msg) && (msg.role === "assistant" || msg.role === "ai")) {
      return { messages: [msg] };
    }
  }
  return outputData;
}
