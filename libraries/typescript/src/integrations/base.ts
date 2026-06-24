/** Base utilities and adapter class shared by all framework integrations. */

import { type Client, getClient } from "../client.js";
import { safeSerialize } from "../util.js";

// Re-export shared serialization (single source of truth in ../util.js) so
// integration sub-packages keep importing it from `../base.js`.
export { safeSerialize };

// ---------------------------------------------------------------------------
// Shared model-parameter utilities
// ---------------------------------------------------------------------------

export const SAFE_MODEL_PARAM_KEYS: Set<string> = new Set([
  "temperature",
  "top_p",
  "top_k",
  "seed",
  "n",
  "candidate_count",
  "max_tokens",
  "max_output_tokens",
  "max_completion_tokens",
  "frequency_penalty",
  "presence_penalty",
  "stop",
  "stop_sequences",
  "response_format",
  "response_modalities",
  "response_mime_type",
  "reasoning_effort",
  "reasoning",
  "thinking",
  "thinking_level",
  "thinking_budget",
  "stream_options",
  "service_tier",
  "truncation",
]);

/** Convert a config object or plain dict to a dict, dropping null/undefined values. */
export function configToDict(config: unknown): Record<string, unknown> {
  if (config === null || config === undefined || typeof config !== "object") {
    return {};
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(config as Record<string, unknown>)) {
    if (v !== null && v !== undefined && !k.startsWith("_")) {
      out[k] = v;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Base adapter class
// ---------------------------------------------------------------------------

export interface AdapterOptions {
  client?: Client | null;
  sessionId?: string | null;
  userId?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export class BaseIntegrationAdapter {
  protected client: Client | null;
  protected sessionId: string | null;
  protected userId: string | null;
  protected tags: string[];
  protected metadata: Record<string, unknown>;

  constructor(options: AdapterOptions = {}) {
    this.client = options.client ?? null;
    this.sessionId = options.sessionId ?? null;
    this.userId = options.userId ?? null;
    this.tags = options.tags ?? [];
    this.metadata = options.metadata ?? {};
  }

  /** Return the explicit client or fall back to the global singleton. */
  protected resolveClient(): Client {
    if (this.client !== null) {
      return this.client;
    }
    const client = getClient();
    if (client === null) {
      throw new Error(
        "No PandaProbe client available. Set PANDAPROBE_API_KEY and PANDAPROBE_PROJECT_NAME, " +
          "call init() first, or pass { client } to the adapter.",
      );
    }
    return client;
  }
}
