/**
 * Schema types and serializers matching the PandaProbe backend API.
 *
 * Mirrors the Python `schemas.py` Pydantic models and their `to_api_dict()`
 * serialization (omit null/empty fields, stringify UUIDs, ISO-8601 datetimes,
 * enum string values).
 */

// ---------------------------------------------------------------------------
// Enums (string-literal unions backed by const objects)
// ---------------------------------------------------------------------------

export const SpanKind = {
  AGENT: "AGENT",
  TOOL: "TOOL",
  LLM: "LLM",
  RETRIEVER: "RETRIEVER",
  CHAIN: "CHAIN",
  EMBEDDING: "EMBEDDING",
  OTHER: "OTHER",
} as const;
export type SpanKind = (typeof SpanKind)[keyof typeof SpanKind];

export const SpanStatusCode = {
  UNSET: "UNSET",
  OK: "OK",
  ERROR: "ERROR",
} as const;
export type SpanStatusCode = (typeof SpanStatusCode)[keyof typeof SpanStatusCode];

export const TraceStatus = {
  PENDING: "PENDING",
  RUNNING: "RUNNING",
  COMPLETED: "COMPLETED",
  ERROR: "ERROR",
} as const;
export type TraceStatus = (typeof TraceStatus)[keyof typeof TraceStatus];

export const ScoreDataType = {
  NUMERIC: "NUMERIC",
  BOOLEAN: "BOOLEAN",
  CATEGORICAL: "CATEGORICAL",
} as const;
export type ScoreDataType = (typeof ScoreDataType)[keyof typeof ScoreDataType];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a Date as an ISO-8601 string (UTC). */
export function iso(dt: Date): string {
  return dt.toISOString();
}

/** Ensure a value is JSON-serializable; fall back to String(value). */
export function safeJson(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }
  const t = typeof value;
  if (t === "string" || t === "number" || t === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((v) => safeJson(v));
  }
  if (t === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[String(k)] = safeJson(v);
    }
    return out;
  }
  try {
    JSON.stringify(value);
    return value;
  } catch {
    return String(value);
  }
}

/** Generate a UUID v4 string. */
export function newUuid(): string {
  return crypto.randomUUID();
}

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

export interface SpanDataInit {
  spanId?: string;
  parentSpanId?: string | null;
  name: string;
  kind?: SpanKind;
  status?: SpanStatusCode;
  input?: unknown;
  output?: unknown;
  model?: string | null;
  tokenUsage?: Record<string, number> | null;
  metadata?: Record<string, unknown>;
  startedAt: Date;
  endedAt?: Date | null;
  error?: string | null;
  completionStartTime?: Date | null;
  modelParameters?: Record<string, unknown> | null;
  cost?: Record<string, number> | null;
}

export class SpanData {
  spanId: string;
  parentSpanId: string | null;
  name: string;
  kind: SpanKind;
  status: SpanStatusCode;
  input: unknown;
  output: unknown;
  model: string | null;
  tokenUsage: Record<string, number> | null;
  metadata: Record<string, unknown>;
  startedAt: Date;
  endedAt: Date | null;
  error: string | null;
  completionStartTime: Date | null;
  modelParameters: Record<string, unknown> | null;
  cost: Record<string, number> | null;

  constructor(init: SpanDataInit) {
    this.spanId = init.spanId ?? newUuid();
    this.parentSpanId = init.parentSpanId ?? null;
    this.name = init.name;
    this.kind = init.kind ?? SpanKind.OTHER;
    this.status = init.status ?? SpanStatusCode.UNSET;
    this.input = init.input ?? null;
    this.output = init.output ?? null;
    this.model = init.model ?? null;
    this.tokenUsage = init.tokenUsage ?? null;
    this.metadata = init.metadata ?? {};
    this.startedAt = init.startedAt;
    this.endedAt = init.endedAt ?? null;
    this.error = init.error ?? null;
    this.completionStartTime = init.completionStartTime ?? null;
    this.modelParameters = init.modelParameters ?? null;
    this.cost = init.cost ?? null;
  }

  /** Serialize to a dict matching the backend POST /traces span schema. */
  toApiDict(): Record<string, unknown> {
    const d: Record<string, unknown> = {};
    d.span_id = this.spanId;
    if (this.parentSpanId !== null) {
      d.parent_span_id = this.parentSpanId;
    }
    d.name = this.name;
    d.kind = this.kind;
    d.status = this.status;
    if (this.input !== null && this.input !== undefined) {
      d.input = safeJson(this.input);
    }
    if (this.output !== null && this.output !== undefined) {
      d.output = safeJson(this.output);
    }
    if (this.model !== null) {
      d.model = this.model;
    }
    if (this.tokenUsage !== null) {
      d.token_usage = this.tokenUsage;
    }
    if (this.metadata && Object.keys(this.metadata).length > 0) {
      d.metadata = this.metadata;
    }
    d.started_at = iso(this.startedAt);
    if (this.endedAt !== null) {
      d.ended_at = iso(this.endedAt);
    }
    if (this.error !== null) {
      d.error = this.error;
    }
    if (this.completionStartTime !== null) {
      d.completion_start_time = iso(this.completionStartTime);
    }
    if (this.modelParameters !== null) {
      d.model_parameters = this.modelParameters;
    }
    if (this.cost !== null) {
      d.cost = this.cost;
    }
    return d;
  }
}

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

export interface TraceDataInit {
  traceId?: string;
  name: string;
  status?: TraceStatus;
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown>;
  startedAt: Date;
  endedAt?: Date | null;
  sessionId?: string | null;
  userId?: string | null;
  tags?: string[];
  environment?: string | null;
  release?: string | null;
  spans?: SpanData[];
}

export class TraceData {
  traceId: string;
  name: string;
  status: TraceStatus;
  input: unknown;
  output: unknown;
  metadata: Record<string, unknown>;
  startedAt: Date;
  endedAt: Date | null;
  sessionId: string | null;
  userId: string | null;
  tags: string[];
  environment: string | null;
  release: string | null;
  spans: SpanData[];

  constructor(init: TraceDataInit) {
    this.traceId = init.traceId ?? newUuid();
    this.name = init.name;
    this.status = init.status ?? TraceStatus.COMPLETED;
    this.input = init.input ?? null;
    this.output = init.output ?? null;
    this.metadata = init.metadata ?? {};
    this.startedAt = init.startedAt;
    this.endedAt = init.endedAt ?? null;
    this.sessionId = init.sessionId ?? null;
    this.userId = init.userId ?? null;
    this.tags = init.tags ?? [];
    this.environment = init.environment ?? null;
    this.release = init.release ?? null;
    this.spans = init.spans ?? [];
  }

  /** Serialize to a dict matching the backend POST /traces schema. */
  toApiDict(): Record<string, unknown> {
    const d: Record<string, unknown> = {
      trace_id: this.traceId,
      name: this.name,
      status: this.status,
      started_at: iso(this.startedAt),
    };
    if (this.input !== null && this.input !== undefined) {
      d.input = safeJson(this.input);
    }
    if (this.output !== null && this.output !== undefined) {
      d.output = safeJson(this.output);
    }
    if (this.metadata && Object.keys(this.metadata).length > 0) {
      d.metadata = this.metadata;
    }
    if (this.endedAt !== null) {
      d.ended_at = iso(this.endedAt);
    }
    if (this.sessionId !== null) {
      d.session_id = this.sessionId;
    }
    if (this.userId !== null) {
      d.user_id = this.userId;
    }
    if (this.tags.length > 0) {
      d.tags = this.tags;
    }
    if (this.environment !== null) {
      d.environment = this.environment;
    }
    if (this.release !== null) {
      d.release = this.release;
    }
    if (this.spans.length > 0) {
      d.spans = this.spans.map((s) => s.toApiDict());
    }
    return d;
  }
}

// ---------------------------------------------------------------------------
// Score
// ---------------------------------------------------------------------------

export interface ScoreDataInit {
  traceId: string;
  name: string;
  value: string;
  dataType?: ScoreDataType;
  source?: string;
  reason?: string | null;
  metadata?: Record<string, unknown>;
}

export class ScoreData {
  traceId: string;
  name: string;
  value: string;
  dataType: ScoreDataType;
  source: string;
  reason: string | null;
  metadata: Record<string, unknown>;

  constructor(init: ScoreDataInit) {
    this.traceId = init.traceId;
    this.name = init.name;
    this.value = init.value;
    this.dataType = init.dataType ?? ScoreDataType.NUMERIC;
    this.source = init.source ?? "PROGRAMMATIC";
    this.reason = init.reason ?? null;
    this.metadata = init.metadata ?? {};
  }

  toApiDict(): Record<string, unknown> {
    const d: Record<string, unknown> = {
      trace_id: this.traceId,
      name: this.name,
      value: this.value,
      data_type: this.dataType,
      source: this.source,
    };
    if (this.reason !== null) {
      d.reason = this.reason;
    }
    if (this.metadata && Object.keys(this.metadata).length > 0) {
      d.metadata = this.metadata;
    }
    return d;
  }
}
