/** Invocation-owned state shared by native agent callbacks and model middleware. */
import { AsyncLocalStorage, AsyncResource } from "node:async_hooks";
import type { Client } from "../../client.js";
import { SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus } from "../../schemas.js";
import { getCurrentTrace } from "../../tracing/context.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { extractLastUserMessage } from "../../validation.js";
import { type AdapterOptions, safeSerialize } from "../base.js";
import { extractVercelOutput, extractVercelText, extractVercelUsage, normalizeVercelInput } from "./utils.js";

type Any = any;
export const vercelRuns = new AsyncLocalStorage<VercelRun>();

export class VercelRun {
  readonly ambient = getCurrentTrace();
  readonly chain: SpanData;
  readonly agent: SpanData;
  readonly spans: SpanData[];
  private current?: SpanData;
  private observed = new Set<SpanData>();
  private tools = new Map<string, SpanData>();
  private ended = false;
  private streamFailure?: unknown;
  private input: unknown;
  private output: unknown = null;
  private sessionId = getCurrentSessionId();
  private userId = getCurrentUserId();
  private abortCleanup?: () => void;

  constructor(
    private client: Client,
    private options: AdapterOptions & { name?: string },
    input: Any,
    name?: string,
  ) {
    this.input = extractLastUserMessage(normalizeVercelInput({ prompt: input.messages ?? input.prompt }));
    this.chain = new SpanData({
      name: options.name ?? "VercelAI",
      kind: SpanKind.CHAIN,
      parentSpanId: this.ambient?.spanStack.at(-1) ?? null,
      input: this.input,
      startedAt: new Date(),
    });
    this.agent = new SpanData({
      name: name ?? "ToolLoopAgent",
      kind: SpanKind.AGENT,
      parentSpanId: this.chain.spanId,
      input: this.input,
      startedAt: this.chain.startedAt,
    });
    this.spans = [this.chain, this.agent];
    // Abort listeners and stream consumers may run in a different async frame.
    // Keep export/capture attached to the invocation that owns this trace.
    this.finish = AsyncResource.bind(this.finish.bind(this));
  }

  watch(signal?: AbortSignal) {
    if (!signal) return;
    const abort = () => this.finish(signal.reason ?? new Error("Vercel agent aborted"));
    signal.addEventListener("abort", abort, { once: true });
    this.abortCleanup = () => signal.removeEventListener("abort", abort);
    if (signal.aborted) abort();
  }

  start(event: Any) {
    if (this.ended) return;
    this.agent.model = event.model?.modelId ?? null;
  }

  stepStart(event: Any) {
    if (this.ended) return;
    this.current = new SpanData({
      name: "vercel-ai-generate",
      kind: SpanKind.LLM,
      parentSpanId: this.agent.spanId,
      model: event.model?.modelId ?? null,
      input: normalizeVercelInput({
        prompt: [
          ...(typeof event.system === "string"
            ? [{ role: "system", content: event.system }]
            : event.system
              ? [event.system].flat()
              : []),
          ...(event.messages ?? []),
        ],
      }),
      metadata: { step: event.stepNumber, timing_source: "native_step_callbacks" },
      startedAt: new Date(),
    });
    this.spans.push(this.current);
  }

  /** Middleware supplies exact provider-call timing, including individual retries. */
  model(params: Any, model: Any, name: string) {
    if (this.ended) return null;
    if (!this.current || this.observed.has(this.current)) {
      this.stepStart({ model, messages: [] });
    }
    const span = this.current!;
    this.observed.add(span);
    span.name = name;
    span.model = model?.modelId ?? model?.modelID ?? span.model;
    span.input = normalizeVercelInput(params);
    span.startedAt = new Date();
    span.metadata.timing_source = "model_middleware";
    return {
      setOutput: (value: unknown) => {
        if (!this.ended) span.output = safeSerialize(value);
      },
      setTokenUsage: (value: Record<string, number>) => {
        if (!this.ended) span.tokenUsage = value;
      },
      setCompletionStartTime: (value: Date) => {
        if (!this.ended) span.completionStartTime = value;
      },
      end: (error?: unknown) => this.endSpan(span, error),
    };
  }

  stepEnd(event: Any) {
    if (!this.current || this.ended) return;
    this.current.output = extractVercelOutput(event);
    this.current.tokenUsage = extractVercelUsage(event.usage);
    if (!this.current.endedAt) this.endSpan(this.current);
  }

  toolStart(event: Any) {
    if (this.ended) return;
    // Without model middleware, first tool execution bounds the model step.
    if (this.current && !this.observed.has(this.current) && !this.current.endedAt) this.endSpan(this.current);
    const call = event.toolCall;
    if (!call || this.tools.has(call.toolCallId)) return;
    const span = new SpanData({
      name: call.toolName,
      kind: SpanKind.TOOL,
      parentSpanId: this.current?.spanId ?? this.agent.spanId,
      input: safeSerialize(call.input),
      metadata: { tool_call_id: call.toolCallId },
      startedAt: new Date(),
    });
    this.tools.set(call.toolCallId, span);
    this.spans.push(span);
  }

  toolEnd(event: Any) {
    const span = this.tools.get(event.toolCall?.toolCallId);
    if (!span || this.ended) return;
    const result = event.toolOutput ?? event;
    span.output = safeSerialize(result.output);
    this.endSpan(
      span,
      event.success === false || result.type === "tool-error"
        ? (result.error ?? new Error("Tool execution failed"))
        : undefined,
    );
    this.tools.delete(event.toolCall.toolCallId);
  }

  result(event: Any) {
    if (this.ended) return;
    const text = extractVercelText(event);
    this.output = text === null ? null : { messages: [{ role: "assistant", content: text }] };
    this.chain.output = this.output;
    this.agent.output = this.output;
  }

  streamError(error: unknown) {
    this.streamFailure = error;
  }

  private endSpan(span: SpanData, error?: unknown) {
    if (span.endedAt) return;
    span.endedAt = new Date();
    span.status = error == null ? SpanStatusCode.OK : SpanStatusCode.ERROR;
    if (error != null) span.error = String(error instanceof Error ? error.message : error);
  }

  finish(error?: unknown) {
    if (this.ended) return;
    const failure = error ?? this.streamFailure;
    this.ended = true;
    this.abortCleanup?.();
    for (const span of this.spans) this.endSpan(span, failure);
    try {
      if (this.ambient) {
        for (const span of this.spans) this.ambient.addSpanData(span);
      } else {
        this.client.logTrace(
          new TraceData({
            name: this.chain.name,
            input: this.input,
            output: this.output,
            status: failure == null ? TraceStatus.COMPLETED : TraceStatus.ERROR,
            startedAt: this.chain.startedAt,
            endedAt: this.chain.endedAt,
            sessionId: this.options.sessionId ?? this.sessionId,
            userId: this.options.userId ?? this.userId,
            tags: this.options.tags,
            metadata: this.options.metadata,
            spans: this.spans,
          }),
        );
      }
    } catch {
      /* Instrumentation must not change the native result or error. */
    }
  }
}
