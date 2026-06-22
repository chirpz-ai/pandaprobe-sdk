/**
 * OpenAI Agents SDK integration.
 *
 * Mirrors the Python adapter's approach: register a custom tracing processor
 * with the `@openai/agents` SDK (its native tracing hooks). Each SDK trace maps
 * to one PandaProbe trace; SDK spans map to PandaProbe spans by type.
 */

import { logger } from "../../logger.js";
import { SpanData, SpanStatusCode, TraceData, TraceStatus } from "../../schemas.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { type AdapterOptions, BaseIntegrationAdapter } from "../base.js";
import { extractAgentsUsage, extractSpanIo, mapSpanKind } from "./utils.js";

type Any = any;

interface TraceState {
  name: string;
  startedAt: Date;
  spans: SpanData[];
}

/** The TracingProcessor object shape expected by `@openai/agents`. */
export interface TraceProcessor {
  onTraceStart(trace: Any): void;
  onTraceEnd(trace: Any): void;
  onSpanStart(span: Any): void;
  onSpanEnd(span: Any): void;
  forceFlush?(): void;
  shutdown?(): void;
}

export class OpenAIAgentsAdapter extends BaseIntegrationAdapter {
  private states = new Map<string, TraceState>();

  /** Build the tracing processor object (register it with the Agents SDK). */
  createTraceProcessor(): TraceProcessor {
    return {
      onTraceStart: (trace: Any) => {
        const traceId = String(trace?.traceId ?? trace?.trace_id ?? "");
        this.states.set(traceId, {
          name: trace?.name ?? "OpenAIAgents",
          startedAt: new Date(),
          spans: [],
        });
      },
      onSpanStart: () => {
        // Spans are materialized on end (when input/output/usage are known).
      },
      onSpanEnd: (span: Any) => {
        const traceId = String(span?.traceId ?? span?.trace_id ?? "");
        const state = this.states.get(traceId);
        if (!state) {
          return;
        }
        const spanData = span?.spanData ?? span?.span_data ?? {};
        const type = spanData?.type;
        const io = extractSpanIo(spanData);
        state.spans.push(
          new SpanData({
            spanId: String(span?.spanId ?? span?.span_id ?? ""),
            parentSpanId: span?.parentId ?? span?.parent_id ?? null,
            name: spanData?.name ?? type ?? "span",
            kind: mapSpanKind(type),
            input: io.input,
            output: io.output,
            model: io.model ?? null,
            tokenUsage: extractAgentsUsage(spanData?.usage),
            status: span?.error ? SpanStatusCode.ERROR : SpanStatusCode.OK,
            error: span?.error ? String(span.error?.message ?? span.error) : null,
            startedAt: span?.startedAt ? new Date(span.startedAt) : new Date(),
            endedAt: span?.endedAt ? new Date(span.endedAt) : new Date(),
          }),
        );
      },
      onTraceEnd: (trace: Any) => {
        const traceId = String(trace?.traceId ?? trace?.trace_id ?? "");
        const state = this.states.get(traceId);
        if (!state) {
          return;
        }
        this.states.delete(traceId);
        try {
          const client = this.resolveClient();
          client.logTrace(
            new TraceData({
              name: state.name,
              status: TraceStatus.COMPLETED,
              metadata: { ...this.metadata },
              startedAt: state.startedAt,
              endedAt: new Date(),
              sessionId: this.sessionId ?? getCurrentSessionId(),
              userId: this.userId ?? getCurrentUserId(),
              tags: [...this.tags],
              spans: state.spans,
            }),
          );
        } catch (exc) {
          logger.error(`OpenAIAgents adapter failed to submit trace: ${String(exc)}`);
        }
      },
      forceFlush: () => {},
      shutdown: () => {},
    };
  }

  /**
   * Register the processor with the `@openai/agents` SDK. Returns true on
   * success, false if the SDK is unavailable or exposes no known hook.
   */
  async instrument(): Promise<boolean> {
    let agents: Any;
    try {
      // Indirect the specifier so the bundler/typechecker treats it as a
      // runtime-only optional import (the SDK is an optional peer dependency).
      const moduleName: string = "@openai/agents";
      agents = await import(moduleName);
    } catch {
      return false;
    }
    const processor = this.createTraceProcessor();
    if (typeof agents.addTraceProcessor === "function") {
      agents.addTraceProcessor(processor);
      return true;
    }
    if (typeof agents.setTraceProcessors === "function") {
      agents.setTraceProcessors([processor]);
      return true;
    }
    const tracing = agents.tracing ?? agents.getGlobalTraceProvider?.();
    if (tracing && typeof tracing.registerProcessor === "function") {
      tracing.registerProcessor(processor);
      return true;
    }
    return false;
  }
}
