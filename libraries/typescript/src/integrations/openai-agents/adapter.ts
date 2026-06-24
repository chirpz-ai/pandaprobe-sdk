/**
 * OpenAI Agents SDK integration.
 *
 * Mirrors the Python adapter: register a tracing processor with the
 * `@openai/agents` SDK and build, per SDK trace, a synthetic root CHAIN span
 * that parents all SDK spans (remapped to PandaProbe UUIDs). Each SDK span is
 * filled per type (response/generation/function/agent/handoff/guardrail/custom)
 * with normalized message I/O, token usage, model parameters and reasoning; LLM
 * spans propagate I/O to their parent AGENT span and drive chain/trace-level I/O.
 */

import { logger } from "../../logger.js";
import { SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus, newUuid } from "../../schemas.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { extractLastUserMessage } from "../../validation.js";
import { type AdapterOptions, BaseIntegrationAdapter, safeSerialize } from "../base.js";
import {
  extractGenerationModelParameters,
  extractReasoning,
  extractResponseModelParameters,
  extractTokenUsage,
  mapSpanKind,
  normalizeGenerationInput,
  normalizeGenerationOutput,
  normalizeResponseInput,
  normalizeResponseOutput,
  resolveSpanName,
  serializeToolIo,
} from "./utils.js";

type Any = any;

const TRACE_NAME = "OpenAIAgentsSDK";

interface TraceState {
  rootSpanId: string;
  startedAt: Date;
  spans: Map<string, SpanData>;
  sdkToPp: Map<string, string>;
  traceInput: unknown;
  traceOutput: unknown;
  chainInput: unknown;
  chainOutput: unknown;
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

function messagesOf(io: unknown): Any[] {
  if (io && typeof io === "object" && Array.isArray((io as Any).messages)) {
    return (io as Any).messages;
  }
  return [];
}

export class OpenAIAgentsAdapter extends BaseIntegrationAdapter {
  private states = new Map<string, TraceState>();

  /** Build the tracing processor object (register it with the Agents SDK). */
  createTraceProcessor(): TraceProcessor {
    return {
      onTraceStart: (trace: Any) => {
        const traceId = String(trace?.traceId ?? trace?.trace_id ?? "");
        if (!traceId) {
          return;
        }
        const rootSpanId = newUuid();
        const startedAt = new Date();
        const spans = new Map<string, SpanData>();
        spans.set(
          rootSpanId,
          new SpanData({
            spanId: rootSpanId,
            parentSpanId: null,
            name: trace?.name ?? TRACE_NAME,
            kind: SpanKind.CHAIN,
            startedAt,
          }),
        );
        this.states.set(traceId, {
          rootSpanId,
          startedAt,
          spans,
          sdkToPp: new Map(),
          traceInput: null,
          traceOutput: null,
          chainInput: null,
          chainOutput: null,
        });
      },

      onSpanStart: (span: Any) => {
        const traceId = String(span?.traceId ?? span?.trace_id ?? "");
        const state = this.states.get(traceId);
        if (!state) {
          return;
        }
        const sdkSpanId = String(span?.spanId ?? span?.span_id ?? "");
        if (!sdkSpanId) {
          return;
        }
        const spanData = span?.spanData ?? span?.span_data;
        const ppId = newUuid();
        state.sdkToPp.set(sdkSpanId, ppId);

        const rawParent = span?.parentId ?? span?.parent_id;
        const sdkParent = rawParent != null ? String(rawParent) : null;
        const parentPp =
          sdkParent === null || sdkParent === traceId
            ? state.rootSpanId
            : (state.sdkToPp.get(sdkParent) ?? state.rootSpanId);

        state.spans.set(
          ppId,
          new SpanData({
            spanId: ppId,
            parentSpanId: parentPp,
            name: resolveSpanName(span),
            kind: mapSpanKind(spanData?.type),
            startedAt: span?.startedAt ? new Date(span.startedAt) : new Date(),
          }),
        );
      },

      onSpanEnd: (span: Any) => {
        const traceId = String(span?.traceId ?? span?.trace_id ?? "");
        const state = this.states.get(traceId);
        if (!state) {
          return;
        }
        const sdkSpanId = String(span?.spanId ?? span?.span_id ?? "");
        const ppId = state.sdkToPp.get(sdkSpanId);
        const sd = ppId ? state.spans.get(ppId) : undefined;
        if (!sd) {
          return;
        }
        sd.endedAt = span?.endedAt ? new Date(span.endedAt) : new Date();
        if (span?.error) {
          sd.error = String(span.error?.message ?? span.error);
          sd.status = SpanStatusCode.ERROR;
        } else {
          sd.status = SpanStatusCode.OK;
        }

        const spanData = span?.spanData ?? span?.span_data;
        if (!spanData) {
          return;
        }
        switch (spanData.type) {
          case "response":
            this.fillResponseSpan(sd, spanData, state);
            break;
          case "generation":
            this.fillGenerationSpan(sd, spanData, state);
            break;
          case "function":
            this.fillFunctionSpan(sd, spanData);
            break;
          case "agent":
            this.fillAgentSpan(sd, spanData);
            break;
          case "handoff":
            this.fillHandoffSpan(sd, spanData);
            break;
          case "guardrail":
            sd.metadata.triggered = spanData.triggered;
            break;
          case "custom":
            if (spanData.data) {
              sd.metadata.custom_data = safeSerialize(spanData.data);
            }
            break;
        }
      },

      onTraceEnd: (trace: Any) => {
        const traceId = String(trace?.traceId ?? trace?.trace_id ?? "");
        const state = this.states.get(traceId);
        if (!state) {
          return;
        }
        this.states.delete(traceId);
        try {
          const now = new Date();
          const hasError = [...state.spans.values()].some((s) => s.status === SpanStatusCode.ERROR);
          const root = state.spans.get(state.rootSpanId);
          if (root) {
            root.input = state.chainInput;
            root.output = state.chainOutput;
            root.endedAt = now;
            root.status = hasError ? SpanStatusCode.ERROR : SpanStatusCode.OK;
          }
          this.resolveClient().logTrace(
            new TraceData({
              name: TRACE_NAME,
              status: hasError ? TraceStatus.ERROR : TraceStatus.COMPLETED,
              input: state.traceInput,
              output: state.traceOutput,
              metadata: { ...this.metadata },
              startedAt: state.startedAt,
              endedAt: now,
              sessionId: this.sessionId ?? getCurrentSessionId(),
              userId: this.userId ?? getCurrentUserId(),
              tags: [...this.tags],
              spans: [...state.spans.values()],
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

  // ------------------------------------------------------------------
  // Per-span-type fills
  // ------------------------------------------------------------------

  private propagateLlmToParentAgent(span: SpanData, state: TraceState): void {
    const parentId = span.parentSpanId;
    if (!parentId) {
      return;
    }
    const parent = state.spans.get(parentId);
    if (!parent || parent.kind !== SpanKind.AGENT) {
      return;
    }
    if ((parent.input === null || parent.input === undefined) && span.input) {
      parent.input = span.input;
    }
    if (span.output) {
      parent.output = span.output;
    }
  }

  private updateChainAndTraceIo(span: SpanData, state: TraceState): void {
    if ((state.chainInput === null || state.chainInput === undefined) && span.input) {
      state.chainInput = span.input;
    }
    const chainMessages = [...messagesOf(span.input), ...messagesOf(span.output)];
    if (chainMessages.length > 0) {
      state.chainOutput = { messages: chainMessages };
    } else if (span.output) {
      state.chainOutput = span.output;
    }
    if ((state.traceInput === null || state.traceInput === undefined) && span.input) {
      state.traceInput = extractLastUserMessage(span.input);
    }
    if (span.output) {
      state.traceOutput = span.output;
    }
  }

  private fillResponseSpan(span: SpanData, spanData: Any, state: TraceState): void {
    span.input = normalizeResponseInput(spanData);
    span.output = normalizeResponseOutput(spanData);
    const response = spanData.response;
    if (response) {
      if (response.model) {
        span.model = String(response.model);
        span.name = String(response.model);
      }
      if (response.usage) {
        span.tokenUsage = extractTokenUsage(response.usage);
      }
      span.modelParameters = extractResponseModelParameters(response);
      const reasoning = extractReasoning(response);
      if (reasoning) {
        span.metadata.reasoning_summary = reasoning;
      }
    }
    this.propagateLlmToParentAgent(span, state);
    this.updateChainAndTraceIo(span, state);
  }

  private fillGenerationSpan(span: SpanData, spanData: Any, state: TraceState): void {
    span.input = normalizeGenerationInput(spanData);
    span.output = normalizeGenerationOutput(spanData);
    if (spanData.model) {
      span.model = String(spanData.model);
      span.name = String(spanData.model);
    }
    if (spanData.usage) {
      span.tokenUsage = extractTokenUsage(spanData.usage);
    }
    span.modelParameters = extractGenerationModelParameters(spanData);
    this.propagateLlmToParentAgent(span, state);
    this.updateChainAndTraceIo(span, state);
  }

  private fillFunctionSpan(span: SpanData, spanData: Any): void {
    if (spanData.input != null) {
      span.input = serializeToolIo(spanData.input);
    }
    if (spanData.output != null) {
      span.output = serializeToolIo(spanData.output);
    }
    if (spanData.name) {
      span.name = String(spanData.name);
    }
    if (spanData.mcp_data) {
      span.metadata.mcp_data = safeSerialize(spanData.mcp_data);
    }
  }

  private fillAgentSpan(span: SpanData, spanData: Any): void {
    if (spanData.tools) {
      span.metadata.tools = safeSerialize(spanData.tools);
    }
    if (spanData.handoffs) {
      span.metadata.handoffs = safeSerialize(spanData.handoffs);
    }
    if (spanData.output_type) {
      span.metadata.output_type = String(spanData.output_type);
    }
  }

  private fillHandoffSpan(span: SpanData, spanData: Any): void {
    const from = spanData.from_agent ? String(spanData.from_agent) : null;
    const to = spanData.to_agent ? String(spanData.to_agent) : null;
    span.input = { from_agent: from };
    span.output = { to_agent: to };
    span.metadata.from_agent = from;
    span.metadata.to_agent = to;
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
    const tracing = agents.tracing ?? agents.getGlobalTraceProvider?.();
    if (tracing && typeof tracing.registerProcessor === "function") {
      tracing.registerProcessor(processor);
      return true;
    }
    return false;
  }
}
