/**
 * Generic LangChain-core callback handler shared by every LangChain-based
 * integration (langchain, langgraph, deepagents).
 *
 * Implements the LangChain.js `CallbackHandlerMethods` interface (a class with
 * `handle*` methods + a `name`), so it can be passed in
 * `config: { callbacks: [handler] }` without a hard dependency on
 * `@langchain/core`. Maps run events keyed by `runId` / `parentRunId` to
 * PandaProbe spans and submits one trace when the outermost chain finishes.
 *
 * Per-run state is keyed by the **root run id** (each run id is routed to its
 * root via its parent chain), so a single handler instance can be shared across
 * concurrent `invoke`/`stream` calls without interleaving their spans into one
 * trace. State is released when each root finishes.
 */

import { logger } from "../../logger.js";
import { SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus } from "../../schemas.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { extractLastUserMessage } from "../../validation.js";
import { BaseIntegrationAdapter } from "../base.js";
import {
  extractModelParameters,
  extractName,
  extractReasoningFromGeneration,
  extractTokenUsage,
  normalizeLangchainInput,
  normalizeLangchainOutput,
  normalizeLlmGenerationOutput,
  normalizeTypeToRole,
  safeOutput,
} from "./utils.js";

type Any = any;

/** Per-invocation state, keyed by the root run id. */
interface TraceState {
  rootRunId: string;
  spans: Map<string, SpanData>;
  traceInput: unknown;
  traceOutput: unknown;
  traceStartedAt: Date;
  traceName: string;
  runIds: Set<string>;
}

export class BasePandaProbeCallbackHandler extends BaseIntegrationAdapter {
  /** Required by LangChain's CallbackHandlerMethods. */
  readonly name: string = "PandaProbeCallbackHandler";
  /** Trace name used until the root chain start overwrites it. */
  protected static DEFAULT_TRACE_NAME = "LangChain";

  /** Active traces keyed by root run id. */
  private traces = new Map<string, TraceState>();
  /** Maps any run id → its root run id, so interleaved events route correctly. */
  private runToRoot = new Map<string, string>();

  // ------------------------------------------------------------------
  // Override hooks
  // ------------------------------------------------------------------

  protected classifyChainKind(hasParent: boolean): SpanKind {
    return hasParent ? SpanKind.AGENT : SpanKind.CHAIN;
  }

  protected filterRootChainName(name: string): string {
    return name;
  }

  // ------------------------------------------------------------------
  // Per-run state routing
  // ------------------------------------------------------------------

  private get defaultTraceName(): string {
    return (this.constructor as typeof BasePandaProbeCallbackHandler).DEFAULT_TRACE_NAME;
  }

  /** Resolve (creating if needed) the trace state a starting run belongs to. */
  private startState(rid: string, pid: string | null): TraceState {
    if (pid !== null) {
      const root = this.runToRoot.get(pid);
      const state = root !== undefined ? this.traces.get(root) : undefined;
      if (root !== undefined && state) {
        this.runToRoot.set(rid, root);
        state.runIds.add(rid);
        return state;
      }
    }
    // Root run (no parent), or a run whose parent is unknown → start a new trace.
    const state: TraceState = {
      rootRunId: rid,
      spans: new Map(),
      traceInput: null,
      traceOutput: null,
      traceStartedAt: new Date(),
      traceName: this.defaultTraceName,
      runIds: new Set([rid]),
    };
    this.traces.set(rid, state);
    this.runToRoot.set(rid, rid);
    return state;
  }

  /** Look up the live span (and its trace) for a run id, if tracked. */
  private spanFor(rid: string): { state: TraceState; span: SpanData } | undefined {
    const root = this.runToRoot.get(rid);
    if (root === undefined) {
      return undefined;
    }
    const state = this.traces.get(root);
    const span = state?.spans.get(rid);
    if (!state || !span) {
      return undefined;
    }
    return { state, span };
  }

  // ------------------------------------------------------------------
  // Chain callbacks
  // ------------------------------------------------------------------

  handleChainStart(
    chain: Any,
    inputs: Any,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    _metadata?: Any,
    _runType?: string,
    runName?: string,
  ): void {
    const rid = String(runId);
    const pid = parentRunId ? String(parentRunId) : null;
    const state = this.startState(rid, pid);
    let name = runName || extractName(chain, "chain");

    if (state.rootRunId === rid) {
      name = this.filterRootChainName(name);
      state.traceName = name;
      state.traceInput = extractLastUserMessage(normalizeLangchainInput(safeOutput(inputs)));
    }

    state.spans.set(
      rid,
      new SpanData({
        spanId: rid,
        parentSpanId: pid,
        name,
        kind: this.classifyChainKind(pid !== null),
        input: normalizeLangchainInput(safeOutput(inputs)),
        startedAt: new Date(),
      }),
    );
  }

  handleChainEnd(outputs: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    const { state, span } = found;
    span.output = normalizeTypeToRole(safeOutput(outputs));
    span.status = SpanStatusCode.OK;
    span.endedAt = new Date();
    if (String(runId) === state.rootRunId) {
      state.traceOutput = normalizeLangchainOutput(safeOutput(outputs));
      this.finalizeTrace(state, false);
    }
  }

  handleChainError(error: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    const { state, span } = found;
    span.error = String(error?.message ?? error);
    span.status = SpanStatusCode.ERROR;
    span.endedAt = new Date();
    if (String(runId) === state.rootRunId) {
      this.finalizeTrace(state, true);
    }
  }

  // ------------------------------------------------------------------
  // LLM callbacks
  // ------------------------------------------------------------------

  handleLLMStart(
    llm: Any,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Any,
    _tags?: string[],
    _metadata?: Any,
    runName?: string,
  ): void {
    const rid = String(runId);
    const pid = parentRunId ? String(parentRunId) : null;
    const state = this.startState(rid, pid);
    const params = extraParams?.invocation_params ?? {};
    state.spans.set(
      rid,
      new SpanData({
        spanId: rid,
        parentSpanId: pid,
        name: runName || extractName(llm, "llm"),
        kind: SpanKind.LLM,
        input: safeOutput(prompts),
        model: params.model ?? params.model_name ?? null,
        modelParameters: extractModelParameters(params),
        startedAt: new Date(),
      }),
    );
  }

  handleChatModelStart(
    llm: Any,
    messages: Any[],
    runId: string,
    parentRunId?: string,
    extraParams?: Any,
    _tags?: string[],
    _metadata?: Any,
    runName?: string,
  ): void {
    const rid = String(runId);
    const pid = parentRunId ? String(parentRunId) : null;
    const state = this.startState(rid, pid);
    const params = extraParams?.invocation_params ?? {};

    const serializedMsgs: unknown[] = [];
    const first = Array.isArray(messages) ? messages[0] : undefined;
    if (Array.isArray(first)) {
      for (const msg of first) {
        serializedMsgs.push(safeOutput(msg));
      }
    }

    state.spans.set(
      rid,
      new SpanData({
        spanId: rid,
        parentSpanId: pid,
        name: runName || extractName(llm, "llm"),
        kind: SpanKind.LLM,
        input: normalizeTypeToRole({ messages: serializedMsgs }),
        model: params.model ?? params.model_name ?? null,
        modelParameters: extractModelParameters(params),
        startedAt: new Date(),
      }),
    );
  }

  handleLLMEnd(output: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    const { span } = found;
    try {
      const normalized = normalizeLlmGenerationOutput(output);
      if (normalized !== null) {
        span.output = normalized;
      }
      span.tokenUsage = extractTokenUsage(output);
      const reasoning = extractReasoningFromGeneration(output);
      if (reasoning) {
        span.metadata.reasoning_summary = reasoning;
      }
    } catch (exc) {
      logger.debug(`Error extracting LLM response: ${String(exc)}`);
    }
    span.status = SpanStatusCode.OK;
    span.endedAt = new Date();
  }

  handleLLMError(error: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    found.span.error = String(error?.message ?? error);
    found.span.status = SpanStatusCode.ERROR;
    found.span.endedAt = new Date();
  }

  // ------------------------------------------------------------------
  // Tool callbacks
  // ------------------------------------------------------------------

  handleToolStart(
    tool: Any,
    input: Any,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    _metadata?: Any,
    runName?: string,
  ): void {
    const rid = String(runId);
    const pid = parentRunId ? String(parentRunId) : null;
    const state = this.startState(rid, pid);
    state.spans.set(
      rid,
      new SpanData({
        spanId: rid,
        parentSpanId: pid,
        name: runName || extractName(tool, "tool"),
        kind: SpanKind.TOOL,
        input: normalizeTypeToRole(safeOutput(input)),
        startedAt: new Date(),
      }),
    );
  }

  handleToolEnd(output: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    found.span.output = normalizeTypeToRole(safeOutput(output));
    found.span.status = SpanStatusCode.OK;
    found.span.endedAt = new Date();
  }

  handleToolError(error: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    found.span.error = String(error?.message ?? error);
    found.span.status = SpanStatusCode.ERROR;
    found.span.endedAt = new Date();
  }

  // ------------------------------------------------------------------
  // Retriever callbacks
  // ------------------------------------------------------------------

  handleRetrieverStart(
    retriever: Any,
    query: Any,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    _metadata?: Any,
    runName?: string,
  ): void {
    const rid = String(runId);
    const pid = parentRunId ? String(parentRunId) : null;
    const state = this.startState(rid, pid);
    state.spans.set(
      rid,
      new SpanData({
        spanId: rid,
        parentSpanId: pid,
        name: runName || extractName(retriever, "retriever"),
        kind: SpanKind.RETRIEVER,
        input: normalizeTypeToRole(safeOutput(query)),
        startedAt: new Date(),
      }),
    );
  }

  handleRetrieverEnd(documents: Any, runId: string): void {
    const found = this.spanFor(String(runId));
    if (!found) {
      return;
    }
    found.span.output = normalizeTypeToRole(safeOutput(documents));
    found.span.status = SpanStatusCode.OK;
    found.span.endedAt = new Date();
  }

  // ------------------------------------------------------------------
  // Finalization
  // ------------------------------------------------------------------

  private finalizeTrace(state: TraceState, error: boolean): void {
    try {
      const client = this.resolveClient();
      const trace = new TraceData({
        name: state.traceName,
        status: error ? TraceStatus.ERROR : TraceStatus.COMPLETED,
        input: state.traceInput,
        output: state.traceOutput,
        metadata: { ...this.metadata },
        startedAt: state.traceStartedAt,
        endedAt: new Date(),
        sessionId: this.sessionId ?? getCurrentSessionId(),
        userId: this.userId ?? getCurrentUserId(),
        tags: [...this.tags],
        spans: Array.from(state.spans.values()),
      });
      client.logTrace(trace);
    } catch (exc) {
      logger.error(`${this.name} failed to submit trace: ${String(exc)}`);
    } finally {
      this.traces.delete(state.rootRunId);
      for (const rid of state.runIds) {
        this.runToRoot.delete(rid);
      }
    }
  }
}
