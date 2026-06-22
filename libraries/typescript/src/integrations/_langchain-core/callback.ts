/**
 * Generic LangChain-core callback handler shared by every LangChain-based
 * integration (langchain, langgraph, deepagents).
 *
 * Implements the LangChain.js `CallbackHandlerMethods` interface (a class with
 * `handle*` methods + a `name`), so it can be passed in
 * `config: { callbacks: [handler] }` without a hard dependency on
 * `@langchain/core`. Maps run events keyed by `runId` / `parentRunId` to
 * PandaProbe spans and submits one trace when the outermost chain finishes.
 */

import { logger } from "../../logger.js";
import { SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus } from "../../schemas.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { extractLastUserMessage } from "../../validation.js";
import { type AdapterOptions, BaseIntegrationAdapter } from "../base.js";
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

export class BasePandaProbeCallbackHandler extends BaseIntegrationAdapter {
  /** Required by LangChain's CallbackHandlerMethods. */
  readonly name: string = "PandaProbeCallbackHandler";
  /** Trace name used until the root chain start overwrites it. */
  protected static DEFAULT_TRACE_NAME = "LangChain";

  private spans = new Map<string, SpanData>();
  private parents = new Map<string, string | null>();
  private rootRunId: string | null = null;
  private traceInput: unknown = null;
  private traceOutput: unknown = null;
  private traceStartedAt: Date | null = null;
  private traceName: string;

  constructor(options: AdapterOptions = {}) {
    super(options);
    this.traceName = (this.constructor as typeof BasePandaProbeCallbackHandler).DEFAULT_TRACE_NAME;
  }

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
    this.parents.set(rid, pid);
    let name = runName || extractName(chain, "chain");

    if (pid === null) {
      name = this.filterRootChainName(name);
      this.rootRunId = rid;
      const normalized = normalizeLangchainInput(safeOutput(inputs));
      this.traceInput = extractLastUserMessage(normalized);
      this.traceStartedAt = new Date();
      this.traceName = name;
    }

    const span = new SpanData({
      spanId: rid,
      parentSpanId: pid,
      name,
      kind: this.classifyChainKind(pid !== null),
      input: normalizeLangchainInput(safeOutput(inputs)),
      startedAt: new Date(),
    });
    this.spans.set(rid, span);
  }

  handleChainEnd(outputs: Any, runId: string): void {
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
    span.output = normalizeTypeToRole(safeOutput(outputs));
    span.status = SpanStatusCode.OK;
    span.endedAt = new Date();
    if (String(runId) === this.rootRunId) {
      this.traceOutput = normalizeLangchainOutput(safeOutput(outputs));
      this.finalizeTrace(false);
    }
  }

  handleChainError(error: Any, runId: string): void {
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
    span.error = String(error?.message ?? error);
    span.status = SpanStatusCode.ERROR;
    span.endedAt = new Date();
    if (String(runId) === this.rootRunId) {
      this.finalizeTrace(true);
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
    this.parents.set(rid, pid);
    const params = extraParams?.invocation_params ?? {};
    this.spans.set(
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
    this.parents.set(rid, pid);
    const params = extraParams?.invocation_params ?? {};

    const serializedMsgs: unknown[] = [];
    const first = Array.isArray(messages) ? messages[0] : undefined;
    if (Array.isArray(first)) {
      for (const msg of first) {
        serializedMsgs.push(safeOutput(msg));
      }
    }

    this.spans.set(
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
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
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
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
    span.error = String(error?.message ?? error);
    span.status = SpanStatusCode.ERROR;
    span.endedAt = new Date();
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
    this.parents.set(rid, pid);
    this.spans.set(
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
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
    span.output = normalizeTypeToRole(safeOutput(output));
    span.status = SpanStatusCode.OK;
    span.endedAt = new Date();
  }

  handleToolError(error: Any, runId: string): void {
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
    span.error = String(error?.message ?? error);
    span.status = SpanStatusCode.ERROR;
    span.endedAt = new Date();
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
    this.parents.set(rid, pid);
    this.spans.set(
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
    const span = this.spans.get(String(runId));
    if (!span) {
      return;
    }
    span.output = normalizeTypeToRole(safeOutput(documents));
    span.status = SpanStatusCode.OK;
    span.endedAt = new Date();
  }

  // ------------------------------------------------------------------
  // Finalization
  // ------------------------------------------------------------------

  private finalizeTrace(error: boolean): void {
    try {
      const client = this.resolveClient();
      const spans = Array.from(this.spans.values());
      const sessionId = this.sessionId ?? getCurrentSessionId();
      const userId = this.userId ?? getCurrentUserId();
      const trace = new TraceData({
        name: this.traceName,
        status: error ? TraceStatus.ERROR : TraceStatus.COMPLETED,
        input: this.traceInput,
        output: this.traceOutput,
        metadata: { ...this.metadata },
        startedAt: this.traceStartedAt ?? new Date(),
        endedAt: new Date(),
        sessionId,
        userId,
        tags: [...this.tags],
        spans,
      });
      client.logTrace(trace);
    } catch (exc) {
      logger.error(`${this.name} failed to submit trace: ${String(exc)}`);
    } finally {
      this.spans.clear();
      this.parents.clear();
      this.rootRunId = null;
      this.traceInput = null;
      this.traceOutput = null;
    }
  }
}
