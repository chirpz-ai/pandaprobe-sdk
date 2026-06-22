/**
 * Claude Agent SDK integration.
 *
 * JS ESM bindings can't be monkey-patched the way Python's wrapt patches the
 * SDK internals, so this exposes `wrapQuery(query)` — wrap the SDK's `query`
 * function and it emits a trace (AGENT root → LLM per assistant turn → TOOL per
 * tool use) by inspecting the streamed messages, while transparently
 * re-yielding them to the caller.
 */

import { SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus, newUuid } from "../../schemas.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { type AdapterOptions, BaseIntegrationAdapter, safeSerialize } from "../base.js";
import {
  extractClaudeUsage,
  extractPromptText,
  extractToolResults,
  extractToolUses,
  splitAssistantContent,
} from "./utils.js";

type Any = any;

type QueryFn = (args: Any) => AsyncIterable<Any>;

export class ClaudeAgentSDKAdapter extends BaseIntegrationAdapter {
  /** Wrap the SDK `query` function so each call is traced. */
  wrapQuery(query: QueryFn): QueryFn {
    return (args: Any): AsyncIterable<Any> => this.run(query, args);
  }

  private async *run(query: QueryFn, args: Any): AsyncGenerator<Any> {
    let client: ReturnType<BaseIntegrationAdapter["resolveClient"]> | null = null;
    try {
      client = this.resolveClient();
    } catch {
      // No client — pass through untraced.
      yield* query(args);
      return;
    }

    const startedAt = new Date();
    const spans: SpanData[] = [];
    const agentSpanId = newUuid();
    const promptText = extractPromptText(args?.prompt);
    const toolSpansById = new Map<string, SpanData>();
    let lastAssistantText = "";
    let model: string | null = null;
    let usage: Record<string, number> | null = null;
    let errored = false;

    try {
      for await (const msg of query(args)) {
        try {
          this.handleMessage(msg, { agentSpanId, spans, toolSpansById, startedAt }, (m, u, t) => {
            if (m) model = m;
            if (u) usage = u;
            if (t) lastAssistantText = t;
          });
        } catch {
          // never let instrumentation break iteration
        }
        yield msg;
      }
    } catch (exc) {
      errored = true;
      this.submit({
        client,
        agentSpanId,
        spans,
        startedAt,
        promptText,
        lastAssistantText,
        model,
        usage,
        error: String((exc as Error)?.message ?? exc),
      });
      throw exc;
    }

    if (!errored) {
      this.submit({ client, agentSpanId, spans, startedAt, promptText, lastAssistantText, model, usage, error: null });
    }
  }

  private handleMessage(
    msg: Any,
    ctx: { agentSpanId: string; spans: SpanData[]; toolSpansById: Map<string, SpanData>; startedAt: Date },
    update: (model: string | null, usage: Record<string, number> | null, text: string | null) => void,
  ): void {
    if (msg?.type === "assistant") {
      const content = msg.message?.content;
      const { text, thinking } = splitAssistantContent(content);
      const llm = new SpanData({
        parentSpanId: ctx.agentSpanId,
        name: "claude-llm",
        kind: SpanKind.LLM,
        output: { messages: [{ role: "assistant", content: text }] },
        model: msg.message?.model ?? null,
        tokenUsage: extractClaudeUsage(msg.message?.usage),
        metadata: thinking ? { reasoning_summary: thinking } : {},
        status: SpanStatusCode.OK,
        startedAt: new Date(),
        endedAt: new Date(),
      });
      ctx.spans.push(llm);
      update(msg.message?.model ?? null, extractClaudeUsage(msg.message?.usage), text || null);

      for (const use of extractToolUses(content)) {
        const toolSpan = new SpanData({
          parentSpanId: ctx.agentSpanId,
          name: use.name,
          kind: SpanKind.TOOL,
          input: safeSerialize(use.input),
          status: SpanStatusCode.OK,
          startedAt: new Date(),
          endedAt: new Date(),
        });
        ctx.spans.push(toolSpan);
        if (use.id) {
          ctx.toolSpansById.set(use.id, toolSpan);
        }
      }
    } else if (msg?.type === "user") {
      for (const result of extractToolResults(msg.message?.content)) {
        const toolSpan = ctx.toolSpansById.get(result.toolUseId);
        if (toolSpan) {
          toolSpan.output = safeSerialize(result.content);
        }
      }
    } else if (msg?.type === "result") {
      const u = extractClaudeUsage(msg.usage);
      if (u) {
        update(null, u, null);
      }
      if (typeof msg.result === "string" && msg.result) {
        update(null, null, msg.result);
      }
    }
  }

  private submit(args: {
    client: ReturnType<BaseIntegrationAdapter["resolveClient"]>;
    agentSpanId: string;
    spans: SpanData[];
    startedAt: Date;
    promptText: string;
    lastAssistantText: string;
    model: string | null;
    usage: Record<string, number> | null;
    error: string | null;
  }): void {
    const agentSpan = new SpanData({
      spanId: args.agentSpanId,
      name: "ClaudeAgent",
      kind: SpanKind.AGENT,
      input: { messages: [{ role: "user", content: args.promptText }] },
      output: { messages: [{ role: "assistant", content: args.lastAssistantText }] },
      model: args.model,
      tokenUsage: args.usage,
      status: args.error ? SpanStatusCode.ERROR : SpanStatusCode.OK,
      error: args.error,
      startedAt: args.startedAt,
      endedAt: new Date(),
    });

    const trace = new TraceData({
      name: "ClaudeAgent",
      status: args.error ? TraceStatus.ERROR : TraceStatus.COMPLETED,
      input: { messages: [{ role: "user", content: args.promptText }] },
      output: { messages: [{ role: "assistant", content: args.lastAssistantText }] },
      metadata: { ...this.metadata },
      startedAt: args.startedAt,
      endedAt: new Date(),
      sessionId: this.sessionId ?? getCurrentSessionId(),
      userId: this.userId ?? getCurrentUserId(),
      tags: [...this.tags],
      spans: [agentSpan, ...args.spans],
    });
    args.client.logTrace(trace);
  }
}

/** Wrap the Claude Agent SDK `query` function so each call is traced. */
export function wrapClaudeAgentQuery(query: QueryFn, options: AdapterOptions = {}): QueryFn {
  return new ClaudeAgentSDKAdapter(options).wrapQuery(query);
}
