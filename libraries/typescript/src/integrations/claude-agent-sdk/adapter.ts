/**
 * Claude Agent SDK integration.
 *
 * JS ESM bindings can't be monkey-patched the way Python's wrapt patches the
 * SDK internals, so this exposes `wrapQuery(query)` — wrap the SDK's `query`
 * function and it emits a trace by inspecting the streamed messages, while
 * transparently re-yielding them to the caller.
 *
 * Emits the same trace shape as the Python adapter:
 *   CHAIN ("ClaudeAgentSDK")
 *   └── AGENT ("ClaudeAgentSDK")
 *       ├── LLM   (one per non-thinking-only assistant turn; tool_calls on output)
 *       ├── TOOL  (one per tool_use block, output from the matching tool_result)
 *       └── AGENT (subagent, for `Agent` tool_use blocks)
 *
 * Trace input = last user message of the initial prompt; trace output = the last
 * assistant message (thinking stripped). Per-call only: unlike Python's per-client
 * patching, history is not persisted across separate `query()` invocations.
 */

import { SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus, newUuid } from "../../schemas.js";
import { getCurrentSessionId, getCurrentUserId } from "../../tracing/session.js";
import { extractLastUserMessage } from "../../validation.js";
import { type AdapterOptions, BaseIntegrationAdapter, safeSerialize } from "../base.js";
import {
  extractModelParameters,
  extractPromptText,
  extractSystemPrompt,
  extractThinkingFromContent,
  extractTokenUsage,
  extractToolResults,
  extractToolUses,
  isThinkingOnly,
  normalizeContentToText,
  serializeToolResponse,
} from "./utils.js";

type Any = any;

const NAME = "ClaudeAgentSDK";
type QueryFn = (args: Any) => AsyncIterable<Any>;

interface RunState {
  startedAt: Date;
  chainSpanId: string;
  agentSpanId: string;
  initial: Any[];
  collected: Any[];
  llmSpans: SpanData[];
  toolSpans: Map<string, SpanData>;
  subagentSpans: Map<string, SpanData>;
  pendingThinking: string | null;
  lastLlm: SpanData | null;
  model: string | null;
  modelParams: Record<string, unknown> | null;
}

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
      yield* query(args);
      return;
    }

    const systemPrompt = extractSystemPrompt(args?.options);
    const initial: Any[] = [];
    if (systemPrompt) {
      initial.push({ role: "system", content: systemPrompt });
    }
    initial.push({ role: "user", content: extractPromptText(args?.prompt) });

    const state: RunState = {
      startedAt: new Date(),
      chainSpanId: newUuid(),
      agentSpanId: newUuid(),
      initial,
      collected: [...initial],
      llmSpans: [],
      toolSpans: new Map(),
      subagentSpans: new Map(),
      pendingThinking: null,
      lastLlm: null,
      model: null,
      modelParams: extractModelParameters(args?.options),
    };

    let errored: string | null = null;
    try {
      for await (const msg of query(args)) {
        try {
          this.handleMessage(msg, state);
        } catch {
          // never let instrumentation break iteration
        }
        yield msg;
      }
    } catch (exc) {
      errored = String((exc as Error)?.message ?? exc);
      this.submit(client, state, errored);
      throw exc;
    }
    this.submit(client, state, null);
  }

  private handleMessage(msg: Any, state: RunState): void {
    if (msg?.type === "assistant") {
      this.handleAssistant(msg, state);
    } else if (msg?.type === "user") {
      this.handleUser(msg, state);
    } else if (msg?.type === "result") {
      const usage = extractTokenUsage(msg.usage);
      if (usage && state.lastLlm) {
        state.lastLlm.tokenUsage = usage;
      }
      const now = new Date();
      for (const sub of state.subagentSpans.values()) {
        sub.endedAt = now;
        sub.status = SpanStatusCode.OK;
      }
    }
  }

  private handleAssistant(msg: Any, state: RunState): void {
    const content = msg.message?.content;
    const thinking = extractThinkingFromContent(content);

    // The SDK emits a separate thinking-only turn before the real reply; buffer
    // its reasoning and don't create a span for it.
    if (isThinkingOnly(content)) {
      state.pendingThinking = [state.pendingThinking, thinking].filter(Boolean).join("\n\n") || null;
      return;
    }

    const reasoning = [state.pendingThinking, thinking].filter(Boolean).join("\n\n") || null;
    state.pendingThinking = null;

    const text = normalizeContentToText(content) ?? "";
    const toolUses = extractToolUses(content);
    if (msg.message?.model) {
      state.model = String(msg.message.model);
    }

    // Parent to a subagent if this turn belongs to one, else the main agent.
    const parentToolUseId = msg.parent_tool_use_id;
    const parentSpanId =
      parentToolUseId && state.subagentSpans.has(parentToolUseId)
        ? state.subagentSpans.get(parentToolUseId)!.spanId
        : state.agentSpanId;

    const assistantMsg: Record<string, unknown> = { role: "assistant", content: text };
    const callTools = toolUses.filter((u) => u.name !== "Agent");
    if (callTools.length > 0) {
      assistantMsg.tool_calls = callTools.map((u) => ({ id: u.id, name: u.name, arguments: u.input }));
    }

    const llm = new SpanData({
      parentSpanId,
      name: "claude-llm",
      kind: SpanKind.LLM,
      input: { messages: state.collected.map((m) => ({ ...m })) },
      output: { messages: [assistantMsg] },
      model: state.model,
      modelParameters: state.modelParams,
      metadata: reasoning ? { reasoning_summary: reasoning } : {},
      status: SpanStatusCode.OK,
      startedAt: new Date(),
      endedAt: new Date(),
    });
    state.llmSpans.push(llm);
    state.lastLlm = llm;
    state.collected.push(assistantMsg);

    for (const use of toolUses) {
      if (use.name === "Agent") {
        const input = (use.input ?? {}) as Any;
        const subName = input.agent_name || input.description || "subagent";
        state.subagentSpans.set(
          use.id,
          new SpanData({
            parentSpanId: state.agentSpanId,
            name: String(subName),
            kind: SpanKind.AGENT,
            input: safeSerialize(use.input),
            status: SpanStatusCode.OK,
            startedAt: new Date(),
            endedAt: new Date(),
          }),
        );
      } else {
        const tool = new SpanData({
          parentSpanId: state.agentSpanId,
          name: use.name,
          kind: SpanKind.TOOL,
          input: safeSerialize(use.input),
          status: SpanStatusCode.OK,
          startedAt: new Date(),
          endedAt: new Date(),
        });
        if (use.id) {
          state.toolSpans.set(use.id, tool);
        }
      }
    }
  }

  private handleUser(msg: Any, state: RunState): void {
    for (const result of extractToolResults(msg.message?.content)) {
      const out = serializeToolResponse(result.content);
      const tool = state.toolSpans.get(result.toolUseId);
      if (tool) {
        tool.output = out;
        if (result.isError) {
          tool.status = SpanStatusCode.ERROR;
          tool.error = out;
        }
      }
      state.collected.push({ role: "tool", content: out, tool_call_id: result.toolUseId });
    }
  }

  private submit(
    client: ReturnType<BaseIntegrationAdapter["resolveClient"]>,
    state: RunState,
    error: string | null,
  ): void {
    const now = new Date();
    const isError = error !== null;
    const status = isError ? SpanStatusCode.ERROR : SpanStatusCode.OK;

    const lastAssistant = [...state.collected].reverse().find((m) => m.role === "assistant");
    const traceOutput = lastAssistant ? { messages: [{ role: "assistant", content: lastAssistant.content }] } : null;
    const chainInput = { messages: state.initial };

    const chain = new SpanData({
      spanId: state.chainSpanId,
      parentSpanId: null,
      name: NAME,
      kind: SpanKind.CHAIN,
      input: chainInput,
      output: { messages: state.collected },
      status,
      error,
      startedAt: state.startedAt,
      endedAt: now,
    });
    const agent = new SpanData({
      spanId: state.agentSpanId,
      parentSpanId: state.chainSpanId,
      name: NAME,
      kind: SpanKind.AGENT,
      input: chainInput,
      output: traceOutput,
      model: state.model,
      modelParameters: state.modelParams,
      status,
      error,
      startedAt: state.startedAt,
      endedAt: now,
    });

    const trace = new TraceData({
      name: NAME,
      status: isError ? TraceStatus.ERROR : TraceStatus.COMPLETED,
      input: extractLastUserMessage(chainInput),
      output: traceOutput,
      metadata: { ...this.metadata },
      startedAt: state.startedAt,
      endedAt: now,
      sessionId: this.sessionId ?? getCurrentSessionId(),
      userId: this.userId ?? getCurrentUserId(),
      tags: [...this.tags],
      spans: [chain, agent, ...state.llmSpans, ...state.toolSpans.values(), ...state.subagentSpans.values()],
    });
    client.logTrace(trace);
  }
}

/** Wrap the Claude Agent SDK `query` function so each call is traced. */
export function wrapClaudeAgentQuery(query: QueryFn, options: AdapterOptions = {}): QueryFn {
  return new ClaudeAgentSDKAdapter(options).wrapQuery(query);
}
