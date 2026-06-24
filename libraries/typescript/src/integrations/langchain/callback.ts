/** LangChain CallbackHandler for PandaProbe tracing. */

import { BasePandaProbeCallbackHandler } from "../_langchain-core/callback.js";

const INTERNAL_ROOT_NAMES = new Set(["LangGraph", "RunnableSequence"]);

/**
 * LangChain callback handler for `create_agent`, LCEL chains, and custom
 * Runnables. Pass via `config: { callbacks: [handler] }`.
 */
export class LangChainCallbackHandler extends BasePandaProbeCallbackHandler {
  readonly name = "PandaProbeLangChainCallbackHandler";
  protected static DEFAULT_TRACE_NAME = "LangChain";

  protected filterRootChainName(name: string): string {
    return INTERNAL_ROOT_NAMES.has(name) ? "LangChain" : name;
  }
}
