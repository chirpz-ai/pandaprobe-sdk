/** DeepAgents CallbackHandler for PandaProbe tracing. */

import { BasePandaProbeCallbackHandler } from "../_langchain-core/callback.js";

const INTERNAL_ROOT_NAMES = new Set(["LangGraph"]);

/**
 * DeepAgents callback handler. A deep agent is a LangGraph compiled graph under
 * the hood, so the standard LangChain callback interface captures the full
 * trace tree (including sub-agent invocations). Pass via
 * `config: { callbacks: [handler] }`.
 */
export class DeepAgentsCallbackHandler extends BasePandaProbeCallbackHandler {
  readonly name = "PandaProbeDeepAgentsCallbackHandler";
  protected static DEFAULT_TRACE_NAME = "DeepAgents";

  protected filterRootChainName(name: string): string {
    return INTERNAL_ROOT_NAMES.has(name) ? "DeepAgents" : name;
  }
}
