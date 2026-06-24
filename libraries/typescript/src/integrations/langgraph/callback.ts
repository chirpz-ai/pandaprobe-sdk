/** LangGraph CallbackHandler for PandaProbe tracing. */

import { BasePandaProbeCallbackHandler } from "../_langchain-core/callback.js";

/**
 * LangGraph callback handler that maps graph events to PandaProbe traces.
 * Pass via `config: { callbacks: [handler] }`.
 */
export class LangGraphCallbackHandler extends BasePandaProbeCallbackHandler {
  readonly name = "PandaProbeLangGraphCallbackHandler";
  protected static DEFAULT_TRACE_NAME = "LangGraph";
}
