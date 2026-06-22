/**
 * DeepAgents integration — callback-based instrumentation.
 *
 * Install: pnpm add deepagents @langchain/core
 * Run:     pnpm exec tsx examples/deepagents/agent.ts
 *
 * A deep agent is a LangGraph compiled graph under the hood, so the standard
 * LangChain callback interface captures the full trace tree (incl. sub-agents).
 */

import { createDeepAgent } from "deepagents";
import { flush } from "pandaprobe";
import { DeepAgentsCallbackHandler } from "pandaprobe/integrations/deepagents";

async function main(): Promise<void> {
  const agent = createDeepAgent({
    tools: [],
    instructions: "You are a helpful research assistant. Be concise.",
  });

  const handler = new DeepAgentsCallbackHandler({ tags: ["deepagents"] });
  const result = await agent.invoke(
    { messages: [{ role: "user", content: "What is the capital of France?" }] },
    { callbacks: [handler] },
  );
  console.log("Bot:", result.messages.at(-1)?.content);

  await flush();
}

main();
