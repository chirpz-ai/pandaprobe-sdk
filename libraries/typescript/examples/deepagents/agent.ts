/**
 * DeepAgents integration — callback-based instrumentation.
 *
 * Requires: OPENAI_API_KEY
 * Install:  make install-deepagents
 * Run:     pnpm exec tsx examples/deepagents/agent.ts
 *
 * A deep agent is a LangGraph compiled graph under the hood, so the standard
 * LangChain callback interface captures the full trace tree (incl. sub-agents).
 */

import { ChatOpenAI } from "@langchain/openai";
import { type DeepAgent, createDeepAgent } from "deepagents";
import { flush } from "pandaprobe";
import { DeepAgentsCallbackHandler } from "pandaprobe/integrations/deepagents";

async function main(): Promise<void> {
  const agent: DeepAgent = createDeepAgent({
    model: new ChatOpenAI({ model: "gpt-5.6-terra", useResponsesApi: true }),
    tools: [],
    systemPrompt: "You are a helpful research assistant. Be concise.",
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
