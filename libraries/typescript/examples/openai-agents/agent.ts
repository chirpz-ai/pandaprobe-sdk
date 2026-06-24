/**
 * OpenAI Agents SDK integration — register a tracing processor.
 *
 * Install: pnpm add @openai/agents
 * Run:     pnpm exec tsx examples/openai-agents/agent.ts
 */

import { Agent, run } from "@openai/agents";
import { flush } from "pandaprobe";
import { OpenAIAgentsAdapter } from "pandaprobe/integrations/openai-agents";

async function main(): Promise<void> {
  const adapter = new OpenAIAgentsAdapter({ tags: ["openai-agents"] });
  await adapter.instrument();

  const agent = new Agent({
    name: "Assistant",
    instructions: "You are concise.",
    model: "gpt-4o-mini",
  });

  const result = await run(agent, "What is the capital of France?");
  console.log("Bot:", result.finalOutput);

  await flush();
}

main();
