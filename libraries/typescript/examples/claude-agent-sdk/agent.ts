/**
 * Claude Agent SDK integration — wrap the `query` function.
 *
 * Install: pnpm add @anthropic-ai/claude-agent-sdk
 * Run:     pnpm exec tsx examples/claude-agent-sdk/agent.ts
 */

import { query } from "@anthropic-ai/claude-agent-sdk";
import { flush } from "pandaprobe";
import { wrapClaudeAgentQuery } from "pandaprobe/integrations/claude-agent-sdk";

async function main(): Promise<void> {
  const tracedQuery = wrapClaudeAgentQuery(query, { tags: ["claude-agent"] });

  for await (const message of tracedQuery({
    prompt: "What is the capital of France?",
    options: { model: "claude-sonnet-4-20250514" },
  })) {
    if (message.type === "result" && "result" in message) {
      console.log("Bot:", message.result);
    }
  }

  await flush();
}

main();
