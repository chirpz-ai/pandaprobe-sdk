/**
 * Anthropic wrapper — automatic LLM span instrumentation.
 *
 * Install: pnpm add @anthropic-ai/sdk
 * Run:     pnpm exec tsx examples/anthropic/chat.ts
 */

import Anthropic from "@anthropic-ai/sdk";
import { flush } from "pandaprobe";
import { wrapAnthropic } from "pandaprobe/wrappers/anthropic";

async function main(): Promise<void> {
  const client = wrapAnthropic(new Anthropic());

  const message = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 256,
    system: "You are concise.",
    messages: [{ role: "user", content: "What is the capital of France?" }],
  });
  const text = message.content.find((b) => b.type === "text");
  console.log("Bot:", text && "text" in text ? text.text : "");

  await flush();
}

main();
