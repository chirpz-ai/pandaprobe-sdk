/**
 * Mistral wrapper — automatic LLM span instrumentation.
 *
 * Install: pnpm add @mistralai/mistralai
 * Run:     pnpm exec tsx examples/mistral/chat.ts
 */

import { Mistral } from "@mistralai/mistralai";
import { flush } from "pandaprobe";
import { wrapMistral } from "pandaprobe/wrappers/mistral";

async function main(): Promise<void> {
  const client = wrapMistral(new Mistral({ apiKey: process.env.MISTRAL_API_KEY }));

  const response = await client.chat.complete({
    model: "mistral-large-latest",
    messages: [{ role: "user", content: "What is the capital of France?" }],
  });
  const content = response.choices?.[0]?.message?.content;
  console.log("Bot:", typeof content === "string" ? content : JSON.stringify(content));

  await flush();
}

main();
