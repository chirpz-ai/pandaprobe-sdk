/**
 * OpenAI wrapper — automatic LLM span instrumentation.
 *
 * Install: pnpm add openai
 * Run:     pnpm exec tsx examples/openai/chat.ts
 */

import OpenAI from "openai";
import { flush } from "pandaprobe";
import { wrapOpenAI } from "pandaprobe/wrappers/openai";

async function main(): Promise<void> {
  const client = wrapOpenAI(new OpenAI());

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "What is the capital of France?" }],
  });
  console.log("Bot:", response.choices[0]?.message.content);

  // Streaming is instrumented too:
  const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Count to 3." }],
    stream: true,
  });
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta.content ?? "");
  }
  process.stdout.write("\n");

  await flush();
}

main();
