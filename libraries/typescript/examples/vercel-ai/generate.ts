/**
 * Vercel AI SDK integration — wrap a language model with PandaProbe middleware.
 *
 * Install: pnpm add ai @ai-sdk/openai
 * Run:     pnpm exec tsx examples/vercel-ai/generate.ts
 */

import { openai } from "@ai-sdk/openai";
import { generateText, streamText, wrapLanguageModel } from "ai";
import { flush } from "pandaprobe";
import { pandaProbeMiddleware } from "pandaprobe/integrations/vercel-ai";

async function main(): Promise<void> {
  const model = wrapLanguageModel({
    model: openai("gpt-4o-mini"),
    middleware: pandaProbeMiddleware(),
  });

  const { text } = await generateText({
    model,
    prompt: "What is the capital of France?",
  });
  console.log("Bot:", text);

  // Streaming is instrumented too:
  const result = streamText({ model, prompt: "Count to 3." });
  for await (const delta of result.textStream) {
    process.stdout.write(delta);
  }
  process.stdout.write("\n");

  await flush();
}

main();
