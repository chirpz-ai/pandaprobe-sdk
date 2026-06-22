/**
 * AWS Bedrock wrapper — automatic LLM span instrumentation (Converse API).
 *
 * Install: pnpm add @aws-sdk/client-bedrock-runtime
 * Run:     pnpm exec tsx examples/bedrock/converse.ts
 * Uses the standard AWS credential chain + AWS_REGION.
 */

import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { flush } from "pandaprobe";
import { wrapBedrock } from "pandaprobe/wrappers/bedrock";

async function main(): Promise<void> {
  const client = wrapBedrock(new BedrockRuntimeClient({ region: process.env.AWS_REGION ?? "us-east-1" }));

  const response = await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      system: [{ text: "You are concise." }],
      messages: [{ role: "user", content: [{ text: "What is the capital of France?" }] }],
      inferenceConfig: { temperature: 0.5, maxTokens: 256 },
    }),
  );
  const block = response.output?.message?.content?.[0];
  console.log("Bot:", block && "text" in block ? block.text : "");

  await flush();
}

main();
