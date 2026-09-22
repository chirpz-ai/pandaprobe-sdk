/**
 * AWS Bedrock wrapper — automatic LLM span instrumentation (Converse API).
 *
 * Install: pnpm add @aws-sdk/client-bedrock-runtime
 * Run:     pnpm exec tsx examples/bedrock/converse.ts
 * Uses the standard AWS credential chain + AWS_REGION.
 * Model:   Claude Sonnet 5 via the global Bedrock inference profile.
 */

import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { flush } from "pandaprobe";
import { wrapBedrock } from "pandaprobe/wrappers/bedrock";

async function main(): Promise<void> {
  const client = wrapBedrock(new BedrockRuntimeClient({ region: process.env.AWS_REGION ?? "us-east-1" }));

  const response = await client.send(
    new ConverseCommand({
      modelId: process.env.AWS_BEDROCK_MODEL_ID ?? "global.anthropic.claude-sonnet-5",
      system: [{ text: "You are concise." }],
      messages: [{ role: "user", content: [{ text: "What is the capital of France?" }] }],
      inferenceConfig: { maxTokens: 256 },
    }),
  );
  const textBlock = response.output?.message?.content?.find((block) => "text" in block);
  console.log("Bot:", textBlock && "text" in textBlock ? textBlock.text : "");

  await flush();
}

main();
