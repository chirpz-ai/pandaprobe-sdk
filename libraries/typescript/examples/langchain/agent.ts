/**
 * LangChain integration — callback-based instrumentation for LCEL chains.
 *
 * Install: pnpm add @langchain/openai @langchain/core
 * Run:     pnpm exec tsx examples/langchain/agent.ts
 */

import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { flush } from "pandaprobe";
import { LangChainCallbackHandler } from "pandaprobe/integrations/langchain";

async function main(): Promise<void> {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are concise."],
    ["user", "{question}"],
  ]);
  const chain = prompt.pipe(new ChatOpenAI({ model: "gpt-4o-mini" })).pipe(new StringOutputParser());

  const handler = new LangChainCallbackHandler({ tags: ["lcel"] });
  const answer = await chain.invoke({ question: "What is the capital of France?" }, { callbacks: [handler] });
  console.log("Bot:", answer);

  await flush();
}

main();
