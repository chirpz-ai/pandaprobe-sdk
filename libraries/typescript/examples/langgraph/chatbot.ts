/**
 * LangGraph integration — callback-based instrumentation.
 *
 * Install: pnpm add @langchain/langgraph @langchain/openai @langchain/core
 * Run:     pnpm exec tsx examples/langgraph/chatbot.ts
 */

import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { flush } from "pandaprobe";
import { LangGraphCallbackHandler } from "pandaprobe/integrations/langgraph";

async function main(): Promise<void> {
  const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("chatbot", async (state) => ({ messages: [await llm.invoke(state.messages)] }))
    .addEdge(START, "chatbot")
    .addEdge("chatbot", END)
    .compile();

  const handler = new LangGraphCallbackHandler({ tags: ["chatbot", "example"] });
  const result = await graph.invoke(
    { messages: [{ role: "user", content: "What is the capital of France?" }] },
    { callbacks: [handler] },
  );
  console.log("Bot:", result.messages.at(-1)?.content);

  await flush();
}

main();
