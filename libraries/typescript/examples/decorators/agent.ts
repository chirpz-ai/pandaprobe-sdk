/**
 * Manual instrumentation with @trace / @span method decorators.
 *
 * Run: pnpm exec tsx examples/decorators/agent.ts
 * Requires: tsconfig with experimentalDecorators (this repo's tsconfig has it).
 */

import { flush, span, trace } from "pandaprobe";
import { SpanKind } from "pandaprobe";

interface Messages {
  messages: { role: string; content: string }[];
}

class SupportAgent {
  @span({ kind: SpanKind.RETRIEVER, name: "retrieve-docs" })
  async retrieve(_query: string): Promise<string[]> {
    return ["Refund policy: 30 days.", "Shipping: 3-5 business days."];
  }

  @span({ kind: SpanKind.LLM, name: "answer" })
  async answer(_input: Messages): Promise<Messages> {
    // Pretend an LLM produced this.
    return { messages: [{ role: "assistant", content: "You can return items within 30 days." }] };
  }

  @trace({ name: "support-agent", tags: ["example", "decorators"] })
  async run(input: Messages): Promise<Messages> {
    const docs = await this.retrieve(input.messages.at(-1)?.content ?? "");
    void docs;
    return this.answer(input);
  }
}

async function main(): Promise<void> {
  const agent = new SupportAgent();
  const result = await agent.run({ messages: [{ role: "user", content: "What is your refund policy?" }] });
  console.log("Bot:", result.messages[0]?.content);
  await flush();
}

main();
