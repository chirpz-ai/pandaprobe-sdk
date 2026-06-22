/**
 * Manual instrumentation with withTrace / withSpan callback wrappers
 * (the idiomatic analog of Python's `with` context managers).
 *
 * Run: pnpm exec tsx examples/context-managers/rag-pipeline.ts
 */

import { SpanKind, flush, withSpan, withTrace } from "pandaprobe";

async function main(): Promise<void> {
  await withTrace(
    "rag-pipeline",
    { input: { messages: [{ role: "user", content: "When was the Eiffel Tower built?" }] }, tags: ["rag"] },
    async (t) => {
      const docs = await withSpan("retrieve", { kind: SpanKind.RETRIEVER }, async (s) => {
        s.setInput({ query: "Eiffel Tower built" });
        const results = ["The Eiffel Tower was completed in 1889."];
        s.setOutput({ documents: results });
        return results;
      });

      const answer = await withSpan("generate", { kind: SpanKind.LLM, model: "gpt-4o" }, async (s) => {
        s.setInput({
          messages: [
            { role: "system", content: `Context: ${docs.join(" ")}` },
            { role: "user", content: "When was the Eiffel Tower built?" },
          ],
        });
        const out = { messages: [{ role: "assistant", content: "It was completed in 1889." }] };
        s.setOutput(out);
        s.setTokenUsage({ promptTokens: 25, completionTokens: 8 });
        return out;
      });

      t.setOutput(answer);
      console.log("Bot:", answer.messages[0]?.content);
    },
  );

  await flush();
}

main();
