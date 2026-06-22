/**
 * Gemini wrapper — automatic LLM span instrumentation.
 *
 * Install: pnpm add @google/genai
 * Run:     pnpm exec tsx examples/gemini/chat.ts
 */

import { GoogleGenAI } from "@google/genai";
import { flush } from "pandaprobe";
import { wrapGemini } from "pandaprobe/wrappers/gemini";

async function main(): Promise<void> {
  const client = wrapGemini(new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY }));

  const response = await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "What is the capital of France?",
    config: { temperature: 0.2 },
  });
  console.log("Bot:", response.text);

  await flush();
}

main();
