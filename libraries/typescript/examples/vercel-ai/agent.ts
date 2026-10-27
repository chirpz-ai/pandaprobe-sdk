/** AI SDK 6/7 agent tracing. Install with make install-vercel-ai. */
import { openai } from "@ai-sdk/openai";
import { ToolLoopAgent, jsonSchema, tool, wrapLanguageModel } from "ai";
import { flush } from "pandaprobe";
import { pandaProbeMiddleware, wrapVercelAgent } from "pandaprobe/integrations/vercel-ai";

const agent = wrapVercelAgent(
  new ToolLoopAgent({
    id: "City guide",
    model: wrapLanguageModel({ model: openai("gpt-5.6-terra"), middleware: pandaProbeMiddleware() }),
    instructions: "Use the city lookup tool and report its result.",
    tools: {
      lookup_city: tool({
        description: "Look up a city in this fictional travel dataset.",
        inputSchema: jsonSchema<{ city: string }>({
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
          additionalProperties: false,
        }),
        execute: async ({ city }) => ({
          city,
          description: "A fictional destination used to demonstrate tool tracing.",
        }),
      }),
    },
  }),
);

try {
  console.log((await agent.generate({ prompt: "Look up Paris." })).text);
} finally {
  await flush();
}
