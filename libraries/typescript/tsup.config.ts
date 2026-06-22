import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/wrappers/openai/index.ts",
    "src/wrappers/anthropic/index.ts",
    "src/wrappers/gemini/index.ts",
    "src/wrappers/bedrock/index.ts",
    "src/wrappers/mistral/index.ts",
    "src/integrations/langchain/index.ts",
    "src/integrations/langgraph/index.ts",
    "src/integrations/deepagents/index.ts",
    "src/integrations/claude-agent-sdk/index.ts",
    "src/integrations/openai-agents/index.ts",
    "src/integrations/vercel-ai/index.ts",
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: false,
  treeshake: true,
  outDir: "dist",
});
