import { defineConfig } from "vitest/config";
export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    setupFiles: ["./test/setup.ts"],
    include: ["test/native/**/*.test.ts"],
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});
