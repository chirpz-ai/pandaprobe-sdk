import { describe, expect, it } from "vitest";
import { Client, getClient, init } from "../src/client.js";
import { flush, score } from "../src/index.js";
import { requestsTo } from "./helpers.js";

describe("client singleton", () => {
  it("init() creates and getClient() returns the same instance", () => {
    const c = init({ apiKey: "k", projectName: "p" });
    expect(getClient()).toBe(c);
  });

  it("re-init drains the previous client's buffered traces (no data loss)", async () => {
    const c1 = init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
    // Buffer a trace in c1 without flushing it.
    await c1.trace("first-client-trace").run(async () => {});
    // Replacing the global client must not drop c1's buffered trace.
    init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
    // Let the previous client's background shutdown-drain complete.
    await new Promise((r) => setTimeout(r, 200));
    const names = requestsTo("/traces").map((t) => (t.body as Record<string, unknown>).name);
    expect(names).toContain("first-client-trace");
  });

  it("auto-initializes from env vars on first getClient()", () => {
    process.env.PANDAPROBE_API_KEY = "envk";
    process.env.PANDAPROBE_PROJECT_NAME = "envp";
    const c = getClient();
    expect(c).not.toBeNull();
    expect(c?.enabled).toBe(true);
  });

  it("auto-init returns null when PANDAPROBE_ENABLED=false", () => {
    process.env.PANDAPROBE_API_KEY = "envk";
    process.env.PANDAPROBE_PROJECT_NAME = "envp";
    process.env.PANDAPROBE_ENABLED = "false";
    expect(getClient()).toBeNull();
  });

  it("auto-init returns null when credentials are missing", () => {
    expect(getClient()).toBeNull();
  });
});

describe("Client.score", () => {
  it("enqueues a programmatic score", async () => {
    init({ apiKey: "k", projectName: "p", flushInterval: 60 });
    score("trace-1", "relevance", "8.5", { reason: "good" });
    await flush();
    const reqs = requestsTo("/evaluations/trace-scores");
    expect(reqs.length).toBe(1);
    expect(reqs[0]?.body).toMatchObject({
      trace_id: "trace-1",
      name: "relevance",
      value: "8.5",
      data_type: "NUMERIC",
      source: "PROGRAMMATIC",
      reason: "good",
    });
  });
});

describe("disabled client", () => {
  it("does not send traces", async () => {
    const c = new Client({ enabled: false });
    const ctx = c.trace("t", { input: { messages: [{ role: "user", content: "x" }] } });
    await ctx.run(async () => {});
    await c.flush();
    expect(requestsTo("/traces").length).toBe(0);
  });
});
