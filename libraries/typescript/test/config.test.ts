import { describe, expect, it } from "vitest";
import { envEnabled, resolveConfig } from "../src/config.js";

describe("resolveConfig", () => {
  it("uses defaults when only required fields are given", () => {
    const c = resolveConfig({ apiKey: "k", projectName: "p" });
    expect(c.endpoint).toBe("http://testserver"); // pinned by setup env
    expect(c.enabled).toBe(true);
    expect(c.batchSize).toBe(10);
    expect(c.flushInterval).toBe(5.0);
    expect(c.maxQueueSize).toBe(1000);
    expect(c.debug).toBe(false);
  });

  it("explicit options override env vars", () => {
    process.env.PANDAPROBE_BATCH_SIZE = "50";
    const c = resolveConfig({ apiKey: "k", projectName: "p", batchSize: 7 });
    expect(c.batchSize).toBe(7);
  });

  it("reads from env vars when option omitted", () => {
    process.env.PANDAPROBE_API_KEY = "envkey";
    process.env.PANDAPROBE_PROJECT_NAME = "envproj";
    process.env.PANDAPROBE_FLUSH_INTERVAL = "2.5";
    const c = resolveConfig();
    expect(c.apiKey).toBe("envkey");
    expect(c.projectName).toBe("envproj");
    expect(c.flushInterval).toBe(2.5);
  });

  it("throws when enabled but missing api key or project name", () => {
    expect(() => resolveConfig({ projectName: "p" })).toThrow(/API key is required/);
    expect(() => resolveConfig({ apiKey: "k" })).toThrow(/project name is required/);
  });

  it("does not require credentials when disabled", () => {
    const c = resolveConfig({ enabled: false });
    expect(c.enabled).toBe(false);
  });

  it("parses PANDAPROBE_ENABLED for gating", () => {
    process.env.PANDAPROBE_ENABLED = "false";
    expect(envEnabled()).toBe(false);
    process.env.PANDAPROBE_ENABLED = "true";
    expect(envEnabled()).toBe(true);
  });
});
