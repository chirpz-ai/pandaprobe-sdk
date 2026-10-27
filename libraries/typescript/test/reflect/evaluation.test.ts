import { expect, it } from "vitest";
import { digest } from "../../src/reflect/models.js";
import { pythonJson } from "../../src/reflect/serialization.js";
import { SignalGate } from "../../src/reflect/signals.js";
import { PlatformFixture, fixture } from "./helpers.js";
it("single-flights repeated evidence and persists the result across instances", async () => {
  const f = fixture();
  const id = f.server.addTrace();
  const options = { executionId: "one", sessionId: "s" };
  const a = f.runtime.submitTraces([id, id], options);
  const b = f.runtime.submitTraces([id], options);
  expect(a).toBe(b);
  expect((await a).notice).not.toBeNull();
  const reopened = fixture({}, f.server, f.workspace);
  expect((await reopened.runtime.settleEvaluation("one")).status).toBe("evaluated");
  expect(f.server.creates).toBe(1);
  expect(() => f.runtime.submitTraces([id], { ...options, completed: false })).toThrow("different evidence");
  expect((await f.runtime.submitTraces([id], { ...options, executionId: "two" })).reason).toContain("belongs");
});
it("persists unknown creates and never recreates lost responses or crash claims", async () => {
  const f = fixture();
  f.server.fault = "lost";
  const id = f.server.addTrace();
  expect((await f.runtime.submitTraces([id], { executionId: "lost", sessionId: "s" })).status).toBe(
    "creation_unknown",
  );
  const reopened = fixture({}, f.server, f.workspace);
  expect((await reopened.runtime.settleEvaluation("lost")).status).toBe("creation_unknown");
  expect(f.server.creates).toBe(1);
  const other = f.server.addTrace();
  const key = digest(pythonJson(["crash", "detect", [other], ["task_completion"]]));
  f.runtime.workspace.transaction((s) => {
    s.evaluations[key] = {
      status: "creating",
      execution_id: "crash",
      stage: "detect",
      trace_ids: [other],
      metrics: ["task_completion"],
    };
  });
  expect((await f.runtime.submitTraces([other], { executionId: "crash", sessionId: "s" })).status).toBe(
    "creation_unknown",
  );
  expect(f.server.creates).toBe(1);
});
it("resumes polling known runs after restart without spending another attempt", async () => {
  const f = fixture({ evaluationMaxRuns: 1 });
  f.server.fault = "running";
  const id = f.server.addTrace();
  expect((await f.runtime.submitTraces([id], { executionId: "run", sessionId: "s" })).status).toBe("pending");
  f.server.fault = "none";
  const reopened = fixture({ evaluationMaxRuns: 1 }, f.server, f.workspace);
  expect((await reopened.runtime.settleEvaluation("run")).status).toBe("evaluated");
  expect(f.server.creates).toBe(1);
});
it("has no default count cap, tracks daily attempts, and resets at UTC midnight", async () => {
  const f = fixture();
  f.runtime.workspace.transaction((s) => {
    s.evaluation_count = 999;
    s.evaluation_day = new Date().toISOString().slice(0, 10);
  });
  await f.runtime.submitTraces([f.server.addTrace()], { executionId: "uncapped", sessionId: "s" });
  expect(f.runtime.workspace.state().evaluation_count).toBe(1000);
  f.runtime.workspace.transaction((s) => {
    s.evaluation_day = "2000-01-01";
  });
  await f.runtime.submitTraces([f.server.addTrace()], { executionId: "next", sessionId: "s" });
  expect(f.runtime.workspace.state().evaluation_count).toBe(1);
});
it("shares an opt-in budget and preserves primary notices when diagnostics cannot run", async () => {
  const f = fixture({ evaluationMaxRuns: 1, diagnosticMetrics: ["tool_correctness"] });
  const result = await f.runtime.submitTraces([f.server.addTrace()], { executionId: "first", sessionId: "s" });
  expect(result.status).toBe("evaluated");
  expect(result.notice).not.toBeNull();
  expect(result.reason).toContain("Daily evaluation-run budget");
  const second = fixture({ evaluationMaxRuns: 1, diagnosticMetrics: ["tool_correctness"] }, f.server, f.workspace);
  const skipped = await second.runtime.submitTraces([f.server.addTrace()], { executionId: "second", sessionId: "s" });
  expect(skipped.status).toBe("skipped");
  expect(f.server.creates).toBe(1);
  f.runtime.workspace.transaction((s) => {
    s.evaluation_day = "2000-01-01";
  });
  expect((await second.runtime.settleEvaluation("second")).status).toBe("skipped");
});
it("charges failed and unknown creation attempts but not polls", async () => {
  for (const fault of ["lost", "auth"] as const) {
    const f = fixture({ evaluationMaxRuns: 1 });
    f.server.fault = fault;
    await f.runtime.submitTraces([f.server.addTrace()], { executionId: fault, sessionId: "s" });
    expect(f.runtime.workspace.state().evaluation_count).toBe(1);
    await f.runtime.settleEvaluation(fault);
    expect(f.runtime.workspace.state().evaluation_count).toBe(1);
  }
});
it("does not fabricate evidence for empty, oversized, sampled or internal activity", async () => {
  const f = fixture({ evaluationMaxTraces: 1 });
  expect((await f.runtime.submitTraces([], { executionId: "empty", sessionId: "s" })).status).toBe("insufficient");
  expect(
    (
      await f.runtime.submitTraces([f.server.addTrace(), f.server.addTrace()], {
        executionId: "large",
        sessionId: "s",
      })
    ).status,
  ).toBe("insufficient");
  f.server.metadata = { "pandaprobe.reflect": { role: "validation" } };
  expect(
    (await f.runtime.submitTraces([f.server.addTrace()], { executionId: "internal", sessionId: "s" })).status,
  ).toBe("skipped");
  const sampled = fixture({ evaluationSampleRate: 0 });
  expect(
    (await sampled.runtime.submitTraces([sampled.server.addTrace()], { executionId: "sample", sessionId: "s" }))
      .status,
  ).toBe("skipped");
  expect(f.server.creates + sampled.server.creates).toBe(0);
});
it.each(["missing", "unnormalized"])("keeps invalid measurements insufficient: %s", async (fault) => {
  const f = fixture();
  if (fault === "missing") f.server.fault = "missing";
  else f.server.score = 2;
  const result = await f.runtime.submitTraces([f.server.addTrace()], { executionId: "bad", sessionId: "s" });
  expect(result.status).toBe("insufficient");
  expect(result.notice).toBeNull();
});
it("uses threshold/regression/stall precedence and does not duplicate observations", () => {
  const f = fixture({ stallWindow: 3, cooldownSeconds: 60 });
  const gate = new SignalGate(f.runtime.workspace, f.runtime.config);
  const observe = (id: string, score: number, completed = false) =>
    gate.observe({
      executionId: id,
      sessionId: "trajectory",
      traceIds: [],
      scores: { task_completion: score },
      reasons: {},
      completed,
    });
  expect(observe("a", 0.3)).toBeNull();
  expect(observe("b", 0.3)).toBeNull();
  expect(observe("c", 0.3)?.signatures).toEqual(["task_completion:stalled"]);
  expect(observe("d", 0.1)?.signatures).toEqual(["task_completion:regression"]);
  expect(observe("e", 0.1, true)?.signatures).toEqual(["task_completion:below_threshold"]);
  expect(observe("f", 0.1, true)).toBeNull();
  expect(() =>
    gate.observe({ executionId: "x", sessionId: "s", traceIds: [], scores: {}, reasons: {}, completed: true }),
  ).toThrow("complete");
});

it("aggregates each metric's worst trace rather than averaging away a failed turn", async () => {
  const server = new PlatformFixture();
  const fetch = server.fetch;
  server.fetch = async (input, options) => {
    const response = await fetch(input, options);
    if (String(input).endsWith("/scores")) {
      const rows = (await response.json()) as Record<string, unknown>[];
      return Response.json(
        rows.map((row: any, index: number) => ({
          ...row,
          value: String(index === 0 ? 0.9 : index === 1 ? 0.4 : index === 2 ? 0.2 : 0.7),
        })),
      );
    }
    return response;
  };
  const f = fixture({ thresholds: { task_completion: 0.5, coherence: 0.5 } }, server);
  const result = await f.runtime.submitTraces([server.addTrace(), server.addTrace()], {
    executionId: "many",
    sessionId: "s",
  });
  expect(result.scores).toEqual({ task_completion: 0.2, coherence: 0.4 });
  expect(result.notice?.signatures).toHaveLength(2);
  expect(f.runtime.workspace.state().evaluation_count).toBe(1);
});
it("uses each instance configuration for the shared daily count, as Python does", async () => {
  const f = fixture({ evaluationMaxRuns: 1 });
  await f.runtime.submitTraces([f.server.addTrace()], { executionId: "one", sessionId: "s" });
  const other = fixture({ evaluationMaxRuns: 2 }, f.server, f.workspace);
  const result = await other.runtime.submitTraces([f.server.addTrace()], { executionId: "two", sessionId: "s" });
  expect(result.status).toBe("evaluated");
  expect(f.server.creates).toBe(2);
  expect(f.runtime.workspace.state()).not.toHaveProperty("evaluation_limit");
  expect(() => f.runtime.submitTraces([], { executionId: 1 as any, sessionId: "s" })).toThrow("identities");
});

it("uses Python ordered-metric dispatch fingerprints for reordered threshold maps", async () => {
  const server = new PlatformFixture();
  const original = server.fetch;
  let entered!: () => void;
  let release!: () => void;
  const started = new Promise<void>((r) => {
    entered = r;
  });
  const gate = new Promise<void>((r) => {
    release = r;
  });
  let attempts = 0;
  server.fetch = async (input, options) => {
    if (options?.method === "POST") {
      attempts++;
      if (attempts === 1) {
        entered();
        await gate;
      }
    }
    return original(input, options);
  };
  const f = fixture({ thresholds: { task_completion: 0.5, coherence: 0.5 } }, server);
  const id = server.addTrace();
  const request = { executionId: "ordered", sessionId: "s" };
  const first = f.runtime.submitTraces([id], request);
  await started;
  try {
    const second = fixture({ thresholds: { coherence: 0.5, task_completion: 0.5 } }, server, f.workspace);
    expect((await second.runtime.submitTraces([id], request)).status).toBe("evaluated");
    expect(attempts).toBe(2);
  } finally {
    release();
    await first;
  }
  expect(server.creates).toBe(2);
});
