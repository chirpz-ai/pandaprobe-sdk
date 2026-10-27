import { expect, it } from "vitest";
import { Outcome, ReplayCase, type TrialCase } from "../../src/reflect/index.js";
import { Rule } from "../../src/reflect/models.js";
import { RULE, WRITE_RULE, fixture } from "./helpers.js";
const cases = () => [
  new ReplayCase({ id: "one", input: { case: 1 } }),
  new ReplayCase({ id: "two", input: { case: 2 } }),
  new ReplayCase({ id: "protected", input: { case: 3 }, protected: true }),
];
async function proposal() {
  const f = fixture();
  await f.runtime.execution({ sessionId: "author" }, async () => {
    await f.runtime.bash(WRITE_RULE);
  });
  return { ...f, p: f.runtime.proposals()[0]! };
}
it("independently validates fresh paired arms and publishes measured improvement", async () => {
  const f = await proposal();
  const executions = new Set<string>();
  const verdict = await f.runtime.validate(f.p.id, {
    cases: cases(),
    replay: async (request) => {
      expect(executions.has(request.execution.id)).toBe(false);
      executions.add(request.execution.id);
      expect(request.execution.shell.context()).toBe("");
      const excerpt = await f.runtime.bash("grep 'Exclude' /rules/*.md");
      const success = request.case.protected || excerpt.includes("merchandise");
      return new Outcome({ scores: { task_completion: success ? 1 : 0 }, success, evidence: { independent: true } });
    },
  });
  expect(verdict.outcome).toBe("accepted");
  expect(executions.size).toBe(6);
  expect(f.runtime.workspace.rules()["refund.md"]?.text).toBe(RULE);
  expect(f.server.creates).toBe(0);
});
it.each(["missing", "same", "changed", "exception", "protection"])(
  "does not accept incomplete or uncertain replay: %s",
  async (mode) => {
    const f = await proposal();
    const verdict = await f.runtime.validate(f.p.id, {
      cases: cases(),
      replay: async (request) => {
        if (mode === "exception") throw Error("host failed");
        if (mode === "changed") await f.runtime.bash("rm /rules/*.md");
        if (mode === "missing") return new Outcome();
        const success = mode === "protection" ? false : request.case.protected;
        return new Outcome({ scores: { task_completion: success ? 1 : 0 }, success });
      },
    });
    expect(verdict.outcome).toBe("pending");
    expect(f.runtime.workspace.rules()).toEqual({});
  },
);
it("rejects protected success regressions despite improvements elsewhere", async () => {
  const f = await proposal();
  const verdict = await f.runtime.validate(f.p.id, {
    cases: cases(),
    replay: async (r) => {
      const success = r.case.protected ? r.arm === "control" : r.arm === "candidate";
      return new Outcome({ scores: { task_completion: success ? 1 : 0 }, success });
    },
  });
  expect(verdict.outcome).toBe("rejected");
});
it("accepts smaller guidance only for measured non-inferiority with explicit reason", async () => {
  const f = fixture();
  const p = f.runtime.workspace.propose(
    f.runtime.workspace.state().head,
    { "rule.md": new Rule({ path: "rule.md", text: RULE }) },
    { sessionId: "seed", executionId: "seed" },
  ).proposal!;
  f.runtime.workspace.decide({
    proposal_id: p.id,
    outcome: "accepted",
    reason: "fixture seed",
    mode: "trial",
    cases: [],
    created_at: new Date().toISOString(),
  });
  await f.runtime.execution({ sessionId: "cleanup" }, async () => {
    await f.runtime.bash("rm /rules/rule.md");
  });
  const cleanup = f.runtime.proposals().find((p) => p.status === "candidate")!;
  const verdict = await f.runtime.validate(cleanup.id, {
    cases: cases(),
    replay: async () => new Outcome({ scores: { task_completion: 1 }, success: true }),
  });
  expect(verdict.outcome).toBe("accepted");
  expect(verdict.reason).toContain("no behavioral improvement claimed");
});
it("rejects malformed host evidence without coercion", () => {
  for (const success of [1, "true", {}]) expect(() => new Outcome({ success: success as boolean })).toThrow();
  for (const value of [true, "1", Number.NaN, Number.POSITIVE_INFINITY, -1, 2])
    expect(() => new Outcome({ scores: { metric: value as number } })).toThrow();
  expect(() => new ReplayCase({ id: "x", input: { bad: undefined } })).toThrow();
});
it("persists partial pending evidence on timeout and cancellation", async () => {
  const f = await proposal();
  const controller = new AbortController();
  const pending = f.runtime.validate(f.p.id, {
    cases: cases(),
    signal: controller.signal,
    replay: async () => {
      controller.abort();
      return new Promise<Outcome>(() => {});
    },
  });
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  expect(f.runtime.workspace.proposal(f.p.id).status).toBe("pending");
  expect(f.runtime.workspace.rules()).toEqual({});
});
it("requires distinct trial arms and keeps identical records idempotent", async () => {
  const f = await proposal();
  const trial = (id: string): TrialCase => ({
    case_id: id,
    base_revision: f.p.base_revision,
    input_hash: "fixture",
    protected: id === "protected",
    control: {
      execution_id: `control-${id}`,
      scores: { task_completion: id === "protected" ? 1 : 0 },
      success: id === "protected",
      evidence: {},
      candidate_ids: [],
    },
    candidate: {
      execution_id: `candidate-${id}`,
      scores: { task_completion: 1 },
      success: true,
      evidence: {},
      candidate_ids: [f.p.id],
    },
  });
  const one = trial("one");
  const first = f.runtime.validator.recordTrial(f.p.id, one);
  expect(f.runtime.validator.recordTrial(f.p.id, one)).toEqual(first);
  expect(() => f.runtime.validator.recordTrial(f.p.id, { ...one, input_hash: "changed" })).toThrow("replaced");
  f.runtime.validator.recordTrial(f.p.id, trial("two"));
  expect(f.runtime.validator.recordTrial(f.p.id, trial("protected")).outcome).toBe("accepted");
});

it("matches Python-generated decision outcomes and reasons across adverse evidence", async () => {
  const { default: fixtureData } = await import("../fixtures/reflect-parity.json");
  const f = fixture();
  const proposal = fixtureData.decision_proposal as any;
  for (const decision of fixtureData.decisions) {
    const verdict = f.runtime.validator.assess(proposal, decision.rows as TrialCase[], "replay");
    expect({ outcome: verdict.outcome, reason: verdict.reason }, decision.name).toEqual({
      outcome: decision.outcome,
      reason: decision.reason,
    });
  }
});
it("a concurrent accepted head makes an otherwise successful replay pending", async () => {
  const f = await proposal();
  let changed = false;
  const verdict = await f.runtime.validate(f.p.id, {
    cases: cases(),
    replay: async (r) => {
      if (!changed) {
        changed = true;
        const other = f.runtime.workspace.propose(
          f.p.base_revision,
          {
            "other.md": new Rule({
              path: "other.md",
              text: "# Other\nApplies to: tasks\nCheck independent evidence.\n",
            }),
          },
          { sessionId: "other", executionId: "other" },
        ).proposal!;
        f.runtime.workspace.decide({
          proposal_id: other.id,
          outcome: "accepted",
          reason: "independent fixture",
          mode: "trial",
          cases: [],
          created_at: new Date().toISOString(),
        });
      }
      return new Outcome({
        scores: { task_completion: r.case.protected || r.arm === "candidate" ? 1 : 0 },
        success: r.case.protected || r.arm === "candidate",
      });
    },
  });
  expect(verdict.outcome).toBe("pending");
  expect(verdict.reason).toContain("rebase");
  expect(Object.keys(f.runtime.workspace.rules())).toEqual(["other.md"]);
});

it("scheduled replay can acquire resources after the task boundary releases them", async () => {
  const f = fixture();
  let release!: () => void;
  const resource = new Promise<void>((resolve) => {
    release = resolve;
  });
  let armed = false;
  const { Reflect: ReflectRuntime } = await import("../../src/reflect/index.js");
  const runtime = new ReflectRuntime(f.client, {
    api: f.api,
    config: f.runtime.config,
    cases: cases(),
    replay: async (request) => {
      armed = true;
      await resource;
      return new Outcome({
        scores: { task_completion: request.case.protected || request.arm === "candidate" ? 1 : 0 },
        success: request.case.protected || request.arm === "candidate",
      });
    },
  });
  try {
    await runtime.execution({ sessionId: "resource" }, async () => {
      await runtime.bash(WRITE_RULE);
    });
    release();
    expect(await runtime.settleValidation({ timeout: 2 })).toBe(true);
    expect(armed).toBe(true);
    expect(runtime.proposals()[0]?.status).toBe("accepted");
  } finally {
    release();
    await runtime.close();
  }
});
