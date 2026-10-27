import fs from "node:fs";
import { join } from "node:path";
import { expect, it, vi } from "vitest";
import { argumentsFor, exampleConfig, preflight } from "../../examples/reflect/_support/cli.js";
import { inspectDelivery, matchRules } from "../../examples/reflect/_support/delivery.js";
import {
  DISCOVERY,
  FOLLOW_UP,
  REFUND_INSTRUCTIONS,
  RefundBatch,
  refundCases,
  refundPrompt,
} from "../../examples/reflect/_support/refund.js";
import { saveReport } from "../../examples/reflect/_support/reporting.js";
import {
  SUPPORT_INSTRUCTIONS,
  SupportDesk,
  expected,
  supportCases,
  supportPrompt,
  tickets,
} from "../../examples/reflect/_support/scenario.js";
import { decoders, numericValue } from "../../src/api/models.js";
import { Outcome, ReflectConfig, ReplayCase } from "../../src/reflect/index.js";
import {
  Rule,
  freezeEvaluation,
  freezeNotice,
  freezeProposal,
  freezeProposalOutcome,
  freezeReport,
  freezeVerdict,
  proposalOutcome,
} from "../../src/reflect/models.js";
import { ReflectShell } from "../../src/reflect/shell.js";
import reference from "../fixtures/python-behavior.json";
import { RULE, fixture } from "./helpers.js";
const plain = (value: unknown) => JSON.parse(JSON.stringify(value));
it.each(reference.records)("matches Python $model record decoding for $input", (row) => {
  const parse = {
    RuleProposal: freezeProposal,
    ProposalOutcome: freezeProposalOutcome,
    ValidationVerdict: freezeVerdict,
    FailureNotice: freezeNotice,
    EvaluationOutcome: freezeEvaluation,
    ExecutionReport: freezeReport,
  }[row.model as "RuleProposal"];
  if (row.accepted) expect(plain(parse(row.input as any))).toEqual((row as any).value);
  else expect(() => parse(row.input as any)).toThrow();
});
it.each(reference.rules)("matches Python's Markdown normalization for $input", (row) => {
  if (row.accepted) expect(plain(new Rule(row.input))).toEqual((row as any).value);
  else expect(() => new Rule(row.input)).toThrow();
});
it("uses Python prompts, held-out inputs, example options and SDK defaults", () => {
  expect(SUPPORT_INSTRUCTIONS).toBe(reference.support_instructions);
  expect(REFUND_INSTRUCTIONS).toBe(reference.refund_instructions);
  expect(tickets(1)).toEqual(reference.round_two);
  expect(plain(refundCases())).toEqual(reference.refund_cases);
  expect([refundPrompt(DISCOVERY), refundPrompt(FOLLOW_UP)]).toEqual(reference.refund_prompts);
  for (const policy of ["returns-v1", "returns-v2"] as const)
    expect(plain(supportCases(policy))).toEqual(reference.support_cases[policy]);
  const defaults = Object.fromEntries(
    Object.entries(new ReflectConfig())
      .filter(([k]) => k !== "workspace")
      .map(([k, v]) => [k.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`), v]),
  );
  expect(defaults).toEqual(reference.config_defaults);
  vi.stubEnv("REFLECT_EXAMPLE_MODEL", undefined);
  for (const row of reference.arguments) {
    const args = argumentsFor("langchain", row.failure, row.argv);
    for (const [key, value] of Object.entries(row.value)) {
      const name = key.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
      expect((args as any)[name] ?? null, key).toEqual(value);
    }
    const config = exampleConfig(args, "example-output", row.failure);
    expect(config.thresholds).toEqual(reference.example_config.thresholds);
    expect(config.reflectionTimeout).toBe(reference.example_config.reflection_timeout);
    expect(config.validationTimeout).toBe(reference.failure_config.validation_timeout);
    expect(config.workspace).toBe(join("example-output", "workspace"));
  }
  vi.unstubAllEnvs();
});
it.each(reference.support)("matches Python's three-turn transcript for $ticket.id", (row) => {
  const desk = new SupportDesk(row.ticket);
  for (const stage of row.stages) {
    desk.begin(stage.stage as any);
    const outputs = [desk.lookup(row.ticket.id), desk.policy("current returns")];
    const target = expected(row.ticket);
    if (stage.stage !== "inspect")
      outputs.push(
        desk.resolve(row.ticket.id, target.action, target.amount_cents, target.policy_version, row.ticket.id),
      );
    expect(supportPrompt(row.ticket, stage.stage as any)).toBe(stage.prompt);
    expect(outputs).toEqual(stage.outputs);
    expect(plain(desk.verify("Recorded result."))).toEqual(stage.verification);
  }
  expect(desk.events).toEqual(row.events);
  expect(desk.ledger).toEqual(row.ledger);
});
it.each(reference.refund)("matches Python's refund service and verifier for amount $amount", (row) => {
  const batch = new RefundBatch(row.order);
  const outputs = [
    batch.lookup(row.order.id),
    batch.submit(row.order.id, row.amount as number, row.order.id),
    batch.submit(row.order.id, row.amount as number, row.order.id),
  ];
  expect(outputs).toEqual(row.outputs);
  expect(batch.events.map(({ attempted_at, ...event }) => event)).toEqual(row.events);
  expect(plain(batch.verify("Recorded result."))).toEqual(row.verification);
});
it.each(reference.audit)("matches Python's example retrieval coverage: $text", (row) => {
  expect(matchRules(row.text, row.files)).toEqual(row.coverage);
});
it("audits only LLM inputs strictly before the submission, including sub-millisecond ordering", () => {
  const files = { "refund.md": RULE };
  const at = "2026-01-01T00:00:00.000002+00:00";
  const result = inspectDelivery(
    [
      {
        trace_id: "t",
        spans: [
          { span_id: "before", kind: "LLM", input: RULE, started_at: "2026-01-01T00:00:00.000001+00:00" },
          { span_id: "same", kind: "LLM", input: RULE, started_at: at },
          { span_id: "after", kind: "LLM", input: RULE, started_at: "2026-01-01T00:00:00.000003+00:00" },
          { span_id: "tool", kind: "TOOL", input: RULE, started_at: "2025-01-01T00:00:00Z" },
        ],
      },
    ],
    files,
    at,
  );
  expect(result.receipts.map((r) => r.before_submission)).toEqual([true, false, false]);
});
it.each(reference.api_counts)("matches Python count decoding for $input", (row) => {
  const decode = () => decoders.page({ items: [], total: row.input, limit: 1, offset: 0 });
  if (row.accepted) expect(decode().total).toBe((row as any).value);
  else expect(decode).toThrow();
});
it.each(reference.numeric_scores)("matches Python numeric score parsing for $input", (row) => {
  expect(numericValue({ value: row.input, status: "SUCCESS", data_type: "NUMERIC" } as any)).toEqual(row.value);
});
it.each(reference.api_nulls)("matches Python explicit-null handling: $input", (row) => {
  if (row.accepted) expect(decoders.trace(row.input)).toEqual((row as any).value);
  else expect(() => decoders.trace(row.input)).toThrow();
});
it.each(reference.shell_results)("matches Python native shell result: $command", async (row) => {
  const shell = await ReflectShell.create(
    new ReflectConfig({ shellTimeout: 0.01, shellMaxLoopIterations: 10000000 }),
    {},
  );
  try {
    expect(await shell.execute(row.command)).toBe(row.output);
    expect(shell.context()).toBe(row.context);
  } finally {
    await shell.close();
  }
});
it("matches frozen record fields while keeping Python's mutable nested containers", async () => {
  const outcome = proposalOutcome("no_proposal");
  expect(() => {
    (outcome as any).reason = "changed";
  }).toThrow();
  const scores = { task_completion: 1 };
  const evidence = { text: "before" };
  const verified = new Outcome({ scores, success: true, evidence });
  scores.task_completion = 0.5;
  evidence.text = "after";
  expect(verified.scores.task_completion).toBe(0.5);
  expect(verified.evidence.text).toBe("after");
  const input = { task: "before" };
  const replay = new ReplayCase({ id: "case", input });
  input.task = "after";
  expect(replay.input.task).toBe("after");
  const f = fixture();
  await f.runtime.execution({ sessionId: "s", executionId: "e" }, async () =>
    f.runtime.bash("echo memory >/context.md"),
  );
  const report = f.runtime.report("e");
  expect(Object.isFrozen(report)).toBe(true);
  expect(report).not.toHaveProperty("evaluation_usage");
  report.trace_ids.push("mutable-container");
  expect(report.trace_ids.at(-1)).toBe("mutable-container");
});
it("preflights metrics and saves partial reports atomically", async () => {
  const f = fixture();
  const metrics = vi.spyOn(f.api.evaluations, "metrics").mockResolvedValue([]);
  await expect(preflight(f.runtime)).rejects.toThrow("required metrics");
  metrics.mockResolvedValue([{ name: "task_completion", category: "test", description: "test" }]);
  await preflight(f.runtime);
  const file = join(f.workspace, "report.json");
  saveReport(file, { status: "running" });
  saveReport(file, { status: "interrupted", turns: [{ error: "failure" }] });
  expect(JSON.parse(fs.readFileSync(file, "utf8"))).toMatchObject({ status: "interrupted" });
  expect(fs.existsSync(join(f.workspace, "report.tmp"))).toBe(false);
});
