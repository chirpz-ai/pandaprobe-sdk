import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { expect, it } from "vitest";
import { inspectDelivery } from "../../examples/reflect/_support/delivery.js";
import { DISCOVERY, RefundBatch } from "../../examples/reflect/_support/refund.js";
import { SupportDesk, expected, tickets } from "../../examples/reflect/_support/scenario.js";
import { Rule } from "../../src/reflect/index.js";
const execute = promisify(execFile);
it("exports real traces over loopback HTTP and completes evaluation, reflection, replay and fresh use", async () => {
  const { stdout } = await execute(process.execPath, ["--import", "tsx", "examples/reflect/offline-demo.ts"], {
    timeout: 20000,
  });
  const report = JSON.parse(stdout);
  expect(report).toMatchObject({
    original_result: 5,
    validation: "accepted",
    later_result: 9,
    real_model_calls: 0,
    accepted_before_validation: [],
    trace_roles: ["reflection", "task", "validation"],
    evaluated_roles: ["task"],
  });
  expect(report.poll_counts.every((n: number) => n >= 2)).toBe(true);
}, 25000);
it("all twelve framework help entry points avoid dependency initialization, credentials and live calls", async () => {
  for (const framework of ["langchain", "langgraph", "deepagents", "openai-agents", "claude-agent-sdk", "vercel-ai"]) {
    for (const file of ["04_reflect.ts", "05_reflect_failure.ts"]) {
      const { stdout } = await execute(
        process.execPath,
        ["--import", "tsx", `examples/${framework}/${file}`, "--help"],
        { timeout: 10000, env: { ...process.env, PANDAPROBE_API_KEY: "", PANDAPROBE_PROJECT_NAME: "" } },
      );
      expect(stdout).toContain("--model NAME");
      expect(stdout).toContain("--task-timeout SECONDS");
      expect(stdout.includes("Model request timeout: 120 seconds; automatic retries disabled.")).toBe(
        framework === "langgraph" && file === "05_reflect_failure.ts",
      );
    }
  }
}, 20000);
it("independent refund verifier rejects retries, incorrect money, unread writes and false success", () => {
  const bad = new RefundBatch(DISCOVERY);
  bad.lookup(DISCOVERY.id);
  bad.submit(DISCOVERY.id, 7200, DISCOVERY.id);
  bad.submit(DISCOVERY.id, 6500, DISCOVERY.id);
  expect(bad.verify("Refund posted").success).toBe(false);
  expect(bad.ledger).toHaveLength(0);
  const unread = new RefundBatch(DISCOVERY);
  unread.submit(DISCOVERY.id, 6500, DISCOVERY.id);
  expect(unread.verify("Done").success).toBe(false);
  const good = new RefundBatch(DISCOVERY);
  good.lookup(DISCOVERY.id);
  good.submit(DISCOVERY.id, 6500, DISCOVERY.id);
  expect(good.verify("Done").success).toBe(true);
  expect(good.verify("").success).toBe(false);
});
it("support verifier enforces authorization, current policy, paid amounts and idempotency", () => {
  for (const ticket of tickets()) {
    const desk = new SupportDesk(ticket);
    const row = expected(ticket);
    desk.lookup(ticket.id);
    desk.policy("policy");
    desk.resolve(ticket.id, row.action, row.amount_cents, row.policy_version, ticket.id);
    expect(desk.verify("done").success).toBe(false);
    desk.begin("resolve");
    desk.resolve(ticket.id, row.action, row.amount_cents, row.policy_version, ticket.id);
    expect(desk.verify("done").success).toBe(true);
    desk.begin("repeat");
    desk.lookup(ticket.id);
    desk.resolve(ticket.id, row.action, row.amount_cents, row.policy_version, ticket.id);
    expect(desk.verify("done").success).toBe(true);
    expect(desk.ledger).toHaveLength(1);
  }
});
it.each(
  ["full", "numbered", "partial", "missing", "malformed", "wrong-version", "late"].flatMap((kind) =>
    [false, true].map((encoded) => ({ kind, encoded })),
  ),
)(
  "example-only retrieval audit handles $kind output (Claude encoding: $encoded) without controlling acceptance",
  ({ kind, encoded }) => {
    const rule = new Rule({
      path: "refund.md",
      text: "# Refund\nApplies to: refunds\nExclude outbound shipping; refund merchandise only.\nKeep the original order ID as the idempotency key.\n",
    });
    const output =
      kind === "full"
        ? rule.text
        : kind === "numbered"
          ? "/rules/refund.md:3:Exclude outbound shipping; refund merchandise only."
          : kind === "partial"
            ? "Exclude outbound shipping; refund merchandise only."
            : kind === "wrong-version"
              ? "# Refund\nApplies to: refunds\nRefund invoice total including shipping."
              : kind === "malformed"
                ? "/rules/refund.md:3:Exclude ... only."
                : kind === "late"
                  ? rule.text
                  : "/rules/refund.md";
    const report = inspectDelivery(
      [
        {
          trace_id: "task",
          spans: [
            {
              kind: "LLM",
              span_id: "model",
              input: encoded
                ? {
                    messages: [
                      {
                        role: "tool",
                        tool_call_id: "read-rule",
                        content: JSON.stringify([{ type: "text", text: output }]),
                      },
                    ],
                  }
                : output,
              started_at: kind === "late" ? "2026-01-01T00:00:02Z" : "2026-01-01T00:00:00Z",
            },
          ],
        },
      ],
      { "refund.md": rule.text },
      "2026-01-01T00:00:01Z",
    );
    expect(report.before_submission_excerpt_versions.length > 0).toBe(["full", "numbered"].includes(kind));
  },
);

it("does not treat arbitrary JSON, malformed tool content or tool-only output as a model-input receipt", () => {
  const rule = new Rule({ path: "refund.md", text: "# Refund\nApplies to: refunds\nExclude shipping.\n" });
  const blocks = JSON.stringify([{ type: "text", text: rule.text }]);
  for (const span of [
    { kind: "LLM", input: { role: "user", content: blocks } },
    { kind: "LLM", input: { role: "tool", content: blocks.slice(0, -1) } },
    { kind: "LLM", input: { role: "tool", content: JSON.stringify({ text: rule.text }) } },
    { kind: "LLM", input: { role: "tool", content: JSON.stringify([{ type: "image", text: rule.text }]) } },
    { kind: "TOOL", input: { role: "tool", content: blocks } },
    { kind: "LLM", input: "I read refund.md", output: { role: "tool", content: blocks } },
  ]) {
    const report = inspectDelivery(
      [{ trace_id: "t", spans: [{ ...span, started_at: "2026-01-01T00:00:00Z" }] }],
      { "refund.md": rule.text },
      "2026-01-01T00:00:01Z",
    );
    expect(report.before_submission_excerpt_versions).toEqual([]);
  }
});

it("verifies actual streamed text and keeps unavailable native billing unknown", async () => {
  const { responseText, nativeTelemetry } = await import("../../examples/reflect/_support/telemetry.js");
  expect(await responseText({ text: Promise.resolve("") })).toBe("");
  expect(new RefundBatch(DISCOVERY).verify(await responseText({ text: Promise.resolve("posted") })).success).toBe(
    false,
  );
  const measured = await nativeTelemetry(
    "vercel-ai",
    { steps: Promise.resolve([{ usage: { inputTokens: 10, outputTokens: 3 } }]) },
    new Set(),
  );
  expect(measured).toMatchObject({ model_calls: 1, tokens: { input: 10, output: 3 }, native_reported_cost_usd: null });
  expect(
    (
      await nativeTelemetry(
        "claude-agent-sdk",
        [{ type: "result", total_cost_usd: 0.02, usage: { input_tokens: 4, output_tokens: 2 } }],
        new Set(),
      )
    ).native_reported_cost_usd,
  ).toBe(0.02);
});
