/** Python runner.py / failure_runner.py orchestration, with Node lifecycle primitives. */
import { type Invocation, Reflect as ReflectRuntime } from "../../../src/reflect/index.js";
import { TaskInputAudit, collectDelivery } from "./delivery.js";
import type { AgentFactory, BusinessTool } from "./native.js";
import {
  DISCOVERY,
  FOLLOW_UP,
  REFUND_INSTRUCTIONS,
  RefundBatch,
  type RefundOrder,
  refundCases,
  refundPrompt,
} from "./refund.js";
import { type RunOptions, appendUsage, identity, runId, save, seconds, taskDeadline, versions } from "./reporting.js";
import {
  SUPPORT_INSTRUCTIONS,
  type Stage,
  SupportDesk,
  type Ticket,
  supportCases,
  supportPrompt,
  tickets,
} from "./scenario.js";
import { responseText } from "./telemetry.js";
export const refundTools = (batch: RefundBatch): BusinessTool[] => [
  {
    name: "lookup_refund_order",
    description: "Read the authorized standard-return order and its payment breakdown.",
    fields: { order_id: "string" },
    execute: (a) => batch.lookup(a.order_id),
  },
  {
    name: "submit_refund",
    description: "Submit this batch's one authorized refund; a rejected submission closes the batch.",
    fields: { order_id: "string", amount_cents: "integer", idempotency_key: "string" },
    execute: (a) => batch.submit(a.order_id, a.amount_cents, a.idempotency_key),
  },
];
export const supportTools = (desk: SupportDesk): BusinessTool[] => [
  {
    name: "lookup_order",
    description: "Look up a support order, including amounts paid, delivery age, and any prior resolution.",
    fields: { order_id: "string" },
    execute: (a) => desk.lookup(a.order_id),
  },
  {
    name: "search_policy",
    description: "Find return-policy articles. Check status/version: search results include an archived article.",
    fields: { query: "string" },
    execute: (a) => desk.policy(a.query),
  },
  {
    name: "resolve_ticket",
    description:
      "Resolve an authorized ticket: action is refund or escalate; cents exclude shipping; key is the order ID.",
    fields: {
      order_id: "string",
      action: "string",
      amount_cents: "integer",
      policy_version: "string",
      idempotency_key: "string",
    },
    execute: (a) => desk.resolve(a.order_id, a.action, a.amount_cents, a.policy_version, a.idempotency_key),
  },
];
function snapshot(runtime: ReflectRuntime, row: any, activeMemory = false) {
  try {
    row.reflect = runtime.report(row.execution_id);
    const state = runtime.workspace.loadExecution(row.execution_id);
    row.phase_traces = state?.phase_traces ?? {};
    row.shell_usage = state?.usage ?? {};
    if (activeMemory) row.active_memory_bytes = Buffer.byteLength(state?.context ?? "");
  } catch {
    /* Python ignores an execution not yet recorded. */
  }
}
function failure(row: any, error: unknown) {
  row.error = error instanceof Error ? error.name : "Error";
  row.error_message = error instanceof Error ? error.message : String(error);
}
function summarize(report: any) {
  return {
    task_turns: report.turns.length,
    verified_successes: report.turns.filter((r: any) => r.verification?.success === true).length,
    task_errors: report.turns.filter((r: any) => "error" in r).length,
    failure_notices: report.turns.filter((r: any) => r.reflect?.evaluation?.notice).length,
    reflection_outcomes: report.turns
      .filter((r: any) => r.reflect?.reflection)
      .map((r: any) => r.reflect.reflection.status),
    validation_outcomes: report.validations.map((r: any) => r.outcome),
    limitation: "Completion/delivery do not prove improvement. Only paired validation measures the tested change.",
  };
}
export async function supportLoop(runtime: ReflectRuntime, factory: AgentFactory, options: RunOptions = {}) {
  const framework = options.framework ?? factory.framework;
  const report: any = {
    framework,
    model: options.model ?? factory.modelName,
    run_id: runId(options),
    versions: versions(framework),
    node: process.versions.node,
    initial_revision: runtime.workspace.state().head,
    settings: {
      rounds: options.rounds ?? 1,
      max_tickets: options.maxTickets ?? null,
      max_validations: options.maxValidations ?? 2,
      task_timeout: options.taskTimeout ?? 360,
      thresholds: runtime.config.thresholds,
    },
    turns: [],
    validations: [],
    replay_arms: [],
    status: "running",
    workload: "Synthetic store; real provider calls and platform evaluations. No external business mutations.",
    workload_version: 1,
  };
  const checked = new Set<string>();
  let queue = Array.from({ length: options.rounds ?? 1 }, (_, i) => tickets(i)).flat();
  if (options.maxTickets) queue = queue.slice(0, options.maxTickets);
  const started = performance.now();
  save(options, report);
  try {
    for (const ticket of queue) {
      const desk = new SupportDesk(ticket);
      const sessionId = `reflect-support:${report.run_id}:${ticket.id}`;
      const agent = await factory(supportTools(desk), SUPPORT_INSTRUCTIONS, { validation: false });
      for (const stage of ["inspect", "resolve", "repeat"] as Stage[]) {
        desk.begin(stage);
        const message = supportPrompt(ticket, stage);
        const executionId = identity();
        const turn: any = {
          ticket: ticket.id,
          policy: ticket.policy,
          stage,
          session_id: sessionId,
          execution_id: executionId,
        };
        const tick = performance.now();
        try {
          const result = await taskDeadline(options, (signal) =>
            runtime.run(agent, message, { sessionId, executionId, signal }),
          );
          turn.reply = await responseText(result);
          turn.verification = desk.verify(turn.reply);
        } catch (error) {
          failure(turn, error);
          throw error;
        } finally {
          turn.seconds = seconds(tick);
          snapshot(runtime, turn, true);
          turn.domain_events = desk.events.slice(desk.stageStart);
          report.turns.push(turn);
          save(options, report);
        }
        const evaluation = turn.reflect?.evaluation ?? {};
        if (evaluation.status !== "evaluated")
          throw new Error(`Platform evidence unavailable: ${evaluation.status ?? "missing"}; see report`);
        for (const proposal of runtime.proposals()) {
          if (checked.has(proposal.id) || !["candidate", "pending"].includes(proposal.status)) continue;
          if (checked.size >= (options.maxValidations ?? 2)) break;
          checked.add(proposal.id);
          const verdict = await runtime.validate(proposal.id, {
            cases: supportCases(ticket.policy),
            signal: options.signal,
            replay: async (request) => {
              const isolated = new SupportDesk(request.case.input as Ticket);
              isolated.begin("resolve");
              const fresh = await factory(supportTools(isolated), SUPPORT_INSTRUCTIONS, { validation: true });
              const tick = performance.now();
              const answer = await runtime.run(fresh, supportPrompt(isolated.ticket, "resolve"), {
                sessionId: request.execution.sessionId,
              });
              const reply = await responseText(answer);
              const outcome = isolated.verify(reply);
              report.replay_arms.push({
                proposal_id: proposal.id,
                case_id: request.case.id,
                arm: request.arm,
                execution_id: request.execution.id,
                seconds: seconds(tick),
                reply,
                verification: outcome,
              });
              save(options, report);
              return outcome;
            },
          });
          report.validations.push(verdict);
          save(options, report);
        }
      }
    }
    report.status = "completed";
  } finally {
    report.seconds = seconds(started);
    report.summary = summarize(report);
    report.proposals = runtime.proposals();
    report.accepted_rules = Object.fromEntries(
      Object.entries(runtime.workspace.rules()).map(([path, rule]) => [
        path,
        { version: rule.version, bytes: Buffer.byteLength(rule.text) },
      ]),
    );
    report.workspace = runtime.workspace.path;
    await appendUsage(runtime, report);
    if (report.status !== "completed") report.status = "interrupted";
    save(options, report);
  }
  return report;
}
async function runRefundTask(
  runtime: ReflectRuntime,
  factory: AgentFactory,
  options: RunOptions,
  report: any,
  order: RefundOrder,
  stage: string,
) {
  const batch = new RefundBatch(order);
  const row: any = {
    stage,
    order: order.id,
    order_data: order,
    input: refundPrompt(order),
    execution_id: identity(),
    session_id: `reflect-refund:${stage}:${identity()}`,
  };
  report.turns.push(row);
  const started = performance.now();
  let initialRules: Record<string, string> = {};
  let taskEvents: any[] = [];
  const audit = new TaskInputAudit();
  try {
    const agent = await factory(refundTools(batch), REFUND_INSTRUCTIONS, { inputAudit: audit });
    const observed = async (request: Invocation<string>) => {
      if (request.phase === "task") {
        row.initial_active_memory_bytes = Buffer.byteLength(request.execution.shell.context());
        row.initial_candidate_ids = [...request.execution.shell.candidateIds];
        row.base_revision = request.execution.baseRevision;
        initialRules = request.execution.shell.files();
        audit.executionId = request.execution.id;
      }
      try {
        return await agent.invokeReflect(request);
      } finally {
        if (request.phase === "task") {
          audit.executionId = null;
          taskEvents = [...batch.events];
        }
      }
    };
    const result = await taskDeadline(options, (signal) =>
      runtime.run(observed, refundPrompt(order), {
        sessionId: row.session_id,
        executionId: row.execution_id,
        signal,
      }),
    );
    row.reply = await responseText(result);
    row.verification = batch.verify(row.reply);
  } catch (error) {
    failure(row, error);
    throw error;
  } finally {
    row.seconds = seconds(started);
    row.domain_events = [...batch.events];
    snapshot(runtime, row);
    row.task_delivery = await collectDelivery(
      runtime,
      row.phase_traces?.task ?? [],
      initialRules,
      taskEvents,
      audit.records,
    );
    save(options, report);
  }
  return row;
}
export async function failureLoop(runtime: ReflectRuntime, factory: AgentFactory, options: RunOptions = {}) {
  const state = runtime.workspace.state();
  if (
    Object.keys(state.executions ?? {}).length ||
    runtime.proposals().length ||
    Object.keys(runtime.workspace.rules()).length
  )
    throw new Error("The incident must start in an unused workspace");
  const framework = options.framework ?? factory.framework;
  const config = Object.fromEntries(
    Object.entries(runtime.config).map(([key, value]) => [key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`), value]),
  );
  const report: any = {
    framework,
    model: options.model ?? factory.modelName,
    node: process.versions.node,
    versions: versions(framework),
    run_id: runId(options),
    workload: "Controlled stale-runbook incident; fictional refunds, real model and platform calls",
    workload_version: 4,
    agent_instructions: REFUND_INSTRUCTIONS,
    validation_inputs: refundCases(),
    initial_revision: state.head,
    settings: { task_timeout: options.taskTimeout ?? 360, config },
    turns: [],
    validations: [],
    replay_arms: [],
    milestones: {
      initial_failure_verified: null,
      platform_failure_detected: null,
      proposal_created: null,
      independently_accepted: null,
      fresh_guidance_retrieved: null,
      fresh_task_verified: null,
    },
    status: "running",
  };
  const started = performance.now();
  const stop = (stage: string, reason: string) => {
    Object.assign(report, { status: "incomplete", stopped_at: stage, reason });
    return report;
  };
  save(options, report);
  try {
    const discovery = await runRefundTask(runtime, factory, options, report, DISCOVERY, "discovery");
    const mismatch = discovery.domain_events.some(
      (event: any) => event.result?.reason === "REFUND_AMOUNT_POLICY_MISMATCH",
    );
    report.milestones.initial_failure_verified = discovery.verification.success === false && mismatch;
    if (!report.milestones.initial_failure_verified)
      return stop("discovery", "The intended policy-mismatch failure did not occur; no failure was forced.");
    const evaluation = discovery.reflect?.evaluation ?? {};
    const detected = evaluation.status === "evaluated" && Boolean(evaluation.notice);
    report.milestones.platform_failure_detected = detected;
    if (!detected)
      return stop(
        "detection",
        [
          "The ledger failed but the platform supplied no usable failure notice.",
          `Evaluation status: ${evaluation.status ?? "unavailable"}.`,
          `Scores: ${JSON.stringify(evaluation.scores ?? {})}; thresholds: ${JSON.stringify(runtime.config.thresholds)}.`,
          "A threshold failure requires a score strictly below its threshold.",
          ...(evaluation.reason ? [`Evaluation reason: ${evaluation.reason}`] : []),
        ].join(" "),
      );
    const reflection = discovery.reflect?.reflection ?? {};
    const record = reflection.proposal;
    const proposed = ["proposed", "duplicate"].includes(reflection.status) && Boolean(record);
    report.milestones.proposal_created = proposed;
    if (!proposed)
      return stop("reflection", `No candidate from the reflection checkpoint: ${reflection.status ?? "None"}.`);
    const proposal = runtime.workspace.proposal(record.id);
    if (proposal.status !== "candidate" || proposal.execution_id !== discovery.execution_id)
      return stop("reflection", "The checkpoint did not yield a new candidate for this incident.");
    const verdict = await runtime.validate(proposal.id, {
      cases: refundCases(),
      signal: options.signal,
      replay: async (request) => {
        const batch = new RefundBatch(request.case.input as RefundOrder);
        const row: any = {
          case_id: request.case.id,
          arm: request.arm,
          protected: request.case.protected,
          execution_id: request.execution.id,
          session_id: request.execution.sessionId,
          initial_active_memory_bytes: Buffer.byteLength(request.execution.shell.context()),
        };
        report.replay_arms.push(row);
        const tick = performance.now();
        try {
          const agent = await factory(refundTools(batch), REFUND_INSTRUCTIONS, { validation: true });
          const answer = await taskDeadline(options, (signal) =>
            runtime.run(agent, refundPrompt(batch.order), { sessionId: request.execution.sessionId, signal }),
          );
          row.reply = await responseText(answer);
          const outcome = batch.verify(row.reply);
          row.verification = outcome;
          return outcome;
        } catch (error) {
          failure(row, error);
          throw error;
        } finally {
          row.seconds = seconds(tick);
          row.domain_events = [...batch.events];
          save(options, report);
        }
      },
    });
    report.validations.push(verdict);
    report.milestones.independently_accepted = verdict.outcome === "accepted";
    if (verdict.outcome !== "accepted") return stop("validation", verdict.reason);
    report.accepted_revision = runtime.workspace.state().head;
    const expected = [
      ...new Set(
        Object.values(proposal.changes)
          .filter((rule) => rule !== null)
          .map((rule) => rule.version),
      ),
    ].sort();
    report.expected_accepted_versions = expected;
    const freshRuntime = new ReflectRuntime(runtime.client, { config: runtime.config, api: runtime.api });
    let fresh: any;
    try {
      const freshFactory = await factory.forRuntime(freshRuntime);
      fresh = await runRefundTask(freshRuntime, freshFactory, options, report, FOLLOW_UP, "fresh");
    } finally {
      await freshRuntime.close();
    }
    const retrieved =
      Boolean(expected.length) &&
      expected.every((v) => fresh.task_delivery.before_submission_excerpt_versions.includes(v)) &&
      fresh.initial_active_memory_bytes === 0 &&
      fresh.initial_candidate_ids.length === 0 &&
      fresh.base_revision === report.accepted_revision;
    report.milestones.fresh_guidance_retrieved = retrieved;
    report.milestones.fresh_task_verified = fresh.verification.success;
    if (!retrieved)
      return stop(
        "fresh_delivery",
        "No verified excerpt from each accepted rule reached model input before the fresh refund attempt. " +
          "See the example-only task_delivery audit; SDK acceptance does not depend on this check.",
      );
    if (!fresh.verification.success)
      return stop("fresh_outcome", "Guidance reached the fresh task but the independent ledger check failed.");
    const freshEvaluation = fresh.reflect?.evaluation ?? {};
    if (freshEvaluation.status !== "evaluated" || freshEvaluation.notice)
      return stop("fresh_evaluation", "Fresh ledger success lacks an agreeing, completed platform evaluation.");
    report.status = "completed";
    return report;
  } finally {
    report.seconds = seconds(started);
    if (report.status === "running") report.status = "interrupted";
    report.workspace = runtime.workspace.path;
    report.proposals = runtime.proposals();
    report.accepted_rules = Object.fromEntries(
      Object.entries(runtime.workspace.rules()).map(([path, rule]) => [
        path,
        { version: rule.version, text: rule.text },
      ]),
    );
    save(options, report);
    await appendUsage(runtime, report);
    save(options, report);
  }
}
