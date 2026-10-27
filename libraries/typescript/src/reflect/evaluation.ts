import * as path from "node:path";
import {
  ApiDeadlineExceeded,
  CreationOutcomeUnknown,
  type EvaluationResult,
  InsufficientEvidence,
  type PlatformClient,
  PlatformError,
  numericValue,
} from "../api/index.js";
import { uuid } from "../api/models.js";
import { Semaphore, type Task, task } from "./async.js";
import type { ReflectConfig } from "./config.js";
import { type EvaluationOutcome, digest, evaluationOutcome, freezeEvaluation } from "./models.js";
import { json, pythonJson } from "./serialization.js";
import { clipBytes } from "./shell.js";
import { SignalGate } from "./signals.js";
import { type Workspace, atomicWrite } from "./workspace.js";
export class BudgetExceeded extends Error {}
export interface SubmissionOptions {
  executionId: string;
  sessionId: string;
  completed?: boolean;
}
export class Evaluator {
  readonly tasks = new Map<string, Task<EvaluationOutcome>>();
  private readonly slots: Semaphore;
  private readonly gate: SignalGate;
  constructor(
    private readonly api: PlatformClient,
    private readonly workspace: Workspace,
    private readonly config: ReflectConfig,
    private readonly signal: AbortSignal,
  ) {
    this.slots = new Semaphore(config.maxConcurrency);
    this.gate = new SignalGate(workspace, config);
  }
  submit(traceIds: string[], options: SubmissionOptions): Promise<EvaluationOutcome> {
    const ids = [...new Set(traceIds.map(uuid))].sort();
    const { executionId, sessionId, completed = true } = options;
    if (
      typeof executionId !== "string" ||
      !executionId ||
      typeof sessionId !== "string" ||
      !sessionId ||
      typeof completed !== "boolean"
    )
      throw new TypeError("Execution/session identities must be nonempty and completed must be boolean");
    const request = {
      trace_ids: ids,
      session_id: sessionId,
      completed,
      thresholds: { ...this.config.thresholds },
      diagnostics: [...this.config.diagnosticMetrics],
    };
    this.workspace.transaction((state) => {
      state.executions ??= Object.create(null);
      const executions = state.executions;
      const existing = executions[executionId];
      if (existing && json(existing.request) !== json(request))
        throw new TypeError("An execution ID cannot identify different evidence or policies");
      executions[executionId] ??= { request };
      state.trace_owners ??= Object.create(null);
      const owners = state.trace_owners;
      if (ids.some((id) => id in owners && owners[id] !== executionId))
        executions[executionId].outcome = evaluationOutcome(
          executionId,
          "skipped",
          "Trace evidence already belongs to another execution",
        );
      else for (const id of ids) owners[id] = executionId;
    });
    const existing = this.tasks.get(executionId);
    if (existing && !existing.done) return existing.promise;
    const record = task(Promise.resolve().then(() => this.evaluate(ids, executionId, sessionId, completed)));
    this.tasks.set(executionId, record);
    return record.promise;
  }
  private save(outcome: EvaluationOutcome): EvaluationOutcome {
    this.workspace.transaction((state) => {
      state.executions[outcome.execution_id].outcome = outcome;
    });
    this.workspace.event("evaluation_outcome", { ...outcome });
    return outcome;
  }
  private async run(
    traceIds: string[],
    metrics: string[],
    executionId: string,
    stage: string,
  ): Promise<EvaluationResult> {
    const key = digest(pythonJson([executionId, stage, traceIds, metrics]));
    const [owner, record] = this.workspace.transaction((state) => {
      const existing = state.evaluations[key];
      if (existing) return [false, existing] as const;
      const day = new Date().toISOString().slice(0, 10);
      if (state.evaluation_day !== day) {
        state.evaluation_day = day;
        state.evaluation_count = 0;
      }
      const limit = this.config.evaluationMaxRuns;
      if (limit !== null && state.evaluation_count >= limit)
        throw new BudgetExceeded(
          `Daily evaluation-run budget reached (${state.evaluation_count}/${limit} creation attempts on ${day} UTC; resets at 00:00 UTC)`,
        );
      const record = { status: "creating", execution_id: executionId, stage, trace_ids: traceIds, metrics };
      state.evaluations[key] = record;
      state.evaluation_count++;
      return [true, record as Record<string, any>] as const;
    });
    if (record.status === "completed") return record.result;
    const save = (values: Record<string, unknown>) =>
      this.workspace.transaction((state) => {
        Object.assign(state.evaluations[key], values);
      });
    if (owner) {
      try {
        const run = await this.api.evaluations.create(traceIds, metrics, {
          name: `Reflect ${stage}: ${executionId.slice(0, 128)}`,
          timeout: this.config.evaluationTimeout,
          signal: this.signal,
        });
        save({ status: "running", run_id: run.id });
        record.run_id = run.id;
      } catch (error) {
        if (error instanceof CreationOutcomeUnknown || this.signal.aborted) save({ status: "creation_unknown" });
        else save({ status: "failed", error: error instanceof Error ? error.name : "Error" });
        throw error;
      }
    } else if (!record.run_id) {
      if (record.status === "failed")
        throw new PlatformError("The original creation request failed; a new execution is required");
      throw new CreationOutcomeUnknown("A run was already dispatched; its ID is unknown. Never recreate it");
    }
    const result = await this.api.evaluations.wait(record.run_id, {
      traceIds,
      metrics,
      timeout: this.config.evaluationTimeout,
      pollInterval: this.config.pollInterval,
      signal: this.signal,
    });
    save({ status: "completed", result });
    return result;
  }
  private runIds(executionId: string): string[] {
    return Object.values(this.workspace.state().evaluations)
      .filter((row) => row.execution_id === executionId && row.run_id)
      .map((row) => row.run_id);
  }
  private async evaluate(
    ids: string[],
    executionId: string,
    sessionId: string,
    completed: boolean,
  ): Promise<EvaluationOutcome> {
    const previous = this.workspace.state().executions[executionId].outcome as EvaluationOutcome | undefined;
    if (previous && !["pending", "cancelled"].includes(previous.status)) return freezeEvaluation(previous);
    const scores: Record<string, number> = Object.create(null);
    try {
      this.signal.throwIfAborted();
      if (!ids.length)
        return this.save(
          evaluationOutcome(executionId, "insufficient", "No completed task trace references were submitted"),
        );
      if (ids.length > this.config.evaluationMaxTraces)
        return this.save(
          evaluationOutcome(executionId, "insufficient", "Execution exceeds the trace evaluation limit"),
        );
      if (Number(BigInt(`0x${digest(executionId).slice(0, 16)}`)) / 2 ** 64 >= this.config.evaluationSampleRate)
        return this.save(evaluationOutcome(executionId, "skipped", "Sampling"));
      return await this.slots.run(async () => {
        const gathered = await Promise.allSettled(
          ids.map((id) =>
            this.api.traces.waitVisible(id, {
              timeout: this.config.traceVisibilityTimeout,
              pollInterval: this.config.pollInterval,
              signal: this.signal,
            }),
          ),
        );
        const traces = gathered.map((row) => {
          if (row.status === "rejected") throw row.reason;
          return row.value;
        });
        if (
          traces.some(
            (trace) =>
              ((trace.metadata["pandaprobe.reflect"] as { role?: string } | undefined)?.role ?? "task") !== "task",
          )
        )
          return this.save(
            evaluationOutcome(executionId, "skipped", "Internal reflection/validation evidence is excluded"),
          );
        const primary = await this.run(ids, Object.keys(this.config.thresholds), executionId, "detect");
        const reasons: Record<string, string> = Object.create(null);
        for (const score of primary.scores) {
          const value = numericValue(score);
          if (value === null || value < 0 || value > 1)
            throw new TypeError("Configured signal scores must be finite and normalized to [0, 1]");
          if (!(score.name in scores) || value < scores[score.name]!) {
            scores[score.name] = value;
            reasons[score.name] = score.reason ?? "";
          }
        }
        const notice = this.gate.observe({ executionId, sessionId, traceIds: ids, scores, reasons, completed });
        const evidence: Record<string, unknown> = { trace_ids: ids, primary };
        let diagnosticReason = "";
        if (notice && this.config.diagnosticMetrics.length) {
          try {
            evidence.diagnostic = await this.run(ids, [...this.config.diagnosticMetrics], executionId, "diagnose");
          } catch (error) {
            if (this.signal.aborted) throw error;
            if (!(error instanceof PlatformError || error instanceof BudgetExceeded)) throw error;
            diagnosticReason = `Diagnostic evidence unavailable: ${error instanceof BudgetExceeded ? error.message : error.name}`;
            evidence.diagnostic_error = diagnosticReason;
          }
        }
        if (notice) {
          evidence.traces = traces;
          let body = JSON.stringify(evidence);
          const limit = this.config.evidenceMaxBytes;
          if (Buffer.byteLength(body) > limit) {
            const marker = "\n[EVIDENCE TRUNCATED: original trace/run IDs identify the complete platform records.]";
            body = clipBytes(body, limit - Buffer.byteLength(marker)) + clipBytes(marker, limit);
          }
          atomicWrite(path.join(this.workspace.path, "evidence", `${notice.id}.txt`), body);
        }
        return this.save(
          evaluationOutcome(executionId, "evaluated", diagnosticReason, {
            scores,
            notice,
            run_ids: this.runIds(executionId),
          }),
        );
      }, this.signal);
    } catch (error) {
      if (this.signal.aborted) {
        this.save(
          evaluationOutcome(executionId, "cancelled", "Evaluation wait cancelled", {
            run_ids: this.runIds(executionId),
          }),
        );
        throw error;
      }
      let status: EvaluationOutcome["status"] = "failed";
      if (error instanceof CreationOutcomeUnknown) status = "creation_unknown";
      else if (error instanceof ApiDeadlineExceeded) status = "pending";
      else if (error instanceof InsufficientEvidence || error instanceof TypeError) status = "insufficient";
      else if (error instanceof BudgetExceeded) status = "skipped";
      return this.save(
        evaluationOutcome(executionId, status, error instanceof Error ? error.message : String(error), {
          scores,
          run_ids: this.runIds(executionId),
        }),
      );
    }
  }
  resume(): Promise<EvaluationOutcome>[] {
    const tasks: Promise<EvaluationOutcome>[] = [];
    for (const [executionId, value] of Object.entries(this.workspace.state().executions ?? {})) {
      const row = value as any;
      if (row.outcome && !["pending", "cancelled"].includes(row.outcome.status)) continue;
      const request = row.request;
      if (
        json(request.thresholds) !== json(this.config.thresholds) ||
        json(request.diagnostics) !== json(this.config.diagnosticMetrics)
      )
        continue;
      tasks.push(
        this.submit(request.trace_ids, { executionId, sessionId: request.session_id, completed: request.completed }),
      );
    }
    return tasks;
  }
}
