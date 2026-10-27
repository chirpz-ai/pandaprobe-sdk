import { AsyncLocalStorage } from "node:async_hooks";
import * as fs from "node:fs";
import * as path from "node:path";
import { abortError } from "../api/client.js";
import { PlatformClient } from "../api/index.js";
import { type Client, getClient } from "../client.js";
import { logger } from "../logger.js";
import { DeadlineExceeded, type Task, task, waitFor } from "./async.js";
import { ReflectConfig, type ReflectOptions } from "./config.js";
import { Evaluator, type SubmissionOptions } from "./evaluation.js";
import { Execution, type ExecutionOptions, type Phase } from "./execution.js";
import {
  type Agent,
  type AgentOptions,
  type Framework,
  type NativeAgent,
  buildAgent,
  invokerFor,
  makeTool,
  resultText,
} from "./integrations.js";
import {
  type EvaluationOutcome,
  type ExecutionReport,
  type ProposalOutcome,
  type RuleProposal,
  type ValidationVerdict,
  digest,
  evaluationOutcome,
  freezeProposalOutcome,
  freezeReport,
  identifier,
  proposalOutcome,
} from "./models.js";
import { type Replay, type ReplayCase, Validator } from "./validation.js";
import { Workspace } from "./workspace.js";
export class Invocation<I = unknown, R = unknown> {
  constructor(
    readonly input: I,
    readonly execution: Execution,
    readonly phase: Phase = "task",
    readonly previousResult: R | undefined = undefined,
    readonly feedback: string | undefined = undefined,
    readonly signal: AbortSignal = execution.signal,
  ) {
    Object.freeze(this);
  }
  get instructions(): string {
    return this.execution.instructions();
  }
}
export interface ReflectInit {
  config?: ReflectConfig | ReflectOptions;
  api?: PlatformClient;
  replay?: Replay;
  cases?: ReplayCase[];
}
class ReflectRuntime {
  readonly client: Client;
  readonly config: ReflectConfig;
  readonly workspace: Workspace;
  readonly api: PlatformClient;
  readonly validator: Validator;
  private readonly current = new AsyncLocalStorage<Execution>();
  private readonly controller = new AbortController();
  private readonly evaluator: Evaluator;
  private readonly ownsApi: boolean;
  private readonly reflections = new Map<string, Task<ProposalOutcome>>();
  private readonly validations = new Map<string, Task<ValidationVerdict>>();
  private readonly hosts = new Set<Task<unknown>>();
  private readonly executions = new Set<Execution>();
  private readonly replay?: Replay;
  private readonly cases?: ReplayCase[];
  private closed = false;
  constructor(client?: Client, options: ReflectInit = {}) {
    const resolved = client ?? getClient();
    if (!resolved) throw new TypeError("Initialize PandaProbe first, or pass its Client to Reflect");
    if (Boolean(options.replay) !== Boolean(options.cases))
      throw new TypeError("Provide replay and its protected cases together");
    this.client = resolved;
    this.config = options.config instanceof ReflectConfig ? options.config : new ReflectConfig(options.config);
    this.replay = options.replay;
    this.cases = options.cases ? [...options.cases] : undefined;
    const identity = this.client.config;
    const namespace = digest(
      `${identity.endpoint.replace(/\/+$/, "")}\n${identity.projectName}\n${identity.apiKey}`,
    ).slice(0, 24);
    this.api = options.api ?? new PlatformClient(this.client);
    this.ownsApi = !options.api;
    this.workspace = new Workspace(path.join(this.config.workspace, namespace), this.config);
    this.evaluator = new Evaluator(this.api, this.workspace, this.config, this.signal);
    this.validator = new Validator(this);
  }
  get signal(): AbortSignal {
    return this.controller.signal;
  }
  /** @internal Instance-owned async context; native tools never use a global router. */
  get currentExecution(): Execution | undefined {
    return this.current.getStore();
  }
  withCurrent<T>(execution: Execution, callback: () => T): T {
    return this.current.run(execution, callback);
  }
  assertOpen(): void {
    if (this.closed) throw new Error("Reflect is closed");
  }
  trackExecution(execution: Execution): void {
    this.executions.add(execution);
  }
  untrackExecution(execution: Execution): void {
    this.executions.delete(execution);
  }
  /** The callback must encompass actual work, including consuming streams. */
  async execution<T>(options: ExecutionOptions, callback: (execution: Execution) => Promise<T>): Promise<T> {
    const execution = await Execution.create(this, options);
    return execution.run(callback);
  }
  async bash(command: string): Promise<string> {
    const execution = this.current.getStore();
    if (!execution) return "Reflect Bash requires an active reflect.run(...) or reflect.execution(...) boundary.";
    return execution.shell.execute(command, { signal: execution.signal });
  }
  tool(framework: Framework = "generic"): Promise<any> {
    return makeTool(this, framework);
  }
  agent(factory: any, options: AgentOptions): Promise<NativeAgent> {
    return buildAgent(this, factory, options);
  }
  submitTraces(traceIds: string[], options: SubmissionOptions): Promise<EvaluationOutcome> {
    this.assertOpen();
    return this.evaluator.submit(traceIds, options);
  }
  waitExport(options: { timeout?: number } = {}): Promise<void> {
    return this.client.flush(options.timeout ?? 30);
  }
  async settleEvaluation(
    executionId: string,
    options: { timeout?: number; signal?: AbortSignal } = {},
  ): Promise<EvaluationOutcome> {
    let existing = this.evaluator.tasks.get(executionId);
    if (!existing || (existing.done && existing.value && ["pending", "cancelled"].includes(existing.value.status))) {
      const row = this.workspace.state().executions?.[executionId];
      if (!row) throw new Error(`Unknown execution: ${executionId}`);
      const request = row.request;
      this.submitTraces(request.trace_ids, {
        executionId,
        sessionId: request.session_id,
        completed: request.completed,
      });
      existing = this.evaluator.tasks.get(executionId)!;
    }
    try {
      return await waitFor(existing.promise, options.timeout, options.signal);
    } catch (error) {
      if (error instanceof DeadlineExceeded) return evaluationOutcome(executionId, "pending", "Caller wait deadline");
      throw error;
    }
  }
  resumeEvaluations(): Promise<EvaluationOutcome>[] {
    this.assertOpen();
    return this.evaluator.resume();
  }
  async continueReflection(
    executionId: string,
    continuation: Agent,
    options: { input?: unknown; previousResult?: unknown; signal?: AbortSignal } = {},
  ): Promise<ProposalOutcome> {
    this.assertOpen();
    let existing = this.reflections.get(executionId);
    if (existing?.done && existing.value?.reason.startsWith("No assigned failure notice")) existing = undefined;
    if (!existing) {
      existing = task(this.reflect(executionId, invokerFor(continuation), options));
      this.reflections.set(executionId, existing);
    }
    return waitFor(existing.promise, undefined, options.signal);
  }
  private async reflect(
    executionId: string,
    continuation: (request: Invocation) => Promise<any>,
    options: { input?: unknown; previousResult?: unknown },
  ): Promise<ProposalOutcome> {
    const evaluation = await this.settleEvaluation(executionId);
    if (!evaluation.notice) return proposalOutcome("no_proposal", `No assigned failure notice (${evaluation.status})`);
    const notice = evaluation.notice;
    const [owner, record] = this.workspace.transaction((state) => {
      const row = state.executions[executionId];
      if (row.reflection) return [false, row.reflection] as const;
      row.reflection = { status: "running" };
      return [true, row.reflection] as const;
    });
    if (!owner)
      return record.outcome
        ? freezeProposalOutcome(record.outcome)
        : proposalOutcome("failed", "Reflection already claimed; interrupted work requires review");
    let execution: Execution | undefined;
    let outcome: ProposalOutcome;
    try {
      execution = await Execution.create(this, {
        sessionId: notice.session_id,
        executionId,
        restore: this.workspace.loadExecution(executionId) !== null,
        role: "reflection",
      });
      const file = path.join(this.workspace.path, "evidence", `${notice.id}.txt`);
      execution.shell.setEvidence(
        notice,
        fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "Evidence unavailable",
      );
      execution.shell.beginReflection();
      const feedback =
        "The task invocation has finished. Reflect on the external evidence in /inbox now. Judge explanations may be wrong: compare them with execution evidence. Search existing /rules, then make at most one coherent improvement or maintenance change set, or leave files unchanged if the issue is already covered, uncertain, or unactionable. Keep /context.md concise. Do not repeat task actions or poll for validation. Finish within this checkpoint.";
      const active = execution;
      const result = await active.activate("reflection", () =>
        this.callHost(
          (signal) =>
            continuation(
              new Invocation(options.input, active, "reflection", options.previousResult, feedback, signal),
            ),
          this.config.reflectionTimeout,
        ),
      );
      outcome = active.seal({ evidenceIds: [notice.id], rationale: resultText(result) });
    } catch (error) {
      outcome = proposalOutcome(
        error instanceof DeadlineExceeded
          ? "timed_out"
          : this.signal.aborted || (error as Error)?.name === "AbortError"
            ? "cancelled"
            : "failed",
        error instanceof DeadlineExceeded
          ? "Reflection checkpoint deadline exceeded"
          : error instanceof Error
            ? `${error.name}: ${error.message}`
            : String(error),
      );
    } finally {
      await execution?.close();
    }
    this.workspace.transaction((state) => {
      state.executions[executionId].reflection = { status: "completed", outcome };
    });
    this.workspace.event("reflection_outcome", { execution_id: executionId, ...outcome });
    this.scheduleValidation(outcome);
    return outcome;
  }
  /** @internal Bound host waiting, signal cancellation and observe eventual errors. */
  async callHost<T>(callback: (signal: AbortSignal) => Promise<T>, timeout: number, signal?: AbortSignal): Promise<T> {
    const controller = new AbortController();
    const abort = () => controller.abort(this.signal.reason ?? signal?.reason ?? abortError());
    this.signal.addEventListener("abort", abort, { once: true });
    signal?.addEventListener("abort", abort, { once: true });
    if (this.signal.aborted || signal?.aborted) abort();
    const record = task(
      Promise.resolve().then(() => {
        controller.signal.throwIfAborted();
        return callback(controller.signal);
      }),
    );
    this.hosts.add(record);
    void record.promise.finally(() => this.hosts.delete(record)).catch(() => {});
    try {
      return await waitFor(record.promise, timeout, controller.signal);
    } finally {
      controller.abort(abortError());
      this.signal.removeEventListener("abort", abort);
      signal?.removeEventListener("abort", abort);
    }
  }
  async settleReflection(
    executionId: string,
    options: { timeout?: number; signal?: AbortSignal } = {},
  ): Promise<ProposalOutcome> {
    const record = this.reflections.get(executionId);
    if (!record) return proposalOutcome("no_proposal", "No host continuation was scheduled");
    try {
      return await waitFor(record.promise, options.timeout, options.signal);
    } catch (error) {
      if (error instanceof DeadlineExceeded)
        return proposalOutcome("timed_out", "Caller wait deadline; checkpoint is still running");
      throw error;
    }
  }
  async run<I, R>(
    agent: Agent<I, R>,
    input: I,
    options: ExecutionOptions & { invokeOptions?: Record<string, unknown> },
  ): Promise<R> {
    const invoke = invokerFor(agent, options.invokeOptions);
    const active = this.current.getStore();
    if (active?.role === "validation") return invoke(new Invocation(input, active, "validation"));
    let execution!: Execution;
    const result = await this.execution(options, async (current) => {
      execution = current;
      return invoke(new Invocation(input, current));
    });
    try {
      await this.continueReflection(execution.id, invoke, { input, previousResult: result, signal: options.signal });
    } catch (error) {
      if (options.signal?.aborted) throw error;
      logger.debug(`Reflect settlement failed: ${String(error)}`);
    }
    return result;
  }
  report(executionId: string): ExecutionReport {
    const snapshot = this.workspace.loadExecution(executionId);
    const state = this.workspace.state();
    const record = state.executions?.[executionId];
    if (!snapshot && !record) throw new Error(`Unknown execution: ${executionId}`);
    return freezeReport({
      execution_id: executionId,
      session_id: snapshot?.session_id ?? record?.request.session_id ?? "",
      task_status: snapshot?.task_status ?? "externally_submitted",
      trace_ids: snapshot?.trace_ids ?? record?.request.trace_ids ?? [],
      evaluation: record?.outcome ?? null,
      reflection: record?.reflection?.outcome ?? null,
      proposals: Object.values(state.proposals).filter((p) => p.execution_id === executionId),
    });
  }
  proposals(): RuleProposal[] {
    return Object.keys(this.workspace.state().proposals).map((id) => this.workspace.proposal(id));
  }
  scheduleValidation(outcome: ProposalOutcome): void {
    if (outcome.status === "proposed" && outcome.proposal && this.replay && this.cases && !this.closed)
      void this.validate(outcome.proposal.id, { cases: this.cases, replay: this.replay }).catch(() => {});
  }
  validate(
    proposalId: string,
    options: { cases: ReplayCase[]; replay: Replay; signal?: AbortSignal },
  ): Promise<ValidationVerdict> {
    this.assertOpen();
    const existing = this.validations.get(proposalId);
    if (existing && !existing.done) return existing.promise;
    const record = task(
      Promise.resolve().then(() => this.validator.replay(proposalId, options.cases, options.replay, options.signal)),
    );
    this.validations.set(proposalId, record);
    return record.promise;
  }
  async settleValidation(options: { timeout?: number; signal?: AbortSignal } = {}): Promise<boolean> {
    try {
      await waitFor(
        Promise.allSettled([...this.validations.values()].map((t) => t.promise)),
        options.timeout,
        options.signal,
      );
      return true;
    } catch (error) {
      if (error instanceof DeadlineExceeded) return false;
      throw error;
    }
  }
  async trialExecution<T>(
    proposalId: string,
    options: { arm: "control" | "candidate"; sessionId?: string; signal?: AbortSignal },
    callback: (execution: Execution) => Promise<T>,
  ): Promise<T> {
    if (!["control", "candidate"].includes(options.arm)) throw new TypeError("Trial arm must be control or candidate");
    const p = this.workspace.proposal(proposalId);
    if (["accepted", "rejected"].includes(p.status))
      throw new TypeError("A decided proposal cannot start a new trial");
    const execution = await Execution.create(this, {
      sessionId: options.sessionId ?? `reflect:trial:${identifier()}`,
      signal: options.signal,
      role: "validation",
      revision: p.base_revision,
      rules: options.arm === "control" ? this.workspace.rules(p.base_revision) : this.workspace.candidateRules(p),
      candidateIds: options.arm === "candidate" ? [p.id] : [],
    });
    return execution.run(callback);
  }
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.controller.abort(abortError());
    const work = [
      ...this.evaluator.tasks.values(),
      ...this.reflections.values(),
      ...this.validations.values(),
      ...this.hosts,
    ].map((t) => t.promise);
    const shells = [...this.executions].map((e) => e.close());
    try {
      await waitFor(Promise.allSettled([...work, ...shells]), this.config.shutdownTimeout);
    } catch (error) {
      if (!(error instanceof DeadlineExceeded)) throw error;
      logger.debug("Reflect closed with host operations still cancelling");
    } finally {
      if (this.ownsApi) await this.api.close();
    }
  }
}

export { ReflectRuntime as Reflect };
