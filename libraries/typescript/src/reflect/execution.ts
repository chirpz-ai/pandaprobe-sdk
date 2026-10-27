import { logger } from "../logger.js";
import { waitFor } from "./async.js";
import { type ProposalOutcome, Rule, digest, identifier, proposalOutcome } from "./models.js";
import type { Reflect as ReflectRuntime } from "./runtime.js";
import { ReflectShell } from "./shell.js";
import { WorkspaceError } from "./workspace.js";
export const INSTRUCTIONS = `Reflect memory is available through reflect_bash.
Search /rules for guidance relevant to the task; read matching files as needed.
Use /context.md for concise active memory. Its current contents are incorporated
at invocation checkpoints. Follow the task's instructions over stored guidance.
At a reflection checkpoint, inspect /inbox and improve or consolidate /rules.
Each rule is a Markdown file with a '# Title', an 'Applies to: ...' line, and
actionable guidance. Add/edit/move/remove files normally; the SDK captures the
change set and independently validates it. Files are proposals, not proof of
success. /decisions contains outcomes. Avoid duplicates and discard obsolete
working guidance. Task work comes first; no polling or acknowledgement is needed.
`;
export type Phase = "task" | "reflection" | "validation";
export interface ExecutionOptions {
  sessionId: string;
  executionId?: string;
  completed?: boolean;
  signal?: AbortSignal;
}
export interface InternalExecutionOptions extends ExecutionOptions {
  role?: Phase;
  revision?: string;
  rules?: Record<string, Rule>;
  candidateIds?: string[];
  restore?: boolean;
}
export class Execution {
  readonly id: string;
  readonly sessionId: string;
  readonly completed: boolean;
  readonly signal: AbortSignal;
  baseRevision: string;
  role: Phase;
  traceIds: string[] = [];
  proposal: ProposalOutcome = proposalOutcome("no_proposal", "No checkpoint completed");
  status = "created";
  taskStatus = "created";
  phaseTraces: Partial<Record<Phase, string[]>> = {};
  shell!: ReflectShell;
  private entered = false;
  private readonly abortController = new AbortController();
  private readonly abortListeners: Array<() => void> = [];
  private constructor(
    readonly runtime: ReflectRuntime,
    private readonly options: InternalExecutionOptions,
  ) {
    if (
      !options.sessionId ||
      typeof options.sessionId !== "string" ||
      (options.executionId !== undefined && (typeof options.executionId !== "string" || !options.executionId))
    )
      throw new TypeError("Execution/session identities must be nonempty strings");
    this.id = options.executionId ?? identifier();
    this.sessionId = options.sessionId;
    this.completed = options.completed ?? true;
    if (typeof this.completed !== "boolean") throw new TypeError("completed must be boolean");
    this.role = options.role ?? "task";
    this.baseRevision = options.revision ?? runtime.workspace.state().head;
    this.signal = this.abortController.signal;
  }
  static async create(runtime: ReflectRuntime, options: InternalExecutionOptions): Promise<Execution> {
    runtime.assertOpen();
    const execution = new Execution(runtime, options);
    const prior = runtime.workspace.loadExecution(execution.id);
    let context = "";
    let rules = options.rules;
    if (prior && !options.restore)
      throw new TypeError("Execution IDs must be unique; use continueReflection to resume pending work");
    if (options.restore) {
      if (!prior || prior.session_id !== execution.sessionId)
        throw new TypeError("Cannot resume an unknown execution/session");
      execution.baseRevision = prior.base_revision;
      rules = Object.fromEntries(
        Object.entries(prior.files).map(([path, text]) => [path, new Rule({ path, text: text as string })]),
      );
      context = prior.context;
      execution.traceIds = prior.trace_ids;
      execution.taskStatus = prior.task_status ?? prior.status;
      execution.phaseTraces = prior.phase_traces ?? {};
    } else {
      if (execution.role === "task") {
        const memory = runtime.workspace.state().memory?.[digest(execution.sessionId)];
        if (memory?.revision === execution.baseRevision) context = memory.context ?? "";
      }
    }
    execution.shell = await ReflectShell.create(
      runtime.config,
      rules ?? runtime.workspace.rules(execution.baseRevision),
      {
        context,
        candidateIds: options.restore ? prior?.candidate_ids : options.candidateIds,
        usage: options.restore ? prior?.usage : undefined,
        onCheckpoint: (files, context) => execution.checkpoint(files, context),
      },
    );
    const decisions = Object.entries(runtime.workspace.state().proposals)
      .slice(-20)
      .map(([id, row]) => [id, { status: row.status, reason: row.decision?.reason ?? "" }]);
    execution.shell.setDecisions(Object.fromEntries(decisions));
    for (const signal of [runtime.signal, options.signal]) {
      if (!signal) continue;
      const abort = () => execution.abortController.abort(signal.reason);
      if (signal.aborted) abort();
      else {
        signal.addEventListener("abort", abort, { once: true });
        execution.abortListeners.push(() => signal.removeEventListener("abort", abort));
      }
    }
    runtime.trackExecution(execution);
    return execution;
  }
  checkpoint(files = this.shell.files(), context = this.shell.context()): void {
    this.runtime.workspace.saveExecution(this.id, {
      execution_id: this.id,
      session_id: this.sessionId,
      role: this.role,
      base_revision: this.baseRevision,
      files,
      context,
      active_memory_hash: digest(context),
      active_memory_unvalidated: Boolean(context),
      trace_ids: this.traceIds,
      status: this.status,
      task_status: this.taskStatus,
      phase_traces: this.phaseTraces,
      usage: this.shell.usage,
      candidate_ids: this.shell.candidateIds,
      proposal: this.proposal,
    });
  }
  instructions(): string {
    const context = this.shell.context();
    return INSTRUCTIONS + (context ? `\n<reflect_active_memory>\n${context}\n</reflect_active_memory>` : "");
  }
  seal(options: { evidenceIds?: string[]; rationale?: string } = {}): ProposalOutcome {
    const rationale = (options.rationale ?? "").slice(0, 4096);
    try {
      this.proposal = this.runtime.workspace.propose(this.baseRevision, this.shell.rules(), {
        sessionId: this.sessionId,
        executionId: this.id,
        evidenceIds: options.evidenceIds,
        rationale,
      });
      if (this.proposal.proposal)
        this.shell.candidateIds = [...new Set([...this.shell.candidateIds, this.proposal.proposal.id])];
      if (["task", "reflection"].includes(this.role) && this.proposal.status === "no_proposal")
        this.runtime.workspace.transaction((state) => {
          state.memory ??= {};
          state.memory[digest(this.sessionId)] = { revision: this.baseRevision, context: this.shell.context() };
        });
    } catch (error) {
      if (!(error instanceof TypeError || error instanceof WorkspaceError)) throw error;
      this.proposal = proposalOutcome("invalid", error.message);
    }
    this.proposal = { ...this.proposal, rationale };
    this.checkpoint();
    return this.proposal;
  }
  async activate<T>(role: Phase, callback: () => Promise<T>): Promise<T> {
    const scope = this.runtime.client.captureTraces({
      metadata: { "pandaprobe.reflect": { role, execution_id: this.id, task_session_id: this.sessionId } },
      sessionId: role === "task" ? this.sessionId : `reflect:${role}:${this.id}`,
      maxTraces: this.runtime.config.evaluationMaxTraces + 1,
    });
    const previous = this.role;
    this.role = role;
    try {
      return await this.runtime.withCurrent(this, () => scope.run(callback));
    } finally {
      this.role = previous;
      if (role === "task") this.traceIds = [...new Set([...this.traceIds, ...scope.traceIds])];
      this.phaseTraces[role] = [...new Set([...(this.phaseTraces[role] ?? []), ...scope.traceIds])];
      try {
        this.runtime.workspace.event("execution_phase", {
          execution_id: this.id,
          role,
          trace_ids: scope.traceIds,
          capture_overflow: scope.overflow,
        });
      } catch (error) {
        logger.debug(`Reflect phase record failed: ${String(error)}`);
      }
    }
  }
  async run<T>(callback: (execution: Execution) => Promise<T>): Promise<T> {
    if (this.entered) throw new Error("Execution cannot be entered twice");
    this.runtime.assertOpen();
    this.entered = true;
    this.status = "running";
    if (this.role === "task") this.taskStatus = "running";
    let failed = true;
    try {
      this.checkpoint();
      const result = await this.activate(this.role, () =>
        waitFor(
          Promise.resolve().then(() => {
            this.signal.throwIfAborted();
            return callback(this);
          }),
          undefined,
          this.signal,
        ),
      );
      failed = false;
      return result;
    } finally {
      this.status = failed ? "failed" : "completed";
      try {
        if (this.role === "task") {
          this.taskStatus = this.status;
          this.seal();
          this.runtime.scheduleValidation(this.proposal);
          void this.runtime
            .submitTraces(this.traceIds, {
              executionId: this.id,
              sessionId: this.sessionId,
              completed: this.completed,
            })
            .catch(() => {});
        } else this.checkpoint();
      } catch (error) {
        logger.debug(`Reflect could not finish execution ${this.id}: ${String(error)}`);
      } finally {
        await this.close();
      }
    }
  }
  async close(): Promise<void> {
    try {
      await this.shell.close();
    } finally {
      for (const remove of this.abortListeners.splice(0)) remove();
      this.runtime.untrackExecution(this);
    }
  }
}
