import { DeadlineExceeded } from "./async.js";
import { Execution } from "./execution.js";
import { type Agent, invokerFor } from "./integrations.js";
import {
  type RuleProposal,
  type TrialArm,
  type TrialCase,
  type ValidationVerdict,
  checkedScores,
  digest,
  freezeVerdict,
  identifier,
  utcNow,
} from "./models.js";
import { Invocation, type Reflect as ReflectRuntime } from "./runtime.js";
import { json, pythonJson } from "./serialization.js";
import { RevisionConflict } from "./workspace.js";
export class ReplayCase<I = unknown> {
  readonly id: string;
  readonly input: I;
  readonly protected: boolean;
  constructor(options: { id: string; input: I; protected?: boolean }) {
    if (typeof options.id !== "string" || !options.id) throw new TypeError("Replay case identity must be nonempty");
    if (options.protected !== undefined && typeof options.protected !== "boolean")
      throw new TypeError("protected must be boolean");
    json(options.input);
    this.id = options.id;
    this.input = options.input;
    this.protected = options.protected ?? false;
    Object.freeze(this);
  }
}
export class Outcome {
  readonly scores: Record<string, number>;
  readonly success: boolean | null;
  readonly evidence: Record<string, unknown>;
  constructor(
    options: { scores?: Record<string, number>; success?: boolean | null; evidence?: Record<string, unknown> } = {},
  ) {
    if (options.success !== undefined && options.success !== null && typeof options.success !== "boolean")
      throw new TypeError("Outcome success must be a boolean or null");
    checkedScores(options.scores ?? {});
    this.scores = options.scores ?? {};
    this.success = options.success ?? null;
    json(options.evidence ?? {});
    this.evidence = options.evidence ?? {};
    Object.freeze(this);
  }
}
export class ReplayRequest<I = unknown> {
  readonly case: ReplayCase<I>;
  constructor(
    case_: ReplayCase<I>,
    readonly execution: Execution,
    readonly arm: "control" | "candidate",
    readonly signal: AbortSignal = execution.signal,
  ) {
    this.case = case_;
    Object.freeze(this);
  }
  run<R>(agent: Agent<I, R>, options: Record<string, unknown> = {}): Promise<R> {
    return invokerFor(
      agent,
      options,
    )(new Invocation(this.case.input, this.execution, "validation", undefined, undefined, this.signal));
  }
}
export type Replay<I = any> = (request: ReplayRequest<I>) => Promise<Outcome>;
export class Validator {
  constructor(private readonly runtime: ReflectRuntime) {}
  async replay(
    proposalId: string,
    cases: ReplayCase[],
    replay: Replay,
    signal?: AbortSignal,
  ): Promise<ValidationVerdict> {
    const proposal = this.runtime.workspace.proposal(proposalId);
    if (["accepted", "rejected"].includes(proposal.status)) return freezeVerdict(proposal.decision!);
    if (new Set(cases.map((c) => c.id)).size !== cases.length) throw new TypeError("Replay case IDs must be unique");
    const rows: TrialCase[] = [];
    let verdict: ValidationVerdict;
    try {
      await this.runtime.callHost(
        (active) => this.collect(proposal, cases, replay, rows, active),
        this.runtime.config.validationTimeout,
        signal,
      );
      verdict = this.assess(proposal, rows, "replay");
    } catch (error) {
      verdict = {
        proposal_id: proposalId,
        outcome: "pending",
        mode: "replay",
        reason:
          error instanceof DeadlineExceeded
            ? "Replay deadline exceeded"
            : signal?.aborted || this.runtime.signal.aborted
              ? "Replay cancelled"
              : `Replay failed: ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`,
        cases: structuredClone(rows),
        created_at: utcNow(),
      };
      const published = this.publish(verdict);
      if (signal?.aborted || this.runtime.signal.aborted) throw error;
      return published;
    }
    return this.publish(verdict);
  }
  private async collect(
    proposal: RuleProposal,
    cases: ReplayCase[],
    replay: Replay,
    rows: TrialCase[],
    signal: AbortSignal,
  ): Promise<void> {
    const baseline = this.runtime.workspace.rules(proposal.base_revision);
    const candidate = this.runtime.workspace.candidateRules(proposal);
    for (const case_ of cases) {
      signal.throwIfAborted();
      const frozen = new ReplayCase({ id: case_.id, input: case_.input, protected: case_.protected });
      const row: { -readonly [K in keyof TrialCase]: TrialCase[K] } = {
        case_id: frozen.id,
        input_hash: digest(pythonJson(frozen.input)),
        protected: frozen.protected,
        base_revision: proposal.base_revision,
      };
      rows.push(row);
      const arms: Array<"control" | "candidate"> =
        Number.parseInt(digest(frozen.id)[0]!, 16) % 2 ? ["candidate", "control"] : ["control", "candidate"];
      for (const arm of arms) {
        signal.throwIfAborted();
        const assigned = arm === "control" ? baseline : candidate;
        const execution = await Execution.create(this.runtime, {
          sessionId: `reflect:validation:${identifier()}`,
          role: "validation",
          revision: proposal.base_revision,
          rules: assigned,
          candidateIds: arm === "candidate" ? [proposal.id] : [],
          signal,
        });
        try {
          const outcome = await execution.run(() =>
            this.runtime.callHost(
              (active) =>
                replay(
                  new ReplayRequest(
                    new ReplayCase({ id: frozen.id, input: frozen.input, protected: frozen.protected }),
                    execution,
                    arm,
                    active,
                  ),
                ),
              this.runtime.config.validationTimeout,
              signal,
            ),
          );
          signal.throwIfAborted();
          if (!(outcome instanceof Outcome)) throw new TypeError("Replay must return an externally verified Outcome");
          if (json(execution.shell.rules()) !== json(assigned))
            throw new TypeError("Validation changed the assigned rule snapshot");
          row[arm] = {
            execution_id: execution.id,
            scores: { ...outcome.scores },
            success: outcome.success,
            evidence: structuredClone(outcome.evidence),
            candidate_ids: [...execution.shell.candidateIds],
          };
        } finally {
          await execution.close();
        }
      }
    }
  }
  assess(proposal: RuleProposal, rows: TrialCase[], mode: "replay" | "trial"): ValidationVerdict {
    const config = this.runtime.config;
    const verdict = (outcome: ValidationVerdict["outcome"], reason: string): ValidationVerdict =>
      freezeVerdict({
        proposal_id: proposal.id,
        outcome,
        reason,
        mode,
        cases: structuredClone(rows),
        created_at: utcNow(),
      });
    if (rows.length < config.validationMinCases || new Set(rows.map((r) => r.case_id)).size !== rows.length)
      return verdict("pending", "Insufficient distinct paired cases");
    if (!rows.some((row) => row.protected))
      return verdict("pending", "At least one protected successful case is required");
    const required = Object.keys(config.thresholds);
    const seen = new Set<string>();
    const deltas: number[] = [];
    for (const row of rows) {
      if (row.base_revision !== proposal.base_revision)
        return verdict("pending", "Case baseline does not match the proposed revision");
      if (!row.control || !row.candidate) return verdict("pending", "Missing paired outcome");
      const { control, candidate } = row;
      for (const arm of [control, candidate]) {
        if (!arm.execution_id || seen.has(arm.execution_id))
          return verdict("pending", "Execution evidence was reused across trial arms or cases");
        seen.add(arm.execution_id);
        try {
          checkedScores(arm.scores);
        } catch {
          return verdict("pending", "Missing or invalid independently verified outcomes");
        }
        if (required.some((metric) => !(metric in arm.scores)) || typeof arm.success !== "boolean")
          return verdict("pending", "Missing or invalid independently verified outcomes");
      }
      if (
        control.candidate_ids?.length ||
        !candidate.candidate_ids ||
        json(candidate.candidate_ids) !== json([proposal.id])
      )
        return verdict("pending", "Trial exposure/assignment does not match control and candidate");
      if (json(Object.keys(control.scores).sort()) !== json(Object.keys(candidate.scores).sort()))
        return verdict("pending", "Paired metric sets differ");
      if (row.protected && !control.success)
        return verdict("pending", "Protected case no longer succeeds under the frozen control");
      if (control.success && !candidate.success)
        return verdict("rejected", "Candidate regressed a successful outcome");
      for (const [name, before] of Object.entries(control.scores)) {
        const change = candidate.scores[name]! - before;
        if (change < -config.validationRegressionTolerance)
          return verdict("rejected", `Candidate regressed ${name} on case ${row.case_id}`);
        if (!row.protected) deltas.push(change);
      }
    }
    if (deltas.length && deltas.reduce((a, b) => a + b, 0) / deltas.length >= config.validationMinImprovement)
      return verdict(
        "accepted",
        "Observed paired improvement within configured regression tolerances in tested cases",
      );
    const size = (rules: ReturnType<ReflectRuntime["workspace"]["rules"]>) =>
      Object.values(rules).reduce((size, r) => size + Buffer.byteLength(r.text), 0);
    if (
      size(this.runtime.workspace.candidateRules(proposal)) <
      size(this.runtime.workspace.rules(proposal.base_revision))
    )
      return verdict(
        "accepted",
        "Reduced guidance size with measured non-inferiority; no behavioral improvement claimed",
      );
    return verdict("pending", "No sufficient measured improvement");
  }
  private publish(verdict: ValidationVerdict): ValidationVerdict {
    try {
      return freezeVerdict(this.runtime.workspace.decide(verdict).decision!);
    } catch (error) {
      if (!(error instanceof RevisionConflict)) throw error;
      const pending = {
        ...verdict,
        outcome: "pending" as const,
        reason: "Accepted guidance changed; rebase and revalidate",
      };
      return freezeVerdict(this.runtime.workspace.decide(pending).decision!);
    }
  }
  recordTrial(proposalId: string, case_: TrialCase): ValidationVerdict {
    const proposal = this.runtime.workspace.proposal(proposalId);
    if (["accepted", "rejected"].includes(proposal.status)) return freezeVerdict(proposal.decision!);
    const allowed = ["case_id", "base_revision", "input_hash", "protected", "control", "candidate"];
    if (
      !case_ ||
      Object.keys(case_).some((k) => !allowed.includes(k)) ||
      typeof case_.case_id !== "string" ||
      !case_.case_id ||
      typeof case_.base_revision !== "string" ||
      case_.base_revision.length !== 64
    )
      throw new TypeError("Invalid matched trial case");
    const inputHash = case_.input_hash === undefined ? "" : case_.input_hash;
    if (typeof inputHash !== "string") throw new TypeError("Invalid input hash");
    const boolean = (value: unknown): boolean => {
      if (typeof value === "boolean") return value;
      if (value === 1 || value === 0) return Boolean(value);
      if (typeof value === "string") {
        if (["1", "true", "t", "yes", "y", "on"].includes(value.toLowerCase())) return true;
        if (["0", "false", "f", "no", "n", "off"].includes(value.toLowerCase())) return false;
      }
      throw new TypeError("Invalid protected flag");
    };
    const arm = (value: TrialArm): TrialArm => {
      if (
        !value ||
        Object.keys(value).some(
          (k) => !["execution_id", "scores", "success", "evidence", "candidate_ids"].includes(k),
        ) ||
        typeof value.execution_id !== "string" ||
        !value.execution_id
      )
        throw new TypeError("Invalid trial arm");
      const candidateIds = value.candidate_ids === undefined ? [] : value.candidate_ids;
      if (!Array.isArray(candidateIds) || candidateIds.some((id) => typeof id !== "string"))
        throw new TypeError("Invalid candidate identities");
      const evidence = value.evidence === undefined ? {} : value.evidence;
      if (!evidence || typeof evidence !== "object" || Array.isArray(evidence))
        throw new TypeError("Invalid trial evidence");
      const scores = value.scores === undefined ? {} : value.scores;
      checkedScores(scores);
      if (value.success != null && typeof value.success !== "boolean") throw new TypeError("Invalid trial success");
      return {
        execution_id: value.execution_id,
        scores,
        evidence,
        candidate_ids: candidateIds,
        ...(value.success == null ? {} : { success: value.success }),
      } as TrialArm;
    };
    const parsed: TrialCase = {
      case_id: case_.case_id,
      base_revision: case_.base_revision,
      input_hash: inputHash,
      protected: boolean(case_.protected === undefined ? false : case_.protected),
      ...(case_.control == null ? {} : { control: arm(case_.control) }),
      ...(case_.candidate == null ? {} : { candidate: arm(case_.candidate) }),
    };
    json(parsed);
    const record = structuredClone(parsed);
    const [rows, inserted] = this.runtime.workspace.transaction((state) => {
      state.trials[proposalId] ??= {};
      const trials = state.trials[proposalId]!;
      if (trials[record.case_id] && json(trials[record.case_id]) !== json(record))
        throw new TypeError("A trial case cannot be replaced with different evidence");
      const inserted = !(record.case_id in trials);
      trials[record.case_id] = record;
      return [Object.values(trials) as TrialCase[], inserted] as const;
    });
    if (!inserted && proposal.decision) return freezeVerdict(proposal.decision);
    return this.publish(this.assess(proposal, rows, "trial"));
  }
}
