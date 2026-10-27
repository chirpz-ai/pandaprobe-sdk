import { createHash, randomUUID } from "node:crypto";
export const digest = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
export const identifier = (): string => randomUUID().replaceAll("-", "");
export const utcNow = (): string => new Date().toISOString().replace("Z", "+00:00");
/** Persisted records keep Python's schema-v1 field names. Host options use camelCase. */
export class Rule {
  readonly path: string;
  readonly text: string;
  constructor(value: { path: string; text: string }) {
    if (Object.keys(value).some((key) => !["path", "text"].includes(key)))
      throw new TypeError("Unexpected Rule field");
    Rule.safePath(value.path);
    if (typeof value.text !== "string") throw new TypeError("Rule text must be a string");
    const text = `${strip(value.text.replaceAll("\r\n", "\n"))}\n`;
    if (
      text.includes("\0") ||
      // biome-ignore lint/suspicious/noControlCharactersInRegex: Python's whitespace includes these separators.
      !/(?:^|\n)Applies to:[\p{White_Space}\u001c-\u001f]*[^\p{White_Space}\u001c-\u001f][^\n]+(?=\n|$)/u.test(text)
    )
      throw new TypeError("Each rule needs an 'Applies to: ...' line with its applicability");
    if (lines(text).length < 3 || !text.startsWith("# "))
      throw new TypeError("Each rule needs a Markdown title, applicability, and behavioral guidance");
    this.path = value.path;
    this.text = text;
    Object.freeze(this);
  }
  static safePath(path: string): void {
    if (typeof path !== "string" || path.length > 200 || !/^[a-zA-Z0-9_-]+(?:\/[a-zA-Z0-9_-]+)*\.md$/.test(path))
      throw new TypeError("Rule paths must be relative Markdown names without traversal or special characters");
  }
  get version(): string {
    return digest(this.text);
  }
  get scope(): string {
    return this.path.includes("/") ? this.path.slice(0, this.path.lastIndexOf("/")) : "global";
  }
  get applicability(): string {
    return strip(
      lines(this.text)
        .find((l) => l.startsWith("Applies to:"))!
        .slice(11),
    );
  }
}
export interface RuleProposal {
  readonly id: string;
  readonly base_revision: string;
  readonly changes: Record<string, Rule | null>;
  readonly fingerprint: string;
  readonly session_id: string;
  readonly execution_id: string;
  readonly evidence_ids: string[];
  readonly created_at: string;
  readonly status: "candidate" | "pending" | "accepted" | "rejected";
  readonly rationale: string;
  readonly decision: ValidationVerdict | null;
}
export interface ProposalOutcome {
  readonly status:
    | "proposed"
    | "duplicate"
    | "no_proposal"
    | "invalid"
    | "conflict"
    | "timed_out"
    | "failed"
    | "cancelled";
  readonly proposal: RuleProposal | null;
  readonly reason: string;
  readonly rationale: string;
}
export const proposalOutcome = (
  status: ProposalOutcome["status"],
  reason = "",
  proposal: RuleProposal | null = null,
  rationale = "",
): ProposalOutcome => freezeProposalOutcome({ status, reason, proposal, rationale });
export interface ValidationVerdict {
  readonly proposal_id: string;
  readonly outcome: "accepted" | "rejected" | "pending";
  readonly reason: string;
  readonly mode: "replay" | "trial";
  readonly cases: TrialCase[];
  readonly created_at: string;
}
export interface FailureNotice {
  readonly id: string;
  readonly execution_id: string;
  readonly session_id: string;
  readonly trace_ids: string[];
  readonly scores: Record<string, number>;
  readonly reasons: Record<string, string>;
  readonly signatures: string[];
  readonly created_at: string;
}
export interface EvaluationOutcome {
  readonly execution_id: string;
  readonly status: "evaluated" | "skipped" | "pending" | "insufficient" | "failed" | "creation_unknown" | "cancelled";
  readonly run_ids: string[];
  readonly scores: Record<string, number>;
  readonly notice: FailureNotice | null;
  readonly reason: string;
}
export const evaluationOutcome = (
  execution_id: string,
  status: EvaluationOutcome["status"],
  reason = "",
  rest: Partial<EvaluationOutcome> = {},
): EvaluationOutcome =>
  freezeEvaluation({ execution_id, status, reason, run_ids: [], scores: {}, notice: null, ...rest });
export interface TrialArm {
  readonly execution_id: string;
  readonly scores: Record<string, number>;
  readonly success: boolean | null;
  readonly evidence: Record<string, unknown>;
  readonly candidate_ids: string[];
}
export interface TrialCase {
  readonly case_id: string;
  readonly base_revision: string;
  readonly input_hash: string;
  readonly protected: boolean;
  readonly control?: TrialArm;
  readonly candidate?: TrialArm;
}
export interface ExecutionReport {
  readonly execution_id: string;
  readonly session_id: string;
  readonly task_status: string;
  readonly trace_ids: string[];
  readonly evaluation: EvaluationOutcome | null;
  readonly reflection: ProposalOutcome | null;
  readonly proposals: RuleProposal[];
}
/** Pydantic frozen records prohibit field assignment, while their containers remain mutable. */
export function freezeNotice(value: FailureNotice): FailureNotice {
  const r = record(value, [
    "id",
    "execution_id",
    "session_id",
    "trace_ids",
    "scores",
    "reasons",
    "signatures",
    "created_at",
  ]);
  return Object.freeze({
    id: text(defaulted(r.id, identifier)),
    execution_id: text(r.execution_id),
    session_id: text(r.session_id),
    trace_ids: array(r.trace_ids, text),
    scores: mapping(r.scores, number),
    reasons: mapping(
      defaulted(r.reasons, () => ({})),
      text,
    ),
    signatures: array(r.signatures, text),
    created_at: text(defaulted(r.created_at, utcNow)),
  });
}
export function freezeProposal(value: RuleProposal): RuleProposal {
  const r = record(value, [
    "id",
    "base_revision",
    "changes",
    "fingerprint",
    "session_id",
    "execution_id",
    "evidence_ids",
    "created_at",
    "status",
    "rationale",
    "decision",
  ]);
  return Object.freeze({
    id: text(defaulted(r.id, identifier)),
    base_revision: text(r.base_revision),
    changes: mapping(r.changes, (rule) => (rule === null ? null : rule instanceof Rule ? rule : new Rule(rule))),
    fingerprint: text(r.fingerprint),
    session_id: text(r.session_id),
    execution_id: text(r.execution_id),
    evidence_ids: array(
      defaulted(r.evidence_ids, () => []),
      text,
    ),
    created_at: text(defaulted(r.created_at, utcNow)),
    status: choice(
      defaulted(r.status, () => "candidate"),
      ["candidate", "pending", "accepted", "rejected"],
    ),
    rationale: text(defaulted(r.rationale, () => "")),
    decision: r.decision == null ? null : (object(r.decision) as unknown as ValidationVerdict),
  });
}
export function freezeProposalOutcome(value: ProposalOutcome): ProposalOutcome {
  const r = record(value, ["status", "proposal", "reason", "rationale"]);
  return Object.freeze({
    status: choice(r.status, [
      "proposed",
      "duplicate",
      "no_proposal",
      "invalid",
      "conflict",
      "timed_out",
      "failed",
      "cancelled",
    ]),
    proposal: r.proposal == null ? null : freezeProposal(r.proposal),
    reason: text(defaulted(r.reason, () => "")),
    rationale: text(defaulted(r.rationale, () => "")),
  });
}
export function freezeEvaluation(value: EvaluationOutcome): EvaluationOutcome {
  const r = record(value, ["execution_id", "status", "run_ids", "scores", "notice", "reason"]);
  return Object.freeze({
    execution_id: text(r.execution_id),
    status: choice(r.status, [
      "evaluated",
      "skipped",
      "pending",
      "insufficient",
      "failed",
      "creation_unknown",
      "cancelled",
    ]),
    run_ids: array(
      defaulted(r.run_ids, () => []),
      text,
    ),
    scores: mapping(
      defaulted(r.scores, () => ({})),
      number,
    ),
    notice: r.notice == null ? null : freezeNotice(r.notice),
    reason: text(defaulted(r.reason, () => "")),
  });
}
export function freezeVerdict(value: ValidationVerdict): ValidationVerdict {
  const r = record(value, ["proposal_id", "outcome", "reason", "mode", "cases", "created_at"]);
  return Object.freeze({
    proposal_id: text(r.proposal_id),
    outcome: choice(r.outcome, ["accepted", "rejected", "pending"]),
    reason: text(r.reason),
    mode: choice(r.mode, ["replay", "trial"]),
    cases: array(
      defaulted(r.cases, () => []),
      object,
    ) as TrialCase[],
    created_at: text(defaulted(r.created_at, utcNow)),
  });
}
export function freezeReport(value: ExecutionReport): ExecutionReport {
  const r = record(value, [
    "execution_id",
    "session_id",
    "task_status",
    "trace_ids",
    "evaluation",
    "reflection",
    "proposals",
  ]);
  return Object.freeze({
    execution_id: text(r.execution_id),
    session_id: text(r.session_id),
    task_status: text(r.task_status),
    trace_ids: array(r.trace_ids, text),
    proposals: array(
      defaulted(r.proposals, () => []),
      freezeProposal,
    ),
    evaluation: r.evaluation == null ? null : freezeEvaluation(r.evaluation),
    reflection: r.reflection == null ? null : freezeProposalOutcome(r.reflection),
  });
}
const strip = (value: string): string =>
  // biome-ignore lint/suspicious/noControlCharactersInRegex: Match Python str.strip exactly.
  value.replace(/^[\p{White_Space}\u001c-\u001f]+|[\p{White_Space}\u001c-\u001f]+$/gu, "");
const lines = (value: string): string[] =>
  value
    // biome-ignore lint/suspicious/noControlCharactersInRegex: Match Python str.splitlines exactly.
    .replace(/(?:\r\n|[\n\r\v\f\u001c-\u001e\u0085\u2028\u2029])$/, "")
    // biome-ignore lint/suspicious/noControlCharactersInRegex: Match Python str.splitlines exactly.
    .split(/\r\n|[\n\r\v\f\u001c-\u001e\u0085\u2028\u2029]/);
const object = (value: unknown): Record<string, any> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new TypeError("Expected a record object");
  return { ...value };
};
const record = (value: unknown, fields: string[]) => {
  const result = object(value);
  if (Object.keys(result).some((key) => !fields.includes(key))) throw new TypeError("Unexpected record field");
  return result;
};
const defaulted = (value: any, fallback: () => any): any => (value === undefined ? fallback() : value);
const text = (value: unknown): string => {
  if (typeof value !== "string") throw new TypeError("Expected a string");
  return value;
};
const number = (value: unknown): number => {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return Number(value);
  const normalized = typeof value === "string" ? value.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "") : "";
  if (/^[+-]?(?:nan|inf(?:inity)?)$/i.test(normalized))
    return /nan/i.test(normalized)
      ? Number.NaN
      : normalized.startsWith("-")
        ? Number.NEGATIVE_INFINITY
        : Number.POSITIVE_INFINITY;
  if (
    typeof value === "string" &&
    /^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:e[+-]?\d(?:_?\d)*)?$/i.test(normalized)
  )
    return Number(normalized.replaceAll("_", ""));
  throw new TypeError("Expected a numeric score");
};
const array = <T>(value: unknown, parse: (value: any) => T): T[] => {
  if (!Array.isArray(value)) throw new TypeError("Expected an array");
  return value.map(parse);
};
const mapping = <T>(value: unknown, parse: (value: any) => T): Record<string, T> =>
  Object.fromEntries(Object.entries(object(value)).map(([key, entry]) => [key, parse(entry)]));
const choice = <T extends string>(value: unknown, choices: T[]): T => {
  if (!choices.includes(value as T)) throw new TypeError("Invalid record status");
  return value as T;
};
export function checkedScores(scores: Record<string, number>): Record<string, number> {
  if (
    !scores ||
    typeof scores !== "object" ||
    Array.isArray(scores) ||
    Object.entries(scores).some(([k, v]) => !k || typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1)
  )
    throw new TypeError("Outcome scores must be named finite values in [0, 1]");
  return { ...scores };
}
