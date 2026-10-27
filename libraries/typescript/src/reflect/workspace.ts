import * as fs from "node:fs";
import * as path from "node:path";
import type { ReflectConfig } from "./config.js";
import { nativeLocks } from "./locks.js";
import {
  type ProposalOutcome,
  Rule,
  type RuleProposal,
  type ValidationVerdict,
  digest,
  freezeProposal,
  identifier,
  proposalOutcome,
  utcNow,
} from "./models.js";
import { json } from "./serialization.js";

export class WorkspaceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}
export class RevisionConflict extends WorkspaceError {}
function syncDirectory(folder: string): void {
  if (process.platform === "win32") return;
  const fd = fs.openSync(folder, "r");
  try {
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}
export function atomicWrite(file: string, content: string): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = path.join(path.dirname(file), `.reflect-${identifier()}`);
  let fd: number | undefined;
  try {
    fd = fs.openSync(temporary, "wx", 0o600);
    fs.writeFileSync(fd, content, "utf8");
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = undefined;
    fs.renameSync(temporary, file);
    syncDirectory(path.dirname(file));
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
    fs.rmSync(temporary, { force: true });
  }
}
export interface WorkspaceState {
  schema_version: 1;
  head: string;
  proposals: Record<string, RuleProposal>;
  evaluations: Record<string, any>;
  trials: Record<string, Record<string, any>>;
  gates: Record<string, any>;
  notices: Record<string, any>;
  evaluation_count: number;
  [key: string]: any;
}
export class Workspace {
  readonly path: string;
  private readonly locks: { tryLock(fd: number): boolean; unlock(fd: number): void };
  constructor(
    folder: string,
    readonly config: ReflectConfig,
  ) {
    this.locks = nativeLocks();
    this.path = path.resolve(folder);
    fs.mkdirSync(this.path, { recursive: true });
    this.locked(() => {
      if (!fs.existsSync(path.join(this.path, "state.json"))) {
        const revisions = path.join(this.path, "revisions");
        if (fs.existsSync(revisions))
          for (const entry of fs.readdirSync(revisions)) {
            const manifest = path.join(revisions, entry, "manifest.json");
            if (fs.existsSync(manifest) && Object.keys(JSON.parse(fs.readFileSync(manifest, "utf8"))).length)
              throw new WorkspaceError(
                "Workspace state is missing but nonempty history exists; restore its authoritative state",
              );
          }
        this.writeState({
          schema_version: 1,
          head: this.writeRevision({}),
          proposals: {},
          evaluations: {},
          trials: {},
          gates: {},
          notices: {},
          evaluation_count: 0,
        });
      }
      this.readState();
    });
  }
  private locked<T>(callback: () => T): T {
    const fd = fs.openSync(path.join(this.path, ".lock"), "a+", 0o600);
    const deadline = performance.now() + 10000;
    let acquired = false;
    try {
      while (!acquired) {
        acquired = this.locks.tryLock(fd);
        if (acquired) break;
        if (performance.now() >= deadline) throw new WorkspaceError("Reflect workspace lock deadline exceeded");
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
      }
      return callback();
    } finally {
      if (acquired) this.locks.unlock(fd);
      fs.closeSync(fd);
    }
  }
  private readState(): WorkspaceState {
    try {
      const state = JSON.parse(fs.readFileSync(path.join(this.path, "state.json"), "utf8"), (_key, value) =>
        value && typeof value === "object" && !Array.isArray(value)
          ? Object.assign(Object.create(null), value)
          : value,
      );
      if (state.schema_version !== 1) throw Error();
      return state;
    } catch {
      throw new WorkspaceError("Reflect workspace state is corrupt; refusing to overwrite it");
    }
  }
  private writeState(state: WorkspaceState): void {
    atomicWrite(path.join(this.path, "state.json"), json(state));
  }
  transaction<T>(update: (state: WorkspaceState) => T): T {
    return this.locked(() => {
      const state = this.readState();
      const result = update(state);
      if (result instanceof Promise) throw new TypeError("Workspace transactions must be synchronous");
      this.writeState(state);
      return result;
    });
  }
  state(): WorkspaceState {
    return this.locked(() => this.readState());
  }
  private writeRevision(rules: Record<string, Rule>): string {
    const manifest = Object.fromEntries(Object.entries(rules).map(([name, rule]) => [name, rule.version]));
    const revision = digest(json(manifest));
    const target = path.join(this.path, "revisions", revision);
    if (fs.existsSync(target)) {
      this.rules(revision);
      return revision;
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const temporary = fs.mkdtempSync(path.join(path.dirname(target), ".revision-"));
    try {
      for (const [name, rule] of Object.entries(rules)) atomicWrite(path.join(temporary, "rules", name), rule.text);
      atomicWrite(path.join(temporary, "manifest.json"), json(manifest));
      fs.renameSync(temporary, target);
      syncDirectory(path.dirname(target));
    } finally {
      fs.rmSync(temporary, { recursive: true, force: true });
    }
    return revision;
  }
  rules(revision?: string): Record<string, Rule> {
    const id = revision ?? this.state().head;
    if (!/^[a-f0-9]{64}$/.test(id)) throw new WorkspaceError("Invalid revision");
    try {
      const folder = path.join(this.path, "revisions", id);
      const manifest = JSON.parse(fs.readFileSync(path.join(folder, "manifest.json"), "utf8"));
      if (digest(json(manifest)) !== id) throw new WorkspaceError("Revision manifest integrity check failed");
      const result: Record<string, Rule> = {};
      for (const [name, version] of Object.entries(manifest)) {
        Rule.safePath(name);
        const file = path.join(folder, "rules", name);
        let cursor = folder;
        for (const component of ["rules", ...name.split("/")]) {
          cursor = path.join(cursor, component);
          if (fs.lstatSync(cursor).isSymbolicLink()) throw new WorkspaceError("Revision contains a symbolic link");
        }
        if (!fs.statSync(file).isFile()) throw new WorkspaceError("Rule must be a regular file");
        const rule = new Rule({ path: name, text: fs.readFileSync(file, "utf8") });
        if (rule.version !== version) throw new WorkspaceError("Rule integrity check failed");
        result[name] = rule;
      }
      return result;
    } catch (error) {
      if (error instanceof WorkspaceError) throw error;
      throw new WorkspaceError(`Cannot read immutable revision: ${String(error)}`);
    }
  }
  checkRules(rules: Record<string, Rule>): void {
    if (Object.keys(rules).length > this.config.rulesMaxCount)
      throw new WorkspaceError("Rule count limit reached; consolidate or remove obsolete rules");
    let total = 0;
    for (const [name, rule] of Object.entries(rules)) {
      if (name !== rule.path) throw new WorkspaceError("Rule path and record disagree");
      const size = Buffer.byteLength(rule.text);
      if (size > this.config.ruleMaxBytes) throw new WorkspaceError(`Rule ${name} exceeds its byte budget`);
      total += size;
    }
    if (total > this.config.rulesMaxBytes) throw new WorkspaceError("Rule corpus exceeds its byte budget");
  }
  propose(
    baseRevision: string,
    rules: Record<string, Rule>,
    options: { sessionId: string; executionId: string; evidenceIds?: string[]; rationale?: string },
  ): ProposalOutcome {
    this.checkRules(rules);
    const baseline = this.rules(baseRevision);
    const changes: Record<string, Rule | null> = {};
    for (const name of [...new Set([...Object.keys(baseline), ...Object.keys(rules)])].sort())
      if (baseline[name]?.text !== rules[name]?.text) changes[name] = rules[name] ?? null;
    if (!Object.keys(changes).length) return proposalOutcome("no_proposal", "The rule revision is unchanged");
    const texts = new Map<string, string>();
    for (const [name, rule] of Object.entries(rules)) {
      const normalized = rule.text.toLowerCase().trim().split(/\s+/u).join(" ");
      const duplicate = texts.get(normalized);
      if (duplicate && (name in changes || duplicate in changes))
        return proposalOutcome("duplicate", `Guidance duplicates ${duplicate}`);
      texts.set(normalized, name);
    }
    const fingerprint = digest(json({ base: baseRevision, changes }));
    const outcome = this.transaction((state) => {
      if (state.head !== baseRevision)
        return proposalOutcome("conflict", "Accepted guidance changed; refresh before proposing");
      for (const p of Object.values(state.proposals))
        if (p.fingerprint === fingerprint)
          return proposalOutcome("duplicate", "This exact change set has already been considered", this.hydrate(p));
      const proposal: RuleProposal = {
        id: identifier(),
        base_revision: baseRevision,
        changes,
        fingerprint,
        session_id: options.sessionId,
        execution_id: options.executionId,
        evidence_ids: [...(options.evidenceIds ?? [])],
        created_at: utcNow(),
        status: "candidate",
        rationale: (options.rationale ?? "").slice(0, 4096),
        decision: null,
      };
      state.proposals[proposal.id] = proposal;
      return proposalOutcome("proposed", "", proposal);
    });
    if (outcome.status === "proposed")
      this.event("proposal_created", { proposal_id: outcome.proposal!.id, execution_id: options.executionId });
    return outcome;
  }
  private hydrate(p: RuleProposal): RuleProposal {
    const changes = Object.fromEntries(
      Object.entries(p.changes).map(([k, v]) => [k, v === null ? null : new Rule(v)]),
    );
    return freezeProposal({
      ...p,
      changes,
    });
  }
  proposal(id: string): RuleProposal {
    const row = this.state().proposals[id];
    if (!row) throw new WorkspaceError(`Unknown proposal: ${id}`);
    return this.hydrate(row);
  }
  candidateRules(proposal: RuleProposal): Record<string, Rule> {
    const rules = this.rules(proposal.base_revision);
    for (const [name, rule] of Object.entries(proposal.changes)) {
      if (rule === null) delete rules[name];
      else rules[name] = new Rule(rule);
    }
    this.checkRules(rules);
    return rules;
  }
  decide(verdict: ValidationVerdict): RuleProposal {
    const proposal = this.transaction((state) => {
      const row = state.proposals[verdict.proposal_id];
      if (!row) throw new WorkspaceError("Unknown proposal");
      const p = this.hydrate(row);
      if (["accepted", "rejected"].includes(p.status)) return p;
      if (verdict.outcome === "accepted") {
        if (state.head !== p.base_revision)
          throw new RevisionConflict("Accepted revision changed during validation; candidate must be revalidated");
        state.head = this.writeRevision(this.candidateRules(p));
      }
      const decided = { ...p, status: verdict.outcome, decision: verdict };
      state.proposals[p.id] = decided;
      return decided;
    });
    this.event("validation_decided", { ...verdict });
    return freezeProposal(proposal);
  }
  event(type: string, data: Record<string, unknown> = {}): string {
    const id = identifier();
    atomicWrite(path.join(this.path, "events", `${id}.json`), json({ id, type, created_at: utcNow(), ...data }));
    return id;
  }
  saveExecution(id: string, data: Record<string, unknown>): void {
    atomicWrite(path.join(this.path, "executions", `${digest(id)}.json`), json(data));
  }
  loadExecution(id: string): Record<string, any> | null {
    const file = path.join(this.path, "executions", `${digest(id)}.json`);
    return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : null;
  }
}
