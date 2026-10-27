import { spawn } from "node:child_process";
import fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, expect, it } from "vitest";
import { ReflectConfig } from "../../src/reflect/config.js";
import { Rule, type ValidationVerdict, digest, utcNow } from "../../src/reflect/models.js";
import { json, pythonJson } from "../../src/reflect/serialization.js";
import { RevisionConflict, Workspace, WorkspaceError } from "../../src/reflect/workspace.js";
import fixture from "../fixtures/reflect-parity.json";
const folders: string[] = [];
const workspace = () => {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), "reflect-test-"));
  folders.push(folder);
  return new Workspace(folder, new ReflectConfig());
};
const rule = (name = "refund.md", text = "# Refund\nApplies to: refunds\nExclude shipping.") =>
  new Rule({ path: name, text });
const propose = (w: Workspace, rules: Record<string, Rule>) =>
  w.propose(w.state().head, rules, { sessionId: "s", executionId: "e" }).proposal!;
const verdict = (id: string): ValidationVerdict => ({
  proposal_id: id,
  outcome: "accepted",
  mode: "replay",
  reason: "tested",
  cases: [],
  created_at: utcNow(),
});
afterEach(() => {
  for (const folder of folders.splice(0)) fs.rmSync(folder, { recursive: true, force: true });
});
it("matches Python rule normalization, manifests, proposal and input fingerprints", () => {
  const w = workspace();
  expect(w.state().head).toBe(fixture.empty_revision);
  const rules: Record<string, Rule> = {};
  for (const record of fixture.rules) {
    const r = new Rule({ path: record.path, text: record.text });
    expect(r.version).toBe(record.version);
    expect(r.scope).toBe(record.scope);
    expect(r.applicability).toBe(record.applicability);
    rules[r.path] = r;
  }
  const p = propose(w, rules);
  expect(p.fingerprint).toBe(fixture.proposal_fingerprint);
  expect(json({ base: p.base_revision, changes: p.changes })).toBe(fixture.proposal_json);
  w.decide(verdict(p.id));
  expect(w.state().head).toBe(fixture.revision);
  expect(pythonJson(fixture.input)).toBe(fixture.input_json);
  expect(digest(pythonJson(fixture.input))).toBe(fixture.input_hash);
});
it("publishes only independent decisions, keeping frozen snapshots and stable decisions", () => {
  const w = workspace();
  const base = w.state().head;
  const p = propose(w, { "refund.md": rule() });
  expect(w.rules()).toEqual({});
  expect(w.state().head).toBe(base);
  w.decide(verdict(p.id));
  expect(w.rules()["refund.md"]?.text).toContain("shipping");
  expect(w.rules(base)).toEqual({});
  expect(w.decide({ ...verdict(p.id), outcome: "rejected" }).status).toBe("accepted");
});
it("supports edits, moves, merges, removals and exact or normalized duplicates", () => {
  const w = workspace();
  const p = propose(w, { "refund.md": rule() });
  expect(w.propose(p.base_revision, { "refund.md": rule() }, { sessionId: "s", executionId: "e2" }).status).toBe(
    "duplicate",
  );
  w.decide(verdict(p.id));
  expect(w.propose(w.state().head, w.rules(), { sessionId: "s", executionId: "e" }).status).toBe("no_proposal");
  expect(
    w.propose(w.state().head, { ...w.rules(), "another.md": rule("another.md") }, { sessionId: "s", executionId: "e" })
      .status,
  ).toBe("duplicate");
  const move = propose(w, { "archive/refund.md": rule("archive/refund.md") });
  expect(move.changes["refund.md"]).toBeNull();
  w.decide(verdict(move.id));
  const remove = propose(w, {});
  expect(remove.changes["archive/refund.md"]).toBeNull();
  w.decide(verdict(remove.id));
  expect(w.rules()).toEqual({});
});
it("prevents stale publication and detects revision tampering or missing authoritative state", () => {
  const w = workspace();
  const a = propose(w, { "a.md": rule("a.md") });
  const b = propose(w, { "b.md": rule("b.md", "# Other\nApplies to: tasks\nInspect evidence.") });
  w.decide(verdict(a.id));
  expect(() => w.decide(verdict(b.id))).toThrow(RevisionConflict);
  expect(w.propose(b.base_revision, w.candidateRules(b), { sessionId: "s", executionId: "e" }).status).toBe(
    "conflict",
  );
  const file = path.join(w.path, "revisions", w.state().head, "rules/a.md");
  fs.writeFileSync(file, "tampered");
  expect(() => w.rules()).toThrow(WorkspaceError);
  fs.rmSync(path.join(w.path, "state.json"));
  expect(() => new Workspace(w.path, w.config)).toThrow("nonempty history");
});
it("rejects corrupt state, symlinks, invalid paths and bounded corpora", () => {
  const w = workspace();
  for (const name of ["../x.md", "/x.md", "x.txt", "a b.md", "x/../../y.md", "x$.md"])
    expect(() => rule(name)).toThrow();
  const p = propose(w, { "a.md": rule("a.md") });
  w.decide(verdict(p.id));
  const file = path.join(w.path, "revisions", w.state().head, "rules/a.md");
  fs.rmSync(file);
  fs.symlinkSync(path.join(w.path, "state.json"), file);
  expect(() => w.rules()).toThrow("symbolic link");
  fs.writeFileSync(path.join(w.path, "state.json"), "{}");
  expect(() => w.state()).toThrow("corrupt");
});
it("hashes arbitrary execution identities as filenames and preserves user data", () => {
  const w = workspace();
  w.saveExecution("../../outside", { context: "memory" });
  expect(w.loadExecution("../../outside")).toEqual({ context: "memory" });
  expect(fs.readdirSync(path.join(w.path, "executions"))).toEqual([`${digest("../../outside")}.json`]);
});
it("coordinates independent processes with actual native locks", async () => {
  const w = workspace();
  const script = `import {Workspace} from './src/reflect/workspace.ts';import {ReflectConfig} from './src/reflect/config.ts';const w=new Workspace(process.argv[1],new ReflectConfig());for(let i=0;i<20;i++)w.transaction(s=>{s.counter=(s.counter??0)+1;});`;
  const child = () =>
    new Promise<void>((resolve, reject) => {
      const process_ = spawn(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script, w.path], {
        cwd: process.cwd(),
        stdio: ["ignore", "pipe", "pipe"],
      });
      let stderr = "";
      process_.stderr.on("data", (d) => {
        stderr += d;
      });
      process_.on("error", reject);
      process_.on("close", (code) => (code === 0 ? resolve() : reject(new Error(stderr))));
    });
  await Promise.all([child(), child(), child()]);
  expect(w.state().counter).toBe(60);
}, 20000);
it("rejects invalid configuration without coercion and copies caller mappings", () => {
  for (const value of [0, -1, true, Number.NaN, Number.POSITIVE_INFINITY, "3"])
    expect(() => new ReflectConfig({ evaluationMaxRuns: value as number })).toThrow();
  for (const value of [true, Number.NaN, -0.1, 1.1])
    expect(() => new ReflectConfig({ thresholds: { test: value as number } })).toThrow();
  const thresholds = { metric: 0.5 };
  const c = new ReflectConfig({ thresholds });
  thresholds.metric = 0.1;
  expect(c.thresholds.metric).toBe(0.5);
  expect(c.evaluationMaxRuns).toBeNull();
});

it("matches Python proposal authority without an added fingerprint verification gate", () => {
  const w = workspace();
  const p = propose(w, { "a.md": rule("a.md") });
  const before = w.state().head;
  w.transaction((s) => {
    s.proposals[p.id]!.changes["a.md"] = new Rule({
      path: "a.md",
      text: "# Changed\nApplies to: refunds\nWrong amount.\n",
    });
  });
  w.decide(verdict(p.id));
  expect(w.state().head).not.toBe(before);
  expect(w.rules()["a.md"]!.text).toContain("Wrong amount");
});
it("keeps the old head if a process exits between immutable revision creation and head replacement", async () => {
  const w = workspace();
  const p = propose(w, { "a.md": rule("a.md") });
  const before = w.state().head;
  const script = `import fs from 'node:fs';import {syncBuiltinESMExports} from 'node:module';const rename=fs.renameSync;fs.renameSync=(from,to)=>{if(String(to).endsWith('state.json'))process.exit(23);return rename(from,to);};syncBuiltinESMExports();const {Workspace}=await import('./src/reflect/workspace.ts');const {ReflectConfig}=await import('./src/reflect/config.ts');const w=new Workspace(process.argv[1],new ReflectConfig());w.decide({proposal_id:process.argv[2],outcome:'accepted',mode:'trial',reason:'test',cases:[],created_at:new Date().toISOString()});`;
  const code = await new Promise<number | null>((resolve, reject) => {
    const child = spawn(process.execPath, ["--import", "tsx", "--input-type=module", "-e", script, w.path, p.id], {
      stdio: "pipe",
    });
    child.on("error", reject);
    child.on("close", resolve);
  });
  expect(code).toBe(23);
  const reopened = new Workspace(w.path, w.config);
  expect(reopened.state().head).toBe(before);
  expect(reopened.proposal(p.id).status).toBe("candidate");
  expect(fs.readdirSync(path.join(w.path, "revisions")).filter((name) => !name.startsWith(".")).length).toBe(2);
  reopened.decide(verdict(p.id));
  expect(Object.keys(reopened.rules())).toEqual(["a.md"]);
});

it("matches Python initialization when state is missing and only the empty revision exists", () => {
  const w = workspace();
  w.transaction((s) => {
    s.evaluations.dispatched = { status: "creating" };
  });
  fs.rmSync(path.join(w.path, "state.json"));
  const reopened = new Workspace(w.path, w.config);
  expect(reopened.state().evaluations).toEqual({});
  expect(fs.existsSync(path.join(w.path, ".initialized"))).toBe(false);
});
