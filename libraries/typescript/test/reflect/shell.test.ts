import { afterEach, expect, it } from "vitest";
import { ReflectConfig } from "../../src/reflect/config.js";
import { Rule } from "../../src/reflect/models.js";
import { ReflectShell, clipBytes } from "../../src/reflect/shell.js";
const shells: ReflectShell[] = [];
async function shell(options: ConstructorParameters<typeof ReflectConfig>[0] = {}) {
  const s = await ReflectShell.create(new ReflectConfig(options), {});
  shells.push(s);
  return s;
}
afterEach(async () => {
  await Promise.all(shells.splice(0).map((s) => s.close()));
});
it("executes ordinary shell lifecycle with persistent variables, loops and real utilities", async () => {
  const s = await shell();
  await s.execute(
    "mkdir -p /rules/support; cat > /rules/support/refunds.md <<'END'\n# Refund\nApplies to: returns\nExclude shipping.\nEND\nvalue=persisted",
  );
  expect(
    await s.execute('for file in /rules/support/*.md; do cat "$file" | grep shipping; done; echo "$value"'),
  ).toContain("persisted");
  await s.execute(
    "sed -i 's/Exclude/Always exclude/' /rules/support/refunds.md; mv /rules/support/refunds.md /rules/refunds.md",
  );
  expect(s.rules()["refunds.md"]?.text).toContain("Always exclude");
  await s.execute("rm /rules/refunds.md");
  expect(s.rules()).toEqual({});
});
it("isolates host files, credentials, network, Python and separate interpreters", async () => {
  process.env.REFLECT_HOST_SECRET = "must-not-leak";
  try {
    const a = await shell();
    const b = await shell();
    await a.execute("secret=private; echo private >/context.md");
    expect(
      await a.execute(`cat '${process.cwd()}/package.json'; printenv; python -c 1; curl https://example.com`),
    ).not.toContain("must-not-leak");
    expect(await b.execute('echo "$secret"; cat /context.md')).not.toContain("private");
  } finally {
    process.env.REFLECT_HOST_SECRET = undefined;
  }
});
it("rolls back invalid filesystem mutations while preserving read-only mounts", async () => {
  const s = await shell({ contextMaxBytes: 5 });
  s.setEvidence({ id: "notice" }, "trusted");
  s.setDecisions({ accepted: false });
  expect(await s.execute("echo too-long >/context.md; echo damaged >/inbox/evidence.txt")).toContain("rolled back");
  expect(s.context()).toBe("");
  expect(await s.execute("echo damaged >/inbox/evidence.txt; cat /inbox/evidence.txt")).toContain("trusted");
  expect(await s.execute("echo damaged >/decisions/status.json; cat /decisions/status.json")).toContain(
    '"accepted":false',
  );
});
it.each([
  "ln -s /context.md /rules/link.md",
  "rm -r /rules; ln -s /tmp /rules",
  "touch '/rules/bad name.md'",
  "rm /context.md; ln -s /etc/passwd /context.md",
])("rejects unsafe snapshot changes: %s", async (command) => {
  const s = await shell();
  expect(await s.execute(command)).toContain("rolled back");
  expect(s.files()).toEqual({});
  expect(s.context()).toBe("");
});
it("does not roll back valid writes just because a command exits nonzero", async () => {
  const s = await shell();
  expect(await s.execute("echo remember >/context.md; false")).toContain("exit_code=1");
  expect(s.context()).toBe("remember\n");
});
it("bounds output including guard replies, distinguishes truncation and closes late calls", async () => {
  const s = await shell({ executionOutputMaxBytes: 700, outputMaxBytes: 300, commandMaxBytes: 100 });
  let bytes = 0;
  for (let i = 0; i < 50; i++) bytes += Buffer.byteLength(await s.execute("x".repeat(101)));
  expect(bytes).toBeLessThanOrEqual(700);
  const t = await shell({ outputMaxBytes: 300 });
  expect(await t.execute("seq 1 1000")).toContain("output truncated");
  await t.close();
  expect(await t.execute("echo late")).toBe("");
  expect(clipBytes("a🐼é", 4)).toBe("a");
  expect(clipBytes("a🐼é", 5)).toBe("a🐼");
});
it("separates reflection commands from task command count", async () => {
  const s = await shell({ reflectionMaxCommands: 1 });
  await s.execute("echo task");
  s.beginReflection();
  await s.execute("echo reflection");
  expect(await s.execute("echo extra")).toContain("command budget");
  s.endReflection();
  expect(await s.execute("echo task")).toContain("task");
});
it("serializes commands, snapshots checkpoints and cancels running native work", async () => {
  const checkpoints: string[] = [];
  const s = await ReflectShell.create(
    new ReflectConfig({ shellTimeout: 0.03, shellMaxLoopIterations: 100000000 }),
    {},
    { onCheckpoint: (_files, context) => checkpoints.push(context) },
  );
  shells.push(s);
  await Promise.all([s.execute("echo first >/context.md"), s.execute("echo second >>/context.md")]);
  expect(s.context()).toBe("first\nsecond\n");
  expect(checkpoints).toHaveLength(2);
  const controller = new AbortController();
  const pending = s.execute("echo mutation >/context.md; sleep 10", { signal: controller.signal });
  setTimeout(() => controller.abort(), 5);
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  expect(s.context()).toBe("first\nsecond\n");
  expect(await s.execute("echo recovered")).toContain("recovered");
});
it("keeps rule byte, traversal and memory budgets", async () => {
  const s = await shell({ ruleMaxBytes: 20 });
  expect(await s.execute("echo aaaaaaaaaaaaaaaaaaaaa >/rules/x.md")).toContain("rolled back");
  const t = await shell({ rulesMaxCount: 1 });
  expect(await t.execute("touch /rules/a.md /rules/b.md")).toContain("rolled back");
  expect(() => new Rule({ path: "a.md", text: "bad" })).toThrow();
});

it("preserves returned interpreter-error edits like Python, and retains usage across restoration", async () => {
  const s = await shell({ shellTimeout: 0.01, shellMaxLoopIterations: 10000000 });
  const output = await s.execute("echo interrupted >/context.md; sleep 5");
  expect(output).toContain("execution timeout");
  expect(s.context()).toBe("interrupted\n");
  const restored = await ReflectShell.create(s.config, {}, { usage: s.usage });
  shells.push(restored);
  expect(restored.commandCount).toBe(1);
  restored.beginReflection();
  await restored.execute("echo continued");
  expect(restored.usage.output_bytes).toBeGreaterThan(s.usage.output_bytes);
});
it("rejects deep and oversized trees and prevents evidence rename/delete", async () => {
  const s = await shell();
  s.setEvidence({ id: "notice" }, "immutable evidence");
  await s.execute("rm -rf /inbox; mv /inbox /moved; echo damaged >/inbox/evidence.txt");
  expect(await s.execute("cat /inbox/evidence.txt")).toContain("immutable evidence");
  const deep = Array.from({ length: 15 }, () => "nested").join("/");
  expect(await s.execute(`mkdir -p /rules/${deep}; touch /rules/${deep}/x.md`)).toContain("rolled back");
  expect(s.files()).toEqual({});
});
