import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, expect, it } from "vitest";

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});
function fixture() {
  const root = mkdtempSync(join(tmpdir(), "native-fixture-"));
  directories.push(root);
  function write(path: string, value: unknown) {
    mkdirSync(dirname(join(root, path)), { recursive: true });
    writeFileSync(join(root, path), JSON.stringify(value));
  }
  write("test/native/package.json", { dependencies: { ai: "6.0.116", "@scope/framework": "1.0.0" } });
  write("test/native/node_modules/ai/package.json", { version: "6.0.116" });
  write("test/native/node_modules/@scope/framework/package.json", { version: "1.0.0" });
  mkdirSync(join(root, "scripts"));
  const script = join(root, "scripts/link-native-tests.mjs");
  copyFileSync(fileURLToPath(new URL("../scripts/link-native-tests.mjs", import.meta.url)), script);
  return {
    root,
    write,
    run: (...args: string[]) => spawnSync(process.execPath, [script, ...args], { encoding: "utf8" }),
  };
}
it("links the pinned fixture for scoped and unscoped imports and is idempotent", () => {
  const f = fixture();
  expect(f.run().status).toBe(0);
  expect(f.run().status).toBe(0);
  for (const name of ["ai", "@scope/framework"])
    expect(realpathSync(join(f.root, "node_modules", name))).toBe(
      realpathSync(join(f.root, "test/native/node_modules", name)),
    );
});
it("rejects installed packages without overwriting them, unless compatibility testing is explicit", () => {
  const f = fixture();
  f.write("node_modules/ai/package.json", { version: "7.0.154" });
  const result = f.run();
  expect(result.status).not.toBe(0);
  expect(result.stderr).toContain("shadows the pinned native fixture");
  expect(JSON.parse(readFileSync(join(f.root, "node_modules/ai/package.json"), "utf8"))).toEqual({
    version: "7.0.154",
  });
  const compatibility = f.run("--allow-installed");
  expect(compatibility.status).toBe(0);
  expect(compatibility.stderr).toContain("Native compatibility override: ai@7.0.154");
});
it("requires the fixture to be installed before creating any links", () => {
  const f = fixture();
  rmSync(join(f.root, "test/native/node_modules/@scope"), { recursive: true });
  const result = f.run();
  expect(result.status).not.toBe(0);
  expect(result.stderr).toContain("Install native fixtures first");
  expect(() => realpathSync(join(f.root, "node_modules/ai"))).toThrow();
});
