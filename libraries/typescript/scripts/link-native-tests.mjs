import { existsSync, mkdirSync, readFileSync, symlinkSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dependencies = JSON.parse(readFileSync(resolve(root, "test/native/package.json"), "utf8")).dependencies;
for (const name of Object.keys(dependencies)) {
  const target = resolve(root, "node_modules", name);
  const source = resolve(root, "test/native/node_modules", name);
  if (!existsSync(source))
    throw new Error(`Install native fixtures first: npm ci --prefix test/native (${name} missing)`);
  if (!existsSync(target)) {
    mkdirSync(dirname(target), { recursive: true });
    symlinkSync(source, target, process.platform === "win32" ? "junction" : "dir");
  }
}
