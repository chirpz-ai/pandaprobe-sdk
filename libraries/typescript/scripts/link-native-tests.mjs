import { existsSync, lstatSync, mkdirSync, readFileSync, realpathSync, symlinkSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dependencies = JSON.parse(readFileSync(resolve(root, "test/native/package.json"), "utf8")).dependencies;
const allowInstalled = process.argv.includes("--allow-installed");
const missing = [];
// Validate every resolution before changing any links. Local example installs
// must neither silently shadow the fixture nor be overwritten by this script.
for (const name of Object.keys(dependencies)) {
  const target = resolve(root, "node_modules", name);
  const source = resolve(root, "test/native/node_modules", name);
  if (!existsSync(source))
    throw new Error(`Install native fixtures first: npm ci --prefix test/native (${name} missing)`);
  if (!lstatSync(target, { throwIfNoEntry: false })) {
    missing.push({ source, target });
  } else if (!existsSync(target) || realpathSync(target) !== realpathSync(source)) {
    if (!allowInstalled || !existsSync(target))
      throw new Error(
        `${name} shadows the pinned native fixture at ${target}. Run pinned tests in a clean install, or use pnpm test:native:installed to explicitly test local versions. No installed packages were replaced.`,
      );
    const installed = JSON.parse(readFileSync(resolve(target, "package.json"), "utf8"));
    console.warn(`Native compatibility override: ${name}@${installed.version} (${realpathSync(target)})`);
  }
}
for (const { source, target } of missing) {
  mkdirSync(dirname(target), { recursive: true });
  symlinkSync(source, target, process.platform === "win32" ? "junction" : "dir");
}
