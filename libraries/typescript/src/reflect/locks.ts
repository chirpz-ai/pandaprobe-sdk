import * as fs from "node:fs";
import { createRequire } from "node:module";
import * as path from "node:path";
/** The pinned package's N-API binary supports Node 18; its current transitive
 * JS resolver uses Array.prototype.with (Node 20). Resolve the packaged binary
 * directly, without global polyfills, altered dependencies or weaker PID locks.
 * These two calls mirror fs-native-extensions 1.5.1's public lock wrappers.
 */
export function nativeLocks(): { tryLock(fd: number): boolean; unlock(fd: number): void } {
  try {
    const require = createRequire(typeof __filename === "string" ? __filename : import.meta.url);
    const packageFile = require.resolve("fs-native-extensions/package.json");
    if (JSON.parse(fs.readFileSync(packageFile, "utf8")).version !== "1.5.1")
      throw new Error("Reflect requires fs-native-extensions 1.5.1");
    const binding = require(
      path.join(
        path.dirname(packageFile),
        "prebuilds",
        `${process.platform}-${process.arch}`,
        "fs-native-extensions.node",
      ),
    );
    return {
      tryLock(fd: number): boolean {
        try {
          binding.tryLock(fd, 0, 0, true);
          return true;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "EAGAIN") return false;
          throw error;
        }
      },
      unlock(fd: number): void {
        binding.unlock(fd, 0, 0);
      },
    };
  } catch (cause) {
    throw new Error(
      `Install Reflect runtimes for ${process.platform}-${process.arch}: npm install @everruns/bashkit@0.19.0 fs-native-extensions@1.5.1`,
      { cause },
    );
  }
}
