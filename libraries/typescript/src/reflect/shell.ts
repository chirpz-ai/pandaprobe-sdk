import type { Bash, FileSystem } from "@everruns/bashkit";
import { abortError } from "../api/client.js";
import type { ReflectConfig } from "./config.js";
import { Rule } from "./models.js";
import { WorkspaceError } from "./workspace.js";
export const TOOL_NAME = "reflect_bash";
export const TOOL_DESCRIPTION =
  "Run Bash commands in the persistent Reflect filesystem. Search/read /rules for learned guidance; edit Markdown rule files to propose improvements and /context.md to maintain active memory. Read /inbox for assigned failure evidence and /decisions for validation results. Supports grep, find, sed, awk, cat, mkdir, mv, rm, pipes, loops and heredocs. This filesystem is isolated from host files and network.";
export const TOOL_PARAMETERS = {
  type: "object",
  properties: { command: { type: "string", description: "Bash command to execute" } },
  required: ["command"],
  additionalProperties: false,
} as const;
export function clipBytes(text: string, maximum: number): string {
  const bytes = Buffer.from(text);
  let end = Math.max(0, Math.min(bytes.length, Math.floor(maximum)));
  if (end === bytes.length) return text;
  while (end > 0 && (bytes[end]! & 0xc0) === 0x80) end--;
  return bytes.subarray(0, end).toString("utf8");
}
export interface ShellOptions {
  context?: string;
  candidateIds?: string[];
  onCheckpoint?: (files: Record<string, string>, context: string) => void;
  usage?: { commands: number; output_bytes: number };
}
export class ReflectShell {
  candidateIds: string[];
  private outputBytes: number;
  private commands: number;
  private commandLimit: number | null = null;
  private closed = false;
  private queue: Promise<unknown> = Promise.resolve();
  private readonly: Record<string, Record<string, string>> = {};
  private constructor(
    readonly config: ReflectConfig,
    private readonly bash: Bash,
    private readonly Fs: typeof FileSystem,
    private readonly options: ShellOptions,
  ) {
    this.candidateIds = [...(options.candidateIds ?? [])];
    this.commands = options.usage?.commands ?? 0;
    this.outputBytes = options.usage?.output_bytes ?? 0;
  }
  static async create(
    config: ReflectConfig,
    rules: Record<string, Rule>,
    options: ShellOptions = {},
  ): Promise<ReflectShell> {
    let module: typeof import("@everruns/bashkit");
    try {
      const dependency: string = "@everruns/bashkit";
      module = await import(dependency);
    } catch {
      throw new Error("Install Reflect runtimes: npm install @everruns/bashkit@0.19.0 fs-native-extensions@1.5.1");
    }
    const bash = new module.Bash({
      cwd: "/",
      env: { PANDAPROBE_REFLECT: "1" },
      maxCommands: config.shellMaxCommands,
      maxLoopIterations: config.shellMaxLoopIterations,
      maxMemory: config.shellMemoryBytes,
      maxInputBytes: config.commandMaxBytes,
      timeoutMs: config.shellTimeout * 1000,
      python: false,
      sqlite: false,
      mounts: [],
      allowedMountPaths: [],
    });
    const shell = new ReflectShell(config, bash, module.FileSystem, options);
    bash.mkdir("/rules", true);
    for (const [name, rule] of Object.entries(rules)) {
      Rule.safePath(name);
      bash.mkdir(`/rules/${name}`.slice(0, `/rules/${name}`.lastIndexOf("/")), true);
      bash.writeFile(`/rules/${name}`, rule.text);
    }
    bash.writeFile("/context.md", clipBytes(options.context ?? "", config.contextMaxBytes));
    shell.mountReadonly("/inbox", {});
    shell.mountReadonly("/decisions", {});
    return shell;
  }
  private mountReadonly(path: string, contents: Record<string, string>): void {
    const fs = new this.Fs();
    for (const [name, content] of Object.entries(contents)) {
      if (name.includes("/") || [".", ".."].includes(name))
        throw new TypeError("Evidence filenames must be simple names");
      fs.writeFile(`/${name}`, content);
    }
    this.bash.mount(path, fs, true);
    this.readonly[path] = { ...contents };
  }
  setEvidence(notice: unknown, evidence: string): void {
    this.bash.unmount("/inbox");
    this.mountReadonly("/inbox", {
      "notice.json": JSON.stringify(notice, null, 2),
      "evidence.txt": clipBytes(evidence, this.config.evidenceMaxBytes),
    });
  }
  setDecisions(decisions: unknown): void {
    this.bash.unmount("/decisions");
    this.mountReadonly("/decisions", { "status.json": JSON.stringify(decisions) });
  }
  beginReflection(): void {
    this.commandLimit = this.commands + this.config.reflectionMaxCommands;
  }
  endReflection(): void {
    this.commandLimit = null;
  }
  get commandCount(): number {
    return this.commands;
  }
  get usage(): { commands: number; output_bytes: number } {
    return { commands: this.commands, output_bytes: this.outputBytes };
  }
  context(): string {
    if (!this.bash.exists("/context.md")) return "";
    const info = this.bash.lstat("/context.md");
    if (info.fileType !== "file" || info.size > this.config.contextMaxBytes)
      throw new WorkspaceError("context.md must be a regular file within its byte budget");
    return this.bash.readFile("/context.md");
  }
  files(): Record<string, string> {
    const files: Record<string, string> = {};
    if (!this.bash.exists("/rules")) return files;
    if (this.bash.lstat("/rules").fileType !== "directory") throw new WorkspaceError("/rules must be a directory");
    const pending = ["/rules"];
    let entries = 0;
    let size = 0;
    while (pending.length) {
      const directory = pending.pop()!;
      for (const entry of this.bash.readDir(directory)) {
        if (++entries > this.config.rulesMaxCount * 4 + 32)
          throw new WorkspaceError("Too many rule directory entries; consolidate the working revision");
        const path = `${directory}/${entry.name}`;
        const info = this.bash.lstat(path);
        if (info.fileType === "directory") {
          if (path.split("/").length - 1 > 12) throw new WorkspaceError("Rule directory nesting exceeds the limit");
          pending.push(path);
        } else if (info.fileType !== "file") throw new WorkspaceError("Rule files cannot be symbolic links");
        else if (path.endsWith(".md")) {
          const name = path.slice(7);
          Rule.safePath(name);
          if (info.size > this.config.ruleMaxBytes) throw new WorkspaceError(`Rule ${name} exceeds its byte budget`);
          size += info.size;
          if (size > this.config.rulesMaxBytes || Object.keys(files).length >= this.config.rulesMaxCount)
            throw new WorkspaceError("Rule corpus exceeds its file or byte budget");
          files[name] = this.bash.readFile(path);
        }
      }
    }
    return files;
  }
  rules(): Record<string, Rule> {
    return Object.fromEntries(Object.entries(this.files()).map(([path, text]) => [path, new Rule({ path, text })]));
  }
  private checkpoint(): void {
    this.options.onCheckpoint?.(this.files(), this.context());
  }
  private control(message: string): string {
    const rendered = clipBytes(
      message,
      Math.min(this.config.outputMaxBytes, this.config.executionOutputMaxBytes - this.outputBytes),
    );
    if (rendered) {
      this.outputBytes += Buffer.byteLength(rendered);
      this.checkpoint();
    }
    return rendered;
  }
  execute(command: string, options: { signal?: AbortSignal } = {}): Promise<string> {
    const task = this.queue.then(() => this.executeSerialized(command, options.signal));
    this.queue = task.catch(() => {});
    return task;
  }
  private restore(snapshot: Uint8Array): void {
    this.bash.restoreSnapshot(snapshot);
    for (const [path, contents] of Object.entries(this.readonly)) {
      this.bash.unmount(path);
      this.mountReadonly(path, contents);
    }
  }
  private async executeSerialized(command: string, signal?: AbortSignal): Promise<string> {
    if (this.closed) return "";
    signal?.throwIfAborted();
    if (typeof command !== "string" || Buffer.byteLength(command) > this.config.commandMaxBytes)
      return this.control("Command exceeds the Reflect input budget.");
    if (this.commandLimit !== null && this.commands >= this.commandLimit)
      return this.control(
        "Reflection command budget reached. Return to the task; unfinished guidance remains pending.",
      );
    const remaining = this.config.executionOutputMaxBytes - this.outputBytes;
    if (remaining < 256) return this.control("Reflect output budget reached for this execution.");
    const snapshot = this.bash.snapshot();
    this.bash.clearCancel();
    this.commands++;
    let output: string;
    let code: number;
    let truncated = false;
    try {
      const result = await this.bash.execute(command, { signal });
      if (signal?.aborted || this.closed) {
        this.restore(snapshot);
        this.checkpoint();
        if (signal?.aborted) throw signal.reason ?? abortError();
        return "";
      }
      output = result.stdout + (result.stderr ? `${result.stdout ? "\n" : ""}${result.stderr}` : "");
      code = result.exitCode;
      truncated = result.stdoutTruncated || result.stderrTruncated;
    } catch (error) {
      this.restore(snapshot);
      this.checkpoint();
      if (signal?.aborted) throw signal.reason ?? abortError();
      if (this.closed) return "";
      return this.control("Reflect command failed; its filesystem changes were rolled back.");
    }
    let context: string;
    try {
      this.files();
      context = this.context();
    } catch (error) {
      this.restore(snapshot);
      output = `Filesystem change rejected and rolled back: ${String(error)}`;
      code = 1;
      context = this.context();
    }
    const limit = Math.min(this.config.outputMaxBytes, remaining);
    truncated ||= Buffer.byteLength(output) > limit - 256;
    const suffix = `\n[exit_code=${code}; context.md=${Buffer.byteLength(context)}/${this.config.contextMaxBytes} bytes${truncated ? "; output truncated: use a narrower search or excerpt" : ""}]`;
    const rendered = clipBytes(clipBytes(output, limit - 256) + suffix, limit);
    this.outputBytes += Buffer.byteLength(rendered);
    this.checkpoint();
    return rendered;
  }
  async close(): Promise<void> {
    this.closed = true;
    this.bash.cancel();
    await this.queue;
  }
}
