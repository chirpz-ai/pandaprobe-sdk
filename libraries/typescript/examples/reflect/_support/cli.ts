/** Example launchers mirror Python runner.py / failure_runner.py; never imported by the SDK. */
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Client } from "../../../src/index.js";
import { type Framework, ReflectConfig, Reflect as ReflectRuntime } from "../../../src/reflect/index.js";
import { failureLoop, supportLoop } from "./loops.js";
import { type NativeOptions, nativeFactory } from "./native.js";
import { type RunOptions, identity } from "./reporting.js";
export interface ExampleArguments extends RunOptions {
  model: string;
  taskTimeout: number;
  rounds: number;
  maxValidations: number;
  workspace?: string;
}
export const defaultModel = (framework: string) =>
  framework === "claude-agent-sdk" ? "claude-haiku-5-5" : "gpt-6-luna";
export function argumentsFor(framework: string, failure: boolean, argv = process.argv.slice(2)): ExampleArguments {
  const result: ExampleArguments = {
    model: process.env.REFLECT_EXAMPLE_MODEL ?? defaultModel(framework),
    taskTimeout: 360,
    rounds: 1,
    maxValidations: 2,
  };
  const allowed = new Set([
    "--model",
    "--output",
    "--task-timeout",
    ...(failure ? [] : ["--rounds", "--max-tickets", "--max-validations", "--workspace"]),
  ]);
  for (let i = 0; i < argv.length; i++) {
    const [key, inline] = argv[i]!.split(/=(.*)/s);
    if (!allowed.has(key!)) throw new TypeError(`Unknown option ${key}`);
    const value = inline ?? argv[++i];
    if (value === undefined || value.startsWith("--")) throw new TypeError(`Missing value for ${key}`);
    if (key === "--model") result.model = value;
    if (key === "--output") result.output = resolve(value);
    if (key === "--workspace") result.workspace = resolve(value);
    if (key === "--task-timeout") {
      const n = Number(value);
      if (!value.trim() || !Number.isFinite(n) || n <= 0)
        throw new TypeError("task-timeout must be finite and positive");
      result.taskTimeout = n;
    }
    if (["--rounds", "--max-tickets", "--max-validations"].includes(key!)) {
      const n = Number(value);
      if (!/^[+-]?\d+$/.test(value) || !Number.isSafeInteger(n) || n < (key === "--max-validations" ? 0 : 1))
        throw new TypeError(`${key} must be ${key === "--max-validations" ? "nonnegative" : "positive"}`);
      if (key === "--rounds") result.rounds = n;
      if (key === "--max-tickets") result.maxTickets = n;
      if (key === "--max-validations") result.maxValidations = n;
    }
  }
  return result;
}
export const exampleConfig = (args: ExampleArguments, output: string, failure: boolean) =>
  new ReflectConfig({
    workspace: (!failure && args.workspace) || join(output, "workspace"),
    thresholds: { task_completion: 0.8 },
    reflectionTimeout: 90,
    validationTimeout: 600,
  });
export async function preflight(runtime: ReflectRuntime): Promise<void> {
  if (!runtime.client.config.enabled) throw new Error("These live examples require PANDAPROBE_ENABLED=true");
  const metrics = new Set((await runtime.api.evaluations.metrics()).map((metric) => metric.name));
  const missing = [...new Set([...Object.keys(runtime.config.thresholds), ...runtime.config.diagnosticMetrics])]
    .filter((metric) => !metrics.has(metric))
    .sort();
  if (missing.length) throw new Error(`Platform lacks required metrics: ${JSON.stringify(missing)}`);
}
export async function main(
  framework: Exclude<Framework, "generic">,
  failure: boolean,
  modelOptions: Pick<NativeOptions, "modelTimeoutMs"> = {},
) {
  if (process.argv.slice(2).some((value) => value === "--help" || value === "-h")) {
    console.log(
      `Live ${framework} Reflect ${failure ? "stale-runbook incident" : "customer-support pressure test"}\n--model NAME (default REFLECT_EXAMPLE_MODEL or ${defaultModel(framework)})\n--output PATH (new run directory; existing directories are refused)\n--task-timeout SECONDS (default 360 per task/evaluation/reflection)\n${
        failure
          ? ""
          : "--rounds COUNT (default 1 = 18 turns) --max-tickets COUNT\n--max-validations COUNT (default 2) --workspace PATH (explicit reuse; default fresh)\n"
      }${modelOptions.modelTimeoutMs === undefined ? "" : `Model request timeout: ${modelOptions.modelTimeoutMs / 1000} seconds; automatic retries disabled.\n`}Real provider calls and platform evaluations; fictional business tools. Requires PANDAPROBE_API_KEY, PANDAPROBE_PROJECT_NAME and provider credentials.`,
    );
    return;
  }
  const args = argumentsFor(framework, failure);
  const required = [
    "PANDAPROBE_API_KEY",
    "PANDAPROBE_PROJECT_NAME",
    framework === "claude-agent-sdk" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY",
  ];
  const missing = required.filter((name) => !process.env[name]);
  if (missing.length) throw new Error(`Export these test credentials before running: ${missing.join(", ")}`);
  const output =
    args.output ??
    join(".pandaprobe/reflect-examples", `${framework}${failure ? "-failure" : ""}`, identity().slice(0, 12));
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(output);
  console.log(`Artifacts: ${resolve(output)}\nReal model/evaluation calls; fictional business tools only.`);
  const client = new Client();
  let runtime: ReflectRuntime | undefined;
  try {
    runtime = new ReflectRuntime(client, { config: exampleConfig(args, output, failure) });
    await preflight(runtime);
    const factory = await nativeFactory(runtime, framework, {
      ...modelOptions,
      model: args.model,
      workdir: output,
      tag: failure ? "reflect-failure" : "reflect-support",
    });
    const options = { ...args, output, framework };
    let report: any;
    try {
      report = await (failure ? failureLoop(runtime, factory, options) : supportLoop(runtime, factory, options));
    } finally {
      if (!failure) await runtime.waitExport();
    }
    console.log(
      JSON.stringify(failure ? { status: report.status, milestones: report.milestones } : report.summary, null, 2),
    );
    console.log(`Report: ${join(output, "report.json")}`);
    if (failure && report.status !== "completed") {
      console.error(`Full loop not demonstrated at ${report.stopped_at}: ${report.reason}`);
      process.exitCode = 1;
    }
  } finally {
    await runtime?.close();
    await client.shutdown();
  }
}
