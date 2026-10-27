/** Reflect policy. Durations are seconds, byte limits count UTF-8 bytes. */
const defaults = {
  workspace: ".pandaprobe/reflect",
  traceVisibilityTimeout: 30,
  evaluationTimeout: 120,
  pollInterval: 1,
  evaluationMaxRuns: null as number | null,
  evaluationMaxTraces: 32,
  evaluationSampleRate: 1,
  maxConcurrency: 4,
  cooldownSeconds: 60,
  reflectionTimeout: 60,
  shutdownTimeout: 5,
  reflectionMaxCommands: 8,
  shellTimeout: 5,
  shellMaxCommands: 2000,
  shellMaxLoopIterations: 5000,
  shellMemoryBytes: 16 * 1024 * 1024,
  commandMaxBytes: 16 * 1024,
  outputMaxBytes: 12 * 1024,
  executionOutputMaxBytes: 96 * 1024,
  contextMaxBytes: 8 * 1024,
  ruleMaxBytes: 8 * 1024,
  rulesMaxBytes: 512 * 1024,
  rulesMaxCount: 256,
  evidenceMaxBytes: 64 * 1024,
  thresholds: { task_completion: 0.5, coherence: 0.5 } as Record<string, number>,
  diagnosticMetrics: ["tool_correctness", "argument_correctness"] as readonly string[],
  stallWindow: 5,
  minimumProgress: 0.02,
  regressionDrop: 0.15,
  validationMinCases: 3,
  validationMinImprovement: 0.05,
  validationRegressionTolerance: 0.02,
  validationTimeout: 120,
};
export type ReflectOptions = Partial<typeof defaults>;
export type ResolvedReflectConfig = Readonly<typeof defaults>;
export class ReflectConfig implements ResolvedReflectConfig {
  readonly workspace!: string;
  readonly traceVisibilityTimeout!: number;
  readonly evaluationTimeout!: number;
  readonly pollInterval!: number;
  readonly evaluationMaxRuns!: number | null;
  readonly evaluationMaxTraces!: number;
  readonly evaluationSampleRate!: number;
  readonly maxConcurrency!: number;
  readonly cooldownSeconds!: number;
  readonly reflectionTimeout!: number;
  readonly shutdownTimeout!: number;
  readonly reflectionMaxCommands!: number;
  readonly shellTimeout!: number;
  readonly shellMaxCommands!: number;
  readonly shellMaxLoopIterations!: number;
  readonly shellMemoryBytes!: number;
  readonly commandMaxBytes!: number;
  readonly outputMaxBytes!: number;
  readonly executionOutputMaxBytes!: number;
  readonly contextMaxBytes!: number;
  readonly ruleMaxBytes!: number;
  readonly rulesMaxBytes!: number;
  readonly rulesMaxCount!: number;
  readonly evidenceMaxBytes!: number;
  readonly thresholds!: Record<string, number>;
  readonly diagnosticMetrics!: readonly string[];
  readonly stallWindow!: number;
  readonly minimumProgress!: number;
  readonly regressionDrop!: number;
  readonly validationMinCases!: number;
  readonly validationMinImprovement!: number;
  readonly validationRegressionTolerance!: number;
  readonly validationTimeout!: number;
  constructor(options: ReflectOptions = {}) {
    for (const key of Object.keys(options))
      if (!(key in defaults)) throw new TypeError(`Unknown Reflect configuration: ${key}`);
    const value = {
      ...defaults,
      workspace: process.env.PANDAPROBE_REFLECT_WORKSPACE ?? defaults.workspace,
      ...options,
    };
    if (typeof value.workspace !== "string" || !value.workspace)
      throw new TypeError("workspace must be a nonempty path");
    const probabilities = [
      "evaluationSampleRate",
      "minimumProgress",
      "regressionDrop",
      "validationMinImprovement",
      "validationRegressionTolerance",
    ];
    const times = [
      "traceVisibilityTimeout",
      "evaluationTimeout",
      "pollInterval",
      "reflectionTimeout",
      "shutdownTimeout",
      "shellTimeout",
      "validationTimeout",
    ];
    for (const [key, n] of Object.entries(value)) {
      if (["workspace", "thresholds", "diagnosticMetrics"].includes(key)) continue;
      if (key === "evaluationMaxRuns" && n === null) continue;
      if (typeof n !== "number" || !Number.isFinite(n)) throw new TypeError(`${key} must be finite`);
      if (probabilities.includes(key)) {
        if (n < 0 || n > 1) throw new TypeError(`${key} must be in [0, 1]`);
      } else if (key === "cooldownSeconds") {
        if (n < 0) throw new TypeError("cooldownSeconds must be nonnegative");
      } else if (n <= 0 || (!times.includes(key) && !Number.isSafeInteger(n)))
        throw new TypeError(`${key} must be positive${times.includes(key) ? "" : " and integral"}`);
    }
    if (
      !value.thresholds ||
      Array.isArray(value.thresholds) ||
      !Object.keys(value.thresholds).length ||
      Object.entries(value.thresholds).some(
        ([k, v]) => !k || typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1,
      )
    )
      throw new TypeError("thresholds must contain named finite values in [0, 1]");
    if (!Array.isArray(value.diagnosticMetrics) || value.diagnosticMetrics.some((v) => typeof v !== "string" || !v))
      throw new TypeError("diagnosticMetrics must contain metric names");
    Object.assign(this, value, {
      thresholds: { ...value.thresholds },
      diagnosticMetrics: Object.freeze([...value.diagnosticMetrics]),
    });
    Object.freeze(this);
  }
}
