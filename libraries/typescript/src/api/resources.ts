import { type PlatformClient, type RequestOptions, pause, positive } from "./client.js";
import {
  ApiDeadlineExceeded,
  CreationOutcomeUnknown,
  EvaluationFailed,
  InsufficientEvidence,
  InvalidResponseError,
  NotFoundError,
} from "./errors.js";
import {
  type EvaluationResult,
  type EvaluationRun,
  type Metric,
  type Score,
  type Trace,
  type TracePage,
  type TraceSummary,
  decode,
  decodeMany,
  decoders,
  numericValue,
  uuid,
} from "./models.js";

type WaitOptions = { timeout?: number; pollInterval?: number; signal?: AbortSignal };
function polling(options: WaitOptions, timeout: number, interval: number) {
  const seconds = options.timeout ?? timeout;
  const poll = options.pollInterval ?? interval;
  positive(seconds, "timeout");
  positive(poll, "pollInterval");
  const deadline = performance.now() + seconds * 1000;
  const remaining = () => {
    const value = (deadline - performance.now()) / 1000;
    if (value <= 0) throw new ApiDeadlineExceeded("Polling deadline exceeded");
    return value;
  };
  return { remaining, pause: () => pause(Math.min(remaining(), poll) * 1000, options.signal) };
}
export class Traces {
  constructor(private readonly client: PlatformClient) {}
  async get(traceId: string, options: RequestOptions = {}): Promise<Trace> {
    const id = uuid(traceId);
    const trace = decode(decoders.trace, await this.client.request("GET", `traces/${id}`, options));
    if (trace.trace_id !== id) throw new InvalidResponseError("Retrieved trace identity does not match the request");
    return trace;
  }
  async list(
    options: RequestOptions & { sessionId?: string; status?: string; limit?: number; offset?: number } = {},
  ): Promise<TracePage> {
    const { sessionId, status, limit = 50, offset = 0 } = options;
    if (!(limit >= 1 && limit <= 200) || offset < 0) throw new TypeError("limit must be 1–200 and offset nonnegative");
    return decode(
      decoders.page,
      await this.client.request("GET", "traces", {
        ...options,
        query: { session_id: sessionId, status, limit, offset },
      }),
    );
  }
  async *iterate(
    options: { sessionId?: string; maxItems?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<TraceSummary> {
    const maximum = options.maxItems ?? 1000;
    let offset = 0;
    while (offset < maximum) {
      const page = await this.list({ ...options, limit: Math.min(200, maximum - offset), offset });
      for (const item of page.items) yield item;
      offset += page.items.length;
      if (!page.items.length || offset >= page.total) break;
    }
  }
  async waitVisible(traceId: string, options: WaitOptions = {}): Promise<Trace> {
    const poll = polling(options, 30, 0.5);
    for (;;) {
      try {
        const trace = await this.get(traceId, { timeout: poll.remaining(), signal: options.signal });
        if (trace.status === "COMPLETED" || trace.status === "ERROR") return trace;
      } catch (error) {
        if (!(error instanceof NotFoundError)) throw error;
      }
      await poll.pause();
    }
  }
}
export class Evaluations {
  constructor(private readonly client: PlatformClient) {}
  async metrics(options: RequestOptions = {}): Promise<Metric[]> {
    return decodeMany(decoders.metric, await this.client.request("GET", "evaluations/trace-metrics", options));
  }
  async create(
    traceIds: string[],
    metrics: string[],
    options: RequestOptions & { name?: string; model?: string } = {},
  ): Promise<EvaluationRun> {
    const ids = [...new Set(traceIds.map(uuid))];
    const names = [...new Set(metrics)];
    if (!ids.length || !names.length || names.some((m) => typeof m !== "string" || !m))
      throw new TypeError("Expected nonempty trace IDs and metric names");
    const data = await this.client.request("POST", "evaluations/trace-runs/batch", {
      ...options,
      body: {
        trace_ids: ids,
        metrics: names,
        ...(options.name === undefined ? {} : { name: options.name }),
        ...(options.model === undefined ? {} : { model: options.model }),
      },
    });
    try {
      return decode(decoders.run, data);
    } catch {
      throw new CreationOutcomeUnknown("The platform returned an invalid response; the run may exist");
    }
  }
  async get(runId: string, options: RequestOptions = {}): Promise<EvaluationRun> {
    const id = uuid(runId);
    const run = decode(decoders.run, await this.client.request("GET", `evaluations/trace-runs/${id}`, options));
    if (run.id !== id) throw new InvalidResponseError("Retrieved evaluation identity does not match the request");
    return run;
  }
  async scores(runId: string, options: RequestOptions = {}): Promise<Score[]> {
    return decodeMany(
      decoders.score,
      await this.client.request("GET", `evaluations/trace-runs/${uuid(runId)}/scores`, options),
    );
  }
  async wait(
    runId: string,
    options: WaitOptions & { traceIds: string[]; metrics: string[] },
  ): Promise<EvaluationResult> {
    const expected = [...new Set(options.traceIds.map(uuid))]
      .sort()
      .flatMap((id) => [...new Set(options.metrics)].sort().map((metric) => [id, metric] as [string, string]));
    if (!expected.length) throw new TypeError("Expected traces and metrics must not be empty");
    const poll = polling(options, 120, 1);
    for (;;) {
      const run = await this.get(runId, { timeout: poll.remaining(), signal: options.signal });
      if (run.status === "FAILED" || run.status === "CANCELLED") throw new EvaluationFailed(run.id, run.status);
      if (run.status === "COMPLETED") {
        const scores = await this.scores(runId, { timeout: poll.remaining(), signal: options.signal });
        const seen = new Set<string>();
        const duplicates = new Set<string>();
        const usable = new Map<string, Score>();
        for (const score of scores) {
          const key = JSON.stringify([score.trace_id, score.name]);
          if (seen.has(key)) duplicates.add(key);
          seen.add(key);
          if (score.eval_run_id === run.id && numericValue(score) !== null) usable.set(key, score);
        }
        const missing = expected.filter(
          (pair) => !usable.has(JSON.stringify(pair)) || duplicates.has(JSON.stringify(pair)),
        );
        if (missing.length) throw new InsufficientEvidence(run.id, missing);
        return { run, scores: expected.map((pair) => usable.get(JSON.stringify(pair))!) };
      }
      await poll.pause();
    }
  }
}
