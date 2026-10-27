import type { ReflectConfig } from "./config.js";
import { type FailureNotice, checkedScores, digest, freezeNotice, identifier, utcNow } from "./models.js";
import type { Workspace } from "./workspace.js";
export class SignalGate {
  constructor(
    private readonly workspace: Workspace,
    private readonly config: ReflectConfig,
  ) {}
  observe(options: {
    executionId: string;
    sessionId: string;
    traceIds: string[];
    scores: Record<string, number>;
    reasons: Record<string, string>;
    completed: boolean;
  }): FailureNotice | null {
    const { executionId, sessionId, traceIds, scores, reasons, completed } = options;
    checkedScores(scores);
    if (Object.keys(this.config.thresholds).some((metric) => !Object.hasOwn(scores, metric)))
      throw new TypeError("Signal gates require complete finite scores in [0, 1]");
    return this.workspace.transaction((state) => {
      for (const record of Object.values(state.notices))
        if (record.execution_id === executionId) return freezeNotice(record);
      state.gates[digest(sessionId)] ??= { history: [], last_notices: {} };
      const gate = state.gates[digest(sessionId)];
      const history: Array<{ execution_id: string; scores: Record<string, number> }> = gate.history;
      if (history.some((row) => row.execution_id === executionId)) return null;
      let signatures: string[] = [];
      for (const [metric, threshold] of Object.entries(this.config.thresholds)) {
        const value = scores[metric]!;
        const previous = history.filter((row) => metric in row.scores).map((row) => row.scores[metric]!);
        if (completed && value < threshold) signatures.push(`${metric}:below_threshold`);
        else if (
          previous.length &&
          previous[previous.length - 1]! - value >= this.config.regressionDrop &&
          value < threshold
        )
          signatures.push(`${metric}:regression`);
        else if (previous.length >= this.config.stallWindow - 1 && value < threshold) {
          const window = [...previous, value].slice(-this.config.stallWindow);
          if (Math.max(...window) - Math.min(...window) < this.config.minimumProgress)
            signatures.push(`${metric}:stalled`);
        }
      }
      history.push({ execution_id: executionId, scores });
      gate.history = history.slice(-this.config.stallWindow);
      const now = Date.now() / 1000;
      signatures = signatures.filter(
        (signature) => now - (gate.last_notices[signature] ?? 0) >= this.config.cooldownSeconds,
      );
      if (!signatures.length) return null;
      for (const signature of signatures) gate.last_notices[signature] = now;
      const notice: FailureNotice = {
        id: identifier(),
        execution_id: executionId,
        session_id: sessionId,
        trace_ids: traceIds,
        scores,
        reasons,
        signatures,
        created_at: utcNow(),
      };
      state.notices[notice.id] = notice;
      return freezeNotice(notice);
    });
  }
}
