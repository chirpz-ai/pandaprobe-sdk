import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { Client } from "../../src/index.js";
import { type Invocation, Outcome, Reflect as ReflectRuntime, ReplayCase } from "../../src/reflect/index.js";
import { localPlatform } from "./local-platform.js";
const RULE =
  "# Complete sums\nApplies to: summing lists of numbers\nUse every number in the input, not only the first two.\n";
export async function demonstrate(workspace: string) {
  const service = await localPlatform((trace, metric) => {
    const values = trace.metadata.values as number[];
    const actual = Number(trace.output.messages[0].content);
    return {
      score: metric === "task_completion" ? Number(actual === values.reduce((a, b) => a + b, 0)) : 1,
      reason: "Compared output with independently computed arithmetic",
    };
  });
  const client = new Client({
    apiKey: "offline-key",
    projectName: "offline-demo",
    endpoint: service.endpoint,
    batchSize: 1,
    flushInterval: 0.01,
  });
  const runtime = new ReflectRuntime(client, {
    config: { workspace, pollInterval: 0.01, cooldownSeconds: 0, traceVisibilityTimeout: 3, evaluationTimeout: 3 },
  });
  const agent = async (request: Invocation<number[]>) => {
    if (request.phase === "reflection")
      return client.trace("scripted reflection").run(async (trace) => {
        await runtime.bash("cat /inbox/notice.json; grep -rl 'Use every number' /rules || true");
        await runtime.bash(`cat > /rules/sums.md <<'RULE'\n${RULE}RULE\ncat /rules/sums.md > /context.md`);
        trace.setOutput({ messages: [{ role: "assistant", content: "Proposed complete summation" }] });
        return "Proposed complete summation";
      });
    return client
      .trace("scripted sum", {
        input: { messages: [{ role: "user", content: `Sum ${request.input}` }] },
        metadata: { values: request.input },
      })
      .run(async (trace) => {
        const guidance = await runtime.bash("grep -r 'Use every number' /rules || true");
        const values = guidance.includes("Use every number") ? request.input : request.input.slice(0, 2);
        const result = values.reduce((a, b) => a + b, 0);
        trace.setOutput({ messages: [{ role: "assistant", content: String(result) }] });
        return result;
      });
  };
  try {
    const first = await runtime.run(agent, [2, 3, 4], { sessionId: "customer", executionId: "failure" });
    const reflection = await runtime.settleReflection("failure");
    if (!reflection.proposal) throw Error(`Expected proposal: ${JSON.stringify(runtime.report("failure"))}`);
    const before = Object.keys(runtime.workspace.rules());
    const cases = [
      new ReplayCase({ id: "one", input: [2, 3, 4] }),
      new ReplayCase({ id: "two", input: [10, 20, 30] }),
      new ReplayCase({ id: "protected", input: [1, 1], protected: true }),
    ];
    const verdict = await runtime.validate(reflection.proposal.id, {
      cases,
      replay: async (request) => {
        const result = await request.run(agent);
        const expected = (request.case.input as number[]).reduce((a, b) => a + b, 0);
        return new Outcome({
          scores: { task_completion: Number(result === expected), coherence: 1 },
          success: result === expected,
          evidence: { actual: result, expected },
        });
      },
    });
    const later = await runtime.run(agent, [2, 3, 4], { sessionId: "fresh", executionId: "after" });
    await runtime.waitExport({ timeout: 3 });
    return {
      original_result: first,
      candidate_status: reflection.status,
      accepted_before_validation: before,
      validation: verdict.outcome,
      validation_reason: verdict.reason,
      later_result: later,
      accepted_files: Object.keys(runtime.workspace.rules()),
      trace_roles: [
        ...new Set([...service.state.traces.values()].map((t) => t.metadata["pandaprobe.reflect"].role)),
      ].sort(),
      evaluated_roles: [
        ...new Set(
          [...service.state.runs.values()].flatMap((r) =>
            r.trace_ids.map((id: string) => service.state.traces.get(id).metadata["pandaprobe.reflect"].role),
          ),
        ),
      ],
      evaluation_creates: service.state.creates,
      poll_counts: [...service.state.polls.values()],
      workspace: runtime.workspace.path,
      real_model_calls: 0,
    };
  } finally {
    await runtime.close();
    await client.shutdown();
    await service.close();
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.includes("--help"))
    console.log("Credential-free Reflect complete loop. Run: pnpm exec tsx examples/reflect/offline-demo.ts");
  else {
    const folder = mkdtempSync(join(tmpdir(), "pandaprobe-reflect-demo-"));
    try {
      console.log(JSON.stringify(await demonstrate(folder), null, 2));
    } finally {
      rmSync(folder, { recursive: true, force: true });
    }
  }
}
