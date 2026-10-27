# PandaProbe Reflect for TypeScript

Trace → Evaluate → Reflect. Your task agent searches a persistent rule filesystem,
receives external failure evidence at a bounded checkpoint, and proposes guidance.
Independent paired outcomes decide whether that guidance becomes shared memory.
Reflect does not select another model or run a separate repair agent.

## Install and run

```sh
npm install pandaprobe @everruns/bashkit@0.19.0 fs-native-extensions@1.5.1
```

The tracing package retains zero required runtime dependencies. `/api` uses Node
fetch; `/reflect` loads the native shell and lock packages only when activated.
The generic runtime supports Node 18/20/22. Native framework fixtures use Node 22
or newer; use the tested versions in [native integrations](native.md).

From this repository, the credential-free example runs real HTTP export, eventual
trace visibility, run polling, real isolated Bash, six independent replay arms,
acceptance, and a fresh task:

```sh
# From libraries/typescript (or make ts-install-reflect at the repository root):
make install-reflect
node --import tsx examples/reflect/offline-demo.ts
```

Its scripted arithmetic task first returns 5, proposes guidance, and returns 9 in
a new execution only after acceptance. This proves plumbing, not model improvement.

## Generic host integration

```ts
import { Client } from "pandaprobe";
import { Reflect as ReflectRuntime, type Invocation } from "pandaprobe/reflect";

const client = new Client(); // PANDAPROBE_API_KEY / PROJECT_NAME / ENDPOINT
const reflect = new ReflectRuntime(client);
const reflectBash = await reflect.tool(); // async (command: string) => string

// Your existing model/tool loop remains responsible for producing the result.
async function agent(request: Invocation<string>) {
  return client.trace("support", {
    input: { messages: [{ role: "user", content: request.input }] },
  }).run(async trace => {
    const output = await yourExistingAgent({
      input: request.phase === "reflection" ? request.feedback : request.input,
      previousResult: request.previousResult,
      instructions: request.instructions,
      tools: request.phase === "reflection" ? { reflect_bash: reflectBash }
        : { ...yourBusinessTools, reflect_bash: reflectBash },
      signal: request.signal,
    });
    trace.setOutput({ messages: [{ role: "assistant", content: output }] });
    return output;
  });
}
try {
  const result = await reflect.run(agent, "Resolve my return", {
    sessionId: "customer-42", executionId: crypto.randomUUID(),
  });
  console.log(result); // original task result, never the reflection's reply
} finally {
  await reflect.close();
  await client.shutdown();
}
```

`yourExistingAgent` and `yourBusinessTools` above are host code, not SDK exports.
The [offline demo](offline-demo.ts) is a complete runnable generic integration;
the [native quickstarts](native.md) supply tools and instruction composition for you.
Keep domain actions inaccessible during generic reflection, including their actual
execution paths. Native adapters enforce their supported administration boundary.

`run` captures existing traces through the explicit client, waits for evaluation,
and re-enters the same host after the task boundary when a notice is assigned.
Original task objects and errors are preserved; a failed task is evaluated but is
not automatically re-entered. `execution` schedules evaluation without invoking a
continuation. External work can call `submitTraces(ids, {executionId, sessionId,
completed})`; IDs must describe task activity and cannot be rebilled under a new
execution. Work must already be traced: Reflect does not manufacture task traces.

For streaming, keep **consumption** inside the boundary:

```ts
await reflect.execution({ sessionId: "stream", executionId: crypto.randomUUID() }, async execution => {
  const stream = await yourStreamingAgent(execution.instructions(), await reflect.tool());
  for await (const event of stream) consume(event);
});
```

Native adapters accept `invokeOptions: { stream: true }` where supported and consume
the native stream before leaving the scope. They return the native completed result;
this convenience is not a token-forwarding UI interface. Use an explicit boundary
for interactive streaming, custom graphs, or specialized provider orchestration.
Claude already exposes its query as a stream and returns its collected messages.

## Validation

Without an independent verifier, proposals stay candidates/pending. Register
`{ replay, cases }` in the constructor to schedule replay after the task releases
its resources, or call `validate(proposal.id, {cases, replay})` explicitly.

```ts
import { Outcome, ReplayCase } from "pandaprobe/reflect";
const cases = [
  new ReplayCase({ id: "shipping-a", input: { order: "held-a" } }),
  new ReplayCase({ id: "shipping-b", input: { order: "held-b" } }),
  new ReplayCase({ id: "protected", input: { order: "held-free" }, protected: true }),
];
const verdict = await reflect.validate(proposal.id, {
  cases,
  replay: async request => {
    const sandbox = await createFreshBusinessEnvironment(request.case.input);
    const agent = await createYourAgent(sandbox);
    const result = await request.run(agent);
    const checked = await independentlyVerify(sandbox, result);
    return new Outcome({ scores: checked.scores, success: checked.success, evidence: checked.evidence });
  },
});
```

The host creates fresh business state; Reflect cannot undo external side effects.
Required metrics must all be independently measured, finite and within `[0,1]`;
`success` must be a boolean. Default minimum: three distinct pairs, one protected
successful control, no regression beyond 0.02, and mean improvement ≥0.05 on
non-protected comparisons. Smaller guidance can be accepted for measured
non-inferiority, explicitly without claiming behavioral improvement. A changed
assigned snapshot, reused execution, incomplete pair, or stale accepted head cannot
publish. `trialExecution` and `validator.recordTrial` support explicitly assigned
forward pairs; coincidental later successes are not trials.

There is **no delivery checker in core**. Acceptance depends on independent
outcomes, including when the model reads a relevant unformatted excerpt. The live
examples' retrieval audit is descriptive and never a promotion condition.

## Waiting, errors and limits

All configuration durations and wait timeouts are **seconds**; file/output budgets
count UTF-8 bytes. See [workspace and limits](workspace.md) for every default.
`settleEvaluation(id, {timeout, signal})`, `settleReflection`, and `settleValidation`
bound caller waiting; timing out a waiter does not cancel shared work. `close()`
cancels owned work, propagates abort signals and bounds shutdown waiting (5 seconds
by default). JavaScript cannot forcibly terminate arbitrary uncooperative host code;
late results are observed and closed shell calls cannot mutate files.

Use `continueReflection(id, sameAgent, {input, previousResult})` for late feedback.
Provide the original native result/history unless a native session owns it.
`resumeEvaluations()` polls durable known run IDs after restart. Creation is
single-attempt: a lost response leaves `creation_unknown` and is never recreated
automatically. Reconcile it against the platform before an operator changes state;
there is no server idempotency-key contract. Read retries are bounded (30-second
API deadline, three GET attempts); writes are not retried and redirects are refused.

Evaluation outcomes: `evaluated`, `skipped`, `pending`, `insufficient`, `failed`,
`creation_unknown`, `cancelled`. Proposal generation: `proposed`, `duplicate`,
`no_proposal`, `invalid`, `conflict`, `timed_out`, `failed`, `cancelled`. Lifecycle:
`candidate`, `pending`, `accepted`, `rejected`. Completed processing is not proof
of repair. `report(id)`, `proposals()`, workspace events and validation reasons keep
these facts separate. API errors distinguish auth, not-found, rate limit, transport,
invalid response, deadline, failed evaluation, insufficient evidence and unknown
creation. Core API numeric scores can lie outside `[0,1]`; Reflect rejects those as
normalized outcome evidence.

There is **no default 100-evaluation quota**. `evaluationMaxRuns: null` is unlimited
count. An optional positive integer caps creation attempts per UTC day across a
shared identity workspace, including diagnostic/failed/unknown attempts. Polling
known runs continues at the cap. Zero is invalid. The attempt counter is shared, and each instance uses its configured limit,
matching Python. This counter is neither a monetary budget nor a platform quota. Already skipped executions do not silently retry after midnight.

## More

- [Native framework quickstarts](native.md)
- [Workspace, security boundary and defaults](workspace.md)
- [Multi-turn support guide](live_support.md)
- [Controlled refund failure guide](live_failure.md)
- [Parity and verification inventory](parity.md)
