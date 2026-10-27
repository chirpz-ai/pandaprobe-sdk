# Workspace and memory

The agent sees a virtual filesystem, with no host mounts, inherited host secrets,
network commands, Python subprocess, or Docker service. `reflect_bash` is a real
embedded Bash interpreter: pipelines, grep/find, sed/awk, loops, heredocs, edits,
renames and deletions operate on its persistent in-memory filesystem.

| Virtual path | Meaning |
| --- | --- |
| `/rules/**/*.md` | Pinned guidance plus this execution's candidate edits |
| `/context.md` | Bounded active memory, included in one replaceable instruction block |
| `/inbox/notice.json`, `/inbox/evidence.txt` | Read-only assigned external evidence |
| `/decisions/status.json` | Read-only recent decisions |

Rules need a Markdown heading, `Applies to: ...`, and actionable text. Rule paths
are relative safe Markdown paths. The SDK seals the whole change set, including
edits/moves/merges/deletions, with duplicate and stale-baseline checks. The model
cannot promote or retire shared guidance through Bash. The SDK never searches,
ranks, or injects the full corpus; the agent decides which files to read.

Host persistence is separate. Under `config.workspace` (environment override
`PANDAPROBE_REFLECT_WORKSPACE`, otherwise `.pandaprobe/reflect`) the identity partition
is SHA-256 of normalized endpoint, project name and API key separated by newlines,
truncated to 24 hex characters. No key is written in plaintext. Rotation changes
the partition; history migration is an explicit operator action.

```
<workspace>/<identity>/
  state.json
  .lock
  revisions/<sha256>/manifest.json
  revisions/<sha256>/rules/**.md
  executions/<sha256-of-id>.json
  evidence/<notice-id>.txt
  events/<event-id>.json
```

`state.json.head` identifies accepted guidance. Immutable rule content and manifests
are hashed; the new revision is fsynced before atomically replacing state. Real OS
advisory locks coordinate processes. Invalid schema, missing state alongside nonempty revision manifests, and invalid
revision checksums fail on read. As in Python, missing state with only the empty
revision is reinitialized. Proposals are decoded and validated as records; there
is no additional proposal-fingerprint tamper gate. Execution
filenames hash arbitrary IDs. Working files and active memory hashes are distinct
from accepted revisions; `active_memory_unvalidated` identifies mutable memory.
Do not modify these files while a runtime is active. Use local filesystems with
working locks/atomic rename; distributed mounts and object storage are unverified.

Memory-only edits can carry into the same session at the same accepted revision.
Candidate-associated memory does not carry into a fresh ordinary execution.
Validation arms receive fresh local memory/conversations and frozen assigned rules.
An authoring conversation may retain candidate content and is not a clean control.
LangChain-family and Vercel refresh the block at model invocation checkpoints;
OpenAI dynamic instructions refresh on native runs; Claude refreshes at query
invocations. No adapter rewrites the host's entire conversation.

## Default policy (camelCase names, seconds and UTF-8 bytes)

| Configuration | Default |
| --- | --- |
| `traceVisibilityTimeout`, `evaluationTimeout`, `pollInterval` | 30, 120, 1 seconds |
| `evaluationMaxRuns`, `evaluationMaxTraces`, `evaluationSampleRate` | null, 32, 1 |
| `maxConcurrency`, `cooldownSeconds` | 4, 60 seconds |
| `reflectionTimeout`, `shutdownTimeout`, `reflectionMaxCommands` | 60 seconds, 5 seconds, 8 |
| `shellTimeout`, `shellMaxCommands`, `shellMaxLoopIterations` | 5 seconds, 2000, 5000 |
| `shellMemoryBytes`, `commandMaxBytes` | 16 MiB, 16 KiB |
| `outputMaxBytes`, `executionOutputMaxBytes` | 12 KiB, 96 KiB |
| `contextMaxBytes`, `ruleMaxBytes` | 8 KiB each |
| `rulesMaxBytes`, `rulesMaxCount`, `evidenceMaxBytes` | 512 KiB, 256, 64 KiB |
| `thresholds` | task_completion: 0.5, coherence: 0.5 |
| `diagnosticMetrics` | tool_correctness, argument_correctness |
| `stallWindow`, `minimumProgress`, `regressionDrop` | 5, 0.02, 0.15 |
| `validationMinCases`, `validationMinImprovement` | 3, 0.05 |
| `validationRegressionTolerance`, `validationTimeout` | 0.02, 120 seconds |

A shell call has bounded bytes, commands, loop iterations, memory and duration.
Filesystem violations, thrown interpreter errors and caller cancellation roll
back to the pre-call snapshot. Returned native error results and nonzero exit
statuses, including exit 124, retain valid edits, matching Python/Bashkit. A
returned resource-limit result is distinct from an exception or cancellation. Replies, including guards, consume the cumulative
output budget; counters survive restoration. This bounds tool/file traffic, not
all model context, model billing, audit-disk growth, or host domain tools.

## Portability

The pinned Bashkit npm package contains native binaries for macOS arm64/x64,
Linux GNU arm64/x64, and Windows x64. `fs-native-extensions` supplies advisory
locking. Its pinned N-API binary is loaded directly because the current transitive
JavaScript resolver uses Node 20-only Array.with; this keeps Node 18 support
without changing globals or weakening locks. Alpine/musl, browser runtimes, edge workers, Bun/Deno, network mounts and
cross-language concurrent writers are not claimed. macOS arm64 (Node 18/20/22) and a Linux arm64 container (Node 22) passed local
core and packed-package checks; native framework tests passed on Node 22 on both.
CI additionally covers Linux x64 and Windows x64. A platform in an upstream binary list is not a
claim that its CI ran in this local implementation session.

Schema-1 artifacts remain readable and representative rule/revision/proposal and
replay hashes match Python-generated fixtures. JavaScript numbers do not preserve
Python's distinction between `1` and `1.0`, arbitrary-size integers, or all exponent
spellings. Do not share a writable workspace between Python and TypeScript; neither
numeric canonicalization across all inputs nor cross-language lock compatibility
is promised. Keep money as safe integer cents and JSON inputs reproducible.

The evaluator preserves Python's ordered metric-array dispatch fingerprint. Two
otherwise equivalent threshold maps inserted in different orders can create two
claims. The per-day attempt counter is shared, while each instance uses its own
configured cap; no additional persisted cap-consistency policy is imposed. These
behaviors are covered by regression tests against the Python source contract.
