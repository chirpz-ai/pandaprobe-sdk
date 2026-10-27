# PandaProbe Reflect: TypeScript implementation handoff

## 1. Mission and source of truth

Implement the TypeScript equivalent of the released Python **PandaProbe Reflect**
inside the existing TypeScript `pandaprobe` SDK. The product decisions are settled.
This is an implementation and verification task, not another architecture research
or brainstorming phase.

The user reports that Python Reflect was merged and released successfully. This
handoff was checked against Python **0.6.0**, local commit `eb5f09d`, on 2026-10-26.
At the start of your session, inspect the actual merged code: the repository may
have advanced. Python source and meaningful behavioral tests are the executable
reference; this document explains the intent, boundaries, and pitfalls.

**Replicate Python behavior 1:1**, using idiomatic TypeScript APIs and appropriate
Node implementation primitives. Preserve the lifecycle, authority boundaries,
failure semantics, configuration defaults, evidence, and developer experience.
Do not literally translate Python concurrency primitives or infer that a Python
framework API exists in JavaScript. Choose and verify the TypeScript mechanisms
yourself. Do not quietly omit difficult behavior, replace it with a stub, or
redesign the product to simplify the port.

If source, tests, and prose disagree, inspect the relevant history, document the
specific discrepancy, and resolve it explicitly. Do not reproduce an established
bug merely to claim parity, or introduce speculative improvements as migration
requirements. Escalate only concrete blockers or a necessary change to the agreed
contract; routine implementation decisions do not need another approval gate.

This document replaces the old root `plan.md` as the active implementation plan.
The old Python brief and implementation log remain in Git history, for example
`git show eb5f09d:plan.md`. They contain superseded architecture options and
delivery-checker designs; they must not override this handoff or current Python.

## 2. Workspace, scope, and working rules

| Reference | Location |
| --- | --- |
| Repository | `/Users/sina/Projects/PandaProbe/pandaprobe-sdk` |
| Target implementation | `libraries/typescript/` |
| Python behavioral reference | `libraries/python/src/pandaprobe/reflect/` |
| Reusable Python platform API | `libraries/python/src/pandaprobe/api/` |
| Python trace capture | `libraries/python/src/pandaprobe/tracing/capture.py` and `client.py` |
| Python tests | `libraries/python/tests/` |
| Python product guide and examples | `libraries/python/examples/reflect/` |
| Existing TS integrations and examples | `libraries/typescript/src/integrations/`, `libraries/typescript/examples/` |
| Backend contracts, if verification is needed | `/Users/sina/Projects/PandaProbe/pandaprobe/backend/` |
| Historical Harness, normally unnecessary | `/Users/sina/Projects/PandaProbe/pandaprobe-harness/` |
| Historical CLI, never a runtime dependency | `/Users/sina/Projects/PandaProbe/pandaprobe-cli/` |

Before editing, read applicable `AGENTS.md`/`CLAUDE.md`, inspect Git status and
current branches, and preserve user changes. Start a new branch from the merged
baseline, such as `feat/typescript-reflect`; do not continue implementation on the
old Python feature branch. Do not stage, commit, push, merge, tag, publish, or bump
a release version unless the user separately asks. The task is to produce a
complete, reviewable implementation with verification results.

Work primarily in `libraries/typescript`, plus focused TypeScript CI changes and
this plan. Keep Python implementation, live workspaces, historical results, and
research artifacts unchanged. Do not edit either the Harness paper or the Context
Language Models paper. There is no need to repeat that literature review. Do not
expand this into backend implementation, CLI work, website work, a general tracing
rewrite, or core-service documentation changes. Report necessary backend changes
as explicit follow-ups rather than silently adding them.

Update the checklist in section 16 as work progresses. Mark a step **DONE** only
when its implementation and relevant verification are complete; record results
and remaining limitations. Continue through implementation, tests, documentation,
and package verification rather than stopping at scaffolding or a design proposal.

## 3. Settled product and architectural decisions

The public product stack is **Trace → Evaluate → Reflect**. The operational loop
is **Detect → Notify → Reflect → Validate**. Reflect changes execution guidance,
not model weights. A plausible reflection is not proof of repair, and accepted
guidance is evidence under tested conditions, not a permanent guarantee.

1. **The developer's task agent performs reflection.** The host retains its model,
   prompts, framework, domain tools, execution loop, and environment. Reflect
   re-enters that agent at an explicit bounded checkpoint with external evidence.
   There is no package-owned second reflection agent, separately selected repair
   model, hidden provider wrapper, or autonomous model loop owned by Reflect.
2. **The filesystem is the memory interface.** The SDK always supplies one general
   `reflect_bash` tool through native framework integration. Agents search, read,
   write, consolidate, move, and remove rule files with normal shell operations.
   Do not resurrect lists of specialized rule/mailbox/proposal/acknowledgement
   tools or disguise them as bespoke shell commands.
3. **Agents select guidance.** The SDK does not rank rules or automatically insert
   the rule corpus. A small instruction block tells the agent how to use the
   filesystem. An editable `context.md` supplies bounded active memory at explicit
   checkpoints. Full CLM-style conversation rewriting/compression is excluded.
4. **Lifecycle administration is deterministic SDK code.** Platform calls,
   evaluation scheduling, file diffs, records, budgets, independent validation,
   and accepted-revision publication are not agent responsibilities. Bash replaces
   agent-facing memory tools, not SDK HTTP APIs or validation authority.
5. **Only independent validation publishes shared guidance.** Agent writes are
   proposals. Additions, edits, merges, moves, and deletions use the same lifecycle.
   Candidate exposure and acceptance are different facts.
6. **The core SDK has no delivery checker.** Do not scan model inputs for rule
   text, parse grep output to authorize acceptance, require whole-file reads,
   persist delivery receipts, or add delivery-observer hooks. Those mechanisms
   were explicitly removed from Python. Example/test-only audits may demonstrate
   retrieval; acceptance depends on independent outcomes.
7. **Developer integration must remain simple.** Normal use should take only a
   few minutes, with SDK-owned defaults and a supplied tool. Do not ask developers
   to choose a shell implementation, build an MCP server, configure a second model,
   or duplicate endpoint/auth/project settings for the normal path.

## 4. Packaging and preservation of existing tracing

Use the existing npm **`pandaprobe`** distribution. Add explicit public subpaths
such as **`pandaprobe/reflect`** and **`pandaprobe/api`**, following repository
conventions. Do not create a separate public Reflect package or depend on Python,
a Python subprocess, or the PandaProbe CLI to implement the TypeScript feature.

Python installs optional runtime dependencies with `[reflect]`; npm does not have
that mechanism. Select a simple optional-dependency/install arrangement appropriate
to this SDK and document the exact working installation command. The invariant is
that ordinary tracing remains lightweight and usable without the shell engine,
frameworks, or model libraries. Do not eagerly load them from the SDK root. Test
the actual published package, not just source imports. Preserve ESM, CommonJS,
declaration files, explicit exports, and the existing base dependency policy.

The inspected TS SDK advertises Node **18+**, tests Node **18/20/22**, uses strict
TypeScript, tsup, Vitest, Biome, pnpm, and optional framework peer dependencies.
Its `.npmrc` disables automatic peer installation. Verify these facts at execution
time. Preserve the current tracing support range; if a framework or shell engine
needs a newer runtime, investigate and state the precise limitation rather than
silently raising the entire SDK's engine requirement or claiming unsupported
compatibility. Reflect's persistent filesystem is a Node capability; do not imply
it works in browsers or filesystem-free edge runtimes without implementation.

Preserve existing tracing adapters, model wrappers, initialization, schemas,
session identity, streaming behavior, exceptions, and export transport. Reflect
does not install a second set of tracing patches. Importing the SDK or enabling
tracing must not create workspaces, invoke models, start evaluation polling, or
activate reflection. Do not run evaluation/reflection in the export worker.
Reflect intentionally enabled has resource and latency overhead; do not advertise
zero overhead.

A small additive trace-capture interface is part of the Python implementation:

- `Client.capture_traces()` observes trace IDs submitted by existing instrumentation
  inside a host-defined scope. It creates no traces and guarantees no persistence.
- It decorates a copied export payload with explicit phase metadata/session, keeps
  unique IDs with an overflow flag, and restores the surrounding capture scope.
- Capture is client-owned and scoped to an async invocation. Multiple clients,
  instances, nested scopes, and concurrent sessions cannot use a global
  last-active-runtime router.
- Descendant work finishing after closure keeps its inherited attribution, but
  cannot extend the completed boundary's collected IDs or observer callbacks.
- Observer failures cannot alter task results or export. The optional submission
  callback is not permission to reintroduce a Reflect delivery checker.

Use the TS SDK's existing context facilities where appropriate. Test async
propagation, nested scopes, late work, cancellation, and independent clients;
do not assume Python `ContextVar` behavior maps automatically to JS promises.

## 5. Python implementation map: read before porting

All paths below are relative to `libraries/python/`.

| Responsibility | Primary sources | Behavioral checks |
| --- | --- | --- |
| Public exports, configuration, records | `src/pandaprobe/reflect/__init__.py`, `config.py`, `models.py` | Configuration checks throughout Reflect tests |
| Native platform HTTP and typed resources | `src/pandaprobe/api/{client,resources,models,errors}.py` | `tests/test_api.py`, `tests/fixtures/platform_contract.json` |
| Capturing existing trace submissions | `src/pandaprobe/tracing/capture.py`, `src/pandaprobe/client.py` | `tests/test_trace_capture.py` |
| Atomic authoritative state and revisions | `src/pandaprobe/reflect/workspace.py` | `tests/test_reflect_workspace.py` |
| General isolated shell and budgets | `src/pandaprobe/reflect/shell.py` | `tests/test_reflect_shell.py` |
| Execution scope, memory, checkpointing | `src/pandaprobe/reflect/execution.py` | `tests/test_reflect_runtime.py` |
| Evaluation scheduling and signal gates | `src/pandaprobe/reflect/evaluation.py`, `signals.py` | `tests/test_reflect_evaluation.py` |
| Orchestration and explicit waits | `src/pandaprobe/reflect/runtime.py` | `tests/test_reflect_runtime.py` |
| Independent replay and matched trials | `src/pandaprobe/reflect/validation.py` | `tests/test_reflect_validation.py` |
| Framework tool conversion and continuation | `src/pandaprobe/reflect/integrations.py`, `adapters/` | `tests/test_reflect_integrations.py` |
| Minimal install and offline complete loop | `examples/reflect/offline_demo.py` | `tests/test_reflect_installation.py`, `test_reflect_e2e.py` |
| Native examples and independent refund verifier | `examples/reflect/_support/`, framework `04_reflect.py` and `05_reflect_failure.py` | `tests/test_reflect_examples.py`, `test_reflect_failure_example.py`, `test_reflect_failure_frameworks.py` |
| Example-only retrieval evidence | `examples/reflect/_support/delivery.py` | `tests/test_reflect_delivery_example.py`, `tests/fixtures/reflect/numbered_grep.json` |
| Current user-facing behavior | `examples/reflect/README.md`, `workspace.md`, `live_support.md`, `live_failure.md`, `framework_quickstarts.py` | Runnable examples and their tests |

Also inspect the corresponding TS client/config/transport, tracing context/session,
integration implementations, `test/`, `package.json`, `tsup.config.ts`, Makefiles,
and `.github/workflows/ci-typescript.yml` / `release-typescript.yml`.

## 6. Public capabilities and execution lifecycle

Provide TypeScript counterparts for the current public Python exports:

- `Reflect`, `ReflectConfig`, `Execution`, `Invocation`.
- `Rule`, `RuleProposal`, `ProposalOutcome`, `EvaluationOutcome`,
  `ExecutionReport`, `ValidationVerdict`.
- `Outcome`, `ReplayCase`, `ReplayRequest`, `TrialCase`, `TrialArm`.
- The independent `PlatformClient`, its trace/evaluation resources, typed results,
  and distinguishable errors from `pandaprobe.api`.

Use idiomatic casing/options and documented JS lifecycle management. Do not
promise the illustrative signatures below until implemented. Preserve these
capabilities rather than mechanically exposing every private Python helper:

| Python capability | Required TypeScript behavior |
| --- | --- |
| `Reflect(client, config=..., api=..., replay=..., cases=...)` | Reuse resolved identity; own a workspace and separate API lifecycle; optional external API and verifier |
| `agent(factory, ...)`, `tool(framework)`, `bash(command)` | Native setup with supplied tool; explicit generic fallback; tool bound to the current instance/execution |
| `run(agent, input, session_id=..., execution_id=...)` | Task invocation followed by bounded evidence/reflection settlement; original task result preserved |
| `execution(...)` | Framework-independent application scope, including custom/streaming loops |
| `submit_traces(...)` | Explicit completed trace references with stable execution and session identities |
| `wait_export()` | Wait for existing export separately from persistence/evaluation |
| `settle_evaluation(id)`, `resume_evaluations()` | Wait/resume durable evidence work without recreating an existing run |
| `continue_reflection(id, continuation, ...)` | Explicit host continuation, including late evidence after task return |
| `settle_reflection(id)` | Wait for a checkpoint already scheduled |
| `validate(id, cases, replay)`, `settle_validation()` | Independent replay outside the task resource barrier |
| `trial_execution(...)`, `validator.record_trial(...)` | Explicit matched control/candidate assignments and host-attested outcomes |
| `report(id)`, `proposals()` | Inspect evidence and lifecycle without a misleading repair-success flag |
| `aclose()` | Bounded cleanup of owned tasks/resources; external API clients remain host-owned |

The actual lifecycle to preserve:

1. Construct Reflect explicitly. Resolve the workspace identity from the same SDK
   client configuration. Do not start a task or model merely by importing code.
2. Enter an execution with a unique invocation ID and a session grouping ID. Pin
   the accepted revision. Create its own virtual shell, working rules, memory,
   trace capture, and persisted execution snapshot.
3. Invoke the developer's agent with existing tracing, the supplied tool, and one
   current memory/instruction block. Preserve domain prompts and tools for task
   work. The application boundary must cover stream consumption and actual work,
   not just obtaining an iterator or initiating a provider request.
4. On boundary exit, record task status, seal any working rule diff into a proposal,
   checkpoint memory/usage, submit captured task trace references, and close the
   shell. Ordinary task writes can propose changes too; proposal generation is
   not exclusively permitted after a judge notice.
5. Discover persisted terminal traces, evaluate configured signals, apply gates,
   and persist an assigned failure notice with diagnostics when warranted.
6. `run()` can await this bounded process and re-enter the same host agent at a
   reflection checkpoint. A generic host uses `Invocation` with input, execution,
   phase, prior result, feedback, and current instructions. Native continuation
   must respect framework-owned conversation state.
7. Reflection reads `/inbox`, checks existing rules, and makes at most one coherent
   change set per checkpoint, or leaves files unchanged with a rationale. It must
   not repeat business actions, poll for acceptance, or acknowledge notices through
   extra tools. Native integrations restrict capabilities to memory administration.
8. Record proposal outcome and schedule configured validation independently. Return
   the original task result. A reflection failure must not replace it; an original
   task exception must remain the original exception. Failed tasks may still have
   evaluable traces, but a failed invocation is not an automatic valid continuation.
9. Fresh ordinary executions see accepted rules. Candidate trials are explicitly
   assigned. No replay is inferred from a trace alone and no acceptance comes from
   a model's self-assessment.

Keep completion of task work, trace export, persistence, evaluation, proposal,
validation, and shutdown distinct. `completed` on submitted work describes the
host's boundary/finality, not proof of business success. A session is not a unique
execution ID. No tracing means insufficient trace evidence, not synthetic success.

Only explicit host continuations can handle feedback arriving after return.
Persist reflection claims so concurrent processes/restarts do not repeat an
already claimed continuation. Interrupted claims need review; do not silently
rerun arbitrary host behavior. Caller wait deadlines must not unintentionally
cancel shared evaluation tasks. Shutdown does cancel owned work. Bound waiting
when host code ignores cancellation, close its shell, observe eventual errors,
and preserve attribution. Do not claim a timeout undoes an accepted provider call
or forcibly stops arbitrary host code.

## 7. Filesystem, rules, context, and authority

### Agent-visible virtual filesystem

| Path | Semantics |
| --- | --- |
| `/rules/**/*.md` | Editable working snapshot, scoped by directory |
| `/context.md` | Editable active memory; bounded and session-local, not accepted guidance |
| `/inbox/notice.json` | Read-only failure notice assigned by SDK code |
| `/inbox/evidence.txt` | Read-only diagnostic evidence with truncation made explicit |
| `/decisions/status.json` | Read-only recent validation summaries; current Python exposes the last 20 proposals |

Python uses an optional embedded **Bashkit** interpreter, not the host's Bash
process. The TS implementation must offer comparable ordinary shell semantics
and isolation. Select and verify a suitable implementation; the choice of a JS,
native, or WASM binding is your implementation work. Do not require developers to
bring an executor. Do not silently fall back to unrestricted `child_process`, a
host shell, a Python bridge, or a tiny hand-written parser that only recognizes
the example commands.

Required operations include grep/find/cat, pipelines, redirection, heredocs,
loops, mkdir, edits with standard utilities, mv, rm, and persistent per-execution
shell/filesystem state. The supplied environment has no host mounts, inherited
credentials, network, arbitrary external programs, or Python execution. Shell
instances do not share interpreter state. SDK-owned immutable state is never
mounted writable inside the shell. If the host separately provides privileged
filesystem tools, document that those tools can bypass OS-level isolation; the
SDK cannot protect storage from another tool with the same host permissions.

Preserve serialized shell operations, timeout/cancellation cleanup, snapshots and
rollback for invalid filesystem changes, thrown interpreter failures and caller
cancellation, read-only evidence after rollback, traversal limits, regular-file
checks, and UTF-8 byte clipping. Per the user clarification, returned native
resource/error results (including exit 124) retain valid edits, as Python does.
Do not export symlinks or paths escaping the rule tree. Distinguish malformed
filesystem changes from a valid shell command returning a nonzero exit status.
Checkpoint accepted working mutations and usage after tool execution.

Tool output includes stdout/stderr and an exit/status/budget summary. Per-call
and cumulative output bounds include control/error replies. Late calls after
closure and fully depleted output allowance can produce empty output; this is
not automatically an engine failure. Truncation must be visible. Limits constrain
this tool's output, not the model's full context, unrelated tools, or historical
conversation accumulation. Do not advertise an end-to-end token guarantee.

### Rule records and proposal diffs

Port the actual `Rule` validation and normalization from `models.py`: relative
safe Markdown paths, no traversal/special characters, maximum path length, title,
explicit `Applies to: ...` line, meaningful guidance, newline normalization, and
content-derived SHA-256 version. Root-level scope is `global`; directories define
other scopes. Do not substitute embeddings or a new memory database.

At sealing, compare the whole working snapshot against its pinned base. A proposal
stores changed/new files and `null` for removals, execution/session attribution,
base revision, deterministic fingerprint, evidence IDs, timestamp, rationale,
status, and later decision. No diff means `no_proposal`.

Preserve exact change-set duplicate detection and normalized identical-guidance
detection across filenames. These are not semantic deduplication. Conflicts,
contradictions, applicability, and stale guidance require agent reasoning and
independent validation. Concurrent changes to accepted head cannot be overwritten
by a stale proposal. Rule cleanup is a validated change, not an administrative
shortcut to deleting shared guidance.

### Durable host filesystem

Current layout, under `.pandaprobe/reflect/<identity-digest>/` by default:

```text
state.json                         # accepted head and authoritative lifecycle records
.lock                              # cross-process coordination
revisions/<revision>/manifest.json  # rule path -> version
revisions/<revision>/rules/**.md    # immutable accepted snapshots
executions/<hashed-id>.json         # working files, memory, attribution, usage
evidence/<notice-id>.txt            # preserved assigned diagnostic evidence
events/<event-id>.json              # append-only event records
```

The namespace is the first 24 hex characters of SHA-256 over the normalized
endpoint, project name, and API key separated by newlines. Do not print credentials
or store them in plaintext in the namespace. Credential rotation changes the
partition. Preserve endpoint/project/credential isolation and explain migration
of history as an explicit operator action.

`state.json` schema version 1 owns accepted head, proposals/decisions, evaluation
dispatch/results, trials, signal history/cooldowns, notices, usage counters,
execution requests, trace ownership, and session memory. Execution filenames
hash arbitrary invocation IDs; IDs are not paths. Execution snapshots distinguish
working rule edits from accepted history and record `active_memory_hash` and
`active_memory_unvalidated`.

Use real cross-process locking plus atomic replace/publication, not just an
in-process mutex. Write a complete immutable revision before moving the accepted
head. Follow Python's schema and revision checksum validation. Missing state with
nonempty revision manifests fails; with only the empty revision it reinitializes.
Per the user clarification, do not add initialization markers or a proposal
fingerprint gate beyond Python's record validation. Test concurrent
instances/processes and interruption between revision creation and head update.
Audit-history growth is separate from bounded active-rule growth; no unrequested
automatic deletion of provenance. Local filesystems with locking/atomic rename
are the supported model, not object storage or unverified distributed mounts.

Preserve record formats and deterministic hashes with Python-generated fixtures.
Check UTF-8, newline normalization, key ordering, JSON whitespace/escaping, numeric
and boolean validation, and the different serialization used for revision hashes,
proposal fingerprints, evaluation keys, and replay input hashes. Do not assume
plain `JSON.stringify` matches every Python `json.dumps` call. Establish readable
artifact/semantic parity; do not claim simultaneous Python/TS workspace writes
are interoperable until both serialization and lock protocols are verified.
If exact interoperability needs a format change, document and discuss that
concrete issue rather than silently changing schema version 1.

### Active-memory lifecycle

The constant instruction preamble plus current `context.md` form one replaceable
memory block. Relevant rules reach context only when the agent reads them through
its tool or writes relevant material into active memory. A file merely existing
on disk does not influence a model.

Refresh/replace the memory block at the native integration's supported invocation
checkpoints without repeatedly appending old copies. Do not claim every framework
refreshes it between every tool call; port and test the actual checkpoint behavior.
Do not rewrite or compact the host's full conversation.

Memory-only changes can persist to the same session at the same accepted revision.
Memory associated with unvalidated rule changes is not carried into a fresh
ordinary execution. New revisions invalidate incompatible session memory, and
validation uses fresh local memory. Authoring conversations may already have read
candidate material; that does not make them clean controls.

## 8. Native platform API, evaluation, and detection

Build reusable native resources outside the Reflect policy module. Use the
existing resolved SDK client identity, including explicitly provided configuration;
do not fall back to environment-only settings. HTTP/auth/project settings should
be configured once. Own a separate asynchronous HTTP lifecycle from trace export.

Current Python resources use these paths relative to the configured endpoint:

| Operation | HTTP |
| --- | --- |
| Trace list, pagination, session filtering | `GET traces` |
| Trace and span evidence | `GET traces/{trace_id}` |
| Evaluation metric metadata | `GET evaluations/trace-metrics` |
| Create a batch trace evaluation | `POST evaluations/trace-runs/batch` |
| Existing run state | `GET evaluations/trace-runs/{run_id}` |
| Run scores | `GET evaluations/trace-runs/{run_id}/scores` |

Authentication uses `X-API-Key` and `X-Project-Name`. Validate identities and avoid
redirecting credentials to another endpoint. Preserve additive response fields
while checking required fields/statuses. The actual Python layer does not need
monitor CRUD or a new session-evaluation orchestration path; do not add those just
because the old Harness/CLI had related commands.

Use `tests/fixtures/platform_contract.json`, `scripts/update_platform_contract.py`,
and `tests/test_api.py` as the contract reference. If necessary verify backend
routes/schemas under `backend/app/api/v1/routes/{traces,evaluations}.py`, dependencies,
and OpenAPI. Do not import backend implementation modules or translate CLI strings
into a facade. Contract fixtures must remain reproducible from OpenAPI.

Preserve these distinct conditions:

- Export accepted but trace not persisted yet: bounded visibility polling;
  both `COMPLETED` and `ERROR` task traces can be terminal evidence.
- Evaluation accepted/running: poll the same run ID.
- Evaluation completed: require one usable finite numeric score for every expected
  trace/metric pair, belonging to that run. Missing, failed, invalid, or duplicate
  scores mean insufficient evidence, never success.
- Failed/cancelled evaluation, auth failure, rate limiting, transport failure,
  invalid response, caller cancellation, and deadline: preserve distinct outcomes.
- Core API numeric score decoding and Reflect's normalized `[0,1]` signal checks
  are separate; do not coerce booleans/strings into valid host outcome evidence.

**Evaluation creation is deliberately single-attempt.** The Python baseline's
backend has no idempotency-key contract. Persist a dispatch claim before POST;
record the returned run ID durably as soon as possible. An ambiguous transport
failure, timeout, server error, malformed successful response, or cancellation
can leave `creation_unknown`. A crash leaving a dispatch claim without an ID
must not trigger another creation. Request IDs are not idempotency keys. Resume
polling known runs after restart; never automatically recreate unknown ones.
Future server-supported idempotency is a separate coordinated backend feature.

Safe GETs have bounded retries with a total deadline, jitter, and bounded
`Retry-After` handling. Defaults are 30 seconds and three read attempts in the
Python PlatformClient. Non-GETs are not retried. A JS `Promise.race` alone does not
cancel an underlying network operation: implement cancellation/resource behavior
and verify it, without promising to undo a server-side action.

Evaluation scheduling must retain:

- Stable normalized trace IDs, explicit execution identity, and matching policies
  on repeat submissions. One trace cannot become new billable evidence under a
  different execution ID. Same-runtime work is single-flight; durable claims
  coordinate across instances/processes and restarts.
- Deterministic execution-ID sampling, trace-count bounds, per-instance concurrency,
  and rejection of reflection/validation activity as task evidence.
- Primary metrics from configured thresholds; aggregate the worst trace per metric
  within an application boundary. Missing measurements do not become a trajectory.
- Threshold breach for completed work; regression/stall gates for intermediate
  work, with the exact precedence/window logic in `signals.py`. Session history
  is bounded, repeated execution IDs are not multiple observations, and cooldowns
  apply per failure signature.
- Diagnostics only when warranted, retaining primary notice/evidence if diagnostics
  are unavailable. Judge explanations are fallible evidence, not authority.
- Bounded evidence text with diagnostics before large raw traces, explicit
  truncation, and original trace/run IDs retained for investigation.

### Evaluation usage is not a default 100-run quota

`evaluation_max_runs` defaults to **None/unlimited count**. An explicitly configured
positive integer caps creation attempts per UTC day across a reused identity
workspace, including detection, diagnostics, failed attempts, and unknown attempts.
Polling existing runs does not count and continues after reaching the cap. Track
usage even without a cap and reset at UTC midnight. The counter is shared; each
instance uses its configured cap, matching Python. Per the user clarification,
do not persist or enforce an additional shared-cap consistency policy. Zero is
not an unlimited sentinel.

At the cap, new detection is skipped with a clear reason; missing diagnostics
must not discard primary evidence. Preserve Python's state/event/report fields;
do not add a separate usage API or report field. Task execution and existing
guidance remain available.
Already skipped executions do not silently retry the next day. This is a local
cost proxy, not a provider/platform quota or a monetary spending limit.

## 9. Independent validation and candidate exposure

Without an independent verifier, proposals remain candidates/pending. A model's
rationale, a successful reflection turn, or evidence that text was delivered is
not acceptance. The model cannot call a promotion/retirement API through Bash.

Port `ReplayCase`, `ReplayRequest`, `Outcome`, matched `TrialCase`/`TrialArm`, and
the exact decision algorithm in `validation.py`:

1. Freeze the candidate's base revision; apply its change set only in candidate
   arms. Use fresh execution IDs, conversations, local memory, and host business
   state for each arm. Controls have no candidate IDs; candidates identify exactly
   the proposal. Never reuse authoring conversation history as a control.
2. The host supplies reproducible JSON inputs and an independently implemented
   verifier. SDK code cannot clone arbitrary external state or reverse side
   effects. The host must reset environments/ledgers and choose meaningful cases.
3. Alternate arm order deterministically by case ID, preserving the Python rule.
   Record input fingerprint, base revision, identities, assignment, scores, success,
   evidence, and protected status. Reject changed assigned rule snapshots.
4. Default minimum is three distinct paired cases, including a protected successful
   case. Required metrics must all be present, finite, and normalized; success
   must be an actual boolean. Arm metric sets must agree; execution evidence
   cannot be reused. Invalid/missing evidence stays pending.
5. A protected control that no longer succeeds is insufficient protection. A
   successful control becoming an unsuccessful candidate is a regression. Any
   metric drop beyond tolerance rejects the proposal even if other cases improve.
6. Accept when the mean metric improvement over non-protected comparisons meets
   the configured threshold and the preceding checks pass. Smaller total guidance
   can instead be accepted for measured non-inferiority, with an explicit reason
   that no behavioral improvement is claimed. Otherwise remain pending.
7. Only the validator updates authoritative proposal status and accepted head.
   If head changed while testing, keep the candidate pending for rebase/revalidation.
   A decided proposal is stable; do not keep trying it until accepted.

Replay runs separately from the task-turn barrier so it can acquire the same
resources later. Host errors, missing evidence, deadline, and cancellation cannot
promote a candidate. Persist partial/pending evidence and propagate cancellation
appropriately. Shutdown remains bounded even for uncooperative host code.

For controlled forward trials, the host explicitly assigns/matches arms and
submits verified outcomes. Duplicate identical case records are idempotent;
replacing a case ID with different evidence is rejected. Unrelated later successes
or coincidental timestamps do not establish a controlled comparison. This is
explicit paired trial support, not automatic attribution of all future outcomes.

## 10. Configuration parity

The Python `ReflectConfig` is authoritative. Preserve defaults and their meaning;
translate naming consistently and document time units rather than silently
interpreting seconds as JavaScript milliseconds. The values below describe the
baseline SDK, not the live examples' deliberate overrides.

| Python field(s) | Default |
| --- | --- |
| `workspace` | `PANDAPROBE_REFLECT_WORKSPACE` or `.pandaprobe/reflect` |
| `trace_visibility_timeout`, `evaluation_timeout`, `poll_interval` | 30 s, 120 s, 1 s |
| `evaluation_max_runs`, `evaluation_max_traces`, `evaluation_sample_rate` | unlimited count, 32, 1.0 |
| `max_concurrency`, `cooldown_seconds` | 4, 60 s |
| `reflection_timeout`, `shutdown_timeout`, `reflection_max_commands` | 60 s, 5 s, 8 |
| `shell_timeout`, `shell_max_commands`, `shell_max_loop_iterations` | 5 s, 2,000, 5,000 |
| `shell_memory_bytes`, `command_max_bytes` | 16 MiB, 16 KiB |
| `output_max_bytes`, `execution_output_max_bytes` | 12 KiB, 96 KiB |
| `context_max_bytes`, `rule_max_bytes` | 8 KiB each |
| `rules_max_bytes`, `rules_max_count`, `evidence_max_bytes` | 512 KiB, 256, 64 KiB |
| `thresholds` | `task_completion: 0.5`, `coherence: 0.5` |
| `diagnostic_metrics` | `tool_correctness`, `argument_correctness` |
| `stall_window`, `minimum_progress`, `regression_drop` | 5, 0.02, 0.15 |
| `validation_min_cases`, `validation_min_improvement` | 3, 0.05 |
| `validation_regression_tolerance`, `validation_timeout` | 0.02, 120 s |

Validate finite ranges, positive counts/timeouts, nonnegative cooldown, and
nonempty threshold maps. Reject invalid booleans/numeric coercion, NaN, and
Infinity where Python rejects them. Copy mutable caller configuration. Do not
invent a second reflection-model configuration or legacy `HARNESS_*` aliases.

Preserve explicit outcome vocabularies:

- Evaluation: `evaluated`, `skipped`, `pending`, `insufficient`, `failed`,
  `creation_unknown`, `cancelled`.
- Proposal generation: `proposed`, `duplicate`, `no_proposal`, `invalid`,
  `conflict`, `timed_out`, `failed`, `cancelled`.
- Proposal lifecycle: `candidate`, `pending`, `accepted`, `rejected`.
- Validation: `accepted`, `rejected`, `pending`, with mode `replay` or `trial`.

Model rationale and deterministic decision reason are separate. A container
status such as a completed episode means processing ended, not behavioral repair.

## 11. Framework integration scope and lessons

At handoff, the existing TS SDK supports **LangChain, LangGraph, DeepAgents,
OpenAI Agents, Claude Agent SDK, and Vercel AI SDK**. Supply Reflect integrations
and examples for these existing surfaces. Reinspect the current inventory first.
Python also has Google ADK and CrewAI; their code is useful behavioral reference,
but this task does not require adding previously absent TS tracing frameworks.
Do not invent JS support merely because a Python adapter exists. Document any
concrete capability gap, and retain the explicit generic host integration.

Use existing tracing integrations. Reflect-specific middleware/tool formatting
and explicit execution adapters are allowed because they deliver context and
control invocation permissions; they must not become duplicate tracing patches.
Preserve host model configuration, prompts, tools, callbacks, sessions, native
results, and supported streaming behavior. Bind supplied tools per instance and
execution, including MCP/native dispatch tasks. Apply administration permissions
both to offered tools and the actual executable path where the framework allows.

### LangChain, LangGraph, and DeepAgents

- Use native tool/message formats and stable replacement of the Reflect memory
  block. Preserve custom prompt composition and unrelated graph state.
- A checkpoint already owns history. Resume reflection with the new feedback
  and updated memory; **do not replay the previous full result into that state**.
  DeepAgents previously deduplicated assistant messages but appended ID-less tool
  replies again, causing an orphan-tool-message provider 400. If checkpoint
  history is absent or cleared, reconstruct it from the previous result as needed.
- Test saved, absent, and cleared checkpoints. Verify tool calls and replies stay
  paired through actual native loops. Do not add a production message sanitizer
  that silently drops malformed history instead of fixing ownership.
- Validation needs fresh thread/conversation identity. Precompiled custom graphs
  may need explicit tool nodes; do not promise arbitrary graphs can be retrofitted
  invisibly. If a required native hook is unavailable, give a clear version error.

### OpenAI Agents

- Preserve original agent options using invocation-local copies. Compose dynamic
  instructions and tools; restrict reflection tools/handoffs.
- Continue a persisted session using only new feedback, or explicitly continue
  the prior result's history when no session owns it. Do not duplicate both.
- Validation cannot reuse session/conversation/previous-response state. Keep
  native tracing hooks intact and test structured results and errors.

### Claude Agent SDK

- Supply the general Bash tool through the native MCP tool format, preserving
  host servers/options for task work. Resume the completed native session for
  reflection; validation starts fresh. Reflection exposes only the Reflect
  server/tool and explicitly restricts unrelated capabilities.
- Create native execution resources inside the correct scope and close them.
  Do not use a process-global active-session router.
- A native error result is an error even if no exception was thrown. Python now
  checks `ResultMessage.is_error`, preserves subtype/provider detail, records a
  failed invocation, and does not automatically continue reflection from it.
  Test the corresponding JS result/stream semantics, including max-turn errors.
- Python's bounded examples disable thinking and use low effort because their
  older bundled CLI sent legacy enabled thinking rejected by Haiku 5.5. This is
  example configuration, not a Reflect-wide model override. Verify JS SDK/model
  compatibility against its actual version; do not copy incompatible flags or
  override application-supplied model settings automatically.

### Vercel AI SDK

There is no Python Vercel adapter to copy. Implement the same Reflect contract on
this SDK's existing Vercel integration, selecting the actual native agent/tool
loop and tool-schema APIs from the installed version. Existing tracing middleware
alone is not an application task boundary. Verify multi-step tools, explicit
reflection continuation, isolated replay, memory delivery, stream completion,
abort/error handling, and preservation of the existing model middleware. Do not
substitute a one-shot text generation demo for an agent exercising the full loop.

### Public integration experience

Provide concise quickstarts for every supported framework and one generic custom
agent/streaming boundary. Native setup supplies Bash automatically and should
not require a developer to duplicate tool schemas or manually construct the
reflection prompt. Preserve explicit boundaries and explain their small required
host integration. Keep normal configuration simple; advanced controls stay optional.

## 12. Examples and behavioral demonstrations

Port the shared Python example structure so each framework uses the same business
scenario and independent verifier, not six unrelated demos. Reuse the code's
business invariants, reporting, stop conditions, and partial-artifact behavior.
Do not compare against old Harness benchmarks in this implementation.

### Offline complete loop

Port `examples/reflect/offline_demo.py`: use a loopback-only HTTP fixture with real
trace export, native API requests, real isolated shell operations, a deterministic
agent, independent replay, protected success, acceptance, and fresh retrieval.
It must run without credentials or paid services. A scripted agent proves lifecycle
wiring, not real-model repair quality. Keep fixtures out of the shipped runtime.

### Long-running support queue

Port `scenario.py`, `runner.py`, and the framework `04_reflect.py` experience:
fictional customer support, multiple conversations and inspect/authorize/resolve/
repeat work, independent ledger verification, idempotency, policy-sensitive cases,
held-out replay, configurable short/full runs, budgets, and useful reports. This
scenario can legitimately succeed without triggering reflection; do not invent a
failure notice just to show activity.

### Controlled failure → accepted guidance → fresh execution

Port `refund_incident.py`, `failure_runner.py`, and `05_reflect_failure.py`:

- The stale runbook defaults to invoice total when no applicable learned rule is
  available. The fictional service's policy actually excludes outbound shipping.
- Initial order `INCIDENT-101`: merchandise 6,500 cents, shipping 700; submitting
  7,200 is rejected. One submission closes the batch, so the same task cannot
  retry away its failure. The verifier checks the ledger, not the agent's claims.
- Platform evaluation must detect the failure. The same agent then reads assigned
  evidence and proposes a reusable rule; do not plant the answer in its filesystem.
- Held-out paired cases: 4,200 + 1,200 shipping; 11,300 + 300 shipping; protected
  2,700 with free shipping. Each arm has fresh state/conversation/memory. Only its
  assigned rule snapshot differs. A control occasionally succeeding does not
  invalidate the test; assess actual paired improvement and protected outcomes.
- After independent acceptance, reopen the workspace in a new Reflect instance
  and use a fresh agent/session. `FRESH-501` has 9,250 merchandise + 850 shipping;
  verify the 9,250-cent refund and the separate evidence of prior retrieval.
- Report initial failure, detection, proposal, acceptance, fresh retrieval, and
  fresh verified outcome as separate milestones. Missing/unattempted milestones
  remain false/unknown, never fabricated. Stop clearly at the first unfulfilled
  stage; do not retry until acceptance or success. Preserve partial artifacts and
  actual provider errors, including cases where native error results are values.

The current Python failure workload is version 4. Its overrides include a
360-second task/evaluation/reflection wait, 90-second reflection limit, 600-second
paired validation limit, and a stricter task-completion threshold. Read the runner
for all settings; do not mistake them for core defaults. Explain run counts,
possible paid activity, and overrides in TS example docs. Do not silently run
paid examples without the user's existing authorization and usable test credentials.

### Example-only retrieval audit

The core runtime stays free of this audit. In examples/tests, distinguish:

1. A rule exists or a filename appears.
2. Verified text from that rule version was in a task-model input before the
   business action.
3. Complete file coverage versus a partial excerpt.
4. Independently verified task outcome and paired improvement.

The current demonstration accepts a verified excerpt before submission; it does
not require all lines. Numbered grep output can be verified against path, line,
and text. Keep complete-coverage fields separate and never promote partial text
to a full-file claim. A tool output alone is not evidence of a subsequent model
request. Reflection-only or late reads cannot satisfy a fresh-task audit. Retain
conservative unknowns where a framework does not expose model inputs; report the
observation limit instead of changing SDK validation or adding hidden tracing
patches. Semantic usefulness/compliance/causality are not proved by text matching.

Keep model IDs configurable. Python examples currently default to `gpt-6-luna`,
`gemini-3.8-flash`, and `claude-haiku-5-5` where applicable. Verify current official
provider documentation and framework-specific tool/thinking support when building
TS examples. Do not expand into rewriting unrelated provider examples.

## 13. Historical verification and known boundaries

These are context, not a substitute for testing the new implementation:

- Live Python LangChain and DeepAgents runs demonstrated the complete controlled
  failure loop, paired validation, accepted guidance, and fresh task success.
  LangGraph also demonstrated acceptance/fresh success; an overly strict
  example-only full-file audit was corrected to recognize verified excerpts.
- DeepAgents conversation duplication and Claude error-result handling were real
  integration bugs fixed in Python. Port their regression cases.
- A Google ADK live attempt made successful tool calls but exhausted the example's
  360-second deadline waiting through slow model requests. The user attributed
  this to the personal API account and chose to leave the investigation aside.
  Do not present that run as a completed Reflect loop or a proven SDK defect.
  Proposed follow-up timeout/cancellation changes from that interrupted
  investigation are not part of the released reference unless current code says so.
- Claude's corrected model settings/error handling passed offline checks; this
  handoff does not establish a successful subsequent live full loop. Native
  telemetry can limit the example's retrieval audit separately from acceptance.
- Current exact/normalized deduplication does not solve semantic contradictions.
  Bounded tool/file bytes do not bound the whole model context or audit disk usage.
  Same-agent continuation can retain candidate exposure; clean replay must be fresh.
- Lost evaluation-create responses remain explicitly unknown until reconciled.
  Native idempotent creation needs future backend work; monitor management is out
  of scope. The historical backend judge guidance was updated separately to
  distinguish actual memory administration from task work and preserve costs.
  Verify deployed contracts/prompts if relevant; do not use a blanket `reflect_*`
  exemption for task mistakes, or require new backend work merely for TS naming.
- Python's successful release and test history do not establish Node compatibility,
  JavaScript framework versions, npm package contents, or TS runtime isolation.

Preserve the source-level distinction between accepted shared revisions, mutable
authoring snapshots, terminal processing status, and measured improvement. Small
reporting discrepancies in old artifacts are not a mandate to add new core policy.

## 14. Required verification

Use tests that exercise behavior, including negative cases. Port useful Python
tests rather than only asserting implementation-shaped mocks. Native framework
tests should run actual framework loops with mocked model/API boundaries. Include
real shell execution and filesystem concurrency tests; a mock shell cannot prove
Bash semantics or isolation.

### Core parity and API

- Trace/evaluation request and response contracts; resolved explicit config;
  endpoint normalization, authentication/project isolation, and redirect handling.
- Eventual trace visibility, pending/failed evaluation, complete score matrices,
  duplicate/missing/invalid scores, bounded reads, Retry-After, deadlines, aborts.
- One creation under concurrent submission/restart; durable dispatch before network
  calls; unknown response/cancellation/crash never causes a duplicate create.
- Signal thresholds, intermediate trajectory gates, deterministic sampling,
  cooldowns, worst-trace aggregation, trace ownership, and internal-activity exclusion.
- Uncapped daily usage by default; optional shared cap, UTC reset, diagnostics,
  cancellation/unknown attempts counted, and continued polling after the cap.
- Original task results/exceptions preserved; failed native result objects handled;
  separate task/reflection/validation sessions; no process-global runtime routing.
- Late feedback and explicit continuation; caller wait versus shutdown cancellation;
  bounded wait for uncooperative host code; no unhandled promise rejections.

### Filesystem, memory, and independent decisions

- Actual read/search/edit/move/merge/delete commands and persisted working state.
- No host credential/file/network access; read-only evidence, symlink/traversal
  rejection, oversized files/context, nested/large trees, rollback, and abort.
- UTF-8 byte limits, repeated guard replies, total-output exhaustion, late calls,
  and budget continuity across reflection/restoration.
- Atomic immutable revision publication, checksum tampering, stale/conflicting
  proposals, corrupt/missing authoritative state, and cross-process concurrent writes.
- Python-generated fixtures for rule normalization/versions, record formats,
  proposal/revision fingerprints, and representative deterministic decision outputs.
- Replacing active memory, same-session persistence, fresh controls, and no leakage
  of candidate-associated memory into ordinary new executions.
- Candidate lifecycle for edits and cleanup; empty/duplicate/uncertain outcomes;
  independently verified scores/success; protected regressions; partial evidence;
  changed replay snapshot; distinct arm IDs; trial assignment and duplicate records.
- Measured improvement and smaller-guidance non-inferiority remain distinct.
  Acceptance must work when an agent uses a partial unformatted relevant excerpt;
  no delivery matcher is needed or present in core.
- Replay runs after releasing task-held resources; interruption cannot publish
  accepted changes and stale baselines cannot overwrite newer acceptance.

### Frameworks, tracing, and packaging

- Every existing TS framework gets native task tools, memory, bounded reflection,
  isolated validation, error propagation, and concurrent-instance coverage.
- Checkpoint-owned history is not duplicated; tool requests/replies remain valid;
  provider-native error values/streams are not mistaken for successful results.
- Preserve existing tracing tests for decorators, wrappers, sessions, spans,
  streaming, cancellation, callbacks, and initialization.
- Test minimal tracing imports/install without Reflect or framework dependencies,
  explicit Reflect activation, missing-dependency errors, and relevant optional
  dependency combinations on declared Node versions.
- Run ESM and CommonJS consumers against an actual packed npm artifact, including
  `pandaprobe/reflect`, `pandaprobe/api`, and type declarations. Verify any interpreter
  binaries/WASM/resources are included and load correctly. Do not bundle tests,
  examples' audit code, workspaces, credentials, or Python into runtime outputs.
- Cover Linux/macOS/Windows where the selected runtime claims support. Exercise
  Node 18/20/22 base support and the honest framework-specific minimum versions.
  Do not let all meaningful integration tests pass by being skipped.
- Add focused CI jobs only where necessary to cover native optional frameworks,
  OS behavior, and packed/minimal installation. Preserve the intended trigger:
  pushes to `main`, PRs into `main`, reusable workflow calls, and manual dispatch;
  do not reintroduce arbitrary feature-branch push runs accidentally.

### Examples and claims

- A credential-free local HTTP complete-loop test, with real export and shell.
- Native mocked-provider multi-turn support and failure scenarios for every target
  framework, including Vercel AI. Verify failure branches do not become success.
- Example-only audit tests for exact, numbered, partial, missing, malformed,
  wrong-version, and late text. Include the existing numbered-grep fixture if useful;
  keep it test-only.
- Live tests, when authorized and configured, report task outcome, proposal,
  validation, retrieval, latency, model calls/tokens, and available cost separately.
  Missing cost telemetry is unknown, not free. Preserve real artifacts and report
  exactly which frameworks/stages were actually tested live.
- No requirement to benchmark against historical Harness. Unit or scripted test
  success is not a measured claim of broad real-model improvement.

## 15. Documentation and release readiness

Update TypeScript README/exports/install instructions, framework quickstarts,
generic and streaming integration, workspace guide, limits, error meanings,
candidate/acceptance semantics, verifier responsibilities, and both live example
guides. Keep short happy-path setup genuinely runnable in the installed package.

Explain the isolated virtual filesystem versus durable host records, how to find
accepted head, where active memory lives, why no second agent/model is needed,
and why writing a rule is not the same as accepting or using it. State explicitly
that core has no delivery checker and no default 100-evaluation quota.

Release verification includes build, declarations, lint/format/typecheck, tests,
packed artifact contents, minimal installations, optional runtime resources, and
the existing npm OIDC workflow. At handoff `src/version.ts` is `0.4.2` while
`package.json` is `0.4.0`; the existing release workflow synchronizes package
version from `src/version.ts`. Inspect current state instead of inferring a new
version from Python 0.6.0. Do not publish, tag, or change release identity as part
of implementation without a separate release instruction.

Core-service docs/website changes remain follow-up work, not this task. Final
delivery must say what changed, which tests ran and passed, what was skipped,
which live runs were performed, and any operational/portability limitations.
Clearly identify any material behavior not yet at Python parity.

## 16. Implementation checklist and progress log

This ordering describes dependencies, not a prescribed TypeScript implementation.
Keep the settled contract above; decide the concrete mechanisms through code
inspection and focused feasibility checks. Record test commands/results beneath
each completed step. Do not mark blocked or untested work done.

- [x] **0. Baseline and branch — DONE.** Inspect instructions/Git, preserve changes, base a
  new `feat/typescript-reflect` branch on the merged Python release, record exact
  Python/TS revisions and current Node/framework support. Read the source map.
- [x] **1. Parity inventory — DONE.** Map every public capability, record, default,
  lifecycle outcome, and Python test area to its TS counterpart. Record only
  concrete platform/version gaps. Verify the isolated Bash runtime and packaging
  are viable before building an incomplete policy layer around a placeholder.
- [x] **2. Platform API — DONE.** Implement independently reusable typed resources,
  resolved config, separate HTTP lifecycle, contract fixtures, safe reads,
  visibility/polling, error distinctions, and single-attempt creation tests.
- [x] **3. Trace capture — DONE.** Add explicit client-owned scoped capture to existing
  submissions, phase attribution, overflow/late-work behavior, and preservation
  of tracing contracts. No new framework tracing patches or export-worker policy.
- [x] **4. Records and workspace — DONE.** Port config/record validation, immutable
  revisions, atomic state, process locks, proposal diffs, dedup/conflicts,
  evidence/events/memory, and deterministic cross-language fixtures.
- [x] **5. Shell and active memory — DONE.** Implement one supplied `reflect_bash` with
  real isolated filesystem operations, budgets, read-only mounts, checkpoints,
  rollback, abort/close, and memory construction. Verify with the actual engine.
- [x] **6. Detection and orchestration — DONE.** Port explicit boundaries and submission,
  evaluation scheduler/gates/usage, external notices, same-agent bounded
  reflection, result preservation, attribution, recovery, and separate waits.
- [x] **7. Independent validation — DONE.** Implement replay, paired live trials,
  fresh assignments/controls, protected outcomes, improvement/non-inferiority,
  pending/rejected states, resource separation, and atomic acceptance.
- [x] **8. Native framework integration — DONE.** Implement and test every existing TS
  framework surface, including Vercel AI, with native loops, preserved tracing,
  tool restrictions, correct continuation/history, and a generic host fallback.
- [x] **9. End-to-end examples — DONE.** Add offline complete-loop and shared support /
  controlled-failure examples to each framework, independent business verifiers,
  example-only retrieval audit, partial reports, and side-effect-free help.
- [ ] **10. Compatibility and package verification — LOCAL CHECKS PASS; WINDOWS CI PENDING.** Run meaningful full and
  framework/OS/Node checks, minimal and optional installs, ESM/CJS/types, npm pack
  inspection, tracing regressions, and focused CI updates. Resolve failures.
- [x] **11. Documentation and final review — DONE.** Finish runnable docs, update this
  plan's evidence log, inspect the diff for scope/accidental artifacts, report
  parity and limitations. Leave changes unstaged/uncommitted for user review.

### Evidence log — strict Python-parity revision

**Baseline and scope.** Branch `feat/typescript-reflect` remains at merged baseline
`5e44571` (PR #32, same release tree as Python 0.6.0 `eb5f09d`). The source of truth
is the unchanged Python source/tests. Existing user changes/files are preserved;
the user's `plan.md` is present again and has not been edited in this revision.
Package version remains 0.4.0, source version 0.4.2; release identity/workflow are
unchanged. No stage, commit, push, publish, tag or live provider call.

**User clarification applied.** Separate helper files are organizational only.
Examples must use Python's exact workload/configuration and control flow. Core
must preserve Python behavior, with only necessary native language/framework
adaptations. This evidence supersedes the earlier implementation's verification.

**Implementation and parity inventory.** See
`libraries/typescript/examples/reflect/parity.md` for the module/capability map and
necessary native translations. Live support/failure examples now use Python's
instructions, prompts, tool descriptions and responses, held-out cases, verifiers,
argument/model defaults, preflight, fresh output/workspace behavior, thresholds
(task_completion 0.8), reflection/validation deadlines (90/600), per-task deadline
(360), validation limits, reporting/stop conditions, and fresh Reflect-instance
reopening. `_support/cli.ts` is only the example argument/startup equivalent of
Python argparse; it is excluded from the SDK artifact.

Removed the discretionary returned-error/exit-124 rollback, `.initialized`,
execution-ID claims, persisted cap consistency, reordered-metric fallback
single-flight, proposal-fingerprint tamper gate, and extra evaluation-usage API /
report field. The ordered metrics dispatch hash, shared attempt counter with
per-instance caps, empty-revision state reinitialization, and shell result semantics
now follow Python. Frozen record fields retain Python's mutable nested containers;
record defaults/extras/nulls and API coercion are covered by differential fixtures.

| Area | Current verification |
| --- | --- |
| Core and tracing | Full Vitest suite under Node 18, 20 and 22: **393 tests in 26 files**, passing, no skips. Includes **103 original tracing tests**, API/capture, real Bash/locks, durable dispatch/recovery, independent validation and examples. The lock subprocess test allows 20 seconds for instrumented/process startup; its contention assertions are unchanged. |
| Python differential checks | `libraries/python/.venv/bin/python -B libraries/typescript/scripts/update-python-parity.py` executes unchanged Python business logic, argument/config defaults, records, audit matcher, API decoders and actual Bashkit results. `python-parity.test.ts`: **177 tests** pass. Existing `update-reflect-fixtures.py` covers normalization/hashes and ten Python-assessed validation decisions. |
| Native framework loops | Node 22 native suite: **54 tests in 3 files**, passing, no skips. Real LangChain, LangGraph, DeepAgents, OpenAI Agents, Claude MCP and Vercel loops with scripted providers. All six run the 18-turn support workload; five prove the complete incident including actual model-input delivery. Claude proves the fresh ledger but honestly reports incomplete delivery evidence. Tests also cover missing evidence, limits, partial/interrupted reports, fresh platform disagreement, concurrency, continuation, native errors and streaming. |
| Example-only delivery | Uses actual task LLM inputs; matches Python full/verbatim or exact numbered excerpt, versions and strictly-before-submission timestamps. The old tool-output-only proof was removed. Node performance timeOrigin supplies a high-resolution wall clock, avoiding a rounded Date.now baseline. No delivery gate exists in core. |
| Coverage | Node 22 full suite: Reflect core **91.26% lines / 84.73% branches**; API **94.74% / 88.13%**. Native adapters are tested separately and are not included in those figures. |
| Lint/format/types/build | `pnpm exec biome check .`, `pnpm format:check`, `pnpm typecheck`, `pnpm typecheck:native`, `pnpm build`: pass. |
| Packed package | `scripts/verify-package.mjs` under Node 18/20/22 with npm 10: minimal imports/install, clear missing-runtime error, optional native binaries, ESM/CJS consumers, shared tracing/client context, strict NodeNext types and AI SDK 5.0.100 native tracing all pass. Artifact: **138 files, 441211 bytes**. No examples/audit code, tests, workspaces, credentials or Python included; zero required runtime dependencies. |
| Clean installation | Fresh temporary source/config copy: frozen install, verified absence of framework packages, typecheck/build and **393 tests on Node 22** pass. An initial host Node 24 run lost a shell-test worker during concurrent verification; that run is not counted as passing. Its 14 shell tests pass when rerun in isolation on the same host runtime. |
| Linux | Official `node:22-bookworm-slim` **Linux arm64** disposable container: frozen install, typecheck, **393 base tests**, build, packed ESM/CJS/types/resources, native install, **54 native tests** and native typecheck pass. Only the credential-free source archive was mounted read-only; the container was removed automatically. Docker was restored to its prior stopped state. |
| Scope review | `git diff --check`, whitespace checks for all **77 new package files**, CI YAML/trigger/default-directory checks pass. Index empty; Python, release workflow and source version unchanged. No generated/private artifacts among new package files. |

**Necessary native differences retained.** Promise/callback scopes,
AsyncLocalStorage, AbortSignals, fetch, native JS framework hooks/schemas/sessions,
optional peer loading and Node filesystem/advisory-lock primitives. OpenAI's
example uses native MemorySession in place of Python's in-memory SQLiteSession.
Vercel remains the settled TS-specific framework surface. Both inspected shell
runtimes are Bashkit 0.19.0. The pinned lock binding is loaded directly to retain
Node 18 support without changing globals. JS numeric representation cannot encode
all Python int/float distinctions; simultaneous writable Python/TS workspaces and
cross-language lock interoperability are not claimed. Native host hooks remain
trusted code; shutdown cannot forcibly terminate arbitrary uncooperative JS.

**Remaining verification.** Step 10 remains open for Windows x64 / hosted CI.
The local host cannot execute those jobs, and no push is authorized. Focused CI
jobs cover native frameworks, filesystem/platform behavior and packed consumers;
existing main-push/main-PR/reusable/manual triggers are preserved. Alpine/musl,
browser/edge, distributed filesystems and mixed-language concurrent writers remain
outside the tested support model.

**Live evidence and Claude limitation.** No paid/live provider run was performed.
The offline complete loop uses real loopback HTTP export, polling and the native
shell; native suites use real framework loops with fake providers/platforms.
These tests do not establish broad real-model improvement. The tested Claude query
surface does not expose complete model input, so the example reports
`incomplete` at `fresh_delivery`, even with accepted guidance and a correct fresh
ledger. This follows Python's evidence requirement rather than weakening it.

### Strict Python-parity revision checklist

- [x] Compare and port example arguments, model defaults, prompts, business tools,
  workloads, reporting, preflight, deadlines, fresh-runtime reopening and audits.
- [x] Remove discretionary core policy/state changes; match Python record and
  API-decoding semantics, retaining documented native adaptations.
- [x] Add Python-generated differential fixtures and pass core/native examples,
  tracing regressions, types/lint/build and packed-consumer checks.
- [x] Finish clean-install/Linux result review, final documentation/evidence and
  Git scope checks. Windows/hosted CI remains explicitly pending in step 10.

### Install-target follow-up — DONE

Added `make install-reflect` in `libraries/typescript` and the root
`make ts-install-reflect` delegate after the user's setup review. The TS Makefile
now uses separate explicit recipes, matching Python's structure: `install` runs
`pnpm install`; `install-reflect` runs `pnpm add -D --save-exact
@everruns/bashkit@0.19.0 fs-native-extensions@1.5.1`. Like Python's `dev` extra,
TS devDependencies already include Reflect. Updated the development and Reflect
setup guides. Base/Reflect/root targets pass dry-run resolution; the explicit
recipe matches the pinned dev/optional-peer versions, and `git diff --check`
passes. No dependency manifest/lockfile changes or installation were needed for
this follow-up; SDK behavior and Python are unchanged.

### DeepAgents provider-request fix — DONE

The user reported an OpenAI 400 on the first discovery request with DeepAgents
1.16.0, LangChain 1.5.20, LangGraph 1.5.3 and @langchain/openai 1.7.1. The shared
example factory supplied `reasoningEffort: "none"` as a ChatOpenAI constructor
field. The installed JS provider client only reads `reasoning.effort` there;
`reasoningEffort` is a deprecated call option. The resulting Chat Completions
request omitted `reasoning_effort`, violating the model's tool-call requirement.
The example now constructs ChatOpenAI with `reasoning: { effort: "none" }`,
preserving Python's reasoning setting and Chat Completions transport for
LangChain, LangGraph and DeepAgents. Reflect core and tracing are unchanged.

Added `test/native/model-options.test.ts` and pinned @langchain/openai 1.7.1 in
only the isolated native test fixture. These tests use actual ChatOpenAI clients,
framework/tool loops and Reflect phases with mocked HTTP. All three reproduced
HTTP 400 before the fix and pass afterward, checking all 16 requests per framework
through task, reflection and six replay arms. The prior scripted-model tests did
not exercise provider request serialization, which is why they missed this bug.

Verification: **57 native tests / 4 files pass** on the user's installed versions
(Node 24.3.0). The three new request regressions also pass in a temporary source
copy resolving the pinned older framework fixture (DeepAgents 1.15.1, LangChain
1.5.19, LangGraph 1.5.2) with @langchain/openai 1.7.1. Base/native typechecks,
focused Biome checks and `git diff --check` pass. The temporary copy was removed.
No live provider calls were made. Hash checks confirm the user's root TS package
manifest/lockfile and failed report `bcebb911890f/report.json` are unchanged.
Python and the Git index remain unchanged. No dependency reinstall is needed to
rerun the live example; it creates a new output directory by default.
