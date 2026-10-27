# Python 0.6.0 parity inventory

Reference: Python release commit `eb5f09d`, merged baseline `5e44571`. Python is
unchanged. The architecture and public outcome vocabularies follow the root
`typescript-reflect-plan.md`; idiomatic JS names use camelCase and async callbacks.

| Python capability / test area | TypeScript implementation | Behavioral verification |
| --- | --- | --- |
| API models/client/resources, OpenAPI fixture | `src/api/{models,client,resources}.ts`, `/api` export | `test/api.test.ts`; `scripts/update-platform-contract.py` consumes exported OpenAPI |
| Explicit client-owned submission capture | `src/tracing/capture.ts`, additive Client hook | `test/trace-capture.test.ts`, all original tracing tests |
| ReflectConfig defaults/strict ranges | `src/reflect/config.ts` | workspace/config and runtime tests; [complete defaults](workspace.md) |
| Rule/notice/proposal/evaluation/report/trial records | `models.ts`, `validation.ts` | Python-generated `reflect-parity.json` / `python-behavior.json`, record decoding, decision reasons and host outcome tests |
| Hashes, immutable snapshots, locks, atomic state | `workspace.ts`, `serialization.ts` | actual subprocess contention, interrupted publication, corrupt state/hash/symlink/stale-head tests |
| Bash tool, read-only evidence, rollback/budgets | `shell.ts` with native Bashkit | real `test/reflect/shell.test.ts`, UTF-8/abort/restoration tests |
| Active memory/session pinning | `execution.ts` | fresh memory/candidate isolation, replacing context, streaming boundary tests |
| Threshold/regression/stall/cooldown | `signals.ts` | evaluator signal-gate tests |
| Single flight, durable create/unknown recovery | `evaluation.ts`, `async.ts` | concurrency, dispatch-before-POST, known-run restart, lost create, usage/day/reset/cap tests |
| Explicit task boundary, same agent continuation | `runtime.ts`, `Invocation` | original object/errors, delayed notices, cancellation/shutdown/closed tools, concurrent instances |
| Replay, paired live trials, independent publication | `validation.ts` | protected outcomes, partial evidence, stale head, no delivery gate, cleanup non-inferiority, duplicate trial records |
| LangChain / LangGraph / DeepAgents | native middleware/ToolNode guards | real loops with saved/absent/cleared checkpoints and existing callbacks |
| OpenAI Agents / Claude Agent SDK | invocation-local Agent clone / execution-bound native MCP | real native run/tool paths, native errors, continuation/fresh validation |
| Vercel AI (TS-specific surface) | native ToolLoopAgent Reflect adapter plus `wrapVercelAgent` tracing | CHAIN/AGENT/LLM/TOOL trees on AI SDK 6/7, real tool results/errors, existing middleware/callbacks, same-agent reflection, streaming/cancellation |
| Python offline/support/failure examples | `examples/reflect` and twelve framework entry points | real loopback HTTP complete loop, native mocked-provider 18-turn support and six-arm failure/fresh loop |
| Build/import/runtime compatibility | split shared ESM/CJS chunks, optional peers | packed/minimal consumer script and focused CI |

Fixture regeneration (development only; Python is never an SDK runtime dependency):

```sh
# From repository root, with the Python development environment installed:
libraries/python/.venv/bin/python -B libraries/typescript/scripts/update-reflect-fixtures.py
libraries/python/.venv/bin/python -B libraries/typescript/scripts/update-python-parity.py
# From libraries/typescript, from an independently exported OpenAPI document:
python3 scripts/update-platform-contract.py /path/to/openapi.json
```

Native frameworks live in a separate npm fixture so base Node 18 dev installs do
not accidentally acquire packages requiring newer Node. Native tests mock model
and platform boundaries, not the framework loops, tool schemas or Bash engine.
They are mandatory in their CI job; no meaningful native tests silently skip for
missing dependencies. Base tests remain runnable without those framework packages.

## What differs, and why

| Area | Necessary TS adaptation |
| --- | --- |
| Module layout | `index.ts` is the export counterpart of `__init__.py`. `serialization.ts`, `async.ts` and `locks.ts` separate implementation helpers; they do not add framework policy. |
| Async lifecycle | Promise-returning methods and callback scopes replace Python coroutines/context managers. AsyncLocalStorage carries invocation context; AbortSignal carries cancellation; native fetch replaces httpx. Arbitrary uncooperative host JS cannot be forcibly terminated. |
| Public models | Readonly TS properties plus runtime decoding/freezing replace Pydantic records. Extra fields, defaults, explicit nulls, statuses and nested records are validated. As in Python, frozen fields do not make nested containers immutable. Outcome/ReplayCase retain their input references like Python's frozen dataclasses. |
| Native frameworks | Native JS middleware, schemas, ToolNode boundaries, sessions and MCP objects implement the same task/reflection/validation lifecycle. OpenAI examples use MemorySession in place of Python's in-memory SQLiteSession. See [native.md](native.md) for matching model options. |
| Workspace locks | The same filesystem transactions use Node filesystem primitives and native advisory locks. The pinned N-API binding is loaded directly to retain Node 18 compatibility. |
| Numbers and encoding | Representative persisted hashes match Python. JS has one Number representation rather than Python's separate int/float and arbitrary-size ints; exact cross-language numeric serialization for every input is not possible. Use separate writable workspaces. |
| Framework coverage | Vercel is a TS-native adapter using the shared workload. Google ADK/CrewAI remain outside the settled existing-TS-framework matrix. |
| Example helpers | `_support/cli.ts` corresponds to argument parsing/startup in Python's runners; it is excluded from the npm artifact. Native model-input observation and microsecond report timestamps adapt Python's audit to JS surfaces. |

The support and incident examples match Python's prompts, tool descriptions,
outputs, money amounts, policies, verifiers, replay inputs, protected cases,
threshold 0.8, reflection timeout 90, validation timeout 600, per-task timeout 360,
argument defaults, preflight, fresh workspace rules, report lifecycle, and fresh
Reflect-instance reopening. Fixture generation executes the unchanged Python
source, including service transcripts and delivery-audit coverage. LLM replies
remain provider-dependent. The Claude example audit decodes the tracing adapter's
JSON-encoded MCP text blocks within recorded LLM tool messages; exact rule matching
and before-action timing still apply. Missing delivery evidence leaves the incident
`incomplete` at `fresh_delivery`, as required by the shared evidence criterion.

## Removed discretionary differences

- Returned Bashkit errors and exit 124 no longer trigger extra rollback. Python's
  distinction between returned results and thrown/cancelled calls is preserved.
- No `.initialized` marker, additional execution-ID claims, or persisted shared-cap
  consistency requirement. Missing state with only the empty revision follows
  Python's reinitialization behavior.
- No execution/stage fallback deduplication for reordered metric arrays; dispatch
  keys use Python's ordered metrics. The regression expects two claims in that case.
- No extra proposal-fingerprint validation at publication.
- No additional `evaluationUsage()` API or `evaluation_usage` report field.
- No shortened prompts, threshold 0.5 in live examples, global example deadline,
  fixed default workspace, disabled DeepAgents subagents, or tool-output-only
  delivery proof. The model must actually receive the matching rule excerpt.

The API fixtures also cover Python's integer coercion, numeric-score grammar and
missing-versus-null behavior. Markdown fixtures cover Python whitespace and line
normalization; record fixtures exercise accepted defaults and rejected extra/null
or wrong-type fields. Decision fixtures compare ten independently assessed cases.
Python is used only to regenerate development fixtures, never at SDK runtime.

## Verification boundaries

Native framework dependencies are isolated from the base package's dev install.
The tests run real framework/tool/shell loops with fake providers and a mocked
platform. They do not establish live-model improvement. Native tests are mandatory
in their CI job, with no silent skips for missing dependencies.

See the root plan evidence log for current commands/results. Base SDK support is
Node 18/20/22; the tested native framework combination uses Node 22 or newer.
Windows x64 and hosted CI cannot be verified locally on this macOS host. Upstream
binary availability is not a claim that those jobs ran. Alpine/musl, browser/edge,
distributed filesystems and concurrent mixed-language writers are not claimed.
The Bashkit runtime is 0.19.0 in both the inspected Python environment and npm
fixture. No Python changes, server contract changes, second reflection model,
core delivery gate, tracing rewrite, version bump, or live paid runs are included.
