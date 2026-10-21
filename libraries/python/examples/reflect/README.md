# PandaProbe Reflect

Reflect connects **Detect → Notify → Reflect → Validate**. Your task agent reads
and maintains a rule filesystem through one supplied Bash tool. PandaProbe
evaluations provide external failure evidence; independent host replay or paired
trials decide whether a proposed change becomes shared guidance.

Reflection is a proposal, not proof of repair. An accepted change has evidence
under tested conditions; it is not a permanent guarantee.

## Start with an existing traced agent

```bash
pip install "pandaprobe[reflect,langchain]"
```

Keep your model, prompts, domain tools, and existing PandaProbe tracing setup.
For a LangChain/LangGraph message-state agent:

```python
from langchain.agents import create_agent
from pandaprobe.reflect import Reflect

async with Reflect(sdk_client) as reflect:
    agent = reflect.agent(create_agent, model=existing_model, tools=domain_tools)
    result = await reflect.run(agent, "Do the task", session_id="customer-123")
```

`reflect.agent(...)` supplies Bash automatically. There is no shell provider,
container, second model, or separate PandaProbe CLI to configure. For OpenAI
Agents and CrewAI, `reflect.run(existing_agent, ...)` can add the tool on an
invocation copy. Claude uses `ClaudeAgentOptions` before connecting its client.
See [framework_quickstarts.py](framework_quickstarts.py) for each integration,
including explicit use of the existing SDK tracing adapters.

An already compiled custom graph cannot acquire new tool nodes automatically.
Construct it with Reflect's supplied tool, or use the explicit boundary below.
If tracing is absent, Reflect reports insufficient trace evidence; it does not
invent an execution from a session name or install tracing patches.

`run()` returns the original task result. After task completion it waits for
bounded evaluation and, when warranted, re-enters your agent for a bounded
reflection checkpoint. Reflection errors cannot replace a successful task result
or its original exception. Enabled Reflect adds latency, model calls, and tokens.

## Try the complete loop without credentials

From `libraries/python`:

```bash
pip install -e ".[reflect]"
python examples/reflect/offline_demo.py
```

This starts a temporary **loopback-only** HTTP fixture and uses real trace export,
native HTTP API calls, and Bash. A deliberately faulty scripted agent proposes a
rule; an independent arithmetic verifier checks fresh controls, candidates, and
a protected successful case. A later execution retrieves the accepted rule.
It makes zero real model calls and demonstrates orchestration, not real-model
repair performance. The temporary workspace is removed when the demo exits.

## Files and authority

The agent sees these paths through `reflect_bash`:

| Path | Purpose |
| --- | --- |
| `/rules/**/*.md` | Working rule revision; search, read, edit, move, merge, or remove with ordinary Bash |
| `/context.md` | Editable, bounded active memory |
| `/inbox/notice.json`, `/inbox/evidence.txt` | Read-only assigned failure evidence |
| `/decisions/status.json` | Read-only recent validation summaries |

Each rule contains a title, explicit applicability, and actionable guidance:

```markdown
# Verify account identifiers
Applies to: customer-account lookup before account changes
Resolve the account by its stable identifier and confirm that the returned
account matches the request before submitting a change.
```

Directories provide scope. The agent selects relevant guidance with commands
such as `grep -rl 'account' /rules`, then reads matching files. Reflect does not
rank or automatically insert the entire rule corpus. Exact duplicates are
detected; semantic overlaps and contradictions require agent reasoning and
independent validation. File/count limits prevent unbounded active-corpus growth.

The embedded Bash runtime supports common file utilities, pipes, loops and
heredocs. It has no host files, inherited credentials, network, external programs,
or Python execution. It is a general shell interface within the memory workspace,
not a full operating-system shell. Working files are checkpointed into durable
filesystem records; immutable accepted revisions and lifecycle state are never
mounted inside this tool.

All shared additions, edits, merges, moves and deletions become change sets.
Only validation publishes an accepted revision. A candidate is readable in its
authoring execution and in explicitly assigned trials. Ordinary new rule views
start from accepted guidance. Failed or interrupted administration cannot publish
partial changes. Concurrent publication against a stale base stays pending.

`context.md` is session-local working memory and may contain unvalidated thinking.
Its hash and status are recorded. It is incorporated at invocation checkpoints
through a replaceable memory block; it is not a shared accepted rule. Context
associated with unvalidated rule edits is not reused by a fresh ordinary
execution, and validation controls always start with fresh local memory.
Existing conversation history is not rewritten or compressed by Reflect.

The supplied tool enforces its own file and output budgets. Those limits do not
bound a framework's full conversation history or tools supplied separately by
the host. If the agent also has unrestricted host filesystem tools, the host must
keep Reflect's authoritative storage outside their writable environment; Python
SDK code cannot enforce isolation against another tool running with the same OS
permissions.

## Validation and observable outcomes

Without an independent verifier, proposals remain candidates awaiting validation. A trace cannot
reconstruct an arbitrary agent's tools, environment, or side effects. Supply
reproducible cases and a host-owned verifier when replay is available:

```python
from pandaprobe.reflect import Outcome, ReplayCase

cases = [
    ReplayCase(id="failure-a", input={"query": "..."}),
    ReplayCase(id="failure-b", input={"query": "..."}),
    ReplayCase(id="protected-success", input={"query": "..."}, protected=True),
]


async def replay(request):
    result = await request.run(agent)  # isolated, explicitly assigned rule view
    passed = independent_verifier(request.case.input, result)
    return Outcome(
        scores={"task_completion": float(passed), "coherence": 1.0},
        success=passed,
        evidence={"verifier": "my-verifier-v1"},
    )


proposal = reflect.proposals()[0]
verdict = await reflect.validate(proposal.id, cases=cases, replay=replay)
```

Use genuinely independent measurements for every configured metric; the constant
coherence above is appropriate only when the host has established that invariant.
The [offline demo](offline_demo.py) uses a deterministic verifier whose expected
answers do not depend on the proposal or judge explanation.

Alternatively, construct `Reflect(sdk_client, cases=cases, replay=replay)` once.
New proposals then schedule validation automatically. Replay runs separately from
the task/reflect barrier; call `await reflect.settle_validation()` after releasing
resources that replay itself needs. Do not await replay while holding its required
semaphore, transaction, or environment lease.

By default validation requires three distinct paired cases, at least one
protected successful case, complete finite outcomes, no regression beyond the configured tolerance,
and actual delivery of changed guidance. Removed guidance must have been
exercised in controls. Smaller guidance can be accepted for measured
non-inferiority, with no claim of behavioral improvement. Otherwise results are
accepted, rejected, or pending with recorded reasons.

For live trials, use `reflect.trial_execution(proposal_id, arm="control")` and
`arm="candidate"` to obtain explicit isolated assignments. Submit matched
`TrialCase`/`TrialArm` evidence to `reflect.validator.record_trial(...)`. Include
independently verified scores, success, the execution IDs, submitted rule
versions from `execution.delivered_versions`, and assigned candidate IDs.
Unrelated successful sessions, duplicate cases, missing outcomes, and unknown
exposure do not establish improvement. The host owns matching and environment
reset; the SDK does not infer a causal trial from timestamps.

Pass an explicit `execution_id` to `run()` when you want to retrieve
`reflect.report(execution_id)`. Reports distinguish evaluation, proposal creation,
duplicate/no-proposal, timeout/failure, delivery, and independent validation.
The agent's reported rationale is retained separately from the SDK's decision
reason, including when it leaves files unchanged because the evidence is uncertain.
There is deliberately no boolean that calls a completed reflection a repair.

## Explicit boundaries and custom agents

An invocation ID identifies one execution; a session ID only groups related work.
The boundary must enclose consumption of streaming results and completion of
the work whose traces it submits:

```python
async with reflect.execution(session_id="customer", execution_id="turn-42") as turn:
    instructions = turn.instructions()
    # Supply these instructions and reflect.tool() to your own model loop.
    # Invoke reflect.bash(command) when its supplied tool is called.
    result = await my_agent_loop(instructions, reflect.tool(), user_input)
    # At your actual model-input boundary, attest the exact submitted text:
    # turn.submitted(text, channel="tool_output")

evaluation = await reflect.settle_evaluation("turn-42")
```

The generic `reflect.run(callback, input, ...)` callback is async and accepts an
`Invocation` containing the input, execution, instructions, phase, feedback and
previous task result. It owns model invocation and continuation. During reflection,
it should expose only the supplied filesystem tool and avoid repeating task
actions. Native factory adapters apply this restriction where the framework
provides the relevant hooks. Custom graph/tool nodes remain host-owned.

The SDK records tool output as offered, and records submitted text separately.
A filename, grep snippet, accessible file, or successful `cat` alone does not
prove that the full rule reached a model request. Native model-input hooks and
existing traced LLM inputs confirm exact rule versions. Claude's SDK does not
generally expose full model requests: tool-body confirmation may require host
attestation; unknown delivery keeps validation pending. Delivery is never proof
of compliance.

For externally completed work:

```python
reflect.submit_traces(trace_ids, execution_id="turn-42", session_id="customer")
evaluation = await reflect.settle_evaluation("turn-42")
proposal = await reflect.continue_reflection("turn-42", my_async_continuation)
```

Evaluation can finish after the original agent returns. The notice remains
durable; it does not resurrect an agent without a host continuation. On restart,
`resume_evaluations()` polls known pending runs. An interrupted reflection claim
is not silently rerun, because a continuation may have external side effects.

Wait operations are deliberately separate:

| Operation | What it waits for |
| --- | --- |
| `wait_export()` | The existing SDK export worker |
| `settle_evaluation(id)` | Trace visibility and evaluation evidence |
| `continue_reflection(id, callback)` | A bounded host continuation |
| `settle_reflection(id)` | An already scheduled checkpoint |
| `settle_validation()` | Independent replay tasks |
| `aclose()` | Cancellation/cleanup of owned work and HTTP resources |

Host callbacks must cooperate with cancellation. Reflect bounds its waiting and
closes their supplied shell; it cannot forcibly stop arbitrary Python code or
undo a model request already accepted by a provider. Cancellation of a caller's
settlement wait does not silently recreate an evaluation run.

## Defaults, storage, and portability

`ReflectConfig` owns defaults; ordinary integration needs none. Shared endpoint,
API key and project come from the same resolved `Client`, including explicit
constructor values. `PANDAPROBE_REFLECT_WORKSPACE` changes only workspace location.
Default storage is `.pandaprobe/reflect/<identity-digest>/`, partitioned by endpoint,
project and credential to avoid cross-tenant collisions. Credential rotation
creates a distinct partition; transfer existing history only after verifying its
project ownership.

Default limits include 100 evaluation runs per UTC day, four concurrent evaluations
per instance, 32 traces per invocation, a 60-second reflection checkpoint with
eight shell calls, 256 rule files/512 KiB total, 8 KiB per rule/active-memory file,
12 KiB per tool output and 96 KiB total output per execution. These are byte
bounds, not tokenizer-independent token guarantees. The agent chooses how to
consolidate guidance; validation checks the proposed change.
Guard/error responses consume the same allowance. A fully depleted or closed
execution returns no further tool text; framework message-envelope overhead and
the host's conversation/turn limits remain separate.

Immutable revisions, decisions, execution snapshots and events retain provenance.
Active corpus limits do not impose automatic audit-history deletion. Operators
own filesystem permissions, disk monitoring, backup, archival and retention.
Use a local filesystem with functioning advisory locks and atomic rename;
distributed/object-store mounts are not an interchangeable workspace backend.

Reflect supports Python 3.10+. Deep Agents independently requires Python 3.11+.
Bashkit supplies ABI3 wheels for supported Linux/macOS/Windows architectures;
unsupported platforms fail installation rather than falling back to a host shell.
Framework integrations are lazy. A tracing-only install imports no Bash engine,
framework, reflection model or workspace, and starts no evaluation work.

The native integration tests cover the repository's locked versions (LangChain
1.3, LangGraph Prebuilt 1.1, Deep Agents 0.6, OpenAI Agents 0.13, ADK 1.33,
CrewAI 1.6 and Claude Agent SDK 0.1.50), plus newer releases tested during
implementation. Custom graphs still need explicit tool nodes. LangGraph's
`create_react_agent` is deprecated upstream; `langchain.agents.create_agent`
is the preferred factory when LangChain is installed. Streaming custom loops
use `reflect.execution(...)` and consume their stream inside that boundary.

## Native platform API and operational limits

`pandaprobe.api.PlatformClient(sdk_client)` is independently usable for trace
retrieval, evaluation creation and polling. It owns a separate asynchronous HTTP
client; it does not reuse or alter trace export. API errors distinguish auth,
rate limits, transport, deadlines, failed evaluation and insufficient scores.

API wire contracts are pinned from current backend OpenAPI, including nested
response types and request schemas. Maintainers can refresh them with
`python scripts/update_platform_contract.py /path/to/openapi.json`, then run
`pytest tests/test_api.py`. This development utility has no backend dependency.

The current backend has no evaluation-creation idempotency-key contract. Reflect
durably records dispatch, polls an existing run, and **never retries an ambiguous
POST**. A lost response can leave a potentially billable run with unknown identity;
the result is `creation_unknown`, requiring operator reconciliation. Request IDs
do not solve this. Safe GETs use bounded retries. Empty/missing/failed scores are
never accepted as a successful measurement.

Reflection and validation traces use distinct attribution and sessions and do not
recursively trigger task evaluation. Their costs remain visible. The associated
backend prompt update distinguishes genuine memory operations from task attempts;
it does not exempt arbitrary Bash or hide administration overhead.

Live model behavior, historical Harness comparisons, TypeScript Reflect, full
conversation rewriting, and monitor management are outside this release's test
claims. Core-service documentation changes are deferred until operational review.
