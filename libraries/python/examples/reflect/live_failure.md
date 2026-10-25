# Live failure → proposal → validation → fresh retrieval

Each framework's `05_reflect_failure.py` complements the successful support
workload with a **controlled stale-runbook incident**. It uses a real native
agent and the ordinary Reflect runtime. Business tools operate on fictional,
in-memory refunds; model calls and platform evaluations are real and billable.

From `libraries/python`, with the same exported credentials used for
[`04_reflect.py`](live_support.md#run):

```bash
uv sync --extra dev --extra langchain --locked
uv run python examples/langchain/05_reflect_failure.py
```

Use the corresponding framework folder and install extra:

| Framework folder | Extra | Default model | Provider credential |
| --- | --- | --- | --- |
| `langchain` | `langchain` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `langgraph` | `langgraph` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `deepagents` | `deepagents` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `openai_agents` | `openai-agents` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `google_adk` | `google-adk` | `gemini-3.8-flash` | `GOOGLE_API_KEY` |
| `crewai` | `crewai` | `openai/gpt-6-luna` | `OPENAI_API_KEY` |
| `claude_agent_sdk` | `claude-agent-sdk` | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |

For example, `uv run python examples/google_adk/05_reflect_failure.py` runs the
same incident with Google ADK. Install one framework at a time; Deep Agents and
CrewAI require Python 3.11+. The existing support and new failure examples share
native agent factories in `_support/agents.py`, with separate task instructions.

The Claude examples explicitly disable thinking and use low effort. The bundled
Claude runtime in Agent SDK 0.1.50 otherwise selects legacy budget-based thinking
for Haiku 5.5, which that model rejects. This is example configuration; Reflect
preserves the application's thinking settings. Provider error results stop the
invocation, and the report retains the error type and message.

Required shared variables: `PANDAPROBE_API_KEY` and `PANDAPROBE_PROJECT_NAME`;
optionally `PANDAPROBE_ENDPOINT`. Use a test project with evaluation judges
configured. `--model` or `REFLECT_EXAMPLE_MODEL` overrides the task model.
Platform judge configuration is separate. `--help` requires neither credentials
nor the selected framework. No extra SDK settings are needed.

**Retrieval audit:** this example reads recorded task traces to check whether
accepted guidance appeared in a model request before the fresh refund. This
conservative check lives in `_support/delivery.py`, outside the distributed SDK.
For the LangChain family, an example-owned middleware/callback also observes task
inputs, covering Python 3.10 paths where existing traces omit model inputs.
The report identifies each observation's source. The audit does not participate
in SDK validation or alter the agent's prompts, tools, or decisions.
Claude Agent SDK traces generally do not expose complete model inputs: its rule
can pass behavioral validation while the example remains incomplete at
`fresh_delivery` because retrieval could not be confirmed.

## What makes the failure observable

The deployed refund service refunds paid merchandise only. The agent starts with
an explicitly legacy fallback that uses the invoice total, including outbound
shipping. Its instructions allow applicable learned policy updates to supersede
that fallback. There is no initial rule or host-authored correction.

The discovery order has 6,500 cents of merchandise and 700 cents of shipping.
Submitting 7,200 cents produces a visible `REFUND_AMOUNT_POLICY_MISMATCH`, explains
the current policy, and leaves the refund ledger empty. The simulated settlement
batch allows one submission and quarantines rejected batches for operations
review. The agent cannot erase the failure by retrying within that task.

This one-submission constraint and obsolete runbook are deliberate fault fixtures,
applied identically in discovery, replay, and follow-up. This tests a known policy
migration, not how often an otherwise strong agent naturally fails. If the model
independently chooses the correct amount, the script records that result and stops;
it does not override the model or fabricate a failure.

## Stages and checks

1. **Discover:** the original failed task result is preserved. An independent
   verifier checks the actual ledger, lookup before submission, a single attempt,
   and a nonempty response. It does not grade prose accuracy. The tool's rejected
   result is present in the native trace. Platform evaluation must independently
   issue a failure notice before Reflect continues the agent.
2. **Propose:** the same agent reads `/inbox` and manages `/rules` with
   `reflect_bash`. Host code supplies no rule text and cannot approve a proposal.
   A context.md-only edit is not persistent accepted guidance and cannot complete
   this demonstration.
3. **Validate once:** the SDK runs three held-out cases in both control and
   candidate arms. The cases use different order IDs and amounts: 4,200 + 1,200
   shipping, 11,300 + 300 shipping, and a protected 2,700-cent order with free
   shipping. Every arm gets identical prompts/tools, a fresh agent conversation,
   an empty active-memory file, and a new business ledger. Only the pinned rule
   snapshot differs. The protected control must succeed, candidates must avoid
   regressions, and measured improvement must meet the SDK's validation threshold.
   SDK acceptance does not require rule text to match a model input. Pending/rejected results
   stop the demonstration; there is no retry-until-accepted loop.
4. **Retrieve on a fresh execution:** after acceptance, a new Reflect instance
   reopens the workspace from disk. A new agent/checkpointer and customer session
   handle a new 9,250 + 850 shipping order. The example verifies the accepted
   revision, empty initial active memory, and no candidate trial assignment. Its
   own trace audit verifies that text from each accepted rule reached a task model
   request before the first refund attempt. A verified excerpt suffices; reading
   the whole file is not required. The independent ledger and platform evaluation
   are recorded separately.

The example audit uses only **task inputs**, excluding subsequent reflection.
It compares the recorded model-request time with the first refund attempt, not
the time the audit runs. Reading a rule only after the refund submission cannot
satisfy this demonstration. Reading guidance and succeeding once still do not
establish general reliability. `before_submission_excerpt_versions` records
versions with verified text in model input before the action, whether partial
or complete. A filename or search command alone does not count. For numbered
grep output, each matched path, line number, and line text must agree with the
current file. Retrieval does not prove that the excerpt was relevant or used;
business success and paired improvement remain separate evidence.

The report still distinguishes complete coverage (`model_input_versions` and
`before_submission_versions`) from partial snippets (`rule_coverage` and
`partial_model_input_versions`). Only verbatim full text or numbered output
covering every nonblank line counts as complete. Separate excerpts are never
combined into a claim of whole-file coverage. These checks live only in the
example and do not affect SDK rule acceptance.

## Bounds and artifacts

At most **eight task invocations** run: discovery, six replay arms, and one fresh
follow-up. Reflection continuations are bounded separately. Multiple model/tool
calls can occur within an invocation. Discovery and follow-up use real platform
evaluation (normally three run creations including discovery diagnostics; at most
four if the follow-up also triggers diagnostics). Replay uses the host ledger
verifier and does not recursively create platform evaluations.

The default task/evaluation/reflection deadline is 360 seconds per task
(`--task-timeout`); reflection itself gets 90 seconds and the full paired replay
gets 600 seconds. The SDK's other normal limits apply. These bounds are not a
monetary spending limit, and missing trace cost telemetry does not mean zero cost.
The example's later retrieval audit allows up to 10 seconds for export and
30 seconds for reading existing task traces; it creates no evaluations.

Each invocation of the script creates a new directory:

```text
.pandaprobe/reflect-examples/<framework>-failure/<run-id>/
  report.json
  workspace/<identity-namespace>/...
```

`--output` selects a new artifact directory; existing directories are refused.
There is intentionally no workspace reuse option for the initial incident, to
prevent previously learned guidance from contaminating the baseline. Reopening
the accepted workspace for the final task happens automatically within the run.
Historical `langchain/04_reflect.py` artifacts are untouched.
See the [workspace guide](workspace.md) for the contents below `workspace/`.

`report.json` is updated after each task and replay arm and contains:

- `milestones`: verified initial failure, platform notice, proposal, independent
  acceptance, fresh retrieval, and fresh task success. Unattempted stages are null.
- Original task replies, tool requests/results, immutable expected ledger outcomes,
  and separate platform evaluation/reflection records.
- The full paired verdict with distinct execution IDs and independently verified outcomes,
  accepted rule text and versions, and the separate example-only task-input audit.
- Configuration, timing, workspace path, task/reflection/validation trace IDs, and
  best-effort model token/cost telemetry. Judge cost remains on evaluation runs.

`status: completed` means every stage was observed, including an agreeing fresh
platform evaluation. `status: incomplete` names `stopped_at` and `reason` and exits
nonzero; it is a measured outcome, not proof that the runtime is broken.
An incomplete retrieval audit does not revoke SDK acceptance or change the
verified task result.
Operational exceptions preserve an `interrupted` report and propagate. Authentication
and metric discovery are checked before model spend; a preflight failure can occur
before a report exists.

Offline tests exercise all seven native framework loops with scripted model and
API fixtures, including Claude's accepted candidate with unverified fresh retrieval. LangChain additionally
covers protected regressions, no-proposal and judge-miss outcomes, missing/late
retrieval failures, verified partial retrieval, and operational failure reports. A recorded numbered-grep
output from the live LangChain incident is retained as a regression fixture. **These
fixtures do not establish that the live model will produce an accepted rule.**
Inspect the live report before making any improvement claim. This is a focused
integration demonstration, not a benchmark against the historical Harness.
