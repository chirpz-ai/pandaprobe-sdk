# Controlled stale-runbook refund incident

This mirrors Python's `failure_runner.py`, workload version 4: the same instructions,
orders, payment breakdowns, tool schemas, prompts, one-submission service, held-out
cases, protected case, verifier, stop conditions and report milestones.

```sh
node --import tsx examples/openai-agents/05_reflect_failure.ts --task-timeout 360
```

Entry points also exist for LangChain, LangGraph, DeepAgents, Claude Agent SDK and
Vercel AI. Install [native dependencies](native.md) and configure platform/provider
credentials as in the [support guide](live_support.md). Calls to the provider and
platform are real; the refund service is fictional.

Arguments are exactly the Python controls: `--model`, `--output`, and
`--task-timeout` (360 seconds per task, including each replay arm). Defaults are
`REFLECT_EXAMPLE_MODEL` or `gpt-6-luna`, with `claude-haiku-5-5` for Claude.
An output directory must be new. Its workspace must contain no executions,
proposals or accepted rules. This example has no workspace-reuse option.
The example threshold is `task_completion: 0.8`, with 90-second reflection and
600-second validation timeouts. Required platform metrics are checked first.

The run proceeds through these Python milestones:

1. Independently verify the discovery task failed with `REFUND_AMOUNT_POLICY_MISMATCH`.
2. Obtain a completed platform evaluation with a failure notice.
3. Obtain a new candidate from that incident's reflection checkpoint.
4. Independently validate three paired cases (six fresh replay arms), including
   the protected zero-shipping case, and accept through the SDK.
5. Reopen the same workspace in a **new Reflect instance**, with a fresh agent and
   native conversation. Verify accepted guidance reached task model input before
   the refund attempt, with empty initial memory and no candidate assignment.
6. Independently verify the fresh 9250-cent refund and an agreeing completed
   platform evaluation without a failure notice.

The example-only delivery audit follows Python's exact matching rules: full text
or an exact, numbered `/rules/...` excerpt in an actual task LLM input. A filename,
a tool response by itself, an unnumbered partial fragment, a model's claim, or a
late read does not establish delivery. This audit does not control core acceptance.
It uses platform LLM inputs and native LangChain-family model-input observations.
Claude's tracing adapter records MCP text-block arrays as JSON strings inside
tool messages in subsequent LLM inputs. The example audit decodes that known
representation before applying the same exact-text and before-submission checks.
Tracing payloads are unchanged. Missing inputs still cannot be inferred from
successful tool calls or the final answer. All six native example tests verify
the complete failure/validation/fresh-delivery loop with mocked providers.

Platform judges can disagree with the deterministic ledger check. For example,
`task_completion=0.8` does not trigger a failure at the `0.8` threshold: the score
must be **strictly below** it, as in Python. Such a run remains incomplete at
`detection`; its reason includes evaluation status, scores and thresholds. The
example does not force a notice, change the threshold or retry evaluations until
it obtains a desired score.

Status is `completed` only after all milestones. Missing evidence or an unmet
milestone produces `incomplete`, with `stopped_at` and `reason`; exceptions retain
an `interrupted` report and propagate. Both incomplete and error runs exit nonzero.
The intended failure is observed rather than forced by the runner.

`report.json` is written atomically throughout, including partial turns/replay arms,
errors, independent evidence, delivery receipts, accepted text/versions, SDK
outcomes, timings and available platform trace usage. Unavailable cost is unknown.
The native tests use real framework loops with scripted providers and a mocked
platform. No live provider run or real-model improvement measurement is claimed.
