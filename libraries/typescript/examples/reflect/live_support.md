# Customer-support pressure test

This is a port of Python's `runner.py` and support workload. The six conversations
have the same inspect → resolve → repeat turns, prompts, policy versions, tools,
integer-cent amounts, idempotency checks and independent verifier. One round is
18 task turns. The inspect turn is evaluated as completed, just like Python.

From `libraries/typescript`, install the native dependencies in [native.md](native.md),
then set `PANDAPROBE_API_KEY`, `PANDAPROBE_PROJECT_NAME` and `OPENAI_API_KEY`
(`ANTHROPIC_API_KEY` for Claude). Set `PANDAPROBE_ENDPOINT` for your platform.

```sh
node --import tsx examples/langchain/04_reflect.ts --rounds 1 --task-timeout 360
```

The same entry point is available under `langgraph`, `deepagents`, `openai-agents`,
`claude-agent-sdk`, and `vercel-ai`. Vercel is a native TS extension; it runs the
same business workload. Real model and platform evaluation calls may incur costs.
The business tools use fictional local data.

| Argument | Python-matching behavior |
| --- | --- |
| `--model NAME` | `REFLECT_EXAMPLE_MODEL`, otherwise `gpt-6-luna`; Claude defaults to `claude-haiku-5-5` |
| `--rounds N` | 1; positive integer |
| `--max-tickets N` | Optional positive bound on conversations across rounds |
| `--max-validations N` | 2; zero disables replay validation |
| `--task-timeout SECONDS` | 360 per task, including evaluation/reflection; positive finite seconds |
| `--output PATH` | A new output directory; existing directories are refused |
| `--workspace PATH` | Explicit reuse; otherwise a fresh workspace inside the output directory |

Without `--output`, a new run is created below `.pandaprobe/reflect-examples`.
Example configuration matches Python: `task_completion: 0.8`, reflection timeout
90 seconds, validation timeout 600 seconds. Other SDK defaults are unchanged.
Credentials, enabled tracing, and required platform metrics are checked before
agent construction. `--help` needs no credentials or optional framework packages.
`_support/cli.ts` only parses these example arguments and launches the runner,
like Python's argparse code. It is not part of the SDK and adds no Reflect CLI.

Each ticket gets a fresh native agent/conversation. Its three turns share that
conversation. Every turn must obtain an `evaluated` platform result. Eligible
candidate/pending proposals are validated at most once per run, up to the configured
limit, using fresh business environments and native conversations for each arm.
Support replay uses the overall validation deadline, as Python does.

`report.json` is atomically replaced after stages and retained on failure. It
contains turns, replies, verifier evidence, Reflect outcomes, phase trace IDs,
shell usage, memory sizes, replay arms, validation decisions, accepted rule
versions/bytes, elapsed time, and platform trace usage when available. Final status
is `completed` or `interrupted`. Completion can include independently failed tasks;
it does not mean the workload was repaired. Unavailable telemetry remains unknown.

The model configuration and native session translations are documented in
[native.md](native.md). Scripted-provider tests exercise real native framework
loops. No live-model improvement claim follows from those tests.
