# Live customer-support exercises

Each framework's `04_reflect.py` builds a real model-backed customer-support agent
and enables Reflect with its native tool format and existing tracing integration.
The shared application lives in `examples/reflect/_support/`; it is not SDK code.

The store is fictional. Order lookup, policy search, and refund/escalation actions
operate on an in-memory ledger. **Model calls and PandaProbe evaluations are real
and may incur charges.** No real refunds or customer messages are sent.

## Run

From `libraries/python`, install **one framework at a time**:

| Example | Install | Provider credential |
| --- | --- | --- |
| `langchain/04_reflect.py` | `uv sync --extra dev --extra langchain --locked` | `OPENAI_API_KEY` |
| `langgraph/04_reflect.py` | `uv sync --extra dev --extra langgraph --locked` | `OPENAI_API_KEY` |
| `deepagents/04_reflect.py` | `uv sync --extra dev --extra deepagents --locked` | `OPENAI_API_KEY`; Python 3.11+ |
| `openai_agents/04_reflect.py` | `uv sync --extra dev --extra openai-agents --locked` | `OPENAI_API_KEY` |
| `google_adk/04_reflect.py` | `uv sync --extra dev --extra google-adk --locked` | `GOOGLE_API_KEY` |
| `crewai/04_reflect.py` | `uv sync --extra dev --extra crewai --locked` | `OPENAI_API_KEY` |
| `claude_agent_sdk/04_reflect.py` | `uv sync --extra dev --extra claude-agent-sdk --locked` | `ANTHROPIC_API_KEY` |

`dev` includes `[reflect]`, provider examples, and test dependencies. For an
application installation, use `pip install 'pandaprobe[reflect,<framework>]'`
plus the chosen model provider. The LangChain family examples use `langchain-openai`.

Set the same identity used by the tracing examples, pointing at a **test project**:

```bash
export PANDAPROBE_API_KEY="..."
export PANDAPROBE_PROJECT_NAME="reflect-support-test"
export PANDAPROBE_ENDPOINT="https://api.pandaprobe.com/"
export OPENAI_API_KEY="..."  # Or the provider key in the table.
```

The platform must support `task_completion`, `tool_correctness`, and
`argument_correctness` evaluations and have its judge provider configured.
The example checks metric discovery and authentication before spending model tokens.
An unavailable evaluation, persistence timeout, or unknown creation outcome stops
the run with partial artifacts; it never silently substitutes offline scores.
The accompanying backend Reflect judge guidance should be deployed for meaningful
assessment of memory administration.

Start with three task turns and no replay, then run the full workload:

```bash
uv run python examples/langchain/04_reflect.py --max-tickets 1 --max-validations 0
uv run python examples/langchain/04_reflect.py
# Longer stress run: 36 task turns across 12 fresh customer conversations.
uv run python examples/langchain/04_reflect.py --rounds 2 --max-validations 3
```

Use the corresponding file for another framework. `--model` (or
`REFLECT_EXAMPLE_MODEL`) overrides the existing examples' provider model default.
CrewAI model names include a provider prefix, e.g. `openai/gpt-6-luna`.
`--help` works without credentials or installing that framework.

## Workload and boundaries

The default is six customer conversations, each with **inspect → authorize →
repeat request** turns. The native conversation persists within a ticket; a fresh
conversation starts for the next customer. Accepted rule storage persists across
all tickets. The workload includes:

- An inflated customer refund claim: use the discounted merchandise payment.
- An ordinary return within the original 30-day window.
- A claimed VIP exception to a final-sale restriction.
- A policy change to 14 days: previously useful 30-day guidance becomes stale.
- A damaged-arrival exception that overrides age and final-sale restrictions.
- Another discounted return with a request to include nonrefundable shipping.

Archived policy articles appear beside the effective policy, and duplicate
requests test idempotency. The task tools enforce authorization and business
invariants. Rejected attempts remain visible to the verifier even if the agent
subsequently recovers. There is no planted wrong answer, prewritten reflection,
automatic rule injection, or guaranteed failure. A strong agent can complete the
workload without generating any reflection.

Reflect supplies `reflect_bash`, `/rules`, and editable `/context.md`; the agent
chooses what to search, read, retain, or propose. SDK defaults bound administration.
These examples use a `task_completion` threshold of 0.8 so the independent replay
verifier can measure precisely that metric; they do not fabricate coherence scores.
Reflection gets 90 seconds; each task/evaluation/reflection cycle gets 360 seconds.
Reflect has no daily evaluation-run count cap by default
(`ReflectConfig.evaluation_max_runs=None`). To set one, pass a positive integer
in the example runner's `ReflectConfig`, such as `evaluation_max_runs=1000`.
This local cost control counts detection and diagnostic creation attempts,
including failed attempts, per UTC day across a reused workspace. Polling does
not consume a slot. At the cap, further run creations are skipped and the reason
appears in the report and warning logs; task execution continues. Existing
concurrency, sampling, timeout, deduplication, and reflection bounds still apply.
A run count is not a monetary budget; runs can include multiple traces and metrics.

After a proposal, the host runs up to **two** candidate validations by default.
Each uses three held-out orders and fresh control/candidate environments, including
one protected ordinary return. That is up to **12 additional agent invocations**,
plus bounded reflection continuations and platform judge calls. `--max-validations`
controls this example's replay allowance. Unvalidated proposals stay pending.
Both arms get identical domain instructions and tools; only their assigned rule
snapshot differs. Shared conversation history, task memory, and mutated business
state are excluded. Validation settles after the task releases its resources.

The verifier measures ledger state, authorization, invalid tool attempts, and a
nonempty reply. It does **not** establish prose accuracy or subjective coherence.
The platform judge independently determines failure notices. Disagreement with
the verifier remains visible in the report and is useful debugging evidence.

## Inspect the results

Every run creates a fresh `.pandaprobe/reflect-examples/<framework>/<run-id>/`
folder (gitignored). `--output` selects a new output directory. `--workspace`
explicitly reuses prior guidance; otherwise each run starts from an empty corpus.
Do not reuse a workspace across frameworks when comparing their independent runs.

`report.json` is saved after each task and replay arm. It records:

- Actual replies and domain tool events, independent outcomes, and elapsed time.
- Task execution/session/trace IDs, platform scores and notices, and reflection outcomes.
- Proposals, accepted rule versions/sizes, and validation decisions with paired evidence.
- Guidance delivery receipts, shell usage, active-memory bytes, and replay-arm delivery.
- Observed LLM span token usage separated by task, reflection, and validation;
  missing telemetry is marked unavailable, not zero. Evaluation-model charges are
  separate and remain visible on the platform's evaluation runs.

The workspace retains rule revisions, candidates, evidence, receipts, and decisions.
Use trace IDs and evaluation-run IDs from the report to inspect model/tool execution
and total operational cost in PandaProbe. If a proposal is accepted, inspect the
paired outcomes and later fresh-customer delivery before claiming improvement.
Acceptance only covers tested cases. A pending result is valid, including lack of
improvement, missing delivery, or a protected case that fails its control.

Claude's SDK generally does not expose complete model inputs. Actual tool-body
delivery may remain unknown and validation can stay pending; this example does
not invent host attestations. Claude runs with only the explicit MCP tools, no
host Bash/filesystem tools or project/user plugins. Deep Agents uses its in-memory
scratch filesystem; its Reflect workspace is separately accessed by `reflect_bash`.

## Verification status

Unit/native-framework tests use deterministic local models and never credentials.
They verify the scenario, verifier, tool schemas, conversation handling, and report
logic. They do not establish hosted model behavior. Live runs require the credentials
above; inspect the generated reports for the actual tested models and results.
There is no comparison against the historical Harness in this workload.
