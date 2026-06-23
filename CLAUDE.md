# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This is a multi-language SDK monorepo. Both the Python SDK (`libraries/python/`) and the TypeScript SDK (`libraries/typescript/`) are implemented and mirror the same architecture, trace/span schema, and `CHAIN → AGENT → LLM → TOOL` normalization, so traces are cross-compatible.

The root `Makefile` is a thin wrapper that delegates every `py-*` target into `libraries/python/Makefile` and every `ts-*` target into `libraries/typescript/Makefile`. Run commands from the repo root.

## Common Commands

```bash
make py-install       # uv sync --extra dev (creates .venv, installs dev deps)
make py-lint          # uv run ruff check .
make py-format        # uv run ruff format .
make py-format-check  # check formatting without writing (what CI runs)
make py-test          # uv run pytest tests/ -v
make py-test-cov      # tests with coverage
make py-lock          # regenerate uv.lock after editing pyproject.toml
make py-build         # build wheel via `uv build`
```

Optional framework deps are installed via separate targets — `make py-install-langgraph`, `py-install-google-adk`, `py-install-claude-agent-sdk`, `py-install-crewai`, `py-install-openai-agents`. These are designed to be installed **one at a time** (heavy agent frameworks frequently conflict on transitive deps).

Run a single test:
```bash
cd libraries/python && uv run pytest tests/test_decorators.py::test_name -v
```

Run an example (requires API key):
```bash
cd libraries/python
export PANDAPROBE_API_KEY=... PANDAPROBE_PROJECT_NAME=... PANDAPROBE_ENDPOINT=http://localhost:8000
uv run python examples/decorators/...
```

CI runs `make format-check`, `make lint`, and `make test-cov` against Python 3.10–3.13. Ruff is configured with `line-length = 119` in `pyproject.toml`.

Note: `CONTRIBUTING.md` references `make py-sync` — that target does not exist. Use `make py-install`.

## Architecture

The SDK is layered. Each layer above the core delegates to the one below it; understanding the trace/span lifecycle in **Tracing Core** is a prerequisite for working in any other layer.

### Tracing Core (`src/pandaprobe/tracing/`, `client.py`, `transport.py`)

- `client.Client` is the SDK entry point. A module-level singleton (`_global_client`) is created by `pandaprobe.init(...)` or **auto-initialized on first `get_client()` call** from `PANDAPROBE_API_KEY` + `PANDAPROBE_PROJECT_NAME` env vars. Setting `PANDAPROBE_ENABLED=false` disables auto-init silently. Most code paths call `get_client()` and no-op if it returns `None`.
- `tracing.context.TraceContext` is a context manager representing one trace. It pushes itself onto a `ContextVar` (`_current_trace`); nested `SpanContext` objects push onto `_span_stack`. This is how decorators, wrappers, and integrations all discover the active trace without explicit plumbing.
- `tracing.session` exposes contextvars for the current `session_id` / `user_id`. `set_session` / `with session(...)` / `set_user` / `with user(...)` propagate across **every layer** (decorators, wrappers, integrations). Explicit kwargs always win over contextvar values.
- `transport.Transport` runs a background `threading.Thread` consuming a `queue.Queue` of trace/span/score payloads. Sends are batched (`PANDAPROBE_BATCH_SIZE`) and periodic (`PANDAPROBE_FLUSH_INTERVAL`). Retries on 429/5xx with backoff; gives up immediately on 401/403/422. An `atexit` handler flushes on interpreter shutdown.
- All trace/span I/O follows a strict schema: `{"messages": [{"role": "...", "content": "..."}]}`. `validation.py` enforces this and `warn_if_invalid_messages` is called wherever user-supplied input enters the system. Trace input should be the **current turn's user message only**, not the full conversation history.

### Decorators (`src/pandaprobe/decorators.py`)

`@trace` and `@span` wrap sync **and** async functions (`asyncio.iscoroutinefunction` switch). They look up the current trace via the contextvar — `@span` without an active `@trace` is a no-op. Function args become input, return value becomes output (after message-shape extraction via `validation.extract_last_user_message` / `extract_last_assistant_message`).

### Wrappers (`src/pandaprobe/wrappers/<provider>/`)

These **monkey-patch SDK clients** (OpenAI, Anthropic, Gemini) to emit LLM spans automatically. Each provider has its own subdirectory with `wrapper.py` (the patching) and `utils.py` (provider-specific serialization). Shared parameter whitelists and serializers live in `wrappers/_base.py` (see `SAFE_INVOCATION_PARAMS` — only these kwargs are recorded; anything else is dropped to avoid leaking secrets/large payloads).

### Integrations (`src/pandaprobe/integrations/<framework>/`)

For agent frameworks (LangGraph, Google ADK, Claude Agent SDK, CrewAI, OpenAI Agents). All subclass `_base.BaseIntegrationAdapter`, which provides `_resolve_client()` (explicit client or global singleton) and standard serialization helpers (`safe_serialize`, `config_to_dict`, `SAFE_MODEL_PARAM_KEYS`).

- **LangGraph** uses callback-based instrumentation (`LangGraphCallbackHandler` plugs into `config={"callbacks": [...]}`).
- **Google ADK / Claude Agent SDK / CrewAI** use `wrapt`-based monkey-patching via `adapter.instrument()`.
- **OpenAI Agents** uses the SDK's native tracing hooks.

When adding a new wrapper or integration: mirror an existing sibling, add the optional dep under `[project.optional-dependencies]` in `pyproject.toml`, and run `make py-lock`. Wrappers and integrations share serialization/safety helpers from their respective `_base.py` — extend those rather than reimplementing.

### Schemas (`src/pandaprobe/schemas.py`)

Pydantic models mirroring the backend API. The `StrEnum` shim near the top supports Python 3.10 (where `enum.StrEnum` doesn't exist). `SpanKind` (`AGENT`/`TOOL`/`LLM`/`RETRIEVER`/`CHAIN`/`EMBEDDING`/`OTHER`) is the canonical taxonomy used by every layer.

## Testing Conventions

- `pytest-asyncio` is in auto mode (`asyncio_mode = "auto"` in `pyproject.toml`) — async tests don't need `@pytest.mark.asyncio`.
- HTTP is mocked with `respx`. Tests must never hit a real backend.
- Per-integration tests live under `tests/test_integrations/<framework>/` and `tests/test_wrappers/<provider>/`, mirroring `src/`.

## TypeScript SDK (`libraries/typescript/`)

The TS SDK is a faithful port of the Python SDK. Common commands (from repo root): `make ts-install`, `ts-typecheck`, `ts-lint`, `ts-format-check`, `ts-test`, `ts-test-cov`, `ts-build`, `ts-clean`. CI (`.github/workflows/ci-typescript.yml`) runs lint + format-check + typecheck + test across Node 18/20/22. Package manager is **pnpm**; build is **tsup** (dual ESM+CJS + `.d.ts`); lint/format is **Biome** (line width 119); tests are **vitest**.

Dependency model mirrors Python's extras: the core has **zero runtime dependencies**. `make ts-install` installs dev tooling + the lightweight LLM provider SDKs + LangChain glue (`devDependencies` — the `dev`+`examples` analog). Heavyweight agent frameworks are **optional `peerDependencies`** and are NOT auto-installed (`auto-install-peers=false` in `libraries/typescript/.npmrc`); install them **one at a time** via `make ts-install-{langgraph,langchain,deepagents,claude-agent-sdk,openai-agents,vercel-ai}` (each runs `pnpm add -D`, the `uv sync --extra <name>` analog; reset with `git checkout -- package.json pnpm-lock.yaml && make ts-install`). Because wrappers/integrations type provider clients structurally (`any`) and never hard-import the SDK type packages — and the one dynamic import (`@openai/agents` in the openai-agents adapter) is indirected through a `string` specifier — typecheck/test/build all pass without any framework installed.

Architecture mirrors Python layer-for-layer:

- **Tracing Core** (`src/tracing/`, `client.ts`, `transport.ts`, `schemas.ts`, `validation.ts`, `config.ts`). Key translations: Python `contextvars` → a single `AsyncLocalStorage` per concern (`tracing/context.ts` holds the current trace + span stack; `tracing/session.ts` holds session/user); the background thread + queue → an in-memory buffer drained by a `setInterval` timer + on `batchSize`, flushed on `beforeExit`/`SIGTERM`/`SIGINT`; `httpx` → native `fetch`; Pydantic `to_api_dict()` → hand-written `toApiDict()` preserving the exact omit-null/empty + ISO-8601 + enum-string serialization. The HTTP contract (routes, headers, retry rules) is identical to Python's.
- **Manual instrumentation** (`src/decorators.ts`): JS has no standalone function decorators, so there are two forms — callback wrappers `withTrace`/`withSpan`/`startTrace` (the `with`-context-manager analog) **and** TS class-method `@trace`/`@span` decorators (require `experimentalDecorators`). Both preserve the emit-shape parity (trace input = last user message; output = last assistant message).
- **Wrappers** (`src/wrappers/<provider>/`): `wrapOpenAI`/`wrapAnthropic`/`wrapGemini`/`wrapBedrock`/`wrapMistral`. They replace bound methods on the client instance (the `wrapt`-monkey-patch analog). Shared engine in `wrappers/base.ts` (`openLlmSpan`/`closeLlmSpan`/`errorLlmSpan`, `wrapAsyncStream` for streaming-leak prevention, `SAFE_INVOCATION_PARAMS`). Bedrock is command-based: it patches `client.send` and branches on the command constructor name.
- **Integrations** (`src/integrations/<framework>/`): LangChain family (`langchain`/`langgraph`/`deepagents`) share `_langchain-core/callback.ts`, a `CallbackHandlerMethods`-shaped handler passed via `config: { callbacks: [handler] }` (no hard `@langchain/core` dependency). Adapter-based: `claude-agent-sdk` exposes `wrapClaudeAgentQuery` (wraps the `query` async generator — ESM can't be monkey-patched like Python's wrapt); `openai-agents` registers a tracing processor; `vercel-ai` is net-new (no Python equivalent) — a `pandaProbeMiddleware()` for the AI SDK's `wrapLanguageModel`. Google ADK and CrewAI are intentionally omitted (no mature official JS SDK).

Provider/framework SDKs are **optional `peerDependencies`** (the optional-extras analog); wrappers/integrations type their clients structurally (`any`) and never hard-import the SDK type packages, so the core install stays dependency-free. Subpath `exports` (`pandaprobe/wrappers/openai`, `pandaprobe/integrations/langgraph`) mirror Python's import paths.

### TS Testing Conventions

- **vitest** with `globals: true`; HTTP is mocked by stubbing `global.fetch` (the `respx` analog) via `test/helpers.ts`, which records outgoing payloads for assertions. `test/setup.ts` clears `PANDAPROBE_*` env, pins the endpoint to `http://testserver`, and resets the client singleton between tests (the `conftest.py` analog). Tests must never hit a real backend.
- Tests mirror `src/` under `test/`, `test/wrappers/<provider>`, and `test/integrations/<framework>`.

### TS Versioning & Release

`src/version.ts` (`VERSION`) is the single source of truth (the `_version.py` analog) — it's both the runtime User-Agent version and the release trigger. The release workflow (`.github/workflows/release-typescript.yml`, triggered by pushes to `main` touching `src/version.ts`) gates on the full CI, syncs `package.json`'s version from `version.ts` (`npm pkg set version`), builds, tags `typescript-v<version>`, creates a GitHub release, and publishes to npm. Publishing is **tokenless via OIDC Trusted Publishing** (the PyPI trusted-publisher analog) — no `NPM_TOKEN` secret; auth uses the workflow `id-token` and provenance is automatic. Requires a Trusted Publisher configured on the npm package and npm CLI ≥ 11.5.1 (the workflow upgrades npm). The first-ever publish is manual (claims the name), then Trusted Publishing handles CI releases. The TS package version is intentionally aligned with the Python package version for cross-language parity.
