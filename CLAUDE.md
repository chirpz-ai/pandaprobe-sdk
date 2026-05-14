# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This is a multi-language SDK monorepo. Only the Python SDK is implemented; `libraries/typescript/` is a placeholder. All real source lives under `libraries/python/`.

The root `Makefile` is a thin wrapper that delegates every `py-*` target into `libraries/python/Makefile`. Run commands from the repo root.

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
