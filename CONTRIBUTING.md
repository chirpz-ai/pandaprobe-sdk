# Contributing to PandaProbe SDK

Thanks for your interest in contributing! This guide will help you get set up and submit your first pull request.

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Git**

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Getting Started

1. Fork the repo and clone your fork:

```bash
git clone https://github.com/<your-username>/pandaprobe-sdk.git
cd pandaprobe-sdk
```

2. Install the Python SDK with all development dependencies:

```bash
make py-sync
```

This runs `uv sync --all-extras` inside `libraries/python/`, which creates a virtual environment and installs all dependencies (including optional ones like `openai` and `langgraph`) from the lockfile.

3. Verify everything works:

```bash
make py-lint
make py-test
```

## Repository Structure

```
pandaprobe-sdk/
├── libraries/
│   └── python/              # Python SDK
│       ├── src/pandaprobe/  # Source code
│       ├── tests/           # Unit tests
│       ├── examples/        # Example scripts
│       ├── pyproject.toml   # Package metadata & dependencies
│       └── Makefile         # Python-specific targets
├── Makefile                 # Root targets (delegates to language-specific Makefiles)
├── .github/                 # CI workflows, issue templates
└── CONTRIBUTING.md          # This file
```

## Development Workflow

### Make Targets

All commands are run from the repo root. Each `py-*` target delegates to `libraries/python/Makefile`.

| Command | Description |
|---|---|
| `make py-sync` | Install all deps from lockfile (creates venv) |
| `make py-lint` | Run ruff linter |
| `make py-format` | Auto-format code with ruff |
| `make py-format-check` | Check formatting without modifying files |
| `make py-test` | Run tests |
| `make py-test-cov` | Run tests with coverage report |
| `make py-lock` | Regenerate `uv.lock` after changing dependencies |
| `make py-build` | Build the package |
| `make py-clean` | Remove build artifacts |

### Making Changes

1. Create a branch from `main`:

```bash
git checkout -b feat/my-feature
```

2. Make your changes in `libraries/python/src/pandaprobe/`.

3. Add or update tests in `libraries/python/tests/`.

4. Run lint and tests before committing:

```bash
make py-format
make py-lint
make py-test
```

5. If you added or changed dependencies in `pyproject.toml`, regenerate the lockfile:

```bash
make py-lock
```

### Running Examples

The `libraries/python/examples/` directory contains scripts you can run against a local backend:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_ENDPOINT="http://localhost:8000"
export PANDAPROBE_PROJECT_NAME="my-project"

cd libraries/python
uv run python examples/01_decorators.py
```

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
- Line length limit is **119 characters** (configured in `pyproject.toml`).
- No unnecessary comments — code should be self-explanatory. Comments are for non-obvious intent or trade-offs only.
- Type hints are encouraged for all public APIs.

## Testing

- All tests live under `libraries/python/tests/`.
- Tests use `pytest` with `pytest-asyncio` for async support.
- HTTP calls are mocked with `respx` — tests should never hit a real backend.
- Aim for coverage on any new code paths you introduce.

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a PR against `main` on the upstream repo.
3. Fill out the PR template — link related issues, describe your changes, and check the contributor checklist.
4. CI will run lint, format check, and tests across Python 3.10–3.13. All checks must pass.
5. A maintainer will review your PR. Address any feedback, then it gets merged.

## Adding a New Wrapper or Integration

The SDK is designed to be extensible:

- **Wrappers** (monkey-patching external clients): Add a new subdirectory under `libraries/python/src/pandaprobe/wrappers/<provider>/`. Follow the structure of `wrappers/openai/` — shared helpers live in `wrappers/_base.py`.
- **Integrations** (callback-based instrumentation): Add a new subdirectory under `libraries/python/src/pandaprobe/integrations/<framework>/`. Follow the structure of `integrations/langgraph/`.

If your wrapper or integration requires new optional dependencies, add them under `[project.optional-dependencies]` in `pyproject.toml` and regenerate the lockfile.

## Reporting Bugs & Requesting Features

- **Bugs**: Use the [Bug Report](https://github.com/chirpz-ai/pandaprobe-sdk/issues/new?template=bug_report.yml) template.
- **Features**: Use the [Feature Request](https://github.com/chirpz-ai/pandaprobe-sdk/issues/new?template=feature_request.yml) template.
- **Questions**: Start a [Discussion](https://github.com/chirpz-ai/pandaprobe-sdk/discussions).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
