.PHONY: install install-dev sync lock lint format format-check test test-cov build clean nuke-venv

install:
	uv pip install -e .

install-dev:
	uv sync --all-extras

sync:
	uv sync --all-extras

lock:
	uv lock

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=pandaprobe --cov-report=term-missing

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache

# iCloud / cloud sync can leave read-only bits that break plain `rm -rf .venv`.
nuke-venv:
	@if [ -d .venv ]; then chmod -R u+w .venv && rm -rf .venv && echo "Removed .venv"; else echo "No .venv to remove"; fi
