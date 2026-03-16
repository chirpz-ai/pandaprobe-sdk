.PHONY: install install-dev sync lock lint format format-check test test-cov build clean

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,openai,langgraph]"

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
