.PHONY: install install-crewai lock lint format format-check test test-cov build clean

install:
	uv sync --extra dev

install-crewai:
	uv sync --extra dev --extra crewai

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