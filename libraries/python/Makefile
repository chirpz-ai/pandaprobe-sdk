.PHONY: install install-langgraph install-google-adk install-claude-agent-sdk install-crewai lock lint format format-check test test-cov build clean

install:
	uv sync --extra dev

install-langgraph:
	uv sync --extra dev --extra langgraph

install-google-adk:
	uv sync --extra dev --extra google-adk

install-claude-agent-sdk:
	uv sync --extra dev --extra claude-agent-sdk

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