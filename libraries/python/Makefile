.PHONY: install install-dev lint format test test-cov build clean

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,openai,langgraph]"

sync:
	uv sync --all-extras

lock:
	uv lock

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=pandaprobe --cov-report=term-missing

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
